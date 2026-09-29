"""Unit tests for DroneDisturbed3DEnv, WindField3D, and DisturbanceRecoveryTracker."""

from typing import Tuple

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from adaptive_rl.environments import make_env
from adaptive_rl.environments.disturbed_drone import (
    DisturbanceRecoveryTracker,
    DroneDisturbed3DEnv,
    DynamicObstacleSphere3D,
    WindField3D,
)


def test_disturbed_drone_gymnasium_checker() -> None:
    """Verify DroneDisturbed3DEnv passes Gymnasium's check_env."""
    env = DroneDisturbed3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20)
    check_env(env)
    env.close()


def test_disturbed_drone_registry_instantiation() -> None:
    """Verify environment can be created via registry under both aliases."""
    env1 = make_env("drone_disturbed", max_steps=25)
    env2 = make_env("drone_disturbance", max_steps=25)

    assert isinstance(env1, DroneDisturbed3DEnv)
    assert isinstance(env2, DroneDisturbed3DEnv)

    env1.close()
    env2.close()


def test_disturbed_drone_spaces() -> None:
    """Verify action and observation space bounds and shapes (standard 29-dim observation)."""
    env = DroneDisturbed3DEnv(bounds=(50.0, 50.0, 25.0), num_lidar_rays=16)

    assert env.action_space.shape == (3,)
    assert env.action_space.dtype == np.float32
    np.testing.assert_allclose(env.action_space.low, -np.ones(3, dtype=np.float32))
    np.testing.assert_allclose(env.action_space.high, np.ones(3, dtype=np.float32))

    # Standard unified 29-dim observation: 13 kinematics/goal + 16 lidar = 29
    assert env.observation_space.shape == (29,)
    assert env.observation_space.dtype == np.float32
    np.testing.assert_allclose(env.observation_space.low, -np.ones(29, dtype=np.float32))
    np.testing.assert_allclose(env.observation_space.high, np.ones(29, dtype=np.float32))

    env.close()


def test_wind_field_steady_and_shear() -> None:
    """Verify steady wind scaling with altitude shear."""
    field = WindField3D(
        steady_wind=(2.0, 0.0, 0.0),
        gust_theta=0.15,
        gust_sigma=0.0,
        altitude_shear=0.05,
    )
    # At z=0
    w0 = field.get_wind(np.array([10.0, 10.0, 0.0]))
    np.testing.assert_allclose(w0.steady, [2.0, 0.0, 0.0])
    assert w0.speed == pytest.approx(2.0)

    # At z=10: factor = 1 + 0.05*10 = 1.5 -> steady = [3.0, 0.0, 0.0]
    w10 = field.get_wind(np.array([10.0, 10.0, 10.0]))
    np.testing.assert_allclose(w10.steady, [3.0, 0.0, 0.0])
    assert w10.speed == pytest.approx(3.0)


def test_wind_field_stochastic_gusts() -> None:
    """Verify OU stochastic gust generation and clamping."""
    field = WindField3D(
        steady_wind=(0.0, 0.0, 0.0),
        gust_theta=0.2,
        gust_sigma=1.0,
        max_gust=3.0,
        dt=0.1,
    )
    rng1 = np.random.default_rng(123)
    field.reset()

    # Step for 50 steps
    gusts = []
    for _ in range(50):
        g = field.step_gust(rng1)
        gusts.append(g)
        assert np.linalg.norm(g) <= 3.0 + 1e-6

    # Verify deterministic reproducibility with same seed
    rng2 = np.random.default_rng(123)
    field.reset()
    for expected in gusts:
        g2 = field.step_gust(rng2)
        np.testing.assert_allclose(g2, expected)


def test_wind_causes_physical_drift() -> None:
    """Verify steady wind exerts aerodynamic force on drone trajectory."""
    # Zero wind env
    env_calm = DroneDisturbed3DEnv(
        bounds=(50.0, 50.0, 25.0),
        start_pos=(10.0, 10.0, 10.0),
        wind_speed=0.0,
        steady_wind=(0.0, 0.0, 0.0),
        gust_sigma=0.0,
        num_obstacles=0,
    )
    # Positive X wind env (5.0 m/s)
    env_wind = DroneDisturbed3DEnv(
        bounds=(50.0, 50.0, 25.0),
        start_pos=(10.0, 10.0, 10.0),
        wind_speed=5.0,
        steady_wind=(5.0, 0.0, 0.0),
        gust_sigma=0.0,
        num_obstacles=0,
    )

    env_calm.reset(seed=42)
    env_wind.reset(seed=42)

    zero_action = np.zeros(3, dtype=np.float32)

    # Step 10 times with zero control action
    for _ in range(10):
        obs_c, _, _, _, info_c = env_calm.step(zero_action)
        obs_w, _, _, _, info_w = env_wind.step(zero_action)

    # In calm, drone should stay near initial position (0 velocity, 0 net force)
    assert info_c["position"][0] == pytest.approx(10.0, abs=1e-3)

    # In wind along +X, aerodynamic drag induces positive drift along X (acc ~ 0.25 m/s^2)
    assert info_w["position"][0] > 10.1
    assert info_w["velocity"][0] > 0.1

    env_calm.close()
    env_wind.close()


def test_dynamic_obstacles_movement_and_bounce() -> None:
    """Verify dynamic obstacles move and bounce within bounds."""
    dyn_obs = DynamicObstacleSphere3D(
        position=np.array([28.0, 10.0, 10.0]),
        velocity=np.array([2.0, 0.0, 0.0]),
        radius=1.0,
    )
    bounds: Tuple[float, float, float] = (30.0, 20.0, 20.0)

    # Step until it bounces off X boundary (bounds[0] - radius = 29.0)
    for _ in range(10):
        dyn_obs.step(0.1, bounds)

    # Velocity along X should have flipped negative after hitting boundary
    assert dyn_obs.velocity[0] < 0.0
    assert dyn_obs.position[0] <= 29.0


def test_disturbance_recovery_tracker() -> None:
    """Verify DisturbanceRecoveryTracker detects events, onset, and recovery."""
    tracker = DisturbanceRecoveryTracker()
    assert tracker.telemetry()["recovery_events"] == 0

    # Step 1: calm (mag 0.2 < threshold 0.7)
    tracker.update(
        step=1,
        magnitude=0.2,
        speed=1.0,
        threshold=0.7,
        speed_tolerance=0.5,
        hold_steps=2,
    )
    assert not tracker.telemetry()["disturbance_active"]

    # Step 2: disturbance onset (mag 1.2 >= threshold 0.7)
    tracker.update(
        step=2,
        magnitude=1.2,
        speed=1.0,
        threshold=0.7,
        speed_tolerance=0.5,
        hold_steps=2,
    )
    assert tracker.telemetry()["disturbance_active"]
    assert tracker.telemetry()["disturbance_onset_step"] == 2

    # Step 3: disturbance subsides below threshold, speed within tolerance (streak 1)
    tracker.update(
        step=3,
        magnitude=0.3,
        speed=1.1,
        threshold=0.7,
        speed_tolerance=0.5,
        hold_steps=2,
    )
    assert tracker.telemetry()["disturbance_active"]
    assert not tracker.telemetry()["recovered"]

    # Step 4: calm sustained for hold_steps (streak 2 -> recovery completed!)
    tracker.update(
        step=4,
        magnitude=0.3,
        speed=1.1,
        threshold=0.7,
        speed_tolerance=0.5,
        hold_steps=2,
    )
    assert not tracker.telemetry()["disturbance_active"]
    assert tracker.telemetry()["recovered"]
    assert tracker.telemetry()["recovery_completed"] == 1
    assert tracker.telemetry()["recovery_times"] == [2]  # step 4 - step 2 = 2 steps


def test_set_and_get_effective_parameters() -> None:
    """Verify dynamic parameter updating via set_parameters and get_effective_parameters."""
    env = DroneDisturbed3DEnv(bounds=(50.0, 50.0, 25.0), wind_speed=0.5, num_obstacles=8)
    params = env.get_effective_parameters()
    assert params["wind_speed"] == pytest.approx(0.5)
    assert params["num_obstacles"] == 8

    # Apply TEST-B overrides
    env.set_parameters(wind_speed=4.0, gust_sigma=0.6, num_obstacles=12)
    params_b = env.get_effective_parameters()
    assert params_b["wind_speed"] == pytest.approx(4.0)
    assert params_b["gust_sigma"] == pytest.approx(0.6)
    assert params_b["num_obstacles"] == 12

    env.close()
