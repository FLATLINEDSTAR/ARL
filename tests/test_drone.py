"""Comprehensive tests for 3D Drone Navigation, kinematics, 3D LiDAR, and continuous control."""

import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.environments import make_env
from adaptive_rl.environments.drone import (
    DroneKinematics3D,
    DroneNavigation3DEnv,
    ObstacleSphere3D,
    compute_lidar_3d_readings,
    generate_drone_obstacles,
    generate_lidar_3d_ray_directions,
    ray_cast_box_boundaries_3d,
    ray_cast_sphere_3d,
)
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.training.trainer import PPOTrainer


def test_drone_gymnasium_checker() -> None:
    """Verify DroneNavigation3DEnv passes Farama Gymnasium check_env."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20)
    check_env(env)
    env.close()


def test_drone_registry_instantiation() -> None:
    """Verify drone can be instantiated via make_env under aliases."""
    env1 = make_env("drone", max_steps=25)
    env2 = make_env("drone_3d", max_steps=25)

    assert isinstance(env1, DroneNavigation3DEnv)
    assert isinstance(env2, DroneNavigation3DEnv)

    env1.close()
    env2.close()


def test_drone_spaces() -> None:
    """Verify action and observation space bounds and shapes."""
    env = DroneNavigation3DEnv(bounds=(50.0, 50.0, 25.0), num_lidar_rays=16)
    assert env.action_space.shape == (3,)
    assert env.action_space.dtype == np.float32
    np.testing.assert_allclose(env.action_space.low, np.array([-1.0, -1.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(env.action_space.high, np.array([1.0, 1.0, 1.0], dtype=np.float32))

    # 13 base + 16 lidar = 29
    assert env.observation_space.shape == (29,)
    assert env.observation_space.dtype == np.float32
    np.testing.assert_allclose(env.observation_space.low, -np.ones(29, dtype=np.float32))
    np.testing.assert_allclose(env.observation_space.high, np.ones(29, dtype=np.float32))
    env.close()


def test_drone_deterministic_seeding() -> None:
    """Verify identical seeds produce identical obstacle layouts and observations."""
    env1 = DroneNavigation3DEnv(bounds=(30.0, 30.0, 15.0), num_obstacles=5)
    env2 = DroneNavigation3DEnv(bounds=(30.0, 30.0, 15.0), num_obstacles=5)

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)

    np.testing.assert_array_almost_equal(obs1, obs2)
    np.testing.assert_array_equal(info1["position"], info2["position"])
    assert len(env1._obstacles) == len(env2._obstacles)
    for o1, o2 in zip(env1._obstacles, env2._obstacles):
        np.testing.assert_array_almost_equal(o1.center, o2.center)
        assert o1.radius == o2.radius

    # Step both environments with identical action
    act = np.array([0.5, -0.2, 0.8], dtype=np.float32)
    o1, r1, t1, tr1, i1 = env1.step(act)
    o2, r2, t2, tr2, i2 = env2.step(act)

    np.testing.assert_array_almost_equal(o1, o2)
    assert r1 == r2
    assert t1 == t2
    assert tr1 == tr2

    env1.close()
    env2.close()


def test_drone_kinematics_equations_of_motion() -> None:
    """Verify 3D kinematics integration, acceleration, drag damping, and speed limits."""
    kinematics = DroneKinematics3D(
        dt=0.1,
        max_velocity=10.0,
        max_acceleration=5.0,
        linear_damping=0.1,
    )
    init_pos = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    kinematics.reset(init_pos)

    # Apply forward X acceleration of 4.0 m/s^2
    pos, vel = kinematics.step(np.array([4.0, 0.0, 0.0]))
    # dt = 0.1: dv = (4.0 - 0.1 * 0) * 0.1 = 0.4 m/s
    assert vel[0] == pytest.approx(0.4)
    assert vel[1] == pytest.approx(0.0)
    assert vel[2] == pytest.approx(0.0)
    # dx = 0.4 * 0.1 = 0.04 m -> pos = 10.04
    assert pos[0] == pytest.approx(10.04)

    # Step without acceleration: damping should decelerate velocity
    pos2, vel2 = kinematics.step(np.array([0.0, 0.0, 0.0]))
    assert vel2[0] < vel[0]

    # Test maximum speed ceiling
    for _ in range(50):
        pos_fast, vel_fast = kinematics.step(np.array([5.0, 5.0, 5.0]))
    speed = float(np.linalg.norm(vel_fast))
    assert speed <= 10.0 + 1e-6


def test_drone_ray_cast_sphere_3d() -> None:
    """Verify 3D analytical ray-sphere intersection tests."""
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_dir = np.array([1.0, 0.0, 0.0])
    sphere_center = np.array([10.0, 0.0, 0.0])
    radius = 2.0

    # Ray straight towards sphere center at distance 10.0 with radius 2.0 -> hit at 8.0
    dist = ray_cast_sphere_3d(ray_origin, ray_dir, sphere_center, radius, max_range=20.0)
    assert dist == pytest.approx(8.0)

    # Ray directed away from sphere (-X) -> misses
    dist_away = ray_cast_sphere_3d(ray_origin, -ray_dir, sphere_center, radius, max_range=20.0)
    assert dist_away == pytest.approx(20.0)

    # Inside sphere -> returns 0.0
    dist_inside = ray_cast_sphere_3d(sphere_center, ray_dir, sphere_center, radius, max_range=20.0)
    assert dist_inside == pytest.approx(0.0)


def test_drone_ray_cast_box_boundaries_3d() -> None:
    """Verify 3D ray-box boundary slab intersection."""
    bounds = (50.0, 50.0, 25.0)
    ray_origin = np.array([10.0, 20.0, 15.0])

    # Ray pointing along +X (toward X=50.0) -> distance is 50.0 - 10.0 = 40.0
    d_x = ray_cast_box_boundaries_3d(ray_origin, np.array([1.0, 0.0, 0.0]), bounds, max_range=100.0)
    assert d_x == pytest.approx(40.0)

    # Ray pointing along -Z (toward Z=0.0) -> distance is 15.0 - 0.0 = 15.0
    d_z = ray_cast_box_boundaries_3d(
        ray_origin, np.array([0.0, 0.0, -1.0]), bounds, max_range=100.0
    )
    assert d_z == pytest.approx(15.0)


def test_drone_lidar_3d_readings() -> None:
    """Verify 16-ray 3D rangefinder readings normalized to [0, 1]."""
    ray_dirs = generate_lidar_3d_ray_directions(num_rays=16)
    assert ray_dirs.shape == (16, 3)

    bounds = (40.0, 40.0, 20.0)
    origin = np.array([20.0, 20.0, 10.0])
    obstacles = [ObstacleSphere3D(center=np.array([25.0, 20.0, 10.0]), radius=2.0)]

    readings = compute_lidar_3d_readings(
        origin=origin,
        ray_directions=ray_dirs,
        obstacles=obstacles,
        bounds=bounds,
        max_range=20.0,
    )
    assert readings.shape == (16,)
    assert np.all(readings >= 0.0)
    assert np.all(readings <= 1.0)
    assert float(readings.min()) <= 0.16


def test_drone_procedural_obstacles_clearance() -> None:
    """Verify procedurally generated obstacles respect start and goal clearance."""
    bounds = (50.0, 50.0, 25.0)
    start = np.array([5.0, 5.0, 5.0])
    goal = np.array([45.0, 45.0, 20.0])
    obstacles = generate_drone_obstacles(
        bounds=bounds,
        start_pos=start,
        goal_pos=goal,
        num_obstacles=6,
        obstacle_radius=2.0,
        clearance_radius=4.0,
        rng=np.random.default_rng(42),
    )
    assert len(obstacles) == 6
    for obs in obstacles:
        assert np.linalg.norm(obs.center - start) >= 6.0
        assert np.linalg.norm(obs.center - goal) >= 6.0


def test_drone_goal_reached_termination() -> None:
    """Verify reaching the target waypoint awards goal reward and terminates."""
    env = DroneNavigation3DEnv(
        bounds=(20.0, 20.0, 10.0),
        start_pos=np.array([10.0, 10.0, 5.0]),
        goal_pos=np.array([10.5, 10.0, 5.0]),  # 0.5m away
        target_radius=1.5,
        num_obstacles=0,
    )
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0]))

    assert terminated is True
    assert info["success"] is True
    assert reward == env.goal_reward
    env.close()


def test_drone_boundary_collision_termination() -> None:
    """Verify colliding with perimeter wall triggers collision termination."""
    env = DroneNavigation3DEnv(
        bounds=(20.0, 20.0, 10.0),
        start_pos=np.array([1.0, 10.0, 5.0]),
        collision_radius=1.2,  # Start pos 1.0 - 1.2 = -0.2 <= 0.0 -> collision!
        num_obstacles=0,
        terminate_on_collision=True,
    )
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0]))

    assert terminated is True
    assert info["collision"] is True
    assert reward == env.collision_reward
    env.close()


def test_drone_truncation_at_max_steps() -> None:
    """Verify episode truncates at max_steps."""
    max_steps = 10
    env = DroneNavigation3DEnv(
        bounds=(50.0, 50.0, 25.0),
        num_obstacles=0,
        max_steps=max_steps,
        terminate_on_collision=False,
    )
    env.reset(seed=42)
    for _ in range(max_steps - 1):
        _, _, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
        assert term is False
        assert trunc is False

    _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
    assert trunc is True
    assert term is False
    assert info["step"] == max_steps
    env.close()


def test_drone_render_modes() -> None:
    """Verify textual 3D flight dashboard rendering."""
    env = DroneNavigation3DEnv(render_mode="ansi")
    env.reset(seed=42)
    rendered = env.render()
    assert isinstance(rendered, str)
    assert "SIMULATED 3D DRONE FLIGHT DECK" in rendered
    assert "Altitude (Z)" in rendered
    assert "Waypoint" in rendered

    # Human mode
    env_h = DroneNavigation3DEnv(render_mode="human")
    env_h.reset(seed=42)
    env_h.step(np.zeros(3, dtype=np.float32))
    env_h.close()
    env.close()


def test_drone_ppo_training(tmp_path: Path) -> None:
    """Integration test: Train continuous PPO on 3D drone navigation."""
    exp_cfg = ExperimentConfig(
        name="test_drone_ppo",
        seed=42,
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=0.0003,
            gamma=0.99,
            batch_size=32,
            parameters={"n_steps": 64, "n_epochs": 2},
        ),
        environment=EnvironmentConfig(
            name="drone",
            max_steps=20,
            parameters={"bounds": [20.0, 20.0, 10.0], "num_obstacles": 2},
        ),
        training=TrainingConfig(
            total_timesteps=128,
            checkpoint_freq=64,
            log_interval=1,
        ),
        evaluation=EvaluationConfig(
            eval_episodes=2,
            deterministic=True,
        ),
    )

    trainer = PPOTrainer(config=exp_cfg)
    result = trainer.fit()

    assert result.total_timesteps == 128
    assert result.final_model_path.exists()
    assert result.metadata_path is not None and result.metadata_path.exists()


def test_drone_evaluator_benchmarking(tmp_path: Path) -> None:
    """Integration test: Benchmark 3D drone environment using Evaluator."""
    env = make_env("drone", bounds=(30.0, 30.0, 15.0), max_steps=25, num_obstacles=2)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = Evaluator(algorithm=algo, env=env)
    metrics = evaluator.evaluate(num_episodes=4, deterministic=True, base_seed=100)

    assert metrics.episodes == 4
    assert isinstance(metrics.mean_reward, float)
    assert metrics.mean_episode_length == 25.0

    report_path = tmp_path / "drone_eval.json"
    evaluator.save_report(metrics, report_path)
    assert report_path.exists()
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_data["episodes"] == 4
    env.close()


def test_drone_obstacle_counts_scaling() -> None:
    """Verify drone environment handles varying obstacle counts (4, 6, 8)."""
    for count in [4, 6, 8]:
        env = DroneNavigation3DEnv(bounds=(30.0, 30.0, 15.0), num_obstacles=count)
        obs, info = env.reset(seed=42)
        assert obs.shape == (29,)
        assert len(env._obstacles) == count
        act = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(act)
        assert next_obs.shape == (29,)
        assert isinstance(reward, float)
        env.close()


def test_drone_random_rollout_lifecycle() -> None:
    """Verify complete episode rollout using random actions terminates or truncates cleanly."""
    env = DroneNavigation3DEnv(bounds=(25.0, 25.0, 12.0), max_steps=30, num_obstacles=3)
    obs, info = env.reset(seed=99)
    done = False
    step_count = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, term, trunc, step_info = env.step(action)
        step_count += 1
        done = term or trunc

    assert step_count <= 30
    assert "position" in step_info
    assert "distance_to_goal" in step_info
    env.close()
