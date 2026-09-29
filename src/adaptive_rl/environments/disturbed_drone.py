"""Gymnasium-compatible 3D drone navigation under environmental disturbances and constraints.

Extends DroneNavigation3DEnv with:
1. Atmospheric wind fields: steady prevailing wind, altitude gradient/shear, and
   stochastic Ornstein-Uhlenbeck (OU) gusts.
2. Optional moving dynamic spherical obstacles with boundary-bounce physics.
3. Hidden injected OU disturbance (disturbance_strength) perturbing dynamics via
   linear aerodynamic drag force.
4. Step-indexed disturbance/recovery event tracking (DisturbanceRecoveryTracker).
5. Configurable sensor realism (LiDAR Gaussian noise, ray dropout, minimum blind zone).
6. Strict Gymnasium API compliance with standard 29-dimensional observation space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adaptive_rl.environments.drone import (
    DroneNavigation3DEnv,
    ObstacleSphere3D,
    compute_lidar_3d_readings,
)


@dataclass
class WindState3D:
    """Instantaneous state of the 3D wind velocity vector."""

    steady: np.ndarray  # [wx, wy, wz] steady prevailing wind in m/s
    gust: np.ndarray  # [gx, gy, gz] stochastic turbulence component in m/s
    total: np.ndarray  # combined wind vector in m/s

    @property
    def speed(self) -> float:
        """Total instantaneous wind speed magnitude in m/s."""
        return float(np.linalg.norm(self.total))


class WindField3D:
    """Simulates 3D atmospheric wind fields with steady currents and stochastic gusts.

    Implements:
    1. Steady prevailing wind vector: w_steady = [wx, wy, wz].
    2. Altitude wind gradient: wind speed scales with altitude z.
    3. Ornstein-Uhlenbeck (OU) stochastic gust turbulence:
       dg = -theta * g * dt + sigma * sqrt(dt) * N(0, I)
    """

    def __init__(
        self,
        steady_wind: Tuple[float, float, float] = (1.5, 0.5, 0.0),
        gust_theta: float = 0.15,
        gust_sigma: float = 0.0,
        max_gust: float = 4.0,
        altitude_shear: float = 0.02,
        dt: float = 0.1,
    ) -> None:
        if gust_theta < 0.0:
            raise ValueError(f"gust_theta cannot be negative, got {gust_theta}")
        if gust_sigma < 0.0:
            raise ValueError(f"gust_sigma cannot be negative, got {gust_sigma}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.steady_wind = np.array(steady_wind, dtype=np.float64)
        self.gust_theta = float(gust_theta)
        self.gust_sigma = float(gust_sigma)
        self.max_gust = float(max_gust)
        self.altitude_shear = float(altitude_shear)
        self.dt = float(dt)

        self._current_gust = np.zeros(3, dtype=np.float64)

    def reset(self, initial_gust: Optional[np.ndarray] = None) -> None:
        """Reset the gust turbulence state to zero or specified initial vector."""
        if initial_gust is not None:
            self._current_gust = np.asarray(initial_gust, dtype=np.float64).copy()
        else:
            self._current_gust = np.zeros(3, dtype=np.float64)

    def step_gust(self, rng: np.random.Generator) -> np.ndarray:
        """Advance the stochastic Ornstein-Uhlenbeck gust turbulence by one time step dt.

        dg = -theta * g * dt + sigma * sqrt(dt) * noise
        """
        if self.gust_sigma > 0.0:
            noise = rng.normal(loc=0.0, scale=1.0, size=3)
            dg = (
                -self.gust_theta * self._current_gust * self.dt
                + self.gust_sigma * np.sqrt(self.dt) * noise
            )
            self._current_gust += dg

            # Clamp gust magnitude to prevent explosive divergence
            gust_mag = float(np.linalg.norm(self._current_gust))
            if gust_mag > self.max_gust:
                self._current_gust = (self._current_gust / gust_mag) * self.max_gust
        else:
            self._current_gust = np.zeros(3, dtype=np.float64)

        return self._current_gust.copy()

    def get_wind(self, position: np.ndarray) -> WindState3D:
        """Calculate total instantaneous 3D wind velocity at a specific drone position.

        w_total = w_steady * (1 + shear * z) + w_gust
        """
        z = max(0.0, float(position[2]))
        altitude_factor = 1.0 + self.altitude_shear * z

        scaled_steady = self.steady_wind * altitude_factor
        total_wind = scaled_steady + self._current_gust

        return WindState3D(
            steady=scaled_steady.copy(),
            gust=self._current_gust.copy(),
            total=total_wind.copy(),
        )


class DisturbanceRecoveryTracker:
    """Step-indexed disturbance/recovery event tracker (environment steps, never wall-clock).

    Tracks transient disturbance onset and recovery according to the contract:
    1. Disturbance event: ||gust + injected_disturbance|| >= threshold.
    2. Onset: first step magnitude crosses at-or-above threshold while calm.
    3. Recovery: magnitude subsides below threshold AND drone speed returns to within
       speed_tolerance of onset speed for hold_steps consecutive steps.
    4. Censoring: events still active when episode terminates are flagged as censored.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all recorded events and active-event state for a new episode."""
        self._events: List[Dict[str, Optional[int]]] = []
        self._active_onset: Optional[int] = None
        self._ref_speed: float = 0.0
        self._calm_streak: int = 0
        self._recovered_this_step: bool = False
        self._completed_this_step: Optional[int] = None

    def update(
        self,
        *,
        step: int,
        magnitude: float,
        speed: float,
        threshold: float,
        speed_tolerance: float,
        hold_steps: int,
    ) -> None:
        """Advance the tracker by one environment step."""
        self._recovered_this_step = False
        self._completed_this_step = None
        mag = float(magnitude)

        if self._active_onset is None:
            if threshold > 0.0 and mag >= threshold:
                self._active_onset = int(step)
                self._ref_speed = float(speed)
                self._calm_streak = 0
                self._events.append({"onset": int(step), "recovery": None})
            return

        calm = (mag < threshold) and (abs(float(speed) - self._ref_speed) <= speed_tolerance)
        if calm:
            self._calm_streak += 1
        else:
            self._calm_streak = 0

        if self._calm_streak >= max(1, int(hold_steps)):
            recovery_time = int(step) - int(self._active_onset)
            self._events[-1]["recovery"] = recovery_time
            self._completed_this_step = recovery_time
            self._recovered_this_step = True
            self._active_onset = None
            self._calm_streak = 0

    def telemetry(self) -> Dict[str, Any]:
        """Return the per-step telemetry contract consumed by evaluation code."""
        completed = [e["recovery"] for e in self._events if e["recovery"] is not None]
        recovery_times = [int(r) for r in completed if r is not None]
        return {
            "disturbance_active": self._active_onset is not None,
            "disturbance_onset_step": self._active_onset,
            "recovered": self._recovered_this_step,
            "recovery_time": self._completed_this_step,
            "recovery_events": len(self._events),
            "recovery_completed": len(completed),
            "recovery_open": 1 if self._active_onset is not None else 0,
            "recovery_censored": 1 if self._active_onset is not None else 0,
            "recovery_times": recovery_times,
            "mean_recovery_time": float(np.mean(recovery_times)) if recovery_times else None,
        }


@dataclass
class DynamicObstacleSphere3D:
    """Moving 3D spherical obstacle with velocity and boundary bounce physics."""

    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    radius: float = 2.0

    def step(self, dt: float, bounds: Tuple[float, float, float]) -> None:
        """Advance obstacle position and bounce off domain boundaries."""
        self.position = self.position + self.velocity * dt
        for i in range(3):
            if self.position[i] - self.radius < 0.0 and self.velocity[i] < 0.0:
                self.velocity[i] = -self.velocity[i]
                self.position[i] = self.radius
            elif self.position[i] + self.radius > bounds[i] and self.velocity[i] > 0.0:
                self.velocity[i] = -self.velocity[i]
                self.position[i] = bounds[i] - self.radius

    def distance_to(self, point: np.ndarray) -> float:
        """Compute Euclidean distance from the obstacle surface to an external 3D point."""
        d_center = float(np.linalg.norm(self.position - point))
        return max(0.0, d_center - self.radius)

    def collides_with(self, point: np.ndarray, radius: float) -> bool:
        """Check collision with an external sphere."""
        d_center = float(np.linalg.norm(self.position - point))
        return d_center <= (self.radius + radius)


def generate_dynamic_drone_obstacles(
    num_dynamic_obstacles: int,
    bounds: Tuple[float, float, float],
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_radius: float = 2.0,
    speed: float = 1.5,
    rng: Optional[np.random.Generator] = None,
) -> List[DynamicObstacleSphere3D]:
    """Procedurally place dynamic spherical obstacles away from start and goal."""
    if num_dynamic_obstacles <= 0:
        return []

    r = rng if rng is not None else np.random.default_rng()
    obstacles: List[DynamicObstacleSphere3D] = []
    margin = obstacle_radius + 1.0

    for _ in range(num_dynamic_obstacles):
        pos = np.array(
            [
                r.uniform(margin, bounds[0] - margin),
                r.uniform(margin, bounds[1] - margin),
                r.uniform(margin, bounds[2] - margin),
            ],
            dtype=np.float64,
        )
        direction = r.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0, 0.0])
        vel = direction * speed
        obstacles.append(
            DynamicObstacleSphere3D(position=pos, velocity=vel, radius=obstacle_radius)
        )

    return obstacles


class DroneDisturbed3DEnv(DroneNavigation3DEnv):
    """Gymnasium environment simulating 3D drone navigation under environmental disturbances.

    Extends DroneNavigation3DEnv with:
    - Steady wind force (c_d * w_steady) and Ornstein-Uhlenbeck stochastic wind gusts.
    - Procedural dynamic obstacles with boundary bouncing.
    - Hidden injected disturbance (disturbance_strength).
    - Disturbance recovery tracking.
    - Configurable LiDAR noise and dropout.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float] = (50.0, 50.0, 25.0),
        start_pos: Optional[Tuple[float, float, float] | np.ndarray] = None,
        goal_pos: Optional[Tuple[float, float, float] | np.ndarray] = None,
        num_obstacles: int = 8,
        num_dynamic_obstacles: int = 0,
        obstacle_radius: float = 2.0,
        dynamic_obstacle_speed: float = 1.5,
        target_radius: float = 1.5,
        collision_radius: float = 0.8,
        lidar_range: float = 20.0,
        num_lidar_rays: int = 16,
        lidar_noise_std: float = 0.0,
        lidar_dropout_prob: float = 0.0,
        lidar_min_range: float = 0.0,
        dt: float = 0.1,
        max_velocity: float = 8.0,
        max_acceleration: float = 4.0,
        linear_damping: float = 0.05,
        max_steps: int = 300,
        step_penalty: float = -0.05,
        goal_reward: float = 100.0,
        collision_reward: float = -100.0,
        progress_weight: float = 2.0,
        action_penalty_weight: float = 0.01,
        terminate_on_collision: bool = True,
        render_mode: Optional[str] = None,
        split: Optional[str] = None,
        # Environmental disturbance parameters
        wind_speed: Optional[float] = None,
        steady_wind: Tuple[float, float, float] = (1.5, 0.5, 0.0),
        gust_theta: float = 0.15,
        gust_sigma: float = 0.0,
        max_gust: float = 4.0,
        altitude_shear: float = 0.02,
        disturbance_strength: float = 0.0,
        disturbance_theta: float = 0.15,
        disturbance_event_threshold_override: Optional[float] = None,
        disturbance_event_factor: float = 1.5,
        recovery_speed_tolerance: float = 1.0,
        recovery_hold_steps: int = 5,
        wind_direction: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if wind_direction is not None:
            steady_wind = wind_direction
        super().__init__(
            bounds=bounds,
            start_pos=start_pos,
            goal_pos=goal_pos,
            num_obstacles=num_obstacles,
            obstacle_radius=obstacle_radius,
            target_radius=target_radius,
            collision_radius=collision_radius,
            lidar_range=lidar_range,
            num_lidar_rays=num_lidar_rays,
            lidar_noise_std=lidar_noise_std,
            lidar_dropout_prob=lidar_dropout_prob,
            lidar_min_range=lidar_min_range,
            dt=dt,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            linear_damping=linear_damping,
            max_steps=max_steps,
            step_penalty=step_penalty,
            goal_reward=goal_reward,
            collision_reward=collision_reward,
            progress_weight=progress_weight,
            action_penalty_weight=action_penalty_weight,
            terminate_on_collision=terminate_on_collision,
            render_mode=render_mode,
            split=split,
        )

        self.dt = float(dt)
        self.linear_damping = float(linear_damping)
        self.max_acceleration = float(max_acceleration)
        self.num_dynamic_obstacles = int(num_dynamic_obstacles)
        self.dynamic_obstacle_speed = float(dynamic_obstacle_speed)
        self._configured_wind_speed: Optional[float] = (
            None if wind_speed is None else float(wind_speed)
        )
        self.disturbance_strength = float(disturbance_strength)
        self.disturbance_theta = float(disturbance_theta)
        self.disturbance_event_threshold_override = (
            None
            if disturbance_event_threshold_override is None
            else float(disturbance_event_threshold_override)
        )
        self.disturbance_event_factor = float(disturbance_event_factor)
        self.recovery_speed_tolerance = float(recovery_speed_tolerance)
        self.recovery_hold_steps = int(recovery_hold_steps)

        self.wind_field = WindField3D(
            steady_wind=steady_wind,
            gust_theta=gust_theta,
            gust_sigma=gust_sigma,
            max_gust=max_gust,
            altitude_shear=altitude_shear,
            dt=self.dt,
        )
        if self._configured_wind_speed is not None:
            self._apply_wind_speed(self._configured_wind_speed)

        self._dynamic_obstacles: List[DynamicObstacleSphere3D] = []
        self._injected_disturbance = np.zeros(3, dtype=np.float64)
        self._last_disturbance_magnitude = 0.0
        self._recovery = DisturbanceRecoveryTracker()

    def _apply_wind_speed(self, value: float) -> None:
        """Rescale the steady wind vector to the requested magnitude (m/s)."""
        val = float(value)
        if val < 0.0:
            raise ValueError(f"wind_speed cannot be negative, got {value}")
        current = np.asarray(self.wind_field.steady_wind, dtype=np.float64)
        norm = float(np.linalg.norm(current))
        if norm > 0.0:
            direction = current / norm
        else:
            default = np.array([1.5, 0.5, 0.0], dtype=np.float64)
            direction = default / float(np.linalg.norm(default))
        self.wind_field.steady_wind = direction * val

    def _advance_disturbance(self) -> np.ndarray:
        """Advance the hidden injected OU disturbance by one time step dt."""
        if self.disturbance_strength > 0.0:
            if self.disturbance_theta > 0.0:
                sigma = self.disturbance_strength * math.sqrt(2.0 * self.disturbance_theta)
            else:
                sigma = self.disturbance_strength
            noise = self.np_random.normal(loc=0.0, scale=1.0, size=3)
            self._injected_disturbance += (
                -self.disturbance_theta * self._injected_disturbance * self.dt
                + sigma * math.sqrt(self.dt) * noise
            )
            cap = 4.0 * self.disturbance_strength
            mag = float(np.linalg.norm(self._injected_disturbance))
            if mag > cap:
                self._injected_disturbance = (self._injected_disturbance / mag) * cap
        return self._injected_disturbance

    @property
    def disturbance_event_threshold(self) -> float:
        """Magnitude threshold (m/s) at which a disturbance event starts."""
        if self.disturbance_event_threshold_override is not None:
            return float(self.disturbance_event_threshold_override)
        theta = float(self.wind_field.gust_theta)
        sigma = float(self.wind_field.gust_sigma)
        gust_stat = sigma / math.sqrt(2.0 * theta) if theta > 0.0 else sigma
        total_var = gust_stat * gust_stat + self.disturbance_strength * self.disturbance_strength
        if total_var <= 0.0:
            return 0.0
        return float(self.disturbance_event_factor * math.sqrt(3.0 * total_var))

    def get_effective_parameters(self) -> Dict[str, Any]:
        """Return the concrete parameter set governing current environment dynamics."""
        steady = [float(v) for v in list(np.asarray(self.wind_field.steady_wind).tolist())]
        return {
            "bounds": [float(b) for b in self.bounds],
            "num_obstacles": int(self.num_obstacles),
            "num_dynamic_obstacles": int(self.num_dynamic_obstacles),
            "obstacle_radius": float(self.obstacle_radius),
            "dynamic_obstacle_speed": float(self.dynamic_obstacle_speed),
            "steady_wind": steady,
            "wind_speed": float(np.linalg.norm(self.wind_field.steady_wind)),
            "gust_theta": float(self.wind_field.gust_theta),
            "gust_sigma": float(self.wind_field.gust_sigma),
            "disturbance_strength": float(self.disturbance_strength),
            "disturbance_event_threshold": float(self.disturbance_event_threshold),
            "lidar_noise_std": float(self.lidar_noise_std),
            "lidar_dropout_prob": float(self.lidar_dropout_prob),
            "lidar_min_range": float(self.lidar_min_range),
            "max_steps": int(self.max_steps),
        }

    def set_parameters(self, **kwargs: Any) -> None:
        """Dynamically update environment parameters for distribution shifts or curriculum."""
        for key, value in kwargs.items():
            if key == "wind_speed":
                if value is not None:
                    self._apply_wind_speed(float(value))
                    self._configured_wind_speed = float(value)
            elif key == "steady_wind":
                self.wind_field.steady_wind = np.array(value, dtype=np.float64)
            elif key == "gust_sigma":
                self.wind_field.gust_sigma = float(value)
            elif key == "gust_theta":
                self.wind_field.gust_theta = float(value)
            elif key == "num_obstacles":
                self.num_obstacles = int(value)
            elif key == "num_dynamic_obstacles":
                self.num_dynamic_obstacles = int(value)
            elif key == "obstacle_radius":
                self.obstacle_radius = float(value)
            elif key == "dynamic_obstacle_speed":
                self.dynamic_obstacle_speed = float(value)
            elif key == "disturbance_strength":
                if float(value) < 0.0:
                    raise ValueError(f"disturbance_strength cannot be negative, got {value}")
                self.disturbance_strength = float(value)
            elif key == "disturbance_event_threshold_override":
                self.disturbance_event_threshold_override = None if value is None else float(value)
            elif key == "lidar_noise_std":
                self.lidar_noise_std = float(value)
            elif key == "lidar_dropout_prob":
                self.lidar_dropout_prob = float(value)
            elif key == "lidar_min_range":
                self.lidar_min_range = float(value)
            elif key == "max_steps":
                self.max_steps = int(value)

    def _all_obstacles_as_spheres(self) -> List[ObstacleSphere3D]:
        """Combine static and dynamic obstacles as spherical obstacles for LiDAR/collision."""
        combined: List[ObstacleSphere3D] = list(self._obstacles)
        for dyn in self._dynamic_obstacles:
            combined.append(ObstacleSphere3D(center=dyn.position.copy(), radius=dyn.radius))
        return combined

    def _get_obs(self) -> np.ndarray:
        """Construct standard 29-dimensional observation vector."""
        bx, by, bz = self.bounds
        v_max = self.kinematics.max_velocity

        norm_pos = [self._position[0] / bx, self._position[1] / by, self._position[2] / bz]
        norm_vel = [
            self._velocity[0] / v_max,
            self._velocity[1] / v_max,
            self._velocity[2] / v_max,
        ]
        norm_goal = [self._goal[0] / bx, self._goal[1] / by, self._goal[2] / bz]
        rel_goal = [
            (self._goal[0] - self._position[0]) / bx,
            (self._goal[1] - self._position[1]) / by,
            (self._goal[2] - self._position[2]) / bz,
        ]
        curr_dist = float(np.linalg.norm(self._goal - self._position))
        norm_dist = [min(1.0, curr_dist / self.max_diagonal)]

        all_obs = self._all_obstacles_as_spheres()
        lidar_readings = compute_lidar_3d_readings(
            origin=self._position,
            ray_directions=self.lidar_rays,
            obstacles=all_obs,
            bounds=self.bounds,
            max_range=self.lidar_range,
            noise_std=self.lidar_noise_std,
            dropout_prob=self.lidar_dropout_prob,
            min_range=self.lidar_min_range,
            rng=self.np_random,
        )

        raw = np.concatenate([norm_pos, norm_vel, norm_goal, rel_goal, norm_dist, lidar_readings])
        return np.asarray(np.clip(raw, -1.0, 1.0), dtype=np.float32)

    def _check_collision(self, pos: np.ndarray) -> Tuple[bool, str]:
        """Check collision against boundaries and all static/dynamic obstacles."""
        r = self.collision_radius
        bx, by, bz = self.bounds

        if pos[0] - r <= 0.0 or pos[0] + r >= bx:
            return True, "boundary_x"
        if pos[1] - r <= 0.0 or pos[1] + r >= by:
            return True, "boundary_y"
        if pos[2] - r <= 0.0 or pos[2] + r >= bz:
            return True, "boundary_z"

        for obs in self._all_obstacles_as_spheres():
            if obs.collides_with(pos, r):
                return True, "obstacle"

        return False, "none"

    def _get_info(self) -> Dict[str, Any]:
        """Construct detailed telemetry dictionary with wind and recovery data."""
        info = super()._get_info()
        wind_state = self.wind_field.get_wind(self._position)
        info["wind_vector"] = wind_state.total.copy()
        info["wind_speed"] = float(wind_state.speed)
        info["gust_vector"] = wind_state.gust.copy()
        info["disturbance_magnitude"] = float(self._last_disturbance_magnitude)
        info["num_dynamic_obstacles"] = len(self._dynamic_obstacles)
        info.update(self._recovery.telemetry())
        return info

    def _wind_velocity(self, position: np.ndarray) -> np.ndarray:
        return self.wind_field.get_wind(position).total.copy()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment, obstacles, wind fields, and disturbance state."""
        super().reset(seed=seed, options=options)

        self.wind_field.reset()
        if self._configured_wind_speed is not None:
            self._apply_wind_speed(self._configured_wind_speed)

        self._injected_disturbance = np.zeros(3, dtype=np.float64)
        self._last_disturbance_magnitude = 0.0
        self._recovery.reset()

        if self.num_dynamic_obstacles > 0:
            self._dynamic_obstacles = generate_dynamic_drone_obstacles(
                num_dynamic_obstacles=self.num_dynamic_obstacles,
                bounds=self.bounds,
                start=self._position,
                goal=self._goal,
                obstacle_radius=self.obstacle_radius,
                speed=self.dynamic_obstacle_speed,
                rng=self.np_random,
            )
        else:
            self._dynamic_obstacles = []

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment with wind disturbances, dynamic obstacles, and recovery tracking."""
        act_arr = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(act_arr):
            act_arr = np.clip(act_arr, -1.0, 1.0)

        self._current_step += 1

        # 1. Advance stochastic gusts and injected disturbance
        self.wind_field.step_gust(self.np_random)
        self._advance_disturbance()
        wind_state = self.wind_field.get_wind(self._position)

        # 2. Advance dynamic obstacles
        for dyn in self._dynamic_obstacles:
            dyn.step(self.dt, self.bounds)

        # 3. Aerodynamic drag drift force: F_wind = linear_damping * (w_total + disturbance)
        # Net acceleration = a_cmd + linear_damping * (w_total + disturbance) - linear_damping * v
        raw_cmd_acc = act_arr * self.kinematics.max_acceleration
        effective_acc_cmd = raw_cmd_acc + self.linear_damping * (
            wind_state.total + self._injected_disturbance
        )

        new_pos, new_vel = self.kinematics.step(effective_acc_cmd)
        self._position = new_pos
        self._velocity = new_vel

        curr_distance = float(np.linalg.norm(self._goal - self._position))
        dist_delta = self._prev_distance_to_goal - curr_distance
        self._prev_distance_to_goal = curr_distance

        # 4. Update disturbance/recovery event tracker
        transient = wind_state.gust + self._injected_disturbance
        self._last_disturbance_magnitude = float(np.linalg.norm(transient))
        self._recovery.update(
            step=self._current_step,
            magnitude=self._last_disturbance_magnitude,
            speed=float(np.linalg.norm(self._velocity)),
            threshold=self.disturbance_event_threshold,
            speed_tolerance=self.recovery_speed_tolerance,
            hold_steps=self.recovery_hold_steps,
        )

        # 5. Check collision against boundaries, static, and dynamic obstacles
        is_collision, collision_type = self._check_collision(self._position)
        is_success = curr_distance <= self.target_radius

        terminated = False
        truncated = False
        info = self._get_info()
        info["collision"] = is_collision
        info["collision_type"] = collision_type
        info["success"] = is_success
        info["is_success"] = is_success

        if is_collision:
            reward = self.collision_reward
            info["success"] = False
            info["is_success"] = False
            if self.terminate_on_collision:
                terminated = True
        elif is_success:
            reward = self.goal_reward
            info["success"] = True
            info["is_success"] = True
            terminated = True
        else:
            action_effort = float(np.sum(np.square(act_arr)))
            progress_reward = self.progress_weight * dist_delta
            effort_penalty = self.action_penalty_weight * action_effort
            reward = float(progress_reward + self.step_penalty - effort_penalty)

        if self._current_step >= self.max_steps and not terminated:
            truncated = True

        info["terminated"] = terminated
        info["truncated"] = truncated
        info["TimeLimit.truncated"] = truncated

        if self.render_mode == "human":
            print(self.render())

        return self._get_obs(), float(reward), terminated, truncated, info


# Aliases for backwards compatibility
DroneDisturbance3DEnv = DroneDisturbed3DEnv

__all__ = [
    "DisturbanceRecoveryTracker",
    "DroneDisturbance3DEnv",
    "DroneDisturbed3DEnv",
    "DynamicObstacleSphere3D",
    "WindField3D",
    "WindState3D",
    "generate_dynamic_drone_obstacles",
]
