"""Kinematic 3D Drone Navigation Environment for AdaptiveRL.

This module provides a simulated 3D drone navigation environment featuring
point-mass 3-DOF translational kinematics with linear drag damping, procedural
spherical obstacles, 3D LiDAR rangefinding rays, and Gymnasium API compliance.

Mathematical Model & Architecture:
----------------------------------
1. Dynamics:
   Translational point-mass model integrated via semi-implicit Euler stepping:
     a_cmd = action * max_acceleration
     v_{t+1} = clamp(v_t + (a_cmd - linear_damping * v_t) * dt, speed <= max_velocity)
     p_{t+1} = p_t + v_{t+1} * dt

2. Action Space:
   Box(-1.0, 1.0, shape=(3,), dtype=float32)
   Continuous 3D acceleration command along [ax, ay, az].

3. Observation Space:
   Box(-1.0, 1.0, shape=(29,), dtype=float32)
   - [0:3]   : Normalized 3D position [x/X_max, y/Y_max, z/Z_max] in [0, 1]
   - [3:6]   : Normalized 3D velocity [vx/v_max, vy/v_max, vz/v_max] in [-1, 1]
   - [6:9]   : Normalized target position [gx/X_max, gy/Y_max, gz/Z_max] in [0, 1]
   - [9:12]  : Relative target vector [(gx - x)/X_max, (gy - y)/Y_max, (gz - z)/Z_max] in [-1, 1]
   - [12]    : Normalized distance to goal ||goal - pos|| / max_diagonal in [0, 1]
   - [13:29] : 16-ray 3D LiDAR distance rangefinder readings in [0, 1]

4. Reward Function:
   - Target reached (distance <= target_radius): +goal_reward (+100.0)
   - Collision (obstacle or arena boundary): +collision_reward (-100.0)
   - Step progress: progress_weight * (dist_prev - dist_curr)
   - Step time penalty: step_penalty (-0.05)
   - Control effort penalty: -action_penalty_weight * ||action||^2

5. Limitations:
   This is a kinematic simulation of 3D point-mass translation. It does NOT model
   quadrotor rotational attitude (roll/pitch/yaw), rotor aerodynamics, blade-flapping,
   ground effects, or motor torque/voltage curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from adaptive_rl.environments.base import AdaptiveRLEnv


@dataclass
class DroneState3D:
    """Represents the continuous 3D kinematic state of a drone."""

    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in meters/second
    acceleration: np.ndarray  # [ax, ay, az] in meters/second^2

    def copy(self) -> DroneState3D:
        """Create a deep copy of the current kinematic state."""
        return DroneState3D(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
        )


class DroneKinematics3D:
    """3D point-mass translational kinematic equations of motion.

    Simulates continuous 3D translation under commanded accelerations with linear
    aerodynamic drag damping and physical speed limits:
        dv/dt = a - c_d * v
        dp/dt = v
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_velocity: float = 8.0,
        max_acceleration: float = 4.0,
        linear_damping: float = 0.05,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if max_velocity <= 0.0:
            raise ValueError(f"max_velocity must be positive, got {max_velocity}")
        if max_acceleration <= 0.0:
            raise ValueError(f"max_acceleration must be positive, got {max_acceleration}")
        if linear_damping < 0.0:
            raise ValueError(f"linear_damping cannot be negative, got {linear_damping}")

        self.dt = float(dt)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.linear_damping = float(linear_damping)

        self.state = DroneState3D(
            position=np.zeros(3, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            acceleration=np.zeros(3, dtype=np.float64),
        )

    def reset(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
    ) -> DroneState3D:
        pos = np.asarray(position, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(f"Position must have shape (3,), got {pos.shape}")

        vel = (
            np.asarray(velocity, dtype=np.float64)
            if velocity is not None
            else np.zeros(3, dtype=np.float64)
        )
        if vel.shape != (3,):
            raise ValueError(f"Velocity must have shape (3,), got {vel.shape}")

        self.state = DroneState3D(
            position=pos.copy(),
            velocity=vel.copy(),
            acceleration=np.zeros(3, dtype=np.float64),
        )
        return self.state.copy()

    def step(self, action_acceleration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raw_acc = np.asarray(action_acceleration, dtype=np.float64)
        if raw_acc.shape != (3,):
            raise ValueError(f"Acceleration command must have shape (3,), got {raw_acc.shape}")
        clamped_acc = np.clip(raw_acc, -self.max_acceleration, self.max_acceleration)
        self.state.acceleration = clamped_acc

        effective_acc = clamped_acc - self.linear_damping * self.state.velocity
        new_velocity = self.state.velocity + effective_acc * self.dt

        speed = float(np.linalg.norm(new_velocity))
        if speed > self.max_velocity:
            new_velocity = (new_velocity / speed) * self.max_velocity

        new_position = self.state.position + new_velocity * self.dt
        self.state.velocity = new_velocity
        self.state.position = new_position

        return self.state.position.copy(), self.state.velocity.copy()


@dataclass
class ObstacleSphere3D:
    """A spherical 3D obstacle defined by center coordinates and radius."""

    center: np.ndarray  # [x, y, z] in meters
    radius: float  # radius in meters

    def distance_to(self, point: np.ndarray) -> float:
        """Compute surface distance from point to obstacle (negative inside)."""
        dist_to_center = float(np.linalg.norm(point - self.center))
        return dist_to_center - self.radius

    def collides_with(self, point: np.ndarray, collision_radius: float) -> bool:
        """Check if a spherical volume collides with this obstacle."""
        dist_to_center = float(np.linalg.norm(point - self.center))
        return dist_to_center <= (self.radius + collision_radius)


def ray_cast_sphere_3d(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    center: np.ndarray,
    radius: float,
    max_range: float = 20.0,
) -> float:
    """Compute ray intersection distance with a 3D sphere."""
    orig = np.asarray(ray_origin, dtype=np.float64)
    direction = np.asarray(ray_direction, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)

    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-9:
        return float(max_range)
    d = direction / dir_norm

    m = orig - c
    b = float(np.dot(m, d))
    c_const = float(np.dot(m, m)) - radius * radius

    if c_const <= 0.0:
        return 0.0

    if b > 0.0 and c_const > 0.0:
        return float(max_range)

    discriminant = b * b - c_const
    if discriminant < 0.0:
        return float(max_range)

    t = -b - float(np.sqrt(discriminant))
    if 0.0 <= t <= max_range:
        return float(t)
    return float(max_range)


def ray_cast_box_boundaries_3d(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    bounds: Tuple[float, float, float],
    max_range: float = 20.0,
) -> float:
    """Compute distance from inside a 3D bounding box to the nearest boundary."""
    orig = np.asarray(ray_origin, dtype=np.float64)
    direction = np.asarray(ray_direction, dtype=np.float64)

    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-9:
        return float(max_range)
    d = direction / dir_norm

    t_exit = float("inf")
    for axis in range(3):
        axis_dir = d[axis]
        if abs(axis_dir) > 1e-7:
            if axis_dir > 0.0:
                t = (bounds[axis] - orig[axis]) / axis_dir
            else:
                t = (0.0 - orig[axis]) / axis_dir
            if 0.0 <= t < t_exit:
                t_exit = t

    return float(min(t_exit, max_range))


def generate_lidar_3d_ray_directions(num_rays: int = 16) -> np.ndarray:
    """Generate unit vector directions for 3D rangefinder rays.

    Default 16 rays:
    - 8 horizontal rays (elevation 0 deg, azimuth evenly spaced [0..315] deg)
    - 4 upper hemispheric rays (elevation +45 deg, azimuth [0, 90, 180, 270] deg)
    - 4 lower hemispheric rays (elevation -45 deg, azimuth [0, 90, 180, 270] deg)
    """
    if num_rays == 16:
        rays: List[List[float]] = []
        for h_deg in np.linspace(0.0, 315.0, 8):
            az_h = float(np.radians(float(h_deg)))
            rays.append([float(np.cos(az_h)), float(np.sin(az_h)), 0.0])

        el_up = float(np.radians(45.0))
        for u_deg in [0.0, 90.0, 180.0, 270.0]:
            az_u = float(np.radians(u_deg))
            rays.append(
                [
                    float(np.cos(az_u) * np.cos(el_up)),
                    float(np.sin(az_u) * np.cos(el_up)),
                    float(np.sin(el_up)),
                ]
            )

        el_down = float(np.radians(-45.0))
        for d_deg in [0.0, 90.0, 180.0, 270.0]:
            az_d = float(np.radians(d_deg))
            rays.append(
                [
                    float(np.cos(az_d) * np.cos(el_down)),
                    float(np.sin(az_d) * np.cos(el_down)),
                    float(np.sin(el_down)),
                ]
            )

        arr = np.array(rays, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return (arr / norms).astype(np.float32)

    indices = np.arange(0, num_rays, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / num_rays)
    theta = np.pi * (1.0 + 5.0**0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def compute_lidar_3d_readings(
    origin: np.ndarray,
    ray_directions: np.ndarray,
    obstacles: List[ObstacleSphere3D],
    bounds: Tuple[float, float, float],
    max_range: float = 20.0,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    min_range: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute normalized LiDAR range readings along 3D ray directions with optional noise/dropout.

    Args:
        origin: 3D point of sensor origin [x, y, z].
        ray_directions: Array of normalized unit vectors for each beam.
        obstacles: List of spherical obstacles.
        bounds: Arena bounding box (X_max, Y_max, Z_max).
        max_range: Maximum sensor detection range in meters.
        noise_std: Standard deviation of zero-mean Gaussian measurement noise in meters.
        dropout_prob: Probability in [0, 1] that an individual beam fails to return an echo.
                      Dropped beams report max_range (normalized 1.0).
        min_range: Minimum detectable distance in meters (blind zone). Distances below
                   min_range are clamped to min_range.
        rng: Optional seeded NumPy random generator for stochastic noise and dropout.
    """
    orig = np.asarray(origin, dtype=np.float64)
    readings: List[float] = []

    for ray_dir in ray_directions:
        min_dist = ray_cast_box_boundaries_3d(orig, ray_dir, bounds, max_range=max_range)
        for obs in obstacles:
            dist = ray_cast_sphere_3d(orig, ray_dir, obs.center, obs.radius, max_range=max_range)
            if dist < min_dist:
                min_dist = dist

        # Apply stochastic sensor model if configured
        if dropout_prob > 0.0 and rng is not None:
            if float(rng.uniform(0.0, 1.0)) < dropout_prob:
                readings.append(1.0)
                continue

        final_dist = min_dist
        if noise_std > 0.0 and rng is not None:
            noise = float(rng.normal(0.0, noise_std))
            final_dist += noise

        if min_range > 0.0:
            final_dist = max(min_range, final_dist)

        norm_reading = min(1.0, max(0.0, final_dist / max_range))
        readings.append(norm_reading)

    return np.asarray(readings, dtype=np.float32)


def generate_drone_obstacles(
    bounds: Tuple[float, float, float],
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    num_obstacles: int = 8,
    obstacle_radius: float = 2.0,
    clearance_radius: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> List[ObstacleSphere3D]:
    """Procedurally place spherical obstacles ensuring clearance from start and goal."""
    if rng is None:
        rng = np.random.default_rng()

    obstacles: List[ObstacleSphere3D] = []
    start = np.asarray(start_pos, dtype=np.float64)
    goal = np.asarray(goal_pos, dtype=np.float64)

    min_x, max_x = obstacle_radius + 1.0, bounds[0] - obstacle_radius - 1.0
    min_y, max_y = obstacle_radius + 1.0, bounds[1] - obstacle_radius - 1.0
    min_z, max_z = obstacle_radius + 1.0, bounds[2] - obstacle_radius - 1.0

    min_x, max_x = min(min_x, max_x), max(min_x, max_x)
    min_y, max_y = min(min_y, max_y), max(min_y, max_y)
    min_z, max_z = min(min_z, max_z), max(min_z, max_z)

    attempts = 0
    max_attempts = num_obstacles * 100

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        candidate = np.array(
            [
                float(rng.uniform(min_x, max_x)),
                float(rng.uniform(min_y, max_y)),
                float(rng.uniform(min_z, max_z)),
            ],
            dtype=np.float64,
        )

        if float(np.linalg.norm(candidate - start)) < (obstacle_radius + clearance_radius):
            continue
        if float(np.linalg.norm(candidate - goal)) < (obstacle_radius + clearance_radius):
            continue

        overlap = False
        for existing in obstacles:
            if float(np.linalg.norm(candidate - existing.center)) < (
                obstacle_radius + existing.radius + 1.0
            ):
                overlap = True
                break

        if not overlap:
            obstacles.append(ObstacleSphere3D(center=candidate, radius=obstacle_radius))

    return obstacles


class DroneNavigation3DEnv(AdaptiveRLEnv[np.ndarray, np.ndarray]):
    """Gymnasium environment simulating kinematic 3D drone navigation.

    Translational motion is governed by commanded accelerations, velocity caps,
    and linear aerodynamic drag. The agent senses its 3D environment through
    normalized relative coordinates and a 16-ray 3D LiDAR rangefinder.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0),
        start_pos: Optional[Tuple[float, float, float] | np.ndarray] = None,
        goal_pos: Optional[Tuple[float, float, float] | np.ndarray] = None,
        num_obstacles: int = 4,
        obstacle_radius: float = 2.0,
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
        max_steps: int = 200,
        step_penalty: float = -0.05,
        goal_reward: float = 100.0,
        collision_reward: float = -100.0,
        progress_weight: float = 2.0,
        action_penalty_weight: float = 0.01,
        terminate_on_collision: bool = True,
        render_mode: Optional[str] = None,
        split: Optional[str] = None,
        wind_speed: float = 0.0,
        gust_sigma: float = 0.0,
        gust_theta: float = 0.15,
        wind_direction: Tuple[float, float, float] = (1.5, 0.5, 0.0),
    ) -> None:
        super().__init__()
        if any(b <= 0.0 for b in bounds):
            raise ValueError(f"Bounds must be strictly positive, got {bounds}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if target_radius <= 0.0:
            raise ValueError(f"target_radius must be positive, got {target_radius}")
        if collision_radius <= 0.0:
            raise ValueError(f"collision_radius must be positive, got {collision_radius}")
        if not np.isfinite(wind_speed) or wind_speed < 0.0:
            raise ValueError(f"wind_speed must be finite and non-negative, got {wind_speed}")
        if not np.isfinite(gust_sigma) or gust_sigma < 0.0:
            raise ValueError(f"gust_sigma must be finite and non-negative, got {gust_sigma}")
        if not np.isfinite(gust_theta) or gust_theta <= 0.0:
            raise ValueError(f"gust_theta must be finite and positive, got {gust_theta}")
        direction = np.asarray(wind_direction, dtype=np.float64)
        if (
            direction.shape != (3,)
            or not np.isfinite(direction).all()
            or np.linalg.norm(direction) == 0
        ):
            raise ValueError("wind_direction must be a finite, non-zero 3-vector")
        if lidar_noise_std < 0.0:
            raise ValueError(f"lidar_noise_std cannot be negative, got {lidar_noise_std}")
        if not (0.0 <= lidar_dropout_prob <= 1.0):
            raise ValueError(
                f"lidar_dropout_prob must be between 0.0 and 1.0, got {lidar_dropout_prob}"
            )
        if not (0.0 <= lidar_min_range < lidar_range):
            raise ValueError(
                f"lidar_min_range must be >= 0.0 and < lidar_range ({lidar_range}), got {lidar_min_range}"
            )

        self.split: Optional[str] = None
        if split is not None:
            clean_split = str(split).lower().strip()
            from adaptive_rl.evaluation.generalization import VALID_SPLITS

            if clean_split not in VALID_SPLITS:
                raise ValueError(f"Invalid split '{split}'. Expected one of: {VALID_SPLITS}")
            self.split = clean_split

        self._active_split: Optional[str] = self.split
        self._split_episode_index: int = 0
        self._last_split_seed: Optional[int] = None

        self.bounds = (float(bounds[0]), float(bounds[1]), float(bounds[2]))
        self.default_start = (
            np.asarray(start_pos, dtype=np.float64)
            if start_pos is not None
            else np.array([5.0, 5.0, 5.0], dtype=np.float64)
        )
        self.default_goal = (
            np.asarray(goal_pos, dtype=np.float64)
            if goal_pos is not None
            else np.array(
                [self.bounds[0] - 5.0, self.bounds[1] - 5.0, self.bounds[2] - 5.0],
                dtype=np.float64,
            )
        )
        self.num_obstacles = num_obstacles
        self.obstacle_radius = float(obstacle_radius)
        self.target_radius = float(target_radius)
        self.collision_radius = float(collision_radius)
        self.lidar_range = float(lidar_range)
        self.num_lidar_rays = num_lidar_rays
        self.lidar_noise_std = float(lidar_noise_std)
        self.lidar_dropout_prob = float(lidar_dropout_prob)
        self.lidar_min_range = float(lidar_min_range)
        self.max_steps = max_steps
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.collision_reward = float(collision_reward)
        self.progress_weight = float(progress_weight)
        self.action_penalty_weight = float(action_penalty_weight)
        self.terminate_on_collision = terminate_on_collision
        self.render_mode = render_mode
        self.wind_speed = float(wind_speed)
        self.gust_sigma = float(gust_sigma)
        self.gust_theta = float(gust_theta)
        self.wind_direction = direction / np.linalg.norm(direction)
        self._gust_velocity = np.zeros(3, dtype=np.float64)

        self.max_diagonal = float(np.linalg.norm(self.bounds))

        # Farama Gymnasium spaces: continuous 3D acceleration in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # Observation space: 29 continuous values in [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(13 + self.num_lidar_rays,),
            dtype=np.float32,
        )

        self.lidar_rays = generate_lidar_3d_ray_directions(num_rays=self.num_lidar_rays)
        self.kinematics = DroneKinematics3D(
            dt=dt,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            linear_damping=linear_damping,
        )

        self._position: np.ndarray = self.default_start.copy()
        self._velocity: np.ndarray = np.zeros(3, dtype=np.float64)
        self._goal: np.ndarray = self.default_goal.copy()
        self._obstacles: List[ObstacleSphere3D] = []
        self._current_step = 0
        self._prev_distance_to_goal: float = float(np.linalg.norm(self._goal - self._position))

    def _get_obs(self) -> np.ndarray:
        bx, by, bz = self.bounds
        v_max = self.kinematics.max_velocity

        norm_pos = [self._position[0] / bx, self._position[1] / by, self._position[2] / bz]
        norm_vel = [self._velocity[0] / v_max, self._velocity[1] / v_max, self._velocity[2] / v_max]
        norm_goal = [self._goal[0] / bx, self._goal[1] / by, self._goal[2] / bz]
        rel_goal = [
            (self._goal[0] - self._position[0]) / bx,
            (self._goal[1] - self._position[1]) / by,
            (self._goal[2] - self._position[2]) / bz,
        ]
        curr_dist = float(np.linalg.norm(self._goal - self._position))
        norm_dist = [min(1.0, curr_dist / self.max_diagonal)]

        lidar_readings = compute_lidar_3d_readings(
            origin=self._position,
            ray_directions=self.lidar_rays,
            obstacles=self._obstacles,
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
        r = self.collision_radius
        bx, by, bz = self.bounds

        if pos[0] - r <= 0.0 or pos[0] + r >= bx:
            return True, "boundary_x"
        if pos[1] - r <= 0.0 or pos[1] + r >= by:
            return True, "boundary_y"
        if pos[2] - r <= 0.0 or pos[2] + r >= bz:
            return True, "boundary_z"

        for obs in self._obstacles:
            if obs.collides_with(pos, r):
                return True, "obstacle"

        return False, "none"

    def _get_info(self) -> Dict[str, Any]:
        dist = float(np.linalg.norm(self._goal - self._position))
        speed = float(np.linalg.norm(self._velocity))

        min_obs_dist = float("inf")
        for obs in self._obstacles:
            d = obs.distance_to(self._position) - self.collision_radius
            if d < min_obs_dist:
                min_obs_dist = d

        info = {
            "step": self._current_step,
            "max_steps": self.max_steps,
            "position": self._position.copy(),
            "velocity": self._velocity.copy(),
            "acceleration": self.kinematics.state.acceleration.copy(),
            "speed": speed,
            "goal": self._goal.copy(),
            "distance_to_goal": dist,
            "num_obstacles": len(self._obstacles),
            "min_obstacle_distance": min_obs_dist if self._obstacles else float("inf"),
            "altitude": float(self._position[2]),
            "wind_velocity": self._wind_velocity(self._position).copy(),
            "lidar_noise_std": self.lidar_noise_std,
            "lidar_dropout_prob": self.lidar_dropout_prob,
            "lidar_min_range": self.lidar_min_range,
        }
        current_split = self._active_split if self._active_split is not None else self.split
        if current_split is not None:
            info["split"] = current_split
            info["split_seed"] = self._last_split_seed
        return info

    def _wind_velocity(self, position: np.ndarray) -> np.ndarray:
        altitude_factor = 1.0 + 0.02 * max(0.0, float(position[2]))
        return self.wind_direction * self.wind_speed * altitude_factor + self._gust_velocity

    def get_effective_parameters(self) -> Dict[str, Any]:
        """Return the physical environment parameters used by this instance."""
        return {
            "bounds": list(self.bounds),
            "num_obstacles": self.num_obstacles,
            "obstacle_radius": self.obstacle_radius,
            "wind_speed": self.wind_speed,
            "gust_sigma": self.gust_sigma,
            "gust_theta": self.gust_theta,
            "wind_direction": self.wind_direction.tolist(),
            "wind_altitude_shear": 0.02,
            "max_gust": 4.0,
            "linear_damping": self.kinematics.linear_damping,
            "max_acceleration": self.kinematics.max_acceleration,
            "max_steps": self.max_steps,
        }

    @property
    def drone_state(self) -> DroneState3D:
        """Convenience property exposing the underlying drone state."""
        return self.kinematics.state

    @property
    def target(self) -> np.ndarray:
        """Convenience property exposing the target waypoint."""
        return self._goal

    @property
    def obstacles(self) -> List[ObstacleSphere3D]:
        """Public read-only snapshot of the active obstacle set.

        Evaluation and visualization code should use this telemetry
        interface instead of the private ``_obstacles`` attribute. The list
        is copied so callers cannot mutate environment state.
        """
        return list(self._obstacles)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        active_split = self.split
        if options and "split" in options and options["split"] is not None:
            clean_split = str(options["split"]).lower().strip()
            from adaptive_rl.evaluation.generalization import VALID_SPLITS

            if clean_split not in VALID_SPLITS:
                raise ValueError(
                    f"Invalid split '{options['split']}'. Expected one of: {VALID_SPLITS}"
                )
            active_split = clean_split
        self._active_split = active_split

        effective_seed = seed
        if active_split is not None:
            from adaptive_rl.evaluation.generalization import (
                TEST_SEED_END,
                TEST_SEED_START,
                TRAIN_SEED_END,
                TRAIN_SEED_START,
                validate_split_seed,
            )

            if effective_seed is not None:
                validate_split_seed(effective_seed, active_split)
            else:
                if active_split == "train":
                    capacity = TRAIN_SEED_END - TRAIN_SEED_START
                    effective_seed = TRAIN_SEED_START + (self._split_episode_index % capacity)
                else:
                    capacity = TEST_SEED_END - TEST_SEED_START
                    effective_seed = TEST_SEED_START + (self._split_episode_index % capacity)
                self._split_episode_index += 1

        self._last_split_seed = effective_seed
        super().reset(seed=effective_seed, options=options)
        self._current_step = 0

        num_obs = self.num_obstacles
        if options and "num_obstacles" in options:
            num_obs = options["num_obstacles"]

        if options and "lidar_noise_std" in options:
            n_std = float(options["lidar_noise_std"])
            if n_std < 0.0:
                raise ValueError(f"lidar_noise_std cannot be negative, got {n_std}")
            self.lidar_noise_std = n_std
        if options and "lidar_dropout_prob" in options:
            d_prob = float(options["lidar_dropout_prob"])
            if not (0.0 <= d_prob <= 1.0):
                raise ValueError(f"lidar_dropout_prob must be between 0.0 and 1.0, got {d_prob}")
            self.lidar_dropout_prob = d_prob
        if options and "lidar_min_range" in options:
            m_range = float(options["lidar_min_range"])
            if not (0.0 <= m_range < self.lidar_range):
                raise ValueError(
                    f"lidar_min_range must be >= 0.0 and < lidar_range ({self.lidar_range}), got {m_range}"
                )
            self.lidar_min_range = m_range

        self._position = self.default_start.copy()
        self._velocity = np.zeros(3, dtype=np.float64)
        self._goal = self.default_goal.copy()
        self._gust_velocity = np.zeros(3, dtype=np.float64)

        self.kinematics.reset(self._position, self._velocity)

        self._obstacles = generate_drone_obstacles(
            bounds=self.bounds,
            start_pos=self._position,
            goal_pos=self._goal,
            num_obstacles=num_obs,
            obstacle_radius=self.obstacle_radius,
            clearance_radius=self.collision_radius + 2.0,
            rng=self.np_random,
        )

        self._prev_distance_to_goal = float(np.linalg.norm(self._goal - self._position))

        if self.render_mode == "human":
            print(self.render())

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act_arr = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(act_arr):
            # Clip gently to handle float precision issues
            act_arr = np.clip(act_arr, -1.0, 1.0)

        self._current_step += 1

        # Exact discrete OU update: stationary per-axis gust standard deviation
        # approaches sigma/sqrt(2*theta), as specified by the shift config.
        if self.gust_sigma > 0.0:
            noise = self.np_random.normal(size=3)
            self._gust_velocity += (
                -self.gust_theta * self._gust_velocity * self.kinematics.dt
                + self.gust_sigma * np.sqrt(self.kinematics.dt) * noise
            )
            gust_magnitude = float(np.linalg.norm(self._gust_velocity))
            if gust_magnitude > 4.0:
                self._gust_velocity *= 4.0 / gust_magnitude
        wind_acceleration = self.kinematics.linear_damping * self._wind_velocity(self._position)
        acc_command = act_arr * self.kinematics.max_acceleration + wind_acceleration
        new_pos, new_vel = self.kinematics.step(acc_command)
        self._position = new_pos
        self._velocity = new_vel

        curr_distance = float(np.linalg.norm(self._goal - self._position))
        dist_delta = self._prev_distance_to_goal - curr_distance
        self._prev_distance_to_goal = curr_distance

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

    def render(self) -> Optional[str]:
        dist = float(np.linalg.norm(self._goal - self._position))
        speed = float(np.linalg.norm(self._velocity))
        pos_str = f"[{self._position[0]:5.1f}, {self._position[1]:5.1f}, {self._position[2]:5.1f}]"
        vel_str = f"[{self._velocity[0]:5.1f}, {self._velocity[1]:5.1f}, {self._velocity[2]:5.1f}]"
        goal_str = f"[{self._goal[0]:5.1f}, {self._goal[1]:5.1f}, {self._goal[2]:5.1f}]"

        lines = [
            "+----------------------------------------------------------------+",
            "|               SIMULATED 3D DRONE FLIGHT DECK                   |",
            "+----------------------------------------------------------------+",
            f"| Step: {self._current_step:03d}/{self.max_steps:03d} | Altitude (Z): {self._position[2]:5.1f}m | Speed: {speed:4.1f} m/s          |",
            f"| Position [X, Y, Z]: {pos_str:<28} |",
            f"| Velocity [Vx,Vy,Vz]: {vel_str:<28} |",
            f"| Waypoint [Gx,Gy,Gz]: {goal_str:<28} |",
            f"| Range to Target: {dist:5.1f}m | Obstacles in Arena: {len(self._obstacles):02d}                |",
            "+----------------------------------------------------------------+",
            f"  Arena Boundaries: [0..{self.bounds[0]:.0f}, 0..{self.bounds[1]:.0f}, 0..{self.bounds[2]:.0f}] m",
            "+----------------------------------------------------------------+",
        ]
        dashboard = "\n".join(lines)

        if self.render_mode == "human":
            print(dashboard)
            return None
        return dashboard
