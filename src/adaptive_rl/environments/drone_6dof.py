"""Rigid-body 6-DOF Quadrotor Dynamics Simulation Environment for AdaptiveRL.

This module implements a theoretical rigid-body 6-DOF quadrotor flight dynamics
model with quaternion attitude kinematics, Euler rotational equations, aerodynamic
drag, motor thrust mapping, procedural 3D spherical obstacles, and 16-ray LiDAR.

Coordinate Frame Conventions:
-----------------------------
1. World Frame: ENU (East-North-Up) Cartesian coordinate system:
   - +X_W: East
   - +Y_W: North
   - +Z_W: Upward (gravity g = 9.81 m/s^2 acts along -Z_W)

2. Body Frame: Fixed to the quadrotor center of mass:
   - +X_B: Forward
   - +Y_B: Left
   - +Z_B: Upward (aligned with the collective thrust vector)

3. Attitude Representation:
   Unit quaternion q = [q_w, q_x, q_y, q_z] representing the rotation from body
   frame to world frame: v_W = R(q) * v_B.
   Quaternion kinematics:
     dot(q) = 0.5 * q (x) [0, omega]
   where (x) is the quaternion product and omega = [p, q, r] is the body angular velocity.

4. Rotor Configuration:
   Quadrotor "X" configuration with arm length L = 0.2 m (d = L / sqrt(2)):
   - Rotor 1: Front-Right (+d, -d, 0), CCW rotation (reaction torque: +c_tau along +Z_B)
   - Rotor 2: Front-Left  (+d, +d, 0), CW rotation  (reaction torque: -c_tau along +Z_B)
   - Rotor 3: Rear-Left   (-d, +d, 0), CCW rotation (reaction torque: +c_tau along +Z_B)
   - Rotor 4: Rear-Right  (-d, -d, 0), CW rotation  (reaction torque: -c_tau along +Z_B)

   Thrust & Torques:
     T_total = F1 + F2 + F3 + F4 (along +Z_B)
     tau_x = d * (-F1 + F2 + F3 - F4)  [Roll torque]
     tau_y = d * (-F1 - F2 + F3 + F4)  [Pitch torque]
     tau_z = c_tau * (+F1 - F2 + F3 - F4) [Yaw reaction torque]

Equations of Motion:
--------------------
- Translational acceleration:
    m * ddot(p) = [0, 0, -m*g]^T + R(q) * [0, 0, T_total]^T - D_v * v
- Rotational angular acceleration:
    I * dot(omega) = tau - omega x (I * omega) - D_omega * omega
where I = diag(Ixx, Iyy, Izz) is the rotational inertia tensor, D_v is linear translational
drag, and D_omega is rotational damping.

Numerical Integration:
----------------------
Runge-Kutta 4th-order (RK4) integration with substeps:
Environment step dt = 0.05 s, with 5 substeps of dt_sub = 0.01 s.
Quaternion normalization is enforced after every substep to prevent numerical drift.

NOTE:
-----
This is a theoretical, educational rigid-body quadrotor simulation model. It has not
been validated against physical hardware telemetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from adaptive_rl.environments.base import AdaptiveRLEnv
from adaptive_rl.environments.drone import (
    ObstacleSphere3D,
    compute_lidar_3d_readings,
    generate_drone_obstacles,
    generate_lidar_3d_ray_directions,
)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix (body to world)."""
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # Normalize for safety
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm > 1e-12:
        qw /= norm
        qx /= norm
        qy /= norm
        qz /= norm
    else:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qw * qx),
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Compute time derivative of quaternion given body angular velocity omega [p, q, r]."""
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    p, q_rate, r = float(omega[0]), float(omega[1]), float(omega[2])

    dq_w = 0.5 * (-qx * p - qy * q_rate - qz * r)
    dq_x = 0.5 * (qw * p + qy * r - qz * q_rate)
    dq_y = 0.5 * (qw * q_rate - qx * r + qz * p)
    dq_z = 0.5 * (qw * r + qx * q_rate - qy * p)

    return np.array([dq_w, dq_x, dq_y, dq_z], dtype=np.float64)


@dataclass
class DroneState6DOF:
    """Represents the complete rigid-body 6-DOF state of a quadrotor."""

    position: np.ndarray  # [x, y, z] in meters (world frame)
    velocity: np.ndarray  # [vx, vy, vz] in m/s (world frame)
    quaternion: np.ndarray  # [qw, qx, qy, qz] attitude (body to world)
    angular_velocity: np.ndarray  # [p, q, r] in rad/s (body frame)

    def copy(self) -> DroneState6DOF:
        """Return a deep copy of the 6-DOF state."""
        return DroneState6DOF(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )


class DroneDynamics6DOF:
    """Rigid-body 6-DOF quadrotor flight dynamics with Runge-Kutta 4 integration.

    Implements equations of motion for translation (world frame) and rotation (body frame)
    under collective motor thrusts, body torques, gravity, and aerodynamic damping.
    """

    def __init__(
        self,
        mass: float = 1.0,
        arm_length: float = 0.2,
        inertia: Optional[Tuple[float, float, float]] = None,
        gravity: float = 9.81,
        max_thrust_per_motor: float = 5.0,
        torque_to_thrust_ratio: float = 0.01,
        linear_damping: float = 0.1,
        angular_damping: float = 0.02,
        dt: float = 0.05,
        substeps: int = 5,
    ) -> None:
        if mass <= 0.0:
            raise ValueError(f"Mass must be positive, got {mass}")
        if arm_length <= 0.0:
            raise ValueError(f"Arm length must be positive, got {arm_length}")
        if max_thrust_per_motor <= 0.0:
            raise ValueError(f"max_thrust_per_motor must be positive, got {max_thrust_per_motor}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")

        self.mass = float(mass)
        self.arm_length = float(arm_length)
        self.d = self.arm_length / math.sqrt(2.0)  # distance along X and Y axes
        self.gravity = float(gravity)
        self.max_thrust_per_motor = float(max_thrust_per_motor)
        self.c_tau = float(torque_to_thrust_ratio)
        self.linear_damping = float(linear_damping)
        self.angular_damping = float(angular_damping)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.substep_dt = self.dt / self.substeps

        # Moments of inertia (diagonal tensor)
        if inertia is None:
            self.inertia_diag = np.array([0.01, 0.01, 0.02], dtype=np.float64)
        else:
            self.inertia_diag = np.array(
                [float(inertia[0]), float(inertia[1]), float(inertia[2])], dtype=np.float64
            )
            if any(val <= 0.0 for val in self.inertia_diag):
                raise ValueError(f"Inertia components must be strictly positive, got {inertia}")

        self.inv_inertia = 1.0 / self.inertia_diag
        self.hover_thrust_per_motor = (self.mass * self.gravity) / 4.0

        self.state = DroneState6DOF(
            position=np.zeros(3, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
        )

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        quaternion: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
    ) -> None:
        """Reset the internal physical state."""
        self.state.position = (
            np.asarray(position, dtype=np.float64).copy()
            if position is not None
            else np.zeros(3, dtype=np.float64)
        )
        self.state.velocity = (
            np.asarray(velocity, dtype=np.float64).copy()
            if velocity is not None
            else np.zeros(3, dtype=np.float64)
        )
        if quaternion is not None:
            q = np.asarray(quaternion, dtype=np.float64)
            q_norm = np.linalg.norm(q)
            self.state.quaternion = (
                (q / q_norm).copy() if q_norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])
            )
        else:
            self.state.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.state.angular_velocity = (
            np.asarray(angular_velocity, dtype=np.float64).copy()
            if angular_velocity is not None
            else np.zeros(3, dtype=np.float64)
        )

    def compute_forces_and_torques(self, motor_thrusts: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute collective thrust and body torques from 4 individual rotor thrusts.

        Rotor configuration (X-quadrotor):
          F1: Front-Right (+d, -d) CCW
          F2: Front-Left  (+d, +d) CW
          F3: Rear-Left   (-d, +d) CCW
          F4: Rear-Right  (-d, -d) CW

        Returns:
            (total_thrust, torques_vector [tau_x, tau_y, tau_z])
        """
        f = np.clip(np.asarray(motor_thrusts, dtype=np.float64), 0.0, self.max_thrust_per_motor)
        f1, f2, f3, f4 = f[0], f[1], f[2], f[3]

        total_thrust = float(f1 + f2 + f3 + f4)
        tau_x = float(self.d * (-f1 + f2 + f3 - f4))
        tau_y = float(self.d * (-f1 - f2 + f3 + f4))
        tau_z = float(self.c_tau * (+f1 - f2 + f3 - f4))

        return total_thrust, np.array([tau_x, tau_y, tau_z], dtype=np.float64)

    def _state_derivatives(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        total_thrust: float,
        torques: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute time derivatives [d_pos, d_vel, d_quat, d_omega]."""
        # 1. Position derivative: dp/dt = vel
        d_pos = vel.copy()

        # 2. Velocity derivative: dv/dt = [0, 0, -g] + (1/m)*R(q)*[0, 0, T] - (D_v/m)*v
        r_mat = quaternion_to_rotation_matrix(quat)
        thrust_world = r_mat @ np.array([0.0, 0.0, total_thrust], dtype=np.float64)
        gravity_world = np.array([0.0, 0.0, -self.mass * self.gravity], dtype=np.float64)
        drag_force = -self.linear_damping * vel
        d_vel = (gravity_world + thrust_world + drag_force) / self.mass

        # 3. Quaternion derivative: dq/dt = 0.5 * q (x) [0, omega]
        d_quat = quaternion_derivative(quat, omega)

        # 4. Angular velocity derivative: I * d_omega = tau - omega x (I * omega) - D_omega * omega
        i_omega = self.inertia_diag * omega
        gyroscopic = np.cross(omega, i_omega)
        angular_damping_torque = self.angular_damping * omega
        d_omega = self.inv_inertia * (torques - gyroscopic - angular_damping_torque)

        return d_pos, d_vel, d_quat, d_omega

    def step_rk4(self, motor_thrusts: np.ndarray) -> DroneState6DOF:
        """Advance physical state by dt using Runge-Kutta 4th-order integration over substeps."""
        total_thrust, torques = self.compute_forces_and_torques(motor_thrusts)
        h = self.substep_dt

        pos = self.state.position.copy()
        vel = self.state.velocity.copy()
        quat = self.state.quaternion.copy()
        omega = self.state.angular_velocity.copy()

        for _ in range(self.substeps):
            # k1
            k1_p, k1_v, k1_q, k1_w = self._state_derivatives(
                pos, vel, quat, omega, total_thrust, torques
            )

            # k2
            p2 = pos + 0.5 * h * k1_p
            v2 = vel + 0.5 * h * k1_v
            q2 = quat + 0.5 * h * k1_q
            q2_norm = np.linalg.norm(q2)
            if q2_norm > 1e-12:
                q2 /= q2_norm
            w2 = omega + 0.5 * h * k1_w
            k2_p, k2_v, k2_q, k2_w = self._state_derivatives(p2, v2, q2, w2, total_thrust, torques)

            # k3
            p3 = pos + 0.5 * h * k2_p
            v3 = vel + 0.5 * h * k2_v
            q3 = quat + 0.5 * h * k2_q
            q3_norm = np.linalg.norm(q3)
            if q3_norm > 1e-12:
                q3 /= q3_norm
            w3 = omega + 0.5 * h * k2_w
            k3_p, k3_v, k3_q, k3_w = self._state_derivatives(p3, v3, q3, w3, total_thrust, torques)

            # k4
            p4 = pos + h * k3_p
            v4 = vel + h * k3_v
            q4 = quat + h * k3_q
            q4_norm = np.linalg.norm(q4)
            if q4_norm > 1e-12:
                q4 /= q4_norm
            w4 = omega + h * k3_w
            k4_p, k4_v, k4_q, k4_w = self._state_derivatives(p4, v4, q4, w4, total_thrust, torques)

            # RK4 update
            pos += (h / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
            vel += (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
            quat += (h / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
            q_norm = np.linalg.norm(quat)
            if q_norm > 1e-12:
                quat /= q_norm
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            omega += (h / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)

        self.state.position = pos
        self.state.velocity = vel
        self.state.quaternion = quat
        self.state.angular_velocity = omega

        return self.state.copy()


class Drone6DOFEnv(AdaptiveRLEnv[np.ndarray, np.ndarray]):
    """Gymnasium environment simulating full 6-DOF rigid-body quadrotor navigation.

    Features:
    - 6-DOF translation and attitude dynamics with quaternion kinematics and body torques.
    - Continuous 4-motor action space mapped to physical rotor thrusts.
    - 36-dimensional observation space (position, velocity, quaternion, angular velocity,
      goal, relative target vector, target distance, and 16-ray 3D LiDAR).
    - Procedural spherical obstacles and arena boundary collision detection.
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
        mass: float = 1.0,
        arm_length: float = 0.2,
        inertia: Optional[Tuple[float, float, float]] = None,
        gravity: float = 9.81,
        max_thrust_per_motor: float = 5.0,
        torque_to_thrust_ratio: float = 0.01,
        linear_damping: float = 0.1,
        angular_damping: float = 0.02,
        dt: float = 0.05,
        substeps: int = 5,
        max_steps: int = 400,
        step_penalty: float = -0.05,
        goal_reward: float = 100.0,
        collision_reward: float = -100.0,
        progress_weight: float = 2.0,
        action_penalty_weight: float = 0.01,
        angular_rate_penalty_weight: float = 0.05,
        tilt_penalty_weight: float = 0.1,
        terminate_on_collision: bool = True,
        render_mode: Optional[str] = None,
        split: Optional[str] = None,
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

        self.num_obstacles = int(num_obstacles)
        self.obstacle_radius = float(obstacle_radius)
        self.target_radius = float(target_radius)
        self.collision_radius = float(collision_radius)
        self.lidar_range = float(lidar_range)
        self.num_lidar_rays = int(num_lidar_rays)
        self.max_steps = int(max_steps)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.collision_reward = float(collision_reward)
        self.progress_weight = float(progress_weight)
        self.action_penalty_weight = float(action_penalty_weight)
        self.angular_rate_penalty_weight = float(angular_rate_penalty_weight)
        self.tilt_penalty_weight = float(tilt_penalty_weight)
        self.terminate_on_collision = bool(terminate_on_collision)
        self.render_mode = render_mode

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

        self.max_diagonal = float(np.linalg.norm(self.bounds))
        self.max_velocity = 10.0  # normalization scale for velocity
        self.max_angular_velocity = 10.0  # normalization scale for angular velocity

        self.dynamics = DroneDynamics6DOF(
            mass=mass,
            arm_length=arm_length,
            inertia=inertia,
            gravity=gravity,
            max_thrust_per_motor=max_thrust_per_motor,
            torque_to_thrust_ratio=torque_to_thrust_ratio,
            linear_damping=linear_damping,
            angular_damping=angular_damping,
            dt=dt,
            substeps=substeps,
        )

        # Action space: 4 motor commands in [-1, 1], mapped to [0, max_thrust]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Observation space: 36 continuous variables in [-1, 1]
        # pos (3) + vel (3) + quat (4) + omega (3) + goal (3) + rel_goal (3) + dist (1) + lidar (16) = 36
        obs_dim = 3 + 3 + 4 + 3 + 3 + 3 + 1 + self.num_lidar_rays
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.lidar_rays = generate_lidar_3d_ray_directions(num_rays=self.num_lidar_rays)
        self._position: np.ndarray = self.default_start.copy()
        self._goal: np.ndarray = self.default_goal.copy()
        self._obstacles: List[ObstacleSphere3D] = []
        self._current_step = 0
        self._prev_distance_to_goal: float = float(np.linalg.norm(self._goal - self._position))

    def _action_to_motor_thrusts(self, action: np.ndarray) -> np.ndarray:
        """Map normalized action [-1, 1]^4 to motor thrusts [0, max_thrust]^4."""
        act_arr = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return (act_arr + 1.0) * 0.5 * self.dynamics.max_thrust_per_motor

    def _get_obs(self) -> np.ndarray:
        bx, by, bz = self.bounds
        state = self.dynamics.state

        norm_pos = [state.position[0] / bx, state.position[1] / by, state.position[2] / bz]
        norm_vel = [
            state.velocity[0] / self.max_velocity,
            state.velocity[1] / self.max_velocity,
            state.velocity[2] / self.max_velocity,
        ]
        quat = [state.quaternion[0], state.quaternion[1], state.quaternion[2], state.quaternion[3]]
        norm_omega = [
            state.angular_velocity[0] / self.max_angular_velocity,
            state.angular_velocity[1] / self.max_angular_velocity,
            state.angular_velocity[2] / self.max_angular_velocity,
        ]
        norm_goal = [self._goal[0] / bx, self._goal[1] / by, self._goal[2] / bz]
        rel_goal = [
            (self._goal[0] - state.position[0]) / bx,
            (self._goal[1] - state.position[1]) / by,
            (self._goal[2] - state.position[2]) / bz,
        ]
        curr_dist = float(np.linalg.norm(self._goal - state.position))
        norm_dist = [min(1.0, curr_dist / self.max_diagonal)]

        lidar_readings = compute_lidar_3d_readings(
            origin=state.position,
            ray_directions=self.lidar_rays,
            obstacles=self._obstacles,
            bounds=self.bounds,
            max_range=self.lidar_range,
        )

        raw = np.concatenate(
            [norm_pos, norm_vel, quat, norm_omega, norm_goal, rel_goal, norm_dist, lidar_readings]
        )
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
        state = self.dynamics.state
        dist = float(np.linalg.norm(self._goal - state.position))
        speed = float(np.linalg.norm(state.velocity))
        omega_mag = float(np.linalg.norm(state.angular_velocity))

        min_obs_dist = float("inf")
        for obs in self._obstacles:
            d = obs.distance_to(state.position) - self.collision_radius
            if d < min_obs_dist:
                min_obs_dist = d

        info = {
            "step": self._current_step,
            "max_steps": self.max_steps,
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "quaternion": state.quaternion.copy(),
            "angular_velocity": state.angular_velocity.copy(),
            "speed": speed,
            "angular_speed": omega_mag,
            "goal": self._goal.copy(),
            "distance_to_goal": dist,
            "num_obstacles": len(self._obstacles),
            "min_obstacle_distance": min_obs_dist if self._obstacles else float("inf"),
            "altitude": float(state.position[2]),
        }
        current_split = self._active_split if self._active_split is not None else self.split
        if current_split is not None:
            info["split"] = current_split
            info["split_seed"] = self._last_split_seed
        return info

    @property
    def drone_state(self) -> DroneState6DOF:
        """Expose current 6-DOF drone state."""
        return self.dynamics.state

    @property
    def target(self) -> np.ndarray:
        """Expose current target waypoint."""
        return self._goal

    @property
    def obstacles(self) -> List[ObstacleSphere3D]:
        """Public read-only copy of active obstacle set."""
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
            num_obs = int(options["num_obstacles"])

        self._position = self.default_start.copy()
        self._goal = self.default_goal.copy()

        self.dynamics.reset(position=self._position)

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
            act_arr = np.clip(act_arr, -1.0, 1.0)

        self._current_step += 1

        thrusts = self._action_to_motor_thrusts(act_arr)
        new_state = self.dynamics.step_rk4(thrusts)
        self._position = new_state.position

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
            omega_sq = float(np.sum(np.square(new_state.angular_velocity)))
            # Tilt penalty: penalize deviation from upright (upright means q = [1, 0, 0, 0] so r33 = 1)
            # Body z in world z:
            r_mat = quaternion_to_rotation_matrix(new_state.quaternion)
            upright_alignment = float(r_mat[2, 2])  # 1.0 when perfectly upright
            tilt_penalty = 1.0 - max(-1.0, min(1.0, upright_alignment))

            progress_reward = self.progress_weight * dist_delta
            effort_penalty = self.action_penalty_weight * action_effort
            rate_penalty = self.angular_rate_penalty_weight * omega_sq
            tilt_pen = self.tilt_penalty_weight * tilt_penalty

            reward = float(
                progress_reward + self.step_penalty - effort_penalty - rate_penalty - tilt_pen
            )

        if self._current_step >= self.max_steps and not terminated:
            truncated = True

        info["terminated"] = terminated
        info["truncated"] = truncated
        info["TimeLimit.truncated"] = truncated

        if self.render_mode == "human":
            print(self.render())

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        state = self.dynamics.state
        dist = float(np.linalg.norm(self._goal - state.position))
        speed = float(np.linalg.norm(state.velocity))
        pos_str = f"[{state.position[0]:5.1f}, {state.position[1]:5.1f}, {state.position[2]:5.1f}]"
        vel_str = f"[{state.velocity[0]:5.1f}, {state.velocity[1]:5.1f}, {state.velocity[2]:5.1f}]"
        quat_str = f"[{state.quaternion[0]:4.2f}, {state.quaternion[1]:4.2f}, {state.quaternion[2]:4.2f}, {state.quaternion[3]:4.2f}]"
        omega_str = f"[{state.angular_velocity[0]:4.2f}, {state.angular_velocity[1]:4.2f}, {state.angular_velocity[2]:4.2f}]"
        goal_str = f"[{self._goal[0]:5.1f}, {self._goal[1]:5.1f}, {self._goal[2]:5.1f}]"

        lines = [
            "+----------------------------------------------------------------+",
            "|               SIMULATED 6-DOF QUADROTOR FLIGHT DECK            |",
            "+----------------------------------------------------------------+",
            f"| Step: {self._current_step:03d}/{self.max_steps:03d} | Altitude (Z): {state.position[2]:5.1f}m | Speed: {speed:4.1f} m/s          |",
            f"| Position [X, Y, Z]: {pos_str:<28} |",
            f"| Velocity [Vx,Vy,Vz]: {vel_str:<28} |",
            f"| Attitude Quat [w,x,y,z]: {quat_str:<24} |",
            f"| Angular Vel [p,q,r]:     {omega_str:<24} |",
            f"| Waypoint [Gx,Gy,Gz]:     {goal_str:<24} |",
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
