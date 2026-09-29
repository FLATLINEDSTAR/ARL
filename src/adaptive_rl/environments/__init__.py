"""Environment interfaces, metadata, and registration for AdaptiveRL."""

from __future__ import annotations

from adaptive_rl.environments.base import AdaptiveRLEnv
from adaptive_rl.environments.drone import (
    DroneKinematics3D,
    DroneNavigation3DEnv,
    DroneState3D,
    ObstacleSphere3D,
)
from adaptive_rl.environments.metadata import EnvironmentMetadata
from adaptive_rl.environments.registry import (
    EnvironmentRegistry,
    RegistryError,
    create_environment,
    get,
    get_metadata,
    list_all_metadata,
    list_environments,
    make_env,
    register,
    registry,
)


def register_default_environments() -> None:
    """Register built-in drone environments into the global registry."""
    if "drone" not in list_environments():
        register(
            "drone",
            lambda **kwargs: DroneNavigation3DEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="drone",
                description="Simulated kinematic 3D drone navigation with continuous translation and 3D LiDAR.",
                observation_type="box",
                action_type="continuous",
                version="0.1.0",
                max_episode_steps=200,
                reward_range=(-100.0, 100.0),
                tags=["continuous", "drone", "3d", "kinematics", "lidar", "navigation"],
            ),
        )

    if "drone_3d" not in list_environments():
        register(
            "drone_3d",
            lambda **kwargs: DroneNavigation3DEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="drone_3d",
                description="Simulated kinematic 3D drone navigation with continuous translation and 3D LiDAR.",
                observation_type="box",
                action_type="continuous",
                version="0.1.0",
                max_episode_steps=200,
                reward_range=(-100.0, 100.0),
                tags=["continuous", "drone", "3d", "kinematics", "lidar", "navigation"],
            ),
        )

    if "drone_disturbed" not in list_environments():
        from adaptive_rl.environments.disturbed_drone import DroneDisturbed3DEnv

        register(
            "drone_disturbed",
            lambda **kwargs: DroneDisturbed3DEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="drone_disturbed",
                description="Drone navigation with deterministic steady-wind and OU gust parameters.",
                observation_type="box",
                action_type="continuous",
                version="0.2.0",
                max_episode_steps=200,
                reward_range=(-100.0, 100.0),
                tags=["continuous", "drone", "distribution-shift", "wind", "gust"],
            ),
        )

    if "drone-6dof" not in list_environments():
        from adaptive_rl.environments.drone_6dof import Drone6DOFEnv

        register(
            "drone-6dof",
            lambda **kwargs: Drone6DOFEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="drone-6dof",
                description="Rigid-body 6-DOF quadrotor flight dynamics simulation with attitude quaternion and 4-motor thrusts.",
                observation_type="box",
                action_type="continuous",
                version="0.1.0",
                max_episode_steps=400,
                reward_range=(-100.0, 100.0),
                tags=[
                    "continuous",
                    "drone",
                    "6dof",
                    "dynamics",
                    "quaternion",
                    "lidar",
                    "navigation",
                ],
            ),
        )

    if "drone_6dof" not in list_environments():
        from adaptive_rl.environments.drone_6dof import Drone6DOFEnv

        register(
            "drone_6dof",
            lambda **kwargs: Drone6DOFEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="drone_6dof",
                description="Rigid-body 6-DOF quadrotor flight dynamics simulation with attitude quaternion and 4-motor thrusts.",
                observation_type="box",
                action_type="continuous",
                version="0.1.0",
                max_episode_steps=400,
                reward_range=(-100.0, 100.0),
                tags=[
                    "continuous",
                    "drone",
                    "6dof",
                    "dynamics",
                    "quaternion",
                    "lidar",
                    "navigation",
                ],
            ),
        )

    if "drone_disturbed" not in list_environments():
        from adaptive_rl.environments.disturbed_drone import DroneDisturbed3DEnv

        register(
            "drone_disturbed",
            lambda **kwargs: DroneDisturbed3DEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="drone_disturbed",
                description="Kinematic 3D drone navigation under environmental wind, gusts, and dynamic obstacles.",
                observation_type="box",
                action_type="continuous",
                version="0.1.0",
                max_episode_steps=300,
                reward_range=(-100.0, 100.0),
                tags=[
                    "continuous",
                    "drone",
                    "disturbed",
                    "wind",
                    "gusts",
                    "dynamic_obstacles",
                    "lidar",
                    "navigation",
                ],
            ),
        )

    if "drone_disturbance" not in list_environments():
        from adaptive_rl.environments.disturbed_drone import DroneDisturbed3DEnv

        register(
            "drone_disturbance",
            lambda **kwargs: DroneDisturbed3DEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="drone_disturbance",
                description="Kinematic 3D drone navigation under environmental wind, gusts, and dynamic obstacles.",
                observation_type="box",
                action_type="continuous",
                version="0.1.0",
                max_episode_steps=300,
                reward_range=(-100.0, 100.0),
                tags=[
                    "continuous",
                    "drone",
                    "disturbed",
                    "wind",
                    "gusts",
                    "dynamic_obstacles",
                    "lidar",
                    "navigation",
                ],
            ),
        )


# Automatically register default environments
register_default_environments()

__all__ = [
    "AdaptiveRLEnv",
    "DisturbanceRecoveryTracker",
    "Drone6DOFEnv",
    "DroneDisturbance3DEnv",
    "DroneDisturbed3DEnv",
    "DroneDynamics6DOF",
    "DroneKinematics3D",
    "DroneNavigation3DEnv",
    "DroneState3D",
    "DroneState6DOF",
    "DynamicObstacleSphere3D",
    "EnvironmentMetadata",
    "EnvironmentRegistry",
    "ObstacleSphere3D",
    "RegistryError",
    "WindField3D",
    "WindState3D",
    "create_environment",
    "get",
    "get_metadata",
    "list_all_metadata",
    "list_environments",
    "make_env",
    "register",
    "register_default_environments",
    "registry",
]
