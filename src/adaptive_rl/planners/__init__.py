"""Planning baselines and utilities for AdaptiveRL.

Provides classical 3D motion-planning algorithms for benchmarking against
reinforcement learning policies. Planners in this module operate independently
of PyTorch and trained neural network model weights.
"""

from .astar3d import (
    AStar3DPlanner,
    PlanningResult,
    check_segment_boundary_collision,
    check_segment_collision,
    check_segment_sphere_collision,
    compute_path_min_obstacle_clearance,
    is_point_valid,
    is_segment_valid,
    segment_sphere_distance,
)

__all__ = [
    "AStar3DPlanner",
    "PlanningResult",
    "check_segment_boundary_collision",
    "check_segment_collision",
    "check_segment_sphere_collision",
    "compute_path_min_obstacle_clearance",
    "is_point_valid",
    "is_segment_valid",
    "segment_sphere_distance",
]
