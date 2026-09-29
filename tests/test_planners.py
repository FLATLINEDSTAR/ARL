"""Unit tests for AStar3DPlanner and 3D analytical clearance geometry.

Covers the seven core verification scenarios required by Issue #250:
1. Empty-arena optimal straight-line navigation
2. Single-obstacle detour
3. Multi-obstacle detour
4. Unreachable goal handling and graceful termination
5. Strict segment-to-obstacle analytical clearance
6. Arena boundary validation and clearance enforcement
7. Determinism across repeated queries
"""

from __future__ import annotations

import math
import sys
from typing import List

import numpy as np
import pytest

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.planners.astar3d import (
    AStar3DPlanner,
    check_segment_boundary_collision,
    check_segment_sphere_collision,
    compute_path_min_obstacle_clearance,
    is_segment_valid,
    segment_sphere_distance,
)

# =============================================================================
# 1. Empty-Arena Optimal Straight Line
# =============================================================================


def test_empty_arena_optimal_straight_line() -> None:
    """In an unobstructed arena, planner produces the optimal direct straight line."""
    bounds = (30.0, 30.0, 15.0)
    start = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    goal = np.array([25.0, 25.0, 10.0], dtype=np.float64)
    obstacles: List[ObstacleSphere3D] = []

    planner = AStar3DPlanner(resolution=1.0, connectivity=26)
    result = planner.plan_detailed(
        start, goal, bounds=bounds, obstacles=obstacles, collision_radius=0.8
    )

    assert result.success is True
    assert result.path is not None
    # Empty-arena shortcut returns direct line [start, goal] with no unnecessary waypoints
    assert len(result.path) == 2
    assert np.allclose(result.path[0], start)
    assert np.allclose(result.path[1], goal)

    expected_dist = float(np.linalg.norm(goal - start))
    assert math.isclose(result.path_length, expected_dist, rel_tol=1e-5)
    assert math.isclose(result.straight_line_distance, expected_dist, rel_tol=1e-5)
    assert result.nodes_expanded == 0
    assert result.status == "success"


# =============================================================================
# 2. Single-Obstacle Detour
# =============================================================================


def test_single_obstacle_detour() -> None:
    """When an obstacle blocks the direct line, planner finds a valid collision-free detour."""
    bounds = (30.0, 30.0, 15.0)
    start = np.array([5.0, 15.0, 7.5], dtype=np.float64)
    goal = np.array([25.0, 15.0, 7.5], dtype=np.float64)
    collision_radius = 0.8

    # Place obstacle directly in the center of the start-goal line
    obs_center = np.array([15.0, 15.0, 7.5], dtype=np.float64)
    obs_radius = 3.0
    obstacles = [ObstacleSphere3D(center=obs_center, radius=obs_radius)]

    # Verify that the direct line is indeed blocked
    assert not is_segment_valid(start, goal, bounds, obstacles, collision_radius=collision_radius)

    planner = AStar3DPlanner(resolution=0.5, connectivity=26, max_iterations=20_000)
    result = planner.plan_detailed(
        start, goal, bounds=bounds, obstacles=obstacles, collision_radius=collision_radius
    )

    assert result.success is True
    assert result.path is not None
    assert len(result.path) >= 3

    # Verify path starts at start and reaches goal
    assert np.allclose(result.path[0], start)
    assert np.allclose(result.path[-1], goal)

    # Detour must be strictly longer than the unobstructed straight-line distance
    straight_dist = float(np.linalg.norm(goal - start))
    assert result.path_length > straight_dist

    # Every single segment must satisfy clearance with no collisions
    for p_a, p_b in zip(result.path[:-1], result.path[1:]):
        assert is_segment_valid(p_a, p_b, bounds, obstacles, collision_radius=collision_radius)
        # Minimum distance to effective obstacle sphere must be >= 0
        eff_radius = obs_radius + collision_radius
        clearance = segment_sphere_distance(p_a, p_b, obs_center, eff_radius)
        assert clearance >= -1e-6


# =============================================================================
# 3. Multi-Obstacle Detour
# =============================================================================


def test_multi_obstacle_detour() -> None:
    """Planner successfully navigates around multiple obstacles distributed across the space."""
    bounds = (30.0, 30.0, 15.0)
    start = np.array([3.0, 3.0, 3.0], dtype=np.float64)
    goal = np.array([27.0, 27.0, 12.0], dtype=np.float64)
    collision_radius = 0.8

    # Staggered obstacles along the diagonal path
    obstacles = [
        ObstacleSphere3D(center=np.array([10.0, 10.0, 5.0]), radius=2.5),
        ObstacleSphere3D(center=np.array([15.0, 18.0, 8.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([20.0, 15.0, 9.0]), radius=2.5),
    ]

    planner = AStar3DPlanner(resolution=1.0, connectivity=26, max_iterations=30_000)
    result = planner.plan_detailed(
        start, goal, bounds=bounds, obstacles=obstacles, collision_radius=collision_radius
    )

    assert result.success is True
    assert result.path is not None
    assert np.allclose(result.path[0], start)
    assert np.allclose(result.path[-1], goal)

    for p_a, p_b in zip(result.path[:-1], result.path[1:]):
        assert is_segment_valid(p_a, p_b, bounds, obstacles, collision_radius=collision_radius)


# =============================================================================
# 4. Unreachable Goals and Graceful Failure
# =============================================================================


def test_unreachable_goal_enclosed() -> None:
    """Goal enclosed by obstacles or budget exhaustion terminates gracefully returning None."""
    bounds = (30.0, 30.0, 15.0)
    start = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    goal = np.array([25.0, 25.0, 10.0], dtype=np.float64)
    collision_radius = 0.8

    # Enclose goal by placing a large obstacle directly covering the goal
    obstacles = [ObstacleSphere3D(center=np.array([25.0, 25.0, 10.0]), radius=3.0)]

    planner = AStar3DPlanner(resolution=1.0, connectivity=26)
    result = planner.plan_detailed(
        start, goal, bounds=bounds, obstacles=obstacles, collision_radius=collision_radius
    )

    assert result.success is False
    assert result.path is None
    assert result.status == "goal_in_collision"

    # Also test with start in collision
    obs_start = [ObstacleSphere3D(center=np.array([5.0, 5.0, 5.0]), radius=2.0)]
    result_start = planner.plan_detailed(
        start, goal, bounds=bounds, obstacles=obs_start, collision_radius=collision_radius
    )
    assert result_start.success is False
    assert result_start.path is None
    assert result_start.status == "start_in_collision"

    # Test search budget exhaustion
    blocked_obs = [ObstacleSphere3D(center=np.array([15.0, 15.0, 7.5]), radius=5.0)]
    budget_planner = AStar3DPlanner(resolution=0.5, connectivity=26, max_iterations=5)
    result_budget = budget_planner.plan_detailed(
        start, goal, bounds=bounds, obstacles=blocked_obs, collision_radius=collision_radius
    )
    assert result_budget.success is False
    assert result_budget.path is None
    assert result_budget.status == "budget_exhausted"


# =============================================================================
# 5. Strict Segment-to-Obstacle Clearance
# =============================================================================


def test_strict_segment_to_obstacle_clearance() -> None:
    """Analytical geometry correctly checks segments grazing, intersecting, and clearing spheres."""
    center = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    radius = 2.0

    # Case A: Segment penetrating sphere center
    p0 = np.array([5.0, 10.0, 10.0])
    p1 = np.array([15.0, 10.0, 10.0])
    dist = segment_sphere_distance(p0, p1, center, radius)
    # Closest point is center (10, 10, 10), distance to center is 0, minus radius = -2.0
    assert math.isclose(dist, -2.0, abs_tol=1e-6)
    assert check_segment_sphere_collision(p0, p1, center, radius) is True

    # Case B: Segment endpoints outside sphere, but segment cuts through sphere interior
    p0 = np.array([7.0, 11.0, 10.0])
    p1 = np.array([13.0, 11.0, 10.0])
    dist = segment_sphere_distance(p0, p1, center, radius)
    # Closest point is (10, 11, 10) which is distance 1.0 from center. Minus radius 2.0 = -1.0
    assert math.isclose(dist, -1.0, abs_tol=1e-6)
    assert check_segment_sphere_collision(p0, p1, center, radius) is True

    # Case C: Segment exactly grazing tangent to sphere surface
    p0 = np.array([5.0, 12.0, 10.0])
    p1 = np.array([15.0, 12.0, 10.0])
    dist = segment_sphere_distance(p0, p1, center, radius)
    # Closest point is (10, 12, 10) which is distance 2.0 from center. Minus radius 2.0 = 0.0
    assert math.isclose(dist, 0.0, abs_tol=1e-6)
    # Grazing violates clearance
    assert check_segment_sphere_collision(p0, p1, center, radius) is True

    # Case D: Segment clearly separated from sphere
    p0 = np.array([5.0, 15.0, 10.0])
    p1 = np.array([15.0, 15.0, 10.0])
    dist = segment_sphere_distance(p0, p1, center, radius)
    # Closest point is (10, 15, 10) distance 5.0, minus radius 2.0 = 3.0
    assert math.isclose(dist, 3.0, abs_tol=1e-6)
    assert check_segment_sphere_collision(p0, p1, center, radius) is False

    # Case E: Degenerate zero-length segment
    p_deg = np.array([10.0, 14.0, 10.0])
    dist_deg = segment_sphere_distance(p_deg, p_deg, center, radius)
    assert math.isclose(dist_deg, 2.0, abs_tol=1e-6)


# =============================================================================
# 6. Boundary Validation
# =============================================================================


def test_boundary_validation() -> None:
    """Planner rejects out-of-bounds start/goal and segments violating drone radius clearance."""
    bounds = (30.0, 30.0, 15.0)
    collision_radius = 0.8
    planner = AStar3DPlanner()

    # Start violating minimum x boundary (pos[0] - r <= 0)
    start_invalid = np.array([0.5, 15.0, 7.5])  # 0.5 - 0.8 = -0.3 <= 0
    goal_valid = np.array([25.0, 15.0, 7.5])
    assert planner.plan(start_invalid, goal_valid, bounds, [], collision_radius) is None

    # Goal violating maximum z boundary (pos[2] + r >= 15.0)
    goal_invalid = np.array([25.0, 15.0, 14.5])  # 14.5 + 0.8 = 15.3 >= 15.0
    start_valid = np.array([5.0, 15.0, 7.5])
    assert planner.plan(start_valid, goal_invalid, bounds, [], collision_radius) is None

    # Exact boundary condition (pos[0] == r) is considered touching / collides
    start_exact_boundary = np.array([0.8, 15.0, 7.5])
    collides, col_type = check_segment_boundary_collision(
        start_exact_boundary, start_exact_boundary, bounds, collision_radius=0.8
    )
    assert collides is True
    assert col_type == "boundary_x"

    # Segment crossing outside arena
    p0 = np.array([5.0, 5.0, 5.0])
    p1 = np.array([-1.0, 5.0, 5.0])
    collides_seg, _ = check_segment_boundary_collision(p0, p1, bounds, collision_radius=0.8)
    assert collides_seg is True


# =============================================================================
# 7. Determinism Across Repeated Calls
# =============================================================================


def test_determinism_across_repeated_calls() -> None:
    """Repeated planning queries on identical inputs produce bitwise identical paths."""
    bounds = (30.0, 30.0, 15.0)
    start = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    goal = np.array([25.0, 25.0, 10.0], dtype=np.float64)
    collision_radius = 0.8

    obstacles = [
        ObstacleSphere3D(center=np.array([12.0, 12.0, 6.0]), radius=2.5),
        ObstacleSphere3D(center=np.array([18.0, 18.0, 8.0]), radius=2.5),
    ]

    planner = AStar3DPlanner(resolution=1.0, connectivity=26, max_iterations=20_000)

    first_result = planner.plan_detailed(start, goal, bounds, obstacles, collision_radius)
    assert first_result.success is True
    assert first_result.path is not None

    for _ in range(5):
        run_result = planner.plan_detailed(start, goal, bounds, obstacles, collision_radius)
        assert run_result.success is True
        assert len(run_result.path) == len(first_result.path)
        for p1, p2 in zip(first_result.path, run_result.path):
            assert np.array_equal(p1, p2)
        assert math.isclose(run_result.path_length, first_result.path_length, abs_tol=1e-9)


# =============================================================================
# 8. Planner Independence & Configuration
# =============================================================================


def test_planner_does_not_load_torch() -> None:
    """AStar3DPlanner executes independently without importing PyTorch."""
    import subprocess

    cmd = [
        sys.executable,
        "-c",
        "import sys, adaptive_rl.planners.astar3d; sys.exit(0 if 'torch' not in sys.modules else 1)",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"Importing planner inadvertently imported torch: {proc.stderr}"


def test_planner_configuration_validation() -> None:
    """Planner rejects invalid resolution, connectivity, or iteration budgets."""
    with pytest.raises(ValueError, match="resolution must be strictly positive"):
        AStar3DPlanner(resolution=0.0)

    with pytest.raises(ValueError, match="connectivity must be 6 or 26"):
        AStar3DPlanner(connectivity=18)

    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        AStar3DPlanner(max_iterations=0)


# =============================================================================
# 9. Analytical Segment Clearance Tests
# =============================================================================


def test_compute_path_min_obstacle_clearance_segment_vs_waypoint() -> None:
    """Verify segment clearance detects obstacle proximity along line segments where endpoints are safe."""
    # Sphere at (10, 0, 0) with radius 2.0
    obs = ObstacleSphere3D(center=np.array([10.0, 0.0, 0.0]), radius=2.0)
    # Endpoints at (0, 0, 0) and (20, 0, 0)
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([20.0, 0.0, 0.0])
    path = [p0, p1]

    # Waypoint-only clearance:
    # dist(p0, center) = 10.0 - 2.0 = 8.0
    # dist(p1, center) = 10.0 - 2.0 = 8.0
    # Analytical segment clearance:
    # closest point on segment is (10, 0, 0) which is distance 0 from center.
    # clearance = 0 - 2.0 - 0.0 = -2.0 (penetration)
    clearance_no_col_radius = compute_path_min_obstacle_clearance(path, [obs], collision_radius=0.0)
    assert clearance_no_col_radius is not None
    assert math.isclose(clearance_no_col_radius, -2.0, abs_tol=1e-5)

    clearance_with_col_radius = compute_path_min_obstacle_clearance(
        path, [obs], collision_radius=0.5
    )
    assert clearance_with_col_radius is not None
    assert math.isclose(clearance_with_col_radius, -2.5, abs_tol=1e-5)

    # Now an obstacle off to the side: center (10, 5, 0), radius 2.0
    # Closest point on segment is (10, 0, 0), dist to center is 5.0
    # clearance = 5.0 - 2.0 - 0.8 = 2.2
    obs_side = ObstacleSphere3D(center=np.array([10.0, 5.0, 0.0]), radius=2.0)
    clearance_side = compute_path_min_obstacle_clearance(path, [obs_side], collision_radius=0.8)
    assert clearance_side is not None
    assert math.isclose(clearance_side, 2.2, abs_tol=1e-5)


def test_compute_path_min_obstacle_clearance_empty_and_single_point() -> None:
    """Verify compute_path_min_obstacle_clearance handles edge cases (empty path/obstacles, single point)."""
    assert (
        compute_path_min_obstacle_clearance(
            [], [ObstacleSphere3D(center=np.array([0, 0, 0]), radius=1.0)]
        )
        is None
    )
    assert compute_path_min_obstacle_clearance([np.array([1.0, 2.0, 3.0])], []) is None

    # Single point
    obs = ObstacleSphere3D(center=np.array([5.0, 0.0, 0.0]), radius=1.0)
    single_pt = [np.array([0.0, 0.0, 0.0])]
    clearance = compute_path_min_obstacle_clearance(single_pt, [obs], collision_radius=0.5)
    assert clearance is not None
    # 5.0 - 1.0 - 0.5 = 3.5
    assert math.isclose(clearance, 3.5, abs_tol=1e-5)


def test_planner_invalid_bounds_and_radius() -> None:
    """Planner rejects non-positive bounds and negative collision radius with invalid_inputs."""
    planner = AStar3DPlanner()
    start = np.array([5.0, 5.0, 5.0])
    goal = np.array([10.0, 10.0, 5.0])

    # Non-positive bounds
    res_bounds = planner.plan_detailed(
        start, goal, bounds=(30.0, 30.0, 0.0), obstacles=[], collision_radius=0.8
    )
    assert res_bounds.success is False
    assert res_bounds.status == "invalid_inputs"
    assert res_bounds.path is None

    # Negative collision radius
    res_radius = planner.plan_detailed(
        start, goal, bounds=(30.0, 30.0, 15.0), obstacles=[], collision_radius=-0.5
    )
    assert res_radius.success is False
    assert res_radius.status == "invalid_inputs"


def test_planner_identical_start_and_goal() -> None:
    """Planner handles coincident start and goal points as immediate zero-distance path."""
    planner = AStar3DPlanner()
    start = np.array([5.0, 5.0, 5.0])
    res = planner.plan_detailed(
        start, start, bounds=(30.0, 30.0, 15.0), obstacles=[], collision_radius=0.8
    )
    assert res.success is True
    assert res.status == "success"
    assert res.path is not None
    assert len(res.path) == 2
    assert np.allclose(res.path[0], start)
    assert np.allclose(res.path[1], start)
    assert res.path_length == pytest.approx(0.0)
    assert res.straight_line_distance == pytest.approx(0.0)
