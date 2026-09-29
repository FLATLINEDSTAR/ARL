"""A* motion planner for 3D point-mass spatial navigation.

Provides a classical, deterministic 3D motion-planning baseline operating on
a discretized spatial lattice (6-connected or 26-connected). Collision and
clearance checking uses analytical segment-to-sphere and segment-to-boundary
geometry, guaranteeing that every segment along the generated path satisfies
clearance requirements.

This module is fully independent of PyTorch, trained weights, and reinforcement
learning algorithms.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from adaptive_rl.environments.drone import ObstacleSphere3D

# =============================================================================
# Analytical Geometry and Clearance Verification
# =============================================================================


def segment_sphere_distance(
    p0: np.ndarray,
    p1: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> float:
    """Compute the minimum Euclidean surface distance from a segment to a sphere.

    Parameters
    ----------
    p0: np.ndarray
        Start point of the line segment [x, y, z].
    p1: np.ndarray
        End point of the line segment [x, y, z].
    center: np.ndarray
        Center coordinates of the sphere [x, y, z].
    radius: float
        Radius of the sphere.

    Returns
    -------
    float
        Signed distance to the sphere surface:
        - > 0: segment is separated from the sphere surface.
        - = 0: segment grazes / touches the sphere surface.
        - < 0: segment penetrates the sphere (interior collision).
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)

    d = p1 - p0
    d_sq = float(np.dot(d, d))

    if d_sq < 1e-14:
        # Degenerate zero-length segment: point-to-sphere distance
        dist_to_center = float(np.linalg.norm(p0 - center))
        return dist_to_center - radius

    # Project sphere center onto the line segment: p(t) = p0 + t * (p1 - p0)
    # Minimizing ||p0 + t*d - center||^2 leads to:
    # t* = - dot(p0 - center, d) / dot(d, d) = dot(center - p0, d) / ||d||^2
    v = center - p0
    t = float(np.dot(v, d)) / d_sq
    t_clamped = max(0.0, min(1.0, t))

    closest_pt = p0 + t_clamped * d
    dist_to_center = float(np.linalg.norm(closest_pt - center))
    return dist_to_center - radius


def compute_path_min_obstacle_clearance(
    path: Sequence[Union[np.ndarray, Sequence[float]]],
    obstacles: Sequence[Any],
    collision_radius: float = 0.0,
) -> Optional[float]:
    """Compute the analytical minimum effective obstacle clearance across complete path segments.

    Evaluates the continuous closest Euclidean distance from each 3D line segment to
    every spherical obstacle center, subtracting the obstacle radius and drone collision radius:
        effective_clearance = dist(segment, obstacle.center) - obstacle.radius - collision_radius

    Parameters
    ----------
    path : Sequence of 3D coordinates
        Ordered sequence of waypoints along the planned trajectory.
    obstacles : Sequence of obstacles
        Spherical obstacles with center [x, y, z] and radius r.
    collision_radius : float
        Drone bounding collision sphere radius.

    Returns
    -------
    Optional[float]
        The minimum analytical clearance across all segments and obstacles,
        or None if path has fewer than 1 point or no obstacles exist.
    """
    if not obstacles or len(path) == 0:
        return None

    path_pts = [np.asarray(p, dtype=np.float64) for p in path]
    r = float(collision_radius)

    clearances: List[float] = []

    if len(path_pts) == 1:
        p0 = path_pts[0]
        for obs in obstacles:
            center = np.asarray(getattr(obs, "center", obs), dtype=np.float64)
            radius = float(getattr(obs, "radius", 0.0))
            dist = float(np.linalg.norm(p0 - center)) - radius - r
            clearances.append(dist)
    else:
        for p0, p1 in zip(path_pts[:-1], path_pts[1:]):
            for obs in obstacles:
                center = np.asarray(getattr(obs, "center", obs), dtype=np.float64)
                radius = float(getattr(obs, "radius", 0.0))
                dist = segment_sphere_distance(p0, p1, center, radius + r)
                clearances.append(dist)

    return float(min(clearances)) if clearances else None


def check_segment_sphere_collision(
    p0: np.ndarray,
    p1: np.ndarray,
    center: np.ndarray,
    radius: float,
    tolerance: float = 1e-7,
) -> bool:
    """Return True if the line segment intersects or grazes the sphere."""
    return segment_sphere_distance(p0, p1, center, radius) <= tolerance


def check_segment_boundary_collision(
    p0: np.ndarray,
    p1: np.ndarray,
    bounds: Tuple[float, float, float],
    collision_radius: float = 0.0,
    tolerance: float = 1e-7,
) -> Tuple[bool, str]:
    """Check if any point along the segment violates or touches the arena boundary.

    Because a line segment is linear, coordinate extrema along [p0, p1] occur
    strictly at the endpoints. The drone with collision_radius r collides if:
        pos[i] - r <= 0.0  or  pos[i] + r >= bounds[i]
    matching DroneNavigation3DEnv._check_collision semantics.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    r = float(collision_radius)
    axis_names = ("boundary_x", "boundary_y", "boundary_z")

    for i in range(3):
        min_val = min(p0[i], p1[i]) - r
        max_val = max(p0[i], p1[i]) + r
        if min_val <= tolerance or max_val >= (bounds[i] - tolerance):
            return True, axis_names[i]

    return False, "none"


def check_segment_collision(
    p0: np.ndarray,
    p1: np.ndarray,
    bounds: Tuple[float, float, float],
    obstacles: Sequence[ObstacleSphere3D],
    collision_radius: float = 0.0,
    tolerance: float = 1e-7,
) -> Tuple[bool, str]:
    """Verify segment clearance against arena boundaries and all obstacles.

    Matches DroneNavigation3DEnv collision checking rules:
    - Boundary collision if pos - r <= 0 or pos + r >= bounds
    - Obstacle collision if distance from segment to obstacle center <= obstacle.radius + r

    Returns
    -------
    Tuple[bool, str]
        (is_collision, collision_type) where collision_type is 'none',
        'boundary_x', 'boundary_y', 'boundary_z', or 'obstacle'.
    """
    bound_coll, bound_type = check_segment_boundary_collision(
        p0, p1, bounds=bounds, collision_radius=collision_radius, tolerance=tolerance
    )
    if bound_coll:
        return True, bound_type

    r = float(collision_radius)
    for obs in obstacles:
        eff_radius = float(obs.radius) + r
        if check_segment_sphere_collision(p0, p1, obs.center, eff_radius, tolerance=tolerance):
            return True, "obstacle"

    return False, "none"


def is_segment_valid(
    p0: np.ndarray,
    p1: np.ndarray,
    bounds: Tuple[float, float, float],
    obstacles: Sequence[ObstacleSphere3D],
    collision_radius: float = 0.0,
    tolerance: float = 1e-7,
) -> bool:
    """Return True if the segment is fully collision-free within bounds."""
    collides, _ = check_segment_collision(
        p0,
        p1,
        bounds=bounds,
        obstacles=obstacles,
        collision_radius=collision_radius,
        tolerance=tolerance,
    )
    return not collides


def is_point_valid(
    point: np.ndarray,
    bounds: Tuple[float, float, float],
    obstacles: Sequence[ObstacleSphere3D],
    collision_radius: float = 0.0,
    tolerance: float = 1e-7,
) -> bool:
    """Return True if a single point satisfies boundary and obstacle clearance."""
    return is_segment_valid(
        point,
        point,
        bounds=bounds,
        obstacles=obstacles,
        collision_radius=collision_radius,
        tolerance=tolerance,
    )


# =============================================================================
# Planning Result Data Structures
# =============================================================================


@dataclass
class PlanningResult:
    """Detailed result of a 3D A* planning query.

    Attributes
    ----------
    success: bool
        Whether a collision-free path reaching the goal was found.
    path: Optional[List[np.ndarray]]
        Ordered sequence of 3D waypoint coordinates, or None on failure.
    planning_time_ms: float
        Wall-clock time in milliseconds spent in plan(). Includes search,
        collision checking, and path reconstruction.
    nodes_expanded: int
        Total number of lattice nodes expanded during the search.
    status: str
        Explanatory status string: 'success', 'unreachable', 'budget_exhausted',
        'start_in_collision', 'goal_in_collision', 'out_of_bounds', or 'invalid_inputs'.
    path_length: Optional[float]
        Cumulative Euclidean length of the path in meters, or None.
    straight_line_distance: Optional[float]
        Direct Euclidean distance from start to goal in meters.
    """

    success: bool
    path: Optional[List[np.ndarray]]
    planning_time_ms: float
    nodes_expanded: int
    status: str
    path_length: Optional[float] = None
    straight_line_distance: Optional[float] = None


# =============================================================================
# Classical 3D A* Motion Planner
# =============================================================================


class AStar3DPlanner:
    """Classical deterministic 3D A* motion planner on a spatial lattice.

    Operates independently of PyTorch and trained neural network weights.
    Finds a collision-free path through spherical obstacles within 3D arena
    bounds, verifying that every segment respects the required clearance.

    Parameters
    ----------
    resolution: float, default 0.5
        Grid cell size of the 3D lattice in meters.
    connectivity: int, default 26
        Lattice connectivity: 6 (axis-aligned) or 26 (including face and cube diagonals).
    max_iterations: int, default 20_000
        Maximum number of node expansions before terminating with a budget failure.
    tolerance: float, default 1e-7
        Numerical tolerance for boundary and clearance checks.
    """

    def __init__(
        self,
        resolution: float = 0.5,
        connectivity: int = 26,
        max_iterations: int = 20_000,
        tolerance: float = 1e-7,
    ):
        if resolution <= 0.0:
            raise ValueError(f"resolution must be strictly positive, got {resolution}")
        if connectivity not in (6, 26):
            raise ValueError(f"connectivity must be 6 or 26, got {connectivity}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

        self.resolution = float(resolution)
        self.connectivity = connectivity
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self._neighbour_offsets = self._generate_neighbour_offsets(connectivity)

    # -------------------------------------------------------------------------
    # Public Planning API
    # -------------------------------------------------------------------------

    def plan(
        self,
        start_pos: np.ndarray | Sequence[float],
        goal_pos: np.ndarray | Sequence[float],
        bounds: Tuple[float, float, float] | Sequence[float],
        obstacles: Sequence[ObstacleSphere3D],
        collision_radius: float = 0.8,
    ) -> Optional[List[np.ndarray]]:
        """Plan a collision-free 3D waypoint sequence from start to goal.

        Parameters
        ----------
        start_pos: 3D coordinates [x, y, z] of the starting position.
        goal_pos: 3D coordinates [x, y, z] of the goal position.
        bounds: Arena dimensions (X_max, Y_max, Z_max).
        obstacles: Sequence of ObstacleSphere3D instances.
        collision_radius: Drone spherical collision radius in meters (default: 0.8).

        Returns
        -------
        Optional[List[np.ndarray]]
            Ordered list of 3D waypoint arrays including start and goal,
            or None if no collision-free path is found within search budget.
        """
        result = self.plan_detailed(
            start_pos=start_pos,
            goal_pos=goal_pos,
            bounds=bounds,
            obstacles=obstacles,
            collision_radius=collision_radius,
        )
        return result.path

    def plan_detailed(
        self,
        start_pos: np.ndarray | Sequence[float],
        goal_pos: np.ndarray | Sequence[float],
        bounds: Tuple[float, float, float] | Sequence[float],
        obstacles: Sequence[ObstacleSphere3D],
        collision_radius: float = 0.8,
    ) -> PlanningResult:
        """Execute 3D A* search and return comprehensive diagnostic results."""
        t0 = time.perf_counter()

        start = np.asarray(start_pos, dtype=np.float64)
        goal = np.asarray(goal_pos, dtype=np.float64)
        bounds_tuple = (float(bounds[0]), float(bounds[1]), float(bounds[2]))
        r = float(collision_radius)
        dist_sg = float(np.linalg.norm(goal - start))

        # 1. Input dimension and bounds sanity check
        if (
            start.shape != (3,)
            or goal.shape != (3,)
            or any(b <= 0.0 for b in bounds_tuple)
            or r < 0.0
        ):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return PlanningResult(
                success=False,
                path=None,
                planning_time_ms=elapsed_ms,
                nodes_expanded=0,
                status="invalid_inputs",
                straight_line_distance=dist_sg,
            )

        # 2. Check if start or goal violate arena bounds
        bound_coll_s, _ = check_segment_boundary_collision(
            start, start, bounds=bounds_tuple, collision_radius=r, tolerance=self.tolerance
        )
        bound_coll_g, _ = check_segment_boundary_collision(
            goal, goal, bounds=bounds_tuple, collision_radius=r, tolerance=self.tolerance
        )
        if bound_coll_s or bound_coll_g:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return PlanningResult(
                success=False,
                path=None,
                planning_time_ms=elapsed_ms,
                nodes_expanded=0,
                status="out_of_bounds",
                straight_line_distance=dist_sg,
            )

        # 3. Check if start or goal collide with obstacles
        if not is_point_valid(
            start,
            bounds=bounds_tuple,
            obstacles=obstacles,
            collision_radius=r,
            tolerance=self.tolerance,
        ):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return PlanningResult(
                success=False,
                path=None,
                planning_time_ms=elapsed_ms,
                nodes_expanded=0,
                status="start_in_collision",
                straight_line_distance=dist_sg,
            )

        if not is_point_valid(
            goal,
            bounds=bounds_tuple,
            obstacles=obstacles,
            collision_radius=r,
            tolerance=self.tolerance,
        ):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return PlanningResult(
                success=False,
                path=None,
                planning_time_ms=elapsed_ms,
                nodes_expanded=0,
                status="goal_in_collision",
                straight_line_distance=dist_sg,
            )

        # 4. Empty-arena / unobstructed direct path shortcut
        # If the straight-line segment from start to goal is collision-free,
        # return optimal direct path immediately without unnecessary intermediate waypoints.
        if is_segment_valid(
            start,
            goal,
            bounds=bounds_tuple,
            obstacles=obstacles,
            collision_radius=r,
            tolerance=self.tolerance,
        ):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            path = [start.copy(), goal.copy()]
            return PlanningResult(
                success=True,
                path=path,
                planning_time_ms=elapsed_ms,
                nodes_expanded=0,
                status="success",
                path_length=dist_sg,
                straight_line_distance=dist_sg,
            )

        # 5. A* Search on the 3D Lattice
        res = self.resolution
        start_idx = self._pos_to_idx(start)
        goal_idx = self._pos_to_idx(goal)

        # Identify candidate entry lattice nodes near start
        # If start is already an exact lattice point: single entry.
        # Otherwise, consider surrounding lattice vertices with collision-free segments from start.
        entry_nodes: List[Tuple[Tuple[int, int, int], float]] = []
        start_pt = self._idx_to_pos(start_idx)
        if np.allclose(start, start_pt, atol=1e-5):
            entry_nodes.append((start_idx, 0.0))
        else:
            base_coords = [int(math.floor(start[i] / res)) for i in range(3)]
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        cand_idx = (base_coords[0] + dx, base_coords[1] + dy, base_coords[2] + dz)
                        cand_pt = self._idx_to_pos(cand_idx)
                        if is_segment_valid(
                            start,
                            cand_pt,
                            bounds=bounds_tuple,
                            obstacles=obstacles,
                            collision_radius=r,
                            tolerance=self.tolerance,
                        ):
                            entry_nodes.append((cand_idx, float(np.linalg.norm(cand_pt - start))))

        if not entry_nodes:
            # If no surrounding lattice node can be reached directly from start
            entry_nodes.append((start_idx, float(np.linalg.norm(start_pt - start))))

        # Priority queue entries: (f_score, h_score, tie_breaker_counter, node_idx)
        open_pq: List[Tuple[float, float, int, Tuple[int, int, int]]] = []
        g_scores: Dict[Tuple[int, int, int], float] = {}
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        closed_set: Set[Tuple[int, int, int]] = set()

        counter = 0
        for cand_idx, init_g in entry_nodes:
            cand_pt = self._idx_to_pos(cand_idx)
            h = float(np.linalg.norm(goal - cand_pt))
            f = init_g + h
            g_scores[cand_idx] = init_g
            heapq.heappush(open_pq, (f, h, counter, cand_idx))
            counter += 1

        nodes_expanded = 0
        found_target_idx: Optional[Tuple[int, int, int]] = None

        while open_pq and nodes_expanded < self.max_iterations:
            f, h, _, current_idx = heapq.heappop(open_pq)

            if current_idx in closed_set:
                continue

            closed_set.add(current_idx)
            nodes_expanded += 1
            current_pt = self._idx_to_pos(current_idx)

            # Check if goal lattice node reached
            if current_idx == goal_idx:
                # Verify segment from lattice node to exact goal is clear
                if is_segment_valid(
                    current_pt,
                    goal,
                    bounds=bounds_tuple,
                    obstacles=obstacles,
                    collision_radius=r,
                    tolerance=self.tolerance,
                ):
                    found_target_idx = current_idx
                    break

            # Explore 3D lattice neighbours
            for offset, step_cost in self._neighbour_offsets:
                neigh_idx = (
                    current_idx[0] + offset[0],
                    current_idx[1] + offset[1],
                    current_idx[2] + offset[2],
                )

                if neigh_idx in closed_set:
                    continue

                neigh_pt = self._idx_to_pos(neigh_idx)

                # Segment-to-boundary and segment-to-obstacle clearance check
                if not is_segment_valid(
                    current_pt,
                    neigh_pt,
                    bounds=bounds_tuple,
                    obstacles=obstacles,
                    collision_radius=r,
                    tolerance=self.tolerance,
                ):
                    continue

                tentative_g = g_scores[current_idx] + step_cost
                if tentative_g < g_scores.get(neigh_idx, float("inf")):
                    g_scores[neigh_idx] = tentative_g
                    came_from[neigh_idx] = current_idx
                    neigh_h = float(np.linalg.norm(goal - neigh_pt))
                    neigh_f = tentative_g + neigh_h
                    heapq.heappush(open_pq, (neigh_f, neigh_h, counter, neigh_idx))
                    counter += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if found_target_idx is None:
            status = "budget_exhausted" if nodes_expanded >= self.max_iterations else "unreachable"
            return PlanningResult(
                success=False,
                path=None,
                planning_time_ms=elapsed_ms,
                nodes_expanded=nodes_expanded,
                status=status,
                straight_line_distance=dist_sg,
            )

        # 6. Path Reconstruction
        raw_indices: List[Tuple[int, int, int]] = [found_target_idx]
        curr = found_target_idx
        while curr in came_from:
            curr = came_from[curr]
            raw_indices.append(curr)
        raw_indices.reverse()

        # Build continuous waypoint sequence
        waypoints: List[np.ndarray] = [start.copy()]
        for idx in raw_indices:
            pt = self._idx_to_pos(idx)
            if not np.allclose(pt, waypoints[-1], atol=1e-5):
                waypoints.append(pt)

        if not np.allclose(waypoints[-1], goal, atol=1e-5):
            waypoints.append(goal.copy())

        # Strict post-verification: every single segment must satisfy clearance
        for p_a, p_b in zip(waypoints[:-1], waypoints[1:]):
            if not is_segment_valid(
                p_a,
                p_b,
                bounds=bounds_tuple,
                obstacles=obstacles,
                collision_radius=r,
                tolerance=self.tolerance,
            ):
                # If a segment unexpectedly collides, reject path
                return PlanningResult(
                    success=False,
                    path=None,
                    planning_time_ms=elapsed_ms,
                    nodes_expanded=nodes_expanded,
                    status="unreachable",
                    straight_line_distance=dist_sg,
                )

        # Compute total path length
        total_len = float(
            sum(np.linalg.norm(p_b - p_a) for p_a, p_b in zip(waypoints[:-1], waypoints[1:]))
        )

        return PlanningResult(
            success=True,
            path=waypoints,
            planning_time_ms=elapsed_ms,
            nodes_expanded=nodes_expanded,
            status="success",
            path_length=total_len,
            straight_line_distance=dist_sg,
        )

    # -------------------------------------------------------------------------
    # Internal Coordinate and Neighborhood Helpers
    # -------------------------------------------------------------------------

    def _pos_to_idx(self, pos: np.ndarray) -> Tuple[int, int, int]:
        return (
            int(round(float(pos[0]) / self.resolution)),
            int(round(float(pos[1]) / self.resolution)),
            int(round(float(pos[2]) / self.resolution)),
        )

    def _idx_to_pos(self, idx: Tuple[int, int, int]) -> np.ndarray:
        return np.array(
            [idx[0] * self.resolution, idx[1] * self.resolution, idx[2] * self.resolution],
            dtype=np.float64,
        )

    def _generate_neighbour_offsets(
        self, connectivity: int
    ) -> List[Tuple[Tuple[int, int, int], float]]:
        offsets: List[Tuple[Tuple[int, int, int], float]] = []
        res = self.resolution

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    manhattan = abs(dx) + abs(dy) + abs(dz)
                    if connectivity == 6 and manhattan != 1:
                        continue
                    dist = res * math.sqrt(dx * dx + dy * dy + dz * dz)
                    offsets.append(((dx, dy, dz), float(dist)))

        # Sort neighbour offsets by step cost for consistent deterministic expansion
        offsets.sort(key=lambda item: (item[1], item[0]))
        return offsets


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
