"""Evaluation metrics data structures and standardization schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Backward-compatible re-exports from canonical adaptive_rl.metrics module
from adaptive_rl.metrics import (
    EpisodeMetrics as EpisodeMetrics,
)
from adaptive_rl.metrics import (
    EpisodeMetricsAccumulator as EpisodeMetricsAccumulator,
)
from adaptive_rl.metrics import (
    compute_rate as compute_rate,
)
from adaptive_rl.metrics import (
    extract_episode_metrics as extract_episode_metrics,
)


def compute_trajectory_metrics(
    positions: Sequence[Any],
    goal: Optional[Any] = None,
    velocities: Optional[Sequence[Any]] = None,
    accelerations: Optional[Sequence[Any]] = None,
    obstacles: Optional[Sequence[Any]] = None,
) -> Dict[str, Optional[float]]:
    """Compute trajectory-quality and safety metrics for a single episode.

    Calculates:
    - path_length: L = sum(||p[t+1] - p[t]||) (m)
    - straight_line_distance: D0 = ||goal - p[0]|| (m)
    - path_efficiency: eta = D0 / L clamped to [0.0, 1.0] (0.0 if L <= 0)
    - min_obstacle_clearance: minimum distance from drone position to obstacle surface (m)
    - max_velocity: max(||v[t]||) (m/s)
    - max_acceleration: max(||a[t]||) (m/s^2)

    Args:
        positions: Sequence of 3D positions [[x0, y0, z0], [x1, y1, z1], ...].
        velocities: Optional sequence of 3D velocity vectors.
        accelerations: Optional sequence of 3D acceleration vectors.
        goal: Optional 3D goal coordinate [gx, gy, gz].
        obstacles: Optional sequence of obstacles with distance_to(pos) or (center, radius).

    Returns:
        Dictionary containing scalar trajectory and safety metrics.
    """
    if not positions:
        return {
            "path_length": 0.0,
            "straight_line_distance": 0.0,
            "path_efficiency": 0.0,
            "min_obstacle_clearance": None,
            "max_velocity": 0.0,
            "max_acceleration": 0.0,
        }

    pos_arr = np.asarray(positions, dtype=np.float64)

    # 1. Path length: L = sum(||p[t+1] - p[t]||)
    if len(pos_arr) >= 2:
        step_diffs = pos_arr[1:] - pos_arr[:-1]
        step_dists = np.linalg.norm(step_diffs, axis=1)
        path_length = float(np.sum(step_dists))
    else:
        path_length = 0.0

    # 2. Straight-line distance: D0 = ||goal - p[0]||
    if goal is not None and len(pos_arr) >= 1:
        goal_arr = np.asarray(goal, dtype=np.float64)
        straight_line_distance = float(np.linalg.norm(goal_arr - pos_arr[0]))
    else:
        straight_line_distance = 0.0

    # 3. Path efficiency: eta = D0 / L clamped to [0.0, 1.0]
    if path_length > 0.0:
        path_efficiency = float(min(1.0, max(0.0, straight_line_distance / path_length)))
    else:
        path_efficiency = 0.0

    # 4. Minimum obstacle surface clearance
    min_obstacle_clearance: Optional[float] = None
    if obstacles is not None and len(obstacles) > 0 and len(pos_arr) >= 1:
        clearances: List[float] = []
        for p in pos_arr:
            for obs in obstacles:
                if hasattr(obs, "distance_to"):
                    clearances.append(float(obs.distance_to(p)))
                elif hasattr(obs, "center") and hasattr(obs, "radius"):
                    center = np.asarray(obs.center, dtype=np.float64)
                    dist = float(np.linalg.norm(p - center)) - float(obs.radius)
                    clearances.append(dist)
        if clearances:
            min_obstacle_clearance = float(min(clearances))

    # 5. Maximum velocity: max(||v[t]||)
    if velocities is not None and len(velocities) > 0:
        vel_arr = np.asarray(velocities, dtype=np.float64)
        if vel_arr.ndim == 1:
            max_velocity = float(np.max(np.abs(vel_arr)))
        else:
            speeds = np.linalg.norm(vel_arr, axis=1)
            max_velocity = float(np.max(speeds))
    else:
        max_velocity = 0.0

    # 6. Maximum acceleration: max(||a[t]||)
    if accelerations is not None and len(accelerations) > 0:
        acc_arr = np.asarray(accelerations, dtype=np.float64)
        if acc_arr.ndim == 1:
            max_acceleration = float(np.max(np.abs(acc_arr)))
        else:
            acc_norms = np.linalg.norm(acc_arr, axis=1)
            max_acceleration = float(np.max(acc_norms))
    else:
        max_acceleration = 0.0

    return {
        "path_length": path_length,
        "straight_line_distance": straight_line_distance,
        "path_efficiency": path_efficiency,
        "min_obstacle_clearance": min_obstacle_clearance,
        "max_velocity": max_velocity,
        "max_acceleration": max_acceleration,
    }


class EvaluationMetrics(BaseModel):
    """Container for reinforcement learning evaluation results.

    Tracks core episodic return statistics, episode-level outcome rates,
    and optional domain-specific telemetry. Metrics unavailable for a given
    environment (e.g. collision_rate in non-spatial environments) are explicitly
    set to None, never silently collapsed into 0.0.

    Rate Denominator Semantics:
        Outcome rates (success_rate, collision_rate, overflow_rate) are calculated
        among episodes where the metric is defined (non-None). Episodes where
        the metric was not tracked or unavailable (None) are excluded from both
        numerator and denominator. If a metric is unavailable across all episodes
        (all None), the rate evaluates to None.

    Recovery-Time Aggregation Semantics (event-weighted, censoring-aware):
        `recovery_time` is the mean over *completed recovery events* pooled across
        all episodes (not a mean of per-episode means), so episodes with many
        events contribute proportionally more. Three episode classes are
        distinguished (counts recorded under `additional_metrics`):
        - measured (`recovery_episodes_measured`): at least one disturbance event
          completed with a valid recovery time in environment steps;
        - unavailable (`recovery_episodes_unavailable`): no disturbance event
          occurred, so recovery is undefined (never 0.0);
        - censored (`recovery_episodes_censored`): disturbance events occurred
          but none recovered before episode end; excluded from the mean, never 0.0.
        Event-level accounting is first-class: `recovery_events` (total),
        `recovery_completed_events`, `recovery_censored_events`,
        `recovery_completion_rate` (completed/total, None when total is 0), and
        `recovery_censoring_rate`. When the environment exposes no disturbance
        telemetry at all, every recovery field stays None. If no event completed,
        `recovery_time` is None. Because censored events are excluded from the
        mean, `recovery_time` is explicitly conditional and must never be read
        as an unconditional robustness score (see the shift-benchmark gap rule
        that suppresses recovery time gaps under censoring).
    """

    model_config = ConfigDict(extra="ignore")

    episodes: int = Field(..., gt=0, description="Total evaluation episodes executed")
    mean_reward: float = Field(..., description="Mean cumulative episodic reward")
    std_reward: float = Field(0.0, description="Standard deviation of episodic reward")
    min_reward: float = Field(0.0, description="Minimum episodic reward observed")
    max_reward: float = Field(0.0, description="Maximum episodic reward observed")
    success_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of episodes reaching target among episodes where success is defined (None if unavailable)",
    )
    collision_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of episodes ending in collision among episodes where collision is defined (None if unavailable)",
    )
    overflow_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of episodes experiencing queue overflow (traffic only; None if unavailable)",
    )
    truncation_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes reaching max step limit"
    )
    timeout_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of episodes ending in timeout (None if unavailable)",
    )
    recovery_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Event-weighted mean completed recovery time in environment steps "
        "(mean over completed recovery events pooled across episodes; None when no "
        "event completed or recovery is unavailable; never 0.0 as a placeholder; "
        "explicitly conditional on recovery, not an unconditional robustness score)",
    )
    recovery_events: Optional[int] = Field(
        None,
        ge=0,
        description="Total disturbance events observed across episodes "
        "(None when the environment exposes no disturbance telemetry)",
    )
    recovery_completed_events: Optional[int] = Field(
        None,
        ge=0,
        description="Total disturbance events that completed with a valid recovery "
        "(None when the environment exposes no disturbance telemetry)",
    )
    recovery_censored_events: Optional[int] = Field(
        None,
        ge=0,
        description="Total disturbance events still open at episode end "
        "(None when the environment exposes no disturbance telemetry)",
    )
    recovery_completion_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Completed events / total events (None when total is 0 or telemetry absent)",
    )
    recovery_censoring_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Censored events / total events (None when total is 0 or telemetry absent)",
    )
    mean_episode_length: float = Field(..., ge=0.0, description="Mean step count per episode")
    std_episode_length: float = Field(
        0.0, ge=0.0, description="Standard deviation of episode length"
    )
    mean_path_length: Optional[float] = Field(
        None, ge=0.0, description="Mean total Euclidean path length in meters"
    )
    std_path_length: Optional[float] = Field(
        None, ge=0.0, description="Standard deviation of path length in meters"
    )
    mean_straight_line_distance: Optional[float] = Field(
        None, ge=0.0, description="Mean straight-line distance from start to goal in meters"
    )
    mean_path_efficiency: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Mean path efficiency (ratio of straight-line distance to actual path length)",
    )
    mean_min_obstacle_clearance: Optional[float] = Field(
        None, description="Mean minimum obstacle surface clearance in meters"
    )
    mean_max_velocity: Optional[float] = Field(
        None, ge=0.0, description="Mean maximum linear velocity per episode in m/s"
    )
    mean_max_acceleration: Optional[float] = Field(
        None, ge=0.0, description="Mean maximum acceleration per episode in m/s^2"
    )
    obstacle_collision_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes ending in obstacle collision"
    )
    boundary_collision_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes ending in arena boundary collision"
    )
    obstacle_collision_count: Optional[int] = Field(
        None, ge=0, description="Total count of episodes ending in obstacle collision"
    )
    boundary_collision_count: Optional[int] = Field(
        None, ge=0, description="Total count of episodes ending in arena boundary collision"
    )
    additional_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment-specific metrics (e.g. energy consumption, path length, traffic telemetry)",
    )


class PlannerEvaluationMetrics(BaseModel):
    """Container for classical 3D motion planner evaluation results."""

    model_config = ConfigDict(extra="ignore")

    episodes: int = Field(..., gt=0, description="Total evaluation episodes executed")
    success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of queries reaching target without collision"
    )
    collision_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of queries violating clearance"
    )
    mean_planning_time_ms: float = Field(
        0.0, ge=0.0, description="Mean planning wall-clock time in milliseconds"
    )
    mean_planning_time: float = Field(
        0.0, ge=0.0, description="Mean planning wall-clock time in seconds (for standardization)"
    )
    mean_path_length: Optional[float] = Field(
        None, ge=0.0, description="Mean cumulative 3D path length in meters"
    )
    std_path_length: Optional[float] = Field(
        None, ge=0.0, description="Standard deviation of path length in meters"
    )
    mean_straight_line_distance: Optional[float] = Field(
        None, ge=0.0, description="Mean straight-line start-to-goal distance in meters"
    )
    mean_path_efficiency: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Mean ratio of straight-line distance to path length"
    )
    mean_min_obstacle_clearance: Optional[float] = Field(
        None, description="Mean minimum obstacle surface clearance in meters"
    )
    episode_records: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-episode planning query records and telemetry"
    )
    additional_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Additional domain-specific metrics"
    )


def _first_not_none(*values: Any) -> Any:
    """Return the first value that is not None, or None if all are None."""
    for val in values:
        if val is not None:
            return val
    return None


class StandardizedExperimentMetrics(BaseModel):
    """Standardized cross-paradigm evaluation metrics schema for AdaptiveRL.

    Provides a stable, uniform schema across reinforcement learning policies
    and classical deterministic/sampling planners. Metrics unavailable for a
    particular algorithm or environment are explicitly set to None (JSON null),
    never silently defaulted to 0.

    Aggregation Rules for Traffic Telemetry:
        - mean_queue_length: Mean of per-timestep total queue lengths averaged across all evaluation timesteps.
        - mean_max_wait_time: Mean of per-episode maximum vehicle waiting times across all evaluation episodes.
        - mean_wait_time: Mean of per-timestep mean vehicle waiting times averaged across all evaluation timesteps.
        - mean_total_departures: Mean cumulative vehicle departures per episode across all evaluation episodes.
        - total_departures: Total vehicle departures summed across all evaluation episodes.
        - cumulative_delay: Mean cumulative vehicle delay per episode across all evaluation episodes.
        - overflow_rate: Fraction of evaluation episodes with at least one queue overflow event.
    """

    model_config = ConfigDict(extra="ignore")

    episodes: int = Field(..., gt=0, description="Total evaluation episodes executed")
    episode_return: Optional[float] = Field(
        None, description="Mean cumulative episodic return (RL only; null for classical planners)"
    )
    success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes reaching goal or objective"
    )
    collision_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes ending in collision"
    )
    overflow_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of episodes experiencing queue overflow (traffic only)",
    )
    truncation_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes truncated by max step limit"
    )
    timeout_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes ending in timeout"
    )
    episode_length: Optional[float] = Field(None, ge=0.0, description="Mean step count per episode")
    path_length: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean geometric path length (grid steps or Euclidean distance)",
    )
    path_efficiency: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Ratio of optimal/straight-line distance to actual path length",
    )
    straight_line_distance: Optional[float] = Field(
        None, ge=0.0, description="Mean straight-line distance from start to goal in meters"
    )
    min_obstacle_clearance: Optional[float] = Field(
        None, description="Mean minimum obstacle surface clearance in meters"
    )
    max_velocity: Optional[float] = Field(
        None, ge=0.0, description="Mean maximum linear velocity per episode in m/s"
    )
    max_acceleration: Optional[float] = Field(
        None, ge=0.0, description="Mean maximum acceleration per episode in m/s^2"
    )
    obstacle_collision_count: Optional[int] = Field(
        None, ge=0, description="Total episodes ending in obstacle collision"
    )
    boundary_collision_count: Optional[int] = Field(
        None, ge=0, description="Total episodes ending in arena boundary collision"
    )
    obstacle_collision_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes ending in obstacle collision"
    )
    boundary_collision_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes ending in arena boundary collision"
    )
    planning_time: Optional[float] = Field(
        None, ge=0.0, description="Mean planning/search or inference wall-clock time in seconds"
    )
    generalization_gap: Optional[float] = Field(
        None, description="Performance gap between training and unseen test distributions"
    )
    battery_remaining: Optional[float] = Field(
        None, ge=0.0, description="Mean remaining battery level (drone environments)"
    )
    battery_used: Optional[float] = Field(
        None, ge=0.0, description="Mean battery energy consumed (drone environments)"
    )
    dynamic_collision_count: Optional[int] = Field(
        None, ge=0, description="Total dynamic obstacle collision events"
    )
    mean_queue_length: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean of per-timestep total queue lengths across all evaluation timesteps (traffic only)",
    )
    mean_max_wait_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean of per-episode maximum vehicle waiting times across all evaluation episodes (traffic only)",
    )
    mean_wait_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean of per-timestep mean vehicle waiting times across all evaluation timesteps (traffic only)",
    )
    mean_total_departures: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean cumulative vehicle departures per episode across all evaluation episodes (traffic only)",
    )
    total_departures: Optional[int] = Field(
        None,
        ge=0,
        description="Total vehicle departures summed across all evaluation episodes (traffic only)",
    )
    cumulative_delay: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean cumulative vehicle delay per episode across all evaluation episodes (traffic only)",
    )
    additional_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary domain-specific metric dictionary"
    )

    @classmethod
    def from_rl_metrics(
        cls,
        eval_metrics: EvaluationMetrics,
        generalization_gap: Optional[float] = None,
    ) -> StandardizedExperimentMetrics:
        """Construct standardized metrics from RL EvaluationMetrics."""
        extra = eval_metrics.additional_metrics

        # Extract traffic-specific aggregates
        mean_q = _first_not_none(extra.get("mean_queue_length"), extra.get("mean_queue"))
        mean_mw = _first_not_none(extra.get("mean_max_wait_time"), extra.get("mean_max_wait"))
        mean_w = _first_not_none(extra.get("mean_wait_time"), extra.get("mean_wait"))
        mean_dep = _first_not_none(extra.get("mean_total_departures"), extra.get("mean_departures"))
        tot_dep = extra.get("total_departures")
        cum_del = _first_not_none(extra.get("cumulative_delay"), extra.get("mean_cumulative_delay"))
        overflow_r = _first_not_none(eval_metrics.overflow_rate, extra.get("overflow_rate"))
        truncation_r = _first_not_none(eval_metrics.truncation_rate, extra.get("truncation_rate"))
        timeout_r = _first_not_none(
            eval_metrics.timeout_rate,
            extra.get("timeout_rate"),
            truncation_r,
        )
        path_eff = _first_not_none(
            eval_metrics.mean_path_efficiency,
            extra.get("mean_path_efficiency"),
            extra.get("path_efficiency"),
        )

        handled_keys = {
            "mean_path_length",
            "path_length",
            "std_path_length",
            "path_efficiency",
            "mean_path_efficiency",
            "straight_line_distance",
            "mean_straight_line_distance",
            "min_obstacle_clearance",
            "mean_min_obstacle_clearance",
            "max_velocity",
            "mean_max_velocity",
            "max_acceleration",
            "mean_max_acceleration",
            "obstacle_collision_count",
            "boundary_collision_count",
            "obstacle_collision_rate",
            "boundary_collision_rate",
            "mean_planning_time",
            "planning_time",
            "generalization_gap",
            "mean_battery_remaining",
            "battery_remaining",
            "mean_battery_used",
            "battery_used",
            "dynamic_collision_count",
            "mean_queue_length",
            "mean_queue",
            "mean_max_wait_time",
            "mean_max_wait",
            "mean_wait_time",
            "mean_wait",
            "mean_total_departures",
            "mean_departures",
            "total_departures",
            "cumulative_delay",
            "mean_cumulative_delay",
            "overflow_rate",
            "truncation_rate",
            "timeout_rate",
        }

        path_len = _first_not_none(
            eval_metrics.mean_path_length,
            extra.get("mean_path_length"),
            extra.get("path_length"),
        )
        path_eff = _first_not_none(
            eval_metrics.mean_path_efficiency,
            extra.get("mean_path_efficiency"),
            extra.get("path_efficiency"),
        )
        straight_dist = _first_not_none(
            eval_metrics.mean_straight_line_distance,
            extra.get("mean_straight_line_distance"),
            extra.get("straight_line_distance"),
        )
        min_clear = _first_not_none(
            eval_metrics.mean_min_obstacle_clearance,
            extra.get("mean_min_obstacle_clearance"),
            extra.get("min_obstacle_clearance"),
        )
        max_vel = _first_not_none(
            eval_metrics.mean_max_velocity,
            extra.get("mean_max_velocity"),
            extra.get("max_velocity"),
        )
        max_acc = _first_not_none(
            eval_metrics.mean_max_acceleration,
            extra.get("mean_max_acceleration"),
            extra.get("max_acceleration"),
        )
        obs_coll_cnt = _first_not_none(
            eval_metrics.obstacle_collision_count,
            extra.get("obstacle_collision_count"),
        )
        bound_coll_cnt = _first_not_none(
            eval_metrics.boundary_collision_count,
            extra.get("boundary_collision_count"),
        )
        obs_coll_rate = _first_not_none(
            eval_metrics.obstacle_collision_rate,
            extra.get("obstacle_collision_rate"),
        )
        bound_coll_rate = _first_not_none(
            eval_metrics.boundary_collision_rate,
            extra.get("boundary_collision_rate"),
        )

        return cls(
            episodes=eval_metrics.episodes,
            episode_return=eval_metrics.mean_reward,
            success_rate=eval_metrics.success_rate,
            collision_rate=eval_metrics.collision_rate,
            overflow_rate=overflow_r,
            truncation_rate=truncation_r,
            timeout_rate=timeout_r,
            episode_length=eval_metrics.mean_episode_length,
            path_length=path_len,
            path_efficiency=path_eff,
            straight_line_distance=straight_dist,
            min_obstacle_clearance=min_clear,
            max_velocity=max_vel,
            max_acceleration=max_acc,
            obstacle_collision_count=obs_coll_cnt,
            boundary_collision_count=bound_coll_cnt,
            obstacle_collision_rate=obs_coll_rate,
            boundary_collision_rate=bound_coll_rate,
            planning_time=_first_not_none(
                extra.get("mean_planning_time"), extra.get("planning_time")
            ),
            generalization_gap=_first_not_none(generalization_gap, extra.get("generalization_gap")),
            battery_remaining=_first_not_none(
                extra.get("mean_battery_remaining"), extra.get("battery_remaining")
            ),
            battery_used=_first_not_none(extra.get("mean_battery_used"), extra.get("battery_used")),
            dynamic_collision_count=extra.get("dynamic_collision_count"),
            mean_queue_length=mean_q,
            mean_max_wait_time=mean_mw,
            mean_wait_time=mean_w,
            mean_total_departures=mean_dep,
            total_departures=tot_dep,
            cumulative_delay=cum_del,
            additional_metrics={k: v for k, v in extra.items() if k not in handled_keys},
        )

    @classmethod
    def from_planner_metrics(
        cls,
        planner_metrics: Any,
    ) -> StandardizedExperimentMetrics:
        """Construct standardized metrics from PlannerEvaluationMetrics."""
        extra = getattr(planner_metrics, "additional_metrics", {})
        mean_path = getattr(planner_metrics, "mean_path_length", None)
        mean_time = getattr(planner_metrics, "mean_planning_time", None)

        path_eff = _first_not_none(
            getattr(planner_metrics, "mean_path_efficiency", None),
            extra.get("path_efficiency"),
            extra.get("mean_path_efficiency"),
        )
        straight_dist = _first_not_none(
            getattr(planner_metrics, "mean_straight_line_distance", None),
            extra.get("mean_straight_line_distance"),
            extra.get("straight_line_distance"),
        )
        min_clear = _first_not_none(
            getattr(planner_metrics, "mean_min_obstacle_clearance", None),
            extra.get("mean_min_obstacle_clearance"),
            extra.get("min_obstacle_clearance"),
        )

        return cls(
            episodes=getattr(planner_metrics, "episodes", 1),
            episode_return=None,  # Classical planners do not accumulate RL reward returns
            success_rate=getattr(planner_metrics, "success_rate", None),
            collision_rate=getattr(planner_metrics, "collision_rate", None),
            overflow_rate=None,
            truncation_rate=None,
            episode_length=None,  # Independent: RL step count is not applicable to geometric paths
            path_length=mean_path,
            path_efficiency=path_eff,
            straight_line_distance=straight_dist,
            min_obstacle_clearance=min_clear,
            planning_time=mean_time,
            generalization_gap=None,
            battery_remaining=None,
            battery_used=None,
            dynamic_collision_count=None,
            additional_metrics=extra,
        )

    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert flat scalar fields to a dictionary suitable for CSV serialization."""
        data = self.model_dump(exclude={"additional_metrics"})
        flat: Dict[str, Any] = {}
        for k, v in data.items():
            if v is not None:
                flat[k] = v
            else:
                flat[k] = ""
        return flat


__all__ = [
    "EvaluationMetrics",
    "PlannerEvaluationMetrics",
    "StandardizedExperimentMetrics",
    "compute_trajectory_metrics",
]
