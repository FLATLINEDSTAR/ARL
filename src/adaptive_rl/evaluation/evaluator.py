"""Evaluation engine for AdaptiveRL."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.algorithms.random_policy import RandomPolicy
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.metrics import (
    EvaluationMetrics,
    PlannerEvaluationMetrics,
    StandardizedExperimentMetrics,
    compute_trajectory_metrics,
)
from adaptive_rl.evaluation.statistics import MetricStatistics, summarize_seed_values


def derive_episode_reset_seed(
    evaluation_group_seed: int,
    episode_index: int,
    *,
    split: Optional[str] = None,
) -> int:
    """Derive a stable reset seed from a seed group and episode index.

    For custom distributions, Cantor pairing maps every non-negative integer
    pair injectively to a non-negative integer:
    ``((g + i) * (g + i + 1)) // 2 + i``. It is independent of the requested
    episode count and does not use Python's randomized ``hash``.

    Configured train/test splits have finite seed namespaces, so a SHA-256
    digest is reduced into that split's documented interval. Sampling with
    replacement is possible in these finite partitions and is identified in
    the benchmark metadata.
    """
    if isinstance(evaluation_group_seed, bool) or not isinstance(evaluation_group_seed, int):
        raise ValueError("Evaluation group seed must be an integer.")
    if evaluation_group_seed < 0:
        raise ValueError("Evaluation group seed must be non-negative.")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise ValueError("Episode index must be an integer.")
    if episode_index < 0:
        raise ValueError("Episode index must be non-negative.")

    if split is None:
        total = evaluation_group_seed + episode_index
        return total * (total + 1) // 2 + episode_index

    from adaptive_rl.evaluation.generalization import (
        TEST_SEED_END,
        TEST_SEED_START,
        TRAIN_SEED_END,
        TRAIN_SEED_START,
        validate_split_seed,
    )

    clean_split = split.strip().lower()
    if clean_split == "train":
        start, end = TRAIN_SEED_START, TRAIN_SEED_END
    elif clean_split == "test":
        start, end = TEST_SEED_START, TEST_SEED_END
    else:
        raise ValueError(f"Unknown evaluation split {split!r}.")
    validate_split_seed(evaluation_group_seed, clean_split)
    capacity = end - start
    payload = (
        f"adaptive-rl-evaluation-reset-v1:{clean_split}:{evaluation_group_seed}:{episode_index}"
    )
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return start + int.from_bytes(digest[:8], "big") % capacity


@dataclass(frozen=True)
class EpisodeEvaluationRecord:
    """Record for a single evaluation episode.

    Seed semantics:
        ``evaluation_group_seed`` is the requested evaluation seed and is the
        statistical grouping unit used by :meth:`Evaluator.evaluate_seeds`.
        It is ``None`` for single-seed :meth:`Evaluator.evaluate` runs, which
        are not organized into seed groups. ``episode_reset_seed`` is the value
        actually passed to ``env.reset(seed=...)`` for this episode.

    Legacy fields:
        ``seed`` and ``episode_seed`` are retained for backward compatibility.
        ``episode_seed`` mirrors ``episode_reset_seed``. ``seed`` holds the
        reset seed for single-seed runs and the group seed for multi-seed runs;
        prefer the explicit accessors above when interpreting results.
    """

    episode_index: int
    seed: Optional[int]
    return_value: float
    episode_length: int
    success: Optional[bool]
    collision: Optional[bool]
    truncated: bool
    episode_seed: Optional[int] = None
    evaluation_group_seed: Optional[int] = None
    collision_type: Optional[str] = None
    path_length: Optional[float] = None
    straight_line_distance: Optional[float] = None
    path_efficiency: Optional[float] = None
    min_obstacle_clearance: Optional[float] = None
    max_velocity: Optional[float] = None
    max_acceleration: Optional[float] = None
    timeout: bool = False

    @property
    def episode_reset_seed(self) -> Optional[int]:
        """Actual ``env.reset(seed=...)`` value for this episode."""
        return self.episode_seed

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "episode_index": self.episode_index,
            "evaluation_group_seed": self.evaluation_group_seed,
            "episode_reset_seed": self.episode_reset_seed,
            "seed": self.seed,
            "return": self.return_value,
            "episode_length": self.episode_length,
            "success": self.success,
            "collision": self.collision,
            "truncated": self.truncated,
            "timeout": self.timeout,
            "episode_seed": self.episode_seed,
        }
        if self.collision_type is not None:
            data["collision_type"] = self.collision_type
        if self.path_length is not None:
            data["path_length"] = self.path_length
        if self.straight_line_distance is not None:
            data["straight_line_distance"] = self.straight_line_distance
        if self.path_efficiency is not None:
            data["path_efficiency"] = self.path_efficiency
        if self.min_obstacle_clearance is not None:
            data["min_obstacle_clearance"] = self.min_obstacle_clearance
        if self.max_velocity is not None:
            data["max_velocity"] = self.max_velocity
        if self.max_acceleration is not None:
            data["max_acceleration"] = self.max_acceleration
        return data


@dataclass(frozen=True)
class SeedEvaluationSummary:
    """Episode-level summary for one evaluation seed group.

    ``seed`` is the requested evaluation seed (the statistical grouping unit),
    not an episode reset seed. :attr:`evaluation_group_seed` exposes the same
    value under an explicit name.
    """

    seed: int
    episodes: int
    success_rate: float | None
    collision_rate: float | None
    truncation_rate: float | None
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    std_episode_length: float
    path_length: float | None

    @property
    def evaluation_group_seed(self) -> int:
        """Requested evaluation seed; the unit used for cross-seed statistics."""
        return self.seed

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "seed": self.seed,
            "evaluation_group_seed": self.evaluation_group_seed,
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "truncation_rate": self.truncation_rate,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_episode_length": self.mean_episode_length,
            "std_episode_length": self.std_episode_length,
            "path_length": self.path_length,
        }


@dataclass(frozen=True)
class MultiSeedEvaluationResult:
    """Complete multi-seed result retaining episodes and seed-level identity.

    ``seeds`` are evaluation group seeds; each group runs
    ``episodes_per_seed`` episodes whose actual reset seeds are derived from
    the group seed and recorded per episode as ``episode_reset_seed``.
    """

    seeds: list[int]
    episodes_per_seed: int
    deterministic: bool
    per_seed: list[SeedEvaluationSummary]
    episodes: list[EpisodeEvaluationRecord]
    aggregate: dict[str, MetricStatistics]
    environment: str

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    @property
    def evaluation_group_seeds(self) -> list[int]:
        """Requested evaluation seeds; the grouping unit for cross-seed statistics."""
        return list(self.seeds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "seeds": list(self.seeds),
                "evaluation_group_seeds": list(self.seeds),
                "seed_count": len(self.seeds),
                "seed_semantics": (
                    "seeds are evaluation group seeds (the statistical grouping unit); "
                    "episodes within a group use derived episode_reset_seed values"
                ),
                "episode_reset_seed_mapping": (
                    "custom: Cantor pairing ((group_seed + episode_index) * "
                    "(group_seed + episode_index + 1)) // 2 + episode_index; "
                    "configured split: SHA-256 of "
                    "'adaptive-rl-evaluation-reset-v1:<split>:<group_seed>:<episode_index>' "
                    "reduced into the finite split interval"
                ),
                "episodes_per_seed": self.episodes_per_seed,
                "total_episodes": self.total_episodes,
                "deterministic": self.deterministic,
                "environment": self.environment,
                "confidence_interval": "two-sided 95% Student's t interval across seed summaries; "
                "sample standard deviation; unavailable when fewer than two values exist",
                "duplicate_seed_policy": "rejected",
            },
            "episodes": [record.to_dict() for record in self.episodes],
            "per_seed": [summary.to_dict() for summary in self.per_seed],
            "aggregate": {
                name: statistics.to_dict() for name, statistics in self.aggregate.items()
            },
        }


class Evaluator:
    """Standardized multi-episode evaluation engine for drone navigation.

    Environment telemetry contract:
        Every episode always records return, length, and the Gymnasium
        truncation flag. Additional outcomes come from the environment ``info``
        dict and are optional: ``success``/``is_success``, ``collision``,
        ``collision_type``, ``position``, ``velocity``, ``acceleration``.
        When an environment does not report an outcome, the metric is
        ``None`` (never ``0.0``), so non-spatial environments remain valid
        evaluation targets.

    Drone-specific telemetry:
        Collision typing, obstacle clearance, path length, and path
        efficiency describe the 3D drone navigation task and are only
        available when the environment exposes positions and obstacles.
        They must not be read as universal environment metrics. Obstacle
        geometry is read from the public ``unwrapped.obstacles`` interface
        when available, falling back to the private ``_obstacles`` attribute
        for environments that predate it.
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: Optional[gym.Env] = None,
        env_name: str = "drone",
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.algorithm = algorithm
        self.env_kwargs = dict(env_kwargs or {})
        self.env_name = env_name

        if env is not None:
            self.env = env
        else:
            self.env = make_env(self.env_name, **self.env_kwargs)

        self.last_episode_records: List[EpisodeEvaluationRecord] = []

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        base_seed: Optional[int] = None,
        seeds: Optional[Sequence[int]] = None,
        split: Optional[str] = None,
    ) -> EvaluationMetrics:
        """Execute evaluation rollouts and compute aggregated metrics.

        Args:
            num_episodes: Number of episodes to run (overridden by len(seeds) if seeds is given).
            deterministic: Whether to use deterministic policy actions.
            base_seed: Starting seed for sequential seed generation (seed = base_seed + ep).
            seeds: Explicit sequence of integer seeds to evaluate against.
            split: Benchmark split ('train' or 'test'). When supplied without seeds,
                seeds are deterministically derived from the split partition.
        """
        if split is not None:
            from adaptive_rl.evaluation.generalization import get_split_seeds, validate_split_seed

            if seeds is None:
                seeds = get_split_seeds(split, num_episodes=num_episodes)
            else:
                for s in seeds:
                    validate_split_seed(s, split)

        if seeds is not None:
            seeds = list(seeds)
            num_episodes = len(seeds)

        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")

        self.last_episode_records = []
        rewards: List[float] = []
        lengths: List[int] = []
        successes: List[Optional[bool]] = []
        collisions: List[Optional[bool]] = []
        truncations: List[bool] = []
        obstacle_collisions: List[bool] = []
        boundary_collisions: List[bool] = []
        timeouts: List[bool] = []

        path_lengths: List[float] = []
        straight_line_dists: List[float] = []
        path_efficiencies: List[float] = []
        min_clearances: List[float] = []
        max_velocities: List[float] = []
        max_accelerations: List[float] = []

        for ep in range(num_episodes):
            if seeds is not None:
                seed = seeds[ep]
            elif base_seed is not None:
                seed = base_seed + ep
            else:
                seed = None

            reset_options = {"split": split} if split is not None else None
            if reset_options is not None:
                try:
                    obs, info = self.env.reset(seed=seed, options=reset_options)
                except TypeError:
                    obs, info = self.env.reset(seed=seed)
            else:
                obs, info = self.env.reset(seed=seed)
            ep_reward = 0.0
            ep_length = 0
            done = False
            last_info = dict(info or {})
            was_truncated = False
            previous_position = self._position_from_info(last_info)
            path_length = 0.0 if previous_position is not None else None

            # Trajectory tracking for trajectory-quality and safety metrics
            positions: List[np.ndarray] = []
            velocities: List[np.ndarray] = []
            accelerations: List[np.ndarray] = []

            init_pos = last_info.get("position")
            if init_pos is not None:
                positions.append(np.asarray(init_pos, dtype=np.float64).copy())
            init_vel = last_info.get("velocity")
            if init_vel is not None:
                velocities.append(np.asarray(init_vel, dtype=np.float64).copy())
            init_acc = last_info.get("acceleration")
            if init_acc is not None:
                accelerations.append(np.asarray(init_acc, dtype=np.float64).copy())

            goal = last_info.get("goal")
            unwrapped_env = getattr(self.env, "unwrapped", self.env)
            obstacles = getattr(unwrapped_env, "obstacles", None)
            if obstacles is None:
                obstacles = getattr(unwrapped_env, "_obstacles", None)

            while not done:
                if self.algorithm is not None:
                    action, _ = self.algorithm.predict(obs, deterministic=deterministic)
                elif hasattr(self.env, "action_space") and self.env.action_space is not None:
                    action = self.env.action_space.sample()
                else:
                    action = np.zeros(3, dtype=np.float32)

                obs, reward, terminated, truncated, step_info = self.env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                last_info = dict(step_info or {})
                was_truncated = bool(truncated)
                current_position = self._position_from_info(step_info)
                if (
                    path_length is not None
                    and previous_position is not None
                    and current_position is not None
                    and current_position.shape == previous_position.shape
                ):
                    path_length += float(np.linalg.norm(current_position - previous_position))
                else:
                    path_length = None
                previous_position = current_position
                done = terminated or truncated
                step_pos = last_info.get("position")
                if step_pos is not None:
                    positions.append(np.asarray(step_pos, dtype=np.float64).copy())
                step_vel = last_info.get("velocity")
                if step_vel is not None:
                    velocities.append(np.asarray(step_vel, dtype=np.float64).copy())
                step_acc = last_info.get("acceleration")
                if step_acc is not None:
                    accelerations.append(np.asarray(step_acc, dtype=np.float64).copy())
                elif hasattr(unwrapped_env, "kinematics"):
                    accelerations.append(unwrapped_env.kinematics.state.acceleration.copy())

            success_value = last_info.get("success", last_info.get("is_success"))
            collision_value = last_info.get("collision")
            is_success = bool(success_value) if success_value is not None else None
            is_collision = bool(collision_value) if collision_value is not None else None
            collision_type = str(last_info.get("collision_type", "none"))
            is_obs_coll = bool(is_collision) and collision_type == "obstacle"
            is_bound_coll = bool(is_collision) and collision_type.startswith("boundary")
            is_timeout = was_truncated

            rewards.append(ep_reward)
            lengths.append(ep_length)
            successes.append(is_success)
            collisions.append(is_collision)
            truncations.append(was_truncated)
            if not math.isfinite(ep_reward):
                raise ValueError(f"Episode {ep} produced a non-finite cumulative reward.")
            obstacle_collisions.append(is_obs_coll)
            boundary_collisions.append(is_bound_coll)
            timeouts.append(is_timeout)

            traj_metrics = compute_trajectory_metrics(
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                goal=goal,
                obstacles=obstacles,
            )

            ep_path_len = path_length
            ep_straight_dist = traj_metrics["straight_line_distance"]
            ep_path_eff = traj_metrics["path_efficiency"]
            ep_min_clear = traj_metrics["min_obstacle_clearance"]
            ep_max_vel = traj_metrics["max_velocity"]
            ep_max_acc = traj_metrics["max_acceleration"]

            if ep_path_len is not None:
                path_lengths.append(ep_path_len)
            if ep_straight_dist is not None:
                straight_line_dists.append(ep_straight_dist)
            if ep_path_eff is not None:
                path_efficiencies.append(ep_path_eff)
            if ep_min_clear is not None:
                min_clearances.append(ep_min_clear)
            if ep_max_vel is not None:
                max_velocities.append(ep_max_vel)
            if ep_max_acc is not None:
                max_accelerations.append(ep_max_acc)

            self.last_episode_records.append(
                EpisodeEvaluationRecord(
                    episode_index=ep,
                    seed=seed,
                    return_value=ep_reward,
                    episode_length=ep_length,
                    success=is_success,
                    collision=is_collision,
                    truncated=was_truncated,
                    episode_seed=seed,
                    collision_type=collision_type if is_collision else None,
                    path_length=ep_path_len,
                    straight_line_distance=ep_straight_dist,
                    path_efficiency=ep_path_eff,
                    min_obstacle_clearance=ep_min_clear,
                    max_velocity=ep_max_vel,
                    max_acceleration=ep_max_acc,
                    timeout=is_timeout,
                )
            )

        mean_rew = float(np.mean(rewards))
        std_rew = float(np.std(rewards))
        min_rew = float(np.min(rewards))
        max_rew = float(np.max(rewards))
        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))

        observed_successes = [value for value in successes if value is not None]
        observed_collisions = [value for value in collisions if value is not None]
        succ_rate = (
            float(sum(observed_successes) / len(observed_successes)) if observed_successes else None
        )
        coll_rate = (
            float(sum(observed_collisions) / len(observed_collisions))
            if observed_collisions
            else None
        )
        trunc_rate = float(sum(truncations) / num_episodes)
        time_rate = float(sum(timeouts) / num_episodes)
        mean_path_eff = float(np.mean(path_efficiencies)) if path_efficiencies else None

        obs_coll_count = int(sum(obstacle_collisions))
        bound_coll_count = int(sum(boundary_collisions))
        obs_coll_rate = float(obs_coll_count / num_episodes)
        bound_coll_rate = float(bound_coll_count / num_episodes)

        mean_path_len = float(np.mean(path_lengths)) if path_lengths else None
        std_path_len = float(np.std(path_lengths)) if path_lengths else None
        mean_straight_dist = float(np.mean(straight_line_dists)) if straight_line_dists else None
        mean_path_eff = float(np.mean(path_efficiencies)) if path_efficiencies else None
        mean_min_clear = float(np.mean(min_clearances)) if min_clearances else None
        mean_max_vel = float(np.mean(max_velocities)) if max_velocities else None
        mean_max_acc = float(np.mean(max_accelerations)) if max_accelerations else None

        return EvaluationMetrics(
            episodes=num_episodes,
            mean_reward=mean_rew,
            std_reward=std_rew,
            min_reward=min_rew,
            max_reward=max_rew,
            success_rate=succ_rate,
            collision_rate=coll_rate,
            truncation_rate=time_rate,
            timeout_rate=time_rate,
            mean_episode_length=mean_len,
            std_episode_length=std_len,
            mean_path_length=mean_path_len,
            std_path_length=std_path_len,
            mean_straight_line_distance=mean_straight_dist,
            mean_path_efficiency=mean_path_eff,
            mean_min_obstacle_clearance=mean_min_clear,
            mean_max_velocity=mean_max_vel,
            mean_max_acceleration=mean_max_acc,
            obstacle_collision_rate=obs_coll_rate,
            boundary_collision_rate=bound_coll_rate,
            obstacle_collision_count=obs_coll_count,
            boundary_collision_count=bound_coll_count,
            additional_metrics={
                "all_rewards": rewards,
                "all_lengths": lengths,
                "all_path_lengths": [record.path_length for record in self.last_episode_records],
                "deterministic": deterministic,
                "base_seed": base_seed,
                "truncation_count": int(sum(truncations)),
                "timeout_rate": trunc_rate,
                "split": split,
                "seeds": list(seeds) if seeds is not None else None,
                "mean_path_length": mean_path_len,
                "path_length": mean_path_len,
                "std_path_length": std_path_len,
                "straight_line_distance": mean_straight_dist,
                "mean_straight_line_distance": mean_straight_dist,
                "path_efficiency": mean_path_eff,
                "mean_path_efficiency": mean_path_eff,
                "min_obstacle_clearance": mean_min_clear,
                "mean_min_obstacle_clearance": mean_min_clear,
                "max_velocity": mean_max_vel,
                "mean_max_velocity": mean_max_vel,
                "max_acceleration": mean_max_acc,
                "mean_max_acceleration": mean_max_acc,
                "obstacle_collision_count": obs_coll_count,
                "boundary_collision_count": bound_coll_count,
                "obstacle_collision_rate": obs_coll_rate,
                "boundary_collision_rate": bound_coll_rate,
                "truncation_rate": time_rate,
            },
        )

    @staticmethod
    def _position_from_info(info: dict[str, Any]) -> np.ndarray | None:
        position = info.get("position")
        if position is None:
            return None
        try:
            coordinates = np.asarray(position, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if coordinates.ndim != 1 or coordinates.size == 0:
            return None
        if not np.all(np.isfinite(coordinates)):
            return None
        return coordinates

    def evaluate_seeds(
        self,
        seeds: Sequence[int],
        episodes_per_seed: int,
        deterministic: bool = True,
        split: Optional[str] = None,
    ) -> MultiSeedEvaluationResult:
        """Evaluate a policy independently for each explicit seed group.

        ``seeds`` are evaluation group seeds: the statistical grouping unit
        for cross-seed summaries. Each group runs ``episodes_per_seed``
        episodes whose actual ``env.reset`` seeds are derived independently of
        ``episodes_per_seed``. Custom seeds use injective Cantor pairing;
        configured finite splits use a split-bounded SHA-256 mapping.

        Duplicate seeds are rejected because repeated entries do not represent
        independent test conditions and would over-weight that environment.
        """
        seed_values = list(seeds)
        if not seed_values:
            raise ValueError("Evaluation seeds must not be empty.")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_values):
            raise ValueError("Evaluation seeds must be integers.")
        if any(seed < 0 for seed in seed_values):
            raise ValueError("Evaluation seeds must be non-negative.")
        if len(set(seed_values)) != len(seed_values):
            raise ValueError("Evaluation seeds must be unique; duplicate seeds are not allowed.")
        if isinstance(episodes_per_seed, bool) or not isinstance(episodes_per_seed, int):
            raise ValueError("episodes_per_seed must be an integer.")
        if episodes_per_seed <= 0:
            raise ValueError(f"episodes_per_seed must be positive, got {episodes_per_seed}.")
        if not isinstance(deterministic, bool):
            raise ValueError("deterministic must be a boolean.")

        all_records: list[EpisodeEvaluationRecord] = []
        seed_summaries: list[SeedEvaluationSummary] = []
        for seed in seed_values:
            reset_seeds = [
                derive_episode_reset_seed(seed, index, split=split)
                for index in range(episodes_per_seed)
            ]
            metrics = self.evaluate(
                num_episodes=episodes_per_seed,
                deterministic=deterministic,
                seeds=reset_seeds,
                split=split,
            )
            records = [
                replace(
                    record,
                    seed=seed,
                    evaluation_group_seed=seed,
                    episode_seed=reset_seeds[index],
                    episode_index=index,
                )
                for index, record in enumerate(self.last_episode_records)
            ]
            all_records.extend(records)
            path_lengths = [
                record.path_length for record in records if record.path_length is not None
            ]
            seed_summaries.append(
                SeedEvaluationSummary(
                    seed=seed,
                    episodes=metrics.episodes,
                    success_rate=metrics.success_rate,
                    collision_rate=metrics.collision_rate,
                    truncation_rate=metrics.truncation_rate,
                    mean_reward=metrics.mean_reward,
                    std_reward=metrics.std_reward,
                    mean_episode_length=metrics.mean_episode_length,
                    std_episode_length=metrics.std_episode_length,
                    path_length=(
                        float(math.fsum(path_lengths) / len(path_lengths)) if path_lengths else None
                    ),
                )
            )

        metric_names = (
            "mean_reward",
            "success_rate",
            "collision_rate",
            "truncation_rate",
            "mean_episode_length",
            "std_reward",
            "std_episode_length",
            "path_length",
        )
        aggregate = {
            name: summarize_seed_values([getattr(summary, name) for summary in seed_summaries])
            for name in metric_names
        }
        self.last_episode_records = all_records
        return MultiSeedEvaluationResult(
            seeds=seed_values,
            episodes_per_seed=episodes_per_seed,
            deterministic=deterministic,
            per_seed=seed_summaries,
            episodes=all_records,
            aggregate=aggregate,
            environment=self.env_name,
        )

    @staticmethod
    def save_multiseed_report(
        result: MultiSeedEvaluationResult,
        json_path: str | Path,
        csv_path: str | Path,
    ) -> tuple[Path, Path]:
        """Write the complete multi-seed result to stable JSON and aggregate CSV."""
        json_target = Path(json_path)
        csv_target = Path(csv_path)
        json_target.parent.mkdir(parents=True, exist_ok=True)
        csv_target.parent.mkdir(parents=True, exist_ok=True)

        with json_target.open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, allow_nan=False)

        columns = [
            "metric",
            "mean",
            "std",
            "ci95_lower",
            "ci95_upper",
            "sample_count",
            "seed_count",
            "episodes_per_seed",
            "total_episodes",
        ]
        with csv_target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for metric, statistics in result.aggregate.items():
                writer.writerow(
                    {
                        "metric": metric,
                        **statistics.to_dict(),
                        "seed_count": len(result.seeds),
                        "episodes_per_seed": result.episodes_per_seed,
                        "total_episodes": result.total_episodes,
                    }
                )
        return json_target, csv_target

    @staticmethod
    def save_report(
        metrics: EvaluationMetrics,
        output_path: str | Path,
    ) -> Path:
        """Serialize evaluation metrics to JSON."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "episodes": metrics.episodes,
            "mean_reward": round(metrics.mean_reward, 2),
            "std_reward": round(metrics.std_reward, 2),
            "min_reward": round(metrics.min_reward, 2),
            "max_reward": round(metrics.max_reward, 2),
            "success_rate": round(metrics.success_rate, 4)
            if metrics.success_rate is not None
            else None,
            "collision_rate": round(metrics.collision_rate, 4)
            if metrics.collision_rate is not None
            else None,
            "truncation_rate": round(metrics.truncation_rate, 4)
            if metrics.truncation_rate is not None
            else None,
            "timeout_rate": round(metrics.timeout_rate, 4)
            if metrics.timeout_rate is not None
            else None,
            "mean_episode_length": round(metrics.mean_episode_length, 2),
            "std_episode_length": round(metrics.std_episode_length, 2),
            "mean_path_length": round(metrics.mean_path_length, 2)
            if metrics.mean_path_length is not None
            else None,
            "std_path_length": round(metrics.std_path_length, 2)
            if metrics.std_path_length is not None
            else None,
            "mean_straight_line_distance": round(metrics.mean_straight_line_distance, 2)
            if metrics.mean_straight_line_distance is not None
            else None,
            "mean_path_efficiency": round(metrics.mean_path_efficiency, 4)
            if metrics.mean_path_efficiency is not None
            else None,
            "mean_min_obstacle_clearance": round(metrics.mean_min_obstacle_clearance, 2)
            if metrics.mean_min_obstacle_clearance is not None
            else None,
            "mean_max_velocity": round(metrics.mean_max_velocity, 2)
            if metrics.mean_max_velocity is not None
            else None,
            "mean_max_acceleration": round(metrics.mean_max_acceleration, 2)
            if metrics.mean_max_acceleration is not None
            else None,
            "obstacle_collision_rate": round(metrics.obstacle_collision_rate, 4)
            if metrics.obstacle_collision_rate is not None
            else None,
            "boundary_collision_rate": round(metrics.boundary_collision_rate, 4)
            if metrics.boundary_collision_rate is not None
            else None,
            "obstacle_collision_count": metrics.obstacle_collision_count,
            "boundary_collision_count": metrics.boundary_collision_count,
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return target

    @staticmethod
    def save_csv_report(
        metrics: EvaluationMetrics,
        output_path: str | Path,
    ) -> Path:
        """Serialize standardized evaluation metrics to CSV."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        std_metrics = StandardizedExperimentMetrics.from_rl_metrics(metrics)
        row = std_metrics.to_csv_dict()

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        return target

    def close(self) -> None:
        """Close evaluation environment."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()


def evaluate_random_policy(
    env: Optional[gym.Env] = None,
    num_episodes: int = 20,
    base_seed: Optional[int] = 42,
) -> EvaluationMetrics:
    """Evaluate an untrained uniform-random action baseline policy under controlled seeds.

    Args:
        env: Optional Gymnasium environment instance (defaults to standard DroneNavigation3DEnv).
        num_episodes: Total evaluation episodes to run.
        base_seed: Deterministic base seed.

    Returns:
        EvaluationMetrics containing empirical benchmark metrics.
    """
    close_env = False
    if env is None:
        env = DroneNavigation3DEnv()
        close_env = True
    try:
        policy = RandomPolicy(action_space=env.action_space, seed=base_seed)
        evaluator = Evaluator(algorithm=policy, env=env)
        return evaluator.evaluate(
            num_episodes=num_episodes,
            deterministic=False,
            base_seed=base_seed,
        )
    finally:
        if close_env:
            env.close()


def evaluate_ppo_policy(
    algorithm: BaseAlgorithm,
    env: Optional[gym.Env] = None,
    num_episodes: int = 20,
    base_seed: Optional[int] = 42,
    deterministic: bool = True,
) -> EvaluationMetrics:
    """Evaluate a trained PPO policy under controlled benchmark seeds.

    Args:
        algorithm: Initialized or loaded PPOAlgorithm wrapper.
        env: Optional Gymnasium environment instance (defaults to standard DroneNavigation3DEnv).
        num_episodes: Total evaluation episodes to run.
        base_seed: Deterministic base seed.
        deterministic: Whether to use deterministic mode.

    Returns:
        EvaluationMetrics containing empirical benchmark metrics.
    """
    close_env = False
    if env is None:
        env = DroneNavigation3DEnv()
        close_env = True
    try:
        evaluator = Evaluator(algorithm=algorithm, env=env)
        return evaluator.evaluate(
            num_episodes=num_episodes,
            deterministic=deterministic,
            base_seed=base_seed,
        )
    finally:
        if close_env:
            env.close()


def compare_policies(
    ppo_algorithm: Optional[BaseAlgorithm] = None,
    random_policy: Optional[BaseAlgorithm] = None,
    env: Optional[gym.Env] = None,
    num_episodes: int = 20,
    base_seed: Optional[int] = 42,
    split: Optional[str] = None,
    algorithm: Optional[BaseAlgorithm] = None,
) -> Dict[str, EvaluationMetrics]:
    """Execute head-to-head evaluation between trained policy and Random baseline under identical seeds."""
    target_algo = algorithm if algorithm is not None else ppo_algorithm
    if target_algo is None:
        raise ValueError("Must provide either algorithm or ppo_algorithm.")

    close_env = False
    if env is None:
        env = DroneNavigation3DEnv()
        close_env = True
    try:
        if random_policy is None:
            random_policy = RandomPolicy(action_space=env.action_space, seed=base_seed)

        from adaptive_rl.algorithms.sac import SACAlgorithm

        algo_name = "SAC" if isinstance(target_algo, SACAlgorithm) else "PPO"

        target_eval = Evaluator(algorithm=target_algo, env=env)
        target_metrics = target_eval.evaluate(
            num_episodes=num_episodes,
            deterministic=True,
            base_seed=base_seed if split is None else None,
            split=split,
        )

        rand_eval = Evaluator(algorithm=random_policy, env=env)
        rand_metrics = rand_eval.evaluate(
            num_episodes=num_episodes,
            deterministic=False,
            base_seed=base_seed if split is None else None,
            split=split,
        )

        return {
            algo_name: target_metrics,
            "Random Policy": rand_metrics,
        }
    finally:
        if close_env:
            env.close()


def evaluate_planner(
    planner: Optional[Any] = None,
    env: Optional[gym.Env] = None,
    num_episodes: int = 20,
    base_seed: Optional[int] = 42,
    split: Optional[str] = None,
) -> PlannerEvaluationMetrics:
    """Benchmark AStar3DPlanner on procedural drone navigation environments.

    Evaluates the classical 3D A* planner using the exact same procedural
    environment seeds as RL policies. For each seed, start, goal, bounds,
    and obstacles are extracted and passed to the planner.

    Planning time (ms) measures the complete end-to-end wall-clock duration of
    the planning query (input validation, obstacle clearance, A* graph search,
    and path reconstruction).

    Parameters
    ----------
    planner: Optional[AStar3DPlanner]
        Motion planner instance. If None, instantiates AStar3DPlanner with defaults.
    env: Optional[gym.Env]
        Drone navigation environment. If None, instantiates a default DroneNavigation3DEnv.
    num_episodes: int
        Number of evaluation benchmark episodes.
    base_seed: Optional[int]
        Base seed for procedural environment generation.
    split: Optional[str]
        Environment dataset split ('train' or 'test'). When set, procedural seeds
        are drawn from the standardized split range.

    Returns
    -------
    PlannerEvaluationMetrics
        Aggregated benchmark metrics including success rate, collision rate,
        mean planning time in milliseconds, path length, and path efficiency.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")

    import math

    from adaptive_rl.planners.astar3d import (
        AStar3DPlanner,
        check_segment_collision,
        compute_path_min_obstacle_clearance,
    )

    if planner is None:
        planner = AStar3DPlanner(resolution=0.5, connectivity=26)

    close_env = False
    if env is None:
        env = DroneNavigation3DEnv()
        close_env = True

    try:
        episode_records: List[Dict[str, Any]] = []
        planning_times_ms: List[float] = []
        successes: List[bool] = []
        collisions: List[bool] = []
        path_lengths: List[float] = []
        straight_dists: List[float] = []
        path_efficiencies: List[float] = []
        min_clearances: List[float] = []

        unwrapped = getattr(env, "unwrapped", env)
        bounds = getattr(unwrapped, "bounds", (30.0, 30.0, 15.0))
        collision_radius = getattr(unwrapped, "collision_radius", 0.8)
        target_radius = getattr(unwrapped, "target_radius", 1.5)

        if split is not None:
            from adaptive_rl.evaluation.generalization import get_split_seeds

            seeds: Sequence[Optional[int]] = get_split_seeds(split, num_episodes=num_episodes)
        else:
            seeds = [
                (base_seed + ep) if base_seed is not None else None for ep in range(num_episodes)
            ]

        for ep in range(num_episodes):
            seed = seeds[ep]
            reset_options = {"split": split} if split is not None else None
            obs, info = env.reset(seed=seed, options=reset_options)
            info = info or {}

            start = info.get("position")
            if start is None:
                start = getattr(
                    unwrapped,
                    "_position",
                    getattr(unwrapped, "default_start", np.array([5.0, 5.0, 5.0])),
                )
            start = np.asarray(start, dtype=np.float64)

            goal = info.get("goal")
            if goal is None:
                goal = getattr(
                    unwrapped,
                    "_goal",
                    getattr(unwrapped, "default_goal", np.array([25.0, 25.0, 10.0])),
                )
            goal = np.asarray(goal, dtype=np.float64)

            obstacles = getattr(unwrapped, "_obstacles", getattr(unwrapped, "obstacles", []))

            plan_result = planner.plan_detailed(
                start_pos=start,
                goal_pos=goal,
                bounds=bounds,
                obstacles=obstacles,
                collision_radius=collision_radius,
            )

            planning_times_ms.append(plan_result.planning_time_ms)
            path = plan_result.path

            is_success = False
            is_collision = False
            path_len: Optional[float] = None
            straight_dist: Optional[float] = float(np.linalg.norm(goal - start))
            path_eff: Optional[float] = None
            min_clear: Optional[float] = None

            if path is not None and len(path) >= 2:
                # 1. Success condition: path reaches within target_radius of goal
                # AND does not violate collision constraints
                dist_to_goal = float(np.linalg.norm(path[-1] - goal))
                reaches_goal = dist_to_goal <= target_radius

                # 2. Collision condition: every segment must satisfy clearance
                collided = False
                for p_a, p_b in zip(path[:-1], path[1:]):
                    col, _ = check_segment_collision(
                        p_a,
                        p_b,
                        bounds=bounds,
                        obstacles=obstacles,
                        collision_radius=collision_radius,
                    )
                    if col:
                        collided = True
                        break

                is_collision = collided
                is_success = reaches_goal and (not is_collision)

                # 3. Standard trajectory quality metrics
                traj_metrics = compute_trajectory_metrics(
                    positions=path,
                    goal=goal,
                    obstacles=obstacles,
                )
                path_len = traj_metrics["path_length"]
                straight_dist = traj_metrics["straight_line_distance"]
                path_eff = traj_metrics["path_efficiency"]
                min_clear = compute_path_min_obstacle_clearance(
                    path=path,
                    obstacles=obstacles,
                    collision_radius=collision_radius,
                )
            else:
                is_success = False
                is_collision = False

            successes.append(is_success)
            collisions.append(is_collision)
            if path_len is not None:
                path_lengths.append(path_len)
            if straight_dist is not None:
                straight_dists.append(straight_dist)
            if path_eff is not None:
                path_efficiencies.append(path_eff)
            if min_clear is not None and math.isfinite(min_clear):
                min_clearances.append(min_clear)

            episode_records.append(
                {
                    "episode_index": ep,
                    "seed": seed,
                    "success": is_success,
                    "collision": is_collision,
                    "planning_time_ms": round(plan_result.planning_time_ms, 2),
                    "nodes_expanded": plan_result.nodes_expanded,
                    "status": plan_result.status,
                    "path_length": round(path_len, 2) if path_len is not None else None,
                    "straight_line_distance": round(straight_dist, 2)
                    if straight_dist is not None
                    else None,
                    "path_efficiency": round(path_eff, 4) if path_eff is not None else None,
                    "min_obstacle_clearance": round(min_clear, 2)
                    if min_clear is not None
                    else None,
                    "waypoints_count": len(path) if path is not None else 0,
                }
            )

        n_ep = len(successes)
        mean_time_ms = float(np.mean(planning_times_ms)) if planning_times_ms else 0.0
        mean_path = float(np.mean(path_lengths)) if path_lengths else None
        std_path = float(np.std(path_lengths)) if path_lengths else None
        mean_straight = float(np.mean(straight_dists)) if straight_dists else None
        mean_eff = float(np.mean(path_efficiencies)) if path_efficiencies else None
        mean_clear = float(np.mean(min_clearances)) if min_clearances else None

        succ_rate = sum(successes) / n_ep if n_ep > 0 else 0.0
        coll_rate = sum(collisions) / n_ep if n_ep > 0 else 0.0

        return PlannerEvaluationMetrics(
            episodes=n_ep,
            success_rate=round(succ_rate, 4),
            collision_rate=round(coll_rate, 4),
            mean_planning_time_ms=round(mean_time_ms, 2),
            mean_planning_time=round(mean_time_ms / 1000.0, 5),
            mean_path_length=round(mean_path, 2) if mean_path is not None else None,
            std_path_length=round(std_path, 2) if std_path is not None else None,
            mean_straight_line_distance=round(mean_straight, 2)
            if mean_straight is not None
            else None,
            mean_path_efficiency=round(mean_eff, 4) if mean_eff is not None else None,
            mean_min_obstacle_clearance=round(mean_clear, 2) if mean_clear is not None else None,
            episode_records=episode_records,
            additional_metrics={
                "mean_planning_time_ms": round(mean_time_ms, 2),
                "path_efficiency": round(mean_eff, 4) if mean_eff is not None else None,
                "mean_path_efficiency": round(mean_eff, 4) if mean_eff is not None else None,
                "straight_line_distance": round(mean_straight, 2)
                if mean_straight is not None
                else None,
                "mean_straight_line_distance": round(mean_straight, 2)
                if mean_straight is not None
                else None,
                "min_obstacle_clearance": round(mean_clear, 2) if mean_clear is not None else None,
                "mean_min_obstacle_clearance": round(mean_clear, 2)
                if mean_clear is not None
                else None,
                "planning_time": round(mean_time_ms / 1000.0, 5),
                "feasibility_type": "geometric",
            },
        )
    finally:
        if close_env:
            env.close()


def compare_with_planner(
    ppo_algorithm: BaseAlgorithm,
    planner: Optional[Any] = None,
    random_policy: Optional[BaseAlgorithm] = None,
    env: Optional[gym.Env] = None,
    num_episodes: int = 20,
    base_seed: Optional[int] = 42,
    output_path: Optional[str | Path] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute fair head-to-head comparison across PPO, Classical Planner, and Random Policy.

    All three methods are evaluated on the exact same procedural environment seeds.
    Exports structured benchmark comparison to output_path if provided.
    """
    close_env = False
    if env is None:
        env = DroneNavigation3DEnv()
        close_env = True

    try:
        from adaptive_rl.planners.astar3d import AStar3DPlanner

        effective_base_seed = base_seed if split is None else None
        if planner is None:
            planner = AStar3DPlanner(resolution=0.5, connectivity=26)
        if random_policy is None:
            random_policy = RandomPolicy(action_space=env.action_space, seed=effective_base_seed)

        # 1. PPO Policy evaluation
        ppo_eval = Evaluator(algorithm=ppo_algorithm, env=env)
        ppo_metrics = ppo_eval.evaluate(
            num_episodes=num_episodes,
            deterministic=True,
            base_seed=effective_base_seed,
            split=split,
        )

        # 2. Random Policy baseline evaluation
        rand_eval = Evaluator(algorithm=random_policy, env=env)
        rand_metrics = rand_eval.evaluate(
            num_episodes=num_episodes,
            deterministic=False,
            base_seed=effective_base_seed,
            split=split,
        )

        # 3. Classical A* Planner evaluation
        planner_metrics = evaluate_planner(
            planner=planner,
            env=env,
            num_episodes=num_episodes,
            base_seed=effective_base_seed,
            split=split,
        )

        # Build comparison summary
        seeds_used: List[Optional[int]]
        if split is not None:
            from adaptive_rl.evaluation.generalization import get_split_seeds

            seeds_used = list(get_split_seeds(split, num_episodes=num_episodes))
        else:
            seeds_used = [
                (base_seed + i) if base_seed is not None else None for i in range(num_episodes)
            ]

        comparison_data: Dict[str, Any] = {
            "methodology_note": (
                "A* planner finds geometrically valid collision-free paths through spatial lattice search "
                "without simulating drone dynamics. PPO and Random policies execute control actions in closed-loop "
                "simulation subject to drone inertia, actuation limits, and environment dynamics."
            ),
            "evaluation_config": {
                "num_episodes": num_episodes,
                "base_seed": effective_base_seed,
                "seeds": seeds_used,
                "split": split,
            },
            "planner_config": {
                "name": "AStar3DPlanner",
                "resolution": planner.resolution,
                "connectivity": planner.connectivity,
                "max_iterations": planner.max_iterations,
                "tolerance": planner.tolerance,
                "feasibility_type": "geometric",
            },
            "summary": {
                "PPO": {
                    "feasibility_type": "dynamic",
                    "success_rate": round(ppo_metrics.success_rate, 4)
                    if ppo_metrics.success_rate is not None
                    else None,
                    "collision_rate": round(ppo_metrics.collision_rate, 4)
                    if ppo_metrics.collision_rate is not None
                    else None,
                    "mean_planning_time_ms": None,
                    "mean_path_length": round(ppo_metrics.mean_path_length, 2)
                    if ppo_metrics.mean_path_length is not None
                    else None,
                    "mean_path_efficiency": round(ppo_metrics.mean_path_efficiency, 4)
                    if ppo_metrics.mean_path_efficiency is not None
                    else None,
                    "mean_reward": round(ppo_metrics.mean_reward, 2),
                    "mean_steps": round(ppo_metrics.mean_episode_length, 1),
                },
                "Classical Planner (A*)": {
                    "feasibility_type": "geometric",
                    "feasibility_note": (
                        "A* success denotes finding a geometrically collision-free path. "
                        "The planner path is not executed through the drone dynamics."
                    ),
                    "success_rate": round(planner_metrics.success_rate, 4)
                    if planner_metrics.success_rate is not None
                    else None,
                    "collision_rate": round(planner_metrics.collision_rate, 4)
                    if planner_metrics.collision_rate is not None
                    else None,
                    "mean_planning_time_ms": round(planner_metrics.mean_planning_time_ms, 2),
                    "mean_path_length": round(planner_metrics.mean_path_length, 2)
                    if planner_metrics.mean_path_length is not None
                    else None,
                    "mean_path_efficiency": round(planner_metrics.mean_path_efficiency, 4)
                    if planner_metrics.mean_path_efficiency is not None
                    else None,
                    "mean_reward": None,
                    "mean_steps": None,
                },
                "Random Policy": {
                    "feasibility_type": "dynamic",
                    "success_rate": round(rand_metrics.success_rate, 4)
                    if rand_metrics.success_rate is not None
                    else None,
                    "collision_rate": round(rand_metrics.collision_rate, 4)
                    if rand_metrics.collision_rate is not None
                    else None,
                    "mean_planning_time_ms": None,
                    "mean_path_length": round(rand_metrics.mean_path_length, 2)
                    if rand_metrics.mean_path_length is not None
                    else None,
                    "mean_path_efficiency": round(rand_metrics.mean_path_efficiency, 4)
                    if rand_metrics.mean_path_efficiency is not None
                    else None,
                    "mean_reward": round(rand_metrics.mean_reward, 2),
                    "mean_steps": round(rand_metrics.mean_episode_length, 1),
                },
            },
            "ppo_metrics": ppo_metrics.model_dump(),
            "planner_metrics": planner_metrics.model_dump(),
            "random_policy_metrics": rand_metrics.model_dump(),
        }

        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(comparison_data, f, indent=2)

        return comparison_data
    finally:
        if close_env:
            env.close()


def run_obstacle_density_experiment(
    algorithm: BaseAlgorithm,
    obstacle_counts: Sequence[int] = (4, 6, 8),
    episodes_per_density: int = 10,
    base_seed: int = 42,
    bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0),
    output_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Evaluate a trained agent across varied obstacle densities (e.g. 4, 6, 8 obstacles)."""
    results: List[Dict[str, Any]] = []

    for count in obstacle_counts:
        env = DroneNavigation3DEnv(bounds=bounds, num_obstacles=count)
        try:
            evaluator = Evaluator(algorithm=algorithm, env=env)
            metrics = evaluator.evaluate(
                num_episodes=episodes_per_density,
                deterministic=True,
                base_seed=base_seed,
            )
            entry: Dict[str, Any] = {
                "obstacle_count": int(count),
                "episodes": int(episodes_per_density),
                "success_rate": round(float(metrics.success_rate or 0.0), 4),
                "collision_rate": round(float(metrics.collision_rate or 0.0), 4),
                "mean_reward": round(float(metrics.mean_reward), 2),
                "mean_episode_length": round(float(metrics.mean_episode_length), 2),
            }
            results.append(entry)
        finally:
            env.close()

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def run_reward_ablation_experiment(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Convenience forwarder for reward-function ablation experiment."""
    from adaptive_rl.benchmarking.ablation import run_reward_ablation_experiment as _run

    return _run(*args, **kwargs)


__all__ = [
    "EpisodeEvaluationRecord",
    "Evaluator",
    "compare_policies",
    "compare_with_planner",
    "evaluate_planner",
    "evaluate_ppo_policy",
    "evaluate_random_policy",
    "run_obstacle_density_experiment",
    "run_reward_ablation_experiment",
]
