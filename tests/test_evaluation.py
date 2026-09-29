"""Tests for agent evaluation and JSON report generation."""

import csv
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.environments.drone import DroneNavigation3DEnv, ObstacleSphere3D
from adaptive_rl.evaluation.evaluator import (
    Evaluator,
    compare_policies,
    compare_with_planner,
    derive_episode_reset_seed,
    evaluate_planner,
    evaluate_random_policy,
    run_obstacle_density_experiment,
)
from adaptive_rl.evaluation.metrics import (
    PlannerEvaluationMetrics,
    StandardizedExperimentMetrics,
    compute_trajectory_metrics,
)
from adaptive_rl.evaluation.statistics import (
    student_t_critical_value,
    summarize_descriptive_episodes,
    summarize_seed_values,
)
from adaptive_rl.protocol.seeds import LegacySeedError, ScheduleValidationError, derive_seed


class _SeedOutcomeEnv(gym.Env):
    observation_space = gym.spaces.Box(-1000.0, 1000.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self, *, include_outcomes: bool = True) -> None:
        super().__init__()
        self.include_outcomes = include_outcomes
        self.current_seed = 0
        self.position = np.zeros(2, dtype=np.float64)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_seed = 0 if seed is None else seed
        self.position = np.array([float(self.current_seed), 0.0])
        return np.zeros(1, dtype=np.float32), {"position": self.position.copy()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.position = self.position + np.array([1.0, 0.0])
        info = {"position": self.position.copy()}
        if self.include_outcomes:
            outcome = self.current_seed % 3
            info.update(
                {
                    "success": outcome == 0,
                    "collision": outcome == 1,
                }
            )
            return (
                np.zeros(1, dtype=np.float32),
                float(self.current_seed),
                outcome != 2,
                outcome == 2,
                info,
            )
        return np.zeros(1, dtype=np.float32), float(self.current_seed), True, False, info


class _ZeroPolicy:
    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return np.zeros(1, dtype=np.float32), None


class _ZeroDronePolicy:
    """Zero-action policy for the 3-D drone action space."""

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return np.zeros(3, dtype=np.float32), None


def test_research_evaluation_rejects_legacy_sequential_seeds() -> None:
    evaluator = Evaluator(algorithm=_ZeroPolicy(), env=_SeedOutcomeEnv())  # type: ignore[arg-type]

    with pytest.raises(LegacySeedError, match=r"base_seed \+ episode"):
        evaluator.evaluate(num_episodes=2, base_seed=31001, research_mode=True)

    with pytest.raises(LegacySeedError, match="explicit preregistered"):
        evaluator.evaluate(num_episodes=2, research_mode=True)

    with pytest.raises(ScheduleValidationError, match="do not match"):
        evaluator.evaluate(
            seeds=[31001],
            research_mode=True,
            research_master_seed=31001,
            research_domain="eval_pre",
        )

    result = evaluator.evaluate(
        seeds=[derive_seed(31001, "eval_pre", 1)],
        research_mode=True,
        research_master_seed=31001,
        research_domain="eval_pre",
    )
    assert result.episodes == 1


def test_evaluator_deterministic_evaluation(tmp_path: Path) -> None:
    """Verify evaluation generates deterministic results with fixed seed."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)

    evaluator = Evaluator(algorithm=algo, env=env)
    metrics1 = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=42)
    metrics2 = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=42)

    assert metrics1.episodes == 3
    assert metrics1.mean_reward == metrics2.mean_reward
    assert metrics1.success_rate == metrics2.success_rate
    assert metrics1.collision_rate == metrics2.collision_rate
    assert metrics1.mean_episode_length == metrics2.mean_episode_length

    # Verify JSON report creation and structure
    report_file = tmp_path / "evaluation.json"
    saved = evaluator.save_report(metrics1, report_file)
    assert saved.exists()

    with open(saved, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["episodes"] == 3
    assert "success_rate" in data
    assert "collision_rate" in data
    assert "mean_reward" in data
    assert "mean_episode_length" in data
    env.close()


def test_evaluator_episode_records() -> None:
    """Verify individual episode records are tracked correctly."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = Evaluator(algorithm=algo, env=env)

    evaluator.evaluate(num_episodes=2, deterministic=True, base_seed=10)
    assert len(evaluator.last_episode_records) == 2
    rec = evaluator.last_episode_records[0]
    assert rec.episode_index == 0
    assert rec.seed == 10
    assert isinstance(rec.return_value, float)
    assert isinstance(rec.success, bool)
    assert isinstance(rec.collision, bool)
    env.close()


def test_student_t_statistics_match_analytical_values() -> None:
    assert student_t_critical_value(0.95, 1) == pytest.approx(12.7062047364, rel=1e-9)
    assert student_t_critical_value(0.95, 2) == pytest.approx(4.3026527297, rel=1e-9)
    assert student_t_critical_value(0.95, 5) == pytest.approx(2.5705818356, rel=1e-9)
    assert student_t_critical_value(0.95, 9) == pytest.approx(2.2621571627, rel=1e-9)
    assert student_t_critical_value(0.95, 10) == pytest.approx(2.2281388520, rel=1e-9)
    assert student_t_critical_value(0.95, 30) == pytest.approx(2.0422724563, rel=1e-9)
    assert student_t_critical_value(0.99, 5) == pytest.approx(4.0321429836, rel=1e-9)
    assert student_t_critical_value(0.95, 100) == pytest.approx(1.9839715184, rel=1e-9)
    assert student_t_critical_value(0.95, 1000) == pytest.approx(1.9623390808, rel=1e-9)
    assert student_t_critical_value(0.99, 5) > student_t_critical_value(0.95, 5)
    assert student_t_critical_value(0.95, 5) > student_t_critical_value(0.95, 100)

    stats = summarize_seed_values([1.0, 2.0, 3.0])
    margin = 4.3026527297 / math.sqrt(3.0)
    assert stats.mean == pytest.approx(2.0)
    assert stats.std == pytest.approx(1.0)
    assert stats.ci95_lower == pytest.approx(2.0 - margin)
    assert stats.ci95_upper == pytest.approx(2.0 + margin)
    assert stats.sample_count == 3


def test_student_t_statistics_handle_small_and_constant_samples() -> None:
    one = summarize_seed_values([7.0])
    assert one.mean == 7.0
    assert one.std is None
    assert one.ci95_lower is None
    assert one.ci95_upper is None

    two = summarize_seed_values([0.0, 2.0])
    assert two.mean == 1.0
    assert two.std == pytest.approx(math.sqrt(2.0))
    assert two.ci95_lower is not None and math.isfinite(two.ci95_lower)
    assert two.ci95_upper is not None and math.isfinite(two.ci95_upper)

    constant = summarize_seed_values([3.0, 3.0, 3.0])
    assert constant.mean == 3.0
    assert constant.std == 0.0
    assert constant.ci95_lower == 3.0
    assert constant.ci95_upper == 3.0

    assert summarize_seed_values([None, None]).mean is None
    with pytest.raises(ValueError, match="finite"):
        summarize_seed_values([1.0, float("nan")])


def test_evaluate_seeds_preserves_per_seed_records_and_metrics(tmp_path: Path) -> None:
    env = _SeedOutcomeEnv()
    evaluator = Evaluator(algorithm=_ZeroPolicy(), env=env)  # type: ignore[arg-type]
    requested_seeds = [10, 20]
    result = evaluator.evaluate_seeds(requested_seeds, episodes_per_seed=2)

    assert result.seeds == [10, 20]
    assert requested_seeds == [10, 20]
    assert result.total_episodes == 4
    assert [(record.seed, record.episode_index) for record in result.episodes] == [
        (10, 0),
        (10, 1),
        (20, 0),
        (20, 1),
    ]
    assert [record.episode_seed for record in result.episodes] == [55, 67, 210, 232]
    assert all(record.path_length == 1.0 for record in result.episodes)
    assert [summary.seed for summary in result.per_seed] == requested_seeds
    assert [summary.episodes for summary in result.per_seed] == [2, 2]
    assert [summary.mean_reward for summary in result.per_seed] == [61.0, 221.0]
    assert result.per_seed[0].success_rate == pytest.approx(0.0)
    assert result.per_seed[0].collision_rate == pytest.approx(1.0)
    assert result.per_seed[0].truncation_rate == pytest.approx(0.0)
    assert result.per_seed[0].mean_episode_length == 1.0
    assert result.per_seed[0].path_length == 1.0
    assert result.aggregate["mean_reward"].mean == pytest.approx(141.0)
    assert result.aggregate["mean_reward"].std == pytest.approx(math.sqrt(12800.0))

    json_path = tmp_path / "evaluation_multiseed.json"
    csv_path = tmp_path / "evaluation_multiseed.csv"
    saved_json, saved_csv = evaluator.save_multiseed_report(result, json_path, csv_path)
    assert saved_json.is_file()
    assert saved_csv.is_file()
    document = json.loads(saved_json.read_text(encoding="utf-8"))
    json.dumps(document, allow_nan=False)
    assert document["metadata"] == {
        "seeds": [10, 20],
        "evaluation_group_seeds": [10, 20],
        "seed_count": 2,
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
        "episodes_per_seed": 2,
        "total_episodes": 4,
        "deterministic": True,
        "environment": "drone",
        "confidence_interval": (
            "two-sided 95% Student's t interval across seed summaries; "
            "sample standard deviation; unavailable when fewer than two values exist"
        ),
        "duplicate_seed_policy": "rejected",
    }
    assert len(document["episodes"]) == 4
    assert len(document["per_seed"]) == 2
    assert document["aggregate"]["mean_reward"]["sample_count"] == 2
    assert document["episodes"][0]["episode_seed"] == 55
    assert document["episodes"][0]["path_length"] == 1.0

    expected_headers = [
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
    with saved_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == expected_headers
    assert len(rows) == len(result.aggregate)
    assert rows[0]["metric"] == "mean_reward"
    assert float(rows[0]["mean"]) == pytest.approx(141.0)
    assert rows[0]["seed_count"] == "2"
    assert rows[0]["episodes_per_seed"] == "2"
    assert rows[0]["total_episodes"] == "4"
    evaluator.close()


def test_evaluate_seeds_is_deterministic_and_rejects_invalid_inputs() -> None:
    evaluator = Evaluator(algorithm=_ZeroPolicy(), env=_SeedOutcomeEnv())  # type: ignore[arg-type]
    first = evaluator.evaluate_seeds([7, 13], episodes_per_seed=2, deterministic=True)
    first_data = first.to_dict()
    second = evaluator.evaluate_seeds([7, 13], episodes_per_seed=2, deterministic=True)
    assert second.to_dict() == first_data

    with pytest.raises(ValueError, match="must not be empty"):
        evaluator.evaluate_seeds([], episodes_per_seed=1)
    with pytest.raises(ValueError, match="unique"):
        evaluator.evaluate_seeds([7, 7], episodes_per_seed=1)
    with pytest.raises(ValueError, match="positive"):
        evaluator.evaluate_seeds([7], episodes_per_seed=0)
    with pytest.raises(ValueError, match="non-negative"):
        evaluator.evaluate_seeds([-1], episodes_per_seed=1)
    evaluator.close()


def test_evaluation_group_reset_seeds_are_independent_of_episode_count() -> None:
    observed: dict[int, dict[int, list[int]]] = {}
    for episodes_per_seed in (1, 5, 20, 100):
        env = _SeedOutcomeEnv()
        reset_seeds: list[int] = []
        original_reset = env.reset

        def record_reset(*, seed: int | None = None, options: dict | None = None):
            reset_seeds.append(seed if seed is not None else -1)
            return original_reset(seed=seed, options=options)

        env.reset = record_reset  # type: ignore[method-assign]
        evaluator = Evaluator(algorithm=_ZeroPolicy(), env=env)  # type: ignore[arg-type]
        result = evaluator.evaluate_seeds([11, 13], episodes_per_seed=episodes_per_seed)
        observed[episodes_per_seed] = {
            group_seed: reset_seeds[index * episodes_per_seed : (index + 1) * episodes_per_seed]
            for index, group_seed in enumerate((11, 13))
        }

        group_11_records = {
            record.episode_index: record
            for record in result.episodes
            if record.evaluation_group_seed == 11
        }
        assert group_11_records[0].episode_reset_seed == derive_episode_reset_seed(11, 0)
        if episodes_per_seed > 1:
            assert group_11_records[1].episode_reset_seed == derive_episode_reset_seed(11, 1)
        assert len(result.episodes) == 2 * episodes_per_seed
        evaluator.close()

    for group_seed in (11, 13):
        reference = observed[100][group_seed]
        for episode_count in observed:
            assert (
                observed[episode_count][group_seed][: min(episode_count, 5)]
                == reference[: min(episode_count, 5)]
            )
    assert observed[1][11][0] == observed[5][11][0] == observed[20][11][0] == 66
    assert observed[1][13][0] == observed[5][13][0] == observed[20][13][0]
    assert observed[5][11][1] == observed[20][11][1] == observed[100][11][1] == 79

    seeds_group_11 = {derive_episode_reset_seed(11, index) for index in range(100)}
    seeds_group_13 = {derive_episode_reset_seed(13, index) for index in range(100)}
    assert seeds_group_11.isdisjoint(seeds_group_13)


def test_split_seed_mapping_is_stable_and_respects_partition() -> None:
    train_seed = derive_episode_reset_seed(11, 3, split="train")
    test_seed = derive_episode_reset_seed(1001, 3, split="test")
    assert train_seed == derive_episode_reset_seed(11, 3, split="train")
    assert 0 <= train_seed < 1000
    assert 1000 <= test_seed < 1200


def test_evaluate_seeds_preserves_unavailable_optional_metrics() -> None:
    evaluator = Evaluator(
        algorithm=_ZeroPolicy(),
        env=_SeedOutcomeEnv(include_outcomes=False),  # type: ignore[arg-type]
    )
    result = evaluator.evaluate_seeds([2, 5], episodes_per_seed=1)
    assert all(summary.success_rate is None for summary in result.per_seed)
    assert all(summary.collision_rate is None for summary in result.per_seed)
    assert result.aggregate["success_rate"].mean is None
    evaluator.close()


def test_evaluator_uses_truncated_signal_for_timeout_metrics() -> None:
    terminating_env = _SeedOutcomeEnv(include_outcomes=False)
    terminating_env.max_steps = 1
    evaluator = Evaluator(
        algorithm=_ZeroPolicy(),
        env=terminating_env,  # type: ignore[arg-type]
    )

    terminated = evaluator.evaluate(num_episodes=1, base_seed=7)
    assert terminated.timeout_rate == 0.0
    assert terminated.truncation_rate == 0.0
    assert evaluator.last_episode_records[0].truncated is False
    assert evaluator.last_episode_records[0].timeout is False
    evaluator.close()

    truncating_env = _SeedOutcomeEnv(include_outcomes=True)
    evaluator = Evaluator(
        algorithm=_ZeroPolicy(),
        env=truncating_env,  # type: ignore[arg-type]
    )
    truncated = evaluator.evaluate(num_episodes=1, base_seed=11)
    assert truncated.timeout_rate == 1.0
    assert truncated.truncation_rate == 1.0
    assert evaluator.last_episode_records[0].truncated is True
    assert evaluator.last_episode_records[0].timeout is True
    evaluator.close()


def test_evaluate_random_policy() -> None:
    """Verify uniform-random policy evaluation baseline executes and returns valid metrics."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=2)
    metrics = evaluate_random_policy(env=env, num_episodes=3, base_seed=42)

    assert metrics.episodes == 3
    assert isinstance(metrics.mean_reward, float)
    assert 0.0 <= (metrics.success_rate or 0.0) <= 1.0
    assert 0.0 <= (metrics.collision_rate or 0.0) <= 1.0
    assert metrics.mean_episode_length > 0
    env.close()


def test_compare_policies() -> None:
    """Verify head-to-head comparison between PPO and Random baseline."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)

    comparison = compare_policies(ppo_algorithm=algo, env=env, num_episodes=2, base_seed=42)
    assert "PPO" in comparison
    assert "Random Policy" in comparison
    assert comparison["PPO"].episodes == 2
    assert comparison["Random Policy"].episodes == 2
    env.close()


def test_run_obstacle_density_experiment(tmp_path: Path) -> None:
    """Verify obstacle-density experiment runs across varied obstacle counts."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    output_file = tmp_path / "density_exp.json"

    results = run_obstacle_density_experiment(
        algorithm=algo,
        obstacle_counts=(2, 4),
        episodes_per_density=2,
        base_seed=42,
        bounds=(20.0, 20.0, 10.0),
        output_path=output_file,
    )

    assert len(results) == 2
    assert results[0]["obstacle_count"] == 2
    assert results[1]["obstacle_count"] == 4
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert len(saved_data) == 2
    assert "success_rate" in saved_data[0]
    assert "collision_rate" in saved_data[0]
    env.close()


def test_trajectory_metrics_stationary() -> None:
    """Verify stationary trajectory produces 0 path length and 0 efficiency."""
    positions = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    goal = [5.0, 2.0, 3.0]
    metrics = compute_trajectory_metrics(positions, goal)

    assert metrics["path_length"] == 0.0
    assert metrics["path_efficiency"] == 0.0
    assert metrics["straight_line_distance"] == pytest.approx(4.0)


def test_trajectory_metrics_straight_line() -> None:
    """Verify perfect straight-line trajectory achieves ~1.0 path efficiency."""
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    goal = [5.0, 0.0, 0.0]
    metrics = compute_trajectory_metrics(positions, goal)

    assert metrics["path_length"] == pytest.approx(5.0)
    assert metrics["straight_line_distance"] == pytest.approx(5.0)
    assert metrics["path_efficiency"] == pytest.approx(1.0)


def test_trajectory_metrics_zigzag() -> None:
    """Verify non-straight trajectory yields efficiency strictly less than 1.0."""
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 2.0, 0.0],
        [4.0, 0.0, 0.0],
    ]
    goal = [4.0, 0.0, 0.0]
    metrics = compute_trajectory_metrics(positions, goal)

    assert metrics["straight_line_distance"] == pytest.approx(4.0)
    assert metrics["path_length"] > 4.0
    assert 0.0 < metrics["path_efficiency"] < 1.0


def test_trajectory_metrics_zero_length_and_empty() -> None:
    """Verify zero-length and single-waypoint trajectories do not raise division errors."""
    # Empty positions
    metrics_empty = compute_trajectory_metrics([], goal=[1.0, 1.0, 1.0])
    assert metrics_empty["path_length"] == 0.0
    assert metrics_empty["path_efficiency"] == 0.0
    assert metrics_empty["straight_line_distance"] == 0.0

    # Single position
    metrics_single = compute_trajectory_metrics([[1.0, 1.0, 1.0]], goal=[4.0, 1.0, 1.0])
    assert metrics_single["path_length"] == 0.0
    assert metrics_single["path_efficiency"] == 0.0
    assert metrics_single["straight_line_distance"] == pytest.approx(3.0)


def test_trajectory_metrics_obstacle_clearance_surface() -> None:
    """Verify closest distance calculation measures to spherical obstacle surface, not center."""
    obs = ObstacleSphere3D(center=np.array([5.0, 5.0, 5.0]), radius=1.5)
    # p1: dist to center = 5.0, surface clearance = 5.0 - 1.5 = 3.5
    # p2: dist to center = 2.0, surface clearance = 2.0 - 1.5 = 0.5
    # p3: dist to center = 6.0, surface clearance = 6.0 - 1.5 = 4.5
    positions = [[0.0, 5.0, 5.0], [3.0, 5.0, 5.0], [11.0, 5.0, 5.0]]
    goal = [11.0, 5.0, 5.0]

    metrics = compute_trajectory_metrics(positions, goal, obstacles=[obs])
    assert metrics["min_obstacle_clearance"] == pytest.approx(0.5)


def test_trajectory_metrics_max_velocity() -> None:
    """Verify maximum velocity computation against known synthetic velocity vectors."""
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    goal = [1.0, 0.0, 0.0]
    velocities = [
        [1.0, 2.0, 2.0],  # norm = 3.0
        [0.0, 4.0, 3.0],  # norm = 5.0
        [2.0, 0.0, 0.0],  # norm = 2.0
    ]
    metrics = compute_trajectory_metrics(positions, goal, velocities=velocities)
    assert metrics["max_velocity"] == pytest.approx(5.0)


def test_trajectory_metrics_max_acceleration() -> None:
    """Verify maximum acceleration computation against known synthetic acceleration vectors."""
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    goal = [1.0, 0.0, 0.0]
    accelerations = [
        [0.0, 0.0, 0.0],  # norm = 0.0
        [1.0, 2.0, 2.0],  # norm = 3.0
        [-2.0, 1.0, 2.0],  # norm = 3.0
    ]
    metrics = compute_trajectory_metrics(positions, goal, accelerations=accelerations)
    assert metrics["max_acceleration"] == pytest.approx(3.0)


def test_obstacle_vs_boundary_collision_separation() -> None:
    """Verify obstacle collisions and boundary collisions are tracked separately."""

    class DummyEnv:
        def __init__(self) -> None:
            self.episode = 0
            self.action_space = None
            self.observation_space = None
            self.obstacles: list[Any] = []

        def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
            self.episode += 1
            return np.zeros(29, dtype=np.float32), {
                "drone_position": np.array([0.0, 0.0, 0.0]),
                "target_position": np.array([5.0, 5.0, 5.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "acceleration": np.array([0.0, 0.0, 0.0]),
                "obstacles": [],
            }

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
            if self.episode == 1:
                # Obstacle collision
                info = {
                    "drone_position": np.array([1.0, 0.0, 0.0]),
                    "target_position": np.array([5.0, 5.0, 5.0]),
                    "velocity": np.array([1.0, 0.0, 0.0]),
                    "acceleration": np.array([0.5, 0.0, 0.0]),
                    "collision": True,
                    "collision_type": "obstacle",
                    "success": False,
                    "obstacles": [],
                }
                return np.zeros(29, dtype=np.float32), -50.0, True, False, info
            elif self.episode == 2:
                # Boundary collision
                info = {
                    "drone_position": np.array([0.0, 10.0, 0.0]),
                    "target_position": np.array([5.0, 5.0, 5.0]),
                    "velocity": np.array([0.0, 2.0, 0.0]),
                    "acceleration": np.array([0.0, 1.0, 0.0]),
                    "collision": True,
                    "collision_type": "boundary_y",
                    "success": False,
                    "obstacles": [],
                }
                return np.zeros(29, dtype=np.float32), -50.0, True, False, info
            else:
                # Success
                info = {
                    "drone_position": np.array([5.0, 5.0, 5.0]),
                    "target_position": np.array([5.0, 5.0, 5.0]),
                    "velocity": np.array([0.1, 0.0, 0.0]),
                    "acceleration": np.array([0.0, 0.0, 0.0]),
                    "collision": False,
                    "collision_type": None,
                    "success": True,
                    "obstacles": [],
                }
                return np.zeros(29, dtype=np.float32), 100.0, True, False, info

        def close(self) -> None:
            pass

    dummy_env = DummyEnv()
    evaluator = Evaluator(algorithm=None, env=dummy_env)  # type: ignore[arg-type]
    metrics = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=1)

    assert metrics.episodes == 3
    assert metrics.collision_rate == pytest.approx(2 / 3)
    assert metrics.obstacle_collision_count == 1
    assert metrics.obstacle_collision_rate == pytest.approx(1 / 3)
    assert metrics.boundary_collision_count == 1
    assert metrics.boundary_collision_rate == pytest.approx(1 / 3)
    assert metrics.success_rate == pytest.approx(1 / 3)


def test_evaluator_trajectory_metrics_and_csv_export(tmp_path: Path) -> None:
    """Verify evaluator produces trajectory metrics and exports to JSON and CSV."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = Evaluator(algorithm=algo, env=env)

    metrics = evaluator.evaluate(num_episodes=2, deterministic=True, base_seed=42)

    # Trajectory metrics populated
    assert metrics.mean_path_length is not None
    assert metrics.mean_straight_line_distance is not None
    assert metrics.mean_path_efficiency is not None
    assert metrics.mean_max_velocity is not None
    assert metrics.mean_max_acceleration is not None
    assert metrics.obstacle_collision_count is not None
    assert metrics.boundary_collision_count is not None
    assert metrics.obstacle_collision_rate is not None
    assert metrics.boundary_collision_rate is not None

    # JSON export
    json_path = tmp_path / "eval.json"
    evaluator.save_report(metrics, json_path)
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "mean_path_length" in data
    assert "mean_path_efficiency" in data
    assert "obstacle_collision_count" in data
    assert "boundary_collision_count" in data

    # CSV export
    csv_path = tmp_path / "eval.csv"
    evaluator.save_csv_report(metrics, csv_path)
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        assert "episode_return" in row
        assert "path_length" in row
        assert "path_efficiency" in row
        assert "obstacle_collision_count" in row
        assert "boundary_collision_count" in row

    env.close()


def test_train_test_seeds_disjoint() -> None:
    """Verify that the train and test seed sets are strictly disjoint (zero intersection)."""
    from adaptive_rl.evaluation.generalization import get_split_seeds

    train_seeds = get_split_seeds("train")
    test_seeds = get_split_seeds("test")

    assert len(train_seeds) > 0
    assert len(test_seeds) > 0
    assert set(train_seeds).isdisjoint(set(test_seeds))
    assert len(set(train_seeds).intersection(set(test_seeds))) == 0


def test_train_test_seed_generation_deterministic() -> None:
    """Verify that train and test seed generation is 100% deterministic and repeatable."""
    from adaptive_rl.evaluation.generalization import get_split_seeds

    train_seeds_1 = get_split_seeds("train", num_episodes=20)
    train_seeds_2 = get_split_seeds("train", num_episodes=20)
    assert train_seeds_1 == train_seeds_2
    assert len(train_seeds_1) == 20

    test_seeds_1 = get_split_seeds("test", num_episodes=20)
    test_seeds_2 = get_split_seeds("test", num_episodes=20)
    assert test_seeds_1 == test_seeds_2
    assert len(test_seeds_1) == 20

    # Prefix consistency
    train_prefix = get_split_seeds("train", num_episodes=10)
    assert train_prefix == train_seeds_1[:10]


def test_same_test_seeds_produce_reproducible_layouts() -> None:
    """Verify that identical test seeds produce identical obstacle layouts."""
    env = DroneNavigation3DEnv(bounds=(30.0, 30.0, 15.0), num_obstacles=4)

    test_seed = 1042
    _, _ = env.reset(seed=test_seed)
    obstacles_run1 = [(obs.center.copy(), obs.radius) for obs in env._obstacles]

    _, _ = env.reset(seed=test_seed)
    obstacles_run2 = [(obs.center.copy(), obs.radius) for obs in env._obstacles]

    assert len(obstacles_run1) == len(obstacles_run2) == 4
    for (c1, r1), (c2, r2) in zip(obstacles_run1, obstacles_run2):
        np.testing.assert_allclose(c1, c2, rtol=1e-6)
        assert r1 == pytest.approx(r2)


def test_train_and_test_representative_layouts_are_distinct() -> None:
    """Verify that representative train layouts and test layouts produce distinct geometry."""
    env = DroneNavigation3DEnv(bounds=(30.0, 30.0, 15.0), num_obstacles=4)

    # Train layout (seed 42)
    env.reset(seed=42)
    train_obstacles = [(obs.center.copy(), obs.radius) for obs in env._obstacles]

    # Test layout (seed 1042)
    env.reset(seed=1042)
    test_obstacles = [(obs.center.copy(), obs.radius) for obs in env._obstacles]

    centers_train = np.array([c for c, _ in train_obstacles])
    centers_test = np.array([c for c, _ in test_obstacles])

    # Obstacle coordinate layouts must not be identical
    assert not np.allclose(centers_train, centers_test)
    env.close()


def test_environment_split_enforcement_and_leakage_prevention() -> None:
    """Verify environment enforces split boundaries and strictly blocks cross-split seeds."""
    # 1. Train environment draws only train seeds
    train_env = DroneNavigation3DEnv(split="train")
    for _ in range(5):
        _, info = train_env.reset()
        assert info["split"] == "train"
        assert 0 <= info["split_seed"] < 1000

    # Train environment rejects test seed
    with pytest.raises(ValueError, match="out of bounds for 'train' split"):
        train_env.reset(seed=1005)

    # 2. Test environment draws only test seeds
    test_env = DroneNavigation3DEnv(split="test")
    for _ in range(5):
        _, info = test_env.reset()
        assert info["split"] == "test"
        assert 1000 <= info["split_seed"] < 1200

    # Test environment rejects train seed
    with pytest.raises(ValueError, match="out of bounds for 'test' split"):
        test_env.reset(seed=5)

    # Invalid split name
    with pytest.raises(ValueError, match="Invalid split"):
        DroneNavigation3DEnv(split="validation")

    train_env.close()
    test_env.close()


def test_generalization_gap_calculation_normal() -> None:
    """Verify standard generalization gap computation: Delta = Train - Test."""
    from adaptive_rl.evaluation.generalization import compute_generalization_gap
    from adaptive_rl.evaluation.metrics import EvaluationMetrics

    train_m = EvaluationMetrics(
        episodes=20,
        mean_reward=80.0,
        std_reward=5.0,
        min_reward=70.0,
        max_reward=90.0,
        success_rate=0.85,
        collision_rate=0.10,
        mean_episode_length=120.0,
    )
    test_m = EvaluationMetrics(
        episodes=20,
        mean_reward=30.0,
        std_reward=10.0,
        min_reward=10.0,
        max_reward=50.0,
        success_rate=0.55,
        collision_rate=0.40,
        mean_episode_length=80.0,
    )

    gap = compute_generalization_gap(train_m, test_m)

    assert gap.success_gap == pytest.approx(0.85 - 0.55)  # +0.30
    assert gap.reward_gap == pytest.approx(80.0 - 30.0)  # +50.0
    assert gap.collision_gap == pytest.approx(0.10 - 0.40)  # -0.30


def test_generalization_gap_edge_cases() -> None:
    """Verify generalization gap calculation under boundary conditions."""
    from adaptive_rl.evaluation.generalization import compute_generalization_gap
    from adaptive_rl.evaluation.metrics import EvaluationMetrics

    # Case 1: 100% train success, 0% test success
    m_100 = EvaluationMetrics(
        episodes=10,
        mean_reward=100.0,
        success_rate=1.0,
        collision_rate=0.0,
        mean_episode_length=50.0,
    )
    m_0 = EvaluationMetrics(
        episodes=10,
        mean_reward=-50.0,
        success_rate=0.0,
        collision_rate=1.0,
        mean_episode_length=20.0,
    )
    gap1 = compute_generalization_gap(m_100, m_0)
    assert gap1.success_gap == pytest.approx(1.0)
    assert gap1.reward_gap == pytest.approx(150.0)

    # Case 2: 0% train success, 100% test success
    gap2 = compute_generalization_gap(m_0, m_100)
    assert gap2.success_gap == pytest.approx(-1.0)
    assert gap2.reward_gap == pytest.approx(-150.0)

    # Case 3: 0% train success, 0% test success
    gap3 = compute_generalization_gap(m_0, m_0)
    assert gap3.success_gap == pytest.approx(0.0)
    assert gap3.reward_gap == pytest.approx(0.0)

    # Case 4: 100% train success, 100% test success
    gap4 = compute_generalization_gap(m_100, m_100)
    assert gap4.success_gap == pytest.approx(0.0)
    assert gap4.reward_gap == pytest.approx(0.0)

    # Case 5: Empty episodes error
    m_empty = EvaluationMetrics(
        episodes=1,  # Pydantic requires >0
        mean_reward=0.0,
        mean_episode_length=0.0,
    )
    object.__setattr__(m_empty, "episodes", 0)
    with pytest.raises(ValueError, match="empty evaluation episodes"):
        compute_generalization_gap(m_empty, m_100)

    # Case 6: Non-finite reward error
    m_nan = EvaluationMetrics(
        episodes=10,
        mean_reward=float("nan"),
        mean_episode_length=10.0,
    )
    with pytest.raises(ValueError, match="Non-finite mean reward"):
        compute_generalization_gap(m_nan, m_100)


def test_rerunning_test_evaluation_is_deterministic() -> None:
    """Verify that re-running test split evaluation produces deterministic identical results."""
    from adaptive_rl.algorithms.random_policy import RandomPolicy

    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    policy = RandomPolicy(action_space=env.action_space, seed=123)
    evaluator = Evaluator(algorithm=policy, env=env)

    # First test split evaluation
    metrics1 = evaluator.evaluate(num_episodes=3, deterministic=True, split="test")
    records1 = list(evaluator.last_episode_records)

    # Second test split evaluation under identical conditions
    # Re-instantiate policy with same seed
    policy2 = RandomPolicy(action_space=env.action_space, seed=123)
    evaluator2 = Evaluator(algorithm=policy2, env=env)
    metrics2 = evaluator2.evaluate(num_episodes=3, deterministic=True, split="test")
    records2 = list(evaluator2.last_episode_records)

    assert metrics1.episodes == metrics2.episodes == 3
    assert metrics1.mean_reward == pytest.approx(metrics2.mean_reward)
    assert metrics1.success_rate == metrics2.success_rate
    assert metrics1.collision_rate == metrics2.collision_rate

    # Seeds evaluated are identical
    seeds1 = [r.seed for r in records1]
    seeds2 = [r.seed for r in records2]
    assert seeds1 == seeds2 == [1000, 1001, 1002]

    env.close()


def test_evaluate_generalization_workflow_and_json_export(tmp_path: Path) -> None:
    """Verify evaluate_generalization runs both splits, computes gaps, and exports JSON."""
    from adaptive_rl.algorithms.random_policy import RandomPolicy
    from adaptive_rl.evaluation.generalization import evaluate_generalization

    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    policy = RandomPolicy(action_space=env.action_space, seed=42)

    json_target = tmp_path / "generalization_benchmark.json"
    result = evaluate_generalization(
        algorithm=policy,
        env=env,
        num_episodes=2,
        deterministic=True,
        output_path=json_target,
    )

    assert json_target.exists()

    # Verify result structure
    assert result.train["episodes"] == 2
    assert result.test["episodes"] == 2
    assert len(result.train["seeds"]) == 2
    assert len(result.test["seeds"]) == 2

    # Verify seed disjointness
    assert set(result.train["seeds"]).isdisjoint(set(result.test["seeds"]))

    # Verify gap fields
    assert "success" in result.generalization_gap
    assert "reward" in result.generalization_gap
    assert isinstance(result.generalization_gap["reward"], float)

    # Verify serialized JSON content
    with open(json_target, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "train" in data
    assert "test" in data
    assert "generalization_gap" in data
    assert data["train"]["seeds"] == [0, 1]
    assert data["test"]["seeds"] == [1000, 1001]
    assert "success" in data["generalization_gap"]
    assert "reward" in data["generalization_gap"]

    env.close()


def test_evaluate_planner_basic() -> None:
    """Verify evaluate_planner generates valid benchmark metrics over procedural seeds."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20, num_obstacles=2)
    planner_metrics = evaluate_planner(env=env, num_episodes=3, base_seed=42)

    assert isinstance(planner_metrics, PlannerEvaluationMetrics)
    assert planner_metrics.episodes == 3
    assert 0.0 <= planner_metrics.success_rate <= 1.0
    assert 0.0 <= planner_metrics.collision_rate <= 1.0
    assert planner_metrics.mean_planning_time_ms >= 0.0
    assert len(planner_metrics.episode_records) == 3

    # Check conversion to StandardizedExperimentMetrics
    std_metrics = StandardizedExperimentMetrics.from_planner_metrics(planner_metrics)
    assert std_metrics.episodes == 3
    assert std_metrics.planning_time == planner_metrics.mean_planning_time
    assert std_metrics.success_rate == planner_metrics.success_rate
    assert std_metrics.collision_rate == planner_metrics.collision_rate

    env.close()


def test_compare_with_planner_structure(tmp_path: Path) -> None:
    """Verify compare_with_planner executes fair 3-way evaluation under identical seeds."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=10, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=32, batch_size=16, seed=42)
    report_target = tmp_path / "evaluation_planner_comparison.json"

    comp_results = compare_with_planner(
        ppo_algorithm=algo,
        env=env,
        num_episodes=2,
        base_seed=100,
        output_path=report_target,
    )

    assert "methodology_note" in comp_results
    assert "evaluation_config" in comp_results
    assert "planner_config" in comp_results
    assert comp_results["planner_config"]["feasibility_type"] == "geometric"
    assert "summary" in comp_results
    assert "PPO" in comp_results["summary"]
    assert comp_results["summary"]["PPO"]["feasibility_type"] == "dynamic"
    assert "Classical Planner (A*)" in comp_results["summary"]
    assert comp_results["summary"]["Classical Planner (A*)"]["feasibility_type"] == "geometric"
    assert "feasibility_note" in comp_results["summary"]["Classical Planner (A*)"]
    assert "Random Policy" in comp_results["summary"]
    assert comp_results["summary"]["Random Policy"]["feasibility_type"] == "dynamic"

    assert comp_results["evaluation_config"]["seeds"] == [100, 101]

    # Verify JSON file structure
    assert report_target.exists()
    with open(report_target, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert "ppo_metrics" in saved_data
    assert "planner_metrics" in saved_data
    assert "random_policy_metrics" in saved_data
    assert saved_data["summary"]["Classical Planner (A*)"]["mean_planning_time_ms"] is not None

    env.close()


def test_from_planner_metrics_preserves_straight_line_and_clearance() -> None:
    """Verify StandardizedExperimentMetrics preserves all planner metrics including clearance and straight line dist."""
    planner_m = PlannerEvaluationMetrics(
        episodes=5,
        success_rate=0.8,
        collision_rate=0.0,
        timeout_rate=0.2,
        mean_planning_time_ms=12.5,
        mean_path_length=24.5,
        mean_path_efficiency=0.92,
        mean_straight_line_distance=22.54,
        mean_min_obstacle_clearance=1.75,
        additional_metrics={
            "straight_line_distance": 22.54,
            "min_obstacle_clearance": 1.75,
            "path_efficiency": 0.92,
        },
    )

    std_m = StandardizedExperimentMetrics.from_planner_metrics(planner_m)

    assert std_m.episodes == 5
    assert std_m.success_rate == 0.8
    assert std_m.collision_rate == 0.0
    assert std_m.path_length == 24.5
    assert std_m.path_efficiency == 0.92
    assert std_m.straight_line_distance == 22.54
    assert std_m.min_obstacle_clearance == 1.75
    assert std_m.planning_time == planner_m.mean_planning_time


def test_evaluate_planner_num_episodes_validation() -> None:
    """Verify evaluate_planner validates num_episodes > 0 explicitly."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=10, num_obstacles=1)

    with pytest.raises(ValueError, match="num_episodes must be positive"):
        evaluate_planner(env=env, num_episodes=0)

    with pytest.raises(ValueError, match="num_episodes must be positive"):
        evaluate_planner(env=env, num_episodes=-5)

    env.close()


def test_evaluate_planner_with_generalization_split() -> None:
    """Verify evaluate_planner operates correctly under generalization train and test splits."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=10, num_obstacles=1)

    # Test split evaluation draws seeds from test range [1000, 1200)
    test_metrics = evaluate_planner(env=env, num_episodes=2, split="test")
    assert test_metrics.episodes == 2
    assert [r["seed"] for r in test_metrics.episode_records] == [1000, 1001]

    # Train split evaluation draws seeds from train range [0, 1000)
    train_metrics = evaluate_planner(env=env, num_episodes=2, split="train")
    assert train_metrics.episodes == 2
    assert [r["seed"] for r in train_metrics.episode_records] == [0, 1]

    env.close()


def test_compare_with_planner_with_split(tmp_path: Path) -> None:
    """Verify compare_with_planner respects dataset split and records split in JSON."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=10, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=32, batch_size=16, seed=42)
    report_target = tmp_path / "split_planner_comp.json"

    comp_results = compare_with_planner(
        ppo_algorithm=algo,
        env=env,
        num_episodes=2,
        split="test",
        output_path=report_target,
    )

    assert comp_results["evaluation_config"]["split"] == "test"
    assert comp_results["evaluation_config"]["seeds"] == [1000, 1001]
    assert report_target.exists()

    env.close()


def test_student_t_critical_value_validates_inputs() -> None:
    for confidence in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            student_t_critical_value(confidence, 5)
    for degrees_of_freedom in (0, -3):
        with pytest.raises(ValueError, match="degrees of freedom must be positive"):
            student_t_critical_value(0.95, degrees_of_freedom)
    for confidence in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            student_t_critical_value(confidence, 5)
    for degrees_of_freedom in (True, 1.5):
        with pytest.raises(ValueError, match="degrees of freedom must be an integer"):
            student_t_critical_value(0.95, degrees_of_freedom)  # type: ignore[arg-type]


def test_summarize_descriptive_episodes_pools_every_episode() -> None:
    metrics = summarize_descriptive_episodes(
        rewards=[1.0, 2.0, 3.0],
        episode_lengths=[4.0, 6.0, 8.0],
        successes=[True, False, None],
        collisions=[None, None, None],
        truncations=[False, True, True],
    )
    assert metrics.episodes == 3
    assert metrics.mean_reward == pytest.approx(2.0)
    assert metrics.std_reward == pytest.approx(1.0)
    assert metrics.mean_episode_length == pytest.approx(6.0)
    # Success/collision rates exclude episodes without that outcome, and a
    # rate with no observed outcomes is None, never 0.0.
    assert metrics.success_rate == pytest.approx(0.5)
    assert metrics.collision_rate is None
    # The truncation flag is always available, so it uses every episode.
    assert metrics.timeout_rate == pytest.approx(2.0 / 3.0)
    assert metrics.to_dict()["episodes"] == 3

    single = summarize_descriptive_episodes(
        rewards=[5.0],
        episode_lengths=[2.0],
        successes=[True],
        collisions=[False],
        truncations=[False],
    )
    assert single.episodes == 1
    assert single.std_reward is None


def test_summarize_descriptive_episodes_validates_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        summarize_descriptive_episodes([1.0], [1.0, 2.0], [True], [False], [False])
    with pytest.raises(ValueError, match="At least one episode"):
        summarize_descriptive_episodes([], [], [], [], [])
    for bad_reward in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            summarize_descriptive_episodes([bad_reward], [1.0], [True], [False], [False])
    for bad_length in (-1.0, float("nan"), float("inf"), "1"):
        with pytest.raises(ValueError, match="Episode lengths"):
            summarize_descriptive_episodes([1.0], [bad_length], [True], [False], [False])  # type: ignore[list-item]
    for bad_flag in (1, "false"):
        with pytest.raises(ValueError, match="successes values"):
            summarize_descriptive_episodes([1.0], [1.0], [bad_flag], [False], [False])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="collisions values"):
            summarize_descriptive_episodes([1.0], [1.0], [True], [bad_flag], [False])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="Truncation values"):
            summarize_descriptive_episodes([1.0], [1.0], [True], [False], [bad_flag])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="real numbers"):
        summarize_descriptive_episodes([True], [1.0], [True], [False], [False])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="real numbers"):
        summarize_seed_values(["not-a-number"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="real numbers"):
        summarize_seed_values([True])


def test_pooled_and_seed_level_statistics_use_different_sample_units() -> None:
    """Pooled episodes and seed summaries are different statistical layers."""
    seed_zero = [1.0, 2.0]  # seed mean 1.5
    seed_one = [10.0, 10.0, 10.0, 10.0]  # seed mean 10.0, four episodes

    pooled = summarize_descriptive_episodes(
        rewards=seed_zero + seed_one,
        episode_lengths=[1.0] * 6,
        successes=[True, False, True, True, True, True],
        collisions=[False] * 6,
        truncations=[False] * 6,
    )
    across_seeds = summarize_seed_values([1.5, 10.0])

    assert pooled.episodes == 6
    assert across_seeds.sample_count == 2
    assert pooled.mean_reward == pytest.approx((1.0 + 2.0 + 40.0) / 6.0)
    assert across_seeds.mean == pytest.approx(5.75)
    # Unequal episode counts per seed make the two layers differ.
    assert pooled.mean_reward != pytest.approx(across_seeds.mean)

    # With equal episode counts the values coincide, but the layers are still
    # computed independently with different sample units.
    equal_pooled = summarize_descriptive_episodes(
        rewards=[1.0, 2.0, 10.0, 20.0],
        episode_lengths=[1.0] * 4,
        successes=[True, False, True, True],
        collisions=[False] * 4,
        truncations=[False] * 4,
    )
    equal_across = summarize_seed_values([1.5, 15.0])
    assert equal_pooled.episodes == 4
    assert equal_across.sample_count == 2
    assert equal_pooled.mean_reward == pytest.approx(equal_across.mean)


def test_episode_records_expose_group_and_reset_seed_naming(tmp_path: Path) -> None:
    evaluator = Evaluator(algorithm=_ZeroPolicy(), env=_SeedOutcomeEnv())  # type: ignore[arg-type]
    result = evaluator.evaluate_seeds([10, 20], episodes_per_seed=2)
    records = result.episodes

    assert [record.evaluation_group_seed for record in records] == [10, 10, 20, 20]
    assert [record.episode_reset_seed for record in records] == [55, 67, 210, 232]
    # Legacy names keep their documented meanings: seed is the group seed and
    # episode_seed is the per-episode reset seed.
    assert [record.seed for record in records] == [10, 10, 20, 20]
    assert [record.episode_seed for record in records] == [55, 67, 210, 232]

    payload = records[0].to_dict()
    assert payload["evaluation_group_seed"] == 10
    assert payload["episode_reset_seed"] == 55
    assert payload["seed"] == 10
    assert payload["episode_seed"] == 55

    assert [summary.evaluation_group_seed for summary in result.per_seed] == [10, 20]
    assert [summary.seed for summary in result.per_seed] == [10, 20]

    document = result.to_dict()
    assert document["metadata"]["evaluation_group_seeds"] == [10, 20]
    assert "evaluation group seeds" in document["metadata"]["seed_semantics"]

    saved_json, saved_csv = evaluator.save_multiseed_report(
        result,
        tmp_path / "seed_naming.json",
        tmp_path / "seed_naming.csv",
    )
    exported = json.loads(saved_json.read_text(encoding="utf-8"))
    json.dumps(exported, allow_nan=False)
    assert exported["episodes"][0]["evaluation_group_seed"] == 10
    assert exported["episodes"][0]["episode_reset_seed"] == 55
    assert saved_csv.is_file()
    evaluator.close()


def test_path_length_is_unavailable_without_position_telemetry() -> None:
    class _NoPositionEnv(gym.Env):
        observation_space = gym.spaces.Box(-1000.0, 1000.0, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def __init__(self) -> None:
            super().__init__()
            self.current_seed = 0
            self.steps = 0

        def reset(
            self, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[np.ndarray, dict]:
            super().reset(seed=seed)
            self.current_seed = 0 if seed is None else seed
            self.steps = 0
            return np.zeros(1, dtype=np.float32), {"success": True, "collision": False}

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
            self.steps += 1
            info = {"success": True, "collision": False}
            terminated = self.steps >= 2
            return np.zeros(1, dtype=np.float32), 1.0, terminated, False, info

    evaluator = Evaluator(algorithm=_ZeroPolicy(), env=_NoPositionEnv())  # type: ignore[arg-type]
    result = evaluator.evaluate_seeds([5], episodes_per_seed=1)
    record = result.episodes[0]

    assert record.path_length is None
    assert "path_length" not in record.to_dict()
    assert result.per_seed[0].path_length is None
    document = result.to_dict()
    assert "path_length" not in document["episodes"][0]
    evaluator.close()


def test_evaluator_falls_back_to_private_obstacle_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=10, num_obstacles=2)
    env.reset(seed=5)
    monkeypatch.delattr(type(env), "obstacles")

    evaluator = Evaluator(algorithm=_ZeroDronePolicy(), env=env)  # type: ignore[arg-type]
    result = evaluator.evaluate_seeds([5], episodes_per_seed=1)
    assert result.episodes[0].min_obstacle_clearance is not None
    evaluator.close()


def test_evaluator_reads_obstacle_geometry_from_public_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=10, num_obstacles=2)
    env.reset(seed=5)
    public_obstacles = env.obstacles
    assert public_obstacles

    # Shadow the private attribute while the public interface keeps exposing
    # the real geometry: the evaluator must use ``unwrapped.obstacles``.
    monkeypatch.setattr(type(env), "obstacles", property(lambda self: list(public_obstacles)))
    env._obstacles = []

    evaluator = Evaluator(algorithm=_ZeroDronePolicy(), env=env)  # type: ignore[arg-type]
    result = evaluator.evaluate_seeds([5], episodes_per_seed=1)
    assert result.episodes[0].min_obstacle_clearance is not None
    evaluator.close()
