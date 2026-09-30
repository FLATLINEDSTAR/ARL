"""Focused tests for PPO learning-curve benchmarking."""

import csv
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from adaptive_rl.benchmarking import (
    BenchmarkRunError,
    LearningCurveBenchmarkResult,
    LearningCurvePoint,
    run_learning_curve_benchmark,
    validate_budgets,
)
from adaptive_rl.config import (
    AlgorithmConfig,
    BenchmarkConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
    load_config,
    save_config,
)
from adaptive_rl.evaluation.evaluator import Evaluator

CSV_FIELDNAMES = [
    "budget_timesteps",
    "trained_timesteps",
    "pooled_success_rate",
    "pooled_collision_rate",
    "pooled_timeout_rate",
    "pooled_mean_reward",
    "pooled_std_reward",
    "pooled_mean_episode_length",
    "training_time_seconds",
    "model_path",
    "training_seed",
    "evaluation_seeds",
    "evaluation_episodes",
    "evaluation_group_seed_count",
    "episodes_per_seed",
    "evaluation_split",
    "deterministic",
    "algorithm",
    "environment",
    "environment_fingerprint",
    "device",
]


def _make_point(**overrides: Any) -> LearningCurvePoint:
    """Build a fully populated learning-curve point with overridable defaults."""
    values: dict[str, Any] = {
        "budget_timesteps": 64,
        "trained_timesteps": 64,
        "success_rate": 0.5,
        "collision_rate": 0.25,
        "timeout_rate": 0.0,
        "mean_reward": 1.5,
        "std_reward": None,
        "mean_episode_length": 4.0,
        "training_time_seconds": 0.25,
        "model_path": "artifacts/models/ppo_budget_64_final.zip",
        "training_seed": 17,
        "evaluation_seeds": [11, 12],
        "evaluation_episodes": 2,
        "deterministic": True,
        "algorithm": "ppo",
        "environment": "drone",
    }
    values.update(overrides)
    return LearningCurvePoint(**values)


def _make_result(**overrides: Any) -> LearningCurveBenchmarkResult:
    """Build a benchmark result with overridable defaults."""
    values: dict[str, Any] = {
        "benchmark_name": "ppo_learning_curve",
        "algorithm": "ppo",
        "environment": "drone",
        "budgets": [64],
        "training_seed": 17,
        "evaluation_seeds": [11, 12],
        "evaluation_episodes": 2,
        "deterministic": True,
    }
    values.update(overrides)
    return LearningCurveBenchmarkResult(**values)


def _make_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="benchmark_test",
        seed=31,
        algorithm=AlgorithmConfig(
            name="ppo",
            batch_size=32,
            parameters={"n_steps": 64, "n_epochs": 1, "seed": 999},
        ),
        environment=EnvironmentConfig(
            name="drone",
            max_steps=8,
            parameters={"bounds": [20.0, 20.0, 10.0], "num_obstacles": 1},
        ),
        training=TrainingConfig(total_timesteps=256, checkpoint_freq=0, log_interval=1),
        evaluation=EvaluationConfig(eval_episodes=1, deterministic=True),
        output_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        benchmark=BenchmarkConfig(
            budgets=[128, 64],
            training_seed=17,
            evaluation_seeds=[11, 12],
            evaluation_episodes=1,
            deterministic=True,
        ),
    )


@pytest.mark.parametrize(
    ("raw_budgets", "message"),
    [
        ([], "empty"),
        ([0], "positive"),
        ([-1], "positive"),
        ("foo,128", "Malformed budget value"),
        ("64,,128", "Malformed budget list"),
        ([64, 64], "Duplicate"),
    ],
)
def test_budget_validation_rejects_invalid_values(raw_budgets: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_budgets(raw_budgets)


def test_budget_validation_normalizes_valid_values() -> None:
    assert validate_budgets("128,64") == [64, 128]
    assert validate_budgets([128, 64]) == [64, 128]


@pytest.mark.parametrize("budgets", [[], [0], [-1], [64, 64], ["64"]])
def test_benchmark_config_rejects_invalid_budgets(budgets: Any) -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig(budgets=budgets)


def test_benchmark_config_defaults_and_old_config_compatibility(tmp_path: Path) -> None:
    benchmark = BenchmarkConfig()
    assert benchmark.budgets == [5000, 10000, 25000, 50000]
    assert benchmark.training_seed == 42
    assert benchmark.evaluation_seeds
    assert benchmark.evaluation_episodes > 0

    old_config_path = tmp_path / "old_config.yaml"
    old_config_path.write_text(
        """
name: old_config
seed: 4
algorithm:
  name: ppo
environment:
  name: drone
training:
  total_timesteps: 128
evaluation:
  eval_episodes: 2
""",
        encoding="utf-8",
    )
    loaded = load_config(old_config_path)
    assert loaded.name == "old_config"
    assert loaded.benchmark is None

    benchmark_config_path = tmp_path / "benchmark_config.yaml"
    benchmark_config_path.write_text(
        """
name: configured_benchmark
algorithm:
  name: ppo
environment:
  name: drone
training:
  total_timesteps: 128
evaluation:
  eval_episodes: 2
benchmark:
  budgets: [128, 64]
  training_seed: 7
  evaluation_seeds: [20, 21]
  evaluation_episodes: 3
  deterministic: false
""",
        encoding="utf-8",
    )
    configured = load_config(benchmark_config_path)
    assert configured.benchmark is not None
    assert configured.benchmark.budgets == [64, 128]
    assert configured.benchmark.training_seed == 7
    assert configured.benchmark.evaluation_seeds == [20, 21]
    assert configured.benchmark.evaluation_episodes == 3
    assert configured.benchmark.deterministic is False
    assert configured.benchmark.evaluation_split == "custom"


def test_core_and_benchmark_imports_do_not_load_optional_rl_stack() -> None:
    """Importing the core package or benchmarking must not need torch/sb3/gym."""
    repository_root = Path(__file__).resolve().parent.parent
    script = textwrap.dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder

        class BlockOptionalRLImports(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "gymnasium" or fullname.startswith(
                    ("stable_baselines3", "torch")
                ):
                    raise AssertionError(f"Optional RL dependency imported eagerly: {fullname}")
                return None

        sys.meta_path.insert(0, BlockOptionalRLImports())
        import adaptive_rl
        from adaptive_rl.benchmarking import (
            BenchmarkRunError,
            run_learning_curve_benchmark,
        )

        assert callable(run_learning_curve_benchmark)
        assert issubclass(BenchmarkRunError, RuntimeError)

        # The root namespace stays small: benchmark symbols live in the
        # benchmarking subpackage and must not be re-exported eagerly.
        assert "run_learning_curve_benchmark" not in vars(adaptive_rl)
        assert "run_learning_curve_benchmark" not in adaptive_rl.__all__
        assert "LearningCurvePoint" not in vars(adaptive_rl)
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository_root / "src"), environment.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_learning_curve_benchmark_execution_and_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl.benchmarking import learning_curve
    from adaptive_rl.training.trainer import PPOTrainer

    config = _make_config(tmp_path)
    original = config.model_dump(mode="python")
    real_make_env = learning_curve._make_env
    real_evaluate_model = learning_curve._evaluate_model
    trainers: list[Any] = []
    envs: list[gym.Env] = []
    evaluation_calls: list[dict[str, Any]] = []

    class TrackingTrainer(PPOTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.fit_calls = 0
            trainers.append(self)

        def fit(self) -> Any:
            self.fit_calls += 1
            return super().fit()

    def track_env(env_name: str, **kwargs: Any) -> gym.Env:
        env = real_make_env(env_name, **kwargs)
        envs.append(env)
        return env

    def track_trainer(config: ExperimentConfig, env: gym.Env) -> PPOTrainer:
        return TrackingTrainer(config=config, env=env)

    def track_evaluation(*args: Any, **kwargs: Any) -> Any:
        evaluation_calls.append(kwargs.copy())
        return real_evaluate_model(*args, **kwargs)

    monkeypatch.setattr(learning_curve, "_make_trainer", track_trainer)
    monkeypatch.setattr(learning_curve, "_make_env", track_env)
    monkeypatch.setattr(learning_curve, "_evaluate_model", track_evaluation)

    result = run_learning_curve_benchmark(config, budgets=[128, 64])

    assert result.budgets == [64, 128]
    assert [point.budget_timesteps for point in result.points] == [64, 128]
    assert len(result.points) == len(trainers) == 2
    assert [trainer.fit_calls for trainer in trainers] == [1, 1]
    assert [trainer.config.training.total_timesteps for trainer in trainers] == [64, 128]
    assert [trainer.config.seed for trainer in trainers] == [17, 17]
    assert [trainer.config.name for trainer in trainers] == ["ppo_budget_64", "ppo_budget_128"]
    assert trainers[0].config.output_dir != trainers[1].config.output_dir
    assert trainers[0].config.log_dir != trainers[1].config.log_dir
    assert [
        (trainer.config.training.checkpoint_freq, trainer.config.training.log_interval)
        for trainer in trainers
    ] == [(0, 1), (0, 1)]
    assert [trainer.algorithm.hyperparameters["seed"] for trainer in trainers] == [17, 17]
    expected_algorithm_config = config.algorithm.model_dump()
    expected_algorithm_config["parameters"].pop("seed")
    assert [trainer.config.algorithm.model_dump() for trainer in trainers] == [
        expected_algorithm_config,
        expected_algorithm_config,
    ]
    assert [trainer.config.environment.model_dump() for trainer in trainers] == [
        config.environment.model_dump(),
        config.environment.model_dump(),
    ]
    assert [trainer.config.evaluation.model_dump() for trainer in trainers] == [
        config.evaluation.model_dump(),
        config.evaluation.model_dump(),
    ]
    assert len({id(trainer.env) for trainer in trainers}) == 2
    assert len({id(trainer.algorithm.model) for trainer in trainers}) == 2
    assert [trainer.algorithm.num_timesteps for trainer in trainers] == [64, 128]
    assert all(env is trainer.env for env, trainer in zip((envs[0], envs[2]), trainers))
    assert len({id(env) for env in envs}) == 4

    for call in evaluation_calls:
        assert call["evaluation_seeds"] == [11, 12]
        assert call["evaluation_episodes"] == 1
        assert call["deterministic"] is True
        assert call["env_name"] == config.environment.name
        assert call["env_kwargs"] == config.environment.parameters
    assert len(evaluation_calls) == 2

    model_paths = [Path(point.model_path) for point in result.points]
    assert len(set(model_paths)) == 2
    assert all(path.is_file() for path in model_paths)
    assert all(path.parent.parent.parent.name == "learning_curve" for path in model_paths)
    assert result.json_path == config.output_dir / "benchmarks" / "learning_curve_budget.json"
    assert result.csv_path == config.output_dir / "benchmarks" / "learning_curve_budget.csv"

    for point in result.points:
        assert point.trained_timesteps == point.budget_timesteps
        assert point.success_rate is None or 0.0 <= point.success_rate <= 1.0
        assert point.collision_rate is None or 0.0 <= point.collision_rate <= 1.0
        assert point.timeout_rate is None or 0.0 <= point.timeout_rate <= 1.0
        assert math.isfinite(point.mean_reward)
        assert point.std_reward is None or point.std_reward >= 0.0
        assert point.mean_episode_length >= 0.0
        assert math.isfinite(point.training_time_seconds)
        assert point.training_time_seconds >= 0.0

    assert config.name == original["name"]
    assert config.training.total_timesteps == original["training"]["total_timesteps"]
    assert config.output_dir == original["output_dir"]
    assert config.log_dir == original["log_dir"]
    assert config.evaluation.model_dump() == original["evaluation"]
    assert config.algorithm.parameters == original["algorithm"]["parameters"]
    assert config.model_dump(mode="python") == original

    assert result.json_path is not None and result.json_path.is_file()
    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    json.dumps(data, allow_nan=False)
    assert data["status"] == "completed"
    assert data["completed_budgets"] == [64, 128]
    assert data["failed_budget"] is None
    assert data["error"] is None
    assert data["plot"] == {"requested": False, "path": None, "error": None}
    assert data["benchmark"]["training_seed"] == 17
    assert data["benchmark"]["evaluation_seeds"] == [11, 12]
    assert data["benchmark"]["evaluation_group_seeds"] == [11, 12]
    assert data["benchmark"]["evaluation_episodes"] == 1
    assert data["benchmark"]["episodes_per_seed"] == 1
    assert data["benchmark"]["deterministic"] is True
    assert data["benchmark"]["evaluation_split"] == "custom"
    assert data["benchmark"]["budgets"] == [64, 128]
    assert data["provenance"]["adaptive_rl_version"]
    assert data["provenance"]["python_version"]
    assert set(data["provenance"]["packages"]) == {"torch", "stable-baselines3", "gymnasium"}
    assert data["provenance"]["environment_configuration"]["training"]["name"] == "drone"
    assert data["provenance"]["environment_configuration"]["evaluation"]["name"] == "drone"
    assert len(data["provenance"]["environment_fingerprint_sha256"]) == 64
    assert "evaluation group seeds" in data["benchmark"]["seed_semantics"]
    assert len(data["results"]) == 2
    required_metrics = {
        "budget_timesteps",
        "trained_timesteps",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_reward",
        "std_reward",
        "mean_episode_length",
        "training_time_seconds",
        "model_path",
        "descriptive_metrics",
        "per_seed_summaries",
        "cross_seed_statistics",
        "training_metadata",
        "evaluation_split",
        "environment_fingerprint",
        "device",
    }
    for row in data["results"]:
        assert required_metrics <= row.keys()
        assert Path(row["model_path"]).is_file()
        descriptive = row["descriptive_metrics"]
        assert descriptive["success_rate"] == row["success_rate"]
        assert descriptive["collision_rate"] == row["collision_rate"]
        assert descriptive["timeout_rate"] == row["timeout_rate"]
        assert descriptive["mean_reward"] == row["mean_reward"]
        assert descriptive["std_reward"] == row["std_reward"]
        assert descriptive["mean_episode_length"] == row["mean_episode_length"]
        assert descriptive["episodes"] == 2
        metadata = row["training_metadata"]
        assert metadata["training_seed"] == 17
        assert metadata["evaluation_group_seeds"] == [11, 12]
        assert metadata["episodes_per_seed"] == 1
        assert metadata["model_path"] == row["model_path"]
        assert metadata["training_time_seconds"] == row["training_time_seconds"]
        budget_result_path = Path(row["model_path"]).parents[1] / "budget_result.json"
        budget_result = json.loads(budget_result_path.read_text(encoding="utf-8"))
        assert budget_result["status"] == "completed"
        assert budget_result["result"]["budget_timesteps"] == row["budget_timesteps"]
        budget_dir = Path(row["model_path"]).parents[1]
        assert list(budget_dir.parent.glob(f".{budget_dir.name}.*")) == []
    assert set(data["plot_data"]) == {
        "budgets",
        "trained_timesteps",
        "success_rate",
        "mean_reward",
    }
    assert data["plot_data"]["budgets"] == [64, 128]
    assert data["plot_data"]["trained_timesteps"] == [64, 128]
    assert data["benchmark"]["metric_semantics"]["aggregation"] == (
        "descriptive_metrics (and the legacy top-level metric fields) pool "
        "every evaluated episode and are descriptive only"
    )
    assert (
        "not across pooled episodes"
        in data["benchmark"]["metric_semantics"]["cross_seed_statistics"]
    )
    assert data["benchmark"]["training_time_semantics"].startswith("monotonic elapsed time")
    assert all(len(row["per_seed_summaries"]) == 2 for row in data["results"])
    assert all("success_rate" in row["cross_seed_statistics"] for row in data["results"])

    assert result.csv_path is not None and result.csv_path.is_file()
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == CSV_FIELDNAMES
    assert len(rows) == len(result.points)
    assert not any(field.startswith("Unnamed:") for field in reader.fieldnames or [])
    assert [int(row["budget_timesteps"]) for row in rows] == [64, 128]
    assert all(math.isfinite(float(row["pooled_mean_reward"])) for row in rows)
    assert "mean_reward" not in (reader.fieldnames or [])
    assert all(Path(row["model_path"]).is_file() for row in rows)
    assert all(row["training_seed"] == "17" for row in rows)
    assert all(row["evaluation_seeds"] == "11;12" for row in rows)
    assert all(row["evaluation_episodes"] == "1" for row in rows)
    assert all(row["evaluation_group_seed_count"] == "2" for row in rows)
    assert all(row["episodes_per_seed"] == "1" for row in rows)
    assert all(row["evaluation_split"] == "custom" for row in rows)
    assert all(row["algorithm"] == "ppo" for row in rows)
    assert all(row["environment"] == "drone" for row in rows)
    assert all(len(row["environment_fingerprint"]) == 64 for row in rows)
    assert all(row["deterministic"] == "True" for row in rows)
    provenance = data["provenance"]
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["configuration_fingerprint_sha256"])
    assert provenance["environment_fingerprint_sha256"] == result.environment_fingerprint
    assert provenance["training_seed"] == 17
    assert provenance["evaluation_seeds"] == [11, 12]
    assert provenance["requested_budgets"] == [64, 128]
    assert provenance["git_commit"] is None or re.fullmatch(
        r"[0-9a-f]{40}", provenance["git_commit"]
    )
    assert provenance["working_tree_dirty"] is None or isinstance(
        provenance["working_tree_dirty"], bool
    )


def test_learning_curve_benchmark_repeats_deterministically(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    first = run_learning_curve_benchmark(
        config,
        budgets=[65],
        output_dir=tmp_path / "first",
    )
    second = run_learning_curve_benchmark(
        config,
        budgets=[65],
        output_dir=tmp_path / "second",
    )

    first_point = first.points[0]
    second_point = second.points[0]
    assert first.budgets == second.budgets == [65]
    # Non-aligned budget: PPO collects complete rollouts, so the requested
    # budget and the timesteps actually trained differ (65 -> 128 with
    # n_steps=64). Both are recorded and never conflated.
    assert first_point.budget_timesteps == 65
    assert first_point.trained_timesteps == second_point.trained_timesteps == 128
    assert first_point.budget_timesteps != first_point.trained_timesteps
    assert first.plot_data["budgets"] == [65]
    assert first.plot_data["trained_timesteps"] == [128]
    assert first.training_seed == second.training_seed == 17
    first_provenance = json.loads(first.json_path.read_text(encoding="utf-8"))["provenance"]
    second_provenance = json.loads(second.json_path.read_text(encoding="utf-8"))["provenance"]
    assert (
        first_provenance["configuration_fingerprint_sha256"]
        == second_provenance["configuration_fingerprint_sha256"]
    )
    assert first.evaluation_seeds == second.evaluation_seeds == [11, 12]
    assert first.evaluation_episodes == second.evaluation_episodes == 1
    assert first.deterministic is second.deterministic is True
    for field in (
        "budget_timesteps",
        "trained_timesteps",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_reward",
        "std_reward",
        "mean_episode_length",
        "training_seed",
        "evaluation_seeds",
        "evaluation_episodes",
        "deterministic",
    ):
        assert getattr(first_point, field) == getattr(second_point, field)


def test_test_split_benchmark_records_distribution_and_seed_provenance(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.environment.parameters["split"] = "train"
    assert config.benchmark is not None
    config.benchmark.evaluation_seeds = [1001]
    config.benchmark.evaluation_episodes = 1

    result = run_learning_curve_benchmark(
        config,
        budgets=[64],
        output_dir=tmp_path / "test_split",
        evaluation_split="test",
    )
    document = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert result.evaluation_split == document["benchmark"]["evaluation_split"] == "test"
    assert (
        document["provenance"]["environment_configuration"]["training"]["parameters"]["split"]
        == "train"
    )
    assert (
        document["provenance"]["environment_configuration"]["evaluation"]["parameters"]["split"]
        == "test"
    )
    episode_seed_mapping = document["benchmark"]["episode_reset_seed_mapping"]
    assert "SHA-256" in episode_seed_mapping
    assert result.points[0].per_seed_summaries[0]["seed"] == 1001
    assert result.points[0].evaluation_split == "test"

    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["evaluation_split"] == "test"


def test_split_benchmark_requires_train_partition_for_policy_training(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    assert config.benchmark is not None
    config.benchmark.evaluation_seeds = [1001]
    with pytest.raises(ValueError, match="split 'test' requires"):
        run_learning_curve_benchmark(
            config,
            budgets=[64],
            output_dir=tmp_path / "invalid_split",
            evaluation_split="test",
        )
    assert not (tmp_path / "invalid_split").exists()


def test_benchmark_keeps_single_episode_standard_deviation_unavailable(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    assert config.benchmark is not None
    config.benchmark.evaluation_seeds = [11]
    config.benchmark.evaluation_episodes = 1

    result = run_learning_curve_benchmark(
        config,
        budgets=[64],
        output_dir=tmp_path / "single_episode",
    )

    point = result.points[0]
    assert point.std_reward is None
    assert result.to_dict()["results"][0]["std_reward"] is None
    assert result.csv_path is not None
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["pooled_std_reward"] == ""


def test_json_manifest_replace_is_atomic_on_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import adaptive_rl.benchmarking.learning_curve as learning_curve

    target = tmp_path / "learning_curve_budget.json"
    previous_document = '{"previous": true}\n'
    target.write_text(previous_document, encoding="utf-8")
    result = _make_result(json_path=target)

    def interrupted_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated interruption before atomic replace")

    monkeypatch.setattr(learning_curve.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        learning_curve._write_json(result)

    assert target.read_text(encoding="utf-8") == previous_document
    assert list(tmp_path.glob(".learning_curve_budget.json.*.tmp")) == []


def test_plot_failure_preserves_json_and_csv_benchmark_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import adaptive_rl.benchmarking.learning_curve as learning_curve

    point = _make_point()
    monkeypatch.setattr(
        learning_curve,
        "_run_single_budget",
        lambda *args, **kwargs: point,
    )
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    result = run_learning_curve_benchmark(
        _make_config(tmp_path),
        budgets=[64],
        output_dir=tmp_path / "plot_failure",
        plot=True,
    )

    assert result.status == "completed"
    assert result.plot_error is not None
    assert result.json_path is not None and result.json_path.is_file()
    assert result.csv_path is not None and result.csv_path.is_file()
    document = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert document["status"] == "completed"
    assert document["plot"]["error"] == result.plot_error
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1


def test_default_benchmark_evaluation_does_not_derive_seed_count_from_episode_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl.benchmarking import learning_curve

    config = _make_config(tmp_path)
    config.benchmark = None
    config.evaluation.eval_episodes = 2
    calls: list[dict[str, Any]] = []

    def fake_run(
        config: ExperimentConfig,
        budget: int,
        **kwargs: Any,
    ) -> learning_curve.LearningCurvePoint:
        calls.append({"budget": budget, **kwargs})
        return learning_curve.LearningCurvePoint(
            budget_timesteps=budget,
            trained_timesteps=budget,
            success_rate=None,
            collision_rate=None,
            timeout_rate=0.0,
            mean_reward=1.0,
            std_reward=0.0,
            mean_episode_length=1.0,
            training_time_seconds=0.0,
            model_path=str(tmp_path / "model.zip"),
            training_seed=kwargs["training_seed"],
            evaluation_seeds=list(kwargs["evaluation_seeds"]),
            evaluation_episodes=kwargs["evaluation_episodes"],
            deterministic=kwargs["deterministic"],
            algorithm="ppo",
            environment="drone",
        )

    monkeypatch.setattr(learning_curve, "_run_single_budget", fake_run)
    result = run_learning_curve_benchmark(config, budgets=[64], output_dir=tmp_path / "out")

    assert result.evaluation_seeds == [42, 43, 44, 45, 46]
    assert result.evaluation_episodes == 20
    assert calls[0]["evaluation_seeds"] == [42, 43, 44, 45, 46]
    assert calls[0]["evaluation_episodes"] == 20
    assert len(calls[0]["evaluation_seeds"]) * calls[0]["evaluation_episodes"] == 100


def test_plotting_is_lazy_and_closes_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl.benchmarking.learning_curve import (
        LearningCurveBenchmarkResult,
        plot_learning_curve,
    )

    figure = MagicMock()
    figure.savefig.side_effect = lambda path, dpi, metadata: Path(path).touch()
    axes = [MagicMock(), MagicMock()]
    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.subplots = MagicMock(return_value=(figure, axes))  # type: ignore[attr-defined]
    pyplot.close = MagicMock()  # type: ignore[attr-defined]
    matplotlib = ModuleType("matplotlib")
    matplotlib.use = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    output_path = tmp_path / "curve.png"
    result = LearningCurveBenchmarkResult(
        benchmark_name="ppo_learning_curve",
        algorithm="ppo",
        environment="drone",
        budgets=[],
        training_seed=0,
        evaluation_seeds=[],
        evaluation_episodes=1,
        deterministic=True,
    )
    path = plot_learning_curve(result, output_path)
    assert path == output_path
    assert output_path.is_file()
    assert figure.savefig.call_args.kwargs["metadata"] == {"Software": "AdaptiveRL"}
    pyplot.close.assert_called_once_with(figure)  # type: ignore[attr-defined]


def test_plot_learning_curve_with_none_success_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl.benchmarking.learning_curve import (
        LearningCurveBenchmarkResult,
        LearningCurvePoint,
        plot_learning_curve,
    )

    figure = MagicMock()
    axes = [MagicMock(), MagicMock()]
    pyplot = MagicMock()
    pyplot.subplots.return_value = (figure, axes)
    matplotlib = ModuleType("matplotlib")
    matplotlib.use = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    output_path = tmp_path / "curve_none.png"
    point = LearningCurvePoint(
        budget_timesteps=64,
        trained_timesteps=64,
        success_rate=None,
        collision_rate=None,
        timeout_rate=None,
        mean_reward=10.0,
        std_reward=None,
        mean_episode_length=5.0,
        training_time_seconds=1.0,
        model_path="dummy.zip",
        training_seed=0,
        evaluation_seeds=[1],
        evaluation_episodes=1,
        deterministic=True,
        algorithm="ppo",
        environment="drone",
    )
    result = LearningCurveBenchmarkResult(
        benchmark_name="ppo_learning_curve",
        algorithm="ppo",
        environment="drone",
        budgets=[64],
        training_seed=0,
        evaluation_seeds=[1],
        evaluation_episodes=1,
        deterministic=True,
        points=[point],
    )
    path = plot_learning_curve(result, output_path)
    assert path == output_path
    axes[0].plot.assert_not_called()
    axes[1].plot.assert_called_once()


def test_plotting_reports_missing_matplotlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl.benchmarking.learning_curve import (
        LearningCurveBenchmarkResult,
        plot_learning_curve,
    )

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    result = LearningCurveBenchmarkResult(
        benchmark_name="ppo_learning_curve",
        algorithm="ppo",
        environment="drone",
        budgets=[],
        training_seed=0,
        evaluation_seeds=[],
        evaluation_episodes=1,
        deterministic=True,
    )
    with pytest.raises(RuntimeError, match="requires matplotlib"):
        plot_learning_curve(result, tmp_path / "curve.png")


class _OutcomeEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, outcome: str) -> None:
        super().__init__()
        self.outcome = outcome

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.outcome == "collision":
            return (
                np.zeros(1, dtype=np.float32),
                1.0,
                True,
                False,
                {"success": False, "collision": True},
            )
        if self.outcome == "truncated":
            return (
                np.zeros(1, dtype=np.float32),
                1.0,
                False,
                True,
                {
                    "success": False,
                    "collision": False,
                },
            )
        if self.outcome == "success":
            return (
                np.zeros(1, dtype=np.float32),
                1.0,
                True,
                False,
                {
                    "success": True,
                    "collision": False,
                },
            )
        if self.outcome == "termination":
            return (
                np.zeros(1, dtype=np.float32),
                1.0,
                True,
                False,
                {
                    "success": False,
                    "collision": False,
                },
            )
        return np.zeros(1, dtype=np.float32), 1.0, True, False, {}


class _ConstantPolicy:
    def predict(self, observation: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.zeros(1, dtype=np.float32), None


@pytest.mark.parametrize(
    ("outcome", "expected_success", "expected_collision", "expected_timeout"),
    [
        ("termination", 0.0, 0.0, 0.0),
        ("collision", 0.0, 1.0, 0.0),
        ("truncated", 0.0, 0.0, 1.0),
        ("success", 1.0, 0.0, 0.0),
    ],
)
def test_evaluator_uses_gymnasium_truncation_signal(
    outcome: str,
    expected_success: float,
    expected_collision: float,
    expected_timeout: float,
) -> None:
    env = _OutcomeEnv(outcome)
    evaluator = Evaluator(algorithm=_ConstantPolicy(), env=env)  # type: ignore[arg-type]
    metrics = evaluator.evaluate(num_episodes=1, deterministic=True, base_seed=2)

    assert metrics.success_rate == expected_success
    assert metrics.collision_rate == expected_collision
    assert metrics.truncation_rate == expected_timeout
    assert evaluator.last_episode_records[0].truncated is (expected_timeout == 1.0)
    evaluator.close()


def test_evaluator_preserves_unavailable_outcome_metrics() -> None:
    env = _OutcomeEnv("unavailable")
    evaluator = Evaluator(algorithm=_ConstantPolicy(), env=env)  # type: ignore[arg-type]
    metrics = evaluator.evaluate(num_episodes=1)
    assert metrics.success_rate is None
    assert metrics.collision_rate is None
    evaluator.close()


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _clean(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def test_benchmark_cli_dispatch_and_budget_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl import benchmarking
    from adaptive_rl.cli import app

    config_path = tmp_path / "benchmark.yaml"
    save_config(_make_config(tmp_path), config_path)
    runner = CliRunner()
    benchmark_help = runner.invoke(app, ["benchmark", "--help"])
    assert benchmark_help.exit_code == 0
    assert "budgets" in _clean(benchmark_help.output)
    budget_help = runner.invoke(app, ["benchmark", "budgets", "--help"])
    assert budget_help.exit_code == 0
    clean_budget_help = _clean(budget_help.output)
    for option in (
        "--config",
        "--budgets",
        "--training-seed",
        "--eval-seeds",
        "--episodes",
        "--deterministic",
        "--stochastic",
        "--output-dir",
        "--plot",
        "--no-plot",
        "--plot-x-axis",
        "--evaluation-split",
    ):
        assert option in clean_budget_help
    dispatch: list[dict[str, Any]] = []
    fake_result = LearningCurveBenchmarkResult(
        benchmark_name="ppo_learning_curve",
        algorithm="ppo",
        environment="drone",
        budgets=[64, 128],
        training_seed=17,
        evaluation_seeds=[11, 12],
        evaluation_episodes=1,
        deterministic=True,
        json_path=tmp_path / "learning_curve_budget.json",
        csv_path=tmp_path / "learning_curve_budget.csv",
    )

    def fake_benchmark(config: ExperimentConfig, **kwargs: Any) -> LearningCurveBenchmarkResult:
        dispatch.append({"config": config, **kwargs})
        return fake_result

    monkeypatch.setattr(benchmarking, "run_learning_curve_benchmark", fake_benchmark)
    success = runner.invoke(
        app,
        [
            "benchmark",
            "budgets",
            "--config",
            str(config_path),
            "--budgets",
            "64,128",
            "--training-seed",
            "17",
            "--eval-seeds",
            "11,12",
            "--episodes",
            "1",
            "--evaluation-split",
            "custom",
        ],
    )
    assert success.exit_code == 0, success.output
    clean_success = _clean(success.output)
    assert "learning-curve benchmark complete" in clean_success
    assert "64, 128" in clean_success
    assert len(dispatch) == 1
    assert dispatch[0]["budgets"] == [64, 128]
    assert dispatch[0]["training_seed"] == 17
    assert dispatch[0]["evaluation_seeds"] == [11, 12]
    assert dispatch[0]["evaluation_episodes"] == 1
    assert dispatch[0]["plot_x_axis"] == "trained"
    assert dispatch[0]["evaluation_split"] == "custom"

    invalid_axis = runner.invoke(
        app,
        [
            "benchmark",
            "budgets",
            "--config",
            str(config_path),
            "--plot-x-axis",
            "sideways",
        ],
    )
    assert invalid_axis.exit_code == 1
    assert "Invalid --plot-x-axis" in invalid_axis.output
    assert "Traceback" not in invalid_axis.output
    assert len(dispatch) == 1

    invalid_split = runner.invoke(
        app,
        [
            "benchmark",
            "budgets",
            "--config",
            str(config_path),
            "--evaluation-split",
            "unseen",
        ],
    )
    assert invalid_split.exit_code == 1
    assert "Invalid --evaluation-split" in invalid_split.output
    assert len(dispatch) == 1

    for invalid in ("64,-1", "foo,128"):
        result = runner.invoke(
            app,
            [
                "benchmark",
                "budgets",
                "--config",
                str(config_path),
                "--budgets",
                invalid,
            ],
        )
        assert result.exit_code == 1
        clean_result = _clean(result.output)
        assert "Benchmark failed with error" in clean_result
        assert "Traceback" not in clean_result
    assert len(dispatch) == 1

    malformed_seeds = runner.invoke(
        app,
        [
            "benchmark",
            "budgets",
            "--config",
            str(config_path),
            "--budgets",
            "64,128",
            "--eval-seeds",
            "11,,12",
        ],
    )
    assert malformed_seeds.exit_code == 1
    clean_malformed = _clean(malformed_seeds.output)
    assert "Malformed evaluation seed list" in clean_malformed
    assert "Traceback" not in clean_malformed
    assert len(dispatch) == 1

    invalid_seed_value = runner.invoke(
        app,
        [
            "benchmark",
            "budgets",
            "--config",
            str(config_path),
            "--budgets",
            "64,128",
            "--eval-seeds",
            "foo,12",
        ],
    )
    assert invalid_seed_value.exit_code == 1
    clean_invalid = _clean(invalid_seed_value.output)
    assert "Malformed evaluation seed value: 'foo'" in clean_invalid
    assert len(dispatch) == 1


def _install_fake_matplotlib(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, list[MagicMock], Any]:
    """Install an in-memory matplotlib so plot tests never need the real library."""
    figure = MagicMock()
    figure.savefig.side_effect = lambda path, dpi, metadata: Path(path).touch()
    axes = [MagicMock(), MagicMock()]
    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.subplots = MagicMock(return_value=(figure, axes))  # type: ignore[attr-defined]
    pyplot.close = MagicMock()  # type: ignore[attr-defined]
    matplotlib = ModuleType("matplotlib")
    matplotlib.use = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    return figure, axes, pyplot


def test_plot_x_axis_defaults_to_actual_trained_timesteps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl.benchmarking.learning_curve import plot_learning_curve

    _, axes, _ = _install_fake_matplotlib(monkeypatch)
    # Requested budget 65 rounds up to 128 trained timesteps under n_steps=64.
    point = _make_point(budget_timesteps=65, trained_timesteps=128)
    result = _make_result(points=[point], budgets=[65])

    plot_learning_curve(result, tmp_path / "curve.png")

    assert list(axes[0].plot.call_args.args[0]) == [128]
    axes[0].set_xlabel.assert_called_once_with("Actual trained timesteps")
    assert list(axes[1].plot.call_args.args[0]) == [128]
    axes[1].set_xlabel.assert_called_once_with("Actual trained timesteps")


def test_plot_x_axis_requested_budget_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl.benchmarking.learning_curve import plot_learning_curve

    _, axes, _ = _install_fake_matplotlib(monkeypatch)
    point = _make_point(budget_timesteps=65, trained_timesteps=128)
    result = _make_result(points=[point], budgets=[65])

    plot_learning_curve(result, tmp_path / "curve.png", x_axis="requested")

    assert list(axes[0].plot.call_args.args[0]) == [65]
    axes[0].set_xlabel.assert_called_once_with("Requested training budget (timesteps)")
    assert list(axes[1].plot.call_args.args[0]) == [65]


def test_plot_rejects_unknown_x_axis(tmp_path: Path) -> None:
    from adaptive_rl.benchmarking.learning_curve import plot_learning_curve

    with pytest.raises(ValueError, match="x_axis must be one of"):
        plot_learning_curve(
            _make_result(),
            tmp_path / "curve.png",
            x_axis="sideways",  # type: ignore[arg-type]
        )


def test_benchmark_rejects_unknown_plot_x_axis_before_training(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    with pytest.raises(ValueError, match="plot_x_axis must be one of"):
        run_learning_curve_benchmark(
            config,
            budgets=[64],
            output_dir=tmp_path / "axis",
            plot_x_axis="sideways",  # type: ignore[arg-type]
        )
    assert not (tmp_path / "axis").exists()


def test_benchmark_rejects_non_ppo_algorithms(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.algorithm.name = "sac"
    with pytest.raises(ValueError, match="supports only PPO"):
        run_learning_curve_benchmark(config, budgets=[64], output_dir=tmp_path / "sac")
    assert not (tmp_path / "sac").exists()

    # Whitespace/case variants of the PPO name still pass the gate and fail
    # later for an unrelated reason, proving the gate accepted them.
    config.algorithm.name = "PPO "
    with pytest.raises(ValueError, match="plot_x_axis must be one of"):
        run_learning_curve_benchmark(
            config,
            budgets=[64],
            output_dir=tmp_path / "ppo_variants",
            plot_x_axis="sideways",  # type: ignore[arg-type]
        )
    assert not (tmp_path / "ppo_variants").exists()


def test_benchmark_training_time_is_training_operation_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """training_time_seconds must be the trainer's own measurement, not wall clock."""
    import adaptive_rl.benchmarking.learning_curve as learning_curve_module
    from adaptive_rl.evaluation.statistics import DescriptiveMetrics

    config = _make_config(tmp_path)

    class FakeEnv:
        def close(self) -> None:
            return None

    class FakeAlgorithm:
        num_timesteps = 64

    class FakeTrainer:
        def __init__(self, cfg: ExperimentConfig, env: FakeEnv) -> None:
            self.config = cfg
            self.env = env
            self.algorithm = FakeAlgorithm()

        def fit(self) -> Any:
            model_path = self.config.output_dir / "models" / (f"{self.config.name}_final.zip")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"model")
            return SimpleNamespace(final_model_path=model_path, training_time_seconds=1234.5)

        def close(self) -> None:
            self.env.close()

    def fake_evaluate(*args: Any, **kwargs: Any) -> Any:
        return (
            DescriptiveMetrics(
                episodes=2,
                success_rate=1.0,
                collision_rate=0.0,
                timeout_rate=0.0,
                mean_reward=3.0,
                std_reward=None,
                mean_episode_length=2.0,
            ),
            [{"seed": 11, "episodes": 1}],
            {
                "mean_reward": {
                    "mean": 3.0,
                    "std": None,
                    "ci95_lower": None,
                    "ci95_upper": None,
                    "sample_count": 1,
                }
            },
        )

    monkeypatch.setattr(learning_curve_module, "_make_env", lambda *a, **k: FakeEnv())
    monkeypatch.setattr(
        learning_curve_module,
        "_make_trainer",
        lambda cfg, env: FakeTrainer(cfg, env),
    )
    monkeypatch.setattr(learning_curve_module, "_evaluate_model", fake_evaluate)

    started = time.perf_counter()
    result = run_learning_curve_benchmark(config, budgets=[64], output_dir=tmp_path / "timing")
    elapsed = time.perf_counter() - started

    point = result.points[0]
    # The sentinel proves the value came from PPOAlgorithm.train() itself.
    assert point.training_time_seconds == 1234.5
    assert point.training_time_seconds > elapsed
    document = result.to_dict()
    assert document["results"][0]["training_time_seconds"] == 1234.5
    assert document["results"][0]["training_metadata"]["training_time_seconds"] == 1234.5
    assert document["benchmark"]["training_time_semantics"].startswith("monotonic elapsed time")


def test_benchmark_distinguishes_pooled_and_seed_level_sample_units(
    tmp_path: Path,
) -> None:
    """Pooled descriptive stats count episodes; cross-seed stats count seed groups."""
    config = _make_config(tmp_path)
    assert config.benchmark is not None
    config.benchmark.evaluation_seeds = [11, 12]
    config.benchmark.evaluation_episodes = 2

    result = run_learning_curve_benchmark(config, budgets=[64], output_dir=tmp_path / "units")

    row = result.to_dict()["results"][0]
    descriptive = row["descriptive_metrics"]
    assert descriptive["episodes"] == 4  # 2 seed groups x 2 episodes
    assert row["cross_seed_statistics"]["mean_reward"]["sample_count"] == 2  # 2 groups
    assert len(row["per_seed_summaries"]) == 2
    assert row["per_seed_summaries"][0]["evaluation_group_seed"] == 11
    assert descriptive["mean_reward"] == row["mean_reward"]
    assert row["training_metadata"]["episodes_per_seed"] == 2


def test_partial_benchmark_failure_preserves_completed_budget_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import adaptive_rl.benchmarking.learning_curve as learning_curve_module

    config = _make_config(tmp_path)
    real_run_single = learning_curve_module._run_single_budget

    def flaky(config: ExperimentConfig, budget: int, **kwargs: Any) -> Any:
        if budget == 128:
            raise RuntimeError("injected budget failure")
        return real_run_single(config, budget, **kwargs)

    monkeypatch.setattr(learning_curve_module, "_run_single_budget", flaky)

    with pytest.raises(BenchmarkRunError) as excinfo:
        run_learning_curve_benchmark(config, budgets=[64, 128], output_dir=tmp_path / "partial")

    error = excinfo.value
    result = error.result
    assert "Traceback" not in str(error)
    assert str(error) == ("Benchmark failed at budget 128: RuntimeError: injected budget failure")
    assert result.status == "failed"
    assert result.completed_budgets == [64]
    assert result.failed_budget == 128
    assert result.error == "RuntimeError: injected budget failure"

    model_path = Path(result.points[0].model_path)
    assert model_path.is_file()

    document = json.loads(result.json_path.read_text(encoding="utf-8"))
    json.dumps(document, allow_nan=False)
    assert document["status"] == "failed"
    assert document["completed_budgets"] == [64]
    assert document["failed_budget"] == 128
    assert document["error"] == "RuntimeError: injected budget failure"
    assert len(document["results"]) == 1
    assert document["results"][0]["budget_timesteps"] == 64

    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert int(rows[0]["budget_timesteps"]) == 64


def test_first_budget_failure_still_writes_status_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import adaptive_rl.benchmarking.learning_curve as learning_curve_module

    config = _make_config(tmp_path)
    target_dir = tmp_path / "first_failure"

    def always_fails(config: ExperimentConfig, budget: int, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(learning_curve_module, "_run_single_budget", always_fails)

    with pytest.raises(BenchmarkRunError) as excinfo:
        run_learning_curve_benchmark(config, budgets=[64], output_dir=target_dir)

    result = excinfo.value.result
    assert result.status == "failed"
    assert result.completed_budgets == []
    assert result.failed_budget == 64
    assert result.error is not None and "boom" in result.error

    document = json.loads(result.json_path.read_text(encoding="utf-8"))
    json.dumps(document, allow_nan=False)
    assert document["status"] == "failed"
    assert document["completed_budgets"] == []
    assert document["results"] == []

    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []


def test_failed_budget_does_not_publish_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import adaptive_rl.benchmarking.learning_curve as learning_curve_module

    config = _make_config(tmp_path)
    target_dir = tmp_path / "staged_failure"

    def fail_evaluation(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(learning_curve_module, "_evaluate_model", fail_evaluation)

    with pytest.raises(BenchmarkRunError, match="evaluation failed"):
        run_learning_curve_benchmark(config, budgets=[64], output_dir=target_dir)

    budget_parent = target_dir / "learning_curve"
    assert not (budget_parent / "budget_64").exists()
    assert list(budget_parent.glob(".budget_64.*")) == []


def test_existing_benchmark_output_is_not_overwritten(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    target_dir = tmp_path / "existing_output"
    target_dir.mkdir()
    report = target_dir / "learning_curve_budget.json"
    report.write_text('{"prior_run": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="choose a fresh output directory"):
        run_learning_curve_benchmark(config, budgets=[64], output_dir=target_dir)

    assert report.read_text(encoding="utf-8") == '{"prior_run": true}\n'
    assert not (target_dir / "learning_curve").exists()


def test_cli_reports_partial_benchmark_failure_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl import benchmarking
    from adaptive_rl.cli import app

    config_path = tmp_path / "benchmark.yaml"
    save_config(_make_config(tmp_path), config_path)
    partial = LearningCurveBenchmarkResult(
        benchmark_name="ppo_learning_curve",
        algorithm="ppo",
        environment="drone",
        budgets=[64, 128],
        training_seed=17,
        evaluation_seeds=[11, 12],
        evaluation_episodes=1,
        deterministic=True,
        status="failed",
        failed_budget=128,
        error="RuntimeError: injected budget failure",
        json_path=tmp_path / "learning_curve_budget.json",
        csv_path=tmp_path / "learning_curve_budget.csv",
    )

    def fail_benchmark(config: ExperimentConfig, **kwargs: Any) -> Any:
        raise BenchmarkRunError(
            "Benchmark failed at budget 128: RuntimeError: injected budget failure",
            partial,
        )

    monkeypatch.setattr(benchmarking, "run_learning_curve_benchmark", fail_benchmark)

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "budgets", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "FAILED" in result.output
    assert "Failed budget: 128" in result.output.replace(
        "[bold]Failed budget:[/bold]", "Failed budget:"
    )
    assert "learning-curve benchmark complete" not in result.output
    assert "Traceback" not in result.output


def test_strict_json_rejects_unserializable_values(tmp_path: Path) -> None:
    base = _make_result(points=[_make_point()], budgets=[64])
    json.dumps(base.to_dict(), allow_nan=False)

    unavailable = _make_result(points=[_make_point(success_rate=None)])
    assert unavailable.to_dict()["results"][0]["success_rate"] is None

    nan_point = _make_result(points=[_make_point(mean_reward=float("nan"))])
    with pytest.raises(ValueError, match="Non-finite number at"):
        nan_point.to_dict()

    inf_point = _make_result(points=[_make_point(success_rate=float("inf"))])
    with pytest.raises(ValueError, match="Non-finite number at"):
        inf_point.to_dict()

    numpy_point = _make_result(points=[_make_point(mean_reward=np.float32(1.0))])
    with pytest.raises(ValueError, match="not JSON-serializable: float32"):
        numpy_point.to_dict()

    path_point = _make_result(points=[_make_point(model_path=tmp_path / "model.zip")])
    with pytest.raises(ValueError, match="not JSON-serializable: PosixPath"):
        path_point.to_dict()


def test_plot_failure_preserves_json_and_csv_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the optional-dependency failure path deterministically.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    config = _make_config(tmp_path)
    target_dir = tmp_path / "plotfail"

    result = run_learning_curve_benchmark(config, budgets=[64], output_dir=target_dir, plot=True)

    assert result.status == "completed"
    assert result.plot_error is not None
    assert "matplotlib" in result.plot_error
    assert not (target_dir / "learning_curve_budget.png").exists()

    document = json.loads(result.json_path.read_text(encoding="utf-8"))
    json.dumps(document, allow_nan=False)
    assert document["status"] == "completed"
    assert document["plot"]["requested"] is True
    assert document["plot"]["error"] == result.plot_error
    assert result.csv_path.is_file()
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
