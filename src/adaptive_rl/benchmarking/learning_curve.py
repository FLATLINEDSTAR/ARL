"""Benchmark orchestration for PPO learning curves across training budgets.

The benchmark is PPO-specific (issue #245): it trains a fresh model for each
requested budget under a fixed training seed, evaluates every budget with
identical evaluation conditions, and serializes provenance-rich JSON/CSV
artifacts. Evaluation and cross-seed statistics are delegated to
:mod:`adaptive_rl.evaluation` rather than reimplemented here.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from adaptive_rl.config import BenchmarkConfig, ExperimentConfig

if TYPE_CHECKING:
    from adaptive_rl.evaluation.statistics import DescriptiveMetrics

__all__ = [
    "BenchmarkRunError",
    "LearningCurveBenchmarkResult",
    "LearningCurvePoint",
    "PLOT_X_AXES",
    "plot_learning_curve",
    "run_learning_curve_benchmark",
    "validate_budgets",
]

#: Supported x-axis semantics for :func:`plot_learning_curve`.
PLOT_X_AXES: tuple[str, ...] = ("trained", "requested")

#: Stable CSV header for the per-budget benchmark table.
CSV_FIELDNAMES: tuple[str, ...] = (
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
)


class BenchmarkRunError(RuntimeError):
    """A budget run failed; carries the partial benchmark result.

    Successful budget artifacts are already on disk when this is raised, and
    ``result`` describes exactly which budgets completed and which failed.
    """

    def __init__(self, message: str, result: LearningCurveBenchmarkResult) -> None:
        super().__init__(message)
        self.result = result


def _validate_strict_json_payload(payload: Any, path: str = "$") -> None:
    """Reject values that ``json.dumps(..., allow_nan=False)`` cannot encode.

    Non-finite floats, numpy scalars that are not float subclasses, ``Path``
    objects, and any other non-JSON-native type raise ``ValueError`` with the
    offending location instead of leaking into benchmark artifacts.
    """
    if payload is None or isinstance(payload, (str, bool, int)):
        return
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise ValueError(f"Non-finite number at {path}: {payload!r}.")
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise ValueError(f"Non-string object key at {path}: {key!r}.")
            _validate_strict_json_payload(value, f"{path}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _validate_strict_json_payload(value, f"{path}[{index}]")
        return
    raise ValueError(f"Value at {path} is not JSON-serializable: {type(payload).__name__}.")


def _stable_json_value(value: Any, path: str = "$") -> Any:
    """Normalize configuration values without serializing unstable objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite configuration value at {path}: {value!r}.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Configuration keys must be strings at {path}: {key!r}.")
            normalized[key] = _stable_json_value(item, f"{path}.{key}")
        return normalized
    raise ValueError(
        f"Unsupported configuration value at {path}: {type(value).__name__}; "
        "use JSON-compatible values."
    )


def _environment_configuration(
    config: ExperimentConfig, evaluation_split: str
) -> tuple[dict[str, Any], str]:
    training = _stable_json_value(
        config.environment.model_dump(mode="python"), "$.environment.training"
    )
    evaluation = copy.deepcopy(training)
    evaluation_parameters = evaluation["parameters"]
    if evaluation_split == "custom":
        evaluation_parameters.pop("split", None)
    else:
        evaluation_parameters["split"] = evaluation_split
    normalized = {"training": training, "evaluation": evaluation}
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return normalized, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _runtime_provenance() -> dict[str, Any]:
    """Record stable runtime/version information without importing optional packages."""
    import adaptive_rl

    packages: dict[str, str | None] = {}
    for distribution in ("torch", "stable-baselines3", "gymnasium"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None

    repository_root = Path(__file__).resolve().parents[3]
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    try:
        working_tree_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        working_tree_dirty = None

    return {
        "adaptive_rl_version": adaptive_rl.__version__,
        "python_version": platform.python_version(),
        "packages": packages,
        "git_commit": git_commit,
        "working_tree_dirty": working_tree_dirty,
    }


def _configuration_fingerprint(
    config: ExperimentConfig,
    *,
    budgets: Sequence[int],
    training_seed: int,
    evaluation_seeds: Sequence[int],
    evaluation_episodes: int,
    deterministic: bool,
    evaluation_split: str,
) -> str:
    """Fingerprint the input config together with effective benchmark overrides."""
    experiment_config = config.model_dump(mode="python")
    experiment_config.pop("benchmark", None)
    experiment_config.pop("output_dir", None)
    experiment_config.pop("log_dir", None)
    identity = {
        "experiment_config": experiment_config,
        "effective_benchmark": {
            "budgets": list(budgets),
            "training_seed": training_seed,
            "evaluation_seeds": list(evaluation_seeds),
            "evaluation_episodes": evaluation_episodes,
            "deterministic": deterministic,
            "evaluation_split": evaluation_split,
        },
    }
    canonical = json.dumps(
        _stable_json_value(identity), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write and sync a sibling temporary file before atomically replacing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_budgets(
    raw_budgets: Sequence[int] | str | None, *, allow_empty: bool = False
) -> list[int]:
    """Validate and normalize benchmark budgets.

    Rules:
      - reject empty, zero, negative, non-integer, malformed comma-separated values
      - reject duplicates after normalization
      - canonicalize to ascending order; caller-provided ordering is not preserved
    """
    if raw_budgets is None:
        if allow_empty:
            return []
        raise ValueError("Benchmark budgets cannot be empty.")

    if isinstance(raw_budgets, str):
        values = [part.strip() for part in raw_budgets.split(",")]
        if not values or any(part == "" for part in values):
            raise ValueError("Malformed budget list: expected comma-separated integers.")
        normalized: list[int] = []
        for token in values:
            try:
                normalized.append(int(token))
            except ValueError as exc:  # pragma: no cover - explicit validation path
                raise ValueError(f"Malformed budget value: {token!r}") from exc
    else:
        normalized = list(raw_budgets)

    if not normalized:
        if allow_empty:
            return []
        raise ValueError("Benchmark budgets cannot be empty.")

    cleaned: list[int] = []
    seen: set[int] = set()
    for value in normalized:
        if isinstance(value, bool):
            raise ValueError(f"Budget values must be integers, got boolean {value!r}.")
        if not isinstance(value, int):
            raise ValueError(
                f"Budget values must be integers, got {type(value).__name__}: {value!r}"
            )
        if value <= 0:
            raise ValueError(f"Budget values must be positive, got {value!r}.")
        if value in seen:
            raise ValueError(f"Duplicate budget value detected: {value}.")
        seen.add(value)
        cleaned.append(value)

    cleaned.sort()
    return cleaned


@dataclass(frozen=True)
class LearningCurvePoint:
    """Performance record for a single training-budget evaluation point.

    Three metric layers are kept distinct and must not be conflated:

    * ``descriptive_metrics`` (and the identically valued legacy top-level
      ``success_rate``/``collision_rate``/``timeout_rate``/``mean_reward``/
      ``std_reward``/``mean_episode_length`` fields) pools every evaluated
      episode for this budget — descriptive values only;
    * ``per_seed_summaries`` holds exactly one summary per evaluation seed
      group, where the group seed is the statistical unit;
    * ``cross_seed_statistics`` holds Student's t statistics computed *across*
      those seed-level summaries, never across pooled episodes.

    ``training_metadata`` records how the model behind this point was trained
    and evaluated. ``budget_timesteps`` is the requested budget while
    ``trained_timesteps`` is what Stable-Baselines3 actually collected; the
    two differ whenever the budget is not aligned to a rollout boundary.
    """

    budget_timesteps: int
    trained_timesteps: int
    success_rate: float | None
    collision_rate: float | None
    timeout_rate: float | None
    mean_reward: float
    std_reward: float | None
    mean_episode_length: float
    training_time_seconds: float
    model_path: str
    training_seed: int
    evaluation_seeds: list[int]
    evaluation_episodes: int
    deterministic: bool
    algorithm: str
    environment: str
    evaluation_split: str = "custom"
    environment_fingerprint: str = ""
    device: str = "unknown"
    per_seed_summaries: list[dict[str, int | float | None]] = field(default_factory=list)
    cross_seed_statistics: dict[str, dict[str, float | int | None]] = field(default_factory=dict)

    @property
    def evaluation_group_seeds(self) -> list[int]:
        """Evaluation seed groups (the statistical grouping unit) for this point."""
        return list(self.evaluation_seeds)

    @property
    def episodes_per_seed(self) -> int:
        """Episodes evaluated inside each evaluation seed group."""
        return self.evaluation_episodes

    @property
    def descriptive_metrics(self) -> dict[str, float | int | None]:
        """Pooled episode-level descriptive statistics for this budget.

        These values summarize the evaluated episodes as one sample and are
        generally *not* equal to the means stored in ``cross_seed_statistics``
        when seed groups contribute unequal numbers of episodes.
        """
        return {
            "episodes": len(self.evaluation_seeds) * self.evaluation_episodes,
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "timeout_rate": self.timeout_rate,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_episode_length": self.mean_episode_length,
        }

    @property
    def training_metadata(self) -> dict[str, Any]:
        """Provenance for how this budget's model was trained and evaluated."""
        return {
            "budget_timesteps": self.budget_timesteps,
            "trained_timesteps": self.trained_timesteps,
            "training_seed": self.training_seed,
            "training_time_seconds": self.training_time_seconds,
            "algorithm": self.algorithm,
            "environment": self.environment,
            "model_path": self.model_path,
            "deterministic": self.deterministic,
            "evaluation_group_seeds": list(self.evaluation_seeds),
            "episodes_per_seed": self.evaluation_episodes,
            "evaluation_split": self.evaluation_split,
            "environment_fingerprint": self.environment_fingerprint,
            "device": self.device,
        }


@dataclass
class LearningCurveBenchmarkResult:
    """Top-level container for the ordered learning-curve benchmark output.

    ``status`` makes partial execution explicit: ``"completed"`` means every
    requested budget trained and evaluated, while ``"failed"`` means the run
    stopped at ``failed_budget`` after ``completed_budgets`` finished.
    ``completed_budgets``, ``failed_budget``, and ``error`` (a sanitized
    ``TypeName: message`` string, never a traceback) are always serialized so
    artifact consumers can tell a partial benchmark from a complete one.
    """

    benchmark_name: str
    algorithm: str
    environment: str
    budgets: list[int]
    training_seed: int
    evaluation_seeds: list[int]
    evaluation_episodes: int
    deterministic: bool
    evaluation_split: str = "custom"
    environment_configuration: dict[str, Any] = field(default_factory=dict)
    environment_fingerprint: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    points: list[LearningCurvePoint] = field(default_factory=list)
    plot_data: dict[str, list[float | int | None]] = field(default_factory=dict)
    output_dir: Path | None = None
    json_path: Path | None = None
    csv_path: Path | None = None
    plot_path: Path | None = None
    status: str = "completed"
    failed_budget: int | None = None
    error: str | None = None
    plot_error: str | None = None

    @property
    def completed_budgets(self) -> list[int]:
        """Budgets that trained and evaluated successfully, in run order."""
        return [point.budget_timesteps for point in self.points]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the benchmark result to a strict JSON-friendly dictionary."""
        payload: dict[str, Any] = {
            "status": self.status,
            "completed_budgets": list(self.completed_budgets),
            "failed_budget": self.failed_budget,
            "error": self.error,
            "benchmark": {
                "name": self.benchmark_name,
                "algorithm": self.algorithm,
                "environment": self.environment,
                "training_seed": self.training_seed,
                "evaluation_seeds": self.evaluation_seeds,
                "evaluation_group_seeds": list(self.evaluation_seeds),
                "evaluation_episodes": self.evaluation_episodes,
                "episodes_per_seed": self.evaluation_episodes,
                "evaluation_split": self.evaluation_split,
                "deterministic": self.deterministic,
                "budgets": self.budgets,
                "seed_semantics": (
                    "evaluation_seeds are evaluation group seeds (the statistical grouping "
                    "unit); each group evaluates episodes_per_seed episodes"
                ),
                "metric_semantics": {
                    "aggregation": (
                        "descriptive_metrics (and the legacy top-level metric fields) pool "
                        "every evaluated episode and are descriptive only"
                    ),
                    "success_rate": "pooled over episodes with available success metadata",
                    "collision_rate": "pooled over episodes with available collision metadata",
                    "timeout_rate": "fraction of all evaluated episodes with truncated=True",
                    "mean_reward": "mean of pooled episode returns",
                    "std_reward": "sample standard deviation across pooled episode returns",
                    "mean_episode_length": "mean of pooled episode lengths",
                    "per_seed_summaries": (
                        "one summary per evaluation seed group, keyed by the group seed"
                    ),
                    "cross_seed_statistics": (
                        "Student's t summaries across evaluator per-seed summaries, not across "
                        "pooled episodes; within-seed reward and length standard deviations use "
                        "evaluator semantics"
                    ),
                },
                "training_time_semantics": (
                    "monotonic elapsed time inside PPOAlgorithm.train() only; excludes model "
                    "serialization, metadata writing, evaluation, and artifact export"
                ),
                "episode_reset_seed_mapping": (
                    "custom: injective Cantor pairing "
                    "((group_seed + episode_index) * (group_seed + episode_index + 1)) // 2 "
                    "+ episode_index; configured train/test split: SHA-256 mapping reduced "
                    "into the finite split interval (sampling with replacement is possible)"
                ),
            },
            "provenance": {
                **self.provenance,
                "environment_configuration": self.environment_configuration,
                "environment_fingerprint_sha256": self.environment_fingerprint,
            },
            "results": [
                {
                    "budget_timesteps": point.budget_timesteps,
                    "trained_timesteps": point.trained_timesteps,
                    "success_rate": point.success_rate,
                    "collision_rate": point.collision_rate,
                    "timeout_rate": point.timeout_rate,
                    "mean_reward": point.mean_reward,
                    "std_reward": point.std_reward,
                    "mean_episode_length": point.mean_episode_length,
                    "training_time_seconds": point.training_time_seconds,
                    "model_path": point.model_path,
                    "training_seed": point.training_seed,
                    "evaluation_seeds": point.evaluation_seeds,
                    "evaluation_episodes": point.evaluation_episodes,
                    "deterministic": point.deterministic,
                    "evaluation_split": point.evaluation_split,
                    "algorithm": point.algorithm,
                    "environment": point.environment,
                    "environment_fingerprint": point.environment_fingerprint,
                    "device": point.device,
                    "descriptive_metrics": point.descriptive_metrics,
                    "per_seed_summaries": point.per_seed_summaries,
                    "cross_seed_statistics": point.cross_seed_statistics,
                    "training_metadata": point.training_metadata,
                }
                for point in self.points
            ],
            "plot": {
                "requested": self.plot_path is not None,
                "path": str(self.plot_path) if self.plot_path is not None else None,
                "error": self.plot_error,
            },
            "plot_data": self.plot_data,
        }
        _validate_strict_json_payload(payload)
        return payload


def _resolve_benchmark_config(
    base_config: ExperimentConfig, overrides: dict[str, Any] | None
) -> BenchmarkConfig:
    """Merge benchmark settings into a config object without mutating the original."""
    if base_config.benchmark is not None:
        benchmark_cfg = BenchmarkConfig.model_validate(
            base_config.benchmark.model_dump(mode="python")
        )
    else:
        benchmark_cfg = BenchmarkConfig()

    if overrides:
        benchmark_cfg = benchmark_cfg.model_copy(update=overrides)
    return benchmark_cfg


def _budget_dir(base_output_dir: Path, budget: int) -> Path:
    return base_output_dir / "learning_curve" / f"budget_{budget}"


def _make_env(env_name: str, **env_kwargs: Any) -> Any:
    from adaptive_rl.environments.registry import make_env

    return make_env(env_name, **env_kwargs)


def _make_trainer(config: ExperimentConfig, env: Any) -> Any:
    from adaptive_rl.training.trainer import PPOTrainer

    return PPOTrainer(config=config, env=env)


def _load_evaluator(model_path: Path, env: Any) -> tuple[Any, Any]:
    from adaptive_rl.algorithms.ppo import PPOAlgorithm
    from adaptive_rl.evaluation.evaluator import Evaluator

    algorithm = PPOAlgorithm.from_pretrained(model_path, env=env)
    return algorithm, Evaluator(algorithm=algorithm, env=env)


def _evaluate_model(
    model_path: Path,
    *,
    env_name: str,
    env_kwargs: dict[str, Any],
    evaluation_seeds: Sequence[int],
    evaluation_episodes: int,
    deterministic: bool,
    evaluation_split: str,
) -> tuple[
    DescriptiveMetrics,
    list[dict[str, Any]],
    dict[str, dict[str, float | int | None]],
]:
    """Evaluate one trained model under fixed evaluation conditions.

    Returns pooled descriptive metrics, one summary per evaluation seed
    group, and cross-seed statistics — three deliberately distinct layers.
    """
    from adaptive_rl.evaluation.statistics import summarize_descriptive_episodes

    env = _make_env(env_name, **env_kwargs)
    try:
        _, evaluator = _load_evaluator(model_path, env)
        evaluation = evaluator.evaluate_seeds(
            seeds=evaluation_seeds,
            episodes_per_seed=evaluation_episodes,
            deterministic=deterministic,
            split=None if evaluation_split == "custom" else evaluation_split,
        )
        records = evaluation.episodes
        if not records:
            raise RuntimeError("Evaluation produced no episodes.")
        descriptive = summarize_descriptive_episodes(
            rewards=[record.return_value for record in records],
            episode_lengths=[float(record.episode_length) for record in records],
            successes=[record.success for record in records],
            collisions=[record.collision for record in records],
            truncations=[record.truncated for record in records],
        )
        return (
            descriptive,
            [summary.to_dict() for summary in evaluation.per_seed],
            {name: stats.to_dict() for name, stats in evaluation.aggregate.items()},
        )
    finally:
        env.close()


def _run_single_budget(
    config: ExperimentConfig,
    budget: int,
    *,
    training_seed: int,
    evaluation_seeds: Sequence[int],
    evaluation_episodes: int,
    deterministic: bool,
    output_base_dir: Path,
    evaluation_split: str,
    environment_fingerprint: str,
) -> LearningCurvePoint:
    """Run a single budget as a fresh training process from the same base configuration."""
    benchmark_dir = _budget_dir(output_base_dir, budget)
    benchmark_parent = benchmark_dir.parent
    benchmark_parent.mkdir(parents=True, exist_ok=True)
    if benchmark_dir.exists():
        raise FileExistsError(
            f"Completed artifact directory already exists for budget {budget}: {benchmark_dir}"
        )

    lock_path = benchmark_parent / f".budget_{budget}.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Budget {budget} is already reserved by another or interrupted benchmark: {lock_path}"
        ) from exc

    temporary_dir: Path | None = None
    try:
        os.close(lock_descriptor)
        temporary_dir = Path(tempfile.mkdtemp(prefix=f".budget_{budget}.", dir=benchmark_parent))
        config_copy = config.model_copy(deep=True)
        config_copy.name = f"ppo_budget_{budget}"
        config_copy.seed = training_seed
        config_copy.algorithm.parameters.pop("seed", None)
        if config_copy.training is None:
            raise ValueError("A training section is required to run the learning-curve benchmark.")
        config_copy.training.total_timesteps = budget
        config_copy.output_dir = temporary_dir
        config_copy.log_dir = temporary_dir / "logs"

        env = _make_env(config_copy.environment.name, **config_copy.environment.parameters)
        trainer: Any = None
        try:
            trainer = _make_trainer(config_copy, env)
            result = trainer.fit()
            training_time_seconds = float(result.training_time_seconds)
            trained_timesteps = int(trainer.algorithm.num_timesteps)
            model = getattr(trainer.algorithm, "model", None)
            device = str(
                getattr(model, "device", config_copy.algorithm.parameters.get("device", "auto"))
            )
        finally:
            if trainer is not None:
                trainer.close()
            else:
                env.close()

        staged_model_path = Path(result.final_model_path)
        if not staged_model_path.exists():
            raise FileNotFoundError(
                f"Training budget {budget} did not create model artifact: {staged_model_path}"
            )
        if not math.isfinite(training_time_seconds) or training_time_seconds < 0:
            raise RuntimeError(
                f"Invalid training duration for budget {budget}: {training_time_seconds}"
            )

        evaluation_env_kwargs = dict(config_copy.environment.parameters)
        if evaluation_split == "custom":
            evaluation_env_kwargs.pop("split", None)
        else:
            evaluation_env_kwargs["split"] = evaluation_split
        descriptive, per_seed_summaries, cross_seed_statistics = _evaluate_model(
            staged_model_path,
            env_name=config_copy.environment.name,
            env_kwargs=evaluation_env_kwargs,
            evaluation_seeds=evaluation_seeds,
            evaluation_episodes=evaluation_episodes,
            deterministic=deterministic,
            evaluation_split=evaluation_split,
        )

        final_model_path = benchmark_dir / staged_model_path.relative_to(temporary_dir)
        point = LearningCurvePoint(
            budget_timesteps=budget,
            trained_timesteps=trained_timesteps,
            success_rate=descriptive.success_rate,
            collision_rate=descriptive.collision_rate,
            timeout_rate=descriptive.timeout_rate,
            mean_reward=descriptive.mean_reward,
            std_reward=descriptive.std_reward,
            mean_episode_length=descriptive.mean_episode_length,
            training_time_seconds=training_time_seconds,
            model_path=str(final_model_path),
            training_seed=training_seed,
            evaluation_seeds=list(evaluation_seeds),
            evaluation_episodes=evaluation_episodes,
            deterministic=deterministic,
            algorithm=config_copy.algorithm.name,
            environment=config_copy.environment.name,
            evaluation_split=evaluation_split,
            environment_fingerprint=environment_fingerprint,
            device=device,
            per_seed_summaries=per_seed_summaries,
            cross_seed_statistics=cross_seed_statistics,
        )
        budget_payload = {
            "status": "completed",
            "result": {
                "budget_timesteps": point.budget_timesteps,
                "trained_timesteps": point.trained_timesteps,
                "descriptive_metrics": point.descriptive_metrics,
                "training_metadata": point.training_metadata,
                "per_seed_summaries": point.per_seed_summaries,
                "cross_seed_statistics": point.cross_seed_statistics,
            },
        }
        _validate_strict_json_payload(budget_payload)
        _atomic_write_text(
            temporary_dir / "budget_result.json",
            json.dumps(budget_payload, indent=2, allow_nan=False) + "\n",
        )
        os.rename(temporary_dir, benchmark_dir)
        return point
    except Exception:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def _write_json(result: LearningCurveBenchmarkResult) -> None:
    """Write the strict JSON artifact for a (possibly partial) benchmark."""
    if result.json_path is None:
        raise ValueError("A JSON output path is required to serialize the benchmark.")
    result.json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    _atomic_write_text(result.json_path, serialized)


def _csv_row(point: LearningCurvePoint) -> dict[str, str | float | int | None]:
    """Build one stable CSV row, including evaluation provenance."""
    return {
        "budget_timesteps": point.budget_timesteps,
        "trained_timesteps": point.trained_timesteps,
        "pooled_success_rate": point.success_rate,
        "pooled_collision_rate": point.collision_rate,
        "pooled_timeout_rate": point.timeout_rate,
        "pooled_mean_reward": point.mean_reward,
        "pooled_std_reward": point.std_reward,
        "pooled_mean_episode_length": point.mean_episode_length,
        "training_time_seconds": point.training_time_seconds,
        "model_path": point.model_path,
        "training_seed": point.training_seed,
        "evaluation_seeds": ";".join(str(seed) for seed in point.evaluation_seeds),
        "evaluation_episodes": point.evaluation_episodes,
        "evaluation_group_seed_count": len(point.evaluation_seeds),
        "episodes_per_seed": point.evaluation_episodes,
        "evaluation_split": point.evaluation_split,
        "deterministic": str(bool(point.deterministic)),
        "algorithm": point.algorithm,
        "environment": point.environment,
        "environment_fingerprint": point.environment_fingerprint,
        "device": point.device,
    }


def _write_csv(result: LearningCurveBenchmarkResult) -> None:
    """Write one CSV row per completed budget with fixed headers."""
    if result.csv_path is None:
        raise ValueError("A CSV output path is required to serialize the benchmark.")
    result.csv_path.parent.mkdir(parents=True, exist_ok=True)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    for point in result.points:
        writer.writerow(_csv_row(point))
    _atomic_write_text(result.csv_path, output.getvalue())


def run_learning_curve_benchmark(
    config: ExperimentConfig,
    budgets: Sequence[int] | str | None = None,
    *,
    training_seed: int | None = None,
    evaluation_seeds: Sequence[int] | None = None,
    evaluation_episodes: int | None = None,
    deterministic: bool | None = None,
    output_dir: str | Path | None = None,
    plot: bool = False,
    plot_x_axis: Literal["trained", "requested"] = "trained",
    evaluation_split: Literal["custom", "train", "test"] | None = None,
) -> LearningCurveBenchmarkResult:
    """Run the PPO learning-curve benchmark across training budgets.

    Each budget trains a fresh PPO model from the same base configuration and
    a fixed training seed, then is evaluated under identical conditions
    (same evaluation seed groups, episodes per seed, deterministic setting,
    and environment configuration). Only the training budget changes.

    ``training_time_seconds`` on each point measures PPO optimization only
    (see :class:`~adaptive_rl.training.trainer.TrainingResult`).

    If a budget fails, the JSON/CSV artifacts are still written with
    ``status="failed"``, ``completed_budgets``, ``failed_budget``, and a
    sanitized ``error`` message, and :class:`BenchmarkRunError` is raised with
    the partial result attached. Plotting happens only after JSON/CSV are on
    disk, so a plot failure cannot corrupt benchmark data; the failure is
    recorded in ``plot_error`` and the result is still returned.
    """
    algorithm_name = str(config.algorithm.name).strip().lower()
    if algorithm_name != "ppo":
        raise ValueError(
            f"The learning-curve benchmark supports only PPO, got "
            f"{config.algorithm.name!r}. Set algorithm.name to 'ppo' in the "
            "experiment configuration."
        )
    if config.training is None:
        raise ValueError("A training section is required to run the learning-curve benchmark.")
    if plot_x_axis not in PLOT_X_AXES:
        raise ValueError(f"plot_x_axis must be one of {list(PLOT_X_AXES)}, got {plot_x_axis!r}.")

    benchmark_cfg = _resolve_benchmark_config(config, None)
    normalized = validate_budgets(benchmark_cfg.budgets if budgets is None else budgets)
    final_evaluation_split = (
        benchmark_cfg.evaluation_split if evaluation_split is None else evaluation_split
    )
    if final_evaluation_split not in ("custom", "train", "test"):
        raise ValueError("evaluation_split must be one of 'custom', 'train', or 'test'.")

    training_split = config.environment.parameters.get("split")
    if training_split == "test":
        raise ValueError(
            "The training environment cannot use the held-out 'test' split. "
            "Use environment.parameters.split: train for split-based training."
        )
    if final_evaluation_split in ("train", "test") and training_split != "train":
        raise ValueError(
            f"Evaluation split '{final_evaluation_split}' requires "
            "environment.parameters.split: train so test layouts remain held out."
        )

    if training_seed is not None:
        final_training_seed = training_seed
    elif config.benchmark is not None:
        final_training_seed = config.benchmark.training_seed
    else:
        final_training_seed = config.seed

    final_eval_seeds = (
        list(benchmark_cfg.evaluation_seeds) if evaluation_seeds is None else list(evaluation_seeds)
    )

    if isinstance(final_training_seed, bool) or not isinstance(final_training_seed, int):
        raise ValueError("Training seed must be an integer.")
    if final_training_seed < 0:
        raise ValueError("Training seed must be non-negative.")
    if not final_eval_seeds:
        raise ValueError("Evaluation seeds must not be empty.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in final_eval_seeds):
        raise ValueError("Evaluation seeds must be integers.")
    if any(seed < 0 for seed in final_eval_seeds):
        raise ValueError("Evaluation seeds must be non-negative.")
    if len(set(final_eval_seeds)) != len(final_eval_seeds):
        raise ValueError("Evaluation seeds must not contain duplicates.")
    if final_evaluation_split in ("train", "test"):
        from adaptive_rl.evaluation.generalization import validate_split_seed

        for seed in final_eval_seeds:
            validate_split_seed(seed, final_evaluation_split)

    final_eval_episodes = (
        benchmark_cfg.evaluation_episodes if evaluation_episodes is None else evaluation_episodes
    )
    if isinstance(final_eval_episodes, bool) or not isinstance(final_eval_episodes, int):
        raise ValueError("Evaluation episodes per seed must be an integer.")
    if final_eval_episodes <= 0:
        raise ValueError("Evaluation episodes per seed must be positive.")

    if deterministic is None:
        if config.benchmark is not None:
            final_deterministic = config.benchmark.deterministic
        else:
            final_deterministic = config.evaluation.deterministic
    else:
        if not isinstance(deterministic, bool):
            raise ValueError("Deterministic evaluation setting must be a boolean.")
        final_deterministic = deterministic

    if output_dir is None:
        target_dir = Path(config.output_dir) / "benchmarks"
    else:
        target_dir = Path(output_dir)

    json_path = target_dir / "learning_curve_budget.json"
    csv_path = target_dir / "learning_curve_budget.csv"
    plot_output_path = target_dir / "learning_curve_budget.png"
    plot_path = plot_output_path if plot else None
    existing_outputs = [path for path in (json_path, csv_path, plot_output_path) if path.exists()]
    existing_budgets = [
        _budget_dir(target_dir, budget)
        for budget in normalized
        if _budget_dir(target_dir, budget).exists()
    ]
    if existing_outputs or existing_budgets:
        existing = existing_outputs + existing_budgets
        raise FileExistsError(
            "Learning-curve output already exists; choose a fresh output directory: "
            + ", ".join(str(path) for path in existing)
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    environment_configuration, environment_fingerprint = _environment_configuration(
        config, final_evaluation_split
    )
    runtime_provenance = _runtime_provenance()
    runtime_provenance.update(
        {
            "configuration_fingerprint_sha256": _configuration_fingerprint(
                config,
                budgets=normalized,
                training_seed=final_training_seed,
                evaluation_seeds=final_eval_seeds,
                evaluation_episodes=final_eval_episodes,
                deterministic=final_deterministic,
                evaluation_split=final_evaluation_split,
            ),
            "environment_fingerprint_sha256": environment_fingerprint,
            "training_seed": final_training_seed,
            "evaluation_seeds": list(final_eval_seeds),
            "requested_budgets": list(normalized),
        }
    )

    bench_points: list[LearningCurvePoint] = []
    status = "completed"
    failed_budget: int | None = None
    error: str | None = None
    for budget in normalized:
        try:
            bench_points.append(
                _run_single_budget(
                    config,
                    budget,
                    training_seed=final_training_seed,
                    evaluation_seeds=final_eval_seeds,
                    evaluation_episodes=final_eval_episodes,
                    deterministic=final_deterministic,
                    output_base_dir=target_dir,
                    evaluation_split=final_evaluation_split,
                    environment_fingerprint=environment_fingerprint,
                )
            )
        except Exception as exc:
            status = "failed"
            failed_budget = budget
            error = f"{type(exc).__name__}: {exc}"
            break

    result = LearningCurveBenchmarkResult(
        benchmark_name="ppo_learning_curve",
        algorithm=config.algorithm.name,
        environment=config.environment.name,
        budgets=normalized,
        training_seed=final_training_seed,
        evaluation_seeds=list(final_eval_seeds),
        evaluation_episodes=final_eval_episodes,
        deterministic=final_deterministic,
        evaluation_split=final_evaluation_split,
        environment_configuration=environment_configuration,
        environment_fingerprint=environment_fingerprint,
        provenance=runtime_provenance,
        points=bench_points,
        plot_data={
            "budgets": [int(point.budget_timesteps) for point in bench_points],
            "trained_timesteps": [int(point.trained_timesteps) for point in bench_points],
            "success_rate": [point.success_rate for point in bench_points],
            "mean_reward": [point.mean_reward for point in bench_points],
        },
        output_dir=target_dir,
        json_path=json_path,
        csv_path=csv_path,
        plot_path=plot_path,
        status=status,
        failed_budget=failed_budget,
        error=error,
        plot_error=(
            "plot not attempted because the benchmark failed" if status == "failed" else None
        ),
    )

    # Artifacts are written before any plotting so a rendering failure can
    # never corrupt or truncate valid JSON/CSV benchmark data.
    _write_json(result)
    _write_csv(result)

    if status == "failed":
        raise BenchmarkRunError(f"Benchmark failed at budget {failed_budget}: {error}", result)

    if plot:
        try:
            plot_learning_curve(result, plot_path, x_axis=plot_x_axis)
        except Exception as exc:
            result.plot_error = f"{type(exc).__name__}: {exc}"
            _write_json(result)

    return result


def plot_learning_curve(
    result: LearningCurveBenchmarkResult,
    output_path: str | Path | None = None,
    *,
    x_axis: Literal["trained", "requested"] = "trained",
) -> Path:
    """Render a lightweight learning curve for success rate and mean reward.

    Args:
        result: Completed benchmark result to plot.
        output_path: Destination PNG path (defaults to ``result.plot_path``).
        x_axis: ``"trained"`` (default) plots against the actual timesteps
            Stable-Baselines3 collected, ``"requested"`` plots against the
            requested budget. The two differ when a budget is rounded up to a
            rollout boundary; the axis label always states which is used.

    Requires the optional ``plot`` extra (matplotlib); the figure is always
    closed before returning.
    """
    if x_axis not in PLOT_X_AXES:
        raise ValueError(f"x_axis must be one of {list(PLOT_X_AXES)}, got {x_axis!r}.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Plotting requires matplotlib. Install the optional 'plot' extra "
            '(pip install -e ".[plot]") to enable --plot.'
        ) from exc

    plot_target = Path(output_path) if output_path is not None else result.plot_path
    if plot_target is None:
        raise ValueError("An output path is required for plotting.")
    plot_target.parent.mkdir(parents=True, exist_ok=True)

    if x_axis == "trained":
        x_values = [int(point.trained_timesteps) for point in result.points]
        x_label = "Actual trained timesteps"
    else:
        x_values = [int(point.budget_timesteps) for point in result.points]
        x_label = "Requested training budget (timesteps)"
    success_rates = [point.success_rate for point in result.points]
    rewards = [point.mean_reward for point in result.points]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    try:
        valid_success = [
            (x_value, success_rate)
            for x_value, success_rate in zip(x_values, success_rates)
            if success_rate is not None
        ]
        if valid_success:
            valid_x, valid_rates = zip(*valid_success)
            axes[0].plot(valid_x, valid_rates, marker="o", linewidth=2)
        axes[0].set_title("Success rate vs training budget")
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel("Success rate")
        axes[0].set_ylim(-0.05, 1.05)

        axes[1].plot(x_values, rewards, marker="s", linewidth=2, color="tab:orange")
        axes[1].set_title("Mean reward vs training budget")
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("Mean reward")

        fig.savefig(plot_target, dpi=160, metadata={"Software": "AdaptiveRL"})
    finally:
        plt.close(fig)
    return plot_target
