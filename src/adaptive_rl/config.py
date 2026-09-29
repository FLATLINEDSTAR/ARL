"""Configuration system and schemas for AdaptiveRL experiments.

Provides schema validation, YAML loading, and deterministic configuration
management for environments, algorithms, training, and evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)


class ConfigError(Exception):
    """Exception raised for configuration parsing or validation failures."""

    pass


class AlgorithmConfig(BaseModel):
    """Configuration parameters for a reinforcement learning algorithm."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field("ppo", description="Algorithm name, e.g. 'ppo' or 'sac'")
    learning_rate: float = Field(3e-4, gt=0.0, description="Optimizer learning rate")
    gamma: float = Field(0.99, ge=0.0, le=1.0, description="Discount factor")
    batch_size: int = Field(64, gt=0, description="Minibatch size")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional algorithm-specific hyperparameters"
    )

    @model_validator(mode="before")
    @classmethod
    def _alias_params(cls, data: Any) -> Any:
        if isinstance(data, dict) and "params" in data and "parameters" not in data:
            data["parameters"] = data.pop("params")
        return data


class EnvironmentConfig(BaseModel):
    """Configuration parameters for the Gymnasium environment."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field("drone", description="Registered environment name, e.g. 'drone'")
    max_steps: int = Field(200, gt=0, description="Maximum steps per episode")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment-specific parameters (e.g. bounds, obstacles)",
    )

    @model_validator(mode="before")
    @classmethod
    def _alias_params(cls, data: Any) -> Any:
        if isinstance(data, dict) and "params" in data and "parameters" not in data:
            data["parameters"] = data.pop("params")
        return data


class TrainingConfig(BaseModel):
    """Configuration parameters for the training loop."""

    model_config = ConfigDict(extra="forbid")

    total_timesteps: int = Field(25000, gt=0, description="Total environment steps to train")
    checkpoint_freq: int = Field(
        5000, ge=0, description="Frequency of saving model checkpoints (0 = disabled)"
    )
    log_interval: int = Field(10, gt=0, description="Frequency of logging metrics")


class EvaluationConfig(BaseModel):
    """Configuration parameters for evaluation and benchmarking."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    eval_episodes: int = Field(20, gt=0, description="Number of evaluation episodes")
    deterministic: bool = Field(
        True, description="Whether to use deterministic actions in evaluation"
    )

    @model_validator(mode="before")
    @classmethod
    def _alias_episodes(cls, data: Any) -> Any:
        if isinstance(data, dict) and "episodes" in data and "eval_episodes" not in data:
            data["eval_episodes"] = data.pop("episodes")
        return data


class BenchmarkConfig(BaseModel):
    """Configuration for PPO learning-curve benchmarking across training budgets."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    budgets: list[StrictInt] = Field(
        default_factory=lambda: [5000, 10000, 25000, 50000],
        min_length=1,
        description="Training budgets used for the learning-curve benchmark.",
    )
    training_seed: int = Field(42, ge=0, description="Seed used for all benchmark training runs")
    evaluation_seeds: list[StrictInt] = Field(
        default_factory=lambda: [42, 43, 44, 45, 46],
        min_length=1,
        description="Fixed seed sequence used for evaluation across all budgets.",
    )
    evaluation_episodes: int = Field(
        20, gt=0, description="Episodes per seed for benchmark evaluation"
    )
    deterministic: bool = Field(
        True,
        description="Whether to evaluate using deterministic action selection for all budgets.",
    )
    evaluation_split: Literal["custom", "train", "test"] = Field(
        "custom",
        description=(
            "Evaluation distribution: arbitrary derived seeds ('custom') or the configured "
            "generalization protocol's train/test seed partition."
        ),
    )

    @field_validator("budgets")
    @classmethod
    def _validate_budgets(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("Benchmark budgets must all be positive integers.")
        if len(values) != len(set(values)):
            raise ValueError("Benchmark budgets must not contain duplicates.")
        return sorted(values)

    @field_validator("evaluation_seeds")
    @classmethod
    def _validate_evaluation_seeds(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("Evaluation seeds must be non-negative integers.")
        if len(values) != len(set(values)):
            raise ValueError("Evaluation seeds must not contain duplicates.")
        return values


class AdaptationBenchmarkConfig(BaseModel):
    """Frozen TEST-B cell declaration for Issue #265."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: str = "2.0"
    scenario: str = "TEST-B"
    shift_parameters: Dict[str, Any] = Field(
        default_factory=lambda: {"num_obstacles": 12, "wind_speed": 4.0, "gust_sigma": 0.6}
    )

    @model_validator(mode="after")
    def _validate_test_b(self) -> "AdaptationBenchmarkConfig":
        expected = {"num_obstacles": 12, "wind_speed": 4.0, "gust_sigma": 0.6}
        if self.protocol_version != "2.0":
            raise ValueError("Issue #265 requires protocol_version '2.0'")
        if self.scenario != "TEST-B" or self.shift_parameters != expected:
            raise ValueError(
                "Issue #265's primary cell is exactly TEST-B (12 obstacles, wind 4.0, gust 0.6)"
            )
        return self


class ExperimentConfig(BaseModel):
    """Top-level configuration schema for an AdaptiveRL experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field("drone_ppo", description="Unique experiment identifier")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    training: Optional[TrainingConfig] = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output_dir: Path = Field(
        default_factory=lambda: Path("artifacts"),
        description="Directory for saving models and evaluations",
    )
    log_dir: Path = Field(
        default_factory=lambda: Path("artifacts/logs"),
        description="Directory for logging and metrics",
    )
    benchmark: Optional[BenchmarkConfig] = Field(
        default=None,
        description="Optional benchmark settings for training-budget learning curves.",
    )
    adaptation_benchmark: Optional[AdaptationBenchmarkConfig] = Field(
        default=None,
        description="Frozen Issue #265 TEST-B online-adaptation cell.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_experiment_dict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "experiment" in data and isinstance(data["experiment"], dict):
                exp_dict = data.pop("experiment")
                if "name" in exp_dict and "name" not in data:
                    data["name"] = exp_dict["name"]
                if "seed" in exp_dict and "seed" not in data:
                    data["seed"] = exp_dict["seed"]
        return data


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate an AdaptiveRL experiment configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML file at {path}: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigError(
            f"Configuration file {path} must contain a YAML mapping/dictionary, got {type(raw_data).__name__}"
        )

    try:
        return ExperimentConfig.model_validate(raw_data)
    except ValidationError as exc:
        formatted_errors = []
        for err in exc.errors():
            loc = " -> ".join(str(p) for p in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            formatted_errors.append(f"  - [{loc}]: {msg}")
        errors_str = "\n".join(formatted_errors)
        raise ConfigError(f"Configuration validation failed for {path}:\n{errors_str}") from exc


def save_config(config: ExperimentConfig, target_path: str | Path) -> None:
    """Save an experiment configuration to a YAML file."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="python", exclude_none=True)
    data["output_dir"] = str(data["output_dir"])
    data["log_dir"] = str(data["log_dir"])

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def compute_config_sha256(config: ExperimentConfig) -> str:
    """Compute deterministic SHA-256 hash of the experiment configuration."""
    canonical_dict = config.model_dump(mode="json", exclude={"output_dir", "log_dir"})
    serialized = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
