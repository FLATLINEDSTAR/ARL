"""AdaptiveRL — Reinforcement Learning for Simulated 3D Drone Navigation.

AdaptiveRL trains, evaluates, and demonstrates an RL agent navigating
a simulated 3D drone through obstacles toward a target waypoint.

The root namespace intentionally stays small and free of optional RL
dependencies. Import benchmark and evaluation features from their own
subpackages, for example::

    from adaptive_rl.benchmarking import run_learning_curve_benchmark
    from adaptive_rl.evaluation import Evaluator
"""

from adaptive_rl.config import (
    AlgorithmConfig,
    BenchmarkConfig,
    ConfigError,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
    load_config,
    save_config,
)
from adaptive_rl.manifest import (
    ExperimentManifest,
    create_manifest,
    load_manifest,
    save_manifest,
)
from adaptive_rl.metrics import (
    DefaultOutcomePolicy,
    EpisodeMetrics,
    EpisodeMetricsAccumulator,
    OutcomePolicy,
    compute_rate,
    extract_episode_metrics,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AlgorithmConfig",
    "BenchmarkConfig",
    "ConfigError",
    "DefaultOutcomePolicy",
    "EnvironmentConfig",
    "EpisodeMetrics",
    "EpisodeMetricsAccumulator",
    "EvaluationConfig",
    "ExperimentConfig",
    "ExperimentManifest",
    "OutcomePolicy",
    "TrainingConfig",
    "compute_rate",
    "create_manifest",
    "extract_episode_metrics",
    "load_config",
    "load_manifest",
    "save_config",
    "save_manifest",
]
