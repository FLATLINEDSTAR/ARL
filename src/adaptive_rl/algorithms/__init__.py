"""Algorithm abstraction layer for AdaptiveRL."""

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.random_policy import RandomPolicy
from adaptive_rl.algorithms.registry import (
    AlgorithmMetadata,
    AlgorithmRegistry,
    AlgorithmRegistryError,
    algorithm_registry,
    get_algorithm_factory,
    get_algorithm_metadata,
    list_algorithms,
    list_all_algorithm_metadata,
    load_algorithm_from_pretrained,
    register_algorithm,
)
from adaptive_rl.algorithms.sac import SACAlgorithm

__all__ = [
    "AlgorithmMetadata",
    "AlgorithmRegistry",
    "AlgorithmRegistryError",
    "BaseAlgorithm",
    "PPOAlgorithm",
    "RandomPolicy",
    "SACAlgorithm",
    "algorithm_registry",
    "get_algorithm_factory",
    "get_algorithm_metadata",
    "list_algorithms",
    "list_all_algorithm_metadata",
    "load_algorithm_from_pretrained",
    "register_algorithm",
]
