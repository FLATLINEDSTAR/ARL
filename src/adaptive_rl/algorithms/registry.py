"""Algorithm registry for AdaptiveRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

if TYPE_CHECKING:
    from adaptive_rl.algorithms.base import BaseAlgorithm


@dataclass
class AlgorithmMetadata:
    """Metadata record for a registered RL algorithm."""

    name: str
    description: str = ""
    action_space: str = "continuous"
    trainable: bool = True
    class_name: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class AlgorithmRegistryError(Exception):
    """Exception raised for algorithm registry operation failures."""

    pass


class AlgorithmRegistry:
    """Registry maintaining available algorithms, factories, and associated metadata."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._metadata: Dict[str, AlgorithmMetadata] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        metadata: Optional[AlgorithmMetadata] = None,
    ) -> None:
        if not name or not isinstance(name, str):
            raise AlgorithmRegistryError("Algorithm name must be a non-empty string.")

        clean = name.strip().lower()
        if clean in self._factories:
            raise AlgorithmRegistryError(f"Algorithm '{clean}' is already registered.")
        if not callable(factory):
            raise AlgorithmRegistryError(f"Factory for algorithm '{clean}' must be callable.")

        self._factories[clean] = factory
        if metadata is not None:
            self._metadata[clean] = metadata
        else:
            self._metadata[clean] = AlgorithmMetadata(
                name=clean, class_name=getattr(factory, "__name__", str(factory))
            )

    def get_factory(self, name: str) -> Callable[..., Any]:
        clean = name.strip().lower()
        if clean not in self._factories:
            available = ", ".join(sorted(self._factories.keys())) or "none"
            raise AlgorithmRegistryError(
                f"Unknown algorithm '{clean}'. Available registered algorithms: {available}"
            )
        return self._factories[clean]

    def get_metadata(self, name: str) -> AlgorithmMetadata:
        clean = name.strip().lower()
        if clean not in self._metadata:
            available = ", ".join(sorted(self._metadata.keys())) or "none"
            raise AlgorithmRegistryError(
                f"Unknown algorithm '{clean}'. Available registered algorithms: {available}"
            )
        return self._metadata[clean]

    def list_algorithms(self) -> List[str]:
        return sorted(self._factories.keys())

    def list_all_metadata(self) -> Dict[str, AlgorithmMetadata]:
        return {name: self._metadata[name] for name in sorted(self._metadata.keys())}

    def is_trainable(self, name: str) -> bool:
        return True

    def clear(self) -> None:
        self._factories.clear()
        self._metadata.clear()


algorithm_registry = AlgorithmRegistry()


def _register_defaults() -> None:
    from adaptive_rl.algorithms.ppo import PPOAlgorithm
    from adaptive_rl.algorithms.sac import SACAlgorithm

    if "ppo" not in algorithm_registry.list_algorithms():
        algorithm_registry.register(
            "ppo",
            PPOAlgorithm,
            AlgorithmMetadata(
                name="ppo",
                description="Proximal Policy Optimization (PPO) backed by Stable-Baselines3.",
                action_space="continuous",
                trainable=True,
                class_name="PPOAlgorithm",
                hyperparameters={
                    "learning_rate": 3e-4,
                    "n_steps": 1024,
                    "batch_size": 64,
                    "gamma": 0.99,
                },
                tags=["on-policy", "actor-critic", "sb3", "continuous"],
            ),
        )

    if "sac" not in algorithm_registry.list_algorithms():
        algorithm_registry.register(
            "sac",
            SACAlgorithm,
            AlgorithmMetadata(
                name="sac",
                description="Soft Actor-Critic (SAC) backed by Stable-Baselines3.",
                action_space="continuous",
                trainable=True,
                class_name="SACAlgorithm",
                hyperparameters={
                    "learning_rate": 3e-4,
                    "buffer_size": 100000,
                    "batch_size": 256,
                    "gamma": 0.99,
                    "tau": 0.005,
                },
                tags=["off-policy", "actor-critic", "sb3", "continuous"],
            ),
        )


_register_defaults()


def register_algorithm(
    name: str,
    factory: Callable[..., Any],
    metadata: Optional[AlgorithmMetadata] = None,
) -> None:
    algorithm_registry.register(name, factory, metadata)


def get_algorithm_factory(name: str) -> Callable[..., Any]:
    return algorithm_registry.get_factory(name)


def get_algorithm_metadata(name: str) -> AlgorithmMetadata:
    return algorithm_registry.get_metadata(name)


def list_algorithms() -> List[str]:
    return algorithm_registry.list_algorithms()


def list_all_algorithm_metadata() -> Dict[str, AlgorithmMetadata]:
    return algorithm_registry.list_all_metadata()


def load_algorithm_from_pretrained(
    path: str | Path,
    env: Optional[Any] = None,
    algorithm_name: Optional[str] = None,
) -> BaseAlgorithm:
    """Load a trained algorithm instance from a saved model file.

    Inspects algorithm_name or model archive metadata to instantiate either
    PPOAlgorithm or SACAlgorithm (or registered custom algorithm).

    Args:
        path: Path to model checkpoint file (.zip).
        env: Optional Gymnasium environment to bind to the model.
        algorithm_name: Optional algorithm name hint ('ppo', 'sac').

    Returns:
        Loaded BaseAlgorithm instance.
    """
    model_path = Path(path)
    if not model_path.exists() and model_path.with_suffix(".zip").exists():
        model_path = model_path.with_suffix(".zip")

    detected = algorithm_name.strip().lower() if algorithm_name else None
    if detected is None and model_path.exists():
        try:
            import zipfile

            with zipfile.ZipFile(model_path, "r") as z:
                if "data" in z.namelist():
                    data_str = z.read("data").decode("utf-8", errors="ignore")
                    if '"SAC"' in data_str or "SACAlgorithm" in data_str:
                        detected = "sac"
                    elif '"PPO"' in data_str or "PPOAlgorithm" in data_str:
                        detected = "ppo"
        except Exception:
            pass

    if detected == "sac":
        from adaptive_rl.algorithms.sac import SACAlgorithm

        return SACAlgorithm.from_pretrained(model_path, env=env)
    elif detected == "ppo":
        from adaptive_rl.algorithms.ppo import PPOAlgorithm

        return PPOAlgorithm.from_pretrained(model_path, env=env)
    elif detected is not None and detected in algorithm_registry.list_algorithms():
        factory = algorithm_registry.get_factory(detected)
        if hasattr(factory, "from_pretrained"):
            return cast("BaseAlgorithm", factory.from_pretrained(model_path, env=env))
        return cast("BaseAlgorithm", factory(env=env))

    # Fallback heuristic: try PPO first, then SAC
    try:
        from adaptive_rl.algorithms.ppo import PPOAlgorithm

        return PPOAlgorithm.from_pretrained(model_path, env=env)
    except Exception:
        from adaptive_rl.algorithms.sac import SACAlgorithm

        return SACAlgorithm.from_pretrained(model_path, env=env)


__all__ = [
    "AlgorithmMetadata",
    "AlgorithmRegistry",
    "AlgorithmRegistryError",
    "algorithm_registry",
    "get_algorithm_factory",
    "get_algorithm_metadata",
    "list_algorithms",
    "list_all_algorithm_metadata",
    "load_algorithm_from_pretrained",
    "register_algorithm",
]
