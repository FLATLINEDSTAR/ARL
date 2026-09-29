"""Training engine implementation for AdaptiveRL."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import gymnasium as gym
import numpy as np
import torch

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.registry import get_algorithm_factory
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.config import ExperimentConfig
from adaptive_rl.environments.registry import make_env
from adaptive_rl.training.callbacks import (
    BaseCallback,
    CheckpointCallback,
    MetricLoggerCallback,
    SB3CallbackAdapter,
)
from adaptive_rl.training.checkpointing import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Structured summary and artifacts resulting from a training run.

    ``training_time_seconds`` is the monotonic (``time.perf_counter``) elapsed
    time of the ``algorithm.train(...)`` call only: it excludes model
    serialization, metadata writing, and any evaluation. The legacy
    ``duration_seconds`` value written into the training metadata JSON covers
    the whole ``fit()`` lifecycle and is a different quantity.
    """

    experiment_name: str
    total_timesteps: int
    episodes_completed: int
    mean_reward: float
    final_model_path: Path
    checkpoints: List[dict[str, Any]] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    success_rate: Optional[float] = None
    collision_rate: Optional[float] = None
    metadata_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    training_time_seconds: float = 0.0


class RLTrainer:
    """Trainer orchestrating reinforcement learning policy learning on the drone navigation environment."""

    def __init__(
        self,
        config: ExperimentConfig,
        env: Optional[gym.Env] = None,
        callbacks: Optional[List[BaseCallback]] = None,
    ) -> None:
        self.config = config
        self._set_deterministic_seed(self.config.seed)

        if env is not None:
            self.env = env
        else:
            self.env = make_env(self.config.environment.name, **self.config.environment.parameters)

        checkpoint_dir = self.config.output_dir / "checkpoints" / self.config.name
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        self.metric_logger = MetricLoggerCallback()
        self._callbacks: List[BaseCallback] = [self.metric_logger]

        if self.config.training and self.config.training.checkpoint_freq > 0:
            self._callbacks.append(
                CheckpointCallback(
                    checkpoint_manager=self.checkpoint_manager,
                    save_freq=self.config.training.checkpoint_freq,
                )
            )
        if callbacks:
            self._callbacks.extend(callbacks)

        algo_params = dict(self.config.algorithm.parameters)
        lr = algo_params.pop("learning_rate", self.config.algorithm.learning_rate)
        gamma = algo_params.pop("gamma", self.config.algorithm.gamma)
        batch_size = algo_params.pop("batch_size", self.config.algorithm.batch_size)
        seed = algo_params.pop("seed", self.config.seed)
        algo_name = self.config.algorithm.name.lower()
        self.algorithm: BaseAlgorithm
        if algo_name == "ppo":
            self.algorithm = PPOAlgorithm(
                env=self.env,
                learning_rate=lr,
                gamma=gamma,
                batch_size=batch_size,
                seed=seed,
                **algo_params,
            )
        elif algo_name == "sac":
            self.algorithm = SACAlgorithm(
                env=self.env,
                learning_rate=lr,
                gamma=gamma,
                batch_size=batch_size,
                seed=seed,
                **algo_params,
            )
        else:
            factory = get_algorithm_factory(algo_name)
            self.algorithm = factory(
                env=self.env,
                learning_rate=lr,
                gamma=gamma,
                batch_size=batch_size,
                seed=seed,
                **algo_params,
            )

    @staticmethod
    def _set_deterministic_seed(seed: int) -> None:
        """Enforce deterministic random seeds across libraries."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(self) -> TrainingResult:
        """Execute the training run and save final model and metadata."""
        started_at = time.time()
        import adaptive_rl

        adapter = SB3CallbackAdapter(
            callbacks=self._callbacks,
            algorithm=self.algorithm,
        )

        assert self.config.training is not None
        training_started_at = time.perf_counter()
        self.algorithm.train(
            total_timesteps=self.config.training.total_timesteps,
            callback=adapter,
        )
        training_time_seconds = time.perf_counter() - training_started_at
        if not math.isfinite(training_time_seconds) or training_time_seconds < 0:
            raise RuntimeError(f"Invalid training duration: {training_time_seconds}")

        # Save model artifact
        models_dir = self.config.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        final_model_path = models_dir / f"{self.config.name}_final.zip"
        self.algorithm.save(final_model_path)

        finished_at = time.time()
        duration = finished_at - started_at

        # Save training metadata JSON
        metadata_dir = self.config.output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / f"{self.config.name}_training.json"

        meta_dict = {
            "experiment_name": self.config.name,
            "algorithm": self.config.algorithm.name,
            "environment": self.config.environment.name,
            "seed": self.config.seed,
            "total_timesteps": self.config.training.total_timesteps,
            "episodes_completed": self.metric_logger.total_episodes,
            "mean_reward": float(self.metric_logger.mean_reward),
            "success_rate": self.metric_logger.success_rate,
            "collision_rate": self.metric_logger.collision_rate,
            "final_model_path": str(final_model_path),
            "duration_seconds": round(duration, 2),
            "training_time_seconds": training_time_seconds,
            "created_at": datetime.fromtimestamp(finished_at, tz=timezone.utc).isoformat(),
            "episode_rewards": [round(float(r), 2) for r in self.metric_logger.episode_rewards],
            "episode_lengths": [int(length) for length in self.metric_logger.episode_lengths],
            "version": adaptive_rl.__version__,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        # Generate and save experiment manifest (Issue #248)
        from adaptive_rl.manifest import create_manifest, save_manifest

        manifest = create_manifest(
            experiment_name=self.config.name,
            config_dict=self.config.model_dump(),
            started_at=started_at,
            finished_at=finished_at,
            training_time_seconds=training_time_seconds,
            artifacts={
                "model": final_model_path,
                "metadata": metadata_path,
            },
        )
        manifest_path = metadata_dir / f"{self.config.name}_manifest.json"
        save_manifest(manifest, manifest_path)

        return TrainingResult(
            experiment_name=self.config.name,
            total_timesteps=self.config.training.total_timesteps,
            episodes_completed=self.metric_logger.total_episodes,
            mean_reward=self.metric_logger.mean_reward,
            final_model_path=final_model_path,
            checkpoints=self.checkpoint_manager.list_checkpoints(),
            episode_rewards=list(self.metric_logger.episode_rewards),
            episode_lengths=list(self.metric_logger.episode_lengths),
            success_rate=self.metric_logger.success_rate,
            collision_rate=self.metric_logger.collision_rate,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            training_time_seconds=training_time_seconds,
        )

    def close(self) -> None:
        """Release trainer resources without hiding cleanup failures.

        Environments occasionally raise on close (for example when a render
        backend is already gone). Those failures are logged at ERROR level
        with context instead of being swallowed silently, and they do not
        replace an in-flight exception from :meth:`fit`.
        """
        env = getattr(self, "env", None)
        if env is None:
            return
        experiment_name = getattr(getattr(self, "config", None), "name", None)
        try:
            env.close()
        except Exception:
            logger.exception(
                "Failed to close training environment %s for experiment %r",
                type(env).__name__,
                experiment_name,
            )


PPOTrainer = RLTrainer


def get_trainer(
    config: ExperimentConfig,
    env: Optional[gym.Env] = None,
    callbacks: Optional[List[BaseCallback]] = None,
) -> RLTrainer:
    """Factory returning the trainer based on configuration."""
    return RLTrainer(config=config, env=env, callbacks=callbacks)


__all__ = [
    "PPOTrainer",
    "RLTrainer",
    "TrainingResult",
    "get_trainer",
]
