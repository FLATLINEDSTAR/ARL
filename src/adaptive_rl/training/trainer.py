"""Training engine implementation for AdaptiveRL."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import gymnasium as gym
import numpy as np
import torch

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.config import ExperimentConfig
from adaptive_rl.environments.registry import make_env
from adaptive_rl.training.callbacks import (
    BaseCallback,
    CheckpointCallback,
    MetricLoggerCallback,
    SB3CallbackAdapter,
)
from adaptive_rl.training.checkpointing import CheckpointManager


@dataclass
class TrainingResult:
    """Structured summary and artifacts resulting from a training run."""

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


class PPOTrainer:
    """Trainer orchestrating PPO policy learning on the drone navigation environment."""

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
        self.algorithm = PPOAlgorithm(
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
        self.algorithm.train(
            total_timesteps=self.config.training.total_timesteps,
            callback=adapter,
        )

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
            "episode_rewards": [round(float(r), 2) for r in self.metric_logger.episode_rewards],
            "episode_lengths": [int(length) for length in self.metric_logger.episode_lengths],
            "version": adaptive_rl.__version__,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

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
        )

    def close(self) -> None:
        """Clean up trainer resources."""
        if hasattr(self, "env") and self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass


def get_trainer(
    config: ExperimentConfig,
    env: Optional[gym.Env] = None,
    callbacks: Optional[List[BaseCallback]] = None,
) -> PPOTrainer:
    """Factory returning the trainer based on configuration."""
    return PPOTrainer(config=config, env=env, callbacks=callbacks)
