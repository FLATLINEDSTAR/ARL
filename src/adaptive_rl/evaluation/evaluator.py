"""Evaluation engine for AdaptiveRL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.algorithms.random_policy import RandomPolicy
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.metrics import EvaluationMetrics


@dataclass(frozen=True)
class EpisodeEvaluationRecord:
    """Record for a single evaluation episode."""

    episode_index: int
    seed: Optional[int]
    return_value: float
    episode_length: int
    success: bool
    collision: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "seed": self.seed,
            "return": self.return_value,
            "episode_length": self.episode_length,
            "success": self.success,
            "collision": self.collision,
        }


class Evaluator:
    """Standardized multi-episode evaluation engine for drone navigation."""

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
    ) -> EvaluationMetrics:
        """Execute evaluation rollouts and compute aggregated metrics."""
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")

        self.last_episode_records = []
        rewards: List[float] = []
        lengths: List[int] = []
        successes: List[bool] = []
        collisions: List[bool] = []

        for ep in range(num_episodes):
            seed = (base_seed + ep) if base_seed is not None else None
            obs, info = self.env.reset(seed=seed)
            ep_reward = 0.0
            ep_length = 0
            done = False
            last_info = dict(info or {})

            while not done:
                action, _ = self.algorithm.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                last_info = step_info
                done = terminated or truncated

            is_success = bool(last_info.get("success", False))
            is_collision = bool(last_info.get("collision", False))

            rewards.append(ep_reward)
            lengths.append(ep_length)
            successes.append(is_success)
            collisions.append(is_collision)

            self.last_episode_records.append(
                EpisodeEvaluationRecord(
                    episode_index=ep,
                    seed=seed,
                    return_value=ep_reward,
                    episode_length=ep_length,
                    success=is_success,
                    collision=is_collision,
                )
            )

        mean_rew = float(np.mean(rewards))
        std_rew = float(np.std(rewards))
        min_rew = float(np.min(rewards))
        max_rew = float(np.max(rewards))
        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))

        succ_rate = float(sum(successes) / num_episodes)
        coll_rate = float(sum(collisions) / num_episodes)

        return EvaluationMetrics(
            episodes=num_episodes,
            mean_reward=mean_rew,
            std_reward=std_rew,
            min_reward=min_rew,
            max_reward=max_rew,
            success_rate=succ_rate,
            collision_rate=coll_rate,
            mean_episode_length=mean_len,
            std_episode_length=std_len,
            additional_metrics={
                "all_rewards": rewards,
                "all_lengths": lengths,
                "deterministic": deterministic,
                "base_seed": base_seed,
            },
        )

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
            "mean_episode_length": round(metrics.mean_episode_length, 2),
            "std_episode_length": round(metrics.std_episode_length, 2),
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

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
    ppo_algorithm: BaseAlgorithm,
    random_policy: Optional[BaseAlgorithm] = None,
    env: Optional[gym.Env] = None,
    num_episodes: int = 20,
    base_seed: Optional[int] = 42,
) -> Dict[str, EvaluationMetrics]:
    """Execute head-to-head evaluation between trained PPO and Random baseline under identical seeds."""
    close_env = False
    if env is None:
        env = DroneNavigation3DEnv()
        close_env = True
    try:
        if random_policy is None:
            random_policy = RandomPolicy(action_space=env.action_space, seed=base_seed)

        ppo_eval = Evaluator(algorithm=ppo_algorithm, env=env)
        ppo_metrics = ppo_eval.evaluate(
            num_episodes=num_episodes,
            deterministic=True,
            base_seed=base_seed,
        )

        rand_eval = Evaluator(algorithm=random_policy, env=env)
        rand_metrics = rand_eval.evaluate(
            num_episodes=num_episodes,
            deterministic=False,
            base_seed=base_seed,
        )

        return {
            "PPO": ppo_metrics,
            "Random Policy": rand_metrics,
        }
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


__all__ = [
    "EpisodeEvaluationRecord",
    "Evaluator",
    "compare_policies",
    "evaluate_ppo_policy",
    "evaluate_random_policy",
    "run_obstacle_density_experiment",
]
