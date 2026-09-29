"""Comparative algorithm benchmark framework for AdaptiveRL.

Evaluates reinforcement learning algorithms (e.g. PPO and SAC) under strictly identical
experimental conditions:
- Identical environment geometry and obstacles
- Controlled training timesteps budget
- Identical held-out evaluation seeds and episodes
- Standardized performance, safety, and trajectory metrics
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from adaptive_rl.algorithms.registry import get_algorithm_metadata
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.training.trainer import get_trainer

DEFAULT_ENV_PARAMS: Dict[str, Any] = {
    "bounds": [30.0, 30.0, 15.0],
    "start_pos": [5.0, 5.0, 5.0],
    "goal_pos": [25.0, 25.0, 10.0],
    "num_obstacles": 4,
    "obstacle_radius": 2.0,
    "target_radius": 1.5,
    "collision_radius": 0.8,
    "progress_weight": 2.0,
}


def get_default_algorithm_config(algo_name: str) -> AlgorithmConfig:
    """Return default algorithm configuration for benchmarking."""
    clean = algo_name.strip().lower()
    if clean == "ppo":
        return AlgorithmConfig(
            name="ppo",
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=64,
            parameters={
                "n_steps": 1024,
                "n_epochs": 10,
                "clip_range": 0.2,
                "ent_coef": 0.0,
            },
        )
    elif clean == "sac":
        return AlgorithmConfig(
            name="sac",
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=256,
            parameters={
                "buffer_size": 100000,
                "tau": 0.005,
                "learning_starts": 100,
                "ent_coef": "auto",
            },
        )
    else:
        meta = get_algorithm_metadata(clean)
        return AlgorithmConfig(
            name=clean,
            parameters=dict(meta.hyperparameters),
        )


def export_comparison_json(data: Dict[str, Any], path: str | Path) -> Path:
    """Serialize algorithm comparison benchmark data to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return target


def export_comparison_csv(results: List[Dict[str, Any]], path: str | Path) -> Path:
    """Export algorithm comparison results to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        target.touch()
        return target

    fieldnames = list(results[0].keys())
    with open(target, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return target


def run_algorithm_comparison(
    algorithms: Sequence[str] = ("ppo", "sac"),
    timesteps: int = 25000,
    eval_episodes: int = 20,
    seed: int = 42,
    env_name: str = "drone",
    env_parameters: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path | str] = None,
    output_json: Optional[Path | str] = Path("artifacts/algorithm_comparison.json"),
    output_csv: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Execute head-to-head training and evaluation comparison between RL algorithms.

    Args:
        algorithms: Sequence of registered algorithm names to benchmark (e.g. ('ppo', 'sac')).
        timesteps: Number of training timesteps per algorithm.
        eval_episodes: Number of evaluation episodes on held-out seeds.
        seed: Base random seed for training and evaluation.
        env_name: Name of registered environment.
        env_parameters: Optional environment parameters (defaults to 3D drone navigation).
        output_dir: Directory for storing training checkpoints and run artifacts.
        output_json: Optional path to save structured JSON report.
        output_csv: Optional path to save CSV comparison summary.

    Returns:
        Structured dictionary containing full comparison results and per-algorithm metrics.
    """
    base_out = Path(output_dir) if output_dir is not None else Path("artifacts/benchmarks")
    env_params = dict(env_parameters if env_parameters is not None else DEFAULT_ENV_PARAMS)
    eval_seed_base = seed + 10000

    results: List[Dict[str, Any]] = []

    for algo_name in algorithms:
        clean_algo = algo_name.strip().lower()
        algo_cfg = get_default_algorithm_config(clean_algo)

        exp_name = f"benchmark_{clean_algo}"
        exp_cfg = ExperimentConfig(
            name=exp_name,
            seed=seed,
            algorithm=algo_cfg,
            environment=EnvironmentConfig(
                name=env_name,
                parameters=env_params,
            ),
            training=TrainingConfig(
                total_timesteps=timesteps,
                checkpoint_freq=0,
            ),
            evaluation=EvaluationConfig(
                eval_episodes=eval_episodes,
                deterministic=True,
            ),
            output_dir=base_out / clean_algo,
        )

        train_start = time.time()
        trainer = get_trainer(config=exp_cfg)
        trainer.fit()
        train_duration = time.time() - train_start

        # Evaluate on identical evaluation seeds
        eval_env = make_env(env_name, **env_params)
        evaluator = Evaluator(algorithm=trainer.algorithm, env=eval_env)
        try:
            metrics = evaluator.evaluate(
                num_episodes=eval_episodes,
                deterministic=True,
                base_seed=eval_seed_base,
            )
        finally:
            eval_env.close()
            trainer.close()

        res_record: Dict[str, Any] = {
            "algorithm": clean_algo.upper(),
            "timesteps_trained": timesteps,
            "training_duration_seconds": round(train_duration, 2),
            "eval_episodes": metrics.episodes,
            "success_rate": round(metrics.success_rate, 4)
            if metrics.success_rate is not None
            else 0.0,
            "collision_rate": round(metrics.collision_rate, 4)
            if metrics.collision_rate is not None
            else 0.0,
            "obstacle_collision_rate": (
                round(metrics.obstacle_collision_rate, 4)
                if metrics.obstacle_collision_rate is not None
                else 0.0
            ),
            "boundary_collision_rate": (
                round(metrics.boundary_collision_rate, 4)
                if metrics.boundary_collision_rate is not None
                else 0.0
            ),
            "timeout_rate": round(metrics.timeout_rate, 4)
            if metrics.timeout_rate is not None
            else 0.0,
            "mean_reward": round(metrics.mean_reward, 2),
            "std_reward": round(metrics.std_reward, 2),
            "min_reward": round(metrics.min_reward, 2),
            "max_reward": round(metrics.max_reward, 2),
            "mean_episode_length": round(metrics.mean_episode_length, 2),
            "std_episode_length": round(metrics.std_episode_length, 2),
            "mean_path_length": (
                round(metrics.mean_path_length, 2) if metrics.mean_path_length is not None else None
            ),
            "mean_path_efficiency": (
                round(metrics.mean_path_efficiency, 4)
                if metrics.mean_path_efficiency is not None
                else None
            ),
            "mean_min_obstacle_clearance": (
                round(metrics.mean_min_obstacle_clearance, 2)
                if metrics.mean_min_obstacle_clearance is not None
                else None
            ),
            "mean_max_velocity": (
                round(metrics.mean_max_velocity, 2)
                if metrics.mean_max_velocity is not None
                else None
            ),
            "mean_max_acceleration": (
                round(metrics.mean_max_acceleration, 2)
                if metrics.mean_max_acceleration is not None
                else None
            ),
        }
        results.append(res_record)

    benchmark_data: Dict[str, Any] = {
        "benchmark": "algorithm_comparison",
        "algorithms": [a.upper() for a in algorithms],
        "timesteps": timesteps,
        "eval_episodes": eval_episodes,
        "seed": seed,
        "evaluation_base_seed": eval_seed_base,
        "environment": {
            "name": env_name,
            "parameters": env_params,
        },
        "results": results,
    }

    if output_json is not None:
        export_comparison_json(benchmark_data, output_json)
    if output_csv is not None:
        export_comparison_csv(results, output_csv)

    return benchmark_data
