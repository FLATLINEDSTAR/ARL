"""Reward-function ablation benchmark framework for AdaptiveRL.

Implements controlled scientific ablations of the drone navigation reward
function across four exact variants:
- Variant A: Progress Only
- Variant B: Progress + Collision
- Variant C: Progress + Collision + Step
- Variant D: Full Baseline

Maintains strict experimental controls:
- Identical training timestep budget
- Identical PPO hyperparameters
- Identical initial network initialization and environment seeds
- Identical held-out evaluation seeds and episodes
- Identical environment geometry
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
    load_config,
)
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.training.callbacks import BaseCallback
from adaptive_rl.training.trainer import PPOTrainer


@dataclass(frozen=True)
class RewardAblationVariant:
    """Explicit parameters and metadata for a reward-function ablation variant."""

    id: str
    name: str
    description: str
    progress_weight: float
    goal_reward: float
    step_penalty: float
    action_penalty_weight: float
    collision_reward: float

    def get_reward_parameters(self) -> Dict[str, float]:
        """Return environment keyword arguments corresponding to this variant."""
        return {
            "progress_weight": self.progress_weight,
            "goal_reward": self.goal_reward,
            "step_penalty": self.step_penalty,
            "action_penalty_weight": self.action_penalty_weight,
            "collision_reward": self.collision_reward,
        }


REWARD_ABLATION_VARIANTS: Dict[str, RewardAblationVariant] = {
    "A": RewardAblationVariant(
        id="A",
        name="Variant A (Progress Only)",
        description="Pure progress delta toward target without step, effort, or collision penalties.",
        progress_weight=2.0,
        goal_reward=100.0,
        step_penalty=0.0,
        action_penalty_weight=0.0,
        collision_reward=0.0,
    ),
    "B": RewardAblationVariant(
        id="B",
        name="Variant B (Progress + Collision)",
        description="Progress reward with terminal collision penalty (-100.0), no step or effort penalty.",
        progress_weight=2.0,
        goal_reward=100.0,
        step_penalty=0.0,
        action_penalty_weight=0.0,
        collision_reward=-100.0,
    ),
    "C": RewardAblationVariant(
        id="C",
        name="Variant C (Progress + Collision + Step)",
        description="Progress, collision (-100.0), and step time penalty (-0.05), without effort penalty.",
        progress_weight=2.0,
        goal_reward=100.0,
        step_penalty=-0.05,
        action_penalty_weight=0.0,
        collision_reward=-100.0,
    ),
    "D": RewardAblationVariant(
        id="D",
        name="Variant D (Full Baseline)",
        description="Full standard reward matching default environment (progress, collision, step, effort).",
        progress_weight=2.0,
        goal_reward=100.0,
        step_penalty=-0.05,
        action_penalty_weight=0.01,
        collision_reward=-100.0,
    ),
}


def get_ablation_variant(identifier: str) -> RewardAblationVariant:
    """Retrieve an ablation variant by key or alias (e.g. 'A', 'Variant A', 'Variant A (Progress Only)')."""
    clean = identifier.strip().upper()
    if clean in REWARD_ABLATION_VARIANTS:
        return REWARD_ABLATION_VARIANTS[clean]

    for key, var in REWARD_ABLATION_VARIANTS.items():
        if clean == var.name.upper() or clean == f"VARIANT {key}":
            return var
        if clean == key:
            return var

    available = ", ".join(REWARD_ABLATION_VARIANTS.keys())
    raise ValueError(f"Unknown ablation variant '{identifier}'. Available: {available}")


class ConvergenceEvaluationCallback(BaseCallback):
    """Callback evaluating policy on held-out test seeds to track convergence speed.

    Convergence speed is defined as the first training timestep at which the
    evaluation success rate strictly exceeds the threshold (> 0.70).
    If the threshold is never exceeded during training, convergence_speed is None.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int = 5000,
        eval_episodes: int = 10,
        eval_seed: int = 1042,
        success_threshold: float = 0.70,
        model: Optional[BaseAlgorithm] = None,
    ) -> None:
        self.eval_env = eval_env
        self.eval_freq = max(1, eval_freq)
        self.eval_episodes = max(1, eval_episodes)
        self.eval_seed = eval_seed
        self.success_threshold = float(success_threshold)
        self.model = model

        self.last_eval_step: int = 0
        self.convergence_step: Optional[int] = None
        self.history: List[Dict[str, Any]] = []

    def set_model(self, model: BaseAlgorithm) -> None:
        """Assign model reference for evaluation."""
        self.model = model

    def on_step(self, step: int, locals_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Evaluate when step interval is crossed."""
        if self.model is None and locals_dict and "self" in locals_dict:
            self.model = locals_dict["self"]

        if self.eval_freq > 0 and (step - self.last_eval_step) >= self.eval_freq:
            self._evaluate(step)
        return True

    def on_training_end(self) -> None:
        """Ensure at least one evaluation was performed."""
        if self.last_eval_step == 0:
            self._evaluate(self.last_eval_step)

    def _evaluate(self, step: int) -> None:
        self.last_eval_step = step
        if self.model is None:
            return

        # Preserve global random state during intermediate rollout evaluations
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()

        try:
            evaluator = Evaluator(algorithm=self.model, env=self.eval_env)
            metrics = evaluator.evaluate(
                num_episodes=self.eval_episodes,
                deterministic=True,
                base_seed=self.eval_seed,
            )
            succ = metrics.success_rate if metrics.success_rate is not None else 0.0
            self.history.append(
                {
                    "step": step,
                    "success_rate": succ,
                    "mean_reward": metrics.mean_reward,
                }
            )
            if succ > self.success_threshold and self.convergence_step is None:
                self.convergence_step = step
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)

    def close(self) -> None:
        """Close evaluation environment."""
        if hasattr(self.eval_env, "close"):
            try:
                self.eval_env.close()
            except Exception:
                pass


def export_ablation_csv(results: Sequence[Dict[str, Any]], output_path: str | Path) -> Path:
    """Export benchmark ablation results to CSV."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "variant",
        "seed",
        "training_timesteps",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_reward",
        "mean_path_efficiency",
        "convergence_speed",
    ]

    with open(target, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "variant": r["variant"],
                "seed": r["seed"],
                "training_timesteps": r["training_timesteps"],
                "success_rate": r["success_rate"] if r["success_rate"] is not None else "",
                "collision_rate": r["collision_rate"] if r["collision_rate"] is not None else "",
                "timeout_rate": r["timeout_rate"] if r["timeout_rate"] is not None else "",
                "mean_reward": round(float(r["mean_reward"]), 4)
                if r.get("mean_reward") is not None
                else "",
                "mean_path_efficiency": round(float(r["mean_path_efficiency"]), 4)
                if r.get("mean_path_efficiency") is not None
                else "",
                "convergence_speed": r["convergence_speed"]
                if r.get("convergence_speed") is not None
                else "",
            }
            writer.writerow(row)

    return target


def export_ablation_json(data: Dict[str, Any], output_path: str | Path) -> Path:
    """Export detailed benchmark ablation data to JSON."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return target


def run_reward_ablation_experiment(
    timesteps: int = 25000,
    eval_episodes: int = 20,
    seed: int = 42,
    eval_freq: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
    output_json: Optional[str | Path] = None,
    output_csv: Optional[str | Path] = None,
    variants: Optional[Sequence[str]] = None,
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute controlled reward-function ablation experiment across variants.

    Args:
        timesteps: Total training environment steps per variant.
        eval_episodes: Number of held-out evaluation episodes per variant.
        seed: Deterministic base seed for initialization and environments.
        eval_freq: Intermediate evaluation frequency for convergence tracking.
        output_dir: Base directory for benchmark artifacts.
        output_json: Optional custom path for JSON report.
        output_csv: Optional custom path for CSV report.
        variants: Optional subset of variant keys (default: ['A', 'B', 'C', 'D']).
        config_path: Optional base experiment config path.

    Returns:
        Dictionary containing experiment metadata and per-variant results.
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    if eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be positive, got {eval_episodes}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    variant_keys = list(variants) if variants is not None else ["A", "B", "C", "D"]
    resolved_variants = [get_ablation_variant(k) for k in variant_keys]

    base_output_dir = Path(output_dir) if output_dir is not None else Path("artifacts/benchmarks")
    json_path = (
        Path(output_json) if output_json is not None else base_output_dir / "reward_ablation.json"
    )
    csv_path = (
        Path(output_csv) if output_csv is not None else base_output_dir / "reward_ablation.csv"
    )

    # Load baseline experiment configuration to inherit algorithm and geometry hyperparameters
    base_cfg: Optional[ExperimentConfig] = None
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            base_cfg = load_config(p)
    elif Path("configs/drone_ppo_demo.yaml").exists():
        base_cfg = load_config("configs/drone_ppo_demo.yaml")
    elif Path("configs/drone_ppo.yaml").exists():
        base_cfg = load_config("configs/drone_ppo.yaml")

    if base_cfg is not None:
        base_algo_name = base_cfg.algorithm.name
        base_lr = base_cfg.algorithm.learning_rate
        base_gamma = base_cfg.algorithm.gamma
        base_batch_size = base_cfg.algorithm.batch_size
        base_algo_params = dict(base_cfg.algorithm.parameters)
        base_env_params = dict(base_cfg.environment.parameters)
        base_max_steps = base_cfg.environment.max_steps
    else:
        base_algo_name = "ppo"
        base_lr = 0.0003
        base_gamma = 0.99
        base_batch_size = 64
        base_algo_params = {"n_steps": 1024, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.0}
        base_env_params = {
            "bounds": [30.0, 30.0, 15.0],
            "start_pos": [5.0, 5.0, 5.0],
            "goal_pos": [25.0, 25.0, 10.0],
            "num_obstacles": 4,
            "obstacle_radius": 2.0,
            "target_radius": 1.5,
            "collision_radius": 0.8,
        }
        base_max_steps = 200

    # Ensure PPO rollouts fit within requested budget
    n_steps = base_algo_params.get("n_steps", 1024)
    if timesteps < n_steps:
        n_steps = max(32, timesteps)
        base_algo_params["n_steps"] = n_steps
        base_batch_size = min(base_batch_size, n_steps)

    # Calculate sensible evaluation frequency if not provided
    resolved_eval_freq = eval_freq
    if resolved_eval_freq is None:
        if timesteps >= 25000:
            resolved_eval_freq = 5000
        else:
            resolved_eval_freq = max(100, timesteps // 5)

    # This legacy reward-ablation utility is exploratory and outside the
    # preregistered Issue #271 adaptation research path.
    eval_seed = seed + 1000
    results: List[Dict[str, Any]] = []

    for variant in resolved_variants:
        # Enforce identical initial network weights and environment sequence
        PPOTrainer._set_deterministic_seed(seed)

        # Merge environment geometry with explicit variant reward weights
        env_params = dict(base_env_params)
        env_params.update(variant.get_reward_parameters())

        variant_config = ExperimentConfig(
            name=f"ablation_{variant.id.lower()}",
            seed=seed,
            algorithm=AlgorithmConfig(
                name=base_algo_name,
                learning_rate=base_lr,
                gamma=base_gamma,
                batch_size=base_batch_size,
                parameters=base_algo_params,
            ),
            environment=EnvironmentConfig(
                name="drone",
                max_steps=base_max_steps,
                parameters=env_params,
            ),
            training=TrainingConfig(
                total_timesteps=timesteps,
                checkpoint_freq=0,
                log_interval=10,
            ),
            evaluation=EvaluationConfig(
                eval_episodes=eval_episodes,
                deterministic=True,
            ),
            output_dir=base_output_dir / "runs",
            log_dir=base_output_dir / "runs" / "logs",
        )

        # Standard held-out evaluation environment (identical geometry across all variants)
        eval_env = DroneNavigation3DEnv(
            bounds=tuple(base_env_params.get("bounds", [30.0, 30.0, 15.0])),  # type: ignore[arg-type]
            start_pos=base_env_params.get("start_pos"),
            goal_pos=base_env_params.get("goal_pos"),
            num_obstacles=base_env_params.get("num_obstacles", 4),
            obstacle_radius=base_env_params.get("obstacle_radius", 2.0),
            target_radius=base_env_params.get("target_radius", 1.5),
            collision_radius=base_env_params.get("collision_radius", 0.8),
            max_steps=base_max_steps,
        )

        convergence_cb = ConvergenceEvaluationCallback(
            eval_env=eval_env,
            eval_freq=resolved_eval_freq,
            eval_episodes=min(10, eval_episodes),
            eval_seed=eval_seed,
            success_threshold=0.70,
        )

        trainer = PPOTrainer(config=variant_config, callbacks=[convergence_cb])
        convergence_cb.set_model(trainer.algorithm)

        try:
            trainer.fit()
        finally:
            trainer.close()

        # Final held-out evaluation across all evaluation episodes using identical seed
        final_evaluator = Evaluator(algorithm=trainer.algorithm, env=eval_env)
        final_metrics = final_evaluator.evaluate(
            num_episodes=eval_episodes,
            deterministic=True,
            base_seed=eval_seed,
        )

        eval_env.close()

        record: Dict[str, Any] = {
            "variant": variant.name,
            "variant_id": variant.id,
            "seed": seed,
            "training_timesteps": timesteps,
            "success_rate": round(float(final_metrics.success_rate or 0.0), 4)
            if final_metrics.success_rate is not None
            else None,
            "collision_rate": round(float(final_metrics.collision_rate or 0.0), 4)
            if final_metrics.collision_rate is not None
            else None,
            "timeout_rate": round(float(final_metrics.timeout_rate or 0.0), 4)
            if final_metrics.timeout_rate is not None
            else None,
            "mean_reward": round(float(final_metrics.mean_reward), 2),
            "std_reward": round(float(final_metrics.std_reward), 2),
            "mean_episode_length": round(float(final_metrics.mean_episode_length), 2),
            "mean_path_efficiency": round(float(final_metrics.mean_path_efficiency or 0.0), 4)
            if final_metrics.mean_path_efficiency is not None
            else None,
            "convergence_speed": convergence_cb.convergence_step,
            "reward_weights": variant.get_reward_parameters(),
        }
        results.append(record)

    benchmark_summary: Dict[str, Any] = {
        "experiment": "reward_function_ablation",
        "base_seed": seed,
        "eval_seed": eval_seed,
        "training_timesteps": timesteps,
        "eval_episodes": eval_episodes,
        "convergence_threshold": 0.70,
        "eval_freq": resolved_eval_freq,
        "environment": {
            "name": "drone",
            "bounds": base_env_params.get("bounds", [30.0, 30.0, 15.0]),
            "num_obstacles": base_env_params.get("num_obstacles", 4),
            "max_steps": base_max_steps,
        },
        "algorithm": {
            "name": base_algo_name,
            "learning_rate": base_lr,
            "gamma": base_gamma,
            "batch_size": base_batch_size,
            "n_steps": n_steps,
        },
        "results": results,
    }

    export_ablation_json(benchmark_summary, json_path)
    export_ablation_csv(results, csv_path)

    return benchmark_summary


__all__ = [
    "REWARD_ABLATION_VARIANTS",
    "ConvergenceEvaluationCallback",
    "RewardAblationVariant",
    "export_ablation_csv",
    "export_ablation_json",
    "get_ablation_variant",
    "run_reward_ablation_experiment",
]
