"""Preregistered AdaptiveRL Experiment Runner: Online Adaptation vs Fixed Policy.

Implements the complete research workflow of docs/research/adaptive_rl_hypothesis.md
and docs/research/TREATMENT_CARD.md (Protocol Version 2.0 / Issue #265):

    NOMINAL TRAINING
          |
          v
    TRAIN POLICY ON NOMINAL ENV
          |
          v
     FREEZE CHECKPOINT
          |
          v
   PRE-SHIFT EVALUATION  (K_pre = 15 episodes, shared)
          |
          v
   DISTRIBUTION SHIFT    (TEST-B: moderate compound shift)
          |
          v
  SHARED SHOCK WINDOW    (episodes 1-5, shared single execution)
          |
          v
        FORK
    +-----------+
    |           |
    v           v
  FIXED      ADAPTIVE
    |           |
    |       ONLINE UPDATE (Blocks B5..B14 strictly between episodes)
    |           |
    |       POST-SHIFT EVALUATION (Episodes 6-15, identical seeds)
    |           |
    +-----+-----+
          |
          v
   RECOVERY ANALYSIS     (adaptive_rl.protocol.recovery.compute_recovery)
          |
          v
  STATISTICAL REPORT     (paired t-test, Cohen's d, bootstrap CI, manifest)
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.config import ExperimentConfig
from adaptive_rl.environments.disturbed_drone import DroneDisturbed3DEnv
from adaptive_rl.manifest import create_manifest
from adaptive_rl.protocol.constants import (
    ALPHA,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    K_PRE,
    MIN_VALID_N,
    N_POST,
    N_UPDATE,
    PROTOCOL_VERSION,
    TRAINING_SEEDS,
    WINDOW,
)
from adaptive_rl.protocol.recovery import (
    compute_recovery,
)
from adaptive_rl.protocol.seeds import (
    derive_seed,
)
from adaptive_rl.protocol.statistics import (
    bootstrap_percentile_ci,
    cohen_dz,
    exact_sign_test,
    exact_wilcoxon_signed_rank,
    paired_t_interval,
    paired_t_test,
)


def policy_fingerprint(algorithm_or_model: Any) -> str:
    """Hash policy weights using SHA-256 for auditing checkpoint immutability.

    Accepts BaseAlgorithm, SB3 model, or PyTorch nn.Module.
    """
    model = getattr(algorithm_or_model, "model", algorithm_or_model)
    policy = getattr(model, "policy", model)
    if not hasattr(policy, "state_dict"):
        raise RuntimeError("Target has no 'state_dict()' interface for fingerprinting.")

    digest = hashlib.sha256()
    state = policy.state_dict()
    for name in sorted(state.keys()):
        tensor = state[name]
        digest.update(name.encode("utf-8"))
        try:
            digest.update(bytes(tensor.detach().cpu().numpy().tobytes()))
        except Exception as err:
            raise RuntimeError(f"Cannot fingerprint parameter '{name}': {err}") from err
    return digest.hexdigest()


def parameter_delta_norm(
    dict_before: Dict[str, torch.Tensor],
    dict_after: Dict[str, torch.Tensor],
) -> float:
    """Compute the Euclidean norm ||theta_new - theta_old||_2 across all model parameters."""
    total_sq = 0.0
    for key, t1 in dict_before.items():
        if key in dict_after:
            t2 = dict_after[key]
            if t1.dtype.is_floating_point and t2.dtype.is_floating_point:
                diff = (t2.detach().float() - t1.detach().float()).cpu()
                total_sq += float(torch.sum(diff**2).item())
    return math.sqrt(total_sq)


@dataclass
class Transition:
    """Single environment transition tuple."""

    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    episode_index: int  # 1-based post-shift episode index


class TransitionBuffer:
    """Cumulative buffer storing post-shift transitions with return computation."""

    def __init__(self) -> None:
        self.transitions: List[Transition] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        episode_index: int,
    ) -> None:
        self.transitions.append(
            Transition(
                obs=np.asarray(obs, dtype=np.float32).copy(),
                action=np.asarray(action, dtype=np.float32).copy(),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32).copy(),
                done=bool(done),
                episode_index=int(episode_index),
            )
        )

    def get_transitions(self, max_episode: int) -> List[Transition]:
        """Return all transitions up to max_episode (cumulative window)."""
        return [t for t in self.transitions if t.episode_index <= max_episode]

    def compute_returns_and_advantages(
        self,
        transitions: List[Transition],
        model: Any,
        gamma: float = 0.99,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute discounted Monte-Carlo returns and normalized advantages."""
        if not transitions:
            raise ValueError("Cannot compute returns on an empty transition set.")

        obs_list = [t.obs for t in transitions]
        act_list = [t.action for t in transitions]
        rew_list = [t.reward for t in transitions]
        done_list = [t.done for t in transitions]

        n = len(transitions)
        returns = np.zeros(n, dtype=np.float32)

        # Discounted return per episode sequence
        running_g = 0.0
        for i in reversed(range(n)):
            if done_list[i]:
                running_g = 0.0
            running_g = rew_list[i] + gamma * running_g
            returns[i] = running_g

        device = getattr(model, "device", "cpu")
        obs_tensor = torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device)
        act_tensor = torch.as_tensor(np.array(act_list), dtype=torch.float32, device=device)
        ret_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

        policy = getattr(model, "policy", model)
        with torch.no_grad():
            values, _, _ = policy.evaluate_actions(obs_tensor, act_tensor)
            values_flat = values.flatten()
            advantages = ret_tensor - values_flat
            if len(advantages) > 1:
                adv_std = float(advantages.std().item())
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / adv_std

        return obs_tensor, act_tensor, ret_tensor, advantages


def execute_ppo_adaptation_block(
    model: Any,
    buffer: TransitionBuffer,
    max_episode: int,
    derived_seed: int,
    lr: float = 1e-4,
    n_epochs: int = 2,
    batch_size: int = 32,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> Dict[str, Any]:
    """Execute an online PPO adaptation block B_k according to TREATMENT_CARD.md."""
    # Seed RNGs deterministically
    torch.manual_seed(derived_seed)
    np.random.seed(derived_seed)

    transitions = buffer.get_transitions(max_episode)
    if not transitions:
        raise ValueError(f"No transitions available for adaptation up to episode {max_episode}")

    policy = getattr(model, "policy", model)
    fingerprint_before = policy_fingerprint(policy)
    state_before = {k: v.clone() for k, v in policy.state_dict().items()}

    # Compute targets with current policy
    obs_t, act_t, ret_t, adv_t = buffer.compute_returns_and_advantages(transitions, model)

    with torch.no_grad():
        _, old_log_probs, _ = policy.evaluate_actions(obs_t, act_t)

    optimizer = getattr(policy, "optimizer", None)
    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Set adaptation learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    num_samples = len(transitions)
    indices = np.arange(num_samples)

    rng = np.random.default_rng(derived_seed)

    for _ in range(n_epochs):
        rng.shuffle(indices)
        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            b_obs = obs_t[batch_idx]
            b_act = act_t[batch_idx]
            b_ret = ret_t[batch_idx]
            b_adv = adv_t[batch_idx]
            b_old_log_prob = old_log_probs[batch_idx]

            values, log_probs, entropy = policy.evaluate_actions(b_obs, b_act)

            # Ratio and surrogate objective
            ratio = torch.exp(log_probs - b_old_log_prob)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss and entropy bonus
            value_loss = 0.5 * ((values.flatten() - b_ret) ** 2).mean()
            ent_loss = -entropy.mean()

            loss = policy_loss + vf_coef * value_loss + ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
            optimizer.step()

    fingerprint_after = policy_fingerprint(policy)
    state_after = policy.state_dict()
    delta_norm = parameter_delta_norm(state_before, state_after)

    return {
        "derived_seed": int(derived_seed),
        "num_transitions": len(transitions),
        "data_episodes": [1, max_episode],
        "parameter_delta_norm": float(delta_norm),
        "fingerprint_before": fingerprint_before,
        "fingerprint_after": fingerprint_after,
    }


def evaluate_episode(
    env: Any,
    policy_model: Any,
    seed: int,
    collect_transitions: bool = False,
    buffer: Optional[TransitionBuffer] = None,
    episode_index: int = 1,
) -> Tuple[float, int, bool, bool]:
    """Run one evaluation episode with deterministic action selection."""
    obs, _ = env.reset(seed=int(seed))
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    success = False
    collision = False

    while not (terminated or truncated):
        # Deterministic action prediction
        if hasattr(policy_model, "predict"):
            action, _ = policy_model.predict(obs, deterministic=True)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_dist = policy_model.get_distribution(obs_t)
                action = action_dist.mode().cpu().numpy()[0]

        next_obs, reward, term, trunc, info = env.step(action)
        terminated = bool(term)
        truncated = bool(trunc)
        total_reward += float(reward)
        steps += 1

        if collect_transitions and buffer is not None:
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=(terminated or truncated),
                episode_index=episode_index,
            )

        obs = next_obs
        if info.get("success", False) or info.get("is_success", False):
            success = True
        if info.get("collision", False):
            collision = True

    return float(total_reward), steps, success, collision


@dataclass
class ReplicateReport:
    """Output metrics and artifacts for one replicate (Adaptive vs Fixed)."""

    replicate_index: int
    training_seed: int
    pre_shift_seeds: List[int]
    post_shift_seeds: List[int]
    update_seeds: List[int]
    frozen_fingerprint: str
    pre_shift_returns: List[float]
    shock_returns: List[float]
    fixed_post_returns: List[float]
    adaptive_post_returns: List[float]
    fixed_recovery: Dict[str, Any]
    adaptive_recovery: Dict[str, Any]
    d_i: float  # T_H(Adaptive) - T_H(Fixed)
    block_logs: List[Dict[str, Any]]


def run_adaptive_vs_fixed_replicate(
    replicate_index: int,
    training_seed: int,
    config: Optional[ExperimentConfig] = None,
    training_timesteps: int = 60000,
    quick_test_mode: bool = False,
) -> ReplicateReport:
    """Execute Step 1 through Step 8 of the research protocol for one replicate.

    Args:
        replicate_index: 1-based replicate index (1..10).
        training_seed: Pinned training seed integer from TRAINING_SEEDS.
        config: Optional base ExperimentConfig.
        training_timesteps: Total timesteps for nominal policy training.
        quick_test_mode: If True, uses small training steps and short episodes for fast testing.

    Returns:
        ReplicateReport containing all pre/post returns, fingerprints, block logs, and recovery.
    """
    # 1. Generate deterministic seed schedules
    pre_seeds = [derive_seed(training_seed, "pre", j) for j in range(1, K_PRE + 1)]
    post_seeds = [derive_seed(training_seed, "post", j) for j in range(1, N_POST + 1)]
    update_seeds = [derive_seed(training_seed, "update", b) for b in range(N_UPDATE)]

    # 2. Step 1: Train nominal policy on nominal env (TRAIN parameters: 8 obstacles, low wind)
    nominal_max_steps = 50 if quick_test_mode else 300
    steps_budget = 200 if quick_test_mode else training_timesteps

    train_env = DroneDisturbed3DEnv(
        num_obstacles=8,
        num_dynamic_obstacles=0,
        wind_speed=0.5,
        gust_sigma=0.15,
        disturbance_strength=0.0,
        max_steps=nominal_max_steps,
    )

    ppo_algo = PPOAlgorithm(
        env=train_env,
        learning_rate=3e-4,
        n_steps=64 if quick_test_mode else 1024,
        batch_size=32 if quick_test_mode else 64,
        n_epochs=2 if quick_test_mode else 10,
        seed=int(training_seed),
    )
    ppo_algo.train(total_timesteps=steps_budget)
    train_env.close()
    assert ppo_algo.model is not None, "PPO model was not initialized"

    # 3. Step 2: Freeze checkpoint and record fingerprint
    frozen_fingerprint = policy_fingerprint(ppo_algo)
    frozen_weights = copy.deepcopy(ppo_algo.model.policy.state_dict())

    # 4. Step 3: Shared pre-shift evaluation (K_pre = 15 episodes on nominal env)
    eval_nominal_env = DroneDisturbed3DEnv(
        num_obstacles=8,
        num_dynamic_obstacles=0,
        wind_speed=0.5,
        gust_sigma=0.15,
        disturbance_strength=0.0,
        max_steps=nominal_max_steps,
    )

    pre_shift_returns: List[float] = []
    for j, s in enumerate(pre_seeds, start=1):
        ret, _, _, _ = evaluate_episode(eval_nominal_env, ppo_algo.model, seed=s)
        pre_shift_returns.append(ret)
    eval_nominal_env.close()

    # 5. Step 4 & 5: Shift introduction (TEST-B) & Shared shock window (episodes 1-5)
    # TEST-B: 12 static obstacles, moderate steady wind 4.0 m/s, gust sigma 0.6
    shifted_env = DroneDisturbed3DEnv(
        num_obstacles=12,
        num_dynamic_obstacles=0,
        wind_speed=4.0,
        gust_sigma=0.6,
        disturbance_strength=0.0,
        max_steps=nominal_max_steps,
    )

    shock_returns: List[float] = []
    adaptive_buffer = TransitionBuffer()

    for j in range(1, WINDOW + 1):
        s = post_seeds[j - 1]
        ret, _, _, _ = evaluate_episode(
            shifted_env,
            ppo_algo.model,
            seed=s,
            collect_transitions=True,
            buffer=adaptive_buffer,
            episode_index=j,
        )
        shock_returns.append(ret)

    # 6. Step 6: Fork into Fixed and Adaptive arms
    # Both start from identical frozen checkpoint
    fixed_post_returns: List[float] = list(shock_returns)
    adaptive_post_returns: List[float] = list(shock_returns)

    # Fixed model remains completely frozen
    fixed_model = ppo_algo.model
    fixed_model.policy.load_state_dict(frozen_weights)

    # Adaptive model starts from frozen checkpoint
    adaptive_algo = PPOAlgorithm(
        env=shifted_env,
        learning_rate=1e-4,
        n_steps=64,
        batch_size=32,
        seed=int(training_seed),
    )
    assert adaptive_algo.model is not None, "Adaptive PPO model was not initialized"
    adaptive_algo.model.policy.load_state_dict(copy.deepcopy(frozen_weights))
    adaptive_model = adaptive_algo.model

    block_logs: List[Dict[str, Any]] = []

    # Execute Block B5 (episodes 1-5 data) strictly before episode 6 reset
    b5_log = execute_ppo_adaptation_block(
        model=adaptive_model,
        buffer=adaptive_buffer,
        max_episode=5,
        derived_seed=update_seeds[0],
    )
    b5_log["block_id"] = "B5"
    b5_log["episode_k"] = 5
    block_logs.append(b5_log)

    # 7. Step 7: Post-shift episodes 6..15
    for k in range(6, N_POST + 1):
        s_eval = post_seeds[k - 1]

        # Fixed arm evaluation (frozen policy, identical seed)
        fixed_ret, _, _, _ = evaluate_episode(shifted_env, fixed_model, seed=s_eval)
        fixed_post_returns.append(fixed_ret)

        # Adaptive arm evaluation (adapted policy, identical seed)
        adapt_ret, _, _, _ = evaluate_episode(
            shifted_env,
            adaptive_model,
            seed=s_eval,
            collect_transitions=True,
            buffer=adaptive_buffer,
            episode_index=k,
        )
        adaptive_post_returns.append(adapt_ret)

        # Execute Block B_k if k < N_POST
        if k < N_POST:
            block_idx = k - 5
            b_log = execute_ppo_adaptation_block(
                model=adaptive_model,
                buffer=adaptive_buffer,
                max_episode=k,
                derived_seed=update_seeds[block_idx],
            )
            b_log["block_id"] = f"B{k}"
            b_log["episode_k"] = k
            block_logs.append(b_log)

    shifted_env.close()

    # 8. Step 8: Causal Recovery Analysis
    rec_fixed = compute_recovery(pre_shift_returns, fixed_post_returns)
    rec_adaptive = compute_recovery(pre_shift_returns, adaptive_post_returns)

    # D_i = T_H(Adaptive) - T_H(Fixed)
    d_i = float(rec_adaptive.truncated_recovery_time - rec_fixed.truncated_recovery_time)

    return ReplicateReport(
        replicate_index=int(replicate_index),
        training_seed=int(training_seed),
        pre_shift_seeds=pre_seeds,
        post_shift_seeds=post_seeds,
        update_seeds=update_seeds,
        frozen_fingerprint=frozen_fingerprint,
        pre_shift_returns=pre_shift_returns,
        shock_returns=shock_returns,
        fixed_post_returns=fixed_post_returns,
        adaptive_post_returns=adaptive_post_returns,
        fixed_recovery=asdict(rec_fixed),
        adaptive_recovery=asdict(rec_adaptive),
        d_i=d_i,
        block_logs=block_logs,
    )


class AdaptiveShiftRunner:
    """Orchestrates multi-replicate online adaptation vs fixed benchmark experiments."""

    def __init__(
        self,
        training_seeds: Sequence[int] = TRAINING_SEEDS,
        training_timesteps: int = 60000,
        output_dir: str = "artifacts/benchmarks",
        quick_test_mode: bool = False,
    ) -> None:
        self.training_seeds = [int(s) for s in training_seeds]
        self.training_timesteps = int(training_timesteps)
        self.output_dir = Path(output_dir)
        self.quick_test_mode = bool(quick_test_mode)

    def run(self) -> Dict[str, Any]:
        """Execute full benchmark across all specified training seeds."""
        start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        replicate_reports: List[ReplicateReport] = []
        differences: List[float] = []

        for idx, seed in enumerate(self.training_seeds, start=1):
            rep = run_adaptive_vs_fixed_replicate(
                replicate_index=idx,
                training_seed=seed,
                training_timesteps=self.training_timesteps,
                quick_test_mode=self.quick_test_mode,
            )
            replicate_reports.append(rep)
            differences.append(rep.d_i)

        # Statistical analysis
        min_n = min(2, len(differences)) if self.quick_test_mode else MIN_VALID_N
        t_res = paired_t_test(differences, min_valid_n=min_n)
        ci_res = paired_t_interval(differences, confidence=1.0 - ALPHA, min_valid_n=min_n)
        cohen_d = cohen_dz(differences)
        boot_ci = bootstrap_percentile_ci(
            differences,
            confidence=1.0 - ALPHA,
            reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED,
            min_valid_n=min_n,
        )
        sign_res = exact_sign_test(differences, min_valid_n=min_n)
        wilcox_res = exact_wilcoxon_signed_rank(differences, min_valid_n=min_n)

        finish_time = time.time()
        json_path = self.output_dir / "adaptive_vs_fixed.json"
        csv_path = self.output_dir / "adaptive_vs_fixed.csv"

        # Export CSV summary first so it can be hashed in the manifest
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "replicate",
                    "training_seed",
                    "p_pre_mean",
                    "p0_mean",
                    "fixed_t_h",
                    "fixed_status",
                    "adaptive_t_h",
                    "adaptive_status",
                    "d_i",
                ]
            )
            for r in replicate_reports:
                writer.writerow(
                    [
                        r.replicate_index,
                        r.training_seed,
                        f"{r.fixed_recovery['p_pre']:.4f}",
                        f"{r.fixed_recovery['p0']:.4f}",
                        r.fixed_recovery["truncated_recovery_time"],
                        r.fixed_recovery["status"],
                        r.adaptive_recovery["truncated_recovery_time"],
                        r.adaptive_recovery["status"],
                        r.d_i,
                    ]
                )

        manifest = create_manifest(
            experiment_name="adaptive_vs_fixed_shift",
            config_dict={
                "environment": "drone_disturbed",
                "condition": "TEST-B",
                "algorithm": "ppo",
                "training_seeds": self.training_seeds,
                "protocol_version": PROTOCOL_VERSION,
            },
            started_at=start_time,
            finished_at=finish_time,
            training_time_seconds=finish_time - start_time,
            artifacts={"csv_summary": csv_path},
        )

        report_dict: Dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "experiment": "online_adaptation_vs_fixed_shift",
            "condition": "TEST-B",
            "num_replicates": len(replicate_reports),
            "differences": differences,
            "statistics": {
                "mean_d": float(t_res.mean),
                "std_d": float(t_res.std_dev),
                "se_d": float(t_res.standard_error),
                "t_statistic": float(t_res.t_statistic),
                "p_value_onesided": float(t_res.p_value),
                "confidence_interval_95": [float(ci_res[0]), float(ci_res[1])],
                "cohens_dz": float(cohen_d) if cohen_d is not None else None,
                "bootstrap_ci_95": [float(boot_ci[0]), float(boot_ci[1])],
                "sign_test_p": float(sign_res.p_value),
                "wilcoxon_p": float(wilcox_res.p_value),
            },
            "replicates": [asdict(r) for r in replicate_reports],
            "manifest": manifest.model_dump(),
        }

        # Export JSON artifact
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        return report_dict


# Convenience API
run_adaptive_vs_fixed_experiment = AdaptiveShiftRunner

__all__ = [
    "AdaptiveShiftRunner",
    "ReplicateReport",
    "Transition",
    "TransitionBuffer",
    "execute_ppo_adaptation_block",
    "parameter_delta_norm",
    "policy_fingerprint",
    "run_adaptive_vs_fixed_experiment",
    "run_adaptive_vs_fixed_replicate",
]
