"""Experiment runner for the online adaptation protocol."""

import hashlib
import json
from pathlib import Path

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.environments.registry import make_env
from adaptive_rl.protocol.recovery import compute_recovery


def policy_fingerprint(algo: PPOAlgorithm) -> str:
    """Compute a SHA-256 fingerprint of the policy weights."""
    state_dict = algo.model.policy.state_dict()
    hasher = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        hasher.update(k.encode("utf-8"))
        tensor_bytes = state_dict[k].cpu().numpy().tobytes()
        hasher.update(tensor_bytes)
    return hasher.hexdigest()


def l2_delta(w1, w2) -> float:
    """Compute L2 distance between two sets of policy weights."""
    return sum((w1[k] - w2[k]).float().pow(2).sum() for k in w1.keys()).item()


def get_weights(algo: PPOAlgorithm) -> dict:
    """Extract a copy of the policy weights."""
    return {k: v.clone() for k, v in algo.model.policy.state_dict().items()}


class AdaptiveShiftRunner:
    """Runner that executes the full Adaptive RL shift protocol.

    Includes Train, Freeze, Pre-Shift Eval, Shock Window, Fork,
    Online Adaptation Blocks, and Recovery Analysis.
    """

    def __init__(self, env_name="drone", training_steps=128, k_pre=2, w=2, h=4, block_steps=16, seed=42, output_dir=None):
        self.env_name = env_name
        self.training_steps = training_steps
        self.k_pre = k_pre
        self.w = w
        self.h = h
        self.block_steps = block_steps
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else Path("experiments/smoke_test")

    def _eval_algo(self, algo: PPOAlgorithm, env, episodes: int) -> list:
        returns = []
        for _ in range(episodes):
            obs, _ = env.reset(seed=self.seed)
            done = False
            ret = 0.0
            steps = 0
            while not done and steps < 5:  # Cap steps for fast execution
                action, _ = algo.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ret += float(reward)
                steps += 1
            returns.append(ret)
        return returns

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Train
        train_env = make_env(self.env_name)
        train_env.reset(seed=self.seed)
        algo = PPOAlgorithm(env=train_env, learning_rate=1e-3, n_steps=self.training_steps, batch_size=16, seed=self.seed)
        algo.train(total_timesteps=self.training_steps)

        # 2. Freeze
        fingerprint = policy_fingerprint(algo)
        frozen_weights = get_weights(algo)
        algo.save(self.output_dir / "frozen_policy.zip")

        # 3. Pre-Shift Eval (Nominal)
        eval_env = make_env(self.env_name)
        pre_shift_returns = self._eval_algo(algo, eval_env, self.k_pre)

        # 4. Shock Window (Shift Introduced) & Fork Environments
        fixed_env = make_env(self.env_name, linear_damping=0.5)
        adaptive_env = make_env(self.env_name, linear_damping=0.5)

        # 5. Fork Algorithms
        # Fixed Arm
        fixed_algo = PPOAlgorithm(
            env=fixed_env, learning_rate=1e-3, n_steps=self.block_steps, batch_size=16, seed=self.seed
        )
        fixed_algo.model.policy.load_state_dict(frozen_weights)
        fixed_weights_before = get_weights(fixed_algo)

        # Adaptive Arm
        adaptive_algo = PPOAlgorithm(
            env=adaptive_env, learning_rate=1e-3, n_steps=self.block_steps, batch_size=16, seed=self.seed
        )
        adaptive_algo.model.policy.load_state_dict(frozen_weights)
        adaptive_weights_before = get_weights(adaptive_algo)

        # 6. Evaluate Shock Window
        fixed_post_returns = self._eval_algo(fixed_algo, fixed_env, self.w)
        adaptive_post_returns = self._eval_algo(adaptive_algo, adaptive_env, self.w)

        # 7. Online Adaptation Blocks & Arm Segments
        # Run remaining episodes
        for _ in range(self.h - self.w):
            # Adaptive Arm executes online update block
            adaptive_algo.train(total_timesteps=self.block_steps)

            # Evaluate Adaptive Arm
            adaptive_post_returns.extend(self._eval_algo(adaptive_algo, adaptive_env, 1))

            # Evaluate Fixed Arm
            fixed_post_returns.extend(self._eval_algo(fixed_algo, fixed_env, 1))

        fixed_weights_after = get_weights(fixed_algo)
        adaptive_weights_after = get_weights(adaptive_algo)

        fixed_l2_delta = l2_delta(fixed_weights_before, fixed_weights_after)
        adaptive_l2_delta = l2_delta(adaptive_weights_before, adaptive_weights_after)

        # 8. Recovery Analysis
        fixed_recovery = compute_recovery(pre_shift_returns, fixed_post_returns)
        adaptive_recovery = compute_recovery(pre_shift_returns, adaptive_post_returns)

        # 9. Manifest
        manifest = {
            "fingerprint": fingerprint,
            "pre_shift_returns": pre_shift_returns,
            "fixed_post_returns": fixed_post_returns,
            "adaptive_post_returns": adaptive_post_returns,
            "fixed_l2_delta": fixed_l2_delta,
            "adaptive_l2_delta": adaptive_l2_delta,
            "fixed_recovery_th": fixed_recovery.truncated_recovery_time,
            "adaptive_recovery_th": adaptive_recovery.truncated_recovery_time,
            "checksum": "",
        }

        manifest_str = json.dumps(manifest, sort_keys=True)
        manifest["checksum"] = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        return manifest
