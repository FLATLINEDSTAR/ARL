"""Unit and integration tests for the Adaptive vs Fixed online adaptation experiment runner."""

import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.environments.disturbed_drone import DroneDisturbed3DEnv
from adaptive_rl.experiments.shift_runner import (
    AdaptiveShiftRunner,
    TransitionBuffer,
    execute_ppo_adaptation_block,
    parameter_delta_norm,
    policy_fingerprint,
    run_adaptive_vs_fixed_replicate,
)
from adaptive_rl.protocol.constants import K_PRE, N_POST, N_UPDATE, WINDOW


def test_policy_fingerprint_and_delta_norm() -> None:
    """Verify policy fingerprinting and parameter delta norm calculations."""
    env = DroneDisturbed3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20)
    algo = PPOAlgorithm(env=env, learning_rate=1e-4)

    fp1 = policy_fingerprint(algo)
    assert isinstance(fp1, str)
    assert len(fp1) == 64

    # Identical state produces identical fingerprint and zero delta norm
    state1 = {k: v.clone() for k, v in algo.model.policy.state_dict().items()}
    state2 = {k: v.clone() for k, v in state1.items()}
    assert parameter_delta_norm(state1, state2) == pytest.approx(0.0)

    # Perturb one weight tensor slightly
    for key in state2:
        if "weight" in key and state2[key].dtype.is_floating_point:
            state2[key] += 0.1
            break

    algo.model.policy.load_state_dict(state2)
    fp2 = policy_fingerprint(algo)
    assert fp1 != fp2
    assert parameter_delta_norm(state1, state2) > 0.0

    env.close()


def test_transition_buffer_and_adaptation_block() -> None:
    """Verify transition buffer storage, returns calculation, and adaptation block execution."""
    env = DroneDisturbed3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20)
    algo = PPOAlgorithm(env=env, learning_rate=1e-4)

    buffer = TransitionBuffer()
    obs_dim = 29
    act_dim = 3

    # Add transitions across 5 episodes
    for ep in range(1, 6):
        for _ in range(10):
            buffer.add(
                obs=np.random.randn(obs_dim).astype(np.float32),
                action=np.random.uniform(-1, 1, size=act_dim).astype(np.float32),
                reward=1.0,
                next_obs=np.random.randn(obs_dim).astype(np.float32),
                done=False,
                episode_index=ep,
            )

    t5 = buffer.get_transitions(max_episode=5)
    assert len(t5) == 50
    t3 = buffer.get_transitions(max_episode=3)
    assert len(t3) == 30

    fp_before = policy_fingerprint(algo)
    log = execute_ppo_adaptation_block(
        model=algo.model,
        buffer=buffer,
        max_episode=5,
        derived_seed=42,
        lr=1e-4,
        n_epochs=1,
        batch_size=16,
    )
    fp_after = policy_fingerprint(algo)

    assert log["derived_seed"] == 42
    assert log["num_transitions"] == 50
    assert log["parameter_delta_norm"] > 0.0
    assert log["fingerprint_before"] == fp_before
    assert log["fingerprint_after"] == fp_after
    assert fp_before != fp_after

    env.close()


def test_run_adaptive_vs_fixed_replicate_quick(tmp_path: Path) -> None:
    """Verify complete end-to-end replicate execution in quick test mode."""
    # Use training seed 31001
    rep = run_adaptive_vs_fixed_replicate(
        replicate_index=1,
        training_seed=31001,
        quick_test_mode=True,
    )

    assert rep.replicate_index == 1
    assert rep.training_seed == 31001
    assert len(rep.pre_shift_seeds) == K_PRE == 15
    assert len(rep.post_shift_seeds) == N_POST == 15
    assert len(rep.update_seeds) == N_UPDATE == 10

    # Verification of evaluation trajectories
    assert len(rep.pre_shift_returns) == 15
    assert len(rep.shock_returns) == WINDOW == 5
    assert len(rep.fixed_post_returns) == 15
    assert len(rep.adaptive_post_returns) == 15

    # Shock window returns must be shared identically by construction
    for i in range(5):
        assert rep.fixed_post_returns[i] == pytest.approx(rep.shock_returns[i])
        assert rep.adaptive_post_returns[i] == pytest.approx(rep.shock_returns[i])

    # Exactly 10 update blocks executed (B5..B14)
    assert len(rep.block_logs) == 10
    assert rep.block_logs[0]["block_id"] == "B5"
    assert rep.block_logs[-1]["block_id"] == "B14"

    # Recovery computation
    assert "truncated_recovery_time" in rep.fixed_recovery
    assert "truncated_recovery_time" in rep.adaptive_recovery
    assert rep.d_i == pytest.approx(
        rep.adaptive_recovery["truncated_recovery_time"]
        - rep.fixed_recovery["truncated_recovery_time"]
    )


def test_adaptive_shift_runner_full_workflow(tmp_path: Path) -> None:
    """Verify AdaptiveShiftRunner multi-replicate benchmark execution and artifact export."""
    runner = AdaptiveShiftRunner(
        training_seeds=[31001, 31002],
        output_dir=str(tmp_path / "benchmarks"),
        quick_test_mode=True,
    )

    report = runner.run()

    assert report["protocol_version"] == "2.0"
    assert report["num_replicates"] == 2
    assert len(report["differences"]) == 2
    assert "statistics" in report
    assert "p_value_onesided" in report["statistics"]
    assert "manifest" in report

    json_file = tmp_path / "benchmarks" / "adaptive_vs_fixed.json"
    csv_file = tmp_path / "benchmarks" / "adaptive_vs_fixed.csv"

    assert json_file.exists()
    assert csv_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["experiment"] == "online_adaptation_vs_fixed_shift"
        assert len(loaded["replicates"]) == 2
