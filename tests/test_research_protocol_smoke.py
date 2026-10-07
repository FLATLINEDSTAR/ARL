"""Smoke test for the continuous integration end-to-end research protocol workflow."""

import hashlib
import json
import adaptive_rl.protocol.constants as constants
import adaptive_rl.protocol.recovery as recovery
from adaptive_rl.experiments.shift_runner import AdaptiveShiftRunner, policy_fingerprint
from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.environments.registry import make_env
from adaptive_rl.protocol.recovery import compute_recovery


def test_research_protocol_invariants(monkeypatch, tmp_path):
    """Verify that the central research workflow executes and maintains all invariants."""
    # 1. Miniature research run configuration
    # 1 replicate, 128 training steps, K_pre=2, W=2, H=4, mini-adaptation 16 steps.
    monkeypatch.setattr(constants, "K_PRE", 2)
    monkeypatch.setattr(constants, "WINDOW", 2)
    monkeypatch.setattr(constants, "HORIZON", 4)
    monkeypatch.setattr(constants, "PERSISTENCE", 2)

    monkeypatch.setattr(recovery, "K_PRE", 2)
    monkeypatch.setattr(recovery, "WINDOW", 2)
    monkeypatch.setattr(recovery, "HORIZON", 4)
    monkeypatch.setattr(recovery, "PERSISTENCE", 2)

    runner = AdaptiveShiftRunner(
        env_name="drone",
        training_steps=128,
        k_pre=2,
        w=2,
        h=4,
        block_steps=16,
        seed=42,
        output_dir=tmp_path
    )

    manifest = runner.run()

    # 2. Invariant checks
    # Training produces valid policy checkpoint and SHA-256 fingerprint.
    assert manifest["fingerprint"] is not None
    assert len(manifest["fingerprint"]) == 64  # Valid SHA-256 hex string
    assert (tmp_path / "frozen_policy.zip").exists()
    
    # Validate fingerprint corresponds to the expected policy
    loaded_algo = PPOAlgorithm.from_pretrained(tmp_path / "frozen_policy.zip", env=make_env("drone"))
    recomputed_fingerprint = policy_fingerprint(loaded_algo)
    assert manifest["fingerprint"] == recomputed_fingerprint

    # Pre-shift evaluation runs and records baseline return.
    assert len(manifest["pre_shift_returns"]) == 2

    # Shock window returns are independently evaluated and identical for both Fixed and Adaptive arms initially.
    assert manifest["adaptive_post_returns"][:2] == manifest["fixed_post_returns"][:2]

    # Fixed arm policy weights remain completely unchanged (L2 delta = 0.0).
    assert manifest["fixed_l2_delta"] == 0.0

    # Adaptive arm executes online update blocks and modifies weights (L2 delta > 0.0).
    assert manifest["adaptive_l2_delta"] > 0.0

    # Recovery metrics TH computed correctly via adaptive_rl.protocol.recovery.
    assert "fixed_recovery_th" in manifest
    assert "adaptive_recovery_th" in manifest
    
    recomputed_fixed_recovery = compute_recovery(manifest["pre_shift_returns"], manifest["fixed_post_returns"])
    assert manifest["fixed_recovery_th"] == recomputed_fixed_recovery.truncated_recovery_time

    recomputed_adaptive_recovery = compute_recovery(manifest["pre_shift_returns"], manifest["adaptive_post_returns"])
    assert manifest["adaptive_recovery_th"] == recomputed_adaptive_recovery.truncated_recovery_time

    # Experiment manifest created with valid artifact checksums.
    assert manifest["checksum"] is not None
    assert len(manifest["checksum"]) == 64
    
    # Validate persisted artifact/manifest checksums by recomputation
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    
    with open(manifest_path, "r") as f:
        persisted_data = json.load(f)
        
    persisted_checksum = persisted_data.pop("checksum")
    recomputed_checksum = hashlib.sha256(json.dumps(persisted_data, sort_keys=True).encode("utf-8")).hexdigest()
    
    assert persisted_checksum == recomputed_checksum
    assert manifest["checksum"] == recomputed_checksum
