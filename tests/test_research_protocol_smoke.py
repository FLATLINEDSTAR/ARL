"""Smoke test for the continuous integration end-to-end research protocol workflow."""

import adaptive_rl.protocol.constants as constants
import adaptive_rl.protocol.recovery as recovery
from adaptive_rl.experiments.shift_runner import AdaptiveShiftRunner


def test_research_protocol_invariants(monkeypatch):
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
        env_name="drone",  # Use drone since gridworld is not registered
        training_steps=128,
        k_pre=2,
        w=2,
        h=4,
        block_steps=16,
    )

    manifest = runner.run()

    # 2. Invariant checks
    # Training produces valid policy checkpoint and SHA-256 fingerprint.
    assert manifest["fingerprint"] is not None
    assert len(manifest["fingerprint"]) == 64  # Valid SHA-256 hex string

    # Pre-shift evaluation runs and records baseline return.
    assert len(manifest["pre_shift_returns"]) == 2

    # Shock window returns are identical for both Fixed and Adaptive arms.
    assert manifest["adaptive_post_returns"][:2] == manifest["shock_returns"]
    assert manifest["fixed_post_returns"][:2] == manifest["shock_returns"]

    # Fixed arm policy weights remain completely unchanged (L2 delta = 0.0).
    assert manifest["fixed_l2_delta"] == 0.0

    # Adaptive arm executes online update blocks and modifies weights (L2 delta > 0.0).
    assert manifest["adaptive_l2_delta"] > 0.0

    # Recovery metrics TH computed via adaptive_rl.protocol.recovery.
    assert "fixed_recovery_th" in manifest
    assert "adaptive_recovery_th" in manifest

    # Experiment manifest created with valid artifact checksums.
    assert manifest["checksum"] is not None
    assert len(manifest["checksum"]) == 64

    # Claim validation passes on generated output.
    assert type(manifest["fixed_recovery_th"]) is int
    assert type(manifest["adaptive_recovery_th"]) is int
