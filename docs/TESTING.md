# Testing Strategy

This document outlines the testing strategy for AdaptiveRL.

## End-to-End Pipeline Smoke Test
The primary end-to-end smoke test validates the CLI workflow. See `tests/test_e2e_smoke.py`.

## Research Protocol Smoke Test
To prevent regressions in the central research workflow, a fast, deterministic CI smoke test (`tests/test_research_protocol_smoke.py`) exercises the entire research pipeline end-to-end using tiny step budgets (≤128 steps). It verifies all protocol invariants without slowing down CI.

### Invariants Checked
The smoke test asserts the following protocol invariants:
- **Valid Checkpoints**: Training produces a valid policy checkpoint and SHA-256 fingerprint.
- **Baseline Return**: Pre-shift evaluation runs and records baseline return.
- **Identical Shock Windows**: Shock window returns are identical for both Fixed and Adaptive arms.
- **Fixed Arm Mutability**: Fixed arm policy weights remain completely unchanged (L2 delta = 0.0).
- **Adaptive Arm Mutability**: Adaptive arm executes online update blocks and modifies weights (L2 delta > 0.0).
- **Recovery Metrics**: $T_H$ computed correctly via `adaptive_rl.protocol.recovery`.
- **Experiment Manifest**: A manifest is created with valid artifact checksums.
- **Claim Validation**: Passes on generated output.

This test explicitly verifies **code pipeline integrity**, not scientific hypothesis validity.
