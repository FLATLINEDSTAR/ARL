## Issue #272 implementation update

Implemented on branch `feat/issue-272-experiment-cell-registry` in commit `11f8709cbd46b6020842f6d83e239465962f94da`.

### What changed

- Added `src/adaptive_rl/experiments/matrix.py` with six immutable cell specs, canonical IDs, shared budget/replicate/seed-schedule invariants, derived status, stale-lock recovery, atomic completion records, artifact checksums, single-cell and batch dispatch.
- Added `adaptive-rl benchmark matrix --list`, `--cell`, and `--all` in `src/adaptive_rl/cli.py`.
- Added `docs/BENCHMARKS.md` and an append-only Issue #272 addendum in `docs/research/adaptive_rl_hypothesis.md`; existing preregistered criteria were not changed.
- Fixed `EpisodeRecord.to_dict()` and `post_shift_data()` being nested under `EpisodeTaggedTransition` on the base branch. Added a regression test. No SAC online-adaptation implementation was added.

### Cell status on this branch

| Cell | Status | Reason |
|---|---|---|
| `CELL_PPO_TEST_A` | `UNSUPPORTED` | TEST-A has no supported shift config/schema block. |
| `CELL_PPO_TEST_B` | `AVAILABLE` | Existing PPO TEST-B path passed a one-replicate real smoke; a smoke is not a completed ten-replicate result. |
| `CELL_PPO_TEST_D` | `UNSUPPORTED` | TEST-D is unconfigured; the issue's sensor-noise/wind-gust/dynamic-obstacle definition differs from the preregistered hidden-disturbance value. |
| `CELL_SAC_TEST_A` | `UNSUPPORTED` | TEST-A has no supported shift config/schema block. |
| `CELL_SAC_TEST_B` | `UNSUPPORTED` | Real one-replicate smoke failed during tagged replay insertion. SAC repair belongs to Issue 3 and is out of scope. |
| `CELL_SAC_TEST_D` | `UNSUPPORTED` | TEST-D is unconfigured and its definitions conflict as above. |

TEST-B follows the source YAML/preregistration: 12 obstacles, 4.0 m/s steady wind, gust sigma 0.6; nominal TRAIN is 8 obstacles and 0.5 m/s. Canonical IDs use uppercase underscores; lowercase/hyphen aliases are accepted. The Issue #272 registry is separate from the preregistered six-cell primary family.

### CLI

```bash
adaptive-rl benchmark matrix --list
adaptive-rl benchmark matrix --cell CELL_PPO_TEST_B
adaptive-rl benchmark matrix --all
```

### Verification

- `pytest -q -p no:cacheprovider`: **306 passed, 3 failed**. Failures: PPO rollout buffer position, SAC batch episode tags, and reward-ablation CLI smoke. These are in existing baseline adaptation/ablation paths; the SAC failures and missing helpers are present on base commit `26a4649`.
- `pytest ... tests/test_experiment_matrix.py tests/test_cli.py -k 'not experiment_ablation_smoke'`: **36 passed, 1 deselected**.
- Real available-cell smoke (`test_one_replicate_smoke_for_every_available_cell`): **1 passed in 5.65s** for PPO/TEST-B.
- `ruff check` on changed Python files: **All checks passed**. Repository-wide `ruff check`: **10 existing errors**, primarily undefined SAC helpers in `src/adaptive_rl/algorithms/adaptation.py` and one unused import in the adaptation runner.
- `ruff format --check` on changed/new implementation files: **6 files already formatted**. Repository-wide format check identifies **5 existing unrelated files** that need formatting.
- `mypy src/adaptive_rl`: **10 existing errors** in adaptation modules (including SAC helpers and tagged transition typing). Targeted matrix/CLI invocation reports those same imported baseline errors; the matrix-specific `Any` return issue was fixed.

### Acceptance criteria

- [x] Six registry slots, algorithm/environment metadata, config reference, shared schedule and N=10 are defined.
- [x] Single-cell CLI dispatch is implemented.
- [x] Batch dispatch runs AVAILABLE cells sequentially, continues after errors, and reports final statuses.
- [x] Unknown/unsupported cells fail cleanly before output creation; atomic result/checksum validation is covered by tests.
- [ ] All requested scenario/algorithm cells are runnable: TEST-A/TEST-D lack agreed source configuration, and SAC/TEST-B fails its existing smoke. They remain explicitly unsupported.

The implementation uses existing seed utilities. Roadmap Issues 4/5 and PR #266 could not be checked live because GitHub CLI is not authenticated. No new immutable study validation was built, and this matrix uses the existing non-study runner path.

**PR:** not opened. `gh auth status` reports no authenticated host, and pushing to `origin` was rejected with HTTP 403 for the configured identity `open-source889`.
