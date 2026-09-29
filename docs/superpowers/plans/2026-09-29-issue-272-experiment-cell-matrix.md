# Issue #272 Experiment Cell Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a truthful six-cell PPO/SAC TEST-A/B/D registry, single-cell and batch CLI, tests, and documentation without changing preregistered criteria.

**Architecture:** A new matrix module owns immutable cell metadata, status derivation, locks, artifact validation, and dispatch to the existing adaptation runner. The source YAML and preregistered seed utilities remain authoritative: only PPO/TEST-B and SAC/TEST-B are initially runnable, because TEST-A and TEST-D are not configured and TEST-D's issue description conflicts with the preregistration. CLI and docs expose those facts explicitly.

**Tech Stack:** Python 3.10+, Pydantic v2 config, Typer, Rich, pytest, existing adaptation runner and protocol seed utilities.

**Spec:** `docs/superpowers/specs/2026-09-29-issue-272-experiment-cell-matrix-design.md`

## Global Constraints

* Preserve all preregistered hypotheses, thresholds, seeds, horizon, and statistical criteria.
* Canonical cell IDs are `CELL_<ALGO>_<ENV>` with uppercase underscore forms.
* TEST-B is 12 obstacles, 4.0 m/s steady wind, and gust sigma 0.6; nominal TRAIN is 8 obstacles and 0.5 m/s.
* Planned or unsupported cells cannot create run output; completion requires valid real result artifacts.
* The matrix has ten preregistered replicates and uses the shared derived seed schedule and 60,000 timestep budget.
* Batch runs all AVAILABLE cells sequentially, continues after errors, and exits non-zero when any cell fails.
* No SAC online-adaptation implementation may be added; reuse only the existing tested runner support.

## Review Focus

* Incorrect or stale PID markers — test dead-PID recovery and live-PID RUNNING behavior.
* Malformed, incomplete, or tampered result artifacts — test they never derive COMPLETED.
* Unsupported cells causing filesystem side effects — test a clean temporary directory remains empty.
* Parser spelling/case errors and Rich markup injection — test aliases, invalid IDs, and bracketed values.
* Config/protocol drift — test shared budget, replicate count, evaluation schedule, and config source invariants reject a deviation.

---

### Task 1: Add the immutable matrix registry and artifact state model

**Files:**
- Create: `src/adaptive_rl/experiments/__init__.py`
- Create: `src/adaptive_rl/experiments/matrix.py`
- Test: `tests/test_experiment_matrix.py`

**Interfaces:**
- Consumes: `load_config`, `TRAINING_SEEDS`, `PLANNED_N`, `frozen_schedule`, `schedule_fingerprint`, `K_PRE`, `N_POST`, `N_UPDATE`.
- Produces: `Algorithm`, `ShiftEnv`, `CellStatus`, frozen `CellSpec`, `ExperimentCellRegistry`, `UnsupportedCellError`, `UnknownCellError`, `CellIdError`, `atomic_write_json(path, value)`, and `validate_cell_invariants(cells)`.
- `CellSpec` fields: `cell_id`, `algo`, `env`, `config_path`, `shift_parameters`, `training_budget`, `replicate_count`, `seed_schedule`, `evaluation_config`, `status`, `status_reason`, `result_paths`.
- Canonical IDs are uppercase underscore IDs; accept lowercase/hyphen variants by normalization. Reject malformed IDs and unsupported algorithm/environment tokens clearly.

- [ ] **Step 1: Add registry and parser tests** for exactly six cell slots, canonical and alias IDs, explicit case/format normalization, all three enums, immutable metadata, source config values, and clear errors for unknown/malformed IDs.
- [ ] **Step 2: Run the focused tests and confirm RED** because `adaptive_rl.experiments.matrix` does not yet exist.
- [ ] **Step 3: Implement the enums, frozen spec, ID parser, and registry**. Set all cells to N=10, the configured 60,000 budget, and copies of the validated shared seed/evaluation schedule. Read the base YAML once. Populate TEST-B shift parameters directly from the YAML. Set TEST-A and TEST-D to `UNSUPPORTED` because no configuration/schema support exists and include their source discrepancy in `status_reason`. Set PPO/TEST-B and SAC/TEST-B capability according to the existing runner and deterministic-action constraints.
- [ ] **Step 4: Add invariant checks** that compare every cell's budget, N, and evaluation seed schedule to the one shared protocol source; test that mutating any one value raises a descriptive error.
- [ ] **Step 5: Run registry/parser tests and confirm GREEN.**

### Task 2: Implement derived status, locking, atomic artifacts, and dispatch

**Files:**
- Modify: `src/adaptive_rl/experiments/matrix.py`
- Test: `tests/test_experiment_matrix.py`

**Interfaces:**
- Consumes: Task 1 registry and immutable specs.
- Produces: `list_cells()`, `get(cell_id)`, `by_status(status)`, `result_lookup(cell_id)`, `status(cell_id)`, `run_cell(cell_id, *, output_root=None, smoke=False, runner=run_adaptation_benchmark)`, and `run_available(*, output_root=None, runner=run_adaptation_benchmark)`.

- [ ] **Step 1: Add status and fail-fast tests** for absent results, valid results, malformed/checksum-mismatched results, active process marker, stale PID marker, and unsupported/unknown requests producing no files or directories.
- [ ] **Step 2: Run those focused tests and confirm RED.**
- [ ] **Step 3: Implement derived status and a crash-recoverable lock**. A live PID and fresh timestamp yields RUNNING; a dead PID or malformed/stale marker is removed/recovered. Acquire using exclusive file creation so concurrent launches cannot both train one cell.
- [ ] **Step 4: Implement atomic JSON writes and result validation.** Write through a temp file in the destination directory and atomically publish; completion requires valid JSON schema, cell identity, expected runner completion state, artifact paths, and SHA-256 checksums. Reuse the existing runner's atomic JSON/CSV artifacts and add an atomic per-cell completion record binding their hashes.
- [ ] **Step 5: Implement single-cell dispatch.** Resolve and reject unsupported cells before creating output. Clone the YAML config, choose PPO/SAC with existing runner parameters, enforce the shared ten-seed schedule and budget, and request stochastic action collection for PPO. Run with `study_run_id=None`, clearly keeping matrix execution distinct from Issue #271 immutable prereg-v1 execution. Allow `smoke=True` only as an API/testing machinery check, using runner smoke mode; do not expose it as a normal CLI research result.
- [ ] **Step 6: Implement batch dispatch.** Run each AVAILABLE cell once, continue after a failure, capture per-cell outcome, printable progress data, and final status mapping; mark a failed attempt without presenting it as COMPLETED.
- [ ] **Step 7: Add tests for atomic-write interruption** (simulate failure before publish and confirm destination is absent/unchanged), successful hash validation, batch continues after one failure, and one-replicate real smoke execution for each AVAILABLE cell using tiny runner smoke budgets.
- [ ] **Step 8: Run matrix tests and existing adaptation smoke/regression tests.**

### Task 3: Add `benchmark matrix` CLI

**Files:**
- Modify: `src/adaptive_rl/cli.py`
- Test: `tests/test_experiment_matrix.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ExperimentCellRegistry`, `run_cell`, and `run_available`.
- Produces: `adaptive-rl benchmark matrix --list`, `--cell CELL_ID`, and `--all` (exactly one mode required).

- [ ] **Step 1: Add CliRunner tests** for list output with six statuses/reasons, mocked single-cell dispatch, batch dispatch, unknown and unsupported clean exit codes, no traceback, and bracket-containing input rendered literally.
- [ ] **Step 2: Run the new CLI tests and confirm RED.**
- [ ] **Step 3: Implement a nested Typer command** with mutually exclusive required modes. Escape all user-provided IDs, reasons, and config values using `rich.markup.escape`; render batch progress and final cell/status table; exit 1 on dispatch or cell errors.
- [ ] **Step 4: Run new CLI tests and all existing CLI tests.**

### Task 4: Document status and operational decisions

**Files:**
- Modify: `docs/research/adaptive_rl_hypothesis.md` (append near §12.3/§12.4 only)
- Create or modify: `docs/BENCHMARKS.md`
- Test: `tests/test_experiment_matrix.py` (doc/config synchronization assertions if existing conventions support them)

- [ ] **Step 1: Add documentation checks** for six canonical IDs, status names, config values, and CLI flags.
- [ ] **Step 2: Append a clearly labeled Issue #272 matrix status table** without editing any pre-existing preregistered sentence, criterion, or table. Explain the separate Issue #272 matrix versus the preregistered six-cell family, the TEST-B discrepancy resolution, and why TEST-A/D are unsupported.
- [ ] **Step 3: Document commands, derived states, artifact validity, stale-lock recovery, batch continue-on-error/final exit behavior, and how to add a cell only after config/schema/runner support exists.**
- [ ] **Step 4: Run doc sync and matrix tests.**

### Task 5: Verify, review, commit, and prepare the issue/PR report

**Files:**
- Verify: full repository tests and configured ruff/mypy commands.
- Commit: small logical commits for registry, CLI/docs, and final fixes.

- [ ] **Step 1: Run `pytest`, `ruff check src/ tests/`, `ruff format --check src/ tests/`, and `mypy src/`; record actual output and counts.**
- [ ] **Step 2: Run every available cell smoke execution; do not run unsupported cells.**
- [ ] **Step 3: Review `git diff` for untouched preregistered criteria, no unrelated files, accurate statuses, and atomic/no-write behavior.**
- [ ] **Step 4: Commit the implementation in small logical commits and push the requested branch if credentials permit.**
- [ ] **Step 5: Open a PR with the exact requested title. Use `Fixes #272` only if all acceptance criteria pass; otherwise `Partially addresses #272`.**
- [ ] **Step 6: Write the final Markdown issue comment from actual statuses and command outputs, post with `gh issue comment` only if authenticated, and report a URL only if returned.**
