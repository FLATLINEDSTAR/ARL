# Issue #272 Experiment Cell Matrix Design

## Purpose

Add a truthful six-slot registry and CLI for the Issue #272 PPO/SAC by TEST-A/TEST-B/TEST-D benchmark request. The registry must report executable cells from existing code and configuration, and must not represent planned, unsupported, running, or incomplete work as completed research.

## Scope and scientific boundary

The matrix IDs are `CELL_PPO_TEST_A`, `CELL_PPO_TEST_B`, `CELL_PPO_TEST_D`, `CELL_SAC_TEST_A`, `CELL_SAC_TEST_B`, and `CELL_SAC_TEST_D`. This is an Issue #272 benchmark matrix. It does not replace or amend the preregistered primary family in `adaptive_rl_hypothesis.md` §17, which names `gridworld/ppo`, `traffic_signal/ppo`, `drone_disturbed/ppo`, `drone_disturbed/sac`, `navigation_2d/ppo`, and `navigation_2d/sac`.

The current `configs/drone_distribution_shift.yaml` has nominal TRAIN values of 8 obstacles and 0.5 m/s steady wind, and configures only TEST-B at 12 obstacles, 4.0 m/s wind, and gust sigma 0.6. The current Pydantic configuration schema also freezes the adaptation scenario to TEST-B. The Issue #272 TEST-A and TEST-D descriptions are not represented by scenarios in this YAML. In the preregistration, TEST-A is density-only and TEST-D is described as a hidden disturbance of 6.0; it does not define the issue's sensor-noise/wind-gust/dynamic-obstacle combination. Therefore A and D are `UNSUPPORTED` with this baseline and must not receive guessed parameters.

The existing adaptation runner and smoke tests support PPO and SAC for TEST-B. PPO requires stochastic action sampling for adaptation data; the matrix dispatcher must request stochastic collection for PPO. The immutable Issue #271 study entrypoint remains distinct and has its own frozen-config/clean-tree gates; the matrix must not claim its issue-specific non-study outputs satisfy a full preregistered study. The shared protocol source supplies ten training seeds, ten replicates, derived pre/post/update schedules, 15 evaluation episodes per phase, and the configured 60,000 timestep budget.

## Architecture

Create `src/adaptive_rl/experiments/matrix.py` with string enums, frozen `CellSpec`, cell ID normalization, a registry, status/result inspection, atomic marker/result helpers, and single/batch dispatch. Cell status will be derived from runner/config capability, validated artifacts, and a PID/timestamp marker. A live PID is `RUNNING`; a dead or malformed marker is stale and recoverable. `COMPLETED` requires a schema-valid result artifact whose expected paths and checksums verify. The only initial executable slots are PPO/TEST-B and SAC/TEST-B; remaining cells are `UNSUPPORTED` with explicit reasons.

The dispatcher will clone the source config, use the canonical TEST-B values and the shared preregistered seeds/budget, select the requested algorithm through the existing adaptation runner interface, and avoid its immutable study mode. It will preflight unsupported states before creating output directories. Marker and matrix summary files will be written atomically. Batch mode runs each available cell sequentially, reports progress and a final status table, continues after cell errors, and exits non-zero if any cell fails.

Extend `src/adaptive_rl/cli.py` with `adaptive-rl benchmark matrix --list`, `--cell CELL_ID`, and `--all`. Invalid and unsupported requests receive escaped, concise diagnostics without tracebacks. Append a matrix status/usage note to the preregistration without modifying existing criteria; expand `docs/BENCHMARKS.md` with state derivation, command examples, and the path for adding a future scenario only after it is defined and executable.

## Validation

Add fast registry, parser, invariant, status derivation, stale marker, unsupported fail-fast/no-write, atomic write, result integrity, and CLI tests. Mock the existing training runner for CLI orchestration tests. Run one-replicate tiny smoke execution only for cells the registry reports as `AVAILABLE`; do not smoke-test unsupported cells. Preserve existing adaptation and protocol regression tests.

## Explicit decisions

* Canonical IDs use underscores and uppercase algorithm/environment names; case-insensitive and hyphen aliases may normalize, while malformed IDs get a clear parse error.
* TEST-B follows the YAML/preregistration values (12 obstacles, 4.0 m/s, gust sigma 0.6); nominal TRAIN is 8 obstacles and 0.5 m/s.
* Batch execution means every currently `AVAILABLE` cell; planned/unsupported cells are reported and skipped.
* Roadmap Issues 4/5, PR #266, and issue #272 live status require checking via local history or authenticated GitHub tooling; implementation must not depend on an unmerged PR.

