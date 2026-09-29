# Issue #271: preregistered Adaptive vs Fixed study

## Scope and preregistration

This execution targets only `drone_disturbed/ppo` under TEST-B. The normative
protocol remains [`adaptive_rl_hypothesis.md`](adaptive_rl_hypothesis.md); this
document does not change its hypothesis, thresholds, alpha, horizon, seeds, or
analysis plan. The treatment is specified by
[`TREATMENT_CARD.md`](TREATMENT_CARD.md).

## Before implementation: gap analysis

| Requirement | Before | Evidence |
|---|---|---|
| Train once on nominal parameters; share pre-shift and shock; fork and adapt only between episodes | PARTIAL | `src/adaptive_rl/benchmarking/adaptation_runner.py` implemented the Issue #265 lifecycle, but had no Issue #271 invariant record |
| Exact ten-seed schedule and per-block provenance | PARTIAL | `src/adaptive_rl/protocol/seeds.py` froze the schedule; `--training-seeds` still allowed a subset |
| Recovery endpoint and registered tests/sensitivities | EXISTS | `src/adaptive_rl/protocol/recovery.py`, `src/adaptive_rl/protocol/statistics.py`, `src/adaptive_rl/benchmarking/adaptation_statistics.py` |
| Immutable JSON/CSV plus complete checksummed manifest and clean-tree execution gate | PARTIAL | `src/adaptive_rl/benchmarking/adaptation_artifacts.py` refused JSON/CSV overwrite, but there was no manifest or clean-tree gate |
| Required mutation, integrity, and tidy study CSV checks | PARTIAL | Existing Issue #265 tests covered smoke execution; no manifest validation or invariant mutation tests |
| Full real ten-replicate execution and report | MISSING | `docs/research/issue-265.md` stated that no full run had been collected |

## Implementation

The `adaptive-rl benchmark adaptation --study prereg-v1 --run-id RUN_ID`
entrypoint runs all ten training seeds in preregistered order, rejects subsets
and smoke mode, requires a clean committed tree, and writes into an immutable
run directory. Before training, `study_manifest.json` records the canonical
configuration, the frozen treatment and seed schedule, source revision, and
runtime identity. Its stable JSON representation is SHA-256 hashed. Prereg-v1
also rejects any config or treatment-card hash that differs from the values
frozen here. `--resume` recomputes that identity and accepts only terminal
per-replicate records whose digest, study hash, protocol hash, and seed identity
verify; it never trusts partial training directories. An interrupted seed
without a complete checkpoint is recorded as failed, and remaining unstarted
seeds continue. A present but malformed, stale, or mismatched checkpoint aborts
resume rather than becoming a failed replicate. Final manifest validation also
rejects files added to the completed run directory after its artifact set was
recorded. Replicate checkpoints bind the training files they rely on, and the
runner validates all checkpoint state before resuming any unfinished seed.
Repeating resume after finalization validates and returns the same
immutable result. The JSON stores
the raw trajectories, protocol analysis, seed schedule, outcomes, runtime
invariants, and run status.
The runner returns that same persisted JSON on initial completion and repeated
resume; its artifact paths are stable and relative to the run directory.
The CSV has one row per replicate and arm, with finite-horizon `T_H`, status,
per-episode return vectors, and seed vectors. `manifest.json` checksums every
file in the run directory, including the pre-execution study manifest and
replicate checkpoints; `validate_study_manifest()` detects missing, changed,
malformed, or path-escaping entries. Result files are written atomically and
are not overwritten.

The fixed arm has a prediction-only interface. Exact equality of its initial,
per-episode, and final policy fingerprints establishes a zero weight delta; the
artifact records this as `fixed_parameter_delta_l2: 0.0` and
`fixed_weight_update_count: 0`. Adaptive block logs contain their update seed,
visible episode prefix, finite loss metrics, state fingerprints, and parameter
delta. The runner recomputes invariant checks before writing a successful
replicate.

Right-censored recovery remains the preregistered finite endpoint `T_H = 15`.
Failed replicates are kept in the artifact and CSV; they are omitted pairwise
from the primary analysis and retained in both preregistered imputation bounds.
The statistical artifact includes standard error, t statistic, degrees of
freedom, one-sided p-value, 95% interval, Cohen's `d_z`, exact Wilcoxon and sign
tests, bootstrap interval, and failure-imputation bounds.

## Ambiguity resolutions

* The literal training seeds 31001–31010 identify the ten replicates. Each uses
  the existing SHA-256-derived `pre`, `post`, and `update` phase seeds; the frozen
  schedule fingerprint is recorded and rechecked.
* The shared shock window is the single execution of TEST-B post-shift episodes
  1–5, used by both arms before B5. The shared segment is represented once in
  the raw artifact and referenced by identical per-arm return-vector hashes.
* B5 through B14 use the exact completed post-shift episode prefix required by
  `protocol.adaptation.build_update_batch`. No block runs after episode 15.
* Censoring uses the preregistered finite-horizon value 15, not infinity. This
  differs from the issue prompt's parenthetical `T_H = inf`; the preregistration
  is the specified source of truth.

## Dependency and repository status

No local `ExperimentManifest` or roadmap issue 4/5 implementation was found in
the current checkout, so the study uses the minimal manifest described above.
The checked-out base was `fix/pr-259-review-hardening`, one commit ahead and
seven behind `origin/main`; implementation proceeds on
`feat/issue-271-preregistered-study`. GitHub issue/PR pages were unavailable to
the browsing environment, so the live status of PR #266 and roadmap issues 4/5
could not be independently verified. No dependency on unmerged code is used.

## Execution record

Execution status: **COMPLETE** for this one-cell, ten-replicate study. The
preregistered six-cell family remains **INCONCLUSIVE** because five cells are
not executable in this checkout. These data do not support H1.

The prereg-v1 command below is retained as historical provenance. The runner
rejects it because its frozen deterministic PPO actions are invalid data for
the native PPO update.

```bash
.venv/bin/adaptive-rl benchmark adaptation \
  --config configs/drone_distribution_shift.yaml \
  --output-dir artifacts/issue271 \
  --study prereg-v1 \
  --run-id issue271-prereg-v1-20260929-01
```

The runner enables Torch deterministic algorithms in warn-only mode and cuDNN
deterministic settings for the study, while recording that cross-hardware and
cross-library bitwise reproducibility is not claimed.

## Scientific validity audit finding

The frozen config sets `evaluation.deterministic: true`, and
`evaluate_episode()` passes that setting to PPO while collecting the post-shift
transitions later supplied to PPO's native clipped update. PPO's update assumes
actions were sampled from the recorded behavior distribution. A deterministic
mean action with its Gaussian density recorded as `behavior_log_prob` does not
have that sampling distribution, so the stored rollout is not a valid on-policy
PPO sample. The current preregistration does not define an action-sampling rule
that resolves this conflict. No treatment or analysis change is made here; the
adaptation runner now rejects deterministic PPO collection before training,
including the frozen prereg-v1 configuration. Issue #265's general PPO CLI can
run a machinery check with explicit stochastic collection, but that does not
resolve the conflict with this study's deterministic-evaluation rule. The PPO
treatment must not be described as scientifically validated until a prospective
protocol amendment resolves the action-selection contract.
The recorded prereg-v1 artifact is retained for provenance, but its `COMPLETE`
status describes harness execution only. Its PPO returns are not valid evidence
for the preregistered Adaptive-vs-Fixed claim.

* Run ID: `issue271-prereg-v1-20260929-01`
* Run status: `COMPLETE`; 10 completed, 0 failed, ordered seeds 31001–31010.
* Executed commit: `a5250ffa5efaedaf76ad88c398f9087eb5ef49ca`; clean tree.
* Wall time: approximately 17 minutes 50 seconds (run-directory creation to
  manifest creation); summed PPO training time was 939.38 seconds.
* Hardware/runtime: Linux x86_64, 4 logical CPUs, Python 3.14.7, gymnasium
  1.3.0, stable-baselines3 2.9.0, Torch 2.14.0, NumPy 2.5.3. The platform
  reported no processor model.
* Artifacts: `artifacts/issue271/issue271-prereg-v1-20260929-01/`
  * `adaptive_vs_fixed.json` SHA-256:
    `e7b681d875e8fdfdcc56acada44a34062ac05c92e8881b16c8a3354633b79265`
  * `adaptive_vs_fixed.csv` SHA-256:
    `4d2879f08cf6caa67585576bc67c2cf0d699b617ee8260849c10b7ae7e7ce098`
  * `manifest.json` SHA-256:
    `bb5c3e6f5150a329aecca114dfd47f50366d99ddb1c3e84c515f6fa7d5d480ec`
* Config SHA-256: `0039c298b5048254b2d211cc66967e9d3e1d71275575fcfe28736485c65937b5`.
  Treatment Card SHA-256: `8383e736f02ff31393474b241b06d2dc92b036910c93c26081c1673c92a317e3`.
* Independent audit verified the exact seed list and frozen schedule fingerprint,
  ten passing invariant records, all 100 B5–B14 blocks, 20 tidy CSV rows,
  all manifest-listed checksums, and no `/home/aryan` path in the JSON.

The run predates the follow-up safe-resume addition; it executed from the clean
source commit recorded by the manifest and was not resumed. The artifact's
`execution_command` is normalized to `adaptive-rl` because the entrypoint
records the command's CLI form; the exact shell invocation above includes the
virtual-environment path used to select the installed executable.

## Result from the artifact

For every seed, both arms had `T_H = 0`: 4 replicates were `no_degradation` and
6 were `degradation_below_resolution` in each arm. There were no recovered or
right-censored replicates and no failures. Thus the censoring count was zero;
the finite right-censor endpoint remains `T_H = 15` by protocol.

The paired differences were ten zeros. The artifact reports `N_valid = 10`,
mean difference 0, standard error 0, one-sided paired t statistic 0 on 9 df,
primary p = 0.5, 95% t interval [0, 0], Cohen's `d_z = 0`, exact sign p = 1,
exact Wilcoxon p = 1, and bootstrap interval [0, 0]. The cell is evaluable but
not significant. The six-cell family decision is `INCONCLUSIVE`; no family
claim is made.

## Final acceptance status

| Acceptance area | Status | Evidence |
|---|---|---|
| Ten preregistered PPO replicates and seeds | PASS | JSON seed list, schedule fingerprint, 10 completed records |
| Shared segments, identical fork, fixed lock, ten causal update blocks | PASS | Every replicate's `invariants.all_passed`; mutation coverage in `tests/test_adaptation_invariants.py` |
| Recovery and preregistered paired analyses | PASS | Per-replicate recovery plus `paired_analysis` in JSON |
| Immutable JSON/CSV and validated checksums | PASS | 20-row CSV; `manifest.json`; `validate_study_manifest()` |
| Safe continuation after interruption | PASS in follow-up code | Digest-verified terminal replicate checkpoints; incomplete directories become recorded failures |
| Statistical superiority of Adaptive | FAIL | All observed paired differences are zero; p = 0.5 |
| Six-cell family claim / power ≥ 0.80 | NOT MET | Five cells unavailable; preregistration says power is unquantified at N = 10 |

PR #266 and roadmap issues 4/5 remain unverified live because GitHub was not
available. The run does not depend on them. This is a one-cell study, not
evidence for the broader multi-environment claim.

## Prospective amendment: prereg-v2

Prereg-v1 froze deterministic PPO evaluation, which conflicts with the native
PPO treatment's need for sampled behavior-policy actions. Prereg-v2 resolves
the action-selection contract prospectively: all pre-shift and post-shift PPO
episodes use stochastic policy actions, and those same sampled actions and
behavior log-probabilities form the rollout used by adaptation. The study
schedule, training budget, environment shifts, recovery endpoint, seed schedule,
and statistical analysis remain unchanged. This amendment creates a new
configuration identity; prereg-v1 artifacts and hashes remain untouched.

Run prereg-v2 only with the amended config and a new run ID:

```bash
.venv/bin/adaptive-rl benchmark adaptation \
  --config configs/drone_distribution_shift_prereg_v2.yaml \
  --output-dir artifacts/issue271 \
  --study prereg-v2 \
  --run-id issue271-prereg-v2-YYYYMMDD-01
```

The v2 config sets `evaluation.deterministic: false` and has canonical
SHA-256 `be12f2837dc0fbf00a752104e1649b3d5b41d852de2574f29a50a5d155dbafb8`.
The runner verifies this hash before training, retains the common treatment-card
hash, and stamps `adaptive-vs-fixed/prereg-v2` into both study manifests.
Prereg-v1 remains available for verifying its prior immutable artifact but
cannot start a new run because deterministic PPO collection is rejected.
