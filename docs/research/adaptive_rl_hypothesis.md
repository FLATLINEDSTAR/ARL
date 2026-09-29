# AdaptiveRL Hypothesis and Evaluation Protocol

**PROTOCOL_VERSION = "2.0"** · Pinned repository commit: `faefc5c8e4a39bbcc728d73b1c9855c8e9c5386f`

## 1. Purpose and Status

This document is the normative, pre-specified experimental contract for evaluating a proposed online environment-shift adaptation treatment ("Adaptive") against a matched frozen-policy baseline ("Fixed"). The companion package `src/adaptive_rl/protocol/` is the executable mirror of the frozen constants, the seed schedule, the recovery endpoint, and the statistical analysis plan defined here; `tests/test_protocol_doc_sync.py` fails if this document and that package drift apart.

> [!WARNING]
> **Implementation Status**
>
> **[VERIFIED-CODE]** `src/adaptive_rl/benchmarking/adaptation_runner.py` now implements the Issue #265 drone TEST-B lifecycle: train once on nominal parameters, share the pre-shift and shock episodes, fork independent PPO/SAC policies, update only Adaptive between episodes B5–B14, and compute recovery from the paired trajectories. `tests/test_adaptation_smoke.py` exercises the CI-sized PPO path; both PPO and SAC real-environment smoke paths have been executed.
>
> **[PROTOCOL REQUIREMENT — NOT IMPLEMENTED]** Four non-drone primary cells do not have an Issue #265 shift environment/configuration in this checkout. They remain inconclusive; the six-cell family claim cannot be evaluated from the drone cell alone.
>
> **[NO RESEARCH RESULT]** No full ten-replicate scientific benchmark has been collected. Smoke execution validates software behavior only and does not establish empirical superiority or statistical significance.
>
> **PR #168 / Issue #98** introduced this protocol and its executable constants/seed/recovery/statistics mirror. Issue #265 adds the drone TEST-B harness. Nothing here claims a full research result or empirical validation.

**Status tags used throughout** (a value is what its tag says, nothing more):

| Tag | Meaning |
|---|---|
| `[VERIFIED-CODE]` | Behavior verified in the current repository source; frozen scientific values remain pinned to the protocol version |
| `[SMOKE-VALIDATED]` | CI-sized or one-off smoke execution verified software behavior only; not research evidence |
| `[VERIFIED-CONFIG]` | Value appears literally in the cited config file at the pinned commit |
| `[PROTOCOL REQUIREMENT — NOT IMPLEMENTED]` | Frozen requirement for a future harness; not present in the repository today |
| `[FUTURE PROTOCOL VALUE]` | Pre-registered target that must be configured before that condition is executable |
| `[FUTURE DESIGN DECISION]` | Must be frozen in a signed-off artifact before that cell collects data |

## 2. Research Question

Does an agent equipped with the online adaptation treatment recover performance faster after a distribution shift than a matched fixed-policy baseline (PPO or SAC) trained on the same nominal distribution?

## 3. Hypotheses and Estimand

**Time axis**: completed post-shift episode index. All recovery definitions use this single canonical axis.

Let $P_{pre}$, $P_0$, $P(t)$, $R(t)$ be defined in §6. Let $\tau$ be the earliest persistence-confirmed recovery episode defined in §6.4.

$$T_H = \begin{cases} 0 & \text{status} \in \{\texttt{no\_degradation}, \texttt{degradation\_below\_resolution}\} \\ \tau & \text{status} = \texttt{recovered},\ \tau \in \{6, \dots, 13\} \\ H & \text{status} = \texttt{right\_censored} \end{cases}$$

with $H = 15$ completed post-shift episodes. **$H = 15$ is a protocol-level pre-registered horizon for Issue #98 and is not an ARL-wide evaluation default.**

**Domain of $T_H$** (frozen): $T_H \in \{0\} \cup \{6, 7, \dots, 13\} \cup \{15\}$. Values 1–5 and 14 are unreachable: $P(5) \equiv P_0$ makes $R(5) = 0$ (so $\tau$ can never be 5), and $\tau = 14$ would require a window ending at episode 16, beyond $H$.

The estimand is the **finite-horizon** recovery time $T_H$, not an unobserved true recovery time beyond $H$. Right-censored runs contribute exactly $H$ (no clipping, no extrapolation).

For replicate $i$: $D_i = T_{H,i}(\text{Adaptive}) - T_{H,i}(\text{Fixed})$. **A negative $D_i$ favors Adaptive** (lower is better) everywhere in this protocol. The primary estimand is $\mu_D = \mathbb{E}[D_i]$ over the training-seed distribution.

### H0 — Null
$\mu_D \ge 0$. The treatment does not reduce horizon-truncated recovery time.

### H1 — Alternative
$\mu_D < 0$. The treatment achieves strictly lower horizon-truncated recovery time.

The primary test is one-sided, consistent with H1 (§18.2).

## 4. Treatment Arms and Experimental Design

### 4.1 Factors

* **Algorithm**: PPO or SAC — a separate factor from treatment.
* **Treatment**: Adaptive vs Fixed.
* Within every environment × algorithm cell, base algorithm, hyperparameters, training budget, checkpoint, and all evaluation seeds are **identical** between arms. The sole intervention is the online adaptation mechanism and its compute overhead during post-shift episodes 6–15.

### 4.2 Train-once, clone, fork design

**[SMOKE-VALIDATED]** The drone TEST-B runner executes each replicate as one shared segment followed by two arm segments:

```
TRAIN (seed = training_seed(i))          # one training run per replicate
  → freeze + fingerprint (§16)
  → SHARED pre-shift evaluation          # K_pre = 15 nominal episodes
  → SHIFT INTRODUCTION                   # no policy change at this step
  → SHARED shock window                  # post-shift episodes 1-5, frozen policy
  → FORK (after episode 5, before episode 6):
        Arm A (Adaptive): run update block B5 (§5), then episodes 6..15 with blocks B6..B14
        Arm F (Fixed):    clone of the same checkpoint; no update ever; episodes 6..15
```

Consequences (required invariants, reported per replicate):

1. $P_{pre}$ and $P_0$ are computed **once** in the shared segment; both arms use the same stored numbers by construction. The Adaptive arm performs **no update** during pre-shift evaluation or during post-shift episodes 1–5.
2. The Fixed arm never updates its weights at any point — not during training-prolonging hooks, not during evaluation, not after any episode.
3. Because training happens once per replicate, bit-level determinism of retraining is not required for the arm contrast; the same trained weights serve both arms.

### 4.3 Observation pipeline

No running normalization statistics exist in the repository (no VecNormalize or equivalent is configured) `[VERIFIED-CODE]`. The Adaptive arm must not introduce observation-normalization state that differs between arms; any adaptation-only state (replay buffers, optimizers, schedules) lives in the Adaptive arm, is logged per block (§5), and never affects Fixed.

## 5. Update-Block Schedule (Frozen)

### 5.1 Block schedule

**[SMOKE-VALIDATED]** Exactly `N_update = 10` update blocks execute strictly between the termination of episode $k$ and the reset of episode $k+1$, for $k = 5, 6, \dots, 14$ (block identifiers $B_5, \dots, B_{14}$):

| Block | Executes between | Visible data (frozen) |
|---|---|---|
| $B_5$ (first block) | episode-5 termination ↔ episode-6 reset | post-shift episodes 1–5 only |
| $B_k$, $k = 6..14$ | episode-$k$ termination ↔ episode-$(k+1)$ reset | all post-shift episodes $1..k$ (cumulative) |
| — | after episode 15 | **no block**; policy unchanged for reporting |

Rules:

1. **No update while an episode is running.** The policy is constant within an episode; $B_5$ is strictly after the shock window completes, so $P_0$ is pre-update for both arms by construction (§4.2).
2. The Fixed arm runs the same episode boundaries with **zero** blocks.
3. Adaptation-side randomness (minibatch order, exploration noise if any) uses the block's derived seed from §14.2 (`update` phase). Env interaction during episodes 6–15 uses the episode's `post` seed only.
4. Update-data visibility is limited to post-shift episodes; pre-shift nominal episodes are never replayed into adaptation.
5. Blocks $B_5..B_{14}$ satisfy `N_update = 10` in `src/adaptive_rl/protocol/constants.py`.

### 5.2 Ordering contract (crash-safety)

An update block completes (or fails) **before** the next episode reset. A failed block is a treatment failure (§19), never a silent skip: the harness must not continue episodes 6–15 with a partially applied update.

### 5.3 What is frozen vs. what must still be frozen before execution

* **Frozen now**: schedule of §5.1, data visibility, freeze windows, identical-seed rule, logging requirements, all constants in §24.
* **[SMOKE-VALIDATED]** The adaptation algorithm, buffer construction, native PPO/SAC update details, inherited hyperparameters, and failure behavior are frozen in `docs/research/TREATMENT_CARD.md`. Artifacts record the Card SHA-256. Smoke tests are not evidence of outcome-based tuning or scientific validity.

## 6. Operational Definition of Recovery

Executable implementation: `src/adaptive_rl/protocol/recovery.py`; hand-verified examples: §7; tests: `tests/test_protocol_recovery.py`.

### 6.1 Baselines

* **$P_{pre}$** (pre-shift performance): mean episodic return over the `K_pre = 15` pre-shift episodes (seeds §14.2).
* **$P_0$** (immediate post-shift performance): mean episodic return over post-shift episodes 1–5, i.e. `P0 = mean(post_shift_returns[0:5])`. In `compute_recovery()` this window is derived from the post-shift vector itself, so a mismatched "shock" array cannot exist.

### 6.2 Trailing-window performance

**$P(t)$**: mean return over the causal trailing window of the `WINDOW = 5` most recently completed post-shift episodes ending at episode $t$; defined for $t = 5..15$ (11 values). No future episode is ever used to declare recovery at an earlier index. $P(5) \equiv P_0$ identically.

### 6.3 Recovery ratio and degradation gate

$$R(t) = \frac{P(t) - P_0}{P_{pre} - P_0} \qquad \text{(reported), evaluated as the predicate below (normative)}$$

* **Degradation** $\delta = P_{pre} - P_0$.
* **Minimum measurable degradation**: `delta_min = 2 * SE(delta)` where `SE(delta) = sqrt(var(pre)/15 + var(shock)/5)` (unbiased ddof=1 variances; constant `MIN_DEGRADATION_SE_MULTIPLIER = 2.0`).
* If $\delta \le 0$: status `no_degradation`, recovery not required, $T_H = 0$, $\tau$ undefined, $R(t)$ undefined (all `None`). This is a protocol convention, not evidence of instantaneous adaptation (limitation recorded in §22).
* If $0 < \delta <$ `delta_min`: status `degradation_below_resolution`, recovery not required, $T_H = 0$, label preserved in reporting.
* If $\delta \ge$ `delta_min`: **recovery required**; $R(t)$ defined.

Edge cases (frozen):

1. $R(t) < 0$ is permitted (performance below the post-shift floor) — never clipped.
2. $R(t) > 1$ is permitted (performance above pre-shift) — never clipped.
3. $P_{pre} = P_0$ (zero denominator) is exactly case 1 above.
4. Float64 throughout; **no epsilon** anywhere.

### 6.4 Threshold, persistence, $\tau$

**Threshold predicate (normative, division-free):**

```
10.0 * (p_t - p0) >= 9.0 * degradation
```

(mathematically $R(t) \ge 0.9$ with $0.9 = 9/10$; cross-multiplication fixes the float64 boundary — see §7 case 5/6).

**Persistence**: eligible starts are $\tau \in \{6, \dots, 13\}$ (constant `range` in `recovery.py`: `last_start = HORIZON - PERSISTENCE + 1`). $\tau$ is the **smallest** eligible $t$ such that the predicate holds at all three windows $t, t{+}1, t{+}2$ (all within $H$). The start at $t = 5$ is excluded structurally ($R(5) = 0 < 0.9$ whenever recovery is required). If no such $t$ exists: status `right_censored`, $\tau$ undefined, $T_H = H = 15$.

### 6.5 Status vocabulary (frozen)

`recovered` (τ recorded, $T_H = \tau$), `right_censored` ($T_H = 15$), `no_degradation` ($T_H = 0$), `degradation_below_resolution` ($T_H = 0$).

The two $T_H = 0$ statuses are **arm-invariant with respect to pre-treatment data**: both arms share the same pre-shift segment and shock window by construction (§4.2), so a replicate with `no_degradation` contributes $D_i = 0$ identically — it cannot favor either arm. Reporting must still distinguish the statuses so $T_H = 0$ is never conflated with observed fast recovery.

## 7. Worked Examples (Edge Cases)

Each row is executed verbatim by `tests/test_protocol_recovery.py` (same arrays, same expectations). Returns are per-episode episodic returns; `pre` is 15 values; `post` is episodes 1–15.

| # | pre (15) | post (15) | Expected |
|---|---|---|---|
| 1a | all `10.0` | all `10.0` | `no_degradation`, δ=0, $T_H=0$, R undefined |
| 1b | all `8.0` | all `10.0` | `no_degradation`, δ=−2, $T_H=0$ |
| 2 | all `10.0` | `[5]*5 + [30]*10` | `recovered`, τ=6, $T_H=6$; `P(5)=P0`, R(5)=0 |
| 3 | all `10.0` | `[5]*12 + [30]*3` | `recovered`, τ=13, $T_H=13$ |
| 4 | all `10.0` | all `5.0` | `right_censored`, $T_H=15$ |
| 5 | all `10.0` | `[0]*5 + [9]*10` | τ=10; R(10)=R(11)=R(12)=0.9 exactly; predicate True |
| 6 | predicate | — | `10*(9.0−0)>=9*10` True; `8.999999999999998` False; `9.000000000000002` True |
| 7 | all `20.0` | `[10]*5 + [5]*10` | R(6) = −0.1 (negative permitted); `right_censored` |
| 8 | all `10.0` | `[5]*5 + [20]*10` | R(15) = 3.0 (> 1 permitted) |
| 9 | all `10.0` | all `10.0` | zero denominator ⇒ case 1a |
| 10 | all `10.0` | `[7,8,9,10,11] + [5]*10` | δ=1 < δ_min=√2≈1.4142 ⇒ `degradation_below_resolution`, $T_H=0$ |
| 10b | all `10.0` | `[8.8,9,9,9,9.2] + [5]*10` | δ=1 ≥ δ_min≈0.1265 ⇒ recovery required |
| 11 | all `10.0` | `[0]*5 + [45,−50] + [100]*8` | t=6 passes, t=7 breaks ⇒ first triple at **τ=8** |
| 12 | all `10.0` | `[0]*5 + [40,50,0,−50] + [100]*6` | passes at 7,8 broken at 9 ⇒ **τ=10** |

## 8. Variables

### 8.1 Independent variables
| Variable | Levels |
|---|---|
| Algorithm | PPO, SAC |
| Treatment | Adaptive, Fixed |
| Environment | `gridworld`, `navigation_2d`, `traffic_signal`, `drone_disturbed` |
| Shift condition | Mild, Moderate (primary: drone = TEST-B, §12), Severe |

### 8.2 Dependent variables
* **Primary**: $T_H$ (lower is better; §3, §6).
* **Secondary** (report, no confirmatory claim): final-window episodic return over episodes 11–15 — equal to $P(15)$ by construction (asserted in tests); final-window success rate over episodes 11–15 with `compute_rate` semantics `[VERIFIED-CODE src/adaptive_rl/metrics.py:572]` (`None` excluded from numerator and denominator, `None` when the window has no defined outcome); full per-episode return trajectory (descriptive).

### 8.3 Controlled variables
Pre-registered training budget and config pinned to §15; final-checkpoint rule (§16); identical derived seeds across arms (§14); deterministic evaluation actions (`evaluation_deterministic` in the shift config `[VERIFIED-CONFIG]`).

## 9. Baseline

**Fixed arm**: trained on the nominal distribution; policy frozen and fingerprinted immediately after training; evaluated under shift with weights locked at all times (`src/adaptive_rl/benchmarking/adaptation_runner.py` verifies the Fixed fingerprint against the frozen fingerprint `[SMOKE-VALIDATED]`).

## 10. Evaluation Environments

At the pinned commit `[VERIFIED-CODE src/adaptive_rl/environments/__init__.py:60-104]`:

| Cell id | Config env name | Action space | Note |
|---|---|---|---|
| `gridworld` | `gridworld` | Discrete | |
| `navigation_2d` | `navigation` | Continuous | `navigation` is registered as the **same class** (`ContinuousNavigation2DEnv`) as `navigation_2d` |
| `traffic_signal` | `traffic` | Discrete | `traffic` is registered as the **same class** (`TrafficSignalEnv`) as `traffic_signal` |
| `drone_disturbed` | `drone_disturbed` | Continuous | `max_steps: 300` `[VERIFIED-CONFIG configs/drone_distribution_shift.yaml:42]` |

## 11. Training Distribution

Issue #265 training uses only `environment.parameters` from `configs/drone_distribution_shift.yaml`; the distinct TEST-B `shift_parameters` are merged by the runner only after training and the shared pre-shift evaluation. Derived update/evaluation seeds are validated against the frozen configuration seed pools. The runner's separate configuration fields make the TEST-B values unavailable to the training environment constructor `[SMOKE-VALIDATED]`.

**[SMOKE-VALIDATED]** Each replicate trains with the selected preregistered `training_seed`; the trainer receives a per-replicate config copy and the source config is unchanged.

## 12. Distribution Shifts

`[FUTURE PROTOCOL VALUE]` entries are pre-registered targets that require configuration before that condition is executable. Mild/Severe conditions are **outside** the primary family (§17) and are registered only to prevent silent value shopping later.

### 12.1 GridWorld — `num_obstacles`
Sources: `configs/gridworld_ppo.yaml` (nominal), `configs/generalization_gridworld.yaml` (mild), `configs/benchmark_planners_gridworld.yaml` (moderate as design reference — different benchmark purpose).

| Condition | num_obstacles | Status |
|---|---|---|
| Nominal | 3 | `[VERIFIED-CONFIG]` `gridworld_ppo.yaml:21` |
| Mild | 4 | `[VERIFIED-CONFIG]` `generalization_gridworld.yaml:20` |
| Moderate (primary) | 6 | `[VERIFIED-CONFIG value; FUTURE PROTOCOL VALUE as a shift config]` — appears only in `benchmark_planners_gridworld.yaml:19` |
| Severe | 8 | `[FUTURE PROTOCOL VALUE]` |

No dedicated gridworld shift config exists at the pinned commit: **the gridworld cell is not executable today.**

### 12.2 Navigation 2D — `lidar_noise` and `num_obstacles`

* `lidar_noise` does **not** appear in any source file or config at the pinned commit — `[VERIFIED-ABSENT]`. The doc makes no claim that it exists.
* The observed density mechanism is `num_obstacles`.

| Condition | num_obstacles | Status |
|---|---|---|
| Nominal | 5 | `[VERIFIED-CONFIG]` `navigation.yaml:19` |
| Mild (design reference) | 4 | `[VERIFIED-CONFIG]` `generalization_navigation.yaml:20` (SAC config, different benchmark purpose) |
| Moderate (primary) | 8 | `[FUTURE PROTOCOL VALUE]` |
| Severe | 12 | `[FUTURE PROTOCOL VALUE]` |

**The navigation cells are not executable today** (moderate value not configured; Adaptive harness absent).

### 12.3 Traffic Signal — `arrival_rates` (4-lane tuple)
Source: `configs/traffic_ppo.yaml:19`.

| Condition | arrival_rates | Status |
|---|---|---|
| Nominal | (0.35, 0.35, 0.25, 0.25) | `[VERIFIED-CONFIG]` |
| Mild | (0.5, 0.5, 0.5, 0.5) | `[FUTURE PROTOCOL VALUE]` |
| Moderate (primary) | (0.7, 0.7, 0.7, 0.7) | `[FUTURE PROTOCOL VALUE]` |
| Severe | (1.0, 1.0, 1.0, 1.0) | `[FUTURE PROTOCOL VALUE]` |

No dedicated traffic shift config exists: **the traffic cell is not executable today.**

### 12.4 Drone Disturbed — `wind_speed` (m/s) and `gust_sigma`
Source: `configs/drone_distribution_shift.yaml` (benchmark v1.1). Physical meaning (config header `[VERIFIED-CONFIG]`): `wind_speed` is steady wind (force via `linear_damping = 0.05`, `max_acceleration = 4.0 m/s²`); `gust_sigma` is OU gust volatility with stationary per-axis std ≈ `sigma/sqrt(2*theta)`.

| Condition | wind_speed | gust_sigma | num_obstacles | Status |
|---|---|---|---|---|
| Nominal (TRAIN) | 0.5 | 0.15 | 8 | `[VERIFIED-CONFIG]` TRAIN scenario |
| Mild | 2.0 | 0.3 | 10 | `[FUTURE PROTOCOL VALUE]` |
| **Moderate (primary) = TEST-B** | **4.0** | **0.6** | **12** | `[VERIFIED-CONFIG]` TEST-B (`drone_distribution_shift.yaml:94-103`) |
| Moderate (secondary) = TEST-C | 4.0 | 0.6 | 8 static + 4 moving | `[VERIFIED-CONFIG]` TEST-C (`:104-113`); **not** a primary cell |
| Severe | 8.0 | 1.2 | 12 | `[FUTURE PROTOCOL VALUE]` |

**The drone "moderate" condition is TEST-B** (obstacle-density + moderate-wind compound shift). TEST-A (density only), TEST-C (dynamic obstacles + wind), and TEST-D (hidden disturbance 6.0) are registered in the config but are **not** primary cells; TEST-C is reported only as secondary/exploratory. This resolves the ambiguity between "moderate wind" and "moderate shift".

> Do not confuse the benchmark's `recovery_event_threshold: 0.7` `[VERIFIED-CONFIG :64]` — a per-event completion threshold of the existing disturbance-recovery summary — with the protocol's recovery threshold `0.9` on $R(t)$ (§6.4). They are different quantities on different axes.

## 13. Evaluation Phase Sequence

Mandatory order; any deviation is a protocol violation.

```
Step 1 — TRAIN
  Train once on nominal parameters with config.seed := training_seed(i) (§11).
  Budget per §15. No test/shift parameters visible to training.

Step 2 — FREEZE + FINGERPRINT
  Final training checkpoint (§16). Fingerprint recorded (shift_runner.py:181).
  Weights immutable from here except through Adaptive blocks B5..B14.

Step 3 — SHARED PRE-SHIFT EVALUATION  [K_pre = 15 episodes]
  Deterministic policy; reset(seed = derive_seed(i, "pre", j)), j = 1..15 (§14).
  P_pre = mean of the 15 returns.

Step 4 — SHIFT INTRODUCTION
  Environment parameters → moderate condition (§12). No policy update here.

Step 5 — SHARED SHOCK WINDOW  [post-shift episodes 1..5]
  Frozen policy (no update possible by construction — single shared execution).
  reset(seed = derive_seed(i, "post", j)), j = 1..5.
  P0 = mean of these 5 returns (§6.1).

Step 6 — FORK (after episode 5, before episode 6)
  Clone the frozen checkpoint into both arms.
  Adaptive: execute block B5 (data = episodes 1..5), THEN reset episode 6.
  Fixed:    no block; reset episode 6 with the same seed.

Step 7 — ARM SEGMENT  [post-shift episodes 6..15]
  reset(seed = derive_seed(i, "post", j)), j = 6..15 — SAME seeds in both arms.
  Adaptive executes block B_k after episode k for k = 6..14 (§5.1).
  Fixed executes no block, ever.
  Policy constant within each episode.

Step 8 — CAUSAL RECOVERY MEASUREMENT
  Compute P(t), predicate, persistence, tau, T_H, status (§6) from the shared
  pre-shift returns + per-arm post-shift returns. Derived by
  adaptive_rl.protocol.recovery, not ad-hoc scripts.
```

**Common-random-number property** `[SMOKE-VALIDATED]`: episodes are reseeded at every reset, and drone gust noise is drawn from the seeded environment RNG once per step. Both arms therefore receive the same exogenous noise stream for episode $j$ up to the point where episode length diverges (the number of RNG draws per episode is action-dependent). CRN alignment is exact for the shared segment and partial-by-construction for the arm segment; it is a variance-reduction property, not a claim of identical trajectories.

## 14. Seed Protocol

Executable implementation: `src/adaptive_rl/protocol/seed_schedule.py`; tests: `tests/test_protocol_seed_schedule.py`. The domain API names `eval_pre` and `eval_post` map to the frozen payload tokens `pre` and `post`; `train` is the unchanged replicate master seed. Serialized schedule keys and the preregistered fingerprint remain unchanged. See [`SEED_POLICY.md`](SEED_POLICY.md) for the API mapping, shock-window assignment, validation, manifest fields, and worked example.

### 14.1 Training seeds (replicate identity)

```
TRAINING_SEEDS = [31001, 31002, 31003, 31004, 31005, 31006, 31007, 31008, 31009, 31010]
```

`PLANNED_N = 10` replicates per cell. This count is a pre-registered compute budget, **not** a powered sample size (power undetermined — §25). The integers are `< 2^31`, so they are valid for `np.random.seed` (requires `< 2^32`) `[VERIFIED-CODE src/adaptive_rl/training/trainer.py:120-126]`, `torch.manual_seed`, gymnasium, and Python `random`. They are disjoint from every config seed pool (§14.4).

### 14.2 Derived episode/block seeds — exact specification

**[VERIFIED-CODE]** There are 400 derived values per schedule (10 seeds × (15 pre + 15 post + 10 update)). The derivation is SHA-256, **not** Python's `hash()` (which is salted per process by `PYTHONHASHSEED`):

```
payload  = f"{training_seed}|{phase}|{index}".encode("utf-8")
digest   = hashlib.sha256(payload).digest()
value    = int.from_bytes(digest[:4], byteorder="big", signed=False) & 0x7FFFFFFF
```

* `training_seed` ∈ `TRAINING_SEEDS` (any other value raises).
* Derivation payload tokens are `"pre"`, `"post"`, and `"update"`. The API accepts `"eval_pre"` and `"eval_post"` as descriptive aliases for the first two tokens; `"train"` at index 0 returns the unchanged `training_seed`.
* Index domains (frozen, 1-based for episodes, 0-based for blocks): `pre`: 1..15; `post`: 1..15; `update`: 0..9.
* Range: `0 <= value <= SEED_VALUE_MAX` where `SEED_VALUE_MAX = 0x7FFFFFFF` — valid for every RNG the repo seeds.
* Example regression value: `derive_seed(31001, "pre", 1) = 1280372827`.

`validate_schedule()` enforces, and the test suite checks, all of:

1. **Global uniqueness** across all 400 values (any duplicate raises; collisions cannot pass silently).
2. **Disjointness** from `TRAINING_SEEDS`.
3. **Disjointness** from config pools `[1000..1014]` (TRAIN) and `[2000..2014]` (TEST).
4. **Pre/post/update disjointness** within each replicate.
5. Value range.

### 14.3 Schedule fingerprint (arm-identity check)

`schedule_fingerprint(schedule)` = SHA-256 hex over the canonical serialization of the whole schedule. The frozen fingerprint is:

```
schedule_fingerprint = 65939167572731c99599c382ac50cf3fddbba3cf758305764392f13b2e4efa67
```

The research artifact and each replicate record this fingerprint, shared by both arms; a mismatch against this document invalidates the replicate. Any change to seeds, indices, or phases changes the fingerprint and therefore fails `tests/test_protocol_doc_sync.py`.

### 14.4 RNG initialization and train/test disjointness

Every pre/post episode reset uses `reset(seed = derived_value)` and every update block uses its corresponding derived update seed `[SMOKE-VALIDATED]`. Training-side seeding remains `random.seed` / `np.random.seed` / `torch.manual_seed` / CUDA seeds in `trainer.py`. Disjointness of the derived schedule from training seeds and config pools is enforced by `validate_schedule()` (§14.2). Determinism is **not** claimed across hardware/library versions (§25). Research evaluation rejects the generic sequential `base_seed + episode` fallback; see [`SEED_POLICY.md`](SEED_POLICY.md).

## 15. Training Budgets (Pinned)

Pinned to `faefc5c8e4a39bbcc728d73b1c9855c8e9c5386f`. Same budget for both arms within every cell (arms share one training run anyway, §4.2).

| Cell | Config File | total_timesteps | Status |
|---|---|---|---|
| `gridworld/ppo` | `configs/gridworld_ppo.yaml` | 5,000 | `[VERIFIED-CONFIG]` |
| `traffic_signal/ppo` | `configs/traffic_ppo.yaml` | 10,000 | `[VERIFIED-CONFIG]` |
| `drone_disturbed/ppo` | `configs/drone_distribution_shift.yaml` | 60,000 | `[VERIFIED-CONFIG]` |
| `drone_disturbed/sac` | `configs/drone_disturbed_sac.yaml` | 100,000 | `[VERIFIED-CONFIG]` — the config **exists** at the pinned commit (`name: drone_disturbed_sac`, `algorithm.name: sac`, `environment.name: drone_disturbed`); earlier revisions of this document wrongly called it future work |
| `navigation_2d/ppo` | `configs/navigation.yaml` | 100,000 | `[VERIFIED-CONFIG]` |
| `navigation_2d/sac` | — | — | `[FUTURE CONFIG]` + `[FUTURE PROTOCOL VALUE]` (no SAC navigation config exists) |

> [!NOTE]
> These values are protocol-frozen at the pinned commit. If a YAML file later changes, this table remains the authoritative pre-registered budget.

## 16. Checkpoint Selection

The deterministic pre-declared rule is the **final training checkpoint** at the end of the budget. Post-shift performance never selects checkpoints. A policy fingerprint is computed before evaluation and re-verified per scenario (`shift_runner.py:181,204-205`) `[VERIFIED-CODE]`. In the fork design (§4.2) the fingerprint is captured once at freeze and recorded again at the start of each arm segment; the Fixed arm's fingerprint must be unchanged at experiment end.

## 17. Primary Endpoint and Family

**Primary estimand**: $\mu_D$ in $T_H$ under the moderate condition (drone primary cell = TEST-B, §12.4).

**Preregistered primary family (6 cells, fixed order)** — this is the complete family; no cell may be added or dropped after results are seen:

| # | Cell id | Executable today? |
|---|---|---|
| 1 | `gridworld/ppo` | No — no shift config; moderate is design-reference only |
| 2 | `traffic_signal/ppo` | No — moderate `[FUTURE PROTOCOL VALUE]` |
| 3 | `drone_disturbed/ppo` | Yes — runner and smoke path implemented; full ten-replicate data not collected |
| 4 | `drone_disturbed/sac` | Yes — runner and smoke path implemented; full ten-replicate data not collected |
| 5 | `navigation_2d/ppo` | No — moderate `[FUTURE PROTOCOL VALUE]`; harness absent |
| 6 | `navigation_2d/sac` | No — config absent; harness absent |

### 17.1 Evaluability and the family decision rule (frozen)

* A cell yields a primary p-value only with `MIN_VALID_N = 8` **pairwise-complete** replicates (both arms have an evaluable $T_H$) out of `PLANNED_N = 10` (§18.1).
* The family claim is an **intersection-union test (IUT)**: `SUPPORTED` iff **all six** cells are evaluable **and** every one-sided p-value is `< ALPHA` with `ALPHA = 0.05`.
* **Any** non-evaluable cell ⇒ family decision `INCONCLUSIVE` (the conjunction can be neither supported nor refuted). Partial results are reported descriptively with per-cell Holm-adjusted p-values (§18.5), never as a family claim.
* All evaluable and at least one cell `p >= ALPHA` ⇒ `NOT_SUPPORTED`.

This rule is implemented by `decide_family()` and tested for every branch in `tests/test_protocol_statistics.py`.

## 18. Statistical Analysis Plan

Executable implementation: `src/adaptive_rl/protocol/statistics.py` (pure Python — the repository has **no** scipy dependency; Student-t probabilities come from the regularized incomplete beta continued fraction). Validated against published t-table quantiles and stdlib `statistics` arithmetic in `tests/test_protocol_statistics.py`.

### 18.1 Experimental unit

The independent unit is the **training run / seed** (n = 10 planned, ≥ 8 evaluable). Episodes are repeated observations nested within that unit and are never treated as independent replicates. Pairing is by training seed: both arms of replicate $i$ share the training run and all seeds (§4.2).

### 18.2 Primary test

**One-sample (paired) t-test** on the pairwise-complete differences $D_i = T_{H,i}(\text{Adaptive}) - T_{H,i}(\text{Fixed})$:

$$t = \frac{\bar{D}}{s_D / \sqrt{n}}, \qquad p = F_{t,\,n-1}(t) \quad \text{for } H_1: \mu_D < 0$$

Rules (frozen):

* $n = N_{valid} \ge$ `MIN_VALID_N = 8`, else the cell is **not evaluable** (raises; never returns a number).
* Degrees of freedom are always `n - 1` — never hard-coded to the full planned sample (the CI formula in earlier revisions hard-coded the full-sample critical value; this revision computes the critical value at the actual $n$).
* **Degenerate variance** (`s_D = 0`): $t = -\infty, p = 0$ (mean < 0); $t = +\infty, p = 1$ (mean > 0); $t = 0, p = 0.5$ (mean = 0).
* One-sided, `ALPHA = 0.05`, direction: negative favors Adaptive.

**Why not a sign-flip permutation test as primary** (this replaces the earlier plan): a sign-flip test of the mean is exact only under sign-symmetry/exchangeability of the $D_i$ under $H_0$, an assumption that cannot be checked at $n = 10$; the earlier formulation also hard-coded the enumeration size to the full planned sample, which breaks the moment a replicate is lost. The t-test makes its (standard, stated) normality assumption explicit; sensitivity analyses below probe how much the conclusion depends on it.

**Assumption, honestly stated**: the t-test is valid under approximate normality / finite variance of $D_i$; with $n$ between 8 and 10 this is a modeling assumption, not a guarantee. If the sensitivity analyses (§18.3) disagree with the primary test, the report must say so prominently and the family claim, while still decided by the primary rule, must be flagged as assumption-sensitive.

### 18.3 Sensitivity analyses (different targets — labeled)

None of these replace the primary test; each targets a different quantity:

| Analysis | Target | Notes |
|---|---|---|
| Exact sign test | median of $D$ ($P(D<0)=1/2$) | `p = P(K >= k)`, `K ~ Bin(n', 0.5)`; zeros dropped |
| Exact Wilcoxon signed-rank | pseudomedian under symmetry | sign-flip enumeration over average ranks; zeros dropped; `n' ≤ 20` |
| Percentile bootstrap CI | mean of $D$ (approximate) | `BOOTSTRAP_REPS = 10000`, `BOOTSTRAP_SEED = 168098`, `numpy.random.default_rng` — reproducible for a fixed numpy version (record it) |
| Imputation bounds | mean of $D$ under failure | both directions over the genuine domain `[0, H]` (§19) |

### 18.4 Effect size and interval

* **Cohen's $d_z = \bar{D} / s_D$**; negative favors Adaptive. If $s_D = 0$: $d_z = 0$ when $\bar{D} = 0$; otherwise undefined — report $\bar{D}$ raw (never a division by zero).
* **95% t-interval**: $\bar{D} \pm t_{1-\alpha/2,\,n-1} \cdot s_D/\sqrt{n}$ at the **actual** $n$ (§18.2). This is an uncertainty interval for the mean paired difference, not exact coverage — with $n \le 10$ normality of the $D_i$ is doing real work.

### 18.5 Multiplicity

* **Family claim (§17.1)**: IUT — all six one-sided tests at `ALPHA = 0.05`, no correction required for the conjunction (its size is bounded by $\alpha$ without adjustment).
* **Per-cell claims**: Holm step-down adjusted p-values across the six cells reported alongside raw values; per-cell "significant" statements must use the Holm-adjusted value.
* Secondary endpoints and sensitivities are exploratory: uncorrected, explicitly labeled.

## 19. Failure / Invalid / Censored Runs

Identical rules for both arms; counts reported per arm and per cell.

| Category | Definition | Primary analysis | Sensitivity analysis |
|---|---|---|---|
| Planned | `PLANNED_N = 10` replicates | — | — |
| Completed | training + shared segment + both arm segments succeed | included pairwise | included |
| Failed | training divergence, NaN, crash, failed update block (§5.2) | replicate excluded **pairwise** (both arms dropped for that $i$) | imputed (below) |
| `no_degradation` | $\delta \le 0$ | $T_H = 0$ | $T_H = 0$ |
| `degradation_below_resolution` | $0 < \delta <$ `delta_min` | $T_H = 0$, label kept | $T_H = 0$ |
| `right_censored` | no recovery within $H$ | $T_H = 15$ | $T_H = 15$ |
| Not evaluable cell | $N_{valid} <$ `MIN_VALID_N = 8` | no p-value; family → `INCONCLUSIVE` | reported descriptively |

**Imputation sensitivity** (both directions, genuine bounds): because $T_H \in [0, H]$ always (§3), imputing failed arms within $[0, H]$ produces true bounds on each $D_i$:

* `worst_for_adaptive`: failed Adaptive → $T_H = H$; failed Fixed → $T_H = 0$.
* `best_for_adaptive`: the reverse.

Report the primary test under both imputations of the *planned* 10 replicates. These are bounds **conditional on the domain**; the counterfactual behavior of a crashed run is unobservable, so no imputation is "the truth".

Pairwise-complete exclusion in the primary analysis can create attrition bias if failure rates differ by arm; the per-arm failure counts plus the two-sided imputation table are the pre-registered mitigation.

## 20. Reporting Requirements

### 20.1 Historical Fixed-benchmark artifact (pinned protocol reference)

`ShiftBenchmarkReport` stores (actual field names): `schema_version`, `experiment_name`, `environment_name`, `algorithm_name`, `deterministic`, `total_training_timesteps`, `train_seeds`, `test_seeds`, **`scenarios`** (list of `ScenarioResult`), `recovery_definition`, `training_provenance`, `environment_provenance`, `config_sha256`, `metadata`.

The legacy Fixed-benchmark implementation at the pinned protocol revision stored per scenario: `scenario_name`, `role`, `seeds`, `environment_overrides`, `effective_environment_parameters`, `metrics`, `recovery`, **`episodes`** (list of `EpisodeBenchmarkRecord`), `gaps`, `policy_fingerprint`. That legacy module is not present in the current checkout; Issue #265 writes its own explicit adaptation schema.

Per episode record (`shift_benchmark.py:466-490`): `seed`, `reward`, `length`, `success`, `collision`, `terminated`, `truncated`, `recovery_times`, `recovery_events`, `recovery_completed`, `recovery_censored`.

> Earlier revisions of this document used a wrong field name for the per-scenario list; the real field is `scenarios[]` with `episodes[]` inside. `tests/test_protocol_doc_sync.py` bans the wrong name.

### 20.2 Derived research quantities

$P_{pre}$, $P_0$, $P(t)$, $R(t)$, $\tau$, $T_H$, status, δ, `delta_min`, and the update-block log are derived from episode records; the Issue #265 artifact stores them using `src/adaptive_rl/protocol/recovery.py` (never ad hoc equations).

### 20.3 Issue #265 artifact contents

Per replicate × arm: training seed, all three phase-seed lists used, schedule fingerprint (§14.3), `PROTOCOL_VERSION`, pinned commit SHA, config SHA-256, pre-shift/shock/post-shift return vectors, per-episode success flags, $P_{pre}$, $P_0$, δ, `delta_min`, $P(t)$ and $R(t)$ trajectories, predicate vector, $\tau$, $T_H$, status; per block: block id, episode range, derived update seed, data-episode indices, parameter-delta norm, fingerprint before/after; plus environment/library provenance (`environment_provenance` already exists) including numpy version (bootstrap dependency, §18.3).

## 21. Success Criteria

* **Family claim**: `SUPPORTED` iff all six cells evaluable and every one-sided primary `p < 0.05` (§17.1). Anything else is `NOT_SUPPORTED` or `INCONCLUSIVE` — never a weaker paraphrase like "trend toward support".
* **Per-cell statements**: require Holm-adjusted `p < 0.05` and are secondary to the family claim.
* **Practical significance**: judged by $d_z$ and the 95% interval, reported regardless of significance; statistical `SUPPORTED` with a trivial $d_z$ must be reported as such.
* **Assumption sensitivity**: disagreement between the primary test and §18.3 analyses must be stated in the abstract-level summary of results.

## 22. Threats to Validity

| Threat | Why it matters | Mitigation | Remaining limitation |
|---|---|---|---|
| Treatment-contaminated $P_0$ | Adaptive updating before $P_0$ is measured breaks normalization | Shared single execution of episodes 1–5 (§4.2); $B_5$ strictly after episode 5 | Full ten-replicate protocol execution remains unverified |
| `no_degradation` convention | $T_H = 0$ anchors the distribution without adaptation evidence | Convention retained, status always labeled; arm-invariant so $D_i = 0$ exactly | Can still shift the *level* of $T_H$ vs other studies; comparisons must match conventions |
| Finite-horizon truncation | $T_H = 15$ for non-recovery makes the estimand a truncated mean | Estimand explicitly finite-horizon; right-censoring labeled, never extrapolated | Not a claim about true recovery-time distributions |
| Normality of $D_i$ (t-test) | Primary test assumes it; $n = 8..10$ cannot verify it | Sensitivities (§18.3) target median/pseudomedian/resampling; disagreement must be reported | Type I/II error may deviate from nominal if strongly violated |
| `MIN_VALID_N = 8` threshold | A cell at exactly 8 has less power than one at 10 | Rule frozen pre-data; per-cell $N_{valid}$ always reported | Power is unquantified; no formal power analysis (§25) |
| Failed-run attrition | Excluding crashes can bias the contrast | Pairwise exclusion; per-arm failure counts; two-sided $[0,H]$ imputation bounds | Counterfactual $T_H$ of crashed runs unobservable |
| Partial cell coverage | Only the drone PPO/SAC cells have the Issue #265 runner in this checkout | Missing cells remain explicitly inconclusive; no six-cell family claim | GridWorld, Traffic Signal, and Navigation cells are unimplemented |
| Unequal compute | Adaptive adds online compute | Declared part of the intervention; Fixed budget unchanged | Deployment latency/cost unmeasured |
| Training non-determinism | No cudnn/deterministic-algorithm flags exist (`trainer.py:120-126`) | Train-once design: one run serves both arms (§4.2) | Cross-machine retraining may not reproduce weights; recorded fingerprint detects it |
| CRN partial alignment | Episode lengths are action-dependent | Reseeding per episode; alignment stated as partial-by-construction (§13) | Variance reduction weaker than exact pairing of noise |
| Config/YAML drift | Configs may change in later commits | Pinned commit SHA + `config_sha256` + budget table (§15) | Reproducibility depends on SHA accessibility |
| Environment heterogeneity | 4 simulated environments | 4 environments × 2 algorithms in the family | Small environment count; all simulated; no real deployment data |
| Shift realism | Synthetic parameter shifts | Shifts target documented physical parameters (`wind_speed`, `gust_sigma`, obstacle density, arrival rates) | Simulation-to-reality gap |
| Benchmark-threshold confusion | Config `recovery_event_threshold: 0.7` ≠ protocol `0.9` | Explicit non-equivalence stated (§12.4) | Readers may still conflate them |

## 23. Reproducibility Checklist

* [ ] `PROTOCOL_VERSION` and pinned commit `faefc5c8e4a39bbcc728d73b1c9855c8e9c5386f` recorded in every results artifact.
* [ ] Config YAMLs archived at the pinned commit; `config_sha256` recorded.
* [ ] All 400 derived seeds logged; schedule fingerprint equals `65939167572731c99599c382ac50cf3fddbba3cf758305764392f13b2e4efa67` in both arms.
* [ ] `validate_schedule()` run on the artifact's schedule with zero violations (uniqueness, range, disjointness).
* [ ] Training receives nominal parameters only; the runner withholds TEST-B parameters until after shared pre-shift evaluation, and `validate_schedule()` passes.
* [ ] Train-once/fork design confirmed: one training fingerprint per replicate, both arms fork from it; Fixed arm fingerprint unchanged at end.
* [ ] $P_0$ measured in the shared segment, pre-$B_5$; block ordering log shows $B_5$ strictly between episodes 5 and 6.
* [ ] Per-block log complete for $B_5..B_{14}$ (10 blocks, no block after episode 15).
* [ ] $P_{pre}$, $P_0$, $P(t)$, predicate, $\tau$, $T_H$, status derived via `adaptive_rl.protocol.recovery`.
* [ ] Primary test, sensitivities, Holm, and `decide_family()` executed from `adaptive_rl.protocol.statistics`.
* [ ] Per-cell $N_{valid}$, failure counts by arm, and both imputation directions reported.
* [ ] Doc-sync tests pass: `python -m pytest tests/test_protocol_seed_schedule.py tests/test_protocol_recovery.py tests/test_protocol_statistics.py tests/test_protocol_doc_sync.py`.
* [ ] Treatment Card (§5.3) frozen before a scientific run and its SHA recorded in the artifact.

## 24. Executable Protocol Mirror

| Concern | Module | Tests |
|---|---|---|
| Frozen constants (`PROTOCOL_VERSION`, seeds, `K_pre`, `H`, `N_update`, `MIN_VALID_N`, `ALPHA`, bootstrap config, cells, pools) | `src/adaptive_rl/protocol/constants.py` | `tests/test_protocol_seed_schedule.py`, `tests/test_protocol_doc_sync.py` |
| Seed derivation, schedule, validation, fingerprint | `src/adaptive_rl/protocol/seed_schedule.py` | `tests/test_protocol_seed_schedule.py` (incl. cross-process/`PYTHONHASHSEED` independence via subprocess) |
| Recovery endpoint, statuses, edge cases, secondary endpoints | `src/adaptive_rl/protocol/recovery.py` | `tests/test_protocol_recovery.py` (all §7 cases) |
| Primary t-test, t CI, $d_z$, sensitivities, Holm, IUT decision, imputation | `src/adaptive_rl/protocol/statistics.py` | `tests/test_protocol_statistics.py` |
| Document ↔ code agreement (versions, literals, SHA, bans on superseded claims) | this document | `tests/test_protocol_doc_sync.py` |

Run everything:

```
python -m pytest tests/test_protocol_seed_schedule.py tests/test_protocol_recovery.py \
    tests/test_protocol_statistics.py tests/test_protocol_doc_sync.py
```

Changing any constant requires bumping `PROTOCOL_VERSION` and revising this document in the same change; the doc-sync test is the enforcement point.

## 25. Non-Claims and Limitations (explicit)

This protocol does **not** claim:

1. **Executability** — no Adaptive cell can run today (§1, §17).
2. **Bit-level reproducibility of training** — no deterministic-algorithm flags exist; the design avoids needing them (§4.2, §22).
3. **Statistical exactness of the primary test** — the t-test is assumption-dependent at $n = 8..10$; sensitivities are probes, not replacements (§18.2–18.3).
4. **Power** — `PLANNED_N = 10` is a budget, not a power calculation; power for plausible effect sizes is unknown.
5. **Validation on data** — no result has been collected under this protocol; every worked number in §7 is a hand-computed test vector, not an experimental finding.

## 26. Open Research Questions

* How does the adaptation compute overhead scale with observation dimensionality?
* Can representation learning improve the sample efficiency of the online adaptation phase?
* What $n$ is required to achieve 80% power for a plausible $d_z$ (once pilot estimates exist)?
* Does the choice of update-data window (cumulative post-shift, §5.1) materially change $T_H$ relative to a rolling window — and should that be a registered secondary factor?
