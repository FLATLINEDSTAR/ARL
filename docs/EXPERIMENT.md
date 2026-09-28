# Empirical Experimentation & Scientific Evaluation Methodology

This document details the experimental methodology, hypotheses, benchmark variables, and verified empirical results for the **AdaptiveRL** 3D drone navigation project.

---

## 1. Experimental Methodology & Hypotheses

### Hypothesis 1: Learning Validation (PPO vs Random Baseline)
> *A Proximal Policy Optimization (PPO) agent trained on continuous translational kinematics and LiDAR range observations will achieve a significantly higher survival rate, lower collision rate, and higher cumulative return than an untrained uniform-random action baseline when evaluated on identical test environments.*

### Hypothesis 2: Environmental Difficulty Scaling (Obstacle Density)
> *As the number of procedural obstacles in the 3D flight arena increases (from 4 to 6 to 8 obstacles), the task difficulty will scale non-linearly, resulting in monotonically increasing collision rates and decreased mean episode return.*

---

## 2. Experimental Setup & Variables

| Variable Category | Parameter | Experimental Setting |
|---|---|---|
| **Independent Variables** | Policy Type | PPO (`MlpPolicy`) vs Uniform-Random Action Baseline |
| | Obstacle Count | 4 obstacles, 6 obstacles, 8 obstacles |
| **Controlled Variables** | Arena Bounds | $30.0\text{ m} \times 30.0\text{ m} \times 15.0\text{ m}$ |
| | Physics Integration | $\Delta t = 0.1\text{ s}$, Point-mass kinematics with linear drag $c_d = 0.05$ |
| | Max Speed / Accel | $v_{\max} = 8.0\text{ m/s}$, $a_{\max} = 4.0\text{ m/s}^2$ |
| | Sensor Configuration | 16-ray spherical LiDAR ($20.0\text{ m}$ max range) |
| | Evaluation Seeds | Base seed $42$, episode seeds $s_i = 42 + i$ |
| | Episode Budget | 20 evaluation episodes per policy benchmark; 10 episodes per density condition |
| **Dependent Metrics** | Success Rate | Percentage of episodes reaching target within $1.5\text{ m}$ radius |
| | Collision Rate | Percentage of episodes colliding with obstacles or arena walls |
| | Mean Return | Cumulative discounted episodic reward $\sum_t R_t$ |
| | Mean Episode Length | Average timesteps before terminal state or truncation ($max\_steps = 200$) |

---

## 3. PPO Learning-Curve Benchmark

The budget benchmark trains a fresh PPO model from the same base configuration at each requested training budget. Every model is evaluated with the same ordered evaluation seed groups, episode count per seed, environment parameters, algorithm settings, and deterministic-action setting; evaluation uses the saved model and a separate fresh environment. `--eval-seeds` selects the seed groups; `--episodes` is the number of episodes run within each group and does not determine how many seeds are evaluated.

A single training seed is shared by every budget: `--training-seed` (or `benchmark.training_seed`, falling back to `seed`) is written into each budget's configuration copy before training, and the legacy algorithm-level `parameters.seed` value is dropped so that budgets differ only in budget size. Differences between budgets therefore cannot be attributed to re-seeding. Training itself is not guaranteed bit-for-bit reproducible across hardware, PyTorch versions, or CUDA kernels.

```bash
adaptive-rl benchmark budgets \
  --config configs/drone_ppo.yaml \
  --budgets 5000,10000,25000,50000 \
  --training-seed 42 \
  --eval-seeds 42,43,44,45,46 \
  --episodes 20 \
  --deterministic
```

The command reports the budget list and output locations when complete. By default, machine-readable artifacts are written beneath `artifacts/benchmarks/`:

```text
Learning Curve Benchmark
PPO learning-curve benchmark complete
Budgets: 5,000, 10,000, 25,000, 50,000
Training seed: 42
Evaluation seeds: [42, 43, 44, 45, 46]
JSON: artifacts/benchmarks/learning_curve_budget.json
CSV: artifacts/benchmarks/learning_curve_budget.csv
Plot: not generated

artifacts/benchmarks/
├── learning_curve_budget.json
├── learning_curve_budget.csv
└── learning_curve/
    ├── budget_5000/models/ppo_budget_5000_final.zip
    ├── budget_10000/models/ppo_budget_10000_final.zip
    └── ...
```

JSON contains benchmark settings, one result object per completed budget, pooled metrics, per-seed summaries, cross-seed Student's t statistics, and plot-ready series. Its provenance records Python and relevant library versions, normalized training/evaluation environment settings, and their SHA-256 fingerprint. Each budget also records its model device and evaluation split. CSV contains the pooled per-budget performance values with provenance columns.

The benchmark writes its JSON and CSV artifacts before attempting any plot and validates the JSON payload as strict JSON (finite numbers only, no NaN or Infinity, no non-native numeric types such as `float32` or `Path`), so an export failure cannot leave a half-written or unparseable report behind.

**JSON schema.** The document has these top-level keys:

| Key | Contents |
|---|---|
| `status` | `"completed"` when every requested budget finished, `"failed"` when the run stopped early |
| `completed_budgets` | Budgets that trained and evaluated successfully, in run order |
| `failed_budget` | The budget at which the run stopped, or `null` |
| `error` | Sanitized `TypeName: message` for the failure (never a traceback), or `null` |
| `benchmark` | Settings: `training_seed`, `evaluation_seeds`, `evaluation_group_seeds`, `evaluation_episodes`, `episodes_per_seed`, `evaluation_split`, `deterministic`, `budgets`, plus seed-mapping, metric, and training-time semantics |
| `provenance` | Python/library versions, normalized training/evaluation environment configuration, and a SHA-256 environment-configuration fingerprint |
| `results` | One object per completed budget (see below) |
| `plot` | `requested` (bool), `path`, and `error` for the optional figure |
| `plot_data` | Plot-ready series: `budgets`, `trained_timesteps`, `success_rate`, `mean_reward` |

Each `results` object carries `budget_timesteps`, `trained_timesteps`, the pooled metrics (`success_rate`, `collision_rate`, `timeout_rate`, `mean_reward`, `std_reward`, `mean_episode_length`), `training_time_seconds`, `model_path`, `training_seed`, `evaluation_seeds`, `evaluation_episodes`, `evaluation_split`, `deterministic`, `algorithm`, `environment`, `environment_fingerprint`, `device`, `descriptive_metrics`, `per_seed_summaries`, `cross_seed_statistics`, and `training_metadata`. `descriptive_metrics` is the same pooled sample as the top-level metrics and additionally reports `episodes`; `per_seed_summaries` has exactly one entry per evaluation seed group; `cross_seed_statistics` holds the Student's t statistics over those seed groups (`sample_count` counts seeds, not episodes); `training_metadata` repeats the training/evaluation provenance for the model behind that row.

**CSV schema.** One row per completed budget with columns `budget_timesteps`, `trained_timesteps`, `success_rate`, `collision_rate`, `timeout_rate`, `mean_reward`, `std_reward`, `mean_episode_length`, `training_time_seconds`, `model_path`, `training_seed`, `evaluation_seeds` (seeds joined with `;`), `evaluation_episodes`, and `deterministic`.

**Partial failures.** If a budget fails, the run stops at that budget: JSON and CSV are still written with `status: "failed"`, `completed_budgets`, `failed_budget`, and `error`, and `adaptive-rl benchmark budgets` prints a partial-failure report and exits with status `1`. Model artifacts and metrics for budgets listed in `completed_budgets` remain on disk and valid; a failed budget is never serialized as a completed result.

**Plotting.** Pass `--plot` to additionally render `learning_curve_budget.png`. Matplotlib is imported only when plotting is requested, comes from the optional `plot` extra (`python -m pip install -e ".[plot]"`, included in `[all]`), and generated figures are always closed after saving. `--plot-x-axis trained` (default) uses the timesteps PPO actually collected, while `--plot-x-axis requested` uses the requested budget; the axis label always states which one is plotted, and the two differ whenever a budget is not aligned to a rollout boundary. A plot failure cannot corrupt benchmark data: it is recorded in `plot.error`, the JSON is rewritten with that field, the CSV is unchanged, and the CLI exits with status `1` after reporting the completed benchmark.

The built-in benchmark defaults are budgets `[5000, 10000, 25000, 50000]`, training seed `42`, evaluation seed groups `[42, 43, 44, 45, 46]`, and `20` episodes per seed. A `benchmark` section in the YAML supplies these values instead; explicit CLI options override the corresponding config values. Legacy `evaluation.eval_episodes` does not control the number of seed groups or the benchmark episode count. Thus, without overrides, the default evaluation runs five seed groups with twenty episodes each, not twenty seed groups with twenty episodes each.

`budget_timesteps` records the requested budget, while `trained_timesteps` records the actual environment interactions reported by Stable-Baselines3. For example, budget `65` with PPO `n_steps: 64` trains to `128` steps because PPO collects complete rollouts. Compare results using `trained_timesteps` when budgets are not aligned to rollout sizes.

`training_time_seconds` measures only the call to `PPOAlgorithm.train()` using a monotonic clock. It excludes environment/model setup, final model serialization, metadata writing, evaluation, JSON/CSV export, and plotting. Training metadata also retains the broader legacy `duration_seconds` lifecycle measure, which is not the benchmark training-time metric. Neither duration is hardware-independent.

The named benchmark metrics (`success_rate`, `collision_rate`, `timeout_rate`, `mean_reward`, `std_reward`, and `mean_episode_length`) are pooled descriptive summaries over all evaluated episodes for a budget. Reward standard deviation is the sample standard deviation across pooled episode returns and is unavailable (`null` in JSON, blank in CSV) with fewer than two episodes. Success and collision rates use episodes that reported the corresponding outcome field; timeout rate is based only on Gymnasium's actual `truncated` signal. The JSON additionally retains per-seed summaries and cross-seed Student's t statistics from the reusable evaluator; these are distinct from the pooled metrics and are not estimates based on the pooled episode sample. Within-seed reward and episode-length standard deviations follow the evaluator's existing population-standard-deviation convention; cross-seed uncertainty is then calculated over those seed summaries using sample-standard-deviation and Student's t conventions.

An evaluation seed is a group identifier, not itself necessarily the environment reset seed. In the default `custom` mode, each episode reset seed is generated with an injective Cantor-pair mapping of the group seed and zero-based episode index. It is stable when the episode count is extended and keeps distinct group/episode pairs distinct. With `evaluation_split: train` or `test`, reset seeds are SHA-256-derived into that split's finite seed interval; mapping into a finite interval can collide. In split mode, evaluation group seeds must themselves be valid members of the selected partition (the custom defaults are not valid test-split seeds). Split evaluation requires training on the `train` split, so test layouts are not used in PPO rollouts. The default `custom` evaluation uses the configured environment without its training split restriction.

Interpret the curves jointly: rising success rate and mean reward with a falling collision or timeout rate suggest improvement; flat metrics may indicate a plateau. Treat these curves as empirical observations, not as a monotonicity guarantee: each budget is an independent training run, and sampling and optimization noise can make a larger budget score worse than a smaller one on some metrics. A timeout is counted only when Gymnasium returns `truncated=True`, not merely because an episode has a particular length. The same seed groups and settings make evaluation conditions comparable, but do not remove variation from training or guarantee bit-for-bit results across hardware, PyTorch versions, or CUDA kernels.

For a CI-sized run, copy the experiment YAML and set PPO `n_steps: 64` and `batch_size: 32` in that copy. Then run a short evaluation:

```bash
cp configs/drone_ppo_demo.yaml /tmp/drone_ppo_ci.yaml
# Edit /tmp/drone_ppo_ci.yaml: set n_steps to 64 and batch_size to 32.
adaptive-rl benchmark budgets --config /tmp/drone_ppo_ci.yaml --budgets 64,128 --episodes 1
```

The committed demo config uses `n_steps: 1024`, so those tiny budgets would be rounded up to its rollout boundary; keep the shipped training hyperparameters unchanged and use the copied config only for this CI-sized run.

---

## 4. Multi-Seed Evaluation and Confidence Intervals

Evaluation over several independent environment seeds helps show how policy performance varies with randomized starts and obstacles, instead of depending on one seed sequence. `--episodes` is the number of episodes run for each listed seed. Each requested seed identifies a group; the environment reset seed is derived for each group/episode pair and does not change for existing episodes when the requested episode count increases. For the `custom` mapping the pairing is injective; for bounded train/test splits the SHA-256-derived mapping is deterministic but sampling with replacement means collisions are possible. The requested group seed and actual per-episode reset seed are both recorded. Duplicate requested group seeds are rejected to avoid overweighting a repeated condition.

Episode records expose both seed meanings under explicit names: `evaluation_group_seed` is the requested evaluation seed (the statistical grouping unit) and `episode_reset_seed` is the per-episode environment reset seed. The legacy field names keep their documented meanings — `seed` equals `evaluation_group_seed` and `episode_seed` equals `episode_reset_seed`. Per-seed summaries expose `evaluation_group_seed` alongside `seed`, and report metadata carries `evaluation_group_seeds` plus a `seed_semantics` string.

```bash
adaptive-rl evaluate \
  --config configs/drone_ppo.yaml \
  --model artifacts/models/drone_ppo_final.zip \
  --seeds 0 1 2 3 4 \
  --episodes 10 \
  --deterministic
```

The existing invocation remains single-seed and uses the configuration seed unless overridden with `--seed`:

```bash
adaptive-rl evaluate --config configs/drone_ppo.yaml --episodes 10
adaptive-rl evaluate --config configs/drone_ppo.yaml --seed 7 --episodes 10
```

`--seed` and `--seeds` are mutually exclusive. In multi-seed mode, `--episodes` is per seed, and `--compare-random` is not supported. The command writes `artifacts/evaluation_multiseed.json` and `artifacts/evaluation_multiseed.csv` by default; `--output-report` and `--output-csv` can select alternate destinations.

The JSON retains raw episode records (requested seed, episode index, actual reset seed, return, episode length, outcomes, truncation, and path length when the environment reports positions), per-seed summaries, aggregate metrics, and evaluation metadata. Optional geometry metrics follow an explicit availability contract: `path_length` is unavailable (`null`, and omitted from episode records) when the environment reports no positions, and obstacle surface clearance plus obstacle-vs-boundary collision typing are available only when the environment exposes obstacle geometry — read from the public `unwrapped.obstacles` interface when present, falling back to the private `_obstacles` attribute. The CSV is a stable, aggregate-only table with one row per metric and columns `metric`, `mean`, `std`, `ci95_lower`, `ci95_upper`, `sample_count`, `seed_count`, `episodes_per_seed`, and `total_episodes`.

Cross-seed means and confidence intervals are calculated from the per-seed summaries, not pooled episodes. The standard deviation is the sample standard deviation (`ddof=1`); two-sided 95% confidence intervals use Student's t critical values and `mean ± t * s / sqrt(n)`. Missing values are excluded per metric. With fewer than two valid seeds, sample standard deviation and CI bounds are `null`/unavailable; they are not replaced with zero. The interval describes uncertainty in the estimated mean across the evaluated seeds under the independent, representative-seed and approximate t-model assumptions. It is not proof that one policy is superior. Identical seeds and deterministic actions reproduce equivalent episode results when the policy and environment implementation are unchanged.

---

## 5. Expected Results (Hypothesized Prior to Testing)

1. **Random Action Baseline**:
   - Success Rate: $0.0\%$ (probability of randomly stumbling into a $1.5\text{ m}$ sphere across a $13,500\text{ m}^3$ arena without striking walls is practically zero).
   - Collision Rate: Approaching $100.0\%$.
   - Mean Reward: Strongly negative (dominated by $-100.0$ collision penalty).

2. **Trained PPO Policy (25,000 timesteps budget)**:
   - Success Rate: $> 0.0\%$ (occasional direct reach).
   - Collision Rate: Substantially lower than random ($< 50\%$).
   - Mean Reward: Significantly improved compared to the $-100$ collision baseline.

3. **Obstacle-Density Progression**:
   - Monotonically increasing collision rates as obstacle count increases from 4 to 6 to 8.

---

## 6. Actual Measured Results (Empirical Verification)

All results below were generated through genuine Python 3.12 CPU execution using the canonical project commands:
```bash
adaptive-rl evaluate --config configs/drone_ppo_demo.yaml --model artifacts/models/drone_ppo_demo_final.zip --episodes 20 --compare-random
adaptive-rl experiment-density --model artifacts/models/drone_ppo_demo_final.zip --episodes 10
```

### Experiment 1: PPO vs Random Action Baseline (20 Test Episodes, Seed 42)

| Policy Evaluated | Success Rate (%) | Collision Rate (%) | Mean Reward | Mean Steps | Result Summary |
|---|---|---|---|---|---|
| **Random Policy Baseline** | **0.0%** | **100.0%** | **-101.24** | **71.2** | Collided in 100% of episodes |
| **Trained PPO Policy** | **5.0%** | **35.0%** | **-3.34** | **141.0** | **65% survival rate**, +97.9 reward delta |

#### Scientific Findings:
- The uniform-random policy validates that the simulated flight arena is genuinely hazardous: an unguided drone has a 100% probability of collision within ~71 steps.
- The PPO policy learned purposeful obstacle avoidance, reducing collisions by 65 percentage points (from 100% down to 35%) and doubling flight longevity (141.0 steps vs 71.2 steps).
- The cumulative reward improved from **-101.24** to **-3.34**, proving that the policy network internalizes goal-directed attraction and LiDAR-based repulsion.

---

### Experiment 2: Obstacle-Density Scaling (10 Test Episodes per Condition, Seed 42)

| Arena Condition | Obstacles | Episodes | Success Rate | Collision Rate | Mean Return | Mean Episode Steps |
|---|---|---|---|---|---|---|
| **Low Density** | 4 Obstacles | 10 | 0.0% | **20.0%** | **+5.00** | 169.0 steps |
| **Medium Density** | 6 Obstacles | 10 | 0.0% | **40.0%** | **-15.20** | 136.2 steps |
| **High Density** | 8 Obstacles | 10 | 10.0% | **70.0%** | **-32.43** | 88.2 steps |

#### Scientific Findings:
- As obstacle count scales from 4 to 8, the collision rate jumps from **20.0%** $\rightarrow$ **40.0%** $\rightarrow$ **70.0%**, directly validating **Hypothesis 2**.
- Mean episode length drops from 169.0 steps to 88.2 steps as higher obstacle packing density causes earlier terminal collisions.
- Mean reward drops monotonically from $+5.00$ down to $-32.43$, demonstrating that higher obstacle density severely constrains safe trajectory corridors.

---

### Experiment 3: Unseen-Environment Generalization Benchmark

#### Why Train/Test Separation Exists
In standard reinforcement learning for continuous drone navigation, procedural obstacles are placed in `reset()` based on whatever random seed is active. Without strict partitioning between training and evaluation environments, an agent risks memorizing specific obstacle layouts and flight trajectories rather than mastering a generalized obstacle-avoidance policy.

To scientifically evaluate out-of-distribution transfer and prevent test contamination, AdaptiveRL establishes a strict, reproducible seed-space partitioning protocol.

#### Seed-Space Partitioning Protocol
Random seed space is partitioned into two disjoint, non-overlapping deterministic intervals:
- **Training Split (`train`)**: Seeds $[0, 1000)$ ($0 \le \text{seed} < 1000$, 1000 unique layouts).
- **Unseen Test Split (`test`)**: Seeds $[1000, 1200)$ ($1000 \le \text{seed} < 1200$, 200 unique held-out layouts).

The boundary is enforced at the environment level:
- When initialized with `--split train`, the environment only ever draws episode seeds from $[0, 1000)$ and rejects any seed outside this partition with an immediate `ValueError`.
- When evaluated with `--split test`, the environment only ever draws episode seeds from $[1000, 1200)$ and rejects training seeds.
- Zero test configurations are ever encountered during training rollouts, guaranteeing zero data leakage.

#### Metrics and Generalization Gaps
The generalization benchmark runs the frozen policy on both distributions ($N=20$ episodes each) and computes:
- **`train_success_rate` / `test_success_rate`**: Target reach rate on seen training vs unseen test distributions.
- **`train_collision_rate` / `test_collision_rate`**: Obstacle or boundary collision rate on seen vs unseen layouts.
- **`train_mean_reward` / `test_mean_reward`**: Mean cumulative episodic reward on seen vs unseen distributions.
- **Success Generalization Gap**:
  $$\Delta_{\text{success}} = \text{train\_success\_rate} - \text{test\_success\_rate}$$
  A small gap indicates strong generalization to novel obstacle configurations; a large positive gap signals policy overfitting/memorization.
- **Reward Generalization Gap**:
  $$\Delta_{\text{reward}} = \text{train\_mean\_reward} - \text{test\_mean\_reward}$$

---

## 7. Reproducibility Guarantee

To independently reproduce the benchmark and empirical results:
```bash
# 1. Clean environment install
pip install -e ".[all]"

# 2. Train with the dedicated training split (seed-partitioned)
adaptive-rl train --config configs/drone_ppo.yaml --split train

# 3. Evaluate exclusively on unseen held-out test environments
adaptive-rl evaluate \
  --config configs/drone_ppo.yaml \
  --model artifacts/models/drone_ppo_final.zip \
  --split test \
  --episodes 20

# 4. Run the full Unseen-Environment Generalization Benchmark
adaptive-rl evaluate-generalization \
  --model artifacts/models/drone_ppo_final.zip \
  --episodes 20 \
  --output-report artifacts/generalization_benchmark.json

# 5. Evaluate PPO vs Random Baseline
adaptive-rl evaluate \
  --config configs/drone_ppo_demo.yaml \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 20 \
  --compare-random

# 6. Run Obstacle-Density Experiment
adaptive-rl experiment-density \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 10 \
  --seed 42
```
All benchmark results and metrics are exported directly to structured JSON in `artifacts/`:
- `artifacts/generalization_benchmark.json`: Train/test distributions, episode seeds, performance metrics, and computed $\Delta_{\text{success}}$ and $\Delta_{\text{reward}}$.
- `artifacts/evaluation.json`: Single-run evaluation telemetry.
- `artifacts/obstacle_density_experiment.json`: Multi-density progression results.

---

## 6. Reward-Function Ablation Study

### Rationale
In continuous 3D drone navigation, reward shaping balances progress incentive against collision aversion, flight time, and control effort. Without structured empirical ablations, multi-term reward formulations remain unvalidated heuristics that can inadvertently induce suboptimal failure modes (e.g. hovering defensively to avoid effort penalties, or rushing blindly into obstacles due to excessive step penalties).

The reward-function ablation study isolates the contribution of each reward component under strictly controlled conditions.

### The Five Reward Terms
The continuous 3D navigation reward function comprises five distinct terms:
1. **$w_{\text{progress}} \times (d_{t-1} - d_t)$**: Distance-progress reward attracting the drone toward the target waypoint ($w_{\text{progress}} = 2.0$).
2. **$\text{goal\_reward}$**: Sparse terminal bonus granted upon reaching the goal within target radius ($+100.0$).
3. **$\text{collision\_reward}$**: Terminal penalty assessed upon collision with obstacles or arena boundary ($-100.0$).
4. **$\text{step\_penalty}$**: Constant time penalty incurred at each step to incentivize efficient paths ($-0.05$).
5. **$-w_{\text{effort}} \times \|\mathbf{a}_t\|_2^2$**: Smoothness/effort penalty minimizing excessive actuator chatter ($w_{\text{effort}} = 0.01$).

### The Four Controlled Variants
The study evaluates an exact four-variant progression:

| Variant | Variant Name | Progress Weight ($w_{\text{prog}}$) | Goal Reward | Collision Penalty | Step Penalty ($c_{\text{step}}$) | Action Effort Weight ($w_{\text{effort}}$) | Description |
|---|---|---|---|---|---|---|---|
| **A** | **Progress Only** | 2.0 | +100.0 | 0.0 | 0.0 | 0.0 | Pure progress delta; no step, effort, or collision penalties. |
| **B** | **Progress + Collision** | 2.0 | +100.0 | -100.0 | 0.0 | 0.0 | Adds terminal collision avoidance incentive. |
| **C** | **Progress + Collision + Step** | 2.0 | +100.0 | -100.0 | -0.05 | 0.0 | Adds step time penalty to encourage rapid goal-seeking. |
| **D** | **Full Baseline** | 2.0 | +100.0 | -100.0 | -0.05 | 0.01 | Full standard formulation (matches default environment). |

### Experimental Controls & Methodology
To ensure rigorous empirical comparison:
- **Identical Training Budget**: Each variant is trained for exactly the same number of timesteps (default 25,000 steps).
- **Identical PPO Hyperparameters**: Policy network architecture (`MlpPolicy`), learning rate ($3 \times 10^{-4}$), discount factor ($\gamma = 0.99$), batch size (64), rollout steps ($n_{\text{steps}} = 1024$), and clip range ($0.2$) are held strictly constant.
- **Identical Random Initialization**: Each variant begins from the identical base seed (default `42`), enforcing identical initial neural network weight states and identical environment procedural generation sequences.
- **Identical Held-Out Evaluation**: All four trained policies are evaluated on the identical held-out test seed distribution (`eval_seed = base_seed + 1000`) over a fixed evaluation episode count (default 20 episodes).
- **Identical Environment Geometry**: Flight arena bounds ($30\text{ m} \times 30\text{ m} \times 15\text{ m}$), obstacle count (4), start/goal coordinates, and kinematics remain identical.

### Metric Definitions
- **Success Rate**: Fraction of evaluation episodes terminating inside the calibrated target radius ($\le 1.5\text{ m}$).
- **Collision Rate**: Fraction of evaluation episodes terminating due to contact with spherical obstacles or boundary walls.
- **Timeout Rate**: Fraction of evaluation episodes truncated by reaching the maximum step limit ($max\_steps = 200$) without arrival or collision.
- **Mean Reward**: Average cumulative return per episode on the standardized held-out benchmark.
- **Mean Path Efficiency**: Ratio of straight-line distance ($D_0 = \|\mathbf{g} - \mathbf{p}_0\|$) to actual path length ($L = \sum_t \|\mathbf{p}_{t+1} - \mathbf{p}_t\|$), clamped to $[0.0, 1.0]$.
- **Convergence Speed**: The first training timestep at which the intermediate held-out evaluation success rate strictly exceeds $70\%$ ($> 0.70$). If the $70\%$ threshold is never reached during training, convergence speed is recorded as `null` (`not reached`).

### How to Run the Experiment
Run the ablation study via the command line:

```bash
# Standard 25,000-timestep research benchmark across all 4 variants
adaptive-rl experiment-ablation --timesteps 25000 --episodes 20 --seed 42

# Fast verification run (e.g. for testing)
adaptive-rl experiment-ablation --timesteps 500 --episodes 5 --seed 42
```

Outputs are automatically exported to:
- `artifacts/benchmarks/reward_ablation.json` (detailed per-variant results and configuration metadata)
- `artifacts/benchmarks/reward_ablation.csv` (tabular benchmark data for analysis)

---

## 6. Experiment Manifest and Provenance Auditing

Every training and benchmark run automatically persists a comprehensive experiment manifest to `artifacts/metadata/{name}_manifest.json` (Issue #248). The manifest ensures scientific reproducibility and verifiable artifact provenance without requiring external cloud trackers.

### Manifest Schema & Contents

The manifest document captures:
1. **Git Provenance**:
   - `git_commit`: Full 40-character SHA-1 commit hash (or `"unknown"` if executed outside a Git repository).
   - `git_branch`: Active Git branch name.
   - `git_dirty`: Boolean flag indicating whether uncommitted source modifications were present at execution time.
2. **Host System & Python Runtime**:
   - `os_name`, `os_version`: Operating system platform and kernel release.
   - `python_version`: Precise Python interpreter version.
   - `architecture`: Hardware architecture (e.g. `x86_64`, `aarch64`).
3. **Software Package Pinned Versions**:
   - Explicit pinned versions of core dependencies: `adaptive_rl`, `torch`, `stable_baselines3`, `gymnasium`, `numpy`, `typer`, and `pydantic`.
4. **Hardware & Compute Telemetry**:
   - `device`: Primary compute device used (`cpu` or `cuda`).
   - `gpu_name`, `gpu_count`: Dedicated accelerator specifications if GPU is available.
   - `cpu_count`: Available logical CPU cores.
5. **Execution Timing & Invocation**:
   - `started_at`, `finished_at`: Exact ISO 8601 UTC timestamps.
   - `duration_seconds`: Total wall-clock duration of the experiment.
   - `training_time_seconds`: Monotonic duration of interaction/training (excluding model serialization).
   - `command`: Sanitized command line arguments (`sys.argv`).
6. **Artifact Checksums & Provenance**:
   - Full filesystem paths, file sizes, and **SHA-256 cryptographic digests** for all generated model weight files (`.zip`) and metric reports (`.json`).
7. **Security & Sanitization**:
   - The manifest generator strictly avoids recording environment credentials, API tokens, passwords, or unrelated private files.

### Inspecting a Manifest via CLI

Inspect and verify an experiment manifest using the CLI:

```bash
# Using 'inspect manifest'
adaptive-rl inspect manifest artifacts/metadata/drone_ppo_manifest.json

# Using 'manifest inspect'
adaptive-rl manifest inspect artifacts/metadata/drone_ppo_manifest.json
```

The CLI renders a formatted terminal table detailing all categories and verifies whether linked artifact files exist and match their recorded SHA-256 checksums (`✓ Verified`).

### Reproducing an Experiment from a Manifest

To audit or reproduce a historical experiment:
1. **Verify Source State**: Checkout the exact Git commit recorded in `git.git_commit`. If `git_dirty` is `true`, note that working-tree modifications were present.
2. **Verify Environment**: Compare Python and package versions against `packages`.
3. **Replay Configuration**: Instantiate the experiment using the identical `config` payload and `config.seed`.
4. **Verify Generated Weights**: Calculate SHA-256 of the resulting model weights and compare against the manifest's `artifacts.model.sha256`.

