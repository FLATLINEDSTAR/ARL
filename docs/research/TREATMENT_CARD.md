# Issue #265 Treatment Card

**Status: frozen before benchmark execution.** This card defines the adaptation
treatment. Its SHA-256 is stored in every Issue #265 artifact. No benchmark
outcome may be used to change these choices.

## Fixed treatment choices

- **Scope:** adapt the already trained PPO or SAC policy using only transitions
  collected in completed post-shift episodes. Base training configuration,
  checkpoint, and environment interaction schedule are identical between arms.
- **Mechanism:** one cumulative, episode-bounded batch update at each boundary
  B5 through B14. Each batch is the concatenation of all post-shift transitions
  from episodes 1 through the boundary episode. The batch contains observation,
  action, reward, next observation, terminated, and truncated values. No nominal
  data is retained in or passed to the adaptation path.
- **PPO update:** store the behavior-policy log-probability and value estimate
  with each transition at collection time, plus the behavior value of a
  truncated transition's next observation. At each block, construct a fresh
  Stable-Baselines3 `RolloutBuffer` from the permitted cumulative prefix, use
  the configured `gamma` and `gae_lambda`, bootstrap truncated transitions
  from their recorded next observation, and run the native clipped PPO update
  for the config's `n_epochs` and `batch_size`. The configured entropy and
  value-function coefficients are unchanged. Previously seen prefix data is
  reused at later blocks with its stored behavior quantities.
- **SAC update:** construct a fresh Stable-Baselines3 `ReplayBuffer` for each
  block from the permitted cumulative prefix; the training replay buffer is
  never reused. Its capacity is `max(prefix_transition_count, configured
  batch_size)`. Run native SAC training for the configured `gradient_steps`
  and `batch_size`, retaining the configured learning rate, target update, and
  entropy mechanism. A sampled minibatch may draw with replacement when it is
  larger than the currently available transition count. Time-limit truncations
  are marked according to SB3's timeout-mask convention.
- **Randomness:** each block seeds Python, NumPy, and Torch with its one
  preregistered `update` seed while sampling/updating, then restores the caller's
  RNG states. No environment step occurs in either adapter.
- **Learning rate, batch size, epochs/steps, and regularization:** all values
  come from the base cell config; no additional optimizer or regularizer is
  added. PPO's configured entropy/value terms and SAC's configured entropy
  mechanism remain part of their respective native objectives.
- **Minimum data:** at least one complete episode (the current block's newly
  completed episode) and at least one valid transition are required.
- **Buffer construction:** a fresh immutable snapshot is assembled at each
  block boundary from the allowed prefix only. No shared/global replay buffer
  is used. Transitions are never sampled from a future episode.
- **Update timing:** only after episode termination/truncation and before the
  next reset; no updates at B15. The policy is constant during every episode.
- **Failure behavior:** any exception, non-finite loss/parameter/optimizer
  value, invalid tensor, or shape change invalidates the replicate. The block is
  rolled back before the failed replicate is recorded; execution does not
  continue to another evaluation episode.
- **Diagnostics:** record pre/post SHA-256 model fingerprints, update seed,
  visible episode indices, transition count, status, and global L2 parameter
  delta. Zero delta is recorded as a non-mutating update, not successful
  adaptation.

The Fixed arm is an independent clone of the frozen checkpoint. It performs no
training or optimizer operation after the fork. Both arms use the same derived
post-shift episode seeds. A smoke run validates machinery only and is not
empirical evidence for the research hypothesis.

## Implementation constraint

Algorithm-specific recorded-data adapters are implemented for PPO and SAC.
The complete benchmark experiment runner, environment shift integration,
artifacts, and CLI are still required before the treatment may be run as an
Issue #265 experiment. The runner must fail explicitly for any configuration
that cannot provide the required PPO behavior quantities or recorded transition
data without extra environment interaction.

## SAC online adaptation implementation (Issue #273)

This treatment uses Stable-Baselines3 SAC's own `SAC.train()` implementation;
the SAC actor, critic, and temperature losses are not reimplemented. The runner
is `src/adaptive_rl/benchmarking/adaptation_runner.py`; the isolated update
adapter and tagged replay storage are in
`src/adaptive_rl/algorithms/adaptation.py`. SAC audit data is nested under the
`sac_audit` key on each existing update-block record, so existing episode JSON
keys and CSV columns remain unchanged.

### Protocol mapping

The runner follows the existing eight-stage paired protocol:

1. Train one nominal SAC policy for the configured training budget for each
   replicate.
2. Freeze policy training mode and fingerprint the trained policy state.
3. Evaluate the nominal policy once for the shared `K_pre = 15` pre-shift
   episodes.
4. Introduce the shift and collect shared post-shift shock episodes 1–5
   without an adaptation update.
5. Deep-copy the frozen policy into Adaptive and Fixed arms and verify equal
   model-state fingerprints. The Fixed arm receives no training call.
6. Run adaptive episodes 6–15, collecting each episode to completion before
   another update can run. Run the matching fixed episodes with identical
   derived environment seeds and frozen weights.
7. Run blocks `B_5` through `B_14` only between completed episodes. `B_j` sees
   the ordered post-shift prefix 1..j, including the shock-window episodes.
8. Compute recovery with `protocol.recovery.compute_recovery` and preserve its
   status and `T_H` endpoint in the existing result artifact and analysis.

SAC differs from PPO at the update only: off-policy SAC samples replay
minibatches, while PPO consumes behavior-policy rollout quantities. Each SAC
block builds a fresh replay snapshot from the tagged visible prefix. When the
configured batch exceeds the visible transition count, SB3 replay sampling
retains its normal replacement behavior.

### SAC update settings and provenance

| Setting | Value or source |
|---|---|
| Learning rate | `algorithm.learning_rate` (distribution-shift config: `0.0003`) |
| Discount | `algorithm.gamma` (config: `0.99`) |
| Batch size | `algorithm.batch_size` (config: `128`) |
| Gradient steps | SAC wrapper default `1` per block |
| Replay capacity | `max(visible transition count, batch size)` per fresh block buffer |
| `tau` | SAC wrapper default `0.005` |
| Target update interval | Stable-Baselines3 SAC default `1`; read from instantiated model |
| Entropy coefficient | SAC wrapper default `"auto"`; retain SB3 automatic temperature optimizer |
| `train_freq` | SAC wrapper default `1`; does not collect data during recorded-data updates |
| `learning_starts` | SAC wrapper default `100`; nominal phase only; direct `train()` does not call `learn()` |
| `buffer_size` | SAC wrapper default `100000` for nominal training; never reused for adaptation |

The wrapper defaults are implementation choices where the preregistered
hypothesis does not pin down a SAC-specific setting. They are not tuned on
shifted-environment outcomes. Target update interval comes from the instantiated
SB3 model; target update count is derived from the configured gradient steps
and interval.

### Replay isolation and block audit

Immediately after the fork, the Adaptive clone's nominal replay buffer is reset
and checked to have size zero. The Fixed clone is never passed to the SAC
adapter. At each update, the adapter creates a new `TaggedReplayBuffer` and
inserts only completed post-shift transitions. Each slot records its source
episode index; construction rejects missing, out-of-prefix, and future episode
data. Buffer size and tags are checked before native training. The prior empty
adaptive buffer is restored after the update.
Stable-Baselines3 2.9.0's `BaseAlgorithm._excluded_save_params()` omits the
replay buffer from ordinary model saves. The runner currently adapts the trained
in-memory policy, but the post-fork reset also covers any buffer restored by an
explicit replay-buffer load path.

Each `sac_audit` entry records the derived update seed, gradient steps, batch
size, replay size, episode tags, actor and critic losses, alpha, alpha loss when
SB3 reports it, target-update count and interval, tau, before/after SAC state
fingerprints, and optimizer-state fingerprints. The state fingerprint covers
actor, critics, target critics, and entropy temperature because each can affect
the policy or Bellman targets. Actor and critic weights must change. Every
target parameter update is checked against the Polyak equation, and each block's
after fingerprint must match the next block's before fingerprint.

The Fixed arm is prediction-only and its fingerprint must remain equal to the
frozen fingerprint throughout its 15 post-shift episodes. Existing atomic
artifact writers install JSON and CSV via temporary files. Censored `T_H = 15`
outcomes and failure status are preserved by the shared recovery/artifact path.

### Limitations

These changes implement the SAC treatment and validation machinery; they do not
run the full ten-replicate study or establish an empirical adaptation benefit.
The SAC study requires a SAC-specific frozen cell configuration and execution
identity. Exact bitwise reproducibility is limited to matching CPU/software
environments; GPU kernels and library versions may introduce nondeterminism.
The six-cell primary registry already contains `drone_disturbed/sac`; this
treatment does not create or alter registry cells.
