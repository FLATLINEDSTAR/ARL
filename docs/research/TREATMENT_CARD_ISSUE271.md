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
