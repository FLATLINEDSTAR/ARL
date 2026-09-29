# AdaptiveRL Treatment Card: Online Fine-Tuning Specification

**Protocol Version**: 2.0<br>
**Status**: Preregistered & Frozen<br>
**Governing Protocol**: [`docs/research/adaptive_rl_hypothesis.md`](file:///home/ux0/ARL/docs/research/adaptive_rl_hypothesis.md)<br>
**Target Milestone**: Issue #265 (`[Research] Online adaptation after a distribution shift: Adaptive policy vs Fixed policy`)

---

## 1. Scope & Purpose

This Treatment Card defines the exact algorithmic and optimization hyperparameters governing the **Adaptive** arm during post-shift online fine-tuning. In accordance with §5.1, §5.2, §13, and §14 of the AdaptiveRL Research Protocol, this card freezes all adaptation parameters prior to data collection.

### Core Protocol Constraints
1. **Unmodified Base Hyperparameters**: All base model architecture and environment parameters remain identical to the training configuration unless explicitly declared below.
2. **No Future Leakage**: Block $B_k$ has zero access to transitions, returns, or environment outcomes from episodes $> k$.
3. **Deterministic Seeding**: No pseudo-random draw may occur outside the preregistered derived seed schedule (`derive_seed(training_seed, "update", b)`).
4. **Arm Invariance**: The **Fixed** arm starts from the identical frozen checkpoint and executes **no** update blocks ($B_k = \emptyset$ for all $k$). Both arms share the exact same pre-shift evaluation and shock window (episodes 1–5).

---

## 2. Adaptation Schedule & Block Triggering

The Adaptive arm executes exactly $N_{update} = 10$ update blocks, denoted $B_5, B_6, \dots, B_{14}$:

| Block | Execution Timing | Data Visibility | Derived Seed Call |
|---|---|---|---|
| $B_5$ | Post-shift episode 5 termination $\longleftrightarrow$ episode 6 reset | Episodes 1–5 (shock window) | `derive_seed(seed, "update", 0)` |
| $B_6$ | Post-shift episode 6 termination $\longleftrightarrow$ episode 7 reset | Episodes 1–6 (cumulative) | `derive_seed(seed, "update", 1)` |
| $B_7$ | Post-shift episode 7 termination $\longleftrightarrow$ episode 8 reset | Episodes 1–7 (cumulative) | `derive_seed(seed, "update", 2)` |
| $B_8$ | Post-shift episode 8 termination $\longleftrightarrow$ episode 9 reset | Episodes 1–8 (cumulative) | `derive_seed(seed, "update", 3)` |
| $B_9$ | Post-shift episode 9 termination $\longleftrightarrow$ episode 10 reset | Episodes 1–9 (cumulative) | `derive_seed(seed, "update", 4)` |
| $B_{10}$ | Post-shift episode 10 termination $\longleftrightarrow$ episode 11 reset | Episodes 1–10 (cumulative) | `derive_seed(seed, "update", 5)` |
| $B_{11}$ | Post-shift episode 11 termination $\longleftrightarrow$ episode 12 reset | Episodes 1–11 (cumulative) | `derive_seed(seed, "update", 6)` |
| $B_{12}$ | Post-shift episode 12 termination $\longleftrightarrow$ episode 13 reset | Episodes 1–12 (cumulative) | `derive_seed(seed, "update", 7)` |
| $B_{13}$ | Post-shift episode 13 termination $\longleftrightarrow$ episode 14 reset | Episodes 1–13 (cumulative) | `derive_seed(seed, "update", 8)` |
| $B_{14}$ | Post-shift episode 14 termination $\longleftrightarrow$ episode 15 reset | Episodes 1–14 (cumulative) | `derive_seed(seed, "update", 9)` |

*Note: No update block is executed after episode 15 (end of horizon $H=15$).*

---

## 3. Optimization Hyperparameters (PPO Online Adaptation)

| Hyperparameter | Value | Description / Rationale |
|---|---|---|
| **Optimizer** | Adam | Standard first-order stochastic optimizer |
| **Adaptation Learning Rate ($\alpha_{adapt}$)** | $1.0 \times 10^{-4}$ | Conservative rate to prevent catastrophic forgetting while adapting to shifts |
| **Adam Betas** | $(0.9, 0.999)$ | Standard momentum and variance decay |
| **Adam Epsilon** | $1.0 \times 10^{-8}$ | Numerical stability denominator |
| **Update Epochs ($N_{epochs}$)** | 2 | Small number of passes over post-shift buffer to bound policy drift |
| **Minibatch Size** | 32 | Minibatch sampling size: $\min(32, N_{transitions})$ |
| **Discount Factor ($\gamma$)** | 0.99 | Temporal return discounting matching base policy |
| **PPO Clipping Range ($\epsilon$)** | 0.2 | Standard clipped surrogate objective radius |
| **Value Function Weight ($c_1$)** | 0.5 | Mean squared error value loss coefficient |
| **Entropy Weight ($c_2$)** | 0.01 | Policy entropy bonus coefficient |
| **Max Gradient Norm** | 0.5 | Gradient clipping threshold |

---

## 4. Surrogate Objective Formulation

For each minibatch drawn during block $B_k$, the objective optimized over policy parameters $\theta$ is:

$$L(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) + c_1 L_t^{VF}(\theta) - c_2 S[\pi_\theta](s_t) \right]$$

where:
- Probability ratio: $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$
- Clipped surrogate: $L_t^{CLIP}(\theta) = -\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)$
- Value loss: $L_t^{VF}(\theta) = \frac{1}{2} \left( V_\theta(s_t) - G_t \right)^2$
- Empirical advantage: $\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + 10^{-8}}$, with $A_t = G_t - V_{\theta_{old}}(s_t)$

---

## 5. Audit Logging & Verification Requirements

Each update block execution must record in the experiment output artifact:
1. `block_id`: Identifier ($B_5 \dots B_{14}$)
2. `episode_k`: The completed episode triggering the block ($5 \dots 14$)
3. `derived_seed`: The 31-bit seed applied to PyTorch, NumPy, and Python RNGs
4. `num_transitions`: Total transition tuples in the cumulative buffer
5. `data_episodes`: Range $[1, k]$ of post-shift episodes utilized
6. `parameter_delta_norm`: Euclidean norm of parameter weight changes:
   $$\Delta_\theta = \sqrt{\sum_{p} \|\theta_{new}^{(p)} - \theta_{old}^{(p)}\|_2^2}$$
7. `fingerprint_before`: SHA-256 hash of the policy `state_dict` before block execution
8. `fingerprint_after`: SHA-256 hash of the policy `state_dict` after block execution
