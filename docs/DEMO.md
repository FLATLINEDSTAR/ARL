# Drone RL Demonstration Guide

This guide provides a step-by-step walkthrough for demonstrating the AdaptiveRL 3D drone navigation project during a college evaluation or presentation.

---

## 1. Quick Verification (30 Seconds)

Verify that the environment and CLI are correctly set up:

```bash
# Check CLI commands
adaptive-rl --help

# Inspect the drone environment specification and observation dimensions
adaptive-rl env inspect drone

# Run the test suite
pytest
```

---

## 2. Train the Demonstration Agent (1 Minute)

Run PPO training using the demonstration budget (`25,000` steps):

```bash
adaptive-rl train --config configs/drone_ppo_demo.yaml
```

Expected output:
- Live progress bar showing timesteps and elapsed time (~25 seconds on CPU).
- Completion summary displaying total training duration and saved artifact paths:
  - Model: `artifacts/models/drone_ppo_demo_final.zip`
  - Metadata: `artifacts/metadata/training.json`

---

## 3. Deterministic Evaluation (20 Seconds)

Evaluate the trained policy across 20 deterministic test episodes:

```bash
adaptive-rl evaluate \
  --config configs/drone_ppo_demo.yaml \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 20
```

Expected output:
- A formatted Rich table showing:
  - **Success Rate**: % of episodes where the drone successfully reached within 1.0m of the target.
  - **Collision Rate**: % of episodes where the drone collided with obstacles or arena boundaries.
  - **Mean Reward**: Average cumulative reward per episode.
  - **Mean Episode Length**: Average step count before episode termination.
- JSON metrics saved to `artifacts/evaluation.json`.

---

## 4. Visual Flight Demonstration (30 Seconds)

Run a deterministic single-episode flight with seed `42`:

```bash
adaptive-rl demo-drone \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --seed 42
```

Expected output:
- Step-by-step trajectory trace displaying:
  - Current Position `[x, y, z]`
  - Distance to Goal `d_goal`
  - Minimum Obstacle Proximity `d_obs`
  - Cumulative Reward
- Final banner indicating outcome:
  - `SUCCESS: Reached target coordinate!` or `FAILED: Collided with obstacle!`

---

## 5. Browser 3D Flight Deck GUI (Presentation Mode)

For college project evaluation, presentations, or live committee review, launch the browser GUI:

```bash
streamlit run app.py
# or
adaptive-rl gui
```

### Presentation Walkthrough:
1. **Live Flight Demo**:
   - Select model checkpoint `drone_ppo_demo_final.zip`.
   - Set test seed `42` and max steps `100`.
   - Click **Run Flight Simulation**.
   - Use the **Step Scrubber** to inspect the drone navigating around obstacle spheres towards the green target diamond.
   - Toggle **LiDAR sensor beams** to show real-time 16-ray geometric distance ray-casts.
2. **Training Panel**:
   - Show how PPO hyperparameters (learning rate, timesteps, seed) can be tuned.
   - Run a short demo training loop directly inside the browser.
3. **Evaluation Panel**:
   - Run 10-20 deterministic evaluation episodes.
   - Inspect the interactive return distribution chart and metrics summary.
4. **Architecture Panel**:
   - Walk evaluators through the 3D kinematic equations, 29-dimensional observation vector, and reward formulation.

