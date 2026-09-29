# Architecture Overview

AdaptiveRL is structured around a minimal, modular pipeline consisting of four primary components:

```text
src/adaptive_rl/
├── environments/      # 3D kinematic drone, 6-DOF rigid-body quadrotor, & disturbed drone (Gymnasium)
├── algorithms/        # RL algorithm registry and wrappers (Stable-Baselines3 PPO & SAC)
├── planners/          # Classical motion-planning baselines (A* 3D lattice planner)
├── training/          # Unified RL training loop (RLTrainer/PPOTrainer), checkpointing, and artifact export
├── evaluation/        # Multi-episode deterministic evaluator, policy comparison, and metrics
├── benchmarking/      # Controlled ablations and cross-algorithm benchmarking (PPO vs SAC)
├── experiments/       # Preregistered online adaptation vs fixed shift runner (Protocol v2.0)
├── manifest.py        # Reproducibility metadata manifest generator (Git, OS, hardware, hashes)
├── config.py          # Strongly typed Pydantic configuration schemas
└── cli.py             # User-facing Typer CLI application
```

## Component Details

### 1. Environments (`adaptive_rl.environments`)
- **`DroneNavigation3DEnv` (`drone`, `drone_3d`)**: Point-mass 3-DOF kinematic translation with semi-implicit Euler stepping and linear aerodynamic drag ($c_{\text{drag}} = 0.05$).
- **`Drone6DOFEnv` (`drone-6dof`, `drone_6dof`)**: Rigid-body 6-DOF quadrotor dynamics with quaternion attitude kinematics, Euler rotational equations, 4-rotor "X" configuration thrusts, and RK4 multi-substep numerical integration.
- **`DroneDisturbed3DEnv` (`drone_disturbed`, `drone_disturbance`)**: 3D drone navigation subject to steady prevailing wind, stochastic Ornstein-Uhlenbeck (OU) gusts, moving dynamic obstacles with boundary reflection, and step-indexed disturbance recovery tracking.
- **Observations (Standard 29 Dims)**:
  - Relative target vector (3 dims)
  - Drone velocity (3 dims)
  - Normalized drone position (3 dims)
  - Normalized target position (3 dims)
  - Target distance scalar (1 dim)
  - 16-ray spherical LiDAR rangefinder readings (16 dims) with configurable Gaussian noise and dropout.
- **Actions**: Continuous 3D acceleration commands $a \in [-1.0, 1.0]^3$ (or 4-motor thrusts in 6-DOF).

### 2. Algorithms & Training (`adaptive_rl.algorithms`, `adaptive_rl.training`)
- **PPO (`PPOAlgorithm`)**: On-policy actor-critic algorithm wrapped from Stable-Baselines3.
  - Multi-Layer Perceptron (`MlpPolicy`).
  - Generalized Advantage Estimation (GAE) with $\gamma=0.99, \lambda=0.95$.
- **SAC (`SACAlgorithm`)**: Off-policy maximum-entropy actor-critic algorithm wrapped from Stable-Baselines3.
  - Replay buffer size $100{,}000$, soft target update $\tau=0.005$, learning rate $3 \times 10^{-4}$, batch size $256$.
- **Trainer (`RLTrainer` / `PPOTrainer`)**: Generalized trainer dispatching algorithms by configuration (`config.algorithm.name`), orchestrating callbacks, periodic checkpointing, and structured metadata logging.

### 3. Evaluation & Benchmarking (`adaptive_rl.evaluation`, `adaptive_rl.benchmarking`)
- Evaluates trained checkpoints across deterministic seed sets.
- Tracks binary outcomes: Target Reached (Success), Obstacle Collision, Boundary Violation, Timeout.
- Measures trajectory metrics: path length, path efficiency, obstacle surface clearance, max velocity/acceleration.
- Provides comparative benchmarking workflow (`adaptive-rl benchmark compare-algorithms`) under strictly fair, identical environment configurations and evaluation seeds.
- Serializes evaluation reports to `artifacts/evaluation.json`, `artifacts/evaluation.csv`, and `artifacts/algorithm_comparison.json`.

### 4. Classical Planning Baseline (`adaptive_rl.planners.astar3d`)
- Implements `AStar3DPlanner`: a deterministic classical 3D motion-planning baseline operating on a spatial lattice.
- Independent of PyTorch, trained weights, and reinforcement learning dependencies.
- **Search Space & Neighborhood**: Configurable 3D lattice resolution (default: 0.5 m) supporting 6-connected (axis-aligned) or 26-connected (diagonal) motion.
- **Heuristic**: Admissible and consistent Euclidean distance $h(n) = \|pos(n) - goal\|$.
- **Analytical Clearance Geometry**:
  - Every segment $[p_0, p_1]$ is verified using exact segment-to-sphere analytical distance:
    $$t^* = \text{clamp}\left(\frac{(C - p_0) \cdot (p_1 - p_0)}{\|p_1 - p_0\|^2}, 0, 1\right)$$
    $$\text{dist}(segment, C) = \|p_0 + t^* (p_1 - p_0) - C\|$$
    $$\text{clearance} = \text{dist}(segment, C) - (R_{\text{obs}} + R_{\text{coll}})$$
  - Enforces arena boundary clearance: $\min(p_0[i], p_1[i]) - R_{\text{coll}} > 0$ and $\max(p_0[i], p_1[i]) + R_{\text{coll}} < \text{bounds}[i]$.
- **Empty-Arena Behavior**: Returns direct optimal two-point straight line $[start, goal]$ with 100% path efficiency without intermediate lattice waypoints.
- **Fair Benchmarking Adapter**: `compare_with_planner` evaluates PPO, AStar3D, and Random Policy under identical procedural seeds and exports structured comparison reports to `artifacts/evaluation_planner_comparison.json`.
- **CLI Command**:
  ```bash
  adaptive-rl evaluate --model artifacts/models/drone_ppo_demo_final.zip --compare-planner astar
  ```
- **Limitations & Feasibility Paradigms**:
  - **Dynamic Feasibility (PPO & Random Policy)**: Evaluated through closed-loop physics simulation in `DroneNavigation3DEnv`. The drone must generate continuous acceleration control commands to navigate under inertia, velocity drag, and actuation limits. Success requires dynamically executing the trajectory without colliding.
  - **Geometric Feasibility (A* Planner)**: Evaluated as open-loop 3D spatial path planning. Success denotes finding an obstacle-free, boundary-clearing geometric path from start to goal on the discretized spatial lattice. The path is not executed through drone attitude dynamics or closed-loop tracking control.
