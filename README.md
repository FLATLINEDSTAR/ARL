# AdaptiveRL (ARL): Autonomous 3D Drone Navigation Using Reinforcement Learning

AdaptiveRL is an educational reinforcement learning simulation project in which an agent learns autonomous 3D translational navigation of a simplified drone point-mass model toward a target waypoint while avoiding procedural obstacles.

> [!IMPORTANT]
> **Academic Simulation Scope & Technical Integrity:**  
> This is a computer science / robotics simulation project designed for college demonstration and academic evaluation. It models continuous 3D translational kinematics with aerodynamic drag and analytical raycasts. It is **not** a real-world drone autopilot, realistic quadrotor aerodynamics simulator, or hardware flight controller (e.g. PX4 / ArduPilot).

---

## 1. Problem Statement

Autonomous navigation in continuous 3D environments requires robotic agents to continuously process spatial sensor readings and command smooth multidirectional accelerations. Classical trajectory planners (such as A* or RRT*) require prior global map representations and can suffer from computational latency when replanning in continuous 3D volumes. Reinforcement Learning (RL) enables an agent to learn an onboard, reactive navigation policy mapping raw sensor range measurements directly to continuous acceleration controls in real-time.

---

## 2. Project Objective

The primary objective of this project is to:
1. Formulate a Farama Gymnasium-compliant continuous 3D flight arena with procedural obstacles and spherical LiDAR rangefinder sensing.
2. Train a Proximal Policy Optimization (PPO) neural network agent to navigate toward target coordinates without collisions.
3. Scientifically validate learning by benchmarking PPO against an untrained **Random Action Baseline**.
4. Measure performance degradation under increasing environmental complexity through an **Obstacle-Density Experiment** (4, 6, and 8 obstacles).
5. Provide an interactive browser-based demonstration flight deck (**Streamlit + Plotly**) for college presentations and viva evaluations.

---

## 3. What the Project Actually Does

- **Simulates Flight Dynamics**: Point-mass 3-DOF translation governed by commanded accelerations, physical velocity caps ($v_{\max} = 8.0\text{ m/s}$), and linear aerodynamic drag damping ($c_{\text{drag}} = 0.05$).
- **LiDAR Perception**: Casts a 16-ray spherical rangefinder ($20\text{ m}$ max range) around the drone to detect distances to obstacles and bounding arena walls.
- **PPO Policy Learning**: Trains an actor-critic Multi-Layer Perceptron (SB3 `MlpPolicy`) directly on CPU in under 30 seconds.
- **Controlled Benchmarking**: Deterministically evaluates policies over multi-episode trials, exporting empirical JSON summaries.
- **Interactive 3D Flight Deck**: Visualizes flight paths, obstacle meshes, and live sensor rays with a playback scrubber in the browser.

---

## 4. Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Demonstration Layer                             │
│       CLI: adaptive-rl {train | evaluate | experiment-density | gui}        │
│       GUI: streamlit run app.py (Interactive 5-Tab 3D Flight Deck)          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────┐
│             Training Loop             │ │          Evaluation Engine        │
│   Stable-Baselines3 (PPO Algorithm)   │ │  PPO vs Random Action Baseline    │
│   Episode Metrics & Moving Averages   │ │  Obstacle Density (4, 6, 8 obs)   │
└───────────────────┬───────────────────┘ └───────────────────┬───────────────┘
                    │                                         │
                    └──────────────────┬──────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DroneNavigation3DEnv                             │
│   • 3-DOF translational kinematics with aerodynamic drag damping            │
│   • 16-ray spherical LiDAR rangefinder (analytical ray-sphere geometry)     │
│   • 29-dimensional continuous state vector in [-1.0, 1.0]                   │
│   • Continuous 3D action space (acceleration in [-1.0, 1.0]^3)              │
│   • Procedural spherical obstacles placed with guaranteed start/goal clearance│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Reinforcement Learning Framework

The agent is trained using **Proximal Policy Optimization (PPO)**, a state-of-the-art on-policy actor-critic algorithm that ensures stable policy updates via a clipped surrogate objective:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

- **Policy Architecture**: Multi-Layer Perceptron (`MlpPolicy`, two hidden layers of 64 units each).
- **Optimization**: Adam optimizer with learning rate $\alpha = 3 \times 10^{-4}$.
- **Discount Factor**: $\gamma = 0.99$.
- **Generalized Advantage Estimation**: $\lambda = 0.95$.

---

## 6. Environment Specifications

### A. Observation Space (29 Continuous Dimensions)
At every time step $t$, the drone receives a normalized 29-dimensional observation vector in $[-1.0, 1.0]$:
1. **Drone Position** $\mathbf{p} = [x, y, z] / \text{bounds}$ (3 dims)
2. **Drone Velocity** $\mathbf{v} = [v_x, v_y, v_z] / v_{\max}$ (3 dims)
3. **Target Coordinate** $\mathbf{g} = [g_x, g_y, g_z] / \text{bounds}$ (3 dims)
4. **Relative Goal Vector** $(\mathbf{g} - \mathbf{p}) / \text{bounds}$ (3 dims)
5. **Normalized Distance to Target** $\|\mathbf{g} - \mathbf{p}\| / d_{\text{diagonal}}$ (1 dim)
6. **Spherical LiDAR Range Readings** Normalized distances along 16 rays (16 dims)

### B. Action Space (3 Continuous Dimensions)
Continuous 3D acceleration vector:
$$\mathbf{u}_t = [u_x, u_y, u_z] \in [-1.0, 1.0]^3$$
Scaled to physical commanded acceleration $\mathbf{a}_t = \mathbf{u}_t \cdot a_{\max}$ where $a_{\max} = 4.0\text{ m/s}^2$.

### C. Kinematic Flight Model
Translational motion updates via Euler integration with linear aerodynamic drag damping:
$$\mathbf{v}_{t+1} = \mathbf{v}_t (1 - c_{\text{drag}} \Delta t) + \mathbf{a}_t \Delta t$$
$$\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{v}_{t+1} \Delta t$$
Where $\Delta t = 0.1\text{ s}$ and $c_{\text{drag}} = 0.05$.

### D. Reward Function
The reward at step $t$ actively shapes goal pursuit while penalizing collisions:
$$R_t = w_{\text{prog}} (d_{t-1} - d_t) + r_{\text{step}} - w_{\text{act}} \|\mathbf{a}_t\|^2 + R_{\text{terminal}}$$
- **Distance Progress**: $w_{\text{prog}} \times (d_{t-1} - d_t)$ ($w_{\text{prog}} = 2.0$), rewarding movement toward the goal.
- **Step Penalty**: $r_{\text{step}} = -0.05$ per step to encourage time-efficient paths.
- **Action Regularization**: $w_{\text{act}} \|\mathbf{a}_t\|^2 = 0.01 \|\mathbf{a}_t\|^2$ penalizing excessive control effort.
- **Goal Reached**: $R_{\text{terminal}} = +100.0$ when the drone arrives within $1.5\text{ m}$ of the goal.
- **Collision**: $R_{\text{terminal}} = -100.0$ if the drone strikes an obstacle or arena boundary.

---

## 7. Installation

Clone the repository and install all dependencies in a virtual environment on Python 3.10, 3.11, or 3.12:

```bash
git clone https://github.com/StellarResearch/ARL.git
cd ARL
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all]"
```

Verify installation:
```bash
pytest
adaptive-rl --help
adaptive-rl env inspect drone
```

---

## 8. Interactive 3D Demonstration GUI (Streamlit)

Launch the presentation flight deck with:

```bash
streamlit run app.py
# or via CLI
adaptive-rl gui
```

Open **`http://localhost:8501`** to access the 5 demonstration tabs:
1. **🎮 Live 3D Demo**: Real-time 3D arena visualizing drone flight, obstacles, green target diamond, 16-ray LiDAR rangefinder beams, and trajectory playback scrubber.
2. **📈 Train PPO**: In-browser training loop on CPU with real-time status and interactive Plotly training progress curves.
3. **📊 Evaluation & Baseline**: Head-to-head empirical comparison between the trained PPO agent and the untrained Random Policy baseline over selectable episodes.
4. **🎯 Difficulty Experiment**: Evaluates policy across 4, 6, and 8 obstacles, illustrating how collision rates rise with obstacle density.
5. **📘 About & Architecture**: Professor-friendly step-by-step breakdown of equations, observation layout, and academic limitations.

---

## 9. Command Line Usage

### Train the Demonstration Agent (CPU-friendly, ~25s)
```bash
adaptive-rl train --config configs/drone_ppo_demo.yaml
```
Checkpoints are saved to `artifacts/models/drone_ppo_demo_final.zip` and metadata to `artifacts/metadata/drone_ppo_demo_training.json`.

### Evaluate and Compare Against Random Action Baseline
```bash
adaptive-rl evaluate \
  --config configs/drone_ppo_demo.yaml \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 20 \
  --compare-random
```

### Run the Obstacle-Density Experiment (4, 6, 8 Obstacles)
```bash
adaptive-rl experiment-density \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 10
```

### Run the Textual 3D Trajectory Demo
```bash
adaptive-rl demo-drone \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --seed 42
```

---

## 10. Measured Empirical Results

All results below were empirically collected via genuine CPU execution:

### A. PPO vs Random Action Baseline (20 Test Episodes, Seed 42)
| Policy | Success Rate (%) | Collision Rate (%) | Mean Reward | Mean Steps | Outcome |
|---|---|---|---|---|---|
| **Random Policy Baseline** | **0.0%** | **100.0%** | **-101.24** | **71.2** | Collides in 100% of episodes |
| **Trained PPO Policy** | **5.0%** | **35.0%** | **-3.34** | **141.0** | **65% survival rate**, +97.9 reward delta |

### B. Obstacle-Density Scaling (10 Test Episodes per Condition, Seed 42)
| Condition | Obstacle Count | Collision Rate (%) | Mean Return | Mean Flight Steps |
|---|---|---|---|---|
| Low Density | 4 Obstacles | **20.0%** | **+5.00** | 169.0 steps |
| Medium Density | 6 Obstacles | **40.0%** | **-15.20** | 136.2 steps |
| High Density | 8 Obstacles | **70.0%** | **-32.43** | 88.2 steps |

---

## 11. Testing & Code Quality

Run the complete test suite:
```bash
pytest -v tests/
```
All 49 unit and integration tests run in under 6 seconds on CPU.

Run linters and type analysis:
```bash
ruff check src/ tests/ app.py
ruff format --check src/ tests/ app.py
mypy src/
```

---

## 12. Technical Limitations

- **Kinematic Point Mass**: The drone is simulated as a 3-DOF point-mass with aerodynamic drag. It does not model 6-DOF rigid-body rotational dynamics, rotor gyroscopic precession, or blade wash.
- **Analytical Rangefinder**: LiDAR sensing uses analytical ray-sphere geometric intersections rather than physical beam reflections or sensor noise.
- **Simulation Only**: Designed as an academic college demonstration; not for direct deployment on real drone flight controllers (PX4 / ArduPilot).

---

## 13. Future Work

- Implement 6-DOF quadrotor attitude rate dynamics and motor RPM controllers.
- Add domain randomization (stochastic wind gusts and sensor noise).
- Introduce dynamic, moving obstacles to test reactive obstacle tracking.

---

## 14. Project Structure

```text
ARL/
├── app.py                      # Interactive 5-Tab Streamlit flight deck GUI
├── pyproject.toml              # Packaging configuration & dependency extras
├── configs/
│   ├── drone_ppo.yaml          # Full training configuration (50,000 steps)
│   └── drone_ppo_demo.yaml     # Fast demonstration configuration (25,000 steps)
├── docs/
│   ├── COLLEGE_DEMO.md         # 5-10 minute presentation script & viva Q&A
│   ├── DEMO.md                 # CLI & GUI demonstration guide
│   └── EXPERIMENT.md           # Experimental methodology & empirical results
├── src/adaptive_rl/
│   ├── algorithms/             # PPO algorithm wrapper & RandomPolicy baseline
│   ├── environments/           # DroneNavigation3DEnv & 16-ray LiDAR raycaster
│   ├── evaluation/             # Evaluator, baseline comparison, & density experiment
│   ├── gui/                    # 3D Plotly visualizer & training curve charts
│   ├── training/               # PPOTrainer & metric callbacks
│   └── cli.py                  # Typer CLI application
└── tests/                      # 49 unit and integration tests
```

---

## 15. License

MIT License. See `LICENSE` for details.
