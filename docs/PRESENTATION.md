# College Project Presentation Outline

### Slide 1: Title & Problem Statement
- **Title**: Autonomous 3D Drone Navigation via Reinforcement Learning
- **Problem**: Navigating a simulated drone from origin to coordinate target avoiding obstacles in continuous 3D space.

### Slide 2: Kinematic Simulation & Physics Model
- 3-DOF translation with semi-implicit Euler integration
- Aerodynamic drag damping ($c_{drag}=0.05$)
- 16-ray spherical LiDAR rangefinder for spherical obstacles

### Slide 3: Reinforcement Learning Architecture
- Algorithm: Proximal Policy Optimization (PPO) via Stable-Baselines3
- State Space: 29-dimensional normalized vector (positions, velocities, goal vector, relative target vector, distance ratio, 16 LiDAR ranges)
- Action Space: Continuous 3D thrust acceleration $[-1.0, 1.0]^3$
- Reward Formulation: Distance-progress shaping with collision and step penalties

### Slide 4: Experimental Evaluation & Demonstration
- 25k demonstration budget: ~25 seconds on student laptop CPU
- Deterministic flight demonstration with seed 42
- Live terminal metrics and ASCII trajectory trace
