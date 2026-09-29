# Simulated 3D Drone Environment Specification

## Mathematical Model

The drone is modeled as a 3-DOF kinematic point mass subject to commanded linear acceleration and linear aerodynamic drag damping.

### State Equations

Given environment integration time step $\Delta t = 0.1\text{ s}$:

$$\mathbf{a}_t = \text{clip}(\mathbf{u}_t, -1, 1) \cdot a_{\max}$$

$$\mathbf{v}_{t+1} = \text{clip}\left(\mathbf{v}_t + (\mathbf{a}_t - c_{\text{drag}} \mathbf{v}_t) \Delta t, -v_{\max}, v_{\max}\right)$$

$$\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{v}_{t+1} \Delta t$$

Where:
- $a_{\max} = 4.0\text{ m/s}^2$: Maximum linear acceleration command magnitude
- $v_{\max} = 8.0\text{ m/s}$: Maximum linear speed cap
- $c_{\text{drag}} = 0.05\text{ s}^{-1}$: Linear aerodynamic drag damping coefficient

---

## Sensor Model (16-Ray 3D LiDAR)

The simulated drone carries a 16-ray spherical rangefinder:
- 8 rays in the horizontal plane ($0^\circ, 45^\circ, \dots, 315^\circ$)
- 4 rays elevated at $+45^\circ$ ($0^\circ, 90^\circ, 180^\circ, 270^\circ$)
- 4 rays inclined at $-45^\circ$ ($0^\circ, 90^\circ, 180^\circ, 270^\circ$)

Ray-sphere and ray-box intersection tests are computed analytically against each spherical obstacle and arena boundary up to a maximum range of $20.0\text{ m}$. Readings are normalized to $[0.0, 1.0]$ where $0.0$ indicates immediate proximity and $1.0$ indicates free space or maximum range.

The observation vector has 29 dimensions:
- `[0:3]`: Normalized 3D position $[x/X_{\max}, y/Y_{\max}, z/Z_{\max}]$ in $[0, 1]$
- `[3:6]`: Normalized 3D velocity $[v_x/v_{\max}, v_y/v_{\max}, v_z/v_{\max}]$ in $[-1, 1]$
- `[6:9]`: Normalized target position $[g_x/X_{\max}, g_y/Y_{\max}, g_z/Z_{\max}]$ in $[0, 1]$
- `[9:12]`: Relative target vector in $[-1, 1]$
- `[12]`: Normalized Euclidean distance to target in $[0, 1]$
- `[13:29]`: 16-ray normalized LiDAR distance readings in $[0, 1]$

---

## LiDAR Sensor Noise & Dropout

The 3D drone environment supports configurable LiDAR measurement noise, beam dropout, and minimum detection range (Issue #254).

### Configuration

```yaml
environment:
  name: "drone"
  parameters:
    lidar_noise_std: 0.05
    lidar_dropout_prob: 0.02
    lidar_min_range: 0.2
```

- `lidar_noise_std`: Standard deviation of zero-mean Gaussian ranging noise in meters ($d_{\text{noisy}} = d + \mathcal{N}(0, \sigma^2)$). Default: `0.0` (noiseless).
- `lidar_dropout_prob`: Probability in $[0, 1]$ that an individual beam fails to return an echo. Dropped beams report `max_range` (normalized `1.0`). Default: `0.0` (no dropout).
- `lidar_min_range`: Minimum detectable distance in meters (blind zone). Distances below this threshold are clamped to `lidar_min_range`. Default: `0.0`.

All normalized readings are clipped to $[0.0, 1.0]$. The environment utilizes its seeded NumPy RNG (`self.np_random`) for stochastic sensor behavior, ensuring strict reproducibility when reset with a fixed seed.

### Recommended Usage
- **Noiseless baseline**: `lidar_noise_std: 0.0`, `lidar_dropout_prob: 0.0`, `lidar_min_range: 0.0` (exact backwards compatibility preserved).
- **Robustness training**: `lidar_noise_std: 0.05`, `lidar_dropout_prob: 0.02`, `lidar_min_range: 0.2`.

---

## Reward Formulation

At each step $t$:

$$R_t = w_{\text{progress}} (d_{t-1} - d_t) + c_{\text{step}} - c_{\text{action}} \|\mathbf{u}_t\|^2 + R_{\text{terminal}}$$

Where:
- $d_t = \|\mathbf{p}_t - \mathbf{g}\|_2$: Euclidean distance to target
- $w_{\text{progress}} = 2.0$: Positive reward for moving toward the target
- $c_{\text{step}} = -0.05$: Step time penalty
- $c_{\text{action}} = 0.01$: Effort penalty on acceleration command magnitude squared
- $R_{\text{terminal}}$:
  - $+100.0$ if reached target ($d_t \le r_{\text{target}} = 1.5\text{ m}$)
  - $-100.0$ if collided with obstacle or arena boundary ($r_{\text{collision}} = 0.8\text{ m}$)

---

## Trajectory-Quality & Safety Evaluation Metrics

The evaluation engine computes deterministic trajectory-quality and safety metrics across evaluation rollouts:

### Trajectory-Quality Metrics

- **Path length ($L$)** [$\text{m}$]: Cumulative Euclidean distance traveled across trajectory waypoints:
  $$L = \sum_{t=0}^{T-1} \|\mathbf{p}_{t+1} - \mathbf{p}_t\|_2$$
- **Straight-line distance ($D_0$)** [$\text{m}$]: Euclidean distance from initial drone position to target:
  $$D_0 = \|\mathbf{g} - \mathbf{p}_0\|_2$$
- **Path efficiency ($\eta$)** [dimensionless]: Ratio of straight-line distance to actual path length, clamped to $[0.0, 1.0]$:
  $$\eta = \begin{cases} \text{clip}\left(\frac{D_0}{L}, 0.0, 1.0\right) & \text{if } L > 0 \\ 0.0 & \text{if } L \le 0 \end{cases}$$

### Safety & Dynamics Metrics

- **Minimum obstacle clearance ($d_{\min}$)** [$\text{m}$]: Closest distance from drone position to any spherical obstacle surface over the rollout:
  $$d_{\min} = \min_{t, i} \left(\|\mathbf{p}_t - \mathbf{c}_i\|_2 - r_i\right)$$
- **Maximum velocity ($v_{\max}$)** [$\text{m/s}$]: Peak instantaneous linear speed attained during rollout:
  $$v_{\max} = \max_t \|\mathbf{v}_t\|_2$$
- **Maximum acceleration ($a_{\max}$)** [$\text{m/s}^2$]: Peak instantaneous linear acceleration magnitude:
  $$a_{\max} = \max_t \|\mathbf{a}_t\|_2$$
- **Obstacle collisions**: Count and rate of terminal impacts against spherical obstacle surfaces.
- **Boundary collisions**: Count and rate of terminal impacts against arena bounding planes ($X, Y, Z$).
