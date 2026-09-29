# 6-DOF Rigid-Body Quadrotor Dynamics Environment (`drone-6dof`)

## 1. Overview and Scope

The `drone-6dof` environment simulates the rigid-body flight dynamics of a quadrotor in 3D space, incorporating full rotational attitude kinematics (unit quaternions), Euler rigid-body rotational equations, individual motor thrusts, aerodynamic damping, procedural 3D spherical obstacles, and 16-ray LiDAR sensing.

> [!IMPORTANT]
> **Theoretical Model Disclaimer**:
> This environment implements a canonical rigid-body flight dynamics model intended for educational, algorithmic, and reinforcement learning research. It does **not** represent a validated digital twin of any specific commercial or physical UAV hardware, and has not been tuned or validated against real-world flight telemetry.

---

## 2. Coordinate Frames & Conventions

To prevent ambiguity, the coordinate systems and conventions are defined as follows:

### 2.1 World Frame: ENU (East-North-Up)
* $+X_W$: East
* $+Y_W$: North
* $+Z_W$: Upward against gravity.
* Gravitational acceleration: $\mathbf{g} = [0, 0, -9.81]^T\text{ m/s}^2$.

### 2.2 Body Frame
Fixed to the quadrotor center of mass (CoM):
* $+X_B$: Forward along the longitudinal axis.
* $+Y_B$: Left along the lateral axis.
* $+Z_B$: Upward along the normal axis (aligned with the collective rotor thrust vector).

### 2.3 Attitude Representation: Unit Quaternion
Attitude is represented by a unit quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]^T \in \mathbb{R}^4$ with $\|\mathbf{q}\| = 1$, parameterizing the rotation from body frame to world frame:
$$\mathbf{v}_W = R(\mathbf{q}) \mathbf{v}_B$$

The direction cosine matrix $R(\mathbf{q})$ is given by:
$$R(\mathbf{q}) = \begin{bmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}$$

---

## 3. Rotor Layout & Actuation

The quadrotor operates in an "X" configuration with arm length $L = 0.2\text{ m}$. With $d = \frac{L}{\sqrt{2}} \approx 0.1414\text{ m}$:

| Rotor Index | Position $(x_B, y_B)$ | Direction | Reaction Torque |
| :--- | :--- | :--- | :--- |
| **Rotor 1 (Front-Right)** | $(+d, -d)$ | CCW | $+c_\tau F_1 \hat{z}_B$ |
| **Rotor 2 (Front-Left)** | $(+d, +d)$ | CW | $-c_\tau F_2 \hat{z}_B$ |
| **Rotor 3 (Rear-Left)** | $(-d, +d)$ | CCW | $+c_\tau F_3 \hat{z}_B$ |
| **Rotor 4 (Rear-Right)** | $(-d, -d)$ | CW | $-c_\tau F_4 \hat{z}_B$ |

### Forces and Torques
Each rotor generates thrust $F_i \in [0, F_{\max}]$ directed along $+Z_B$:
1. **Total Collective Thrust**:
   $$T = \sum_{i=1}^4 F_i$$
2. **Body Torques** $\boldsymbol{\tau} = [\tau_x, \tau_y, \tau_z]^T$:
   $$\tau_x = d (-F_1 + F_2 + F_3 - F_4) \quad \text{(Roll torque about } +X_B\text{)}$$
   $$\tau_y = d (-F_1 - F_2 + F_3 + F_4) \quad \text{(Pitch torque about } +Y_B\text{)}$$
   $$\tau_z = c_\tau (F_1 - F_2 + F_3 - F_4) \quad \text{(Yaw torque about } +Z_B\text{)}$$
   where $c_\tau = 0.01\text{ m}$ is the aerodynamic torque-to-thrust ratio.

---

## 4. Equations of Motion

### 4.1 Translational Dynamics (World Frame)
$$m \ddot{\mathbf{p}} = \begin{bmatrix} 0 \\ 0 \\ -mg \end{bmatrix} + R(\mathbf{q}) \begin{bmatrix} 0 \\ 0 \\ T \end{bmatrix} - D_v \mathbf{v}$$
where:
* $m = 1.0\text{ kg}$ is total mass.
* $D_v = 0.1\text{ N}\cdot\text{s/m}$ is the linear aerodynamic translational drag coefficient.
* $\mathbf{v} = \dot{\mathbf{p}}$ is world-frame linear velocity.

### 4.2 Rotational Dynamics (Euler Rigid-Body Equations, Body Frame)
$$I \dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times (I \boldsymbol{\omega}) - D_\omega \boldsymbol{\omega}$$
where:
* $\boldsymbol{\omega} = [p, q, r]^T$ is the body angular velocity.
* $I = \text{diag}(I_{xx}, I_{yy}, I_{zz}) = \text{diag}(0.01, 0.01, 0.02)\text{ kg}\cdot\text{m}^2$ is the moment of inertia tensor.
* $D_\omega = 0.02\text{ N}\cdot\text{m}\cdot\text{s/rad}$ is the rotational aerodynamic damping.

### 4.3 Quaternion Kinematics
$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix}$$
$$\begin{aligned}
\dot{q}_w &= -\frac{1}{2} (q_x p + q_y q + q_z r) \\
\dot{q}_x &= \frac{1}{2} (q_w p + q_y r - q_z q) \\
\dot{q}_y &= \frac{1}{2} (q_w q - q_x r + q_z p) \\
\dot{q}_z &= \frac{1}{2} (q_w r + q_x q - q_y p)
\end{aligned}$$

---

## 5. Numerical Integration & Stability

To avoid numerical drift and instability:
* The environment step is $\Delta t = 0.05\text{ s}$.
* Stepping employs **Runge-Kutta 4th-order (RK4)** numerical integration with $5$ internal substeps ($\Delta t_{\text{sub}} = 0.01\text{ s}$).
* The quaternion $\mathbf{q}$ is explicitly re-normalized after every RK4 substep:
  $$\mathbf{q} \leftarrow \frac{\mathbf{q}}{\|\mathbf{q}\|}$$
* Tests verify zero explosion and strict finite bounds over 1,000 continuous random steps.

---

## 6. Action and Observation Spaces

### 6.1 Action Space
Continuous 4-dimensional box: `Box(-1.0, 1.0, shape=(4,), dtype=float32)`.
Mapped linearly to motor thrusts:
$$F_i = \frac{a_i + 1}{2} F_{\max} \in [0, 5.0]\text{ N}$$
Hover corresponds to $F_i = \frac{mg}{4} = 2.4525\text{ N}$ ($a_i \approx -0.019$).

### 6.2 Observation Space
Continuous 36-dimensional box: `Box(-1.0, 1.0, shape=(36,), dtype=float32)`:
* `[0:3]`: Normalized position $\mathbf{p} / \text{bounds}$ in $[0, 1]^3$
* `[3:6]`: Normalized velocity $\mathbf{v} / v_{\max}$ in $[-1, 1]^3$
* `[6:10]`: Attitude quaternion $[q_w, q_x, q_y, q_z]$ in $[-1, 1]^4$
* `[10:13]`: Normalized angular velocity $\boldsymbol{\omega} / \omega_{\max}$ in $[-1, 1]^3$
* `[13:16]`: Normalized target position $\mathbf{g} / \text{bounds}$ in $[0, 1]^3$
* `[16:19]`: Relative target vector $(\mathbf{g} - \mathbf{p}) / \text{bounds}$ in $[-1, 1]^3$
* `[19]`: Normalized Euclidean distance to target in $[0, 1]$
* `[20:36]`: 16-ray 3D LiDAR normalized distances in $[0, 1]$

---

## 7. Assumptions & Limitations

1. **Rigid Airframe**: The airframe is assumed perfectly rigid with constant moments of inertia.
2. **Simplified Aerodynamics**: Drag is modeled as linear translational and rotational damping. Complex fluid dynamics, such as ground effect, blade flapping, vortex ring state, and aerodynamic downwash are omitted.
3. **Instantaneous Motor Dynamics**: Rotor thrust response is modeled without motor electrical inductance or ESC lag.
4. **Validation**: This model should be treated as an educational simulation platform rather than a physical digital twin.
