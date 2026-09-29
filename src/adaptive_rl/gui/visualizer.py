"""Plotly-based 3D visual arena and sensor visualizer for AdaptiveRL."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from adaptive_rl.environments.drone import ObstacleSphere3D


def _generate_sphere_surface(
    center: np.ndarray | Sequence[float],
    radius: float,
    n_points: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate X, Y, Z coordinate matrices for a 3D sphere surface."""
    u = np.linspace(0.0, 2.0 * np.pi, n_points)
    v = np.linspace(0.0, np.pi, n_points)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def _generate_box_wireframe(
    bounds: Tuple[float, float, float],
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Generate line coordinates for the 12 edges of the 3D arena boundary box."""
    bx, by, bz = bounds
    # Define 8 corners
    # 0: 0,0,0; 1: bx,0,0; 2: bx,by,0; 3: 0,by,0
    # 4: 0,0,bz; 5: bx,0,bz; 6: bx,by,bz; 7: 0,by,bz
    c = [
        (0.0, 0.0, 0.0),
        (bx, 0.0, 0.0),
        (bx, by, 0.0),
        (0.0, by, 0.0),
        (0.0, 0.0, bz),
        (bx, 0.0, bz),
        (bx, by, bz),
        (0.0, by, bz),
    ]

    edges = [
        # Bottom perimeter
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        # Top perimeter
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        # Vertical pillars
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []

    for start_idx, end_idx in edges:
        xs.extend([c[start_idx][0], c[end_idx][0], None])
        ys.extend([c[start_idx][1], c[end_idx][1], None])
        zs.extend([c[start_idx][2], c[end_idx][2], None])

    return xs, ys, zs


def build_arena_3d_figure(
    bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0),
    obstacles: Optional[Sequence[ObstacleSphere3D]] = None,
    target: Optional[np.ndarray | Sequence[float]] = None,
    start_pos: Optional[np.ndarray | Sequence[float]] = None,
    trajectory: Optional[Sequence[np.ndarray | Sequence[float]]] = None,
    current_pos: Optional[np.ndarray | Sequence[float]] = None,
    current_vel: Optional[np.ndarray | Sequence[float]] = None,
    attitude_quaternion: Optional[Sequence[float]] = None,
    wind_vector: Optional[Sequence[float]] = None,
    lidar_rays: Optional[np.ndarray] = None,
    lidar_ranges: Optional[Sequence[float] | np.ndarray] = None,
    show_lidar: bool = True,
    title: str = "3D Drone Flight Arena",
) -> go.Figure:
    """Build an interactive 3D Plotly flight arena visualizer.

    Visualizes:
    - Bounded flight volume with wireframe arena walls
    - Procedural 3D spherical obstacles
    - Goal target waypoint and launch start position
    - Real-time or historical drone trajectory
    - Current drone 3D position and velocity vector
    - 16-ray spherical LiDAR sensor beams and contact points
    """
    fig = go.Figure()

    # 1. Arena Bounding Box Wireframe
    bx, by, bz = bounds
    box_x, box_y, box_z = _generate_box_wireframe(bounds)
    fig.add_trace(
        go.Scatter3d(
            x=box_x,
            y=box_y,
            z=box_z,
            mode="lines",
            line=dict(color="#4a5568", width=3, dash="dot"),
            name="Arena Boundary",
            hoverinfo="skip",
        )
    )

    # 2. Obstacles (3D Spheres)
    if obstacles:
        for idx, obs in enumerate(obstacles):
            ox, oy, oz = _generate_sphere_surface(obs.center, obs.radius, n_points=14)
            # Custom red colorscale for obstacle spheres
            red_scale = [[0.0, "#e53e3e"], [1.0, "#c53030"]]
            fig.add_trace(
                go.Surface(
                    x=ox,
                    y=oy,
                    z=oz,
                    colorscale=red_scale,
                    showscale=False,
                    opacity=0.65,
                    name=f"Obstacle {idx + 1}",
                    hoverinfo="text",
                    text=f"Obstacle {idx + 1}<br>Pos: [{obs.center[0]:.1f}, {obs.center[1]:.1f}, {obs.center[2]:.1f}]<br>Radius: {obs.radius:.1f}m",
                )
            )

    # 3. Target Waypoint
    if target is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[target[0]],
                y=[target[1]],
                z=[target[2]],
                mode="markers+text",
                marker=dict(
                    symbol="diamond",
                    size=12,
                    color="#48bb78",
                    line=dict(color="#ffffff", width=2),
                ),
                text=["TARGET"],
                textposition="top center",
                textfont=dict(color="#48bb78", size=13),
                name="Target Waypoint",
                hoverinfo="text",
                hovertext=f"Target Goal<br>Pos: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]",
            )
        )

    # 4. Start Position
    if start_pos is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[start_pos[0]],
                y=[start_pos[1]],
                z=[start_pos[2]],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=8,
                    color="#4299e1",
                    line=dict(color="#ffffff", width=1),
                ),
                name="Start Position",
                hoverinfo="text",
                hovertext=f"Start<br>Pos: [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}]",
            )
        )

    # 5. Trajectory Line
    if trajectory and len(trajectory) > 1:
        traj_arr = np.asarray(trajectory, dtype=np.float64)
        fig.add_trace(
            go.Scatter3d(
                x=traj_arr[:, 0],
                y=traj_arr[:, 1],
                z=traj_arr[:, 2],
                mode="lines",
                line=dict(color="#ecc94b", width=5),
                name="Flight Path",
                hoverinfo="text",
                hovertext=[
                    f"Step {i}<br>Pos: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]"
                    for i, p in enumerate(traj_arr)
                ],
            )
        )

    # 6. LiDAR Beams
    if (
        show_lidar
        and current_pos is not None
        and lidar_rays is not None
        and lidar_ranges is not None
    ):
        c_pos = np.asarray(current_pos, dtype=np.float64)
        lx: List[Optional[float]] = []
        ly: List[Optional[float]] = []
        lz: List[Optional[float]] = []

        for ray_dir, dist in zip(lidar_rays, lidar_ranges):
            end_pt = c_pos + ray_dir * dist
            lx.extend([c_pos[0], end_pt[0], None])
            ly.extend([c_pos[1], end_pt[1], None])
            lz.extend([c_pos[2], end_pt[2], None])

        fig.add_trace(
            go.Scatter3d(
                x=lx,
                y=ly,
                z=lz,
                mode="lines",
                line=dict(color="rgba(0, 255, 255, 0.45)", width=2),
                name="16-Ray LiDAR",
                hoverinfo="skip",
            )
        )

    # 7. Current Drone Position & Velocity
    if current_pos is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[current_pos[0]],
                y=[current_pos[1]],
                z=[current_pos[2]],
                mode="markers+text",
                marker=dict(
                    symbol="cross",
                    size=10,
                    color="#ed8936",
                    line=dict(color="#ffffff", width=2),
                ),
                text=["DRONE"],
                textposition="bottom center",
                textfont=dict(color="#ed8936", size=13),
                name="Drone",
                hoverinfo="text",
                hovertext=f"Drone<br>Pos: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]",
            )
        )

        # Velocity Vector Arrow
        if current_vel is not None and float(np.linalg.norm(current_vel)) > 0.1:
            vel_endpoint = [
                current_pos[0] + current_vel[0] * 0.8,
                current_pos[1] + current_vel[1] * 0.8,
                current_pos[2] + current_vel[2] * 0.8,
            ]
            fig.add_trace(
                go.Scatter3d(
                    x=[current_pos[0], vel_endpoint[0]],
                    y=[current_pos[1], vel_endpoint[1]],
                    z=[current_pos[2], vel_endpoint[2]],
                    mode="lines",
                    line=dict(color="#ed8936", width=4),
                    name="Velocity Vector",
                    hoverinfo="skip",
                )
            )

        # 6-DOF Rigid-body Attitude Coordinate Frame (Roll/Pitch/Yaw)
        if attitude_quaternion is not None and len(attitude_quaternion) == 4:
            qw, qx, qy, qz = [float(v) for v in attitude_quaternion]
            rot_mat = np.array(
                [
                    [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
                ]
            )
            axis_len = 1.2
            axes_colors = [
                ("#ef4444", "Body X (Roll)"),
                ("#22c55e", "Body Y (Pitch)"),
                ("#3b82f6", "Body Z (Yaw)"),
            ]
            for idx, (col, ax_name) in enumerate(axes_colors):
                vec = rot_mat[:, idx] * axis_len
                fig.add_trace(
                    go.Scatter3d(
                        x=[current_pos[0], current_pos[0] + vec[0]],
                        y=[current_pos[1], current_pos[1] + vec[1]],
                        z=[current_pos[2], current_pos[2] + vec[2]],
                        mode="lines",
                        line=dict(color=col, width=4),
                        name=ax_name,
                        hoverinfo="name",
                    )
                )

        # Environmental Wind Vector
        if wind_vector is not None and float(np.linalg.norm(wind_vector)) > 0.05:
            wx, wy, wz = [float(v) for v in wind_vector]
            # Draw wind indicator near the top corner
            w_origin = [bx * 0.1, by * 0.1, bz * 0.85]
            w_scale = 0.5
            fig.add_trace(
                go.Scatter3d(
                    x=[w_origin[0], w_origin[0] + wx * w_scale],
                    y=[w_origin[1], w_origin[1] + wy * w_scale],
                    z=[w_origin[2], w_origin[2] + wz * w_scale],
                    mode="lines+markers",
                    marker=dict(size=[0, 4], color="#38bdf8"),
                    line=dict(color="#38bdf8", width=5),
                    name=f"Ambient Wind ({np.linalg.norm(wind_vector):.1f} m/s)",
                    hoverinfo="name",
                )
            )

    # Configure 3D Camera, Lighting, and Dark Flight Deck Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        title=dict(text=title, font=dict(size=18, color="#f7fafc"), x=0.05, y=0.95),
        scene=dict(
            xaxis=dict(
                title="X Distance (m)",
                range=[0, bx],
                backgroundcolor="#1a202c",
                gridcolor="#2d3748",
                showbackground=True,
                zerolinecolor="#4a5568",
            ),
            yaxis=dict(
                title="Y Distance (m)",
                range=[0, by],
                backgroundcolor="#1a202c",
                gridcolor="#2d3748",
                showbackground=True,
                zerolinecolor="#4a5568",
            ),
            zaxis=dict(
                title="Z Altitude (m)",
                range=[0, bz],
                backgroundcolor="#1a202c",
                gridcolor="#2d3748",
                showbackground=True,
                zerolinecolor="#4a5568",
            ),
            aspectratio=dict(x=1.0, y=1.0, z=bz / max(bx, by)),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(17, 24, 39, 0.7)",
            font=dict(color="#e2e8f0"),
        ),
    )

    return fig


def get_available_models(models_dir: str | Path = "artifacts/models") -> List[Path]:
    """Scan the artifacts directory for saved PPO model checkpoints."""
    path = Path(models_dir)
    if not path.exists():
        return []
    return sorted(list(path.glob("*.zip")), key=lambda p: p.stat().st_mtime, reverse=True)


def run_drone_simulation_episode(
    env: Any,
    model: Optional[Any] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute a single deterministic simulation episode and capture full trajectory."""
    obs, info = env.reset(seed=seed)
    trajectory: List[np.ndarray] = [info["position"].copy()]
    steps_data: List[Dict[str, Any]] = []

    total_reward = 0.0
    terminated = False
    truncated = False
    step_count = 0

    steps_data.append(
        {
            "step": 0,
            "position": info["position"].copy(),
            "velocity": info["velocity"].copy(),
            "speed": float(info.get("speed", 0.0)),
            "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
            "min_obstacle_distance": float(info.get("min_obstacle_distance", 0.0)),
            "reward": 0.0,
            "cumulative_reward": 0.0,
            "lidar_ranges": (obs[13 : 13 + env.num_lidar_rays] * env.lidar_range).copy(),
            "action": np.zeros(3, dtype=np.float32),
            "collision": False,
            "is_success": False,
            "quaternion": info.get("quaternion"),
            "wind_vector": info.get("wind_vector"),
        }
    )

    while not (terminated or truncated) and step_count < env.max_steps:
        step_count += 1
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        pos = info["position"].copy()
        vel = info["velocity"].copy()
        trajectory.append(pos)

        raw_lidar_norm = obs[13 : 13 + env.num_lidar_rays]
        lidar_ranges = (raw_lidar_norm * env.lidar_range).copy()

        steps_data.append(
            {
                "step": step_count,
                "position": pos,
                "velocity": vel,
                "speed": float(info.get("speed", 0.0)),
                "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
                "min_obstacle_distance": float(info.get("min_obstacle_distance", 0.0)),
                "reward": float(reward),
                "cumulative_reward": float(total_reward),
                "lidar_ranges": lidar_ranges,
                "action": np.asarray(action, dtype=np.float32).copy(),
                "collision": bool(info.get("collision", False)),
                "is_success": bool(info.get("is_success", False) or info.get("success", False)),
                "quaternion": info.get("quaternion"),
                "wind_vector": info.get("wind_vector"),
            }
        )

    outcome = "SUCCESS" if steps_data[-1]["is_success"] else "COLLISION / TIMEOUT"

    return {
        "trajectory": trajectory,
        "steps": steps_data,
        "total_reward": total_reward,
        "total_steps": step_count,
        "outcome": outcome,
        "seed": seed,
        "obstacles": getattr(env, "_obstacles", getattr(env, "obstacles", [])),
        "target": getattr(
            env, "target", getattr(env, "_goal", getattr(env, "goal", np.zeros(3)))
        ).copy(),
        "bounds": env.bounds,
        "start_pos": getattr(env, "default_start", getattr(env, "_start_pos", np.zeros(3))).copy(),
        "lidar_rays": env.lidar_rays,
        "wind_vector": info.get("wind_vector"),
        "quaternion": info.get("quaternion"),
    }


def build_training_curve_figure(
    episode_rewards: Sequence[float],
    window_size: int = 10,
    title: str = "PPO Training Progress",
) -> go.Figure:
    """Construct an interactive Plotly training curve with rolling average."""
    fig = go.Figure()
    if not episode_rewards:
        fig.update_layout(
            template="plotly_dark",
            title=title,
            annotations=[
                dict(
                    text="No training episodes recorded yet.",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                )
            ],
        )
        return fig

    episodes = list(range(1, len(episode_rewards) + 1))
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=list(episode_rewards),
            mode="lines",
            line=dict(color="rgba(100, 180, 255, 0.4)", width=1.5),
            name="Episode Return",
        )
    )

    if len(episode_rewards) >= window_size:
        rolling = np.convolve(
            np.asarray(episode_rewards, dtype=float),
            np.ones(window_size) / window_size,
            mode="valid",
        )
        rolling_x = list(range(window_size, len(episode_rewards) + 1))
        fig.add_trace(
            go.Scatter(
                x=rolling_x,
                y=rolling.tolist(),
                mode="lines",
                line=dict(color="#00E5FF", width=3),
                name=f"Moving Avg (w={window_size})",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Episode",
        yaxis_title="Return (Cumulative Reward)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_comparison_bar_chart(
    metrics_map: Dict[str, Any],
    title: str = "Policy Performance Comparison",
) -> go.Figure:
    """Build side-by-side bar chart comparing PPO vs Random Policy."""
    fig = go.Figure()
    policies = list(metrics_map.keys())
    if not policies:
        return fig

    categories = ["Success Rate (%)", "Collision Rate (%)"]
    colors = ["#00E676", "#FF5252", "#FFD700", "#7C4DFF"]

    for i, policy_name in enumerate(policies):
        m = metrics_map[policy_name]
        succ = (m.success_rate if hasattr(m, "success_rate") else m.get("success_rate", 0.0)) or 0.0
        coll = (
            m.collision_rate if hasattr(m, "collision_rate") else m.get("collision_rate", 0.0)
        ) or 0.0

        fig.add_trace(
            go.Bar(
                name=policy_name,
                x=categories,
                y=[succ * 100.0, coll * 100.0],
                marker_color=colors[i % len(colors)],
                text=[f"{succ * 100:.1f}%", f"{coll * 100:.1f}%"],
                textposition="auto",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        title=title,
        yaxis=dict(title="Percentage (%)", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_density_experiment_figure(
    density_results: List[Dict[str, Any]],
    title: str = "Obstacle Density vs Navigation Performance",
) -> go.Figure:
    """Build performance visualization for the obstacle-density experiment."""
    fig = go.Figure()
    if not density_results:
        return fig

    x_labels = [f"{d['obstacle_count']} Obstacles" for d in density_results]
    success_rates = [(d.get("success_rate", 0.0) or 0.0) * 100.0 for d in density_results]
    collision_rates = [(d.get("collision_rate", 0.0) or 0.0) * 100.0 for d in density_results]

    fig.add_trace(
        go.Bar(
            name="Success Rate (%)",
            x=x_labels,
            y=success_rates,
            marker_color="#00E676",
            text=[f"{s:.1f}%" for s in success_rates],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Collision Rate (%)",
            x=x_labels,
            y=collision_rates,
            marker_color="#FF5252",
            text=[f"{c:.1f}%" for c in collision_rates],
            textposition="auto",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        title=title,
        yaxis=dict(title="Rate (%)", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_distribution_shift_trajectory_figure(
    fixed_trajectory: Sequence[Sequence[float] | np.ndarray],
    adaptive_trajectory: Sequence[Sequence[float] | np.ndarray],
    bounds: Tuple[float, float, float] = (50.0, 50.0, 25.0),
    obstacles: Optional[Sequence[ObstacleSphere3D]] = None,
    start_pos: Optional[Sequence[float] | np.ndarray] = None,
    target: Optional[Sequence[float] | np.ndarray] = None,
    wind_vector: Optional[Sequence[float]] = None,
    title: str = "Post-Shift Flight Comparison: Fixed Policy vs Adaptive Policy",
) -> go.Figure:
    """Build comparative 3D arena figure showing Fixed vs Adaptive trajectories under shift."""
    fig = build_arena_3d_figure(
        bounds=bounds,
        obstacles=obstacles,
        target=target,
        start_pos=start_pos,
        wind_vector=wind_vector,
        show_lidar=False,
        title=title,
    )

    # Fixed Policy Trajectory (Orange)
    if fixed_trajectory and len(fixed_trajectory) > 1:
        fx = [float(p[0]) for p in fixed_trajectory]
        fy = [float(p[1]) for p in fixed_trajectory]
        fz = [float(p[2]) for p in fixed_trajectory]
        fig.add_trace(
            go.Scatter3d(
                x=fx,
                y=fy,
                z=fz,
                mode="lines+markers",
                marker=dict(size=2, color="#f97316"),
                line=dict(color="#f97316", width=5, dash="dash"),
                name="Fixed Policy (No Adaptation)",
            )
        )

    # Adaptive Policy Trajectory (Cyan)
    if adaptive_trajectory and len(adaptive_trajectory) > 1:
        ax = [float(p[0]) for p in adaptive_trajectory]
        ay = [float(p[1]) for p in adaptive_trajectory]
        az = [float(p[2]) for p in adaptive_trajectory]
        fig.add_trace(
            go.Scatter3d(
                x=ax,
                y=ay,
                z=az,
                mode="lines+markers",
                marker=dict(size=3, color="#06b6d4"),
                line=dict(color="#06b6d4", width=5),
                name="Adaptive Policy (Online PPO)",
            )
        )

    return fig


def build_recovery_curve_figure(
    episodes: Sequence[int],
    fixed_returns: Sequence[float],
    adaptive_returns: Sequence[float],
    p_pre: float,
    p0: float,
    fixed_t_h: Optional[int] = None,
    adaptive_t_h: Optional[int] = None,
    title: str = "Online Adaptation Recovery Dynamics (Protocol v2.0)",
) -> go.Figure:
    """Build recovery trajectory figure comparing Fixed vs Adaptive returns across post-shift episodes."""
    fig = go.Figure()

    eps = list(episodes)
    degradation = p_pre - p0
    threshold_val = p0 + 0.9 * degradation if degradation > 0 else p_pre

    # Pre-shift baseline P_pre
    fig.add_trace(
        go.Scatter(
            x=[min(eps), max(eps)],
            y=[p_pre, p_pre],
            mode="lines",
            line=dict(color="#10b981", width=2, dash="dash"),
            name=f"Nominal Baseline P_pre ({p_pre:.1f})",
        )
    )

    # Initial Shock baseline P0
    fig.add_trace(
        go.Scatter(
            x=[min(eps), max(eps)],
            y=[p0, p0],
            mode="lines",
            line=dict(color="#ef4444", width=2, dash="dot"),
            name=f"Shock Baseline P0 ({p0:.1f})",
        )
    )

    # 90% Recovery Threshold
    if degradation > 0:
        fig.add_trace(
            go.Scatter(
                x=[min(eps), max(eps)],
                y=[threshold_val, threshold_val],
                mode="lines",
                line=dict(color="#f59e0b", width=2, dash="dashdot"),
                name=f"90% Recovery Level ({threshold_val:.1f})",
            )
        )

    # Fixed Policy Returns
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=list(fixed_returns),
            mode="lines+markers",
            marker=dict(size=7, color="#f97316"),
            line=dict(color="#f97316", width=3, dash="dash"),
            name="Fixed Arm (Frozen)",
        )
    )

    # Adaptive Policy Returns
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=list(adaptive_returns),
            mode="lines+markers",
            marker=dict(size=8, color="#06b6d4"),
            line=dict(color="#06b6d4", width=4),
            name="Adaptive Arm (Online PPO)",
        )
    )

    # Highlight shock window (episodes 1-5)
    fig.add_vrect(
        x0=0.5,
        x1=5.5,
        fillcolor="#374151",
        opacity=0.3,
        layer="below",
        line_width=0,
        annotation_text="Shared Shock Window (Ep 1-5)",
        annotation_position="top left",
    )

    # Annotate recovery points if available
    if adaptive_t_h is not None and adaptive_t_h < 15:
        idx = adaptive_t_h - 1
        if 0 <= idx < len(adaptive_returns):
            fig.add_annotation(
                x=adaptive_t_h,
                y=adaptive_returns[idx],
                text=f"Adaptive Recovered (T_H={adaptive_t_h})",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#06b6d4",
                font=dict(color="#06b6d4"),
            )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis=dict(title="Post-Shift Episode Index (k)", tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(title="Episodic Return"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig
