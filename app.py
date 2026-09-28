"""AdaptiveRL: Autonomous 3D Drone Navigation Demonstration GUI.

A Streamlit-based presentation interface for college professors and evaluators,
visualizing real-time 3D flight trajectories, LiDAR sensor readings, PPO training,
random baseline comparison, obstacle-density experiments, and system architecture.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
from stable_baselines3 import PPO

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.config import AlgorithmConfig, EnvironmentConfig, ExperimentConfig, TrainingConfig
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.evaluation.evaluator import (
    Evaluator,
    evaluate_ppo_policy,
    evaluate_random_policy,
    run_obstacle_density_experiment,
)
from adaptive_rl.gui import (
    build_arena_3d_figure,
    build_comparison_bar_chart,
    build_density_experiment_figure,
    build_training_curve_figure,
    get_available_models,
    run_drone_simulation_episode,
)
from adaptive_rl.training.trainer import PPOTrainer

# Configure Streamlit Page
st.set_page_config(
    page_title="AdaptiveRL — 3D Drone Navigation Demo",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    # Sidebar: Project Overview
    st.sidebar.title("🚁 AdaptiveRL")
    st.sidebar.markdown(
        "**Autonomous 3D Drone Navigation**\n\n"
        "College Demonstration Prototype using Proximal Policy Optimization (PPO).\n\n"
        "*(Point-mass kinematic simulation for academic evaluation)*"
    )

    available_models = get_available_models()
    model_names = [m.name for m in available_models]

    tabs = st.tabs(
        [
            "🎮 Live 3D Demo",
            "📈 Train PPO",
            "📊 Evaluation & Baseline",
            "🎯 Difficulty Experiment",
            "📘 About & Architecture",
        ]
    )

    # ==========================================
    # TAB 1: LIVE 3D DEMO
    # ==========================================
    with tabs[0]:
        st.subheader("Live 3D Drone Navigation Simulation")
        st.caption(
            "Observe the drone navigate from its launch position toward the green target diamond "
            "while avoiding spherical obstacles using continuous 3D acceleration actions."
        )

        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 1, 1, 1])

        with ctrl_col1:
            policy_options = ["Random Action Baseline"]
            if model_names:
                policy_options = [f"Trained PPO: {m}" for m in model_names] + policy_options

            selected_policy = st.selectbox(
                "Select Policy:",
                options=policy_options,
                index=0,
                help="Choose a trained PPO model checkpoint or test an untrained random baseline.",
            )

        with ctrl_col2:
            demo_seed = st.number_input(
                "Environment Seed:",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                help="Locks procedural obstacle placement and target location for reproducible evaluation.",
            )

        with ctrl_col3:
            num_obs = st.slider(
                "Obstacles:",
                min_value=0,
                max_value=8,
                value=4,
                step=1,
                help="Number of procedural spherical obstacles generated in the arena.",
            )

        with ctrl_col4:
            show_lidar_rays = st.checkbox("Show 16-Ray LiDAR", value=True)

        run_col1, _ = st.columns([1, 4])
        with run_col1:
            launch_clicked = st.button(
                "🚀 Run Flight Simulation", type="primary", use_container_width=True
            )

        if launch_clicked or "flight_data" not in st.session_state:
            with st.spinner("Executing simulation dynamics..."):
                env = DroneNavigation3DEnv(num_obstacles=int(num_obs))
                model_to_use: Optional[PPO] = None

                if selected_policy.startswith("Trained PPO:") and available_models:
                    raw_name = selected_policy.replace("Trained PPO: ", "").strip()
                    matching = [p for p in available_models if p.name == raw_name]
                    if matching:
                        try:
                            model_to_use = PPO.load(str(matching[0]))
                        except Exception as exc:
                            st.warning(f"Could not load checkpoint ({exc}); using random actions.")

                st.session_state["flight_data"] = run_drone_simulation_episode(
                    env=env,
                    model=model_to_use,
                    seed=int(demo_seed),
                )
                env.close()

        flight_data = st.session_state["flight_data"]
        steps = flight_data["steps"]
        total_steps = len(steps) - 1

        # Outcome Banner
        outcome_color = "#38a169" if flight_data["outcome"] == "SUCCESS" else "#e53e3e"
        st.markdown(
            f"""
            <div style="background-color: {outcome_color}22; border-left: 5px solid {outcome_color}; padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;">
                <h4 style="margin: 0; color: {outcome_color};">Flight Outcome: {flight_data["outcome"]}</h4>
                <p style="margin: 4px 0 0 0; color: #cbd5e0; font-size: 14px;">
                    Completed in <b>{flight_data["total_steps"]}</b> timesteps with cumulative reward <b>{flight_data["total_reward"]:.2f}</b> (Seed: {flight_data["seed"]}).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Scrubber Slider for Trajectory Inspection
        scrub_col, _ = st.columns([3, 1])
        with scrub_col:
            selected_step = st.slider(
                "Flight Step Playback:",
                min_value=0,
                max_value=total_steps,
                value=total_steps,
                step=1,
                help="Scrub through flight time to inspect position, speed, and sensor range.",
            )

        step_info = steps[selected_step]
        cur_pos = step_info["position"]
        cur_vel = step_info["velocity"]
        cur_lidar = step_info["lidar_ranges"]

        # Build Interactive 3D Plotly Arena
        fig = build_arena_3d_figure(
            bounds=flight_data["bounds"],
            obstacles=flight_data["obstacles"],
            target=flight_data["target"],
            start_pos=flight_data["start_pos"],
            trajectory=[s["position"] for s in steps[: selected_step + 1]],
            current_pos=cur_pos,
            current_vel=cur_vel,
            lidar_rays=flight_data["lidar_rays"],
            lidar_ranges=cur_lidar,
            show_lidar=show_lidar_rays,
            title=f"3D Flight Arena — Step {selected_step}/{total_steps} (Distance to Goal: {step_info['distance_to_goal']:.2f}m)",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Step Metrics Row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Altitude (Z)", f"{cur_pos[2]:.2f} m")
        m2.metric("Ground Speed", f"{step_info['speed']:.2f} m/s")
        m3.metric("Range to Goal", f"{step_info['distance_to_goal']:.2f} m")
        m4.metric("Nearest Obstacle", f"{step_info['min_obstacle_distance']:.2f} m")
        m5.metric("Step Reward", f"{step_info['reward']:+.2f}")
        m6.metric("Cumulative Reward", f"{step_info['cumulative_reward']:+.2f}")

    # ==========================================
    # TAB 2: TRAIN PPO
    # ==========================================
    with tabs[1]:
        st.subheader("Train PPO Policy")
        st.markdown(
            "Execute real reinforcement learning training using **Stable-Baselines3 PPO** "
            "on your CPU. Training progress and rewards are tracked in real-time."
        )

        train_col1, train_col2, train_col3 = st.columns(3)
        with train_col1:
            train_steps = st.select_slider(
                "Training Budget (Timesteps):",
                options=[5000, 10000, 25000, 50000],
                value=25000,
                help="25,000 steps requires ~25 seconds on a standard student laptop CPU.",
            )
        with train_col2:
            train_lr = st.selectbox(
                "Learning Rate:",
                options=[0.0001, 0.0003, 0.001],
                index=1,
            )
        with train_col3:
            train_seed = st.number_input("Random Seed:", value=42, step=1)

        is_training = st.session_state.get("is_training", False)
        start_train_btn = st.button("▶ Start PPO Training", type="primary", disabled=is_training)

        if start_train_btn:
            st.session_state["is_training"] = True
            st.info(f"Starting training run ({train_steps:,} timesteps on CPU)...")
            prog_bar = st.progress(0.0)
            status_text = st.empty()

            try:
                train_cfg = ExperimentConfig(
                    name="drone_ppo_gui",
                    seed=int(train_seed),
                    algorithm=AlgorithmConfig(
                        name="ppo",
                        learning_rate=float(train_lr),
                    ),
                    environment=EnvironmentConfig(
                        name="drone",
                    ),
                    training=TrainingConfig(
                        total_timesteps=int(train_steps),
                    ),
                    output_dir=Path("artifacts"),
                )

                trainer = PPOTrainer(config=train_cfg)
                t0 = time.time()
                result = trainer.fit()
                duration = time.time() - t0

                prog_bar.progress(1.0)
                status_text.text(f"Training Complete! Duration: {duration:.2f}s")
                st.success(f"Checkpoint saved: `{result.final_model_path}`")

                st.session_state["latest_training_rewards"] = result.episode_rewards

                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Timesteps", f"{result.total_timesteps:,}")
                t2.metric("Episodes Completed", f"{result.episodes_completed}")
                t3.metric("Training Time", f"{duration:.2f}s")
                t4.metric("Mean Final Reward", f"{result.mean_reward:.2f}")

            except Exception as e:
                st.error(f"Training failed: {e}")
            finally:
                st.session_state["is_training"] = False

        # Display Training Curve
        rewards_to_show = st.session_state.get("latest_training_rewards", [])
        if not rewards_to_show:
            demo_meta_path = Path("artifacts/metadata/drone_ppo_demo_training.json")
            if demo_meta_path.exists():
                import json

                try:
                    with open(demo_meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        rewards_to_show = data.get("episode_rewards", [])
                except Exception:
                    pass

        if rewards_to_show:
            st.markdown("### Training Progress Curve")
            fig_curve = build_training_curve_figure(
                episode_rewards=rewards_to_show,
                window_size=10,
                title="PPO Episode Return over Training Episodes",
            )
            st.plotly_chart(fig_curve, use_container_width=True)

    # ==========================================
    # TAB 3: EVALUATION & BASELINE
    # ==========================================
    with tabs[2]:
        st.subheader("Policy Evaluation & Random Baseline Comparison")
        st.markdown(
            "Empirically compare the trained **PPO agent** against an **untrained Random Policy baseline** "
            "under identical environment seeds and episode counts."
        )

        e_col1, e_col2, e_col3 = st.columns([2, 1, 1])
        with e_col1:
            eval_model_name = st.selectbox(
                "Evaluation Model:",
                options=model_names if model_names else ["No models found in artifacts/models"],
                index=0,
            )
        with e_col2:
            eval_episodes = st.slider("Episodes:", min_value=5, max_value=50, value=20, step=5)
        with e_col3:
            eval_mode = st.radio(
                "Mode:", ["Compare Both (PPO vs Random)", "PPO Only", "Random Only"]
            )

        run_eval_btn = st.button("📊 Run Benchmark Evaluation", type="primary")

        if run_eval_btn:
            with st.spinner(f"Running evaluation ({eval_episodes} episodes)..."):
                eval_env = DroneNavigation3DEnv()
                try:
                    metrics_to_plot: Dict[str, Any] = {}

                    # Evaluate PPO
                    if eval_mode in ["Compare Both (PPO vs Random)", "PPO Only"]:
                        if available_models:
                            target_path = next(
                                p for p in available_models if p.name == eval_model_name
                            )
                            ppo_algo = PPOAlgorithm.from_pretrained(target_path, env=eval_env)
                            ppo_res = evaluate_ppo_policy(
                                algorithm=ppo_algo,
                                env=eval_env,
                                num_episodes=int(eval_episodes),
                                base_seed=42,
                            )
                            metrics_to_plot["PPO Policy"] = ppo_res
                            Evaluator.save_report(ppo_res, Path("artifacts/evaluation.json"))
                        else:
                            st.warning("No saved model found. Please train a PPO agent first.")

                    # Evaluate Random Policy
                    if eval_mode in ["Compare Both (PPO vs Random)", "Random Only"]:
                        rand_res = evaluate_random_policy(
                            env=eval_env,
                            num_episodes=int(eval_episodes),
                            base_seed=42,
                        )
                        metrics_to_plot["Random Policy"] = rand_res

                    st.session_state["benchmark_results"] = metrics_to_plot

                except Exception as exc:
                    st.error(f"Evaluation error: {exc}")
                finally:
                    eval_env.close()

        # Display Benchmark Results
        bench_results = st.session_state.get("benchmark_results", {})
        if bench_results:
            st.markdown("### Empirical Benchmark Results")

            # Metrics Table
            table_data = []
            for pol_name, m in bench_results.items():
                succ_str = f"{m.success_rate * 100:.1f}%" if m.success_rate is not None else "N/A"
                coll_str = (
                    f"{m.collision_rate * 100:.1f}%" if m.collision_rate is not None else "N/A"
                )
                table_data.append(
                    {
                        "Policy": pol_name,
                        "Episodes": m.episodes,
                        "Success Rate": succ_str,
                        "Collision Rate": coll_str,
                        "Mean Reward": f"{m.mean_reward:.2f} ± {m.std_reward:.2f}",
                        "Mean Episode Length": f"{m.mean_episode_length:.1f} steps",
                    }
                )
            st.table(table_data)

            # Side-by-Side Comparison Chart
            fig_comp = build_comparison_bar_chart(
                metrics_map=bench_results,
                title="PPO Policy vs Random Baseline (Success & Collision Rates)",
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            st.info("Evaluation report exported to `artifacts/evaluation.json`")
        else:
            st.caption("Click 'Run Benchmark Evaluation' to generate empirical metrics.")

    # ==========================================
    # TAB 4: DIFFICULTY EXPERIMENT
    # ==========================================
    with tabs[3]:
        st.subheader("Obstacle-Density Experiment")
        st.markdown(
            "This experiment demonstrates how environment difficulty impacts agent performance. "
            "The policy is evaluated under identical seeds across **4, 6, and 8 procedural obstacles**."
        )
        st.info(
            "💡 **Core Insight**: *More obstacles increase the navigation difficulty of the environment.*"
        )

        d_col1, d_col2 = st.columns([2, 1])
        with d_col1:
            diff_model_name = st.selectbox(
                "Model to Test:",
                options=model_names if model_names else ["No models found in artifacts/models"],
                index=0,
                key="diff_model",
            )
        with d_col2:
            diff_episodes = st.slider(
                "Episodes per Condition:", min_value=5, max_value=20, value=10, step=5
            )

        run_diff_btn = st.button("🎯 Run Obstacle-Density Experiment", type="primary")

        if run_diff_btn:
            if available_models:
                with st.spinner("Evaluating across 4, 6, and 8 obstacles..."):
                    target_path = next(p for p in available_models if p.name == diff_model_name)
                    dummy_env = DroneNavigation3DEnv()
                    try:
                        algo = PPOAlgorithm.from_pretrained(target_path, env=dummy_env)
                        density_results = run_obstacle_density_experiment(
                            algorithm=algo,
                            obstacle_counts=(4, 6, 8),
                            episodes_per_density=int(diff_episodes),
                            base_seed=42,
                            output_path=Path("artifacts/obstacle_density_experiment.json"),
                        )
                        st.session_state["density_results"] = density_results
                    finally:
                        dummy_env.close()
            else:
                st.warning("No saved model found. Please train a PPO agent first.")

        density_data = st.session_state.get("density_results", None)
        if density_data:
            st.markdown("### Measured Experiment Data")
            display_rows = []
            for row in density_data:
                display_rows.append(
                    {
                        "Obstacle Count": f"{row['obstacle_count']} Obstacles",
                        "Episodes": row["episodes"],
                        "Success Rate": f"{row['success_rate'] * 100:.1f}%",
                        "Collision Rate": f"{row['collision_rate'] * 100:.1f}%",
                        "Mean Reward": f"{row['mean_reward']:.2f}",
                        "Mean Steps": f"{row['mean_episode_length']:.1f}",
                    }
                )
            st.table(display_rows)

            fig_density = build_density_experiment_figure(
                density_results=density_data,
                title="Performance Degradation under Increasing Obstacle Density",
            )
            st.plotly_chart(fig_density, use_container_width=True)
            st.success(
                "Experiment results exported to `artifacts/obstacle_density_experiment.json`"
            )
        else:
            st.warning(
                "Not evaluated — click 'Run Obstacle-Density Experiment' to collect actual data."
            )

    # ==========================================
    # TAB 5: ABOUT & ARCHITECTURE
    # ==========================================
    with tabs[4]:
        st.subheader("System Architecture & College Demonstration Specs")

        st.markdown(
            r"""
            ### How It Works (Step-by-Step)
            1. **Drone Observes**: The drone receives a **29-dimensional continuous state vector** at each time step.
            2. **PPO Predicts**: The neural network (MlpPolicy) computes continuous 3D acceleration commands.
            3. **Action Executed**: Commanded accelerations $(a_x, a_y, a_z) \in [-1.0, 1.0]^3$ are scaled to $\pm 4.0\text{ m/s}^2$.
            4. **Physics Integration**: 3D kinematics updates velocity with aerodynamic drag and integrates position.
            5. **Reward Feedback**: The environment rewards closing distance to the target and penalizes collisions.
            6. **Policy Learning**: During training, PPO updates actor-critic weights using generalized advantage estimation.
            7. **Evaluation**: The trained policy is tested deterministically on unseen random seeds.
            """
        )

        arch_col1, arch_col2 = st.columns(2)

        with arch_col1:
            st.markdown(
                r"""
                ### Specifications
                - **Observation Space (29 Dims)**:
                  - Drone position $\mathbf{p} = [x, y, z]$ (3 dims)
                  - Velocity vector $\mathbf{v} = [v_x, v_y, v_z]$ (3 dims)
                  - Target position $\mathbf{g} = [g_x, g_y, g_z]$ (3 dims)
                  - Relative target offset $\mathbf{g} - \mathbf{p}$ (3 dims)
                  - Target distance ratio (1 dim)
                  - 16-ray spherical LiDAR rangefinder readings (16 dims)
                - **Action Space (3 Dims)**:
                  - Continuous 3D acceleration commands in $[-1.0, 1.0]^3$
                - **Agent**: Stable-Baselines3 PPO (`MlpPolicy`, 2 layers of 64 units)
                """
            )

        with arch_col2:
            st.markdown(
                r"""
                ### Kinematic Flight Model
                $$\mathbf{a}_t = \text{clip}(\mathbf{u}_t, -1, 1) \cdot a_{\max}$$
                $$\mathbf{v}_{t+1} = \mathbf{v}_t (1 - c_{\text{drag}} \Delta t) + \mathbf{a}_t \Delta t$$
                $$\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{v}_{t+1} \Delta t$$
                - **Time Step**: $\Delta t = 0.1\text{ s}$
                - **Max Acceleration**: $a_{\max} = 4.0\text{ m/s}^2$
                - **Max Velocity**: $v_{\max} = 8.0\text{ m/s}$
                - **Linear Drag**: $c_{\text{drag}} = 0.05$
                - **Arena Bounds**: $30\text{m} \times 30\text{m} \times 15\text{m}$
                """
            )

        st.markdown("---")
        st.markdown(
            """
            ### Educational Scope & Honest Technical Limitations
            - **Simulation Only**: Designed as a college computer science/engineering demonstration project.
            - **Point-Mass Kinematics**: Uses a translational 3-DOF model with aerodynamic drag; does not model 6-DOF rotor aerodynamics, blade stall, or motor ESC latency.
            - **Analytical Raycasts**: The 16-ray spherical LiDAR uses geometric ray-sphere and ray-box intersections without sensor noise or atmospheric scattering.
            - **No Real Drone Autopilot**: Not intended for direct deployment on physical UAV flight controllers (PX4/ArduPilot).
            """
        )


if __name__ == "__main__":
    main()
