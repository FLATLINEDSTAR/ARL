"""Browser GUI and interactive 3D visualization utilities for AdaptiveRL."""

from adaptive_rl.gui.visualizer import (
    build_arena_3d_figure,
    build_comparison_bar_chart,
    build_density_experiment_figure,
    build_distribution_shift_trajectory_figure,
    build_recovery_curve_figure,
    build_training_curve_figure,
    get_available_models,
    run_drone_simulation_episode,
)

__all__ = [
    "build_arena_3d_figure",
    "build_comparison_bar_chart",
    "build_density_experiment_figure",
    "build_distribution_shift_trajectory_figure",
    "build_recovery_curve_figure",
    "build_training_curve_figure",
    "get_available_models",
    "run_drone_simulation_episode",
]
