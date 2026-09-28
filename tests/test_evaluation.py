"""Tests for agent evaluation and JSON report generation."""

import json
from pathlib import Path

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.evaluation.evaluator import (
    Evaluator,
    compare_policies,
    evaluate_random_policy,
    run_obstacle_density_experiment,
)


def test_evaluator_deterministic_evaluation(tmp_path: Path) -> None:
    """Verify evaluation generates deterministic results with fixed seed."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)

    evaluator = Evaluator(algorithm=algo, env=env)
    metrics1 = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=42)
    metrics2 = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=42)

    assert metrics1.episodes == 3
    assert metrics1.mean_reward == metrics2.mean_reward
    assert metrics1.success_rate == metrics2.success_rate
    assert metrics1.collision_rate == metrics2.collision_rate
    assert metrics1.mean_episode_length == metrics2.mean_episode_length

    # Verify JSON report creation and structure
    report_file = tmp_path / "evaluation.json"
    saved = evaluator.save_report(metrics1, report_file)
    assert saved.exists()

    with open(saved, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["episodes"] == 3
    assert "success_rate" in data
    assert "collision_rate" in data
    assert "mean_reward" in data
    assert "mean_episode_length" in data
    env.close()


def test_evaluator_episode_records() -> None:
    """Verify individual episode records are tracked correctly."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = Evaluator(algorithm=algo, env=env)

    evaluator.evaluate(num_episodes=2, deterministic=True, base_seed=10)
    assert len(evaluator.last_episode_records) == 2
    rec = evaluator.last_episode_records[0]
    assert rec.episode_index == 0
    assert rec.seed == 10
    assert isinstance(rec.return_value, float)
    assert isinstance(rec.success, bool)
    assert isinstance(rec.collision, bool)
    env.close()


def test_evaluate_random_policy() -> None:
    """Verify uniform-random policy evaluation baseline executes and returns valid metrics."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=2)
    metrics = evaluate_random_policy(env=env, num_episodes=3, base_seed=42)

    assert metrics.episodes == 3
    assert isinstance(metrics.mean_reward, float)
    assert 0.0 <= (metrics.success_rate or 0.0) <= 1.0
    assert 0.0 <= (metrics.collision_rate or 0.0) <= 1.0
    assert metrics.mean_episode_length > 0
    env.close()


def test_compare_policies() -> None:
    """Verify head-to-head comparison between PPO and Random baseline."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)

    comparison = compare_policies(ppo_algorithm=algo, env=env, num_episodes=2, base_seed=42)
    assert "PPO" in comparison
    assert "Random Policy" in comparison
    assert comparison["PPO"].episodes == 2
    assert comparison["Random Policy"].episodes == 2
    env.close()


def test_run_obstacle_density_experiment(tmp_path: Path) -> None:
    """Verify obstacle-density experiment runs across varied obstacle counts."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    output_file = tmp_path / "density_exp.json"

    results = run_obstacle_density_experiment(
        algorithm=algo,
        obstacle_counts=(2, 4),
        episodes_per_density=2,
        base_seed=42,
        bounds=(20.0, 20.0, 10.0),
        output_path=output_file,
    )

    assert len(results) == 2
    assert results[0]["obstacle_count"] == 2
    assert results[1]["obstacle_count"] == 4
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert len(saved_data) == 2
    assert "success_rate" in saved_data[0]
    assert "collision_rate" in saved_data[0]
    env.close()
