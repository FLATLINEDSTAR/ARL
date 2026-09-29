"""Unit and integration tests for SAC algorithm, registration, training, and benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from adaptive_rl.algorithms.registry import (
    algorithm_registry,
    get_algorithm_factory,
    get_algorithm_metadata,
    load_algorithm_from_pretrained,
)
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.benchmarking.comparison import (
    run_algorithm_comparison,
)
from adaptive_rl.cli import app
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
    load_config,
)
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.evaluation.evaluator import Evaluator, compare_policies
from adaptive_rl.training.trainer import PPOTrainer, RLTrainer, get_trainer

runner = CliRunner()


def test_sac_registry_metadata() -> None:
    """Verify SAC algorithm is properly registered in algorithm_registry."""
    algorithms = algorithm_registry.list_algorithms()
    assert "sac" in algorithms
    assert "ppo" in algorithms

    meta = get_algorithm_metadata("sac")
    assert meta.name == "sac"
    assert meta.action_space == "continuous"
    assert meta.trainable is True
    assert meta.class_name == "SACAlgorithm"
    assert "off-policy" in meta.tags
    assert "actor-critic" in meta.tags

    # Verify required default hyperparameters
    hp = meta.hyperparameters
    assert hp["learning_rate"] == pytest.approx(3e-4)
    assert hp["buffer_size"] == 100000
    assert hp["batch_size"] == 256
    assert hp["gamma"] == pytest.approx(0.99)
    assert hp["tau"] == pytest.approx(0.005)


def test_sac_factory_instantiation() -> None:
    """Verify get_algorithm_factory instantiates SACAlgorithm correctly."""
    factory = get_algorithm_factory("sac")
    assert factory is SACAlgorithm

    env = DroneNavigation3DEnv(bounds=(15.0, 15.0, 10.0), max_steps=10)
    try:
        algo = factory(env=env, learning_starts=10)
        assert isinstance(algo, SACAlgorithm)
        assert algo.hyperparameters["learning_starts"] == 10
    finally:
        env.close()


def test_sac_algorithm_lifecycle(tmp_path: Path) -> None:
    """Verify SAC train, predict, save, load, and from_pretrained lifecycle."""
    env = DroneNavigation3DEnv(bounds=(15.0, 15.0, 10.0), max_steps=10, num_obstacles=1)
    try:
        algo = SACAlgorithm(
            env=env,
            learning_rate=1e-3,
            buffer_size=1000,
            learning_starts=20,
            batch_size=32,
            seed=42,
        )

        assert algo.num_timesteps == 0

        # Predict before training
        obs, _ = env.reset(seed=42)
        action, state = algo.predict(obs, deterministic=True)
        assert action.shape == (3,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

        # Train a few timesteps
        algo.train(total_timesteps=40)
        assert algo.num_timesteps >= 40

        trained_action, _ = algo.predict(obs, deterministic=True)

        # Save model
        save_file = tmp_path / "sac_test.zip"
        algo.save(save_file)
        assert save_file.exists()

        # Load model into new instance
        algo_loaded = SACAlgorithm(env=env)
        algo_loaded.load(save_file, env=env)
        act_loaded, _ = algo_loaded.predict(obs, deterministic=True)
        np.testing.assert_allclose(trained_action, act_loaded, atol=1e-4)

        # Test from_pretrained
        algo_pretrained = SACAlgorithm.from_pretrained(save_file, env=env)
        assert isinstance(algo_pretrained, SACAlgorithm)
        act_pre, _ = algo_pretrained.predict(obs, deterministic=True)
        np.testing.assert_allclose(trained_action, act_pre, atol=1e-4)

        # Test generic load_algorithm_from_pretrained
        algo_generic = load_algorithm_from_pretrained(save_file, env=env)
        assert isinstance(algo_generic, SACAlgorithm)

    finally:
        env.close()


def test_trainer_generalization_sac(tmp_path: Path) -> None:
    """Verify RLTrainer correctly instantiates and trains SAC based on config."""
    config = ExperimentConfig(
        name="test_sac_train",
        seed=42,
        algorithm=AlgorithmConfig(
            name="sac",
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=32,
            parameters={"buffer_size": 2000, "learning_starts": 20},
        ),
        environment=EnvironmentConfig(
            name="drone",
            max_steps=10,
            parameters={"bounds": [15.0, 15.0, 10.0], "num_obstacles": 1},
        ),
        training=TrainingConfig(
            total_timesteps=40,
            checkpoint_freq=0,
            log_interval=5,
        ),
        evaluation=EvaluationConfig(
            eval_episodes=2,
            deterministic=True,
        ),
        output_dir=tmp_path,
        log_dir=tmp_path / "logs",
    )

    trainer = get_trainer(config=config)
    assert isinstance(trainer, RLTrainer)
    assert isinstance(trainer.algorithm, SACAlgorithm)
    assert PPOTrainer is RLTrainer

    result = trainer.fit()
    assert result.total_timesteps == 40
    assert result.final_model_path.exists()
    assert result.metadata_path is not None and result.metadata_path.exists()

    with open(result.metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["algorithm"] == "sac"
    assert meta["total_timesteps"] == 40
    trainer.close()


def test_sac_evaluation_and_policy_comparison() -> None:
    """Verify Evaluator and compare_policies handle SAC policy seamlessly."""
    env = DroneNavigation3DEnv(bounds=(15.0, 15.0, 10.0), max_steps=10, num_obstacles=1)
    try:
        algo = SACAlgorithm(
            env=env,
            learning_rate=1e-3,
            buffer_size=1000,
            learning_starts=10,
            batch_size=32,
            seed=42,
        )

        evaluator = Evaluator(algorithm=algo, env=env)
        metrics = evaluator.evaluate(num_episodes=2, deterministic=True, base_seed=42)
        assert metrics.episodes == 2
        assert metrics.success_rate is not None
        assert metrics.collision_rate is not None

        comp = compare_policies(algorithm=algo, env=env, num_episodes=2, base_seed=42)
        assert "SAC" in comp
        assert "Random Policy" in comp
        assert comp["SAC"].episodes == 2
        assert comp["Random Policy"].episodes == 2
    finally:
        env.close()


def test_drone_sac_config_validation() -> None:
    """Verify configs/drone_sac.yaml loads and validates cleanly."""
    sac_config_path = Path("configs/drone_sac.yaml")
    assert sac_config_path.exists()

    cfg = load_config(sac_config_path)
    assert cfg.algorithm.name == "sac"
    assert cfg.algorithm.learning_rate == pytest.approx(3e-4)
    assert cfg.algorithm.gamma == pytest.approx(0.99)
    assert cfg.algorithm.batch_size == 256
    assert cfg.algorithm.parameters["buffer_size"] == 100000
    assert cfg.algorithm.parameters["tau"] == pytest.approx(0.005)

    res = runner.invoke(app, ["config", "validate", str(sac_config_path)])
    assert res.exit_code == 0
    assert "Configuration is valid" in res.output


def test_cli_train_and_evaluate_sac(tmp_path: Path) -> None:
    """Verify CLI train and evaluate end-to-end with SAC configuration."""
    config_dict = {
        "name": "cli_sac_test",
        "seed": 42,
        "algorithm": {
            "name": "sac",
            "learning_rate": 0.001,
            "gamma": 0.99,
            "batch_size": 32,
            "parameters": {
                "buffer_size": 2000,
                "learning_starts": 15,
            },
        },
        "environment": {
            "name": "drone",
            "max_steps": 10,
            "parameters": {
                "bounds": [15.0, 15.0, 10.0],
                "num_obstacles": 1,
            },
        },
        "training": {
            "total_timesteps": 30,
            "checkpoint_freq": 0,
            "log_interval": 5,
        },
        "evaluation": {
            "eval_episodes": 2,
            "deterministic": True,
        },
        "output_dir": str(tmp_path),
        "log_dir": str(tmp_path / "logs"),
    }
    cfg_file = tmp_path / "test_sac.yaml"
    import yaml

    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)

    train_res = runner.invoke(app, ["train", "--config", str(cfg_file)])
    assert train_res.exit_code == 0
    assert "Training Completed Successfully" in train_res.output

    model_file = tmp_path / "models" / "cli_sac_test_final.zip"
    assert model_file.exists()

    report_file = tmp_path / "eval_report.json"
    eval_res = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(cfg_file),
            "--model",
            str(model_file),
            "--episodes",
            "2",
            "--output-report",
            str(report_file),
            "--compare-random",
        ],
    )
    assert eval_res.exit_code == 0
    assert "Benchmark Results" in eval_res.output
    assert "Policy Comparison" in eval_res.output
    assert "SAC" in eval_res.output
    assert report_file.exists()


def test_algorithm_comparison_benchmark(tmp_path: Path) -> None:
    """Verify run_algorithm_comparison runs both PPO and SAC under fair identical conditions."""
    report_json = tmp_path / "comparison.json"
    report_csv = tmp_path / "comparison.csv"

    data = run_algorithm_comparison(
        algorithms=("ppo", "sac"),
        timesteps=30,
        eval_episodes=2,
        seed=42,
        env_parameters={"bounds": [15.0, 15.0, 10.0], "num_obstacles": 1, "max_steps": 10},
        output_dir=tmp_path,
        output_json=report_json,
        output_csv=report_csv,
    )

    assert data["benchmark"] == "algorithm_comparison"
    assert data["algorithms"] == ["PPO", "SAC"]
    assert data["timesteps"] == 30
    assert data["eval_episodes"] == 2
    assert data["seed"] == 42
    assert data["evaluation_base_seed"] == 10042
    assert len(data["results"]) == 2

    ppo_res = data["results"][0]
    sac_res = data["results"][1]

    assert ppo_res["algorithm"] == "PPO"
    assert sac_res["algorithm"] == "SAC"
    assert ppo_res["eval_episodes"] == 2
    assert sac_res["eval_episodes"] == 2
    assert "success_rate" in ppo_res
    assert "collision_rate" in sac_res

    assert report_json.exists()
    assert report_csv.exists()


def test_cli_benchmark_compare_algorithms(tmp_path: Path) -> None:
    """Verify CLI benchmark compare-algorithms command execution."""
    report_json = tmp_path / "cli_compare.json"
    res = runner.invoke(
        app,
        [
            "benchmark",
            "compare-algorithms",
            "--algorithms",
            "ppo,sac",
            "--timesteps",
            "30",
            "--episodes",
            "1",
            "--output-report",
            str(report_json),
        ],
    )
    assert res.exit_code == 0
    assert "Algorithm Comparison Results" in res.output
    assert report_json.exists()
