"""Tests verifying Typer CLI commands and execution."""

import re
from pathlib import Path

from typer.testing import CliRunner

from adaptive_rl.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """Verify adaptive-rl --help prints help and available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AdaptiveRL" in result.output
    assert "train" in result.output
    assert "evaluate" in result.output
    assert "demo-drone" in result.output
    assert "gui" in result.output
    assert "experiment-density" in result.output
    assert "experiment-ablation" in result.output
    assert "config" in result.output
    assert "env" in result.output


def test_cli_gui_help() -> None:
    """Verify adaptive-rl gui --help displays options."""
    result = runner.invoke(app, ["gui", "--help"])
    assert result.exit_code == 0
    clean_output = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.output)
    assert "--port" in clean_output
    assert "--host" in clean_output
    assert "Streamlit" in clean_output


def test_cli_experiment_density_help() -> None:
    """Verify adaptive-rl experiment-density --help displays options."""
    result = runner.invoke(app, ["experiment-density", "--help"])
    assert result.exit_code == 0
    clean_output = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.output)
    assert "--model" in clean_output
    assert "--episodes" in clean_output
    assert "--seed" in clean_output


def test_cli_experiment_ablation_help() -> None:
    """Verify adaptive-rl experiment-ablation --help displays options."""
    result = runner.invoke(app, ["experiment-ablation", "--help"])
    assert result.exit_code == 0
    clean_output = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.output)
    assert "--timesteps" in clean_output
    assert "--episodes" in clean_output
    assert "--seed" in clean_output
    assert "--output-report" in clean_output
    assert "--output-csv" in clean_output


def test_cli_version() -> None:
    """Verify adaptive-rl version displays package version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "AdaptiveRL" in result.output


def test_cli_config_validate_success() -> None:
    """Verify adaptive-rl config validate succeeds for valid YAML configuration."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "drone_ppo.yaml"
    result = runner.invoke(app, ["config", "validate", str(config_path)])
    assert result.exit_code == 0
    assert "Configuration is valid" in result.output


def test_cli_config_validate_failure(tmp_path: Path) -> None:
    """Verify adaptive-rl config validate fails with code 1 for invalid YAML."""
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("invalid: yaml: syntax: [", encoding="utf-8")
    result = runner.invoke(app, ["config", "validate", str(bad_config)])
    assert result.exit_code == 1
    assert "Configuration validation error" in result.output


def test_cli_env_inspect_success() -> None:
    """Verify adaptive-rl env inspect inspects spaces for the drone environment."""
    result = runner.invoke(app, ["env", "inspect", "drone"])
    assert result.exit_code == 0
    assert "verified successfully" in result.output
    assert "Observation Space" in result.output
    assert "Action Space" in result.output


def test_cli_env_inspect_failure() -> None:
    """Verify adaptive-rl env inspect fails gracefully for unknown environment."""
    result = runner.invoke(app, ["env", "inspect", "NonExistentEnv-v999"])
    assert result.exit_code == 1
    assert "Environment inspection failed" in result.output


def test_cli_train_and_evaluate_and_demo(tmp_path: Path) -> None:
    """End-to-end CLI test: train -> evaluate -> demo-drone."""
    test_config = tmp_path / "test_drone_cli.yaml"
    test_config.write_text(
        f"""
name: "cli_drone_test"
seed: 42
algorithm:
  name: "ppo"
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 32
  parameters:
    n_steps: 64
environment:
  name: "drone"
  max_steps: 20
  parameters:
    bounds: [20.0, 20.0, 10.0]
    num_obstacles: 2
training:
  total_timesteps: 64
  checkpoint_freq: 0
  log_interval: 10
evaluation:
  eval_episodes: 2
output_dir: "{(tmp_path / "artifacts").as_posix()}"
log_dir: "{(tmp_path / "logs").as_posix()}"
""",
        encoding="utf-8",
    )

    # 1. Train command
    train_res = runner.invoke(app, ["train", "--config", str(test_config), "--timesteps", "64"])
    if train_res.exit_code != 0:
        print(f"STDOUT from CLI: {train_res.stdout}")
    assert train_res.exit_code == 0
    assert "Training Completed Successfully!" in train_res.output

    model_file = tmp_path / "artifacts" / "models" / "cli_drone_test_final.zip"
    assert model_file.exists()

    # 2. Evaluate command
    eval_report = tmp_path / "eval_report.json"
    eval_csv = tmp_path / "eval_report.csv"
    eval_res = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(test_config),
            "--model",
            str(model_file),
            "--episodes",
            "2",
            "--output-report",
            str(eval_report),
            "--output-csv",
            str(eval_csv),
            "--compare-random",
        ],
    )
    assert eval_res.exit_code == 0
    assert "## Evaluation" in eval_res.output
    assert "Policy Comparison" in eval_res.output
    assert eval_report.exists()
    assert eval_csv.exists()

    # 2b. Experiment-density command
    density_report = tmp_path / "density_test.json"
    dense_res = runner.invoke(
        app,
        [
            "experiment-density",
            "--model",
            str(model_file),
            "--episodes",
            "1",
            "--output-report",
            str(density_report),
        ],
    )
    assert dense_res.exit_code == 0
    assert "Obstacle-Density Results" in dense_res.output
    assert density_report.exists()

    # 3. Demo-drone command
    demo_res = runner.invoke(
        app,
        [
            "demo-drone",
            "--model",
            str(model_file),
            "--config",
            str(test_config),
            "--seed",
            "42",
            "--max-steps",
            "10",
        ],
    )
    assert demo_res.exit_code == 0
    assert ("SUCCESS" in demo_res.output) or ("FAILED" in demo_res.output)


def test_cli_split_options_and_generalization_benchmark(tmp_path: Path) -> None:
    """Verify CLI commands accept --split train/test and evaluate-generalization works."""
    import json

    test_config = tmp_path / "test_drone_split.yaml"
    test_config.write_text(
        f"""
name: "cli_split_test"
seed: 42
algorithm:
  name: "ppo"
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 32
  parameters:
    n_steps: 64
environment:
  name: "drone"
  max_steps: 15
  parameters:
    bounds: [20.0, 20.0, 10.0]
    num_obstacles: 2
training:
  total_timesteps: 32
  checkpoint_freq: 0
  log_interval: 10
evaluation:
  eval_episodes: 2
output_dir: "{(tmp_path / "artifacts").as_posix()}"
log_dir: "{(tmp_path / "logs").as_posix()}"
""",
        encoding="utf-8",
    )

    # 1. Train with --split train
    train_res = runner.invoke(
        app,
        ["train", "--config", str(test_config), "--timesteps", "32", "--split", "train"],
    )
    assert train_res.exit_code == 0
    assert "Split: train" in train_res.output

    # 2. Train with invalid split fails
    bad_train = runner.invoke(
        app,
        ["train", "--config", str(test_config), "--timesteps", "32", "--split", "invalid_split"],
    )
    assert bad_train.exit_code == 1
    assert "Invalid split" in bad_train.output

    model_file = tmp_path / "artifacts" / "models" / "cli_split_test_final.zip"
    assert model_file.exists()

    # 3. Evaluate with --split test
    eval_res = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(test_config),
            "--model",
            str(model_file),
            "--episodes",
            "2",
            "--split",
            "test",
        ],
    )
    assert eval_res.exit_code == 0
    assert "Split: test" in eval_res.output

    # 4. Evaluate with invalid split fails
    bad_eval = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(test_config),
            "--model",
            str(model_file),
            "--split",
            "invalid_split",
        ],
    )
    assert bad_eval.exit_code == 1
    assert "Invalid split" in bad_eval.output

    # 5. Evaluate-generalization command
    bench_report = tmp_path / "generalization_benchmark.json"
    gen_res = runner.invoke(
        app,
        [
            "evaluate-generalization",
            "--model",
            str(model_file),
            "--config",
            str(test_config),
            "--episodes",
            "2",
            "--output-report",
            str(bench_report),
        ],
    )
    assert gen_res.exit_code == 0
    assert "Generalization Benchmark Results" in gen_res.output
    assert "Generalization Gaps" in gen_res.output
    assert bench_report.exists()

    with open(bench_report, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "train" in data
    assert "test" in data
    assert "generalization_gap" in data
    assert data["train"]["seeds"] == [0, 1]
    assert data["test"]["seeds"] == [1000, 1001]
    assert "success" in data["generalization_gap"]
    assert "reward" in data["generalization_gap"]

    # 6. Evaluate-generalization missing model fails
    missing_res = runner.invoke(
        app,
        [
            "evaluate-generalization",
            "--model",
            str(tmp_path / "nonexistent.zip"),
        ],
    )
    assert missing_res.exit_code == 1
    assert "Model file does not exist" in missing_res.output


def test_cli_experiment_ablation_smoke(tmp_path: Path) -> None:
    """Verify adaptive-rl experiment-ablation executes end-to-end and outputs results."""
    json_rep = tmp_path / "ablation.json"
    csv_rep = tmp_path / "ablation.csv"
    res = runner.invoke(
        app,
        [
            "experiment-ablation",
            "--timesteps",
            "64",
            "--episodes",
            "1",
            "--seed",
            "42",
            "--eval-freq",
            "32",
            "--output-report",
            str(json_rep),
            "--output-csv",
            str(csv_rep),
        ],
    )
    assert res.exit_code == 0
    assert "Reward-Function Ablation Benchmark Results" in res.output
    assert json_rep.exists()
    assert csv_rep.exists()
