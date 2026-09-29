"""CI-sized end-to-end Issue #265 protocol smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from adaptive_rl.benchmarking.adaptation_artifacts import (
    write_adaptive_vs_fixed_artifacts,
    write_or_verify_study_manifest,
    write_replicate_checkpoint,
    write_study_manifest,
)
from adaptive_rl.benchmarking.adaptation_runner import (
    _load_resume_replicates,
    _read_completed_study_artifact,
    _resume_replicate_checkpoint,
    run_adaptation_benchmark,
)
from adaptive_rl.cli import app
from adaptive_rl.config import load_config
from adaptive_rl.protocol.constants import TRAINING_SEEDS
from adaptive_rl.protocol.seeds import frozen_schedule, schedule_fingerprint


def test_cli_adaptation_smoke_runs_complete_protocol_and_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "adaptation-smoke"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "adaptation",
            "--smoke",
            "--stochastic",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    artifact = json.loads((output_dir / "adaptation.json").read_text(encoding="utf-8"))
    assert artifact["run_type"] == "smoke"
    replicate = artifact["replicates"][0]
    assert replicate["training_seed"] == TRAINING_SEEDS[0]
    assert replicate["status"] == "completed"
    train_config = replicate["training_provenance"]["effective_config"]
    assert train_config["adaptation_benchmark"] is None
    assert train_config["environment"]["parameters"]["num_obstacles"] == 8
    assert train_config["environment"]["parameters"]["wind_speed"] == 0.5
    assert train_config["environment"]["parameters"]["gust_sigma"] == 0.15
    assert replicate["schedule_fingerprint"] == schedule_fingerprint(frozen_schedule())
    assert len(replicate["shared_pre_shift_episodes"]) == 15
    assert len(replicate["shared_shock_episodes"]) == 5
    assert len(replicate["adaptive_episodes"]) == 10
    assert len(replicate["fixed_episodes"]) == 10
    assert [block["block_episode"] for block in replicate["update_blocks"]] == list(range(5, 15))
    assert [block["visible_episode_indices"] for block in replicate["update_blocks"]] == [
        list(range(1, boundary + 1)) for boundary in range(5, 15)
    ]
    assert [e["episode_seed"] for e in replicate["adaptive_episodes"]] == [
        e["episode_seed"] for e in replicate["fixed_episodes"]
    ]
    assert replicate["fixed_final_fingerprint"] == replicate["frozen_fingerprint"]
    assert artifact["paired_analysis"]["drone_disturbed/ppo"]["status"] == "inconclusive"
    assert (output_dir / "adaptation.csv").is_file()


def test_cli_adaptation_rejects_deterministic_ppo_before_creating_outputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "invalid-deterministic-ppo"
    result = CliRunner().invoke(
        app,
        ["benchmark", "adaptation", "--smoke", "--output-dir", str(output_dir)],
    )
    assert result.exit_code == 1
    assert "not valid on-policy rollout data" in result.output
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--algorithm", "dqn"], "--algorithm must be 'ppo' or 'sac'"),
        (["--resume"], "--resume requires a preregistered --study version and --run-id"),
        (
            ["--study", "prereg-v1", "--run-id", "hash-mismatch", "--stochastic"],
            "frozen Issue #271 configuration",
        ),
    ],
)
def test_cli_adaptation_rejects_invalid_or_mismatched_requests(
    args: list[str], message: str
) -> None:
    result = CliRunner().invoke(app, ["benchmark", "adaptation", *args])
    assert result.exit_code == 1
    assert message in " ".join(result.output.split())


def test_preregistered_study_rejects_subset_before_training(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "adaptation",
            "--study",
            "prereg-v1",
            "--run-id",
            "subset-rejection",
            "--training-seeds",
            "31001",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "all ten seeds in order" in " ".join(result.output.split())


def test_preregistered_study_rejects_changed_scientific_config_before_training() -> None:
    config = load_config("configs/drone_distribution_shift.yaml")
    algorithm = config.algorithm.model_copy(update={"learning_rate": 0.0002}, deep=True)
    changed = config.model_copy(update={"algorithm": algorithm}, deep=True)
    with pytest.raises(ValueError, match="frozen Issue #271 configuration"):
        run_adaptation_benchmark(
            changed,
            study_run_id="changed-config",
            training_seeds=TRAINING_SEEDS,
        )


def test_preregistered_study_rejects_deterministic_ppo_rollout_before_training() -> None:
    with pytest.raises(ValueError, match="not valid on-policy rollout data"):
        run_adaptation_benchmark(
            load_config("configs/drone_distribution_shift.yaml"),
            study_run_id="deterministic-ppo-rejection",
            training_seeds=TRAINING_SEEDS,
        )


def test_issue265_rejects_deterministic_ppo_rollout_before_creating_outputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="not valid on-policy rollout data"):
        run_adaptation_benchmark(
            load_config("configs/drone_distribution_shift.yaml"),
            output_dir=output_dir,
            training_seeds=TRAINING_SEEDS[:1],
        )
    assert not output_dir.exists()


def test_checkpoint_resume_helper_rejects_corrupted_terminal_record(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "seed_31001.json"
    write_replicate_checkpoint(
        {"training_seed": TRAINING_SEEDS[0], "status": "completed"},
        checkpoint_path,
        study_hash="expected-study",
        protocol_hash="expected-protocol",
    )
    checkpoint_path.write_text("truncated", encoding="utf-8")

    with pytest.raises(ValueError, match="resume rejected invalid checkpoint"):
        _resume_replicate_checkpoint(
            checkpoint_path,
            study_hash="expected-study",
            protocol_hash="expected-protocol",
            training_seed=TRAINING_SEEDS[0],
        )


def test_resume_preflight_rejects_orphan_checkpoint_digest_before_training(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "replicate_state"
    state_root.mkdir()
    (state_root / "seed_31001.json.sha256").write_text("0" * 64, encoding="ascii")

    with pytest.raises(ValueError, match="checkpoint or digest is missing for seed 31001"):
        _load_resume_replicates(
            state_root,
            study_hash="study",
            protocol_hash="protocol",
            training_seeds=(TRAINING_SEEDS[0], TRAINING_SEEDS[1]),
        )


def test_completed_study_reader_returns_stable_persisted_artifact(
    tmp_path: Path,
) -> None:
    study_hash = write_or_verify_study_manifest(
        {"protocol": "test", "seed": TRAINING_SEEDS[0]},
        tmp_path / "study_manifest.json",
        resume=False,
    )
    artifact = {
        "schema_version": "1.0",
        "run_id": "idempotent-run",
        "study_hash": study_hash,
        "run_status": "COMPLETE",
        "replicates": [],
        "artifact_paths": {
            "json": "adaptive_vs_fixed.json",
            "csv": "adaptive_vs_fixed.csv",
            "manifest": "manifest.json",
        },
    }
    json_path, csv_path = write_adaptive_vs_fixed_artifacts(artifact, tmp_path)
    manifest_path = tmp_path / "manifest.json"
    write_study_manifest(
        json_path,
        csv_path,
        manifest_path,
        run_id="idempotent-run",
        command="adaptive-rl benchmark adaptation --study prereg-v1 --run-id idempotent-run",
    )

    first_result = _read_completed_study_artifact(
        json_path,
        manifest_path,
        run_id="idempotent-run",
        study_hash=study_hash,
    )
    resumed_result = _read_completed_study_artifact(
        json_path,
        manifest_path,
        run_id="idempotent-run",
        study_hash=study_hash,
    )
    assert first_result == resumed_result == json.loads(json_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"output_dir": Path("/tmp/absolute-study-output")}, "must be relative"),
        (
            {"config_path": Path.cwd() / "configs/drone_distribution_shift.yaml"},
            "config path must be repository-relative",
        ),
    ],
)
def test_preregistered_study_rejects_nonreproducible_paths_before_training(
    kwargs: dict[str, Path], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_adaptation_benchmark(
            load_config("configs/drone_distribution_shift.yaml"),
            study_run_id="invalid-path",
            training_seeds=TRAINING_SEEDS,
            **kwargs,
        )
