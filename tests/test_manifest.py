"""Comprehensive unit tests for experiment manifests and provenance tracking (Issue #248)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from adaptive_rl.cli import app
from adaptive_rl.config import load_config
from adaptive_rl.manifest import (
    ExperimentManifest,
    collect_git_metadata,
    collect_hardware_telemetry,
    collect_host_system,
    collect_software_packages,
    compute_sha256,
    create_manifest,
    load_manifest,
    save_manifest,
)
from adaptive_rl.training.trainer import RLTrainer

runner = CliRunner()


def test_compute_sha256(tmp_path: Path) -> None:
    """Verify compute_sha256 produces exact cryptographic hashes."""
    test_file = tmp_path / "sample.bin"
    payload = b"AdaptiveRL experiment manifest reproducibility test payload 12345"
    test_file.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    actual = compute_sha256(test_file)
    assert actual == expected

    with pytest.raises(FileNotFoundError):
        compute_sha256(tmp_path / "nonexistent.bin")


def test_manifest_schema_and_serialization(tmp_path: Path) -> None:
    """Verify manifest schema generation and serialization roundtrip."""
    dummy_model = tmp_path / "model.zip"
    dummy_model.write_bytes(b"dummy model weights content")

    manifest = create_manifest(
        experiment_name="test_experiment",
        config_dict={"name": "test_experiment", "seed": 42},
        started_at=1700000000.0,
        finished_at=1700000010.5,
        training_time_seconds=9.8,
        artifacts={"model": dummy_model},
        command=["adaptive-rl", "train", "--config", "configs/drone_ppo.yaml"],
    )

    assert isinstance(manifest, ExperimentManifest)
    assert manifest.experiment_name == "test_experiment"
    assert manifest.execution.duration_seconds == 10.5
    assert manifest.execution.training_time_seconds == 9.8
    assert "model" in manifest.artifacts
    assert manifest.artifacts["model"].sha256 == compute_sha256(dummy_model)
    assert manifest.artifacts["model"].size_bytes == len(b"dummy model weights content")

    # Save and reload
    out_file = tmp_path / "experiment_manifest.json"
    save_manifest(manifest, out_file)
    assert out_file.exists()

    reloaded = load_manifest(out_file)
    assert reloaded.experiment_name == manifest.experiment_name
    assert reloaded.git.git_commit == manifest.git.git_commit
    assert reloaded.packages.torch == manifest.packages.torch
    assert reloaded.artifacts["model"].sha256 == manifest.artifacts["model"].sha256


def test_collect_git_metadata_fallback() -> None:
    """Non-git environments gracefully return 'unknown' rather than crashing."""
    with patch("subprocess.run", side_effect=FileNotFoundError("git command not found")):
        meta = collect_git_metadata()
        assert meta.git_commit == "unknown"
        assert meta.git_branch == "unknown"
        assert meta.git_dirty is False

    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["git"], returncode=128, stdout="", stderr="fatal"
        ),
    ):
        meta = collect_git_metadata()
        assert meta.git_commit == "unknown"
        assert meta.git_branch == "unknown"
        assert meta.git_dirty is False


def test_software_and_system_metadata() -> None:
    """Verify host system and software packages metadata collection."""
    host = collect_host_system()
    assert host.os_name
    assert host.python_version
    assert host.architecture

    packages = collect_software_packages()
    assert packages.torch != "unknown"
    assert packages.numpy != "unknown"
    assert packages.pydantic != "unknown"
    assert packages.gymnasium != "unknown"

    hardware = collect_hardware_telemetry(device="cpu")
    assert hardware.device == "cpu"
    assert hardware.cpu_count >= 1


def test_security_sanitization_no_credentials(tmp_path: Path) -> None:
    """Ensure no environment secrets, keys, or passwords leak into manifest payload."""
    dirty_command = [
        "adaptive-rl",
        "train",
        "--token=SECRET_API_TOKEN_12345",
        "--api-key=CONFIDENTIAL_KEY",
        "--config",
        "configs/drone_ppo.yaml",
    ]

    manifest = create_manifest(
        experiment_name="secure_test",
        config_dict={"name": "secure_test", "seed": 42},
        started_at=1700000000.0,
        finished_at=1700000005.0,
        command=dirty_command,
    )

    manifest_json = manifest.model_dump_json()
    assert "SECRET_API_TOKEN" not in manifest_json
    assert "CONFIDENTIAL_KEY" not in manifest_json
    assert "--config" in manifest.execution.command


def test_trainer_fit_automatically_writes_manifest(tmp_path: Path) -> None:
    """Training run automatically creates and persists valid experiment manifest."""
    config = load_config(Path("configs/ci_smoke.yaml"))
    config.training.total_timesteps = 32
    config.training.checkpoint_freq = 0
    config.output_dir = tmp_path / "artifacts"

    trainer = RLTrainer(config)
    result = trainer.fit()
    trainer.close()

    assert result.manifest_path is not None
    assert result.manifest_path.exists()
    assert result.manifest_path.name == f"{config.name}_manifest.json"

    # Validate saved manifest
    manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["experiment_name"] == config.name
    assert "git" in manifest_data
    assert "packages" in manifest_data
    assert "artifacts" in manifest_data
    assert "model" in manifest_data["artifacts"]

    model_path = Path(manifest_data["artifacts"]["model"]["path"])
    assert model_path.exists()
    assert manifest_data["artifacts"]["model"]["sha256"] == compute_sha256(model_path)


def test_cli_inspect_manifest(tmp_path: Path) -> None:
    """Verify CLI inspection works for both inspect manifest and manifest inspect."""
    dummy_file = tmp_path / "artifact.bin"
    dummy_file.write_bytes(b"content")

    manifest = create_manifest(
        experiment_name="cli_inspect_test",
        config_dict={"algorithm": {"name": "ppo"}, "environment": {"name": "drone"}, "seed": 101},
        started_at=1700000000.0,
        finished_at=1700000002.0,
        artifacts={"file": dummy_file},
    )
    manifest_file = tmp_path / "test_manifest.json"
    save_manifest(manifest, manifest_file)

    # Test 'adaptive-rl inspect manifest <path>'
    res1 = runner.invoke(app, ["inspect", "manifest", str(manifest_file)])
    assert res1.exit_code == 0
    assert "cli_inspect_test" in res1.stdout
    assert "Git Metadata" in res1.stdout
    assert "Verified" in res1.stdout

    # Test 'adaptive-rl manifest inspect <path>'
    res2 = runner.invoke(app, ["manifest", "inspect", str(manifest_file)])
    assert res2.exit_code == 0
    assert "cli_inspect_test" in res2.stdout
    assert "Verified" in res2.stdout

    # Test error handling on missing manifest
    res_err = runner.invoke(app, ["inspect", "manifest", str(tmp_path / "missing.json")])
    assert res_err.exit_code == 1
    assert "Failed to inspect manifest" in res_err.stdout
