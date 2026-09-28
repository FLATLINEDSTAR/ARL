"""Tests for experiment provenance and environment manifest generation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from adaptive_rl.cli import app
from adaptive_rl.config import AlgorithmConfig, ExperimentConfig
from adaptive_rl.manifest import (
    ArtifactRecord,
    ExperimentManifest,
    GitMetadata,
    HostMetadata,
    PackageMetadata,
    compute_file_sha256,
    create_artifact_record,
    generate_manifest,
    get_git_metadata,
    get_hardware_metadata,
    get_host_metadata,
    get_package_metadata,
    load_manifest,
    sanitize_config_dict,
    sanitize_path,
)
from adaptive_rl.training.trainer import RLTrainer

runner = CliRunner()


# 1. Valid manifest generation
def test_valid_manifest_generation(tmp_path: Path) -> None:
    """Verify generate_manifest produces a complete and valid ExperimentManifest."""
    config = ExperimentConfig(
        name="test_manifest_exp",
        seed=101,
        algorithm=AlgorithmConfig(name="ppo", learning_rate=0.0003),
        output_dir=tmp_path / "artifacts",
    )

    dummy_model = tmp_path / "artifacts" / "models" / "test_final.zip"
    dummy_model.parent.mkdir(parents=True, exist_ok=True)
    dummy_model.write_bytes(b"dummy model checkpoint weights")

    manifest = generate_manifest(
        config=config,
        started_at=datetime.now(timezone.utc),
        artifacts=[dummy_model],
        base_dir=tmp_path / "artifacts",
    )

    assert manifest.manifest_version == "1.0.0"
    assert manifest.experiment_name == "test_manifest_exp"
    assert manifest.seed == 101
    assert manifest.algorithm == "ppo"
    assert len(manifest.artifacts) == 1
    assert manifest.artifacts[0].path == "models/test_final.zip"
    assert (
        manifest.artifacts[0].sha256
        == hashlib.sha256(b"dummy model checkpoint weights").hexdigest()
    )
    assert manifest.artifacts[0].size_bytes == len(b"dummy model checkpoint weights")

    target = tmp_path / "manifest.json"
    saved = manifest.save(target)
    assert saved.is_file()

    loaded = load_manifest(saved)
    assert loaded.experiment_name == manifest.experiment_name
    assert loaded.seed == manifest.seed
    assert len(loaded.artifacts) == 1


# 2. Pydantic schema validation
def test_pydantic_schema_validation() -> None:
    """Verify ExperimentManifest schema constraints and rejection of invalid data."""
    with pytest.raises(Exception):  # Missing required fields
        ExperimentManifest.model_validate({})

    with pytest.raises(Exception):  # Extra forbidden fields
        ExperimentManifest.model_validate(
            {
                "manifest_version": "1.0.0",
                "experiment_name": "test",
                "git": {},
                "host": {
                    "os_name": "Linux",
                    "os_version": "6.0",
                    "python_version": "3.10",
                    "architecture": "x86_64",
                },
                "packages": {},
                "hardware": {"device": "cpu"},
                "execution": {"started_at": "2026-01-01T00:00:00Z"},
                "experiment": {
                    "name": "test",
                    "algorithm": "ppo",
                    "seed": 42,
                    "config": {},
                },
                "unknown_extra_field": "disallowed",
            }
        )


# 3. Git metadata in normal repository
def test_git_metadata_normal_repository() -> None:
    """Verify get_git_metadata accurately probes a real Git repository."""
    git_meta = get_git_metadata()
    assert isinstance(git_meta, GitMetadata)
    assert git_meta.git_commit != "unknown"
    assert len(git_meta.git_commit) == 40
    assert isinstance(git_meta.git_dirty, bool)


# 4. Non-Git environment
def test_git_metadata_non_git_environment(tmp_path: Path) -> None:
    """Verify get_git_metadata safely returns defaults outside a Git repo."""
    git_meta = get_git_metadata(repo_path=tmp_path)
    assert git_meta.git_commit == "unknown"
    assert git_meta.git_branch is None
    assert git_meta.git_dirty is False


# 5. Git executable unavailable or failure
def test_git_metadata_executable_unavailable_or_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_git_metadata handles missing Git binary or timeouts gracefully."""

    def mock_run_fnf(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("git binary not found")

    monkeypatch.setattr(subprocess, "run", mock_run_fnf)
    git_meta = get_git_metadata()
    assert git_meta.git_commit == "unknown"
    assert git_meta.git_branch is None
    assert git_meta.git_dirty is False

    def mock_run_timeout(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(subprocess, "run", mock_run_timeout)
    git_meta_timeout = get_git_metadata()
    assert git_meta_timeout.git_commit == "unknown"


# 6. Detached HEAD behavior
def test_git_metadata_detached_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_git_metadata records 'detached' when in detached HEAD state."""

    class MockCompletedProcess:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    def mock_run(cmd: list[str], *args: Any, **kwargs: Any) -> MockCompletedProcess:
        if "rev-parse" in cmd:
            return MockCompletedProcess(0, "a" * 40 + "\n")
        elif "symbolic-ref" in cmd:
            return MockCompletedProcess(1, "")  # Detached HEAD
        elif "status" in cmd:
            return MockCompletedProcess(0, "")
        return MockCompletedProcess(0, "")

    monkeypatch.setattr(subprocess, "run", mock_run)
    git_meta = get_git_metadata()
    assert git_meta.git_commit == "a" * 40
    assert git_meta.git_branch == "detached"
    assert git_meta.git_dirty is False


# 7. Dirty working tree behavior
def test_git_metadata_dirty_working_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_git_metadata detects uncommitted modifications."""

    class MockCompletedProcess:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    def mock_run(cmd: list[str], *args: Any, **kwargs: Any) -> MockCompletedProcess:
        if "rev-parse" in cmd:
            return MockCompletedProcess(0, "b" * 40 + "\n")
        elif "symbolic-ref" in cmd:
            return MockCompletedProcess(0, "feature-branch\n")
        elif "status" in cmd:
            return MockCompletedProcess(0, " M modified_file.py\n")
        return MockCompletedProcess(0, "")

    monkeypatch.setattr(subprocess, "run", mock_run)
    git_meta = get_git_metadata()
    assert git_meta.git_dirty is True
    assert git_meta.git_branch == "feature-branch"


# 8. Python/OS/architecture metadata
def test_host_metadata() -> None:
    """Verify get_host_metadata collects valid OS and Python runtime attributes."""
    host_meta = get_host_metadata()
    assert isinstance(host_meta, HostMetadata)
    assert host_meta.os_name == platform.system()
    assert host_meta.python_version == platform.python_version()
    assert host_meta.architecture == platform.machine()
    assert len(host_meta.os_version) > 0


# 9. Package version metadata
def test_package_version_metadata() -> None:
    """Verify get_package_metadata gathers installed versions and handles missing packages."""
    pkg_meta = get_package_metadata()
    assert isinstance(pkg_meta, PackageMetadata)
    assert pkg_meta.adaptive_rl is not None
    assert pkg_meta.numpy is not None
    assert pkg_meta.pydantic is not None

    # Test unknown extra package does not raise
    extra_meta = get_package_metadata(extra_packages=["nonexistent_package_xyz_123"])
    assert getattr(extra_meta, "nonexistent_package_xyz_123", None) is None


# 10. Complete configuration serialization
def test_complete_configuration_serialization(tmp_path: Path) -> None:
    """Verify complete ExperimentConfig is captured and alterations affect config_sha256."""
    config1 = ExperimentConfig(
        name="cfg_test_1",
        seed=42,
        algorithm=AlgorithmConfig(name="ppo", learning_rate=0.0003),
    )
    config2 = ExperimentConfig(
        name="cfg_test_1",
        seed=42,
        algorithm=AlgorithmConfig(name="ppo", learning_rate=0.001),  # Different LR
    )

    m1 = generate_manifest(config=config1, started_at=datetime.now(timezone.utc))
    m2 = generate_manifest(config=config2, started_at=datetime.now(timezone.utc))

    assert m1.experiment.config["algorithm"]["learning_rate"] == 0.0003
    assert m2.experiment.config["algorithm"]["learning_rate"] == 0.001
    assert m1.experiment.config_sha256 != m2.experiment.config_sha256


# 11. Seed preservation
def test_seed_preservation() -> None:
    """Verify seed is preserved across manifest models and properties."""
    for test_seed in (0, 42, 999999):
        config = ExperimentConfig(name="seed_exp", seed=test_seed)
        manifest = generate_manifest(config=config, started_at=datetime.now(timezone.utc))
        assert manifest.seed == test_seed
        assert manifest.experiment.seed == test_seed
        assert manifest.experiment.config["seed"] == test_seed


# 12. CPU/CUDA device metadata behavior
def test_hardware_telemetry_cpu_and_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_hardware_metadata behavior for CPU and mocked CUDA environments."""
    hw_cpu = get_hardware_metadata(device="cpu")
    assert hw_cpu.device == "cpu"
    assert hw_cpu.cuda_device_name is None
    assert hw_cpu.cpu_count is not None and hw_cpu.cpu_count > 0

    # Mock CUDA available
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "NVIDIA RTX 4090 Test")

    hw_cuda = get_hardware_metadata(device="cuda:0")
    assert hw_cuda.device == "cuda:0"
    assert hw_cuda.cuda_device_name == "NVIDIA RTX 4090 Test"

    # Mock CUDA unavailable when requested
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    hw_no_cuda = get_hardware_metadata(device="cuda")
    assert hw_no_cuda.device == "cpu"
    assert hw_no_cuda.cuda_device_name is None

    # Test non-CUDA device (e.g. Apple Silicon mps) is preserved
    hw_mps = get_hardware_metadata(device="mps")
    assert hw_mps.device == "mps"
    assert hw_mps.cuda_device_name is None


# 13. SHA-256 calculation against a known fixture
def test_sha256_calculation_against_known_fixture(tmp_path: Path) -> None:
    """Verify compute_file_sha256 against standard SHA-256 test vectors."""
    # 1. Empty file
    empty_file = tmp_path / "empty.txt"
    empty_file.write_bytes(b"")
    empty_hash = compute_file_sha256(empty_file)
    assert empty_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    # 2. Known string
    msg_file = tmp_path / "message.txt"
    msg_file.write_bytes(b"AdaptiveRL Experiment Manifest\n")
    expected_hash = hashlib.sha256(b"AdaptiveRL Experiment Manifest\n").hexdigest()
    assert compute_file_sha256(msg_file) == expected_hash

    # 3. Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        compute_file_sha256(tmp_path / "nonexistent.bin")


# 14. Artifact paths and checksum records
def test_artifact_paths_and_checksum_records(tmp_path: Path) -> None:
    """Verify create_artifact_record formats relative paths, size, and checksums."""
    model_path = tmp_path / "artifacts" / "models" / "drone_test.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"PK\x03\x04modelweights")

    record = create_artifact_record(
        model_path,
        artifact_type="model",
        base_dir=tmp_path / "artifacts",
    )
    assert isinstance(record, ArtifactRecord)
    assert record.path == "models/drone_test.zip"
    assert record.artifact_type == "model"
    assert record.size_bytes == len(b"PK\x03\x04modelweights")
    assert record.sha256 == hashlib.sha256(b"PK\x03\x04modelweights").hexdigest()


# 15. Secret environment variables do not leak
def test_secrets_and_credentials_do_not_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify environment variables and sensitive configuration values are never serialized."""
    secret_key = "sk_live_very_secret_api_key_12345"
    token_val = "ghp_super_secret_github_token_67890"
    password_val = "SuperSecretPassword!@#$"

    monkeypatch.setenv("OPENAI_API_KEY", secret_key)
    monkeypatch.setenv("GITHUB_TOKEN", token_val)
    monkeypatch.setenv("DB_PASSWORD", password_val)

    config = ExperimentConfig(
        name="security_test",
        algorithm=AlgorithmConfig(
            name="ppo",
            parameters={"api_key": secret_key, "auth_token": token_val},
        ),
        output_dir=tmp_path / "artifacts",
    )

    manifest = generate_manifest(
        config=config,
        started_at=datetime.now(timezone.utc),
        command=f"adaptive-rl train --token {token_val} --config conf.yaml",
    )
    serialized = manifest.model_dump_json()

    assert secret_key not in serialized
    assert token_val not in serialized
    assert password_val not in serialized
    # Check masked values
    assert manifest.experiment.config["algorithm"]["parameters"]["api_key"] == "***"
    assert manifest.experiment.config["algorithm"]["parameters"]["auth_token"] == "***"


# 16. Absolute and private paths are not leaked
def test_absolute_and_private_paths_not_leaked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify sanitize_path avoids exposing home directory paths and handles root home safely."""
    home = Path.home()
    test_path = home / "my_secret_subfolder" / "experiment_output.zip"

    sanitized = sanitize_path(test_path)
    assert str(home) not in sanitized
    assert sanitized.startswith("~/") or not sanitized.startswith("/")

    # Test config dictionary path sanitization
    raw_dict = {"path": test_path, "sub": {"file": test_path}}
    sanitized_dict = sanitize_config_dict(raw_dict)
    assert str(home) not in str(sanitized_dict)

    # Test root home directory edge case does not corrupt paths
    monkeypatch.setattr(Path, "home", lambda: Path("/"))
    root_sanitized = sanitize_path(Path("/usr/bin/python"))
    assert "~" not in root_sanitized
    assert root_sanitized == "python"

    cfg_root = sanitize_config_dict({"cmd": "/usr/bin/python"})
    assert cfg_root["cmd"] == "/usr/bin/python"


# 17. CLI manifest inspection
def test_cli_inspect_manifest_valid(tmp_path: Path) -> None:
    """Verify 'adaptive-rl inspect manifest' succeeds on valid manifests."""
    config = ExperimentConfig(name="cli_inspect_exp", output_dir=tmp_path)
    manifest = generate_manifest(config=config, started_at=datetime.now(timezone.utc))
    manifest_file = tmp_path / "manifest.json"
    manifest.save(manifest_file)

    res = runner.invoke(app, ["inspect", "manifest", str(manifest_file)])
    assert res.exit_code == 0
    assert "Experiment Manifest is valid" in res.output
    assert "cli_inspect_exp" in res.output
    assert "PPO" in res.output


# 18. Malformed manifest handling
def test_cli_inspect_manifest_malformed(tmp_path: Path) -> None:
    """Verify 'adaptive-rl inspect manifest' fails cleanly with exit code 1 on malformed input."""
    # 1. Not JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{this is not valid json", encoding="utf-8")
    res1 = runner.invoke(app, ["inspect", "manifest", str(bad_json)])
    assert res1.exit_code == 1
    assert "Manifest validation error" in res1.output

    # 2. Valid JSON but invalid manifest schema
    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text(json.dumps({"invalid_manifest": 123}), encoding="utf-8")
    res2 = runner.invoke(app, ["inspect", "manifest", str(bad_schema)])
    assert res2.exit_code == 1
    assert "Manifest validation error" in res2.output


# 19. Missing manifest handling
def test_cli_inspect_manifest_missing(tmp_path: Path) -> None:
    """Verify 'adaptive-rl inspect manifest' fails cleanly when file does not exist."""
    missing = tmp_path / "nonexistent_manifest.json"
    res = runner.invoke(app, ["inspect", "manifest", str(missing)])
    assert res.exit_code == 1
    assert "Manifest file not found" in res.output


# 20. Existing training tests still pass and fit() generates manifest automatically
def test_trainer_fit_generates_manifest_automatically(tmp_path: Path) -> None:
    """Verify RLTrainer.fit() writes manifest.json and links it in TrainingResult."""
    test_config = tmp_path / "trainer_manifest_test.yaml"
    test_config.write_text(
        f"""
name: "fit_manifest_test"
seed: 42
algorithm:
  name: "ppo"
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 16
  parameters:
    n_steps: 32
environment:
  name: "drone"
  max_steps: 10
  parameters:
    bounds: [20.0, 20.0, 10.0]
    num_obstacles: 1
training:
  total_timesteps: 32
  checkpoint_freq: 0
  log_interval: 10
evaluation:
  eval_episodes: 1
output_dir: "{tmp_path / "artifacts"}"
log_dir: "{tmp_path / "logs"}"
""",
        encoding="utf-8",
    )

    from adaptive_rl.config import load_config

    cfg = load_config(test_config)
    trainer = RLTrainer(config=cfg)
    result = trainer.fit()

    assert result.manifest_path is not None
    assert result.manifest_path.is_file()
    assert "fit_manifest_test_manifest.json" in str(result.manifest_path)

    # Check canonical experiment.json in experiments dir
    canonical = tmp_path / "artifacts" / "experiments" / "fit_manifest_test" / "experiment.json"
    assert canonical.is_file()

    # Load and verify manifest
    manifest = load_manifest(result.manifest_path)
    assert manifest.experiment_name == "fit_manifest_test"
    assert manifest.seed == 42
    assert len(manifest.artifacts) >= 2  # Model .zip and metadata .json

    # Verify model artifact checksum
    model_record = next(a for a in manifest.artifacts if a.artifact_type == "model")
    assert model_record.sha256 == compute_file_sha256(result.final_model_path)
