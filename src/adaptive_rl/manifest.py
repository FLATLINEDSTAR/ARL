"""Experiment provenance and environment manifest system for AdaptiveRL.

Implements typed Pydantic models and utility functions for capturing, serializing,
and inspecting experiment metadata, including Git provenance, host system info,
dependency versions, hardware telemetry, effective configuration, and artifact
integrity checksums (SHA-256).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from adaptive_rl.config import ExperimentConfig, compute_config_sha256

logger = logging.getLogger(__name__)

# Standard dependencies tracked in software provenance
CORE_PACKAGES = (
    "adaptive_rl",
    "torch",
    "stable_baselines3",
    "gymnasium",
    "numpy",
    "typer",
    "pydantic",
)


class ManifestError(Exception):
    """Raised for manifest generation, validation, or loading failures."""

    pass


class GitMetadata(BaseModel):
    """Git version control provenance."""

    model_config = ConfigDict(extra="forbid")

    git_commit: str = Field("unknown", description="Git commit SHA or 'unknown'")
    git_branch: Optional[str] = Field(None, description="Active branch name, 'detached', or None")
    git_dirty: bool = Field(False, description="Whether uncommitted changes exist in working tree")


class HostMetadata(BaseModel):
    """Host operating system and Python runtime environment."""

    model_config = ConfigDict(extra="forbid")

    os_name: str = Field(..., description="Operating system name, e.g. 'Linux'")
    os_version: str = Field(..., description="Operating system release/kernel version")
    python_version: str = Field(..., description="Python interpreter version")
    architecture: str = Field(..., description="CPU architecture, e.g. 'x86_64'")


class PackageMetadata(BaseModel):
    """Installed versions of key dependencies."""

    model_config = ConfigDict(extra="allow")

    adaptive_rl: Optional[str] = Field(None, description="AdaptiveRL version")
    torch: Optional[str] = Field(None, description="PyTorch version")
    stable_baselines3: Optional[str] = Field(None, description="Stable-Baselines3 version")
    gymnasium: Optional[str] = Field(None, description="Gymnasium version")
    numpy: Optional[str] = Field(None, description="NumPy version")
    typer: Optional[str] = Field(None, description="Typer version")
    pydantic: Optional[str] = Field(None, description="Pydantic version")


class HardwareMetadata(BaseModel):
    """Compute hardware telemetry."""

    model_config = ConfigDict(extra="forbid")

    device: str = Field("cpu", description="Compute device utilized ('cpu' or 'cuda')")
    cuda_device_name: Optional[str] = Field(
        None, description="CUDA GPU model name if GPU was utilized"
    )
    cpu_count: Optional[int] = Field(None, description="Number of logical CPU cores")


class ExecutionMetadata(BaseModel):
    """Execution lifecycle and invocation details."""

    model_config = ConfigDict(extra="forbid")

    started_at: str = Field(..., description="ISO 8601 UTC timestamp of execution start")
    finished_at: Optional[str] = Field(
        None, description="ISO 8601 UTC timestamp of execution completion"
    )
    duration_seconds: Optional[float] = Field(
        None, ge=0.0, description="Total execution duration in seconds"
    )
    command: Optional[str] = Field(None, description="Sanitized invocation command")


class ArtifactRecord(BaseModel):
    """Provenance and cryptographic hash of an output artifact."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Relative path to artifact within workspace")
    sha256: str = Field(..., description="SHA-256 integrity checksum hex digest")
    size_bytes: int = Field(..., ge=0, description="Artifact file size in bytes")
    artifact_type: str = Field(
        "unknown", description="Category: 'model', 'metadata', 'checkpoint', etc."
    )


class ExperimentMetadata(BaseModel):
    """High-level experiment identifiers and complete effective configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Experiment name")
    algorithm: str = Field(..., description="Algorithm name, e.g. 'ppo' or 'sac'")
    seed: int = Field(..., description="Random seed")
    training_budget: Optional[int] = Field(
        None, description="Total timesteps budgeted for training"
    )
    environment_name: str = Field(..., description="Registered environment name")
    environment_parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Environment parameters"
    )
    config: Dict[str, Any] = Field(
        ..., description="Complete deserialized ExperimentConfig dictionary"
    )
    config_sha256: Optional[str] = Field(
        None, description="Deterministic SHA-256 fingerprint of the configuration"
    )


class ExperimentManifest(BaseModel):
    """Complete machine-readable experiment manifest recording environment provenance."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: str = Field("1.0.0", description="Schema version of this manifest")
    experiment_name: str = Field(..., description="Experiment identifier")
    git: GitMetadata = Field(default_factory=GitMetadata)
    host: HostMetadata
    packages: PackageMetadata
    hardware: HardwareMetadata
    execution: ExecutionMetadata
    experiment: ExperimentMetadata
    artifacts: List[ArtifactRecord] = Field(
        default_factory=list, description="Output artifacts and SHA-256 checksums"
    )

    @property
    def git_commit(self) -> str:
        """Convenience property for Git commit hash."""
        return self.git.git_commit

    @property
    def git_branch(self) -> Optional[str]:
        """Convenience property for Git branch name."""
        return self.git.git_branch

    @property
    def git_dirty(self) -> bool:
        """Convenience property for Git dirty working tree flag."""
        return self.git.git_dirty

    @property
    def os_name(self) -> str:
        """Convenience property for operating system name."""
        return self.host.os_name

    @property
    def python_version(self) -> str:
        """Convenience property for Python interpreter version."""
        return self.host.python_version

    @property
    def seed(self) -> int:
        """Convenience property for experiment seed."""
        return self.experiment.seed

    @property
    def algorithm(self) -> str:
        """Convenience property for experiment algorithm name."""
        return self.experiment.algorithm

    def save(self, path: Union[str, Path], atomic: bool = True) -> Path:
        """Save this manifest to a JSON file."""
        return save_manifest(self, path, atomic=atomic)

    @classmethod
    def load(cls, path: Union[str, Path]) -> ExperimentManifest:
        """Load and validate an experiment manifest from a JSON file."""
        return load_manifest(path)


def compute_file_sha256(file_path: Union[Path, str], chunk_size: int = 65536) -> str:
    """Compute the SHA-256 checksum of a file using streaming reads."""
    target = Path(file_path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {target}")

    hasher = hashlib.sha256()
    with open(target, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def sanitize_path(path: Union[Path, str], base_dir: Optional[Path] = None) -> str:
    """Sanitize a file path to avoid leaking absolute home directory paths."""
    p = Path(path)

    # 1. If base_dir is supplied, attempt relative path to base_dir
    if base_dir is not None:
        try:
            return p.resolve().relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            pass

    # 2. Attempt relative path to current working directory
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        pass

    # 3. If within user home directory, mask prefix with '~/'
    try:
        home = Path.home().resolve()
        if str(home) not in ("/", "") and p.resolve().is_relative_to(home):
            rel = p.resolve().relative_to(home)
            return f"~/{rel.as_posix()}"
    except (ValueError, AttributeError):
        pass

    # 4. Fallback to filename
    return p.name


def sanitize_command(args: Optional[Union[str, Sequence[str]]] = None) -> str:
    """Sanitize CLI command string, redacting potential secret arguments and absolute home paths."""
    if args is None:
        tokens = list(sys.argv)
    elif isinstance(args, str):
        tokens = args.split()
    else:
        tokens = list(args)

    if not tokens:
        return ""

    sanitized_tokens: List[str] = []
    redact_next = False
    sensitive_flags = {
        "--api-key",
        "--token",
        "--password",
        "--secret",
        "--auth",
        "--key",
        "-k",
    }

    for token in tokens:
        if redact_next:
            sanitized_tokens.append("***")
            redact_next = False
            continue

        lower = token.lower()
        if any(lower == flag or lower.startswith(f"{flag}=") for flag in sensitive_flags):
            if "=" in token:
                flag_name, _ = token.split("=", 1)
                sanitized_tokens.append(f"{flag_name}=***")
            else:
                sanitized_tokens.append(token)
                redact_next = True
            continue

        # Redact token-like patterns (e.g. ghp_..., sk-...)
        if any(token.startswith(prefix) for prefix in ("ghp_", "sk-", "gho_", "ghu_")):
            sanitized_tokens.append("***")
            continue

        # Sanitize home directory from paths in arguments
        try:
            home = str(Path.home())
            if home not in ("/", "") and home in token:
                token = token.replace(home, "~")
        except Exception:
            pass

        sanitized_tokens.append(token)

    return " ".join(sanitized_tokens)


def sanitize_config_dict(data: Any) -> Any:
    """Recursively sanitize configuration dictionaries to ensure no secrets or private home paths leak."""
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(
                term in k_lower
                for term in ("key", "token", "secret", "password", "auth", "credential")
            ):
                sanitized[k] = "***"
            else:
                sanitized[k] = sanitize_config_dict(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_config_dict(item) for item in data]
    elif isinstance(data, Path):
        return sanitize_path(data)
    elif isinstance(data, str):
        try:
            home = str(Path.home())
            if home not in ("/", "") and home in data:
                return data.replace(home, "~")
        except Exception:
            pass
        return data
    return data


def get_git_metadata(repo_path: Optional[Path] = None) -> GitMetadata:
    """Query Git version control provenance safely without raising exceptions."""
    cwd = str(repo_path) if repo_path is not None else str(Path.cwd())
    commit = "unknown"
    branch = None
    dirty = False

    try:
        # Check commit hash
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            commit = proc.stdout.strip()
        else:
            return GitMetadata(git_commit="unknown", git_branch=None, git_dirty=False)

        # Check branch name
        proc_b = subprocess.run(
            ["git", "symbolic-ref", "--short", "-q", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc_b.returncode == 0 and proc_b.stdout.strip():
            branch = proc_b.stdout.strip()
        else:
            branch = "detached"

        # Check dirty working tree
        proc_d = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc_d.returncode == 0:
            dirty = bool(proc_d.stdout.strip())

    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, Exception):
        commit = "unknown"
        branch = None
        dirty = False

    return GitMetadata(git_commit=commit, git_branch=branch, git_dirty=dirty)


def get_host_metadata() -> HostMetadata:
    """Collect host operating system and Python interpreter runtime metadata."""
    return HostMetadata(
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
        architecture=platform.machine(),
    )


def get_package_metadata(extra_packages: Optional[Sequence[str]] = None) -> PackageMetadata:
    """Retrieve installed versions of core project dependencies via importlib.metadata."""
    targets = list(CORE_PACKAGES)
    if extra_packages:
        targets.extend(extra_packages)

    versions: Dict[str, Optional[str]] = {}
    for pkg in targets:
        ver: Optional[str] = None
        # Try both underscored and hyphenated distribution names
        candidate_names = [pkg, pkg.replace("_", "-")]
        for name in candidate_names:
            try:
                ver = importlib.metadata.version(name)
                break
            except importlib.metadata.PackageNotFoundError:
                pass
            except Exception:
                pass

        if ver is None and pkg == "adaptive_rl":
            try:
                import adaptive_rl

                ver = getattr(adaptive_rl, "__version__", None)
            except Exception:
                pass

        versions[pkg] = ver

    return PackageMetadata(**versions)


def get_hardware_metadata(device: Optional[str] = None) -> HardwareMetadata:
    """Determine compute device and query CUDA telemetry if actively used."""
    dev_str = str(device).lower() if device is not None else "cpu"
    cuda_name: Optional[str] = None

    if "cuda" in dev_str:
        try:
            import torch

            if torch.cuda.is_available():
                cuda_idx = 0
                if ":" in dev_str:
                    try:
                        cuda_idx = int(dev_str.split(":")[-1])
                    except ValueError:
                        cuda_idx = 0
                cuda_name = torch.cuda.get_device_name(cuda_idx)
            else:
                dev_str = "cpu"
        except Exception:
            dev_str = "cpu"

    return HardwareMetadata(
        device=dev_str,
        cuda_device_name=cuda_name,
        cpu_count=os.cpu_count(),
    )


def create_artifact_record(
    file_path: Union[Path, str],
    artifact_type: str = "unknown",
    base_dir: Optional[Path] = None,
) -> ArtifactRecord:
    """Generate an ArtifactRecord with path, SHA-256 checksum, and file size."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Artifact not found: {p}")

    sha256 = compute_file_sha256(p)
    size_bytes = p.stat().st_size
    rel_path = sanitize_path(p, base_dir=base_dir)

    return ArtifactRecord(
        path=rel_path,
        sha256=sha256,
        size_bytes=size_bytes,
        artifact_type=artifact_type,
    )


def generate_manifest(
    config: ExperimentConfig,
    started_at: Union[float, datetime],
    finished_at: Optional[Union[float, datetime]] = None,
    artifacts: Optional[Sequence[Union[Path, str, ArtifactRecord]]] = None,
    device: Optional[str] = None,
    command: Optional[Union[str, Sequence[str]]] = None,
    base_dir: Optional[Path] = None,
    repo_path: Optional[Path] = None,
) -> ExperimentManifest:
    """Construct an ExperimentManifest capturing complete environment, code, and config provenance."""
    # 1. Timestamps and duration
    if isinstance(started_at, (int, float)):
        start_dt = datetime.fromtimestamp(started_at, tz=timezone.utc)
    else:
        start_dt = started_at if started_at.tzinfo else started_at.replace(tzinfo=timezone.utc)

    if finished_at is None:
        finish_dt = datetime.now(timezone.utc)
    elif isinstance(finished_at, (int, float)):
        finish_dt = datetime.fromtimestamp(finished_at, tz=timezone.utc)
    else:
        finish_dt = finished_at if finished_at.tzinfo else finished_at.replace(tzinfo=timezone.utc)

    duration = max(0.0, (finish_dt - start_dt).total_seconds())

    # 2. Metadata groups
    git_meta = get_git_metadata(repo_path=repo_path)
    host_meta = get_host_metadata()
    package_meta = get_package_metadata()
    hardware_meta = get_hardware_metadata(device=device)
    sanitized_cmd = sanitize_command(command)

    execution_meta = ExecutionMetadata(
        started_at=start_dt.isoformat(),
        finished_at=finish_dt.isoformat(),
        duration_seconds=round(duration, 4),
        command=sanitized_cmd,
    )

    # 3. Serialized effective configuration (sanitized)
    raw_config = config.model_dump(mode="json")
    sanitized_config = sanitize_config_dict(raw_config)
    config_sha = compute_config_sha256(config)

    experiment_meta = ExperimentMetadata(
        name=config.name,
        algorithm=config.algorithm.name,
        seed=config.seed,
        training_budget=config.training.total_timesteps if config.training else None,
        environment_name=config.environment.name,
        environment_parameters=sanitize_config_dict(dict(config.environment.parameters)),
        config=sanitized_config,
        config_sha256=config_sha,
    )

    # 4. Artifact records
    artifact_records: List[ArtifactRecord] = []
    if artifacts:
        for item in artifacts:
            if isinstance(item, ArtifactRecord):
                artifact_records.append(item)
            else:
                p = Path(item)
                if not p.is_file():
                    logger.warning("Artifact file not found for manifest: %s", p)
                    continue
                # Guess artifact type from extension
                suffix = p.suffix.lower()
                art_type = (
                    "model"
                    if suffix == ".zip"
                    else "metadata"
                    if suffix == ".json"
                    else "data"
                    if suffix == ".csv"
                    else "checkpoint"
                    if "checkpoint" in p.name.lower()
                    else "artifact"
                )
                artifact_records.append(
                    create_artifact_record(p, artifact_type=art_type, base_dir=base_dir)
                )

    return ExperimentManifest(
        manifest_version="1.0.0",
        experiment_name=config.name,
        git=git_meta,
        host=host_meta,
        packages=package_meta,
        hardware=hardware_meta,
        execution=execution_meta,
        experiment=experiment_meta,
        artifacts=artifact_records,
    )


def save_manifest(
    manifest: ExperimentManifest,
    path: Union[str, Path],
    atomic: bool = True,
) -> Path:
    """Write an ExperimentManifest to disk as formatted JSON, using atomic replace if requested."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump_json(indent=2)

    if atomic:
        tmp_target = target.with_suffix(f".tmp.{os.getpid()}")
        try:
            with open(tmp_target, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_target, target)
        except Exception:
            if tmp_target.exists():
                try:
                    tmp_target.unlink()
                except Exception:
                    pass
            raise
    else:
        with open(target, "w", encoding="utf-8") as f:
            f.write(payload)

    return target


def load_manifest(path: Union[str, Path]) -> ExperimentManifest:
    """Load, parse, and validate an ExperimentManifest from disk."""
    target = Path(path)
    if not target.is_file():
        raise ManifestError(f"Manifest file not found: {target}")

    try:
        with open(target, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Malformed JSON in manifest at {target}: {exc}") from exc
    except Exception as exc:
        raise ManifestError(f"Failed to read manifest at {target}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestError(
            f"Manifest at {target} must contain a JSON object, got {type(raw).__name__}"
        )

    try:
        return ExperimentManifest.model_validate(raw)
    except ValidationError as exc:
        raise ManifestError(f"Manifest validation failed for {target}: {exc}") from exc


__all__ = [
    "ArtifactRecord",
    "CORE_PACKAGES",
    "ExecutionMetadata",
    "ExperimentManifest",
    "ExperimentMetadata",
    "GitMetadata",
    "HardwareMetadata",
    "HostMetadata",
    "ManifestError",
    "PackageMetadata",
    "compute_file_sha256",
    "create_artifact_record",
    "generate_manifest",
    "get_git_metadata",
    "get_hardware_metadata",
    "get_host_metadata",
    "get_package_metadata",
    "load_manifest",
    "sanitize_command",
    "sanitize_config_dict",
    "sanitize_path",
    "save_manifest",
]
