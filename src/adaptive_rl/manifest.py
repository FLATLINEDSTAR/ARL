"""Experiment manifest generation, schema validation, and artifact provenance (Issue #248).

Captures software versions, host metadata, Git commit and dirty status, hardware
telemetry, deserialized experiment configuration, execution duration, and SHA-256
integrity checksums of generated model weights and evaluation reports.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path | str) -> str:
    """Compute deterministic SHA-256 hex digest of a file in chunks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class GitMetadata(BaseModel):
    """Git repository metadata at the time of experiment execution."""

    model_config = ConfigDict(extra="forbid")

    git_commit: str = Field(..., description="Git commit SHA-1 or 'unknown'")
    git_branch: str = Field(..., description="Git branch name or 'unknown'")
    git_dirty: bool = Field(False, description="True if uncommitted changes were present")


class HostSystem(BaseModel):
    """Operating system and Python runtime environment."""

    model_config = ConfigDict(extra="forbid")

    os_name: str = Field(..., description="Operating system name (e.g. Linux, Darwin)")
    os_version: str = Field(..., description="OS release version")
    python_version: str = Field(..., description="Python interpreter version")
    architecture: str = Field(..., description="Hardware machine architecture")


class SoftwarePackages(BaseModel):
    """Pinned library package versions for reproducibility."""

    model_config = ConfigDict(extra="forbid")

    adaptive_rl: str = Field(..., description="adaptive-rl package version")
    torch: str = Field(..., description="PyTorch version")
    stable_baselines3: str = Field(..., description="Stable-Baselines3 version")
    gymnasium: str = Field(..., description="Farama Gymnasium version")
    numpy: str = Field(..., description="NumPy version")
    typer: str = Field(..., description="Typer CLI version")
    pydantic: str = Field(..., description="Pydantic version")


class HardwareTelemetry(BaseModel):
    """Compute hardware and acceleration device information."""

    model_config = ConfigDict(extra="forbid")

    device: str = Field("cpu", description="Primary compute device utilized (e.g. cpu or cuda)")
    gpu_name: Optional[str] = Field(None, description="GPU model name if available")
    gpu_count: int = Field(0, ge=0, description="Number of detected CUDA GPUs")
    cpu_count: int = Field(1, gt=0, description="Logical CPU core count")


class ExecutionMetadata(BaseModel):
    """Temporal and command invocation metadata."""

    model_config = ConfigDict(extra="forbid")

    started_at: str = Field(..., description="ISO 8601 UTC start timestamp")
    finished_at: str = Field(..., description="ISO 8601 UTC completion timestamp")
    duration_seconds: float = Field(..., ge=0.0, description="Total wall-clock duration")
    training_time_seconds: Optional[float] = Field(
        None, ge=0.0, description="Exclusive interaction/training duration"
    )
    command: List[str] = Field(default_factory=list, description="CLI invocation arguments")


class ArtifactProvenance(BaseModel):
    """Checksum and location metadata for an experiment artifact."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Filesystem path of the artifact")
    sha256: str = Field(..., description="SHA-256 cryptographic checksum")
    size_bytes: int = Field(..., ge=0, description="Artifact size in bytes")


class ExperimentManifest(BaseModel):
    """Comprehensive experiment manifest capturing provenance and environment metadata."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: str = Field("1.0", description="Manifest schema version")
    experiment_name: str = Field(..., description="Name identifier of the experiment")
    git: GitMetadata
    host: HostSystem
    packages: SoftwarePackages
    hardware: HardwareTelemetry
    execution: ExecutionMetadata
    config: Dict[str, Any] = Field(..., description="Sanitized experiment configuration")
    artifacts: Dict[str, ArtifactProvenance] = Field(
        default_factory=dict, description="Generated artifacts with checksums"
    )


def collect_git_metadata() -> GitMetadata:
    """Safely query Git repository state with graceful fallback for non-git environments."""
    commit = "unknown"
    branch = "unknown"
    dirty = False

    try:
        res_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res_commit.returncode == 0 and res_commit.stdout.strip():
            commit = res_commit.stdout.strip()

        res_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res_branch.returncode == 0 and res_branch.stdout.strip():
            branch = res_branch.stdout.strip()

        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res_status.returncode == 0:
            dirty = bool(res_status.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError, Exception) as exc:
        logger.debug("Failed to query git metadata: %s", exc)

    return GitMetadata(git_commit=commit, git_branch=branch, git_dirty=dirty)


def collect_host_system() -> HostSystem:
    """Collect host operating system and Python environment information."""
    return HostSystem(
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
        architecture=platform.machine(),
    )


def _safe_package_version(pkg_name: str) -> str:
    """Query installed package version with fallback."""
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:
        pass

    try:
        mod = sys.modules.get(pkg_name)
        if mod and hasattr(mod, "__version__"):
            return str(mod.__version__)
    except Exception:
        pass

    return "unknown"


def collect_software_packages() -> SoftwarePackages:
    """Collect exact pinned versions of critical software packages."""
    return SoftwarePackages(
        adaptive_rl=_safe_package_version("adaptive-rl"),
        torch=_safe_package_version("torch"),
        stable_baselines3=_safe_package_version("stable-baselines3"),
        gymnasium=_safe_package_version("gymnasium"),
        numpy=_safe_package_version("numpy"),
        typer=_safe_package_version("typer"),
        pydantic=_safe_package_version("pydantic"),
    )


def collect_hardware_telemetry(device: Optional[str] = None) -> HardwareTelemetry:
    """Collect hardware compute capabilities (CPU cores and GPU devices)."""
    cpu_count = os.cpu_count() or 1
    gpu_name: Optional[str] = None
    gpu_count = 0
    dev_str = device or "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            if not device:
                dev_str = "cuda"
    except Exception:
        pass

    return HardwareTelemetry(
        device=dev_str,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        cpu_count=cpu_count,
    )


def create_manifest(
    experiment_name: str,
    config_dict: Dict[str, Any],
    started_at: datetime | float,
    finished_at: datetime | float,
    artifacts: Optional[Dict[str, Path | str]] = None,
    training_time_seconds: Optional[float] = None,
    command: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> ExperimentManifest:
    """Create a validated ExperimentManifest instance from execution parameters."""
    if isinstance(started_at, (int, float)):
        start_dt = datetime.fromtimestamp(started_at, tz=timezone.utc)
    else:
        start_dt = started_at.astimezone(timezone.utc)

    if isinstance(finished_at, (int, float)):
        finish_dt = datetime.fromtimestamp(finished_at, tz=timezone.utc)
    else:
        finish_dt = finished_at.astimezone(timezone.utc)

    duration = max(0.0, (finish_dt - start_dt).total_seconds())

    # Record artifacts with SHA-256 hashes
    artifact_records: Dict[str, ArtifactProvenance] = {}
    if artifacts:
        for name, art_path in artifacts.items():
            p = Path(art_path)
            if p.is_file():
                artifact_records[name] = ArtifactProvenance(
                    path=str(p),
                    sha256=compute_sha256(p),
                    size_bytes=p.stat().st_size,
                )

    cmd = command if command is not None else list(sys.argv)
    # Sanitize command from possible credentials/keys if any
    clean_cmd = [
        arg for arg in cmd if not any(kw in arg.lower() for kw in ("token=", "key=", "pass="))
    ]

    return ExperimentManifest(
        experiment_name=experiment_name,
        git=collect_git_metadata(),
        host=collect_host_system(),
        packages=collect_software_packages(),
        hardware=collect_hardware_telemetry(device=device),
        execution=ExecutionMetadata(
            started_at=start_dt.isoformat(),
            finished_at=finish_dt.isoformat(),
            duration_seconds=round(duration, 3),
            training_time_seconds=(
                round(training_time_seconds, 3) if training_time_seconds is not None else None
            ),
            command=clean_cmd,
        ),
        config=config_dict,
        artifacts=artifact_records,
    )


def save_manifest(manifest: ExperimentManifest, dest_path: Path | str) -> Path:
    """Save an experiment manifest to disk as formatted JSON."""
    out_path = Path(dest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
    return out_path


def load_manifest(path: Path | str) -> ExperimentManifest:
    """Load and validate an experiment manifest from disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {p}")
    content = p.read_text(encoding="utf-8")
    return ExperimentManifest.model_validate_json(content)
