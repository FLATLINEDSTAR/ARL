"""JSON and analysis-friendly CSV serialization for Issue #265 runs."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np

STUDY_MANIFEST_SCHEMA_VERSION = "1.0"
STUDY_ARTIFACT_SCHEMA_VERSION = "1.0"
REPLICATE_CHECKPOINT_SCHEMA_VERSION = "1.1"

CSV_FIELDS = (
    "replicate_index",
    "training_seed",
    "arm",
    "phase",
    "episode_index",
    "episode_seed",
    "algorithm",
    "environment",
    "reward",
    "length",
    "success",
    "collision",
    "terminated",
    "truncated",
    "policy_fingerprint_start",
    "policy_fingerprint_end",
    "update_block",
    "update_seed",
    "update_status",
    "parameter_delta_l2",
)

STUDY_CSV_FIELDS = (
    "training_seed",
    "arm",
    "replicate_status",
    "recovery_status",
    "T_H",
    "pre_returns",
    "shock_returns",
    "post_returns",
    "pre_seeds",
    "shock_seeds",
    "post_seeds",
    "update_seeds",
    "failure_reason",
    "json_trajectory_reference",
)


def _plain(value: Any) -> Any:
    """Convert supported scientific data values into strict JSON primitives."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, (str, int, float, bool)) for key in value):
            raise TypeError("artifact mappings must have primitive keys")
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    if isinstance(value, str) and value.startswith("/"):
        candidate = Path(value)
        try:
            return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except (OSError, ValueError):
            return candidate.name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported artifact value type: {type(value).__name__}")


def _episode_rows(artifact: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for replicate_index, replicate in enumerate(artifact.get("replicates", []), start=1):
        common = {
            "replicate_index": replicate_index,
            "training_seed": replicate.get("training_seed"),
            "algorithm": artifact.get("experiment", {}).get("algorithm"),
            "environment": artifact.get("experiment", {}).get("environment"),
        }
        segments = (
            ("shared_pre_shift_episodes", "pre", "shared"),
            ("shared_shock_episodes", "post", "shared"),
            ("adaptive_episodes", "post", "adaptive"),
            ("fixed_episodes", "post", "fixed"),
        )
        for key, phase, arm in segments:
            for episode in replicate.get(key, []):
                row: dict[str, Any] = {field: None for field in CSV_FIELDS}
                row.update(common)
                row.update(
                    arm=arm,
                    phase=phase,
                    episode_index=episode.get("episode_index"),
                    episode_seed=episode.get("episode_seed"),
                    reward=episode.get("reward"),
                    length=episode.get("length"),
                    success=episode.get("success"),
                    collision=episode.get("collision"),
                    terminated=episode.get("terminated"),
                    truncated=episode.get("truncated"),
                    policy_fingerprint_start=episode.get("policy_fingerprint_start"),
                    policy_fingerprint_end=episode.get("policy_fingerprint_end"),
                    update_block=episode.get("update_block"),
                    update_seed=episode.get("update_seed"),
                    update_status=episode.get("update_status"),
                    parameter_delta_l2=episode.get("parameter_delta_l2"),
                )
                yield row


def write_adaptation_artifacts(
    artifact: Mapping[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "adaptation",
) -> tuple[Path, Path]:
    """Write strict JSON and flattened CSV without overwriting prior results.

    Both payloads are first written to temporary files, then installed with
    exclusive hard links. If either target already exists, no target is
    overwritten and the call raises ``FileExistsError``.
    """
    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty filename component")
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / f"{stem}.json"
    csv_path = target_dir / f"{stem}.csv"
    plain = _plain(artifact)
    if not isinstance(plain, dict):
        raise TypeError("artifact root must be a mapping")

    temp_paths: list[Path] = []
    try:
        for suffix, writer in (
            (".json", lambda handle: json.dump(plain, handle, indent=2, allow_nan=False)),
            (".csv", lambda handle: _write_csv(handle, plain)),
        ):
            fd, temp_name = tempfile.mkstemp(prefix=f".{stem}-", suffix=suffix, dir=target_dir)
            temp_path = Path(temp_name)
            temp_paths.append(temp_path)
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
        # link() is atomic and fails if the destination exists.
        os.link(temp_paths[0], json_path)
        try:
            os.link(temp_paths[1], csv_path)
        except BaseException:
            json_path.unlink()
            raise
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
    return json_path, csv_path


def write_adaptive_vs_fixed_artifacts(
    artifact: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    """Write study JSON plus the preregistered one-row-per-arm CSV."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "adaptive_vs_fixed.json"
    csv_path = target_dir / "adaptive_vs_fixed.csv"
    plain = _plain(artifact)
    temp_paths: list[Path] = []
    try:
        for suffix, writer in (
            (".json", lambda handle: json.dump(plain, handle, indent=2, allow_nan=False)),
            (".csv", lambda handle: _write_study_csv(handle, plain)),
        ):
            fd, temp_name = tempfile.mkstemp(
                prefix=".adaptive-vs-fixed-", suffix=suffix, dir=target_dir
            )
            temp_path = Path(temp_name)
            temp_paths.append(temp_path)
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
        os.link(temp_paths[0], json_path)
        try:
            os.link(temp_paths[1], csv_path)
        except BaseException:
            json_path.unlink()
            raise
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
    return json_path, csv_path


def _study_rows(artifact: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for replicate in artifact.get("replicates", []):
        for arm in ("adaptive", "fixed"):
            failed = replicate.get("status") != "completed"
            recovery = replicate.get(f"{arm}_recovery") or {}
            post = replicate.get("shared_shock_episodes", []) + replicate.get(f"{arm}_episodes", [])
            seed_map = replicate.get("seeds", {})
            values = {
                "training_seed": replicate.get("training_seed"),
                "arm": arm,
                "replicate_status": replicate.get("status"),
                "recovery_status": None if failed else recovery.get("status"),
                "T_H": None if failed else recovery.get("T_H"),
                "pre_returns": _json_cell(
                    [item.get("reward") for item in replicate.get("shared_pre_shift_episodes", [])]
                ),
                "shock_returns": _json_cell(
                    [item.get("reward") for item in replicate.get("shared_shock_episodes", [])]
                ),
                "post_returns": _json_cell([item.get("reward") for item in post]),
                "pre_seeds": _json_cell(seed_map.get("pre", [])),
                "shock_seeds": _json_cell(seed_map.get("post", [])[:5]),
                "post_seeds": _json_cell(seed_map.get("post", [])),
                "update_seeds": _json_cell(seed_map.get("update", [])),
                "failure_reason": replicate.get("failure_reason"),
                "json_trajectory_reference": (
                    f"replicates[training_seed={replicate.get('training_seed')}].{arm}_episodes"
                ),
            }
            yield values


def _json_cell(value: Any) -> str:
    return json.dumps(_plain(value), separators=(",", ":"), allow_nan=False)


def _write_study_csv(handle: Any, artifact: Mapping[str, Any]) -> None:
    writer = csv.DictWriter(handle, fieldnames=STUDY_CSV_FIELDS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(_study_rows(artifact))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON data canonically for semantic study identity."""
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def make_study_manifest(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Build the immutable, pre-execution study identity document."""
    plain_inputs = _plain(inputs)
    if not isinstance(plain_inputs, dict):
        raise TypeError("study manifest inputs must be a mapping")
    return {
        "schema_version": STUDY_MANIFEST_SCHEMA_VERSION,
        "study_hash": hashlib.sha256(canonical_json_bytes(plain_inputs)).hexdigest(),
        "inputs": plain_inputs,
    }


def write_or_verify_study_manifest(
    inputs: Mapping[str, Any], manifest_path: str | Path, *, resume: bool
) -> str:
    """Persist the expected study identity or fail closed when resuming."""
    manifest_path = Path(manifest_path)
    expected = make_study_manifest(inputs)
    if resume:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("cannot resume without a valid immutable study manifest") from exc
        if (
            not isinstance(existing, dict)
            or existing.get("schema_version") != STUDY_MANIFEST_SCHEMA_VERSION
            or not isinstance(existing.get("inputs"), dict)
        ):
            raise ValueError("study manifest has an invalid structure")
        actual_hash = hashlib.sha256(canonical_json_bytes(existing["inputs"])).hexdigest()
        if existing.get("study_hash") != actual_hash:
            raise ValueError("study manifest hash is invalid")
        if actual_hash != expected["study_hash"]:
            raise ValueError("resume study hash mismatch: execution inputs changed")
        return actual_hash
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite study manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    fd, temp_name = tempfile.mkstemp(prefix=".study-manifest-", dir=manifest_path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, manifest_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return str(expected["study_hash"])


def write_study_manifest(
    artifact_path: str | Path,
    csv_path: str | Path,
    manifest_path: str | Path,
    *,
    run_id: str,
    command: str,
) -> dict[str, Any]:
    """Write an immutable provenance manifest for a completed study attempt."""
    artifact_path = Path(artifact_path)
    csv_path = Path(csv_path)
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        diff_hash = None
        if dirty:
            diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
            diff_hash = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty, diff_hash = None, None, None
    artifact_files: list[Path] = []
    for path in sorted(manifest_path.parent.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"run directory contains a symlinked artifact: {path.name}")
        if path.is_file() and path != manifest_path:
            artifact_files.append(path)
    manifest = {
        "schema_version": STUDY_MANIFEST_SCHEMA_VERSION,
        "study": "adaptive-vs-fixed/prereg-v1",
        "run_id": run_id,
        "commit_sha": commit,
        "working_tree_dirty": dirty,
        "dirty_diff_sha256": diff_hash,
        "execution_command": command,
        "hardware": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
        },
        "package_versions": {
            name: _distribution_version(name)
            for name in ("adaptive-rl", "gymnasium", "stable-baselines3", "torch", "numpy")
        },
        "determinism": {
            "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
            "torch_deterministic_algorithms": _torch_deterministic_algorithms(),
            "torch_cudnn_deterministic": _torch_cudnn_deterministic(),
            "torch_num_threads": _torch_num_threads(),
            "cuda_available": _torch_cuda_available(),
            "cuda_device_count": _torch_cuda_device_count(),
            "protocol_seed_schedule": "SHA-256 derived seeds; see adaptive_vs_fixed.json",
        },
        "artifacts": {
            str(path.relative_to(manifest_path.parent).as_posix()): sha256_file(path)
            for path in artifact_files
        },
    }
    study_spec_path = manifest_path.parent / "study_manifest.json"
    if study_spec_path.is_file():
        study_spec = json.loads(study_spec_path.read_text(encoding="utf-8"))
        manifest["study_hash"] = study_spec.get("study_hash")
    plain = _plain(manifest)
    encoded = json.dumps(plain, indent=2, allow_nan=False) + "\n"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".manifest-", dir=manifest_path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, manifest_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return dict(plain)


def validate_study_manifest(manifest_path: str | Path) -> None:
    """Raise when a listed immutable run artifact is missing or has changed."""
    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink():
        raise ValueError("study artifact manifest cannot be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != STUDY_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("unsupported or malformed study artifact manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("manifest must list at least one artifact checksum")
    root = manifest_path.parent.resolve()
    for relative_path, expected in artifacts.items():
        if (
            not isinstance(relative_path, str)
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("manifest contains an invalid artifact path or SHA-256 digest")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"manifest artifact path escapes the run directory: {relative_path}")
        path = manifest_path.parent / relative
        try:
            path.resolve().relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"manifest artifact path escapes the run directory: {relative_path}"
            ) from exc
        if path.is_symlink():
            raise ValueError(f"manifest artifact is a symlink: {relative_path}")
        if not path.is_file():
            raise ValueError(f"manifest artifact is missing: {relative_path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"manifest checksum mismatch: {relative_path}")
    actual_artifacts = {
        path.relative_to(manifest_path.parent).as_posix()
        for path in manifest_path.parent.rglob("*")
        if path != manifest_path and (path.is_file() or path.is_symlink())
    }
    unlisted_artifacts = actual_artifacts - set(artifacts)
    if unlisted_artifacts:
        names = ", ".join(sorted(unlisted_artifacts))
        raise ValueError(f"run directory contains unlisted artifacts: {names}")
    if "study_hash" in manifest:
        spec_path = manifest_path.parent / "study_manifest.json"
        if not spec_path.is_file() or "study_manifest.json" not in artifacts:
            raise ValueError("study artifact manifest does not bind its pre-execution study spec")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict) or not isinstance(spec.get("inputs"), dict):
            raise ValueError("pre-execution study manifest is malformed")
        actual_study_hash = hashlib.sha256(canonical_json_bytes(spec["inputs"])).hexdigest()
        if (
            spec.get("study_hash") != actual_study_hash
            or manifest.get("study_hash") != actual_study_hash
        ):
            raise ValueError("study artifact manifest hash does not match its pre-execution spec")


def write_replicate_checkpoint(
    replicate: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    study_hash: str,
    protocol_hash: str,
    artifact_root: str | Path | None = None,
    artifact_directories: Sequence[str | Path] = (),
) -> None:
    """Persist a terminal replicate bound to its study and replicate identity."""
    checkpoint_path = Path(checkpoint_path)
    if replicate.get("status") not in {"completed", "failed"}:
        raise ValueError("only terminal replicate states may be checkpointed")
    if artifact_directories and artifact_root is None:
        raise ValueError("artifact_root is required when checkpoint artifacts are declared")
    plain = _plain(dict(replicate))
    artifact_integrity = (
        _snapshot_artifact_directories(artifact_root, artifact_directories)
        if artifact_root is not None
        else []
    )
    envelope = {
        "schema_version": REPLICATE_CHECKPOINT_SCHEMA_VERSION,
        "study_hash": study_hash,
        "protocol_hash": protocol_hash,
        "replicate_id": plain.get("training_seed"),
        "artifact_integrity": artifact_integrity,
        "replicate": plain,
    }
    encoded = (json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".replicate-", dir=checkpoint_path.parent)
    temp_path = Path(temp_name)
    digest_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    installed_checkpoint = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, checkpoint_path)
        installed_checkpoint = True
        digest = hashlib.sha256(encoded).hexdigest()
        digest_temp_fd, digest_temp_name = tempfile.mkstemp(
            prefix=".replicate-digest-", dir=checkpoint_path.parent
        )
        digest_temp_path = Path(digest_temp_name)
        try:
            with os.fdopen(digest_temp_fd, "w", encoding="ascii") as handle:
                handle.write(f"{digest}\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(digest_temp_path, digest_path)
        finally:
            digest_temp_path.unlink(missing_ok=True)
    except BaseException:
        if installed_checkpoint:
            checkpoint_path.unlink(missing_ok=True)
        raise
    finally:
        temp_path.unlink(missing_ok=True)


def read_replicate_checkpoint(
    checkpoint_path: str | Path,
    *,
    study_hash: str,
    protocol_hash: str,
    training_seed: int,
) -> dict[str, Any]:
    """Read a terminal checkpoint after digest, study, protocol, and ID checks."""
    checkpoint_path = Path(checkpoint_path)
    digest_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    payload = checkpoint_path.read_bytes()
    if not digest_path.is_file():
        raise ValueError(f"replicate checkpoint digest is missing: {checkpoint_path.name}")
    expected = digest_path.read_text(encoding="ascii").strip()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"replicate checkpoint checksum mismatch: {checkpoint_path.name}")
    envelope = json.loads(payload)
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema_version") != REPLICATE_CHECKPOINT_SCHEMA_VERSION
        or not isinstance(envelope.get("replicate"), dict)
    ):
        raise ValueError("replicate checkpoint envelope is malformed")
    if envelope.get("study_hash") != study_hash:
        raise ValueError("replicate checkpoint study hash mismatch")
    if envelope.get("protocol_hash") != protocol_hash:
        raise ValueError("replicate checkpoint protocol hash mismatch")
    replicate = envelope["replicate"]
    _validate_checkpoint_artifacts(
        checkpoint_path.parent.parent,
        envelope.get("artifact_integrity"),
    )
    if (
        envelope.get("replicate_id") != training_seed
        or replicate.get("training_seed") != training_seed
    ):
        raise ValueError("replicate checkpoint identity mismatch")
    result = replicate
    if result.get("status") not in {"completed", "failed"}:
        raise ValueError("replicate checkpoint is not terminal")
    return cast(dict[str, Any], result)


def _snapshot_artifact_directories(
    artifact_root: str | Path,
    artifact_directories: Sequence[str | Path],
) -> list[dict[str, Any]]:
    root = Path(artifact_root).resolve()
    snapshots: list[dict[str, Any]] = []
    seen_directories: set[str] = set()
    for directory_value in artifact_directories:
        directory = Path(directory_value)
        if not directory.is_absolute():
            directory = root / directory
        if directory.is_symlink():
            raise ValueError(f"checkpoint artifact directory is a symlink: {directory}")
        resolved = directory.resolve()
        try:
            relative_directory = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"checkpoint artifact directory escapes its run: {directory}") from exc
        if relative_directory in seen_directories:
            raise ValueError(f"duplicate checkpoint artifact directory: {relative_directory}")
        seen_directories.add(relative_directory)
        exists = directory.exists()
        if exists and not directory.is_dir():
            raise ValueError(f"checkpoint artifact path is not a directory: {directory}")
        files: dict[str, str] = {}
        if exists:
            for path in sorted(directory.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"checkpoint artifact contains a symlink: {path}")
                if path.is_file():
                    relative_path = path.resolve().relative_to(root).as_posix()
                    files[relative_path] = sha256_file(path)
        snapshots.append({"path": relative_directory, "exists": exists, "files": files})
    return snapshots


def _validate_checkpoint_artifacts(artifact_root: str | Path, snapshots: Any) -> None:
    if not isinstance(snapshots, list):
        raise ValueError("replicate checkpoint artifact integrity is malformed")
    root = Path(artifact_root).resolve()
    seen_directories: set[str] = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("replicate checkpoint artifact directory record is malformed")
        relative_directory = snapshot.get("path")
        exists = snapshot.get("exists")
        files = snapshot.get("files")
        if (
            not isinstance(relative_directory, str)
            or not relative_directory
            or Path(relative_directory).is_absolute()
            or ".." in Path(relative_directory).parts
            or not isinstance(exists, bool)
            or not isinstance(files, dict)
            or relative_directory in seen_directories
        ):
            raise ValueError("replicate checkpoint artifact directory record is invalid")
        seen_directories.add(relative_directory)
        directory = root / relative_directory
        try:
            directory.resolve().relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError("replicate checkpoint artifact path escapes its run") from exc
        if not exists:
            if directory.exists() or directory.is_symlink():
                raise ValueError("checkpoint artifact directory appeared after checkpointing")
            if files:
                raise ValueError("absent checkpoint artifact directory has recorded files")
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("checkpoint artifact directory is missing or changed type")
        expected_files: set[str] = set()
        for relative_path, expected_hash in files.items():
            relative = Path(relative_path) if isinstance(relative_path, str) else Path("/")
            if (
                not isinstance(relative_path, str)
                or relative.is_absolute()
                or ".." in relative.parts
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash)
            ):
                raise ValueError("checkpoint artifact file record is invalid")
            if (
                relative.parts[: len(Path(relative_directory).parts)]
                != Path(relative_directory).parts
            ):
                raise ValueError("checkpoint artifact file is outside its recorded directory")
            path = root / relative
            try:
                path.resolve().relative_to(root)
            except (OSError, ValueError) as exc:
                raise ValueError("checkpoint artifact file escapes its run") from exc
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"checkpoint artifact file is missing or changed: {relative_path}")
            if sha256_file(path) != expected_hash:
                raise ValueError(f"checkpoint artifact checksum mismatch: {relative_path}")
            expected_files.add(relative_path)
        actual_files: set[str] = set()
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"checkpoint artifact directory contains a symlink: {path}")
            if path.is_file():
                actual_files.add(path.resolve().relative_to(root).as_posix())
        if actual_files != expected_files:
            raise ValueError("checkpoint artifact file set changed after checkpointing")


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_deterministic_algorithms() -> bool | None:
    try:
        import torch

        return bool(torch.are_deterministic_algorithms_enabled())
    except ImportError:
        return None


def _torch_cudnn_deterministic() -> bool | None:
    try:
        import torch

        return bool(torch.backends.cudnn.deterministic)
    except ImportError:
        return None


def _torch_num_threads() -> int | None:
    try:
        import torch

        return int(torch.get_num_threads())
    except ImportError:
        return None


def _torch_cuda_available() -> bool | None:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return None


def _torch_cuda_device_count() -> int | None:
    try:
        import torch

        return int(torch.cuda.device_count())
    except ImportError:
        return None


def _write_csv(handle: Any, artifact: Mapping[str, Any]) -> None:
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(_episode_rows(artifact))


__all__ = [
    "CSV_FIELDS",
    "STUDY_CSV_FIELDS",
    "canonical_json_bytes",
    "make_study_manifest",
    "sha256_file",
    "read_replicate_checkpoint",
    "validate_study_manifest",
    "write_adaptation_artifacts",
    "write_adaptive_vs_fixed_artifacts",
    "write_replicate_checkpoint",
    "write_or_verify_study_manifest",
    "write_study_manifest",
]
