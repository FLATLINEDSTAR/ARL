"""Fail-closed validation of immutable Issue #271 study result packages.

The package format is the ``manifest.json`` + ``adaptive_vs_fixed.json`` and
``adaptive_vs_fixed.csv`` format emitted by ``benchmarking.adaptation_runner``.
Manifest artifact paths are POSIX paths relative to the run directory. The
validator never changes package contents; an optional certificate is written
only to a separate, previously nonexistent destination.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from adaptive_rl.benchmarking.adaptation_artifacts import (
    STUDY_ARTIFACT_SCHEMA_VERSION,
    STUDY_MANIFEST_SCHEMA_VERSION,
)
from adaptive_rl.benchmarking.adaptation_statistics import (
    analyze_primary_cells,
)
from adaptive_rl.protocol.constants import (
    MIN_VALID_N,
    PLANNED_N,
    PRIMARY_CELLS,
    PROTOCOL_VERSION,
    TRAINING_SEEDS,
)
from adaptive_rl.protocol.recovery import compute_recovery
from adaptive_rl.protocol.seeds import frozen_schedule, schedule_fingerprint
from adaptive_rl.protocol.statistics import decide_family

PASS, FAIL, SKIPPED = "PASS", "FAIL", "SKIPPED"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
REQUIRED_FILES = ("manifest.json", "adaptive_vs_fixed.json", "adaptive_vs_fixed.csv")


@dataclass(frozen=True)
class CheckResult:
    id: str
    title: str
    status: str
    message: str
    evidence: dict[str, Any]


class _Skipped(Exception):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(check_id: str, title: str, fn: Callable[[], tuple[str, dict[str, Any]]]) -> CheckResult:
    try:
        message, evidence = fn()
        return CheckResult(check_id, title, PASS, message, evidence)
    except _Skipped as exc:
        return CheckResult(check_id, title, SKIPPED, str(exc), {})
    except Exception as exc:  # unexpected errors are explicitly failures
        return CheckResult(
            check_id,
            title,
            FAIL,
            f"{type(exc).__name__}: {exc}",
            {"exception_type": type(exc).__name__},
        )


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path must be a non-empty string")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative:
        raise ValueError(f"artifact path is not a safe package-relative path: {relative!r}")
    path = root.joinpath(*posix.parts)
    if path.is_symlink():
        raise ValueError(f"artifact is a symlink: {relative}")
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact escapes package directory: {relative}") from exc
    return path


class _Package:
    def __init__(self, directory: Path) -> None:
        self.root = directory.resolve()
        self.manifest_path = self.root / "manifest.json"
        self.manifest: dict[str, Any] | None = None
        self.artifact: dict[str, Any] | None = None
        self.csv_path = self.root / "adaptive_vs_fixed.csv"

    def need(self, field: str) -> Any:
        value = getattr(self, field)
        if value is None:
            raise _Skipped(f"prerequisite data unavailable: {field}")
        return value


def validate_result_package(
    package_dir: str | Path,
    *,
    certificate_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a package and optionally write a certificate outside it.

    The certificate destination must not exist and must not resolve to any
    package path. A failed verdict is still certified as a validation report.
    """
    package = _Package(Path(package_dir))
    checks: list[CheckResult] = []

    def existence() -> tuple[str, dict[str, Any]]:
        missing = [name for name in REQUIRED_FILES if not (package.root / name).is_file()]
        if package.manifest_path.is_symlink():
            missing.append("manifest.json (symlink)")
        if missing:
            raise FileNotFoundError("missing required package files: " + ", ".join(missing))
        try:
            package.manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(package.manifest, dict):
                raise ValueError("manifest root must be an object")
        except Exception:
            package.manifest = None
            raise
        return "required files are present", {"files": list(REQUIRED_FILES)}

    checks.append(_check("artifact_existence", "Required artifact existence", existence))

    def integrity() -> tuple[str, dict[str, Any]]:
        manifest = package.need("manifest")
        entries = manifest.get("artifacts")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("manifest artifacts must be a non-empty path-to-SHA256 object")
        # JSON object keys cannot repeat after decoding; detect duplicate names by parsing pairs.
        pairs = json.loads(
            package.manifest_path.read_text(encoding="utf-8"), object_pairs_hook=list
        )

        # Traverse raw JSON to identify duplicate keys in the artifact map.
        def find_artifacts(node: Any) -> list[tuple[str, Any]] | None:
            if isinstance(node, list):
                if all(isinstance(item, tuple) and len(item) == 2 for item in node):
                    if any(key == "artifacts" for key, _ in node):
                        raw = next(value for key, value in node if key == "artifacts")
                        if isinstance(raw, list) and all(isinstance(item, tuple) for item in raw):
                            return raw
                    for _, child in node:
                        found = find_artifacts(child)
                        if found is not None:
                            return found
                else:
                    for child in node:
                        found = find_artifacts(child)
                        if found is not None:
                            return found
            return None

        raw_entries = find_artifacts(pairs)
        if raw_entries is None:
            raise ValueError("manifest artifacts field is malformed")
        raw_paths = [key for key, _ in raw_entries]
        if len(raw_paths) != len(set(raw_paths)):
            raise ValueError("manifest lists artifact paths more than once")
        if manifest.get("schema_version") != STUDY_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema_version")
        errors: list[str] = []
        for relative, expected in entries.items():
            try:
                path = _safe_path(package.root, relative)
                if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
                    raise ValueError(f"invalid SHA-256 digest for {relative!r}")
                if not path.is_file():
                    raise FileNotFoundError(f"manifest artifact missing: {relative}")
                actual = _sha256(path)
                if actual.lower() != expected.lower():
                    raise ValueError(f"SHA-256 mismatch: {relative}")
            except Exception as exc:
                errors.append(f"{relative}: {type(exc).__name__}: {exc}")
        listed = set(entries)
        if (package.root / "experiment.json").exists() and "experiment.json" not in listed:
            errors.append("experiment.json exists but is not listed in the manifest")
        for required in ("adaptive_vs_fixed.json", "adaptive_vs_fixed.csv"):
            if required not in listed:
                errors.append(f"required result artifact is not listed in manifest: {required}")
        if not any(
            path.startswith("replicate_state/") and path.endswith(".json") for path in listed
        ):
            errors.append("manifest does not reference per-replicate checkpoint JSON artifacts")
        for path in listed:
            if path.startswith("replicate_state/") and path.endswith(".json"):
                sidecar = path + ".sha256"
                if sidecar not in listed:
                    errors.append(
                        f"replicate checkpoint digest sidecar is not manifest-bound: {path}"
                    )
        unlisted: list[str] = []
        for path in package.root.rglob("*"):
            if path.is_symlink():
                unlisted.append(path.relative_to(package.root).as_posix() + " (symlink)")
            elif path.is_file() and path != package.manifest_path:
                relative = path.relative_to(package.root).as_posix()
                if relative not in listed:
                    unlisted.append(relative)
        if unlisted:
            errors.append("unlisted package files: " + ", ".join(sorted(unlisted)))
        if errors:
            raise ValueError("; ".join(errors))
        return "all listed artifact digests match; no unlisted files or symlinks", {
            "artifact_count": len(entries),
            "paths": sorted(entries),
        }

    checks.append(_check("cryptographic_integrity", "Artifact cryptographic integrity", integrity))

    def load_artifact() -> tuple[str, dict[str, Any]]:
        package.need("manifest")
        value = json.loads((package.root / "adaptive_vs_fixed.json").read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("result JSON root must be an object")
        package.artifact = value
        return "result JSON parsed", {"schema_version": value.get("schema_version")}

    checks.append(_check("result_json", "Result JSON structure", load_artifact))

    def schema_protocol() -> tuple[str, dict[str, Any]]:
        manifest, artifact = package.need("manifest"), package.need("artifact")
        if manifest.get("schema_version") != STUDY_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema_version")
        if artifact.get("schema_version") != STUDY_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported result artifact schema_version")
        if artifact.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("protocol_version does not match this validator")
        if artifact.get("schedule_fingerprint") != schedule_fingerprint(frozen_schedule()):
            raise ValueError("seed schedule fingerprint does not match frozen schedule")
        if artifact.get("run_type") != "prereg-v1" or artifact.get("run_status") != "COMPLETE":
            raise ValueError("result is not a complete prereg-v1 study run")
        experiment = artifact.get("experiment", {})
        if (
            experiment.get("algorithm") != "ppo"
            or experiment.get("environment") != "drone_disturbed"
            or experiment.get("scenario") != "TEST-B"
        ):
            raise ValueError("result is not from the frozen drone_disturbed/ppo TEST-B cell")
        if experiment.get("selected_training_seeds") != list(TRAINING_SEEDS):
            raise ValueError("selected training seeds do not match the full frozen study")
        if manifest.get("working_tree_dirty") is not False:
            raise ValueError("manifest does not positively attest a clean working tree")
        if manifest.get("commit_sha") != artifact.get("provenance", {}).get("git_commit"):
            raise ValueError("manifest and result git commit disagree")
        if manifest.get("study_hash") != artifact.get("study_hash"):
            raise ValueError("manifest and result study hash disagree")
        spec_path = package.root / "study_manifest.json"
        if not spec_path.is_file() or spec_path.is_symlink():
            raise ValueError("immutable study_manifest.json is missing or symlinked")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict) or not isinstance(spec.get("inputs"), dict):
            raise ValueError("immutable study manifest is malformed")
        spec_hash = hashlib.sha256(
            json.dumps(
                spec["inputs"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        if spec.get("study_hash") != spec_hash or spec_hash != manifest.get("study_hash"):
            raise ValueError("study manifest hash does not bind the immutable input specification")
        if artifact.get("artifact_paths") != {
            "json": "adaptive_vs_fixed.json",
            "csv": "adaptive_vs_fixed.csv",
            "manifest": "manifest.json",
        }:
            raise ValueError("artifact paths do not match this package layout")
        digest = artifact.get("experiment", {}).get("config_sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError("canonical config hash is missing or malformed")
        if artifact.get("experiment", {}).get("planned_replicates") != PLANNED_N:
            raise ValueError("planned replicate count differs from preregistration")
        if artifact.get("provenance", {}).get("git_dirty") is not False:
            raise ValueError("result provenance does not positively attest a clean git tree")
        return "manifest and result identify a complete run of the frozen protocol", {
            "protocol_version": PROTOCOL_VERSION,
            "schedule_fingerprint": artifact["schedule_fingerprint"],
            "commit_sha": manifest.get("commit_sha"),
        }

    checks.append(_check("protocol_schema", "Manifest and preregistered protocol", schema_protocol))

    def recovery() -> tuple[str, dict[str, Any]]:
        artifact = package.need("artifact")
        replicates = artifact.get("replicates")
        if not isinstance(replicates, list) or len(replicates) != PLANNED_N:
            raise ValueError(f"expected exactly {PLANNED_N} replicate records")
        if [rep.get("training_seed") for rep in replicates] != list(TRAINING_SEEDS):
            raise ValueError("replicate training seeds do not match preregistered order")
        recomputed = 0
        for index, replicate in enumerate(replicates):
            if not isinstance(replicate, dict) or replicate.get("status") != "completed":
                raise ValueError(f"replicate {index} is not completed")
            pre = [episode["reward"] for episode in replicate["shared_pre_shift_episodes"]]
            shock = [episode["reward"] for episode in replicate["shared_shock_episodes"]]
            if len(pre) != 15 or len(shock) != 5:
                raise ValueError(f"replicate {index} has wrong shared pre/shock episode count")
            for arm in ("adaptive", "fixed"):
                episodes = replicate[f"{arm}_episodes"]
                if len(episodes) != 10:
                    raise ValueError(f"replicate {index} {arm} arm must contain episodes 6..15")
                calculated = compute_recovery(
                    pre, shock + [episode["reward"] for episode in episodes]
                )
                recorded = replicate.get(f"{arm}_recovery")
                expected = asdict(calculated)
                expected["T_H"] = calculated.truncated_recovery_time
                if not isinstance(recorded, dict):
                    raise ValueError(f"replicate {index} {arm} recovery record missing")
                for key, value in expected.items():
                    got = recorded.get(key)
                    if isinstance(value, float):
                        if not isinstance(got, (int, float)) or not math.isclose(
                            float(got), value, rel_tol=1e-12, abs_tol=1e-12
                        ):
                            raise ValueError(f"replicate {index} {arm}.{key} does not recompute")
                    elif isinstance(value, tuple):
                        if list(value) != got:
                            raise ValueError(f"replicate {index} {arm}.{key} does not recompute")
                    elif got != value:
                        raise ValueError(f"replicate {index} {arm}.{key} does not recompute")
                recomputed += 1
        return "all recovery records recompute from stored episode returns", {
            "arm_records_recomputed": recomputed
        }

    checks.append(_check("recovery_recompute", "Recovery endpoint recomputation", recovery))

    def execution_invariants() -> tuple[str, dict[str, Any]]:
        artifact = package.need("artifact")
        replicates = artifact.get("replicates")
        if not isinstance(replicates, list):
            raise ValueError("replicates must be a list")
        failed: list[str] = []
        for index, replicate in enumerate(replicates):
            invariants = replicate.get("invariants") if isinstance(replicate, dict) else None
            if not isinstance(invariants, dict) or invariants.get("all_passed") is not True:
                failed.append(f"replicate[{index}] has no passing invariant audit")
                continue
            if any(value is not True for key, value in invariants.items() if key != "all_passed"):
                failed.append(f"replicate[{index}] has a failed invariant")
            if (
                replicate.get("fixed_parameter_delta_l2") != 0.0
                or replicate.get("fixed_weight_update_count") != 0
            ):
                failed.append(f"replicate[{index}] Fixed policy changed")
            frozen = replicate.get("frozen_fingerprint")
            if not isinstance(frozen, str) or not SHA256_RE.fullmatch(frozen):
                failed.append(f"replicate[{index}] frozen fingerprint missing or malformed")
            if replicate.get("fixed_final_fingerprint") != frozen:
                failed.append(f"replicate[{index}] final Fixed fingerprint differs")
            if replicate.get("fork_fingerprint") != frozen:
                failed.append(f"replicate[{index}] fork fingerprint differs from frozen checkpoint")
        if failed:
            raise ValueError("; ".join(failed))
        return "all ten replicate execution invariants and Fixed-arm immutability checks pass", {
            "replicates": len(replicates)
        }

    checks.append(
        _check(
            "execution_invariants", "Replicate execution and frozen weights", execution_invariants
        )
    )

    def csv_consistency() -> tuple[str, dict[str, Any]]:
        artifact = package.need("artifact")
        with package.csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        expected = {
            (str(r.get("training_seed")), arm): r
            for r in artifact.get("replicates", [])
            for arm in ("adaptive", "fixed")
        }
        expected_fields = {
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
        }
        if set(rows[0]) != expected_fields if rows else True:
            raise ValueError("CSV columns do not match the immutable study CSV schema")
        observed: set[tuple[str, str]] = set()
        for row in rows:
            key = (row.get("training_seed", ""), row.get("arm", ""))
            if key not in expected or key in observed:
                raise ValueError(f"CSV has unknown or duplicate arm row: {key}")
            observed.add(key)
            replicate = expected[key]
            arm = key[1]
            recovery_record = replicate.get(f"{arm}_recovery") or {}
            seeds = replicate.get("seeds", {})
            pre_returns = [
                episode.get("reward") for episode in replicate.get("shared_pre_shift_episodes", [])
            ]
            shock_returns = [
                episode.get("reward") for episode in replicate.get("shared_shock_episodes", [])
            ]
            post_records = replicate.get("shared_shock_episodes", []) + replicate.get(
                f"{arm}_episodes", []
            )
            post_returns = [episode.get("reward") for episode in post_records]
            expected_cells = {
                "replicate_status": replicate.get("status", ""),
                "pre_returns": json.dumps(pre_returns, separators=(",", ":"), allow_nan=False),
                "shock_returns": json.dumps(shock_returns, separators=(",", ":"), allow_nan=False),
                "post_returns": json.dumps(post_returns, separators=(",", ":"), allow_nan=False),
                "pre_seeds": json.dumps(
                    seeds.get("pre", []), separators=(",", ":"), allow_nan=False
                ),
                "shock_seeds": json.dumps(
                    seeds.get("post", [])[:5], separators=(",", ":"), allow_nan=False
                ),
                "post_seeds": json.dumps(
                    seeds.get("post", []), separators=(",", ":"), allow_nan=False
                ),
                "update_seeds": json.dumps(
                    seeds.get("update", []), separators=(",", ":"), allow_nan=False
                ),
                "failure_reason": replicate.get("failure_reason") or "",
                "json_trajectory_reference": f"replicates[training_seed={replicate.get('training_seed')}].{arm}_episodes",
            }
            for field, value in expected_cells.items():
                if row.get(field, "") != str(value):
                    raise ValueError(f"CSV {field} disagrees with JSON for {key}")
            csv_t = row.get("T_H", "")
            expected_t = (
                "" if replicate.get("status") != "completed" else str(recovery_record.get("T_H"))
            )
            if csv_t != expected_t:
                raise ValueError(f"CSV T_H disagrees with JSON for {key}")
            if row.get("recovery_status", "") != (recovery_record.get("status") or ""):
                raise ValueError(f"CSV recovery status disagrees with JSON for {key}")
        if observed != set(expected):
            raise ValueError("CSV does not contain exactly one row per replicate arm")
        return "CSV arm summaries agree with JSON", {"rows": len(rows)}

    checks.append(_check("csv_consistency", "JSON and CSV consistency", csv_consistency))

    def registered_analysis() -> tuple[str, dict[str, Any]]:
        artifact = package.need("artifact")
        reps = artifact.get("replicates")
        fixed = [
            r.get("fixed_recovery", {}).get("T_H") if r.get("status") == "completed" else None
            for r in reps
        ]
        adaptive = [
            r.get("adaptive_recovery", {}).get("T_H") if r.get("status") == "completed" else None
            for r in reps
        ]
        cell = f"drone_disturbed/{artifact.get('experiment', {}).get('algorithm')}"
        recorded = artifact.get("paired_analysis", {}).get(cell)
        if not isinstance(recorded, dict):
            raise ValueError(f"paired analysis is missing preregistered cell {cell}")
        analysis = analyze_primary_cells(
            {
                name: (fixed, adaptive)
                if name == cell
                else ([None] * PLANNED_N, [None] * PLANNED_N)
                for name in PRIMARY_CELLS
            }
        )[cell]
        if recorded != analysis.to_dict():
            raise ValueError("recorded paired analysis does not recompute")
        all_analysis = artifact.get("paired_analysis")
        if not isinstance(all_analysis, dict) or set(all_analysis) != set(PRIMARY_CELLS):
            raise ValueError(
                "paired_analysis does not contain the exact preregistered six-cell family"
            )
        p_values = {
            cell_name: all_analysis[cell_name].get("primary_p_value") for cell_name in PRIMARY_CELLS
        }
        decision = decide_family(p_values)
        if decision != artifact.get("family_decision"):
            raise ValueError("family decision does not recompute")
        if analysis.valid_n < MIN_VALID_N:
            raise ValueError(
                f"only {analysis.valid_n} valid pairs; preregistered minimum is {MIN_VALID_N}"
            )
        return "paired differences and preregistered analysis recompute", {
            "cell": cell,
            "valid_n": analysis.valid_n,
        }

    checks.append(
        _check("registered_statistics", "Preregistered paired statistics", registered_analysis)
    )

    verdict = "PASS" if all(result.status == PASS for result in checks) else "FAIL"
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "validator": "adaptive_rl.experiments.validator",
        "package": str(package.root),
        "verdict": verdict,
        "checks": [asdict(result) for result in checks],
    }
    if certificate_path is not None:
        destination = Path(certificate_path).expanduser().absolute()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite certificate path: {destination}")
        try:
            destination.resolve(strict=False).relative_to(package.root)
            raise ValueError("certificate path must be outside the result package")
        except ValueError as exc:
            if str(exc) == "certificate path must be outside the result package":
                raise
        for path in package.root.rglob("*"):
            if path.resolve(strict=False) == destination.resolve(strict=False):
                raise ValueError("certificate path aliases an input artifact")
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        fd, temporary = tempfile.mkstemp(prefix=".validation-certificate-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return report


__all__ = ["CheckResult", "validate_result_package"]
