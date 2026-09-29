"""Structured artifact serialization and no-overwrite guarantees."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from adaptive_rl.benchmarking.adaptation_artifacts import (
    canonical_json_bytes,
    make_study_manifest,
    read_replicate_checkpoint,
    sha256_file,
    validate_study_manifest,
    write_adaptation_artifacts,
    write_adaptive_vs_fixed_artifacts,
    write_or_verify_study_manifest,
    write_replicate_checkpoint,
    write_study_manifest,
)
from adaptive_rl.config import compute_config_sha256, load_config
from adaptive_rl.protocol.constants import (
    ISSUE271_CONFIG_SHA256,
    ISSUE271_TREATMENT_CARD_SHA256,
)


def _artifact():
    return {
        "schema_version": "1.0",
        "protocol_version": "2.0",
        "issue": "265",
        "experiment": {"algorithm": "ppo", "environment": "drone"},
        "replicates": [
            {
                "training_seed": 31001,
                "shared_pre_shift_episodes": [
                    {
                        "episode_index": 1,
                        "episode_seed": 123,
                        "reward": np.float64(2.0),
                        "length": 15,
                        "success": True,
                        "collision": False,
                        "terminated": True,
                        "truncated": False,
                        "transitions": [{"observation": np.asarray([0.1, 0.2])}],
                    }
                ],
                "shared_shock_episodes": [],
                "adaptive_episodes": [],
                "fixed_episodes": [],
            }
        ],
    }


def test_json_and_flattened_csv_are_written_without_opaque_objects(tmp_path) -> None:
    json_path, csv_path = write_adaptation_artifacts(_artifact(), tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["replicates"][0]["shared_pre_shift_episodes"][0]["transitions"][0][
        "observation"
    ] == [0.1, 0.2]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["arm"] == "shared"
    assert rows[0]["phase"] == "pre"
    assert rows[0]["episode_seed"] == "123"


def test_existing_artifacts_are_never_overwritten(tmp_path) -> None:
    json_path, csv_path = write_adaptation_artifacts(_artifact(), tmp_path)
    before_json = json_path.read_bytes()
    before_csv = csv_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_adaptation_artifacts(_artifact(), tmp_path)
    assert json_path.read_bytes() == before_json
    assert csv_path.read_bytes() == before_csv


def test_nonfinite_values_are_rejected_for_strict_json(tmp_path) -> None:
    data = _artifact()
    data["value"] = float("nan")
    with pytest.raises(ValueError, match="JSON compliant"):
        write_adaptation_artifacts(data, tmp_path)


def test_manifest_hashes_all_artifacts_and_detects_tampering(tmp_path) -> None:
    json_path, csv_path = write_adaptation_artifacts(
        _artifact(), tmp_path, stem="adaptive_vs_fixed"
    )
    training_artifact = tmp_path / "training" / "seed_31001" / "weights.zip"
    training_artifact.parent.mkdir(parents=True)
    training_artifact.write_bytes(b"real-artifact-bytes")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_study_manifest(
        json_path,
        csv_path,
        manifest_path,
        run_id="test-run",
        command="adaptive-rl benchmark adaptation --study prereg-v1 --run-id test-run",
    )
    assert manifest["artifacts"][json_path.name] == sha256_file(json_path)
    assert manifest["artifacts"][csv_path.name] == sha256_file(csv_path)
    assert manifest["artifacts"]["training/seed_31001/weights.zip"] == sha256_file(
        training_artifact
    )
    validate_study_manifest(manifest_path)
    with pytest.raises(FileExistsError):
        write_study_manifest(
            json_path,
            csv_path,
            manifest_path,
            run_id="test-run",
            command="adaptive-rl benchmark adaptation --study prereg-v1 --run-id test-run",
        )
    training_artifact.write_bytes(b"tampered")
    assert manifest["artifacts"]["training/seed_31001/weights.zip"] != sha256_file(
        training_artifact
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_study_manifest(manifest_path)


def test_manifest_rejects_unlisted_files_added_after_completion(tmp_path) -> None:
    json_path, csv_path = write_adaptation_artifacts(
        _artifact(), tmp_path, stem="adaptive_vs_fixed"
    )
    manifest_path = tmp_path / "manifest.json"
    write_study_manifest(
        json_path,
        csv_path,
        manifest_path,
        run_id="test-run",
        command="adaptive-rl benchmark adaptation --study prereg-v1 --run-id test-run",
    )
    validate_study_manifest(manifest_path)

    (tmp_path / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted artifacts: unexpected.json"):
        validate_study_manifest(manifest_path)


def test_completed_manifest_binds_preexecution_study_hash(tmp_path) -> None:
    spec_path = tmp_path / "study_manifest.json"
    study_hash = write_or_verify_study_manifest(
        {"protocol": "prereg-v1", "seed": 31001}, spec_path, resume=False
    )
    json_path, csv_path = write_adaptation_artifacts(
        _artifact(), tmp_path, stem="adaptive_vs_fixed"
    )
    manifest_path = tmp_path / "manifest.json"
    write_study_manifest(
        json_path,
        csv_path,
        manifest_path,
        run_id="bound-study",
        command="adaptive-rl benchmark adaptation --study prereg-v1",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["study_hash"] == study_hash
    validate_study_manifest(manifest_path)

    changed_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    changed_spec["inputs"]["seed"] = 31002
    spec_path.write_text(json.dumps(changed_spec), encoding="utf-8")
    manifest["artifacts"]["study_manifest.json"] = sha256_file(spec_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its pre-execution spec"):
        validate_study_manifest(manifest_path)


def test_manifest_rejects_path_escape_and_unsupported_schema(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "1.0", "artifacts": {"../outside": "0" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes the run directory"):
        validate_study_manifest(manifest_path)
    manifest_path.write_text(
        json.dumps({"schema_version": "2.0", "artifacts": {"x": "0" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported or malformed"):
        validate_study_manifest(manifest_path)


def test_manifest_rejects_symlinked_artifact_outside_run_directory(tmp_path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.write_bytes(b"outside")
    link = tmp_path / "external.bin"
    link.symlink_to(outside)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifacts": {"external.bin": sha256_file(outside)},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes the run directory"):
        validate_study_manifest(manifest_path)


def test_manifest_rejects_internal_symlinked_artifact(tmp_path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"bound-bytes")
    link = tmp_path / "alias.bin"
    link.symlink_to(target)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifacts": {"alias.bin": sha256_file(target)},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact is a symlink"):
        validate_study_manifest(manifest_path)


def test_study_csv_has_one_row_per_arm_and_preserves_finite_censoring(tmp_path) -> None:
    data = {
        "replicates": [
            {
                "training_seed": 31001,
                "status": "completed",
                "shared_pre_shift_episodes": [{"reward": 10.0}],
                "shared_shock_episodes": [{"reward": 2.0}],
                "adaptive_episodes": [{"reward": 3.0}],
                "fixed_episodes": [{"reward": 2.5}],
                "adaptive_recovery": {"status": "right_censored", "T_H": 15},
                "fixed_recovery": {"status": "right_censored", "T_H": 15},
                "seeds": {"pre": [11], "post": [12, 13], "update": [14]},
            }
        ]
    }
    _, csv_path = write_adaptive_vs_fixed_artifacts(data, tmp_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["arm"] for row in rows} == {"adaptive", "fixed"}
    assert [row["T_H"] for row in rows] == ["15", "15"]
    assert [row["recovery_status"] for row in rows] == ["right_censored"] * 2
    assert json.loads(rows[0]["post_returns"]) == [2.0, 3.0]


def test_replicate_checkpoint_is_terminal_hashed_and_tamper_evident(tmp_path) -> None:
    checkpoint = tmp_path / "replicate_state" / "seed_31001.json"
    study_hash = "study-hash"
    protocol_hash = "protocol-hash"
    replicate = {"training_seed": 31001, "status": "failed", "failure_reason": "crash"}
    write_replicate_checkpoint(
        replicate,
        checkpoint,
        study_hash=study_hash,
        protocol_hash=protocol_hash,
    )
    assert (
        read_replicate_checkpoint(
            checkpoint,
            study_hash=study_hash,
            protocol_hash=protocol_hash,
            training_seed=31001,
        )["failure_reason"]
        == "crash"
    )
    original = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        write_replicate_checkpoint(
            {"training_seed": 31001, "status": "failed", "failure_reason": "other"},
            checkpoint,
            study_hash=study_hash,
            protocol_hash=protocol_hash,
        )
    assert checkpoint.read_bytes() == original
    with pytest.raises(ValueError, match="study hash mismatch"):
        read_replicate_checkpoint(
            checkpoint,
            study_hash="different-study",
            protocol_hash=protocol_hash,
            training_seed=31001,
        )
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        read_replicate_checkpoint(
            checkpoint,
            study_hash=study_hash,
            protocol_hash="different-protocol",
            training_seed=31001,
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        read_replicate_checkpoint(
            checkpoint,
            study_hash=study_hash,
            protocol_hash=protocol_hash,
            training_seed=31002,
        )
    checkpoint.write_text('{"training_seed":31001,"status":"completed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_replicate_checkpoint(
            checkpoint,
            study_hash=study_hash,
            protocol_hash=protocol_hash,
            training_seed=31001,
        )


def test_replicate_checkpoint_binds_training_artifact_files(tmp_path) -> None:
    run_root = tmp_path / "run"
    training_dir = run_root / "training" / "seed_31001"
    training_dir.mkdir(parents=True)
    model_path = training_dir / "model.zip"
    model_path.write_bytes(b"frozen-model")
    checkpoint_path = run_root / "replicate_state" / "seed_31001.json"
    write_replicate_checkpoint(
        {"training_seed": 31001, "status": "completed"},
        checkpoint_path,
        study_hash="study-hash",
        protocol_hash="protocol-hash",
        artifact_root=run_root,
        artifact_directories=(training_dir,),
    )

    read_replicate_checkpoint(
        checkpoint_path,
        study_hash="study-hash",
        protocol_hash="protocol-hash",
        training_seed=31001,
    )
    model_path.write_bytes(b"changed-model")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        read_replicate_checkpoint(
            checkpoint_path,
            study_hash="study-hash",
            protocol_hash="protocol-hash",
            training_seed=31001,
        )
    model_path.write_bytes(b"frozen-model")
    (training_dir / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact file set changed"):
        read_replicate_checkpoint(
            checkpoint_path,
            study_hash="study-hash",
            protocol_hash="protocol-hash",
            training_seed=31001,
        )


def test_canonical_study_hash_ignores_mapping_order_and_json_formatting(tmp_path) -> None:
    left = {"config": {"seed": 31001, "lr": 0.001}, "schedule": [1, 2, 3]}
    right = {"schedule": [1, 2, 3], "config": {"lr": 0.001, "seed": 31001}}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert make_study_manifest(left)["study_hash"] == make_study_manifest(right)["study_hash"]

    path = tmp_path / "study_manifest.json"
    digest = write_or_verify_study_manifest(left, path, resume=False)
    path.write_text(json.dumps(json.loads(path.read_text()), indent=4), encoding="utf-8")
    assert write_or_verify_study_manifest(right, path, resume=True) == digest


def test_study_hash_binds_artifact_schema_versions() -> None:
    base = {
        "protocol": "prereg-v1",
        "artifact_schema_version": "1.0",
        "manifest_schema_version": "1.0",
        "replicate_checkpoint_schema_version": "1.1",
    }
    changed = {**base, "artifact_schema_version": "2.0"}
    assert make_study_manifest(base)["study_hash"] != make_study_manifest(changed)["study_hash"]


def test_study_manifest_material_change_and_tampered_hash_fail_resume(tmp_path) -> None:
    path = tmp_path / "study_manifest.json"
    inputs = {"config": {"gamma": 0.99}, "seeds": [1, 2]}
    write_or_verify_study_manifest(inputs, path, resume=False)
    with pytest.raises(ValueError, match="study hash mismatch"):
        write_or_verify_study_manifest(
            {"config": {"gamma": 0.98}, "seeds": [1, 2]}, path, resume=True
        )

    document = json.loads(path.read_text(encoding="utf-8"))
    document["study_hash"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="hash is invalid"):
        write_or_verify_study_manifest(inputs, path, resume=True)


def test_resume_rejects_missing_or_malformed_study_manifest(tmp_path) -> None:
    path = tmp_path / "study_manifest.json"
    with pytest.raises(ValueError, match="valid immutable study manifest"):
        write_or_verify_study_manifest({"a": 1}, path, resume=True)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="valid immutable study manifest"):
        write_or_verify_study_manifest({"a": 1}, path, resume=True)


def test_issue271_preregistered_inputs_match_frozen_hashes() -> None:
    config = load_config("configs/drone_distribution_shift.yaml")
    assert compute_config_sha256(config) == ISSUE271_CONFIG_SHA256
    assert sha256_file("docs/research/TREATMENT_CARD.md") == ISSUE271_TREATMENT_CARD_SHA256
