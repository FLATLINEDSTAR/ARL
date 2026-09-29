"""Fail-closed package validation behavior and immutable certificate output."""

from __future__ import annotations

import hashlib
import json

from adaptive_rl.experiments.validator import validate_result_package


def _write_minimal_package(directory):
    directory.mkdir()
    (directory / "adaptive_vs_fixed.json").write_text("{}", encoding="utf-8")
    (directory / "adaptive_vs_fixed.csv").write_text("training_seed,arm\n", encoding="utf-8")
    payload = b"{}"
    (directory / "replicate_state").mkdir()
    (directory / "replicate_state/seed_31001.json").write_bytes(payload)
    (directory / "replicate_state/seed_31001.json.sha256").write_text(
        hashlib.sha256(payload).hexdigest(), encoding="ascii"
    )
    names = (
        "adaptive_vs_fixed.json",
        "adaptive_vs_fixed.csv",
        "replicate_state/seed_31001.json",
        "replicate_state/seed_31001.json.sha256",
    )
    manifest = {
        "schema_version": "1.1",
        "artifacts": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_missing_result_package_fails_closed_and_emits_certificate(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    certificate = tmp_path / "certificate.json"

    report = validate_result_package(package, certificate_path=certificate)

    assert report["verdict"] == "FAIL"
    assert report["checks"][0]["status"] == "FAIL"
    assert all(check["status"] in {"FAIL", "SKIPPED", "PASS"} for check in report["checks"])
    assert json.loads(certificate.read_text(encoding="utf-8"))["verdict"] == "FAIL"
    assert not (package / "manifest.json").exists()


def test_manifested_hash_mismatch_fails_and_certificate_cannot_replace_input(tmp_path):
    package = tmp_path / "package"
    _write_minimal_package(package)
    result_file = package / "adaptive_vs_fixed.json"
    result_file.write_text('{"changed": true}', encoding="utf-8")

    report = validate_result_package(package)

    assert report["verdict"] == "FAIL"
    integrity = next(item for item in report["checks"] if item["id"] == "cryptographic_integrity")
    assert integrity["status"] == "FAIL"


def test_certificate_path_inside_package_is_rejected(tmp_path):
    package = tmp_path / "package"
    _write_minimal_package(package)

    try:
        validate_result_package(package, certificate_path=package / "certificate.json")
    except ValueError as exc:
        assert "outside the result package" in str(exc)
    else:
        raise AssertionError("certificate inside package was accepted")


def test_existing_certificate_is_never_overwritten(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    certificate = tmp_path / "certificate.json"
    certificate.write_text("original", encoding="utf-8")

    try:
        validate_result_package(package, certificate_path=certificate)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing certificate path was overwritten")
    assert certificate.read_text(encoding="utf-8") == "original"
