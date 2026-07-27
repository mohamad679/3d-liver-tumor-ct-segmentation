"""Integration tests for the Phase 2 development leakage-audit CLI."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from protoem_ct.artifacts import (
    LeakageAuditArtifact,
    hash_leakage_audit,
    phase2_artifact_from_json,
)
from protoem_ct.cli.main import app


def _load_unit_helpers() -> Any:
    helper_path = Path(__file__).parents[1] / "unit" / "test_leakage_audit.py"
    spec = importlib.util.spec_from_file_location(
        "_leakage_audit_unit_helpers",
        helper_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("leakage-audit helper module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_unit_helpers()
CREATED_AT_UTC: str = _HELPERS.CREATED_AT_UTC
GIT_COMMIT: str = _HELPERS.GIT_COMMIT
HASH7: str = _HELPERS.HASH7
_case = _HELPERS._case
_manifest = _HELPERS._manifest
_split = _HELPERS._split
_write_inputs = _HELPERS._write_inputs


def _invoke(*args: str) -> Any:
    runner = CliRunner()
    return runner.invoke(app, [*args])


def _audit_args(paths: tuple[Path, Path], output: Path) -> list[str]:
    return [
        "audit-development-leakage",
        "--manifest",
        str(paths[0]),
        "--split",
        str(paths[1]),
        "--output",
        str(output),
        "--audit-contract-version",
        "development_leakage_audit_v1",
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        CREATED_AT_UTC,
    ]


def _error_text(result: Any) -> str:
    return getattr(result, "stderr", "") or result.output


def test_clean_synthetic_manifest_split_produces_passing_audit(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output = tmp_path / "out" / "audit.json"

    result = _invoke(*_audit_args(paths, output))

    assert result.exit_code == 0, result.output
    assert "development leakage audit completed" in result.output
    assert "audit passed: true" in result.output
    assert "manifest case count: 3" in result.output
    assert "manifest patient count: 3" in result.output
    assert "total finding count: 0" in result.output
    assert "anon-" not in result.output
    assert "images/" not in result.output
    assert "labels/" not in result.output
    artifact = phase2_artifact_from_json(
        output.read_text(encoding="utf-8"),
        LeakageAuditArtifact,
    )
    assert artifact.audit_passed is True
    assert artifact.audit_hash == hash_leakage_audit(artifact)


def test_valid_overlap_fixture_publishes_failing_audit_with_zero_cli_exit(tmp_path: Path) -> None:
    manifest = _manifest((_case(1, label_hash=HASH7), _case(2, label_hash=HASH7), _case(3)))
    split = _split(manifest)
    paths = _write_inputs(tmp_path, manifest=manifest, split=split)
    output = tmp_path / "out" / "audit.json"

    result = _invoke(*_audit_args(paths, output))

    assert result.exit_code == 0, result.output
    assert "audit passed: false" in result.output
    assert "label-hash cross-partition overlap count: 1" in result.output
    artifact = phase2_artifact_from_json(
        output.read_text(encoding="utf-8"),
        LeakageAuditArtifact,
    )
    assert artifact.audit_passed is False
    assert artifact.finding_codes == ("label_hash_cross_partition_overlap",)


def test_independent_cli_outputs_are_byte_identical(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    first = tmp_path / "first" / "audit.json"
    second = tmp_path / "second" / "audit.json"

    first_result = _invoke(*_audit_args(paths, first))
    second_result = _invoke(*_audit_args(paths, second))

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    assert first.read_bytes() == second.read_bytes()


def test_structural_cli_failures_do_not_traceback_or_leak_contents(tmp_path: Path) -> None:
    manifest = _manifest()
    split = _split(manifest)
    tampered_manifest = replace(manifest, manifest_hash="7" * 64)
    paths = _write_inputs(tmp_path / "manifest", manifest=tampered_manifest, split=split)
    result = _invoke(*_audit_args(paths, tmp_path / "out" / "audit.json"))
    assert result.exit_code != 0
    assert "Traceback" not in _error_text(result)
    assert "anon-" not in _error_text(result)

    tampered_split = replace(split, split_hash="7" * 64)
    paths = _write_inputs(tmp_path / "split", manifest=manifest, split=tampered_split)
    split_result = _invoke(*_audit_args(paths, tmp_path / "split-out" / "audit.json"))
    assert split_result.exit_code != 0
    assert "Traceback" not in _error_text(split_result)

    paths = _write_inputs(tmp_path / "existing")
    existing = tmp_path / "existing.json"
    existing.write_text("occupied\n", encoding="utf-8")
    existing_result = _invoke(*_audit_args(paths, existing))
    assert existing_result.exit_code != 0
    assert "Traceback" not in _error_text(existing_result)


def test_no_source_medical_file_is_opened_or_created(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output = tmp_path / "out" / "audit.json"

    result = _invoke(*_audit_args(paths, output))

    assert result.exit_code == 0, result.output
    assert not list(tmp_path.rglob("*.nii"))
    assert not list(tmp_path.rglob("*.nii.gz"))
    assert not list(tmp_path.rglob("*.dcm"))
