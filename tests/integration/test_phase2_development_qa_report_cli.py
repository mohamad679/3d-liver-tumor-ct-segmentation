"""Integration tests for the Phase 2 final development QA JSON report CLI."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from protoem_ct.artifacts import (
    DevelopmentQaArtifact,
    hash_development_qa,
    hash_lesion_components,
    phase2_artifact_from_json,
)
from protoem_ct.cli.main import app


def _load_unit_helpers() -> Any:
    helper_path = Path(__file__).parents[1] / "unit" / "test_development_qa_report.py"
    spec = importlib.util.spec_from_file_location(
        "_development_qa_report_unit_helpers",
        helper_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("development QA report helper module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_unit_helpers()
CREATED_AT_UTC: str = _HELPERS.CREATED_AT_UTC
GIT_COMMIT: str = _HELPERS.GIT_COMMIT
HASH0: str = _HELPERS.HASH0
_chain = _HELPERS._chain
_write_chain = _HELPERS._write_chain


def _invoke(*args: str) -> Any:
    runner = CliRunner()
    return runner.invoke(app, [*args])


def _report_args(paths: tuple[Path, Path, Path, Path, Path], output: Path) -> list[str]:
    return [
        "build-development-qa-report",
        "--manifest",
        str(paths[0]),
        "--split",
        str(paths[1]),
        "--geometry-qa-artifact",
        str(paths[2]),
        "--lesion-artifact",
        str(paths[3]),
        "--development-summary-artifact",
        str(paths[4]),
        "--output",
        str(output),
        "--report-contract-version",
        "development_qa_report_v1",
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        CREATED_AT_UTC,
    ]


def _error_text(result: Any) -> str:
    return getattr(result, "stderr", "") or result.output


def test_complete_synthetic_json_chain_builds_final_report(tmp_path: Path) -> None:
    paths = _write_chain(tmp_path)
    output = tmp_path / "out" / "report.json"

    result = _invoke(*_report_args(paths, output))

    assert result.exit_code == 0, result.output
    assert "development QA JSON report completed" in result.output
    assert "case count: 3" in result.output
    assert "passed case count: 2" in result.output
    assert "failed case count: 1" in result.output
    assert "report config hash: " in result.output
    assert "manifest hash: " in result.output
    assert "split hash: " in result.output
    assert "geometry QA artifact hash: " in result.output
    assert "lesion artifact hash: " in result.output
    assert "development summary artifact hash: " in result.output
    assert "final QA artifact hash: " in result.output
    assert f"output path: {output}" in result.output
    assert "anon-p" not in result.output
    assert "anon-c" not in result.output
    assert "images/" not in result.output
    assert "labels/" not in result.output
    assert "label_value_not_allowed" not in result.output

    artifact = phase2_artifact_from_json(
        output.read_text(encoding="utf-8"),
        DevelopmentQaArtifact,
    )
    assert artifact.qa_artifact_hash == hash_development_qa(artifact)
    assert artifact.passed_case_count == 2
    assert artifact.failed_case_count == 1


def test_independent_cli_outputs_are_byte_identical(tmp_path: Path) -> None:
    paths = _write_chain(tmp_path)
    first_output = tmp_path / "first" / "report.json"
    second_output = tmp_path / "second" / "report.json"

    first = _invoke(*_report_args(paths, first_output))
    second = _invoke(*_report_args(paths, second_output))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert first_output.read_bytes() == second_output.read_bytes()


def test_cli_mixed_passing_failing_cases_return_zero(tmp_path: Path) -> None:
    paths = _write_chain(tmp_path)
    output = tmp_path / "out" / "report.json"

    result = _invoke(*_report_args(paths, output))

    assert result.exit_code == 0, result.output
    artifact = phase2_artifact_from_json(
        output.read_text(encoding="utf-8"),
        DevelopmentQaArtifact,
    )
    assert tuple(record.qa_passed for record in artifact.case_records) == (True, False, True)


def test_cli_structural_failures_do_not_traceback(tmp_path: Path) -> None:
    manifest, split, geometry, lesion, summary = _chain()
    tampered_lesion = replace(lesion, lesion_artifact_hash="f" * 64)
    paths = _write_chain(
        tmp_path / "tampered",
        (manifest, split, geometry, tampered_lesion, summary),
    )
    result = _invoke(*_report_args(paths, tmp_path / "out" / "report.json"))
    assert result.exit_code != 0
    assert "Traceback" not in _error_text(result)
    assert "anon-" not in _error_text(result)

    bad_lesion = replace(lesion, geometry_qa_artifact_hash="f" * 64, lesion_artifact_hash=HASH0)
    bad_lesion = replace(bad_lesion, lesion_artifact_hash=hash_lesion_components(bad_lesion))
    paths = _write_chain(tmp_path / "link", (manifest, split, geometry, bad_lesion, summary))
    linked = _invoke(*_report_args(paths, tmp_path / "link-out" / "report.json"))
    assert linked.exit_code != 0
    assert "Traceback" not in _error_text(linked)

    paths = _write_chain(tmp_path / "existing")
    existing_output = tmp_path / "existing.json"
    existing_output.write_text("occupied\n", encoding="utf-8")
    existing = _invoke(*_report_args(paths, existing_output))
    assert existing.exit_code != 0
    assert "Traceback" not in _error_text(existing)


def test_no_nifti_or_image_file_is_created_or_opened(tmp_path: Path) -> None:
    paths = _write_chain(tmp_path)
    output = tmp_path / "out" / "report.json"

    result = _invoke(*_report_args(paths, output))

    assert result.exit_code == 0, result.output
    assert not list(tmp_path.rglob("*.nii"))
    assert not list(tmp_path.rglob("*.nii.gz"))
