"""Integration tests for the Phase 2 development split CLI."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentSplitManifest,
    hash_dataset_manifest,
    hash_development_split,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
)
from protoem_ct.cli.main import app
from protoem_ct.data import PATIENT_HASH_RANK_POLICY_VERSION

GENERATED_AT_UTC = "2026-07-26T00:00:00Z"
GIT_COMMIT = "30ef8ab"


def _invoke(*args: str) -> Any:
    runner = CliRunner()
    return runner.invoke(app, [*args])


def _case(index: int) -> DatasetCaseRecord:
    return DatasetCaseRecord(
        anonymous_patient_id=f"anon-p{index:04d}",
        anonymous_case_id=f"anon-c{index:04d}",
        relative_image_path=f"images/case-{index:04d}.nii.gz",
        relative_label_path=f"labels/case-{index:04d}.nii.gz",
        image_sha256=f"{index:064x}",
        label_sha256=f"{index + 1000:064x}",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
    )


def _manifest(case_count: int = 6) -> DatasetManifest:
    cases = tuple(_case(index) for index in range(1, case_count + 1))
    manifest_without_hash = DatasetManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DATASET_MANIFEST_TYPE,
        dataset_id="lits-development",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
        adapter_name="lits_style",
        adapter_version="1",
        generated_at_utc="2026-07-25T00:00:00Z",
        git_commit=GIT_COMMIT,
        dataset_root_fingerprint="0" * 64,
        manifest_hash="0" * 64,
        case_count=len(cases),
        cases=cases,
    )
    return replace(
        manifest_without_hash,
        manifest_hash=hash_dataset_manifest(manifest_without_hash),
    )


def _write_manifest(path: Path, manifest: DatasetManifest | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(phase2_artifact_to_json(manifest or _manifest()), encoding="utf-8")


def _split_args(manifest_path: Path, output: Path, *, train_count: int = 3) -> list[str]:
    return [
        "build-development-split",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
        "--policy-version",
        PATIENT_HASH_RANK_POLICY_VERSION,
        "--split-seed",
        "1729",
        "--train-patient-count",
        str(train_count),
        "--validation-patient-count",
        "2",
        "--internal-test-patient-count",
        "1",
        "--git-commit",
        GIT_COMMIT,
        "--generated-at-utc",
        GENERATED_AT_UTC,
    ]


def _error_text(result: Any) -> str:
    return getattr(result, "stderr", "") or result.output


def _partition_sets(
    split_manifest: DevelopmentSplitManifest,
) -> dict[str, tuple[set[str], set[str]]]:
    result: dict[str, tuple[set[str], set[str]]] = {
        "train": (set(), set()),
        "validation": (set(), set()),
        "internal_test": (set(), set()),
    }
    for assignment in split_manifest.assignments:
        patients, cases = result[assignment.partition]
        patients.add(assignment.anonymous_patient_id)
        cases.add(assignment.anonymous_case_id)
    return result


def test_successful_deterministic_split_cli_output_and_artifact(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "out" / "split.json"
    _write_manifest(manifest_path)

    result = _invoke(*_split_args(manifest_path, output))

    assert result.exit_code == 0
    assert "split generation success" in result.output
    assert f"policy version: {PATIENT_HASH_RANK_POLICY_VERSION}" in result.output
    assert "split seed: 1729" in result.output
    assert "train patient count: 3" in result.output
    assert "validation patient count: 2" in result.output
    assert "internal-test patient count: 1" in result.output
    assert "total case count: 6" in result.output
    assert "source manifest hash: " in result.output
    assert "split hash: " in result.output
    assert f"output path: {output}" in result.output
    assert "anon-p" not in result.output
    assert "anon-c" not in result.output
    assert "images/" not in result.output
    assert "labels/" not in result.output
    assert "case-" not in result.output

    split_manifest = phase2_artifact_from_json(
        output.read_text(encoding="utf-8"),
        DevelopmentSplitManifest,
    )
    assert split_manifest.train_patient_count == 3
    assert split_manifest.validation_patient_count == 2
    assert split_manifest.internal_test_patient_count == 1
    assert split_manifest.split_hash == hash_development_split(split_manifest)

    partitions = _partition_sets(split_manifest)
    assert partitions["train"][0].isdisjoint(partitions["validation"][0])
    assert partitions["train"][0].isdisjoint(partitions["internal_test"][0])
    assert partitions["validation"][0].isdisjoint(partitions["internal_test"][0])
    assert partitions["train"][1].isdisjoint(partitions["validation"][1])
    assert partitions["train"][1].isdisjoint(partitions["internal_test"][1])
    assert partitions["validation"][1].isdisjoint(partitions["internal_test"][1])


def test_repeated_independent_cli_runs_are_byte_identical(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    first_output = tmp_path / "first" / "split.json"
    second_output = tmp_path / "second" / "split.json"
    _write_manifest(manifest_path)

    first = _invoke(*_split_args(manifest_path, first_output))
    second = _invoke(*_split_args(manifest_path, second_output))

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first_output.read_bytes() == second_output.read_bytes()


def test_split_cli_failures_are_bounded_and_nonleaking(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "out" / "split.json"
    _write_manifest(manifest_path)

    invalid_counts = _invoke(*_split_args(manifest_path, output, train_count=4))
    assert invalid_counts.exit_code != 0
    assert "Phase 2 split error: " in _error_text(invalid_counts)
    assert "Traceback" not in _error_text(invalid_counts)
    assert "anon-p" not in _error_text(invalid_counts)
    assert "images/" not in _error_text(invalid_counts)

    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["manifest_hash"] = "f" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    altered = _invoke(*_split_args(manifest_path, tmp_path / "altered.json"))
    assert altered.exit_code != 0
    assert "Traceback" not in _error_text(altered)
    assert "anon-p" not in _error_text(altered)
    assert "images/" not in _error_text(altered)

    _write_manifest(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("occupied\n", encoding="utf-8")
    existing = _invoke(*_split_args(manifest_path, output))
    assert existing.exit_code != 0
    assert "Traceback" not in _error_text(existing)
    assert "anon-c" not in _error_text(existing)
    assert "labels/" not in _error_text(existing)
