"""Integration tests for the Phase 2 geometry-label QA CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from typer.testing import CliRunner

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentSplitManifest,
    GeometryLabelQaArtifact,
    SplitAssignment,
    hash_dataset_manifest,
    hash_development_split,
    hash_geometry_label_qa,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
    sha256_file,
)
from protoem_ct.cli.main import app

CREATED_AT_UTC = "2026-07-26T00:00:00Z"
GIT_COMMIT = "4b50df7"
NiftiArray = npt.NDArray[np.generic]
AffineArray = npt.NDArray[np.float64]
WriteNiftiFile = Callable[[Path, NiftiArray, AffineArray], Path]


def _invoke(*args: str) -> Any:
    runner = CliRunner()
    return runner.invoke(app, [*args])


def _case(root: Path, index: int, image_relative: str, label_relative: str) -> DatasetCaseRecord:
    return DatasetCaseRecord(
        anonymous_patient_id=f"anon-p{index:04d}",
        anonymous_case_id=f"anon-c{index:04d}",
        relative_image_path=image_relative,
        relative_label_path=label_relative,
        image_sha256=sha256_file(root / image_relative),
        label_sha256=sha256_file(root / label_relative),
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
    )


def _write_case(
    root: Path,
    *,
    index: int,
    image_data: NiftiArray,
    label_data: NiftiArray,
    affine: AffineArray,
    write_nifti_file: WriteNiftiFile,
) -> DatasetCaseRecord:
    image_relative = f"images/case-{index:04d}.nii.gz"
    label_relative = f"labels/case-{index:04d}.nii.gz"
    image_path = root / image_relative
    label_path = root / label_relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    write_nifti_file(image_path, image_data, affine)
    write_nifti_file(label_path, label_data, affine)
    return _case(root, index, image_relative, label_relative)


def _manifest(cases: tuple[DatasetCaseRecord, ...]) -> DatasetManifest:
    without_hash = DatasetManifest(
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
    return replace(without_hash, manifest_hash=hash_dataset_manifest(without_hash))


def _split(manifest: DatasetManifest) -> DevelopmentSplitManifest:
    partitions = ("train", "validation", "internal_test")
    assignments = tuple(
        SplitAssignment(
            anonymous_patient_id=case.anonymous_patient_id,
            anonymous_case_id=case.anonymous_case_id,
            partition=partitions[(index - 1) % 3],
        )
        for index, case in enumerate(manifest.cases, start=1)
    )
    counts = {
        partition: sum(1 for item in assignments if item.partition == partition)
        for partition in partitions
    }
    without_hash = DevelopmentSplitManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DEVELOPMENT_SPLIT_MANIFEST_TYPE,
        generated_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        source_manifest_hash=manifest.manifest_hash,
        split_policy_version="patient_hash_rank_v1",
        split_seed=1729,
        split_hash="0" * 64,
        assignments=assignments,
        train_patient_count=counts["train"],
        validation_patient_count=counts["validation"],
        internal_test_patient_count=counts["internal_test"],
        train_case_count=counts["train"],
        validation_case_count=counts["validation"],
        internal_test_case_count=counts["internal_test"],
    )
    return replace(without_hash, split_hash=hash_development_split(without_hash))


def _fixture(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
    *,
    mixed_failure: bool = False,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "dataset"
    cases = [
        _write_case(
            root,
            index=1,
            image_data=valid_ct_image,
            label_data=valid_binary_tumor_mask,
            affine=matching_affine,
            write_nifti_file=write_nifti_file,
        )
    ]
    if mixed_failure:
        bad_label = np.full(valid_binary_tumor_mask.shape, 2, dtype=np.uint8)
        cases.append(
            _write_case(
                root,
                index=2,
                image_data=cast(NiftiArray, np.add(valid_ct_image, np.float32(0.001))),
                label_data=bad_label,
                affine=matching_affine,
                write_nifti_file=write_nifti_file,
            )
        )
    manifest = _manifest(tuple(cases))
    split = _split(manifest)
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    split_path = tmp_path / "artifacts" / "split.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(phase2_artifact_to_json(manifest), encoding="utf-8", newline="\n")
    split_path.write_text(phase2_artifact_to_json(split), encoding="utf-8", newline="\n")
    return root, manifest_path, split_path


def _qa_args(root: Path, manifest_path: Path, split_path: Path, output: Path) -> list[str]:
    return [
        "qa-geometry-labels",
        "--manifest",
        str(manifest_path),
        "--split",
        str(split_path),
        "--dataset-root",
        str(root),
        "--output",
        str(output),
        "--allowed-label-value",
        "0",
        "--allowed-label-value",
        "1",
        "--tumor-label-value",
        "1",
        "--affine-tolerance",
        "0.00001",
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        CREATED_AT_UTC,
    ]


def _error_text(result: Any) -> str:
    return getattr(result, "stderr", "") or result.output


def test_successful_geometry_label_qa_cli(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
) -> None:
    root, manifest_path, split_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        valid_ct_image,
        valid_binary_tumor_mask,
    )
    output = tmp_path / "out" / "qa.json"

    result = _invoke(*_qa_args(root, manifest_path, split_path, output))

    assert result.exit_code == 0
    assert "geometry and label QA completed" in result.output
    assert "case count: 1" in result.output
    assert "passed case count: 1" in result.output
    assert "failed case count: 0" in result.output
    assert "config hash: " in result.output
    assert "manifest hash: " in result.output
    assert "split hash: " in result.output
    assert "QA artifact hash: " in result.output
    assert f"output path: {output}" in result.output
    assert str(root) not in result.output
    assert "anon-p" not in result.output
    assert "images/" not in result.output

    artifact = phase2_artifact_from_json(
        output.read_text(encoding="utf-8"),
        GeometryLabelQaArtifact,
    )
    assert artifact.qa_artifact_hash == hash_geometry_label_qa(artifact)
    assert artifact.passed_case_count == 1


def test_mixed_failure_cli_writes_artifact_and_exits_zero_without_identifiers(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
) -> None:
    root, manifest_path, split_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        valid_ct_image,
        valid_binary_tumor_mask,
        mixed_failure=True,
    )
    output = tmp_path / "out" / "qa.json"

    result = _invoke(*_qa_args(root, manifest_path, split_path, output))

    assert result.exit_code == 0
    assert "case count: 2" in result.output
    assert "failed case count: 1" in result.output
    assert "anon-c" not in result.output
    assert "case-0002" not in result.output
    artifact = phase2_artifact_from_json(
        output.read_text(encoding="utf-8"),
        GeometryLabelQaArtifact,
    )
    assert artifact.failed_case_count == 1
    assert artifact.case_records[1].failure_reasons == ("label_value_not_allowed",)


def test_repeated_cli_outputs_are_byte_identical(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
) -> None:
    root, manifest_path, split_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        valid_ct_image,
        valid_binary_tumor_mask,
    )
    first_output = tmp_path / "first" / "qa.json"
    second_output = tmp_path / "second" / "qa.json"

    first = _invoke(*_qa_args(root, manifest_path, split_path, first_output))
    second = _invoke(*_qa_args(root, manifest_path, split_path, second_output))

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first_output.read_bytes() == second_output.read_bytes()


def test_geometry_label_qa_cli_structural_failures_are_bounded(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    valid_ct_image: NiftiArray,
    valid_binary_tumor_mask: NiftiArray,
) -> None:
    root, manifest_path, split_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        valid_ct_image,
        valid_binary_tumor_mask,
    )
    output = tmp_path / "out" / "qa.json"

    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["manifest_hash"] = "f" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    altered = _invoke(*_qa_args(root, manifest_path, split_path, output))
    assert altered.exit_code != 0
    assert "Phase 2 geometry-label QA error: " in _error_text(altered)
    assert "Traceback" not in _error_text(altered)
    assert str(root) not in _error_text(altered)
    assert "anon-p" not in _error_text(altered)

    root, manifest_path, split_path = _fixture(
        tmp_path / "changed",
        write_nifti_file,
        matching_affine,
        valid_ct_image,
        valid_binary_tumor_mask,
    )
    output = tmp_path / "changed-out" / "qa.json"
    (root / "images/case-0001.nii.gz").write_bytes(b"changed")
    changed = _invoke(*_qa_args(root, manifest_path, split_path, output))
    assert changed.exit_code != 0
    assert "Traceback" not in _error_text(changed)
    assert str(root) not in _error_text(changed)
    assert "images/" not in _error_text(changed)

    root, manifest_path, split_path = _fixture(
        tmp_path / "existing",
        write_nifti_file,
        matching_affine,
        valid_ct_image,
        valid_binary_tumor_mask,
    )
    output = tmp_path / "existing-output" / "qa.json"
    output.parent.mkdir(parents=True)
    output.write_text("occupied\n", encoding="utf-8")
    existing = _invoke(*_qa_args(root, manifest_path, split_path, output))
    assert existing.exit_code != 0
    assert "Traceback" not in _error_text(existing)
    assert "anon-c" not in _error_text(existing)

    inside = _invoke(*_qa_args(root, manifest_path, split_path, root / "qa.json"))
    assert inside.exit_code != 0
    assert "Traceback" not in _error_text(inside)
