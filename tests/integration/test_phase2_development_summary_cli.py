"""Integration tests for the Phase 2 development-summary CLI."""

from __future__ import annotations

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
    DevelopmentDataSummaryArtifact,
    DevelopmentSplitManifest,
    GeometryLabelQaArtifact,
    LesionComponentsArtifact,
    SplitAssignment,
    hash_dataset_manifest,
    hash_development_data_summary,
    hash_development_split,
    hash_geometry_label_qa,
    hash_lesion_components,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
    sha256_file,
)
from protoem_ct.cli.main import app

CREATED_AT_UTC = "2026-07-26T00:00:00Z"
GIT_COMMIT = "826a12c"
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
            partition=partitions[index % 3],
        )
        for index, case in enumerate(manifest.cases)
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


def _write_inputs(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    *,
    mixed_failure: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    root = tmp_path / "dataset"
    first_image = np.array(
        [[[-1.0, 0.0, 1.9], [2.0, 3.9, 4.0]], [[5.9, 6.0, 7.9], [8.0, 10.0, 11.0]]],
        dtype=np.float32,
    )
    first_label = np.zeros(first_image.shape, dtype=np.uint8)
    first_label[0, 0, 0] = 1
    cases = [
        _write_case(
            root,
            index=1,
            image_data=cast(NiftiArray, first_image),
            label_data=cast(NiftiArray, first_label),
            affine=matching_affine,
            write_nifti_file=write_nifti_file,
        )
    ]
    if mixed_failure:
        second_image = np.ones(first_image.shape, dtype=np.float32)
        second_label = np.full(first_image.shape, 2, dtype=np.uint8)
        cases.append(
            _write_case(
                root,
                index=2,
                image_data=cast(NiftiArray, second_image),
                label_data=cast(NiftiArray, second_label),
                affine=matching_affine,
                write_nifti_file=write_nifti_file,
            )
        )
    manifest = _manifest(tuple(cases))
    split = _split(manifest)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_root / "manifest.json"
    split_path = artifact_root / "split.json"
    geometry_path = artifact_root / "geometry.json"
    lesion_path = artifact_root / "lesion.json"
    manifest_path.write_text(phase2_artifact_to_json(manifest), encoding="utf-8", newline="\n")
    split_path.write_text(phase2_artifact_to_json(split), encoding="utf-8", newline="\n")

    qa_result = _invoke(
        "qa-geometry-labels",
        "--manifest",
        str(manifest_path),
        "--split",
        str(split_path),
        "--dataset-root",
        str(root),
        "--output",
        str(geometry_path),
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
    )
    assert qa_result.exit_code == 0, qa_result.output

    lesion_result = _invoke(
        "summarize-lesions",
        "--manifest",
        str(manifest_path),
        "--geometry-qa-artifact",
        str(geometry_path),
        "--dataset-root",
        str(root),
        "--output",
        str(lesion_path),
        "--connectivity",
        "26",
        "--tumor-label-value",
        "1",
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        CREATED_AT_UTC,
    )
    assert lesion_result.exit_code == 0, lesion_result.output
    return root, manifest_path, geometry_path, lesion_path, split_path


def _summary_args(
    root: Path,
    manifest_path: Path,
    geometry_path: Path,
    lesion_path: Path,
    output: Path,
) -> list[str]:
    return [
        "summarize-development-data",
        "--manifest",
        str(manifest_path),
        "--geometry-qa-artifact",
        str(geometry_path),
        "--lesion-artifact",
        str(lesion_path),
        "--dataset-root",
        str(root),
        "--output",
        str(output),
        "--histogram-min",
        "0.0",
        "--histogram-max",
        "10.0",
        "--histogram-bin-count",
        "5",
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        CREATED_AT_UTC,
    ]


def _error_text(result: Any) -> str:
    return getattr(result, "stderr", "") or result.output


def test_synthetic_development_summary_cli_success(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
) -> None:
    root, manifest_path, geometry_path, lesion_path, _split_path = _write_inputs(
        tmp_path,
        write_nifti_file,
        matching_affine,
    )
    output = tmp_path / "out" / "summary.json"

    result = _invoke(*_summary_args(root, manifest_path, geometry_path, lesion_path, output))

    assert result.exit_code == 0, result.output
    assert "development data summary completed" in result.output
    assert "case count: 1" in result.output
    assert "analyzed case count: 1" in result.output
    assert "skipped case count: 0" in result.output
    assert "aggregate image voxel count: 12" in result.output
    assert "total lesion count: 1" in result.output
    assert "histogram bin count: 5" in result.output
    assert "config hash: " in result.output
    assert "manifest hash: " in result.output
    assert "split hash: " in result.output
    assert "summary artifact hash: " in result.output
    assert f"output path: {output}" in result.output
    assert str(root) not in result.output
    assert "anon-p" not in result.output
    assert "images/" not in result.output
    assert "labels/" not in result.output

    artifact = phase2_artifact_from_json(
        output.read_text(encoding="utf-8"),
        DevelopmentDataSummaryArtifact,
    )
    assert artifact.aggregate_histogram_counts == (2, 2, 2, 2, 2)
    assert artifact.aggregate_below_histogram_range_count == 1
    assert artifact.aggregate_above_histogram_range_count == 1
    assert hash_development_data_summary(artifact) == artifact.summary_artifact_hash


def test_mixed_case_cli_succeeds_and_reports_skipped_without_identifiers(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
) -> None:
    root, manifest_path, geometry_path, lesion_path, _split_path = _write_inputs(
        tmp_path,
        write_nifti_file,
        matching_affine,
        mixed_failure=True,
    )
    output = tmp_path / "out" / "summary.json"

    result = _invoke(*_summary_args(root, manifest_path, geometry_path, lesion_path, output))

    assert result.exit_code == 0, result.output
    assert "case count: 2" in result.output
    assert "analyzed case count: 1" in result.output
    assert "skipped case count: 1" in result.output
    assert "anon-c" not in result.output
    artifact = phase2_artifact_from_json(
        output.read_text(encoding="utf-8"),
        DevelopmentDataSummaryArtifact,
    )
    assert artifact.skipped_case_count == 1


def test_repeated_independent_outputs_are_byte_identical(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
) -> None:
    root, manifest_path, geometry_path, lesion_path, _split_path = _write_inputs(
        tmp_path,
        write_nifti_file,
        matching_affine,
    )
    first_output = tmp_path / "first" / "summary.json"
    second_output = tmp_path / "second" / "summary.json"

    first = _invoke(*_summary_args(root, manifest_path, geometry_path, lesion_path, first_output))
    second = _invoke(*_summary_args(root, manifest_path, geometry_path, lesion_path, second_output))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert first_output.read_bytes() == second_output.read_bytes()


def test_summary_cli_structural_failures_do_not_traceback(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
) -> None:
    root, manifest_path, geometry_path, lesion_path, _split_path = _write_inputs(
        tmp_path,
        write_nifti_file,
        matching_affine,
    )
    output = tmp_path / "out" / "summary.json"
    (root / "images/case-0001.nii.gz").write_bytes(b"changed")

    changed = _invoke(*_summary_args(root, manifest_path, geometry_path, lesion_path, output))

    assert changed.exit_code != 0
    assert "Traceback" not in _error_text(changed)
    assert str(root) not in _error_text(changed)

    root, manifest_path, geometry_path, lesion_path, _split_path = _write_inputs(
        tmp_path / "tampered",
        write_nifti_file,
        matching_affine,
    )
    lesion = phase2_artifact_from_json(
        lesion_path.read_text(encoding="utf-8"),
        LesionComponentsArtifact,
    )
    lesion_path.write_text(
        phase2_artifact_to_json(replace(lesion, lesion_artifact_hash="f" * 64)),
        encoding="utf-8",
        newline="\n",
    )
    tampered = _invoke(*_summary_args(root, manifest_path, geometry_path, lesion_path, output))
    assert tampered.exit_code != 0
    assert "Traceback" not in _error_text(tampered)

    root, manifest_path, geometry_path, lesion_path, _split_path = _write_inputs(
        tmp_path / "existing",
        write_nifti_file,
        matching_affine,
    )
    existing_output = tmp_path / "existing-output.json"
    existing_output.write_text("occupied\n", encoding="utf-8")
    existing = _invoke(
        *_summary_args(root, manifest_path, geometry_path, lesion_path, existing_output)
    )
    assert existing.exit_code != 0
    assert "Traceback" not in _error_text(existing)


def test_upstream_hashes_verify_for_cli_inputs(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
) -> None:
    _root, _manifest_path, geometry_path, lesion_path, _split_path = _write_inputs(
        tmp_path,
        write_nifti_file,
        matching_affine,
    )
    geometry = phase2_artifact_from_json(
        geometry_path.read_text(encoding="utf-8"),
        GeometryLabelQaArtifact,
    )
    lesion = phase2_artifact_from_json(
        lesion_path.read_text(encoding="utf-8"),
        LesionComponentsArtifact,
    )
    assert hash_geometry_label_qa(geometry) == geometry.qa_artifact_hash
    assert hash_lesion_components(lesion) == lesion.lesion_artifact_hash
