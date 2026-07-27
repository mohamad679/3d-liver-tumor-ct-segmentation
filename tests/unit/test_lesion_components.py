"""Unit tests for Phase 2 lesion connected-component summaries."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    GEOMETRY_LABEL_QA_STAGE,
    LESION_COMPONENTS_STAGE,
    LESION_COMPONENTS_UPSTREAM_FAILURE_REASON,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    GeometryLabelQaArtifact,
    GeometryLabelQaCaseRecord,
    LesionComponentCaseRecord,
    LesionComponentsArtifact,
    LesionSummaryRecord,
    Phase2ArtifactSerializationError,
    Phase2ArtifactStageError,
    Phase2ArtifactValidationError,
    hash_dataset_manifest,
    hash_geometry_label_qa,
    hash_lesion_components,
    lesion_components_hash_payload,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
    sha256_file,
)
from protoem_ct.data import (
    ExistingLesionComponentsOutputError,
    InvalidLesionComponentsConfigError,
    LesionComponentsConfig,
    LesionComponentsHashMismatchError,
    LesionComponentsLinkageError,
    LesionComponentsPublicationError,
    LesionComponentsSourceIntegrityError,
    UnsafeLesionComponentsOutputPathError,
    hash_lesion_components_config,
    run_lesion_component_analysis,
)
from protoem_ct.data import lesion_components as lesion_module

CREATED_AT_UTC = "2026-07-26T00:00:00Z"
GIT_COMMIT = "3133d32"
HASH0 = "0" * 64
NiftiArray = npt.NDArray[np.generic]
AffineArray = npt.NDArray[np.float64]
WriteNiftiFile = Callable[[Path, NiftiArray, AffineArray], Path]


def _config(**overrides: object) -> LesionComponentsConfig:
    values: dict[str, object] = {"connectivity": 26, "tumor_label_value": 1}
    values.update(overrides)
    return LesionComponentsConfig(**cast(Any, values))


def _write_case(
    root: Path,
    *,
    index: int,
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
    image_data = np.full(label_data.shape, float(index), dtype=np.float32)
    write_nifti_file(image_path, cast(NiftiArray, image_data), affine)
    write_nifti_file(label_path, label_data, affine)
    return DatasetCaseRecord(
        anonymous_patient_id=f"anon-p{index:04d}",
        anonymous_case_id=f"anon-c{index:04d}",
        relative_image_path=image_relative,
        relative_label_path=label_relative,
        image_sha256=sha256_file(image_path),
        label_sha256=sha256_file(label_path),
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
    )


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
        dataset_root_fingerprint=HASH0,
        manifest_hash=HASH0,
        case_count=len(cases),
        cases=cases,
    )
    return replace(without_hash, manifest_hash=hash_dataset_manifest(without_hash))


def _geometry_case(
    case: DatasetCaseRecord,
    *,
    label_shape: tuple[int, int, int] = (3, 3, 3),
    label_spacing: tuple[float, float, float] = (1.0, 1.5, 2.0),
    label_affine_array: AffineArray | None = None,
    tumor_voxel_count: int = 1,
    qa_passed: bool = True,
    failure_reasons: tuple[str, ...] = (),
) -> GeometryLabelQaCaseRecord:
    label_affine: tuple[tuple[float, float, float, float], ...]
    if label_affine_array is None:
        label_affine = (
            (label_spacing[0], 0.0, 0.0, 4.0),
            (0.0, label_spacing[1], 0.0, 5.0),
            (0.0, 0.0, label_spacing[2], 6.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    else:
        label_affine = cast(
            tuple[tuple[float, float, float, float], ...],
            tuple(tuple(float(value) for value in row) for row in label_affine_array.tolist()),
        )
    return GeometryLabelQaCaseRecord(
        anonymous_patient_id=case.anonymous_patient_id,
        anonymous_case_id=case.anonymous_case_id,
        partition="train" if case.anonymous_case_id.endswith("1") else "validation",
        dimensionality=3,
        image_shape=label_shape,
        label_shape=label_shape,
        image_dtype="float32",
        label_dtype="uint8",
        image_affine=label_affine,
        label_affine=label_affine,
        image_orientation=("R", "A", "S"),
        label_orientation=("R", "A", "S"),
        image_spacing=label_spacing,
        label_spacing=label_spacing,
        image_finite=True,
        label_finite=True,
        observed_label_values=(0, 1),
        allowed_label_values=(0, 1),
        image_label_shape_match=True,
        image_label_affine_match=True,
        tumor_label_value=1,
        tumor_voxel_count=tumor_voxel_count,
        empty_tumor=tumor_voxel_count == 0,
        qa_passed=qa_passed,
        failure_reasons=failure_reasons,
    )


def _geometry_artifact(
    manifest: DatasetManifest,
    records: tuple[GeometryLabelQaCaseRecord, ...],
) -> GeometryLabelQaArtifact:
    without_hash = GeometryLabelQaArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=GEOMETRY_LABEL_QA_STAGE,
        created_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        config_hash=HASH0,
        manifest_hash=manifest.manifest_hash,
        split_hash="1" * 64,
        case_count=len(records),
        passed_case_count=sum(1 for record in records if record.qa_passed),
        failed_case_count=sum(1 for record in records if not record.qa_passed),
        case_records=records,
        qa_artifact_hash=HASH0,
    )
    return replace(without_hash, qa_artifact_hash=hash_geometry_label_qa(without_hash))


def _write_artifacts(
    tmp_path: Path,
    manifest: DatasetManifest,
    geometry: GeometryLabelQaArtifact,
) -> tuple[Path, Path]:
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    geometry_path = tmp_path / "artifacts" / "geometry.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(phase2_artifact_to_json(manifest), encoding="utf-8", newline="\n")
    geometry_path.write_text(phase2_artifact_to_json(geometry), encoding="utf-8", newline="\n")
    return manifest_path, geometry_path


def _fixture(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    label: NiftiArray,
    affine: AffineArray,
    *,
    qa_passed: bool = True,
    tumor_voxel_count: int | None = None,
) -> tuple[Path, DatasetManifest, GeometryLabelQaArtifact, Path, Path]:
    root = tmp_path / "dataset"
    case = _write_case(
        root,
        index=1,
        label_data=label,
        affine=affine,
        write_nifti_file=write_nifti_file,
    )
    manifest = _manifest((case,))
    geometry_case = _geometry_case(
        case,
        label_shape=cast(tuple[int, int, int], tuple(int(value) for value in label.shape)),
        label_spacing=(float(affine[0, 0]), float(affine[1, 1]), float(affine[2, 2])),
        label_affine_array=affine,
        tumor_voxel_count=int(np.count_nonzero(label == 1))
        if tumor_voxel_count is None
        else tumor_voxel_count,
        qa_passed=qa_passed,
        failure_reasons=() if qa_passed else ("label_value_not_allowed",),
    )
    geometry = _geometry_artifact(manifest, (geometry_case,))
    manifest_path, geometry_path = _write_artifacts(tmp_path, manifest, geometry)
    return root, manifest, geometry, manifest_path, geometry_path


def _run(
    manifest_path: Path,
    geometry_path: Path,
    root: Path,
    output: Path,
    *,
    config: LesionComponentsConfig | None = None,
) -> LesionComponentsArtifact:
    return run_lesion_component_analysis(
        manifest_path,
        geometry_path,
        dataset_root=root,
        output_path=output,
        config=config or _config(),
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )


def test_config_schema_roundtrip_hash_and_immutability() -> None:
    assert hash_lesion_components_config(_config(connectivity=6)) == hash_lesion_components_config(
        _config(connectivity=6)
    )
    for connectivity in (6, 18, 26):
        assert _config(connectivity=connectivity).connectivity == connectivity
    with pytest.raises(InvalidLesionComponentsConfigError):
        _config(connectivity=5)
    with pytest.raises(InvalidLesionComponentsConfigError):
        _config(tumor_label_value=True)
    config = _config()
    with pytest.raises(FrozenInstanceError):
        cast(Any, config).connectivity = 6

    analyzed = LesionComponentCaseRecord(
        anonymous_patient_id="anon-p0001",
        anonymous_case_id="anon-c0001",
        partition="train",
        analysis_performed=True,
        connectivity=26,
        tumor_label_value=1,
        voxel_volume_mm3=2.0,
        tumor_voxel_count=3,
        tumor_physical_volume_mm3=6.0,
        lesion_count=1,
        lesions=(LesionSummaryRecord(1, 3, 6.0),),
        qa_passed=True,
        failure_reasons=(),
    )
    skipped = replace(
        analyzed,
        anonymous_patient_id="anon-p0002",
        anonymous_case_id="anon-c0002",
        partition="validation",
        analysis_performed=False,
        voxel_volume_mm3=None,
        tumor_voxel_count=None,
        tumor_physical_volume_mm3=None,
        lesion_count=None,
        lesions=(),
        qa_passed=False,
        failure_reasons=(LESION_COMPONENTS_UPSTREAM_FAILURE_REASON,),
    )
    without_hash = LesionComponentsArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=LESION_COMPONENTS_STAGE,
        created_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        config_hash=HASH0,
        manifest_hash="1" * 64,
        split_hash="2" * 64,
        geometry_qa_artifact_hash="3" * 64,
        connectivity=26,
        case_count=2,
        analyzed_case_count=1,
        skipped_case_count=1,
        case_records=(analyzed, skipped),
        lesion_artifact_hash=HASH0,
    )
    artifact = replace(without_hash, lesion_artifact_hash=hash_lesion_components(without_hash))
    text = phase2_artifact_to_json(artifact)
    assert phase2_artifact_from_json(text, LesionComponentsArtifact) == artifact
    loaded = phase2_artifact_from_json(text)
    assert isinstance(loaded, LesionComponentsArtifact)
    assert loaded.stage == LESION_COMPONENTS_STAGE
    assert "lesion_artifact_hash" not in lesion_components_hash_payload(artifact)
    assert phase2_artifact_to_json(artifact) == text

    with pytest.raises(Phase2ArtifactStageError):
        replace(artifact, stage="wrong")
    with pytest.raises(Phase2ArtifactValidationError):
        replace(artifact, analyzed_case_count=2)
    with pytest.raises(Phase2ArtifactValidationError):
        replace(skipped, lesion_count=0)
    with pytest.raises(Phase2ArtifactValidationError):
        phase2_artifact_to_json(replace(analyzed, voxel_volume_mm3=math.inf))


def test_connectivity_component_counts_and_volumes(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
) -> None:
    affine = np.diag(np.array([1.0, 2.0, 3.0, 1.0], dtype=np.float64))
    edge_diagonal = np.zeros((3, 3, 3), dtype=np.uint8)
    edge_diagonal[0, 0, 0] = 1
    edge_diagonal[1, 1, 0] = 1
    root, _manifest_value, _geometry, manifest_path, geometry_path = _fixture(
        tmp_path,
        write_nifti_file,
        cast(NiftiArray, edge_diagonal),
        affine,
    )

    conn6 = _run(
        manifest_path,
        geometry_path,
        root,
        tmp_path / "c6" / "lesions.json",
        config=_config(connectivity=6),
    )
    conn18 = _run(
        manifest_path,
        geometry_path,
        root,
        tmp_path / "c18" / "lesions.json",
        config=_config(connectivity=18),
    )
    assert conn6.case_records[0].lesion_count == 2
    assert conn18.case_records[0].lesion_count == 1
    assert conn18.case_records[0].voxel_volume_mm3 == 6.0
    assert conn18.case_records[0].tumor_physical_volume_mm3 == 12.0

    corner = np.zeros((3, 3, 3), dtype=np.uint8)
    corner[0, 0, 0] = 1
    corner[1, 1, 1] = 1
    root, _manifest_value, _geometry, manifest_path, geometry_path = _fixture(
        tmp_path / "corner",
        write_nifti_file,
        cast(NiftiArray, corner),
        affine,
    )
    conn18_corner = _run(
        manifest_path,
        geometry_path,
        root,
        tmp_path / "corner18" / "lesions.json",
        config=_config(connectivity=18),
    )
    conn26_corner = _run(
        manifest_path,
        geometry_path,
        root,
        tmp_path / "corner26" / "lesions.json",
        config=_config(connectivity=26),
    )
    assert conn18_corner.case_records[0].lesion_count == 2
    assert conn26_corner.case_records[0].lesion_count == 1


def test_empty_one_multiple_and_ordered_lesions(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
) -> None:
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    label[0, 0, 0] = 1
    label[0, 0, 1] = 1
    label[0, 0, 2] = 1
    label[2, 2, 2] = 1
    label[3, 0, 0] = 1
    root, _manifest_value, _geometry, manifest_path, geometry_path = _fixture(
        tmp_path,
        write_nifti_file,
        cast(NiftiArray, label),
        matching_affine,
    )

    artifact = _run(manifest_path, geometry_path, root, tmp_path / "out" / "lesions.json")
    record = artifact.case_records[0]

    assert record.lesion_count == 3
    assert tuple(lesion.voxel_count for lesion in record.lesions) == (3, 1, 1)
    assert tuple(lesion.lesion_index for lesion in record.lesions) == (1, 2, 3)
    assert record.tumor_voxel_count == 5
    assert record.tumor_physical_volume_mm3 == 15.0
    assert phase2_artifact_to_json(artifact) == phase2_artifact_to_json(
        _run(manifest_path, geometry_path, root, tmp_path / "again" / "lesions.json")
    )

    empty = np.zeros((3, 3, 3), dtype=np.uint8)
    root, _manifest_value, _geometry, manifest_path, geometry_path = _fixture(
        tmp_path / "empty",
        write_nifti_file,
        cast(NiftiArray, empty),
        matching_affine,
    )
    empty_artifact = _run(
        manifest_path, geometry_path, root, tmp_path / "empty-out" / "lesions.json"
    )
    assert empty_artifact.case_records[0].lesion_count == 0
    assert empty_artifact.case_records[0].lesions == ()


def test_upstream_failed_case_is_skipped_without_opening_label(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
) -> None:
    label = np.zeros((3, 3, 3), dtype=np.uint8)
    root, manifest, _geometry, manifest_path, geometry_path = _fixture(
        tmp_path,
        write_nifti_file,
        cast(NiftiArray, label),
        matching_affine,
        qa_passed=False,
    )
    (root / manifest.cases[0].relative_label_path).unlink()

    artifact = _run(manifest_path, geometry_path, root, tmp_path / "out" / "lesions.json")

    record = artifact.case_records[0]
    assert not record.analysis_performed
    assert record.failure_reasons == (LESION_COMPONENTS_UPSTREAM_FAILURE_REASON,)
    assert record.lesion_count is None


def test_integrity_failures_and_image_files_are_not_opened(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = np.zeros((3, 3, 3), dtype=np.uint8)
    label[0, 0, 0] = 1
    root, manifest, geometry, manifest_path, geometry_path = _fixture(
        tmp_path,
        write_nifti_file,
        cast(NiftiArray, label),
        matching_affine,
    )
    output = tmp_path / "out" / "lesions.json"

    tampered_manifest = replace(manifest, manifest_hash="f" * 64)
    manifest_path.write_text(phase2_artifact_to_json(tampered_manifest), encoding="utf-8")
    with pytest.raises(LesionComponentsHashMismatchError):
        _run(manifest_path, geometry_path, root, output)
    assert not output.exists()

    manifest_path.write_text(phase2_artifact_to_json(manifest), encoding="utf-8")
    tampered_geometry = replace(geometry, qa_artifact_hash="f" * 64)
    geometry_path.write_text(phase2_artifact_to_json(tampered_geometry), encoding="utf-8")
    with pytest.raises(LesionComponentsHashMismatchError):
        _run(manifest_path, geometry_path, root, output)
    assert not output.exists()

    wrong_linkage_without_hash = replace(geometry, manifest_hash="e" * 64, qa_artifact_hash=HASH0)
    wrong_linkage = replace(
        wrong_linkage_without_hash,
        qa_artifact_hash=hash_geometry_label_qa(wrong_linkage_without_hash),
    )
    geometry_path.write_text(phase2_artifact_to_json(wrong_linkage), encoding="utf-8")
    with pytest.raises(LesionComponentsLinkageError):
        _run(manifest_path, geometry_path, root, output)
    assert not output.exists()

    geometry_path.write_text(phase2_artifact_to_json(geometry), encoding="utf-8")
    (root / manifest.cases[0].relative_label_path).write_bytes(b"changed")
    with pytest.raises(LesionComponentsSourceIntegrityError):
        _run(manifest_path, geometry_path, root, output)
    assert not output.exists()

    assert (root / manifest.cases[0].relative_image_path).exists()
    (root / manifest.cases[0].relative_image_path).unlink()
    write_nifti_file(
        root / manifest.cases[0].relative_label_path, cast(NiftiArray, label), matching_affine
    )
    altered_case = replace(
        manifest.cases[0],
        label_sha256=sha256_file(root / manifest.cases[0].relative_label_path),
    )
    altered_manifest = _manifest((altered_case,))
    altered_geometry = _geometry_artifact(
        altered_manifest,
        (_geometry_case(altered_case, label_affine_array=matching_affine),),
    )
    manifest_path.write_text(phase2_artifact_to_json(altered_manifest), encoding="utf-8")
    geometry_path.write_text(phase2_artifact_to_json(altered_geometry), encoding="utf-8")
    _run(manifest_path, geometry_path, root, output)

    def _fail_serialize(_artifact: object) -> str:
        raise Phase2ArtifactSerializationError("synthetic serialization failure")

    fresh_root, _manifest_value, _geometry, fresh_manifest_path, fresh_geometry_path = _fixture(
        tmp_path / "fresh",
        write_nifti_file,
        cast(NiftiArray, label),
        matching_affine,
    )
    monkeypatch.setattr(lesion_module, "phase2_artifact_to_json", _fail_serialize)
    failed_output = tmp_path / "failed" / "lesions.json"
    with pytest.raises(LesionComponentsPublicationError):
        _run(fresh_manifest_path, fresh_geometry_path, fresh_root, failed_output)
    assert not failed_output.exists()
    assert not (failed_output.parent / ".lesions.json.tmp").exists()


def test_metadata_mismatch_missing_symlink_duplicate_and_output_safety_failures(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
) -> None:
    label = np.zeros((3, 3, 3), dtype=np.uint8)
    label[0, 0, 0] = 1
    root, manifest, geometry, manifest_path, geometry_path = _fixture(
        tmp_path,
        write_nifti_file,
        cast(NiftiArray, label),
        matching_affine,
    )
    output = tmp_path / "out" / "lesions.json"

    bad_geometry_case = replace(geometry.case_records[0], tumor_voxel_count=2, empty_tumor=False)
    bad_geometry = _geometry_artifact(manifest, (bad_geometry_case,))
    geometry_path.write_text(phase2_artifact_to_json(bad_geometry), encoding="utf-8")
    with pytest.raises(LesionComponentsSourceIntegrityError):
        _run(manifest_path, geometry_path, root, output)

    geometry_path.write_text(phase2_artifact_to_json(geometry), encoding="utf-8")
    (root / manifest.cases[0].relative_label_path).unlink()
    with pytest.raises(LesionComponentsSourceIntegrityError):
        _run(manifest_path, geometry_path, root, output)

    write_nifti_file(
        root / manifest.cases[0].relative_label_path, cast(NiftiArray, label), matching_affine
    )
    existing = tmp_path / "existing.json"
    existing.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(ExistingLesionComponentsOutputError):
        _run(manifest_path, geometry_path, root, existing)
    with pytest.raises(UnsafeLesionComponentsOutputPathError):
        _run(manifest_path, geometry_path, root, root / "lesions.json")
    with pytest.raises(UnsafeLesionComponentsOutputPathError):
        _run(manifest_path, geometry_path, root, Path("relative.json"))
    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a dir\n", encoding="utf-8")
    with pytest.raises(UnsafeLesionComponentsOutputPathError):
        _run(manifest_path, geometry_path, root, parent_file / "lesions.json")

    second_root = tmp_path / "dupe"
    first_case = _write_case(
        second_root,
        index=1,
        label_data=cast(NiftiArray, label),
        affine=matching_affine,
        write_nifti_file=write_nifti_file,
    )
    second_case = _write_case(
        second_root,
        index=2,
        label_data=cast(NiftiArray, label),
        affine=matching_affine,
        write_nifti_file=write_nifti_file,
    )
    same_label_path = second_root / first_case.relative_label_path
    second_label_path = second_root / second_case.relative_label_path
    second_label_path.unlink()
    second_label_path.symlink_to(same_label_path)
    second_case = replace(second_case, label_sha256=first_case.label_sha256)
    duplicate_manifest = _manifest((first_case, second_case))
    duplicate_geometry = _geometry_artifact(
        duplicate_manifest,
        (
            _geometry_case(first_case, label_affine_array=matching_affine),
            _geometry_case(second_case, label_affine_array=matching_affine),
        ),
    )
    duplicate_manifest_path, duplicate_geometry_path = _write_artifacts(
        tmp_path / "duplicate-artifacts",
        duplicate_manifest,
        duplicate_geometry,
    )
    with pytest.raises(LesionComponentsSourceIntegrityError):
        _run(
            duplicate_manifest_path,
            duplicate_geometry_path,
            second_root,
            tmp_path / "duplicate-out" / "lesions.json",
        )


def test_output_independent_of_current_working_directory_and_no_paths(
    tmp_path: Path,
    write_nifti_file: WriteNiftiFile,
    matching_affine: AffineArray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = np.zeros((3, 3, 3), dtype=np.uint8)
    label[0, 0, 0] = 1
    root, _manifest_value, _geometry, manifest_path, geometry_path = _fixture(
        tmp_path,
        write_nifti_file,
        cast(NiftiArray, label),
        matching_affine,
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    first = _run(manifest_path, geometry_path, root, tmp_path / "first" / "lesions.json")
    second = _run(manifest_path, geometry_path, root, tmp_path / "second" / "lesions.json")

    assert phase2_artifact_to_json(first) == phase2_artifact_to_json(second)
    json_text = phase2_artifact_to_json(first)
    for forbidden in (str(root), str(tmp_path), "images/", "labels/", "case-", "source_case_key"):
        assert forbidden not in json_text
