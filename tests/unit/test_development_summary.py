"""Unit tests for Phase 2 fixed CT histograms and development summaries."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import protoem_ct.data.development_summary as summary_module
from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_DATA_SUMMARY_STAGE,
    DEVELOPMENT_DATA_SUMMARY_UPSTREAM_FAILURE_REASON,
    LESION_COMPONENTS_STAGE,
    LESION_COMPONENTS_UPSTREAM_FAILURE_REASON,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    CtHistogramCaseRecord,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentDataSummaryArtifact,
    GeometryLabelQaArtifact,
    GeometryLabelQaCaseRecord,
    LesionComponentCaseRecord,
    LesionComponentsArtifact,
    LesionSummaryRecord,
    Phase2ArtifactSerializationError,
    Phase2ArtifactStageError,
    Phase2ArtifactValidationError,
    development_data_summary_hash_payload,
    hash_dataset_manifest,
    hash_development_data_summary,
    hash_geometry_label_qa,
    hash_lesion_components,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
    sha256_file,
)
from protoem_ct.data.development_summary import (
    MAX_HISTOGRAM_BIN_COUNT,
    DevelopmentSummaryConfig,
    DevelopmentSummaryHashMismatchError,
    DevelopmentSummaryLinkageError,
    DevelopmentSummaryPublicationError,
    DevelopmentSummarySourceIntegrityError,
    ExistingDevelopmentSummaryOutputError,
    InvalidDevelopmentSummaryConfigError,
    UnsafeDevelopmentSummaryOutputPathError,
    hash_development_summary_config,
    run_development_data_summary,
)

CREATED_AT_UTC = "2026-07-26T00:00:00Z"
GIT_COMMIT = "826a12c"
HASH0 = "0" * 64
HASH1 = "1" * 64
HASH2 = "2" * 64
NiftiArray = npt.NDArray[np.generic]
AffineArray = npt.NDArray[np.float64]


def _config(**overrides: object) -> DevelopmentSummaryConfig:
    values: dict[str, object] = {
        "histogram_min": 0.0,
        "histogram_max": 10.0,
        "histogram_bin_count": 5,
    }
    values.update(overrides)
    return DevelopmentSummaryConfig(**values)  # type: ignore[arg-type]


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
    write_nifti_file: Any,
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
        dataset_root_fingerprint=HASH0,
        manifest_hash=HASH0,
        case_count=len(cases),
        cases=cases,
    )
    return replace(without_hash, manifest_hash=hash_dataset_manifest(without_hash))


def _affine_tuple(affine: AffineArray) -> tuple[tuple[float, float, float, float], ...]:
    return cast(
        tuple[tuple[float, float, float, float], ...],
        tuple(tuple(float(item) for item in row) for row in affine.tolist()),
    )


def _spacing_tuple(affine: AffineArray) -> tuple[float, float, float]:
    matrix = affine[:3, :3]
    spacing = np.sqrt(np.sum(matrix * matrix, axis=0))
    return cast(tuple[float, float, float], tuple(float(item) for item in spacing.tolist()))


def _geometry_case(
    case: DatasetCaseRecord,
    *,
    index: int,
    image_data: NiftiArray,
    label_data: NiftiArray,
    affine: AffineArray,
    qa_passed: bool = True,
) -> GeometryLabelQaCaseRecord:
    tumor_voxels = int(np.count_nonzero(label_data == 1))
    return GeometryLabelQaCaseRecord(
        anonymous_patient_id=case.anonymous_patient_id,
        anonymous_case_id=case.anonymous_case_id,
        partition=("train", "validation", "internal_test")[(index - 1) % 3],
        dimensionality=3,
        image_shape=tuple(int(dimension) for dimension in image_data.shape),
        label_shape=tuple(int(dimension) for dimension in label_data.shape),
        image_dtype=str(image_data.dtype),
        label_dtype=str(label_data.dtype),
        image_affine=_affine_tuple(affine),
        label_affine=_affine_tuple(affine),
        image_orientation=("R", "A", "S"),
        label_orientation=("R", "A", "S"),
        image_spacing=_spacing_tuple(affine),
        label_spacing=_spacing_tuple(affine),
        image_finite=True,
        label_finite=True,
        observed_label_values=(0, 1),
        allowed_label_values=(0, 1),
        image_label_shape_match=True,
        image_label_affine_match=True,
        tumor_label_value=1,
        tumor_voxel_count=tumor_voxels,
        empty_tumor=tumor_voxels == 0,
        qa_passed=qa_passed,
        failure_reasons=() if qa_passed else ("label_value_not_allowed",),
    )


def _geometry_artifact(
    manifest: DatasetManifest,
    cases: tuple[GeometryLabelQaCaseRecord, ...],
) -> GeometryLabelQaArtifact:
    without_hash = GeometryLabelQaArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage="geometry_label_qa",
        created_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        config_hash=HASH1,
        manifest_hash=manifest.manifest_hash,
        split_hash=HASH2,
        case_count=len(cases),
        passed_case_count=sum(1 for case in cases if case.qa_passed),
        failed_case_count=sum(1 for case in cases if not case.qa_passed),
        case_records=cases,
        qa_artifact_hash=HASH0,
    )
    return replace(without_hash, qa_artifact_hash=hash_geometry_label_qa(without_hash))


def _lesion_record(
    geometry_case: GeometryLabelQaCaseRecord,
    *,
    voxel_volume: float,
    lesion_volumes: tuple[float, ...] = (3.0,),
) -> LesionComponentCaseRecord:
    if not geometry_case.qa_passed:
        return LesionComponentCaseRecord(
            anonymous_patient_id=geometry_case.anonymous_patient_id,
            anonymous_case_id=geometry_case.anonymous_case_id,
            partition=geometry_case.partition,
            analysis_performed=False,
            connectivity=26,
            tumor_label_value=1,
            voxel_volume_mm3=None,
            tumor_voxel_count=None,
            tumor_physical_volume_mm3=None,
            lesion_count=None,
            lesions=(),
            qa_passed=False,
            failure_reasons=(LESION_COMPONENTS_UPSTREAM_FAILURE_REASON,),
        )
    lesions = tuple(
        LesionSummaryRecord(
            lesion_index=index,
            voxel_count=max(1, int(round(volume / voxel_volume))),
            physical_volume_mm3=volume,
        )
        for index, volume in enumerate(lesion_volumes, start=1)
    )
    return LesionComponentCaseRecord(
        anonymous_patient_id=geometry_case.anonymous_patient_id,
        anonymous_case_id=geometry_case.anonymous_case_id,
        partition=geometry_case.partition,
        analysis_performed=True,
        connectivity=26,
        tumor_label_value=1,
        voxel_volume_mm3=voxel_volume,
        tumor_voxel_count=sum(lesion.voxel_count for lesion in lesions),
        tumor_physical_volume_mm3=float(sum(lesion.physical_volume_mm3 for lesion in lesions)),
        lesion_count=len(lesions),
        lesions=lesions,
        qa_passed=True,
        failure_reasons=(),
    )


def _lesion_artifact(
    manifest: DatasetManifest,
    geometry: GeometryLabelQaArtifact,
    records: tuple[LesionComponentCaseRecord, ...],
) -> LesionComponentsArtifact:
    without_hash = LesionComponentsArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=LESION_COMPONENTS_STAGE,
        created_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        config_hash=HASH2,
        manifest_hash=manifest.manifest_hash,
        split_hash=geometry.split_hash,
        geometry_qa_artifact_hash=geometry.qa_artifact_hash,
        connectivity=26,
        case_count=len(records),
        analyzed_case_count=sum(1 for record in records if record.analysis_performed),
        skipped_case_count=sum(1 for record in records if not record.analysis_performed),
        case_records=records,
        lesion_artifact_hash=HASH0,
    )
    return replace(without_hash, lesion_artifact_hash=hash_lesion_components(without_hash))


def _write_artifacts(
    tmp_path: Path,
    manifest: DatasetManifest,
    geometry: GeometryLabelQaArtifact,
    lesion: LesionComponentsArtifact,
) -> tuple[Path, Path, Path]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_root / "manifest.json"
    geometry_path = artifact_root / "geometry.json"
    lesion_path = artifact_root / "lesion.json"
    manifest_path.write_text(phase2_artifact_to_json(manifest), encoding="utf-8", newline="\n")
    geometry_path.write_text(phase2_artifact_to_json(geometry), encoding="utf-8", newline="\n")
    lesion_path.write_text(phase2_artifact_to_json(lesion), encoding="utf-8", newline="\n")
    return manifest_path, geometry_path, lesion_path


def _fixture(
    tmp_path: Path,
    write_nifti_file: Any,
    affine: AffineArray,
    *,
    image_values: tuple[NiftiArray, ...],
    qa_passes: tuple[bool, ...] | None = None,
    lesion_volumes: tuple[tuple[float, ...], ...] | None = None,
) -> tuple[
    Path,
    DatasetManifest,
    GeometryLabelQaArtifact,
    LesionComponentsArtifact,
    Path,
    Path,
    Path,
]:
    root = tmp_path / "dataset"
    passes = qa_passes or tuple(True for _ in image_values)
    labels = []
    cases = []
    geometry_cases = []
    for index, image_data in enumerate(image_values, start=1):
        label = np.zeros(image_data.shape, dtype=np.uint8)
        if passes[index - 1]:
            label.flat[0] = 1
        labels.append(label)
        case = _write_case(
            root,
            index=index,
            image_data=image_data,
            label_data=cast(NiftiArray, label),
            affine=affine,
            write_nifti_file=write_nifti_file,
        )
        cases.append(case)
        geometry_cases.append(
            _geometry_case(
                case,
                index=index,
                image_data=image_data,
                label_data=cast(NiftiArray, label),
                affine=affine,
                qa_passed=passes[index - 1],
            )
        )
    manifest = _manifest(tuple(cases))
    geometry = _geometry_artifact(manifest, tuple(geometry_cases))
    voxel_volume = float(math.prod(_spacing_tuple(affine)))
    per_case_volumes = lesion_volumes or tuple((voxel_volume,) for _ in image_values)
    lesion = _lesion_artifact(
        manifest,
        geometry,
        tuple(
            _lesion_record(record, voxel_volume=voxel_volume, lesion_volumes=per_case_volumes[i])
            for i, record in enumerate(geometry.case_records)
        ),
    )
    manifest_path, geometry_path, lesion_path = _write_artifacts(
        tmp_path,
        manifest,
        geometry,
        lesion,
    )
    return root, manifest, geometry, lesion, manifest_path, geometry_path, lesion_path


def _run(
    manifest_path: Path,
    geometry_path: Path,
    lesion_path: Path,
    root: Path,
    output: Path,
    *,
    config: DevelopmentSummaryConfig | None = None,
) -> DevelopmentDataSummaryArtifact:
    return run_development_data_summary(
        manifest_path,
        geometry_path,
        lesion_path,
        dataset_root=root,
        output_path=output,
        config=config or _config(),
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )


def test_config_validation_and_hash() -> None:
    config = _config()
    assert config.histogram_bin_edges == (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
    assert hash_development_summary_config(config) == hash_development_summary_config(_config())
    with pytest.raises(InvalidDevelopmentSummaryConfigError):
        _config(histogram_min=math.nan)
    with pytest.raises(InvalidDevelopmentSummaryConfigError):
        _config(histogram_max=math.inf)
    with pytest.raises(InvalidDevelopmentSummaryConfigError):
        _config(histogram_min=10.0, histogram_max=0.0)
    with pytest.raises(InvalidDevelopmentSummaryConfigError):
        _config(histogram_bin_count=0)
    with pytest.raises(InvalidDevelopmentSummaryConfigError):
        _config(histogram_bin_count=MAX_HISTOGRAM_BIN_COUNT + 1)
    with pytest.raises(FrozenInstanceError):
        cast(Any, config).histogram_bin_count = 2


def test_schema_round_trip_and_validation() -> None:
    analyzed = CtHistogramCaseRecord(
        anonymous_patient_id="anon-p0001",
        anonymous_case_id="anon-c0001",
        partition="train",
        analysis_performed=True,
        image_voxel_count=3,
        intensity_min=0.0,
        intensity_max=10.0,
        intensity_mean=4.0,
        intensity_std=2.0,
        histogram_counts=(1, 1, 1),
        below_histogram_range_count=0,
        above_histogram_range_count=0,
        qa_passed=True,
        failure_reasons=(),
    )
    skipped = replace(
        analyzed,
        anonymous_patient_id="anon-p0002",
        anonymous_case_id="anon-c0002",
        partition="validation",
        analysis_performed=False,
        image_voxel_count=None,
        intensity_min=None,
        intensity_max=None,
        intensity_mean=None,
        intensity_std=None,
        histogram_counts=(),
        below_histogram_range_count=None,
        above_histogram_range_count=None,
        qa_passed=False,
        failure_reasons=(DEVELOPMENT_DATA_SUMMARY_UPSTREAM_FAILURE_REASON,),
    )
    without_hash = DevelopmentDataSummaryArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=DEVELOPMENT_DATA_SUMMARY_STAGE,
        created_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        config_hash=HASH0,
        manifest_hash="1" * 64,
        split_hash="2" * 64,
        geometry_qa_artifact_hash="3" * 64,
        lesion_artifact_hash="4" * 64,
        histogram_bin_edges=(0.0, 5.0, 10.0, 15.0),
        histogram_bin_count=3,
        case_count=2,
        analyzed_case_count=1,
        skipped_case_count=1,
        case_records=(analyzed, skipped),
        aggregate_image_voxel_count=3,
        aggregate_intensity_min=0.0,
        aggregate_intensity_max=10.0,
        aggregate_intensity_mean=4.0,
        aggregate_intensity_std=2.0,
        aggregate_histogram_counts=(1, 1, 1),
        aggregate_below_histogram_range_count=0,
        aggregate_above_histogram_range_count=0,
        total_tumor_voxel_count=1,
        total_tumor_physical_volume_mm3=3.0,
        total_lesion_count=1,
        lesion_volume_min_mm3=3.0,
        lesion_volume_max_mm3=3.0,
        lesion_volume_mean_mm3=3.0,
        lesion_volume_median_mm3=3.0,
        summary_artifact_hash=HASH0,
    )
    artifact = replace(
        without_hash,
        summary_artifact_hash=hash_development_data_summary(without_hash),
    )
    text = phase2_artifact_to_json(artifact)
    assert phase2_artifact_from_json(text, DevelopmentDataSummaryArtifact) == artifact
    assert isinstance(phase2_artifact_from_json(text), DevelopmentDataSummaryArtifact)
    assert phase2_artifact_to_json(artifact) == text
    assert "summary_artifact_hash" not in development_data_summary_hash_payload(artifact)

    with pytest.raises(Phase2ArtifactStageError):
        replace(artifact, stage="wrong")
    with pytest.raises(Phase2ArtifactValidationError):
        replace(artifact, aggregate_histogram_counts=(2, 1, 1))
    with pytest.raises(Phase2ArtifactValidationError):
        replace(skipped, image_voxel_count=1)
    with pytest.raises(Phase2ArtifactValidationError):
        replace(analyzed, intensity_mean=math.nan)
    zero_lesion = replace(
        artifact,
        total_tumor_voxel_count=0,
        total_tumor_physical_volume_mm3=0.0,
        total_lesion_count=0,
        lesion_volume_min_mm3=None,
        lesion_volume_max_mm3=None,
        lesion_volume_mean_mm3=None,
        lesion_volume_median_mm3=None,
    )
    assert zero_lesion.total_lesion_count == 0


def test_histogram_edges_and_case_statistics(
    tmp_path: Path,
    write_nifti_file: Any,
    matching_affine: AffineArray,
) -> None:
    values = np.array(
        [[[-1.0, 0.0, 1.9], [2.0, 3.9, 4.0]], [[5.9, 6.0, 7.9], [8.0, 10.0, 11.0]]],
        dtype=np.float32,
    )
    root, _manifest, _geometry, _lesion, manifest_path, geometry_path, lesion_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        image_values=(cast(NiftiArray, values),),
    )
    before = (root / "images/case-0001.nii.gz").read_bytes()

    artifact = _run(manifest_path, geometry_path, lesion_path, root, tmp_path / "out" / "s.json")
    record = artifact.case_records[0]

    assert record.histogram_counts == (2, 2, 2, 2, 2)
    assert record.below_histogram_range_count == 1
    assert record.above_histogram_range_count == 1
    assert record.image_voxel_count == values.size
    assert record.intensity_min == pytest.approx(float(np.min(values)))
    assert record.intensity_max == pytest.approx(float(np.max(values)))
    assert record.intensity_mean == pytest.approx(float(np.mean(values, dtype=np.float64)))
    assert record.intensity_std == pytest.approx(float(np.std(values, dtype=np.float64, ddof=0)))
    assert artifact.aggregate_histogram_counts == record.histogram_counts
    assert (root / "images/case-0001.nii.gz").read_bytes() == before


def test_weighted_aggregate_statistics_and_lesion_medians(
    tmp_path: Path,
    write_nifti_file: Any,
    matching_affine: AffineArray,
) -> None:
    first = np.array([[[0.0, 2.0], [4.0, 6.0]]], dtype=np.float32)
    second = np.array([[[8.0, 10.0, 12.0, 14.0, 16.0, 18.0]]], dtype=np.float32)
    all_values = np.concatenate((first.reshape(-1), second.reshape(-1))).astype(np.float64)
    root, _manifest, _geometry, _lesion, manifest_path, geometry_path, lesion_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        image_values=(cast(NiftiArray, first), cast(NiftiArray, second)),
        lesion_volumes=((3.0,), (6.0, 12.0)),
    )

    artifact = _run(manifest_path, geometry_path, lesion_path, root, tmp_path / "out" / "s.json")

    assert artifact.aggregate_image_voxel_count == all_values.size
    assert artifact.aggregate_intensity_mean == pytest.approx(float(np.mean(all_values)))
    assert artifact.aggregate_intensity_std == pytest.approx(float(np.std(all_values, ddof=0)))
    expected_histogram = np.histogram(
        all_values,
        bins=np.asarray(_config().histogram_bin_edges),
    )[0]
    assert artifact.aggregate_histogram_counts == tuple(int(value) for value in expected_histogram)
    assert artifact.total_lesion_count == 3
    assert artifact.lesion_volume_min_mm3 == 3.0
    assert artifact.lesion_volume_max_mm3 == 12.0
    assert artifact.lesion_volume_mean_mm3 == 7.0
    assert artifact.lesion_volume_median_mm3 == 6.0

    even_root, *_values, even_manifest_path, even_geometry_path, even_lesion_path = _fixture(
        tmp_path / "even",
        write_nifti_file,
        matching_affine,
        image_values=(cast(NiftiArray, first), cast(NiftiArray, second)),
        lesion_volumes=((3.0,), (9.0,)),
    )
    even = _run(
        even_manifest_path,
        even_geometry_path,
        even_lesion_path,
        even_root,
        tmp_path / "even-out" / "s.json",
    )
    assert even.lesion_volume_median_mm3 == 6.0


def test_zero_lesions_and_upstream_failed_cases_skip_image_access(
    tmp_path: Path,
    write_nifti_file: Any,
    matching_affine: AffineArray,
) -> None:
    images = (
        cast(NiftiArray, np.zeros((2, 2, 2), dtype=np.float32)),
        cast(NiftiArray, np.ones((2, 2, 2), dtype=np.float32)),
    )
    root, manifest, geometry, _lesion, manifest_path, geometry_path, _lesion_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        image_values=images,
        qa_passes=(True, False),
        lesion_volumes=((), ()),
    )
    lesion = _lesion_artifact(
        manifest,
        geometry,
        (
            _lesion_record(geometry.case_records[0], voxel_volume=3.0, lesion_volumes=()),
            _lesion_record(geometry.case_records[1], voxel_volume=3.0, lesion_volumes=()),
        ),
    )
    lesion_path = tmp_path / "artifacts" / "lesion.json"
    lesion_path.write_text(phase2_artifact_to_json(lesion), encoding="utf-8", newline="\n")
    (root / manifest.cases[1].relative_image_path).unlink()

    artifact = _run(manifest_path, geometry_path, lesion_path, root, tmp_path / "out" / "s.json")

    assert artifact.analyzed_case_count == 1
    assert artifact.skipped_case_count == 1
    assert artifact.total_lesion_count == 0
    assert artifact.lesion_volume_min_mm3 is None
    assert artifact.case_records[1].failure_reasons == (
        DEVELOPMENT_DATA_SUMMARY_UPSTREAM_FAILURE_REASON,
    )


def test_integrity_and_linkage_failures_leave_no_output(
    tmp_path: Path,
    write_nifti_file: Any,
    matching_affine: AffineArray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest, geometry, lesion, manifest_path, geometry_path, lesion_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        image_values=(cast(NiftiArray, np.zeros((2, 2, 2), dtype=np.float32)),),
    )
    output = tmp_path / "out" / "s.json"

    manifest_path.write_text(
        phase2_artifact_to_json(replace(manifest, manifest_hash="f" * 64)),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(DevelopmentSummaryHashMismatchError):
        _run(manifest_path, geometry_path, lesion_path, root, output)
    assert not output.exists()

    manifest_path.write_text(phase2_artifact_to_json(manifest), encoding="utf-8", newline="\n")
    geometry_path.write_text(
        phase2_artifact_to_json(replace(geometry, qa_artifact_hash="f" * 64)),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(DevelopmentSummaryHashMismatchError):
        _run(manifest_path, geometry_path, lesion_path, root, output)

    geometry_path.write_text(phase2_artifact_to_json(geometry), encoding="utf-8", newline="\n")
    lesion_path.write_text(
        phase2_artifact_to_json(replace(lesion, lesion_artifact_hash="f" * 64)),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(DevelopmentSummaryHashMismatchError):
        _run(manifest_path, geometry_path, lesion_path, root, output)

    wrong_lesion = replace(lesion, split_hash="e" * 64, lesion_artifact_hash=HASH0)
    wrong_lesion = replace(wrong_lesion, lesion_artifact_hash=hash_lesion_components(wrong_lesion))
    lesion_path.write_text(phase2_artifact_to_json(wrong_lesion), encoding="utf-8", newline="\n")
    with pytest.raises(DevelopmentSummaryLinkageError):
        _run(manifest_path, geometry_path, lesion_path, root, output)

    lesion_path.write_text(phase2_artifact_to_json(lesion), encoding="utf-8", newline="\n")
    (root / manifest.cases[0].relative_image_path).write_bytes(b"changed")
    with pytest.raises(DevelopmentSummarySourceIntegrityError):
        _run(manifest_path, geometry_path, lesion_path, root, output)

    fresh_root, _m, _g, _l, fresh_manifest_path, fresh_geometry_path, fresh_lesion_path = _fixture(
        tmp_path / "fresh",
        write_nifti_file,
        matching_affine,
        image_values=(cast(NiftiArray, np.ones((2, 2, 2), dtype=np.float32)),),
    )

    def _fail_serialize(_artifact: object) -> str:
        raise Phase2ArtifactSerializationError("synthetic serialization failure")

    monkeypatch.setattr(summary_module, "phase2_artifact_to_json", _fail_serialize)
    failed_output = tmp_path / "failed" / "s.json"
    with pytest.raises(DevelopmentSummaryPublicationError):
        _run(fresh_manifest_path, fresh_geometry_path, fresh_lesion_path, fresh_root, failed_output)
    assert not failed_output.exists()
    assert not (failed_output.parent / ".s.json.tmp").exists()


def test_metadata_missing_symlink_duplicate_and_output_safety_failures(
    tmp_path: Path,
    write_nifti_file: Any,
    matching_affine: AffineArray,
) -> None:
    root, manifest, geometry, lesion, manifest_path, geometry_path, lesion_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        image_values=(cast(NiftiArray, np.zeros((2, 2, 2), dtype=np.float32)),),
    )
    output = tmp_path / "out" / "s.json"

    bad_geometry = _geometry_artifact(
        manifest,
        (replace(geometry.case_records[0], image_shape=(3, 2, 2)),),
    )
    bad_lesion = replace(
        lesion,
        geometry_qa_artifact_hash=bad_geometry.qa_artifact_hash,
        lesion_artifact_hash=HASH0,
    )
    bad_lesion = replace(bad_lesion, lesion_artifact_hash=hash_lesion_components(bad_lesion))
    geometry_path.write_text(phase2_artifact_to_json(bad_geometry), encoding="utf-8", newline="\n")
    lesion_path.write_text(phase2_artifact_to_json(bad_lesion), encoding="utf-8", newline="\n")
    with pytest.raises(DevelopmentSummarySourceIntegrityError):
        _run(manifest_path, geometry_path, lesion_path, root, output)

    geometry_path.write_text(phase2_artifact_to_json(geometry), encoding="utf-8", newline="\n")
    lesion_path.write_text(phase2_artifact_to_json(lesion), encoding="utf-8", newline="\n")
    bad_lesion_link = replace(lesion, split_hash="e" * 64, lesion_artifact_hash=HASH0)
    bad_lesion_link = replace(
        bad_lesion_link,
        lesion_artifact_hash=hash_lesion_components(bad_lesion_link),
    )
    lesion_path.write_text(phase2_artifact_to_json(bad_lesion_link), encoding="utf-8", newline="\n")
    with pytest.raises(DevelopmentSummaryLinkageError):
        _run(manifest_path, geometry_path, lesion_path, root, output)

    lesion_path.write_text(phase2_artifact_to_json(lesion), encoding="utf-8", newline="\n")
    outside_image = tmp_path / "outside.nii.gz"
    outside_image.write_bytes(b"not a nifti\n")
    image_path = root / manifest.cases[0].relative_image_path
    image_path.unlink()
    image_path.symlink_to(outside_image)
    with pytest.raises(DevelopmentSummarySourceIntegrityError):
        _run(manifest_path, geometry_path, lesion_path, root, output)

    image_path.unlink()
    write_nifti_file(
        image_path,
        cast(NiftiArray, np.zeros((2, 2, 2), dtype=np.float32)),
        matching_affine,
    )
    geometry_path.write_text(phase2_artifact_to_json(geometry), encoding="utf-8", newline="\n")
    (root / manifest.cases[0].relative_image_path).unlink()
    with pytest.raises(DevelopmentSummarySourceIntegrityError):
        _run(manifest_path, geometry_path, lesion_path, root, output)

    existing = tmp_path / "existing.json"
    existing.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(ExistingDevelopmentSummaryOutputError):
        _run(manifest_path, geometry_path, lesion_path, root, existing)
    with pytest.raises(UnsafeDevelopmentSummaryOutputPathError):
        _run(manifest_path, geometry_path, lesion_path, root, root / "s.json")
    with pytest.raises(UnsafeDevelopmentSummaryOutputPathError):
        _run(manifest_path, geometry_path, lesion_path, root, Path("relative.json"))
    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a dir\n", encoding="utf-8")
    with pytest.raises(UnsafeDevelopmentSummaryOutputPathError):
        _run(manifest_path, geometry_path, lesion_path, root, parent_file / "s.json")

    dupe_root, manifest2, geometry2, lesion2, manifest2_path, geometry2_path, lesion2_path = (
        _fixture(
            tmp_path / "dupe",
            write_nifti_file,
            matching_affine,
            image_values=(
                cast(NiftiArray, np.zeros((2, 2, 2), dtype=np.float32)),
                cast(NiftiArray, np.ones((2, 2, 2), dtype=np.float32)),
            ),
        )
    )
    first_image = dupe_root / manifest2.cases[0].relative_image_path
    second_image = dupe_root / manifest2.cases[1].relative_image_path
    second_image.unlink()
    second_image.symlink_to(first_image)
    with pytest.raises(DevelopmentSummarySourceIntegrityError):
        _run(manifest2_path, geometry2_path, lesion2_path, dupe_root, tmp_path / "dupe-out/s.json")


def test_labels_components_preprocessing_and_cwd_are_not_used(
    tmp_path: Path,
    write_nifti_file: Any,
    matching_affine: AffineArray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest, _geometry, _lesion, manifest_path, geometry_path, lesion_path = _fixture(
        tmp_path,
        write_nifti_file,
        matching_affine,
        image_values=(cast(NiftiArray, np.zeros((2, 2, 2), dtype=np.float32)),),
    )
    label_path = root / manifest.cases[0].relative_label_path
    module_any = cast(Any, summary_module)
    real_sha256_file = module_any.artifact_hashing.sha256_file

    def _guarded_hash(path: Path) -> str:
        assert path != label_path
        return cast(str, real_sha256_file(path))

    real_nib_load = module_any.nib.load

    def _guarded_load(path: str, *args: object, **kwargs: object) -> Any:
        assert Path(path) != label_path
        return real_nib_load(path, *args, **kwargs)

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forbidden downstream code was called")

    monkeypatch.setattr(module_any.artifact_hashing, "sha256_file", _guarded_hash)
    monkeypatch.setattr(module_any.nib, "load", _guarded_load)
    monkeypatch.setattr("protoem_ct.data.lesion_components.scipy_label", _forbidden)
    monkeypatch.setattr("protoem_ct.data.preprocessing.preprocess_synthetic_dataset", _forbidden)
    monkeypatch.chdir(tmp_path)

    first = _run(manifest_path, geometry_path, lesion_path, root, tmp_path / "first" / "s.json")
    second = _run(manifest_path, geometry_path, lesion_path, root, tmp_path / "second" / "s.json")

    assert phase2_artifact_to_json(first) == phase2_artifact_to_json(second)
    json_text = phase2_artifact_to_json(first)
    for forbidden in (str(root), "images/", "labels/", "case-", "source_case_key"):
        assert forbidden not in json_text
