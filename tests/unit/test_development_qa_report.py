"""Unit tests for final Phase 2 development QA JSON report assembly."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest

import protoem_ct.artifacts.hashing as artifact_hashing
import protoem_ct.data.development_summary as summary_module
import protoem_ct.data.lesion_components as lesion_module
from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_DATA_SUMMARY_STAGE,
    DEVELOPMENT_DATA_SUMMARY_UPSTREAM_FAILURE_REASON,
    DEVELOPMENT_QA_STAGE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    GEOMETRY_LABEL_QA_STAGE,
    LESION_COMPONENTS_STAGE,
    LESION_COMPONENTS_UPSTREAM_FAILURE_REASON,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    CtHistogramCaseRecord,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentDataSummaryArtifact,
    DevelopmentQaArtifact,
    DevelopmentSplitManifest,
    GeometryLabelQaArtifact,
    GeometryLabelQaCaseRecord,
    LesionComponentCaseRecord,
    LesionComponentsArtifact,
    LesionSummaryRecord,
    Phase2ArtifactSerializationError,
    SplitAssignment,
    development_qa_hash_payload,
    hash_dataset_manifest,
    hash_development_data_summary,
    hash_development_qa,
    hash_development_split,
    hash_geometry_label_qa,
    hash_lesion_components,
    phase2_artifact_from_json,
    phase2_artifact_to_json,
)
from protoem_ct.data.development_qa_report import (
    DEVELOPMENT_QA_REPORT_CONTRACT_VERSION,
    DevelopmentQaReportCaseAssemblyError,
    DevelopmentQaReportConfig,
    DevelopmentQaReportHashMismatchError,
    DevelopmentQaReportLinkageError,
    DevelopmentQaReportPublicationError,
    ExistingDevelopmentQaReportOutputError,
    InvalidDevelopmentQaReportConfigError,
    InvalidDevelopmentQaReportInputError,
    UnsafeDevelopmentQaReportOutputPathError,
    assemble_development_qa_report,
    hash_development_qa_report_config,
)

CREATED_AT_UTC = "2026-07-26T00:00:00Z"
GIT_COMMIT = "c64bac3"
HASH0 = "0" * 64
HASH1 = "1" * 64
HASH2 = "2" * 64
HASH3 = "3" * 64
HASH4 = "4" * 64
HASH5 = "5" * 64


def _config() -> DevelopmentQaReportConfig:
    return DevelopmentQaReportConfig(report_contract_version=DEVELOPMENT_QA_REPORT_CONTRACT_VERSION)


def _case(index: int) -> DatasetCaseRecord:
    return DatasetCaseRecord(
        anonymous_patient_id=f"anon-p{index:04d}",
        anonymous_case_id=f"anon-c{index:04d}",
        relative_image_path=f"images/case-{index:04d}.nii.gz",
        relative_label_path=f"labels/case-{index:04d}.nii.gz",
        image_sha256=f"{index:x}" * 64,
        label_sha256=f"{index + 8:x}" * 64,
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
    )


def _manifest(cases: tuple[DatasetCaseRecord, ...] | None = None) -> DatasetManifest:
    case_records = cases or (_case(1), _case(2), _case(3))
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
        case_count=len(case_records),
        cases=case_records,
    )
    return replace(without_hash, manifest_hash=hash_dataset_manifest(without_hash))


def _split(manifest: DatasetManifest) -> DevelopmentSplitManifest:
    partitions = ("train", "validation", "internal_test")
    assignments = tuple(
        SplitAssignment(
            anonymous_patient_id=case.anonymous_patient_id,
            anonymous_case_id=case.anonymous_case_id,
            partition=partitions[index],
        )
        for index, case in enumerate(manifest.cases)
    )
    without_hash = DevelopmentSplitManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DEVELOPMENT_SPLIT_MANIFEST_TYPE,
        generated_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        source_manifest_hash=manifest.manifest_hash,
        split_policy_version="patient_hash_rank_v1",
        split_seed=1729,
        split_hash=HASH0,
        assignments=assignments,
        train_patient_count=1,
        validation_patient_count=1,
        internal_test_patient_count=1,
        train_case_count=1,
        validation_case_count=1,
        internal_test_case_count=1,
    )
    return replace(without_hash, split_hash=hash_development_split(without_hash))


def _affine() -> tuple[tuple[float, float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.5, 0.0, 0.0),
        (0.0, 0.0, 2.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _geometry_case(
    case: DatasetCaseRecord,
    partition: str,
    *,
    passed: bool,
) -> GeometryLabelQaCaseRecord:
    return GeometryLabelQaCaseRecord(
        anonymous_patient_id=case.anonymous_patient_id,
        anonymous_case_id=case.anonymous_case_id,
        partition=partition,
        dimensionality=3,
        image_shape=(2, 2, 2),
        label_shape=(2, 2, 2),
        image_dtype="float32",
        label_dtype="uint8",
        image_affine=_affine(),
        label_affine=_affine(),
        image_orientation=("R", "A", "S"),
        label_orientation=("R", "A", "S"),
        image_spacing=(1.0, 1.5, 2.0),
        label_spacing=(1.0, 1.5, 2.0),
        image_finite=True,
        label_finite=passed,
        observed_label_values=(0, 1) if passed else (0, 2),
        allowed_label_values=(0, 1),
        image_label_shape_match=True,
        image_label_affine_match=True,
        tumor_label_value=1,
        tumor_voxel_count=1 if passed and case.anonymous_case_id.endswith("1") else 0,
        empty_tumor=not (passed and case.anonymous_case_id.endswith("1")),
        qa_passed=passed,
        failure_reasons=() if passed else ("label_value_not_allowed",),
    )


def _geometry(
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
) -> GeometryLabelQaArtifact:
    assignment_by_case = {item.anonymous_case_id: item for item in split.assignments}
    cases = tuple(
        _geometry_case(
            case,
            assignment_by_case[case.anonymous_case_id].partition,
            passed=case.anonymous_case_id != "anon-c0002",
        )
        for case in manifest.cases
    )
    without_hash = GeometryLabelQaArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=GEOMETRY_LABEL_QA_STAGE,
        created_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        config_hash=HASH1,
        manifest_hash=manifest.manifest_hash,
        split_hash=split.split_hash,
        case_count=len(cases),
        passed_case_count=2,
        failed_case_count=1,
        case_records=cases,
        qa_artifact_hash=HASH0,
    )
    return replace(without_hash, qa_artifact_hash=hash_geometry_label_qa(without_hash))


def _lesion_case(geometry_case: GeometryLabelQaCaseRecord) -> LesionComponentCaseRecord:
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
    lesions = (LesionSummaryRecord(1, 1, 3.0),) if geometry_case.tumor_voxel_count == 1 else ()
    return LesionComponentCaseRecord(
        anonymous_patient_id=geometry_case.anonymous_patient_id,
        anonymous_case_id=geometry_case.anonymous_case_id,
        partition=geometry_case.partition,
        analysis_performed=True,
        connectivity=26,
        tumor_label_value=1,
        voxel_volume_mm3=3.0,
        tumor_voxel_count=geometry_case.tumor_voxel_count,
        tumor_physical_volume_mm3=float(geometry_case.tumor_voxel_count * 3.0),
        lesion_count=len(lesions),
        lesions=lesions,
        qa_passed=True,
        failure_reasons=(),
    )


def _lesion(
    manifest: DatasetManifest,
    geometry: GeometryLabelQaArtifact,
) -> LesionComponentsArtifact:
    records = tuple(_lesion_case(case) for case in geometry.case_records)
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
        analyzed_case_count=2,
        skipped_case_count=1,
        case_records=records,
        lesion_artifact_hash=HASH0,
    )
    return replace(without_hash, lesion_artifact_hash=hash_lesion_components(without_hash))


def _summary_case(geometry_case: GeometryLabelQaCaseRecord) -> CtHistogramCaseRecord:
    if not geometry_case.qa_passed:
        return CtHistogramCaseRecord(
            anonymous_patient_id=geometry_case.anonymous_patient_id,
            anonymous_case_id=geometry_case.anonymous_case_id,
            partition=geometry_case.partition,
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
    return CtHistogramCaseRecord(
        anonymous_patient_id=geometry_case.anonymous_patient_id,
        anonymous_case_id=geometry_case.anonymous_case_id,
        partition=geometry_case.partition,
        analysis_performed=True,
        image_voxel_count=2,
        intensity_min=0.0,
        intensity_max=1.0,
        intensity_mean=0.5,
        intensity_std=0.5,
        histogram_counts=(1, 1),
        below_histogram_range_count=0,
        above_histogram_range_count=0,
        qa_passed=True,
        failure_reasons=(),
    )


def _summary(
    manifest: DatasetManifest,
    geometry: GeometryLabelQaArtifact,
    lesion: LesionComponentsArtifact,
) -> DevelopmentDataSummaryArtifact:
    records = tuple(_summary_case(case) for case in geometry.case_records)
    without_hash = DevelopmentDataSummaryArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=DEVELOPMENT_DATA_SUMMARY_STAGE,
        created_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        config_hash=HASH3,
        manifest_hash=manifest.manifest_hash,
        split_hash=geometry.split_hash,
        geometry_qa_artifact_hash=geometry.qa_artifact_hash,
        lesion_artifact_hash=lesion.lesion_artifact_hash,
        histogram_bin_edges=(0.0, 0.5, 1.0),
        histogram_bin_count=2,
        case_count=3,
        analyzed_case_count=2,
        skipped_case_count=1,
        case_records=records,
        aggregate_image_voxel_count=4,
        aggregate_intensity_min=0.0,
        aggregate_intensity_max=1.0,
        aggregate_intensity_mean=0.5,
        aggregate_intensity_std=0.5,
        aggregate_histogram_counts=(2, 2),
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
    return replace(without_hash, summary_artifact_hash=hash_development_data_summary(without_hash))


def _chain() -> tuple[
    DatasetManifest,
    DevelopmentSplitManifest,
    GeometryLabelQaArtifact,
    LesionComponentsArtifact,
    DevelopmentDataSummaryArtifact,
]:
    manifest = _manifest()
    split = _split(manifest)
    geometry = _geometry(manifest, split)
    lesion = _lesion(manifest, geometry)
    summary = _summary(manifest, geometry, lesion)
    return manifest, split, geometry, lesion, summary


def _write_chain(
    tmp_path: Path,
    artifacts: tuple[
        DatasetManifest,
        DevelopmentSplitManifest,
        GeometryLabelQaArtifact,
        LesionComponentsArtifact,
        DevelopmentDataSummaryArtifact,
    ]
    | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    manifest, split, geometry, lesion, summary = artifacts or _chain()
    root = tmp_path / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    paths = (
        root / "manifest.json",
        root / "split.json",
        root / "geometry.json",
        root / "lesion.json",
        root / "summary.json",
    )
    for path, artifact in zip(paths, (manifest, split, geometry, lesion, summary), strict=True):
        path.write_text(
            phase2_artifact_to_json(cast(Any, artifact)),
            encoding="utf-8",
            newline="\n",
        )
    return paths


def _assemble(paths: tuple[Path, Path, Path, Path, Path], output: Path) -> DevelopmentQaArtifact:
    return assemble_development_qa_report(
        paths[0],
        paths[1],
        paths[2],
        paths[3],
        paths[4],
        output_path=output,
        config=_config(),
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )


def test_config_validation_hash_and_immutability() -> None:
    config = _config()
    assert hash_development_qa_report_config(config) == hash_development_qa_report_config(_config())
    with pytest.raises(InvalidDevelopmentQaReportConfigError):
        DevelopmentQaReportConfig(report_contract_version="other")
    with pytest.raises(FrozenInstanceError):
        cast(Any, config).report_contract_version = "other"


def test_valid_chain_assembles_final_report_and_round_trips(tmp_path: Path) -> None:
    manifest, split, geometry, lesion, summary = _chain()
    paths = _write_chain(tmp_path, (manifest, split, geometry, lesion, summary))
    output = tmp_path / "out" / "development_qa.json"

    artifact = _assemble(paths, output)

    assert output.exists()
    assert artifact.stage == DEVELOPMENT_QA_STAGE
    assert artifact.case_count == 3
    assert artifact.passed_case_count == 2
    assert artifact.failed_case_count == 1
    assert artifact.manifest_hash == manifest.manifest_hash
    assert artifact.split_hash == split.split_hash
    assert artifact.geometry_qa_artifact_hash == geometry.qa_artifact_hash
    assert artifact.lesion_artifact_hash == lesion.lesion_artifact_hash
    assert artifact.development_summary_artifact_hash == summary.summary_artifact_hash
    assert artifact.aggregate_histogram_counts == summary.aggregate_histogram_counts
    assert artifact.total_lesion_count == summary.total_lesion_count
    assert artifact.qa_artifact_hash == hash_development_qa(artifact)
    assert "qa_artifact_hash" not in development_qa_hash_payload(artifact)
    assert (
        phase2_artifact_from_json(output.read_text(encoding="utf-8"), DevelopmentQaArtifact)
        == artifact
    )


def test_case_assembly_preserves_passing_and_upstream_failed_cases(tmp_path: Path) -> None:
    paths = _write_chain(tmp_path)

    artifact = _assemble(paths, tmp_path / "out" / "development_qa.json")

    passing = artifact.case_records[0]
    failed = artifact.case_records[1]
    assert passing.qa_passed
    assert passing.partition == "train"
    assert passing.intensity_mean == 0.5
    assert passing.lesion_count == 1
    assert failed.partition == "validation"
    assert not failed.qa_passed
    assert failed.failure_reasons == ("label_value_not_allowed",)
    assert failed.intensity_min is None
    assert failed.histogram_counts == ()
    assert failed.tumor_physical_volume_mm3 is None
    assert failed.lesion_count is None
    assert failed.lesions == ()


@pytest.mark.parametrize("index", range(5))
def test_input_hash_mismatches_are_rejected(tmp_path: Path, index: int) -> None:
    artifacts: list[Any] = list(_chain())
    hash_fields = (
        "manifest_hash",
        "split_hash",
        "qa_artifact_hash",
        "lesion_artifact_hash",
        "summary_artifact_hash",
    )
    artifacts[index] = replace(artifacts[index], **{hash_fields[index]: "f" * 64})
    paths = _write_chain(tmp_path, cast(Any, tuple(artifacts)))

    with pytest.raises(DevelopmentQaReportHashMismatchError):
        _assemble(paths, tmp_path / "out" / "development_qa.json")


def test_malformed_and_wrong_type_inputs_are_rejected(tmp_path: Path) -> None:
    paths = list(_write_chain(tmp_path))
    paths[2].write_text("{not json\n", encoding="utf-8")
    with pytest.raises(InvalidDevelopmentQaReportInputError):
        _assemble(cast(Any, tuple(paths)), tmp_path / "out" / "development_qa.json")

    paths = list(_write_chain(tmp_path / "wrong"))
    paths[1].write_text(paths[0].read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(InvalidDevelopmentQaReportInputError):
        _assemble(cast(Any, tuple(paths)), tmp_path / "wrong-out" / "development_qa.json")


def test_linkage_and_case_state_mismatches_are_rejected(tmp_path: Path) -> None:
    manifest, split, geometry, lesion, summary = _chain()
    bad_split = replace(split, source_manifest_hash="f" * 64, split_hash=HASH0)
    bad_split = replace(bad_split, split_hash=hash_development_split(bad_split))
    paths = _write_chain(tmp_path, (manifest, bad_split, geometry, lesion, summary))
    with pytest.raises(DevelopmentQaReportLinkageError):
        _assemble(paths, tmp_path / "out" / "development_qa.json")

    bad_lesion = replace(lesion, geometry_qa_artifact_hash="f" * 64, lesion_artifact_hash=HASH0)
    bad_lesion = replace(bad_lesion, lesion_artifact_hash=hash_lesion_components(bad_lesion))
    paths = _write_chain(tmp_path / "lesion", (manifest, split, geometry, bad_lesion, summary))
    with pytest.raises(DevelopmentQaReportLinkageError):
        _assemble(paths, tmp_path / "lesion-out" / "development_qa.json")

    inconsistent = replace(
        lesion.case_records[0],
        analysis_performed=False,
        voxel_volume_mm3=None,
        tumor_voxel_count=None,
        tumor_physical_volume_mm3=None,
        lesion_count=None,
        lesions=(),
        qa_passed=False,
        failure_reasons=(LESION_COMPONENTS_UPSTREAM_FAILURE_REASON,),
    )
    inconsistent_lesion = replace(
        lesion,
        analyzed_case_count=1,
        skipped_case_count=2,
        case_records=(inconsistent, lesion.case_records[1], lesion.case_records[2]),
        lesion_artifact_hash=HASH0,
    )
    inconsistent_lesion = replace(
        inconsistent_lesion,
        lesion_artifact_hash=hash_lesion_components(inconsistent_lesion),
    )
    inconsistent_summary = replace(
        summary,
        lesion_artifact_hash=inconsistent_lesion.lesion_artifact_hash,
        summary_artifact_hash=HASH0,
    )
    inconsistent_summary = replace(
        inconsistent_summary,
        summary_artifact_hash=hash_development_data_summary(inconsistent_summary),
    )
    paths = _write_chain(
        tmp_path / "state",
        (manifest, split, geometry, inconsistent_lesion, inconsistent_summary),
    )
    with pytest.raises(DevelopmentQaReportCaseAssemblyError):
        _assemble(paths, tmp_path / "state-out" / "development_qa.json")


def test_output_safety_and_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_chain(tmp_path)
    with pytest.raises(UnsafeDevelopmentQaReportOutputPathError):
        _assemble(paths, Path("relative.json"))
    with pytest.raises(UnsafeDevelopmentQaReportOutputPathError):
        _assemble(paths, tmp_path / "out" / "report.md")
    with pytest.raises(UnsafeDevelopmentQaReportOutputPathError):
        _assemble(paths, paths[0])
    existing = tmp_path / "existing.json"
    existing.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(ExistingDevelopmentQaReportOutputError):
        _assemble(paths, existing)
    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(UnsafeDevelopmentQaReportOutputPathError):
        _assemble(paths, parent_file / "report.json")

    def _fail_serialize(_artifact: object) -> str:
        raise Phase2ArtifactSerializationError("synthetic serialization failure")

    monkeypatch.setattr(
        "protoem_ct.data.development_qa_report.phase2_artifact_to_json",
        _fail_serialize,
    )
    failed_output = tmp_path / "failed" / "report.json"
    with pytest.raises(DevelopmentQaReportPublicationError):
        _assemble(paths, failed_output)
    assert not failed_output.exists()
    assert not (failed_output.parent / ".report.json.tmp").exists()


def test_byte_identical_output_no_paths_and_no_medical_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_chain(tmp_path)

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forbidden medical-data operation was called")

    monkeypatch.setattr(artifact_hashing, "sha256_file", _forbidden)
    monkeypatch.setattr(summary_module, "run_development_data_summary", _forbidden)
    monkeypatch.setattr(lesion_module, "scipy_label", _forbidden)
    monkeypatch.chdir(tmp_path)

    first = _assemble(paths, tmp_path / "first" / "report.json")
    second = _assemble(paths, tmp_path / "second" / "report.json")

    assert phase2_artifact_to_json(first) == phase2_artifact_to_json(second)
    text = phase2_artifact_to_json(first)
    for forbidden in (
        "images/",
        "labels/",
        "case-",
        "source_case_key",
        str(tmp_path),
        "mohsen",
        "dataset",
    ):
        assert forbidden not in text


def test_timestamp_changes_json_and_hash(tmp_path: Path) -> None:
    paths = _write_chain(tmp_path)
    first = _assemble(paths, tmp_path / "first" / "report.json")
    second = assemble_development_qa_report(
        paths[0],
        paths[1],
        paths[2],
        paths[3],
        paths[4],
        output_path=tmp_path / "second" / "report.json",
        config=_config(),
        git_commit=GIT_COMMIT,
        created_at_utc="2026-07-26T01:00:00Z",
    )

    assert phase2_artifact_to_json(first) != phase2_artifact_to_json(second)
    assert first.qa_artifact_hash != second.qa_artifact_hash
