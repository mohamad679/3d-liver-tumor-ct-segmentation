"""Unit tests for Phase 2 deterministic hashing contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_QA_STAGE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    GEOMETRY_LABEL_QA_STAGE,
    LEAKAGE_AUDIT_STAGE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    CaseQaRecord,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentQaArtifact,
    DevelopmentSplitManifest,
    GeometryLabelQaArtifact,
    GeometryLabelQaCaseRecord,
    LeakageAuditArtifact,
    LesionSummaryRecord,
    SplitAssignment,
    dataset_manifest_hash_payload,
    dataset_root_fingerprint_payload,
    development_qa_hash_payload,
    development_split_hash_payload,
    geometry_label_qa_hash_payload,
    hash_dataset_manifest,
    hash_dataset_root_fingerprint,
    hash_development_qa,
    hash_development_split,
    hash_geometry_label_qa,
    hash_leakage_audit,
    leakage_audit_hash_payload,
    sha256_json,
)

HASH0 = "0" * 64
HASH1 = "1" * 64
HASH2 = "2" * 64
HASH3 = "3" * 64
HASH4 = "4" * 64
HASH5 = "5" * 64


def _case(index: int = 1, **overrides: object) -> DatasetCaseRecord:
    record = DatasetCaseRecord(
        anonymous_patient_id=f"anon-p{index:03d}",
        anonymous_case_id=f"anon-c{index:03d}",
        relative_image_path=f"images/anon-c{index:03d}.img",
        relative_label_path=f"labels/anon-c{index:03d}.lbl",
        image_sha256=f"{index:x}" * 64,
        label_sha256=f"{index + 8:x}" * 64,
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
    )
    return cast(DatasetCaseRecord, cast(Any, replace)(record, **overrides))


def _manifest(**overrides: object) -> DatasetManifest:
    cases = (_case(1), _case(2))
    manifest = DatasetManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DATASET_MANIFEST_TYPE,
        dataset_id="lits-development",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
        adapter_name="synthetic-lits-layout",
        adapter_version="1",
        generated_at_utc="2026-07-25T00:00:00Z",
        git_commit="ae38dca",
        dataset_root_fingerprint=HASH0,
        manifest_hash=HASH1,
        case_count=len(cases),
        cases=cases,
    )
    return cast(DatasetManifest, cast(Any, replace)(manifest, **overrides))


def _assignment(index: int, partition: str) -> SplitAssignment:
    return SplitAssignment(
        anonymous_patient_id=f"anon-p{index:03d}",
        anonymous_case_id=f"anon-c{index:03d}",
        partition=partition,
    )


def _split(**overrides: object) -> DevelopmentSplitManifest:
    assignments = (
        _assignment(1, "train"),
        _assignment(2, "validation"),
        _assignment(3, "internal_test"),
    )
    manifest = DevelopmentSplitManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DEVELOPMENT_SPLIT_MANIFEST_TYPE,
        generated_at_utc="2026-07-25T00:00:00Z",
        git_commit="ae38dca",
        source_manifest_hash=HASH1,
        split_policy_version="patient-split-v1",
        split_seed=1729,
        split_hash=HASH2,
        assignments=assignments,
        train_patient_count=1,
        validation_patient_count=1,
        internal_test_patient_count=1,
        train_case_count=1,
        validation_case_count=1,
        internal_test_case_count=1,
    )
    return cast(DevelopmentSplitManifest, cast(Any, replace)(manifest, **overrides))


def _qa_case(index: int = 1, **overrides: object) -> CaseQaRecord:
    lesion = LesionSummaryRecord(lesion_index=1, voxel_count=7, physical_volume_mm3=10.5)
    record = CaseQaRecord(
        anonymous_patient_id=f"anon-p{index:03d}",
        anonymous_case_id=f"anon-c{index:03d}",
        partition="train" if index == 1 else "validation",
        dimensionality=3,
        shape=(4, 5, 6),
        image_dtype="float32",
        label_dtype="uint8",
        affine=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        orientation=("R", "A", "S"),
        spacing=(1.0, 1.0, 2.0),
        image_finite=True,
        label_finite=True,
        allowed_label_values=(0, 1),
        image_label_shape_match=True,
        image_label_affine_match=True,
        intensity_min=-100.0,
        intensity_max=200.0,
        intensity_mean=25.0,
        intensity_std=30.0,
        histogram_bin_edges=(-1000.0, 0.0, 1000.0),
        histogram_counts=(8, 22),
        tumor_voxel_count=7,
        tumor_physical_volume_mm3=10.5,
        lesion_count=1,
        lesions=(lesion,),
        empty_tumor=False,
        qa_passed=True,
        failure_reasons=(),
    )
    return cast(CaseQaRecord, cast(Any, replace)(record, **overrides))


def _qa_artifact(**overrides: object) -> DevelopmentQaArtifact:
    cases = (_qa_case(1), _qa_case(2))
    artifact = DevelopmentQaArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=DEVELOPMENT_QA_STAGE,
        created_at_utc="2026-07-25T00:00:00Z",
        git_commit="ae38dca",
        config_hash=HASH0,
        manifest_hash=HASH1,
        split_hash=HASH2,
        geometry_qa_artifact_hash=HASH3,
        lesion_artifact_hash=HASH4,
        development_summary_artifact_hash=HASH5,
        connectivity=26,
        affine_tolerance=1e-5,
        histogram_bin_edges=(-1000.0, 0.0, 1000.0),
        histogram_bin_count=2,
        case_count=2,
        passed_case_count=2,
        failed_case_count=0,
        case_records=cases,
        aggregate_image_voxel_count=60,
        aggregate_intensity_min=-100.0,
        aggregate_intensity_max=200.0,
        aggregate_intensity_mean=25.0,
        aggregate_intensity_std=30.0,
        aggregate_histogram_counts=(16, 44),
        aggregate_below_histogram_range_count=0,
        aggregate_above_histogram_range_count=0,
        total_tumor_voxel_count=14,
        total_tumor_physical_volume_mm3=21.0,
        total_lesion_count=2,
        lesion_volume_min_mm3=10.5,
        lesion_volume_max_mm3=10.5,
        lesion_volume_mean_mm3=10.5,
        lesion_volume_median_mm3=10.5,
        qa_artifact_hash=HASH3,
    )
    return cast(DevelopmentQaArtifact, cast(Any, replace)(artifact, **overrides))


def _audit(**overrides: object) -> LeakageAuditArtifact:
    artifact = LeakageAuditArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=LEAKAGE_AUDIT_STAGE,
        created_at_utc="2026-07-25T00:00:00Z",
        git_commit="ae38dca",
        config_hash=HASH0,
        manifest_hash=HASH1,
        split_hash=HASH2,
        split_policy_version="patient-split-v1",
        split_seed=1729,
        manifest_case_count=3,
        manifest_patient_count=3,
        assignment_complete=True,
        patient_counts_by_partition={"train": 1, "validation": 1, "internal_test": 1},
        case_counts_by_partition={"train": 1, "validation": 1, "internal_test": 1},
        pairwise_patient_overlap_counts={
            "train__validation": 0,
            "train__internal_test": 0,
            "validation__internal_test": 0,
        },
        pairwise_case_overlap_counts={
            "train__validation": 0,
            "train__internal_test": 0,
            "validation__internal_test": 0,
        },
        image_hash_cross_partition_overlap_count=0,
        label_hash_cross_partition_overlap_count=0,
        image_label_pair_cross_partition_overlap_count=0,
        duplicate_image_hash_findings=(),
        duplicate_label_hash_findings=(),
        finding_codes=(),
        near_duplicate_policy="not-run-until-approved",
        lits_msd_equivalence_warning="LiTS and MSD Task03 Liver are not independent cohorts.",
        external_data_accessed=False,
        preprocessing_fitted_on_nontraining_data=False,
        unresolved_findings=(),
        critical_finding_count=0,
        audit_passed=True,
        audit_hash=HASH4,
    )
    return cast(LeakageAuditArtifact, cast(Any, replace)(artifact, **overrides))


def _geometry_case(index: int = 1, **overrides: object) -> GeometryLabelQaCaseRecord:
    record = GeometryLabelQaCaseRecord(
        anonymous_patient_id=f"anon-p{index:03d}",
        anonymous_case_id=f"anon-c{index:03d}",
        partition="train" if index == 1 else "validation",
        dimensionality=3,
        image_shape=(2, 3, 4),
        label_shape=(2, 3, 4),
        image_dtype="float32",
        label_dtype="uint8",
        image_affine=(
            (1.0, 0.0, 0.0, 4.0),
            (0.0, 1.5, 0.0, 5.0),
            (0.0, 0.0, 2.0, 6.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        label_affine=(
            (1.0, 0.0, 0.0, 4.0),
            (0.0, 1.5, 0.0, 5.0),
            (0.0, 0.0, 2.0, 6.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        image_orientation=("R", "A", "S"),
        label_orientation=("R", "A", "S"),
        image_spacing=(1.0, 1.5, 2.0),
        label_spacing=(1.0, 1.5, 2.0),
        image_finite=True,
        label_finite=True,
        observed_label_values=(0, 1),
        allowed_label_values=(0, 1),
        image_label_shape_match=True,
        image_label_affine_match=True,
        tumor_label_value=1,
        tumor_voxel_count=2,
        empty_tumor=False,
        qa_passed=True,
        failure_reasons=(),
    )
    return cast(GeometryLabelQaCaseRecord, cast(Any, replace)(record, **overrides))


def _geometry_artifact(**overrides: object) -> GeometryLabelQaArtifact:
    cases = (_geometry_case(1), _geometry_case(2))
    artifact = GeometryLabelQaArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=GEOMETRY_LABEL_QA_STAGE,
        created_at_utc="2026-07-25T00:00:00Z",
        git_commit="ae38dca",
        config_hash=HASH0,
        manifest_hash=HASH1,
        split_hash=HASH2,
        case_count=2,
        passed_case_count=2,
        failed_case_count=0,
        case_records=cases,
        qa_artifact_hash=HASH5,
    )
    return cast(GeometryLabelQaArtifact, cast(Any, replace)(artifact, **overrides))


def test_dataset_root_fingerprint_payload_excludes_absolute_root_and_timestamps() -> None:
    payload = dataset_root_fingerprint_payload(
        adapter_name="synthetic-lits-layout",
        adapter_version="1",
        cases=_manifest().cases,
    )

    assert "generated_at_utc" not in payload
    assert "dataset_root" not in payload
    assert "root" not in payload
    assert "/Users/" not in str(payload)
    assert hash_dataset_root_fingerprint(
        adapter_name="synthetic-lits-layout",
        adapter_version="1",
        cases=_manifest().cases,
    ) == sha256_json(payload)


def test_mapping_key_order_does_not_affect_hashes() -> None:
    first = _audit(
        patient_counts_by_partition={"train": 1, "validation": 1, "internal_test": 1},
        case_counts_by_partition={"train": 1, "validation": 1, "internal_test": 1},
    )
    second = _audit(
        patient_counts_by_partition={"internal_test": 1, "validation": 1, "train": 1},
        case_counts_by_partition={"internal_test": 1, "validation": 1, "train": 1},
    )

    assert hash_leakage_audit(first) == hash_leakage_audit(second)


def test_ordered_case_payload_changes_affect_manifest_hash() -> None:
    manifest = _manifest()
    payload = dataset_manifest_hash_payload(manifest)
    reversed_payload = dict(payload)
    cases_value = payload["cases"]
    assert isinstance(cases_value, list)
    reversed_payload["cases"] = list(reversed(cases_value))

    assert sha256_json(payload) != sha256_json(reversed_payload)


def test_own_hash_fields_are_excluded_from_hash_payloads() -> None:
    assert hash_dataset_manifest(_manifest(manifest_hash=HASH1)) == hash_dataset_manifest(
        _manifest(manifest_hash=HASH2)
    )
    assert hash_development_split(_split(split_hash=HASH2)) == hash_development_split(
        _split(split_hash=HASH3)
    )
    assert hash_development_qa(_qa_artifact(qa_artifact_hash=HASH3)) == hash_development_qa(
        _qa_artifact(qa_artifact_hash=HASH4)
    )
    assert hash_geometry_label_qa(
        _geometry_artifact(qa_artifact_hash=HASH3)
    ) == hash_geometry_label_qa(_geometry_artifact(qa_artifact_hash=HASH4))
    assert hash_leakage_audit(_audit(audit_hash=HASH4)) == hash_leakage_audit(
        _audit(audit_hash=HASH5)
    )


def test_hash_payloads_exclude_only_their_own_hash_fields() -> None:
    manifest_payload = dataset_manifest_hash_payload(_manifest())
    split_payload = development_split_hash_payload(_split())
    qa_payload = development_qa_hash_payload(_qa_artifact())
    geometry_payload = geometry_label_qa_hash_payload(_geometry_artifact())
    audit_payload = leakage_audit_hash_payload(_audit())

    assert "manifest_hash" not in manifest_payload
    assert "dataset_root_fingerprint" in manifest_payload
    assert "split_hash" not in split_payload
    assert "source_manifest_hash" in split_payload
    assert "qa_artifact_hash" not in qa_payload
    assert "manifest_hash" in qa_payload
    assert "qa_artifact_hash" not in geometry_payload
    assert "manifest_hash" in geometry_payload
    assert "audit_hash" not in audit_payload
    assert "manifest_hash" in audit_payload


def test_hashing_is_independent_of_current_working_directory(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _manifest()
    before = hash_dataset_manifest(manifest)

    monkeypatch.chdir(tmp_path)

    assert hash_dataset_manifest(manifest) == before


def test_root_fingerprint_order_matters_for_ordered_case_records() -> None:
    manifest = _manifest()

    forward = hash_dataset_root_fingerprint(
        adapter_name=manifest.adapter_name,
        adapter_version=manifest.adapter_version,
        cases=manifest.cases,
    )
    reversed_hash = hash_dataset_root_fingerprint(
        adapter_name=manifest.adapter_name,
        adapter_version=manifest.adapter_version,
        cases=tuple(reversed(manifest.cases)),
    )

    assert forward != reversed_hash
