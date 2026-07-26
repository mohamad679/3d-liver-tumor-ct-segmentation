"""Unit tests for Phase 2 artifact schema contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import pytest

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
    Phase2ArtifactSchemaVersionError,
    Phase2ArtifactSerializationError,
    Phase2ArtifactStageError,
    Phase2ArtifactValidationError,
    SplitAssignment,
    phase2_artifact_from_json,
    phase2_artifact_to_dict,
    phase2_artifact_to_json,
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


def _lesion(index: int = 1, **overrides: object) -> LesionSummaryRecord:
    lesion = LesionSummaryRecord(
        lesion_index=index,
        voxel_count=7,
        physical_volume_mm3=10.5,
    )
    return cast(LesionSummaryRecord, cast(Any, replace)(lesion, **overrides))


def _qa_case(index: int = 1, **overrides: object) -> CaseQaRecord:
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
        lesions=(_lesion(1),),
        empty_tumor=False,
        qa_passed=True,
        failure_reasons=(),
    )
    return cast(CaseQaRecord, cast(Any, replace)(record, **overrides))


def _qa_artifact(**overrides: object) -> DevelopmentQaArtifact:
    cases = (
        _qa_case(1),
        _qa_case(2, qa_passed=False, failure_reasons=("shape_mismatch",)),
    )
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
        case_count=len(cases),
        passed_case_count=1,
        failed_case_count=1,
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
        manifest_hash=HASH1,
        split_hash=HASH2,
        split_policy_version="patient-split-v1",
        split_seed=1729,
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
        duplicate_image_hash_findings=(),
        duplicate_label_hash_findings=(),
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
    cases = (
        _geometry_case(1),
        _geometry_case(
            2,
            qa_passed=False,
            failure_reasons=("label_value_not_allowed",),
        ),
    )
    artifact = GeometryLabelQaArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=GEOMETRY_LABEL_QA_STAGE,
        created_at_utc="2026-07-25T00:00:00Z",
        git_commit="ae38dca",
        config_hash=HASH0,
        manifest_hash=HASH1,
        split_hash=HASH2,
        case_count=2,
        passed_case_count=1,
        failed_case_count=1,
        case_records=cases,
        qa_artifact_hash=HASH5,
    )
    return cast(GeometryLabelQaArtifact, cast(Any, replace)(artifact, **overrides))


@pytest.mark.parametrize(
    ("artifact", "artifact_type"),
    [
        (_case(), DatasetCaseRecord),
        (_manifest(), DatasetManifest),
        (_assignment(1, "train"), SplitAssignment),
        (_split(), DevelopmentSplitManifest),
        (_lesion(), LesionSummaryRecord),
        (_qa_case(), CaseQaRecord),
        (_qa_artifact(), DevelopmentQaArtifact),
        (_geometry_case(), GeometryLabelQaCaseRecord),
        (_geometry_artifact(), GeometryLabelQaArtifact),
        (_audit(), LeakageAuditArtifact),
    ],
)
def test_phase2_artifacts_round_trip_with_tuple_types(
    artifact: object,
    artifact_type: type[object],
) -> None:
    json_text = phase2_artifact_to_json(cast(Any, artifact))

    loaded = phase2_artifact_from_json(json_text, cast(Any, artifact_type))

    assert loaded == artifact


def test_phase2_json_is_byte_identical_sorted_and_newline_terminated() -> None:
    manifest = _manifest()

    first = phase2_artifact_to_json(manifest)
    second = phase2_artifact_to_json(manifest)

    assert first == second
    assert first.endswith("\n")
    assert list(json.loads(first)) == sorted(json.loads(first))


def test_phase2_to_dict_uses_json_lists() -> None:
    value = phase2_artifact_to_dict(_qa_case())

    assert value["shape"] == [4, 5, 6]
    assert value["histogram_counts"] == [8, 22]
    assert value["lesions"] == [{"lesion_index": 1, "physical_volume_mm3": 10.5, "voxel_count": 7}]


def test_absolute_paths_and_parent_traversal_are_rejected() -> None:
    with pytest.raises(Phase2ArtifactValidationError):
        _case(relative_image_path="/private/root/images/anon-c001.img")

    with pytest.raises(Phase2ArtifactValidationError):
        _case(relative_image_path="../images/anon-c001.img")

    with pytest.raises(Phase2ArtifactValidationError):
        _case(relative_image_path="images/../anon-c001.img")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("anonymous_patient_id", "John-Doe"),
        ("anonymous_patient_id", "mrn-12345"),
        ("anonymous_case_id", "anon/001"),
        ("anonymous_case_id", "1970-01-01"),
    ],
)
def test_obvious_identifier_like_ids_are_rejected(field_name: str, value: object) -> None:
    with pytest.raises(Phase2ArtifactValidationError):
        cast(Any, _case)(**{field_name: value})


def test_malformed_sha256_and_wrong_cohort_role_are_rejected() -> None:
    with pytest.raises(Phase2ArtifactValidationError):
        _case(image_sha256="ABC")

    with pytest.raises(Phase2ArtifactValidationError):
        _case(cohort_role="external")


def test_dataset_manifest_rejects_duplicates_and_count_mismatches() -> None:
    with pytest.raises(Phase2ArtifactValidationError, match="anonymous_patient_id"):
        _manifest(cases=(_case(1), _case(2, anonymous_patient_id="anon-p001")))

    with pytest.raises(Phase2ArtifactValidationError, match="anonymous_case_id"):
        _manifest(cases=(_case(1), _case(2, anonymous_case_id="anon-c001")))

    with pytest.raises(Phase2ArtifactValidationError, match="image paths"):
        _manifest(cases=(_case(1), _case(2, relative_image_path="images/anon-c001.img")))

    with pytest.raises(Phase2ArtifactValidationError, match="label paths"):
        _manifest(cases=(_case(1), _case(2, relative_label_path="labels/anon-c001.lbl")))

    with pytest.raises(Phase2ArtifactValidationError, match="image hashes"):
        _manifest(cases=(_case(1), _case(2, image_sha256="1" * 64)))

    with pytest.raises(Phase2ArtifactValidationError, match="case_count"):
        _manifest(case_count=3)


def test_dataset_manifest_rejects_unsorted_cases_and_mismatched_case_role() -> None:
    with pytest.raises(Phase2ArtifactValidationError, match="ordered"):
        _manifest(cases=(_case(2), _case(1)))

    bad_case = object.__new__(DatasetCaseRecord)
    for key, value in phase2_artifact_to_dict(_case(2)).items():
        object.__setattr__(bad_case, key, value)
    object.__setattr__(bad_case, "cohort_role", "external")
    with pytest.raises(Phase2ArtifactValidationError, match="cohort"):
        _manifest(cases=(_case(1), bad_case))


def test_split_rejects_invalid_partition_patient_overlap_counts_and_order() -> None:
    with pytest.raises(Phase2ArtifactValidationError, match="partition"):
        SplitAssignment("anon-p001", "anon-c001", "test")

    with pytest.raises(Phase2ArtifactValidationError, match="anonymous_patient_id"):
        _split(assignments=(_assignment(1, "train"), _assignment(1, "validation")))

    with pytest.raises(Phase2ArtifactValidationError, match="train_patient_count"):
        _split(train_patient_count=2)

    with pytest.raises(Phase2ArtifactValidationError, match="ordered"):
        _split(
            assignments=(
                _assignment(2, "validation"),
                _assignment(1, "train"),
                _assignment(3, "internal_test"),
            )
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("shape", (4, 5)),
        ("shape", (4, 0, 6)),
        ("spacing", (1.0, 0.0, 2.0)),
        ("affine", ((1.0, 0.0),)),
        ("orientation", ("R", "L", "S")),
        ("histogram_bin_edges", (-1000.0, 0.0)),
        ("histogram_counts", (8, -1)),
        ("lesions", (_lesion(2),)),
    ],
)
def test_case_qa_rejects_invalid_geometry_histogram_and_lesions(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(Phase2ArtifactValidationError):
        cast(Any, _qa_case)(**{field_name: value})


def test_case_qa_rejects_empty_tumor_and_failure_reason_inconsistency() -> None:
    with pytest.raises(Phase2ArtifactValidationError, match="empty_tumor"):
        _qa_case(empty_tumor=True)

    with pytest.raises(Phase2ArtifactValidationError, match="empty_tumor"):
        _qa_case(
            empty_tumor=False,
            tumor_voxel_count=0,
            tumor_physical_volume_mm3=0.0,
            lesion_count=0,
            lesions=(),
        )

    with pytest.raises(Phase2ArtifactValidationError, match="failure_reasons"):
        _qa_case(qa_passed=True, failure_reasons=("unexpected_label",))

    with pytest.raises(Phase2ArtifactValidationError, match="failure reason"):
        _qa_case(qa_passed=False, failure_reasons=())


def test_development_qa_rejects_aggregate_mismatches_and_unsorted_cases() -> None:
    with pytest.raises(Phase2ArtifactValidationError, match="case_count"):
        _qa_artifact(case_count=3)

    with pytest.raises(Phase2ArtifactValidationError, match="passed_case_count"):
        _qa_artifact(passed_case_count=2)

    with pytest.raises(Phase2ArtifactValidationError, match="histogram_bin_edges"):
        _qa_artifact(
            case_count=1,
            passed_case_count=1,
            failed_case_count=0,
            case_records=(_qa_case(1, histogram_bin_edges=(-1.0, 1.0, 2.0)),),
        )

    with pytest.raises(Phase2ArtifactValidationError, match="ordered"):
        _qa_artifact(
            case_records=(
                _qa_case(2, qa_passed=False, failure_reasons=("shape_mismatch",)),
                _qa_case(1),
            )
        )


def test_geometry_label_qa_schema_rejects_invalid_contract_values() -> None:
    with pytest.raises(Phase2ArtifactValidationError, match="allowed_label_values"):
        _geometry_case(allowed_label_values=(1, 0), tumor_label_value=1)

    with pytest.raises(Phase2ArtifactValidationError, match="tumor_label_value"):
        _geometry_case(tumor_label_value=2)

    with pytest.raises(Phase2ArtifactValidationError, match="empty_tumor"):
        _geometry_case(empty_tumor=True, tumor_voxel_count=1)

    with pytest.raises(Phase2ArtifactValidationError, match="failure_reasons"):
        _geometry_case(qa_passed=True, failure_reasons=("image_nonfinite",))

    with pytest.raises(Phase2ArtifactValidationError, match="deterministic"):
        _geometry_case(
            qa_passed=False,
            failure_reasons=("shape_mismatch", "image_nonfinite"),
        )

    with pytest.raises(Phase2ArtifactStageError, match="stage"):
        _geometry_artifact(stage=DEVELOPMENT_QA_STAGE)

    with pytest.raises(Phase2ArtifactValidationError, match="case_count"):
        _geometry_artifact(case_count=3)


def test_leakage_audit_cannot_pass_with_overlap_external_access_or_preprocessing_leak() -> None:
    with pytest.raises(Phase2ArtifactValidationError, match="overlap"):
        _audit(
            pairwise_patient_overlap_counts={
                "train__validation": 1,
                "train__internal_test": 0,
                "validation__internal_test": 0,
            }
        )

    with pytest.raises(Phase2ArtifactValidationError, match="external_data_accessed"):
        _audit(external_data_accessed=True)

    with pytest.raises(Phase2ArtifactValidationError, match="preprocessing_fitted"):
        _audit(preprocessing_fitted_on_nontraining_data=True)

    with pytest.raises(Phase2ArtifactValidationError, match="critical_finding_count"):
        _audit(critical_finding_count=1)

    with pytest.raises(Phase2ArtifactValidationError, match="LiTS and MSD"):
        _audit(lits_msd_equivalence_warning="warning omitted")


def test_leakage_audit_requires_all_partition_and_pairwise_keys() -> None:
    with pytest.raises(Phase2ArtifactValidationError, match="patient_counts_by_partition"):
        _audit(patient_counts_by_partition={"train": 1, "validation": 1})

    with pytest.raises(Phase2ArtifactValidationError, match="pairwise_case_overlap_counts"):
        _audit(pairwise_case_overlap_counts={"train__validation": 0})


def test_unknown_fields_incompatible_schema_and_wrong_type_or_stage_are_rejected() -> None:
    manifest_dict = phase2_artifact_to_dict(_manifest())
    manifest_dict["unexpected"] = True
    with pytest.raises(Phase2ArtifactValidationError):
        phase2_artifact_from_json(json.dumps(manifest_dict), DatasetManifest)

    manifest_dict = phase2_artifact_to_dict(_manifest())
    manifest_dict["schema_version"] = "99"
    with pytest.raises(Phase2ArtifactSchemaVersionError):
        phase2_artifact_from_json(json.dumps(manifest_dict), DatasetManifest)

    manifest_dict = phase2_artifact_to_dict(_manifest())
    manifest_dict["manifest_type"] = DEVELOPMENT_SPLIT_MANIFEST_TYPE
    with pytest.raises(Phase2ArtifactStageError):
        phase2_artifact_from_json(json.dumps(manifest_dict), DatasetManifest)

    qa_dict = phase2_artifact_to_dict(_qa_artifact())
    qa_dict["stage"] = LEAKAGE_AUDIT_STAGE
    with pytest.raises(Phase2ArtifactStageError):
        phase2_artifact_from_json(json.dumps(qa_dict), DevelopmentQaArtifact)


@pytest.mark.parametrize("bad_constant", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_json_are_rejected(bad_constant: str) -> None:
    json_text = phase2_artifact_to_json(_qa_case()).replace("-100.0", bad_constant, 1)

    with pytest.raises(Phase2ArtifactSerializationError):
        phase2_artifact_from_json(json_text, CaseQaRecord)


def test_serialized_synthetic_metadata_contains_no_local_root_or_medical_identifier() -> None:
    json_text = phase2_artifact_to_json(_manifest())

    forbidden_fragments = (
        "/Users/",
        "/private/",
        "mohsen",
        "dataset-root",
        "data_raw",
        "accession",
        "dicom",
        "mrn",
        "patientname",
    )
    for fragment in forbidden_fragments:
        assert fragment not in json_text.lower()
