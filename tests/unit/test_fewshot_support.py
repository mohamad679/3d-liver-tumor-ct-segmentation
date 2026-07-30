"""Unit tests for deterministic Phase 4 support selection and stratification."""

from __future__ import annotations

from dataclasses import replace

import pytest

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    LESION_COMPONENTS_STAGE,
    LESION_COMPONENTS_UPSTREAM_FAILURE_REASON,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentSplitManifest,
    LesionComponentCaseRecord,
    LesionComponentsArtifact,
    LesionSummaryRecord,
    SplitAssignment,
    hash_dataset_manifest,
    hash_development_split,
    hash_lesion_components,
)
from protoem_ct.fewshot import (
    BUCKET_ORDER,
    DEFAULT_REPLICATE_COUNT,
    FewshotSupportInputError,
    FewshotSupportInsufficientCandidatesError,
    FewshotSupportMismatchError,
    SupportCandidate,
    ValidatedSupportSelectionInputs,
    assign_burden_buckets,
    fewshot_support_manifest_to_json,
    generate_complete_fixed_support_manifest_set,
    generate_support_manifests_for_k,
    validate_support_selection_inputs,
)

HASH0 = "0" * 64
GIT_COMMIT = "0" * 40
CREATED_AT_UTC = "2026-07-30T00:00:00Z"


def _case(index: int) -> DatasetCaseRecord:
    return DatasetCaseRecord(
        anonymous_patient_id=f"anon-p{index:04d}",
        anonymous_case_id=f"anon-c{index:04d}",
        relative_image_path=f"images/case-{index:04d}.nii.gz",
        relative_label_path=f"labels/case-{index:04d}.nii.gz",
        image_sha256=f"{index:064x}",
        label_sha256=f"{index + 500:064x}",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
    )


def _manifest(case_count: int = 30) -> DatasetManifest:
    cases = tuple(_case(index) for index in range(1, case_count + 1))
    without_hash = DatasetManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DATASET_MANIFEST_TYPE,
        dataset_id="synthetic-dev",
        cohort_role=PHASE2_DEVELOPMENT_COHORT_ROLE,
        adapter_name="lits_style",
        adapter_version="1",
        generated_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        dataset_root_fingerprint=HASH0,
        manifest_hash=HASH0,
        case_count=len(cases),
        cases=cases,
    )
    return replace(without_hash, manifest_hash=hash_dataset_manifest(without_hash))


def _split(manifest: DatasetManifest, *, internal_test_count: int = 5) -> DevelopmentSplitManifest:
    assignments: list[SplitAssignment] = []
    internal_test_start = manifest.case_count - internal_test_count + 1
    for index, case in enumerate(manifest.cases, start=1):
        if index >= internal_test_start:
            partition = "internal_test"
        elif index % 2 == 0:
            partition = "validation"
        else:
            partition = "train"
        assignments.append(
            SplitAssignment(
                anonymous_patient_id=case.anonymous_patient_id,
                anonymous_case_id=case.anonymous_case_id,
                partition=partition,
            )
        )
    assignments_tuple = tuple(
        sorted(assignments, key=lambda item: (item.anonymous_patient_id, item.anonymous_case_id))
    )
    patient_counts = {
        partition: len(
            {item.anonymous_patient_id for item in assignments_tuple if item.partition == partition}
        )
        for partition in ("train", "validation", "internal_test")
    }
    case_counts = {
        partition: sum(1 for item in assignments_tuple if item.partition == partition)
        for partition in ("train", "validation", "internal_test")
    }
    without_hash = DevelopmentSplitManifest(
        schema_version=PHASE2_SCHEMA_VERSION,
        manifest_type=DEVELOPMENT_SPLIT_MANIFEST_TYPE,
        generated_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        source_manifest_hash=manifest.manifest_hash,
        split_policy_version="patient_hash_rank_v1",
        split_seed=1729,
        split_hash=HASH0,
        assignments=assignments_tuple,
        train_patient_count=patient_counts["train"],
        validation_patient_count=patient_counts["validation"],
        internal_test_patient_count=patient_counts["internal_test"],
        train_case_count=case_counts["train"],
        validation_case_count=case_counts["validation"],
        internal_test_case_count=case_counts["internal_test"],
    )
    return replace(without_hash, split_hash=hash_development_split(without_hash))


def _lesion_record(case: DatasetCaseRecord, partition: str, rank: int) -> LesionComponentCaseRecord:
    lesion_volume = float(rank * 10)
    tumor_voxel_count = rank * 5
    lesion_count = 1 + (rank % 3)
    lesions = tuple(
        LesionSummaryRecord(
            index + 1, tumor_voxel_count // lesion_count, lesion_volume / lesion_count
        )
        for index in range(lesion_count)
    )
    total_voxel_count = sum(item.voxel_count for item in lesions)
    total_volume = sum(item.physical_volume_mm3 for item in lesions)
    return LesionComponentCaseRecord(
        anonymous_patient_id=case.anonymous_patient_id,
        anonymous_case_id=case.anonymous_case_id,
        partition=partition,
        analysis_performed=True,
        connectivity=26,
        tumor_label_value=1,
        voxel_volume_mm3=1.0,
        tumor_voxel_count=total_voxel_count,
        tumor_physical_volume_mm3=total_volume,
        lesion_count=len(lesions),
        lesions=lesions,
        qa_passed=True,
        failure_reasons=(),
    )


def _skipped_lesion_record(case: DatasetCaseRecord, partition: str) -> LesionComponentCaseRecord:
    return LesionComponentCaseRecord(
        anonymous_patient_id=case.anonymous_patient_id,
        anonymous_case_id=case.anonymous_case_id,
        partition=partition,
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


def _lesion_artifact(
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
    *,
    skipped_case_ids: frozenset[str] = frozenset(),
) -> LesionComponentsArtifact:
    case_by_pair = {
        (case.anonymous_patient_id, case.anonymous_case_id): case for case in manifest.cases
    }
    records: list[LesionComponentCaseRecord] = []
    analyzed_case_count = 0
    skipped_case_count = 0
    for rank, assignment in enumerate(split.assignments, start=1):
        case = case_by_pair[(assignment.anonymous_patient_id, assignment.anonymous_case_id)]
        if case.anonymous_case_id in skipped_case_ids:
            records.append(_skipped_lesion_record(case, assignment.partition))
            skipped_case_count += 1
        else:
            records.append(_lesion_record(case, assignment.partition, rank))
            analyzed_case_count += 1
    records_tuple = tuple(
        sorted(records, key=lambda item: (item.anonymous_patient_id, item.anonymous_case_id))
    )
    without_hash = LesionComponentsArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=LESION_COMPONENTS_STAGE,
        created_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        config_hash=HASH0,
        manifest_hash=manifest.manifest_hash,
        split_hash=split.split_hash,
        geometry_qa_artifact_hash="1" * 64,
        connectivity=26,
        case_count=len(records_tuple),
        analyzed_case_count=analyzed_case_count,
        skipped_case_count=skipped_case_count,
        case_records=records_tuple,
        lesion_artifact_hash=HASH0,
    )
    return replace(without_hash, lesion_artifact_hash=hash_lesion_components(without_hash))


def _fixture(
    case_count: int = 30, *, internal_test_count: int = 5
) -> tuple[DatasetManifest, DevelopmentSplitManifest, LesionComponentsArtifact]:
    manifest = _manifest(case_count)
    split = _split(manifest, internal_test_count=internal_test_count)
    lesion_artifact = _lesion_artifact(manifest, split)
    return manifest, split, lesion_artifact


def test_exclusion_of_internal_test_patients_and_cases_and_zero_overlap() -> None:
    manifest, split, lesion_artifact = _fixture()
    support_sets = generate_complete_fixed_support_manifest_set(manifest, split, lesion_artifact)
    internal_test_patients = {
        assignment.anonymous_patient_id
        for assignment in split.assignments
        if assignment.partition == "internal_test"
    }
    internal_test_cases = {
        assignment.anonymous_case_id
        for assignment in split.assignments
        if assignment.partition == "internal_test"
    }

    for manifests in support_sets.values():
        for support_manifest in manifests:
            selected_patients = {
                assignment.anonymous_patient_id for assignment in support_manifest.assignments
            }
            selected_cases = {
                assignment.anonymous_case_id for assignment in support_manifest.assignments
            }
            assert selected_patients.isdisjoint(internal_test_patients)
            assert selected_cases.isdisjoint(internal_test_cases)


def test_exact_k_selection_and_three_replicates_per_k() -> None:
    manifest, split, lesion_artifact = _fixture()
    support_sets = generate_complete_fixed_support_manifest_set(manifest, split, lesion_artifact)

    assert set(support_sets) == {1, 2, 5, 10, 20}
    for requested_k, manifests in support_sets.items():
        assert len(manifests) == DEFAULT_REPLICATE_COUNT
        assert {len(item.assignments) for item in manifests} == {requested_k}


def test_deterministic_repeatability_and_byte_identical_serialization() -> None:
    manifest, split, lesion_artifact = _fixture()

    first = generate_complete_fixed_support_manifest_set(manifest, split, lesion_artifact)
    second = generate_complete_fixed_support_manifest_set(manifest, split, lesion_artifact)

    assert {
        requested_k: tuple(fewshot_support_manifest_to_json(item) for item in manifests)
        for requested_k, manifests in first.items()
    } == {
        requested_k: tuple(fewshot_support_manifest_to_json(item) for item in manifests)
        for requested_k, manifests in second.items()
    }


def test_different_explicit_replicate_seeds_produce_valid_fixed_manifests() -> None:
    manifest, split, lesion_artifact = _fixture()
    inputs = validate_support_selection_inputs(manifest, split, lesion_artifact)

    manifests = generate_support_manifests_for_k(inputs, requested_k=5)

    assert [item.selection_seed for item in manifests] == [5101, 5202, 5303]
    assert (
        len(
            {
                tuple(assignment.anonymous_patient_id for assignment in item.assignments)
                for item in manifests
            }
        )
        >= 2
    )


def test_lesion_burden_stratification_when_feasible() -> None:
    manifest, split, lesion_artifact = _fixture()
    inputs = validate_support_selection_inputs(manifest, split, lesion_artifact)

    manifests = generate_support_manifests_for_k(inputs, requested_k=5)

    assert all(item.stratification_status == "stratified" for item in manifests)
    assert all(item.stratification_reason is None for item in manifests)


def test_deterministic_fallback_when_requested_k_is_too_small() -> None:
    manifest, split, lesion_artifact = _fixture()
    inputs = validate_support_selection_inputs(manifest, split, lesion_artifact)

    manifests = generate_support_manifests_for_k(inputs, requested_k=2)

    assert all(item.stratification_status == "fallback_unstratified" for item in manifests)
    assert {item.stratification_reason for item in manifests} == {"requested_k_below_three"}


def test_deterministic_fallback_when_eligible_burden_is_infeasible() -> None:
    manifest, split, lesion_artifact = _fixture()
    lesion_artifact = _lesion_artifact(
        manifest,
        split,
        skipped_case_ids=frozenset({"anon-c0001"}),
    )
    inputs = validate_support_selection_inputs(manifest, split, lesion_artifact)

    manifests = generate_support_manifests_for_k(inputs, requested_k=5)

    assert all(item.stratification_status == "fallback_unstratified" for item in manifests)
    assert {item.stratification_reason for item in manifests} == {"lesion_summary_not_provided"}


def test_insufficient_candidate_failure() -> None:
    manifest, split, lesion_artifact = _fixture(case_count=8, internal_test_count=4)
    inputs = validate_support_selection_inputs(manifest, split, lesion_artifact)

    with pytest.raises(FewshotSupportInsufficientCandidatesError):
        generate_support_manifests_for_k(inputs, requested_k=5)


def test_manifest_split_lesion_summary_id_mismatch_failure() -> None:
    manifest, split, lesion_artifact = _fixture()
    broken_without_hash = replace(
        lesion_artifact,
        manifest_hash="f" * 64,
        lesion_artifact_hash=HASH0,
    )
    broken = replace(
        broken_without_hash,
        lesion_artifact_hash=hash_lesion_components(broken_without_hash),
    )

    with pytest.raises(FewshotSupportMismatchError):
        validate_support_selection_inputs(manifest, split, broken)


def test_duplicate_patient_case_rejection_in_validated_inputs() -> None:
    manifest, split, lesion_artifact = _fixture()
    duplicate_candidates = (
        SupportCandidate("anon-p0001", "anon-c0001", "train"),
        SupportCandidate("anon-p0001", "anon-c0002", "validation"),
    )
    internal_test_patients = frozenset({"anon-p0029", "anon-p0030"})
    internal_test_cases = frozenset({"anon-c0029", "anon-c0030"})
    with pytest.raises(FewshotSupportInputError):
        ValidatedSupportSelectionInputs(
            manifest=manifest,
            split=split,
            eligible_candidates=duplicate_candidates,
            internal_test_patient_ids=internal_test_patients,
            internal_test_case_ids=internal_test_cases,
            immutable_test_cohort_hash="a" * 64,
            source_lesion_summary_hash=lesion_artifact.lesion_artifact_hash,
            eligible_candidate_burdens=(
                # Not reached because duplicate candidates fail first.
                # Keep pair count aligned for constructor shape only.
            ),
        )


def test_ranked_burden_buckets_follow_fixed_high_medium_low_cycle() -> None:
    manifest, split, lesion_artifact = _fixture()
    inputs = validate_support_selection_inputs(manifest, split, lesion_artifact)
    assert inputs.eligible_candidate_burdens is not None

    bucketed = assign_burden_buckets(inputs.eligible_candidate_burdens[:6])

    assert [item.bucket for item in bucketed[:3]] == list(BUCKET_ORDER)
