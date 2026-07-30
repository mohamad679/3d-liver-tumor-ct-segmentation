"""Unit tests for deterministic Phase 4 protocol construction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    LESION_COMPONENTS_STAGE,
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
from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.fewshot import (
    FIXED_PROTOCOL_ADAPTATION_MODES,
    FIXED_PROTOCOL_ROW_COUNT,
    FewshotAdaptationConfig,
    FewshotInitializationReference,
    FewshotProtocolValidationError,
    FewshotSupportManifest,
    build_all_fewshot_adaptation_configs,
    build_default_support_replicate_plans,
    build_fewshot_adaptation_config,
    build_fewshot_protocol_table,
    fewshot_adaptation_config_to_json,
    fewshot_protocol_table_to_json,
    generate_complete_fixed_support_manifest_set,
)

HASH0 = "0" * 64
GIT_COMMIT = "0" * 40
CREATED_AT_UTC = "2026-07-30T00:00:00Z"


def _reference() -> FewshotInitializationReference:
    return FewshotInitializationReference(
        reference_type="baseline_provenance",
        reference_identifier="baseline_init_001",
        artifact_sha256="1" * 64,
        checkpoint_sha256=None,
    )


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
    return LesionComponentCaseRecord(
        anonymous_patient_id=case.anonymous_patient_id,
        anonymous_case_id=case.anonymous_case_id,
        partition=partition,
        analysis_performed=True,
        connectivity=26,
        tumor_label_value=1,
        voxel_volume_mm3=1.0,
        tumor_voxel_count=sum(item.voxel_count for item in lesions),
        tumor_physical_volume_mm3=sum(item.physical_volume_mm3 for item in lesions),
        lesion_count=len(lesions),
        lesions=lesions,
        qa_passed=True,
        failure_reasons=(),
    )


def _lesion_artifact(
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
) -> LesionComponentsArtifact:
    case_by_pair = {
        (case.anonymous_patient_id, case.anonymous_case_id): case for case in manifest.cases
    }
    records = tuple(
        sorted(
            (
                _lesion_record(
                    case_by_pair[(assignment.anonymous_patient_id, assignment.anonymous_case_id)],
                    assignment.partition,
                    rank,
                )
                for rank, assignment in enumerate(split.assignments, start=1)
            ),
            key=lambda item: (item.anonymous_patient_id, item.anonymous_case_id),
        )
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
        case_count=len(records),
        analyzed_case_count=len(records),
        skipped_case_count=0,
        case_records=records,
        lesion_artifact_hash=HASH0,
    )
    return replace(without_hash, lesion_artifact_hash=hash_lesion_components(without_hash))


def _fixture() -> dict[int, tuple[FewshotSupportManifest, ...]]:
    manifest = _manifest()
    split = _split(manifest)
    lesion_artifact = _lesion_artifact(manifest, split)
    return generate_complete_fixed_support_manifest_set(manifest, split, lesion_artifact)


def _rebuild_support_manifest(
    support_manifest: FewshotSupportManifest,
    *,
    manifest_id: str | None = None,
    replicate_id: str | None = None,
    selection_seed: int | None = None,
    source_development_manifest_hash: str | None = None,
    source_development_split_hash: str | None = None,
    immutable_test_cohort_hash: str | None = None,
) -> FewshotSupportManifest:
    next_manifest_id = support_manifest.manifest_id if manifest_id is None else manifest_id
    next_replicate_id = support_manifest.replicate_id if replicate_id is None else replicate_id
    next_selection_seed = (
        support_manifest.selection_seed if selection_seed is None else selection_seed
    )
    next_source_manifest_hash = (
        support_manifest.source_development_manifest_hash
        if source_development_manifest_hash is None
        else source_development_manifest_hash
    )
    next_source_split_hash = (
        support_manifest.source_development_split_hash
        if source_development_split_hash is None
        else source_development_split_hash
    )
    next_immutable_test_hash = (
        support_manifest.immutable_test_cohort_hash
        if immutable_test_cohort_hash is None
        else immutable_test_cohort_hash
    )
    payload_without_hash = {
        "schema_version": support_manifest.schema_version,
        "manifest_id": next_manifest_id,
        "requested_k": support_manifest.requested_k,
        "replicate_id": next_replicate_id,
        "selection_seed": next_selection_seed,
        "selection_policy_version": support_manifest.selection_policy_version,
        "stratification_status": support_manifest.stratification_status,
        "stratification_reason": support_manifest.stratification_reason,
        "source_development_manifest_hash": next_source_manifest_hash,
        "source_development_split_hash": next_source_split_hash,
        "source_lesion_summary_hash": support_manifest.source_lesion_summary_hash,
        "immutable_test_cohort_hash": next_immutable_test_hash,
        "assignments": [
            {
                "anonymous_patient_id": assignment.anonymous_patient_id,
                "anonymous_case_id": assignment.anonymous_case_id,
            }
            for assignment in support_manifest.assignments
        ],
    }
    return FewshotSupportManifest(
        artifact_hash=sha256_json(payload_without_hash),
        schema_version=support_manifest.schema_version,
        manifest_id=next_manifest_id,
        requested_k=support_manifest.requested_k,
        replicate_id=next_replicate_id,
        selection_seed=next_selection_seed,
        selection_policy_version=support_manifest.selection_policy_version,
        stratification_status=support_manifest.stratification_status,
        stratification_reason=support_manifest.stratification_reason,
        source_development_manifest_hash=next_source_manifest_hash,
        source_development_split_hash=next_source_split_hash,
        source_lesion_summary_hash=support_manifest.source_lesion_summary_hash,
        immutable_test_cohort_hash=next_immutable_test_hash,
        assignments=support_manifest.assignments,
    )


def _config(
    support_manifest: FewshotSupportManifest,
    *,
    adaptation_mode: str = "head_only",
    seed: int = 1729,
) -> FewshotAdaptationConfig:
    return build_fewshot_adaptation_config(
        support_manifest,
        adaptation_mode=adaptation_mode,
        initialization_reference=_reference(),
        seed=seed,
    )


def test_stable_adaptation_config_serialization_and_hash() -> None:
    support_sets = _fixture()

    first = _config(support_sets[1][0], adaptation_mode="decoder_only", seed=9876)
    second = _config(support_sets[1][0], adaptation_mode="decoder_only", seed=9876)

    assert first == second
    assert first.artifact_hash == second.artifact_hash
    assert fewshot_adaptation_config_to_json(first) == fewshot_adaptation_config_to_json(second)


def test_invalid_support_manifest_hash_rejected() -> None:
    support_sets = _fixture()
    support_manifest = support_sets[1][0]
    object.__setattr__(support_manifest, "artifact_hash", "0" * 64)

    with pytest.raises(FewshotProtocolValidationError):
        _config(support_manifest)


def test_invalid_adaptation_mode_rejected() -> None:
    support_sets = _fixture()

    with pytest.raises(FewshotProtocolValidationError):
        _config(support_sets[1][0], adaptation_mode="encoder_only")


def test_all_45_protocol_rows_generated() -> None:
    support_sets = _fixture()
    table = build_fewshot_protocol_table(
        support_sets,
        initialization_reference=_reference(),
        base_seed=1729,
    )

    assert len(table.rows) == FIXED_PROTOCOL_ROW_COUNT


def test_exact_coverage_of_every_k_replicate_mode_combination() -> None:
    support_sets = _fixture()
    table = build_fewshot_protocol_table(
        support_sets,
        initialization_reference=_reference(),
        base_seed=1729,
    )

    assert {(row.requested_k, row.replicate_id, row.adaptation_mode) for row in table.rows} == {
        (requested_k, plan.replicate_id, adaptation_mode)
        for requested_k in (1, 2, 5, 10, 20)
        for plan in build_default_support_replicate_plans(requested_k)
        for adaptation_mode in FIXED_PROTOCOL_ADAPTATION_MODES
    }


def test_deterministic_row_ordering_and_byte_identical_repeated_generation() -> None:
    support_sets = _fixture()

    first = build_fewshot_protocol_table(
        support_sets,
        initialization_reference=_reference(),
        base_seed=1729,
    )
    second = build_fewshot_protocol_table(
        support_sets,
        initialization_reference=_reference(),
        base_seed=1729,
    )

    assert first.rows == tuple(
        sorted(
            first.rows,
            key=lambda row: (
                row.requested_k,
                row.replicate_id,
                row.adaptation_mode,
                row.planned_output_identifier,
                row.support_manifest_hash,
                row.adaptation_config_hash,
                row.initialization_reference.reference_type,
                row.initialization_reference.reference_identifier,
                row.run_status,
            ),
        )
    )
    assert fewshot_protocol_table_to_json(first) == fewshot_protocol_table_to_json(second)


def test_mixed_source_manifest_hash_rejected() -> None:
    support_sets = _fixture()
    support_sets[1] = (
        _rebuild_support_manifest(
            support_sets[1][0],
            source_development_manifest_hash="9" * 64,
        ),
        support_sets[1][1],
        support_sets[1][2],
    )

    with pytest.raises(FewshotProtocolValidationError):
        build_fewshot_protocol_table(
            support_sets,
            initialization_reference=_reference(),
            base_seed=1729,
        )


def test_mixed_split_hash_rejected() -> None:
    support_sets = _fixture()
    support_sets[2] = (
        _rebuild_support_manifest(
            support_sets[2][0],
            source_development_split_hash="8" * 64,
        ),
        support_sets[2][1],
        support_sets[2][2],
    )

    with pytest.raises(FewshotProtocolValidationError):
        build_fewshot_protocol_table(
            support_sets,
            initialization_reference=_reference(),
            base_seed=1729,
        )


def test_mixed_immutable_test_hash_rejected() -> None:
    support_sets = _fixture()
    support_sets[5] = (
        _rebuild_support_manifest(
            support_sets[5][0],
            immutable_test_cohort_hash="7" * 64,
        ),
        support_sets[5][1],
        support_sets[5][2],
    )

    with pytest.raises(FewshotProtocolValidationError):
        build_fewshot_protocol_table(
            support_sets,
            initialization_reference=_reference(),
            base_seed=1729,
        )


def test_missing_replicate_rejected() -> None:
    support_sets = _fixture()
    support_sets[10] = support_sets[10][:2]

    with pytest.raises(FewshotProtocolValidationError):
        build_fewshot_protocol_table(
            support_sets,
            initialization_reference=_reference(),
            base_seed=1729,
        )


def test_duplicate_identity_rejected() -> None:
    support_sets = _fixture()
    duplicate_identity = _rebuild_support_manifest(
        support_sets[1][1],
        manifest_id="k01_replicate_01_b",
        replicate_id="replicate_01",
        selection_seed=9999,
    )
    support_sets[1] = (
        support_sets[1][0],
        duplicate_identity,
        support_sets[1][2],
    )

    with pytest.raises(FewshotProtocolValidationError):
        build_fewshot_protocol_table(
            support_sets,
            initialization_reference=_reference(),
            base_seed=1729,
        )


def test_planned_output_identifier_contains_no_absolute_path_or_patient_case_identifier() -> None:
    support_sets = _fixture()
    table = build_fewshot_protocol_table(
        support_sets,
        initialization_reference=_reference(),
        base_seed=1729,
    )
    assignment_pairs = {
        (assignment.anonymous_patient_id, assignment.anonymous_case_id)
        for manifests in support_sets.values()
        for support_manifest in manifests
        for assignment in support_manifest.assignments
    }

    for row in table.rows:
        assert not row.planned_output_identifier.startswith("/")
        assert "/" not in row.planned_output_identifier
        assert "\\" not in row.planned_output_identifier
        assert ":" not in row.planned_output_identifier
        for patient_id, case_id in assignment_pairs:
            assert patient_id not in row.planned_output_identifier
            assert case_id not in row.planned_output_identifier


def test_build_all_adaptation_configs_returns_45_unique_hashes() -> None:
    support_sets = _fixture()

    configs = build_all_fewshot_adaptation_configs(
        support_sets,
        initialization_reference=_reference(),
        base_seed=1729,
    )

    assert len(configs) == FIXED_PROTOCOL_ROW_COUNT
    assert len({config.artifact_hash for config in configs.values()}) == FIXED_PROTOCOL_ROW_COUNT
