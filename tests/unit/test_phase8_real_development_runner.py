"""Unit tests for the Phase 8 real-development planning scaffold (Substage 2)."""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    PHASE2_SCHEMA_VERSION,
    DatasetCaseRecord,
    DatasetManifest,
    DevelopmentSplitManifest,
    SplitAssignment,
    hash_dataset_manifest,
    hash_development_split,
    phase2_artifact_to_json,
)
from protoem_ct.artifacts.hashing import sha256_file, sha256_json
from protoem_ct.cli.main import app
from protoem_ct.data.phase2_paths import InvalidRelativeDatasetPathError
from protoem_ct.external.internal_evidence import (
    ArtifactReference,
    Phase8FixedCandidateInventory,
    phase8_fixed_candidate_inventory_from_mapping,
)
from protoem_ct.external.real_development_runner import (
    EXECUTION_RELEASE_SCAFFOLD_ONLY,
    PHASE8_REAL_DEVELOPMENT_CASE_BINDINGS_FILENAME,
    PHASE8_REAL_DEVELOPMENT_INPUT_BINDING_FILENAME,
    PHASE8_REAL_DEVELOPMENT_PLAN_SUMMARY_FILENAME,
    PHASE8_REAL_DEVELOPMENT_RUN_PLAN_FILENAME,
    PIXEL_ACCESS_NOT_STARTED,
    Phase8RealDevelopmentCaseBinding,
    Phase8RealDevelopmentExecutionNotReleasedError,
    Phase8RealDevelopmentHashError,
    Phase8RealDevelopmentInputBinding,
    Phase8RealDevelopmentPublicationError,
    Phase8RealDevelopmentRunPlan,
    Phase8RealDevelopmentSerializationError,
    Phase8RealDevelopmentValidationError,
    build_phase8_real_development_case_binding_collection,
    build_phase8_real_development_case_bindings,
    build_phase8_real_development_input_binding,
    build_phase8_real_development_run_plan,
    execute_phase8_real_development_run,
    hash_phase8_real_development_case_binding,
    hash_phase8_real_development_case_binding_collection,
    hash_phase8_real_development_input_binding,
    hash_phase8_real_development_run_plan,
    phase8_real_development_case_binding_collection_from_mapping,
    phase8_real_development_case_binding_collection_to_dict,
    phase8_real_development_case_binding_from_mapping,
    phase8_real_development_case_binding_to_dict,
    phase8_real_development_input_binding_from_mapping,
    phase8_real_development_input_binding_to_dict,
    phase8_real_development_run_plan_from_mapping,
    phase8_real_development_run_plan_to_dict,
    run_phase8_real_development_plan_publication,
)

GIT_COMMIT = "0" * 40
CREATED_AT_UTC = "2026-07-30T00:00:00Z"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64

# ---------------------------------------------------------------------------
# Synthetic Phase 2 manifest/split fixtures (mirrors tests/unit/test_fewshot_protocol.py)
# ---------------------------------------------------------------------------


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


def _manifest(case_count: int = 6) -> DatasetManifest:
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
        dataset_root_fingerprint="0" * 64,
        manifest_hash="0" * 64,
        case_count=len(cases),
        cases=cases,
    )
    return replace(without_hash, manifest_hash=hash_dataset_manifest(without_hash))


def _split(manifest: DatasetManifest) -> DevelopmentSplitManifest:
    # cases 1-3 -> train, 4-5 -> validation, 6 -> internal_test (1:1 patient:case).
    assignments: list[SplitAssignment] = []
    for index, case in enumerate(manifest.cases, start=1):
        if index <= 3:
            partition = "train"
        elif index <= 5:
            partition = "validation"
        else:
            partition = "internal_test"
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
        split_hash="0" * 64,
        assignments=assignments_tuple,
        train_patient_count=patient_counts["train"],
        validation_patient_count=patient_counts["validation"],
        internal_test_patient_count=patient_counts["internal_test"],
        train_case_count=patient_counts["train"],
        validation_case_count=patient_counts["validation"],
        internal_test_case_count=patient_counts["internal_test"],
    )
    return replace(without_hash, split_hash=hash_development_split(without_hash))


def _write_manifest_and_split(tmp_path: Path) -> tuple[Path, Path, DatasetManifest]:
    manifest = _manifest()
    split = _split(manifest)
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    phase2_artifact_to_json(manifest, manifest_path)
    phase2_artifact_to_json(split, split_path)
    return manifest_path, split_path, manifest


def _build_input_binding(
    tmp_path: Path,
) -> tuple[Phase8RealDevelopmentInputBinding, DatasetManifest]:
    manifest_path, split_path, manifest = _write_manifest_and_split(tmp_path)
    binding = build_phase8_real_development_input_binding(
        manifest_path=manifest_path,
        split_path=split_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        approved_development_artifact_set_identity="phase2-real-lits-v2",
    )
    return binding, manifest


# ---------------------------------------------------------------------------
# Synthetic Phase 8 candidate inventory fixture
# ---------------------------------------------------------------------------


def _reference(role: str, artifact_hash: str) -> dict[str, Any]:
    return {
        "schema_name": f"phase8_{role}",
        "schema_version": "v1",
        "artifact_hash": artifact_hash,
        "artifact_role": role,
    }


def _rehash(mapping: dict[str, Any], hash_field: str) -> dict[str, Any]:
    result = deepcopy(mapping)
    payload = deepcopy(result)
    payload.pop(hash_field, None)
    result[hash_field] = sha256_json(payload)
    return result


def _inventory_mapping(
    *, training_config_hash: str = HASH_A, preprocessing_hash: str = HASH_B
) -> dict[str, Any]:
    mapping = {
        "schema_name": "phase8_fixed_candidate_inventory",
        "schema_version": "v1",
        "inventory_state": "freeze_ready",
        "candidate_set_locked": True,
        "selection_started": True,
        "supports_single_candidate_without_comparison": True,
        "candidates": [
            {
                "candidate_id": "candidate_a",
                "model_family": "monai_segresnet",
                "method_identity": "monai_segresnet_baseline",
                "training_adaptation_mode": "baseline_training",
                "training_config_reference": _reference("training_config", training_config_hash),
                "preprocessing_evidence_hash": preprocessing_hash,
                "fixed_seeds": [1729],
                "executable_readiness_state": "executable",
                "failure_handling_policy": "single_candidate_no_comparison",
                "uses_external_artifacts": False,
            }
        ],
    }
    return _rehash(mapping, "inventory_hash")


def _inventory(**kwargs: str) -> Phase8FixedCandidateInventory:
    return phase8_fixed_candidate_inventory_from_mapping(_inventory_mapping(**kwargs))


def _build_run_plan(
    tmp_path: Path,
) -> tuple[Phase8RealDevelopmentRunPlan, Phase8RealDevelopmentInputBinding, DatasetManifest]:
    binding, manifest = _build_input_binding(tmp_path)
    inventory = _inventory()
    training_config_reference = ArtifactReference(
        schema_name="phase8_training_config",
        schema_version="v1",
        artifact_hash=HASH_A,
        artifact_role="training_config",
    )
    preprocessing_decision_reference = ArtifactReference(
        schema_name="phase8_preprocessing_decision",
        schema_version="v1",
        artifact_hash=HASH_B,
        artifact_role="preprocessing_decision",
    )
    plan = build_phase8_real_development_run_plan(
        input_binding=binding,
        candidate_inventory=inventory,
        candidate_id="candidate_a",
        training_config_reference=training_config_reference,
        preprocessing_decision_reference=preprocessing_decision_reference,
        fixed_seeds=(1729,),
        expected_checkpoint_metadata_schema=("phase8_checkpoint_metadata", "v1"),
        expected_validation_evidence_schema=("phase8_validation_evidence_reference", "v1"),
        expected_output_artifact_names=("checkpoint_metadata", "validation_evidence"),
    )
    return plan, binding, manifest


# ---------------------------------------------------------------------------
# 1 + 2: strict round-trip, self-hash validation, unknown-field rejection
# ---------------------------------------------------------------------------


def test_input_binding_round_trips_and_rejects_unknown_fields(tmp_path: Path) -> None:
    binding, _manifest_obj = _build_input_binding(tmp_path)

    mapping = phase8_real_development_input_binding_to_dict(binding)
    round_trip = phase8_real_development_input_binding_from_mapping(mapping)
    assert round_trip == binding
    assert hash_phase8_real_development_input_binding(binding) == binding.input_binding_hash

    extra = dict(mapping)
    extra["unexpected"] = "field"
    with pytest.raises(Phase8RealDevelopmentSerializationError):
        phase8_real_development_input_binding_from_mapping(extra)

    tampered = dict(mapping)
    tampered["input_binding_hash"] = HASH_C
    with pytest.raises(Phase8RealDevelopmentHashError):
        phase8_real_development_input_binding_from_mapping(tampered)


def test_run_plan_round_trips_and_rejects_unknown_fields(tmp_path: Path) -> None:
    plan, _binding, _manifest_obj = _build_run_plan(tmp_path)

    mapping = phase8_real_development_run_plan_to_dict(plan)
    round_trip = phase8_real_development_run_plan_from_mapping(mapping)
    assert round_trip == plan
    assert hash_phase8_real_development_run_plan(plan) == plan.run_plan_hash

    extra = dict(mapping)
    extra["unexpected"] = "field"
    with pytest.raises(Phase8RealDevelopmentSerializationError):
        phase8_real_development_run_plan_from_mapping(extra)


def test_case_binding_collection_round_trips_and_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    plan, _binding, manifest = _build_run_plan(tmp_path)
    case_bindings = build_phase8_real_development_case_bindings(run_plan=plan, manifest=manifest)
    collection = build_phase8_real_development_case_binding_collection(
        run_plan=plan, case_bindings=case_bindings
    )

    mapping = phase8_real_development_case_binding_collection_to_dict(collection)
    round_trip = phase8_real_development_case_binding_collection_from_mapping(mapping)
    assert round_trip == collection
    assert (
        hash_phase8_real_development_case_binding_collection(collection)
        == collection.collection_hash
    )

    extra = dict(mapping)
    extra["unexpected"] = "field"
    with pytest.raises(Phase8RealDevelopmentSerializationError):
        phase8_real_development_case_binding_collection_from_mapping(extra)

    single = phase8_real_development_case_binding_to_dict(case_bindings[0])
    single_round_trip = phase8_real_development_case_binding_from_mapping(single)
    assert single_round_trip == case_bindings[0]
    assert hash_phase8_real_development_case_binding(case_bindings[0]) == (
        case_bindings[0].case_binding_hash
    )
    single_extra = dict(single)
    single_extra["unexpected"] = "field"
    with pytest.raises(Phase8RealDevelopmentSerializationError):
        phase8_real_development_case_binding_from_mapping(single_extra)


# ---------------------------------------------------------------------------
# 3-5: manifest/split file hash mismatch and identity mismatch
# ---------------------------------------------------------------------------


def test_manifest_file_hash_mismatch_rejected(tmp_path: Path) -> None:
    manifest_path, split_path, _manifest_obj = _write_manifest_and_split(tmp_path)
    with pytest.raises(Phase8RealDevelopmentHashError):
        build_phase8_real_development_input_binding(
            manifest_path=manifest_path,
            split_path=split_path,
            expected_manifest_sha256=HASH_C,
            expected_split_sha256=sha256_file(split_path),
            approved_development_artifact_set_identity="phase2-real-lits-v2",
        )


def test_split_file_hash_mismatch_rejected(tmp_path: Path) -> None:
    manifest_path, split_path, _manifest_obj = _write_manifest_and_split(tmp_path)
    with pytest.raises(Phase8RealDevelopmentHashError):
        build_phase8_real_development_input_binding(
            manifest_path=manifest_path,
            split_path=split_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_split_sha256=HASH_C,
            approved_development_artifact_set_identity="phase2-real-lits-v2",
        )


def test_manifest_split_identity_mismatch_rejected(tmp_path: Path) -> None:
    manifest_path, _split_path, manifest = _write_manifest_and_split(tmp_path)
    other_manifest = _manifest(case_count=3)
    mismatched_split = _split(other_manifest)
    mismatched_split_path = tmp_path / "mismatched_split.json"
    phase2_artifact_to_json(mismatched_split, mismatched_split_path)

    with pytest.raises(Phase8RealDevelopmentValidationError):
        build_phase8_real_development_input_binding(
            manifest_path=manifest_path,
            split_path=mismatched_split_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_split_sha256=sha256_file(mismatched_split_path),
            approved_development_artifact_set_identity="phase2-real-lits-v2",
        )
    assert manifest.manifest_hash != other_manifest.manifest_hash


# ---------------------------------------------------------------------------
# 6-7: partition overlap and internal-test rejection
# ---------------------------------------------------------------------------


def test_input_binding_rejects_overlapping_partitions(tmp_path: Path) -> None:
    binding, _manifest_obj = _build_input_binding(tmp_path)
    mapping = phase8_real_development_input_binding_to_dict(binding)
    tampered = dict(mapping)
    tampered["validation_patient_ids"] = deepcopy(tampered["train_patient_ids"])
    tampered = _rehash(tampered, "input_binding_hash")
    with pytest.raises(Phase8RealDevelopmentValidationError):
        phase8_real_development_input_binding_from_mapping(tampered)


def test_case_binding_rejects_internal_test_partition(tmp_path: Path) -> None:
    plan, _binding, manifest = _build_run_plan(tmp_path)
    case_bindings = build_phase8_real_development_case_bindings(run_plan=plan, manifest=manifest)
    assert all(item.partition in {"train", "validation"} for item in case_bindings)
    assert not any(item.partition == "internal_test" for item in case_bindings)

    valid = phase8_real_development_case_binding_to_dict(case_bindings[0])
    tampered = dict(valid)
    tampered["partition"] = "internal_test"
    tampered = _rehash(tampered, "case_binding_hash")
    with pytest.raises(Phase8RealDevelopmentValidationError):
        phase8_real_development_case_binding_from_mapping(tampered)


# ---------------------------------------------------------------------------
# 8: duplicate case rejection
# ---------------------------------------------------------------------------


def test_case_binding_collection_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    plan, _binding, manifest = _build_run_plan(tmp_path)
    case_bindings = build_phase8_real_development_case_bindings(run_plan=plan, manifest=manifest)
    duplicated = case_bindings + (case_bindings[0],)
    with pytest.raises(Phase8RealDevelopmentValidationError):
        build_phase8_real_development_case_binding_collection(
            run_plan=plan, case_bindings=duplicated
        )


# ---------------------------------------------------------------------------
# 9: unsafe relative path rejection propagates
# ---------------------------------------------------------------------------


def test_case_binding_rejects_unsafe_relative_path() -> None:
    with pytest.raises(InvalidRelativeDatasetPathError):
        Phase8RealDevelopmentCaseBinding(
            schema_name="phase8_real_development_case_binding",
            schema_version="v1",
            anonymous_patient_id="anon-p0001",
            anonymous_case_id="anon-c0001",
            partition="train",
            manifest_record_identity_hash=HASH_A,
            relative_image_path="../escape/image.nii.gz",
            relative_label_path="labels/case-0001.nii.gz",
            pixel_access_state=PIXEL_ACCESS_NOT_STARTED,
            case_binding_hash=HASH_B,
        )


# ---------------------------------------------------------------------------
# 10-11: candidate/preprocessing-reference mismatch
# ---------------------------------------------------------------------------


def test_run_plan_builder_rejects_unknown_candidate_id(tmp_path: Path) -> None:
    binding, _manifest_obj = _build_input_binding(tmp_path)
    inventory = _inventory()
    training_config_reference = ArtifactReference(
        schema_name="phase8_training_config",
        schema_version="v1",
        artifact_hash=HASH_A,
        artifact_role="training_config",
    )
    preprocessing_decision_reference = ArtifactReference(
        schema_name="phase8_preprocessing_decision",
        schema_version="v1",
        artifact_hash=HASH_B,
        artifact_role="preprocessing_decision",
    )
    with pytest.raises(Phase8RealDevelopmentValidationError):
        build_phase8_real_development_run_plan(
            input_binding=binding,
            candidate_inventory=inventory,
            candidate_id="candidate_missing",
            training_config_reference=training_config_reference,
            preprocessing_decision_reference=preprocessing_decision_reference,
            fixed_seeds=(1729,),
            expected_checkpoint_metadata_schema=("phase8_checkpoint_metadata", "v1"),
            expected_validation_evidence_schema=(
                "phase8_validation_evidence_reference",
                "v1",
            ),
            expected_output_artifact_names=("checkpoint_metadata",),
        )


def test_run_plan_builder_rejects_preprocessing_reference_mismatch(tmp_path: Path) -> None:
    binding, _manifest_obj = _build_input_binding(tmp_path)
    inventory = _inventory()
    training_config_reference = ArtifactReference(
        schema_name="phase8_training_config",
        schema_version="v1",
        artifact_hash=HASH_A,
        artifact_role="training_config",
    )
    mismatched_preprocessing_reference = ArtifactReference(
        schema_name="phase8_preprocessing_decision",
        schema_version="v1",
        artifact_hash=HASH_C,
        artifact_role="preprocessing_decision",
    )
    with pytest.raises(Phase8RealDevelopmentValidationError):
        build_phase8_real_development_run_plan(
            input_binding=binding,
            candidate_inventory=inventory,
            candidate_id="candidate_a",
            training_config_reference=training_config_reference,
            preprocessing_decision_reference=mismatched_preprocessing_reference,
            fixed_seeds=(1729,),
            expected_checkpoint_metadata_schema=("phase8_checkpoint_metadata", "v1"),
            expected_validation_evidence_schema=(
                "phase8_validation_evidence_reference",
                "v1",
            ),
            expected_output_artifact_names=("checkpoint_metadata",),
        )


# ---------------------------------------------------------------------------
# 12: external-data declaration rejection
# ---------------------------------------------------------------------------


def test_input_binding_rejects_external_data_declaration(tmp_path: Path) -> None:
    binding, _manifest_obj = _build_input_binding(tmp_path)
    mapping = phase8_real_development_input_binding_to_dict(binding)
    tampered = dict(mapping)
    tampered["no_external_data"] = False
    tampered = _rehash(tampered, "input_binding_hash")
    with pytest.raises(Phase8RealDevelopmentValidationError):
        phase8_real_development_input_binding_from_mapping(tampered)


# ---------------------------------------------------------------------------
# 13-14: scaffold_only / not_started enforcement, no release path
# ---------------------------------------------------------------------------


def test_run_plan_enforces_scaffold_only_and_not_started(tmp_path: Path) -> None:
    plan, _binding, _manifest_obj = _build_run_plan(tmp_path)
    assert plan.pixel_access_state == PIXEL_ACCESS_NOT_STARTED
    assert plan.execution_release_state == EXECUTION_RELEASE_SCAFFOLD_ONLY

    mapping = phase8_real_development_run_plan_to_dict(plan)
    released_pixel_access = dict(mapping)
    released_pixel_access["pixel_access_state"] = "released"
    released_pixel_access = _rehash(released_pixel_access, "run_plan_hash")
    with pytest.raises(Phase8RealDevelopmentValidationError):
        phase8_real_development_run_plan_from_mapping(released_pixel_access)

    released_execution = dict(mapping)
    released_execution["execution_release_state"] = "released"
    released_execution = _rehash(released_execution, "run_plan_hash")
    with pytest.raises(Phase8RealDevelopmentValidationError):
        phase8_real_development_run_plan_from_mapping(released_execution)


def test_execute_real_development_run_always_raises(tmp_path: Path) -> None:
    plan, _binding, _manifest_obj = _build_run_plan(tmp_path)

    with pytest.raises(Phase8RealDevelopmentExecutionNotReleasedError):
        execute_phase8_real_development_run(plan)

    class _FakeExecutor:
        def construct_model(self, plan: Any) -> object:
            raise AssertionError("must never be called")

        def load_training_data(self, plan: Any, case_bindings: Any) -> object:
            raise AssertionError("must never be called")

        def run_validation_inference(self, plan: Any, case_bindings: Any) -> object:
            raise AssertionError("must never be called")

        def persist_checkpoint(self, plan: Any, model: Any) -> object:
            raise AssertionError("must never be called")

        def publish_validation_metrics(self, plan: Any, predictions: Any) -> object:
            raise AssertionError("must never be called")

        def publish_preprocessing_evidence(self, plan: Any) -> object:
            raise AssertionError("must never be called")

        def publish_checkpoint_metadata(self, plan: Any, checkpoint: Any) -> object:
            raise AssertionError("must never be called")

    with pytest.raises(Phase8RealDevelopmentExecutionNotReleasedError):
        execute_phase8_real_development_run(plan, executor=_FakeExecutor())

    with pytest.raises(Phase8RealDevelopmentExecutionNotReleasedError):
        execute_phase8_real_development_run(plan, executor=_FakeExecutor(), extra_flag=True)


# ---------------------------------------------------------------------------
# 16-18: output-root safety
# ---------------------------------------------------------------------------


def _publish(tmp_path: Path, output_root: Path, repository_root: Path | None = None) -> Any:
    plan, binding, manifest = _build_run_plan(tmp_path)
    case_bindings = build_phase8_real_development_case_bindings(run_plan=plan, manifest=manifest)
    return run_phase8_real_development_plan_publication(
        input_binding=binding,
        run_plan=plan,
        case_bindings=case_bindings,
        output_root=output_root,
        repository_root=repository_root or (tmp_path / "repo"),
    )


def test_publication_rejects_output_root_inside_repository(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = repository_root / "runs" / "phase8-real-dev"
    with pytest.raises(Phase8RealDevelopmentPublicationError):
        _publish(tmp_path, output_root, repository_root=repository_root)


def test_publication_rejects_symlink_output_root(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    real_target = tmp_path / "real-target"
    real_target.mkdir()
    output_root = tmp_path / "symlinked-output"
    output_root.symlink_to(real_target, target_is_directory=True)
    with pytest.raises(Phase8RealDevelopmentPublicationError):
        _publish(tmp_path, output_root, repository_root=repository_root)


def test_publication_rejects_nonempty_output_root(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "runs" / "phase8-real-dev"
    output_root.mkdir(parents=True)
    (output_root / "preexisting.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Phase8RealDevelopmentPublicationError):
        _publish(tmp_path, output_root, repository_root=repository_root)


# ---------------------------------------------------------------------------
# 19-20: determinism
# ---------------------------------------------------------------------------


def test_case_bindings_are_deterministically_ordered(tmp_path: Path) -> None:
    plan, _binding, manifest = _build_run_plan(tmp_path)
    first = build_phase8_real_development_case_bindings(run_plan=plan, manifest=manifest)
    second = build_phase8_real_development_case_bindings(run_plan=plan, manifest=manifest)
    assert first == second
    assert [item.partition for item in first] == sorted(item.partition for item in first)


def test_publication_is_byte_identical_across_two_output_roots(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    manifest_path, split_path, manifest = _write_manifest_and_split(tmp_path)

    def _make_plan() -> tuple[Phase8RealDevelopmentRunPlan, Phase8RealDevelopmentInputBinding]:
        binding = build_phase8_real_development_input_binding(
            manifest_path=manifest_path,
            split_path=split_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_split_sha256=sha256_file(split_path),
            approved_development_artifact_set_identity="phase2-real-lits-v2",
        )
        inventory = _inventory()
        plan = build_phase8_real_development_run_plan(
            input_binding=binding,
            candidate_inventory=inventory,
            candidate_id="candidate_a",
            training_config_reference=ArtifactReference(
                schema_name="phase8_training_config",
                schema_version="v1",
                artifact_hash=HASH_A,
                artifact_role="training_config",
            ),
            preprocessing_decision_reference=ArtifactReference(
                schema_name="phase8_preprocessing_decision",
                schema_version="v1",
                artifact_hash=HASH_B,
                artifact_role="preprocessing_decision",
            ),
            fixed_seeds=(1729,),
            expected_checkpoint_metadata_schema=("phase8_checkpoint_metadata", "v1"),
            expected_validation_evidence_schema=(
                "phase8_validation_evidence_reference",
                "v1",
            ),
            expected_output_artifact_names=("checkpoint_metadata", "validation_evidence"),
        )
        return plan, binding

    (tmp_path / "runs").mkdir()
    output_root_1 = tmp_path / "runs" / "attempt-1"
    output_root_2 = tmp_path / "runs" / "attempt-2"

    plan_1, binding_1 = _make_plan()
    case_bindings_1 = build_phase8_real_development_case_bindings(
        run_plan=plan_1, manifest=manifest
    )
    result_1 = run_phase8_real_development_plan_publication(
        input_binding=binding_1,
        run_plan=plan_1,
        case_bindings=case_bindings_1,
        output_root=output_root_1,
        repository_root=repository_root,
    )

    plan_2, binding_2 = _make_plan()
    case_bindings_2 = build_phase8_real_development_case_bindings(
        run_plan=plan_2, manifest=manifest
    )
    result_2 = run_phase8_real_development_plan_publication(
        input_binding=binding_2,
        run_plan=plan_2,
        case_bindings=case_bindings_2,
        output_root=output_root_2,
        repository_root=repository_root,
    )

    assert result_1.artifact_hashes == result_2.artifact_hashes
    for filename in sorted(result_1.artifact_hashes):
        assert (output_root_1 / filename).read_bytes() == (output_root_2 / filename).read_bytes()


# ---------------------------------------------------------------------------
# 21-22: CLI
# ---------------------------------------------------------------------------


def test_cli_help_lists_new_command() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "plan-phase8-real-development-run" in result.output

    command_help = CliRunner().invoke(app, ["plan-phase8-real-development-run", "--help"])
    assert command_help.exit_code == 0, command_help.output
    assert "--manifest-path" in command_help.output
    assert "--split-path" in command_help.output
    assert "--candidate-inventory-path" in command_help.output
    assert "--preprocessing-decision" in command_help.output


def test_cli_successful_synthetic_plan_only_run(tmp_path: Path) -> None:
    manifest_path, split_path, _manifest_obj = _write_manifest_and_split(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    (tmp_path / "runs").mkdir()
    output_root = tmp_path / "runs" / "phase8-real-dev-cli"

    preprocessing_mapping = _preprocessing_decision_mapping()
    preprocessing_path = tmp_path / "preprocessing.json"
    preprocessing_path.write_text(json.dumps(preprocessing_mapping), encoding="utf-8")

    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            _inventory_mapping(
                preprocessing_hash=preprocessing_mapping["preprocessing_decision_hash"]
            )
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "plan-phase8-real-development-run",
            "--manifest-path",
            str(manifest_path),
            "--split-path",
            str(split_path),
            "--expected-manifest-sha256",
            sha256_file(manifest_path),
            "--expected-split-sha256",
            sha256_file(split_path),
            "--candidate-inventory-path",
            str(inventory_path),
            "--candidate-id",
            "candidate_a",
            "--preprocessing-decision-path",
            str(preprocessing_path),
            "--approved-development-artifact-set-identity",
            "phase2-real-lits-v2",
            "--output-root",
            str(output_root),
            "--repository-root",
            str(repository_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "scaffold_only" in result.output
    assert "pixel_access_not_started" in result.output
    assert (output_root / PHASE8_REAL_DEVELOPMENT_INPUT_BINDING_FILENAME).is_file()
    assert (output_root / PHASE8_REAL_DEVELOPMENT_RUN_PLAN_FILENAME).is_file()
    assert (output_root / PHASE8_REAL_DEVELOPMENT_CASE_BINDINGS_FILENAME).is_file()
    assert (output_root / PHASE8_REAL_DEVELOPMENT_PLAN_SUMMARY_FILENAME).is_file()


def _preprocessing_decision_mapping() -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "schema_name": "phase8_preprocessing_decision",
        "schema_version": "v1",
        "candidate_id": "candidate_a",
        "development_manifest_hash": HASH_A,
        "development_split_hash": HASH_C,
        "preprocessing_config_schema_name": "phase8_preprocessing_config",
        "preprocessing_config_schema_version": "v1",
        "preprocessing_config_hash": HASH_A,
        "fit_scope": "no_data_dependent_fit",
        "fit_artifact_reference": None,
        "image_interpolation_policy": "bilinear",
        "label_interpolation_policy": "nearest",
        "orientation_policy_reference": _reference("orientation_policy", HASH_A),
        "spacing_policy_reference": _reference("spacing_policy", HASH_A),
        "intensity_policy_reference": _reference("intensity_policy", HASH_A),
        "crop_roi_policy_reference": _reference("crop_roi_policy", HASH_A),
        "normalization_policy_reference": _reference("normalization_policy", HASH_A),
        "originating_git_commit": "0" * 40,
        "no_external_data": True,
        "evidence_status": "resolved",
    }
    return _rehash(mapping, "preprocessing_decision_hash")


# ---------------------------------------------------------------------------
# 23: no real medical-imaging library is ever imported or referenced
# ---------------------------------------------------------------------------


def test_module_never_imports_or_references_medical_imaging_libraries() -> None:
    import protoem_ct.external.real_development_runner as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    forbidden_names = {"nibabel", "SimpleITK", "torch", "monai"}

    tree = ast.parse(source)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_names.add(node.module.split(".")[0])

    assert imported_names.isdisjoint(forbidden_names)
    for name in forbidden_names:
        assert name not in source
