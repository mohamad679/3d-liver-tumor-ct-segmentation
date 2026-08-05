"""Unit tests for the Phase 8 bounded real-development training pilot (Substage 4B)."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

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
    phase2_artifact_to_json,
)
from protoem_ct.artifacts.hashing import sha256_file, sha256_json
from protoem_ct.cli.main import app
from protoem_ct.external.definitive_training_pilot import (
    APPROVED_CANDIDATE_ID,
    APPROVED_MODEL_FAMILY,
    DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
    RAW_LABEL_TUMOR_VALUE,
    REQUIRED_MAX_OPTIMIZER_STEPS,
    REQUIRED_NEGATIVE_PATCHES_FROM_EMPTY_CASE,
    REQUIRED_NEGATIVE_PATCHES_FROM_POSITIVE_CASE,
    REQUIRED_PATCH_SIZE,
    REQUIRED_POSITIVE_PATCHES_FROM_POSITIVE_CASE,
    REQUIRED_SLIDING_WINDOW_BATCH_SIZE,
    REQUIRED_SLIDING_WINDOW_OVERLAP,
    REQUIRED_SLIDING_WINDOW_ROI_SIZE,
    REQUIRED_TOTAL_PATCHES,
    Phase8BoundedPilotCaseSelectionError,
    Phase8BoundedPilotConfigError,
    Phase8BoundedPilotDataError,
    Phase8BoundedPilotError,
    Phase8BoundedPilotLabelDomainError,
    Phase8BoundedPilotOutputRootError,
    Phase8BoundedPilotPrerequisitesError,
    Phase8BoundedPilotProcessWatchdogTimeoutError,
    Phase8BoundedPilotResult,
    Phase8BoundedPilotRuntimeError,
    Phase8BoundedPilotSamplingScheduleError,
    Phase8BoundedPilotSelectedCase,
    Phase8BoundedPilotWatchdogTimeoutError,
    _extract_padded_patch,
    _import_monai,
    _import_torch,
    _load_pilot_training_patch,
    _normalize_hu,
    _reload_and_verify_checkpoint_from_bytes,
    _run_bounded_training_steps,
    _run_pilot_validation_sliding_window,
    _validate_raw_case_pair_geometry,
    build_phase8_bounded_pilot_access_ledger,
    build_phase8_bounded_pilot_checkpoint_metadata,
    build_phase8_bounded_pilot_config,
    build_phase8_bounded_pilot_sampling_schedule,
    phase8_bounded_pilot_access_ledger_to_dict,
    phase8_bounded_pilot_checkpoint_metadata_to_dict,
    phase8_bounded_pilot_config_to_dict,
    run_phase8_bounded_pilot_with_watchdog,
    select_phase8_bounded_pilot_cases,
    validate_phase8_bounded_pilot_output_root,
    verify_phase8_bounded_pilot_prerequisites,
)
from protoem_ct.external.internal_evidence import (
    Phase8CheckpointMetadata,
    Phase8InternalEvidenceValidationError,
)
from protoem_ct.external.real_development_runner import (
    build_phase8_real_development_input_binding,
    phase8_real_development_input_binding_to_json,
)

requires_baseline_environment = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("monai") is None,
    reason="Phase 8 bounded pilot requires the isolated baseline environment.",
)

GIT_COMMIT = "a" * 40
CREATED_AT_UTC = "2026-08-05T00:00:00Z"
PACKAGE_VERSIONS = {"torch": "2.2.0", "monai": "1.3.0"}
VOLUME_SHAPE = (100, 100, 70)


# ---------------------------------------------------------------------------
# Synthetic Phase 2 manifest/split/lesion-components fixtures
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
    # cases 1-3 -> train, 4-5 -> validation, 6 -> internal_test.
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
        split_policy_version="v1",
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


def _lesion_case_record(
    *,
    case: DatasetCaseRecord,
    partition: str,
    lesion_count: int,
) -> LesionComponentCaseRecord:
    if lesion_count == 0:
        return LesionComponentCaseRecord(
            anonymous_patient_id=case.anonymous_patient_id,
            anonymous_case_id=case.anonymous_case_id,
            partition=partition,
            analysis_performed=True,
            connectivity=26,
            tumor_label_value=RAW_LABEL_TUMOR_VALUE,
            voxel_volume_mm3=1.0,
            tumor_voxel_count=0,
            tumor_physical_volume_mm3=0.0,
            lesion_count=0,
            lesions=(),
            qa_passed=True,
            failure_reasons=(),
        )
    lesions = tuple(
        LesionSummaryRecord(
            lesion_index=lesion_index,
            voxel_count=10,
            physical_volume_mm3=10.0,
        )
        for lesion_index in range(1, lesion_count + 1)
    )
    tumor_voxel_count = sum(lesion.voxel_count for lesion in lesions)
    tumor_physical_volume_mm3 = sum(lesion.physical_volume_mm3 for lesion in lesions)
    return LesionComponentCaseRecord(
        anonymous_patient_id=case.anonymous_patient_id,
        anonymous_case_id=case.anonymous_case_id,
        partition=partition,
        analysis_performed=True,
        connectivity=26,
        tumor_label_value=RAW_LABEL_TUMOR_VALUE,
        voxel_volume_mm3=1.0,
        tumor_voxel_count=tumor_voxel_count,
        tumor_physical_volume_mm3=tumor_physical_volume_mm3,
        lesion_count=lesion_count,
        lesions=lesions,
        qa_passed=True,
        failure_reasons=(),
    )


def _lesion_components(
    manifest: DatasetManifest, split: DevelopmentSplitManifest
) -> LesionComponentsArtifact:
    partition_by_case_id = {
        assignment.anonymous_case_id: assignment.partition for assignment in split.assignments
    }
    case_records: list[LesionComponentCaseRecord] = []
    for index, case in enumerate(manifest.cases, start=1):
        partition = partition_by_case_id[case.anonymous_case_id]
        # Case 2 (the first train case) is the tumor-positive train case; case 1
        # (also train) is the empty-target train case; every other case is
        # empty-target too, so selection is unambiguous.
        lesion_count = 2 if index == 2 else 0
        case_records.append(
            _lesion_case_record(case=case, partition=partition, lesion_count=lesion_count)
        )
    case_records_tuple = tuple(sorted(case_records, key=lambda record: record.anonymous_case_id))
    without_hash = LesionComponentsArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=LESION_COMPONENTS_STAGE,
        created_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        config_hash="0" * 64,
        manifest_hash=manifest.manifest_hash,
        split_hash=split.split_hash,
        geometry_qa_artifact_hash="0" * 64,
        connectivity=26,
        case_count=len(case_records_tuple),
        analyzed_case_count=sum(1 for record in case_records_tuple if record.analysis_performed),
        skipped_case_count=sum(1 for record in case_records_tuple if not record.analysis_performed),
        case_records=case_records_tuple,
        lesion_artifact_hash="0" * 64,
    )
    return replace(without_hash, lesion_artifact_hash=hash_lesion_components(without_hash))


def _write_manifest_split_lesions(
    tmp_path: Path,
) -> tuple[Path, Path, Path, DatasetManifest, DevelopmentSplitManifest, LesionComponentsArtifact]:
    manifest = _manifest()
    split = _split(manifest)
    lesion_components = _lesion_components(manifest, split)
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    lesion_components_path = tmp_path / "lesion_components.json"
    phase2_artifact_to_json(manifest, manifest_path)
    phase2_artifact_to_json(split, split_path)
    phase2_artifact_to_json(lesion_components, lesion_components_path)
    return (
        manifest_path,
        split_path,
        lesion_components_path,
        manifest,
        split,
        lesion_components,
    )


def _reference(role: str, artifact_hash: str) -> dict[str, Any]:
    return {
        "schema_name": f"phase8_{role}",
        "schema_version": "v1",
        "artifact_hash": artifact_hash,
        "artifact_role": role,
    }


def _rehash(mapping: dict[str, Any], hash_field: str) -> dict[str, Any]:
    from copy import deepcopy

    result = deepcopy(mapping)
    payload = deepcopy(result)
    payload.pop(hash_field, None)
    result[hash_field] = sha256_json(payload)
    return result


def _preprocessing_decision_mapping(*, manifest_hash: str, split_hash: str) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "schema_name": "phase8_preprocessing_decision",
        "schema_version": "v1",
        "candidate_id": APPROVED_CANDIDATE_ID,
        "development_manifest_hash": manifest_hash,
        "development_split_hash": split_hash,
        "preprocessing_config_schema_name": "phase8_preprocessing_config",
        "preprocessing_config_schema_version": "v1",
        "preprocessing_config_hash": "d" * 64,
        "fit_scope": "no_data_dependent_fit",
        "fit_artifact_reference": None,
        "image_interpolation_policy": "none",
        "label_interpolation_policy": "none",
        "orientation_policy_reference": _reference("orientation_policy", "1" * 64),
        "spacing_policy_reference": _reference("spacing_policy", "2" * 64),
        "intensity_policy_reference": _reference("intensity_policy", "3" * 64),
        "crop_roi_policy_reference": _reference("crop_roi_policy", "4" * 64),
        "normalization_policy_reference": _reference("normalization_policy", "5" * 64),
        "originating_git_commit": GIT_COMMIT,
        "no_external_data": True,
        "evidence_status": "freeze_ready",
    }
    return _rehash(mapping, "preprocessing_decision_hash")


def _candidate_inventory_mapping(*, preprocessing_decision_hash: str) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "schema_name": "phase8_fixed_candidate_inventory",
        "schema_version": "v1",
        "inventory_state": "freeze_ready",
        "candidate_set_locked": True,
        "selection_started": True,
        "supports_single_candidate_without_comparison": True,
        "candidates": [
            {
                "candidate_id": APPROVED_CANDIDATE_ID,
                "model_family": APPROVED_MODEL_FAMILY,
                "method_identity": "monai_segresnet_baseline_v1",
                "training_adaptation_mode": "baseline_training",
                "training_config_reference": _reference("training_config", "6" * 64),
                "preprocessing_evidence_hash": preprocessing_decision_hash,
                "fixed_seeds": [1729],
                "executable_readiness_state": "executable",
                "failure_handling_policy": "single_candidate_no_comparison",
                "uses_external_artifacts": False,
            }
        ],
    }
    return _rehash(mapping, "inventory_hash")


def _write_prerequisites(
    tmp_path: Path,
    *,
    manifest_path: Path,
    split_path: Path,
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
) -> dict[str, Path | str]:
    prereq_dir = tmp_path / "prerequisites"
    prereq_dir.mkdir()

    input_binding = build_phase8_real_development_input_binding(
        manifest_path=manifest_path,
        split_path=split_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        approved_development_artifact_set_identity="phase8-substage4b-v1",
    )
    input_binding_path = prereq_dir / "phase8_real_development_input_binding.json"
    input_binding_path.write_bytes(phase8_real_development_input_binding_to_json(input_binding))

    preprocessing_mapping = _preprocessing_decision_mapping(
        manifest_hash=manifest.manifest_hash, split_hash=split.split_hash
    )
    preprocessing_decision_path = prereq_dir / "phase8_preprocessing_decision.json"
    preprocessing_decision_path.write_text(
        json.dumps(preprocessing_mapping, sort_keys=True), encoding="utf-8"
    )

    inventory_mapping = _candidate_inventory_mapping(
        preprocessing_decision_hash=preprocessing_mapping["preprocessing_decision_hash"]
    )
    candidate_inventory_path = prereq_dir / "phase8_fixed_candidate_inventory.json"
    candidate_inventory_path.write_text(
        json.dumps(inventory_mapping, sort_keys=True), encoding="utf-8"
    )

    return {
        "input_binding_path": input_binding_path,
        "candidate_inventory_path": candidate_inventory_path,
        "preprocessing_decision_path": preprocessing_decision_path,
        "expected_input_binding_sha256": sha256_file(input_binding_path),
        "expected_candidate_inventory_sha256": sha256_file(candidate_inventory_path),
        "expected_preprocessing_decision_sha256": sha256_file(preprocessing_decision_path),
    }


def _write_dataset_root(tmp_path: Path, *, shape: tuple[int, int, int] = VOLUME_SHAPE) -> Path:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images").mkdir(parents=True)
    (dataset_root / "labels").mkdir(parents=True)
    affine = np.eye(4, dtype=np.float64)
    for index in range(1, 7):
        rng = np.random.default_rng(index)
        image_array = rng.uniform(-500.0, 500.0, size=shape).astype(np.float32)
        label_array = np.zeros(shape, dtype=np.uint8)
        if index == 2:
            # Tumor-positive train case: a block of tumor (raw==2) plus some liver (raw==1).
            label_array[10:20, 10:20, 10:20] = RAW_LABEL_TUMOR_VALUE
            label_array[30:40, 30:40, 10:20] = 1
        image = nib.Nifti1Image(image_array, affine)  # type: ignore[no-untyped-call]
        label = nib.Nifti1Image(label_array, affine)  # type: ignore[no-untyped-call]
        nib.save(image, str(dataset_root / "images" / f"case-{index:04d}.nii.gz"))
        nib.save(label, str(dataset_root / "labels" / f"case-{index:04d}.nii.gz"))
    return dataset_root


def _minimal_supervisor_kwargs(tmp_path: Path) -> dict[str, Any]:
    """Syntactically valid arguments for ``run_phase8_bounded_pilot_with_watchdog``.

    Used only by tests that override ``_target`` with a fake, synthetic callable
    which never actually opens any of these paths -- so the paths need not
    exist and the hashes need not correspond to any real file. Real,
    fixture-backed arguments (``_write_manifest_split_lesions`` etc.) are used
    instead wherever a test exercises the real ``run_phase8_bounded_pilot``.
    """

    return {
        "manifest_path": tmp_path / "manifest.json",
        "split_path": tmp_path / "split.json",
        "lesion_components_path": tmp_path / "lesion_components.json",
        "expected_manifest_sha256": "0" * 64,
        "expected_split_sha256": "0" * 64,
        "expected_lesion_components_sha256": "0" * 64,
        "input_binding_path": tmp_path / "input_binding.json",
        "candidate_inventory_path": tmp_path / "candidate_inventory.json",
        "preprocessing_decision_path": tmp_path / "preprocessing_decision.json",
        "expected_input_binding_sha256": "0" * 64,
        "expected_candidate_inventory_sha256": "0" * 64,
        "expected_preprocessing_decision_sha256": "0" * 64,
        "dataset_root": tmp_path / "dataset",
        "output_root": tmp_path / "output",
        "repository_root": tmp_path / "repo",
        "git_commit": GIT_COMMIT,
        "package_versions": PACKAGE_VERSIONS,
        "seed": 1729,
    }


def _selected_case(
    *, case_id: str, patient_id: str, partition: str, role: str
) -> Phase8BoundedPilotSelectedCase:
    return Phase8BoundedPilotSelectedCase(
        anonymous_patient_id=patient_id,
        anonymous_case_id=case_id,
        partition=partition,
        pilot_role=role,
        relative_image_path=f"images/{case_id}.nii.gz",
        relative_label_path=f"labels/{case_id}.nii.gz",
    )


def _scan_for_forbidden_strings(value: Any, forbidden: tuple[str, ...]) -> list[str]:
    """Recursively scan a JSON-like structure for any forbidden substrings."""

    hits: list[str] = []
    text = json.dumps(value, sort_keys=True)
    for needle in forbidden:
        if needle in text:
            hits.append(needle)
    return hits


def _scan_for_absolute_path_like_strings(value: Any) -> list[str]:
    """Recursively scan a JSON-like structure for strings that look like absolute paths."""

    hits: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            if (
                node.startswith("/")
                or node.startswith("\\")
                or (len(node) > 1 and node[1] == ":" and node[0].isalpha())
            ):
                hits.append(node)
        elif isinstance(node, dict):
            for item in node.values():
                _walk(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(value)
    return hits


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_builds_with_hard_coded_limits() -> None:
    config = build_phase8_bounded_pilot_config(
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
    )
    assert config.device_type == "cpu"
    assert config.max_optimizer_steps == REQUIRED_MAX_OPTIMIZER_STEPS == 20
    assert config.patch_size == REQUIRED_PATCH_SIZE
    assert config.sliding_window_roi_size == REQUIRED_SLIDING_WINDOW_ROI_SIZE
    assert config.sliding_window_overlap == REQUIRED_SLIDING_WINDOW_OVERLAP
    assert (
        config.positive_patches_from_positive_case == REQUIRED_POSITIVE_PATCHES_FROM_POSITIVE_CASE
    )
    assert (
        config.negative_patches_from_positive_case == REQUIRED_NEGATIVE_PATCHES_FROM_POSITIVE_CASE
    )
    assert config.negative_patches_from_empty_case == REQUIRED_NEGATIVE_PATCHES_FROM_EMPTY_CASE
    assert config.resume_disabled is True
    assert config.hyperparameter_search_prohibited is True
    assert config.candidate_comparison_prohibited is True
    assert config.augmentation_disabled is True
    assert config.wall_clock_limit_seconds == DEFAULT_WALL_CLOCK_LIMIT_SECONDS
    assert config.config_hash


def test_config_cannot_broaden_scope_via_replace() -> None:
    config = build_phase8_bounded_pilot_config(
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
    )
    with pytest.raises(Phase8BoundedPilotConfigError):
        replace(config, max_optimizer_steps=1_000_000)
    with pytest.raises(Phase8BoundedPilotConfigError):
        replace(config, device_type="cuda")
    with pytest.raises(Phase8BoundedPilotConfigError):
        replace(config, hyperparameter_search_prohibited=False)


@pytest.mark.parametrize(
    ("field_name", "broadened_value"),
    [
        ("amp_enabled", True),
        ("train_case_count", 3),
        ("validation_case_count", 2),
        ("batch_size", 4),
        ("gradient_accumulation_steps", 2),
        ("max_epochs", 5),
        ("num_workers", 4),
        ("positive_patches_from_positive_case", 20),
        ("negative_patches_from_positive_case", 0),
        ("negative_patches_from_empty_case", 0),
        ("optimizer_name", "sgd"),
        ("learning_rate", 1.0),
        ("weight_decay", 0.0),
        ("loss_name", "cross_entropy"),
        ("class_weighting", "inverse_frequency"),
        ("sliding_window_overlap", 0.0),
        ("sliding_window_batch_size", 4),
        ("hu_window_min", -2000.0),
        ("hu_window_max", 2000.0),
        ("resume_disabled", False),
        ("candidate_comparison_prohibited", False),
        ("augmentation_disabled", False),
    ],
)
def test_config_rejects_every_hard_coded_field_broadened_via_replace(
    field_name: str, broadened_value: object
) -> None:
    """Every hard compute limit must independently reject a broadening `replace()`.

    A plausible broken implementation could validate only a subset of fields
    in ``Phase8BoundedPilotConfig.__post_init__`` (as the reconstructed test
    file originally did, covering only 3 of ~20 fields); this parametrized
    test proves every field in the E-section training-configuration contract
    is actually enforced, not merely declared.
    """

    config = build_phase8_bounded_pilot_config(
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
    )
    with pytest.raises(Phase8BoundedPilotConfigError):
        replace(cast(Any, config), **{field_name: broadened_value})


def test_config_rejects_broadened_patch_size_and_roi_via_replace() -> None:
    config = build_phase8_bounded_pilot_config(
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
    )
    with pytest.raises(Phase8BoundedPilotConfigError):
        replace(config, patch_size=(128, 128, 64))
    with pytest.raises(Phase8BoundedPilotConfigError):
        replace(config, sliding_window_roi_size=(192, 192, 128))


def test_config_rejects_nonpositive_wall_clock_limit() -> None:
    with pytest.raises(Phase8BoundedPilotConfigError):
        build_phase8_bounded_pilot_config(
            seed=1729,
            originating_git_commit=GIT_COMMIT,
            package_versions=PACKAGE_VERSIONS,
            wall_clock_limit_seconds=0.0,
        )


def test_config_accepts_tiny_wall_clock_limit_for_watchdog_tests() -> None:
    config = build_phase8_bounded_pilot_config(
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
        wall_clock_limit_seconds=0.0000001,
    )
    assert config.wall_clock_limit_seconds == 0.0000001


# ---------------------------------------------------------------------------
# Case selection (pure metadata, no pixel access)
# ---------------------------------------------------------------------------


def test_case_selection_picks_positive_empty_and_validation_cases(tmp_path: Path) -> None:
    (
        _manifest_path,
        _split_path,
        _lesion_components_path,
        manifest,
        split,
        lesion_components,
    ) = _write_manifest_split_lesions(tmp_path)

    train_positive, train_empty, validation = select_phase8_bounded_pilot_cases(
        manifest=manifest, split=split, lesion_components=lesion_components
    )
    assert train_positive.pilot_role == "train_positive"
    assert train_empty.pilot_role == "train_empty"
    assert validation.pilot_role == "validation"
    assert train_positive.anonymous_case_id != train_empty.anonymous_case_id
    assert train_positive.partition == "train"
    assert train_empty.partition == "train"
    assert validation.partition == "validation"
    # Case 2 is the only train case with lesion_count > 0 in this fixture.
    assert train_positive.anonymous_case_id == "anon-c0002"


def test_case_selection_blocked_without_opening_nifti_when_no_positive_case(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    split = _split(manifest)
    # No train case is tumor-positive: every lesion count is zero.
    case_records = tuple(
        _lesion_case_record(
            case=case,
            partition=next(
                a.partition
                for a in split.assignments
                if a.anonymous_case_id == case.anonymous_case_id
            ),
            lesion_count=0,
        )
        for case in manifest.cases
    )
    without_hash = LesionComponentsArtifact(
        schema_version=PHASE2_SCHEMA_VERSION,
        stage=LESION_COMPONENTS_STAGE,
        created_at_utc=CREATED_AT_UTC,
        git_commit=GIT_COMMIT,
        config_hash="0" * 64,
        manifest_hash=manifest.manifest_hash,
        split_hash=split.split_hash,
        geometry_qa_artifact_hash="0" * 64,
        connectivity=26,
        case_count=len(case_records),
        analyzed_case_count=len(case_records),
        skipped_case_count=0,
        case_records=case_records,
        lesion_artifact_hash="0" * 64,
    )
    lesion_components = replace(
        without_hash, lesion_artifact_hash=hash_lesion_components(without_hash)
    )

    with pytest.raises(Phase8BoundedPilotCaseSelectionError):
        select_phase8_bounded_pilot_cases(
            manifest=manifest, split=split, lesion_components=lesion_components
        )


# ---------------------------------------------------------------------------
# Output-root path safety
# ---------------------------------------------------------------------------


def test_output_root_rejects_symlink(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    real_target = tmp_path / "real_target"
    real_target.mkdir()
    symlink_root = tmp_path / "symlinked_output"
    symlink_root.symlink_to(real_target, target_is_directory=True)

    with pytest.raises(Phase8BoundedPilotOutputRootError):
        validate_phase8_bounded_pilot_output_root(symlink_root, repository_root=repository_root)


def test_output_root_rejects_existing_nonempty_directory(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "existing_output"
    output_root.mkdir()
    (output_root / "leftover.txt").write_text("x", encoding="utf-8")

    with pytest.raises(Phase8BoundedPilotOutputRootError):
        validate_phase8_bounded_pilot_output_root(output_root, repository_root=repository_root)


def test_output_root_rejects_path_inside_repository(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = repository_root / "nested_output"

    with pytest.raises(Phase8BoundedPilotOutputRootError):
        validate_phase8_bounded_pilot_output_root(output_root, repository_root=repository_root)


def test_output_root_accepts_valid_nonexistent_external_root(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "external_output"

    paths = validate_phase8_bounded_pilot_output_root(output_root, repository_root=repository_root)
    assert paths.output_root == output_root.resolve()
    assert not paths.output_root.exists()


# ---------------------------------------------------------------------------
# Access ledger / checkpoint metadata / internal evidence rejection
# ---------------------------------------------------------------------------


def test_access_ledger_contains_anonymous_ids_only_and_six_accesses() -> None:
    train_positive = _selected_case(
        case_id="anon-c0002", patient_id="anon-p0002", partition="train", role="train_positive"
    )
    train_empty = _selected_case(
        case_id="anon-c0001", patient_id="anon-p0001", partition="train", role="train_empty"
    )
    validation = _selected_case(
        case_id="anon-c0004", patient_id="anon-p0004", partition="validation", role="validation"
    )
    ledger = build_phase8_bounded_pilot_access_ledger(
        train_positive_case=train_positive,
        train_empty_case=train_empty,
        validation_case=validation,
    )
    assert ledger.internal_test_opened is False
    assert ledger.external_data_opened is False
    assert ledger.prior_checkpoint_loaded is False
    assert ledger.file_access_count == 6
    assert len(ledger.records) == 3
    payload = phase8_bounded_pilot_access_ledger_to_dict(ledger)
    # Raw filenames and absolute paths must never leak into the ledger; only
    # anonymous IDs (already asserted present via file_access_count above) are
    # permitted.
    assert _scan_for_absolute_path_like_strings(payload) == []
    forbidden = _scan_for_forbidden_strings(
        payload, ("case-0001.nii.gz", "case-0002.nii.gz", "case-0004.nii.gz", "/", "images/")
    )
    assert forbidden == []


def test_checkpoint_metadata_ineligibility_flags_are_hard_set() -> None:
    config = build_phase8_bounded_pilot_config(
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
    )
    metadata = build_phase8_bounded_pilot_checkpoint_metadata(
        config=config,
        development_manifest_hash="1" * 64,
        development_split_hash="2" * 64,
        checkpoint_sha256="3" * 64,
        checkpoint_byte_size=1024,
    )
    assert metadata.pilot_only is True
    assert metadata.tiny_verification_only is False
    assert metadata.selection_eligible is False
    assert metadata.definitive_training is False
    assert metadata.scientific_metric_eligible is False
    assert metadata.checkpoint_metadata.freeze_eligible is False
    assert metadata.checkpoint_metadata.completion_status == "synthetic_smoke"
    payload = phase8_bounded_pilot_checkpoint_metadata_to_dict(metadata)
    assert payload["wrapper_hash"] == metadata.wrapper_hash


def test_internal_evidence_package_rejects_freeze_eligible_pilot_checkpoint() -> None:
    config = build_phase8_bounded_pilot_config(
        seed=1729,
        originating_git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
    )
    metadata = build_phase8_bounded_pilot_checkpoint_metadata(
        config=config,
        development_manifest_hash="1" * 64,
        development_split_hash="2" * 64,
        checkpoint_sha256="3" * 64,
        checkpoint_byte_size=1024,
    )
    # The bounded pilot's own checkpoint metadata uses completion_status
    # "synthetic_smoke". Attempting to force freeze_eligible=True onto a
    # checkpoint that is not "completed" must be structurally rejected by
    # Phase8CheckpointMetadata itself -- this is what actually prevents a
    # pilot-only checkpoint from ever becoming freeze eligible.
    with pytest.raises(Phase8InternalEvidenceValidationError):
        replace(metadata.checkpoint_metadata, freeze_eligible=True)
    assert isinstance(metadata.checkpoint_metadata, Phase8CheckpointMetadata)


# ---------------------------------------------------------------------------
# Sampling schedule
# ---------------------------------------------------------------------------


def _positive_label_array(shape: tuple[int, int, int] = VOLUME_SHAPE) -> np.ndarray:
    label = np.zeros(shape, dtype=np.uint8)
    label[10:30, 10:30, 10:30] = RAW_LABEL_TUMOR_VALUE
    label[40:60, 40:60, 10:30] = 1
    return label


def _empty_label_array(shape: tuple[int, int, int] = VOLUME_SHAPE) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def test_sampling_schedule_produces_exact_10_10_accounting() -> None:
    schedule = build_phase8_bounded_pilot_sampling_schedule(
        positive_case_label=_positive_label_array(),
        empty_case_label=_empty_label_array(),
        seed=1729,
    )
    assert len(schedule) == REQUIRED_TOTAL_PATCHES == 20
    positive_count = sum(1 for sample in schedule if sample.is_positive)
    assert positive_count == REQUIRED_POSITIVE_PATCHES_FROM_POSITIVE_CASE == 10
    assert len(schedule) - positive_count == 10
    train_positive_negative = sum(
        1 for sample in schedule if sample.pilot_role == "train_positive" and not sample.is_positive
    )
    train_empty_negative = sum(
        1 for sample in schedule if sample.pilot_role == "train_empty" and not sample.is_positive
    )
    assert train_positive_negative == REQUIRED_NEGATIVE_PATCHES_FROM_POSITIVE_CASE == 5
    assert train_empty_negative == REQUIRED_NEGATIVE_PATCHES_FROM_EMPTY_CASE == 5


def test_sampling_schedule_is_deterministic_across_calls() -> None:
    schedule_a = build_phase8_bounded_pilot_sampling_schedule(
        positive_case_label=_positive_label_array(),
        empty_case_label=_empty_label_array(),
        seed=1729,
    )
    schedule_b = build_phase8_bounded_pilot_sampling_schedule(
        positive_case_label=_positive_label_array(),
        empty_case_label=_empty_label_array(),
        seed=1729,
    )
    assert schedule_a == schedule_b


def test_sampling_schedule_fails_closed_when_too_few_foreground_voxels() -> None:
    sparse_label = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
    sparse_label[0, 0, 0] = RAW_LABEL_TUMOR_VALUE  # only one foreground voxel; 10 are required.
    with pytest.raises(Phase8BoundedPilotSamplingScheduleError):
        build_phase8_bounded_pilot_sampling_schedule(
            positive_case_label=sparse_label,
            empty_case_label=_empty_label_array(),
            seed=1729,
        )


# ---------------------------------------------------------------------------
# Raw geometry validation
# ---------------------------------------------------------------------------


def test_geometry_validation_accepts_ras_oriented_pair(tmp_path: Path) -> None:
    affine = np.eye(4, dtype=np.float64)
    image = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.zeros(VOLUME_SHAPE, dtype=np.float32), affine
    )
    label = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.zeros(VOLUME_SHAPE, dtype=np.uint8), affine
    )
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(image, str(image_path))
    nib.save(label, str(label_path))

    _validate_raw_case_pair_geometry(image_path, label_path)


def test_geometry_validation_rejects_non_ras_orientation(tmp_path: Path) -> None:
    from protoem_ct.external.definitive_training_pilot import Phase8BoundedPilotOrientationError

    lps_affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    image = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.zeros(VOLUME_SHAPE, dtype=np.float32), lps_affine
    )
    label = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.zeros(VOLUME_SHAPE, dtype=np.uint8), lps_affine
    )
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(image, str(image_path))
    nib.save(label, str(label_path))

    with pytest.raises(Phase8BoundedPilotOrientationError):
        _validate_raw_case_pair_geometry(image_path, label_path)


def test_geometry_validation_rejects_shape_mismatch(tmp_path: Path) -> None:
    from protoem_ct.external.definitive_training_pilot import Phase8BoundedPilotDataError

    affine = np.eye(4, dtype=np.float64)
    image = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.zeros(VOLUME_SHAPE, dtype=np.float32), affine
    )
    mismatched_shape = (VOLUME_SHAPE[0] + 1, VOLUME_SHAPE[1], VOLUME_SHAPE[2])
    label = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.zeros(mismatched_shape, dtype=np.uint8), affine
    )
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(image, str(image_path))
    nib.save(label, str(label_path))

    with pytest.raises(Phase8BoundedPilotDataError):
        _validate_raw_case_pair_geometry(image_path, label_path)


# ---------------------------------------------------------------------------
# Source-level scientific-metric guard
# ---------------------------------------------------------------------------


def test_no_forbidden_scientific_metric_field_names_in_source() -> None:
    """The pilot module must never *compute* a scientific segmentation metric.

    The module's own docstrings legitimately *mention* metric names (to
    document that they are never computed), so this checks for the absence
    of metric-computation call sites/imports, not for the absence of the bare
    words "dice" or "hd95" anywhere in prose.
    """

    import protoem_ct.external.definitive_training_pilot as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "protoem_ct.evaluation" not in source
    for forbidden in (
        "dice_from_counts",
        "compute_binary_confusion_counts",
        "evaluate_predictions",
        "compute_binary_calibration_ece",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# CLI help and approval gating
# ---------------------------------------------------------------------------


def test_cli_help_mentions_pilot_only_bounded_and_approval() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run-phase8-bounded-real-development-pilot", "--help"])
    assert result.exit_code == 0, result.output
    lowered = result.output.lower()
    assert "pilot" in lowered
    assert "bounded" in lowered
    assert "approve-bounded-pilot" in lowered
    assert "not definitive training" in lowered


def test_cli_refuses_without_approval_before_opening_any_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import protoem_ct.cli.main as cli_main_module

    supervised_entrypoint_calls: list[object] = []

    def _spy_supervised_entrypoint(**kwargs: object) -> object:
        supervised_entrypoint_calls.append(kwargs)
        raise AssertionError(
            "run_phase8_bounded_pilot_with_watchdog must not be called without approval"
        )

    monkeypatch.setattr(
        cli_main_module, "run_phase8_bounded_pilot_with_watchdog", _spy_supervised_entrypoint
    )

    (
        manifest_path,
        split_path,
        lesion_components_path,
        manifest,
        split,
        _lesion_components_obj,
    ) = _write_manifest_split_lesions(tmp_path)
    prereqs = _write_prerequisites(
        tmp_path,
        manifest_path=manifest_path,
        split_path=split_path,
        manifest=manifest,
        split=split,
    )
    dataset_root = tmp_path / "does-not-exist"
    output_root = tmp_path / "output"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-phase8-bounded-real-development-pilot",
            "--manifest-path",
            str(manifest_path),
            "--split-path",
            str(split_path),
            "--lesion-components-path",
            str(lesion_components_path),
            "--expected-manifest-sha256",
            sha256_file(manifest_path),
            "--expected-split-sha256",
            sha256_file(split_path),
            "--expected-lesion-components-sha256",
            sha256_file(lesion_components_path),
            "--input-binding-path",
            str(prereqs["input_binding_path"]),
            "--candidate-inventory-path",
            str(prereqs["candidate_inventory_path"]),
            "--preprocessing-decision-path",
            str(prereqs["preprocessing_decision_path"]),
            "--expected-input-binding-sha256",
            str(prereqs["expected_input_binding_sha256"]),
            "--expected-candidate-inventory-sha256",
            str(prereqs["expected_candidate_inventory_sha256"]),
            "--expected-preprocessing-decision-sha256",
            str(prereqs["expected_preprocessing_decision_sha256"]),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--git-commit",
            GIT_COMMIT,
            "--package-versions-json",
            json.dumps(PACKAGE_VERSIONS),
        ],
    )
    assert result.exit_code != 0
    assert not output_root.exists()
    assert supervised_entrypoint_calls == [], (
        "the CLI must refuse before ever calling the supervised entry point, "
        "so --approve-bounded-pilot gating is not weakened or bypassed by the "
        "new process-supervision code path"
    )


# ---------------------------------------------------------------------------
# Prerequisites reconstruction / cross-check (metadata only, no NIfTI access)
# ---------------------------------------------------------------------------


def test_prerequisites_mismatch_blocks_before_any_nifti_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from protoem_ct.external.definitive_training_pilot import run_phase8_bounded_pilot

    (
        manifest_path,
        split_path,
        lesion_components_path,
        manifest,
        split,
        _lesion_components_obj,
    ) = _write_manifest_split_lesions(tmp_path)
    prereqs = _write_prerequisites(
        tmp_path,
        manifest_path=manifest_path,
        split_path=split_path,
        manifest=manifest,
        split=split,
    )
    # Existing but empty: the point of this test is that a prerequisites
    # mismatch blocks before any NIfTI file is opened, not that the dataset
    # root itself is invalid, so the dataset root must pass path validation.
    dataset_root = tmp_path / "dataset-not-inspected"
    dataset_root.mkdir()
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    opened_nifti_paths: list[str] = []
    original_load = nib.load

    def _spy_load(path: str, *args: object, **kwargs: object) -> object:
        opened_nifti_paths.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", _spy_load)

    with pytest.raises(Phase8BoundedPilotPrerequisitesError):
        run_phase8_bounded_pilot(
            manifest_path=manifest_path,
            split_path=split_path,
            lesion_components_path=lesion_components_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_split_sha256=sha256_file(split_path),
            expected_lesion_components_sha256=sha256_file(lesion_components_path),
            input_binding_path=prereqs["input_binding_path"],  # type: ignore[arg-type]
            candidate_inventory_path=prereqs["candidate_inventory_path"],  # type: ignore[arg-type]
            preprocessing_decision_path=prereqs["preprocessing_decision_path"],  # type: ignore[arg-type]
            expected_input_binding_sha256="0" * 64,  # deliberately wrong.
            expected_candidate_inventory_sha256=prereqs[  # type: ignore[arg-type]
                "expected_candidate_inventory_sha256"
            ],
            expected_preprocessing_decision_sha256=prereqs[  # type: ignore[arg-type]
                "expected_preprocessing_decision_sha256"
            ],
            dataset_root=dataset_root,
            output_root=output_root,
            repository_root=repository_root,
            git_commit=GIT_COMMIT,
            package_versions=PACKAGE_VERSIONS,
            seed=1729,
        )
    assert opened_nifti_paths == []
    assert not output_root.exists()


# ---------------------------------------------------------------------------
# Watchdog (in-process elapsed-time check)
# ---------------------------------------------------------------------------


def test_watchdog_triggers_blocked_with_synthetic_tiny_limit_and_no_publication(
    tmp_path: Path,
) -> None:
    from protoem_ct.external.definitive_training_pilot import run_phase8_bounded_pilot

    (
        manifest_path,
        split_path,
        lesion_components_path,
        manifest,
        split,
        _lesion_components_obj,
    ) = _write_manifest_split_lesions(tmp_path)
    prereqs = _write_prerequisites(
        tmp_path,
        manifest_path=manifest_path,
        split_path=split_path,
        manifest=manifest,
        split=split,
    )
    dataset_root = _write_dataset_root(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    with pytest.raises(Phase8BoundedPilotWatchdogTimeoutError):
        run_phase8_bounded_pilot(
            manifest_path=manifest_path,
            split_path=split_path,
            lesion_components_path=lesion_components_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_split_sha256=sha256_file(split_path),
            expected_lesion_components_sha256=sha256_file(lesion_components_path),
            input_binding_path=prereqs["input_binding_path"],  # type: ignore[arg-type]
            candidate_inventory_path=prereqs["candidate_inventory_path"],  # type: ignore[arg-type]
            preprocessing_decision_path=prereqs["preprocessing_decision_path"],  # type: ignore[arg-type]
            expected_input_binding_sha256=prereqs["expected_input_binding_sha256"],  # type: ignore[arg-type]
            expected_candidate_inventory_sha256=prereqs[  # type: ignore[arg-type]
                "expected_candidate_inventory_sha256"
            ],
            expected_preprocessing_decision_sha256=prereqs[  # type: ignore[arg-type]
                "expected_preprocessing_decision_sha256"
            ],
            dataset_root=dataset_root,
            output_root=output_root,
            repository_root=repository_root,
            git_commit=GIT_COMMIT,
            package_versions=PACKAGE_VERSIONS,
            seed=1729,
            wall_clock_limit_seconds=0.0000001,
        )
    assert not output_root.exists()


# ---------------------------------------------------------------------------
# Full synthetic pipeline (requires torch/monai)
# ---------------------------------------------------------------------------


@requires_baseline_environment
def test_full_synthetic_pilot_produces_all_artifacts_and_no_leakage(tmp_path: Path) -> None:
    from protoem_ct.external.definitive_training_pilot import (
        PHASE8_BOUNDED_PILOT_ACCESS_LEDGER_FILENAME,
        PHASE8_BOUNDED_PILOT_CHECKPOINT_METADATA_FILENAME,
        PHASE8_BOUNDED_PILOT_CHECKPOINT_SUBDIRECTORY_NAME,
        PHASE8_BOUNDED_PILOT_CONFIG_FILENAME,
        PHASE8_BOUNDED_PILOT_SUMMARY_FILENAME,
        run_phase8_bounded_pilot,
    )

    (
        manifest_path,
        split_path,
        lesion_components_path,
        manifest,
        split,
        _lesion_components_obj,
    ) = _write_manifest_split_lesions(tmp_path)
    prereqs = _write_prerequisites(
        tmp_path,
        manifest_path=manifest_path,
        split_path=split_path,
        manifest=manifest,
        split=split,
    )
    dataset_root = _write_dataset_root(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    result = run_phase8_bounded_pilot(
        manifest_path=manifest_path,
        split_path=split_path,
        lesion_components_path=lesion_components_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        expected_lesion_components_sha256=sha256_file(lesion_components_path),
        input_binding_path=prereqs["input_binding_path"],  # type: ignore[arg-type]
        candidate_inventory_path=prereqs["candidate_inventory_path"],  # type: ignore[arg-type]
        preprocessing_decision_path=prereqs["preprocessing_decision_path"],  # type: ignore[arg-type]
        expected_input_binding_sha256=prereqs["expected_input_binding_sha256"],  # type: ignore[arg-type]
        expected_candidate_inventory_sha256=prereqs[  # type: ignore[arg-type]
            "expected_candidate_inventory_sha256"
        ],
        expected_preprocessing_decision_sha256=prereqs[  # type: ignore[arg-type]
            "expected_preprocessing_decision_sha256"
        ],
        dataset_root=dataset_root,
        output_root=output_root,
        repository_root=repository_root,
        git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
        seed=1729,
    )

    assert (output_root / PHASE8_BOUNDED_PILOT_CONFIG_FILENAME).is_file()
    assert (output_root / PHASE8_BOUNDED_PILOT_ACCESS_LEDGER_FILENAME).is_file()
    assert (output_root / PHASE8_BOUNDED_PILOT_CHECKPOINT_METADATA_FILENAME).is_file()
    assert (output_root / PHASE8_BOUNDED_PILOT_SUMMARY_FILENAME).is_file()
    checkpoint_dir = output_root / PHASE8_BOUNDED_PILOT_CHECKPOINT_SUBDIRECTORY_NAME
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoint_files) == 1

    summary = json.loads(
        (output_root / PHASE8_BOUNDED_PILOT_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert summary["completion_status"] == "completed"
    assert summary["executed_step_count"] == REQUIRED_TOTAL_PATCHES == 20
    assert summary["positive_patch_count"] == 10
    assert summary["negative_patch_count"] == 10
    assert summary["checkpoint_round_trip_verified"] is True
    assert summary["pilot_only"] is True
    assert summary["definitive_training"] is False
    assert summary["freeze_eligible"] is False
    assert summary["selection_eligible"] is False
    assert summary["scientific_metric_eligible"] is False
    assert summary["sliding_window_count"] >= 1
    assert _scan_for_absolute_path_like_strings(summary) == []

    for name in (
        PHASE8_BOUNDED_PILOT_CONFIG_FILENAME,
        PHASE8_BOUNDED_PILOT_ACCESS_LEDGER_FILENAME,
        PHASE8_BOUNDED_PILOT_CHECKPOINT_METADATA_FILENAME,
        PHASE8_BOUNDED_PILOT_SUMMARY_FILENAME,
    ):
        payload = json.loads((output_root / name).read_text(encoding="utf-8"))
        forbidden = _scan_for_forbidden_strings(
            payload,
            (
                '"dice"',
                '"iou"',
                '"hd95"',
                '"nsd"',
                "lesion_recall",
                "lesion_precision",
                "lesion_f1",
                "false_positive_lesion",
                "volume_error",
                "case-0001.nii.gz",
                "case-0002.nii.gz",
                "case-0004.nii.gz",
                str(repository_root),
            ),
        )
        assert forbidden == [], f"{name} contains forbidden metric keys: {forbidden}"
        assert _scan_for_absolute_path_like_strings(payload) == []

    assert result.config_hash
    assert result.access_ledger_hash
    assert result.checkpoint_metadata_hash
    assert result.summary_hash


@requires_baseline_environment
def test_cli_smoke_execution_creates_expected_files_only_under_tmp_path(tmp_path: Path) -> None:
    (
        manifest_path,
        split_path,
        lesion_components_path,
        manifest,
        split,
        _lesion_components_obj,
    ) = _write_manifest_split_lesions(tmp_path)
    prereqs = _write_prerequisites(
        tmp_path,
        manifest_path=manifest_path,
        split_path=split_path,
        manifest=manifest,
        split=split,
    )
    dataset_root = _write_dataset_root(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-phase8-bounded-real-development-pilot",
            "--approve-bounded-pilot",
            "--manifest-path",
            str(manifest_path),
            "--split-path",
            str(split_path),
            "--lesion-components-path",
            str(lesion_components_path),
            "--expected-manifest-sha256",
            sha256_file(manifest_path),
            "--expected-split-sha256",
            sha256_file(split_path),
            "--expected-lesion-components-sha256",
            sha256_file(lesion_components_path),
            "--input-binding-path",
            str(prereqs["input_binding_path"]),
            "--candidate-inventory-path",
            str(prereqs["candidate_inventory_path"]),
            "--preprocessing-decision-path",
            str(prereqs["preprocessing_decision_path"]),
            "--expected-input-binding-sha256",
            str(prereqs["expected_input_binding_sha256"]),
            "--expected-candidate-inventory-sha256",
            str(prereqs["expected_candidate_inventory_sha256"]),
            "--expected-preprocessing-decision-sha256",
            str(prereqs["expected_preprocessing_decision_sha256"]),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--git-commit",
            GIT_COMMIT,
            "--package-versions-json",
            json.dumps(PACKAGE_VERSIONS),
            "--repository-root",
            str(repository_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_root.is_dir()
    published_files = sorted(p.name for p in output_root.iterdir() if p.is_file())
    assert published_files == [
        "phase8_definitive_training_pilot_access_ledger.json",
        "phase8_definitive_training_pilot_checkpoint_metadata.json",
        "phase8_definitive_training_pilot_config.json",
        "phase8_definitive_training_pilot_summary.json",
    ]
    assert not any(repository_root.rglob("*.json"))


@requires_baseline_environment
def test_internal_test_case_is_never_opened_by_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from protoem_ct.external.definitive_training_pilot import run_phase8_bounded_pilot

    (
        manifest_path,
        split_path,
        lesion_components_path,
        manifest,
        split,
        _lesion_components_obj,
    ) = _write_manifest_split_lesions(tmp_path)
    internal_test_case_ids = {
        a.anonymous_case_id for a in split.assignments if a.partition == "internal_test"
    }
    assert internal_test_case_ids
    internal_test_basenames = {
        Path(record.relative_image_path).name
        for record in manifest.cases
        if record.anonymous_case_id in internal_test_case_ids
    } | {
        Path(record.relative_label_path).name
        for record in manifest.cases
        if record.anonymous_case_id in internal_test_case_ids
    }

    prereqs = _write_prerequisites(
        tmp_path,
        manifest_path=manifest_path,
        split_path=split_path,
        manifest=manifest,
        split=split,
    )
    dataset_root = _write_dataset_root(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    original_load = nib.load
    opened_names: list[str] = []

    def _spy_load(path: str, *args: object, **kwargs: object) -> object:
        opened_names.append(Path(path).name)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", _spy_load)

    run_phase8_bounded_pilot(
        manifest_path=manifest_path,
        split_path=split_path,
        lesion_components_path=lesion_components_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        expected_lesion_components_sha256=sha256_file(lesion_components_path),
        input_binding_path=prereqs["input_binding_path"],  # type: ignore[arg-type]
        candidate_inventory_path=prereqs["candidate_inventory_path"],  # type: ignore[arg-type]
        preprocessing_decision_path=prereqs["preprocessing_decision_path"],  # type: ignore[arg-type]
        expected_input_binding_sha256=prereqs["expected_input_binding_sha256"],  # type: ignore[arg-type]
        expected_candidate_inventory_sha256=prereqs[  # type: ignore[arg-type]
            "expected_candidate_inventory_sha256"
        ],
        expected_preprocessing_decision_sha256=prereqs[  # type: ignore[arg-type]
            "expected_preprocessing_decision_sha256"
        ],
        dataset_root=dataset_root,
        output_root=output_root,
        repository_root=repository_root,
        git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
        seed=1729,
    )

    for name in opened_names:
        assert name not in internal_test_basenames


@requires_baseline_environment
def test_deterministic_byte_identical_publication_across_two_output_roots(tmp_path: Path) -> None:
    from protoem_ct.external.definitive_training_pilot import run_phase8_bounded_pilot

    (
        manifest_path,
        split_path,
        lesion_components_path,
        manifest,
        split,
        _lesion_components_obj,
    ) = _write_manifest_split_lesions(tmp_path)
    prereqs = _write_prerequisites(
        tmp_path,
        manifest_path=manifest_path,
        split_path=split_path,
        manifest=manifest,
        split=split,
    )
    dataset_root = _write_dataset_root(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    output_root_a = tmp_path / "output_a"
    output_root_b = tmp_path / "output_b"

    kwargs = dict(
        manifest_path=manifest_path,
        split_path=split_path,
        lesion_components_path=lesion_components_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        expected_lesion_components_sha256=sha256_file(lesion_components_path),
        input_binding_path=prereqs["input_binding_path"],
        candidate_inventory_path=prereqs["candidate_inventory_path"],
        preprocessing_decision_path=prereqs["preprocessing_decision_path"],
        expected_input_binding_sha256=prereqs["expected_input_binding_sha256"],
        expected_candidate_inventory_sha256=prereqs["expected_candidate_inventory_sha256"],
        expected_preprocessing_decision_sha256=prereqs["expected_preprocessing_decision_sha256"],
        dataset_root=dataset_root,
        repository_root=repository_root,
        git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
        seed=1729,
    )

    result_a = run_phase8_bounded_pilot(output_root=output_root_a, **kwargs)  # type: ignore[arg-type]
    result_b = run_phase8_bounded_pilot(output_root=output_root_b, **kwargs)  # type: ignore[arg-type]

    # Config, access ledger, and checkpoint metadata contain no wall-clock timing and
    # must be byte-identical given identical synthetic inputs and a fixed seed.
    for name in (
        "phase8_definitive_training_pilot_config.json",
        "phase8_definitive_training_pilot_access_ledger.json",
        "phase8_definitive_training_pilot_checkpoint_metadata.json",
    ):
        text_a = (output_root_a / name).read_text(encoding="utf-8")
        text_b = (output_root_b / name).read_text(encoding="utf-8")
        assert text_a == text_b, f"{name} differed across two runs with identical inputs."

    # The summary embeds real wall-clock elapsed_seconds, so it is intentionally *not*
    # byte-identical; every other field (including the deterministic checkpoint hash) must be.
    summary_a = json.loads(
        (output_root_a / "phase8_definitive_training_pilot_summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary_b = json.loads(
        (output_root_b / "phase8_definitive_training_pilot_summary.json").read_text(
            encoding="utf-8"
        )
    )
    non_deterministic_fields = {"elapsed_seconds", "summary_hash"}
    for key in sorted(set(summary_a) | set(summary_b)):
        if key in non_deterministic_fields:
            continue
        assert summary_a[key] == summary_b[key], f"summary field {key!r} differed across runs."

    assert result_a.config_hash == result_b.config_hash
    assert result_a.access_ledger_hash == result_b.access_ledger_hash


# ---------------------------------------------------------------------------
# Process-level supervision (watchdog) -- run_phase8_bounded_pilot_with_watchdog
# ---------------------------------------------------------------------------
#
# These tests exercise the process-level supervisor in isolation, using
# synthetic, non-medical, fake ``_target`` callables in place of the real
# ``run_phase8_bounded_pilot``. They never touch real data or /Volumes, never
# run real training or inference, and never compute any scientific metric.
#
# ``_target`` and ``_multiprocessing_context`` are private testing-only seams
# on ``run_phase8_bounded_pilot_with_watchdog``. Tests use
# ``multiprocessing.get_context("fork")`` (rather than the production
# "spawn" default) specifically so that fake targets can be ordinary local
# closures: fork duplicates the whole parent process image (no pickling of
# the target/args is required), whereas spawn would require every fake
# target to be a separately importable top-level module attribute. Production
# code (via the CLI) never overrides either seam, and always uses "spawn".


def test_supervised_pilot_kills_blocking_child_before_it_returns(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("fork")
    call_count = ctx.Value("i", 0)

    def _blocking_target(*, output_root: Path, **_kwargs: object) -> Phase8BoundedPilotResult:
        with call_count.get_lock():
            call_count.value += 1
        time.sleep(5.0)  # far longer than the tiny test deadline below.
        # If the process-level kill did not happen, this would eventually run
        # and publish something -- proving the kill happens strictly before
        # any output is produced.
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "should_never_be_published.json").write_text("{}", encoding="utf-8")
        raise AssertionError("unreachable: the child must be killed before returning")

    kwargs = _minimal_supervisor_kwargs(tmp_path)
    output_root = kwargs["output_root"]

    with pytest.raises(Phase8BoundedPilotProcessWatchdogTimeoutError) as exc_info:
        run_phase8_bounded_pilot_with_watchdog(
            **kwargs,
            wall_clock_limit_seconds=0.2,
            _target=_blocking_target,
            _multiprocessing_context=ctx,
            _termination_grace_seconds=1.0,
        )

    # Distinguishable from the plain in-process watchdog timeout: it is a
    # dedicated subclass and it self-documents that it was a process-level kill.
    assert exc_info.value.killed_at_process_level is True
    assert isinstance(exc_info.value, Phase8BoundedPilotWatchdogTimeoutError)
    assert type(exc_info.value) is Phase8BoundedPilotProcessWatchdogTimeoutError

    assert not output_root.exists()
    assert multiprocessing.active_children() == []
    # No automatic retry: the (fake) target was invoked exactly once.
    assert call_count.value == 1


def test_supervised_pilot_deadline_covers_publication_not_just_earlier_stages(
    tmp_path: Path,
) -> None:
    ctx = multiprocessing.get_context("fork")
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    def _slow_publish_target(**_kwargs: object) -> Phase8BoundedPilotResult:
        (marker_dir / "reached_training").write_text("1", encoding="utf-8")
        time.sleep(0.05)  # fast "training" -- well within the deadline.
        (marker_dir / "reached_publish").write_text("1", encoding="utf-8")
        time.sleep(2.0)  # slow "publish" -- alone exceeds the deadline.
        (marker_dir / "reached_after_publish").write_text("1", encoding="utf-8")
        raise AssertionError("unreachable")

    kwargs = _minimal_supervisor_kwargs(tmp_path)

    with pytest.raises(Phase8BoundedPilotProcessWatchdogTimeoutError):
        run_phase8_bounded_pilot_with_watchdog(
            **kwargs,
            wall_clock_limit_seconds=0.5,
            _target=_slow_publish_target,
            _multiprocessing_context=ctx,
            _termination_grace_seconds=1.0,
        )

    assert (marker_dir / "reached_training").exists()
    assert (marker_dir / "reached_publish").exists()
    assert not (marker_dir / "reached_after_publish").exists()
    assert multiprocessing.active_children() == []


def test_supervisor_deadline_accounts_for_time_already_spent_before_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate significant parent-side elapsed time between recording
    # supervisor_start_time and joining the child (e.g. slow process startup).
    # Only the *first* call to time.monotonic() (the supervisor's own
    # ``supervisor_start_time = time.monotonic()``) is faked, and it is faked
    # by shifting the true clock 100s into the past -- every later call
    # (including calls made internally by ``multiprocessing``'s own
    # ``Process.join(timeout=...)`` polling loop) sees the real, unshifted
    # clock. This is deliberately narrow: patching every call to a fixed
    # value would also corrupt ``join``'s own internal timeout bookkeeping
    # and produce a meaningless test. If the supervisor silently used the
    # full wall_clock_limit_seconds as the join timeout (ignoring the 100s
    # already "elapsed" before it even started the child), this child (which
    # only needs 2s) would finish comfortably and no timeout would fire.
    # Patching the ``time`` module's own ``monotonic`` attribute (rather than
    # a copy of it) affects every importer of ``time``, including the pilot
    # module's ``import time; time.monotonic()`` calls, since they all refer
    # to the one global ``time`` module object.
    call_state = {"count": 0}
    true_monotonic = time.monotonic

    def _fake_monotonic() -> float:
        call_state["count"] += 1
        if call_state["count"] == 1:
            return true_monotonic() - 100.0
        return true_monotonic()

    monkeypatch.setattr(time, "monotonic", _fake_monotonic)

    ctx = multiprocessing.get_context("fork")

    def _quick_target(**_kwargs: object) -> Phase8BoundedPilotResult:
        time.sleep(2.0)
        raise AssertionError("unreachable")

    kwargs = _minimal_supervisor_kwargs(tmp_path)

    with pytest.raises(Phase8BoundedPilotProcessWatchdogTimeoutError):
        run_phase8_bounded_pilot_with_watchdog(
            **kwargs,
            wall_clock_limit_seconds=10.0,
            _target=_quick_target,
            _multiprocessing_context=ctx,
            _termination_grace_seconds=1.0,
        )

    assert multiprocessing.active_children() == []


def test_supervised_pilot_returns_target_result_unchanged_for_fast_execution(
    tmp_path: Path,
) -> None:
    ctx = multiprocessing.get_context("fork")
    expected_result = Phase8BoundedPilotResult(
        output_root=tmp_path / "out",
        checkpoint_path=tmp_path / "out" / "checkpoints" / "ckpt.pt",
        config_hash="a" * 64,
        access_ledger_hash="b" * 64,
        checkpoint_metadata_hash="c" * 64,
        summary_hash="d" * 64,
        artifact_hashes={"x.json": "e" * 64},
    )

    def _fast_target(**_kwargs: object) -> Phase8BoundedPilotResult:
        return expected_result

    kwargs = _minimal_supervisor_kwargs(tmp_path)
    result = run_phase8_bounded_pilot_with_watchdog(
        **kwargs,
        wall_clock_limit_seconds=30.0,
        _target=_fast_target,
        _multiprocessing_context=ctx,
    )
    assert result == expected_result
    assert multiprocessing.active_children() == []


def test_supervised_pilot_propagates_child_exception_unchanged(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("fork")

    def _failing_target(**_kwargs: object) -> Phase8BoundedPilotResult:
        raise Phase8BoundedPilotRuntimeError("synthetic_failure_marker")

    kwargs = _minimal_supervisor_kwargs(tmp_path)
    with pytest.raises(Phase8BoundedPilotRuntimeError, match="synthetic_failure_marker"):
        run_phase8_bounded_pilot_with_watchdog(
            **kwargs,
            wall_clock_limit_seconds=30.0,
            _target=_failing_target,
            _multiprocessing_context=ctx,
        )
    assert multiprocessing.active_children() == []


def test_supervised_config_matches_direct_config_field_for_field(tmp_path: Path) -> None:
    """The config built inside the supervised child must match the direct path.

    Uses the exact same pure ``build_phase8_bounded_pilot_config`` builder
    that ``run_phase8_bounded_pilot`` itself calls internally, executed
    inside the supervised child process, and compares it field-for-field
    (and hash-for-hash) against calling that builder directly in this test
    process. This proves the supervisor does not alter any scientific/
    compute-limit configuration -- it only adds process supervision.
    """

    ctx = multiprocessing.get_context("fork")

    def _config_capturing_target(
        *,
        seed: int,
        git_commit: str,
        package_versions: dict[str, str],
        wall_clock_limit_seconds: float,
        **_ignored: object,
    ) -> dict[str, Any]:
        config = build_phase8_bounded_pilot_config(
            seed=seed,
            originating_git_commit=git_commit,
            package_versions=package_versions,
            wall_clock_limit_seconds=wall_clock_limit_seconds,
        )
        return phase8_bounded_pilot_config_to_dict(config)

    kwargs = _minimal_supervisor_kwargs(tmp_path)
    supervised_config_dict = cast(
        "dict[str, Any]",
        run_phase8_bounded_pilot_with_watchdog(
            **kwargs,
            wall_clock_limit_seconds=30.0,
            _target=cast(Any, _config_capturing_target),
            _multiprocessing_context=ctx,
        ),
    )

    direct_config = build_phase8_bounded_pilot_config(
        seed=kwargs["seed"],
        originating_git_commit=kwargs["git_commit"],
        package_versions=kwargs["package_versions"],
        wall_clock_limit_seconds=30.0,
    )
    direct_config_dict = phase8_bounded_pilot_config_to_dict(direct_config)

    assert supervised_config_dict == direct_config_dict
    assert sha256_json(supervised_config_dict) == sha256_json(direct_config_dict)
    assert multiprocessing.active_children() == []


@requires_baseline_environment
def test_full_synthetic_pilot_via_supervised_entrypoint_matches_direct_path(
    tmp_path: Path,
) -> None:
    """Production path: the default target and default (spawn) context.

    Exercises exactly what the CLI now calls -- no ``_target`` or
    ``_multiprocessing_context`` override -- so the real
    ``run_phase8_bounded_pilot`` runs inside a spawned child process,
    re-importing torch/monai fresh in that child, exactly as it would in
    production.
    """

    (
        manifest_path,
        split_path,
        lesion_components_path,
        manifest,
        split,
        _lesion_components_obj,
    ) = _write_manifest_split_lesions(tmp_path)
    prereqs = _write_prerequisites(
        tmp_path,
        manifest_path=manifest_path,
        split_path=split_path,
        manifest=manifest,
        split=split,
    )
    dataset_root = _write_dataset_root(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    result = run_phase8_bounded_pilot_with_watchdog(
        manifest_path=manifest_path,
        split_path=split_path,
        lesion_components_path=lesion_components_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        expected_lesion_components_sha256=sha256_file(lesion_components_path),
        input_binding_path=prereqs["input_binding_path"],  # type: ignore[arg-type]
        candidate_inventory_path=prereqs["candidate_inventory_path"],  # type: ignore[arg-type]
        preprocessing_decision_path=prereqs["preprocessing_decision_path"],  # type: ignore[arg-type]
        expected_input_binding_sha256=prereqs["expected_input_binding_sha256"],  # type: ignore[arg-type]
        expected_candidate_inventory_sha256=prereqs[  # type: ignore[arg-type]
            "expected_candidate_inventory_sha256"
        ],
        expected_preprocessing_decision_sha256=prereqs[  # type: ignore[arg-type]
            "expected_preprocessing_decision_sha256"
        ],
        dataset_root=dataset_root,
        output_root=output_root,
        repository_root=repository_root,
        git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
        seed=1729,
    )

    assert isinstance(result, Phase8BoundedPilotResult)
    assert output_root.is_dir()
    assert result.config_hash
    assert result.access_ledger_hash
    assert result.checkpoint_metadata_hash
    assert result.summary_hash
    assert multiprocessing.active_children() == []


@requires_baseline_environment
def test_cli_smoke_execution_via_supervised_entrypoint_creates_expected_files(
    tmp_path: Path,
) -> None:
    (
        manifest_path,
        split_path,
        lesion_components_path,
        manifest,
        split,
        _lesion_components_obj,
    ) = _write_manifest_split_lesions(tmp_path)
    prereqs = _write_prerequisites(
        tmp_path,
        manifest_path=manifest_path,
        split_path=split_path,
        manifest=manifest,
        split=split,
    )
    dataset_root = _write_dataset_root(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-phase8-bounded-real-development-pilot",
            "--approve-bounded-pilot",
            "--manifest-path",
            str(manifest_path),
            "--split-path",
            str(split_path),
            "--lesion-components-path",
            str(lesion_components_path),
            "--expected-manifest-sha256",
            sha256_file(manifest_path),
            "--expected-split-sha256",
            sha256_file(split_path),
            "--expected-lesion-components-sha256",
            sha256_file(lesion_components_path),
            "--input-binding-path",
            str(prereqs["input_binding_path"]),
            "--candidate-inventory-path",
            str(prereqs["candidate_inventory_path"]),
            "--preprocessing-decision-path",
            str(prereqs["preprocessing_decision_path"]),
            "--expected-input-binding-sha256",
            str(prereqs["expected_input_binding_sha256"]),
            "--expected-candidate-inventory-sha256",
            str(prereqs["expected_candidate_inventory_sha256"]),
            "--expected-preprocessing-decision-sha256",
            str(prereqs["expected_preprocessing_decision_sha256"]),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--git-commit",
            GIT_COMMIT,
            "--package-versions-json",
            json.dumps(PACKAGE_VERSIONS),
            "--repository-root",
            str(repository_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_root.is_dir()
    published_files = sorted(p.name for p in output_root.iterdir() if p.is_file())
    assert published_files == [
        "phase8_definitive_training_pilot_access_ledger.json",
        "phase8_definitive_training_pilot_checkpoint_metadata.json",
        "phase8_definitive_training_pilot_config.json",
        "phase8_definitive_training_pilot_summary.json",
    ]
    assert multiprocessing.active_children() == []


# ---------------------------------------------------------------------------
# Post-incident recovery audit remediation: label domain, HU normalization,
# padded-patch boundary behavior, sampling-center correctness, nonfinite
# fail-closed paths, sliding-window call parameters, self-hash tamper
# rejection, prerequisite cross-hash mismatches, and the definitive-training/
# pilot-approval decoupling guarantee.
# ---------------------------------------------------------------------------


def test_geometry_validation_rejects_out_of_domain_label_value(tmp_path: Path) -> None:
    """A raw label value outside {0,1,2} must fail closed before training.

    This is the exact defect class documented in
    docs/phase8/SUPERVISOR_HANDOFF.md's Substage 3 incident (a three-valued
    LiTS/MSD label misread as binary); the reconstructed test file omitted
    direct coverage of this rejection despite implementing the geometry
    tests around it.
    """

    affine = np.eye(4, dtype=np.float64)
    image = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.zeros(VOLUME_SHAPE, dtype=np.float32), affine
    )
    label_array = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
    label_array[0, 0, 0] = 9  # outside the approved {0, 1, 2} domain.
    label = nib.Nifti1Image(label_array, affine)  # type: ignore[no-untyped-call]
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(image, str(image_path))
    nib.save(label, str(label_path))

    with pytest.raises(Phase8BoundedPilotLabelDomainError):
        _validate_raw_case_pair_geometry(image_path, label_path)


def test_geometry_validation_rejects_nonfinite_image_values(tmp_path: Path) -> None:
    affine = np.eye(4, dtype=np.float64)
    image_array = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    image_array[0, 0, 0] = np.nan
    image = nib.Nifti1Image(image_array, affine)  # type: ignore[no-untyped-call]
    label = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.zeros(VOLUME_SHAPE, dtype=np.uint8), affine
    )
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(image, str(image_path))
    nib.save(label, str(label_path))

    with pytest.raises(Phase8BoundedPilotDataError):
        _validate_raw_case_pair_geometry(image_path, label_path)


def test_geometry_validation_rejects_affine_only_mismatch(tmp_path: Path) -> None:
    """Same shape, differing affine (not just differing shape) must also fail."""

    image_affine = np.eye(4, dtype=np.float64)
    label_affine = np.eye(4, dtype=np.float64)
    label_affine[0, 3] = 50.0  # translated origin, same shape.
    image = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.zeros(VOLUME_SHAPE, dtype=np.float32), image_affine
    )
    label = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.zeros(VOLUME_SHAPE, dtype=np.uint8), label_affine
    )
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(image, str(image_path))
    nib.save(label, str(label_path))

    with pytest.raises(Phase8BoundedPilotDataError):
        _validate_raw_case_pair_geometry(image_path, label_path)


def test_normalize_hu_clips_and_rescales_to_minus_one_one() -> None:
    """HU clipping is exactly [-1000, 1000], then linear rescale to exactly [-1, 1]."""

    config = build_phase8_bounded_pilot_config(
        seed=1729, originating_git_commit=GIT_COMMIT, package_versions=PACKAGE_VERSIONS
    )
    raw = np.array([-5000.0, -1000.0, 0.0, 1000.0, 5000.0], dtype=np.float32).reshape(5, 1, 1)
    normalized = _normalize_hu(raw, config=config)
    assert np.allclose(normalized.reshape(-1), [-1.0, -1.0, 0.0, 1.0, 1.0], atol=1e-6)
    assert float(normalized.min()) >= -1.0
    assert float(normalized.max()) <= 1.0


def test_extract_padded_patch_preserves_shape_and_fill_at_volume_boundary() -> None:
    """A patch center near the volume edge must still yield the exact required shape.

    The out-of-volume region must be filled with ``fill_value``, not silently
    shifted to stay in bounds -- this is the "safely pad at volume
    boundaries" contract, previously untested at the unit level.
    """

    volume = np.ones((10, 10, 10), dtype=np.float32)
    patch = _extract_padded_patch(volume, center=(0, 0, 0), patch_size=(8, 8, 8), fill_value=-1.0)
    assert patch.shape == (8, 8, 8)
    # Center (0,0,0) with patch_size 8 means half=4, so start=-4 on each axis:
    # the first 4 voxels on each axis are out-of-bounds padding.
    assert np.all(patch[:4, :4, :4] == -1.0)
    assert np.all(patch[4:, 4:, 4:] == 1.0)


def test_sampling_schedule_positive_centers_index_tumor_voxels_and_negative_do_not() -> None:
    """Positive centers must actually land on foreground voxels, and vice versa.

    A plausible broken implementation could satisfy the 10:10 count contract
    while drawing centers from the wrong voxel population (e.g. swapping the
    positive/negative coordinate arrays); only checking counts (as the
    reconstructed file did) would not catch that class of bug.
    """

    positive_label = _positive_label_array()
    empty_label = _empty_label_array()
    schedule = build_phase8_bounded_pilot_sampling_schedule(
        positive_case_label=positive_label, empty_case_label=empty_label, seed=1729
    )
    for sample in schedule:
        source = positive_label if sample.pilot_role == "train_positive" else empty_label
        voxel_value = source[sample.center]
        if sample.is_positive:
            assert voxel_value == RAW_LABEL_TUMOR_VALUE
        else:
            assert voxel_value != RAW_LABEL_TUMOR_VALUE


def test_prerequisites_rejects_mismatched_preprocessing_evidence_hash(tmp_path: Path) -> None:
    manifest = _manifest()
    split = _split(manifest)
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    phase2_artifact_to_json(manifest, manifest_path)
    phase2_artifact_to_json(split, split_path)

    prereq_dir = tmp_path / "prerequisites"
    prereq_dir.mkdir()
    input_binding = build_phase8_real_development_input_binding(
        manifest_path=manifest_path,
        split_path=split_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        approved_development_artifact_set_identity="phase8-substage4b-v1",
    )
    input_binding_path = prereq_dir / "input_binding.json"
    input_binding_path.write_bytes(phase8_real_development_input_binding_to_json(input_binding))

    preprocessing_mapping = _preprocessing_decision_mapping(
        manifest_hash=manifest.manifest_hash, split_hash=split.split_hash
    )
    preprocessing_decision_path = prereq_dir / "preprocessing_decision.json"
    preprocessing_decision_path.write_text(
        json.dumps(preprocessing_mapping, sort_keys=True), encoding="utf-8"
    )

    # Deliberately bind the candidate to a different preprocessing decision hash.
    inventory_mapping = _candidate_inventory_mapping(preprocessing_decision_hash="f" * 64)
    candidate_inventory_path = prereq_dir / "candidate_inventory.json"
    candidate_inventory_path.write_text(
        json.dumps(inventory_mapping, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(Phase8BoundedPilotPrerequisitesError):
        verify_phase8_bounded_pilot_prerequisites(
            input_binding_path=input_binding_path,
            candidate_inventory_path=candidate_inventory_path,
            preprocessing_decision_path=preprocessing_decision_path,
            expected_input_binding_sha256=sha256_file(input_binding_path),
            expected_candidate_inventory_sha256=sha256_file(candidate_inventory_path),
            expected_preprocessing_decision_sha256=sha256_file(preprocessing_decision_path),
            manifest=manifest,
            split=split,
        )


def test_prerequisites_rejects_wrong_candidate_model_family(tmp_path: Path) -> None:
    manifest = _manifest()
    split = _split(manifest)
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    phase2_artifact_to_json(manifest, manifest_path)
    phase2_artifact_to_json(split, split_path)

    prereq_dir = tmp_path / "prerequisites"
    prereq_dir.mkdir()
    input_binding = build_phase8_real_development_input_binding(
        manifest_path=manifest_path,
        split_path=split_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        approved_development_artifact_set_identity="phase8-substage4b-v1",
    )
    input_binding_path = prereq_dir / "input_binding.json"
    input_binding_path.write_bytes(phase8_real_development_input_binding_to_json(input_binding))

    preprocessing_mapping = _preprocessing_decision_mapping(
        manifest_hash=manifest.manifest_hash, split_hash=split.split_hash
    )
    preprocessing_decision_path = prereq_dir / "preprocessing_decision.json"
    preprocessing_decision_path.write_text(
        json.dumps(preprocessing_mapping, sort_keys=True), encoding="utf-8"
    )

    inventory_mapping = _candidate_inventory_mapping(
        preprocessing_decision_hash=preprocessing_mapping["preprocessing_decision_hash"]
    )
    inventory_mapping["candidates"][0]["model_family"] = "not_monai_segresnet"
    inventory_mapping = _rehash(inventory_mapping, "inventory_hash")
    candidate_inventory_path = prereq_dir / "candidate_inventory.json"
    candidate_inventory_path.write_text(
        json.dumps(inventory_mapping, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(Phase8BoundedPilotPrerequisitesError):
        verify_phase8_bounded_pilot_prerequisites(
            input_binding_path=input_binding_path,
            candidate_inventory_path=candidate_inventory_path,
            preprocessing_decision_path=preprocessing_decision_path,
            expected_input_binding_sha256=sha256_file(input_binding_path),
            expected_candidate_inventory_sha256=sha256_file(candidate_inventory_path),
            expected_preprocessing_decision_sha256=sha256_file(preprocessing_decision_path),
            manifest=manifest,
            split=split,
        )


def test_prerequisites_rejects_candidate_using_external_artifacts(tmp_path: Path) -> None:
    manifest = _manifest()
    split = _split(manifest)
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    phase2_artifact_to_json(manifest, manifest_path)
    phase2_artifact_to_json(split, split_path)

    prereq_dir = tmp_path / "prerequisites"
    prereq_dir.mkdir()
    input_binding = build_phase8_real_development_input_binding(
        manifest_path=manifest_path,
        split_path=split_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        approved_development_artifact_set_identity="phase8-substage4b-v1",
    )
    input_binding_path = prereq_dir / "input_binding.json"
    input_binding_path.write_bytes(phase8_real_development_input_binding_to_json(input_binding))

    preprocessing_mapping = _preprocessing_decision_mapping(
        manifest_hash=manifest.manifest_hash, split_hash=split.split_hash
    )
    preprocessing_decision_path = prereq_dir / "preprocessing_decision.json"
    preprocessing_decision_path.write_text(
        json.dumps(preprocessing_mapping, sort_keys=True), encoding="utf-8"
    )

    inventory_mapping = _candidate_inventory_mapping(
        preprocessing_decision_hash=preprocessing_mapping["preprocessing_decision_hash"]
    )
    inventory_mapping["candidates"][0]["uses_external_artifacts"] = True
    inventory_mapping = _rehash(inventory_mapping, "inventory_hash")
    candidate_inventory_path = prereq_dir / "candidate_inventory.json"
    candidate_inventory_path.write_text(
        json.dumps(inventory_mapping, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(Phase8BoundedPilotPrerequisitesError):
        verify_phase8_bounded_pilot_prerequisites(
            input_binding_path=input_binding_path,
            candidate_inventory_path=candidate_inventory_path,
            preprocessing_decision_path=preprocessing_decision_path,
            expected_input_binding_sha256=sha256_file(input_binding_path),
            expected_candidate_inventory_sha256=sha256_file(candidate_inventory_path),
            expected_preprocessing_decision_sha256=sha256_file(preprocessing_decision_path),
            manifest=manifest,
            split=split,
        )


def test_config_self_hash_tamper_rejected() -> None:
    config = build_phase8_bounded_pilot_config(
        seed=1729, originating_git_commit=GIT_COMMIT, package_versions=PACKAGE_VERSIONS
    )
    with pytest.raises(Phase8BoundedPilotError):
        replace(config, config_hash="f" * 64)


def test_access_ledger_self_hash_tamper_rejected() -> None:
    train_positive = _selected_case(
        case_id="anon-c0002", patient_id="anon-p0002", partition="train", role="train_positive"
    )
    train_empty = _selected_case(
        case_id="anon-c0001", patient_id="anon-p0001", partition="train", role="train_empty"
    )
    validation = _selected_case(
        case_id="anon-c0004", patient_id="anon-p0004", partition="validation", role="validation"
    )
    ledger = build_phase8_bounded_pilot_access_ledger(
        train_positive_case=train_positive,
        train_empty_case=train_empty,
        validation_case=validation,
    )
    with pytest.raises(Phase8BoundedPilotError):
        replace(ledger, ledger_hash="f" * 64)


def test_checkpoint_metadata_wrapper_self_hash_tamper_rejected() -> None:
    config = build_phase8_bounded_pilot_config(
        seed=1729, originating_git_commit=GIT_COMMIT, package_versions=PACKAGE_VERSIONS
    )
    metadata = build_phase8_bounded_pilot_checkpoint_metadata(
        config=config,
        development_manifest_hash="1" * 64,
        development_split_hash="2" * 64,
        checkpoint_sha256="3" * 64,
        checkpoint_byte_size=1024,
    )
    with pytest.raises(Phase8BoundedPilotError):
        replace(metadata, wrapper_hash="f" * 64)


def test_definitive_training_release_cannot_substitute_for_pilot_approval() -> None:
    """The pilot's approval gate must be structurally independent of Substage 4A.

    Static-scan test (mirrors the existing
    ``test_no_forbidden_scientific_metric_field_names_in_source`` pattern):
    the bounded-pilot module must never import or reference the Substage 4A
    definitive-training release contract, so a ``released``
    ``Phase8DefinitiveExecutionRelease`` object can never be used to bypass
    ``--approve-bounded-pilot`` or any pilot prerequisite check.
    """

    import protoem_ct.external.definitive_training_pilot as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "from protoem_ct.external.definitive_training import" not in source
    assert "import protoem_ct.external.definitive_training\n" not in source
    assert "Phase8DefinitiveExecutionRelease" not in source
    assert "Phase8DefinitiveTrainingConfig" not in source
    assert "release_state" not in source
    assert "execution_release_state" not in source


@requires_baseline_environment
def test_load_pilot_training_patch_binarizes_raw_label_domain_correctly() -> None:
    """Raw label 2 (tumor) maps to foreground; raw 0/1 (background/liver) map to background.

    Direct unit test of the fixed, non-data-dependent binarization rule,
    independent of the full training pipeline.
    """

    torch = _import_torch()
    config = build_phase8_bounded_pilot_config(
        seed=1729, originating_git_commit=GIT_COMMIT, package_versions=PACKAGE_VERSIONS
    )
    image_array = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    label_array = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
    center = (50, 50, 35)
    half = tuple(size // 2 for size in config.patch_size)
    # Voxel exactly at the patch's top-left-front corner -> raw tumor value.
    label_array[center[0] - half[0], center[1] - half[1], center[2] - half[2]] = (
        RAW_LABEL_TUMOR_VALUE
    )
    # Voxel one step further along axis 0 -> raw liver value (must be background).
    label_array[center[0] - half[0] + 1, center[1] - half[1], center[2] - half[2]] = 1

    _, label_tensor = _load_pilot_training_patch(
        image_array=image_array,
        label_array=label_array,
        center=center,
        config=config,
        torch=torch,
    )
    label_np = label_tensor.numpy()[0, 0]
    assert label_np[0, 0, 0] == 1
    assert label_np[1, 0, 0] == 0
    assert set(np.unique(label_np).tolist()) <= {0, 1}


@requires_baseline_environment
def test_run_bounded_training_steps_fails_closed_on_nonfinite_loss() -> None:
    torch = _import_torch()
    config = build_phase8_bounded_pilot_config(
        seed=1729, originating_git_commit=GIT_COMMIT, package_versions=PACKAGE_VERSIONS
    )
    from protoem_ct.external.definitive_training_pilot import Phase8BoundedPilotPatchSample

    model = torch.nn.Conv3d(1, 2, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def _nan_loss(logits: object, labels: object) -> object:
        return cast(Any, logits).sum() * float("nan")

    schedule = [
        Phase8BoundedPilotPatchSample(
            pilot_role="train_positive", is_positive=True, center=(5, 5, 5)
        )
    ]
    case_arrays = {
        "train_positive": (
            np.zeros(VOLUME_SHAPE, dtype=np.float32),
            np.zeros(VOLUME_SHAPE, dtype=np.uint8),
        )
    }
    with pytest.raises(Phase8BoundedPilotRuntimeError, match="nonfinite_loss"):
        _run_bounded_training_steps(
            model=model,
            optimizer=optimizer,
            loss_function=_nan_loss,
            schedule=schedule,
            case_arrays=case_arrays,
            config=config,
            torch=torch,
            start_time=time.monotonic(),
            wall_clock_limit_seconds=DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
        )


@requires_baseline_environment
def test_run_bounded_training_steps_fails_closed_on_nonfinite_gradient() -> None:
    torch = _import_torch()
    config = build_phase8_bounded_pilot_config(
        seed=1729, originating_git_commit=GIT_COMMIT, package_versions=PACKAGE_VERSIONS
    )
    from protoem_ct.external.definitive_training_pilot import Phase8BoundedPilotPatchSample

    class _NaNGradFn(cast(Any, torch).autograd.Function):  # type: ignore[misc]
        @staticmethod
        def forward(ctx: object, x: object) -> object:
            return cast(Any, x).clone()

        @staticmethod
        def backward(ctx: object, grad_output: object) -> object:
            return cast(Any, grad_output) * float("nan")

    model = torch.nn.Conv3d(1, 2, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def _identity_finite_loss_with_nan_grad(logits: object, labels: object) -> object:
        perturbed = _NaNGradFn.apply(logits)
        return cast(Any, perturbed).sum() * 0.0 + cast(Any, logits).float().mean() * 0.0

    schedule = [
        Phase8BoundedPilotPatchSample(
            pilot_role="train_positive", is_positive=True, center=(5, 5, 5)
        )
    ]
    case_arrays = {
        "train_positive": (
            np.zeros(VOLUME_SHAPE, dtype=np.float32),
            np.zeros(VOLUME_SHAPE, dtype=np.uint8),
        )
    }
    with pytest.raises(Phase8BoundedPilotRuntimeError, match="nonfinite_gradient"):
        _run_bounded_training_steps(
            model=model,
            optimizer=optimizer,
            loss_function=_identity_finite_loss_with_nan_grad,
            schedule=schedule,
            case_arrays=case_arrays,
            config=config,
            torch=torch,
            start_time=time.monotonic(),
            wall_clock_limit_seconds=DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
        )


@requires_baseline_environment
def test_sliding_window_validation_uses_required_parameters_and_fails_closed_on_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spy on the sliding-window call to prove ROI/overlap/batch-size and single-call use,
    and that nonfinite logits fail closed."""

    torch = _import_torch()
    monai = _import_monai()
    config = build_phase8_bounded_pilot_config(
        seed=1729, originating_git_commit=GIT_COMMIT, package_versions=PACKAGE_VERSIONS
    )
    model = torch.nn.Conv3d(1, 2, kernel_size=1)

    calls: list[dict[str, Any]] = []
    original = monai.inferers.sliding_window_inference

    def _spy_sliding_window_inference(inputs: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        shape = (1, 2, *cast(Any, inputs).shape[2:])
        return torch.full(shape, float("nan"))

    monkeypatch.setattr(monai.inferers, "sliding_window_inference", _spy_sliding_window_inference)
    try:
        with pytest.raises(Phase8BoundedPilotRuntimeError, match="nonfinite_validation_logits"):
            _run_pilot_validation_sliding_window(
                torch=torch,
                monai=monai,
                model=model,
                image_array=np.zeros(VOLUME_SHAPE, dtype=np.float32),
                label_array=np.zeros(VOLUME_SHAPE, dtype=np.uint8),
                config=config,
            )
    finally:
        monkeypatch.setattr(monai.inferers, "sliding_window_inference", original)

    assert len(calls) == 1
    assert calls[0]["roi_size"] == REQUIRED_SLIDING_WINDOW_ROI_SIZE
    assert calls[0]["overlap"] == REQUIRED_SLIDING_WINDOW_OVERLAP
    assert calls[0]["sw_batch_size"] == REQUIRED_SLIDING_WINDOW_BATCH_SIZE


@requires_baseline_environment
def test_checkpoint_reload_fails_closed_on_corrupted_bytes() -> None:
    torch = _import_torch()
    monai = _import_monai()
    with pytest.raises(Phase8BoundedPilotRuntimeError, match="checkpoint_reload_failed"):
        _reload_and_verify_checkpoint_from_bytes(
            torch=torch, monai=monai, checkpoint_bytes=b"not a valid torch checkpoint"
        )


@requires_baseline_environment
def test_full_pipeline_optimizer_and_loss_match_declared_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The *executed* optimizer/loss must match the config's declared identity.

    Spies the real construction call sites inside
    ``run_phase8_bounded_pilot`` so a mismatch between ``optimizer_name``/
    ``loss_name`` and what is actually built would be caught, not just the
    declared config field.
    """

    from protoem_ct.external.definitive_training_pilot import run_phase8_bounded_pilot

    torch = _import_torch()
    monai = _import_monai()

    (
        manifest_path,
        split_path,
        lesion_components_path,
        manifest,
        split,
        _lesion_components_obj,
    ) = _write_manifest_split_lesions(tmp_path)
    prereqs = _write_prerequisites(
        tmp_path,
        manifest_path=manifest_path,
        split_path=split_path,
        manifest=manifest,
        split=split,
    )
    dataset_root = _write_dataset_root(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "output"

    constructed_optimizers: list[Any] = []
    constructed_losses: list[Any] = []
    original_adamw = torch.optim.AdamW
    original_dice_ce = monai.losses.DiceCELoss

    def _spy_adamw(*args: object, **kwargs: object) -> object:
        instance = original_adamw(*args, **kwargs)
        constructed_optimizers.append((instance, kwargs))
        return instance

    def _spy_dice_ce(*args: object, **kwargs: object) -> object:
        instance = original_dice_ce(*args, **kwargs)
        constructed_losses.append(instance)
        return instance

    monkeypatch.setattr(torch.optim, "AdamW", _spy_adamw)
    monkeypatch.setattr(monai.losses, "DiceCELoss", _spy_dice_ce)

    torch_load_calls: list[object] = []
    original_torch_load = torch.load

    def _spy_torch_load(*args: object, **kwargs: object) -> object:
        torch_load_calls.append(args[0] if args else kwargs.get("f"))
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy_torch_load)

    run_phase8_bounded_pilot(
        manifest_path=manifest_path,
        split_path=split_path,
        lesion_components_path=lesion_components_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_split_sha256=sha256_file(split_path),
        expected_lesion_components_sha256=sha256_file(lesion_components_path),
        input_binding_path=prereqs["input_binding_path"],  # type: ignore[arg-type]
        candidate_inventory_path=prereqs["candidate_inventory_path"],  # type: ignore[arg-type]
        preprocessing_decision_path=prereqs["preprocessing_decision_path"],  # type: ignore[arg-type]
        expected_input_binding_sha256=prereqs["expected_input_binding_sha256"],  # type: ignore[arg-type]
        expected_candidate_inventory_sha256=prereqs[  # type: ignore[arg-type]
            "expected_candidate_inventory_sha256"
        ],
        expected_preprocessing_decision_sha256=prereqs[  # type: ignore[arg-type]
            "expected_preprocessing_decision_sha256"
        ],
        dataset_root=dataset_root,
        output_root=output_root,
        repository_root=repository_root,
        git_commit=GIT_COMMIT,
        package_versions=PACKAGE_VERSIONS,
        seed=1729,
    )

    assert len(constructed_optimizers) == 1
    _optimizer_instance, optimizer_kwargs = constructed_optimizers[0]
    assert optimizer_kwargs["lr"] == 1e-4
    assert optimizer_kwargs["weight_decay"] == 1e-5
    assert len(constructed_losses) == 1
    # Exactly one torch.load call: the checkpoint-round-trip self-verification
    # of the bytes just written in this same run (never a pre-existing file).
    assert len(torch_load_calls) == 1
