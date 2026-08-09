"""Unit tests for Phase 8 definitive-development training contracts (Substage 4A)."""

from __future__ import annotations

import ast
import builtins
import inspect
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.utils import strip_ansi  # type: ignore[attr-defined]
from typer.testing import CliRunner

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.cli.main import app
from protoem_ct.external.definitive_training import (
    AWAITING_EXPLICIT_USER_AUTHORIZATION,
    DEFINITIVE_TIE_BREAK_ORDER,
    PHASE8_DEFINITIVE_EXECUTION_RELEASE_SCHEMA_NAME,
    PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION,
    RELEASE_AUTHORIZED_BY_EXPLICIT_USER_AUTHORIZATION,
    RELEASE_STATE_NOT_RELEASED,
    RELEASE_STATE_RELEASED,
    REQUIRED_CANDIDATE_ID,
    Phase8DefinitiveExecutionRelease,
    Phase8DefinitiveTrainingConfig,
    Phase8DefinitiveTrainingHashError,
    Phase8DefinitiveTrainingNotReleasedError,
    Phase8DefinitiveTrainingPublicationError,
    Phase8DefinitiveTrainingSerializationError,
    Phase8DefinitiveTrainingValidationError,
    build_phase8_definitive_training_config,
    build_unreleased_definitive_execution_release,
    execute_phase8_definitive_training,
    hash_phase8_definitive_execution_release,
    hash_phase8_definitive_training_config,
    phase8_definitive_execution_release_from_mapping,
    phase8_definitive_execution_release_to_dict,
    phase8_definitive_training_config_from_mapping,
    phase8_definitive_training_config_to_dict,
    publish_phase8_definitive_checkpoint_metadata,
    publish_phase8_definitive_model_selection,
    publish_phase8_definitive_preprocessing_evidence,
    publish_phase8_definitive_support_policy,
    publish_phase8_definitive_threshold_decision,
    publish_phase8_definitive_training_plan,
    publish_phase8_definitive_validation_evidence,
    validate_phase8_definitive_output_root,
)
from protoem_ct.external.internal_evidence import (
    ArtifactReference,
    CandidateDefinition,
    Phase8CheckpointMetadata,
    Phase8FixedCandidateInventory,
    Phase8InternalEvidenceValidationError,
    Phase8PreprocessingDecision,
    Phase8ValidationEvidenceReference,
    candidate_definition_to_dict,
    phase8_checkpoint_metadata_from_mapping,
    phase8_fixed_candidate_inventory_from_mapping,
    phase8_preprocessing_decision_from_mapping,
    phase8_validation_evidence_reference_from_mapping,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
GIT_COMMIT = "0" * 40


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _reference(role: str, artifact_hash: str) -> ArtifactReference:
    return ArtifactReference(
        schema_name=f"phase8_{role}",
        schema_version="v1",
        artifact_hash=artifact_hash,
        artifact_role=role,
    )


def _reference_mapping(role: str, artifact_hash: str) -> dict[str, Any]:
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
    if hash_field == "checkpoint_metadata_hash":
        # Phase8CheckpointMetadata's identity payload excludes amp_state and
        # device_type (see phase8_checkpoint_metadata_identity_payload).
        payload.pop("amp_state", None)
        payload.pop("device_type", None)
    result[hash_field] = sha256_json(payload)
    return result


def _preprocessing_decision(
    *,
    candidate_id: str = REQUIRED_CANDIDATE_ID,
    no_external_data: bool = True,
    development_manifest_hash: str = HASH_A,
) -> Phase8PreprocessingDecision:
    mapping: dict[str, Any] = {
        "schema_name": "phase8_preprocessing_decision",
        "schema_version": "v1",
        "candidate_id": candidate_id,
        "development_manifest_hash": development_manifest_hash,
        "development_split_hash": HASH_C,
        "preprocessing_config_schema_name": "phase8_preprocessing_config",
        "preprocessing_config_schema_version": "v1",
        "preprocessing_config_hash": HASH_A,
        "fit_scope": "no_data_dependent_fit",
        "fit_artifact_reference": None,
        "image_interpolation_policy": "bilinear",
        "label_interpolation_policy": "nearest",
        "orientation_policy_reference": _reference_mapping("orientation_policy", HASH_A),
        "spacing_policy_reference": _reference_mapping("spacing_policy", HASH_A),
        "intensity_policy_reference": _reference_mapping("intensity_policy", HASH_A),
        "crop_roi_policy_reference": _reference_mapping("crop_roi_policy", HASH_A),
        "normalization_policy_reference": _reference_mapping("normalization_policy", HASH_A),
        "originating_git_commit": GIT_COMMIT,
        "no_external_data": no_external_data,
        "evidence_status": "resolved",
    }
    mapping = _rehash(mapping, "preprocessing_decision_hash")
    return phase8_preprocessing_decision_from_mapping(mapping)


def _candidate(
    *,
    candidate_id: str = REQUIRED_CANDIDATE_ID,
    training_config_hash: str = HASH_A,
    preprocessing_evidence_hash: str,
) -> CandidateDefinition:
    return CandidateDefinition(
        candidate_id=candidate_id,
        model_family="monai_segresnet",
        method_identity="monai_segresnet_baseline",
        training_adaptation_mode="baseline_training",
        training_config_reference=_reference("training_config", training_config_hash),
        preprocessing_evidence_hash=preprocessing_evidence_hash,
        fixed_seeds=(1729,),
        executable_readiness_state="executable",
        failure_handling_policy="single_candidate_no_comparison",
        uses_external_artifacts=False,
    )


def _inventory(*, candidates: tuple[CandidateDefinition, ...]) -> Phase8FixedCandidateInventory:
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    mapping: dict[str, Any] = {
        "schema_name": "phase8_fixed_candidate_inventory",
        "schema_version": "v1",
        "inventory_state": "freeze_ready",
        "candidate_set_locked": True,
        "selection_started": True,
        "supports_single_candidate_without_comparison": len(ordered) == 1,
        "candidates": [candidate_definition_to_dict(item) for item in ordered],
    }
    mapping = _rehash(mapping, "inventory_hash")
    return phase8_fixed_candidate_inventory_from_mapping(mapping)


def _config(
    *,
    preprocessing_decision: Phase8PreprocessingDecision | None = None,
    inventory: Phase8FixedCandidateInventory | None = None,
    training_config_hash: str = HASH_A,
) -> tuple[
    Phase8DefinitiveTrainingConfig, Phase8PreprocessingDecision, Phase8FixedCandidateInventory
]:
    preprocessing_decision = preprocessing_decision or _preprocessing_decision()
    candidate = _candidate(
        training_config_hash=training_config_hash,
        preprocessing_evidence_hash=preprocessing_decision.preprocessing_decision_hash,
    )
    inventory = inventory or _inventory(candidates=(candidate,))
    preprocessing_decision_reference = ArtifactReference(
        schema_name=preprocessing_decision.schema_name,
        schema_version=preprocessing_decision.schema_version,
        artifact_hash=preprocessing_decision.preprocessing_decision_hash,
        artifact_role="preprocessing_decision",
    )
    config = build_phase8_definitive_training_config(
        input_binding_hash=HASH_B,
        fixed_candidate_inventory_hash=inventory.inventory_hash,
        preprocessing_decision_reference=preprocessing_decision_reference,
        training_config_reference=candidate.training_config_reference,
        fixed_seeds=candidate.fixed_seeds,
    )
    return config, preprocessing_decision, inventory


def _released_release(config: Phase8DefinitiveTrainingConfig) -> Phase8DefinitiveExecutionRelease:
    """Construct a ``released`` release directly (test-only bypass).

    No function reachable from :mod:`protoem_ct.external.definitive_training`
    can produce this; it is constructed here, out of band, purely to exercise
    the hypothetical-release validation path.
    """

    payload = {
        "authorized_by": RELEASE_AUTHORIZED_BY_EXPLICIT_USER_AUTHORIZATION,
        "authorized_max_training_steps": 1000,
        "bound_config_hash": config.config_hash,
        "release_scope_description": "test-only-direct-release",
        "release_state": RELEASE_STATE_RELEASED,
        "schema_name": PHASE8_DEFINITIVE_EXECUTION_RELEASE_SCHEMA_NAME,
        "schema_version": PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION,
    }
    return Phase8DefinitiveExecutionRelease(
        schema_name=PHASE8_DEFINITIVE_EXECUTION_RELEASE_SCHEMA_NAME,
        schema_version=PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION,
        release_state=RELEASE_STATE_RELEASED,
        bound_config_hash=config.config_hash,
        release_scope_description="test-only-direct-release",
        authorized_max_training_steps=1000,
        authorized_by=RELEASE_AUTHORIZED_BY_EXPLICIT_USER_AUTHORIZATION,
        release_hash=sha256_json(payload),
    )


def _checkpoint_metadata(
    *,
    config: Phase8DefinitiveTrainingConfig,
    completion_status: str = "completed",
    freeze_eligible: bool = True,
    checkpoint_byte_size: int = 1024,
    training_config_hash: str | None = None,
    candidate_id: str | None = None,
) -> Phase8CheckpointMetadata:
    validation_reference = (
        _reference_mapping("validation_metric", HASH_C) if freeze_eligible else None
    )
    mapping: dict[str, Any] = {
        "schema_name": "phase8_checkpoint_metadata",
        "schema_version": "v1",
        "checkpoint_sha256": HASH_D,
        "checkpoint_byte_size": checkpoint_byte_size,
        "serialization_format": "torch_state_dict",
        "model_family": "monai_segresnet",
        "candidate_id": candidate_id or config.candidate_id,
        "seed": 1729,
        "training_adaptation_mode": "baseline_training",
        "originating_git_commit": GIT_COMMIT,
        "package_environment_reference": _reference_mapping("package_environment", HASH_A),
        "training_config_hash": training_config_hash
        or config.training_config_reference.artifact_hash,
        "preprocessing_evidence_hash": config.preprocessing_decision_reference.artifact_hash,
        "development_manifest_hash": HASH_A,
        "development_split_hash": HASH_C,
        "validation_metric_artifact_reference": validation_reference,
        "device_type": "cpu",
        "amp_state": "disabled",
        "completion_status": completion_status,
        "failure_codes": [],
        "no_external_data": True,
        "freeze_eligible": freeze_eligible,
    }
    mapping = _rehash(mapping, "checkpoint_metadata_hash")
    return phase8_checkpoint_metadata_from_mapping(mapping)


def _validation_evidence(
    *, config: Phase8DefinitiveTrainingConfig, empty_target_role: str = "empty_target_accounting"
) -> Phase8ValidationEvidenceReference:
    mapping: dict[str, Any] = {
        "schema_name": "phase8_validation_evidence_reference",
        "schema_version": "v1",
        "candidate_id": config.candidate_id,
        "checkpoint_hash": HASH_D,
        "validation_cohort_split_hash": HASH_C,
        "metric_artifact_schema_name": "phase8_metric_artifact",
        "metric_artifact_schema_version": "v1",
        "metric_artifact_hash": HASH_A,
        "primary_metric_name": "tumor_dice",
        "valid_case_count": 5,
        "empty_target_accounting_reference": _reference_mapping(empty_target_role, HASH_A),
        "development_validation_only": True,
        "internal_test_not_used": True,
        "external_data_not_used": True,
    }
    mapping = _rehash(mapping, "validation_evidence_hash")
    return phase8_validation_evidence_reference_from_mapping(mapping)


# ---------------------------------------------------------------------------
# 1-2: strict round-trip, self-hash validation, unknown-field rejection
# ---------------------------------------------------------------------------


def test_config_round_trips_and_rejects_tamper() -> None:
    config, _preprocessing, _inventory_obj = _config()
    mapping = phase8_definitive_training_config_to_dict(config)
    restored = phase8_definitive_training_config_from_mapping(mapping)
    assert restored == config

    tampered = deepcopy(mapping)
    tampered["candidate_id"] = REQUIRED_CANDIDATE_ID
    tampered["input_binding_hash"] = HASH_C
    with pytest.raises(Phase8DefinitiveTrainingHashError):
        phase8_definitive_training_config_from_mapping(tampered)


def test_config_unknown_field_rejected() -> None:
    config, _preprocessing, _inventory_obj = _config()
    mapping = phase8_definitive_training_config_to_dict(config)
    mapping["unexpected_field"] = "x"
    with pytest.raises(Phase8DefinitiveTrainingSerializationError):
        phase8_definitive_training_config_from_mapping(mapping)


def test_release_round_trips_and_rejects_tamper() -> None:
    config, _preprocessing, _inventory_obj = _config()
    release = build_unreleased_definitive_execution_release(
        bound_config_hash=config.config_hash,
        release_scope_description="test-scope",
        authorized_max_training_steps=10,
    )
    mapping = phase8_definitive_execution_release_to_dict(release)
    restored = phase8_definitive_execution_release_from_mapping(mapping)
    assert restored == release

    tampered = deepcopy(mapping)
    tampered["authorized_max_training_steps"] = 99
    with pytest.raises(Phase8DefinitiveTrainingHashError):
        phase8_definitive_execution_release_from_mapping(tampered)


def test_release_unknown_field_rejected() -> None:
    config, _preprocessing, _inventory_obj = _config()
    release = build_unreleased_definitive_execution_release(
        bound_config_hash=config.config_hash,
        release_scope_description="test-scope",
        authorized_max_training_steps=10,
    )
    mapping = phase8_definitive_execution_release_to_dict(release)
    mapping["unexpected_field"] = "x"
    with pytest.raises(Phase8DefinitiveTrainingSerializationError):
        phase8_definitive_execution_release_from_mapping(mapping)


def test_hash_helpers_match_self_hash() -> None:
    config, _preprocessing, _inventory_obj = _config()
    assert hash_phase8_definitive_training_config(config) == config.config_hash
    release = build_unreleased_definitive_execution_release(
        bound_config_hash=config.config_hash,
        release_scope_description="test-scope",
        authorized_max_training_steps=10,
    )
    assert hash_phase8_definitive_execution_release(release) == release.release_hash


# ---------------------------------------------------------------------------
# 3: config fixed-value invariants
# ---------------------------------------------------------------------------


def test_config_builder_always_awaiting_approval() -> None:
    config, _preprocessing, _inventory_obj = _config()
    assert config.execution_release_state == AWAITING_EXPLICIT_USER_AUTHORIZATION
    assert config.device_type == "cpu"
    assert config.amp_enabled is False
    assert config.threshold_policy_type == "fixed_constant"
    assert config.threshold_value == 0.5
    assert config.support_policy_type == "no_support"
    assert config.no_external_data is True
    assert config.internal_test_excluded is True


def test_config_rejects_wrong_candidate_id_direct_construction() -> None:
    config, _preprocessing, _inventory_obj = _config()
    mapping = phase8_definitive_training_config_to_dict(config)
    mapping["candidate_id"] = "some_other_candidate"
    mapping["config_hash"] = sha256_json(
        {key: value for key, value in mapping.items() if key != "config_hash"}
    )
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        phase8_definitive_training_config_from_mapping(mapping)


def test_config_rejects_external_data_direct_construction() -> None:
    config, _preprocessing, _inventory_obj = _config()
    mapping = phase8_definitive_training_config_to_dict(config)
    mapping["no_external_data"] = False
    mapping["config_hash"] = sha256_json(
        {key: value for key, value in mapping.items() if key != "config_hash"}
    )
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        phase8_definitive_training_config_from_mapping(mapping)


def test_config_rejects_internal_test_not_excluded_direct_construction() -> None:
    config, _preprocessing, _inventory_obj = _config()
    mapping = phase8_definitive_training_config_to_dict(config)
    mapping["internal_test_excluded"] = False
    mapping["config_hash"] = sha256_json(
        {key: value for key, value in mapping.items() if key != "config_hash"}
    )
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        phase8_definitive_training_config_from_mapping(mapping)


def test_config_rejects_gpu_device_type_direct_construction() -> None:
    config, _preprocessing, _inventory_obj = _config()
    mapping = phase8_definitive_training_config_to_dict(config)
    mapping["device_type"] = "cuda"
    mapping["config_hash"] = sha256_json(
        {key: value for key, value in mapping.items() if key != "config_hash"}
    )
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        phase8_definitive_training_config_from_mapping(mapping)


def test_config_rejects_amp_enabled_direct_construction() -> None:
    config, _preprocessing, _inventory_obj = _config()
    mapping = phase8_definitive_training_config_to_dict(config)
    mapping["amp_enabled"] = True
    mapping["config_hash"] = sha256_json(
        {key: value for key, value in mapping.items() if key != "config_hash"}
    )
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        phase8_definitive_training_config_from_mapping(mapping)


def test_config_rejects_non_awaiting_execution_release_state() -> None:
    config, _preprocessing, _inventory_obj = _config()
    mapping = phase8_definitive_training_config_to_dict(config)
    mapping["execution_release_state"] = "released"
    mapping["config_hash"] = sha256_json(
        {key: value for key, value in mapping.items() if key != "config_hash"}
    )
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        phase8_definitive_training_config_from_mapping(mapping)


# ---------------------------------------------------------------------------
# 4: release fixed-value invariants
# ---------------------------------------------------------------------------


def test_unreleased_release_builder_never_produces_released() -> None:
    config, _preprocessing, _inventory_obj = _config()
    release = build_unreleased_definitive_execution_release(
        bound_config_hash=config.config_hash,
        release_scope_description="scope",
        authorized_max_training_steps=5,
    )
    assert release.release_state == RELEASE_STATE_NOT_RELEASED
    assert release.authorized_by is None


def test_release_requires_authorized_by_when_released() -> None:
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        Phase8DefinitiveExecutionRelease(
            schema_name=PHASE8_DEFINITIVE_EXECUTION_RELEASE_SCHEMA_NAME,
            schema_version=PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION,
            release_state=RELEASE_STATE_RELEASED,
            bound_config_hash=HASH_A,
            release_scope_description="scope",
            authorized_max_training_steps=1,
            authorized_by=None,
            release_hash=HASH_A,
        )


def test_release_rejects_authorized_by_when_not_released() -> None:
    payload = {
        "authorized_by": RELEASE_AUTHORIZED_BY_EXPLICIT_USER_AUTHORIZATION,
        "authorized_max_training_steps": 1,
        "bound_config_hash": HASH_A,
        "release_scope_description": "scope",
        "release_state": RELEASE_STATE_NOT_RELEASED,
        "schema_name": PHASE8_DEFINITIVE_EXECUTION_RELEASE_SCHEMA_NAME,
        "schema_version": PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION,
    }
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        Phase8DefinitiveExecutionRelease(
            schema_name=PHASE8_DEFINITIVE_EXECUTION_RELEASE_SCHEMA_NAME,
            schema_version=PHASE8_DEFINITIVE_TRAINING_SCHEMA_VERSION,
            release_state=RELEASE_STATE_NOT_RELEASED,
            bound_config_hash=HASH_A,
            release_scope_description="scope",
            authorized_max_training_steps=1,
            authorized_by=RELEASE_AUTHORIZED_BY_EXPLICIT_USER_AUTHORIZATION,
            release_hash=sha256_json(payload),
        )


# ---------------------------------------------------------------------------
# 5: explicit release required for execution and all publishers
# ---------------------------------------------------------------------------


def test_execute_without_release_raises_not_released() -> None:
    config, _preprocessing, _inventory_obj = _config()
    release = build_unreleased_definitive_execution_release(
        bound_config_hash=config.config_hash,
        release_scope_description="scope",
        authorized_max_training_steps=5,
    )
    with pytest.raises(Phase8DefinitiveTrainingNotReleasedError):
        execute_phase8_definitive_training(config, release)


def test_execute_with_mismatched_release_raises_validation_error_even_if_released() -> None:
    config, _preprocessing, _inventory_obj = _config()
    other_config, _p2, _i2 = _config(training_config_hash=HASH_B)
    mismatched_released = _released_release(other_config)
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        execute_phase8_definitive_training(config, mismatched_released)


def test_execute_with_released_but_no_executor_still_fails_closed() -> None:
    config, _preprocessing, _inventory_obj = _config()
    released = _released_release(config)
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        execute_phase8_definitive_training(config, released)


@pytest.mark.parametrize(
    "publisher_name",
    [
        "publish_phase8_definitive_preprocessing_evidence",
        "publish_phase8_definitive_checkpoint_metadata",
        "publish_phase8_definitive_validation_evidence",
        "publish_phase8_definitive_model_selection",
        "publish_phase8_definitive_threshold_decision",
        "publish_phase8_definitive_support_policy",
    ],
)
def test_all_publishers_require_released_release(publisher_name: str, tmp_path: Path) -> None:
    config, preprocessing, inventory = _config()
    checkpoint_metadata = _checkpoint_metadata(config=config)
    validation_evidence = _validation_evidence(config=config)
    not_released = build_unreleased_definitive_execution_release(
        bound_config_hash=config.config_hash,
        release_scope_description="scope",
        authorized_max_training_steps=5,
    )
    output_root = tmp_path / "out"
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    kwargs_by_publisher: dict[str, dict[str, Any]] = {
        "publish_phase8_definitive_preprocessing_evidence": {
            "preprocessing_decision": preprocessing,
        },
        "publish_phase8_definitive_checkpoint_metadata": {
            "checkpoint_metadata": checkpoint_metadata,
        },
        "publish_phase8_definitive_validation_evidence": {
            "validation_evidence": validation_evidence,
        },
        "publish_phase8_definitive_model_selection": {
            "candidate_inventory": inventory,
            "selected_checkpoint_metadata": checkpoint_metadata,
            "selected_validation_evidence": validation_evidence,
        },
        "publish_phase8_definitive_threshold_decision": {"checkpoint_hash": HASH_D},
        "publish_phase8_definitive_support_policy": {"checkpoint_hash": HASH_D},
    }
    publishers_by_name = {
        "publish_phase8_definitive_preprocessing_evidence": (
            publish_phase8_definitive_preprocessing_evidence
        ),
        "publish_phase8_definitive_checkpoint_metadata": (
            publish_phase8_definitive_checkpoint_metadata
        ),
        "publish_phase8_definitive_validation_evidence": (
            publish_phase8_definitive_validation_evidence
        ),
        "publish_phase8_definitive_model_selection": publish_phase8_definitive_model_selection,
        "publish_phase8_definitive_threshold_decision": (
            publish_phase8_definitive_threshold_decision
        ),
        "publish_phase8_definitive_support_policy": publish_phase8_definitive_support_policy,
    }
    publisher: Any = publishers_by_name[publisher_name]

    with pytest.raises(Phase8DefinitiveTrainingNotReleasedError):
        publisher(
            config=config,
            release=not_released,
            output_root=output_root,
            repository_root=repository_root,
            **kwargs_by_publisher[publisher_name],
        )
    assert not output_root.exists()


# ---------------------------------------------------------------------------
# 6: release/config binding mismatch
# ---------------------------------------------------------------------------


def test_release_config_hash_mismatch_rejected_even_when_released() -> None:
    config, _preprocessing, _inventory_obj = _config()
    other_config, _p2, _i2 = _config(training_config_hash=HASH_C)
    released_for_other = _released_release(other_config)
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        execute_phase8_definitive_training(config, released_for_other)


# ---------------------------------------------------------------------------
# 7: executor boundary proves zero file access before raising
# ---------------------------------------------------------------------------


def test_execute_raises_before_any_file_open(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    release = build_unreleased_definitive_execution_release(
        bound_config_hash=config.config_hash,
        release_scope_description="scope",
        authorized_max_training_steps=5,
    )
    real_open = builtins.open
    calls: list[Any] = []

    def _spy_open(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real_open(*args, **kwargs)

    with (
        patch("builtins.open", side_effect=_spy_open),
        pytest.raises(Phase8DefinitiveTrainingNotReleasedError),
    ):
        execute_phase8_definitive_training(config, release)
    assert calls == []


def test_publisher_raises_before_any_file_open(tmp_path: Path) -> None:
    config, preprocessing, _inventory_obj = _config()
    not_released = build_unreleased_definitive_execution_release(
        bound_config_hash=config.config_hash,
        release_scope_description="scope",
        authorized_max_training_steps=5,
    )
    output_root = tmp_path / "out"
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    real_open = builtins.open
    calls: list[Any] = []

    def _spy_open(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real_open(*args, **kwargs)

    with (
        patch("builtins.open", side_effect=_spy_open),
        pytest.raises(Phase8DefinitiveTrainingNotReleasedError),
    ):
        publish_phase8_definitive_preprocessing_evidence(
            config=config,
            release=not_released,
            preprocessing_decision=preprocessing,
            output_root=output_root,
            repository_root=repository_root,
        )
    assert calls == []
    assert not output_root.exists()


# ---------------------------------------------------------------------------
# 8: cross-hash mismatch rejections (via a directly-constructed released release)
# ---------------------------------------------------------------------------


def test_preprocessing_evidence_publisher_rejects_wrong_hash_binding(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    released = _released_release(config)
    wrong_preprocessing = _preprocessing_decision(
        candidate_id=REQUIRED_CANDIDATE_ID, development_manifest_hash=HASH_D
    )
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        publish_phase8_definitive_preprocessing_evidence(
            config=config,
            release=released,
            preprocessing_decision=wrong_preprocessing,
            output_root=tmp_path / "out",
            repository_root=tmp_path,
        )


def test_checkpoint_metadata_publisher_rejects_wrong_training_config_hash(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    released = _released_release(config)
    checkpoint_metadata = _checkpoint_metadata(config=config, training_config_hash=HASH_C)
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        publish_phase8_definitive_checkpoint_metadata(
            config=config,
            release=released,
            checkpoint_metadata=checkpoint_metadata,
            output_root=tmp_path / "out",
            repository_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# 9: single-candidate cardinality enforcement
# ---------------------------------------------------------------------------


def test_model_selection_rejects_two_candidate_inventory(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    released = _released_release(config)
    checkpoint_metadata = _checkpoint_metadata(config=config)
    validation_evidence = _validation_evidence(config=config)
    other_preprocessing = _preprocessing_decision(candidate_id="other_candidate")
    two_candidate_inventory = _inventory(
        candidates=(
            _candidate(
                preprocessing_evidence_hash=(config.preprocessing_decision_reference.artifact_hash)
            ),
            _candidate(
                candidate_id="other_candidate",
                preprocessing_evidence_hash=other_preprocessing.preprocessing_decision_hash,
            ),
        )
    )
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        publish_phase8_definitive_model_selection(
            config=config,
            release=released,
            candidate_inventory=two_candidate_inventory,
            selected_checkpoint_metadata=checkpoint_metadata,
            selected_validation_evidence=validation_evidence,
            output_root=tmp_path / "out",
            repository_root=tmp_path,
        )


def test_model_selection_rejects_mismatched_candidate_id(tmp_path: Path) -> None:
    config, _preprocessing, inventory = _config()
    released = _released_release(config)
    checkpoint_metadata = _checkpoint_metadata(config=config, candidate_id="mismatched_candidate")
    validation_evidence = _validation_evidence(config=config)
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        publish_phase8_definitive_model_selection(
            config=config,
            release=released,
            candidate_inventory=inventory,
            selected_checkpoint_metadata=checkpoint_metadata,
            selected_validation_evidence=validation_evidence,
            output_root=tmp_path / "out",
            repository_root=tmp_path,
        )


def test_model_selection_success_uses_fixed_tie_break_order(tmp_path: Path) -> None:
    config, _preprocessing, inventory = _config()
    released = _released_release(config)
    checkpoint_metadata = _checkpoint_metadata(config=config)
    validation_evidence = _validation_evidence(config=config)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "out"
    published_hash = publish_phase8_definitive_model_selection(
        config=config,
        release=released,
        candidate_inventory=inventory,
        selected_checkpoint_metadata=checkpoint_metadata,
        selected_validation_evidence=validation_evidence,
        output_root=output_root,
        repository_root=repository_root,
    )
    assert len(published_hash) == 64
    from protoem_ct.external.definitive_training import PHASE8_DEFINITIVE_MODEL_SELECTION_FILENAME

    content = (output_root / PHASE8_DEFINITIVE_MODEL_SELECTION_FILENAME).read_text(encoding="utf-8")
    assert "single_candidate_preregistered" in content
    for rule in DEFINITIVE_TIE_BREAK_ORDER:
        assert rule in content


# ---------------------------------------------------------------------------
# 10: internal-test / external-data rejection
# ---------------------------------------------------------------------------


def test_config_direct_construction_rejects_internal_test_included() -> None:
    config, _preprocessing, _inventory_obj = _config()
    mapping = phase8_definitive_training_config_to_dict(config)
    mapping["internal_test_excluded"] = False
    mapping["config_hash"] = sha256_json(
        {key: value for key, value in mapping.items() if key != "config_hash"}
    )
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        phase8_definitive_training_config_from_mapping(mapping)


def test_validation_evidence_publisher_rejects_internal_test_role(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    released = _released_release(config)
    validation_evidence = _validation_evidence(
        config=config, empty_target_role="internal_test_accounting"
    )
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        publish_phase8_definitive_validation_evidence(
            config=config,
            release=released,
            validation_evidence=validation_evidence,
            output_root=tmp_path / "out",
            repository_root=tmp_path,
        )


def test_validation_evidence_publisher_rejects_external_role(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    released = _released_release(config)
    validation_evidence = _validation_evidence(
        config=config, empty_target_role="external_accounting"
    )
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        publish_phase8_definitive_validation_evidence(
            config=config,
            release=released,
            validation_evidence=validation_evidence,
            output_root=tmp_path / "out",
            repository_root=tmp_path,
        )


def test_validation_evidence_publisher_accepts_clean_role(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    released = _released_release(config)
    validation_evidence = _validation_evidence(config=config)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    published_hash = publish_phase8_definitive_validation_evidence(
        config=config,
        release=released,
        validation_evidence=validation_evidence,
        output_root=tmp_path / "out",
        repository_root=repository_root,
    )
    assert len(published_hash) == 64


# ---------------------------------------------------------------------------
# 11: fixed threshold exactly 0.5, no search parameter
# ---------------------------------------------------------------------------


def test_threshold_publisher_signature_has_no_threshold_value_or_grid_parameter() -> None:
    signature = inspect.signature(publish_phase8_definitive_threshold_decision)
    names = set(signature.parameters)
    assert "threshold_value" not in names
    assert not any("grid" in name or "search" in name for name in names)


def test_threshold_publisher_always_outputs_fixed_half(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    released = _released_release(config)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    from protoem_ct.external.definitive_training import (
        PHASE8_DEFINITIVE_THRESHOLD_DECISION_FILENAME,
    )

    output_root = tmp_path / "out"
    publish_phase8_definitive_threshold_decision(
        config=config,
        release=released,
        checkpoint_hash=HASH_D,
        output_root=output_root,
        repository_root=repository_root,
    )
    content = (output_root / PHASE8_DEFINITIVE_THRESHOLD_DECISION_FILENAME).read_text(
        encoding="utf-8"
    )
    assert '"threshold_value":0.5' in content
    assert '"policy_type":"fixed_constant"' in content


# ---------------------------------------------------------------------------
# 12: no_support behavior and signature check
# ---------------------------------------------------------------------------


def test_support_publisher_signature_has_no_manifest_parameter() -> None:
    signature = inspect.signature(publish_phase8_definitive_support_policy)
    names = set(signature.parameters)
    assert not any("manifest" in name for name in names)


def test_support_publisher_always_outputs_no_support(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    released = _released_release(config)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    from protoem_ct.external.definitive_training import PHASE8_DEFINITIVE_SUPPORT_POLICY_FILENAME

    output_root = tmp_path / "out"
    publish_phase8_definitive_support_policy(
        config=config,
        release=released,
        checkpoint_hash=HASH_D,
        output_root=output_root,
        repository_root=repository_root,
    )
    content = (output_root / PHASE8_DEFINITIVE_SUPPORT_POLICY_FILENAME).read_text(encoding="utf-8")
    assert '"policy_type":"no_support"' in content


# ---------------------------------------------------------------------------
# 13-14: synthetic/incomplete checkpoint rejection
# ---------------------------------------------------------------------------


def test_synthetic_smoke_checkpoint_cannot_be_freeze_eligible() -> None:
    with pytest.raises(Phase8InternalEvidenceValidationError):
        _checkpoint_metadata(
            config=_config()[0], completion_status="synthetic_smoke", freeze_eligible=True
        )


def test_checkpoint_publisher_guards_freeze_eligible_completion_status(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    released = _released_release(config)
    valid_checkpoint = _checkpoint_metadata(config=config, completion_status="completed")
    # Bypass the frozen dataclass's own __post_init__ guard via object.__setattr__ to
    # reach this module's redundant defensive guard directly.
    tampered_checkpoint = replace(valid_checkpoint)
    object.__setattr__(tampered_checkpoint, "completion_status", "synthetic_smoke")
    with pytest.raises(Phase8DefinitiveTrainingValidationError):
        publish_phase8_definitive_checkpoint_metadata(
            config=config,
            release=released,
            checkpoint_metadata=tampered_checkpoint,
            output_root=tmp_path / "out",
            repository_root=tmp_path,
        )


def test_incomplete_checkpoint_byte_size_rejected() -> None:
    with pytest.raises(Phase8InternalEvidenceValidationError):
        _checkpoint_metadata(config=_config()[0], checkpoint_byte_size=0, freeze_eligible=False)


def test_freeze_eligible_without_validation_reference_rejected() -> None:
    config, _preprocessing, _inventory_obj = _config()
    with pytest.raises(Phase8InternalEvidenceValidationError):
        Phase8CheckpointMetadata(
            schema_name="phase8_checkpoint_metadata",
            schema_version="v1",
            checkpoint_sha256=HASH_D,
            checkpoint_byte_size=1024,
            serialization_format="torch_state_dict",
            model_family="monai_segresnet",
            candidate_id=config.candidate_id,
            seed=1729,
            training_adaptation_mode="baseline_training",
            originating_git_commit=GIT_COMMIT,
            package_environment_reference=_reference("package_environment", HASH_A),
            training_config_hash=config.training_config_reference.artifact_hash,
            preprocessing_evidence_hash=config.preprocessing_decision_reference.artifact_hash,
            development_manifest_hash=HASH_A,
            development_split_hash=HASH_C,
            validation_metric_artifact_reference=None,
            device_type="cpu",
            amp_state="disabled",
            completion_status="completed",
            failure_codes=(),
            no_external_data=True,
            freeze_eligible=True,
            checkpoint_metadata_hash=HASH_A,
        )


# ---------------------------------------------------------------------------
# 15: output-root safety
# ---------------------------------------------------------------------------


def test_output_root_rejects_symlink(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    real_target = tmp_path / "real-target"
    real_target.mkdir()
    output_root = tmp_path / "symlinked-output"
    output_root.symlink_to(real_target, target_is_directory=True)
    with pytest.raises(Phase8DefinitiveTrainingPublicationError):
        validate_phase8_definitive_output_root(
            output_root=output_root, repository_root=repository_root
        )


def test_output_root_rejects_inside_repository(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = repository_root / "runs" / "phase8-definitive"
    with pytest.raises(Phase8DefinitiveTrainingPublicationError):
        validate_phase8_definitive_output_root(
            output_root=output_root, repository_root=repository_root
        )


def test_plan_publication_rejects_nonempty_output_root(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    release = build_unreleased_definitive_execution_release(
        bound_config_hash=config.config_hash,
        release_scope_description="scope",
        authorized_max_training_steps=5,
    )
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "preexisting.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Phase8DefinitiveTrainingPublicationError):
        publish_phase8_definitive_training_plan(
            config=config, release=release, output_root=output_root, repository_root=repository_root
        )


# ---------------------------------------------------------------------------
# 16: deterministic two-root plan publication
# ---------------------------------------------------------------------------


def test_plan_publication_is_byte_identical_across_two_output_roots(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    release = build_unreleased_definitive_execution_release(
        bound_config_hash=config.config_hash,
        release_scope_description="scope",
        authorized_max_training_steps=5,
    )
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    (tmp_path / "runs").mkdir()
    output_root_1 = tmp_path / "runs" / "attempt-1"
    output_root_2 = tmp_path / "runs" / "attempt-2"

    result_1 = publish_phase8_definitive_training_plan(
        config=config,
        release=release,
        output_root=output_root_1,
        repository_root=repository_root,
    )
    result_2 = publish_phase8_definitive_training_plan(
        config=config,
        release=release,
        output_root=output_root_2,
        repository_root=repository_root,
    )
    assert result_1.artifact_hashes == result_2.artifact_hashes
    for filename in sorted(result_1.artifact_hashes):
        assert (output_root_1 / filename).read_bytes() == (output_root_2 / filename).read_bytes()


def test_plan_publication_rejects_release_config_mismatch(tmp_path: Path) -> None:
    config, _preprocessing, _inventory_obj = _config()
    other_config, _p2, _i2 = _config(training_config_hash=HASH_C)
    release = build_unreleased_definitive_execution_release(
        bound_config_hash=other_config.config_hash,
        release_scope_description="scope",
        authorized_max_training_steps=5,
    )
    with pytest.raises(Phase8DefinitiveTrainingPublicationError):
        publish_phase8_definitive_training_plan(
            config=config,
            release=release,
            output_root=tmp_path / "out",
            repository_root=tmp_path / "repo",
        )


# ---------------------------------------------------------------------------
# 17-18: CLI --help and synthetic plan-only smoke run
# ---------------------------------------------------------------------------


def test_cli_help_lists_new_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "plan-phase8-definitive-development-training" in output
    assert "run-phase8-definitive-development-training" in output

    plan_help = CliRunner().invoke(app, ["plan-phase8-definitive-development-training", "--help"])
    assert plan_help.exit_code == 0, plan_help.output
    plan_output = strip_ansi(plan_help.output)
    assert "--input-binding-hash" in plan_output
    assert "--candidate-inventory-path" in plan_output
    assert "--preprocessing-decision" in plan_output
    assert "awaiting_explicit_user_authorization" in plan_output

    run_help = CliRunner().invoke(app, ["run-phase8-definitive-development-training", "--help"])
    assert run_help.exit_code == 0, run_help.output
    run_output = strip_ansi(run_help.output)
    assert "--config-path" in run_output
    assert "--release-path" in run_output
    assert "refuses" in run_output


def _inventory_mapping(preprocessing_hash: str) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "schema_name": "phase8_fixed_candidate_inventory",
        "schema_version": "v1",
        "inventory_state": "freeze_ready",
        "candidate_set_locked": True,
        "selection_started": True,
        "supports_single_candidate_without_comparison": True,
        "candidates": [
            {
                "candidate_id": REQUIRED_CANDIDATE_ID,
                "model_family": "monai_segresnet",
                "method_identity": "monai_segresnet_baseline",
                "training_adaptation_mode": "baseline_training",
                "training_config_reference": {
                    "schema_name": "phase8_training_config",
                    "schema_version": "v1",
                    "artifact_hash": HASH_A,
                    "artifact_role": "training_config",
                },
                "preprocessing_evidence_hash": preprocessing_hash,
                "fixed_seeds": [1729],
                "executable_readiness_state": "executable",
                "failure_handling_policy": "single_candidate_no_comparison",
                "uses_external_artifacts": False,
            }
        ],
    }
    payload = deepcopy(mapping)
    payload.pop("inventory_hash", None)
    mapping["inventory_hash"] = sha256_json(payload)
    return mapping


def test_cli_successful_synthetic_plan_only_run(tmp_path: Path) -> None:
    import json

    preprocessing_decision = _preprocessing_decision()
    preprocessing_path = tmp_path / "preprocessing.json"
    from protoem_ct.external.internal_evidence import phase8_preprocessing_decision_to_dict

    preprocessing_path.write_text(
        json.dumps(phase8_preprocessing_decision_to_dict(preprocessing_decision)),
        encoding="utf-8",
    )
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            _inventory_mapping(
                preprocessing_hash=preprocessing_decision.preprocessing_decision_hash
            )
        ),
        encoding="utf-8",
    )
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    (tmp_path / "runs").mkdir()
    output_root = tmp_path / "runs" / "definitive-plan-cli"

    result = CliRunner().invoke(
        app,
        [
            "plan-phase8-definitive-development-training",
            "--input-binding-hash",
            HASH_B,
            "--candidate-inventory-path",
            str(inventory_path),
            "--candidate-id",
            REQUIRED_CANDIDATE_ID,
            "--preprocessing-decision-path",
            str(preprocessing_path),
            "--release-scope-description",
            "cli-smoke-test",
            "--authorized-max-training-steps",
            "10",
            "--output-root",
            str(output_root),
            "--repository-root",
            str(repository_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "awaiting_explicit_user_authorization" in result.output
    assert "not_released" in result.output
    assert (output_root / "phase8_definitive_training_config.json").is_file()
    assert (output_root / "phase8_definitive_execution_release.json").is_file()

    run_result = CliRunner().invoke(
        app,
        [
            "run-phase8-definitive-development-training",
            "--config-path",
            str(output_root / "phase8_definitive_training_config.json"),
            "--release-path",
            str(output_root / "phase8_definitive_execution_release.json"),
        ],
    )
    assert run_result.exit_code == 1
    assert "Phase8DefinitiveTrainingNotReleasedError" in run_result.output


# ---------------------------------------------------------------------------
# 19: no real training/checkpoint/metric/medical-file access
# ---------------------------------------------------------------------------


def test_module_never_unconditionally_imports_heavy_or_medical_libraries() -> None:
    import protoem_ct.external.definitive_training as module

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
    # Confirm no bare "import <heavy-lib>" or "from <heavy-lib> import ..." statement
    # appears anywhere in the source (including inside function bodies as a lazy
    # import), while still allowing the required domain vocabulary
    # "monai_segresnet_baseline" to appear in docstrings/identifiers.
    for name in forbidden_names:
        assert re.search(rf"^\s*(import|from)\s+{name}\b", source, flags=re.MULTILINE) is None


def test_executor_protocol_methods_all_raise_not_implemented() -> None:
    from protoem_ct.external.definitive_training import Phase8DefinitiveTrainingExecutor

    class _Spy:
        def construct_model(self, config: Any) -> object:
            raise NotImplementedError

        def load_training_data(self, config: Any) -> object:
            raise NotImplementedError

        def run_validation_inference(self, config: Any) -> object:
            raise NotImplementedError

        def persist_checkpoint(self, config: Any, model: Any) -> object:
            raise NotImplementedError

        def publish_validation_metrics(self, config: Any, predictions: Any) -> object:
            raise NotImplementedError

        def publish_preprocessing_evidence(self, config: Any) -> object:
            raise NotImplementedError

        def publish_checkpoint_metadata(self, config: Any, checkpoint: Any) -> object:
            raise NotImplementedError

    spy: Phase8DefinitiveTrainingExecutor = _Spy()
    config, _preprocessing, _inventory_obj = _config()
    released = _released_release(config)
    with pytest.raises(NotImplementedError):
        execute_phase8_definitive_training(config, released, executor=spy)
