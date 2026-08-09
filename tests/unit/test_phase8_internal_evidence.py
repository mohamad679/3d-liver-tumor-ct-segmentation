"""Unit tests for Phase 8 internal evidence contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.external.internal_evidence import (
    PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME,
    PHASE8_FIXED_CANDIDATE_INVENTORY_SCHEMA_NAME,
    PHASE8_INTERNAL_EVIDENCE_PACKAGE_SCHEMA_NAME,
    PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
    PHASE8_MODEL_SELECTION_DECISION_SCHEMA_NAME,
    PHASE8_PREPROCESSING_DECISION_SCHEMA_NAME,
    PHASE8_SUPPORT_POLICY_SCHEMA_NAME,
    PHASE8_THRESHOLD_DECISION_SCHEMA_NAME,
    PHASE8_VALIDATION_EVIDENCE_REFERENCE_SCHEMA_NAME,
    Phase8InternalEvidenceHashError,
    Phase8InternalEvidenceSerializationError,
    Phase8InternalEvidenceValidationError,
    phase8_fixed_candidate_inventory_from_mapping,
    phase8_internal_evidence_package_from_mapping,
    phase8_internal_evidence_package_to_dict,
    phase8_internal_evidence_package_to_json,
    phase8_model_selection_decision_from_mapping,
    phase8_preprocessing_decision_from_mapping,
    phase8_support_policy_from_mapping,
    phase8_threshold_decision_from_mapping,
    verify_checkpoint_file_identity_values,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64
HASH_1 = "1" * 64
HASH_2 = "2" * 64
HASH_3 = "3" * 64
HASH_4 = "4" * 64
HASH_5 = "5" * 64
HASH_6 = "6" * 64
HASH_7 = "7" * 64


def test_ready_package_reconstructs_and_serializes_canonically() -> None:
    """A complete development-only package reconstructs with stable hashes."""

    package_mapping = _ready_package_mapping()

    package = phase8_internal_evidence_package_from_mapping(package_mapping)
    round_trip = phase8_internal_evidence_package_from_mapping(
        phase8_internal_evidence_package_to_dict(package)
    )

    assert round_trip == package
    assert package.package_readiness_state == "READY"
    assert package.blocker_reason_codes == ()
    assert phase8_internal_evidence_package_to_json(package).endswith(b"\n")
    assert (
        package_mapping["package_hash"]
        == phase8_internal_evidence_package_to_dict(package)["package_hash"]
    )


def test_schema_names_and_versions_are_explicit_v1() -> None:
    """Every substage schema has the required explicit v1 identity."""

    mapping = _ready_package_mapping()

    assert mapping["schema_name"] == PHASE8_INTERNAL_EVIDENCE_PACKAGE_SCHEMA_NAME
    assert mapping["schema_version"] == PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION
    assert mapping["preprocessing_decision"]["schema_name"] == (
        PHASE8_PREPROCESSING_DECISION_SCHEMA_NAME
    )
    assert mapping["candidate_inventory"]["schema_name"] == (
        PHASE8_FIXED_CANDIDATE_INVENTORY_SCHEMA_NAME
    )
    assert mapping["checkpoint_metadata"]["schema_name"] == PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME
    assert mapping["validation_evidence"]["schema_name"] == (
        PHASE8_VALIDATION_EVIDENCE_REFERENCE_SCHEMA_NAME
    )
    assert mapping["model_selection_decision"]["schema_name"] == (
        PHASE8_MODEL_SELECTION_DECISION_SCHEMA_NAME
    )
    assert mapping["threshold_decision"]["schema_name"] == PHASE8_THRESHOLD_DECISION_SCHEMA_NAME
    assert mapping["support_policy"]["schema_name"] == PHASE8_SUPPORT_POLICY_SCHEMA_NAME


def test_unknown_fields_and_self_hash_tampering_are_rejected() -> None:
    """Strict reconstruction rejects extra fields and mismatched self-hashes."""

    package_mapping = _ready_package_mapping()
    extra_mapping = deepcopy(package_mapping)
    extra_mapping["unexpected"] = "field"

    with pytest.raises(Phase8InternalEvidenceSerializationError):
        phase8_internal_evidence_package_from_mapping(extra_mapping)

    tampered = deepcopy(package_mapping)
    tampered["threshold_decision"]["threshold_value"] = 0.7
    with pytest.raises(Phase8InternalEvidenceHashError):
        phase8_internal_evidence_package_from_mapping(tampered)


def test_candidate_inventory_rejects_duplicates_empty_and_external_artifacts() -> None:
    """Fixed inventory fails closed for mutable or external-dependent candidates."""

    inventory = _inventory_mapping()
    duplicate = deepcopy(inventory)
    duplicate["candidates"].append(deepcopy(duplicate["candidates"][0]))
    duplicate = _rehash(duplicate, "inventory_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_fixed_candidate_inventory_from_mapping(duplicate)

    empty = deepcopy(inventory)
    empty["candidates"] = []
    empty = _rehash(empty, "inventory_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_fixed_candidate_inventory_from_mapping(empty)

    external = deepcopy(inventory)
    external["candidates"][0]["uses_external_artifacts"] = True
    external = _rehash(external, "inventory_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_fixed_candidate_inventory_from_mapping(external)


def test_preprocessing_rejects_leakage_unresolved_freeze_and_path_identity() -> None:
    """Preprocessing cannot freeze when external, unresolved, or path-like."""

    preprocessing = _preprocessing_mapping()
    external = deepcopy(preprocessing)
    external["no_external_data"] = False
    external = _rehash(external, "preprocessing_decision_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_preprocessing_decision_from_mapping(external)

    unresolved = deepcopy(preprocessing)
    unresolved["orientation_policy_reference"] = None
    unresolved = _rehash(unresolved, "preprocessing_decision_hash")
    with pytest.raises(Phase8InternalEvidenceSerializationError):
        phase8_preprocessing_decision_from_mapping(unresolved)

    absolute_path = deepcopy(preprocessing)
    absolute_path["candidate_id"] = "/tmp/candidate"
    absolute_path = _rehash(absolute_path, "preprocessing_decision_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_preprocessing_decision_from_mapping(absolute_path)


def test_checkpoint_freeze_eligibility_requires_real_completed_validation_provenance() -> None:
    """Synthetic or provenance-missing checkpoints cannot become freeze eligible."""

    checkpoint = _checkpoint_mapping()
    synthetic = deepcopy(checkpoint)
    synthetic["completion_status"] = "synthetic_smoke"
    synthetic = _rehash(synthetic, "checkpoint_metadata_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_internal_evidence_package_from_mapping(
            _package_with_component(checkpoint_metadata=synthetic, readiness="BLOCKED")
        )

    missing_metric = deepcopy(checkpoint)
    missing_metric["validation_metric_artifact_reference"] = None
    missing_metric = _rehash(missing_metric, "checkpoint_metadata_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_internal_evidence_package_from_mapping(
            _package_with_component(checkpoint_metadata=missing_metric, readiness="BLOCKED")
        )


def test_validation_and_selection_reject_internal_test_or_external_evidence() -> None:
    """Selection evidence must be development-validation only."""

    package_mapping = _ready_package_mapping()
    internal_test_used = deepcopy(package_mapping)
    internal_test_used["validation_evidence"]["internal_test_not_used"] = False
    internal_test_used["validation_evidence"] = _rehash(
        internal_test_used["validation_evidence"],
        "validation_evidence_hash",
    )
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_internal_evidence_package_from_mapping(internal_test_used)

    selection_external = _selection_mapping()
    selection_external["external_data_not_used"] = False
    selection_external = _rehash(selection_external, "model_selection_decision_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_model_selection_decision_from_mapping(selection_external)


def test_threshold_and_support_policy_negative_cases() -> None:
    """Threshold/support policies reject unresolved or external-support states."""

    threshold = _threshold_mapping()
    threshold["threshold_value"] = float("nan")
    threshold["threshold_decision_hash"] = HASH_A
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_threshold_decision_from_mapping(threshold)

    unresolved_threshold = _threshold_mapping()
    unresolved_threshold["freeze_status"] = "freeze_ready"
    unresolved_threshold["threshold_value"] = None
    unresolved_threshold = _rehash(unresolved_threshold, "threshold_decision_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_threshold_decision_from_mapping(unresolved_threshold)

    support = _support_mapping()
    support["internal_support_manifest_reference"] = _reference("support_manifest", HASH_7)
    support = _rehash(support, "support_policy_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_support_policy_from_mapping(support)

    external_support = _support_mapping()
    external_support["external_support_state"] = "allowed"
    external_support = _rehash(external_support, "support_policy_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_support_policy_from_mapping(external_support)


def test_cross_contract_mismatches_fail_closed_and_blocked_codes_must_match() -> None:
    """Aggregate validation rejects inconsistent IDs and blocker accounting."""

    package_mapping = _ready_package_mapping()
    mismatch = deepcopy(package_mapping)
    mismatch["threshold_decision"]["candidate_id"] = "other_candidate"
    mismatch["threshold_decision"] = _rehash(
        mismatch["threshold_decision"],
        "threshold_decision_hash",
    )
    mismatch["package_readiness_state"] = "BLOCKED"
    mismatch["blocker_reason_codes"] = ["threshold_candidate_mismatch"]
    mismatch = _rehash(mismatch, "package_hash")
    package = phase8_internal_evidence_package_from_mapping(mismatch)
    assert package.blocker_reason_codes == ("threshold_candidate_mismatch",)

    bad_ready = deepcopy(mismatch)
    bad_ready["package_readiness_state"] = "READY"
    bad_ready = _rehash(bad_ready, "package_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_internal_evidence_package_from_mapping(bad_ready)

    wrong_blocker = deepcopy(mismatch)
    wrong_blocker["blocker_reason_codes"] = ["support_candidate_mismatch"]
    wrong_blocker = _rehash(wrong_blocker, "package_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_internal_evidence_package_from_mapping(wrong_blocker)


def test_multi_candidate_selected_workflow_mismatches_fail_closed() -> None:
    """READY cannot borrow preprocessing or training config from another candidate."""

    package_mapping = _ready_package_mapping()
    other_preprocessing_hash = sha256_json({"candidate": "other", "kind": "preprocessing"})
    other_training_hash = sha256_json({"candidate": "other", "kind": "training"})
    other_candidate = deepcopy(package_mapping["candidate_inventory"]["candidates"][0])
    other_candidate["candidate_id"] = "candidate_b"
    package_mapping["candidate_inventory"]["candidates"].append(other_candidate)
    package_mapping["model_selection_decision"]["selection_status"] = "freeze_ready"

    borrowed_preprocessing = deepcopy(package_mapping)
    borrowed_preprocessing["candidate_inventory"]["candidates"][0][
        "preprocessing_evidence_hash"
    ] = other_preprocessing_hash
    borrowed_preprocessing["candidate_inventory"] = _rehash(
        borrowed_preprocessing["candidate_inventory"],
        "inventory_hash",
    )
    borrowed_preprocessing["model_selection_decision"]["fixed_candidate_inventory_hash"] = (
        borrowed_preprocessing["candidate_inventory"]["inventory_hash"]
    )
    borrowed_preprocessing["model_selection_decision"] = _rehash(
        borrowed_preprocessing["model_selection_decision"],
        "model_selection_decision_hash",
    )
    borrowed_preprocessing["package_readiness_state"] = "BLOCKED"
    borrowed_preprocessing["blocker_reason_codes"] = ["candidate_preprocessing_reference_mismatch"]
    borrowed_preprocessing = _rehash(borrowed_preprocessing, "package_hash")
    package = phase8_internal_evidence_package_from_mapping(borrowed_preprocessing)
    assert package.blocker_reason_codes == ("candidate_preprocessing_reference_mismatch",)

    borrowed_training = deepcopy(package_mapping)
    borrowed_training["candidate_inventory"]["candidates"][0]["training_config_reference"][
        "artifact_hash"
    ] = other_training_hash
    borrowed_training["candidate_inventory"] = _rehash(
        borrowed_training["candidate_inventory"],
        "inventory_hash",
    )
    borrowed_training["model_selection_decision"]["fixed_candidate_inventory_hash"] = (
        borrowed_training["candidate_inventory"]["inventory_hash"]
    )
    borrowed_training["model_selection_decision"] = _rehash(
        borrowed_training["model_selection_decision"],
        "model_selection_decision_hash",
    )
    borrowed_training["package_readiness_state"] = "BLOCKED"
    borrowed_training["blocker_reason_codes"] = ["training_config_hash_mismatch"]
    borrowed_training = _rehash(borrowed_training, "package_hash")
    package = phase8_internal_evidence_package_from_mapping(borrowed_training)
    assert package.blocker_reason_codes == ("training_config_hash_mismatch",)

    borrowed_checkpoint_preprocessing = deepcopy(package_mapping)
    borrowed_checkpoint_preprocessing["preprocessing_decision"]["preprocessing_config_hash"] = (
        other_preprocessing_hash
    )
    borrowed_checkpoint_preprocessing["preprocessing_decision"] = _rehash(
        borrowed_checkpoint_preprocessing["preprocessing_decision"],
        "preprocessing_decision_hash",
    )
    borrowed_checkpoint_preprocessing["checkpoint_metadata"]["preprocessing_evidence_hash"] = (
        borrowed_checkpoint_preprocessing["preprocessing_decision"]["preprocessing_decision_hash"]
    )
    borrowed_checkpoint_preprocessing["checkpoint_metadata"] = _rehash(
        borrowed_checkpoint_preprocessing["checkpoint_metadata"],
        "checkpoint_metadata_hash",
    )
    borrowed_checkpoint_preprocessing["candidate_inventory"]["candidates"][1][
        "preprocessing_evidence_hash"
    ] = borrowed_checkpoint_preprocessing["preprocessing_decision"]["preprocessing_decision_hash"]
    borrowed_checkpoint_preprocessing["candidate_inventory"]["candidates"][0][
        "preprocessing_evidence_hash"
    ] = other_preprocessing_hash
    borrowed_checkpoint_preprocessing["candidate_inventory"] = _rehash(
        borrowed_checkpoint_preprocessing["candidate_inventory"],
        "inventory_hash",
    )
    borrowed_checkpoint_preprocessing["model_selection_decision"][
        "fixed_candidate_inventory_hash"
    ] = borrowed_checkpoint_preprocessing["candidate_inventory"]["inventory_hash"]
    borrowed_checkpoint_preprocessing["model_selection_decision"] = _rehash(
        borrowed_checkpoint_preprocessing["model_selection_decision"],
        "model_selection_decision_hash",
    )
    borrowed_checkpoint_preprocessing["package_readiness_state"] = "BLOCKED"
    borrowed_checkpoint_preprocessing["blocker_reason_codes"] = [
        "candidate_preprocessing_reference_mismatch"
    ]
    borrowed_checkpoint_preprocessing = _rehash(
        borrowed_checkpoint_preprocessing,
        "package_hash",
    )
    package = phase8_internal_evidence_package_from_mapping(borrowed_checkpoint_preprocessing)
    assert package.blocker_reason_codes == ("candidate_preprocessing_reference_mismatch",)

    borrowed_checkpoint_training = deepcopy(package_mapping)
    borrowed_checkpoint_training["checkpoint_metadata"]["training_config_hash"] = (
        other_preprocessing_hash
    )
    borrowed_checkpoint_training["checkpoint_metadata"] = _rehash(
        borrowed_checkpoint_training["checkpoint_metadata"],
        "checkpoint_metadata_hash",
    )
    borrowed_checkpoint_training["candidate_inventory"]["candidates"][1][
        "training_config_reference"
    ]["artifact_hash"] = other_preprocessing_hash
    borrowed_checkpoint_training["candidate_inventory"] = _rehash(
        borrowed_checkpoint_training["candidate_inventory"],
        "inventory_hash",
    )
    borrowed_checkpoint_training["model_selection_decision"]["fixed_candidate_inventory_hash"] = (
        borrowed_checkpoint_training["candidate_inventory"]["inventory_hash"]
    )
    borrowed_checkpoint_training["model_selection_decision"] = _rehash(
        borrowed_checkpoint_training["model_selection_decision"],
        "model_selection_decision_hash",
    )
    borrowed_checkpoint_training["package_readiness_state"] = "BLOCKED"
    borrowed_checkpoint_training["blocker_reason_codes"] = ["training_config_hash_mismatch"]
    borrowed_checkpoint_training = _rehash(borrowed_checkpoint_training, "package_hash")
    package = phase8_internal_evidence_package_from_mapping(borrowed_checkpoint_training)
    assert package.blocker_reason_codes == ("training_config_hash_mismatch",)


def test_checkpoint_candidate_identity_and_metric_mismatches_fail_closed() -> None:
    """READY requires selected checkpoint provenance and validation metric consistency."""

    package_mapping = _ready_package_mapping()
    checkpoint_model = deepcopy(package_mapping)
    checkpoint_model["checkpoint_metadata"]["model_family"] = "nnunet_v2"
    checkpoint_model["checkpoint_metadata"] = _rehash(
        checkpoint_model["checkpoint_metadata"],
        "checkpoint_metadata_hash",
    )
    checkpoint_model["package_readiness_state"] = "BLOCKED"
    checkpoint_model["blocker_reason_codes"] = ["checkpoint_model_family_mismatch"]
    checkpoint_model = _rehash(checkpoint_model, "package_hash")
    package = phase8_internal_evidence_package_from_mapping(checkpoint_model)
    assert package.blocker_reason_codes == ("checkpoint_model_family_mismatch",)

    checkpoint_mode = deepcopy(package_mapping)
    checkpoint_mode["checkpoint_metadata"]["training_adaptation_mode"] = "head_only"
    checkpoint_mode["checkpoint_metadata"] = _rehash(
        checkpoint_mode["checkpoint_metadata"],
        "checkpoint_metadata_hash",
    )
    checkpoint_mode["package_readiness_state"] = "BLOCKED"
    checkpoint_mode["blocker_reason_codes"] = ["checkpoint_training_mode_mismatch"]
    checkpoint_mode = _rehash(checkpoint_mode, "package_hash")
    package = phase8_internal_evidence_package_from_mapping(checkpoint_mode)
    assert package.blocker_reason_codes == ("checkpoint_training_mode_mismatch",)

    checkpoint_seed = deepcopy(package_mapping)
    checkpoint_seed["checkpoint_metadata"]["seed"] = 42
    checkpoint_seed["checkpoint_metadata"] = _rehash(
        checkpoint_seed["checkpoint_metadata"],
        "checkpoint_metadata_hash",
    )
    checkpoint_seed["package_readiness_state"] = "BLOCKED"
    checkpoint_seed["blocker_reason_codes"] = ["checkpoint_seed_not_in_candidate_fixed_seeds"]
    checkpoint_seed = _rehash(checkpoint_seed, "package_hash")
    package = phase8_internal_evidence_package_from_mapping(checkpoint_seed)
    assert package.blocker_reason_codes == ("checkpoint_seed_not_in_candidate_fixed_seeds",)

    metric_mismatch = deepcopy(package_mapping)
    metric_mismatch["model_selection_decision"]["primary_metric"] = "lesion_f1"
    metric_mismatch["model_selection_decision"] = _rehash(
        metric_mismatch["model_selection_decision"],
        "model_selection_decision_hash",
    )
    metric_mismatch["package_readiness_state"] = "BLOCKED"
    metric_mismatch["blocker_reason_codes"] = ["selection_primary_metric_mismatch"]
    metric_mismatch = _rehash(metric_mismatch, "package_hash")
    package = phase8_internal_evidence_package_from_mapping(metric_mismatch)
    assert package.blocker_reason_codes == ("selection_primary_metric_mismatch",)


def test_single_candidate_selection_is_explicit_without_comparison_claim() -> None:
    """Single-candidate inventory and selection are allowed only when explicit."""

    mapping = _ready_package_mapping()
    selection = mapping["model_selection_decision"]
    assert selection["selection_status"] == "single_candidate_preregistered"
    phase8_internal_evidence_package_from_mapping(mapping)

    multi_candidate = deepcopy(mapping)
    second_candidate = deepcopy(multi_candidate["candidate_inventory"]["candidates"][0])
    second_candidate["candidate_id"] = "candidate_b"
    second_candidate["method_identity"] = "monai_segresnet_baseline_seed2"
    second_candidate["fixed_seeds"] = [1730]
    second_candidate["failure_handling_policy"] = "record_and_exclude"
    multi_candidate["candidate_inventory"]["candidates"].append(second_candidate)
    multi_candidate["candidate_inventory"] = _rehash(
        multi_candidate["candidate_inventory"],
        "inventory_hash",
    )
    multi_candidate["model_selection_decision"]["fixed_candidate_inventory_hash"] = multi_candidate[
        "candidate_inventory"
    ]["inventory_hash"]
    multi_candidate["model_selection_decision"] = _rehash(
        multi_candidate["model_selection_decision"],
        "model_selection_decision_hash",
    )
    multi_candidate["package_readiness_state"] = "BLOCKED"
    multi_candidate["blocker_reason_codes"] = [
        "single_candidate_preregistered_inventory_cardinality_mismatch"
    ]
    multi_candidate = _rehash(multi_candidate, "package_hash")
    package = phase8_internal_evidence_package_from_mapping(multi_candidate)
    assert package.package_readiness_state == "BLOCKED"
    assert package.blocker_reason_codes == (
        "single_candidate_preregistered_inventory_cardinality_mismatch",
    )

    ready_multi_candidate = deepcopy(multi_candidate)
    ready_multi_candidate["package_readiness_state"] = "READY"
    ready_multi_candidate["blocker_reason_codes"] = []
    ready_multi_candidate = _rehash(ready_multi_candidate, "package_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_internal_evidence_package_from_mapping(ready_multi_candidate)

    inventory = _inventory_mapping()
    inventory["supports_single_candidate_without_comparison"] = False
    inventory = _rehash(inventory, "inventory_hash")
    with pytest.raises(Phase8InternalEvidenceValidationError):
        phase8_fixed_candidate_inventory_from_mapping(inventory)


def test_checkpoint_identity_validator_is_pure_and_strict() -> None:
    """Checkpoint digest verification uses supplied values and no filesystem path."""

    verify_checkpoint_file_identity_values(
        expected_checkpoint_sha256=HASH_A,
        expected_checkpoint_byte_size=1234,
        observed_checkpoint_sha256=HASH_A,
        observed_checkpoint_byte_size=1234,
    )

    with pytest.raises(Phase8InternalEvidenceHashError):
        verify_checkpoint_file_identity_values(
            expected_checkpoint_sha256=HASH_A,
            expected_checkpoint_byte_size=1234,
            observed_checkpoint_sha256=HASH_B,
            observed_checkpoint_byte_size=1234,
        )

    with pytest.raises(Phase8InternalEvidenceValidationError):
        verify_checkpoint_file_identity_values(
            expected_checkpoint_sha256=HASH_A,
            expected_checkpoint_byte_size=1234,
            observed_checkpoint_sha256=HASH_A,
            observed_checkpoint_byte_size=1235,
        )


def _ready_package_mapping() -> dict[str, Any]:
    return _package_with_component(readiness="READY")


def _package_with_component(
    *,
    preprocessing_decision: dict[str, Any] | None = None,
    candidate_inventory: dict[str, Any] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
    validation_evidence: dict[str, Any] | None = None,
    model_selection_decision: dict[str, Any] | None = None,
    threshold_decision: dict[str, Any] | None = None,
    support_policy: dict[str, Any] | None = None,
    readiness: str = "READY",
) -> dict[str, Any]:
    preprocessing = preprocessing_decision or _preprocessing_mapping()
    inventory = candidate_inventory or _inventory_mapping(
        preprocessing_hash=preprocessing["preprocessing_decision_hash"]
    )
    checkpoint = checkpoint_metadata or _checkpoint_mapping(
        preprocessing_hash=preprocessing["preprocessing_decision_hash"]
    )
    validation = validation_evidence or _validation_mapping()
    selection = model_selection_decision or _selection_mapping(
        inventory_hash=inventory["inventory_hash"],
        validation_hash=validation["validation_evidence_hash"],
    )
    threshold = threshold_decision or _threshold_mapping()
    support = support_policy or _support_mapping()
    blockers = [] if readiness == "READY" else ["checkpoint_not_freeze_eligible"]
    package = {
        "schema_name": PHASE8_INTERNAL_EVIDENCE_PACKAGE_SCHEMA_NAME,
        "schema_version": PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
        "preprocessing_decision": preprocessing,
        "candidate_inventory": inventory,
        "checkpoint_metadata": checkpoint,
        "validation_evidence": validation,
        "model_selection_decision": selection,
        "threshold_decision": threshold,
        "support_policy": support,
        "package_readiness_state": readiness,
        "blocker_reason_codes": blockers,
    }
    if readiness == "BLOCKED":
        package["blocker_reason_codes"] = _expected_blockers(package)
    return _rehash(package, "package_hash")


def _reference(role: str, artifact_hash: str = HASH_1) -> dict[str, Any]:
    return {
        "schema_name": f"phase8_{role}",
        "schema_version": "v1",
        "artifact_hash": artifact_hash,
        "artifact_role": role,
    }


def _preprocessing_mapping() -> dict[str, Any]:
    mapping = {
        "schema_name": PHASE8_PREPROCESSING_DECISION_SCHEMA_NAME,
        "schema_version": PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
        "candidate_id": "candidate_a",
        "development_manifest_hash": HASH_B,
        "development_split_hash": HASH_C,
        "preprocessing_config_schema_name": "phase8_preprocessing_config",
        "preprocessing_config_schema_version": "v1",
        "preprocessing_config_hash": HASH_D,
        "fit_scope": "no_data_dependent_fit",
        "fit_artifact_reference": None,
        "image_interpolation_policy": "linear",
        "label_interpolation_policy": "nearest",
        "orientation_policy_reference": _reference("orientation_policy", HASH_1),
        "spacing_policy_reference": _reference("spacing_policy", HASH_2),
        "intensity_policy_reference": _reference("intensity_policy", HASH_3),
        "crop_roi_policy_reference": _reference("crop_roi_policy", HASH_4),
        "normalization_policy_reference": _reference("normalization_policy", HASH_5),
        "originating_git_commit": "1ac48c3",
        "no_external_data": True,
        "evidence_status": "freeze_ready",
    }
    return _rehash(mapping, "preprocessing_decision_hash")


def _inventory_mapping(*, preprocessing_hash: str | None = None) -> dict[str, Any]:
    mapping = {
        "schema_name": PHASE8_FIXED_CANDIDATE_INVENTORY_SCHEMA_NAME,
        "schema_version": PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
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
                "training_config_reference": _reference("training_config", HASH_E),
                "preprocessing_evidence_hash": preprocessing_hash
                or _preprocessing_mapping()["preprocessing_decision_hash"],
                "fixed_seeds": [1729],
                "executable_readiness_state": "executable",
                "failure_handling_policy": "single_candidate_no_comparison",
                "uses_external_artifacts": False,
            }
        ],
    }
    return _rehash(mapping, "inventory_hash")


def _checkpoint_mapping(*, preprocessing_hash: str | None = None) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "schema_name": PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME,
        "schema_version": PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
        "checkpoint_sha256": HASH_A,
        "checkpoint_byte_size": 1234,
        "serialization_format": "torch_state_dict",
        "model_family": "monai_segresnet",
        "candidate_id": "candidate_a",
        "seed": 1729,
        "training_adaptation_mode": "baseline_training",
        "originating_git_commit": "1ac48c3",
        "package_environment_reference": _reference("package_environment", HASH_6),
        "training_config_hash": HASH_E,
        "preprocessing_evidence_hash": preprocessing_hash
        or _preprocessing_mapping()["preprocessing_decision_hash"],
        "development_manifest_hash": HASH_B,
        "development_split_hash": HASH_C,
        "validation_metric_artifact_reference": _reference("validation_metrics", HASH_F),
        "device_type": "cpu",
        "amp_state": "disabled",
        "completion_status": "completed",
        "failure_codes": [],
        "no_external_data": True,
        "freeze_eligible": True,
    }
    return _rehash(mapping, "checkpoint_metadata_hash")


def _validation_mapping() -> dict[str, Any]:
    mapping = {
        "schema_name": PHASE8_VALIDATION_EVIDENCE_REFERENCE_SCHEMA_NAME,
        "schema_version": PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
        "candidate_id": "candidate_a",
        "checkpoint_hash": HASH_A,
        "validation_cohort_split_hash": HASH_C,
        "metric_artifact_schema_name": "phase3_metric_report",
        "metric_artifact_schema_version": "v1",
        "metric_artifact_hash": HASH_F,
        "primary_metric_name": "tumor_dice",
        "valid_case_count": 20,
        "empty_target_accounting_reference": _reference("empty_target_accounting", HASH_7),
        "development_validation_only": True,
        "internal_test_not_used": True,
        "external_data_not_used": True,
    }
    return _rehash(mapping, "validation_evidence_hash")


def _selection_mapping(
    *,
    inventory_hash: str | None = None,
    validation_hash: str | None = None,
) -> dict[str, Any]:
    mapping = {
        "schema_name": PHASE8_MODEL_SELECTION_DECISION_SCHEMA_NAME,
        "schema_version": PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
        "fixed_candidate_inventory_hash": inventory_hash or _inventory_mapping()["inventory_hash"],
        "selection_rule_version": "validation_dice_tiebreak_v1",
        "primary_metric": "tumor_dice",
        "deterministic_tie_break_order": [
            "higher_tumor_dice",
            "higher_lesion_f1",
            "lower_hd95",
            "lower_false_positive_lesions",
            "smaller_trainable_parameter_count",
            "lexicographic_candidate_id",
        ],
        "selected_candidate_id": "candidate_a",
        "selected_checkpoint_hash": HASH_A,
        "selected_validation_evidence_hash": validation_hash
        or _validation_mapping()["validation_evidence_hash"],
        "failed_candidate_accounting": [],
        "selection_status": "single_candidate_preregistered",
        "internal_test_not_used": True,
        "external_data_not_used": True,
    }
    return _rehash(mapping, "model_selection_decision_hash")


def _threshold_mapping() -> dict[str, Any]:
    mapping = {
        "schema_name": PHASE8_THRESHOLD_DECISION_SCHEMA_NAME,
        "schema_version": PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
        "policy_type": "fixed_constant",
        "threshold_value": 0.5,
        "candidate_id": "candidate_a",
        "checkpoint_hash": HASH_A,
        "selection_objective": None,
        "candidate_threshold_grid_reference": None,
        "development_metric_reference": None,
        "fixed_constant_rationale": "predeclared_standard_probability_cutoff",
        "development_only_provenance": True,
        "internal_test_not_used": True,
        "external_data_not_used": True,
        "freeze_status": "freeze_ready",
    }
    return _rehash(mapping, "threshold_decision_hash")


def _support_mapping() -> dict[str, Any]:
    mapping = {
        "schema_name": PHASE8_SUPPORT_POLICY_SCHEMA_NAME,
        "schema_version": PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
        "policy_type": "no_support",
        "selected_candidate_id": "candidate_a",
        "selected_checkpoint_hash": HASH_A,
        "internal_support_manifest_reference": None,
        "support_k": None,
        "support_seed_reference": None,
        "external_support_labels_forbidden": True,
        "external_prototype_construction_forbidden": True,
        "external_support_state": "forbidden",
        "freeze_status": "freeze_ready",
    }
    return _rehash(mapping, "support_policy_hash")


def _rehash(mapping: dict[str, Any], hash_field: str) -> dict[str, Any]:
    result = deepcopy(mapping)
    payload = deepcopy(result)
    payload.pop(hash_field, None)
    if hash_field == "checkpoint_metadata_hash":
        payload.pop("amp_state", None)
        payload.pop("device_type", None)
    result[hash_field] = sha256_json(payload)
    return result


def _expected_blockers(package: dict[str, Any]) -> list[str]:
    blockers: set[str] = set()
    if not package["checkpoint_metadata"]["freeze_eligible"]:
        blockers.add("checkpoint_not_freeze_eligible")
    if package["checkpoint_metadata"]["completion_status"] != "completed":
        blockers.add("checkpoint_not_completed")
    if (
        package["threshold_decision"]["candidate_id"]
        != package["model_selection_decision"]["selected_candidate_id"]
    ):
        blockers.add("threshold_candidate_mismatch")
    return sorted(blockers)
