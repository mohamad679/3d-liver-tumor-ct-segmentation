"""Unit tests for Phase 8 external preregistration contracts."""

from __future__ import annotations

import dataclasses

import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.external.preregistration import (
    PHASE8_ARTIFACT_REFERENCE_SCHEMA_NAME,
    PHASE8_ARTIFACT_REFERENCE_SCHEMA_VERSION,
    PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME,
    PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION,
    PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_NAME,
    PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_VERSION,
    Phase8ExternalPreregistration,
    Phase8PreregistrationHashError,
    Phase8PreregistrationSerializationError,
    Phase8PreregistrationValidationError,
    hash_phase8_external_preregistration,
    phase8_external_preregistration_from_json,
    phase8_external_preregistration_identity_payload,
    phase8_external_preregistration_to_dict,
    phase8_external_preregistration_to_json,
)

HEX_1 = "1" * 64
HEX_2 = "2" * 64
HEX_3 = "3" * 64
HEX_4 = "4" * 64
HEX_5 = "5" * 64
HEX_6 = "6" * 64
HEX_7 = "7" * 64
HEX_8 = "8" * 64
HEX_9 = "9" * 64
HEX_A = "a" * 64
HEX_B = "b" * 64
ZERO_HASH = "0" * 64


def _reference_payload(
    referenced_schema_name: str,
    artifact_hash: str | None,
    *,
    reference_state: str = "resolved",
) -> dict[str, object]:
    return {
        "schema_name": PHASE8_ARTIFACT_REFERENCE_SCHEMA_NAME,
        "schema_version": PHASE8_ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "referenced_schema_name": referenced_schema_name,
        "referenced_schema_version": "v1",
        "artifact_hash": artifact_hash,
        "reference_state": reference_state,
    }


def _inclusion_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_name": PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_NAME,
        "schema_version": PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_VERSION,
        "inclusion_state": "included",
        "policy_reference": _reference_payload("phase7_run_summary_policy", HEX_9),
    }
    payload.update(overrides)
    return payload


def _preregistration_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_name": PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME,
        "schema_version": PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION,
        "lifecycle_state": "evaluation_ready",
        "external_cohort_identifier": "3d_ircadb_01",
        "frozen_decision_inventory_hash": HEX_1,
        "anonymous_manifest_reference": _reference_payload("phase8_external_image_manifest", HEX_2),
        "eligibility_exclusion_policy_reference": _reference_payload(
            "phase8_eligibility_exclusion_policy", HEX_3
        ),
        "label_mapping_policy_reference": _reference_payload("phase8_label_mapping_policy", HEX_4),
        "domain_shift_record_reference": _reference_payload("phase8_domain_shift_record", HEX_5),
        "preregistered_segmentation_metrics": [
            "tumor_dice",
            "tumor_iou",
            "tumor_hd95",
            "tumor_normalized_surface_dice",
            "lesion_wise_recall",
            "lesion_wise_precision",
            "lesion_f1",
            "false_positive_lesions_per_scan",
            "tumor_volume_error",
        ],
        "bootstrap_policy_config_reference": _reference_payload(
            "phase8_bootstrap_policy_config", HEX_6
        ),
        "internal_external_comparison_policy_reference": _reference_payload(
            "phase8_internal_external_comparison_policy", HEX_7
        ),
        "qualitative_output_policy_reference": _reference_payload(
            "phase8_qualitative_output_policy", HEX_8
        ),
        "robustness_uncertainty_inclusion": _inclusion_payload(),
        "permitted_deviation_policy": "no_deviations_without_versioned_addendum",
        "no_tuning_declaration": True,
        "external_label_access_state": "unavailable_before_prediction_lock",
        "prediction_lock_requirement": "required_before_label_access",
    }
    payload.update(overrides)
    return payload


def _hashed_payload(**overrides: object) -> dict[str, object]:
    payload = _preregistration_payload(**overrides)
    return {"preregistration_hash": sha256_json(payload), **payload}


def _preregistration(**overrides: object) -> Phase8ExternalPreregistration:
    return phase8_external_preregistration_from_json(
        canonical_json_bytes(_hashed_payload(**overrides))
    )


def test_phase8_preregistration_round_trip() -> None:
    preregistration = _preregistration()

    restored = phase8_external_preregistration_from_json(
        phase8_external_preregistration_to_json(preregistration)
    )

    assert restored == preregistration
    assert phase8_external_preregistration_to_dict(restored) == (
        phase8_external_preregistration_to_dict(preregistration)
    )


def test_phase8_preregistration_canonical_byte_stability() -> None:
    preregistration = _preregistration()
    first = phase8_external_preregistration_to_json(preregistration)
    second = phase8_external_preregistration_to_json(
        phase8_external_preregistration_from_json(first)
    )

    assert first == second
    assert first.endswith(b"\n")
    assert (
        canonical_json_bytes(phase8_external_preregistration_to_dict(preregistration)) + b"\n"
        == first
    )


def test_phase8_preregistration_rejects_unknown_fields() -> None:
    payload = _hashed_payload()
    payload["created_at_utc"] = "2026-08-04T00:00:00Z"

    with pytest.raises(Phase8PreregistrationSerializationError, match="extra"):
        phase8_external_preregistration_from_json(canonical_json_bytes(payload))


def test_phase8_preregistration_rejects_invalid_self_hash() -> None:
    payload = _hashed_payload()
    payload["preregistration_hash"] = HEX_B

    with pytest.raises(Phase8PreregistrationHashError, match="preregistration_hash"):
        phase8_external_preregistration_from_json(canonical_json_bytes(payload))


def test_phase8_preregistration_accepts_unresolved_references_only_in_explicit_draft() -> None:
    draft = _preregistration(
        lifecycle_state="draft",
        frozen_decision_inventory_hash=None,
        anonymous_manifest_reference=_reference_payload(
            "phase8_external_image_manifest",
            None,
            reference_state="unresolved",
        ),
        robustness_uncertainty_inclusion=_inclusion_payload(
            inclusion_state="unresolved",
            policy_reference=None,
        ),
    )

    assert draft.lifecycle_state == "draft"
    assert draft.anonymous_manifest_reference.reference_state == "unresolved"

    with pytest.raises(Phase8PreregistrationValidationError, match="evaluation_ready"):
        _preregistration(
            lifecycle_state="evaluation_ready",
            anonymous_manifest_reference=_reference_payload(
                "phase8_external_image_manifest",
                None,
                reference_state="unresolved",
            ),
        )


def test_phase8_preregistration_evaluation_ready_rejects_missing_or_placeholder_references() -> (
    None
):
    with pytest.raises(Phase8PreregistrationValidationError, match="frozen_decision"):
        _preregistration(frozen_decision_inventory_hash=None)

    with pytest.raises(Phase8PreregistrationValidationError, match="non-placeholder"):
        _preregistration(
            bootstrap_policy_config_reference=_reference_payload(
                "phase8_bootstrap_policy_config",
                ZERO_HASH,
            )
        )


def test_phase8_preregistration_requires_no_tuning_declaration() -> None:
    with pytest.raises(Phase8PreregistrationValidationError, match="no_tuning_declaration"):
        _preregistration(no_tuning_declaration=False)


def test_phase8_preregistration_requires_external_labels_unavailable_before_prediction_lock() -> (
    None
):
    with pytest.raises(Phase8PreregistrationValidationError, match="external labels"):
        _preregistration(external_label_access_state="available_after_prediction_lock")


def test_phase8_preregistration_compatibility_critical_changes_affect_identity() -> None:
    original = _preregistration()
    changed_payload = phase8_external_preregistration_identity_payload(original)
    changed_payload["permitted_deviation_policy"] = "minor_documented_deviations_only"

    assert sha256_json(changed_payload) != hash_phase8_external_preregistration(original)

    changed = dataclasses.replace(
        original,
        permitted_deviation_policy="minor_documented_deviations_only",
        preregistration_hash=sha256_json(changed_payload),
    )
    assert changed.preregistration_hash != original.preregistration_hash
