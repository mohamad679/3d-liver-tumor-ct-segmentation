"""Unit tests for Phase 8 preregistered label-mapping policy contracts."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes
from protoem_ct.external.label_mapping import (
    PHASE8_LABEL_MAPPING_EXTERNAL_COHORT_IDENTITY,
    PHASE8_LABEL_MAPPING_POLICY_SCHEMA_NAME,
    PHASE8_LABEL_MAPPING_POLICY_SCHEMA_VERSION,
    PHASE8_LABEL_MAPPING_TARGET_CLASSES,
    PHASE8_LABEL_MAPPING_TARGET_TASK_IDENTITY,
    Phase8LabelMappingHashError,
    Phase8LabelMappingSerializationError,
    Phase8LabelMappingValidationError,
    apply_phase8_label_mapping_policy,
    build_default_phase8_label_mapping_policy,
    hash_phase8_label_mapping_policy,
    phase8_label_mapping_policy_from_json,
    phase8_label_mapping_policy_to_dict,
    phase8_label_mapping_policy_to_json,
)


def _policy_payload() -> dict[str, object]:
    return dict(phase8_label_mapping_policy_to_dict(build_default_phase8_label_mapping_policy()))


def test_default_label_mapping_policy_round_trip_and_canonical_bytes() -> None:
    policy = build_default_phase8_label_mapping_policy()
    encoded = phase8_label_mapping_policy_to_json(policy)

    restored = phase8_label_mapping_policy_from_json(encoded)

    assert restored == policy
    assert restored.schema_name == PHASE8_LABEL_MAPPING_POLICY_SCHEMA_NAME
    assert restored.schema_version == PHASE8_LABEL_MAPPING_POLICY_SCHEMA_VERSION
    assert restored.external_cohort_identity == PHASE8_LABEL_MAPPING_EXTERNAL_COHORT_IDENTITY
    assert restored.target_task_identity == PHASE8_LABEL_MAPPING_TARGET_TASK_IDENTITY
    assert restored.target_classes == PHASE8_LABEL_MAPPING_TARGET_CLASSES
    assert restored.policy_hash == hash_phase8_label_mapping_policy(restored)
    assert encoded == phase8_label_mapping_policy_to_json(restored)
    assert encoded == canonical_json_bytes(json.loads(encoded)) + b"\n"


def test_policy_canonicalizes_roles_and_rules_independent_of_input_order() -> None:
    policy = build_default_phase8_label_mapping_policy()
    reversed_policy = dataclasses.replace(
        policy,
        permitted_source_roles=tuple(reversed(policy.permitted_source_roles)),
        mapping_rules=tuple(reversed(policy.mapping_rules)),
    )

    assert reversed_policy == policy
    assert phase8_label_mapping_policy_to_json(reversed_policy) == (
        phase8_label_mapping_policy_to_json(policy)
    )
    assert [role.role_id for role in reversed_policy.permitted_source_roles] == [
        "liver_context_mask",
        "tumor_lesion_mask",
    ]
    assert [rule.rule_id for rule in reversed_policy.mapping_rules] == [
        "exclude_liver_context_from_target",
        "map_tumor_lesion_nonzero_to_foreground",
    ]


def test_unknown_field_and_self_hash_tampering_are_rejected() -> None:
    payload = _policy_payload()
    payload["created_at_utc"] = "2026-08-04T00:00:00Z"

    with pytest.raises(Phase8LabelMappingSerializationError, match="extra"):
        phase8_label_mapping_policy_from_json(canonical_json_bytes(payload))

    payload = _policy_payload()
    payload["policy_hash"] = "0" * 64
    with pytest.raises(Phase8LabelMappingHashError, match="policy_hash"):
        phase8_label_mapping_policy_from_json(canonical_json_bytes(payload))


def test_policy_rejects_local_paths_timestamps_and_empirical_claims_before_label_ledger() -> None:
    payload = _policy_payload()
    payload["source_basis_references"] = [
        "/Volumes/Lexar/labels",
        "docs_phase8_plan_label_mapping_policy",
    ]
    with pytest.raises(Phase8LabelMappingValidationError, match="local paths"):
        phase8_label_mapping_policy_from_json(canonical_json_bytes(payload))

    payload = _policy_payload()
    payload["verification_state"] = "empirically_verified_after_label_ledger"
    with pytest.raises(Phase8LabelMappingValidationError, match="expected/documented"):
        phase8_label_mapping_policy_from_json(canonical_json_bytes(payload))


def test_policy_requires_no_tuning_declaration_and_exact_task_classes() -> None:
    payload = _policy_payload()
    payload["no_tuning_declaration"] = False
    with pytest.raises(Phase8LabelMappingValidationError, match="no_tuning_declaration"):
        phase8_label_mapping_policy_from_json(canonical_json_bytes(payload))

    payload = _policy_payload()
    payload["target_classes"] = ["background", "liver_context", "tumor_foreground"]
    with pytest.raises(Phase8LabelMappingValidationError, match="target_classes"):
        phase8_label_mapping_policy_from_json(canonical_json_bytes(payload))


def test_policy_rejects_liver_context_as_target_foreground() -> None:
    payload = _policy_payload()
    rules_value = payload["mapping_rules"]
    assert isinstance(rules_value, list)
    rules = list(rules_value)
    liver_rule = dict(cast(Mapping[str, object], rules[0]))
    liver_rule["target_class"] = "tumor_foreground"
    liver_rule["rule_action"] = "nonzero_voxels_to_tumor_foreground"
    liver_rule["aggregation_operator"] = "union"
    rules[0] = liver_rule
    payload["mapping_rules"] = rules

    with pytest.raises(Phase8LabelMappingValidationError, match="liver_context_mask"):
        phase8_label_mapping_policy_from_json(canonical_json_bytes(payload))


def test_synthetic_nonzero_tumor_maps_to_foreground_and_liver_is_ignored() -> None:
    policy = build_default_phase8_label_mapping_policy()
    tumor = np.zeros((3, 4, 5), dtype=np.uint8)
    liver = np.ones((3, 4, 5), dtype=np.uint8)
    tumor[0, 0, 0] = 1
    tumor[2, 3, 4] = 7

    mapped = apply_phase8_label_mapping_policy(
        policy,
        {"liver_context_mask": liver, "tumor_lesion_mask": tumor},
    )

    assert mapped.dtype == np.bool_
    assert mapped.sum() == 2
    assert mapped[0, 0, 0]
    assert mapped[2, 3, 4]
    assert not mapped[1, 1, 1]


def test_empty_target_is_allowed_but_missing_unknown_or_incompatible_sources_fail_closed() -> None:
    policy = build_default_phase8_label_mapping_policy()
    empty = np.zeros((2, 2, 2), dtype=np.uint8)

    mapped = apply_phase8_label_mapping_policy(policy, {"tumor_lesion_mask": empty})

    assert not mapped.any()

    with pytest.raises(Phase8LabelMappingValidationError, match="missing"):
        apply_phase8_label_mapping_policy(policy, {"liver_context_mask": empty})
    with pytest.raises(Phase8LabelMappingValidationError, match="unknown source label roles"):
        apply_phase8_label_mapping_policy(
            policy,
            {"tumor_lesion_mask": empty, "source_label_1": empty},
        )
    with pytest.raises(Phase8LabelMappingValidationError, match="incompatible"):
        apply_phase8_label_mapping_policy(
            policy,
            {
                "liver_context_mask": np.zeros((2, 2, 3), dtype=np.uint8),
                "tumor_lesion_mask": empty,
            },
        )
    with pytest.raises(Phase8LabelMappingValidationError, match="finite"):
        apply_phase8_label_mapping_policy(
            policy,
            {"tumor_lesion_mask": np.asarray([[[np.nan]]], dtype=np.float32)},
        )
