"""Unit tests for Phase 8 frozen decision artifact contracts."""

from __future__ import annotations

import dataclasses
import json

import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.external.artifacts import (
    PHASE8_DECISION_FREEZE_SCHEMA_NAME,
    PHASE8_DECISION_FREEZE_SCHEMA_VERSION,
    PHASE8_FROZEN_DECISION_REFERENCE_SCHEMA_NAME,
    PHASE8_FROZEN_DECISION_REFERENCE_SCHEMA_VERSION,
    PHASE8_REQUIRED_DECISION_CATEGORIES,
    FrozenDecisionReference,
    Phase8ArtifactHashError,
    Phase8ArtifactSerializationError,
    Phase8ArtifactValidationError,
    Phase8DecisionFreeze,
    frozen_decision_reference_to_dict,
    hash_phase8_decision_freeze,
    phase8_decision_freeze_from_json,
    phase8_decision_freeze_identity_payload,
    phase8_decision_freeze_to_dict,
    phase8_decision_freeze_to_json,
)

HEX_A = "a" * 64


def _sha(seed: str) -> str:
    return sha256_json({"seed": seed})


def _reference(
    category: str, *, source_artifact_hash: str | None = None
) -> FrozenDecisionReference:
    return FrozenDecisionReference(
        schema_name=PHASE8_FROZEN_DECISION_REFERENCE_SCHEMA_NAME,
        schema_version=PHASE8_FROZEN_DECISION_REFERENCE_SCHEMA_VERSION,
        category=category,
        source_schema_name=f"{category}_artifact",
        source_schema_version="v1",
        source_phase="phase8_wave1_synthetic_fixture",
        source_artifact_hash=_sha(category)
        if source_artifact_hash is None
        else source_artifact_hash,
        rationale_code=f"{category}_freeze",
        provenance_reference="docs.phase8.provenance_record",
        frozen=True,
        freeze_state="frozen",
    )


def _references() -> tuple[FrozenDecisionReference, ...]:
    return tuple(_reference(category) for category in sorted(PHASE8_REQUIRED_DECISION_CATEGORIES))


def _inventory_payload(
    *,
    references: tuple[FrozenDecisionReference, ...] | None = None,
    freeze_state: str = "evaluation_ready",
    frozen_before_external_evaluation: bool = True,
) -> dict[str, object]:
    resolved_references = _references() if references is None else references
    return {
        "decision_references": [
            frozen_decision_reference_to_dict(item)
            for item in sorted(resolved_references, key=lambda item: item.category)
        ],
        "freeze_state": freeze_state,
        "frozen_before_external_evaluation": frozen_before_external_evaluation,
        "schema_name": PHASE8_DECISION_FREEZE_SCHEMA_NAME,
        "schema_version": PHASE8_DECISION_FREEZE_SCHEMA_VERSION,
    }


def _inventory(
    *,
    references: tuple[FrozenDecisionReference, ...] | None = None,
    freeze_state: str = "evaluation_ready",
    frozen_before_external_evaluation: bool = True,
) -> Phase8DecisionFreeze:
    payload = _inventory_payload(
        references=references,
        freeze_state=freeze_state,
        frozen_before_external_evaluation=frozen_before_external_evaluation,
    )
    return Phase8DecisionFreeze(
        schema_name=PHASE8_DECISION_FREEZE_SCHEMA_NAME,
        schema_version=PHASE8_DECISION_FREEZE_SCHEMA_VERSION,
        freeze_inventory_hash=sha256_json(payload),
        freeze_state=freeze_state,
        frozen_before_external_evaluation=frozen_before_external_evaluation,
        decision_references=_references() if references is None else references,
    )


def test_round_trip_and_canonical_byte_stability() -> None:
    inventory = _inventory()
    encoded = phase8_decision_freeze_to_json(inventory)

    parsed = phase8_decision_freeze_from_json(encoded)

    assert parsed == inventory
    assert hash_phase8_decision_freeze(parsed) == inventory.freeze_inventory_hash
    assert encoded == phase8_decision_freeze_to_json(parsed)
    assert encoded == canonical_json_bytes(json.loads(encoded)) + b"\n"
    assert phase8_decision_freeze_identity_payload(inventory)["schema_name"] == (
        PHASE8_DECISION_FREEZE_SCHEMA_NAME
    )


def test_unknown_field_rejection() -> None:
    payload = phase8_decision_freeze_to_dict(_inventory())
    payload["unexpected"] = True

    with pytest.raises(Phase8ArtifactSerializationError, match="extra"):
        phase8_decision_freeze_from_json(json.dumps(payload).encode("utf-8"))


def test_invalid_self_hash_rejection() -> None:
    payload = phase8_decision_freeze_to_dict(_inventory())
    payload["freeze_inventory_hash"] = "0" * 64

    with pytest.raises(Phase8ArtifactHashError, match="freeze_inventory_hash"):
        phase8_decision_freeze_from_json(json.dumps(payload).encode("utf-8"))


def test_missing_category_rejected() -> None:
    references = tuple(
        item for item in _references() if item.category != "publication_configuration"
    )
    payload = {
        "freeze_inventory_hash": HEX_A,
        **_inventory_payload(references=references),
    }

    with pytest.raises(Phase8ArtifactValidationError, match="missing required categories"):
        phase8_decision_freeze_from_json(json.dumps(payload).encode("utf-8"))


def test_duplicate_category_rejected() -> None:
    references = _references()
    duplicated = references + (
        dataclasses.replace(references[0], source_artifact_hash=_sha("dupe")),
    )
    payload = {
        "freeze_inventory_hash": HEX_A,
        **_inventory_payload(references=duplicated),
    }

    with pytest.raises(Phase8ArtifactValidationError, match="duplicate categories"):
        phase8_decision_freeze_from_json(json.dumps(payload).encode("utf-8"))


def test_input_order_independent_canonical_ordering_and_hash() -> None:
    ordered = _inventory(references=_references())
    reversed_inventory = _inventory(references=tuple(reversed(_references())))

    assert [item.category for item in reversed_inventory.decision_references] == sorted(
        PHASE8_REQUIRED_DECISION_CATEGORIES
    )
    assert reversed_inventory == ordered
    assert reversed_inventory.freeze_inventory_hash == ordered.freeze_inventory_hash
    assert phase8_decision_freeze_to_json(reversed_inventory) == phase8_decision_freeze_to_json(
        ordered
    )


def test_lowercase_sha_validation() -> None:
    with pytest.raises(Phase8ArtifactValidationError, match="lowercase"):
        dataclasses.replace(_references()[0], source_artifact_hash=("A" * 64))


def test_identity_changes_when_compatibility_critical_reference_fields_change() -> None:
    baseline = _inventory()
    changed_references = list(_references())
    changed_references[0] = dataclasses.replace(
        changed_references[0],
        source_artifact_hash=_sha("changed-critical-reference"),
    )

    changed = _inventory(references=tuple(changed_references))

    assert changed.freeze_inventory_hash != baseline.freeze_inventory_hash


def test_evaluation_ready_inventory_rejects_unresolved_or_draft_references() -> None:
    references = list(_references())
    references[0] = dataclasses.replace(
        references[0],
        rationale_code="unresolved",
        frozen=False,
        freeze_state="draft",
    )

    with pytest.raises(Phase8ArtifactValidationError, match="unresolved|frozen"):
        _inventory(references=tuple(references))
