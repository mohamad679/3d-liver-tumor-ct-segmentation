"""Phase 8 external-validation frozen decision artifact contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json

PHASE8_DECISION_FREEZE_SCHEMA_NAME: Final[str] = "phase8_decision_freeze"
PHASE8_DECISION_FREEZE_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_FROZEN_DECISION_REFERENCE_SCHEMA_NAME: Final[str] = "phase8_frozen_decision_reference"
PHASE8_FROZEN_DECISION_REFERENCE_SCHEMA_VERSION: Final[str] = "v1"

PHASE8_REQUIRED_DECISION_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "bootstrap_configuration",
        "checkpoint_metadata",
        "label_mapping_policy",
        "metric_configuration",
        "model_selection_decision",
        "preprocessing_decision",
        "publication_configuration",
        "support_policy",
        "threshold_decision",
    }
)
PHASE8_INVENTORY_FREEZE_STATES: Final[frozenset[str]] = frozenset(
    {"draft", "frozen", "evaluation_ready"}
)
PHASE8_REFERENCE_FREEZE_STATES: Final[frozenset[str]] = frozenset({"draft", "frozen"})
PHASE8_UNRESOLVED_TOKENS: Final[frozenset[str]] = frozenset(
    {"pending", "placeholder", "tbd", "todo", "unknown", "unresolved"}
)

MappingLike: TypeAlias = Mapping[str, object]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_SCHEMA_VERSION_RE = re.compile(r"^v[1-9][0-9]*$")


class Phase8ArtifactError(ValueError):
    """Base error for Phase 8 artifact contracts."""


class Phase8ArtifactValidationError(Phase8ArtifactError):
    """Raised when a Phase 8 artifact violates its schema contract."""


class Phase8ArtifactSerializationError(Phase8ArtifactError):
    """Raised when a Phase 8 artifact mapping cannot be reconstructed safely."""


class Phase8ArtifactHashError(Phase8ArtifactError):
    """Raised when a Phase 8 artifact self-hash does not match its content."""


@dataclass(frozen=True, slots=True)
class FrozenDecisionReference:
    """Immutable reference to one already-decided compatibility-critical artifact."""

    schema_name: str
    schema_version: str
    category: str
    source_schema_name: str
    source_schema_version: str
    source_phase: str
    source_artifact_hash: str
    rationale_code: str | None
    provenance_reference: str | None
    frozen: bool
    freeze_state: str

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            expected=PHASE8_FROZEN_DECISION_REFERENCE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PHASE8_FROZEN_DECISION_REFERENCE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_allowed(
            self.category,
            PHASE8_REQUIRED_DECISION_CATEGORIES,
            field_name="category",
        )
        _require_identifier(self.source_schema_name, field_name="source_schema_name")
        _require_version_token(self.source_schema_version, field_name="source_schema_version")
        _require_identifier(self.source_phase, field_name="source_phase")
        _require_sha256(self.source_artifact_hash, field_name="source_artifact_hash")
        _require_optional_identifier(
            self.rationale_code,
            field_name="rationale_code",
            allow_unresolved=True,
        )
        _require_optional_identifier(
            self.provenance_reference,
            field_name="provenance_reference",
            allow_unresolved=True,
        )
        _require_allowed(
            self.freeze_state,
            PHASE8_REFERENCE_FREEZE_STATES,
            field_name="freeze_state",
        )
        if self.rationale_code is None and self.provenance_reference is None:
            raise Phase8ArtifactValidationError(
                "each frozen decision reference requires rationale_code or provenance_reference."
            )
        if self.frozen and self.freeze_state != "frozen":
            raise Phase8ArtifactValidationError("frozen references must use freeze_state='frozen'.")
        if self.freeze_state == "frozen" and not self.frozen:
            raise Phase8ArtifactValidationError("freeze_state='frozen' requires frozen=true.")

    @property
    def ordering_key(self) -> str:
        """Return the deterministic inventory ordering key."""

        return self.category


@dataclass(frozen=True, slots=True)
class Phase8DecisionFreeze:
    """Immutable inventory of all frozen Phase 8 external-validation decisions."""

    schema_name: str
    schema_version: str
    freeze_inventory_hash: str
    freeze_state: str
    frozen_before_external_evaluation: bool
    decision_references: tuple[FrozenDecisionReference, ...]

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            expected=PHASE8_DECISION_FREEZE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PHASE8_DECISION_FREEZE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.freeze_inventory_hash, field_name="freeze_inventory_hash")
        _require_allowed(
            self.freeze_state,
            PHASE8_INVENTORY_FREEZE_STATES,
            field_name="freeze_state",
        )
        ordered = tuple(sorted(self.decision_references, key=lambda item: item.ordering_key))
        object.__setattr__(self, "decision_references", ordered)
        categories = [item.category for item in self.decision_references]
        if len(set(categories)) != len(categories):
            raise Phase8ArtifactValidationError(
                "decision_references must not contain duplicate categories."
            )
        missing = sorted(PHASE8_REQUIRED_DECISION_CATEGORIES - set(categories))
        if missing:
            raise Phase8ArtifactValidationError(
                f"decision_references missing required categories: {missing!r}."
            )
        if self.freeze_state in {"frozen", "evaluation_ready"}:
            if not self.frozen_before_external_evaluation:
                raise Phase8ArtifactValidationError(
                    "frozen or evaluation-ready inventories must be frozen before evaluation."
                )
            for reference in self.decision_references:
                _require_resolved_reference(reference)
        if self.freeze_state == "draft" and self.frozen_before_external_evaluation:
            raise Phase8ArtifactValidationError(
                "draft inventories must not claim frozen_before_external_evaluation."
            )
        expected_hash = hash_phase8_decision_freeze(self)
        if self.freeze_inventory_hash != expected_hash:
            raise Phase8ArtifactHashError(
                "freeze_inventory_hash does not match deterministic content."
            )


def frozen_decision_reference_to_dict(
    reference: FrozenDecisionReference,
) -> dict[str, JsonValue]:
    """Convert one frozen decision reference to a canonical mapping."""

    return {
        "category": reference.category,
        "freeze_state": reference.freeze_state,
        "frozen": reference.frozen,
        "provenance_reference": reference.provenance_reference,
        "rationale_code": reference.rationale_code,
        "schema_name": reference.schema_name,
        "schema_version": reference.schema_version,
        "source_artifact_hash": reference.source_artifact_hash,
        "source_phase": reference.source_phase,
        "source_schema_name": reference.source_schema_name,
        "source_schema_version": reference.source_schema_version,
    }


def phase8_decision_freeze_identity_payload(
    inventory: Phase8DecisionFreeze,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one Phase 8 freeze inventory."""

    ordered = sorted(inventory.decision_references, key=lambda item: item.ordering_key)
    return {
        "decision_references": [frozen_decision_reference_to_dict(item) for item in ordered],
        "freeze_state": inventory.freeze_state,
        "frozen_before_external_evaluation": inventory.frozen_before_external_evaluation,
        "schema_name": inventory.schema_name,
        "schema_version": inventory.schema_version,
    }


def phase8_decision_freeze_to_dict(inventory: Phase8DecisionFreeze) -> dict[str, JsonValue]:
    """Convert one Phase 8 freeze inventory to a canonical mapping."""

    payload = phase8_decision_freeze_identity_payload(inventory)
    payload["freeze_inventory_hash"] = inventory.freeze_inventory_hash
    return payload


def hash_phase8_decision_freeze(inventory: Phase8DecisionFreeze) -> str:
    """Return the lowercase SHA-256 identity hash for one Phase 8 freeze inventory."""

    return sha256_json(phase8_decision_freeze_identity_payload(inventory))


def frozen_decision_reference_from_mapping(mapping: MappingLike) -> FrozenDecisionReference:
    """Reconstruct one frozen decision reference from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_FROZEN_DECISION_REFERENCE_FIELDS,
        object_name="FrozenDecisionReference",
    )
    return FrozenDecisionReference(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        category=_expect_string(mapping["category"], field_name="category"),
        source_schema_name=_expect_string(
            mapping["source_schema_name"],
            field_name="source_schema_name",
        ),
        source_schema_version=_expect_string(
            mapping["source_schema_version"],
            field_name="source_schema_version",
        ),
        source_phase=_expect_string(mapping["source_phase"], field_name="source_phase"),
        source_artifact_hash=_expect_string(
            mapping["source_artifact_hash"],
            field_name="source_artifact_hash",
        ),
        rationale_code=_expect_optional_string(
            mapping["rationale_code"],
            field_name="rationale_code",
        ),
        provenance_reference=_expect_optional_string(
            mapping["provenance_reference"],
            field_name="provenance_reference",
        ),
        frozen=_expect_bool(mapping["frozen"], field_name="frozen"),
        freeze_state=_expect_string(mapping["freeze_state"], field_name="freeze_state"),
    )


def phase8_decision_freeze_from_mapping(mapping: MappingLike) -> Phase8DecisionFreeze:
    """Reconstruct a Phase 8 freeze inventory from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_PHASE8_DECISION_FREEZE_FIELDS,
        object_name="Phase8DecisionFreeze",
    )
    references_value = mapping["decision_references"]
    if not isinstance(references_value, list):
        raise Phase8ArtifactSerializationError("decision_references must be a list.")
    return Phase8DecisionFreeze(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        freeze_inventory_hash=_expect_string(
            mapping["freeze_inventory_hash"],
            field_name="freeze_inventory_hash",
        ),
        freeze_state=_expect_string(mapping["freeze_state"], field_name="freeze_state"),
        frozen_before_external_evaluation=_expect_bool(
            mapping["frozen_before_external_evaluation"],
            field_name="frozen_before_external_evaluation",
        ),
        decision_references=tuple(
            frozen_decision_reference_from_mapping(
                _expect_mapping(item, field_name=f"decision_references[{index}]")
            )
            for index, item in enumerate(references_value)
        ),
    )


def phase8_decision_freeze_to_json(inventory: Phase8DecisionFreeze) -> bytes:
    """Serialize one Phase 8 freeze inventory to canonical JSON bytes."""

    return canonical_json_bytes(phase8_decision_freeze_to_dict(inventory)) + b"\n"


def phase8_decision_freeze_from_json(data: bytes | str) -> Phase8DecisionFreeze:
    """Deserialize one Phase 8 freeze inventory from JSON bytes or text."""

    return phase8_decision_freeze_from_mapping(_json_to_mapping(data))


_FROZEN_DECISION_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "category",
        "source_schema_name",
        "source_schema_version",
        "source_phase",
        "source_artifact_hash",
        "rationale_code",
        "provenance_reference",
        "frozen",
        "freeze_state",
    }
)
_PHASE8_DECISION_FREEZE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "freeze_inventory_hash",
        "freeze_state",
        "frozen_before_external_evaluation",
        "decision_references",
    }
)


def _json_to_mapping(data: bytes | str) -> MappingLike:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Phase8ArtifactSerializationError("invalid JSON artifact payload.") from exc
    if not isinstance(decoded, Mapping):
        raise Phase8ArtifactSerializationError("artifact JSON root must be an object.")
    return cast(MappingLike, decoded)


def _expect_mapping(value: object, *, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase8ArtifactSerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase8ArtifactSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase8ArtifactSerializationError(f"{field_name} must be a boolean.")
    return value


def _require_exact_fields(
    mapping: MappingLike,
    *,
    required_fields: frozenset[str],
    object_name: str,
) -> None:
    keys = set(mapping)
    missing = sorted(required_fields - keys)
    extra = sorted(keys - required_fields)
    if missing or extra:
        raise Phase8ArtifactSerializationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _require_schema(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise Phase8ArtifactValidationError(f"{field_name} must be {expected!r}.")


def _require_schema_version(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise Phase8ArtifactValidationError(f"{field_name} must be {expected!r}.")


def _require_version_token(value: str, *, field_name: str) -> None:
    if not _SCHEMA_VERSION_RE.fullmatch(value):
        raise Phase8ArtifactValidationError(f"{field_name} must be a version token like 'v1'.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise Phase8ArtifactValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise Phase8ArtifactValidationError(f"{field_name} must be a conservative identifier.")
    if value in PHASE8_UNRESOLVED_TOKENS:
        raise Phase8ArtifactValidationError(f"{field_name} must not be unresolved.")


def _require_optional_identifier(
    value: str | None,
    *,
    field_name: str,
    allow_unresolved: bool = False,
) -> None:
    if value is not None:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise Phase8ArtifactValidationError(f"{field_name} must be a conservative identifier.")
        if not allow_unresolved and value in PHASE8_UNRESOLVED_TOKENS:
            raise Phase8ArtifactValidationError(f"{field_name} must not be unresolved.")


def _require_allowed(value: str, allowed: frozenset[str], *, field_name: str) -> None:
    if value not in allowed:
        raise Phase8ArtifactValidationError(f"{field_name} must be one of {sorted(allowed)!r}.")


def _require_resolved_reference(reference: FrozenDecisionReference) -> None:
    if not reference.frozen or reference.freeze_state != "frozen":
        raise Phase8ArtifactValidationError(
            "evaluation-ready freeze inventories require every reference to be frozen."
        )
    values = (
        reference.category,
        reference.source_schema_name,
        reference.source_schema_version,
        reference.source_phase,
        reference.rationale_code,
        reference.provenance_reference,
    )
    for value in values:
        if value in PHASE8_UNRESOLVED_TOKENS:
            raise Phase8ArtifactValidationError(
                "frozen or evaluation-ready inventories must not contain unresolved references."
            )


__all__ = [
    "PHASE8_DECISION_FREEZE_SCHEMA_NAME",
    "PHASE8_DECISION_FREEZE_SCHEMA_VERSION",
    "PHASE8_FROZEN_DECISION_REFERENCE_SCHEMA_NAME",
    "PHASE8_FROZEN_DECISION_REFERENCE_SCHEMA_VERSION",
    "PHASE8_INVENTORY_FREEZE_STATES",
    "PHASE8_REFERENCE_FREEZE_STATES",
    "PHASE8_REQUIRED_DECISION_CATEGORIES",
    "FrozenDecisionReference",
    "Phase8ArtifactError",
    "Phase8ArtifactHashError",
    "Phase8ArtifactSerializationError",
    "Phase8ArtifactValidationError",
    "Phase8DecisionFreeze",
    "frozen_decision_reference_from_mapping",
    "frozen_decision_reference_to_dict",
    "hash_phase8_decision_freeze",
    "phase8_decision_freeze_from_json",
    "phase8_decision_freeze_from_mapping",
    "phase8_decision_freeze_identity_payload",
    "phase8_decision_freeze_to_dict",
    "phase8_decision_freeze_to_json",
]
