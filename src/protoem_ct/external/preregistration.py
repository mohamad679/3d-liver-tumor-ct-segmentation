"""Phase 8 external-evaluation preregistration contract.

This module defines the schema and strict validation for the preregistration
artifact only. It performs no filesystem I/O and does not create a completed
Phase 8 preregistration from real external data.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json

PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME: Final[str] = "phase8_external_preregistration"
PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_ARTIFACT_REFERENCE_SCHEMA_NAME: Final[str] = "phase8_artifact_reference"
PHASE8_ARTIFACT_REFERENCE_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_NAME: Final[str] = (
    "phase8_robustness_uncertainty_inclusion"
)
PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_VERSION: Final[str] = "v1"

PHASE8_PREREGISTRATION_LIFECYCLE_STATES: Final[frozenset[str]] = frozenset(
    {"draft", "evaluation_ready"}
)
PHASE8_REFERENCE_STATES: Final[frozenset[str]] = frozenset({"resolved", "unresolved"})
PHASE8_ROBUSTNESS_UNCERTAINTY_STATES: Final[frozenset[str]] = frozenset(
    {"included", "not_included", "unresolved"}
)
PHASE8_EXTERNAL_LABEL_ACCESS_STATES: Final[frozenset[str]] = frozenset(
    {"unavailable_before_prediction_lock", "available_after_prediction_lock"}
)
PHASE8_PREDICTION_LOCK_REQUIREMENTS: Final[frozenset[str]] = frozenset(
    {"required_before_label_access"}
)
PHASE8_PERMITTED_DEVIATION_POLICIES: Final[frozenset[str]] = frozenset(
    {"no_deviations_without_versioned_addendum", "minor_documented_deviations_only"}
)

MappingLike: TypeAlias = Mapping[str, object]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_ZERO_SHA256: Final[str] = "0" * 64


class Phase8PreregistrationError(ValueError):
    """Base error for Phase 8 preregistration contract failures."""


class Phase8PreregistrationValidationError(Phase8PreregistrationError):
    """Raised when preregistration content violates the schema contract."""


class Phase8PreregistrationSerializationError(Phase8PreregistrationError):
    """Raised when preregistration JSON cannot be parsed strictly."""


class Phase8PreregistrationHashError(Phase8PreregistrationError):
    """Raised when the preregistration self-hash does not match its identity."""


@dataclass(frozen=True, slots=True)
class Phase8ArtifactReference:
    """Reference to a preregistered Phase 8 dependency artifact."""

    schema_name: str
    schema_version: str
    referenced_schema_name: str
    referenced_schema_version: str
    artifact_hash: str | None
    reference_state: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PHASE8_ARTIFACT_REFERENCE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PHASE8_ARTIFACT_REFERENCE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_identifier(self.referenced_schema_name, field_name="referenced_schema_name")
        _require_identifier(
            self.referenced_schema_version,
            field_name="referenced_schema_version",
        )
        _require_allowed(
            self.reference_state,
            PHASE8_REFERENCE_STATES,
            field_name="reference_state",
        )
        _require_optional_sha256(self.artifact_hash, field_name="artifact_hash")
        if self.reference_state == "resolved" and self.artifact_hash is None:
            raise Phase8PreregistrationValidationError(
                "resolved artifact references require artifact_hash."
            )
        if self.reference_state == "unresolved" and self.artifact_hash is not None:
            raise Phase8PreregistrationValidationError(
                "unresolved artifact references must not include artifact_hash."
            )


@dataclass(frozen=True, slots=True)
class Phase8RobustnessUncertaintyInclusion:
    """Optional Phase 8 robustness and uncertainty inclusion decision."""

    schema_name: str
    schema_version: str
    inclusion_state: str
    policy_reference: Phase8ArtifactReference | None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_allowed(
            self.inclusion_state,
            PHASE8_ROBUSTNESS_UNCERTAINTY_STATES,
            field_name="inclusion_state",
        )
        if self.inclusion_state == "included" and self.policy_reference is None:
            raise Phase8PreregistrationValidationError(
                "included robustness/uncertainty reporting requires a policy_reference."
            )
        if self.inclusion_state == "not_included" and self.policy_reference is not None:
            raise Phase8PreregistrationValidationError(
                "not_included robustness/uncertainty reporting must not include a policy_reference."
            )


@dataclass(frozen=True, slots=True)
class Phase8ExternalPreregistration:
    """Immutable Phase 8 external-evaluation preregistration identity."""

    schema_name: str
    schema_version: str
    preregistration_hash: str
    lifecycle_state: str
    external_cohort_identifier: str
    frozen_decision_inventory_hash: str | None
    anonymous_manifest_reference: Phase8ArtifactReference
    eligibility_exclusion_policy_reference: Phase8ArtifactReference
    label_mapping_policy_reference: Phase8ArtifactReference
    domain_shift_record_reference: Phase8ArtifactReference
    preregistered_segmentation_metrics: tuple[str, ...]
    bootstrap_policy_config_reference: Phase8ArtifactReference
    internal_external_comparison_policy_reference: Phase8ArtifactReference
    qualitative_output_policy_reference: Phase8ArtifactReference
    robustness_uncertainty_inclusion: Phase8RobustnessUncertaintyInclusion
    permitted_deviation_policy: str
    no_tuning_declaration: bool
    external_label_access_state: str
    prediction_lock_requirement: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.preregistration_hash, field_name="preregistration_hash")
        _require_allowed(
            self.lifecycle_state,
            PHASE8_PREREGISTRATION_LIFECYCLE_STATES,
            field_name="lifecycle_state",
        )
        _require_identifier(
            self.external_cohort_identifier,
            field_name="external_cohort_identifier",
        )
        _require_optional_sha256(
            self.frozen_decision_inventory_hash,
            field_name="frozen_decision_inventory_hash",
        )
        _require_metrics(self.preregistered_segmentation_metrics)
        object.__setattr__(
            self,
            "preregistered_segmentation_metrics",
            tuple(self.preregistered_segmentation_metrics),
        )
        _require_allowed(
            self.permitted_deviation_policy,
            PHASE8_PERMITTED_DEVIATION_POLICIES,
            field_name="permitted_deviation_policy",
        )
        if not self.no_tuning_declaration:
            raise Phase8PreregistrationValidationError("no_tuning_declaration must be true.")
        _require_allowed(
            self.external_label_access_state,
            PHASE8_EXTERNAL_LABEL_ACCESS_STATES,
            field_name="external_label_access_state",
        )
        _require_allowed(
            self.prediction_lock_requirement,
            PHASE8_PREDICTION_LOCK_REQUIREMENTS,
            field_name="prediction_lock_requirement",
        )
        if self.external_label_access_state != "unavailable_before_prediction_lock":
            raise Phase8PreregistrationValidationError(
                "external labels must be unavailable before prediction lock."
            )
        if self.lifecycle_state == "evaluation_ready":
            _require_evaluation_ready(self)
        expected_hash = hash_phase8_external_preregistration(self)
        if self.preregistration_hash != expected_hash:
            raise Phase8PreregistrationHashError(
                "preregistration_hash does not match deterministic preregistration content."
            )


def phase8_artifact_reference_identity_payload(
    reference: Phase8ArtifactReference,
) -> dict[str, JsonValue]:
    """Return the canonical payload for one artifact reference."""

    return {
        "artifact_hash": reference.artifact_hash,
        "reference_state": reference.reference_state,
        "referenced_schema_name": reference.referenced_schema_name,
        "referenced_schema_version": reference.referenced_schema_version,
        "schema_name": reference.schema_name,
        "schema_version": reference.schema_version,
    }


def phase8_robustness_uncertainty_inclusion_identity_payload(
    inclusion: Phase8RobustnessUncertaintyInclusion,
) -> dict[str, JsonValue]:
    """Return the canonical payload for the robustness/uncertainty inclusion decision."""

    return {
        "inclusion_state": inclusion.inclusion_state,
        "policy_reference": None
        if inclusion.policy_reference is None
        else phase8_artifact_reference_identity_payload(inclusion.policy_reference),
        "schema_name": inclusion.schema_name,
        "schema_version": inclusion.schema_version,
    }


def phase8_external_preregistration_identity_payload(
    preregistration: Phase8ExternalPreregistration,
) -> dict[str, JsonValue]:
    """Return the compatibility-critical canonical identity payload."""

    return {
        "anonymous_manifest_reference": phase8_artifact_reference_identity_payload(
            preregistration.anonymous_manifest_reference
        ),
        "bootstrap_policy_config_reference": phase8_artifact_reference_identity_payload(
            preregistration.bootstrap_policy_config_reference
        ),
        "domain_shift_record_reference": phase8_artifact_reference_identity_payload(
            preregistration.domain_shift_record_reference
        ),
        "eligibility_exclusion_policy_reference": phase8_artifact_reference_identity_payload(
            preregistration.eligibility_exclusion_policy_reference
        ),
        "external_cohort_identifier": preregistration.external_cohort_identifier,
        "external_label_access_state": preregistration.external_label_access_state,
        "frozen_decision_inventory_hash": preregistration.frozen_decision_inventory_hash,
        "internal_external_comparison_policy_reference": (
            phase8_artifact_reference_identity_payload(
                preregistration.internal_external_comparison_policy_reference
            )
        ),
        "label_mapping_policy_reference": phase8_artifact_reference_identity_payload(
            preregistration.label_mapping_policy_reference
        ),
        "lifecycle_state": preregistration.lifecycle_state,
        "no_tuning_declaration": preregistration.no_tuning_declaration,
        "permitted_deviation_policy": preregistration.permitted_deviation_policy,
        "prediction_lock_requirement": preregistration.prediction_lock_requirement,
        "preregistered_segmentation_metrics": list(
            preregistration.preregistered_segmentation_metrics
        ),
        "qualitative_output_policy_reference": phase8_artifact_reference_identity_payload(
            preregistration.qualitative_output_policy_reference
        ),
        "robustness_uncertainty_inclusion": (
            phase8_robustness_uncertainty_inclusion_identity_payload(
                preregistration.robustness_uncertainty_inclusion
            )
        ),
        "schema_name": preregistration.schema_name,
        "schema_version": preregistration.schema_version,
    }


def phase8_external_preregistration_to_dict(
    preregistration: Phase8ExternalPreregistration,
) -> dict[str, JsonValue]:
    """Convert one preregistration artifact to a canonical mapping."""

    payload = phase8_external_preregistration_identity_payload(preregistration)
    payload["preregistration_hash"] = preregistration.preregistration_hash
    return payload


def hash_phase8_external_preregistration(
    preregistration: Phase8ExternalPreregistration,
) -> str:
    """Return the canonical SHA-256 identity hash for one preregistration."""

    return sha256_json(phase8_external_preregistration_identity_payload(preregistration))


def phase8_external_preregistration_to_json(
    preregistration: Phase8ExternalPreregistration,
) -> bytes:
    """Serialize one preregistration to canonical JSON bytes."""

    return canonical_json_bytes(phase8_external_preregistration_to_dict(preregistration)) + b"\n"


def phase8_artifact_reference_from_mapping(mapping: MappingLike) -> Phase8ArtifactReference:
    """Reconstruct one strict artifact reference from a mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_ARTIFACT_REFERENCE_FIELDS,
        object_name="Phase8ArtifactReference",
    )
    return Phase8ArtifactReference(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        referenced_schema_name=_expect_string(
            mapping["referenced_schema_name"],
            field_name="referenced_schema_name",
        ),
        referenced_schema_version=_expect_string(
            mapping["referenced_schema_version"],
            field_name="referenced_schema_version",
        ),
        artifact_hash=_expect_optional_string(mapping["artifact_hash"], field_name="artifact_hash"),
        reference_state=_expect_string(mapping["reference_state"], field_name="reference_state"),
    )


def phase8_robustness_uncertainty_inclusion_from_mapping(
    mapping: MappingLike,
) -> Phase8RobustnessUncertaintyInclusion:
    """Reconstruct one strict robustness/uncertainty inclusion state."""

    _require_exact_fields(
        mapping,
        required_fields=_ROBUSTNESS_UNCERTAINTY_INCLUSION_FIELDS,
        object_name="Phase8RobustnessUncertaintyInclusion",
    )
    policy_value = mapping["policy_reference"]
    return Phase8RobustnessUncertaintyInclusion(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        inclusion_state=_expect_string(mapping["inclusion_state"], field_name="inclusion_state"),
        policy_reference=None
        if policy_value is None
        else phase8_artifact_reference_from_mapping(
            _expect_mapping(policy_value, field_name="policy_reference")
        ),
    )


def phase8_external_preregistration_from_mapping(
    mapping: MappingLike,
) -> Phase8ExternalPreregistration:
    """Reconstruct one strict preregistration artifact from a mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_EXTERNAL_PREREGISTRATION_FIELDS,
        object_name="Phase8ExternalPreregistration",
    )
    metrics = _expect_string_sequence(
        mapping["preregistered_segmentation_metrics"],
        field_name="preregistered_segmentation_metrics",
    )
    return Phase8ExternalPreregistration(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        preregistration_hash=_expect_string(
            mapping["preregistration_hash"],
            field_name="preregistration_hash",
        ),
        lifecycle_state=_expect_string(mapping["lifecycle_state"], field_name="lifecycle_state"),
        external_cohort_identifier=_expect_string(
            mapping["external_cohort_identifier"],
            field_name="external_cohort_identifier",
        ),
        frozen_decision_inventory_hash=_expect_optional_string(
            mapping["frozen_decision_inventory_hash"],
            field_name="frozen_decision_inventory_hash",
        ),
        anonymous_manifest_reference=phase8_artifact_reference_from_mapping(
            _expect_mapping(
                mapping["anonymous_manifest_reference"],
                field_name="anonymous_manifest_reference",
            )
        ),
        eligibility_exclusion_policy_reference=phase8_artifact_reference_from_mapping(
            _expect_mapping(
                mapping["eligibility_exclusion_policy_reference"],
                field_name="eligibility_exclusion_policy_reference",
            )
        ),
        label_mapping_policy_reference=phase8_artifact_reference_from_mapping(
            _expect_mapping(
                mapping["label_mapping_policy_reference"],
                field_name="label_mapping_policy_reference",
            )
        ),
        domain_shift_record_reference=phase8_artifact_reference_from_mapping(
            _expect_mapping(
                mapping["domain_shift_record_reference"],
                field_name="domain_shift_record_reference",
            )
        ),
        preregistered_segmentation_metrics=metrics,
        bootstrap_policy_config_reference=phase8_artifact_reference_from_mapping(
            _expect_mapping(
                mapping["bootstrap_policy_config_reference"],
                field_name="bootstrap_policy_config_reference",
            )
        ),
        internal_external_comparison_policy_reference=phase8_artifact_reference_from_mapping(
            _expect_mapping(
                mapping["internal_external_comparison_policy_reference"],
                field_name="internal_external_comparison_policy_reference",
            )
        ),
        qualitative_output_policy_reference=phase8_artifact_reference_from_mapping(
            _expect_mapping(
                mapping["qualitative_output_policy_reference"],
                field_name="qualitative_output_policy_reference",
            )
        ),
        robustness_uncertainty_inclusion=(
            phase8_robustness_uncertainty_inclusion_from_mapping(
                _expect_mapping(
                    mapping["robustness_uncertainty_inclusion"],
                    field_name="robustness_uncertainty_inclusion",
                )
            )
        ),
        permitted_deviation_policy=_expect_string(
            mapping["permitted_deviation_policy"],
            field_name="permitted_deviation_policy",
        ),
        no_tuning_declaration=_expect_bool(
            mapping["no_tuning_declaration"],
            field_name="no_tuning_declaration",
        ),
        external_label_access_state=_expect_string(
            mapping["external_label_access_state"],
            field_name="external_label_access_state",
        ),
        prediction_lock_requirement=_expect_string(
            mapping["prediction_lock_requirement"],
            field_name="prediction_lock_requirement",
        ),
    )


def phase8_external_preregistration_from_json(
    data: bytes | str,
) -> Phase8ExternalPreregistration:
    """Deserialize one preregistration artifact from JSON bytes or text."""

    return phase8_external_preregistration_from_mapping(_json_to_mapping(data))


_ARTIFACT_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "referenced_schema_name",
        "referenced_schema_version",
        "artifact_hash",
        "reference_state",
    }
)
_ROBUSTNESS_UNCERTAINTY_INCLUSION_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_name", "schema_version", "inclusion_state", "policy_reference"}
)
_EXTERNAL_PREREGISTRATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "preregistration_hash",
        "lifecycle_state",
        "external_cohort_identifier",
        "frozen_decision_inventory_hash",
        "anonymous_manifest_reference",
        "eligibility_exclusion_policy_reference",
        "label_mapping_policy_reference",
        "domain_shift_record_reference",
        "preregistered_segmentation_metrics",
        "bootstrap_policy_config_reference",
        "internal_external_comparison_policy_reference",
        "qualitative_output_policy_reference",
        "robustness_uncertainty_inclusion",
        "permitted_deviation_policy",
        "no_tuning_declaration",
        "external_label_access_state",
        "prediction_lock_requirement",
    }
)


def _json_to_mapping(data: bytes | str) -> MappingLike:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Phase8PreregistrationSerializationError(
            "invalid preregistration JSON payload."
        ) from exc
    if not isinstance(decoded, Mapping):
        raise Phase8PreregistrationSerializationError(
            "preregistration JSON root must be an object."
        )
    return cast(MappingLike, decoded)


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
        raise Phase8PreregistrationSerializationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _expect_mapping(value: object, *, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase8PreregistrationSerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase8PreregistrationSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase8PreregistrationSerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_string_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise Phase8PreregistrationSerializationError(f"{field_name} must be a list of strings.")
    if not isinstance(value, list):
        raise Phase8PreregistrationSerializationError(f"{field_name} must be a list of strings.")
    return tuple(
        _expect_string(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _require_schema_name(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise Phase8PreregistrationValidationError(
            f"{field_name} must equal {expected!r}, got {value!r}."
        )


def _require_schema_version(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise Phase8PreregistrationValidationError(
            f"{field_name} must equal {expected!r}, got {value!r}."
        )


def _require_allowed(value: str, allowed: frozenset[str], *, field_name: str) -> None:
    if value not in allowed:
        raise Phase8PreregistrationValidationError(
            f"{field_name} must be one of {sorted(allowed)!r}, got {value!r}."
        )


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise Phase8PreregistrationValidationError(
            f"{field_name} must be a lowercase safe identifier."
        )


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise Phase8PreregistrationValidationError(
            f"{field_name} must be a lowercase SHA-256 hex digest."
        )


def _require_optional_sha256(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_sha256(value, field_name=field_name)


def _require_metrics(metrics: Sequence[str]) -> None:
    if not metrics:
        raise Phase8PreregistrationValidationError(
            "preregistered_segmentation_metrics must be non-empty."
        )
    normalized = tuple(metrics)
    if len(set(normalized)) != len(normalized):
        raise Phase8PreregistrationValidationError(
            "preregistered_segmentation_metrics must not contain duplicates."
        )
    for index, metric in enumerate(normalized):
        _require_identifier(metric, field_name=f"preregistered_segmentation_metrics[{index}]")


def _require_evaluation_ready(preregistration: Phase8ExternalPreregistration) -> None:
    if preregistration.frozen_decision_inventory_hash in (None, _ZERO_SHA256):
        raise Phase8PreregistrationValidationError(
            "evaluation_ready preregistrations require a resolved frozen_decision_inventory_hash."
        )
    for field_name, reference in _iter_references(preregistration):
        _require_resolved_reference(reference, field_name=field_name)
    inclusion = preregistration.robustness_uncertainty_inclusion
    if inclusion.inclusion_state == "unresolved":
        raise Phase8PreregistrationValidationError(
            "evaluation_ready preregistrations must resolve robustness/uncertainty inclusion."
        )


def _iter_references(
    preregistration: Phase8ExternalPreregistration,
) -> tuple[tuple[str, Phase8ArtifactReference], ...]:
    references: list[tuple[str, Phase8ArtifactReference]] = [
        ("anonymous_manifest_reference", preregistration.anonymous_manifest_reference),
        (
            "eligibility_exclusion_policy_reference",
            preregistration.eligibility_exclusion_policy_reference,
        ),
        ("label_mapping_policy_reference", preregistration.label_mapping_policy_reference),
        ("domain_shift_record_reference", preregistration.domain_shift_record_reference),
        (
            "bootstrap_policy_config_reference",
            preregistration.bootstrap_policy_config_reference,
        ),
        (
            "internal_external_comparison_policy_reference",
            preregistration.internal_external_comparison_policy_reference,
        ),
        (
            "qualitative_output_policy_reference",
            preregistration.qualitative_output_policy_reference,
        ),
    ]
    policy_reference = preregistration.robustness_uncertainty_inclusion.policy_reference
    if policy_reference is not None:
        references.append(("robustness_uncertainty_policy_reference", policy_reference))
    return tuple(references)


def _require_resolved_reference(reference: Phase8ArtifactReference, *, field_name: str) -> None:
    if reference.reference_state != "resolved" or reference.artifact_hash in (None, _ZERO_SHA256):
        raise Phase8PreregistrationValidationError(
            f"evaluation_ready preregistrations require resolved non-placeholder {field_name}."
        )


__all__ = [
    "PHASE8_ARTIFACT_REFERENCE_SCHEMA_NAME",
    "PHASE8_ARTIFACT_REFERENCE_SCHEMA_VERSION",
    "PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME",
    "PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION",
    "PHASE8_EXTERNAL_LABEL_ACCESS_STATES",
    "PHASE8_PREDICTION_LOCK_REQUIREMENTS",
    "PHASE8_PREREGISTRATION_LIFECYCLE_STATES",
    "PHASE8_REFERENCE_STATES",
    "PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_NAME",
    "PHASE8_ROBUSTNESS_UNCERTAINTY_INCLUSION_SCHEMA_VERSION",
    "PHASE8_ROBUSTNESS_UNCERTAINTY_STATES",
    "Phase8ArtifactReference",
    "Phase8ExternalPreregistration",
    "Phase8PreregistrationHashError",
    "Phase8PreregistrationSerializationError",
    "Phase8PreregistrationValidationError",
    "Phase8RobustnessUncertaintyInclusion",
    "hash_phase8_external_preregistration",
    "phase8_artifact_reference_from_mapping",
    "phase8_artifact_reference_identity_payload",
    "phase8_external_preregistration_from_json",
    "phase8_external_preregistration_from_mapping",
    "phase8_external_preregistration_identity_payload",
    "phase8_external_preregistration_to_dict",
    "phase8_external_preregistration_to_json",
    "phase8_robustness_uncertainty_inclusion_from_mapping",
    "phase8_robustness_uncertainty_inclusion_identity_payload",
]
