"""Phase 8 preregistered external label-mapping policy contract.

This module is policy-only. It does not discover, open, list, or infer any real external label
artifact. The optional mapping helper operates only on caller-provided synthetic arrays so tests can
verify the frozen binary-tumor semantics without touching 3D-IRCADb-01 labels.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json

PHASE8_LABEL_MAPPING_POLICY_SCHEMA_NAME: Final[str] = "phase8_label_mapping_policy"
PHASE8_LABEL_MAPPING_POLICY_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_SOURCE_LABEL_ROLE_SCHEMA_NAME: Final[str] = "phase8_source_label_role"
PHASE8_SOURCE_LABEL_ROLE_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_SOURCE_TO_TARGET_MAPPING_RULE_SCHEMA_NAME: Final[str] = (
    "phase8_source_to_target_mapping_rule"
)
PHASE8_SOURCE_TO_TARGET_MAPPING_RULE_SCHEMA_VERSION: Final[str] = "v1"

PHASE8_LABEL_MAPPING_EXTERNAL_COHORT_IDENTITY: Final[str] = "3d_ircadb_01"
PHASE8_LABEL_MAPPING_TARGET_TASK_IDENTITY: Final[str] = "binary_liver_tumor_segmentation"
PHASE8_LABEL_MAPPING_TARGET_CLASSES: Final[tuple[str, str]] = (
    "background",
    "tumor_foreground",
)
PHASE8_LABEL_MAPPING_PERMITTED_SOURCE_ROLES: Final[frozenset[str]] = frozenset(
    {"liver_context_mask", "tumor_lesion_mask"}
)
PHASE8_LABEL_MAPPING_VERIFICATION_STATES: Final[frozenset[str]] = frozenset(
    {"expected_documented_not_empirically_verified", "empirically_verified_after_label_ledger"}
)
PHASE8_LABEL_MAPPING_SOURCE_TARGET_USAGES: Final[frozenset[str]] = frozenset(
    {"binary_foreground_source", "excluded_context_only"}
)
PHASE8_LABEL_MAPPING_RULE_ACTIONS: Final[frozenset[str]] = frozenset(
    {"exclude_from_target", "nonzero_voxels_to_tumor_foreground"}
)
PHASE8_LABEL_MAPPING_RULE_OPERATORS: Final[frozenset[str]] = frozenset(
    {"ignore_for_target", "union"}
)
PHASE8_LABEL_MAPPING_REQUIRED_RULE_IDS: Final[tuple[str, str]] = (
    "exclude_liver_context_from_target",
    "map_tumor_lesion_nonzero_to_foreground",
)
PHASE8_LABEL_MAPPING_SOURCE_VALUE_POLICIES: Final[frozenset[str]] = frozenset(
    {"any_nonzero_voxel", "not_used_for_target"}
)

MappingLike: TypeAlias = Mapping[str, object]
BoolArray: TypeAlias = NDArray[np.bool_]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


class Phase8LabelMappingError(ValueError):
    """Base error for Phase 8 label-mapping policy failures."""


class Phase8LabelMappingValidationError(Phase8LabelMappingError):
    """Raised when label-mapping policy content violates the frozen contract."""


class Phase8LabelMappingSerializationError(Phase8LabelMappingError):
    """Raised when label-mapping policy JSON cannot be reconstructed strictly."""


class Phase8LabelMappingHashError(Phase8LabelMappingError):
    """Raised when the label-mapping policy self-hash does not match its content."""


@dataclass(frozen=True, slots=True)
class Phase8SourceLabelRole:
    """Preregistered abstract source-label role, not an observed archive member name."""

    schema_name: str
    schema_version: str
    role_id: str
    target_usage: str
    source_value_policy: str
    verification_state: str
    source_basis_reference: str
    provenance_reference: str

    def __post_init__(self) -> None:
        _require_exact(self.schema_name, PHASE8_SOURCE_LABEL_ROLE_SCHEMA_NAME, "schema_name")
        _require_exact(
            self.schema_version,
            PHASE8_SOURCE_LABEL_ROLE_SCHEMA_VERSION,
            "schema_version",
        )
        _require_allowed(
            self.role_id,
            PHASE8_LABEL_MAPPING_PERMITTED_SOURCE_ROLES,
            "role_id",
        )
        _require_allowed(
            self.target_usage,
            PHASE8_LABEL_MAPPING_SOURCE_TARGET_USAGES,
            "target_usage",
        )
        _require_allowed(
            self.source_value_policy,
            PHASE8_LABEL_MAPPING_SOURCE_VALUE_POLICIES,
            "source_value_policy",
        )
        _require_allowed(
            self.verification_state,
            PHASE8_LABEL_MAPPING_VERIFICATION_STATES,
            "verification_state",
        )
        _require_safe_token(self.source_basis_reference, "source_basis_reference")
        _require_safe_token(self.provenance_reference, "provenance_reference")
        if self.role_id == "tumor_lesion_mask":
            _require_exact(self.target_usage, "binary_foreground_source", "target_usage")
            _require_exact(self.source_value_policy, "any_nonzero_voxel", "source_value_policy")
        if self.role_id == "liver_context_mask":
            _require_exact(self.target_usage, "excluded_context_only", "target_usage")
            _require_exact(self.source_value_policy, "not_used_for_target", "source_value_policy")

    @property
    def ordering_key(self) -> str:
        """Return deterministic ordering key for canonical policy serialization."""

        return self.role_id


@dataclass(frozen=True, slots=True)
class Phase8SourceToTargetMappingRule:
    """Preregistered abstract source-to-target mapping rule."""

    schema_name: str
    schema_version: str
    rule_id: str
    source_role_id: str
    target_class: str | None
    rule_action: str
    aggregation_operator: str
    verification_state: str
    provenance_reference: str

    def __post_init__(self) -> None:
        _require_exact(
            self.schema_name,
            PHASE8_SOURCE_TO_TARGET_MAPPING_RULE_SCHEMA_NAME,
            "schema_name",
        )
        _require_exact(
            self.schema_version,
            PHASE8_SOURCE_TO_TARGET_MAPPING_RULE_SCHEMA_VERSION,
            "schema_version",
        )
        _require_safe_token(self.rule_id, "rule_id")
        _require_allowed(
            self.source_role_id,
            PHASE8_LABEL_MAPPING_PERMITTED_SOURCE_ROLES,
            "source_role_id",
        )
        if self.target_class is not None:
            _require_allowed(
                self.target_class,
                frozenset(PHASE8_LABEL_MAPPING_TARGET_CLASSES),
                "target_class",
            )
        _require_allowed(self.rule_action, PHASE8_LABEL_MAPPING_RULE_ACTIONS, "rule_action")
        _require_allowed(
            self.aggregation_operator,
            PHASE8_LABEL_MAPPING_RULE_OPERATORS,
            "aggregation_operator",
        )
        _require_allowed(
            self.verification_state,
            PHASE8_LABEL_MAPPING_VERIFICATION_STATES,
            "verification_state",
        )
        _require_safe_token(self.provenance_reference, "provenance_reference")
        if self.source_role_id == "tumor_lesion_mask":
            _require_exact(self.target_class, "tumor_foreground", "target_class")
            _require_exact(
                self.rule_action,
                "nonzero_voxels_to_tumor_foreground",
                "rule_action",
            )
            _require_exact(self.aggregation_operator, "union", "aggregation_operator")
        if self.source_role_id == "liver_context_mask":
            if self.target_class is not None:
                raise Phase8LabelMappingValidationError(
                    "liver_context_mask must not map to a target class."
                )
            _require_exact(self.rule_action, "exclude_from_target", "rule_action")
            _require_exact(self.aggregation_operator, "ignore_for_target", "aggregation_operator")

    @property
    def ordering_key(self) -> str:
        """Return deterministic ordering key for canonical policy serialization."""

        return self.rule_id


@dataclass(frozen=True, slots=True)
class Phase8LabelMappingPolicy:
    """Immutable preregistered mapping from external label roles to project binary tumor labels."""

    schema_name: str
    schema_version: str
    policy_hash: str
    external_cohort_identity: str
    target_task_identity: str
    target_classes: tuple[str, ...]
    permitted_source_roles: tuple[Phase8SourceLabelRole, ...]
    mapping_rules: tuple[Phase8SourceToTargetMappingRule, ...]
    liver_mask_handling: str
    lesion_tumor_aggregation: str
    overlap_resolution: str
    background_policy: str
    missing_source_policy: str
    incompatible_source_policy: str
    multiple_component_policy: str
    empty_target_policy: str
    geometry_alignment_policy: str
    interpolation_policy: str
    connectivity_policy: str
    verification_state: str
    source_basis_references: tuple[str, ...]
    no_tuning_declaration: bool

    def __post_init__(self) -> None:
        _require_exact(self.schema_name, PHASE8_LABEL_MAPPING_POLICY_SCHEMA_NAME, "schema_name")
        _require_exact(
            self.schema_version,
            PHASE8_LABEL_MAPPING_POLICY_SCHEMA_VERSION,
            "schema_version",
        )
        _require_sha256(self.policy_hash, "policy_hash")
        _require_exact(
            self.external_cohort_identity,
            PHASE8_LABEL_MAPPING_EXTERNAL_COHORT_IDENTITY,
            "external_cohort_identity",
        )
        _require_exact(
            self.target_task_identity,
            PHASE8_LABEL_MAPPING_TARGET_TASK_IDENTITY,
            "target_task_identity",
        )
        if tuple(self.target_classes) != PHASE8_LABEL_MAPPING_TARGET_CLASSES:
            raise Phase8LabelMappingValidationError(
                "target_classes must be ('background', 'tumor_foreground') in canonical order."
            )
        ordered_roles = tuple(
            sorted(self.permitted_source_roles, key=lambda item: item.ordering_key)
        )
        ordered_rules = tuple(sorted(self.mapping_rules, key=lambda item: item.ordering_key))
        object.__setattr__(self, "permitted_source_roles", ordered_roles)
        object.__setattr__(self, "mapping_rules", ordered_rules)
        _require_exact_role_set(ordered_roles)
        _require_exact_rule_set(ordered_rules)
        _require_exact(
            self.liver_mask_handling,
            "excluded_context_only_never_target_foreground",
            "liver_mask_handling",
        )
        _require_exact(
            self.lesion_tumor_aggregation,
            "union_all_permitted_tumor_lesion_sources",
            "lesion_tumor_aggregation",
        )
        _require_exact(
            self.overlap_resolution,
            "tumor_foreground_precedence_over_context_background_else_background",
            "overlap_resolution",
        )
        _require_exact(
            self.background_policy,
            "all_non_foreground_voxels_background",
            "background_policy",
        )
        _require_exact(
            self.missing_source_policy,
            "fail_closed_missing_required_tumor_source",
            "missing_source_policy",
        )
        _require_exact(
            self.incompatible_source_policy,
            "fail_closed_incompatible_geometry_or_values",
            "incompatible_source_policy",
        )
        _require_exact(
            self.multiple_component_policy,
            "preserve_all_components_no_filtering",
            "multiple_component_policy",
        )
        _require_exact(
            self.empty_target_policy,
            "allow_empty_target_with_explicit_empty_status",
            "empty_target_policy",
        )
        _require_exact(
            self.geometry_alignment_policy,
            "require_same_grid_shape_affine_spacing_orientation",
            "geometry_alignment_policy",
        )
        _require_exact(
            self.interpolation_policy,
            "nearest_neighbor_for_labels_only_no_resampling_in_policy",
            "interpolation_policy",
        )
        _require_exact(
            self.connectivity_policy,
            "3d_26_connectivity_for_component_accounting_only",
            "connectivity_policy",
        )
        _require_allowed(
            self.verification_state,
            PHASE8_LABEL_MAPPING_VERIFICATION_STATES,
            "verification_state",
        )
        _require_sorted_unique_tokens(self.source_basis_references, "source_basis_references")
        if self.verification_state != "expected_documented_not_empirically_verified":
            raise Phase8LabelMappingValidationError(
                "policy must remain expected/documented until label-ledger QA empirically "
                "verifies it."
            )
        if not self.no_tuning_declaration:
            raise Phase8LabelMappingValidationError("no_tuning_declaration must be true.")
        expected_hash = hash_phase8_label_mapping_policy(self)
        if self.policy_hash != expected_hash:
            raise Phase8LabelMappingHashError(
                "policy_hash does not match deterministic label-mapping content."
            )


def phase8_source_label_role_to_dict(role: Phase8SourceLabelRole) -> dict[str, JsonValue]:
    """Convert one source-label role to a canonical mapping."""

    return {
        "provenance_reference": role.provenance_reference,
        "role_id": role.role_id,
        "schema_name": role.schema_name,
        "schema_version": role.schema_version,
        "source_basis_reference": role.source_basis_reference,
        "source_value_policy": role.source_value_policy,
        "target_usage": role.target_usage,
        "verification_state": role.verification_state,
    }


def phase8_source_to_target_mapping_rule_to_dict(
    rule: Phase8SourceToTargetMappingRule,
) -> dict[str, JsonValue]:
    """Convert one mapping rule to a canonical mapping."""

    return {
        "aggregation_operator": rule.aggregation_operator,
        "provenance_reference": rule.provenance_reference,
        "rule_action": rule.rule_action,
        "rule_id": rule.rule_id,
        "schema_name": rule.schema_name,
        "schema_version": rule.schema_version,
        "source_role_id": rule.source_role_id,
        "target_class": rule.target_class,
        "verification_state": rule.verification_state,
    }


def phase8_label_mapping_policy_identity_payload(
    policy: Phase8LabelMappingPolicy,
) -> dict[str, JsonValue]:
    """Return canonical label-mapping identity payload, excluding ``policy_hash``."""

    roles = tuple(sorted(policy.permitted_source_roles, key=lambda item: item.ordering_key))
    rules = tuple(sorted(policy.mapping_rules, key=lambda item: item.ordering_key))
    return {
        "background_policy": policy.background_policy,
        "connectivity_policy": policy.connectivity_policy,
        "empty_target_policy": policy.empty_target_policy,
        "external_cohort_identity": policy.external_cohort_identity,
        "geometry_alignment_policy": policy.geometry_alignment_policy,
        "incompatible_source_policy": policy.incompatible_source_policy,
        "interpolation_policy": policy.interpolation_policy,
        "lesion_tumor_aggregation": policy.lesion_tumor_aggregation,
        "liver_mask_handling": policy.liver_mask_handling,
        "mapping_rules": [phase8_source_to_target_mapping_rule_to_dict(item) for item in rules],
        "missing_source_policy": policy.missing_source_policy,
        "multiple_component_policy": policy.multiple_component_policy,
        "no_tuning_declaration": policy.no_tuning_declaration,
        "overlap_resolution": policy.overlap_resolution,
        "permitted_source_roles": [phase8_source_label_role_to_dict(item) for item in roles],
        "schema_name": policy.schema_name,
        "schema_version": policy.schema_version,
        "source_basis_references": list(policy.source_basis_references),
        "target_classes": list(policy.target_classes),
        "target_task_identity": policy.target_task_identity,
        "verification_state": policy.verification_state,
    }


def phase8_label_mapping_policy_to_dict(
    policy: Phase8LabelMappingPolicy,
) -> dict[str, JsonValue]:
    """Convert one label-mapping policy to a canonical mapping."""

    payload = phase8_label_mapping_policy_identity_payload(policy)
    payload["policy_hash"] = policy.policy_hash
    return payload


def hash_phase8_label_mapping_policy(policy: Phase8LabelMappingPolicy) -> str:
    """Return the canonical SHA-256 identity hash for a label-mapping policy."""

    return sha256_json(phase8_label_mapping_policy_identity_payload(policy))


def phase8_label_mapping_policy_to_json(policy: Phase8LabelMappingPolicy) -> bytes:
    """Serialize one label-mapping policy to canonical JSON bytes."""

    return canonical_json_bytes(phase8_label_mapping_policy_to_dict(policy)) + b"\n"


def phase8_source_label_role_from_mapping(mapping: MappingLike) -> Phase8SourceLabelRole:
    """Reconstruct one source-label role from a strict mapping."""

    _require_exact_fields(mapping, _SOURCE_LABEL_ROLE_FIELDS, "Phase8SourceLabelRole")
    return Phase8SourceLabelRole(
        schema_name=_expect_string(mapping["schema_name"], "schema_name"),
        schema_version=_expect_string(mapping["schema_version"], "schema_version"),
        role_id=_expect_string(mapping["role_id"], "role_id"),
        target_usage=_expect_string(mapping["target_usage"], "target_usage"),
        source_value_policy=_expect_string(mapping["source_value_policy"], "source_value_policy"),
        verification_state=_expect_string(mapping["verification_state"], "verification_state"),
        source_basis_reference=_expect_string(
            mapping["source_basis_reference"],
            "source_basis_reference",
        ),
        provenance_reference=_expect_string(
            mapping["provenance_reference"], "provenance_reference"
        ),
    )


def phase8_source_to_target_mapping_rule_from_mapping(
    mapping: MappingLike,
) -> Phase8SourceToTargetMappingRule:
    """Reconstruct one source-to-target mapping rule from a strict mapping."""

    _require_exact_fields(
        mapping, _SOURCE_TO_TARGET_MAPPING_RULE_FIELDS, "Phase8SourceToTargetMappingRule"
    )
    return Phase8SourceToTargetMappingRule(
        schema_name=_expect_string(mapping["schema_name"], "schema_name"),
        schema_version=_expect_string(mapping["schema_version"], "schema_version"),
        rule_id=_expect_string(mapping["rule_id"], "rule_id"),
        source_role_id=_expect_string(mapping["source_role_id"], "source_role_id"),
        target_class=_expect_optional_string(mapping["target_class"], "target_class"),
        rule_action=_expect_string(mapping["rule_action"], "rule_action"),
        aggregation_operator=_expect_string(
            mapping["aggregation_operator"], "aggregation_operator"
        ),
        verification_state=_expect_string(mapping["verification_state"], "verification_state"),
        provenance_reference=_expect_string(
            mapping["provenance_reference"], "provenance_reference"
        ),
    )


def phase8_label_mapping_policy_from_mapping(mapping: MappingLike) -> Phase8LabelMappingPolicy:
    """Reconstruct one label-mapping policy from a strict mapping."""

    _require_exact_fields(mapping, _LABEL_MAPPING_POLICY_FIELDS, "Phase8LabelMappingPolicy")
    roles_value = _expect_list(mapping["permitted_source_roles"], "permitted_source_roles")
    rules_value = _expect_list(mapping["mapping_rules"], "mapping_rules")
    return Phase8LabelMappingPolicy(
        schema_name=_expect_string(mapping["schema_name"], "schema_name"),
        schema_version=_expect_string(mapping["schema_version"], "schema_version"),
        policy_hash=_expect_string(mapping["policy_hash"], "policy_hash"),
        external_cohort_identity=_expect_string(
            mapping["external_cohort_identity"],
            "external_cohort_identity",
        ),
        target_task_identity=_expect_string(
            mapping["target_task_identity"],
            "target_task_identity",
        ),
        target_classes=_expect_string_tuple(mapping["target_classes"], "target_classes"),
        permitted_source_roles=tuple(
            phase8_source_label_role_from_mapping(
                _expect_mapping(item, f"permitted_source_roles[{index}]")
            )
            for index, item in enumerate(roles_value)
        ),
        mapping_rules=tuple(
            phase8_source_to_target_mapping_rule_from_mapping(
                _expect_mapping(item, f"mapping_rules[{index}]")
            )
            for index, item in enumerate(rules_value)
        ),
        liver_mask_handling=_expect_string(mapping["liver_mask_handling"], "liver_mask_handling"),
        lesion_tumor_aggregation=_expect_string(
            mapping["lesion_tumor_aggregation"],
            "lesion_tumor_aggregation",
        ),
        overlap_resolution=_expect_string(mapping["overlap_resolution"], "overlap_resolution"),
        background_policy=_expect_string(mapping["background_policy"], "background_policy"),
        missing_source_policy=_expect_string(
            mapping["missing_source_policy"],
            "missing_source_policy",
        ),
        incompatible_source_policy=_expect_string(
            mapping["incompatible_source_policy"],
            "incompatible_source_policy",
        ),
        multiple_component_policy=_expect_string(
            mapping["multiple_component_policy"],
            "multiple_component_policy",
        ),
        empty_target_policy=_expect_string(mapping["empty_target_policy"], "empty_target_policy"),
        geometry_alignment_policy=_expect_string(
            mapping["geometry_alignment_policy"],
            "geometry_alignment_policy",
        ),
        interpolation_policy=_expect_string(
            mapping["interpolation_policy"],
            "interpolation_policy",
        ),
        connectivity_policy=_expect_string(mapping["connectivity_policy"], "connectivity_policy"),
        verification_state=_expect_string(mapping["verification_state"], "verification_state"),
        source_basis_references=_expect_string_tuple(
            mapping["source_basis_references"],
            "source_basis_references",
        ),
        no_tuning_declaration=_expect_bool(
            mapping["no_tuning_declaration"],
            "no_tuning_declaration",
        ),
    )


def phase8_label_mapping_policy_from_json(data: bytes | str) -> Phase8LabelMappingPolicy:
    """Deserialize one label-mapping policy from JSON bytes or text."""

    return phase8_label_mapping_policy_from_mapping(_json_to_mapping(data))


def build_default_phase8_label_mapping_policy() -> Phase8LabelMappingPolicy:
    """Build the preregistered expected/documented Phase 8 binary tumor mapping policy."""

    roles = (
        Phase8SourceLabelRole(
            schema_name=PHASE8_SOURCE_LABEL_ROLE_SCHEMA_NAME,
            schema_version=PHASE8_SOURCE_LABEL_ROLE_SCHEMA_VERSION,
            role_id="liver_context_mask",
            target_usage="excluded_context_only",
            source_value_policy="not_used_for_target",
            verification_state="expected_documented_not_empirically_verified",
            source_basis_reference="project_spec_liver_context_not_primary_target",
            provenance_reference="docs_phase8_plan_label_mapping_policy",
        ),
        Phase8SourceLabelRole(
            schema_name=PHASE8_SOURCE_LABEL_ROLE_SCHEMA_NAME,
            schema_version=PHASE8_SOURCE_LABEL_ROLE_SCHEMA_VERSION,
            role_id="tumor_lesion_mask",
            target_usage="binary_foreground_source",
            source_value_policy="any_nonzero_voxel",
            verification_state="expected_documented_not_empirically_verified",
            source_basis_reference="project_spec_binary_tumor_foreground",
            provenance_reference="docs_phase8_plan_label_mapping_policy",
        ),
    )
    rules = (
        Phase8SourceToTargetMappingRule(
            schema_name=PHASE8_SOURCE_TO_TARGET_MAPPING_RULE_SCHEMA_NAME,
            schema_version=PHASE8_SOURCE_TO_TARGET_MAPPING_RULE_SCHEMA_VERSION,
            rule_id="exclude_liver_context_from_target",
            source_role_id="liver_context_mask",
            target_class=None,
            rule_action="exclude_from_target",
            aggregation_operator="ignore_for_target",
            verification_state="expected_documented_not_empirically_verified",
            provenance_reference="docs_phase8_plan_label_mapping_policy",
        ),
        Phase8SourceToTargetMappingRule(
            schema_name=PHASE8_SOURCE_TO_TARGET_MAPPING_RULE_SCHEMA_NAME,
            schema_version=PHASE8_SOURCE_TO_TARGET_MAPPING_RULE_SCHEMA_VERSION,
            rule_id="map_tumor_lesion_nonzero_to_foreground",
            source_role_id="tumor_lesion_mask",
            target_class="tumor_foreground",
            rule_action="nonzero_voxels_to_tumor_foreground",
            aggregation_operator="union",
            verification_state="expected_documented_not_empirically_verified",
            provenance_reference="docs_phase8_plan_label_mapping_policy",
        ),
    )
    payload: dict[str, JsonValue] = {
        "background_policy": "all_non_foreground_voxels_background",
        "connectivity_policy": "3d_26_connectivity_for_component_accounting_only",
        "empty_target_policy": "allow_empty_target_with_explicit_empty_status",
        "external_cohort_identity": PHASE8_LABEL_MAPPING_EXTERNAL_COHORT_IDENTITY,
        "geometry_alignment_policy": "require_same_grid_shape_affine_spacing_orientation",
        "incompatible_source_policy": "fail_closed_incompatible_geometry_or_values",
        "interpolation_policy": "nearest_neighbor_for_labels_only_no_resampling_in_policy",
        "lesion_tumor_aggregation": "union_all_permitted_tumor_lesion_sources",
        "liver_mask_handling": "excluded_context_only_never_target_foreground",
        "mapping_rules": [phase8_source_to_target_mapping_rule_to_dict(item) for item in rules],
        "missing_source_policy": "fail_closed_missing_required_tumor_source",
        "multiple_component_policy": "preserve_all_components_no_filtering",
        "no_tuning_declaration": True,
        "overlap_resolution": "tumor_foreground_precedence_over_context_background_else_background",
        "permitted_source_roles": [phase8_source_label_role_to_dict(item) for item in roles],
        "schema_name": PHASE8_LABEL_MAPPING_POLICY_SCHEMA_NAME,
        "schema_version": PHASE8_LABEL_MAPPING_POLICY_SCHEMA_VERSION,
        "source_basis_references": [
            "docs_phase8_plan_label_mapping_policy",
            "phase8_label_access_boundary_record",
            "project_spec_binary_liver_tumor_segmentation",
        ],
        "target_classes": list(PHASE8_LABEL_MAPPING_TARGET_CLASSES),
        "target_task_identity": PHASE8_LABEL_MAPPING_TARGET_TASK_IDENTITY,
        "verification_state": "expected_documented_not_empirically_verified",
    }
    return Phase8LabelMappingPolicy(
        schema_name=PHASE8_LABEL_MAPPING_POLICY_SCHEMA_NAME,
        schema_version=PHASE8_LABEL_MAPPING_POLICY_SCHEMA_VERSION,
        policy_hash=sha256_json(payload),
        external_cohort_identity=PHASE8_LABEL_MAPPING_EXTERNAL_COHORT_IDENTITY,
        target_task_identity=PHASE8_LABEL_MAPPING_TARGET_TASK_IDENTITY,
        target_classes=PHASE8_LABEL_MAPPING_TARGET_CLASSES,
        permitted_source_roles=roles,
        mapping_rules=rules,
        liver_mask_handling="excluded_context_only_never_target_foreground",
        lesion_tumor_aggregation="union_all_permitted_tumor_lesion_sources",
        overlap_resolution="tumor_foreground_precedence_over_context_background_else_background",
        background_policy="all_non_foreground_voxels_background",
        missing_source_policy="fail_closed_missing_required_tumor_source",
        incompatible_source_policy="fail_closed_incompatible_geometry_or_values",
        multiple_component_policy="preserve_all_components_no_filtering",
        empty_target_policy="allow_empty_target_with_explicit_empty_status",
        geometry_alignment_policy="require_same_grid_shape_affine_spacing_orientation",
        interpolation_policy="nearest_neighbor_for_labels_only_no_resampling_in_policy",
        connectivity_policy="3d_26_connectivity_for_component_accounting_only",
        verification_state="expected_documented_not_empirically_verified",
        source_basis_references=(
            "docs_phase8_plan_label_mapping_policy",
            "phase8_label_access_boundary_record",
            "project_spec_binary_liver_tumor_segmentation",
        ),
        no_tuning_declaration=True,
    )


def apply_phase8_label_mapping_policy(
    policy: Phase8LabelMappingPolicy,
    source_role_masks: Mapping[str, NDArray[np.generic]],
) -> BoolArray:
    """Apply the frozen policy to caller-provided masks and return binary tumor foreground.

    This helper is intentionally role-based. It never accepts paths, archive names, class names from
    source files, or observed external label values.
    """

    if "tumor_lesion_mask" not in source_role_masks:
        raise Phase8LabelMappingValidationError(policy.missing_source_policy)
    unknown_roles = sorted(set(source_role_masks) - PHASE8_LABEL_MAPPING_PERMITTED_SOURCE_ROLES)
    if unknown_roles:
        raise Phase8LabelMappingValidationError(f"unknown source label roles: {unknown_roles!r}")
    tumor_mask = _as_3d_mask(source_role_masks["tumor_lesion_mask"], "tumor_lesion_mask")
    for role_id, role_mask in source_role_masks.items():
        candidate = _as_3d_mask(role_mask, role_id)
        if candidate.shape != tumor_mask.shape:
            raise Phase8LabelMappingValidationError(policy.incompatible_source_policy)
    return tumor_mask


_SOURCE_LABEL_ROLE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "role_id",
        "target_usage",
        "source_value_policy",
        "verification_state",
        "source_basis_reference",
        "provenance_reference",
    }
)
_SOURCE_TO_TARGET_MAPPING_RULE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "rule_id",
        "source_role_id",
        "target_class",
        "rule_action",
        "aggregation_operator",
        "verification_state",
        "provenance_reference",
    }
)
_LABEL_MAPPING_POLICY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "policy_hash",
        "external_cohort_identity",
        "target_task_identity",
        "target_classes",
        "permitted_source_roles",
        "mapping_rules",
        "liver_mask_handling",
        "lesion_tumor_aggregation",
        "overlap_resolution",
        "background_policy",
        "missing_source_policy",
        "incompatible_source_policy",
        "multiple_component_policy",
        "empty_target_policy",
        "geometry_alignment_policy",
        "interpolation_policy",
        "connectivity_policy",
        "verification_state",
        "source_basis_references",
        "no_tuning_declaration",
    }
)


def _json_to_mapping(data: bytes | str) -> MappingLike:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Phase8LabelMappingSerializationError("invalid label-mapping JSON payload.") from exc
    if not isinstance(decoded, Mapping):
        raise Phase8LabelMappingSerializationError("label-mapping JSON root must be an object.")
    return cast(MappingLike, decoded)


def _expect_mapping(value: object, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase8LabelMappingSerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase8LabelMappingSerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name)


def _expect_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase8LabelMappingSerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise Phase8LabelMappingSerializationError(f"{field_name} must be a list.")
    return value


def _expect_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Phase8LabelMappingSerializationError(f"{field_name} must be a list of strings.")
    return tuple(_expect_string(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _require_exact_fields(
    mapping: MappingLike,
    expected_fields: frozenset[str],
    object_name: str,
) -> None:
    keys = set(mapping)
    missing = sorted(expected_fields - keys)
    extra = sorted(keys - expected_fields)
    if missing or extra:
        raise Phase8LabelMappingSerializationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _require_exact(value: object, expected: object, field_name: str) -> None:
    if value != expected:
        raise Phase8LabelMappingValidationError(f"{field_name} must be {expected!r}.")


def _require_allowed(value: str, allowed: frozenset[str], field_name: str) -> None:
    if value not in allowed:
        raise Phase8LabelMappingValidationError(f"{field_name} must be one of {sorted(allowed)!r}.")


def _require_sha256(value: str, field_name: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise Phase8LabelMappingValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_safe_token(value: str, field_name: str) -> None:
    lowered = value.lower()
    if (
        value.startswith("/")
        or value.startswith("~")
        or "\\" in value
        or _WINDOWS_ABSOLUTE_PATH_RE.match(value)
        or "/volumes" in lowered
        or "timestamp" in lowered
        or "localhost" in lowered
    ):
        raise Phase8LabelMappingValidationError(
            f"{field_name} must not contain local paths or machine-specific metadata."
        )
    if _TOKEN_RE.fullmatch(value) is None:
        raise Phase8LabelMappingValidationError(f"{field_name} must be a safe token.")


def _require_sorted_unique_tokens(values: tuple[str, ...], field_name: str) -> None:
    if not values:
        raise Phase8LabelMappingValidationError(f"{field_name} must be non-empty.")
    if tuple(sorted(values)) != values:
        raise Phase8LabelMappingValidationError(f"{field_name} must be canonical sorted.")
    if len(set(values)) != len(values):
        raise Phase8LabelMappingValidationError(f"{field_name} must not contain duplicates.")
    for index, value in enumerate(values):
        _require_safe_token(value, f"{field_name}[{index}]")


def _require_exact_role_set(roles: tuple[Phase8SourceLabelRole, ...]) -> None:
    role_ids = tuple(role.role_id for role in roles)
    if role_ids != tuple(sorted(PHASE8_LABEL_MAPPING_PERMITTED_SOURCE_ROLES)):
        raise Phase8LabelMappingValidationError(
            "permitted_source_roles must contain the exact preregistered role set."
        )


def _require_exact_rule_set(rules: tuple[Phase8SourceToTargetMappingRule, ...]) -> None:
    rule_ids = tuple(rule.rule_id for rule in rules)
    if rule_ids != PHASE8_LABEL_MAPPING_REQUIRED_RULE_IDS:
        raise Phase8LabelMappingValidationError(
            "mapping_rules must contain the exact preregistered rule set."
        )


def _as_3d_mask(value: NDArray[np.generic], role_id: str) -> BoolArray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise Phase8LabelMappingValidationError(f"{role_id} must be a 3D mask.")
    if not np.isfinite(array).all():
        raise Phase8LabelMappingValidationError(f"{role_id} must contain only finite values.")
    return cast(BoolArray, array != 0)


__all__ = [
    "PHASE8_LABEL_MAPPING_EXTERNAL_COHORT_IDENTITY",
    "PHASE8_LABEL_MAPPING_POLICY_SCHEMA_NAME",
    "PHASE8_LABEL_MAPPING_POLICY_SCHEMA_VERSION",
    "PHASE8_LABEL_MAPPING_TARGET_CLASSES",
    "PHASE8_LABEL_MAPPING_TARGET_TASK_IDENTITY",
    "PHASE8_SOURCE_LABEL_ROLE_SCHEMA_NAME",
    "PHASE8_SOURCE_LABEL_ROLE_SCHEMA_VERSION",
    "PHASE8_SOURCE_TO_TARGET_MAPPING_RULE_SCHEMA_NAME",
    "PHASE8_SOURCE_TO_TARGET_MAPPING_RULE_SCHEMA_VERSION",
    "Phase8LabelMappingError",
    "Phase8LabelMappingHashError",
    "Phase8LabelMappingPolicy",
    "Phase8LabelMappingSerializationError",
    "Phase8LabelMappingValidationError",
    "Phase8SourceLabelRole",
    "Phase8SourceToTargetMappingRule",
    "apply_phase8_label_mapping_policy",
    "build_default_phase8_label_mapping_policy",
    "hash_phase8_label_mapping_policy",
    "phase8_label_mapping_policy_from_json",
    "phase8_label_mapping_policy_from_mapping",
    "phase8_label_mapping_policy_identity_payload",
    "phase8_label_mapping_policy_to_dict",
    "phase8_label_mapping_policy_to_json",
    "phase8_source_label_role_from_mapping",
    "phase8_source_label_role_to_dict",
    "phase8_source_to_target_mapping_rule_from_mapping",
    "phase8_source_to_target_mapping_rule_to_dict",
]
