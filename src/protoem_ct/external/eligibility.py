"""Phase 8 immutable external eligibility and label-compatibility accounting.

The contracts here are pure accounting surfaces. They perform no filesystem I/O, open no label
archives, and persist only anonymous case IDs plus versioned status/reason codes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.external.manifest import (
    PHASE8_IRCADB_CASE_COUNT,
    Phase8ExternalImageCase,
    Phase8ExternalImageManifest,
)

PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_NAME: Final[str] = "phase8_eligibility_accounting"
PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_ELIGIBILITY_CASE_SCHEMA_NAME: Final[str] = "phase8_eligibility_case"
PHASE8_ELIGIBILITY_CASE_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_ELIGIBILITY_POLICY_SCHEMA_NAME: Final[str] = "phase8_eligibility_policy"
PHASE8_ELIGIBILITY_POLICY_SCHEMA_VERSION: Final[str] = "v1"

PHASE8_ELIGIBILITY_ACCOUNTING_STATES: Final[frozenset[str]] = frozenset(
    {"pre_label_access_pending", "post_label_access_accounted"}
)
PHASE8_LABEL_COMPATIBILITY_STATUSES: Final[frozenset[str]] = frozenset(
    {"pending", "compatible", "incompatible", "not_applicable"}
)
PHASE8_ELIGIBILITY_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "image_qa_failed",
        "image_not_discovered",
        "image_not_readable",
        "inference_ineligible",
        "label_compatibility_pending",
        "label_incompatible",
    }
)
PHASE8_LABEL_COMPATIBILITY_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "compatible_binary_tumor_target",
        "label_archive_missing",
        "label_geometry_incompatible",
        "label_mapping_policy_mismatch",
        "label_unreadable",
        "non_binary_target_after_mapping",
        "pending_label_access",
        "tumor_target_absent",
    }
)

MappingLike: TypeAlias = Mapping[str, object]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ANONYMOUS_CASE_ID_RE = re.compile(r"^ext-ircadb-[0-9]{3}$")


class Phase8EligibilityError(ValueError):
    """Base error for Phase 8 eligibility accounting contracts."""


class Phase8EligibilityValidationError(Phase8EligibilityError):
    """Raised when eligibility accounting violates immutable status rules."""


class Phase8EligibilitySerializationError(Phase8EligibilityError):
    """Raised when eligibility JSON cannot be reconstructed strictly."""


class Phase8EligibilityHashError(Phase8EligibilityError):
    """Raised when an eligibility self-hash does not match canonical content."""


@dataclass(frozen=True, slots=True)
class Phase8EligibilityPolicy:
    """Self-validating preregistered eligibility and exclusion policy artifact."""

    schema_name: str
    schema_version: str
    policy_hash: str
    accounting_states: tuple[str, ...]
    evaluation_requires: tuple[str, ...]
    label_compatibility_statuses: tuple[str, ...]
    post_label_access_rule: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_exact(self.schema_name, PHASE8_ELIGIBILITY_POLICY_SCHEMA_NAME, "schema_name")
        _require_exact(
            self.schema_version,
            PHASE8_ELIGIBILITY_POLICY_SCHEMA_VERSION,
            "schema_version",
        )
        _require_sha256(self.policy_hash, "policy_hash")
        object.__setattr__(self, "accounting_states", tuple(sorted(self.accounting_states)))
        object.__setattr__(
            self,
            "label_compatibility_statuses",
            tuple(sorted(self.label_compatibility_statuses)),
        )
        object.__setattr__(self, "reason_codes", tuple(sorted(self.reason_codes)))
        if set(self.accounting_states) != PHASE8_ELIGIBILITY_ACCOUNTING_STATES:
            raise Phase8EligibilityValidationError(
                "eligibility policy accounting states do not match the frozen contract."
            )
        if self.evaluation_requires != _PHASE8_EVALUATION_REQUIREMENTS:
            raise Phase8EligibilityValidationError(
                "eligibility policy evaluation requirements changed."
            )
        if set(self.label_compatibility_statuses) != PHASE8_LABEL_COMPATIBILITY_STATUSES:
            raise Phase8EligibilityValidationError(
                "eligibility policy label compatibility statuses changed."
            )
        _require_exact(
            self.post_label_access_rule,
            "evaluation_eligible may become true only after compatible label accounting",
            "post_label_access_rule",
        )
        if set(self.reason_codes) != PHASE8_ELIGIBILITY_REASON_CODES:
            raise Phase8EligibilityValidationError("eligibility policy reason codes changed.")
        expected_hash = hash_phase8_eligibility_policy(self)
        if self.policy_hash != expected_hash:
            raise Phase8EligibilityHashError(
                "policy_hash does not match deterministic eligibility policy content."
            )


@dataclass(frozen=True, slots=True)
class Phase8EligibilityCase:
    """Anonymous eligibility state for one external case."""

    schema_name: str
    schema_version: str
    anonymous_case_id: str
    discovered: bool
    image_readable: bool
    image_qa_eligible: bool
    inference_eligible: bool
    label_compatibility_status: str
    label_compatibility_reason_codes: tuple[str, ...]
    evaluation_eligible: bool
    excluded: bool
    deferred: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_exact(self.schema_name, PHASE8_ELIGIBILITY_CASE_SCHEMA_NAME, "schema_name")
        _require_exact(
            self.schema_version,
            PHASE8_ELIGIBILITY_CASE_SCHEMA_VERSION,
            "schema_version",
        )
        _require_anonymous_case_id(self.anonymous_case_id)
        _require_allowed(
            self.label_compatibility_status,
            PHASE8_LABEL_COMPATIBILITY_STATUSES,
            "label_compatibility_status",
        )
        ordered_reasons = tuple(sorted(self.reason_codes))
        ordered_label_reasons = tuple(sorted(self.label_compatibility_reason_codes))
        object.__setattr__(self, "reason_codes", ordered_reasons)
        object.__setattr__(
            self,
            "label_compatibility_reason_codes",
            ordered_label_reasons,
        )
        _require_reason_codes(ordered_reasons, PHASE8_ELIGIBILITY_REASON_CODES, "reason_codes")
        _require_reason_codes(
            ordered_label_reasons,
            PHASE8_LABEL_COMPATIBILITY_REASON_CODES,
            "label_compatibility_reason_codes",
        )
        _require_case_status_consistency(self)

    @property
    def ordering_key(self) -> str:
        """Return the deterministic accounting ordering key."""

        return self.anonymous_case_id


@dataclass(frozen=True, slots=True)
class Phase8EligibilityAccounting:
    """Self-validating immutable eligibility accounting artifact."""

    schema_name: str
    schema_version: str
    accounting_hash: str
    accounting_state: str
    external_image_manifest_hash: str
    preregistered_policy_hash: str
    label_access_ledger_hash: str | None
    case_count: int
    discovered_count: int
    image_readable_count: int
    image_qa_eligible_count: int
    inference_eligible_count: int
    label_compatibility_pending_count: int
    evaluation_eligible_count: int
    excluded_count: int
    deferred_count: int
    cases: tuple[Phase8EligibilityCase, ...]

    def __post_init__(self) -> None:
        _require_exact(
            self.schema_name,
            PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_NAME,
            "schema_name",
        )
        _require_exact(
            self.schema_version,
            PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_VERSION,
            "schema_version",
        )
        _require_sha256(self.accounting_hash, "accounting_hash")
        _require_allowed(
            self.accounting_state,
            PHASE8_ELIGIBILITY_ACCOUNTING_STATES,
            "accounting_state",
        )
        _require_sha256(self.external_image_manifest_hash, "external_image_manifest_hash")
        _require_sha256(self.preregistered_policy_hash, "preregistered_policy_hash")
        if self.label_access_ledger_hash is not None:
            _require_sha256(self.label_access_ledger_hash, "label_access_ledger_hash")
        ordered = tuple(sorted(self.cases, key=lambda item: item.ordering_key))
        object.__setattr__(self, "cases", ordered)
        _require_accounting_state(self)
        _require_counts(self)
        expected_hash = hash_phase8_eligibility_accounting(self)
        if self.accounting_hash != expected_hash:
            raise Phase8EligibilityHashError(
                "accounting_hash does not match deterministic eligibility content."
            )


def phase8_eligibility_policy_identity_payload(
    policy: Phase8EligibilityPolicy,
) -> dict[str, JsonValue]:
    """Return the eligibility policy identity payload excluding ``policy_hash``."""

    return {
        "accounting_states": list(policy.accounting_states),
        "evaluation_requires": list(policy.evaluation_requires),
        "label_compatibility_statuses": list(policy.label_compatibility_statuses),
        "post_label_access_rule": policy.post_label_access_rule,
        "reason_codes": list(policy.reason_codes),
        "schema_name": PHASE8_ELIGIBILITY_POLICY_SCHEMA_NAME,
        "schema_version": PHASE8_ELIGIBILITY_POLICY_SCHEMA_VERSION,
    }


def phase8_eligibility_policy_payload() -> dict[str, JsonValue]:
    """Return the default preregistered eligibility policy payload with self-hash."""

    return phase8_eligibility_policy_to_dict(build_phase8_eligibility_policy())


def phase8_eligibility_policy_to_dict(policy: Phase8EligibilityPolicy) -> dict[str, JsonValue]:
    """Convert one eligibility policy artifact to canonical JSON."""

    payload = phase8_eligibility_policy_identity_payload(policy)
    payload["policy_hash"] = policy.policy_hash
    return payload


def hash_phase8_eligibility_policy(policy: Phase8EligibilityPolicy | None = None) -> str:
    """Return the deterministic hash for the preregistered eligibility policy."""

    if policy is None:
        payload = _default_phase8_eligibility_policy_identity_payload()
    else:
        payload = phase8_eligibility_policy_identity_payload(policy)
    return sha256_json(payload)


def phase8_eligibility_policy_to_json(policy: Phase8EligibilityPolicy) -> bytes:
    """Serialize one eligibility policy artifact to canonical JSON bytes."""

    return canonical_json_bytes(phase8_eligibility_policy_to_dict(policy)) + b"\n"


def phase8_eligibility_policy_from_mapping(mapping: MappingLike) -> Phase8EligibilityPolicy:
    """Reconstruct one eligibility policy from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_ELIGIBILITY_POLICY_FIELDS,
        object_name="Phase8EligibilityPolicy",
    )
    return Phase8EligibilityPolicy(
        schema_name=_expect_string(mapping["schema_name"], "schema_name"),
        schema_version=_expect_string(mapping["schema_version"], "schema_version"),
        policy_hash=_expect_string(mapping["policy_hash"], "policy_hash"),
        accounting_states=_expect_string_tuple(mapping["accounting_states"], "accounting_states"),
        evaluation_requires=_expect_string_tuple(
            mapping["evaluation_requires"],
            "evaluation_requires",
        ),
        label_compatibility_statuses=_expect_string_tuple(
            mapping["label_compatibility_statuses"],
            "label_compatibility_statuses",
        ),
        post_label_access_rule=_expect_string(
            mapping["post_label_access_rule"],
            "post_label_access_rule",
        ),
        reason_codes=_expect_string_tuple(mapping["reason_codes"], "reason_codes"),
    )


def phase8_eligibility_policy_from_json(data: bytes | str) -> Phase8EligibilityPolicy:
    """Deserialize one eligibility policy from JSON bytes or text."""

    return phase8_eligibility_policy_from_mapping(_json_to_mapping(data))


def build_phase8_eligibility_policy() -> Phase8EligibilityPolicy:
    """Build the preregistered Phase 8 eligibility policy artifact."""

    payload = _default_phase8_eligibility_policy_identity_payload()
    return Phase8EligibilityPolicy(
        schema_name=PHASE8_ELIGIBILITY_POLICY_SCHEMA_NAME,
        schema_version=PHASE8_ELIGIBILITY_POLICY_SCHEMA_VERSION,
        policy_hash=sha256_json(payload),
        accounting_states=tuple(sorted(PHASE8_ELIGIBILITY_ACCOUNTING_STATES)),
        evaluation_requires=_PHASE8_EVALUATION_REQUIREMENTS,
        label_compatibility_statuses=tuple(sorted(PHASE8_LABEL_COMPATIBILITY_STATUSES)),
        post_label_access_rule=(
            "evaluation_eligible may become true only after compatible label accounting"
        ),
        reason_codes=tuple(sorted(PHASE8_ELIGIBILITY_REASON_CODES)),
    )


def phase8_eligibility_case_to_dict(case: Phase8EligibilityCase) -> dict[str, JsonValue]:
    """Convert one eligibility case to a canonical mapping."""

    return {
        "anonymous_case_id": case.anonymous_case_id,
        "deferred": case.deferred,
        "discovered": case.discovered,
        "evaluation_eligible": case.evaluation_eligible,
        "excluded": case.excluded,
        "image_qa_eligible": case.image_qa_eligible,
        "image_readable": case.image_readable,
        "inference_eligible": case.inference_eligible,
        "label_compatibility_reason_codes": list(case.label_compatibility_reason_codes),
        "label_compatibility_status": case.label_compatibility_status,
        "reason_codes": list(case.reason_codes),
        "schema_name": case.schema_name,
        "schema_version": case.schema_version,
    }


def phase8_eligibility_accounting_identity_payload(
    accounting: Phase8EligibilityAccounting,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload excluding ``accounting_hash``."""

    ordered = tuple(sorted(accounting.cases, key=lambda item: item.ordering_key))
    return {
        "accounting_state": accounting.accounting_state,
        "case_count": accounting.case_count,
        "cases": [phase8_eligibility_case_to_dict(case) for case in ordered],
        "deferred_count": accounting.deferred_count,
        "discovered_count": accounting.discovered_count,
        "evaluation_eligible_count": accounting.evaluation_eligible_count,
        "excluded_count": accounting.excluded_count,
        "external_image_manifest_hash": accounting.external_image_manifest_hash,
        "image_qa_eligible_count": accounting.image_qa_eligible_count,
        "image_readable_count": accounting.image_readable_count,
        "inference_eligible_count": accounting.inference_eligible_count,
        "label_access_ledger_hash": accounting.label_access_ledger_hash,
        "label_compatibility_pending_count": accounting.label_compatibility_pending_count,
        "preregistered_policy_hash": accounting.preregistered_policy_hash,
        "schema_name": accounting.schema_name,
        "schema_version": accounting.schema_version,
    }


def phase8_eligibility_accounting_to_dict(
    accounting: Phase8EligibilityAccounting,
) -> dict[str, JsonValue]:
    """Convert one eligibility accounting artifact to a canonical mapping."""

    payload = phase8_eligibility_accounting_identity_payload(accounting)
    payload["accounting_hash"] = accounting.accounting_hash
    return payload


def hash_phase8_eligibility_accounting(accounting: Phase8EligibilityAccounting) -> str:
    """Return the lowercase SHA-256 identity hash for eligibility accounting."""

    return sha256_json(phase8_eligibility_accounting_identity_payload(accounting))


def phase8_eligibility_accounting_to_json(accounting: Phase8EligibilityAccounting) -> bytes:
    """Serialize one eligibility accounting artifact to canonical JSON bytes."""

    return canonical_json_bytes(phase8_eligibility_accounting_to_dict(accounting)) + b"\n"


def phase8_eligibility_case_from_mapping(mapping: MappingLike) -> Phase8EligibilityCase:
    """Reconstruct one eligibility case from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_ELIGIBILITY_CASE_FIELDS,
        object_name="Phase8EligibilityCase",
    )
    return Phase8EligibilityCase(
        schema_name=_expect_string(mapping["schema_name"], "schema_name"),
        schema_version=_expect_string(mapping["schema_version"], "schema_version"),
        anonymous_case_id=_expect_string(mapping["anonymous_case_id"], "anonymous_case_id"),
        discovered=_expect_bool(mapping["discovered"], "discovered"),
        image_readable=_expect_bool(mapping["image_readable"], "image_readable"),
        image_qa_eligible=_expect_bool(mapping["image_qa_eligible"], "image_qa_eligible"),
        inference_eligible=_expect_bool(mapping["inference_eligible"], "inference_eligible"),
        label_compatibility_status=_expect_string(
            mapping["label_compatibility_status"],
            "label_compatibility_status",
        ),
        label_compatibility_reason_codes=_expect_string_tuple(
            mapping["label_compatibility_reason_codes"],
            "label_compatibility_reason_codes",
        ),
        evaluation_eligible=_expect_bool(
            mapping["evaluation_eligible"],
            "evaluation_eligible",
        ),
        excluded=_expect_bool(mapping["excluded"], "excluded"),
        deferred=_expect_bool(mapping["deferred"], "deferred"),
        reason_codes=_expect_string_tuple(mapping["reason_codes"], "reason_codes"),
    )


def phase8_eligibility_accounting_from_mapping(
    mapping: MappingLike,
) -> Phase8EligibilityAccounting:
    """Reconstruct one eligibility accounting artifact from a strict mapping."""

    _require_exact_fields(
        mapping,
        required_fields=_ELIGIBILITY_ACCOUNTING_FIELDS,
        object_name="Phase8EligibilityAccounting",
    )
    case_values = mapping["cases"]
    if not isinstance(case_values, list):
        raise Phase8EligibilitySerializationError("cases must be a list.")
    return Phase8EligibilityAccounting(
        schema_name=_expect_string(mapping["schema_name"], "schema_name"),
        schema_version=_expect_string(mapping["schema_version"], "schema_version"),
        accounting_hash=_expect_string(mapping["accounting_hash"], "accounting_hash"),
        accounting_state=_expect_string(mapping["accounting_state"], "accounting_state"),
        external_image_manifest_hash=_expect_string(
            mapping["external_image_manifest_hash"],
            "external_image_manifest_hash",
        ),
        preregistered_policy_hash=_expect_string(
            mapping["preregistered_policy_hash"],
            "preregistered_policy_hash",
        ),
        label_access_ledger_hash=_expect_optional_string(
            mapping["label_access_ledger_hash"],
            "label_access_ledger_hash",
        ),
        case_count=_expect_int(mapping["case_count"], "case_count"),
        discovered_count=_expect_int(mapping["discovered_count"], "discovered_count"),
        image_readable_count=_expect_int(mapping["image_readable_count"], "image_readable_count"),
        image_qa_eligible_count=_expect_int(
            mapping["image_qa_eligible_count"],
            "image_qa_eligible_count",
        ),
        inference_eligible_count=_expect_int(
            mapping["inference_eligible_count"],
            "inference_eligible_count",
        ),
        label_compatibility_pending_count=_expect_int(
            mapping["label_compatibility_pending_count"],
            "label_compatibility_pending_count",
        ),
        evaluation_eligible_count=_expect_int(
            mapping["evaluation_eligible_count"],
            "evaluation_eligible_count",
        ),
        excluded_count=_expect_int(mapping["excluded_count"], "excluded_count"),
        deferred_count=_expect_int(mapping["deferred_count"], "deferred_count"),
        cases=tuple(
            phase8_eligibility_case_from_mapping(_expect_mapping(item, f"cases[{index}]"))
            for index, item in enumerate(case_values)
        ),
    )


def phase8_eligibility_accounting_from_json(
    data: bytes | str,
) -> Phase8EligibilityAccounting:
    """Deserialize one eligibility accounting artifact from JSON bytes or text."""

    return phase8_eligibility_accounting_from_mapping(_json_to_mapping(data))


def build_phase8_prelabel_eligibility_accounting(
    *,
    image_manifest: Phase8ExternalImageManifest,
    preregistered_policy_hash: str | None = None,
) -> Phase8EligibilityAccounting:
    """Build image-only eligibility accounting with label compatibility still pending."""

    policy_hash = preregistered_policy_hash or hash_phase8_eligibility_policy()
    cases = tuple(_prelabel_case_from_manifest_case(case) for case in image_manifest.cases)
    return _build_accounting(
        accounting_state="pre_label_access_pending",
        external_image_manifest_hash=image_manifest.manifest_hash,
        preregistered_policy_hash=policy_hash,
        label_access_ledger_hash=None,
        cases=cases,
    )


def build_phase8_post_label_eligibility_accounting(
    *,
    previous_accounting: Phase8EligibilityAccounting,
    label_access_ledger_hash: str,
    compatibility_updates: Sequence[Phase8EligibilityCase],
) -> Phase8EligibilityAccounting:
    """Build post-label-access accounting from explicit anonymous compatibility updates.

    The caller supplies already-accounted anonymous compatibility states. This function validates
    that image and inference eligibility inherited from the pre-label artifact did not change.
    """

    _require_sha256(label_access_ledger_hash, "label_access_ledger_hash")
    if previous_accounting.accounting_state != "pre_label_access_pending":
        raise Phase8EligibilityValidationError(
            "post-label accounting must start from pre_label_access_pending accounting."
        )
    updates_by_id = {case.anonymous_case_id: case for case in compatibility_updates}
    if len(updates_by_id) != len(compatibility_updates):
        raise Phase8EligibilityValidationError("compatibility updates duplicate anonymous IDs.")
    previous_ids = {case.anonymous_case_id for case in previous_accounting.cases}
    if set(updates_by_id) != previous_ids:
        raise Phase8EligibilityValidationError(
            "compatibility updates must cover exactly the previous accounting cases."
        )
    updated_cases: list[Phase8EligibilityCase] = []
    for original in previous_accounting.cases:
        updated = updates_by_id[original.anonymous_case_id]
        _require_unchanged_image_accounting(original, updated)
        updated_cases.append(updated)
    return _build_accounting(
        accounting_state="post_label_access_accounted",
        external_image_manifest_hash=previous_accounting.external_image_manifest_hash,
        preregistered_policy_hash=previous_accounting.preregistered_policy_hash,
        label_access_ledger_hash=label_access_ledger_hash,
        cases=tuple(updated_cases),
    )


def build_phase8_label_compatible_case(
    case: Phase8EligibilityCase,
    *,
    compatible: bool,
    label_compatibility_reason_codes: Sequence[str],
) -> Phase8EligibilityCase:
    """Return an explicit post-label compatibility status for one anonymous case."""

    label_reasons = tuple(label_compatibility_reason_codes)
    if compatible:
        status = "compatible"
        reason_codes = tuple(
            item for item in case.reason_codes if item != "label_compatibility_pending"
        )
    else:
        status = "incompatible"
        reason_codes = tuple(
            sorted(
                {
                    *(item for item in case.reason_codes if item != "label_compatibility_pending"),
                    "label_incompatible",
                }
            )
        )
    evaluation_eligible = compatible and case.inference_eligible
    return Phase8EligibilityCase(
        schema_name=PHASE8_ELIGIBILITY_CASE_SCHEMA_NAME,
        schema_version=PHASE8_ELIGIBILITY_CASE_SCHEMA_VERSION,
        anonymous_case_id=case.anonymous_case_id,
        discovered=case.discovered,
        image_readable=case.image_readable,
        image_qa_eligible=case.image_qa_eligible,
        inference_eligible=case.inference_eligible,
        label_compatibility_status=status if case.inference_eligible else "not_applicable",
        label_compatibility_reason_codes=label_reasons,
        evaluation_eligible=evaluation_eligible,
        excluded=not evaluation_eligible,
        deferred=False,
        reason_codes=reason_codes,
    )


def _prelabel_case_from_manifest_case(case: Phase8ExternalImageCase) -> Phase8EligibilityCase:
    readable = case.image_member_count > 0 and "readable_archive_failed" not in case.qa_reason_codes
    image_qa_eligible = case.qa_status == "passed"
    inference_eligible = case.inclusion_eligible_for_inference
    reason_codes: set[str] = set()
    if not readable:
        reason_codes.add("image_not_readable")
    if not image_qa_eligible:
        reason_codes.add("image_qa_failed")
    if not inference_eligible:
        reason_codes.add("inference_ineligible")
    if inference_eligible:
        reason_codes.add("label_compatibility_pending")
    return Phase8EligibilityCase(
        schema_name=PHASE8_ELIGIBILITY_CASE_SCHEMA_NAME,
        schema_version=PHASE8_ELIGIBILITY_CASE_SCHEMA_VERSION,
        anonymous_case_id=case.anonymous_case_id,
        discovered=True,
        image_readable=readable,
        image_qa_eligible=image_qa_eligible,
        inference_eligible=inference_eligible,
        label_compatibility_status="pending" if inference_eligible else "not_applicable",
        label_compatibility_reason_codes=("pending_label_access",) if inference_eligible else (),
        evaluation_eligible=False,
        excluded=not inference_eligible,
        deferred=inference_eligible,
        reason_codes=tuple(sorted(reason_codes)),
    )


def _build_accounting(
    *,
    accounting_state: str,
    external_image_manifest_hash: str,
    preregistered_policy_hash: str,
    label_access_ledger_hash: str | None,
    cases: tuple[Phase8EligibilityCase, ...],
) -> Phase8EligibilityAccounting:
    ordered = tuple(sorted(cases, key=lambda item: item.ordering_key))
    deferred_count = sum(1 for case in ordered if case.deferred)
    discovered_count = sum(1 for case in ordered if case.discovered)
    evaluation_eligible_count = sum(1 for case in ordered if case.evaluation_eligible)
    excluded_count = sum(1 for case in ordered if case.excluded)
    image_qa_eligible_count = sum(1 for case in ordered if case.image_qa_eligible)
    image_readable_count = sum(1 for case in ordered if case.image_readable)
    inference_eligible_count = sum(1 for case in ordered if case.inference_eligible)
    label_compatibility_pending_count = sum(
        1 for case in ordered if case.label_compatibility_status == "pending"
    )
    payload: dict[str, JsonValue] = {
        "accounting_state": accounting_state,
        "case_count": len(ordered),
        "cases": [phase8_eligibility_case_to_dict(case) for case in ordered],
        "deferred_count": deferred_count,
        "discovered_count": discovered_count,
        "evaluation_eligible_count": evaluation_eligible_count,
        "excluded_count": excluded_count,
        "external_image_manifest_hash": external_image_manifest_hash,
        "image_qa_eligible_count": image_qa_eligible_count,
        "image_readable_count": image_readable_count,
        "inference_eligible_count": inference_eligible_count,
        "label_access_ledger_hash": label_access_ledger_hash,
        "label_compatibility_pending_count": label_compatibility_pending_count,
        "preregistered_policy_hash": preregistered_policy_hash,
        "schema_name": PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_NAME,
        "schema_version": PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_VERSION,
    }
    return Phase8EligibilityAccounting(
        schema_name=PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_NAME,
        schema_version=PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_VERSION,
        accounting_hash=sha256_json(payload),
        accounting_state=accounting_state,
        external_image_manifest_hash=external_image_manifest_hash,
        preregistered_policy_hash=preregistered_policy_hash,
        label_access_ledger_hash=label_access_ledger_hash,
        case_count=len(ordered),
        discovered_count=discovered_count,
        image_readable_count=image_readable_count,
        image_qa_eligible_count=image_qa_eligible_count,
        inference_eligible_count=inference_eligible_count,
        label_compatibility_pending_count=label_compatibility_pending_count,
        evaluation_eligible_count=evaluation_eligible_count,
        excluded_count=excluded_count,
        deferred_count=deferred_count,
        cases=ordered,
    )


def _require_case_status_consistency(case: Phase8EligibilityCase) -> None:
    if not case.discovered:
        raise Phase8EligibilityValidationError("Phase 8 accounting must not silently omit cases.")
    if not case.image_readable and "image_not_readable" not in case.reason_codes:
        raise Phase8EligibilityValidationError("unreadable images require image_not_readable.")
    if not case.image_qa_eligible and "image_qa_failed" not in case.reason_codes:
        raise Phase8EligibilityValidationError("image QA failures require image_qa_failed.")
    if not case.inference_eligible and "inference_ineligible" not in case.reason_codes:
        raise Phase8EligibilityValidationError("inference-ineligible cases require a reason.")
    if case.evaluation_eligible:
        if not (
            case.discovered
            and case.image_readable
            and case.image_qa_eligible
            and case.inference_eligible
            and case.label_compatibility_status == "compatible"
        ):
            raise Phase8EligibilityValidationError(
                "evaluation_eligible requires image, inference, and compatible label accounting."
            )
        if case.excluded or case.deferred or case.reason_codes:
            raise Phase8EligibilityValidationError(
                "evaluation-eligible cases must not be excluded, deferred, or reason-coded."
            )
    if case.label_compatibility_status == "pending":
        if not case.deferred or case.evaluation_eligible:
            raise Phase8EligibilityValidationError("pending label compatibility must be deferred.")
        if "label_compatibility_pending" not in case.reason_codes:
            raise Phase8EligibilityValidationError(
                "pending label compatibility requires label_compatibility_pending."
            )
        if "pending_label_access" not in case.label_compatibility_reason_codes:
            raise Phase8EligibilityValidationError(
                "pending label compatibility requires pending_label_access."
            )
    if case.label_compatibility_status == "incompatible":
        if not case.excluded or case.deferred or case.evaluation_eligible:
            raise Phase8EligibilityValidationError(
                "label-incompatible cases must be excluded, not deferred."
            )
        if "label_incompatible" not in case.reason_codes:
            raise Phase8EligibilityValidationError(
                "label-incompatible cases require label_incompatible."
            )
    if case.excluded and case.deferred:
        raise Phase8EligibilityValidationError("cases cannot be both excluded and deferred.")


def _require_accounting_state(accounting: Phase8EligibilityAccounting) -> None:
    if accounting.accounting_state == "pre_label_access_pending":
        if accounting.label_access_ledger_hash is not None:
            raise Phase8EligibilityValidationError(
                "pre-label accounting must not include label_access_ledger_hash."
            )
        if any(case.evaluation_eligible for case in accounting.cases):
            raise Phase8EligibilityValidationError(
                "pre-label accounting cannot claim evaluation eligibility."
            )
    if accounting.accounting_state == "post_label_access_accounted":
        if accounting.label_access_ledger_hash is None:
            raise Phase8EligibilityValidationError(
                "post-label accounting requires label_access_ledger_hash."
            )
        if any(case.label_compatibility_status == "pending" for case in accounting.cases):
            raise Phase8EligibilityValidationError(
                "post-label accounting cannot retain pending label compatibility."
            )


def _require_counts(accounting: Phase8EligibilityAccounting) -> None:
    if accounting.case_count != len(accounting.cases):
        raise Phase8EligibilityValidationError("case_count must equal len(cases).")
    if accounting.case_count != PHASE8_IRCADB_CASE_COUNT:
        raise Phase8EligibilityValidationError("Phase 8 accounting requires exactly 20 cases.")
    ids = [case.anonymous_case_id for case in accounting.cases]
    if len(set(ids)) != len(ids):
        raise Phase8EligibilityValidationError("cases must not duplicate anonymous IDs.")
    expected_ids = [f"ext-ircadb-{index:03d}" for index in range(1, PHASE8_IRCADB_CASE_COUNT + 1)]
    if ids != expected_ids:
        raise Phase8EligibilityValidationError(
            "cases must contain ext-ircadb-001 through ext-ircadb-020 exactly once."
        )
    expected_counts = {
        "deferred_count": sum(1 for case in accounting.cases if case.deferred),
        "discovered_count": sum(1 for case in accounting.cases if case.discovered),
        "evaluation_eligible_count": sum(
            1 for case in accounting.cases if case.evaluation_eligible
        ),
        "excluded_count": sum(1 for case in accounting.cases if case.excluded),
        "image_qa_eligible_count": sum(1 for case in accounting.cases if case.image_qa_eligible),
        "image_readable_count": sum(1 for case in accounting.cases if case.image_readable),
        "inference_eligible_count": sum(1 for case in accounting.cases if case.inference_eligible),
        "label_compatibility_pending_count": sum(
            1 for case in accounting.cases if case.label_compatibility_status == "pending"
        ),
    }
    for field_name, expected in expected_counts.items():
        if getattr(accounting, field_name) != expected:
            raise Phase8EligibilityValidationError(f"{field_name} does not reconcile.")


def _require_unchanged_image_accounting(
    original: Phase8EligibilityCase,
    updated: Phase8EligibilityCase,
) -> None:
    fields = (
        "anonymous_case_id",
        "discovered",
        "image_readable",
        "image_qa_eligible",
        "inference_eligible",
    )
    for field_name in fields:
        if getattr(original, field_name) != getattr(updated, field_name):
            raise Phase8EligibilityValidationError(
                f"post-label compatibility cannot change {field_name}."
            )


def _json_to_mapping(data: bytes | str) -> MappingLike:
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Phase8EligibilitySerializationError("invalid eligibility JSON.") from exc
    if not isinstance(decoded, Mapping):
        raise Phase8EligibilitySerializationError("eligibility JSON root must be an object.")
    return cast(MappingLike, decoded)


def _expect_mapping(value: object, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase8EligibilitySerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase8EligibilitySerializationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name)


def _expect_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase8EligibilitySerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase8EligibilitySerializationError(f"{field_name} must be an integer.")
    return value


def _expect_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Phase8EligibilitySerializationError(f"{field_name} must be a list.")
    return tuple(_expect_string(item, f"{field_name}[{index}]") for index, item in enumerate(value))


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
        raise Phase8EligibilitySerializationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _require_exact(value: str, expected: str, field_name: str) -> None:
    if value != expected:
        raise Phase8EligibilityValidationError(f"{field_name} must be {expected!r}.")


def _require_allowed(value: str, allowed: frozenset[str], field_name: str) -> None:
    if value not in allowed:
        raise Phase8EligibilityValidationError(f"{field_name} must be one of {sorted(allowed)!r}.")


def _require_sha256(value: str, field_name: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise Phase8EligibilityValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_anonymous_case_id(value: str) -> None:
    if _ANONYMOUS_CASE_ID_RE.fullmatch(value) is None:
        raise Phase8EligibilityValidationError(
            "anonymous_case_id must use the ext-ircadb-NNN format."
        )


def _require_reason_codes(
    values: tuple[str, ...], allowed: frozenset[str], field_name: str
) -> None:
    if len(values) != len(set(values)):
        raise Phase8EligibilityValidationError(f"{field_name} must not contain duplicates.")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise Phase8EligibilityValidationError(f"{field_name} contains unknown codes: {unknown!r}.")


_ELIGIBILITY_CASE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "anonymous_case_id",
        "discovered",
        "image_readable",
        "image_qa_eligible",
        "inference_eligible",
        "label_compatibility_status",
        "label_compatibility_reason_codes",
        "evaluation_eligible",
        "excluded",
        "deferred",
        "reason_codes",
    }
)
_PHASE8_EVALUATION_REQUIREMENTS: Final[tuple[str, ...]] = (
    "discovered",
    "image_readable",
    "image_qa_eligible",
    "inference_eligible",
    "post_label_access_compatible",
)
_ELIGIBILITY_POLICY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "policy_hash",
        "accounting_states",
        "evaluation_requires",
        "label_compatibility_statuses",
        "post_label_access_rule",
        "reason_codes",
    }
)
_ELIGIBILITY_ACCOUNTING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "accounting_hash",
        "accounting_state",
        "external_image_manifest_hash",
        "preregistered_policy_hash",
        "label_access_ledger_hash",
        "case_count",
        "discovered_count",
        "image_readable_count",
        "image_qa_eligible_count",
        "inference_eligible_count",
        "label_compatibility_pending_count",
        "evaluation_eligible_count",
        "excluded_count",
        "deferred_count",
        "cases",
    }
)


def _default_phase8_eligibility_policy_identity_payload() -> dict[str, JsonValue]:
    return {
        "accounting_states": list(sorted(PHASE8_ELIGIBILITY_ACCOUNTING_STATES)),
        "evaluation_requires": list(_PHASE8_EVALUATION_REQUIREMENTS),
        "label_compatibility_statuses": list(sorted(PHASE8_LABEL_COMPATIBILITY_STATUSES)),
        "post_label_access_rule": (
            "evaluation_eligible may become true only after compatible label accounting"
        ),
        "reason_codes": list(sorted(PHASE8_ELIGIBILITY_REASON_CODES)),
        "schema_name": PHASE8_ELIGIBILITY_POLICY_SCHEMA_NAME,
        "schema_version": PHASE8_ELIGIBILITY_POLICY_SCHEMA_VERSION,
    }


__all__ = [
    "PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_NAME",
    "PHASE8_ELIGIBILITY_ACCOUNTING_SCHEMA_VERSION",
    "PHASE8_ELIGIBILITY_CASE_SCHEMA_NAME",
    "PHASE8_ELIGIBILITY_CASE_SCHEMA_VERSION",
    "PHASE8_ELIGIBILITY_POLICY_SCHEMA_NAME",
    "PHASE8_ELIGIBILITY_POLICY_SCHEMA_VERSION",
    "PHASE8_ELIGIBILITY_REASON_CODES",
    "PHASE8_LABEL_COMPATIBILITY_REASON_CODES",
    "Phase8EligibilityAccounting",
    "Phase8EligibilityCase",
    "Phase8EligibilityHashError",
    "Phase8EligibilitySerializationError",
    "Phase8EligibilityValidationError",
    "Phase8EligibilityPolicy",
    "build_phase8_eligibility_policy",
    "build_phase8_label_compatible_case",
    "build_phase8_post_label_eligibility_accounting",
    "build_phase8_prelabel_eligibility_accounting",
    "hash_phase8_eligibility_accounting",
    "hash_phase8_eligibility_policy",
    "phase8_eligibility_accounting_from_json",
    "phase8_eligibility_accounting_identity_payload",
    "phase8_eligibility_accounting_to_dict",
    "phase8_eligibility_accounting_to_json",
    "phase8_eligibility_case_to_dict",
    "phase8_eligibility_policy_from_json",
    "phase8_eligibility_policy_identity_payload",
    "phase8_eligibility_policy_payload",
    "phase8_eligibility_policy_to_dict",
    "phase8_eligibility_policy_to_json",
]
