"""Pure Phase 8 external-label leakage contracts and validators.

The contracts in this module intentionally perform no filesystem I/O and carry no timestamps,
hostnames, usernames, hardware details, runtime details, local paths, labels, predictions, metrics,
or scientific result values. They validate only frozen decision identities and label-access order.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, TypeAlias, TypeVar, cast

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json

PHASE8_FROZEN_DECISIONS_SCHEMA_NAME: Final[str] = "phase8_frozen_decisions"
PHASE8_FROZEN_DECISIONS_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME: Final[str] = "phase8_external_preregistration"
PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_NAME: Final[str] = "phase8_external_prediction_lock_record"
PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_NAME: Final[str] = "phase8_external_evaluation_access_event"
PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_DEVIATION_RECORD_SCHEMA_NAME: Final[str] = "phase8_deviation_record"
PHASE8_DEVIATION_RECORD_SCHEMA_VERSION: Final[str] = "v1"
EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_NAME: Final[str] = "external_label_access_ledger"
EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_NAME: Final[str] = "phase8_post_result_decision_lock"
PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_VERSION: Final[str] = "v1"

FROZEN_DECISION_FIELDS: Final[tuple[str, ...]] = (
    "preprocessing_hash",
    "support_policy_hash",
    "threshold_hash",
    "checkpoint_or_model_selection_hash",
    "label_mapping_policy_hash",
)
LABEL_ACCESS_KIND: Final[str] = "external_label_read"
EVALUATION_ACCESS_KIND: Final[str] = "external_evaluation_read"
SUPPORTED_ACCESS_KINDS: Final[tuple[str, ...]] = (
    LABEL_ACCESS_KIND,
    EVALUATION_ACCESS_KIND,
)

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,95}$")
_WINDOWS_ABSOLUTE_PATH_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]:[\\/]")

JsonObject: TypeAlias = dict[str, Any]
_T = TypeVar("_T")


class Phase8LeakageContractError(ValueError):
    """Base error for Phase 8 leakage contracts."""


class Phase8LeakageSerializationError(Phase8LeakageContractError):
    """Raised when leakage-contract JSON cannot be parsed safely."""


class Phase8LeakageValidationError(Phase8LeakageContractError):
    """Raised when a leakage contract violates Phase 8 invariants."""


class Phase8LeakageVersionError(Phase8LeakageContractError):
    """Raised when a leakage contract has an unsupported schema name or version."""


class Phase8LeakageHashError(Phase8LeakageContractError):
    """Raised when a leakage contract self-hash does not match its canonical content."""


@dataclass(frozen=True, slots=True)
class Phase8FrozenDecisions:
    """Immutable identities for decisions frozen before external evaluation."""

    schema_name: str
    schema_version: str
    preprocessing_hash: str
    support_policy_hash: str
    threshold_hash: str
    checkpoint_or_model_selection_hash: str
    label_mapping_policy_hash: str

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            PHASE8_FROZEN_DECISIONS_SCHEMA_NAME,
            PHASE8_FROZEN_DECISIONS_SCHEMA_VERSION,
        )
        _require_sha256(self.preprocessing_hash, "preprocessing_hash")
        _require_sha256(self.support_policy_hash, "support_policy_hash")
        _require_sha256(self.threshold_hash, "threshold_hash")
        _require_sha256(
            self.checkpoint_or_model_selection_hash,
            "checkpoint_or_model_selection_hash",
        )
        _require_sha256(self.label_mapping_policy_hash, "label_mapping_policy_hash")


@dataclass(frozen=True, slots=True)
class Phase8LeakagePreregistrationLock:
    """Minimal self-validating preregistration identity for leakage enforcement."""

    schema_name: str
    schema_version: str
    preregistration_hash: str
    decision_freeze_hash: str
    frozen_decisions: Phase8FrozenDecisions
    label_access_policy: str
    deviation_policy_hash: str
    preregistered_before_label_access: bool

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME,
            PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION,
        )
        _require_sha256(self.preregistration_hash, "preregistration_hash")
        _require_sha256(self.decision_freeze_hash, "decision_freeze_hash")
        _require_sha256(self.deviation_policy_hash, "deviation_policy_hash")
        _require_token(self.label_access_policy, "label_access_policy")
        if self.label_access_policy != "evaluation_only_after_prediction_lock":
            msg = "label_access_policy must be evaluation_only_after_prediction_lock"
            raise Phase8LeakageValidationError(msg)
        if not self.preregistered_before_label_access:
            msg = "preregistration must be frozen before external label access"
            raise Phase8LeakageValidationError(msg)
        _require_no_forbidden_identity_strings(_preregistration_payload(self))
        _require_self_hash(
            self.preregistration_hash,
            _preregistration_payload_without_hash(self),
            "preregistration_hash",
        )


@dataclass(frozen=True, slots=True)
class Phase8PredictionLockRecord:
    """Immutable record that external predictions are locked before label access."""

    schema_name: str
    schema_version: str
    prediction_lock_hash: str
    preregistration_hash: str
    decision_freeze_hash: str
    prediction_manifest_hash: str
    predictions_frozen: bool
    one_time_inference_complete: bool

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_NAME,
            PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_VERSION,
        )
        _require_sha256(self.prediction_lock_hash, "prediction_lock_hash")
        _require_sha256(self.preregistration_hash, "preregistration_hash")
        _require_sha256(self.decision_freeze_hash, "decision_freeze_hash")
        _require_sha256(self.prediction_manifest_hash, "prediction_manifest_hash")
        if not self.predictions_frozen:
            msg = "prediction lock requires predictions_frozen=True"
            raise Phase8LeakageValidationError(msg)
        if not self.one_time_inference_complete:
            msg = "prediction lock requires one_time_inference_complete=True"
            raise Phase8LeakageValidationError(msg)
        _require_no_forbidden_identity_strings(_prediction_lock_payload(self))
        _require_self_hash(
            self.prediction_lock_hash,
            _prediction_lock_payload_without_hash(self),
            "prediction_lock_hash",
        )


@dataclass(frozen=True, slots=True)
class Phase8ExternalEvaluationAccessEvent:
    """Versioned event authorizing one external-evaluation access operation."""

    schema_name: str
    schema_version: str
    access_event_hash: str
    access_event_id: str
    access_kind: str
    purpose: str
    preregistration_hash: str
    decision_freeze_hash: str
    prediction_lock_hash: str

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_NAME,
            PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_VERSION,
        )
        _require_sha256(self.access_event_hash, "access_event_hash")
        _require_identifier(self.access_event_id, "access_event_id")
        if self.access_kind not in SUPPORTED_ACCESS_KINDS:
            msg = f"unsupported access_kind: {self.access_kind!r}"
            raise Phase8LeakageValidationError(msg)
        _require_token(self.purpose, "purpose")
        _require_sha256(self.preregistration_hash, "preregistration_hash")
        _require_sha256(self.decision_freeze_hash, "decision_freeze_hash")
        _require_sha256(self.prediction_lock_hash, "prediction_lock_hash")
        _require_no_forbidden_identity_strings(_access_event_payload(self))
        _require_self_hash(
            self.access_event_hash,
            _access_event_payload_without_hash(self),
            "access_event_hash",
        )


@dataclass(frozen=True, slots=True)
class Phase8DeviationRecord:
    """Explicit reported deviation record that cannot silently reopen frozen decisions."""

    schema_name: str
    schema_version: str
    deviation_hash: str
    deviation_id: str
    affected_field: str
    reason_code: str
    disposition: str
    reported: bool
    changes_frozen_decision: bool

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            PHASE8_DEVIATION_RECORD_SCHEMA_NAME,
            PHASE8_DEVIATION_RECORD_SCHEMA_VERSION,
        )
        _require_sha256(self.deviation_hash, "deviation_hash")
        _require_identifier(self.deviation_id, "deviation_id")
        if self.affected_field not in (*FROZEN_DECISION_FIELDS, "none"):
            msg = f"unsupported affected_field: {self.affected_field!r}"
            raise Phase8LeakageValidationError(msg)
        _require_token(self.reason_code, "reason_code")
        if self.disposition not in ("reported_no_decision_change", "rejected_reopen"):
            msg = "disposition must report no decision change or reject reopening"
            raise Phase8LeakageValidationError(msg)
        if not self.reported:
            msg = "deviations must be explicitly reported"
            raise Phase8LeakageValidationError(msg)
        if self.changes_frozen_decision:
            msg = "deviations cannot change frozen decisions"
            raise Phase8LeakageValidationError(msg)
        _require_no_forbidden_identity_strings(_deviation_payload(self))
        _require_self_hash(
            self.deviation_hash,
            _deviation_payload_without_hash(self),
            "deviation_hash",
        )


@dataclass(frozen=True, slots=True)
class ExternalLabelAccessLedger:
    """Self-validating ledger for the first external label access boundary."""

    schema_name: str
    schema_version: str
    ledger_hash: str
    preregistration_hash: str
    decision_freeze_hash: str
    prediction_lock_hash: str
    label_access_permitted: bool
    access_events: tuple[Phase8ExternalEvaluationAccessEvent, ...] = field(default_factory=tuple)
    deviations: tuple[Phase8DeviationRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_NAME,
            EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_VERSION,
        )
        _require_sha256(self.ledger_hash, "ledger_hash")
        _require_sha256(self.preregistration_hash, "preregistration_hash")
        _require_sha256(self.decision_freeze_hash, "decision_freeze_hash")
        _require_sha256(self.prediction_lock_hash, "prediction_lock_hash")
        if not self.label_access_permitted:
            msg = "label access ledger must explicitly permit evaluation-only label access"
            raise Phase8LeakageValidationError(msg)
        if not self.access_events:
            msg = "label access ledger requires at least one access event"
            raise Phase8LeakageValidationError(msg)
        _require_sorted_unique(tuple(event.access_event_id for event in self.access_events))
        _require_sorted_unique(tuple(item.deviation_id for item in self.deviations))
        for event in self.access_events:
            if event.preregistration_hash != self.preregistration_hash:
                msg = "access event preregistration_hash must match ledger"
                raise Phase8LeakageValidationError(msg)
            if event.decision_freeze_hash != self.decision_freeze_hash:
                msg = "access event decision_freeze_hash must match ledger"
                raise Phase8LeakageValidationError(msg)
            if event.prediction_lock_hash != self.prediction_lock_hash:
                msg = "access event prediction_lock_hash must match ledger"
                raise Phase8LeakageValidationError(msg)
        if not any(event.access_kind == LABEL_ACCESS_KIND for event in self.access_events):
            msg = "label access ledger requires an external_label_read event"
            raise Phase8LeakageValidationError(msg)
        _require_no_forbidden_identity_strings(_ledger_payload(self))
        _require_self_hash(self.ledger_hash, _ledger_payload_without_hash(self), "ledger_hash")


@dataclass(frozen=True, slots=True)
class Phase8PostResultDecisionLock:
    """Post-result proof that external result values did not mutate frozen decisions."""

    schema_name: str
    schema_version: str
    decision_lock_hash: str
    preregistration_hash: str
    decision_freeze_hash: str
    prediction_lock_hash: str
    ledger_hash: str
    frozen_decisions: Phase8FrozenDecisions
    external_result_values_used_for_decisions: bool
    frozen_decision_reopen_attempted: bool
    deviations: tuple[Phase8DeviationRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_NAME,
            PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_VERSION,
        )
        _require_sha256(self.decision_lock_hash, "decision_lock_hash")
        _require_sha256(self.preregistration_hash, "preregistration_hash")
        _require_sha256(self.decision_freeze_hash, "decision_freeze_hash")
        _require_sha256(self.prediction_lock_hash, "prediction_lock_hash")
        _require_sha256(self.ledger_hash, "ledger_hash")
        _require_sorted_unique(tuple(item.deviation_id for item in self.deviations))
        if self.external_result_values_used_for_decisions:
            msg = "external result values cannot modify frozen decisions"
            raise Phase8LeakageValidationError(msg)
        if self.frozen_decision_reopen_attempted and not self.deviations:
            msg = "silent reopening of frozen decisions is rejected; record an explicit deviation"
            raise Phase8LeakageValidationError(msg)
        _require_no_forbidden_identity_strings(_post_result_lock_payload(self))
        _require_self_hash(
            self.decision_lock_hash,
            _post_result_lock_payload_without_hash(self),
            "decision_lock_hash",
        )


def build_phase8_leakage_preregistration_lock(
    *,
    decision_freeze_hash: str,
    frozen_decisions: Phase8FrozenDecisions,
    deviation_policy_hash: str,
    label_access_policy: str = "evaluation_only_after_prediction_lock",
    preregistered_before_label_access: bool = True,
) -> Phase8LeakagePreregistrationLock:
    """Build a deterministic preregistration record with its self-hash."""

    payload = {
        "decision_freeze_hash": decision_freeze_hash,
        "deviation_policy_hash": deviation_policy_hash,
        "label_access_policy": label_access_policy,
        "preregistered_before_label_access": preregistered_before_label_access,
        "schema_name": PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME,
        "schema_version": PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION,
    }
    hash_payload = {"frozen_decisions": _frozen_decisions_payload(frozen_decisions), **payload}
    return Phase8LeakagePreregistrationLock(
        preregistration_hash=sha256_json(hash_payload),
        schema_name=PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME,
        schema_version=PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION,
        decision_freeze_hash=decision_freeze_hash,
        frozen_decisions=frozen_decisions,
        label_access_policy=label_access_policy,
        deviation_policy_hash=deviation_policy_hash,
        preregistered_before_label_access=preregistered_before_label_access,
    )


def build_phase8_prediction_lock_record(
    *,
    preregistration_hash: str,
    decision_freeze_hash: str,
    prediction_manifest_hash: str,
    predictions_frozen: bool = True,
    one_time_inference_complete: bool = True,
) -> Phase8PredictionLockRecord:
    """Build a deterministic prediction-lock record with its self-hash."""

    payload = {
        "decision_freeze_hash": decision_freeze_hash,
        "one_time_inference_complete": one_time_inference_complete,
        "prediction_manifest_hash": prediction_manifest_hash,
        "predictions_frozen": predictions_frozen,
        "preregistration_hash": preregistration_hash,
        "schema_name": PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_NAME,
        "schema_version": PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_VERSION,
    }
    return Phase8PredictionLockRecord(
        schema_name=PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_NAME,
        schema_version=PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_VERSION,
        prediction_lock_hash=sha256_json(payload),
        preregistration_hash=preregistration_hash,
        decision_freeze_hash=decision_freeze_hash,
        prediction_manifest_hash=prediction_manifest_hash,
        predictions_frozen=predictions_frozen,
        one_time_inference_complete=one_time_inference_complete,
    )


def build_phase8_external_evaluation_access_event(
    *,
    access_event_id: str,
    access_kind: str,
    purpose: str,
    preregistration_hash: str,
    decision_freeze_hash: str,
    prediction_lock_hash: str,
) -> Phase8ExternalEvaluationAccessEvent:
    """Build a deterministic external-evaluation access event with its self-hash."""

    payload = {
        "access_event_id": access_event_id,
        "access_kind": access_kind,
        "decision_freeze_hash": decision_freeze_hash,
        "prediction_lock_hash": prediction_lock_hash,
        "preregistration_hash": preregistration_hash,
        "purpose": purpose,
        "schema_name": PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_NAME,
        "schema_version": PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_VERSION,
    }
    return Phase8ExternalEvaluationAccessEvent(
        schema_name=PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_NAME,
        schema_version=PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_VERSION,
        access_event_hash=sha256_json(payload),
        access_event_id=access_event_id,
        access_kind=access_kind,
        purpose=purpose,
        preregistration_hash=preregistration_hash,
        decision_freeze_hash=decision_freeze_hash,
        prediction_lock_hash=prediction_lock_hash,
    )


def build_phase8_deviation_record(
    *,
    deviation_id: str,
    affected_field: str,
    reason_code: str,
    disposition: str = "reported_no_decision_change",
    reported: bool = True,
    changes_frozen_decision: bool = False,
) -> Phase8DeviationRecord:
    """Build a deterministic explicit deviation record with its self-hash."""

    payload = {
        "affected_field": affected_field,
        "changes_frozen_decision": changes_frozen_decision,
        "deviation_id": deviation_id,
        "disposition": disposition,
        "reason_code": reason_code,
        "reported": reported,
        "schema_name": PHASE8_DEVIATION_RECORD_SCHEMA_NAME,
        "schema_version": PHASE8_DEVIATION_RECORD_SCHEMA_VERSION,
    }
    return Phase8DeviationRecord(
        schema_name=PHASE8_DEVIATION_RECORD_SCHEMA_NAME,
        schema_version=PHASE8_DEVIATION_RECORD_SCHEMA_VERSION,
        deviation_hash=sha256_json(payload),
        deviation_id=deviation_id,
        affected_field=affected_field,
        reason_code=reason_code,
        disposition=disposition,
        reported=reported,
        changes_frozen_decision=changes_frozen_decision,
    )


def build_external_label_access_ledger(
    *,
    preregistration_hash: str,
    decision_freeze_hash: str,
    prediction_lock_hash: str,
    access_events: Sequence[Phase8ExternalEvaluationAccessEvent],
    deviations: Sequence[Phase8DeviationRecord] = (),
    label_access_permitted: bool = True,
) -> ExternalLabelAccessLedger:
    """Build a deterministic external-label access ledger with its self-hash."""

    sorted_events = tuple(sorted(access_events, key=lambda item: item.access_event_id))
    sorted_deviations = tuple(sorted(deviations, key=lambda item: item.deviation_id))
    payload = {
        "access_events": [_access_event_payload(item) for item in sorted_events],
        "decision_freeze_hash": decision_freeze_hash,
        "deviations": [_deviation_payload(item) for item in sorted_deviations],
        "label_access_permitted": label_access_permitted,
        "prediction_lock_hash": prediction_lock_hash,
        "preregistration_hash": preregistration_hash,
        "schema_name": EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_NAME,
        "schema_version": EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_VERSION,
    }
    return ExternalLabelAccessLedger(
        schema_name=EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_NAME,
        schema_version=EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_VERSION,
        ledger_hash=sha256_json(payload),
        preregistration_hash=preregistration_hash,
        decision_freeze_hash=decision_freeze_hash,
        prediction_lock_hash=prediction_lock_hash,
        label_access_permitted=label_access_permitted,
        access_events=sorted_events,
        deviations=sorted_deviations,
    )


def build_phase8_post_result_decision_lock(
    *,
    preregistration_hash: str,
    decision_freeze_hash: str,
    prediction_lock_hash: str,
    ledger_hash: str,
    frozen_decisions: Phase8FrozenDecisions,
    deviations: Sequence[Phase8DeviationRecord] = (),
    external_result_values_used_for_decisions: bool = False,
    frozen_decision_reopen_attempted: bool = False,
) -> Phase8PostResultDecisionLock:
    """Build a deterministic post-result decision lock with its self-hash."""

    sorted_deviations = tuple(sorted(deviations, key=lambda item: item.deviation_id))
    payload = {
        "decision_freeze_hash": decision_freeze_hash,
        "deviations": [_deviation_payload(item) for item in sorted_deviations],
        "external_result_values_used_for_decisions": external_result_values_used_for_decisions,
        "frozen_decision_reopen_attempted": frozen_decision_reopen_attempted,
        "ledger_hash": ledger_hash,
        "prediction_lock_hash": prediction_lock_hash,
        "preregistration_hash": preregistration_hash,
        "schema_name": PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_NAME,
        "schema_version": PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_VERSION,
    }
    hash_payload = {"frozen_decisions": _frozen_decisions_payload(frozen_decisions), **payload}
    return Phase8PostResultDecisionLock(
        schema_name=PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_NAME,
        schema_version=PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_VERSION,
        decision_lock_hash=sha256_json(hash_payload),
        preregistration_hash=preregistration_hash,
        decision_freeze_hash=decision_freeze_hash,
        prediction_lock_hash=prediction_lock_hash,
        ledger_hash=ledger_hash,
        frozen_decisions=frozen_decisions,
        external_result_values_used_for_decisions=external_result_values_used_for_decisions,
        frozen_decision_reopen_attempted=frozen_decision_reopen_attempted,
        deviations=sorted_deviations,
    )


def authorize_external_label_access(
    *,
    preregistration: Phase8LeakagePreregistrationLock | None,
    prediction_lock: Phase8PredictionLockRecord | None,
    ledger: ExternalLabelAccessLedger,
) -> None:
    """Fail closed unless label access follows valid preregistration and prediction lock."""

    if preregistration is None:
        msg = "external label access requires valid preregistration"
        raise Phase8LeakageValidationError(msg)
    if prediction_lock is None:
        msg = "external label access requires prediction lock"
        raise Phase8LeakageValidationError(msg)
    if ledger.preregistration_hash != preregistration.preregistration_hash:
        msg = "label ledger preregistration_hash does not match preregistration"
        raise Phase8LeakageValidationError(msg)
    if ledger.decision_freeze_hash != preregistration.decision_freeze_hash:
        msg = "label ledger decision_freeze_hash does not match preregistration"
        raise Phase8LeakageValidationError(msg)
    if prediction_lock.preregistration_hash != preregistration.preregistration_hash:
        msg = "prediction lock preregistration_hash does not match preregistration"
        raise Phase8LeakageValidationError(msg)
    if prediction_lock.decision_freeze_hash != preregistration.decision_freeze_hash:
        msg = "prediction lock decision_freeze_hash does not match preregistration"
        raise Phase8LeakageValidationError(msg)
    if ledger.prediction_lock_hash != prediction_lock.prediction_lock_hash:
        msg = "label ledger prediction_lock_hash does not match prediction lock"
        raise Phase8LeakageValidationError(msg)


def validate_frozen_decisions_unchanged(
    preregistration: Phase8LeakagePreregistrationLock,
    proposed_decisions: Phase8FrozenDecisions,
) -> None:
    """Reject post-preregistration changes to leakage-critical frozen decisions."""

    for field_name in FROZEN_DECISION_FIELDS:
        if getattr(preregistration.frozen_decisions, field_name) != getattr(
            proposed_decisions,
            field_name,
        ):
            msg = f"{field_name} cannot change after preregistration"
            raise Phase8LeakageValidationError(msg)


def validate_post_result_decision_lock(
    *,
    preregistration: Phase8LeakagePreregistrationLock,
    prediction_lock: Phase8PredictionLockRecord,
    ledger: ExternalLabelAccessLedger,
    post_result_lock: Phase8PostResultDecisionLock,
) -> None:
    """Reject any post-result decision lock that changes frozen identities."""

    authorize_external_label_access(
        preregistration=preregistration,
        prediction_lock=prediction_lock,
        ledger=ledger,
    )
    if post_result_lock.preregistration_hash != preregistration.preregistration_hash:
        msg = "post-result lock preregistration_hash does not match preregistration"
        raise Phase8LeakageValidationError(msg)
    if post_result_lock.decision_freeze_hash != preregistration.decision_freeze_hash:
        msg = "post-result lock decision_freeze_hash does not match preregistration"
        raise Phase8LeakageValidationError(msg)
    if post_result_lock.prediction_lock_hash != prediction_lock.prediction_lock_hash:
        msg = "post-result lock prediction_lock_hash does not match prediction lock"
        raise Phase8LeakageValidationError(msg)
    if post_result_lock.ledger_hash != ledger.ledger_hash:
        msg = "post-result lock ledger_hash does not match label-access ledger"
        raise Phase8LeakageValidationError(msg)
    validate_frozen_decisions_unchanged(preregistration, post_result_lock.frozen_decisions)
    if post_result_lock.frozen_decision_reopen_attempted and not post_result_lock.deviations:
        msg = "silent reopening of frozen decisions is rejected"
        raise Phase8LeakageValidationError(msg)


def require_explicit_reported_deviation(
    deviations: Sequence[Phase8DeviationRecord],
    *,
    deviation_required: bool,
) -> None:
    """Require deviations to be explicit, versioned, hashed, reported records."""

    if deviation_required and not deviations:
        msg = "required preregistration deviation is missing"
        raise Phase8LeakageValidationError(msg)
    for deviation in deviations:
        if not deviation.reported:
            msg = "deviations must be reported"
            raise Phase8LeakageValidationError(msg)
        _require_self_hash(
            deviation.deviation_hash,
            _deviation_payload_without_hash(deviation),
            "deviation_hash",
        )


def phase8_leakage_artifact_to_json(artifact: object) -> bytes:
    """Serialize a Phase 8 leakage contract deterministically."""

    return canonical_json_bytes(phase8_leakage_artifact_to_dict(artifact)) + b"\n"


def phase8_leakage_artifact_to_dict(artifact: object) -> JsonObject:
    """Return a deterministic JSON-compatible dictionary for a leakage contract."""

    if isinstance(artifact, Phase8FrozenDecisions):
        return _frozen_decisions_payload(artifact)
    if isinstance(artifact, Phase8LeakagePreregistrationLock):
        return _preregistration_payload(artifact)
    if isinstance(artifact, Phase8PredictionLockRecord):
        return _prediction_lock_payload(artifact)
    if isinstance(artifact, Phase8ExternalEvaluationAccessEvent):
        return _access_event_payload(artifact)
    if isinstance(artifact, Phase8DeviationRecord):
        return _deviation_payload(artifact)
    if isinstance(artifact, ExternalLabelAccessLedger):
        return _ledger_payload(artifact)
    if isinstance(artifact, Phase8PostResultDecisionLock):
        return _post_result_lock_payload(artifact)
    msg = f"unsupported Phase 8 leakage artifact: {type(artifact).__name__}"
    raise Phase8LeakageSerializationError(msg)


def phase8_leakage_artifact_from_json(
    payload: bytes | str,
    expected_type: type[_T] | None = None,
) -> object | _T:
    """Parse and validate a Phase 8 leakage contract JSON payload."""

    parsed = _parse_json_object(payload)
    schema_name = _expect_string(parsed.get("schema_name"), "schema_name")
    match schema_name:
        case "phase8_frozen_decisions":
            artifact: object = _frozen_decisions_from_mapping(parsed)
        case "phase8_external_preregistration":
            artifact = _preregistration_from_mapping(parsed)
        case "phase8_external_prediction_lock_record":
            artifact = _prediction_lock_from_mapping(parsed)
        case "phase8_external_evaluation_access_event":
            artifact = _access_event_from_mapping(parsed)
        case "phase8_deviation_record":
            artifact = _deviation_from_mapping(parsed)
        case "external_label_access_ledger":
            artifact = _ledger_from_mapping(parsed)
        case "phase8_post_result_decision_lock":
            artifact = _post_result_lock_from_mapping(parsed)
        case _:
            msg = f"unsupported Phase 8 leakage schema_name: {schema_name!r}"
            raise Phase8LeakageVersionError(msg)
    if expected_type is not None and not isinstance(artifact, expected_type):
        msg = f"expected {expected_type.__name__}, got {type(artifact).__name__}"
        raise Phase8LeakageSerializationError(msg)
    return cast(_T, artifact) if expected_type is not None else artifact


def _frozen_decisions_payload(artifact: Phase8FrozenDecisions) -> JsonObject:
    return {
        "checkpoint_or_model_selection_hash": artifact.checkpoint_or_model_selection_hash,
        "label_mapping_policy_hash": artifact.label_mapping_policy_hash,
        "preprocessing_hash": artifact.preprocessing_hash,
        "schema_name": artifact.schema_name,
        "schema_version": artifact.schema_version,
        "support_policy_hash": artifact.support_policy_hash,
        "threshold_hash": artifact.threshold_hash,
    }


def _preregistration_payload(artifact: Phase8LeakagePreregistrationLock) -> JsonObject:
    payload = _preregistration_payload_without_hash(artifact)
    payload["preregistration_hash"] = artifact.preregistration_hash
    return payload


def _preregistration_payload_without_hash(artifact: Phase8LeakagePreregistrationLock) -> JsonObject:
    return {
        "decision_freeze_hash": artifact.decision_freeze_hash,
        "deviation_policy_hash": artifact.deviation_policy_hash,
        "frozen_decisions": _frozen_decisions_payload(artifact.frozen_decisions),
        "label_access_policy": artifact.label_access_policy,
        "preregistered_before_label_access": artifact.preregistered_before_label_access,
        "schema_name": artifact.schema_name,
        "schema_version": artifact.schema_version,
    }


def _prediction_lock_payload(artifact: Phase8PredictionLockRecord) -> JsonObject:
    payload = _prediction_lock_payload_without_hash(artifact)
    payload["prediction_lock_hash"] = artifact.prediction_lock_hash
    return payload


def _prediction_lock_payload_without_hash(artifact: Phase8PredictionLockRecord) -> JsonObject:
    return {
        "decision_freeze_hash": artifact.decision_freeze_hash,
        "one_time_inference_complete": artifact.one_time_inference_complete,
        "prediction_manifest_hash": artifact.prediction_manifest_hash,
        "predictions_frozen": artifact.predictions_frozen,
        "preregistration_hash": artifact.preregistration_hash,
        "schema_name": artifact.schema_name,
        "schema_version": artifact.schema_version,
    }


def _access_event_payload(artifact: Phase8ExternalEvaluationAccessEvent) -> JsonObject:
    payload = _access_event_payload_without_hash(artifact)
    payload["access_event_hash"] = artifact.access_event_hash
    return payload


def _access_event_payload_without_hash(
    artifact: Phase8ExternalEvaluationAccessEvent,
) -> JsonObject:
    return {
        "access_event_id": artifact.access_event_id,
        "access_kind": artifact.access_kind,
        "decision_freeze_hash": artifact.decision_freeze_hash,
        "prediction_lock_hash": artifact.prediction_lock_hash,
        "preregistration_hash": artifact.preregistration_hash,
        "purpose": artifact.purpose,
        "schema_name": artifact.schema_name,
        "schema_version": artifact.schema_version,
    }


def _deviation_payload(artifact: Phase8DeviationRecord) -> JsonObject:
    payload = _deviation_payload_without_hash(artifact)
    payload["deviation_hash"] = artifact.deviation_hash
    return payload


def _deviation_payload_without_hash(artifact: Phase8DeviationRecord) -> JsonObject:
    return {
        "affected_field": artifact.affected_field,
        "changes_frozen_decision": artifact.changes_frozen_decision,
        "deviation_id": artifact.deviation_id,
        "disposition": artifact.disposition,
        "reason_code": artifact.reason_code,
        "reported": artifact.reported,
        "schema_name": artifact.schema_name,
        "schema_version": artifact.schema_version,
    }


def _ledger_payload(artifact: ExternalLabelAccessLedger) -> JsonObject:
    payload = _ledger_payload_without_hash(artifact)
    payload["ledger_hash"] = artifact.ledger_hash
    return payload


def _ledger_payload_without_hash(artifact: ExternalLabelAccessLedger) -> JsonObject:
    return {
        "access_events": [_access_event_payload(event) for event in artifact.access_events],
        "decision_freeze_hash": artifact.decision_freeze_hash,
        "deviations": [_deviation_payload(item) for item in artifact.deviations],
        "label_access_permitted": artifact.label_access_permitted,
        "prediction_lock_hash": artifact.prediction_lock_hash,
        "preregistration_hash": artifact.preregistration_hash,
        "schema_name": artifact.schema_name,
        "schema_version": artifact.schema_version,
    }


def _post_result_lock_payload(artifact: Phase8PostResultDecisionLock) -> JsonObject:
    payload = _post_result_lock_payload_without_hash(artifact)
    payload["decision_lock_hash"] = artifact.decision_lock_hash
    return payload


def _post_result_lock_payload_without_hash(artifact: Phase8PostResultDecisionLock) -> JsonObject:
    return {
        "decision_freeze_hash": artifact.decision_freeze_hash,
        "deviations": [_deviation_payload(item) for item in artifact.deviations],
        "external_result_values_used_for_decisions": (
            artifact.external_result_values_used_for_decisions
        ),
        "frozen_decision_reopen_attempted": artifact.frozen_decision_reopen_attempted,
        "frozen_decisions": _frozen_decisions_payload(artifact.frozen_decisions),
        "ledger_hash": artifact.ledger_hash,
        "prediction_lock_hash": artifact.prediction_lock_hash,
        "preregistration_hash": artifact.preregistration_hash,
        "schema_name": artifact.schema_name,
        "schema_version": artifact.schema_version,
    }


def _frozen_decisions_from_mapping(payload: Mapping[str, Any]) -> Phase8FrozenDecisions:
    _require_exact_fields(
        payload,
        {
            "checkpoint_or_model_selection_hash",
            "label_mapping_policy_hash",
            "preprocessing_hash",
            "schema_name",
            "schema_version",
            "support_policy_hash",
            "threshold_hash",
        },
        "frozen decisions",
    )
    return Phase8FrozenDecisions(
        schema_name=_expect_string(payload["schema_name"], "schema_name"),
        schema_version=_expect_string(payload["schema_version"], "schema_version"),
        preprocessing_hash=_expect_string(payload["preprocessing_hash"], "preprocessing_hash"),
        support_policy_hash=_expect_string(payload["support_policy_hash"], "support_policy_hash"),
        threshold_hash=_expect_string(payload["threshold_hash"], "threshold_hash"),
        checkpoint_or_model_selection_hash=_expect_string(
            payload["checkpoint_or_model_selection_hash"],
            "checkpoint_or_model_selection_hash",
        ),
        label_mapping_policy_hash=_expect_string(
            payload["label_mapping_policy_hash"],
            "label_mapping_policy_hash",
        ),
    )


def _preregistration_from_mapping(payload: Mapping[str, Any]) -> Phase8LeakagePreregistrationLock:
    _require_exact_fields(
        payload,
        {
            "decision_freeze_hash",
            "deviation_policy_hash",
            "frozen_decisions",
            "label_access_policy",
            "preregistered_before_label_access",
            "preregistration_hash",
            "schema_name",
            "schema_version",
        },
        "external preregistration",
    )
    return Phase8LeakagePreregistrationLock(
        schema_name=_expect_string(payload["schema_name"], "schema_name"),
        schema_version=_expect_string(payload["schema_version"], "schema_version"),
        preregistration_hash=_expect_string(
            payload["preregistration_hash"],
            "preregistration_hash",
        ),
        decision_freeze_hash=_expect_string(
            payload["decision_freeze_hash"],
            "decision_freeze_hash",
        ),
        frozen_decisions=_frozen_decisions_from_mapping(
            _expect_mapping(payload["frozen_decisions"], "frozen_decisions")
        ),
        label_access_policy=_expect_string(payload["label_access_policy"], "label_access_policy"),
        deviation_policy_hash=_expect_string(
            payload["deviation_policy_hash"], "deviation_policy_hash"
        ),
        preregistered_before_label_access=_expect_bool(
            payload["preregistered_before_label_access"],
            "preregistered_before_label_access",
        ),
    )


def _prediction_lock_from_mapping(payload: Mapping[str, Any]) -> Phase8PredictionLockRecord:
    _require_exact_fields(
        payload,
        {
            "decision_freeze_hash",
            "one_time_inference_complete",
            "prediction_lock_hash",
            "prediction_manifest_hash",
            "predictions_frozen",
            "preregistration_hash",
            "schema_name",
            "schema_version",
        },
        "prediction lock record",
    )
    return Phase8PredictionLockRecord(
        schema_name=_expect_string(payload["schema_name"], "schema_name"),
        schema_version=_expect_string(payload["schema_version"], "schema_version"),
        prediction_lock_hash=_expect_string(
            payload["prediction_lock_hash"],
            "prediction_lock_hash",
        ),
        preregistration_hash=_expect_string(
            payload["preregistration_hash"],
            "preregistration_hash",
        ),
        decision_freeze_hash=_expect_string(
            payload["decision_freeze_hash"],
            "decision_freeze_hash",
        ),
        prediction_manifest_hash=_expect_string(
            payload["prediction_manifest_hash"],
            "prediction_manifest_hash",
        ),
        predictions_frozen=_expect_bool(payload["predictions_frozen"], "predictions_frozen"),
        one_time_inference_complete=_expect_bool(
            payload["one_time_inference_complete"],
            "one_time_inference_complete",
        ),
    )


def _access_event_from_mapping(payload: Mapping[str, Any]) -> Phase8ExternalEvaluationAccessEvent:
    _require_exact_fields(
        payload,
        {
            "access_event_hash",
            "access_event_id",
            "access_kind",
            "decision_freeze_hash",
            "prediction_lock_hash",
            "preregistration_hash",
            "purpose",
            "schema_name",
            "schema_version",
        },
        "external evaluation access event",
    )
    return Phase8ExternalEvaluationAccessEvent(
        schema_name=_expect_string(payload["schema_name"], "schema_name"),
        schema_version=_expect_string(payload["schema_version"], "schema_version"),
        access_event_hash=_expect_string(payload["access_event_hash"], "access_event_hash"),
        access_event_id=_expect_string(payload["access_event_id"], "access_event_id"),
        access_kind=_expect_string(payload["access_kind"], "access_kind"),
        purpose=_expect_string(payload["purpose"], "purpose"),
        preregistration_hash=_expect_string(
            payload["preregistration_hash"],
            "preregistration_hash",
        ),
        decision_freeze_hash=_expect_string(
            payload["decision_freeze_hash"],
            "decision_freeze_hash",
        ),
        prediction_lock_hash=_expect_string(
            payload["prediction_lock_hash"],
            "prediction_lock_hash",
        ),
    )


def _deviation_from_mapping(payload: Mapping[str, Any]) -> Phase8DeviationRecord:
    _require_exact_fields(
        payload,
        {
            "affected_field",
            "changes_frozen_decision",
            "deviation_hash",
            "deviation_id",
            "disposition",
            "reason_code",
            "reported",
            "schema_name",
            "schema_version",
        },
        "deviation record",
    )
    return Phase8DeviationRecord(
        schema_name=_expect_string(payload["schema_name"], "schema_name"),
        schema_version=_expect_string(payload["schema_version"], "schema_version"),
        deviation_hash=_expect_string(payload["deviation_hash"], "deviation_hash"),
        deviation_id=_expect_string(payload["deviation_id"], "deviation_id"),
        affected_field=_expect_string(payload["affected_field"], "affected_field"),
        reason_code=_expect_string(payload["reason_code"], "reason_code"),
        disposition=_expect_string(payload["disposition"], "disposition"),
        reported=_expect_bool(payload["reported"], "reported"),
        changes_frozen_decision=_expect_bool(
            payload["changes_frozen_decision"],
            "changes_frozen_decision",
        ),
    )


def _ledger_from_mapping(payload: Mapping[str, Any]) -> ExternalLabelAccessLedger:
    _require_exact_fields(
        payload,
        {
            "access_events",
            "decision_freeze_hash",
            "deviations",
            "label_access_permitted",
            "ledger_hash",
            "prediction_lock_hash",
            "preregistration_hash",
            "schema_name",
            "schema_version",
        },
        "external label access ledger",
    )
    access_events = tuple(
        _access_event_from_mapping(_expect_mapping(item, "access_events[]"))
        for item in _expect_list(payload["access_events"], "access_events")
    )
    deviations = tuple(
        _deviation_from_mapping(_expect_mapping(item, "deviations[]"))
        for item in _expect_list(payload["deviations"], "deviations")
    )
    return ExternalLabelAccessLedger(
        schema_name=_expect_string(payload["schema_name"], "schema_name"),
        schema_version=_expect_string(payload["schema_version"], "schema_version"),
        ledger_hash=_expect_string(payload["ledger_hash"], "ledger_hash"),
        preregistration_hash=_expect_string(
            payload["preregistration_hash"],
            "preregistration_hash",
        ),
        decision_freeze_hash=_expect_string(
            payload["decision_freeze_hash"],
            "decision_freeze_hash",
        ),
        prediction_lock_hash=_expect_string(
            payload["prediction_lock_hash"],
            "prediction_lock_hash",
        ),
        label_access_permitted=_expect_bool(
            payload["label_access_permitted"],
            "label_access_permitted",
        ),
        access_events=access_events,
        deviations=deviations,
    )


def _post_result_lock_from_mapping(payload: Mapping[str, Any]) -> Phase8PostResultDecisionLock:
    _require_exact_fields(
        payload,
        {
            "decision_freeze_hash",
            "decision_lock_hash",
            "deviations",
            "external_result_values_used_for_decisions",
            "frozen_decision_reopen_attempted",
            "frozen_decisions",
            "ledger_hash",
            "prediction_lock_hash",
            "preregistration_hash",
            "schema_name",
            "schema_version",
        },
        "post-result decision lock",
    )
    deviations = tuple(
        _deviation_from_mapping(_expect_mapping(item, "deviations[]"))
        for item in _expect_list(payload["deviations"], "deviations")
    )
    return Phase8PostResultDecisionLock(
        schema_name=_expect_string(payload["schema_name"], "schema_name"),
        schema_version=_expect_string(payload["schema_version"], "schema_version"),
        decision_lock_hash=_expect_string(payload["decision_lock_hash"], "decision_lock_hash"),
        preregistration_hash=_expect_string(
            payload["preregistration_hash"],
            "preregistration_hash",
        ),
        decision_freeze_hash=_expect_string(
            payload["decision_freeze_hash"],
            "decision_freeze_hash",
        ),
        prediction_lock_hash=_expect_string(
            payload["prediction_lock_hash"],
            "prediction_lock_hash",
        ),
        ledger_hash=_expect_string(payload["ledger_hash"], "ledger_hash"),
        frozen_decisions=_frozen_decisions_from_mapping(
            _expect_mapping(payload["frozen_decisions"], "frozen_decisions")
        ),
        external_result_values_used_for_decisions=_expect_bool(
            payload["external_result_values_used_for_decisions"],
            "external_result_values_used_for_decisions",
        ),
        frozen_decision_reopen_attempted=_expect_bool(
            payload["frozen_decision_reopen_attempted"],
            "frozen_decision_reopen_attempted",
        ),
        deviations=deviations,
    )


def _parse_json_object(payload: bytes | str) -> JsonObject:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        msg = "Phase 8 leakage contract JSON could not be parsed"
        raise Phase8LeakageSerializationError(msg) from exc
    if not isinstance(parsed, dict):
        msg = "Phase 8 leakage contract JSON must be an object"
        raise Phase8LeakageSerializationError(msg)
    return parsed


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(payload)
    extra = actual - expected
    missing = expected - actual
    if extra or missing:
        details: list[str] = []
        if extra:
            details.append(f"unknown={sorted(extra)!r}")
        if missing:
            details.append(f"missing={sorted(missing)!r}")
        msg = f"{context} fields mismatch: {', '.join(details)}"
        raise Phase8LeakageSerializationError(msg)


def _require_schema(
    actual_name: str,
    actual_version: str,
    expected_name: str,
    expected_version: str,
) -> None:
    if actual_name != expected_name:
        msg = f"unsupported schema_name: {actual_name!r}"
        raise Phase8LeakageVersionError(msg)
    if actual_version != expected_version:
        msg = f"unsupported schema_version: {actual_version!r}"
        raise Phase8LeakageVersionError(msg)


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        msg = f"{field_name} must be a lowercase SHA-256 hex digest"
        raise Phase8LeakageValidationError(msg)


def _require_identifier(value: str, field_name: str) -> None:
    if _IDENTIFIER_RE.fullmatch(value) is None:
        msg = f"{field_name} must be a lowercase stable identifier"
        raise Phase8LeakageValidationError(msg)


def _require_token(value: str, field_name: str) -> None:
    if _TOKEN_RE.fullmatch(value) is None:
        msg = f"{field_name} must be a lowercase token"
        raise Phase8LeakageValidationError(msg)


def _require_self_hash(
    actual_hash: str,
    payload_without_hash: Mapping[str, Any],
    field: str,
) -> None:
    expected_hash = sha256_json(payload_without_hash)
    if actual_hash != expected_hash:
        msg = f"{field} does not match canonical contract content"
        raise Phase8LeakageHashError(msg)


def _require_sorted_unique(values: tuple[str, ...]) -> None:
    if tuple(sorted(values)) != values:
        msg = "records must be sorted by stable identifier"
        raise Phase8LeakageValidationError(msg)
    if len(set(values)) != len(values):
        msg = "records must use unique stable identifiers"
        raise Phase8LeakageValidationError(msg)


def _expect_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        msg = f"{field_name} must be a string"
        raise Phase8LeakageSerializationError(msg)
    return value


def _expect_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{field_name} must be a boolean"
        raise Phase8LeakageSerializationError(msg)
    return value


def _expect_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        msg = f"{field_name} must be a JSON object"
        raise Phase8LeakageSerializationError(msg)
    return value


def _expect_list(value: object, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        msg = f"{field_name} must be a JSON array"
        raise Phase8LeakageSerializationError(msg)
    return value


def _require_no_forbidden_identity_strings(value: object) -> None:
    if isinstance(value, str):
        _require_safe_identity_string(value)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_safe_identity_string(str(key))
            _require_no_forbidden_identity_strings(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | str):
        for item in value:
            _require_no_forbidden_identity_strings(item)


def _require_safe_identity_string(value: str) -> None:
    if value.startswith("/") or value.startswith("~") or _WINDOWS_ABSOLUTE_PATH_RE.match(value):
        msg = "scientific identities must not contain local paths"
        raise Phase8LeakageValidationError(msg)
    forbidden_fragments = (
        "/volumes",
        "\\",
        "@",
        "localhost",
        ".local",
        "hostname",
        "username",
        "hardware",
        "runtime",
        "timestamp",
    )
    lowered = value.lower()
    if any(fragment in lowered for fragment in forbidden_fragments):
        msg = "scientific identities must not contain machine-specific metadata"
        raise Phase8LeakageValidationError(msg)


__all__ = [
    "EVALUATION_ACCESS_KIND",
    "EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_NAME",
    "EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_VERSION",
    "FROZEN_DECISION_FIELDS",
    "ExternalLabelAccessLedger",
    "LABEL_ACCESS_KIND",
    "PHASE8_DEVIATION_RECORD_SCHEMA_NAME",
    "PHASE8_DEVIATION_RECORD_SCHEMA_VERSION",
    "PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_NAME",
    "PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_VERSION",
    "PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_NAME",
    "PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_VERSION",
    "PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_NAME",
    "PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION",
    "PHASE8_FROZEN_DECISIONS_SCHEMA_NAME",
    "PHASE8_FROZEN_DECISIONS_SCHEMA_VERSION",
    "PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_NAME",
    "PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_VERSION",
    "Phase8DeviationRecord",
    "Phase8ExternalEvaluationAccessEvent",
    "Phase8LeakagePreregistrationLock",
    "Phase8FrozenDecisions",
    "Phase8LeakageContractError",
    "Phase8LeakageHashError",
    "Phase8LeakageSerializationError",
    "Phase8LeakageValidationError",
    "Phase8LeakageVersionError",
    "Phase8PostResultDecisionLock",
    "Phase8PredictionLockRecord",
    "build_external_label_access_ledger",
    "build_phase8_deviation_record",
    "build_phase8_external_evaluation_access_event",
    "build_phase8_leakage_preregistration_lock",
    "build_phase8_post_result_decision_lock",
    "build_phase8_prediction_lock_record",
    "authorize_external_label_access",
    "phase8_leakage_artifact_from_json",
    "phase8_leakage_artifact_to_dict",
    "phase8_leakage_artifact_to_json",
    "require_explicit_reported_deviation",
    "validate_frozen_decisions_unchanged",
    "validate_post_result_decision_lock",
]
