"""Unit tests for pure Phase 8 leakage contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, TypeVar, cast

import pytest

from protoem_ct.external.leakage import (
    EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_VERSION,
    FROZEN_DECISION_FIELDS,
    LABEL_ACCESS_KIND,
    PHASE8_DEVIATION_RECORD_SCHEMA_VERSION,
    PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_VERSION,
    PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_VERSION,
    PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION,
    PHASE8_FROZEN_DECISIONS_SCHEMA_NAME,
    PHASE8_FROZEN_DECISIONS_SCHEMA_VERSION,
    PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_VERSION,
    ExternalLabelAccessLedger,
    Phase8DeviationRecord,
    Phase8ExternalEvaluationAccessEvent,
    Phase8FrozenDecisions,
    Phase8LeakageHashError,
    Phase8LeakagePreregistrationLock,
    Phase8LeakageSerializationError,
    Phase8LeakageValidationError,
    Phase8PostResultDecisionLock,
    Phase8PredictionLockRecord,
    authorize_external_label_access,
    build_external_label_access_ledger,
    build_phase8_deviation_record,
    build_phase8_external_evaluation_access_event,
    build_phase8_leakage_preregistration_lock,
    build_phase8_post_result_decision_lock,
    build_phase8_prediction_lock_record,
    phase8_leakage_artifact_from_json,
    phase8_leakage_artifact_to_json,
    require_explicit_reported_deviation,
    validate_frozen_decisions_unchanged,
    validate_post_result_decision_lock,
)

HASH0 = "0" * 64
HASH1 = "1" * 64
HASH2 = "2" * 64
HASH3 = "3" * 64
HASH4 = "4" * 64
HASH5 = "5" * 64
HASH6 = "6" * 64
HASH7 = "7" * 64
HASH8 = "8" * 64

_T = TypeVar("_T")


def _decisions(**overrides: object) -> Phase8FrozenDecisions:
    decisions = Phase8FrozenDecisions(
        schema_name=PHASE8_FROZEN_DECISIONS_SCHEMA_NAME,
        schema_version=PHASE8_FROZEN_DECISIONS_SCHEMA_VERSION,
        preprocessing_hash=HASH0,
        support_policy_hash=HASH1,
        threshold_hash=HASH2,
        checkpoint_or_model_selection_hash=HASH3,
        label_mapping_policy_hash=HASH4,
    )
    return cast(Phase8FrozenDecisions, cast(Any, replace)(decisions, **overrides))


def _preregistration(
    *,
    decisions: Phase8FrozenDecisions | None = None,
    decision_freeze_hash: str = HASH5,
) -> Phase8LeakagePreregistrationLock:
    return build_phase8_leakage_preregistration_lock(
        decision_freeze_hash=decision_freeze_hash,
        frozen_decisions=decisions or _decisions(),
        deviation_policy_hash=HASH6,
    )


def _prediction_lock(
    preregistration: Phase8LeakagePreregistrationLock,
) -> Phase8PredictionLockRecord:
    return build_phase8_prediction_lock_record(
        preregistration_hash=preregistration.preregistration_hash,
        decision_freeze_hash=preregistration.decision_freeze_hash,
        prediction_manifest_hash=HASH7,
    )


def _event(
    preregistration: Phase8LeakagePreregistrationLock,
    prediction_lock: Phase8PredictionLockRecord,
) -> Phase8ExternalEvaluationAccessEvent:
    return build_phase8_external_evaluation_access_event(
        access_event_id="label_access_001",
        access_kind=LABEL_ACCESS_KIND,
        purpose="evaluation_only",
        preregistration_hash=preregistration.preregistration_hash,
        decision_freeze_hash=preregistration.decision_freeze_hash,
        prediction_lock_hash=prediction_lock.prediction_lock_hash,
    )


def _deviation(**overrides: object) -> Phase8DeviationRecord:
    return build_phase8_deviation_record(
        deviation_id=cast(str, overrides.pop("deviation_id", "deviation_001")),
        affected_field=cast(str, overrides.pop("affected_field", "none")),
        reason_code=cast(str, overrides.pop("reason_code", "documented_protocol_exception")),
        disposition=cast(str, overrides.pop("disposition", "reported_no_decision_change")),
        reported=cast(bool, overrides.pop("reported", True)),
        changes_frozen_decision=cast(bool, overrides.pop("changes_frozen_decision", False)),
    )


def _ledger(
    preregistration: Phase8LeakagePreregistrationLock,
    prediction_lock: Phase8PredictionLockRecord,
    *,
    deviations: tuple[Phase8DeviationRecord, ...] = (),
) -> ExternalLabelAccessLedger:
    return build_external_label_access_ledger(
        preregistration_hash=preregistration.preregistration_hash,
        decision_freeze_hash=preregistration.decision_freeze_hash,
        prediction_lock_hash=prediction_lock.prediction_lock_hash,
        access_events=(_event(preregistration, prediction_lock),),
        deviations=deviations,
    )


def _post_result_lock(
    preregistration: Phase8LeakagePreregistrationLock,
    prediction_lock: Phase8PredictionLockRecord,
    ledger: ExternalLabelAccessLedger,
    *,
    decisions: Phase8FrozenDecisions | None = None,
    deviations: tuple[Phase8DeviationRecord, ...] = (),
    external_result_values_used_for_decisions: bool = False,
    frozen_decision_reopen_attempted: bool = False,
) -> Phase8PostResultDecisionLock:
    return build_phase8_post_result_decision_lock(
        preregistration_hash=preregistration.preregistration_hash,
        decision_freeze_hash=preregistration.decision_freeze_hash,
        prediction_lock_hash=prediction_lock.prediction_lock_hash,
        ledger_hash=ledger.ledger_hash,
        frozen_decisions=decisions or preregistration.frozen_decisions,
        deviations=deviations,
        external_result_values_used_for_decisions=external_result_values_used_for_decisions,
        frozen_decision_reopen_attempted=frozen_decision_reopen_attempted,
    )


@pytest.mark.parametrize(
    ("artifact", "expected_type", "schema_version"),
    [
        (_decisions(), Phase8FrozenDecisions, PHASE8_FROZEN_DECISIONS_SCHEMA_VERSION),
        (
            _preregistration(),
            Phase8LeakagePreregistrationLock,
            PHASE8_EXTERNAL_PREREGISTRATION_SCHEMA_VERSION,
        ),
        (
            _prediction_lock(_preregistration()),
            Phase8PredictionLockRecord,
            PHASE8_EXTERNAL_PREDICTION_LOCK_SCHEMA_VERSION,
        ),
        (
            _event(_preregistration(), _prediction_lock(_preregistration())),
            Phase8ExternalEvaluationAccessEvent,
            PHASE8_EXTERNAL_ACCESS_EVENT_SCHEMA_VERSION,
        ),
        (_deviation(), Phase8DeviationRecord, PHASE8_DEVIATION_RECORD_SCHEMA_VERSION),
    ],
)
def test_contract_round_trip_and_canonical_stability(
    artifact: object,
    expected_type: type[_T],
    schema_version: str,
) -> None:
    encoded = phase8_leakage_artifact_to_json(artifact)
    parsed = phase8_leakage_artifact_from_json(encoded, expected_type)

    assert parsed == artifact
    assert phase8_leakage_artifact_to_json(parsed) == encoded
    assert json.loads(encoded)["schema_version"] == schema_version
    assert encoded.endswith(b"\n")


def test_ledger_and_post_result_lock_round_trip_and_canonical_stability() -> None:
    preregistration = _preregistration()
    prediction_lock = _prediction_lock(preregistration)
    deviation = _deviation()
    ledger = _ledger(preregistration, prediction_lock, deviations=(deviation,))
    post_result_lock = _post_result_lock(
        preregistration,
        prediction_lock,
        ledger,
        deviations=(deviation,),
        frozen_decision_reopen_attempted=True,
    )

    assert (
        phase8_leakage_artifact_from_json(
            phase8_leakage_artifact_to_json(ledger),
            ExternalLabelAccessLedger,
        )
        == ledger
    )
    assert (
        phase8_leakage_artifact_from_json(
            phase8_leakage_artifact_to_json(post_result_lock),
            Phase8PostResultDecisionLock,
        )
        == post_result_lock
    )
    assert ledger.schema_version == EXTERNAL_LABEL_ACCESS_LEDGER_SCHEMA_VERSION
    assert post_result_lock.schema_version == PHASE8_POST_RESULT_DECISION_LOCK_SCHEMA_VERSION


def test_unknown_field_and_bad_self_hash_rejection() -> None:
    preregistration = _preregistration()
    payload = json.loads(phase8_leakage_artifact_to_json(preregistration))
    payload["unexpected"] = True
    with pytest.raises(Phase8LeakageSerializationError, match="unknown"):
        phase8_leakage_artifact_from_json(json.dumps(payload), Phase8LeakagePreregistrationLock)

    payload = json.loads(phase8_leakage_artifact_to_json(preregistration))
    payload["preregistration_hash"] = HASH8
    with pytest.raises(Phase8LeakageHashError, match="preregistration_hash"):
        phase8_leakage_artifact_from_json(json.dumps(payload), Phase8LeakagePreregistrationLock)


def test_label_access_before_preregistration_rejected() -> None:
    preregistration = _preregistration()
    prediction_lock = _prediction_lock(preregistration)
    ledger = _ledger(preregistration, prediction_lock)

    with pytest.raises(Phase8LeakageValidationError, match="preregistration"):
        authorize_external_label_access(
            preregistration=None,
            prediction_lock=prediction_lock,
            ledger=ledger,
        )


def test_label_access_before_prediction_lock_rejected() -> None:
    preregistration = _preregistration()
    prediction_lock = _prediction_lock(preregistration)
    ledger = _ledger(preregistration, prediction_lock)

    with pytest.raises(Phase8LeakageValidationError, match="prediction lock"):
        authorize_external_label_access(
            preregistration=preregistration,
            prediction_lock=None,
            ledger=ledger,
        )


def test_valid_label_access_after_preregistration_and_lock_accepted() -> None:
    preregistration = _preregistration()
    prediction_lock = _prediction_lock(preregistration)
    ledger = _ledger(preregistration, prediction_lock)

    authorize_external_label_access(
        preregistration=preregistration,
        prediction_lock=prediction_lock,
        ledger=ledger,
    )


@pytest.mark.parametrize("field_name", FROZEN_DECISION_FIELDS)
def test_post_preregistration_frozen_decision_changes_rejected(field_name: str) -> None:
    preregistration = _preregistration()
    proposed = _decisions(**{field_name: HASH8})

    with pytest.raises(Phase8LeakageValidationError, match=field_name):
        validate_frozen_decisions_unchanged(preregistration, proposed)


def test_external_result_values_cannot_modify_frozen_decisions() -> None:
    preregistration = _preregistration()
    prediction_lock = _prediction_lock(preregistration)
    ledger = _ledger(preregistration, prediction_lock)

    with pytest.raises(Phase8LeakageValidationError, match="external result values"):
        _post_result_lock(
            preregistration,
            prediction_lock,
            ledger,
            external_result_values_used_for_decisions=True,
        )

    changed_decisions = _decisions(threshold_hash=HASH8)
    post_result_lock = _post_result_lock(
        preregistration,
        prediction_lock,
        ledger,
        decisions=changed_decisions,
    )
    with pytest.raises(Phase8LeakageValidationError, match="threshold_hash"):
        validate_post_result_decision_lock(
            preregistration=preregistration,
            prediction_lock=prediction_lock,
            ledger=ledger,
            post_result_lock=post_result_lock,
        )


def test_deviations_must_be_explicit_and_reported_silent_reopen_rejected() -> None:
    preregistration = _preregistration()
    prediction_lock = _prediction_lock(preregistration)
    ledger = _ledger(preregistration, prediction_lock)

    with pytest.raises(Phase8LeakageValidationError, match="missing"):
        require_explicit_reported_deviation((), deviation_required=True)

    deviation = _deviation(affected_field="threshold_hash")
    require_explicit_reported_deviation((deviation,), deviation_required=True)

    with pytest.raises(Phase8LeakageValidationError, match="reported"):
        _deviation(reported=False)
    with pytest.raises(Phase8LeakageValidationError, match="cannot change"):
        _deviation(changes_frozen_decision=True)
    with pytest.raises(Phase8LeakageValidationError, match="silent reopening"):
        _post_result_lock(
            preregistration,
            prediction_lock,
            ledger,
            frozen_decision_reopen_attempted=True,
        )

    post_result_lock = _post_result_lock(
        preregistration,
        prediction_lock,
        ledger,
        deviations=(deviation,),
        frozen_decision_reopen_attempted=True,
    )
    validate_post_result_decision_lock(
        preregistration=preregistration,
        prediction_lock=prediction_lock,
        ledger=ledger,
        post_result_lock=post_result_lock,
    )


def test_machine_specific_scientific_identity_values_rejected() -> None:
    with pytest.raises(Phase8LeakageValidationError, match="machine-specific"):
        build_phase8_external_evaluation_access_event(
            access_event_id="label_access_001",
            access_kind=LABEL_ACCESS_KIND,
            purpose="localhost",
            preregistration_hash=HASH0,
            decision_freeze_hash=HASH1,
            prediction_lock_hash=HASH2,
        )
