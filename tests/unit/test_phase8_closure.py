"""Synthetic tests for the Phase 8 final closure / reproduction record contract.

All tests use synthetic hash/commit strings and temp-path publication targets. No real `/Volumes`
path, medical data, or PHI is referenced anywhere in this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from protoem_ct.external.closure import (
    Phase8ClosureHashError,
    Phase8ClosurePublicationError,
    Phase8ClosureValidationError,
    build_phase8_closure_reproduction_record,
    phase8_closure_reproduction_record_from_json,
    phase8_closure_reproduction_record_to_dict,
    publish_phase8_closure_reproduction_record,
)

_FAKE_SHA256 = "a" * 64
_OTHER_SHA256 = "b" * 64
_FAKE_COMMIT = "c" * 40


def _build_record(**overrides: object) -> object:
    base = dict(
        closure_status="closed_negative_external_validation_result",
        freeze_artifact_sha256=_FAKE_SHA256,
        checkpoint_sha256=_FAKE_SHA256,
        preregistration_hash=_FAKE_SHA256,
        prediction_lock_hash=_FAKE_SHA256,
        external_metric_report_hash=_FAKE_SHA256,
        domain_shift_record_hash=_FAKE_SHA256,
        original_comparison_artifact_sha256=_FAKE_SHA256,
        corrected_comparison_artifact_relative_path=(
            "phase8_internal_external_comparison_corrected_v1.json"
        ),
        provenance_fix_commit=_FAKE_COMMIT,
        final_report_relative_path="docs/phase8/FINAL_REPORT.md",
        external_drive_accessible_this_session=True,
    )
    base.update(overrides)
    return build_phase8_closure_reproduction_record(**base)  # type: ignore[arg-type]


def test_build_record_is_self_consistent_and_round_trips() -> None:
    record = _build_record()
    payload = phase8_closure_reproduction_record_to_dict(record)  # type: ignore[arg-type]

    import json

    reconstructed = phase8_closure_reproduction_record_from_json(
        json.dumps(payload).encode("utf-8")
    )

    assert reconstructed == record
    assert reconstructed.closure_status == "closed_negative_external_validation_result"
    assert reconstructed.scientific_computation_rerun is False


def test_record_rejects_claiming_scientific_computation_was_rerun() -> None:
    from protoem_ct.external.closure import (
        PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_NAME,
        PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_VERSION,
        Phase8ClosureReproductionRecord,
    )

    # The "no scientific computation rerun" guard must fire even when the caller supplies an
    # otherwise well-formed self-hash, so use any 64-hex placeholder: this construction must be
    # rejected before the hash comparison is ever consulted.
    with pytest.raises(Phase8ClosureValidationError):
        Phase8ClosureReproductionRecord(
            schema_name=PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_NAME,
            schema_version=PHASE8_CLOSURE_REPRODUCTION_RECORD_SCHEMA_VERSION,
            closure_record_hash=_FAKE_SHA256,
            closure_status="closed_negative_external_validation_result",
            freeze_artifact_sha256=_FAKE_SHA256,
            checkpoint_sha256=_FAKE_SHA256,
            preregistration_hash=_FAKE_SHA256,
            prediction_lock_hash=_FAKE_SHA256,
            external_metric_report_hash=_FAKE_SHA256,
            domain_shift_record_hash=_FAKE_SHA256,
            original_comparison_artifact_sha256=_FAKE_SHA256,
            corrected_comparison_artifact_relative_path="corrected.json",
            provenance_fix_commit=_FAKE_COMMIT,
            final_report_relative_path="docs/phase8/FINAL_REPORT.md",
            external_drive_accessible_this_session=True,
            scientific_computation_rerun=True,
        )


def test_tampered_field_fails_self_hash_validation() -> None:
    record = _build_record()
    payload = phase8_closure_reproduction_record_to_dict(record)  # type: ignore[arg-type]
    tampered = dict(payload)
    tampered["checkpoint_sha256"] = _OTHER_SHA256

    import json

    with pytest.raises(Phase8ClosureHashError):
        phase8_closure_reproduction_record_from_json(json.dumps(tampered).encode("utf-8"))


def test_publish_writes_canonical_json_and_refuses_overwrite(tmp_path: Path) -> None:
    record = _build_record()
    output_path = tmp_path / "phase8_final_closure_reproduction_record.json"

    publish_phase8_closure_reproduction_record(record, output_path)  # type: ignore[arg-type]
    assert output_path.is_file()

    reloaded = phase8_closure_reproduction_record_from_json(output_path.read_bytes())
    assert reloaded == record

    with pytest.raises(Phase8ClosurePublicationError):
        publish_phase8_closure_reproduction_record(record, output_path)
