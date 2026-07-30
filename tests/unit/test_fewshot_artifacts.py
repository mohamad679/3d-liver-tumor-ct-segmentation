"""Unit tests for deterministic Phase 4 few-shot artifact contracts."""

from __future__ import annotations

import json

import pytest

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.fewshot import (
    FEWSHOT_ADAPTATION_CONFIG_VERSION,
    FEWSHOT_PROTOCOL_TABLE_VERSION,
    FEWSHOT_RUN_SUMMARY_VERSION,
    FEWSHOT_SUPPORT_MANIFEST_VERSION,
    FewshotAdaptationConfig,
    FewshotArtifactSerializationError,
    FewshotArtifactValidationError,
    FewshotInitializationReference,
    FewshotProtocolTable,
    FewshotProtocolTableRow,
    FewshotRunSummary,
    FewshotSupportAssignment,
    FewshotSupportManifest,
    fewshot_adaptation_config_from_json,
    fewshot_adaptation_config_to_json,
    fewshot_artifact_from_json,
    fewshot_protocol_table_from_json,
    fewshot_protocol_table_to_json,
    fewshot_run_summary_from_json,
    fewshot_run_summary_to_json,
    fewshot_support_manifest_from_json,
    fewshot_support_manifest_to_json,
)


def _reference(
    *,
    reference_type: str = "baseline_provenance",
    artifact_sha256: str | None = "1" * 64,
    checkpoint_sha256: str | None = None,
) -> FewshotInitializationReference:
    return FewshotInitializationReference(
        reference_type=reference_type,
        reference_identifier="baseline_init_001",
        artifact_sha256=artifact_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )


def _support_manifest(
    *,
    requested_k: int = 2,
    assignments: tuple[FewshotSupportAssignment, ...] | None = None,
) -> FewshotSupportManifest:
    actual_assignments = assignments or tuple(
        FewshotSupportAssignment(
            anonymous_patient_id=f"anon-p{i:04d}",
            anonymous_case_id=f"anon-c{i:04d}",
        )
        for i in range(1, requested_k + 1)
    )
    schema_version = FEWSHOT_SUPPORT_MANIFEST_VERSION
    manifest_id = "support_manifest_001"
    replicate_id = "replicate_a"
    selection_seed = 1729
    selection_policy_version = "support_policy_v1"
    stratification_status = "fallback_unstratified"
    stratification_reason = "insufficient_bucket_size"
    source_development_manifest_hash = "2" * 64
    source_development_split_hash = "3" * 64
    source_lesion_summary_hash = "4" * 64
    immutable_test_cohort_hash = "5" * 64
    payload_without_hash = {
        "schema_version": schema_version,
        "manifest_id": manifest_id,
        "requested_k": requested_k,
        "replicate_id": replicate_id,
        "selection_seed": selection_seed,
        "selection_policy_version": selection_policy_version,
        "stratification_status": stratification_status,
        "stratification_reason": stratification_reason,
        "source_development_manifest_hash": source_development_manifest_hash,
        "source_development_split_hash": source_development_split_hash,
        "source_lesion_summary_hash": source_lesion_summary_hash,
        "immutable_test_cohort_hash": immutable_test_cohort_hash,
        "assignments": [
            {
                "anonymous_patient_id": item.anonymous_patient_id,
                "anonymous_case_id": item.anonymous_case_id,
            }
            for item in actual_assignments
        ],
    }
    return FewshotSupportManifest(
        artifact_hash=sha256_json(payload_without_hash),
        schema_version=schema_version,
        manifest_id=manifest_id,
        requested_k=requested_k,
        replicate_id=replicate_id,
        selection_seed=selection_seed,
        selection_policy_version=selection_policy_version,
        stratification_status=stratification_status,
        stratification_reason=stratification_reason,
        source_development_manifest_hash=source_development_manifest_hash,
        source_development_split_hash=source_development_split_hash,
        source_lesion_summary_hash=source_lesion_summary_hash,
        immutable_test_cohort_hash=immutable_test_cohort_hash,
        assignments=actual_assignments,
    )


def _adaptation_config(
    *,
    adaptation_mode: str = "head_only",
) -> FewshotAdaptationConfig:
    reference = _reference()
    schema_version = FEWSHOT_ADAPTATION_CONFIG_VERSION
    seed = 1729
    baseline_family = "monai_segresnet"
    optimizer_name = "adam"
    learning_rate = 0.001
    weight_decay = 0.0
    max_adaptation_steps = 8
    support_batch_size = 1
    gradient_accumulation_steps = 1
    amp_enabled = False
    payload_without_hash = {
        "schema_version": schema_version,
        "adaptation_mode": adaptation_mode,
        "seed": seed,
        "baseline_family": baseline_family,
        "initialization_reference": {
            "reference_type": reference.reference_type,
            "reference_identifier": reference.reference_identifier,
            "artifact_sha256": reference.artifact_sha256,
            "checkpoint_sha256": reference.checkpoint_sha256,
        },
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_adaptation_steps": max_adaptation_steps,
        "support_batch_size": support_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "amp_enabled": amp_enabled,
    }
    return FewshotAdaptationConfig(
        artifact_hash=sha256_json(payload_without_hash),
        schema_version=schema_version,
        adaptation_mode=adaptation_mode,
        seed=seed,
        baseline_family=baseline_family,
        initialization_reference=reference,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_adaptation_steps=max_adaptation_steps,
        support_batch_size=support_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        amp_enabled=amp_enabled,
    )


def _protocol_table() -> FewshotProtocolTable:
    reference = _reference()
    row_b = FewshotProtocolTableRow(
        requested_k=10,
        replicate_id="replicate_b",
        support_manifest_hash="6" * 64,
        adaptation_mode="full_finetune",
        adaptation_config_hash="7" * 64,
        initialization_reference=reference,
        planned_output_identifier="planned_b",
        run_status="planned",
    )
    row_a = FewshotProtocolTableRow(
        requested_k=1,
        replicate_id="replicate_a",
        support_manifest_hash="8" * 64,
        adaptation_mode="head_only",
        adaptation_config_hash="9" * 64,
        initialization_reference=reference,
        planned_output_identifier="planned_a",
        run_status="planned",
    )
    payload_without_hash = {
        "schema_version": FEWSHOT_PROTOCOL_TABLE_VERSION,
        "source_development_manifest_hash": "a" * 64,
        "source_development_split_hash": "b" * 64,
        "source_lesion_summary_hash": None,
        "immutable_test_cohort_hash": "c" * 64,
        "rows": [
            {
                "requested_k": row.requested_k,
                "replicate_id": row.replicate_id,
                "support_manifest_hash": row.support_manifest_hash,
                "adaptation_mode": row.adaptation_mode,
                "adaptation_config_hash": row.adaptation_config_hash,
                "initialization_reference": {
                    "reference_type": row.initialization_reference.reference_type,
                    "reference_identifier": row.initialization_reference.reference_identifier,
                    "artifact_sha256": row.initialization_reference.artifact_sha256,
                    "checkpoint_sha256": row.initialization_reference.checkpoint_sha256,
                },
                "planned_output_identifier": row.planned_output_identifier,
                "run_status": row.run_status,
            }
            for row in sorted(
                (row_b, row_a), key=lambda item: (item.requested_k, item.replicate_id)
            )
        ],
    }
    return FewshotProtocolTable(
        artifact_hash=sha256_json(payload_without_hash),
        schema_version=FEWSHOT_PROTOCOL_TABLE_VERSION,
        source_development_manifest_hash="a" * 64,
        source_development_split_hash="b" * 64,
        source_lesion_summary_hash=None,
        immutable_test_cohort_hash="c" * 64,
        rows=(row_b, row_a),
    )


def _run_summary(
    *,
    duration_seconds: float | None = None,
    memory_status: str = "unavailable",
    peak_allocated_memory_bytes: int | None = None,
    peak_reserved_memory_bytes: int | None = None,
    peak_host_memory_bytes: int | None = None,
) -> FewshotRunSummary:
    reference = _reference()
    payload_without_hash = {
        "schema_version": FEWSHOT_RUN_SUMMARY_VERSION,
        "support_manifest_hash": "d" * 64,
        "adaptation_config_hash": "e" * 64,
        "initialization_reference": {
            "reference_type": reference.reference_type,
            "reference_identifier": reference.reference_identifier,
            "artifact_sha256": reference.artifact_sha256,
            "checkpoint_sha256": reference.checkpoint_sha256,
        },
        "run_status": "completed",
        "trainable_parameter_count": 2048,
        "duration_seconds": duration_seconds,
        "memory_availability_status": memory_status,
        "peak_allocated_memory_bytes": peak_allocated_memory_bytes,
        "peak_reserved_memory_bytes": peak_reserved_memory_bytes,
        "peak_host_memory_bytes": peak_host_memory_bytes,
        "metric_artifact_references": [],
        "failure_code": None,
        "failure_message": None,
    }
    return FewshotRunSummary(
        artifact_hash=sha256_json(payload_without_hash),
        schema_version=FEWSHOT_RUN_SUMMARY_VERSION,
        support_manifest_hash="d" * 64,
        adaptation_config_hash="e" * 64,
        initialization_reference=reference,
        run_status="completed",
        trainable_parameter_count=2048,
        duration_seconds=duration_seconds,
        memory_availability_status=memory_status,
        peak_allocated_memory_bytes=peak_allocated_memory_bytes,
        peak_reserved_memory_bytes=peak_reserved_memory_bytes,
        peak_host_memory_bytes=peak_host_memory_bytes,
        metric_artifact_references=(),
        failure_code=None,
        failure_message=None,
    )


def test_support_manifest_schema_validation_and_round_trip() -> None:
    artifact = _support_manifest()
    encoded = fewshot_support_manifest_to_json(artifact)
    parsed = fewshot_support_manifest_from_json(encoded)

    assert parsed == artifact
    assert fewshot_artifact_from_json(encoded, FewshotSupportManifest) == artifact
    assert encoded.endswith(b"\n")


def test_invalid_k_rejected() -> None:
    with pytest.raises(FewshotArtifactValidationError):
        _support_manifest(requested_k=3)


def test_duplicate_patient_and_case_ids_rejected() -> None:
    duplicate_assignments = (
        FewshotSupportAssignment("anon-p0001", "anon-c0001"),
        FewshotSupportAssignment("anon-p0001", "anon-c0002"),
    )
    with pytest.raises(FewshotArtifactValidationError):
        _support_manifest(assignments=duplicate_assignments)

    duplicate_case_assignments = (
        FewshotSupportAssignment("anon-p0001", "anon-c0001"),
        FewshotSupportAssignment("anon-p0002", "anon-c0001"),
    )
    with pytest.raises(FewshotArtifactValidationError):
        _support_manifest(assignments=duplicate_case_assignments)


def test_assignment_count_mismatch_rejected() -> None:
    with pytest.raises(FewshotArtifactValidationError):
        _support_manifest(
            requested_k=2,
            assignments=(FewshotSupportAssignment("anon-p0001", "anon-c0001"),),
        )


def test_invalid_adaptation_mode_rejected() -> None:
    with pytest.raises(FewshotArtifactValidationError):
        _adaptation_config(adaptation_mode="encoder_only")


def test_canonical_serialization_and_hash_stability() -> None:
    support = _support_manifest()
    adaptation = _adaptation_config()
    protocol = _protocol_table()
    summary = _run_summary()

    assert fewshot_support_manifest_to_json(support) == fewshot_support_manifest_to_json(support)
    assert fewshot_adaptation_config_to_json(adaptation) == fewshot_adaptation_config_to_json(
        adaptation
    )
    assert fewshot_protocol_table_to_json(protocol) == fewshot_protocol_table_to_json(protocol)
    assert fewshot_run_summary_to_json(summary) == fewshot_run_summary_to_json(summary)

    assert (
        fewshot_support_manifest_from_json(fewshot_support_manifest_to_json(support)).artifact_hash
        == support.artifact_hash
    )
    assert (
        fewshot_adaptation_config_from_json(
            fewshot_adaptation_config_to_json(adaptation)
        ).artifact_hash
        == adaptation.artifact_hash
    )
    assert (
        fewshot_protocol_table_from_json(fewshot_protocol_table_to_json(protocol)).artifact_hash
        == protocol.artifact_hash
    )
    assert (
        fewshot_run_summary_from_json(fewshot_run_summary_to_json(summary)).artifact_hash
        == summary.artifact_hash
    )


def test_protocol_rows_are_sorted_deterministically() -> None:
    table = _protocol_table()

    assert [row.requested_k for row in table.rows] == [1, 10]
    assert table.rows[0].replicate_id == "replicate_a"
    assert table.rows[1].replicate_id == "replicate_b"


def test_nullable_duration_and_unavailable_memory_fields() -> None:
    summary = _run_summary(duration_seconds=None, memory_status="unavailable")
    parsed = fewshot_run_summary_from_json(fewshot_run_summary_to_json(summary))

    assert parsed.duration_seconds is None
    assert parsed.memory_availability_status == "unavailable"
    assert parsed.peak_allocated_memory_bytes is None
    assert parsed.peak_reserved_memory_bytes is None
    assert parsed.peak_host_memory_bytes is None


def test_available_memory_requires_metric_values() -> None:
    with pytest.raises(FewshotArtifactValidationError):
        _run_summary(memory_status="available")


def test_unknown_field_rejected() -> None:
    artifact = _support_manifest()
    payload = json.loads(fewshot_support_manifest_to_json(artifact))
    payload["unexpected"] = True

    with pytest.raises(FewshotArtifactSerializationError):
        fewshot_support_manifest_from_json(json.dumps(payload).encode("utf-8"))
