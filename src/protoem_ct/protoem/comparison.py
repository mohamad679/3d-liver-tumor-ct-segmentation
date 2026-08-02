"""Deterministic Phase 6 ProtoEM ablation-matrix execution and comparison artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.evaluation.metrics import (
    compute_binary_confusion_counts,
    dice_from_counts,
    iou_from_counts,
)
from protoem_ct.protoem.ablation import (
    PROTOEM_ABLATION_VARIANT_ORDER,
    ProtoEMAblationPlanEntry,
    build_protoem_ablation_execution_plan,
)
from protoem_ct.protoem.artifacts import ProtoEMObjectiveWeights
from protoem_ct.protoem.inference import ProtoEMExecutionPackage
from protoem_ct.protoem.publication import Phase6PublicationSettings
from protoem_ct.retrieval.publication import (
    Phase5ComparisonRecord,
    Phase5ComparisonTable,
)

PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_NAME: Final[str] = "protoem_ablation_comparison_record"
PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_NAME: Final[str] = "protoem_ablation_comparison_table"
PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_NAME: Final[str] = "protoem_ablation_run_inventory"
PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_VERSION: Final[str] = "v1"
PHASE6_ABLATION_COMPARISON_JSON_NAME: Final[str] = "ablation_comparison.json"
PHASE6_ABLATION_COMPARISON_MARKDOWN_NAME: Final[str] = "ablation_comparison_table.md"
PHASE6_ABLATION_RUN_INVENTORY_JSON_NAME: Final[str] = "ablation_run_inventory.json"

SUPPORTED_COMPARISON_STATUSES: Final[frozenset[str]] = frozenset(
    {"executed", "phase5_baseline_link", "provenance_only", "unsupported", "failed"}
)
SUPPORTED_METRIC_AVAILABILITY: Final[frozenset[str]] = frozenset({"available", "unavailable"})
_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")


class ProtoEMComparisonError(ValueError):
    """Base error for deterministic Phase 6 ablation comparisons."""


class ProtoEMComparisonValidationError(ProtoEMComparisonError):
    """Raised when one comparison artifact is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class ProtoEMAblationComparisonRecord:
    """One deterministic Phase 6 ablation comparison row."""

    schema_name: str
    schema_version: str
    record_identity_hash: str
    variant_name: str
    execution_status: str
    config_hash: str | None
    baseline_artifact_hash: str | None
    initialization_identity_hash: str | None
    objective_trace_hash: str | None
    stopping_reason: str | None
    completed_iteration_count: int | None
    final_inference_identity_hash: str | None
    final_prediction_content_hash: str | None
    support_objective: float | None
    query_entropy_objective: float | None
    class_balance_objective: float | None
    consistency_objective: float | None
    proximal_objective: float | None
    total_objective: float | None
    metrics_availability_status: str
    dice: float | None
    iou: float | None
    reason_code: str | None
    reason_message: str | None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.record_identity_hash, field_name="record_identity_hash")
        if self.variant_name not in PROTOEM_ABLATION_VARIANT_ORDER:
            raise ProtoEMComparisonValidationError(
                f"variant_name must be one of {list(PROTOEM_ABLATION_VARIANT_ORDER)!r}."
            )
        if self.execution_status not in SUPPORTED_COMPARISON_STATUSES:
            raise ProtoEMComparisonValidationError(
                f"execution_status must be one of {sorted(SUPPORTED_COMPARISON_STATUSES)!r}."
            )
        for field_name in (
            "config_hash",
            "baseline_artifact_hash",
            "initialization_identity_hash",
            "objective_trace_hash",
            "final_inference_identity_hash",
            "final_prediction_content_hash",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_sha256(value, field_name=field_name)
        if self.completed_iteration_count is not None and self.completed_iteration_count < 0:
            raise ProtoEMComparisonValidationError(
                "completed_iteration_count must be nonnegative when present."
            )
        for field_name in (
            "support_objective",
            "query_entropy_objective",
            "class_balance_objective",
            "consistency_objective",
            "proximal_objective",
            "total_objective",
            "dice",
            "iou",
        ):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ProtoEMComparisonValidationError(f"{field_name} must be finite when present.")
        if self.metrics_availability_status not in SUPPORTED_METRIC_AVAILABILITY:
            supported_metric_availability = sorted(SUPPORTED_METRIC_AVAILABILITY)
            raise ProtoEMComparisonValidationError(
                f"metrics_availability_status must be one of {supported_metric_availability!r}."
            )
        if self.metrics_availability_status == "available":
            if self.dice is None or self.iou is None:
                raise ProtoEMComparisonValidationError(
                    "available metrics require both dice and iou."
                )
        else:
            if self.dice is not None or self.iou is not None:
                raise ProtoEMComparisonValidationError(
                    "unavailable metrics must not supply dice or iou."
                )
        if self.execution_status == "executed" and (
            self.config_hash is None or self.initialization_identity_hash is None
        ):
            raise ProtoEMComparisonValidationError(
                "executed comparison rows require config_hash and initialization_identity_hash."
            )
        if self.execution_status == "phase5_baseline_link" and self.baseline_artifact_hash is None:
            raise ProtoEMComparisonValidationError(
                "phase5_baseline_link rows require baseline_artifact_hash."
            )
        if self.execution_status in {"unsupported", "provenance_only", "failed"} and (
            self.reason_code is None or self.reason_message is None
        ):
            raise ProtoEMComparisonValidationError(
                f"{self.execution_status} rows require reason_code and reason_message."
            )
        if self.record_identity_hash != hash_protoem_ablation_comparison_record(self):
            raise ProtoEMComparisonValidationError(
                "record_identity_hash does not match deterministic comparison content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMAblationComparisonTable:
    """Canonical ordered Phase 6 ablation comparison table."""

    schema_name: str
    schema_version: str
    comparison_table_hash: str
    records: tuple[ProtoEMAblationComparisonRecord, ...]

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.comparison_table_hash, field_name="comparison_table_hash")
        variants = tuple(record.variant_name for record in self.records)
        if variants != PROTOEM_ABLATION_VARIANT_ORDER:
            raise ProtoEMComparisonValidationError(
                "comparison records must use the exact canonical 12-variant order."
            )
        if self.comparison_table_hash != hash_protoem_ablation_comparison_table(self):
            raise ProtoEMComparisonValidationError(
                "comparison_table_hash does not match deterministic comparison content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMAblationRunInventory:
    """Deterministic inventory of every Phase 6 ablation execution or linked baseline."""

    schema_name: str
    schema_version: str
    inventory_hash: str
    records: tuple[ProtoEMAblationComparisonRecord, ...]

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.inventory_hash, field_name="inventory_hash")
        variants = tuple(record.variant_name for record in self.records)
        if variants != PROTOEM_ABLATION_VARIANT_ORDER:
            raise ProtoEMComparisonValidationError(
                "inventory records must use the exact canonical 12-variant order."
            )
        if self.inventory_hash != hash_protoem_ablation_run_inventory(self):
            raise ProtoEMComparisonValidationError(
                "inventory_hash does not match deterministic inventory content."
            )


def build_protoem_ablation_comparison_table(
    *,
    settings: Phase6PublicationSettings,
    reference_mask: np.ndarray | None = None,
) -> ProtoEMAblationComparisonTable:
    """Execute or link the complete Phase 6 ablation matrix in canonical order."""

    effective_reference_mask = _comparison_reference_mask(reference_mask)
    phase5_table = _phase5_comparison_table(reference_mask=None)
    phase5_by_method = {record.method_name: record for record in phase5_table.records}
    plan = build_protoem_ablation_execution_plan()
    records = tuple(
        _build_variant_record(
            entry=entry,
            settings=settings,
            phase5_table=phase5_table,
            phase5_by_method=phase5_by_method,
            reference_mask=effective_reference_mask,
        )
        for entry in plan.entries
    )
    return ProtoEMAblationComparisonTable(
        schema_name=PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_NAME,
        schema_version=PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_VERSION,
        comparison_table_hash=sha256_json(
            {
                "schema_name": PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_NAME,
                "schema_version": PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_VERSION,
                "records": [
                    protoem_ablation_comparison_record_to_dict(record) for record in records
                ],
            }
        ),
        records=records,
    )


def build_protoem_ablation_run_inventory(
    *,
    comparison_table: ProtoEMAblationComparisonTable,
) -> ProtoEMAblationRunInventory:
    """Build one deterministic ablation-run inventory from a validated comparison table."""

    return ProtoEMAblationRunInventory(
        schema_name=PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_NAME,
        schema_version=PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_VERSION,
        inventory_hash=sha256_json(
            {
                "schema_name": PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_NAME,
                "schema_version": PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_VERSION,
                "records": [
                    protoem_ablation_comparison_record_to_dict(record)
                    for record in comparison_table.records
                ],
            }
        ),
        records=comparison_table.records,
    )


def render_protoem_ablation_comparison_markdown_from_json(
    payload: bytes | str,
) -> bytes:
    """Render ablation comparison markdown strictly from validated JSON."""

    table = protoem_ablation_comparison_table_from_json(payload)
    lines = [
        "# Phase 6 Ablation Comparison",
        "",
        "| variant | status | stop_reason | iterations | final_inference | dice | iou | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in table.records:
        completed_iteration_count = (
            record.completed_iteration_count
            if record.completed_iteration_count is not None
            else "unavailable"
        )
        lines.append(
            "| "
            f"{record.variant_name} | "
            f"{record.execution_status} | "
            f"{record.stopping_reason or 'unavailable'} | "
            f"{completed_iteration_count} | "
            f"{record.final_inference_identity_hash or 'unavailable'} | "
            f"{record.dice if record.dice is not None else 'unavailable'} | "
            f"{record.iou if record.iou is not None else 'unavailable'} | "
            f"{record.reason_code or 'none'} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def protoem_ablation_comparison_record_to_dict(
    record: ProtoEMAblationComparisonRecord,
) -> dict[str, JsonValue]:
    """Convert one comparison record to a canonical mapping."""

    return {
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "record_identity_hash": record.record_identity_hash,
        "variant_name": record.variant_name,
        "execution_status": record.execution_status,
        "config_hash": record.config_hash,
        "baseline_artifact_hash": record.baseline_artifact_hash,
        "initialization_identity_hash": record.initialization_identity_hash,
        "objective_trace_hash": record.objective_trace_hash,
        "stopping_reason": record.stopping_reason,
        "completed_iteration_count": record.completed_iteration_count,
        "final_inference_identity_hash": record.final_inference_identity_hash,
        "final_prediction_content_hash": record.final_prediction_content_hash,
        "support_objective": record.support_objective,
        "query_entropy_objective": record.query_entropy_objective,
        "class_balance_objective": record.class_balance_objective,
        "consistency_objective": record.consistency_objective,
        "proximal_objective": record.proximal_objective,
        "total_objective": record.total_objective,
        "metrics_availability_status": record.metrics_availability_status,
        "dice": record.dice,
        "iou": record.iou,
        "reason_code": record.reason_code,
        "reason_message": record.reason_message,
    }


def _protoem_ablation_comparison_record_identity_payload(
    record: ProtoEMAblationComparisonRecord,
) -> dict[str, JsonValue]:
    payload = protoem_ablation_comparison_record_to_dict(record)
    del payload["record_identity_hash"]
    return payload


def protoem_ablation_comparison_table_to_dict(
    table: ProtoEMAblationComparisonTable,
) -> dict[str, JsonValue]:
    """Convert one comparison table to a canonical mapping."""

    return {
        "schema_name": table.schema_name,
        "schema_version": table.schema_version,
        "comparison_table_hash": table.comparison_table_hash,
        "records": [protoem_ablation_comparison_record_to_dict(record) for record in table.records],
    }


def _protoem_ablation_comparison_table_identity_payload(
    table: ProtoEMAblationComparisonTable,
) -> dict[str, JsonValue]:
    payload = protoem_ablation_comparison_table_to_dict(table)
    del payload["comparison_table_hash"]
    return payload


def protoem_ablation_run_inventory_to_dict(
    inventory: ProtoEMAblationRunInventory,
) -> dict[str, JsonValue]:
    """Convert one ablation inventory to a canonical mapping."""

    return {
        "schema_name": inventory.schema_name,
        "schema_version": inventory.schema_version,
        "inventory_hash": inventory.inventory_hash,
        "records": [
            protoem_ablation_comparison_record_to_dict(record) for record in inventory.records
        ],
    }


def _protoem_ablation_run_inventory_identity_payload(
    inventory: ProtoEMAblationRunInventory,
) -> dict[str, JsonValue]:
    payload = protoem_ablation_run_inventory_to_dict(inventory)
    del payload["inventory_hash"]
    return payload


def protoem_ablation_comparison_table_to_json(
    table: ProtoEMAblationComparisonTable,
) -> bytes:
    """Serialize one comparison table to canonical JSON."""

    return canonical_json_bytes(protoem_ablation_comparison_table_to_dict(table)) + b"\n"


def protoem_ablation_run_inventory_to_json(inventory: ProtoEMAblationRunInventory) -> bytes:
    """Serialize one ablation inventory to canonical JSON."""

    return canonical_json_bytes(protoem_ablation_run_inventory_to_dict(inventory)) + b"\n"


def protoem_ablation_comparison_table_from_json(
    payload: bytes | str,
) -> ProtoEMAblationComparisonTable:
    """Parse one comparison table from canonical JSON."""

    mapping = _parse_json_mapping(payload, object_name="ProtoEMAblationComparisonTable")
    _require_schema_name(
        _expect_string(mapping["schema_name"], field_name="schema_name"),
        expected=PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_NAME,
        field_name="schema_name",
    )
    _require_schema_version(
        _expect_string(mapping["schema_version"], field_name="schema_version"),
        expected=PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_VERSION,
        field_name="schema_version",
    )
    records_payload = mapping["records"]
    if not isinstance(records_payload, list):
        raise ProtoEMComparisonValidationError("records must be a JSON array.")
    records = tuple(
        _comparison_record_from_mapping(cast(dict[str, object], item)) for item in records_payload
    )
    return ProtoEMAblationComparisonTable(
        schema_name=PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_NAME,
        schema_version=PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_VERSION,
        comparison_table_hash=_expect_string(
            mapping["comparison_table_hash"],
            field_name="comparison_table_hash",
        ),
        records=records,
    )


def hash_protoem_ablation_comparison_record(record: ProtoEMAblationComparisonRecord) -> str:
    """Return the deterministic comparison-record hash."""

    return sha256_json(_protoem_ablation_comparison_record_identity_payload(record))


def hash_protoem_ablation_comparison_table(table: ProtoEMAblationComparisonTable) -> str:
    """Return the deterministic comparison-table hash."""

    return sha256_json(_protoem_ablation_comparison_table_identity_payload(table))


def hash_protoem_ablation_run_inventory(inventory: ProtoEMAblationRunInventory) -> str:
    """Return the deterministic inventory hash."""

    return sha256_json(_protoem_ablation_run_inventory_identity_payload(inventory))


def _build_variant_record(
    *,
    entry: ProtoEMAblationPlanEntry,
    settings: Phase6PublicationSettings,
    phase5_table: Phase5ComparisonTable,
    phase5_by_method: dict[str, Phase5ComparisonRecord],
    reference_mask: np.ndarray | None,
) -> ProtoEMAblationComparisonRecord:
    variant = entry.ablation_definition.variant_name
    if variant == "no_retrieval":
        phase5_record = phase5_by_method["no_retrieval"]
        return _phase5_baseline_record(
            variant_name=variant,
            baseline_artifact_hash=phase5_table.comparison_identity_sha256,
            phase5_record=phase5_record,
        )
    if variant == "no_transduction":
        phase5_record = phase5_by_method["no_retrieval"]
        return _phase5_baseline_record(
            variant_name=variant,
            baseline_artifact_hash=phase5_record.inference_identity_sha256,
            phase5_record=phase5_record,
        )
    if variant == "multiple_prototypes":
        return _negative_record(
            variant_name=variant,
            status="unsupported",
            reason_code="multiple_prototypes_unsupported",
            reason_message="multiple_prototypes remains unsupported in the current Phase 6 path.",
        )
    if variant in {"head_only", "decoder_only", "full_finetune"}:
        return _negative_record(
            variant_name=variant,
            status="provenance_only",
            reason_code=f"{variant}_provenance_only",
            reason_message=(
                f"{variant} remains provenance-only without a real executable adaptation path."
            ),
        )
    if variant == "learned_update_schedule" and settings.positive_step_schedule is None:
        return _negative_record(
            variant_name=variant,
            status="unsupported",
            reason_code="missing_positive_step_schedule",
            reason_message=(
                "learned_update_schedule requires explicit parameterized positive-step values."
            ),
        )
    variant_settings = _settings_for_variant(
        settings=settings,
        variant_name=variant,
    )
    try:
        package = _execute_variant_package(variant_settings)
    except Exception as exc:
        return _negative_record(
            variant_name=variant,
            status="failed",
            reason_code="execution_failure",
            reason_message=f"{type(exc).__name__}: {exc}",
        )
    return _executed_record(
        variant_name=variant,
        package=package,
        reference_mask=reference_mask,
    )


def _execute_variant_package(settings: Phase6PublicationSettings) -> Any:
    from protoem_ct.protoem.publication import build_phase6_synthetic_execution_package

    return build_phase6_synthetic_execution_package(settings)


def _executed_record(
    *,
    variant_name: str,
    package: ProtoEMExecutionPackage,
    reference_mask: np.ndarray | None,
) -> ProtoEMAblationComparisonRecord:
    final_record = (
        package.objective_trace.iterations[-1] if package.objective_trace is not None else None
    )
    dice: float | None = None
    iou: float | None = None
    metrics_status = "unavailable"
    if package.final_inference_result is not None and reference_mask is not None:
        counts = compute_binary_confusion_counts(
            ground_truth_mask=np.asarray(reference_mask, dtype=bool),
            prediction_mask=np.asarray(package.final_inference_result.prediction_map, dtype=bool),
        )
        dice = float(dice_from_counts(counts))
        iou = float(iou_from_counts(counts))
        metrics_status = "available"
    execution_status = (
        "executed" if package.optimization_result.execution_status == "completed" else "failed"
    )
    return _materialize_record(
        variant_name=variant_name,
        execution_status=execution_status,
        config_hash=package.config.config_hash,
        baseline_artifact_hash=None,
        initialization_identity_hash=package.initialization_bundle.initialization_identity_hash,
        objective_trace_hash=(
            package.objective_trace.objective_trace_hash
            if package.objective_trace is not None
            else None
        ),
        stopping_reason=(
            package.stopping_record.stop_reason if package.stopping_record is not None else None
        ),
        completed_iteration_count=(
            package.stopping_record.completed_iteration_count
            if package.stopping_record is not None
            else None
        ),
        final_inference_identity_hash=(
            package.final_inference_result.inference_identity_hash
            if package.final_inference_result is not None
            else None
        ),
        final_prediction_content_hash=(
            package.final_inference_result.prediction_content_hash
            if package.final_inference_result is not None
            else None
        ),
        support_objective=(final_record.support_objective if final_record is not None else None),
        query_entropy_objective=(
            final_record.query_entropy_objective if final_record is not None else None
        ),
        class_balance_objective=(
            final_record.class_balance_objective if final_record is not None else None
        ),
        consistency_objective=(
            final_record.consistency_objective if final_record is not None else None
        ),
        proximal_objective=(final_record.proximal_objective if final_record is not None else None),
        total_objective=final_record.total_objective if final_record is not None else None,
        metrics_availability_status=metrics_status,
        dice=dice,
        iou=iou,
        reason_code=package.optimization_result.failure_code
        if execution_status == "failed"
        else None,
        reason_message=package.optimization_result.failure_message
        if execution_status == "failed"
        else None,
    )


def _phase5_baseline_record(
    *,
    variant_name: str,
    baseline_artifact_hash: str,
    phase5_record: Phase5ComparisonRecord,
) -> ProtoEMAblationComparisonRecord:
    metrics_status = phase5_record.metrics_availability_status
    return _materialize_record(
        variant_name=variant_name,
        execution_status="phase5_baseline_link",
        config_hash=None,
        baseline_artifact_hash=baseline_artifact_hash,
        initialization_identity_hash=None,
        objective_trace_hash=None,
        stopping_reason=None,
        completed_iteration_count=None,
        final_inference_identity_hash=phase5_record.inference_identity_sha256,
        final_prediction_content_hash=phase5_record.prediction_content_sha256,
        support_objective=None,
        query_entropy_objective=None,
        class_balance_objective=None,
        consistency_objective=None,
        proximal_objective=None,
        total_objective=None,
        metrics_availability_status=metrics_status,
        dice=phase5_record.dice,
        iou=phase5_record.iou,
        reason_code=None,
        reason_message=None,
    )


def _negative_record(
    *,
    variant_name: str,
    status: str,
    reason_code: str,
    reason_message: str,
) -> ProtoEMAblationComparisonRecord:
    return _materialize_record(
        variant_name=variant_name,
        execution_status=status,
        config_hash=None,
        baseline_artifact_hash=None,
        initialization_identity_hash=None,
        objective_trace_hash=None,
        stopping_reason=None,
        completed_iteration_count=None,
        final_inference_identity_hash=None,
        final_prediction_content_hash=None,
        support_objective=None,
        query_entropy_objective=None,
        class_balance_objective=None,
        consistency_objective=None,
        proximal_objective=None,
        total_objective=None,
        metrics_availability_status="unavailable",
        dice=None,
        iou=None,
        reason_code=reason_code,
        reason_message=reason_message,
    )


def _materialize_record(
    *,
    variant_name: str,
    execution_status: str,
    config_hash: str | None,
    baseline_artifact_hash: str | None,
    initialization_identity_hash: str | None,
    objective_trace_hash: str | None,
    stopping_reason: str | None,
    completed_iteration_count: int | None,
    final_inference_identity_hash: str | None,
    final_prediction_content_hash: str | None,
    support_objective: float | None,
    query_entropy_objective: float | None,
    class_balance_objective: float | None,
    consistency_objective: float | None,
    proximal_objective: float | None,
    total_objective: float | None,
    metrics_availability_status: str,
    dice: float | None,
    iou: float | None,
    reason_code: str | None,
    reason_message: str | None,
) -> ProtoEMAblationComparisonRecord:
    payload = {
        "schema_name": PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_NAME,
        "schema_version": PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_VERSION,
        "variant_name": variant_name,
        "execution_status": execution_status,
        "config_hash": config_hash,
        "baseline_artifact_hash": baseline_artifact_hash,
        "initialization_identity_hash": initialization_identity_hash,
        "objective_trace_hash": objective_trace_hash,
        "stopping_reason": stopping_reason,
        "completed_iteration_count": completed_iteration_count,
        "final_inference_identity_hash": final_inference_identity_hash,
        "final_prediction_content_hash": final_prediction_content_hash,
        "support_objective": support_objective,
        "query_entropy_objective": query_entropy_objective,
        "class_balance_objective": class_balance_objective,
        "consistency_objective": consistency_objective,
        "proximal_objective": proximal_objective,
        "total_objective": total_objective,
        "metrics_availability_status": metrics_availability_status,
        "dice": dice,
        "iou": iou,
        "reason_code": reason_code,
        "reason_message": reason_message,
    }
    return ProtoEMAblationComparisonRecord(
        schema_name=PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_NAME,
        schema_version=PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_VERSION,
        record_identity_hash=sha256_json(payload),
        variant_name=variant_name,
        execution_status=execution_status,
        config_hash=config_hash,
        baseline_artifact_hash=baseline_artifact_hash,
        initialization_identity_hash=initialization_identity_hash,
        objective_trace_hash=objective_trace_hash,
        stopping_reason=stopping_reason,
        completed_iteration_count=completed_iteration_count,
        final_inference_identity_hash=final_inference_identity_hash,
        final_prediction_content_hash=final_prediction_content_hash,
        support_objective=support_objective,
        query_entropy_objective=query_entropy_objective,
        class_balance_objective=class_balance_objective,
        consistency_objective=consistency_objective,
        proximal_objective=proximal_objective,
        total_objective=total_objective,
        metrics_availability_status=metrics_availability_status,
        dice=dice,
        iou=iou,
        reason_code=reason_code,
        reason_message=reason_message,
    )


def _settings_for_variant(
    *,
    settings: Phase6PublicationSettings,
    variant_name: str,
) -> Phase6PublicationSettings:
    weights = settings.objective_weights
    if variant_name == "no_class_balance":
        objective_weights = ProtoEMObjectiveWeights(
            schema_name=weights.schema_name,
            schema_version=weights.schema_version,
            support=weights.support,
            query_entropy=weights.query_entropy,
            class_balance=0.0,
            consistency=weights.consistency,
            proximal=weights.proximal,
        )
    elif variant_name == "no_proximal":
        objective_weights = ProtoEMObjectiveWeights(
            schema_name=weights.schema_name,
            schema_version=weights.schema_version,
            support=weights.support,
            query_entropy=weights.query_entropy,
            class_balance=weights.class_balance,
            consistency=weights.consistency,
            proximal=0.0,
        )
    else:
        objective_weights = weights
    update_schedule = settings.update_schedule
    positive_step_schedule = settings.positive_step_schedule
    if variant_name == "fixed_update_schedule":
        update_schedule = "fixed_em_like"
        positive_step_schedule = None
    elif variant_name == "learned_update_schedule":
        update_schedule = "learned_positive_step"
    elif variant_name == "full_protoem":
        update_schedule = settings.update_schedule
    elif variant_name == "single_prototype":
        pass
    return Phase6PublicationSettings(
        schema_version=settings.schema_version,
        synthetic_mode_only=settings.synthetic_mode_only,
        explicit_seed=settings.explicit_seed,
        max_iterations=settings.max_iterations,
        minimum_iterations=settings.minimum_iterations,
        convergence_tolerance=settings.convergence_tolerance,
        temperature=settings.temperature,
        confidence_threshold=settings.confidence_threshold,
        foreground_prior=settings.foreground_prior,
        update_schedule=update_schedule,
        prototype_mode="single_prototype",
        objective_weights=objective_weights,
        phase5_initialization_schema_name=settings.phase5_initialization_schema_name,
        phase5_prototype_schema_name=settings.phase5_prototype_schema_name,
        phase5_inference_schema_name=settings.phase5_inference_schema_name,
        positive_step_schedule=positive_step_schedule,
    )


def _phase5_comparison_table(*, reference_mask: np.ndarray | None) -> Phase5ComparisonTable:
    from protoem_ct.retrieval.publication import (
        build_phase5_synthetic_fixture,
        run_phase5_comparison,
    )

    fixture = build_phase5_synthetic_fixture()
    return run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=fixture.support_records,
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=fixture.query_reference_mask
        if reference_mask is None
        else reference_mask,
        emit_confidence_margin=True,
    )


def _comparison_reference_mask(reference_mask: np.ndarray | None) -> np.ndarray:
    if reference_mask is not None:
        return np.asarray(reference_mask, dtype=np.uint8)
    return np.asarray([[[[[1, 0, 1]]]]], dtype=np.uint8)


def _comparison_record_from_mapping(
    mapping: dict[str, object],
) -> ProtoEMAblationComparisonRecord:
    return ProtoEMAblationComparisonRecord(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        record_identity_hash=_expect_string(
            mapping["record_identity_hash"],
            field_name="record_identity_hash",
        ),
        variant_name=_expect_string(mapping["variant_name"], field_name="variant_name"),
        execution_status=_expect_string(
            mapping["execution_status"],
            field_name="execution_status",
        ),
        config_hash=_expect_optional_string(mapping["config_hash"], field_name="config_hash"),
        baseline_artifact_hash=_expect_optional_string(
            mapping["baseline_artifact_hash"],
            field_name="baseline_artifact_hash",
        ),
        initialization_identity_hash=_expect_optional_string(
            mapping["initialization_identity_hash"],
            field_name="initialization_identity_hash",
        ),
        objective_trace_hash=_expect_optional_string(
            mapping["objective_trace_hash"],
            field_name="objective_trace_hash",
        ),
        stopping_reason=_expect_optional_string(
            mapping["stopping_reason"],
            field_name="stopping_reason",
        ),
        completed_iteration_count=_expect_optional_int(
            mapping["completed_iteration_count"],
            field_name="completed_iteration_count",
        ),
        final_inference_identity_hash=_expect_optional_string(
            mapping["final_inference_identity_hash"],
            field_name="final_inference_identity_hash",
        ),
        final_prediction_content_hash=_expect_optional_string(
            mapping["final_prediction_content_hash"],
            field_name="final_prediction_content_hash",
        ),
        support_objective=_expect_optional_float(
            mapping["support_objective"],
            field_name="support_objective",
        ),
        query_entropy_objective=_expect_optional_float(
            mapping["query_entropy_objective"],
            field_name="query_entropy_objective",
        ),
        class_balance_objective=_expect_optional_float(
            mapping["class_balance_objective"],
            field_name="class_balance_objective",
        ),
        consistency_objective=_expect_optional_float(
            mapping["consistency_objective"],
            field_name="consistency_objective",
        ),
        proximal_objective=_expect_optional_float(
            mapping["proximal_objective"],
            field_name="proximal_objective",
        ),
        total_objective=_expect_optional_float(
            mapping["total_objective"],
            field_name="total_objective",
        ),
        metrics_availability_status=_expect_string(
            mapping["metrics_availability_status"],
            field_name="metrics_availability_status",
        ),
        dice=_expect_optional_float(mapping["dice"], field_name="dice"),
        iou=_expect_optional_float(mapping["iou"], field_name="iou"),
        reason_code=_expect_optional_string(mapping["reason_code"], field_name="reason_code"),
        reason_message=_expect_optional_string(
            mapping["reason_message"],
            field_name="reason_message",
        ),
    )


def _parse_json_mapping(payload: bytes | str, *, object_name: str) -> dict[str, object]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProtoEMComparisonValidationError(f"{object_name} JSON could not be decoded.") from exc
    if not isinstance(decoded, dict):
        raise ProtoEMComparisonValidationError(f"{object_name} JSON must decode to an object.")
    return cast(dict[str, object], decoded)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ProtoEMComparisonValidationError(f"{field_name} must be a string.")
    return value


def _expect_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, field_name=field_name)


def _expect_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtoEMComparisonValidationError(f"{field_name} must be an integer when present.")
    return value


def _expect_optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProtoEMComparisonValidationError(f"{field_name} must be numeric when present.")
    return float(value)


def _require_schema_name(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise ProtoEMComparisonValidationError(f"{field_name} must equal {expected!r}.")


def _require_schema_version(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise ProtoEMComparisonValidationError(f"{field_name} must equal {expected!r}.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise ProtoEMComparisonValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )


__all__ = [
    "PHASE6_ABLATION_COMPARISON_JSON_NAME",
    "PHASE6_ABLATION_COMPARISON_MARKDOWN_NAME",
    "PHASE6_ABLATION_RUN_INVENTORY_JSON_NAME",
    "PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_NAME",
    "PROTOEM_ABLATION_COMPARISON_RECORD_SCHEMA_VERSION",
    "PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_NAME",
    "PROTOEM_ABLATION_COMPARISON_TABLE_SCHEMA_VERSION",
    "PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_NAME",
    "PROTOEM_ABLATION_RUN_INVENTORY_SCHEMA_VERSION",
    "ProtoEMAblationComparisonRecord",
    "ProtoEMAblationComparisonTable",
    "ProtoEMAblationRunInventory",
    "ProtoEMComparisonError",
    "ProtoEMComparisonValidationError",
    "build_protoem_ablation_comparison_table",
    "build_protoem_ablation_run_inventory",
    "hash_protoem_ablation_comparison_record",
    "hash_protoem_ablation_comparison_table",
    "hash_protoem_ablation_run_inventory",
    "protoem_ablation_comparison_record_to_dict",
    "protoem_ablation_comparison_table_from_json",
    "protoem_ablation_comparison_table_to_dict",
    "protoem_ablation_comparison_table_to_json",
    "protoem_ablation_run_inventory_to_dict",
    "protoem_ablation_run_inventory_to_json",
    "render_protoem_ablation_comparison_markdown_from_json",
]
