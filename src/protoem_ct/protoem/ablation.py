"""Deterministic ProtoEM-CT ablation planning and execution-result wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.protoem.artifacts import (
    PROTOEM_ABLATION_DEFINITION_SCHEMA_NAME,
    PROTOEM_ABLATION_DEFINITION_SCHEMA_VERSION,
    PROTOEM_ABLATION_OVERRIDE_SCHEMA_NAME,
    PROTOEM_ABLATION_OVERRIDE_SCHEMA_VERSION,
    ProtoEMAblationDefinition,
    ProtoEMAblationOverride,
    ProtoEMConfig,
    ProtoEMObjectiveWeights,
)
from protoem_ct.protoem.inference import ProtoEMExecutionPackage
from protoem_ct.retrieval.publication import NO_RETRIEVAL_METHOD

PROTOEM_ABLATION_PLAN_ENTRY_SCHEMA_NAME: Final[str] = "protoem_ablation_plan_entry"
PROTOEM_ABLATION_PLAN_ENTRY_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_NAME: Final[str] = "protoem_ablation_execution_plan"
PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_VERSION: Final[str] = "v1"
PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_NAME: Final[str] = "protoem_ablation_execution_result"
PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_VERSION: Final[str] = "v1"

PROTOEM_ABLATION_VARIANT_ORDER: Final[tuple[str, ...]] = (
    "full_protoem",
    "no_retrieval",
    "no_transduction",
    "no_class_balance",
    "no_proximal",
    "single_prototype",
    "multiple_prototypes",
    "fixed_update_schedule",
    "learned_update_schedule",
    "head_only",
    "decoder_only",
    "full_finetune",
)

SUPPORTED_ABLATION_EXECUTION_MODES: Final[frozenset[str]] = frozenset(
    {"protoem_execution", "phase5_baseline", "unsupported"}
)
SUPPORTED_PHASE5_BASELINE_KINDS: Final[frozenset[str]] = frozenset(
    {"no_retrieval", "prototype_only_inference"}
)
_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")


class ProtoEMAblationError(ValueError):
    """Base error for deterministic ProtoEM ablation planning."""


class AblationPlanError(ProtoEMAblationError):
    """Raised when one ablation plan or override contract is invalid."""


class UnsupportedAblationExecutionError(ProtoEMAblationError):
    """Raised when one requested ablation remains explicitly unsupported."""


@dataclass(frozen=True, slots=True)
class ProtoEMAblationPlanEntry:
    """One deterministic ablation-plan entry with explicit execution semantics."""

    schema_name: str
    schema_version: str
    entry_identity_hash: str
    ablation_definition: ProtoEMAblationDefinition
    execution_mode: str
    phase5_baseline_kind: str | None
    phase5_method_name: str | None
    unsupported_code: str | None
    unsupported_message: str | None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_ABLATION_PLAN_ENTRY_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_ABLATION_PLAN_ENTRY_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.entry_identity_hash, field_name="entry_identity_hash")
        if self.execution_mode not in SUPPORTED_ABLATION_EXECUTION_MODES:
            raise AblationPlanError(
                f"execution_mode must be one of {sorted(SUPPORTED_ABLATION_EXECUTION_MODES)!r}."
            )
        if self.execution_mode == "protoem_execution":
            if any(
                value is not None
                for value in (
                    self.phase5_baseline_kind,
                    self.phase5_method_name,
                    self.unsupported_code,
                    self.unsupported_message,
                )
            ):
                raise AblationPlanError(
                    "protoem_execution entries must not carry baseline or unsupported metadata."
                )
        elif self.execution_mode == "phase5_baseline":
            if self.phase5_baseline_kind not in SUPPORTED_PHASE5_BASELINE_KINDS:
                raise AblationPlanError(
                    "phase5_baseline entries must declare one supported baseline kind."
                )
            if (
                self.phase5_baseline_kind == "no_retrieval"
                and self.phase5_method_name != NO_RETRIEVAL_METHOD
            ):
                raise AblationPlanError(
                    "no_retrieval phase5 baselines must use phase5_method_name='no_retrieval'."
                )
            if (
                self.phase5_baseline_kind == "prototype_only_inference"
                and self.phase5_method_name is not None
            ):
                raise AblationPlanError(
                    "prototype_only_inference baselines must not declare one Phase 5 method name."
                )
            if self.unsupported_code is not None or self.unsupported_message is not None:
                raise AblationPlanError(
                    "phase5_baseline entries must not carry unsupported metadata."
                )
        else:
            if self.unsupported_code is None or self.unsupported_message is None:
                raise AblationPlanError(
                    "unsupported ablation entries require unsupported_code and unsupported_message."
                )
            if self.phase5_baseline_kind is not None or self.phase5_method_name is not None:
                raise AblationPlanError(
                    "unsupported ablation entries must not masquerade as Phase 5 baselines."
                )
        if self.entry_identity_hash != hash_protoem_ablation_plan_entry(self):
            raise AblationPlanError(
                "entry_identity_hash does not match deterministic ablation-plan content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMAblationExecutionPlan:
    """Deterministic ordered ProtoEM ablation plan."""

    schema_name: str
    schema_version: str
    plan_identity_hash: str
    entries: tuple[ProtoEMAblationPlanEntry, ...]

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.plan_identity_hash, field_name="plan_identity_hash")
        variants = tuple(entry.ablation_definition.variant_name for entry in self.entries)
        if variants != PROTOEM_ABLATION_VARIANT_ORDER:
            raise AblationPlanError(
                "ablation plan entries must use the exact canonical Phase 6 variant order."
            )
        if len(set(variants)) != len(variants):
            raise AblationPlanError("ablation plan entries must not repeat one variant.")
        if self.plan_identity_hash != hash_protoem_ablation_execution_plan(self):
            raise AblationPlanError(
                "plan_identity_hash does not match deterministic ablation-plan content."
            )


@dataclass(frozen=True, slots=True)
class ProtoEMAblationExecutionResult:
    """One deterministic ablation execution or explicit unsupported/baseline result."""

    schema_name: str
    schema_version: str
    result_identity_hash: str
    plan_entry: ProtoEMAblationPlanEntry
    result_mode: str
    effective_config: ProtoEMConfig | None
    execution_package: ProtoEMExecutionPackage | None
    phase5_reference_schema_name: str | None
    phase5_reference_artifact_hash: str | None
    unsupported_code: str | None
    unsupported_message: str | None

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(self.result_identity_hash, field_name="result_identity_hash")
        if self.result_mode != self.plan_entry.execution_mode:
            raise AblationPlanError("result_mode must match the plan entry execution_mode.")
        if self.result_mode == "protoem_execution":
            if self.execution_package is None or self.effective_config is None:
                raise AblationPlanError(
                    "protoem_execution results require effective_config and execution_package."
                )
            if self.execution_package.ablation_definition != self.plan_entry.ablation_definition:
                raise AblationPlanError(
                    "execution_package.ablation_definition must match the plan entry."
                )
            if self.execution_package.config != self.effective_config:
                raise AblationPlanError("execution_package.config must equal effective_config.")
            if any(
                value is not None
                for value in (
                    self.phase5_reference_schema_name,
                    self.phase5_reference_artifact_hash,
                    self.unsupported_code,
                    self.unsupported_message,
                )
            ):
                raise AblationPlanError(
                    "protoem_execution results must not carry baseline or unsupported metadata."
                )
        elif self.result_mode == "phase5_baseline":
            if self.execution_package is not None:
                raise AblationPlanError(
                    "phase5_baseline results must not carry one ProtoEM execution package."
                )
            if (
                self.phase5_reference_schema_name is None
                or self.phase5_reference_artifact_hash is None
            ):
                raise AblationPlanError(
                    "phase5_baseline results require Phase 5 reference schema/hash."
                )
            _require_sha256(
                self.phase5_reference_artifact_hash,
                field_name="phase5_reference_artifact_hash",
            )
            if self.unsupported_code is not None or self.unsupported_message is not None:
                raise AblationPlanError(
                    "phase5_baseline results must not claim unsupported status."
                )
        else:
            if self.execution_package is not None:
                raise AblationPlanError("unsupported results must not carry execution_package.")
            if self.unsupported_code is None or self.unsupported_message is None:
                raise AblationPlanError(
                    "unsupported results require unsupported_code and unsupported_message."
                )
            if (
                self.phase5_reference_schema_name is not None
                or self.phase5_reference_artifact_hash is not None
            ):
                raise AblationPlanError("unsupported results must not masquerade as Phase 5 links.")
        if self.result_identity_hash != hash_protoem_ablation_execution_result(self):
            raise AblationPlanError(
                "result_identity_hash does not match deterministic ablation-result content."
            )


def build_protoem_ablation_definition(*, variant_name: str) -> ProtoEMAblationDefinition:
    """Build one deterministic ProtoEM ablation definition."""

    overrides = _overrides_for_variant(variant_name)
    payload = {
        "schema_name": PROTOEM_ABLATION_DEFINITION_SCHEMA_NAME,
        "schema_version": PROTOEM_ABLATION_DEFINITION_SCHEMA_VERSION,
        "variant_name": variant_name,
        "overrides": [protoem_ablation_override_to_dict(item) for item in overrides],
    }
    return ProtoEMAblationDefinition(
        schema_name=PROTOEM_ABLATION_DEFINITION_SCHEMA_NAME,
        schema_version=PROTOEM_ABLATION_DEFINITION_SCHEMA_VERSION,
        ablation_definition_hash=sha256_json(payload),
        variant_name=variant_name,
        overrides=overrides,
    )


def build_protoem_ablation_execution_plan() -> ProtoEMAblationExecutionPlan:
    """Build the exact deterministic 12-variant ProtoEM ablation plan."""

    entries = tuple(
        _build_plan_entry(variant_name) for variant_name in PROTOEM_ABLATION_VARIANT_ORDER
    )
    payload = {
        "schema_name": PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_NAME,
        "schema_version": PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_VERSION,
        "entries": [protoem_ablation_plan_entry_identity_payload(item) for item in entries],
    }
    return ProtoEMAblationExecutionPlan(
        schema_name=PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_NAME,
        schema_version=PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_VERSION,
        plan_identity_hash=sha256_json(payload),
        entries=entries,
    )


def build_protoem_ablation_execution_result(
    *,
    plan_entry: ProtoEMAblationPlanEntry,
    base_config: ProtoEMConfig,
    execution_package: ProtoEMExecutionPackage | None = None,
    phase5_reference_schema_name: str | None = None,
    phase5_reference_artifact_hash: str | None = None,
) -> ProtoEMAblationExecutionResult:
    """Build one deterministic ablation execution result."""

    effective_config = apply_protoem_ablation_to_config(
        config=base_config,
        definition=plan_entry.ablation_definition,
    )
    if plan_entry.execution_mode == "unsupported":
        payload = {
            "schema_name": PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_NAME,
            "schema_version": PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_VERSION,
            "plan_entry_hash": plan_entry.entry_identity_hash,
            "result_mode": "unsupported",
            "effective_config_hash": effective_config.config_hash,
            "execution_package_hash": None,
            "phase5_reference_schema_name": None,
            "phase5_reference_artifact_hash": None,
            "unsupported_code": plan_entry.unsupported_code,
            "unsupported_message": plan_entry.unsupported_message,
        }
        return ProtoEMAblationExecutionResult(
            schema_name=PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_NAME,
            schema_version=PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_VERSION,
            result_identity_hash=sha256_json(payload),
            plan_entry=plan_entry,
            result_mode="unsupported",
            effective_config=effective_config,
            execution_package=None,
            phase5_reference_schema_name=None,
            phase5_reference_artifact_hash=None,
            unsupported_code=plan_entry.unsupported_code,
            unsupported_message=plan_entry.unsupported_message,
        )
    if plan_entry.execution_mode == "phase5_baseline":
        if phase5_reference_schema_name is None or phase5_reference_artifact_hash is None:
            raise AblationPlanError(
                "phase5_baseline ablation results require explicit Phase 5 schema/hash inputs."
            )
        payload = {
            "schema_name": PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_NAME,
            "schema_version": PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_VERSION,
            "plan_entry_hash": plan_entry.entry_identity_hash,
            "result_mode": "phase5_baseline",
            "effective_config_hash": effective_config.config_hash,
            "execution_package_hash": None,
            "phase5_reference_schema_name": phase5_reference_schema_name,
            "phase5_reference_artifact_hash": phase5_reference_artifact_hash,
            "unsupported_code": None,
            "unsupported_message": None,
        }
        return ProtoEMAblationExecutionResult(
            schema_name=PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_NAME,
            schema_version=PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_VERSION,
            result_identity_hash=sha256_json(payload),
            plan_entry=plan_entry,
            result_mode="phase5_baseline",
            effective_config=effective_config,
            execution_package=None,
            phase5_reference_schema_name=phase5_reference_schema_name,
            phase5_reference_artifact_hash=phase5_reference_artifact_hash,
            unsupported_code=None,
            unsupported_message=None,
        )
    if execution_package is None:
        raise AblationPlanError(
            "protoem_execution ablation results require one ProtoEMExecutionPackage."
        )
    payload = {
        "schema_name": PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_NAME,
        "schema_version": PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_VERSION,
        "plan_entry_hash": plan_entry.entry_identity_hash,
        "result_mode": "protoem_execution",
        "effective_config_hash": effective_config.config_hash,
        "execution_package_hash": execution_package.package_identity_hash,
        "phase5_reference_schema_name": None,
        "phase5_reference_artifact_hash": None,
        "unsupported_code": None,
        "unsupported_message": None,
    }
    return ProtoEMAblationExecutionResult(
        schema_name=PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_NAME,
        schema_version=PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_VERSION,
        result_identity_hash=sha256_json(payload),
        plan_entry=plan_entry,
        result_mode="protoem_execution",
        effective_config=effective_config,
        execution_package=execution_package,
        phase5_reference_schema_name=None,
        phase5_reference_artifact_hash=None,
        unsupported_code=None,
        unsupported_message=None,
    )


def apply_protoem_ablation_to_config(
    *,
    config: ProtoEMConfig,
    definition: ProtoEMAblationDefinition,
) -> ProtoEMConfig:
    """Apply one explicit ablation definition to one base ProtoEM config."""

    overrides = {item.field_name: item.override_value for item in definition.overrides}
    weights = config.objective_weights
    if "objective_weights.class_balance" in overrides:
        weights = ProtoEMObjectiveWeights(
            schema_name=weights.schema_name,
            schema_version=weights.schema_version,
            support=weights.support,
            query_entropy=weights.query_entropy,
            class_balance=_float_override(
                overrides["objective_weights.class_balance"],
                field_name="objective_weights.class_balance",
            ),
            consistency=weights.consistency,
            proximal=weights.proximal,
        )
    if "objective_weights.proximal" in overrides:
        weights = ProtoEMObjectiveWeights(
            schema_name=weights.schema_name,
            schema_version=weights.schema_version,
            support=weights.support,
            query_entropy=weights.query_entropy,
            class_balance=weights.class_balance,
            consistency=weights.consistency,
            proximal=_float_override(
                overrides["objective_weights.proximal"],
                field_name="objective_weights.proximal",
            ),
        )
    update_schedule = str(overrides.get("update_schedule", config.update_schedule))
    prototype_mode = str(overrides.get("prototype_mode", config.prototype_mode))
    payload = {
        "schema_name": config.schema_name,
        "schema_version": config.schema_version,
        "explicit_seed": config.explicit_seed,
        "max_iterations": config.max_iterations,
        "minimum_iterations": config.minimum_iterations,
        "convergence_tolerance": config.convergence_tolerance,
        "temperature": config.temperature,
        "confidence_threshold": config.confidence_threshold,
        "foreground_prior": config.foreground_prior,
        "update_schedule": update_schedule,
        "prototype_mode": prototype_mode,
        "objective_weights": {
            "schema_name": weights.schema_name,
            "schema_version": weights.schema_version,
            "support": weights.support,
            "query_entropy": weights.query_entropy,
            "class_balance": weights.class_balance,
            "consistency": weights.consistency,
            "proximal": weights.proximal,
        },
        "phase5_initialization_schema_name": config.phase5_initialization_schema_name,
        "phase5_initialization_artifact_hash": config.phase5_initialization_artifact_hash,
        "phase5_prototype_schema_name": config.phase5_prototype_schema_name,
        "phase5_prototype_artifact_hash": config.phase5_prototype_artifact_hash,
        "phase5_inference_schema_name": config.phase5_inference_schema_name,
        "phase5_inference_artifact_hash": config.phase5_inference_artifact_hash,
    }
    return ProtoEMConfig(
        schema_name=config.schema_name,
        schema_version=config.schema_version,
        config_hash=sha256_json(payload),
        explicit_seed=config.explicit_seed,
        max_iterations=config.max_iterations,
        minimum_iterations=config.minimum_iterations,
        convergence_tolerance=config.convergence_tolerance,
        temperature=config.temperature,
        confidence_threshold=config.confidence_threshold,
        foreground_prior=config.foreground_prior,
        update_schedule=update_schedule,
        prototype_mode=prototype_mode,
        objective_weights=weights,
        phase5_initialization_schema_name=config.phase5_initialization_schema_name,
        phase5_initialization_artifact_hash=config.phase5_initialization_artifact_hash,
        phase5_prototype_schema_name=config.phase5_prototype_schema_name,
        phase5_prototype_artifact_hash=config.phase5_prototype_artifact_hash,
        phase5_inference_schema_name=config.phase5_inference_schema_name,
        phase5_inference_artifact_hash=config.phase5_inference_artifact_hash,
    )


def protoem_ablation_plan_entry_identity_payload(
    entry: ProtoEMAblationPlanEntry,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one ablation-plan entry."""

    return {
        "schema_name": entry.schema_name,
        "schema_version": entry.schema_version,
        "ablation_definition_hash": entry.ablation_definition.ablation_definition_hash,
        "execution_mode": entry.execution_mode,
        "phase5_baseline_kind": entry.phase5_baseline_kind,
        "phase5_method_name": entry.phase5_method_name,
        "unsupported_code": entry.unsupported_code,
        "unsupported_message": entry.unsupported_message,
    }


def protoem_ablation_execution_plan_identity_payload(
    plan: ProtoEMAblationExecutionPlan,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one ablation execution plan."""

    return {
        "schema_name": plan.schema_name,
        "schema_version": plan.schema_version,
        "entries": [protoem_ablation_plan_entry_identity_payload(item) for item in plan.entries],
    }


def protoem_ablation_execution_result_identity_payload(
    result: ProtoEMAblationExecutionResult,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one ablation execution result."""

    return {
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "plan_entry_hash": result.plan_entry.entry_identity_hash,
        "result_mode": result.result_mode,
        "effective_config_hash": (
            result.effective_config.config_hash if result.effective_config is not None else None
        ),
        "execution_package_hash": (
            result.execution_package.package_identity_hash
            if result.execution_package is not None
            else None
        ),
        "phase5_reference_schema_name": result.phase5_reference_schema_name,
        "phase5_reference_artifact_hash": result.phase5_reference_artifact_hash,
        "unsupported_code": result.unsupported_code,
        "unsupported_message": result.unsupported_message,
    }


def hash_protoem_ablation_plan_entry(entry: ProtoEMAblationPlanEntry) -> str:
    """Return the deterministic hash for one ablation-plan entry."""

    return sha256_json(protoem_ablation_plan_entry_identity_payload(entry))


def hash_protoem_ablation_execution_plan(plan: ProtoEMAblationExecutionPlan) -> str:
    """Return the deterministic hash for one ablation execution plan."""

    return sha256_json(protoem_ablation_execution_plan_identity_payload(plan))


def hash_protoem_ablation_execution_result(result: ProtoEMAblationExecutionResult) -> str:
    """Return the deterministic hash for one ablation execution result."""

    return sha256_json(protoem_ablation_execution_result_identity_payload(result))


def protoem_ablation_override_to_dict(
    override: ProtoEMAblationOverride,
) -> dict[str, JsonValue]:
    return {
        "schema_name": override.schema_name,
        "schema_version": override.schema_version,
        "field_name": override.field_name,
        "override_value": override.override_value,
    }


def query_label_not_part_of_ablation_api() -> bool:
    """Return whether query labels are absent from the public ablation API."""

    return True


def _build_plan_entry(variant_name: str) -> ProtoEMAblationPlanEntry:
    definition = build_protoem_ablation_definition(variant_name=variant_name)
    execution_mode, baseline_kind, phase5_method_name, unsupported_code, unsupported_message = (
        _execution_semantics_for_variant(variant_name)
    )
    payload = {
        "schema_name": PROTOEM_ABLATION_PLAN_ENTRY_SCHEMA_NAME,
        "schema_version": PROTOEM_ABLATION_PLAN_ENTRY_SCHEMA_VERSION,
        "ablation_definition_hash": definition.ablation_definition_hash,
        "execution_mode": execution_mode,
        "phase5_baseline_kind": baseline_kind,
        "phase5_method_name": phase5_method_name,
        "unsupported_code": unsupported_code,
        "unsupported_message": unsupported_message,
    }
    return ProtoEMAblationPlanEntry(
        schema_name=PROTOEM_ABLATION_PLAN_ENTRY_SCHEMA_NAME,
        schema_version=PROTOEM_ABLATION_PLAN_ENTRY_SCHEMA_VERSION,
        entry_identity_hash=sha256_json(payload),
        ablation_definition=definition,
        execution_mode=execution_mode,
        phase5_baseline_kind=baseline_kind,
        phase5_method_name=phase5_method_name,
        unsupported_code=unsupported_code,
        unsupported_message=unsupported_message,
    )


def _execution_semantics_for_variant(
    variant_name: str,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    if variant_name == "no_retrieval":
        return ("phase5_baseline", "no_retrieval", NO_RETRIEVAL_METHOD, None, None)
    if variant_name == "no_transduction":
        return ("phase5_baseline", "prototype_only_inference", None, None, None)
    if variant_name == "multiple_prototypes":
        return (
            "unsupported",
            None,
            None,
            "multiple_prototypes_unsupported",
            "multiple_prototypes remains unsupported until the dedicated "
            "multi-prototype execution substage.",
        )
    if variant_name == "learned_update_schedule":
        return ("protoem_execution", None, None, None, None)
    if variant_name in {"head_only", "decoder_only", "full_finetune"}:
        return (
            "unsupported",
            None,
            None,
            f"{variant_name}_provenance_only",
            f"{variant_name} remains provenance-only until the dedicated "
            "adaptation execution substage.",
        )
    return ("protoem_execution", None, None, None, None)


def _overrides_for_variant(variant_name: str) -> tuple[ProtoEMAblationOverride, ...]:
    if variant_name not in PROTOEM_ABLATION_VARIANT_ORDER:
        raise AblationPlanError(
            f"variant_name must be one of {list(PROTOEM_ABLATION_VARIANT_ORDER)!r}."
        )
    override_specs: tuple[tuple[str, JsonValue], ...]
    if variant_name == "no_class_balance":
        override_specs = (("objective_weights.class_balance", 0.0),)
    elif variant_name == "no_proximal":
        override_specs = (("objective_weights.proximal", 0.0),)
    elif variant_name == "single_prototype":
        override_specs = (("prototype_mode", "single_prototype"),)
    elif variant_name == "multiple_prototypes":
        override_specs = (("prototype_mode", "multiple_prototypes"),)
    elif variant_name == "fixed_update_schedule":
        override_specs = (("update_schedule", "fixed_em_like"),)
    elif variant_name == "learned_update_schedule":
        override_specs = (("update_schedule", "learned_positive_step"),)
    else:
        override_specs = ()
    return tuple(
        ProtoEMAblationOverride(
            schema_name=PROTOEM_ABLATION_OVERRIDE_SCHEMA_NAME,
            schema_version=PROTOEM_ABLATION_OVERRIDE_SCHEMA_VERSION,
            field_name=field_name,
            override_value=override_value,
        )
        for field_name, override_value in override_specs
    )


def _require_schema_name(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise AblationPlanError(f"{field_name} must equal {expected!r}.")


def _require_schema_version(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise AblationPlanError(f"{field_name} must equal {expected!r}.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise AblationPlanError(f"{field_name} must be one lowercase hexadecimal SHA-256 digest.")


def _float_override(value: JsonValue, *, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise AblationPlanError(f"{field_name} override must be numeric.")
    return float(value)


__all__ = [
    "AblationPlanError",
    "PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_NAME",
    "PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_VERSION",
    "PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_NAME",
    "PROTOEM_ABLATION_EXECUTION_RESULT_SCHEMA_VERSION",
    "PROTOEM_ABLATION_PLAN_ENTRY_SCHEMA_NAME",
    "PROTOEM_ABLATION_PLAN_ENTRY_SCHEMA_VERSION",
    "PROTOEM_ABLATION_VARIANT_ORDER",
    "ProtoEMAblationError",
    "ProtoEMAblationExecutionPlan",
    "ProtoEMAblationExecutionResult",
    "ProtoEMAblationPlanEntry",
    "UnsupportedAblationExecutionError",
    "apply_protoem_ablation_to_config",
    "build_protoem_ablation_definition",
    "build_protoem_ablation_execution_plan",
    "build_protoem_ablation_execution_result",
    "hash_protoem_ablation_execution_plan",
    "hash_protoem_ablation_execution_result",
    "hash_protoem_ablation_plan_entry",
    "protoem_ablation_execution_plan_identity_payload",
    "protoem_ablation_execution_result_identity_payload",
    "protoem_ablation_plan_entry_identity_payload",
    "query_label_not_part_of_ablation_api",
]
