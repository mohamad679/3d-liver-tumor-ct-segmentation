"""Deterministic Phase 6 ProtoEM-CT publication, CLI settings, and synthetic execution."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.data import InvalidDatasetRootError, validate_explicit_external_output_root
from protoem_ct.models.interfaces import (
    EmbeddingMetadata,
    FeatureEncoding3D,
    FeatureResolution3D,
)
from protoem_ct.protoem.ablation import (
    PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_NAME,
    PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_VERSION,
    ProtoEMAblationExecutionPlan,
    ProtoEMAblationPlanEntry,
    build_protoem_ablation_definition,
    build_protoem_ablation_execution_plan,
)
from protoem_ct.protoem.artifacts import (
    PROTOEM_CONFIG_SCHEMA_NAME,
    PROTOEM_CONFIG_SCHEMA_VERSION,
    PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
    PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
    ProtoEMConfig,
    ProtoEMObjectiveWeights,
    protoem_collapse_record_to_json,
    protoem_config_from_mapping,
    protoem_config_to_dict,
    protoem_objective_trace_from_json,
    protoem_objective_trace_to_json,
    protoem_run_summary_from_json,
    protoem_run_summary_to_json,
    protoem_stopping_record_from_json,
    protoem_stopping_record_to_json,
)
from protoem_ct.protoem.contracts import (
    PROTOEM_INFERENCE_INPUT_SCHEMA_NAME,
    PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION,
    PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME,
    PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION,
    PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_NAME,
    PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_VERSION,
    PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
    PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
    PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_NAME,
    PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_VERSION,
    PROTOEM_SUPPORT_PROVENANCE_SCHEMA_NAME,
    PROTOEM_SUPPORT_PROVENANCE_SCHEMA_VERSION,
    ProtoEMInferenceInput,
    ProtoEMPhase5ArtifactReference,
    ProtoEMPhase5ReferenceBundle,
    ProtoEMPrototypeInput,
    ProtoEMQueryFeatureInput,
    ProtoEMSupportProvenance,
)
from protoem_ct.protoem.inference import (
    ProtoEMExecutionPackage,
    build_protoem_execution_package,
    protoem_final_inference_to_dict,
)
from protoem_ct.protoem.initialize import (
    ProtoEMInitializationBundle,
    build_protoem_initialization_bundle,
    protoem_initialization_bundle_to_json,
)
from protoem_ct.protoem.learned_schedule import (
    PROTOEM_PARAMETERIZED_POSITIVE_STEP_SCHEDULE_NAME,
    ProtoEMPositiveStepSchedule,
    build_protoem_positive_step_schedule,
    protoem_positive_step_schedule_identity_payload,
)
from protoem_ct.protoem.optimize import run_protoem_optimization
from protoem_ct.retrieval.inference import (
    PrototypeInferenceResult,
    infer_query_from_prototypes,
)
from protoem_ct.retrieval.prototypes import (
    SupportPrototypeInput,
    SupportPrototypeMemory,
    build_support_prototype_memory,
)

PHASE6_CONFIG_ROOT_KEY: Final[str] = "phase6_protoem_ct"
PHASE6_CONFIG_SCHEMA_VERSION: Final[str] = "v1"
PHASE6_CONVERGENCE_PLOT_NAME: Final[str] = "convergence_plot.png"
PHASE6_ABLATION_PLAN_NAME: Final[str] = "ablation_plan.json"
PHASE6_EFFECTIVE_CONFIG_NAME: Final[str] = "effective_config.json"
PHASE6_INITIALIZATION_SUMMARY_NAME: Final[str] = "initialization_summary.json"
PHASE6_OBJECTIVE_TRACE_NAME: Final[str] = "objective_trace.json"
PHASE6_STOPPING_RECORD_NAME: Final[str] = "stopping_record.json"
PHASE6_COLLAPSE_RECORD_NAME: Final[str] = "collapse_record.json"
PHASE6_FINAL_INFERENCE_NAME: Final[str] = "final_inference.json"
PHASE6_RUN_SUMMARY_NAME: Final[str] = "run_summary.json"
PHASE6_SUMMARY_MARKDOWN_NAME: Final[str] = "phase6_summary.md"

_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_MASK_DTYPE: Final[np.dtype[np.uint8]] = np.dtype(np.uint8)
_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_PHASE6_EXPECTED_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "synthetic_mode_only",
        "explicit_seed",
        "max_iterations",
        "minimum_iterations",
        "convergence_tolerance",
        "temperature",
        "confidence_threshold",
        "foreground_prior",
        "update_schedule",
        "prototype_mode",
        "objective_weights",
        "phase5_initialization_schema_name",
        "phase5_prototype_schema_name",
        "phase5_inference_schema_name",
        "positive_step_schedule",
    }
)
_OBJECTIVE_WEIGHT_KEYS: Final[frozenset[str]] = frozenset(
    {"support", "query_entropy", "class_balance", "consistency", "proximal"}
)
_POSITIVE_STEP_KEYS: Final[frozenset[str]] = frozenset({"epsilon", "raw_step_parameters"})


class Phase6PublicationError(ValueError):
    """Base error for deterministic Phase 6 synthetic execution and publication."""


class Phase6PublicationConfigError(Phase6PublicationError):
    """Raised when the Phase 6 config file is malformed or unsupported."""


class Phase6PublicationPathError(Phase6PublicationError):
    """Raised when the output root violates the explicit external publication contract."""


class Phase6PublicationCollisionError(Phase6PublicationError):
    """Raised when publication would overwrite non-identical artifacts."""


class Phase6PublicationIOError(Phase6PublicationError):
    """Raised when staged filesystem publication fails."""


class Phase6PublicationValidationError(Phase6PublicationError):
    """Raised when persisted JSON surfaces are invalid for markdown or plot derivation."""


@dataclass(frozen=True, slots=True)
class Phase6PositiveStepScheduleSettings:
    """Explicit positive-step parameters supplied through the Phase 6 YAML config."""

    epsilon: float
    raw_step_parameters: tuple[float, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise Phase6PublicationConfigError("positive_step_schedule.epsilon must be positive.")
        if not self.raw_step_parameters:
            raise Phase6PublicationConfigError(
                "positive_step_schedule.raw_step_parameters must be non-empty."
            )
        for index, value in enumerate(self.raw_step_parameters):
            if not math.isfinite(value):
                raise Phase6PublicationConfigError(
                    f"positive_step_schedule.raw_step_parameters[{index}] must be finite."
                )


@dataclass(frozen=True, slots=True)
class Phase6PublicationSettings:
    """Bounded synthetic-only Phase 6 execution settings loaded from YAML."""

    schema_version: str
    synthetic_mode_only: bool
    explicit_seed: int
    max_iterations: int
    minimum_iterations: int
    convergence_tolerance: float
    temperature: float
    confidence_threshold: float
    foreground_prior: float
    update_schedule: str
    prototype_mode: str
    objective_weights: ProtoEMObjectiveWeights
    phase5_initialization_schema_name: str
    phase5_prototype_schema_name: str
    phase5_inference_schema_name: str
    positive_step_schedule: Phase6PositiveStepScheduleSettings | None

    def __post_init__(self) -> None:
        if self.schema_version != PHASE6_CONFIG_SCHEMA_VERSION:
            raise Phase6PublicationConfigError(
                f"schema_version must equal {PHASE6_CONFIG_SCHEMA_VERSION!r}."
            )
        if not self.synthetic_mode_only:
            raise Phase6PublicationConfigError(
                "Phase 6 substage 8 supports synthetic_mode_only=true only."
            )
        if self.prototype_mode != "single_prototype":
            raise Phase6PublicationConfigError(
                "Phase 6 substage 8 supports prototype_mode='single_prototype' only."
            )
        if self.update_schedule == "fixed_em_like" and self.positive_step_schedule is not None:
            raise Phase6PublicationConfigError(
                "positive_step_schedule must be omitted for fixed_em_like execution."
            )
        if (
            self.update_schedule == PROTOEM_PARAMETERIZED_POSITIVE_STEP_SCHEDULE_NAME
            and self.positive_step_schedule is None
        ):
            raise Phase6PublicationConfigError(
                "learned_positive_step requires one explicit positive_step_schedule block."
            )
        if (
            self.positive_step_schedule is not None
            and len(self.positive_step_schedule.raw_step_parameters) != self.max_iterations
        ):
            raise Phase6PublicationConfigError(
                "positive_step_schedule.raw_step_parameters length must equal max_iterations."
            )


@dataclass(frozen=True, slots=True)
class Phase6PublicationResult:
    """Deterministic summary of one synthetic Phase 6 ProtoEM publication."""

    output_root: Path
    reused_existing_output: bool
    execution_status: str
    config_hash: str
    initialization_identity_hash: str
    stopping_reason: str
    run_summary_path: Path
    effective_config_path: Path
    initialization_summary_path: Path
    objective_trace_path: Path
    stopping_record_path: Path
    collapse_record_path: Path | None
    final_inference_path: Path | None
    ablation_plan_path: Path
    convergence_plot_path: Path
    summary_markdown_path: Path


def load_phase6_protoem_settings(config_path: Path) -> Phase6PublicationSettings:
    """Load one bounded synthetic-only Phase 6 ProtoEM YAML config."""

    try:
        raw_config = OmegaConf.load(config_path)
    except OSError as exc:
        raise Phase6PublicationConfigError(
            f"failed to read Phase 6 config at {config_path}: {exc}"
        ) from exc
    except Exception as exc:
        raise Phase6PublicationConfigError(
            f"failed to parse Phase 6 config at {config_path}: {exc}"
        ) from exc
    if not isinstance(raw_config, DictConfig):
        raise Phase6PublicationConfigError("Phase 6 config must contain a mapping.")
    container = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(container, dict):
        raise Phase6PublicationConfigError("Phase 6 config must resolve to a mapping.")
    resolved = cast(dict[str, object], container)
    if set(resolved) != {PHASE6_CONFIG_ROOT_KEY}:
        raise Phase6PublicationConfigError(
            f"Phase 6 config must contain only {PHASE6_CONFIG_ROOT_KEY!r}."
        )
    settings_mapping = resolved[PHASE6_CONFIG_ROOT_KEY]
    if not isinstance(settings_mapping, dict):
        raise Phase6PublicationConfigError("phase6_protoem_ct must resolve to a mapping.")
    typed = cast(dict[str, object], settings_mapping)
    unknown_keys = set(typed) - _PHASE6_EXPECTED_CONFIG_KEYS
    missing_keys = _PHASE6_EXPECTED_CONFIG_KEYS - set(typed)
    if unknown_keys:
        raise Phase6PublicationConfigError(
            f"unknown Phase 6 setting keys: {sorted(unknown_keys)!r}."
        )
    if missing_keys:
        raise Phase6PublicationConfigError(
            f"missing Phase 6 setting keys: {sorted(missing_keys)!r}."
        )
    weights_mapping = _expect_mapping(typed["objective_weights"], field_name="objective_weights")
    if set(weights_mapping) != _OBJECTIVE_WEIGHT_KEYS:
        raise Phase6PublicationConfigError(
            "objective_weights must contain exactly support, query_entropy, "
            "class_balance, consistency, and proximal."
        )
    weights = ProtoEMObjectiveWeights(
        schema_name=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_NAME,
        schema_version=PROTOEM_OBJECTIVE_WEIGHTS_SCHEMA_VERSION,
        support=_expect_float(weights_mapping["support"], field_name="objective_weights.support"),
        query_entropy=_expect_float(
            weights_mapping["query_entropy"],
            field_name="objective_weights.query_entropy",
        ),
        class_balance=_expect_float(
            weights_mapping["class_balance"],
            field_name="objective_weights.class_balance",
        ),
        consistency=_expect_float(
            weights_mapping["consistency"],
            field_name="objective_weights.consistency",
        ),
        proximal=_expect_float(
            weights_mapping["proximal"],
            field_name="objective_weights.proximal",
        ),
    )
    positive_step_schedule = _parse_optional_positive_step_schedule(typed["positive_step_schedule"])
    return Phase6PublicationSettings(
        schema_version=_expect_string(typed["schema_version"], field_name="schema_version"),
        synthetic_mode_only=_expect_bool(
            typed["synthetic_mode_only"],
            field_name="synthetic_mode_only",
        ),
        explicit_seed=_expect_int(typed["explicit_seed"], field_name="explicit_seed"),
        max_iterations=_expect_int(typed["max_iterations"], field_name="max_iterations"),
        minimum_iterations=_expect_int(
            typed["minimum_iterations"],
            field_name="minimum_iterations",
        ),
        convergence_tolerance=_expect_float(
            typed["convergence_tolerance"],
            field_name="convergence_tolerance",
        ),
        temperature=_expect_float(typed["temperature"], field_name="temperature"),
        confidence_threshold=_expect_float(
            typed["confidence_threshold"],
            field_name="confidence_threshold",
        ),
        foreground_prior=_expect_float(
            typed["foreground_prior"],
            field_name="foreground_prior",
        ),
        update_schedule=_expect_string(typed["update_schedule"], field_name="update_schedule"),
        prototype_mode=_expect_string(typed["prototype_mode"], field_name="prototype_mode"),
        objective_weights=weights,
        phase5_initialization_schema_name=_expect_string(
            typed["phase5_initialization_schema_name"],
            field_name="phase5_initialization_schema_name",
        ),
        phase5_prototype_schema_name=_expect_string(
            typed["phase5_prototype_schema_name"],
            field_name="phase5_prototype_schema_name",
        ),
        phase5_inference_schema_name=_expect_string(
            typed["phase5_inference_schema_name"],
            field_name="phase5_inference_schema_name",
        ),
        positive_step_schedule=positive_step_schedule,
    )


def build_phase6_synthetic_execution_package(
    settings: Phase6PublicationSettings,
) -> ProtoEMExecutionPackage:
    """Build one deterministic synthetic ProtoEM execution package."""

    bundle, config, positive_step_schedule = _build_synthetic_initialization_and_config(settings)
    optimization_result = run_protoem_optimization(
        config=config,
        initialization_bundle=bundle,
        positive_step_schedule=positive_step_schedule,
    )
    return build_protoem_execution_package(
        config=config,
        ablation_definition=build_protoem_ablation_definition(variant_name="full_protoem"),
        initialization_bundle=bundle,
        optimization_result=optimization_result,
    )


def build_phase6_convergence_plot_png(objective_trace_json: bytes | str) -> bytes:
    """Render one deterministic convergence plot strictly from objective_trace.json bytes."""

    trace = protoem_objective_trace_from_json(objective_trace_json)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise Phase6PublicationIOError("failed to import matplotlib for Phase 6 plotting.") from exc

    iterations = [record.iteration_index for record in trace.iterations]
    figure, axis = plt.subplots(figsize=(8.0, 4.5), dpi=120)
    axis.plot(
        iterations,
        [record.total_objective for record in trace.iterations],
        label="total",
    )
    axis.plot(
        iterations, [record.support_objective for record in trace.iterations], label="support"
    )
    axis.plot(
        iterations,
        [record.query_entropy_objective for record in trace.iterations],
        label="query_entropy",
    )
    axis.plot(
        iterations,
        [record.class_balance_objective for record in trace.iterations],
        label="class_balance",
    )
    axis.plot(
        iterations,
        [record.consistency_objective for record in trace.iterations],
        label="consistency",
    )
    axis.plot(
        iterations,
        [record.proximal_objective for record in trace.iterations],
        label="proximal",
    )
    axis.set_xlabel("iteration")
    axis.set_ylabel("objective")
    axis.set_title("ProtoEM-CT objective trace")
    axis.set_xticks(iterations)
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(loc="best", frameon=False)
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(
        output,
        format="png",
        dpi=120,
        metadata={"Software": "protoem_ct"},
    )
    plt.close(figure)
    return output.getvalue()


def render_phase6_summary_markdown_from_json_artifacts(
    *,
    effective_config_json: bytes,
    initialization_summary_json: bytes,
    objective_trace_json: bytes,
    stopping_record_json: bytes,
    run_summary_json: bytes,
    final_inference_json: bytes | None,
) -> bytes:
    """Render one deterministic Phase 6 markdown summary strictly from validated JSON bytes."""

    effective = _validate_effective_config_json(effective_config_json)
    initialization = _validate_initialization_summary_json(initialization_summary_json)
    trace = protoem_objective_trace_from_json(objective_trace_json)
    stopping = protoem_stopping_record_from_json(stopping_record_json)
    run_summary = protoem_run_summary_from_json(run_summary_json)
    final_inference = (
        _validate_final_inference_json(final_inference_json)
        if final_inference_json is not None
        else None
    )

    config = cast(ProtoEMConfig, effective["protoem_config"])
    lines = [
        "# Phase 6 ProtoEM-CT Summary",
        "",
        f"- configuration_identity: `{config.config_hash}`",
        f"- initialization_identity: `{initialization['initialization_identity_hash']}`",
        f"- update_schedule: `{config.update_schedule}`",
        f"- completed_iteration_count: `{stopping.completed_iteration_count}`",
        f"- stopping_reason: `{stopping.stop_reason}`",
        f"- converged: `{str(stopping.converged).lower()}`",
        f"- failed: `{str(stopping.failed).lower()}`",
        f"- final_state_identity: `{run_summary.final_output_artifact_hash or 'unavailable'}`",
        (
            f"- final_inference_identity: `{final_inference['inference_identity_hash']}`"
            if final_inference is not None
            else "- final_inference_identity: `unavailable`"
        ),
        "",
        (
            "- parameterized_positive_step: explicit caller-supplied parameters were used; "
            "they were not learned from data."
            if effective["parameterized_positive_step_parameters_supplied"]
            else "- parameterized_positive_step: no explicit positive-step parameters were "
            "supplied for this run; Phase 6 does not learn them from data."
        ),
        "",
        (
            "| iteration | support | query_entropy | class_balance | consistency | "
            "proximal | total | confident_voxels | foreground_assignments | "
            "background_assignments | foreground_fraction | convergence_delta |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in trace.iterations:
        convergence_delta = (
            "null" if record.convergence_delta is None else f"{record.convergence_delta:.12g}"
        )
        lines.append(
            "| "
            f"{record.iteration_index} | "
            f"{record.support_objective:.12g} | "
            f"{record.query_entropy_objective:.12g} | "
            f"{record.class_balance_objective:.12g} | "
            f"{record.consistency_objective:.12g} | "
            f"{record.proximal_objective:.12g} | "
            f"{record.total_objective:.12g} | "
            f"{record.confident_voxel_count} | "
            f"{record.foreground_assignment_count} | "
            f"{record.background_assignment_count} | "
            f"{record.foreground_fraction:.12g} | "
            f"{convergence_delta} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def run_and_publish_phase6_protoem(
    *,
    output_root: Path,
    settings: Phase6PublicationSettings,
) -> Phase6PublicationResult:
    """Run bounded synthetic Phase 6 ProtoEM and publish deterministic artifacts."""

    execution_package = build_phase6_synthetic_execution_package(settings)
    if execution_package.objective_trace is None or execution_package.stopping_record is None:
        raise Phase6PublicationValidationError(
            "Phase 6 publication requires one objective_trace and one stopping_record."
        )
    if execution_package.run_summary is None:
        raise Phase6PublicationValidationError(
            "Phase 6 publication requires one run_summary when artifacts are published."
        )

    positive_step_payload: dict[str, JsonValue] | None = None
    if execution_package.optimization_result.learned_schedule_result is not None:
        schedule = _build_positive_step_schedule_from_settings(settings)
        if schedule is None:
            raise Phase6PublicationValidationError(
                "learned schedule execution requires one explicit positive-step schedule."
            )
        positive_step_payload = {
            "schedule_identity_hash": schedule.schedule_identity_hash,
            **protoem_positive_step_schedule_identity_payload(schedule),
        }

    effective_config_payload = {
        PHASE6_CONFIG_ROOT_KEY: {
            "schema_version": settings.schema_version,
            "synthetic_mode_only": settings.synthetic_mode_only,
            "protoem_config": cast(JsonValue, protoem_config_to_dict(execution_package.config)),
            "positive_step_schedule": cast(JsonValue, positive_step_payload),
            "parameterized_positive_step_parameters_supplied": (positive_step_payload is not None),
        }
    }
    effective_config_json = canonical_json_bytes(effective_config_payload) + b"\n"
    initialization_summary_json = protoem_initialization_bundle_to_json(
        execution_package.initialization_bundle
    )
    objective_trace_json = protoem_objective_trace_to_json(execution_package.objective_trace)
    stopping_record_json = protoem_stopping_record_to_json(execution_package.stopping_record)
    collapse_record_json = (
        protoem_collapse_record_to_json(execution_package.collapse_record)
        if execution_package.collapse_record is not None
        else None
    )
    final_inference_json = (
        canonical_json_bytes(
            protoem_final_inference_to_dict(execution_package.final_inference_result)
        )
        + b"\n"
        if execution_package.final_inference_result is not None
        else None
    )
    run_summary_json = protoem_run_summary_to_json(execution_package.run_summary)
    ablation_plan_json = _phase6_ablation_plan_to_json(build_protoem_ablation_execution_plan())

    summary_markdown = render_phase6_summary_markdown_from_json_artifacts(
        effective_config_json=effective_config_json,
        initialization_summary_json=initialization_summary_json,
        objective_trace_json=objective_trace_json,
        stopping_record_json=stopping_record_json,
        run_summary_json=run_summary_json,
        final_inference_json=final_inference_json,
    )
    convergence_plot_png = build_phase6_convergence_plot_png(objective_trace_json)

    artifact_bytes = {
        PHASE6_EFFECTIVE_CONFIG_NAME: effective_config_json,
        PHASE6_INITIALIZATION_SUMMARY_NAME: initialization_summary_json,
        PHASE6_OBJECTIVE_TRACE_NAME: objective_trace_json,
        PHASE6_STOPPING_RECORD_NAME: stopping_record_json,
        PHASE6_RUN_SUMMARY_NAME: run_summary_json,
        PHASE6_ABLATION_PLAN_NAME: ablation_plan_json,
        PHASE6_CONVERGENCE_PLOT_NAME: convergence_plot_png,
        PHASE6_SUMMARY_MARKDOWN_NAME: summary_markdown,
    }
    if collapse_record_json is not None:
        artifact_bytes[PHASE6_COLLAPSE_RECORD_NAME] = collapse_record_json
    if final_inference_json is not None:
        artifact_bytes[PHASE6_FINAL_INFERENCE_NAME] = final_inference_json

    validated_output_root = _validate_phase6_output_root(output_root)
    reused_existing_output = _publish_directory_tree_if_absent_or_equal(
        output_root=validated_output_root,
        artifact_bytes=artifact_bytes,
    )
    return Phase6PublicationResult(
        output_root=validated_output_root,
        reused_existing_output=reused_existing_output,
        execution_status=execution_package.optimization_result.execution_status,
        config_hash=execution_package.config.config_hash,
        initialization_identity_hash=execution_package.initialization_bundle.initialization_identity_hash,
        stopping_reason=execution_package.stopping_record.stop_reason,
        run_summary_path=validated_output_root / PHASE6_RUN_SUMMARY_NAME,
        effective_config_path=validated_output_root / PHASE6_EFFECTIVE_CONFIG_NAME,
        initialization_summary_path=validated_output_root / PHASE6_INITIALIZATION_SUMMARY_NAME,
        objective_trace_path=validated_output_root / PHASE6_OBJECTIVE_TRACE_NAME,
        stopping_record_path=validated_output_root / PHASE6_STOPPING_RECORD_NAME,
        collapse_record_path=(
            validated_output_root / PHASE6_COLLAPSE_RECORD_NAME
            if collapse_record_json is not None
            else None
        ),
        final_inference_path=(
            validated_output_root / PHASE6_FINAL_INFERENCE_NAME
            if final_inference_json is not None
            else None
        ),
        ablation_plan_path=validated_output_root / PHASE6_ABLATION_PLAN_NAME,
        convergence_plot_path=validated_output_root / PHASE6_CONVERGENCE_PLOT_NAME,
        summary_markdown_path=validated_output_root / PHASE6_SUMMARY_MARKDOWN_NAME,
    )


def _build_synthetic_initialization_and_config(
    settings: Phase6PublicationSettings,
) -> tuple[ProtoEMInitializationBundle, ProtoEMConfig, ProtoEMPositiveStepSchedule | None]:
    metadata = _synthetic_metadata(input_identity="query_case_001")
    support_inputs = _synthetic_supports()
    prototype_memory = build_support_prototype_memory(supports=support_inputs)
    query_feature_encoding = FeatureEncoding3D(
        feature_data=np.asarray(
            [[[[[1.0, 0.0, 0.85]]], [[[0.0, 1.0, 0.15]]]]],
            dtype=np.float32,
        ),
        metadata=metadata,
    )
    inference_result = infer_query_from_prototypes(
        query_feature_encoding=query_feature_encoding,
        foreground_prototype=prototype_memory.foreground_prototype,
        background_prototype=prototype_memory.background_prototype,
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        query_dataset_manifest_hash=_synthetic_dataset_manifest_hash(),
        emit_confidence_margin=True,
    )
    references = _build_synthetic_phase5_references(
        settings=settings,
        prototype_memory=prototype_memory,
        inference_result=inference_result,
    )
    config = _build_protoem_config(settings=settings, references=references)
    query_input = ProtoEMQueryFeatureInput(
        schema_name=PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_QUERY_FEATURE_INPUT_SCHEMA_VERSION,
        query_feature_encoding=query_feature_encoding,
        query_identity="query_case_001",
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        dataset_manifest_hash=_synthetic_dataset_manifest_hash(),
    )
    prototype_input = ProtoEMPrototypeInput(
        schema_name=PROTOEM_PROTOTYPE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_PROTOTYPE_INPUT_SCHEMA_VERSION,
        prototype_memory=prototype_memory,
        support_provenance=_synthetic_support_provenance(),
    )
    inference_input = ProtoEMInferenceInput(
        schema_name=PROTOEM_INFERENCE_INPUT_SCHEMA_NAME,
        schema_version=PROTOEM_INFERENCE_INPUT_SCHEMA_VERSION,
        inference_result=inference_result,
    )
    bundle = build_protoem_initialization_bundle(
        config=config,
        query_input=query_input,
        prototype_input=prototype_input,
        inference_input=inference_input,
        phase5_references=references,
    )
    return bundle, config, _build_positive_step_schedule_from_settings(settings)


def _build_protoem_config(
    *,
    settings: Phase6PublicationSettings,
    references: ProtoEMPhase5ReferenceBundle,
) -> ProtoEMConfig:
    payload = {
        "schema_name": PROTOEM_CONFIG_SCHEMA_NAME,
        "schema_version": PROTOEM_CONFIG_SCHEMA_VERSION,
        "explicit_seed": settings.explicit_seed,
        "max_iterations": settings.max_iterations,
        "minimum_iterations": settings.minimum_iterations,
        "convergence_tolerance": settings.convergence_tolerance,
        "temperature": settings.temperature,
        "confidence_threshold": settings.confidence_threshold,
        "foreground_prior": settings.foreground_prior,
        "update_schedule": settings.update_schedule,
        "prototype_mode": settings.prototype_mode,
        "objective_weights": {
            "schema_name": settings.objective_weights.schema_name,
            "schema_version": settings.objective_weights.schema_version,
            "support": settings.objective_weights.support,
            "query_entropy": settings.objective_weights.query_entropy,
            "class_balance": settings.objective_weights.class_balance,
            "consistency": settings.objective_weights.consistency,
            "proximal": settings.objective_weights.proximal,
        },
        "phase5_initialization_schema_name": settings.phase5_initialization_schema_name,
        "phase5_initialization_artifact_hash": references.initialization_reference.artifact_hash,
        "phase5_prototype_schema_name": settings.phase5_prototype_schema_name,
        "phase5_prototype_artifact_hash": references.prototype_reference.artifact_hash,
        "phase5_inference_schema_name": settings.phase5_inference_schema_name,
        "phase5_inference_artifact_hash": references.inference_reference.artifact_hash,
    }
    return ProtoEMConfig(
        schema_name=PROTOEM_CONFIG_SCHEMA_NAME,
        schema_version=PROTOEM_CONFIG_SCHEMA_VERSION,
        config_hash=sha256_json(payload),
        explicit_seed=settings.explicit_seed,
        max_iterations=settings.max_iterations,
        minimum_iterations=settings.minimum_iterations,
        convergence_tolerance=settings.convergence_tolerance,
        temperature=settings.temperature,
        confidence_threshold=settings.confidence_threshold,
        foreground_prior=settings.foreground_prior,
        update_schedule=settings.update_schedule,
        prototype_mode=settings.prototype_mode,
        objective_weights=settings.objective_weights,
        phase5_initialization_schema_name=settings.phase5_initialization_schema_name,
        phase5_initialization_artifact_hash=references.initialization_reference.artifact_hash,
        phase5_prototype_schema_name=settings.phase5_prototype_schema_name,
        phase5_prototype_artifact_hash=references.prototype_reference.artifact_hash,
        phase5_inference_schema_name=settings.phase5_inference_schema_name,
        phase5_inference_artifact_hash=references.inference_reference.artifact_hash,
    )


def _build_positive_step_schedule_from_settings(
    settings: Phase6PublicationSettings,
) -> ProtoEMPositiveStepSchedule | None:
    if settings.positive_step_schedule is None:
        return None
    return build_protoem_positive_step_schedule(
        raw_step_parameters=settings.positive_step_schedule.raw_step_parameters,
        epsilon=settings.positive_step_schedule.epsilon,
        max_iteration_compatibility=settings.max_iterations,
    )


def _build_synthetic_phase5_references(
    *,
    settings: Phase6PublicationSettings,
    prototype_memory: SupportPrototypeMemory,
    inference_result: PrototypeInferenceResult,
) -> ProtoEMPhase5ReferenceBundle:
    prototype_artifact_hash = sha256_json(
        {
            "schema_name": settings.phase5_prototype_schema_name,
            "foreground_prototype_identity": (
                prototype_memory.foreground_prototype.prototype_identity_sha256
            ),
            "background_prototype_identity": (
                prototype_memory.background_prototype.prototype_identity_sha256
            ),
        }
    )
    initialization_artifact_hash = sha256_json(
        {
            "schema_name": settings.phase5_initialization_schema_name,
            "query_identity": "query_case_001",
            "support_identifiers": ["support_case_001", "support_case_002"],
            "prototype_artifact_hash": prototype_artifact_hash,
            "inference_artifact_hash": inference_result.inference_identity_sha256,
        }
    )
    return ProtoEMPhase5ReferenceBundle(
        schema_name=PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_NAME,
        schema_version=PROTOEM_PHASE5_REFERENCE_BUNDLE_SCHEMA_VERSION,
        initialization_reference=ProtoEMPhase5ArtifactReference(
            schema_name=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME,
            schema_version=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION,
            artifact_schema_name=settings.phase5_initialization_schema_name,
            artifact_hash=initialization_artifact_hash,
        ),
        prototype_reference=ProtoEMPhase5ArtifactReference(
            schema_name=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME,
            schema_version=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION,
            artifact_schema_name=settings.phase5_prototype_schema_name,
            artifact_hash=prototype_artifact_hash,
        ),
        inference_reference=ProtoEMPhase5ArtifactReference(
            schema_name=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_NAME,
            schema_version=PROTOEM_PHASE5_ARTIFACT_REFERENCE_SCHEMA_VERSION,
            artifact_schema_name=settings.phase5_inference_schema_name,
            artifact_hash=inference_result.inference_identity_sha256,
        ),
    )


def _synthetic_metadata(*, input_identity: str) -> EmbeddingMetadata:
    resolution = FeatureResolution3D(
        input_spatial_shape=(1, 1, 3),
        feature_spatial_shape=(1, 1, 3),
        downsample_factors=(1, 1, 1),
    )
    return EmbeddingMetadata(
        encoder_identity="segresnet_encoder_v1",
        preprocessing_hash=_sha256_text("synthetic_preprocessing"),
        checkpoint_hash=_sha256_text("synthetic_checkpoint"),
        input_identity=input_identity,
        feature_stage="final_encoder",
        normalization_name="l2_channel",
        feature_channels=2,
        resolution=resolution,
    )


def _synthetic_supports() -> tuple[SupportPrototypeInput, ...]:
    dataset_manifest_hash = _synthetic_dataset_manifest_hash()
    support_1 = SupportPrototypeInput(
        support_identifier="support_case_001",
        support_patient_id="support_patient_001",
        support_case_id="support_case_001",
        dataset_manifest_hash=dataset_manifest_hash,
        feature_encoding=FeatureEncoding3D(
            feature_data=np.asarray(
                [[[[[1.0, 0.0, 0.9]]], [[[0.0, 1.0, 0.1]]]]],
                dtype=np.float32,
            ),
            metadata=_synthetic_metadata(input_identity="support_case_001"),
        ),
        binary_mask=np.asarray([[[[1, 0, 1]]]], dtype=_MASK_DTYPE),
    )
    support_2 = SupportPrototypeInput(
        support_identifier="support_case_002",
        support_patient_id="support_patient_002",
        support_case_id="support_case_002",
        dataset_manifest_hash=dataset_manifest_hash,
        feature_encoding=FeatureEncoding3D(
            feature_data=np.asarray(
                [[[[[0.8, 0.1, 0.95]]], [[[0.2, 0.9, 0.05]]]]],
                dtype=np.float32,
            ),
            metadata=_synthetic_metadata(input_identity="support_case_002"),
        ),
        binary_mask=np.asarray([[[[1, 0, 1]]]], dtype=_MASK_DTYPE),
    )
    return (support_1, support_2)


def _synthetic_support_provenance() -> tuple[ProtoEMSupportProvenance, ...]:
    preprocessing_hash = _sha256_text("synthetic_preprocessing")
    checkpoint_hash = _sha256_text("synthetic_checkpoint")
    dataset_manifest_hash = _synthetic_dataset_manifest_hash()
    return (
        ProtoEMSupportProvenance(
            schema_name=PROTOEM_SUPPORT_PROVENANCE_SCHEMA_NAME,
            schema_version=PROTOEM_SUPPORT_PROVENANCE_SCHEMA_VERSION,
            support_identifier="support_case_001",
            support_patient_id="support_patient_001",
            support_case_id="support_case_001",
            source_input_identity="support_case_001",
            support_manifest_hash=_sha256_text("support_manifest_001"),
            support_assignment_index=0,
            dataset_manifest_hash=dataset_manifest_hash,
            encoder_identity="segresnet_encoder_v1",
            preprocessing_hash=preprocessing_hash,
            checkpoint_hash=checkpoint_hash,
            feature_stage="final_encoder",
            normalization_name="l2_channel",
        ),
        ProtoEMSupportProvenance(
            schema_name=PROTOEM_SUPPORT_PROVENANCE_SCHEMA_NAME,
            schema_version=PROTOEM_SUPPORT_PROVENANCE_SCHEMA_VERSION,
            support_identifier="support_case_002",
            support_patient_id="support_patient_002",
            support_case_id="support_case_002",
            source_input_identity="support_case_002",
            support_manifest_hash=_sha256_text("support_manifest_002"),
            support_assignment_index=0,
            dataset_manifest_hash=dataset_manifest_hash,
            encoder_identity="segresnet_encoder_v1",
            preprocessing_hash=preprocessing_hash,
            checkpoint_hash=checkpoint_hash,
            feature_stage="final_encoder",
            normalization_name="l2_channel",
        ),
    )


def _synthetic_dataset_manifest_hash() -> str:
    return _sha256_text("synthetic_dataset_manifest")


def _phase6_ablation_plan_to_json(plan: ProtoEMAblationExecutionPlan) -> bytes:
    payload = {
        "schema_name": PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_NAME,
        "schema_version": PROTOEM_ABLATION_EXECUTION_PLAN_SCHEMA_VERSION,
        "plan_identity_hash": plan.plan_identity_hash,
        "entries": [_ablation_entry_to_publication_dict(entry) for entry in plan.entries],
    }
    return canonical_json_bytes(payload) + b"\n"


def _ablation_entry_to_publication_dict(entry: ProtoEMAblationPlanEntry) -> dict[str, JsonValue]:
    status: str
    if entry.execution_mode == "protoem_execution":
        status = "executable"
    elif entry.execution_mode == "phase5_baseline":
        status = "phase5_baseline_link"
    elif entry.unsupported_code is not None and entry.unsupported_code.endswith("_provenance_only"):
        status = "provenance_only"
    else:
        status = "unsupported"
    return {
        "entry_identity_hash": entry.entry_identity_hash,
        "variant_name": entry.ablation_definition.variant_name,
        "ablation_definition_hash": entry.ablation_definition.ablation_definition_hash,
        "execution_mode": entry.execution_mode,
        "status": status,
        "phase5_baseline_kind": entry.phase5_baseline_kind,
        "phase5_method_name": entry.phase5_method_name,
        "unsupported_code": entry.unsupported_code,
        "unsupported_message": entry.unsupported_message,
    }


def _validate_effective_config_json(payload: bytes | str) -> dict[str, object]:
    mapping = _parse_json_mapping(payload, object_name="effective_config")
    if set(mapping) != {PHASE6_CONFIG_ROOT_KEY}:
        raise Phase6PublicationValidationError(
            "effective_config.json must contain only the phase6_protoem_ct root key."
        )
    root = _expect_mapping(mapping[PHASE6_CONFIG_ROOT_KEY], field_name=PHASE6_CONFIG_ROOT_KEY)
    if (
        _expect_string(root["schema_version"], field_name="schema_version")
        != PHASE6_CONFIG_SCHEMA_VERSION
    ):
        raise Phase6PublicationValidationError("effective_config.json schema_version is invalid.")
    protoem_config_mapping = _expect_mapping(root["protoem_config"], field_name="protoem_config")
    config = protoem_config_from_mapping(protoem_config_mapping)
    parameterized_positive_step = _expect_bool(
        root["parameterized_positive_step_parameters_supplied"],
        field_name="parameterized_positive_step_parameters_supplied",
    )
    positive_step_schedule = root["positive_step_schedule"]
    if parameterized_positive_step != (positive_step_schedule is not None):
        raise Phase6PublicationValidationError(
            "effective_config.json positive-step flags are inconsistent."
        )
    return {
        "protoem_config": config,
        "parameterized_positive_step_parameters_supplied": parameterized_positive_step,
    }


def _validate_initialization_summary_json(payload: bytes | str) -> dict[str, object]:
    mapping = _parse_json_mapping(payload, object_name="initialization_summary")
    required = {
        "schema_name",
        "schema_version",
        "initialization_identity_hash",
        "query_identity",
        "query_patient_id",
        "query_case_id",
    }
    missing = required - set(mapping)
    if missing:
        raise Phase6PublicationValidationError(
            f"initialization_summary.json is missing keys: {sorted(missing)!r}."
        )
    initialization_identity_hash = _expect_string(
        mapping["initialization_identity_hash"],
        field_name="initialization_identity_hash",
    )
    _require_sha256(initialization_identity_hash, field_name="initialization_identity_hash")
    return {
        "initialization_identity_hash": initialization_identity_hash,
        "query_identity": _expect_string(mapping["query_identity"], field_name="query_identity"),
    }


def _validate_final_inference_json(payload: bytes | str) -> dict[str, object]:
    mapping = _parse_json_mapping(payload, object_name="final_inference")
    required = {
        "schema_name",
        "schema_version",
        "inference_identity_hash",
        "final_state_identity_hash",
        "stopping_reason",
        "completed_iteration_count",
    }
    missing = required - set(mapping)
    if missing:
        raise Phase6PublicationValidationError(
            f"final_inference.json is missing keys: {sorted(missing)!r}."
        )
    inference_identity_hash = _expect_string(
        mapping["inference_identity_hash"],
        field_name="inference_identity_hash",
    )
    _require_sha256(inference_identity_hash, field_name="inference_identity_hash")
    return {"inference_identity_hash": inference_identity_hash}


def _validate_phase6_output_root(output_root: Path) -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        resolved_output_root = validate_explicit_external_output_root(
            output_root,
            forbidden_roots=(repository_root,),
        )
    except InvalidDatasetRootError as exc:
        raise Phase6PublicationPathError(str(exc)) from exc
    _require_no_parent_traversal_components(output_root)
    _require_no_symlink_components(output_root, resolved_output_root=resolved_output_root)
    return resolved_output_root


def _publish_directory_tree_if_absent_or_equal(
    *,
    output_root: Path,
    artifact_bytes: dict[str, bytes],
) -> bool:
    expected_directory_paths = _expected_directory_paths(artifact_bytes)
    if output_root.exists():
        if not output_root.is_dir():
            raise Phase6PublicationPathError("output_root must be a directory when it exists.")
        _require_no_symlinks_in_tree(output_root)
        if any(output_root.iterdir()):
            if _existing_tree_matches(
                output_root=output_root,
                artifact_bytes=artifact_bytes,
                expected_directory_paths=expected_directory_paths,
            ):
                return True
            raise Phase6PublicationCollisionError(
                "output_root already contains non-identical published artifacts."
            )

    staging_root = output_root.parent / f".{output_root.name}.phase6-publication.tmp"
    if staging_root.exists():
        raise Phase6PublicationCollisionError("staging output root already exists.")
    try:
        staging_root.mkdir(mode=0o700)
        for relative_directory in expected_directory_paths:
            (staging_root / relative_directory).mkdir(parents=True, exist_ok=True)
        for relative_path, content in dict(sorted(artifact_bytes.items())).items():
            _write_bytes_in_staging_root(
                staging_root=staging_root,
                relative_path=relative_path,
                payload=content,
            )
        if output_root.exists():
            if any(output_root.iterdir()):
                raise Phase6PublicationCollisionError(
                    "output_root became non-empty before staged publication."
                )
            output_root.rmdir()
        os.replace(staging_root, output_root)
    except Phase6PublicationError:
        _remove_tree_if_present(staging_root)
        raise
    except OSError as exc:
        _remove_tree_if_present(staging_root)
        raise Phase6PublicationIOError("failed to publish Phase 6 artifacts.") from exc
    return False


def _write_bytes_in_staging_root(
    *,
    staging_root: Path,
    relative_path: str,
    payload: bytes,
) -> None:
    normalized_relative_path = _normalize_relative_publication_path(relative_path)
    output_path = staging_root / normalized_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)


def _normalize_relative_publication_path(relative_path: str) -> str:
    from protoem_ct.data.phase2_paths import normalize_safe_relative_posix_path

    try:
        return normalize_safe_relative_posix_path(relative_path)
    except Exception as exc:
        raise Phase6PublicationPathError("relative publication path is unsafe.") from exc


def _expected_directory_paths(artifact_bytes: Mapping[str, bytes]) -> set[str]:
    directories: set[str] = set()
    for relative_path in artifact_bytes:
        parts = relative_path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    return directories


def _existing_tree_matches(
    *,
    output_root: Path,
    artifact_bytes: Mapping[str, bytes],
    expected_directory_paths: set[str],
) -> bool:
    existing_directory_paths: set[str] = set()
    existing_file_paths: dict[str, bytes] = {}
    for path in sorted(output_root.rglob("*")):
        relative_path = path.relative_to(output_root).as_posix()
        if path.is_symlink():
            raise Phase6PublicationPathError("output_root must not contain symlinked paths.")
        if path.is_dir():
            existing_directory_paths.add(relative_path)
        elif path.is_file():
            existing_file_paths[relative_path] = path.read_bytes()
        else:
            raise Phase6PublicationPathError("output_root must contain only directories or files.")
    if existing_directory_paths != expected_directory_paths:
        return False
    return existing_file_paths == artifact_bytes


def _require_no_parent_traversal_components(path: Path) -> None:
    if ".." in path.parts:
        raise Phase6PublicationPathError("output_root must not contain parent traversal.")


def _require_no_symlink_components(path: Path, *, resolved_output_root: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.exists() and current.is_symlink():
            if _is_allowed_output_alias(current, resolved_output_root=resolved_output_root):
                continue
            raise Phase6PublicationPathError("output_root must not traverse symlinked paths.")
        if current.exists() and not current.is_dir() and current != path:
            raise Phase6PublicationPathError("output_root parent chain must contain directories.")


def _is_allowed_output_alias(symlink_path: Path, *, resolved_output_root: Path) -> bool:
    if symlink_path != Path("/tmp"):
        return False
    try:
        resolved_symlink = symlink_path.resolve(strict=True)
    except OSError:
        return False
    return _path_is_equal_or_nested(resolved_output_root, resolved_symlink)


def _path_is_equal_or_nested(path: Path, root: Path) -> bool:
    if path == root:
        return True
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_no_symlinks_in_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Phase6PublicationPathError("output_root must not contain symlinked paths.")


def _remove_tree_if_present(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _parse_optional_positive_step_schedule(
    value: object,
) -> Phase6PositiveStepScheduleSettings | None:
    if value is None:
        return None
    mapping = _expect_mapping(value, field_name="positive_step_schedule")
    if set(mapping) != _POSITIVE_STEP_KEYS:
        raise Phase6PublicationConfigError(
            "positive_step_schedule must contain exactly epsilon and raw_step_parameters."
        )
    raw_values = mapping["raw_step_parameters"]
    if not isinstance(raw_values, list):
        raise Phase6PublicationConfigError(
            "positive_step_schedule.raw_step_parameters must be a list."
        )
    return Phase6PositiveStepScheduleSettings(
        epsilon=_expect_float(mapping["epsilon"], field_name="positive_step_schedule.epsilon"),
        raw_step_parameters=tuple(
            _expect_float(item, field_name="positive_step_schedule.raw_step_parameters")
            for item in raw_values
        ),
    )


def _parse_json_mapping(payload: bytes | str, *, object_name: str) -> dict[str, object]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise Phase6PublicationValidationError(f"{object_name} JSON could not be decoded.") from exc
    if not isinstance(decoded, dict):
        raise Phase6PublicationValidationError(f"{object_name} JSON must decode to an object.")
    return cast(dict[str, object], decoded)


def _expect_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Phase6PublicationConfigError(f"{field_name} must be a mapping.")
    return cast(dict[str, object], value)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase6PublicationConfigError(f"{field_name} must be a string.")
    return value


def _expect_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Phase6PublicationConfigError(f"{field_name} must be an integer.")
    return value


def _expect_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Phase6PublicationConfigError(f"{field_name} must be numeric.")
    return float(value)


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase6PublicationConfigError(f"{field_name} must be boolean.")
    return value


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise Phase6PublicationValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "PHASE6_ABLATION_PLAN_NAME",
    "PHASE6_CONFIG_ROOT_KEY",
    "PHASE6_CONVERGENCE_PLOT_NAME",
    "PHASE6_EFFECTIVE_CONFIG_NAME",
    "PHASE6_FINAL_INFERENCE_NAME",
    "PHASE6_INITIALIZATION_SUMMARY_NAME",
    "PHASE6_OBJECTIVE_TRACE_NAME",
    "PHASE6_RUN_SUMMARY_NAME",
    "PHASE6_STOPPING_RECORD_NAME",
    "PHASE6_SUMMARY_MARKDOWN_NAME",
    "PHASE6_COLLAPSE_RECORD_NAME",
    "PHASE6_CONFIG_SCHEMA_VERSION",
    "Phase6PositiveStepScheduleSettings",
    "Phase6PublicationCollisionError",
    "Phase6PublicationConfigError",
    "Phase6PublicationError",
    "Phase6PublicationIOError",
    "Phase6PublicationPathError",
    "Phase6PublicationResult",
    "Phase6PublicationSettings",
    "Phase6PublicationValidationError",
    "build_phase6_convergence_plot_png",
    "build_phase6_synthetic_execution_package",
    "load_phase6_protoem_settings",
    "render_phase6_summary_markdown_from_json_artifacts",
    "run_and_publish_phase6_protoem",
]
