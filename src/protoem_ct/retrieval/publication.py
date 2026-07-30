"""Deterministic Phase 5 comparison artifacts, publication, and synthetic CLI surface."""

from __future__ import annotations

import hashlib
import json
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
from protoem_ct.evaluation.metrics import (
    compute_binary_confusion_counts,
    dice_from_counts,
    iou_from_counts,
)
from protoem_ct.models.interfaces import (
    EmbeddingMetadata,
    FeatureEncoding3D,
    FeatureResolution3D,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalResult,
)
from protoem_ct.retrieval.cosine import retrieve_top_k_by_exact_cosine
from protoem_ct.retrieval.inference import (
    PrototypeInferenceResult,
    hash_prototype_inference_identity,
    infer_query_from_prototypes,
)
from protoem_ct.retrieval.prototypes import (
    SupportPrototypeInput,
    SupportPrototypeMemory,
    build_support_prototype_memory,
    hash_support_prototype_identity,
)

PHASE5_METHOD_DEFINITION_SCHEMA_NAME: Final[str] = "phase5_method_definition"
PHASE5_METHOD_DEFINITION_SCHEMA_VERSION: Final[str] = "v1"
PHASE5_COMPARISON_RECORD_SCHEMA_NAME: Final[str] = "phase5_comparison_record"
PHASE5_COMPARISON_RECORD_SCHEMA_VERSION: Final[str] = "v1"
PHASE5_COMPARISON_TABLE_SCHEMA_NAME: Final[str] = "phase5_comparison_table"
PHASE5_COMPARISON_TABLE_SCHEMA_VERSION: Final[str] = "v1"
PHASE5_RUN_SUMMARY_SCHEMA_NAME: Final[str] = "phase5_run_summary"
PHASE5_RUN_SUMMARY_SCHEMA_VERSION: Final[str] = "v1"
PHASE5_SUPPORT_RECORD_SCHEMA_NAME: Final[str] = "phase5_support_record"
PHASE5_SUPPORT_RECORD_SCHEMA_VERSION: Final[str] = "v1"

PHASE5_CONFIG_ROOT_KEY: Final[str] = "phase5_foundation_retrieval"
PHASE5_CONFIG_SCHEMA_VERSION: Final[str] = "v1"
PHASE5_COMPARISON_JSON_NAME: Final[str] = "phase5_comparison.json"
PHASE5_COMPARISON_MARKDOWN_NAME: Final[str] = "phase5_comparison_table.md"
PHASE5_RUN_SUMMARY_JSON_NAME: Final[str] = "phase5_run_summary.json"
PHASE5_EFFECTIVE_CONFIG_JSON_NAME: Final[str] = "effective_config.json"

NO_RETRIEVAL_METHOD: Final[str] = "no_retrieval"
NEAREST_SUPPORT_RETRIEVAL_METHOD: Final[str] = "nearest_support_retrieval"
FIXED_METHOD_ORDER: Final[tuple[str, str]] = (
    NO_RETRIEVAL_METHOD,
    NEAREST_SUPPORT_RETRIEVAL_METHOD,
)
FIXED_NEAREST_SUPPORT_TOP_K: Final[int] = 1
METRICS_UNAVAILABLE_STATUS: Final[str] = "unavailable"
METRICS_AVAILABLE_STATUS: Final[str] = "available"
EMPTY_MASK_BEHAVIOR: Final[str] = "empty_reference_and_prediction_yield_dice_iou_1.0"

_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "synthetic_mode_only",
        "nearest_support_top_k",
        "emit_confidence_margin",
        "foundation_model_adapter_enabled",
    }
)
_METHOD_NAMES: Final[frozenset[str]] = frozenset(FIXED_METHOD_ORDER)
_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_PREDICTION_DTYPE: Final[np.dtype[np.uint8]] = np.dtype(np.uint8)
_BINARY_VALUES: Final[frozenset[int]] = frozenset({0, 1})


class Phase5ComparisonError(ValueError):
    """Base error for deterministic Phase 5 comparison and publication."""


class Phase5ComparisonConfigError(Phase5ComparisonError):
    """Raised when the Phase 5 config file is missing or invalid."""


class Phase5ComparisonValidationError(Phase5ComparisonError):
    """Raised when a Phase 5 comparison artifact or input contract is malformed."""


class Phase5ComparisonLeakageError(Phase5ComparisonError):
    """Raised when support/query overlap violates the Phase 5 comparison contract."""


class Phase5ComparisonPublicationError(Phase5ComparisonError):
    """Raised when Phase 5 publication cannot complete safely."""


class Phase5ComparisonPathError(Phase5ComparisonPublicationError):
    """Raised when publication paths are unsafe."""


class Phase5ComparisonCollisionError(Phase5ComparisonPublicationError):
    """Raised when publication would overwrite non-identical existing artifacts."""


class Phase5ComparisonIOError(Phase5ComparisonPublicationError):
    """Raised when atomic publication fails."""


@dataclass(frozen=True, slots=True)
class Phase5Settings:
    """Validated Phase 5 synthetic comparison settings."""

    schema_version: str
    synthetic_mode_only: bool
    nearest_support_top_k: int
    emit_confidence_margin: bool
    foundation_model_adapter_enabled: bool

    def __post_init__(self) -> None:
        if self.schema_version != PHASE5_CONFIG_SCHEMA_VERSION:
            raise Phase5ComparisonConfigError(
                f"schema_version must be {PHASE5_CONFIG_SCHEMA_VERSION!r}."
            )
        if self.synthetic_mode_only is not True:
            raise Phase5ComparisonConfigError("synthetic_mode_only must be true in Phase 5.")
        if self.nearest_support_top_k != FIXED_NEAREST_SUPPORT_TOP_K:
            raise Phase5ComparisonConfigError("nearest_support_top_k must be exactly 1.")
        if self.foundation_model_adapter_enabled is not False:
            raise Phase5ComparisonConfigError(
                "foundation_model_adapter_enabled must be false for Phase 5."
            )


@dataclass(frozen=True, slots=True)
class Phase5SupportRecord:
    """Explicit support feature, mask, and provenance for one comparison input."""

    schema_name: str
    schema_version: str
    support_identifier: str
    support_patient_id: str
    support_case_id: str
    support_manifest_hash: str
    dataset_manifest_hash: str | None
    feature_encoding: FeatureEncoding3D
    binary_mask: np.ndarray

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PHASE5_SUPPORT_RECORD_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PHASE5_SUPPORT_RECORD_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_identifier(self.support_identifier, field_name="support_identifier")
        _require_identifier(self.support_patient_id, field_name="support_patient_id")
        _require_identifier(self.support_case_id, field_name="support_case_id")
        _require_sha256(self.support_manifest_hash, field_name="support_manifest_hash")
        if self.dataset_manifest_hash is not None:
            _require_sha256(self.dataset_manifest_hash, field_name="dataset_manifest_hash")


@dataclass(frozen=True, slots=True)
class Phase5MethodDefinition:
    """One fixed method definition for the Phase 5 comparison table."""

    schema_name: str
    schema_version: str
    method_name: str
    uses_retrieval: bool
    retrieval_top_k: int | None
    uses_full_support_set: bool
    query_label_usage: str
    prototype_update_policy: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PHASE5_METHOD_DEFINITION_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PHASE5_METHOD_DEFINITION_SCHEMA_VERSION,
            field_name="schema_version",
        )
        if self.method_name not in _METHOD_NAMES:
            raise Phase5ComparisonValidationError(
                f"method_name must be one of {sorted(_METHOD_NAMES)!r}."
            )
        if self.method_name == NO_RETRIEVAL_METHOD:
            if self.uses_retrieval:
                raise Phase5ComparisonValidationError("no_retrieval must not use retrieval.")
            if self.retrieval_top_k is not None:
                raise Phase5ComparisonValidationError(
                    "no_retrieval must not declare retrieval_top_k."
                )
            if not self.uses_full_support_set:
                raise Phase5ComparisonValidationError(
                    "no_retrieval must use the full allowed support set."
                )
        else:
            if not self.uses_retrieval:
                raise Phase5ComparisonValidationError(
                    "nearest_support_retrieval must use retrieval."
                )
            if self.retrieval_top_k != FIXED_NEAREST_SUPPORT_TOP_K:
                raise Phase5ComparisonValidationError(
                    "nearest_support_retrieval must use top_k = 1."
                )
            if self.uses_full_support_set:
                raise Phase5ComparisonValidationError(
                    "nearest_support_retrieval must not use the full support set."
                )
        if self.query_label_usage != "evaluation_only":
            raise Phase5ComparisonValidationError("query_label_usage must be 'evaluation_only'.")
        if self.prototype_update_policy != "frozen":
            raise Phase5ComparisonValidationError("prototype_update_policy must be 'frozen'.")


@dataclass(frozen=True, slots=True)
class Phase5ComparisonRecord:
    """One deterministic method-specific Phase 5 comparison record."""

    schema_name: str
    schema_version: str
    method_name: str
    selected_support_identifiers: tuple[str, ...]
    support_count_used: int
    support_set_identity_sha256: str
    retrieval_result_identity_sha256: str | None
    retrieved_support_identifier: str | None
    foreground_prototype_identity_sha256: str
    background_prototype_identity_sha256: str
    inference_identity_sha256: str
    prediction_content_sha256: str
    foreground_score_content_sha256: str
    background_score_content_sha256: str
    confidence_margin_content_sha256: str | None
    metrics_availability_status: str
    dice: float | None
    iou: float | None
    empty_mask_behavior: str

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PHASE5_COMPARISON_RECORD_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PHASE5_COMPARISON_RECORD_SCHEMA_VERSION,
            field_name="schema_version",
        )
        if self.method_name not in _METHOD_NAMES:
            raise Phase5ComparisonValidationError(
                f"method_name must be one of {sorted(_METHOD_NAMES)!r}."
            )
        if not self.selected_support_identifiers:
            raise Phase5ComparisonValidationError(
                "selected_support_identifiers must contain at least one item."
            )
        normalized_identifiers = tuple(sorted(self.selected_support_identifiers))
        for item in normalized_identifiers:
            _require_identifier(item, field_name="selected_support_identifiers")
        if len(set(normalized_identifiers)) != len(normalized_identifiers):
            raise Phase5ComparisonValidationError("selected_support_identifiers must be unique.")
        object.__setattr__(self, "selected_support_identifiers", normalized_identifiers)
        if self.support_count_used != len(self.selected_support_identifiers):
            raise Phase5ComparisonValidationError(
                "support_count_used must equal len(selected_support_identifiers)."
            )
        _require_sha256(self.support_set_identity_sha256, field_name="support_set_identity_sha256")
        if self.retrieval_result_identity_sha256 is not None:
            _require_sha256(
                self.retrieval_result_identity_sha256,
                field_name="retrieval_result_identity_sha256",
            )
        if self.retrieved_support_identifier is not None:
            _require_identifier(
                self.retrieved_support_identifier,
                field_name="retrieved_support_identifier",
            )
        _require_sha256(
            self.foreground_prototype_identity_sha256,
            field_name="foreground_prototype_identity_sha256",
        )
        _require_sha256(
            self.background_prototype_identity_sha256,
            field_name="background_prototype_identity_sha256",
        )
        _require_sha256(self.inference_identity_sha256, field_name="inference_identity_sha256")
        _require_sha256(self.prediction_content_sha256, field_name="prediction_content_sha256")
        _require_sha256(
            self.foreground_score_content_sha256,
            field_name="foreground_score_content_sha256",
        )
        _require_sha256(
            self.background_score_content_sha256,
            field_name="background_score_content_sha256",
        )
        if self.confidence_margin_content_sha256 is not None:
            _require_sha256(
                self.confidence_margin_content_sha256,
                field_name="confidence_margin_content_sha256",
            )
        if self.metrics_availability_status not in {
            METRICS_AVAILABLE_STATUS,
            METRICS_UNAVAILABLE_STATUS,
        }:
            raise Phase5ComparisonValidationError(
                "metrics_availability_status must be 'available' or 'unavailable'."
            )
        if self.metrics_availability_status == METRICS_AVAILABLE_STATUS:
            if self.dice is None or self.iou is None:
                raise Phase5ComparisonValidationError(
                    "available metrics require dice and iou values."
                )
            _require_metric(self.dice, field_name="dice")
            _require_metric(self.iou, field_name="iou")
        else:
            if self.dice is not None or self.iou is not None:
                raise Phase5ComparisonValidationError(
                    "unavailable metrics must keep dice and iou as null."
                )
        if self.empty_mask_behavior != EMPTY_MASK_BEHAVIOR:
            raise Phase5ComparisonValidationError(
                f"empty_mask_behavior must be {EMPTY_MASK_BEHAVIOR!r}."
            )


@dataclass(frozen=True, slots=True)
class Phase5ComparisonTable:
    """Deterministic Phase 5 comparison table over the two fixed methods."""

    schema_name: str
    schema_version: str
    comparison_identity_sha256: str
    query_identity: str
    query_patient_id: str
    query_case_id: str
    dataset_manifest_hash: str | None
    support_set_identity_sha256: str
    ordered_method_definitions: tuple[Phase5MethodDefinition, ...]
    records: tuple[Phase5ComparisonRecord, ...]
    metrics_reference_available: bool

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PHASE5_COMPARISON_TABLE_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PHASE5_COMPARISON_TABLE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(
            self.comparison_identity_sha256,
            field_name="comparison_identity_sha256",
        )
        _require_identifier(self.query_identity, field_name="query_identity")
        _require_identifier(self.query_patient_id, field_name="query_patient_id")
        _require_identifier(self.query_case_id, field_name="query_case_id")
        if self.dataset_manifest_hash is not None:
            _require_sha256(self.dataset_manifest_hash, field_name="dataset_manifest_hash")
        _require_sha256(self.support_set_identity_sha256, field_name="support_set_identity_sha256")
        if len(self.ordered_method_definitions) != 2:
            raise Phase5ComparisonValidationError(
                "ordered_method_definitions must contain exactly two method definitions."
            )
        if len(self.records) != 2:
            raise Phase5ComparisonValidationError("records must contain exactly two methods.")
        method_order_index = {
            method_name: index for index, method_name in enumerate(FIXED_METHOD_ORDER)
        }
        normalized_definitions = tuple(
            sorted(
                self.ordered_method_definitions,
                key=lambda item: method_order_index[item.method_name],
            )
        )
        normalized_records = tuple(
            sorted(
                self.records,
                key=lambda item: method_order_index[item.method_name],
            )
        )
        object.__setattr__(self, "ordered_method_definitions", normalized_definitions)
        object.__setattr__(self, "records", normalized_records)
        if tuple(item.method_name for item in normalized_definitions) != FIXED_METHOD_ORDER:
            raise Phase5ComparisonValidationError(
                "ordered_method_definitions must contain the fixed method order."
            )
        if tuple(item.method_name for item in normalized_records) != FIXED_METHOD_ORDER:
            raise Phase5ComparisonValidationError("records must contain the fixed method order.")


@dataclass(frozen=True, slots=True)
class Phase5RunSummary:
    """Machine-readable deterministic Phase 5 run summary."""

    schema_name: str
    schema_version: str
    comparison_identity_sha256: str
    support_set_identity_sha256: str
    method_count: int
    nearest_support_top_k: int
    metrics_reference_available: bool
    foundation_model_adapter_enabled: bool
    published_filenames: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_schema_name(
            self.schema_name,
            expected=PHASE5_RUN_SUMMARY_SCHEMA_NAME,
            field_name="schema_name",
        )
        _require_schema_version(
            self.schema_version,
            expected=PHASE5_RUN_SUMMARY_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_sha256(
            self.comparison_identity_sha256,
            field_name="comparison_identity_sha256",
        )
        _require_sha256(
            self.support_set_identity_sha256,
            field_name="support_set_identity_sha256",
        )
        if self.method_count != 2:
            raise Phase5ComparisonValidationError("method_count must be exactly 2.")
        if self.nearest_support_top_k != FIXED_NEAREST_SUPPORT_TOP_K:
            raise Phase5ComparisonValidationError("nearest_support_top_k must be exactly 1.")
        expected_files = (
            PHASE5_COMPARISON_JSON_NAME,
            PHASE5_COMPARISON_MARKDOWN_NAME,
            PHASE5_EFFECTIVE_CONFIG_JSON_NAME,
            PHASE5_RUN_SUMMARY_JSON_NAME,
        )
        if tuple(sorted(self.published_filenames)) != tuple(sorted(expected_files)):
            raise Phase5ComparisonValidationError(
                "published_filenames must contain the fixed Phase 5 artifact set."
            )


@dataclass(frozen=True, slots=True)
class Phase5PublicationResult:
    """Deterministic publication summary returned to the CLI and tests."""

    output_root: Path
    comparison_identity_sha256: str
    support_set_identity_sha256: str
    reused_existing_output: bool
    comparison_path: Path
    markdown_path: Path
    run_summary_path: Path
    effective_config_path: Path


@dataclass(frozen=True, slots=True)
class _MethodExecutionResult:
    method_name: str
    selected_support_identifiers: tuple[str, ...]
    prototype_memory: SupportPrototypeMemory
    inference_result: PrototypeInferenceResult
    retrieval_result: RetrievalResult | None
    retrieval_result_identity_sha256: str | None
    retrieved_support_identifier: str | None


@dataclass(frozen=True, slots=True)
class _SyntheticFixture:
    query_feature_encoding: FeatureEncoding3D
    query_patient_id: str
    query_case_id: str
    query_identity: str
    dataset_manifest_hash: str
    support_records: tuple[Phase5SupportRecord, ...]
    query_reference_mask: np.ndarray


def load_phase5_foundation_retrieval_settings(config_path: Path) -> Phase5Settings:
    """Load bounded Phase 5 synthetic retrieval settings from OmegaConf YAML."""

    try:
        raw_config = OmegaConf.load(config_path)
    except OSError as exc:
        raise Phase5ComparisonConfigError(
            f"failed to read Phase 5 config at {config_path}: {exc}"
        ) from exc
    except Exception as exc:
        raise Phase5ComparisonConfigError(
            f"failed to parse Phase 5 config at {config_path}: {exc}"
        ) from exc
    if not isinstance(raw_config, DictConfig):
        raise Phase5ComparisonConfigError("Phase 5 config must contain a mapping.")
    container = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(container, dict):
        raise Phase5ComparisonConfigError("Phase 5 config must resolve to a mapping.")
    resolved = cast(dict[str, object], container)
    if set(resolved) != {PHASE5_CONFIG_ROOT_KEY}:
        raise Phase5ComparisonConfigError(
            f"Phase 5 config must contain only {PHASE5_CONFIG_ROOT_KEY!r}."
        )
    settings_mapping = resolved[PHASE5_CONFIG_ROOT_KEY]
    if not isinstance(settings_mapping, dict):
        raise Phase5ComparisonConfigError("phase5_foundation_retrieval must resolve to a mapping.")
    typed_settings = cast(dict[str, object], settings_mapping)
    unknown_keys = set(typed_settings) - _CONFIG_KEYS
    missing_keys = _CONFIG_KEYS - set(typed_settings)
    if unknown_keys:
        raise Phase5ComparisonConfigError(
            f"unknown Phase 5 setting keys: {sorted(unknown_keys)!r}."
        )
    if missing_keys:
        raise Phase5ComparisonConfigError(
            f"missing Phase 5 setting keys: {sorted(missing_keys)!r}."
        )
    return Phase5Settings(
        schema_version=_expect_string(
            typed_settings["schema_version"], field_name="schema_version"
        ),
        synthetic_mode_only=_expect_bool(
            typed_settings["synthetic_mode_only"],
            field_name="synthetic_mode_only",
        ),
        nearest_support_top_k=_expect_int(
            typed_settings["nearest_support_top_k"],
            field_name="nearest_support_top_k",
        ),
        emit_confidence_margin=_expect_bool(
            typed_settings["emit_confidence_margin"],
            field_name="emit_confidence_margin",
        ),
        foundation_model_adapter_enabled=_expect_bool(
            typed_settings["foundation_model_adapter_enabled"],
            field_name="foundation_model_adapter_enabled",
        ),
    )


def run_and_publish_phase5_retrieval(
    *,
    output_root: Path,
    settings: Phase5Settings,
) -> Phase5PublicationResult:
    """Run the bounded synthetic Phase 5 comparison and publish deterministic artifacts."""

    fixture = build_phase5_synthetic_fixture()
    comparison_table = run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=fixture.support_records,
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=fixture.query_reference_mask,
        emit_confidence_margin=settings.emit_confidence_margin,
        nearest_support_top_k=settings.nearest_support_top_k,
    )
    run_summary = Phase5RunSummary(
        schema_name=PHASE5_RUN_SUMMARY_SCHEMA_NAME,
        schema_version=PHASE5_RUN_SUMMARY_SCHEMA_VERSION,
        comparison_identity_sha256=comparison_table.comparison_identity_sha256,
        support_set_identity_sha256=comparison_table.support_set_identity_sha256,
        method_count=len(comparison_table.records),
        nearest_support_top_k=settings.nearest_support_top_k,
        metrics_reference_available=comparison_table.metrics_reference_available,
        foundation_model_adapter_enabled=settings.foundation_model_adapter_enabled,
        published_filenames=(
            PHASE5_COMPARISON_JSON_NAME,
            PHASE5_COMPARISON_MARKDOWN_NAME,
            PHASE5_EFFECTIVE_CONFIG_JSON_NAME,
            PHASE5_RUN_SUMMARY_JSON_NAME,
        ),
    )
    effective_config_payload = {
        PHASE5_CONFIG_ROOT_KEY: {
            "schema_version": settings.schema_version,
            "synthetic_mode_only": settings.synthetic_mode_only,
            "nearest_support_top_k": settings.nearest_support_top_k,
            "emit_confidence_margin": settings.emit_confidence_margin,
            "foundation_model_adapter_enabled": settings.foundation_model_adapter_enabled,
        }
    }
    artifact_bytes = {
        PHASE5_COMPARISON_JSON_NAME: phase5_comparison_table_to_json(comparison_table),
        PHASE5_COMPARISON_MARKDOWN_NAME: _render_phase5_comparison_markdown(comparison_table),
        PHASE5_RUN_SUMMARY_JSON_NAME: phase5_run_summary_to_json(run_summary),
        PHASE5_EFFECTIVE_CONFIG_JSON_NAME: canonical_json_bytes(effective_config_payload) + b"\n",
    }
    validated_output_root = _validate_phase5_output_root(output_root)
    reused_existing_output = _publish_directory_tree_if_absent_or_equal(
        output_root=validated_output_root,
        artifact_bytes=artifact_bytes,
    )
    return Phase5PublicationResult(
        output_root=validated_output_root,
        comparison_identity_sha256=comparison_table.comparison_identity_sha256,
        support_set_identity_sha256=comparison_table.support_set_identity_sha256,
        reused_existing_output=reused_existing_output,
        comparison_path=validated_output_root / PHASE5_COMPARISON_JSON_NAME,
        markdown_path=validated_output_root / PHASE5_COMPARISON_MARKDOWN_NAME,
        run_summary_path=validated_output_root / PHASE5_RUN_SUMMARY_JSON_NAME,
        effective_config_path=validated_output_root / PHASE5_EFFECTIVE_CONFIG_JSON_NAME,
    )


def run_phase5_comparison(
    *,
    query_feature_encoding: FeatureEncoding3D,
    support_records: tuple[Phase5SupportRecord, ...],
    query_patient_id: str,
    query_case_id: str,
    query_identity: str,
    dataset_manifest_hash: str | None,
    query_reference_mask: np.ndarray | None,
    emit_confidence_margin: bool,
    nearest_support_top_k: int = FIXED_NEAREST_SUPPORT_TOP_K,
) -> Phase5ComparisonTable:
    """Run the two fixed Phase 5 methods and build one deterministic comparison artifact."""

    if nearest_support_top_k != FIXED_NEAREST_SUPPORT_TOP_K:
        raise Phase5ComparisonValidationError("nearest_support_top_k must be exactly 1.")
    normalized_supports = _normalize_support_records(support_records)
    _validate_query_support_disjointness(
        query_patient_id=query_patient_id,
        query_case_id=query_case_id,
        query_identity=query_identity,
        support_records=normalized_supports,
    )
    support_set_identity_sha256 = hash_phase5_support_set(normalized_supports)
    method_definitions = (
        Phase5MethodDefinition(
            schema_name=PHASE5_METHOD_DEFINITION_SCHEMA_NAME,
            schema_version=PHASE5_METHOD_DEFINITION_SCHEMA_VERSION,
            method_name=NO_RETRIEVAL_METHOD,
            uses_retrieval=False,
            retrieval_top_k=None,
            uses_full_support_set=True,
            query_label_usage="evaluation_only",
            prototype_update_policy="frozen",
        ),
        Phase5MethodDefinition(
            schema_name=PHASE5_METHOD_DEFINITION_SCHEMA_NAME,
            schema_version=PHASE5_METHOD_DEFINITION_SCHEMA_VERSION,
            method_name=NEAREST_SUPPORT_RETRIEVAL_METHOD,
            uses_retrieval=True,
            retrieval_top_k=FIXED_NEAREST_SUPPORT_TOP_K,
            uses_full_support_set=False,
            query_label_usage="evaluation_only",
            prototype_update_policy="frozen",
        ),
    )
    method_results = (
        _run_no_retrieval_method(
            query_feature_encoding=query_feature_encoding,
            support_records=normalized_supports,
            query_patient_id=query_patient_id,
            query_case_id=query_case_id,
            query_identity=query_identity,
            dataset_manifest_hash=dataset_manifest_hash,
            emit_confidence_margin=emit_confidence_margin,
        ),
        _run_nearest_support_method(
            query_feature_encoding=query_feature_encoding,
            support_records=normalized_supports,
            query_patient_id=query_patient_id,
            query_case_id=query_case_id,
            query_identity=query_identity,
            dataset_manifest_hash=dataset_manifest_hash,
            emit_confidence_margin=emit_confidence_margin,
            nearest_support_top_k=nearest_support_top_k,
        ),
    )
    normalized_reference_mask = (
        _normalize_reference_mask(
            reference_mask=query_reference_mask,
            expected_shape=method_results[0].inference_result.prediction_mask.shape,
        )
        if query_reference_mask is not None
        else None
    )
    records = tuple(
        _build_comparison_record(
            method_result=item,
            support_set_identity_sha256=support_set_identity_sha256,
            reference_mask=normalized_reference_mask,
        )
        for item in method_results
    )
    draft = Phase5ComparisonTable(
        schema_name=PHASE5_COMPARISON_TABLE_SCHEMA_NAME,
        schema_version=PHASE5_COMPARISON_TABLE_SCHEMA_VERSION,
        comparison_identity_sha256="0" * 64,
        query_identity=query_identity,
        query_patient_id=query_patient_id,
        query_case_id=query_case_id,
        dataset_manifest_hash=dataset_manifest_hash,
        support_set_identity_sha256=support_set_identity_sha256,
        ordered_method_definitions=method_definitions,
        records=records,
        metrics_reference_available=(normalized_reference_mask is not None),
    )
    return Phase5ComparisonTable(
        schema_name=draft.schema_name,
        schema_version=draft.schema_version,
        comparison_identity_sha256=hash_phase5_comparison_table(draft),
        query_identity=draft.query_identity,
        query_patient_id=draft.query_patient_id,
        query_case_id=draft.query_case_id,
        dataset_manifest_hash=draft.dataset_manifest_hash,
        support_set_identity_sha256=draft.support_set_identity_sha256,
        ordered_method_definitions=draft.ordered_method_definitions,
        records=draft.records,
        metrics_reference_available=draft.metrics_reference_available,
    )


def build_phase5_synthetic_fixture() -> _SyntheticFixture:
    """Return the bounded deterministic synthetic Phase 5 smoke fixture."""

    dataset_manifest_hash = "d" * 64
    encoder_identity = "segresnet_encoder_v1"
    preprocessing_hash = "a" * 64
    checkpoint_hash = "b" * 64
    feature_stage = "final_encoder"
    normalization_name = "l2_channel"

    def _metadata(input_identity: str) -> EmbeddingMetadata:
        return EmbeddingMetadata(
            encoder_identity=encoder_identity,
            preprocessing_hash=preprocessing_hash,
            checkpoint_hash=checkpoint_hash,
            input_identity=input_identity,
            feature_stage=feature_stage,
            normalization_name=normalization_name,
            feature_channels=2,
            resolution=FeatureResolution3D(
                input_spatial_shape=(1, 1, 2),
                feature_spatial_shape=(1, 1, 2),
                downsample_factors=(1, 1, 1),
            ),
        )

    support_a = Phase5SupportRecord(
        schema_name=PHASE5_SUPPORT_RECORD_SCHEMA_NAME,
        schema_version=PHASE5_SUPPORT_RECORD_SCHEMA_VERSION,
        support_identifier="support_case_001",
        support_patient_id="support_patient_001",
        support_case_id="support_case_001",
        support_manifest_hash="1" * 64,
        dataset_manifest_hash=dataset_manifest_hash,
        feature_encoding=FeatureEncoding3D(
            feature_data=np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32),
            metadata=_metadata("support_case_001"),
        ),
        binary_mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
    )
    support_b = Phase5SupportRecord(
        schema_name=PHASE5_SUPPORT_RECORD_SCHEMA_NAME,
        schema_version=PHASE5_SUPPORT_RECORD_SCHEMA_VERSION,
        support_identifier="support_case_002",
        support_patient_id="support_patient_002",
        support_case_id="support_case_002",
        support_manifest_hash="2" * 64,
        dataset_manifest_hash=dataset_manifest_hash,
        feature_encoding=FeatureEncoding3D(
            feature_data=np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32),
            metadata=_metadata("support_case_002"),
        ),
        binary_mask=np.asarray([[[[1, 0]]]], dtype=np.uint8),
    )
    query = FeatureEncoding3D(
        feature_data=np.asarray([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], dtype=np.float32),
        metadata=_metadata("query_case_001"),
    )
    query_reference_mask = np.asarray([[[[[1, 0]]]]], dtype=np.uint8)
    return _SyntheticFixture(
        query_feature_encoding=query,
        query_patient_id="query_patient_001",
        query_case_id="query_case_001",
        query_identity="query_case_001",
        dataset_manifest_hash=dataset_manifest_hash,
        support_records=(support_a, support_b),
        query_reference_mask=query_reference_mask,
    )


def hash_phase5_comparison_table(table: Phase5ComparisonTable) -> str:
    """Return the deterministic comparison identity hash for one Phase 5 table."""

    return sha256_json(phase5_comparison_table_identity_payload(table))


def phase5_comparison_table_identity_payload(
    table: Phase5ComparisonTable,
) -> dict[str, JsonValue]:
    """Return the exact canonical identity payload for one Phase 5 comparison table."""

    return {
        "dataset_manifest_hash": table.dataset_manifest_hash,
        "ordered_method_definitions": [
            phase5_method_definition_to_dict(item) for item in table.ordered_method_definitions
        ],
        "query_case_id": table.query_case_id,
        "query_identity": table.query_identity,
        "query_patient_id": table.query_patient_id,
        "records": [
            {
                "background_prototype_identity_sha256": item.background_prototype_identity_sha256,
                "background_score_content_sha256": item.background_score_content_sha256,
                "confidence_margin_content_sha256": item.confidence_margin_content_sha256,
                "dice": item.dice,
                "empty_mask_behavior": item.empty_mask_behavior,
                "foreground_prototype_identity_sha256": item.foreground_prototype_identity_sha256,
                "foreground_score_content_sha256": item.foreground_score_content_sha256,
                "inference_identity_sha256": item.inference_identity_sha256,
                "iou": item.iou,
                "method_name": item.method_name,
                "metrics_availability_status": item.metrics_availability_status,
                "prediction_content_sha256": item.prediction_content_sha256,
                "retrieval_result_identity_sha256": item.retrieval_result_identity_sha256,
                "retrieved_support_identifier": item.retrieved_support_identifier,
                "selected_support_identifiers": list(item.selected_support_identifiers),
                "support_count_used": item.support_count_used,
                "support_set_identity_sha256": item.support_set_identity_sha256,
            }
            for item in table.records
        ],
        "schema_name": table.schema_name,
        "schema_version": table.schema_version,
        "support_set_identity_sha256": table.support_set_identity_sha256,
    }


def phase5_method_definition_to_dict(
    definition: Phase5MethodDefinition,
) -> dict[str, JsonValue]:
    """Convert one method definition to a canonical JSON-compatible mapping."""

    return {
        "method_name": definition.method_name,
        "prototype_update_policy": definition.prototype_update_policy,
        "query_label_usage": definition.query_label_usage,
        "retrieval_top_k": definition.retrieval_top_k,
        "schema_name": definition.schema_name,
        "schema_version": definition.schema_version,
        "uses_full_support_set": definition.uses_full_support_set,
        "uses_retrieval": definition.uses_retrieval,
    }


def phase5_method_definition_from_mapping(
    mapping: Mapping[str, object],
) -> Phase5MethodDefinition:
    """Reconstruct one method definition from a strict mapping."""

    _require_exact_keys(
        mapping,
        required={
            "method_name",
            "prototype_update_policy",
            "query_label_usage",
            "retrieval_top_k",
            "schema_name",
            "schema_version",
            "uses_full_support_set",
            "uses_retrieval",
        },
        field_name="Phase5MethodDefinition",
    )
    return Phase5MethodDefinition(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        method_name=_expect_string(mapping["method_name"], field_name="method_name"),
        uses_retrieval=_expect_bool(mapping["uses_retrieval"], field_name="uses_retrieval"),
        retrieval_top_k=(
            _expect_int(mapping["retrieval_top_k"], field_name="retrieval_top_k")
            if mapping["retrieval_top_k"] is not None
            else None
        ),
        uses_full_support_set=_expect_bool(
            mapping["uses_full_support_set"],
            field_name="uses_full_support_set",
        ),
        query_label_usage=_expect_string(
            mapping["query_label_usage"],
            field_name="query_label_usage",
        ),
        prototype_update_policy=_expect_string(
            mapping["prototype_update_policy"],
            field_name="prototype_update_policy",
        ),
    )


def phase5_comparison_record_to_dict(record: Phase5ComparisonRecord) -> dict[str, JsonValue]:
    """Convert one comparison record to a canonical JSON-compatible mapping."""

    return {
        "background_prototype_identity_sha256": record.background_prototype_identity_sha256,
        "background_score_content_sha256": record.background_score_content_sha256,
        "confidence_margin_content_sha256": record.confidence_margin_content_sha256,
        "dice": record.dice,
        "empty_mask_behavior": record.empty_mask_behavior,
        "foreground_prototype_identity_sha256": record.foreground_prototype_identity_sha256,
        "foreground_score_content_sha256": record.foreground_score_content_sha256,
        "inference_identity_sha256": record.inference_identity_sha256,
        "iou": record.iou,
        "method_name": record.method_name,
        "metrics_availability_status": record.metrics_availability_status,
        "prediction_content_sha256": record.prediction_content_sha256,
        "retrieval_result_identity_sha256": record.retrieval_result_identity_sha256,
        "retrieved_support_identifier": record.retrieved_support_identifier,
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "selected_support_identifiers": list(record.selected_support_identifiers),
        "support_count_used": record.support_count_used,
        "support_set_identity_sha256": record.support_set_identity_sha256,
    }


def phase5_comparison_record_from_mapping(
    mapping: Mapping[str, object],
) -> Phase5ComparisonRecord:
    """Reconstruct one comparison record from a strict mapping."""

    _require_exact_keys(
        mapping,
        required={
            "background_prototype_identity_sha256",
            "background_score_content_sha256",
            "confidence_margin_content_sha256",
            "dice",
            "empty_mask_behavior",
            "foreground_prototype_identity_sha256",
            "foreground_score_content_sha256",
            "inference_identity_sha256",
            "iou",
            "method_name",
            "metrics_availability_status",
            "prediction_content_sha256",
            "retrieval_result_identity_sha256",
            "retrieved_support_identifier",
            "schema_name",
            "schema_version",
            "selected_support_identifiers",
            "support_count_used",
            "support_set_identity_sha256",
        },
        field_name="Phase5ComparisonRecord",
    )
    selected_support_identifiers = _expect_string_tuple(
        mapping["selected_support_identifiers"],
        field_name="selected_support_identifiers",
    )
    return Phase5ComparisonRecord(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        method_name=_expect_string(mapping["method_name"], field_name="method_name"),
        selected_support_identifiers=selected_support_identifiers,
        support_count_used=_expect_int(
            mapping["support_count_used"], field_name="support_count_used"
        ),
        support_set_identity_sha256=_expect_string(
            mapping["support_set_identity_sha256"],
            field_name="support_set_identity_sha256",
        ),
        retrieval_result_identity_sha256=(
            _expect_string(
                mapping["retrieval_result_identity_sha256"],
                field_name="retrieval_result_identity_sha256",
            )
            if mapping["retrieval_result_identity_sha256"] is not None
            else None
        ),
        retrieved_support_identifier=(
            _expect_string(
                mapping["retrieved_support_identifier"],
                field_name="retrieved_support_identifier",
            )
            if mapping["retrieved_support_identifier"] is not None
            else None
        ),
        foreground_prototype_identity_sha256=_expect_string(
            mapping["foreground_prototype_identity_sha256"],
            field_name="foreground_prototype_identity_sha256",
        ),
        background_prototype_identity_sha256=_expect_string(
            mapping["background_prototype_identity_sha256"],
            field_name="background_prototype_identity_sha256",
        ),
        inference_identity_sha256=_expect_string(
            mapping["inference_identity_sha256"],
            field_name="inference_identity_sha256",
        ),
        prediction_content_sha256=_expect_string(
            mapping["prediction_content_sha256"],
            field_name="prediction_content_sha256",
        ),
        foreground_score_content_sha256=_expect_string(
            mapping["foreground_score_content_sha256"],
            field_name="foreground_score_content_sha256",
        ),
        background_score_content_sha256=_expect_string(
            mapping["background_score_content_sha256"],
            field_name="background_score_content_sha256",
        ),
        confidence_margin_content_sha256=(
            _expect_string(
                mapping["confidence_margin_content_sha256"],
                field_name="confidence_margin_content_sha256",
            )
            if mapping["confidence_margin_content_sha256"] is not None
            else None
        ),
        metrics_availability_status=_expect_string(
            mapping["metrics_availability_status"],
            field_name="metrics_availability_status",
        ),
        dice=(
            _expect_float(mapping["dice"], field_name="dice")
            if mapping["dice"] is not None
            else None
        ),
        iou=(
            _expect_float(mapping["iou"], field_name="iou") if mapping["iou"] is not None else None
        ),
        empty_mask_behavior=_expect_string(
            mapping["empty_mask_behavior"],
            field_name="empty_mask_behavior",
        ),
    )


def phase5_comparison_table_to_dict(table: Phase5ComparisonTable) -> dict[str, JsonValue]:
    """Convert one comparison table to a canonical JSON-compatible mapping."""

    return {
        "comparison_identity_sha256": table.comparison_identity_sha256,
        "dataset_manifest_hash": table.dataset_manifest_hash,
        "metrics_reference_available": table.metrics_reference_available,
        "ordered_method_definitions": [
            phase5_method_definition_to_dict(item) for item in table.ordered_method_definitions
        ],
        "query_case_id": table.query_case_id,
        "query_identity": table.query_identity,
        "query_patient_id": table.query_patient_id,
        "records": [phase5_comparison_record_to_dict(item) for item in table.records],
        "schema_name": table.schema_name,
        "schema_version": table.schema_version,
        "support_set_identity_sha256": table.support_set_identity_sha256,
    }


def phase5_comparison_table_from_mapping(
    mapping: Mapping[str, object],
) -> Phase5ComparisonTable:
    """Reconstruct one comparison table from a strict mapping."""

    _require_exact_keys(
        mapping,
        required={
            "comparison_identity_sha256",
            "dataset_manifest_hash",
            "metrics_reference_available",
            "ordered_method_definitions",
            "query_case_id",
            "query_identity",
            "query_patient_id",
            "records",
            "schema_name",
            "schema_version",
            "support_set_identity_sha256",
        },
        field_name="Phase5ComparisonTable",
    )
    method_definitions = tuple(
        phase5_method_definition_from_mapping(item)
        for item in _expect_mapping_tuple(
            mapping["ordered_method_definitions"],
            field_name="ordered_method_definitions",
        )
    )
    records = tuple(
        phase5_comparison_record_from_mapping(item)
        for item in _expect_mapping_tuple(mapping["records"], field_name="records")
    )
    return Phase5ComparisonTable(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        comparison_identity_sha256=_expect_string(
            mapping["comparison_identity_sha256"],
            field_name="comparison_identity_sha256",
        ),
        query_identity=_expect_string(mapping["query_identity"], field_name="query_identity"),
        query_patient_id=_expect_string(
            mapping["query_patient_id"],
            field_name="query_patient_id",
        ),
        query_case_id=_expect_string(mapping["query_case_id"], field_name="query_case_id"),
        dataset_manifest_hash=(
            _expect_string(mapping["dataset_manifest_hash"], field_name="dataset_manifest_hash")
            if mapping["dataset_manifest_hash"] is not None
            else None
        ),
        support_set_identity_sha256=_expect_string(
            mapping["support_set_identity_sha256"],
            field_name="support_set_identity_sha256",
        ),
        ordered_method_definitions=method_definitions,
        records=records,
        metrics_reference_available=_expect_bool(
            mapping["metrics_reference_available"],
            field_name="metrics_reference_available",
        ),
    )


def phase5_run_summary_to_dict(summary: Phase5RunSummary) -> dict[str, JsonValue]:
    """Convert one run summary to a canonical JSON-compatible mapping."""

    return {
        "comparison_identity_sha256": summary.comparison_identity_sha256,
        "foundation_model_adapter_enabled": summary.foundation_model_adapter_enabled,
        "method_count": summary.method_count,
        "metrics_reference_available": summary.metrics_reference_available,
        "nearest_support_top_k": summary.nearest_support_top_k,
        "published_filenames": list(summary.published_filenames),
        "schema_name": summary.schema_name,
        "schema_version": summary.schema_version,
        "support_set_identity_sha256": summary.support_set_identity_sha256,
    }


def phase5_run_summary_from_mapping(mapping: Mapping[str, object]) -> Phase5RunSummary:
    """Reconstruct one run summary from a strict mapping."""

    _require_exact_keys(
        mapping,
        required={
            "comparison_identity_sha256",
            "foundation_model_adapter_enabled",
            "method_count",
            "metrics_reference_available",
            "nearest_support_top_k",
            "published_filenames",
            "schema_name",
            "schema_version",
            "support_set_identity_sha256",
        },
        field_name="Phase5RunSummary",
    )
    return Phase5RunSummary(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        comparison_identity_sha256=_expect_string(
            mapping["comparison_identity_sha256"],
            field_name="comparison_identity_sha256",
        ),
        support_set_identity_sha256=_expect_string(
            mapping["support_set_identity_sha256"],
            field_name="support_set_identity_sha256",
        ),
        method_count=_expect_int(mapping["method_count"], field_name="method_count"),
        nearest_support_top_k=_expect_int(
            mapping["nearest_support_top_k"],
            field_name="nearest_support_top_k",
        ),
        metrics_reference_available=_expect_bool(
            mapping["metrics_reference_available"],
            field_name="metrics_reference_available",
        ),
        foundation_model_adapter_enabled=_expect_bool(
            mapping["foundation_model_adapter_enabled"],
            field_name="foundation_model_adapter_enabled",
        ),
        published_filenames=_expect_string_tuple(
            mapping["published_filenames"],
            field_name="published_filenames",
        ),
    )


def phase5_comparison_table_to_json(table: Phase5ComparisonTable) -> bytes:
    """Serialize one comparison table as canonical deterministic JSON bytes."""

    return canonical_json_bytes(phase5_comparison_table_to_dict(table)) + b"\n"


def phase5_comparison_table_from_json(value: bytes | str) -> Phase5ComparisonTable:
    """Reconstruct one comparison table from canonical JSON text or bytes."""

    try:
        payload = json.loads(value.decode("utf-8") if isinstance(value, bytes) else value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase5ComparisonValidationError("comparison table JSON is malformed.") from exc
    if not isinstance(payload, dict):
        raise Phase5ComparisonValidationError("comparison table JSON must contain a mapping.")
    return phase5_comparison_table_from_mapping(payload)


def phase5_run_summary_to_json(summary: Phase5RunSummary) -> bytes:
    """Serialize one run summary as canonical deterministic JSON bytes."""

    return canonical_json_bytes(phase5_run_summary_to_dict(summary)) + b"\n"


def phase5_run_summary_from_json(value: bytes | str) -> Phase5RunSummary:
    """Reconstruct one run summary from canonical JSON text or bytes."""

    try:
        payload = json.loads(value.decode("utf-8") if isinstance(value, bytes) else value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase5ComparisonValidationError("run summary JSON is malformed.") from exc
    if not isinstance(payload, dict):
        raise Phase5ComparisonValidationError("run summary JSON must contain a mapping.")
    return phase5_run_summary_from_mapping(payload)


def _run_no_retrieval_method(
    *,
    query_feature_encoding: FeatureEncoding3D,
    support_records: tuple[Phase5SupportRecord, ...],
    query_patient_id: str,
    query_case_id: str,
    query_identity: str,
    dataset_manifest_hash: str | None,
    emit_confidence_margin: bool,
) -> _MethodExecutionResult:
    prototype_memory = build_support_prototype_memory(
        supports=tuple(_prototype_input(item) for item in support_records)
    )
    inference_result = infer_query_from_prototypes(
        query_feature_encoding=query_feature_encoding,
        foreground_prototype=prototype_memory.foreground_prototype,
        background_prototype=prototype_memory.background_prototype,
        query_patient_id=query_patient_id,
        query_case_id=query_case_id,
        query_identity=query_identity,
        query_dataset_manifest_hash=dataset_manifest_hash,
        emit_confidence_margin=emit_confidence_margin,
    )
    return _MethodExecutionResult(
        method_name=NO_RETRIEVAL_METHOD,
        selected_support_identifiers=tuple(item.support_identifier for item in support_records),
        prototype_memory=prototype_memory,
        inference_result=inference_result,
        retrieval_result=None,
        retrieval_result_identity_sha256=None,
        retrieved_support_identifier=None,
    )


def _run_nearest_support_method(
    *,
    query_feature_encoding: FeatureEncoding3D,
    support_records: tuple[Phase5SupportRecord, ...],
    query_patient_id: str,
    query_case_id: str,
    query_identity: str,
    dataset_manifest_hash: str | None,
    emit_confidence_margin: bool,
    nearest_support_top_k: int,
) -> _MethodExecutionResult:
    support_candidates = tuple(
        RetrievalCandidate(
            support_identifier=item.support_identifier,
            support_patient_id=item.support_patient_id,
            support_case_id=item.support_case_id,
            support_manifest_hash=item.support_manifest_hash,
            support_assignment_index=index,
            embedding_metadata=item.feature_encoding.metadata,
        )
        for index, item in enumerate(support_records)
    )
    request = RetrievalRequest(
        query_embedding_metadata=query_feature_encoding.metadata,
        support_candidates=support_candidates,
        similarity_metric="cosine",
        top_k=nearest_support_top_k,
    )
    retrieval_result = retrieve_top_k_by_exact_cosine(
        request=request,
        query_embedding=query_feature_encoding,
        support_embeddings={
            item.support_identifier: item.feature_encoding for item in support_records
        },
        query_patient_id=query_patient_id,
        query_case_id=query_case_id,
        query_dataset_manifest_hash=dataset_manifest_hash,
        support_dataset_manifest_hashes={
            item.support_identifier: item.dataset_manifest_hash for item in support_records
        },
    )
    retrieved_identifier = retrieval_result.ordered_matches[0].support_identifier
    selected_support = tuple(
        item for item in support_records if item.support_identifier == retrieved_identifier
    )
    if len(selected_support) != 1:
        raise Phase5ComparisonValidationError(
            "nearest support retrieval must select exactly one support."
        )
    prototype_memory = build_support_prototype_memory(
        supports=tuple(_prototype_input(item) for item in selected_support)
    )
    inference_result = infer_query_from_prototypes(
        query_feature_encoding=query_feature_encoding,
        foreground_prototype=prototype_memory.foreground_prototype,
        background_prototype=prototype_memory.background_prototype,
        query_patient_id=query_patient_id,
        query_case_id=query_case_id,
        query_identity=query_identity,
        query_dataset_manifest_hash=dataset_manifest_hash,
        emit_confidence_margin=emit_confidence_margin,
    )
    return _MethodExecutionResult(
        method_name=NEAREST_SUPPORT_RETRIEVAL_METHOD,
        selected_support_identifiers=(retrieved_identifier,),
        prototype_memory=prototype_memory,
        inference_result=inference_result,
        retrieval_result=retrieval_result,
        retrieval_result_identity_sha256=hash_phase5_retrieval_result(
            retrieval_result,
            query_identity=query_identity,
        ),
        retrieved_support_identifier=retrieved_identifier,
    )


def _build_comparison_record(
    *,
    method_result: _MethodExecutionResult,
    support_set_identity_sha256: str,
    reference_mask: np.ndarray | None,
) -> Phase5ComparisonRecord:
    dice: float | None = None
    iou: float | None = None
    metrics_status = METRICS_UNAVAILABLE_STATUS
    if reference_mask is not None:
        prediction_mask = method_result.inference_result.prediction_mask.astype(bool, copy=False)
        counts = compute_binary_confusion_counts(reference_mask, prediction_mask)
        dice = float(dice_from_counts(counts))
        iou = float(iou_from_counts(counts))
        metrics_status = METRICS_AVAILABLE_STATUS
    return Phase5ComparisonRecord(
        schema_name=PHASE5_COMPARISON_RECORD_SCHEMA_NAME,
        schema_version=PHASE5_COMPARISON_RECORD_SCHEMA_VERSION,
        method_name=method_result.method_name,
        selected_support_identifiers=method_result.selected_support_identifiers,
        support_count_used=len(method_result.selected_support_identifiers),
        support_set_identity_sha256=support_set_identity_sha256,
        retrieval_result_identity_sha256=method_result.retrieval_result_identity_sha256,
        retrieved_support_identifier=method_result.retrieved_support_identifier,
        foreground_prototype_identity_sha256=hash_support_prototype_identity(
            method_result.prototype_memory.foreground_prototype
        ),
        background_prototype_identity_sha256=hash_support_prototype_identity(
            method_result.prototype_memory.background_prototype
        ),
        inference_identity_sha256=hash_prototype_inference_identity(method_result.inference_result),
        prediction_content_sha256=method_result.inference_result.prediction_content_sha256,
        foreground_score_content_sha256=method_result.inference_result.foreground_score_content_sha256,
        background_score_content_sha256=method_result.inference_result.background_score_content_sha256,
        confidence_margin_content_sha256=method_result.inference_result.confidence_margin_content_sha256,
        metrics_availability_status=metrics_status,
        dice=dice,
        iou=iou,
        empty_mask_behavior=EMPTY_MASK_BEHAVIOR,
    )


def hash_phase5_support_set(support_records: tuple[Phase5SupportRecord, ...]) -> str:
    """Return the deterministic support-set identity hash for one support collection."""

    return sha256_json(
        {
            "schema_name": "phase5_support_set_identity",
            "schema_version": "v1",
            "supports": [
                {
                    "binary_mask_sha256": _array_content_sha256(
                        _normalize_binary_mask(item.binary_mask)
                    ),
                    "case_id": item.support_case_id,
                    "checkpoint_hash": item.feature_encoding.metadata.checkpoint_hash,
                    "dataset_manifest_hash": item.dataset_manifest_hash,
                    "encoder_identity": item.feature_encoding.metadata.encoder_identity,
                    "feature_content_sha256": _array_content_sha256(
                        _normalize_feature_array(item.feature_encoding.feature_data)
                    ),
                    "feature_stage": item.feature_encoding.metadata.feature_stage,
                    "normalization_name": item.feature_encoding.metadata.normalization_name,
                    "patient_id": item.support_patient_id,
                    "preprocessing_hash": item.feature_encoding.metadata.preprocessing_hash,
                    "support_identifier": item.support_identifier,
                    "support_manifest_hash": item.support_manifest_hash,
                }
                for item in support_records
            ],
        }
    )


def hash_phase5_retrieval_result(result: RetrievalResult, *, query_identity: str) -> str:
    """Return a deterministic identity hash for one retrieval result."""

    return sha256_json(
        {
            "ordered_matches": [
                {
                    "similarity_score": item.similarity_score,
                    "support_assignment_index": item.support_assignment_index,
                    "support_case_id": item.support_case_id,
                    "support_identifier": item.support_identifier,
                    "support_manifest_hash": item.support_manifest_hash,
                    "support_patient_id": item.support_patient_id,
                }
                for item in result.ordered_matches
            ],
            "query_identity": query_identity,
            "requested_top_k": result.requested_top_k,
            "schema_name": "phase5_retrieval_result_identity",
            "schema_version": "v1",
            "similarity_metric": result.similarity_metric,
        }
    )


def _normalize_support_records(
    support_records: tuple[Phase5SupportRecord, ...],
) -> tuple[Phase5SupportRecord, ...]:
    if not support_records:
        raise Phase5ComparisonValidationError("support_records must contain at least two supports.")
    if len(support_records) < 2:
        raise Phase5ComparisonValidationError("support_records must contain at least two supports.")
    normalized = tuple(
        sorted(
            support_records,
            key=lambda item: (
                item.support_patient_id,
                item.support_case_id,
                item.support_identifier,
            ),
        )
    )
    identifiers = [item.support_identifier for item in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise Phase5ComparisonValidationError("support identifiers must be unique.")
    patient_case_pairs = [(item.support_patient_id, item.support_case_id) for item in normalized]
    if len(set(patient_case_pairs)) != len(patient_case_pairs):
        raise Phase5ComparisonValidationError("support patient/case assignments must be unique.")
    first = normalized[0]
    first_array = _normalize_feature_array(first.feature_encoding.feature_data)
    for item in normalized[1:]:
        array = _normalize_feature_array(item.feature_encoding.feature_data)
        comparisons = (
            ("feature shape", first_array.shape, array.shape),
            (
                "encoder identity",
                first.feature_encoding.metadata.encoder_identity,
                item.feature_encoding.metadata.encoder_identity,
            ),
            (
                "preprocessing hash",
                first.feature_encoding.metadata.preprocessing_hash,
                item.feature_encoding.metadata.preprocessing_hash,
            ),
            (
                "checkpoint hash",
                first.feature_encoding.metadata.checkpoint_hash,
                item.feature_encoding.metadata.checkpoint_hash,
            ),
            (
                "dataset manifest hash",
                first.dataset_manifest_hash,
                item.dataset_manifest_hash,
            ),
            (
                "feature stage",
                first.feature_encoding.metadata.feature_stage,
                item.feature_encoding.metadata.feature_stage,
            ),
            (
                "normalization name",
                first.feature_encoding.metadata.normalization_name,
                item.feature_encoding.metadata.normalization_name,
            ),
        )
        for field_name, expected, observed in comparisons:
            if expected != observed:
                raise Phase5ComparisonValidationError(
                    f"support records must match on {field_name}."
                )
    return normalized


def _validate_query_support_disjointness(
    *,
    query_patient_id: str,
    query_case_id: str,
    query_identity: str,
    support_records: tuple[Phase5SupportRecord, ...],
) -> None:
    _require_identifier(query_patient_id, field_name="query_patient_id")
    _require_identifier(query_case_id, field_name="query_case_id")
    _require_identifier(query_identity, field_name="query_identity")
    for item in support_records:
        if item.support_patient_id == query_patient_id:
            raise Phase5ComparisonLeakageError("query/support patient overlap is not allowed.")
        if item.support_case_id == query_case_id:
            raise Phase5ComparisonLeakageError("query/support case overlap is not allowed.")
        if item.support_identifier == query_identity:
            raise Phase5ComparisonLeakageError(
                "query identity must not match a support identifier."
            )


def _prototype_input(record: Phase5SupportRecord) -> SupportPrototypeInput:
    return SupportPrototypeInput(
        support_identifier=record.support_identifier,
        support_patient_id=record.support_patient_id,
        support_case_id=record.support_case_id,
        dataset_manifest_hash=record.dataset_manifest_hash,
        feature_encoding=record.feature_encoding,
        binary_mask=record.binary_mask,
    )


def _normalize_reference_mask(
    *,
    reference_mask: np.ndarray | None,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if reference_mask is None:
        raise Phase5ComparisonValidationError("reference_mask must not be None here.")
    if not isinstance(reference_mask, np.ndarray):
        raise Phase5ComparisonValidationError("reference_mask must be a NumPy ndarray.")
    if reference_mask.ndim == 4:
        normalized = reference_mask[:, np.newaxis, :, :, :]
    elif reference_mask.ndim == 5:
        normalized = reference_mask
    else:
        raise Phase5ComparisonValidationError(
            "reference_mask must have shape [1,D,H,W] or [1,1,D,H,W]."
        )
    if tuple(int(value) for value in normalized.shape) != tuple(
        int(value) for value in expected_shape
    ):
        raise Phase5ComparisonValidationError(
            "reference_mask must match the prediction spatial output shape."
        )
    if not np.isfinite(normalized).all():
        raise Phase5ComparisonValidationError("reference_mask must not contain NaN or Infinity.")
    unique_values = {int(value) for value in np.unique(normalized)}
    if unique_values - _BINARY_VALUES:
        raise Phase5ComparisonValidationError("reference_mask must contain binary values only.")
    return normalized.astype(bool, copy=False)


def _normalize_feature_array(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise Phase5ComparisonValidationError("feature_data must be a NumPy ndarray.")
    if value.ndim != 5:
        raise Phase5ComparisonValidationError("feature_data must have shape [B,C,D,H,W].")
    if value.shape[0] != 1:
        raise Phase5ComparisonValidationError("feature_data must use batch size 1.")
    if value.dtype.kind != "f":
        raise Phase5ComparisonValidationError("feature_data must use a floating dtype.")
    if not np.isfinite(value).all():
        raise Phase5ComparisonValidationError("feature_data must not contain NaN or Infinity.")
    return np.ascontiguousarray(value)


def _normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise Phase5ComparisonValidationError("binary_mask must be a NumPy ndarray.")
    if mask.ndim == 4:
        normalized = mask
    elif mask.ndim == 5 and mask.shape[1] == 1:
        normalized = mask[:, 0, :, :, :]
    else:
        raise Phase5ComparisonValidationError(
            "binary_mask must have shape [1,D,H,W] or [1,1,D,H,W]."
        )
    if normalized.shape[0] != 1:
        raise Phase5ComparisonValidationError("binary_mask must use batch size 1.")
    if not np.isfinite(normalized).all():
        raise Phase5ComparisonValidationError("binary_mask must not contain NaN or Infinity.")
    unique_values = {int(value) for value in np.unique(normalized)}
    if unique_values - _BINARY_VALUES:
        raise Phase5ComparisonValidationError("binary_mask must contain binary values only.")
    return normalized.astype(np.uint8, copy=False)


def _array_content_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _render_phase5_comparison_markdown(table: Phase5ComparisonTable) -> bytes:
    lines = [
        "# Phase 5 Retrieval Comparison",
        "",
        (
            "| Method | Support Count | Retrieved Support | Prediction Hash | "
            "Dice | IoU | Metrics Status |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in table.records:
        lines.append(
            "| "
            f"{record.method_name} | "
            f"{record.support_count_used} | "
            f"{record.retrieved_support_identifier or '-'} | "
            f"{record.prediction_content_sha256[:12]} | "
            f"{record.dice if record.dice is not None else 'unavailable'} | "
            f"{record.iou if record.iou is not None else 'unavailable'} | "
            f"{record.metrics_availability_status} |"
        )
    if len(table.records) != 2:
        raise Phase5ComparisonValidationError(
            "comparison markdown must contain exactly two data rows."
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _validate_phase5_output_root(output_root: Path) -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        resolved_output_root = validate_explicit_external_output_root(
            output_root,
            forbidden_roots=(repository_root,),
        )
    except InvalidDatasetRootError as exc:
        raise Phase5ComparisonPathError(str(exc)) from exc
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
            raise Phase5ComparisonPathError("output_root must be a directory when it exists.")
        _require_no_symlinks_in_tree(output_root)
        if any(output_root.iterdir()):
            if _existing_tree_matches(
                output_root=output_root,
                artifact_bytes=artifact_bytes,
                expected_directory_paths=expected_directory_paths,
            ):
                return True
            raise Phase5ComparisonCollisionError(
                "output_root already contains non-identical published artifacts."
            )

    staging_root = output_root.parent / f".{output_root.name}.phase5-publication.tmp"
    if staging_root.exists():
        raise Phase5ComparisonCollisionError("staging output root already exists.")
    try:
        staging_root.mkdir(mode=0o700)
        for relative_directory in expected_directory_paths:
            (staging_root / relative_directory).mkdir(parents=True, exist_ok=True)
        for relative_path, payload in dict(sorted(artifact_bytes.items())).items():
            _write_bytes_in_staging_root(
                staging_root=staging_root,
                relative_path=relative_path,
                payload=payload,
            )
        if output_root.exists():
            if any(output_root.iterdir()):
                raise Phase5ComparisonCollisionError(
                    "output_root became non-empty before staged publication."
                )
            output_root.rmdir()
        os.replace(staging_root, output_root)
    except Phase5ComparisonError:
        _remove_tree_if_present(staging_root)
        raise
    except OSError as exc:
        _remove_tree_if_present(staging_root)
        raise Phase5ComparisonIOError("failed to publish Phase 5 retrieval artifacts.") from exc
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
        raise Phase5ComparisonPathError("relative publication path is unsafe.") from exc


def _expected_directory_paths(artifact_bytes: dict[str, bytes]) -> set[str]:
    directories: set[str] = set()
    for relative_path in artifact_bytes:
        parts = relative_path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    return directories


def _existing_tree_matches(
    *,
    output_root: Path,
    artifact_bytes: dict[str, bytes],
    expected_directory_paths: set[str],
) -> bool:
    existing_directory_paths: set[str] = set()
    existing_file_paths: dict[str, bytes] = {}
    for path in sorted(output_root.rglob("*")):
        relative_path = path.relative_to(output_root).as_posix()
        if path.is_symlink():
            raise Phase5ComparisonPathError("output_root must not contain symlinked paths.")
        if path.is_dir():
            existing_directory_paths.add(relative_path)
        elif path.is_file():
            existing_file_paths[relative_path] = path.read_bytes()
        else:
            raise Phase5ComparisonPathError("output_root must contain only directories or files.")
    if existing_directory_paths != expected_directory_paths:
        return False
    return existing_file_paths == artifact_bytes


def _require_no_parent_traversal_components(path: Path) -> None:
    """Reject lexical parent traversal before publishing generated artifacts."""
    if ".." in path.parts:
        raise Phase5ComparisonPathError("output_root must not contain parent traversal.")


def _require_no_symlink_components(path: Path, *, resolved_output_root: Path) -> None:
    """Reject symlink escapes while allowing the canonical macOS /tmp alias."""
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.exists() and current.is_symlink():
            if _is_allowed_phase5_output_alias(
                current,
                resolved_output_root=resolved_output_root,
            ):
                continue
            raise Phase5ComparisonPathError("output_root must not traverse symlinked paths.")
        if current.exists() and not current.is_dir() and current != path:
            raise Phase5ComparisonPathError("output_root parent chain must contain directories.")


def _is_allowed_phase5_output_alias(
    symlink_path: Path,
    *,
    resolved_output_root: Path,
) -> bool:
    """Allow only the platform /tmp alias when it contains the canonical output root."""
    if symlink_path != Path("/tmp"):
        return False
    try:
        resolved_symlink = symlink_path.resolve(strict=True)
    except OSError:
        return False
    return _path_is_equal_or_nested(resolved_output_root, resolved_symlink)


def _path_is_equal_or_nested(path: Path, root: Path) -> bool:
    """Return whether a resolved path is equal to or contained by a resolved root."""
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
            raise Phase5ComparisonPathError("output_root must not contain symlinked paths.")


def _remove_tree_if_present(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _require_schema_name(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise Phase5ComparisonValidationError(f"{field_name} must be {expected!r}, got {value!r}.")


def _require_schema_version(value: str, *, expected: str, field_name: str) -> None:
    if value != expected:
        raise Phase5ComparisonValidationError(f"{field_name} must be {expected!r}, got {value!r}.")


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise Phase5ComparisonValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest."
        )


def _require_identifier(value: str, *, field_name: str) -> None:
    if not value or value != value.strip():
        raise Phase5ComparisonValidationError(f"{field_name} must be a non-empty identifier.")


def _require_metric(value: float, *, field_name: str) -> None:
    if not isinstance(value, float | int) or not np.isfinite(float(value)):
        raise Phase5ComparisonValidationError(f"{field_name} must be finite.")
    if float(value) < 0.0 or float(value) > 1.0:
        raise Phase5ComparisonValidationError(f"{field_name} must lie in [0.0, 1.0].")


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase5ComparisonValidationError(f"{field_name} must be a string.")
    return value


def _expect_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Phase5ComparisonValidationError(f"{field_name} must be an integer.")
    return value


def _expect_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise Phase5ComparisonValidationError(f"{field_name} must be numeric.")
    return float(value)


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase5ComparisonValidationError(f"{field_name} must be boolean.")
    return value


def _expect_string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Phase5ComparisonValidationError(f"{field_name} must be a JSON array.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise Phase5ComparisonValidationError(f"{field_name} must contain only string items.")
        result.append(item)
    return tuple(result)


def _expect_mapping_tuple(
    value: object,
    *,
    field_name: str,
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise Phase5ComparisonValidationError(f"{field_name} must be a JSON array.")
    items: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise Phase5ComparisonValidationError(f"{field_name}[{index}] must be a JSON object.")
        items.append(cast(Mapping[str, object], item))
    return tuple(items)


def _require_exact_keys(
    mapping: Mapping[str, object],
    *,
    required: set[str],
    field_name: str,
) -> None:
    observed = set(mapping)
    if observed != required:
        extra = sorted(observed - required)
        missing = sorted(required - observed)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing!r}")
        if extra:
            details.append(f"unknown {extra!r}")
        raise Phase5ComparisonValidationError(f"{field_name} keys mismatch: {', '.join(details)}.")


__all__ = [
    "EMPTY_MASK_BEHAVIOR",
    "FIXED_METHOD_ORDER",
    "FIXED_NEAREST_SUPPORT_TOP_K",
    "METRICS_AVAILABLE_STATUS",
    "METRICS_UNAVAILABLE_STATUS",
    "NEAREST_SUPPORT_RETRIEVAL_METHOD",
    "NO_RETRIEVAL_METHOD",
    "PHASE5_COMPARISON_JSON_NAME",
    "PHASE5_COMPARISON_MARKDOWN_NAME",
    "PHASE5_COMPARISON_RECORD_SCHEMA_NAME",
    "PHASE5_COMPARISON_RECORD_SCHEMA_VERSION",
    "PHASE5_COMPARISON_TABLE_SCHEMA_NAME",
    "PHASE5_COMPARISON_TABLE_SCHEMA_VERSION",
    "PHASE5_EFFECTIVE_CONFIG_JSON_NAME",
    "PHASE5_METHOD_DEFINITION_SCHEMA_NAME",
    "PHASE5_METHOD_DEFINITION_SCHEMA_VERSION",
    "PHASE5_RUN_SUMMARY_JSON_NAME",
    "PHASE5_RUN_SUMMARY_SCHEMA_NAME",
    "PHASE5_RUN_SUMMARY_SCHEMA_VERSION",
    "PHASE5_SUPPORT_RECORD_SCHEMA_NAME",
    "PHASE5_SUPPORT_RECORD_SCHEMA_VERSION",
    "Phase5ComparisonCollisionError",
    "Phase5ComparisonConfigError",
    "Phase5ComparisonError",
    "Phase5ComparisonIOError",
    "Phase5ComparisonLeakageError",
    "Phase5ComparisonPathError",
    "Phase5ComparisonRecord",
    "Phase5ComparisonTable",
    "Phase5ComparisonValidationError",
    "Phase5MethodDefinition",
    "Phase5PublicationResult",
    "Phase5RunSummary",
    "Phase5Settings",
    "Phase5SupportRecord",
    "build_phase5_synthetic_fixture",
    "hash_phase5_comparison_table",
    "hash_phase5_retrieval_result",
    "hash_phase5_support_set",
    "load_phase5_foundation_retrieval_settings",
    "phase5_comparison_record_from_mapping",
    "phase5_comparison_record_to_dict",
    "phase5_comparison_table_from_json",
    "phase5_comparison_table_from_mapping",
    "phase5_comparison_table_identity_payload",
    "phase5_comparison_table_to_dict",
    "phase5_comparison_table_to_json",
    "phase5_method_definition_from_mapping",
    "phase5_method_definition_to_dict",
    "phase5_run_summary_from_json",
    "phase5_run_summary_from_mapping",
    "phase5_run_summary_to_dict",
    "phase5_run_summary_to_json",
    "run_and_publish_phase5_retrieval",
    "run_phase5_comparison",
]
