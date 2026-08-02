"""Deterministic binary predictive entropy for Phase 7 uncertainty assessment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.uncertainty.artifacts import (
    PHASE7_UNCERTAINTY_RESULT_SCHEMA_NAME,
    PHASE7_UNCERTAINTY_RESULT_SCHEMA_VERSION,
    Phase7UncertaintyResult,
)
from protoem_ct.uncertainty.contracts import (
    BinaryPredictiveProbabilityMap,
    PredictiveEntropyInput,
    PredictiveEntropyResultMetadata,
    ProbabilityMapValidationError,
    array_content_sha256,
    build_predictive_entropy_input,
    build_predictive_entropy_result_metadata,
)


class PredictiveEntropyError(ValueError):
    """Raised when deterministic predictive entropy cannot be computed."""


@dataclass(frozen=True, slots=True)
class PredictiveEntropyComputationResult:
    """Pure in-memory result for one binary predictive entropy computation."""

    entropy_input: PredictiveEntropyInput
    entropy_map: np.ndarray
    entropy_map_content_hash: str
    entropy_result_metadata: PredictiveEntropyResultMetadata
    uncertainty_result: Phase7UncertaintyResult
    valid_voxel_count: int


def compute_predictive_entropy(
    probability_map: BinaryPredictiveProbabilityMap,
    *,
    entropy_input: PredictiveEntropyInput | None = None,
) -> PredictiveEntropyComputationResult:
    """Compute binary entropy from validated foreground/background probabilities.

    The formula is ``H = -(p_fg log(p_fg) + p_bg log(p_bg))`` using natural
    logarithms. Values with probability exactly zero contribute exactly zero to
    the sum, implementing the standard ``0 log 0 = 0`` convention without
    clipping probabilities.
    """

    _revalidate_probability_map(probability_map)
    resolved_entropy_input = entropy_input or build_predictive_entropy_input(probability_map)
    _revalidate_entropy_input(resolved_entropy_input, probability_map)
    if probability_map.source_prediction_identity_hash is None:
        raise PredictiveEntropyError(
            "source_prediction_identity_hash is required to build a Phase 7 uncertainty artifact."
        )

    foreground = np.ascontiguousarray(probability_map.foreground_probability_map, dtype=np.float64)
    background = np.ascontiguousarray(probability_map.background_probability_map, dtype=np.float64)
    entropy_map = np.ascontiguousarray(
        _zero_safe_entropy_term(foreground) + _zero_safe_entropy_term(background),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(entropy_map)):
        raise PredictiveEntropyError("predictive entropy output must contain only finite values.")

    entropy_hash = array_content_sha256(entropy_map)
    valid_voxel_count = int(entropy_map.size)
    metadata = build_predictive_entropy_result_metadata(
        entropy_input=resolved_entropy_input,
        entropy_map_content_hash=entropy_hash,
        valid_voxel_count=valid_voxel_count,
    )
    uncertainty_result = _build_available_uncertainty_result(
        probability_map=probability_map,
        entropy_map_content_hash=entropy_hash,
        mean_uncertainty=float(np.mean(entropy_map)),
        max_uncertainty=float(np.max(entropy_map)),
    )
    return PredictiveEntropyComputationResult(
        entropy_input=resolved_entropy_input,
        entropy_map=entropy_map,
        entropy_map_content_hash=entropy_hash,
        entropy_result_metadata=metadata,
        uncertainty_result=uncertainty_result,
        valid_voxel_count=valid_voxel_count,
    )


def query_label_not_part_of_predictive_entropy_api() -> bool:
    """Return True because predictive entropy accepts no query labels or masks."""

    return True


def _zero_safe_entropy_term(probabilities: np.ndarray) -> np.ndarray:
    contribution = np.zeros_like(probabilities, dtype=np.float64)
    positive = probabilities > 0.0
    contribution[positive] = -probabilities[positive] * np.log(probabilities[positive])
    return contribution


def _revalidate_probability_map(probability_map: BinaryPredictiveProbabilityMap) -> None:
    BinaryPredictiveProbabilityMap(
        schema_name=probability_map.schema_name,
        schema_version=probability_map.schema_version,
        probability_map_identity_hash=probability_map.probability_map_identity_hash,
        foreground_probability_map=probability_map.foreground_probability_map,
        background_probability_map=probability_map.background_probability_map,
        foreground_probability_content_hash=probability_map.foreground_probability_content_hash,
        background_probability_content_hash=probability_map.background_probability_content_hash,
        probability_shape=probability_map.probability_shape,
        probability_sum_tolerance=probability_map.probability_sum_tolerance,
        source_prediction_identity_hash=probability_map.source_prediction_identity_hash,
        common_grid_identity_hash=probability_map.common_grid_identity_hash,
    )


def _revalidate_entropy_input(
    entropy_input: PredictiveEntropyInput,
    probability_map: BinaryPredictiveProbabilityMap,
) -> None:
    PredictiveEntropyInput(
        schema_name=entropy_input.schema_name,
        schema_version=entropy_input.schema_version,
        entropy_input_identity_hash=entropy_input.entropy_input_identity_hash,
        probability_map_identity_hash=entropy_input.probability_map_identity_hash,
        probability_shape=entropy_input.probability_shape,
        log_base=entropy_input.log_base,
        zero_probability_policy=entropy_input.zero_probability_policy,
    )
    if entropy_input.probability_map_identity_hash != probability_map.probability_map_identity_hash:
        raise ProbabilityMapValidationError(
            "entropy_input probability_map_identity_hash must match the probability map."
        )
    if entropy_input.probability_shape != probability_map.probability_shape:
        raise ProbabilityMapValidationError(
            "entropy_input probability_shape must match the probability map."
        )


def _build_available_uncertainty_result(
    *,
    probability_map: BinaryPredictiveProbabilityMap,
    entropy_map_content_hash: str,
    mean_uncertainty: float,
    max_uncertainty: float,
) -> Phase7UncertaintyResult:
    schema_name = PHASE7_UNCERTAINTY_RESULT_SCHEMA_NAME
    schema_version = PHASE7_UNCERTAINTY_RESULT_SCHEMA_VERSION
    source_prediction_hash = probability_map.source_prediction_identity_hash
    if source_prediction_hash is None:
        raise PredictiveEntropyError("source_prediction_identity_hash is required.")
    source_probability_hash = probability_map.probability_map_identity_hash
    uncertainty_type = "predictive_entropy"
    probability_space = "binary_foreground"
    log_base = "natural"
    sample_count = 1
    variance_map_content_hash = None
    availability_status = "available"
    unavailable_reason = None
    payload = {
        "schema_name": PHASE7_UNCERTAINTY_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_UNCERTAINTY_RESULT_SCHEMA_VERSION,
        "source_prediction_hash": source_prediction_hash,
        "source_probability_hash": source_probability_hash,
        "uncertainty_type": uncertainty_type,
        "probability_space": probability_space,
        "log_base": log_base,
        "sample_count": sample_count,
        "uncertainty_map_content_hash": entropy_map_content_hash,
        "variance_map_content_hash": variance_map_content_hash,
        "mean_uncertainty": mean_uncertainty,
        "max_uncertainty": max_uncertainty,
        "availability_status": availability_status,
        "unavailable_reason": unavailable_reason,
    }
    return Phase7UncertaintyResult(
        schema_name=schema_name,
        schema_version=schema_version,
        uncertainty_result_hash=sha256_json(payload),
        source_prediction_hash=source_prediction_hash,
        source_probability_hash=source_probability_hash,
        uncertainty_type=uncertainty_type,
        probability_space=probability_space,
        log_base=log_base,
        sample_count=sample_count,
        uncertainty_map_content_hash=entropy_map_content_hash,
        variance_map_content_hash=variance_map_content_hash,
        mean_uncertainty=mean_uncertainty,
        max_uncertainty=max_uncertainty,
        availability_status=availability_status,
        unavailable_reason=unavailable_reason,
    )
