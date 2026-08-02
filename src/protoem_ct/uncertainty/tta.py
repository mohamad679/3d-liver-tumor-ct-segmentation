"""Deterministic TTA and ensemble probability variance for Phase 7."""

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
    TTAContractValidationError,
    TTAProbabilitySamples,
    UncertaintyIdentityMismatchError,
    array_content_sha256,
)


class TTAVarianceError(ValueError):
    """Raised when deterministic TTA variance cannot be computed."""


class TTAVarianceAlignmentError(TTAVarianceError):
    """Raised when TTA samples are not aligned on one common grid."""


@dataclass(frozen=True, slots=True)
class TTAVarianceResult:
    """Pure in-memory result for deterministic population variance across samples."""

    probability_samples: TTAProbabilitySamples
    foreground_variance_map: np.ndarray
    background_variance_map: np.ndarray
    foreground_variance_content_hash: str
    background_variance_content_hash: str
    mean_foreground_variance: float
    max_foreground_variance: float
    uncertainty_result: Phase7UncertaintyResult


def compute_tta_probability_variance(
    probability_samples: TTAProbabilitySamples,
    *,
    uncertainty_type: str = "tta_variance",
) -> TTAVarianceResult:
    """Compute population variance from aligned binary probability samples.

    For ``N`` samples already restored to one common grid, this computes
    ``var_p(x) = (1/N) * sum_i (p_i(x) - mean_p(x))^2`` independently for
    foreground and background probabilities. The Phase 7 uncertainty artifact
    uses the foreground variance map as the canonical binary foreground
    uncertainty map; the background variance map remains available in the typed
    in-memory result.
    """

    if uncertainty_type not in {"tta_variance", "ensemble_variance"}:
        raise TTAVarianceError("uncertainty_type must be tta_variance or ensemble_variance.")
    _revalidate_probability_samples(probability_samples)
    _validate_common_grid_identity(probability_samples)
    _validate_prediction_sources(probability_samples)

    foreground_stack = _stack_probability_maps(
        probability_samples,
        probability_field="foreground",
    )
    background_stack = _stack_probability_maps(
        probability_samples,
        probability_field="background",
    )
    foreground_variance = _population_variance(foreground_stack)
    background_variance = _population_variance(background_stack)

    foreground_hash = array_content_sha256(foreground_variance)
    background_hash = array_content_sha256(background_variance)
    mean_foreground_variance = float(np.mean(foreground_variance))
    max_foreground_variance = float(np.max(foreground_variance))
    uncertainty_result = _build_available_uncertainty_result(
        probability_samples=probability_samples,
        uncertainty_type=uncertainty_type,
        foreground_variance_content_hash=foreground_hash,
        mean_foreground_variance=mean_foreground_variance,
        max_foreground_variance=max_foreground_variance,
    )
    return TTAVarianceResult(
        probability_samples=probability_samples,
        foreground_variance_map=foreground_variance,
        background_variance_map=background_variance,
        foreground_variance_content_hash=foreground_hash,
        background_variance_content_hash=background_hash,
        mean_foreground_variance=mean_foreground_variance,
        max_foreground_variance=max_foreground_variance,
        uncertainty_result=uncertainty_result,
    )


def compute_ensemble_probability_variance(
    probability_samples: TTAProbabilitySamples,
) -> TTAVarianceResult:
    """Compute the same deterministic variance for explicit ensemble samples."""

    return compute_tta_probability_variance(
        probability_samples,
        uncertainty_type="ensemble_variance",
    )


def query_label_not_part_of_tta_api() -> bool:
    """Return True because TTA variance accepts no query labels, masks, or models."""

    return True


def _revalidate_probability_samples(probability_samples: TTAProbabilitySamples) -> None:
    try:
        TTAProbabilitySamples(
            schema_name=probability_samples.schema_name,
            schema_version=probability_samples.schema_version,
            tta_probability_samples_identity_hash=(
                probability_samples.tta_probability_samples_identity_hash
            ),
            sample_manifest=probability_samples.sample_manifest,
            probability_maps=probability_samples.probability_maps,
            sample_count=probability_samples.sample_count,
            variance_population_policy=probability_samples.variance_population_policy,
        )
    except (TTAContractValidationError, UncertaintyIdentityMismatchError) as exc:
        raise TTAVarianceAlignmentError("TTA probability samples are invalid.") from exc


def _validate_common_grid_identity(probability_samples: TTAProbabilitySamples) -> None:
    manifest_grid_hashes = {
        record.common_grid_identity_hash
        for record in probability_samples.sample_manifest.sample_records
    }
    map_grid_hashes = {
        probability_map.common_grid_identity_hash
        for probability_map in probability_samples.probability_maps
    }
    if len(manifest_grid_hashes) != 1 or len(map_grid_hashes) != 1:
        raise TTAVarianceAlignmentError("TTA samples must share exactly one common grid.")
    manifest_grid_hash = next(iter(manifest_grid_hashes))
    map_grid_hash = next(iter(map_grid_hashes))
    if map_grid_hash is None or manifest_grid_hash != map_grid_hash:
        raise TTAVarianceAlignmentError(
            "TTA probability maps must match the manifest common-grid identity."
        )


def _validate_prediction_sources(probability_samples: TTAProbabilitySamples) -> None:
    missing_sources = [
        probability_map
        for probability_map in probability_samples.probability_maps
        if probability_map.source_prediction_identity_hash is None
    ]
    if missing_sources:
        raise TTAVarianceError(
            "source_prediction_identity_hash is required for every TTA probability sample."
        )


def _stack_probability_maps(
    probability_samples: TTAProbabilitySamples,
    *,
    probability_field: str,
) -> np.ndarray:
    maps: list[np.ndarray] = []
    for probability_map in probability_samples.probability_maps:
        maps.append(_extract_probability_map(probability_map, probability_field=probability_field))
    stacked = np.stack(maps, axis=0).astype(np.float64, copy=False)
    contiguous = np.ascontiguousarray(stacked)
    if not np.all(np.isfinite(contiguous)):
        raise TTAVarianceError("TTA probability samples must contain only finite values.")
    return contiguous


def _extract_probability_map(
    probability_map: BinaryPredictiveProbabilityMap,
    *,
    probability_field: str,
) -> np.ndarray:
    if probability_field == "foreground":
        array = probability_map.foreground_probability_map
    elif probability_field == "background":
        array = probability_map.background_probability_map
    else:
        raise TTAVarianceError("probability_field must be foreground or background.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _population_variance(samples: np.ndarray) -> np.ndarray:
    mean = np.mean(samples, axis=0, dtype=np.float64)
    variance = np.mean((samples - mean) ** 2, axis=0, dtype=np.float64)
    contiguous = np.ascontiguousarray(variance, dtype=np.float64)
    if not np.all(np.isfinite(contiguous)):
        raise TTAVarianceError("TTA variance output must contain only finite values.")
    if np.any(contiguous < 0.0):
        raise TTAVarianceError("TTA variance output must be nonnegative.")
    return contiguous


def _build_available_uncertainty_result(
    *,
    probability_samples: TTAProbabilitySamples,
    uncertainty_type: str,
    foreground_variance_content_hash: str,
    mean_foreground_variance: float,
    max_foreground_variance: float,
) -> Phase7UncertaintyResult:
    schema_name = PHASE7_UNCERTAINTY_RESULT_SCHEMA_NAME
    schema_version = PHASE7_UNCERTAINTY_RESULT_SCHEMA_VERSION
    source_prediction_hash = probability_samples.sample_manifest.sample_manifest_identity_hash
    source_probability_hash = probability_samples.tta_probability_samples_identity_hash
    probability_space = "binary_foreground"
    log_base = "natural"
    sample_count = probability_samples.sample_count
    availability_status = "available"
    unavailable_reason = None
    payload = {
        "schema_name": schema_name,
        "schema_version": schema_version,
        "source_prediction_hash": source_prediction_hash,
        "source_probability_hash": source_probability_hash,
        "uncertainty_type": uncertainty_type,
        "probability_space": probability_space,
        "log_base": log_base,
        "sample_count": sample_count,
        "uncertainty_map_content_hash": foreground_variance_content_hash,
        "variance_map_content_hash": foreground_variance_content_hash,
        "mean_uncertainty": mean_foreground_variance,
        "max_uncertainty": max_foreground_variance,
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
        uncertainty_map_content_hash=foreground_variance_content_hash,
        variance_map_content_hash=foreground_variance_content_hash,
        mean_uncertainty=mean_foreground_variance,
        max_uncertainty=max_foreground_variance,
        availability_status=availability_status,
        unavailable_reason=unavailable_reason,
    )
