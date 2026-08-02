"""Typed Phase 7 uncertainty contracts without uncertainty computation."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, sha256_json

PHASE7_BINARY_PROBABILITY_MAP_SCHEMA_NAME: Final[str] = "phase7_binary_probability_map"
PHASE7_BINARY_PROBABILITY_MAP_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_ENTROPY_INPUT_SCHEMA_NAME: Final[str] = "phase7_entropy_input"
PHASE7_ENTROPY_INPUT_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_ENTROPY_RESULT_METADATA_SCHEMA_NAME: Final[str] = "phase7_entropy_result_metadata"
PHASE7_ENTROPY_RESULT_METADATA_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_TTA_SAMPLE_RECORD_SCHEMA_NAME: Final[str] = "phase7_tta_sample_record"
PHASE7_TTA_SAMPLE_RECORD_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_TTA_SAMPLE_MANIFEST_SCHEMA_NAME: Final[str] = "phase7_tta_sample_manifest"
PHASE7_TTA_SAMPLE_MANIFEST_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_TTA_PROBABILITY_SAMPLES_SCHEMA_NAME: Final[str] = "phase7_tta_probability_samples"
PHASE7_TTA_PROBABILITY_SAMPLES_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_UNCERTAINTY_SUMMARY_INPUT_SCHEMA_NAME: Final[str] = "phase7_uncertainty_summary_input"
PHASE7_UNCERTAINTY_SUMMARY_INPUT_SCHEMA_VERSION: Final[str] = "v1"
PHASE7_EVALUATION_ELIGIBILITY_SCHEMA_NAME: Final[str] = "phase7_evaluation_eligibility"
PHASE7_EVALUATION_ELIGIBILITY_SCHEMA_VERSION: Final[str] = "v1"

SUPPORTED_ENTROPY_LOG_BASES: Final[frozenset[str]] = frozenset({"natural"})
SUPPORTED_UNCERTAINTY_SOURCE_TYPES: Final[frozenset[str]] = frozenset(
    {"predictive_entropy", "tta_variance", "ensemble_variance"}
)
SUPPORTED_ELIGIBILITY_STATUSES: Final[frozenset[str]] = frozenset({"eligible", "unavailable"})
SUPPORTED_EVALUATION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "predictive_entropy",
        "tta_variance",
        "calibration",
        "risk_coverage",
        "uncertainty_error_correlation",
        "failure_detection_auroc",
        "lesion_subgroup",
        "degradation",
    }
)

EligibilityStatus = Literal["eligible", "unavailable"]

_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_BOOL_DTYPE: Final[np.dtype[np.bool_]] = np.dtype(np.bool_)
_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_PROBABILITY_SUM_TOLERANCE: Final[float] = 1e-12
_SAFE_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class Phase7UncertaintyContractError(ValueError):
    """Base error for Phase 7 uncertainty contract validation."""


class ProbabilityMapValidationError(Phase7UncertaintyContractError):
    """Raised when a predictive probability map is invalid."""


class TTAContractValidationError(Phase7UncertaintyContractError):
    """Raised when a deterministic TTA sample contract is invalid."""


class UncertaintyEligibilityValidationError(Phase7UncertaintyContractError):
    """Raised when an uncertainty eligibility contract is invalid."""


class UncertaintyIdentityMismatchError(Phase7UncertaintyContractError):
    """Raised when a Phase 7 uncertainty contract self-identity hash mismatches."""


@dataclass(frozen=True, slots=True)
class BinaryPredictiveProbabilityMap:
    """Foreground/background predictive probabilities for one binary 3D case."""

    schema_name: str
    schema_version: str
    probability_map_identity_hash: str
    foreground_probability_map: np.ndarray
    background_probability_map: np.ndarray
    foreground_probability_content_hash: str
    background_probability_content_hash: str
    probability_shape: tuple[int, int, int, int, int]
    probability_sum_tolerance: float
    source_prediction_identity_hash: str | None = None
    common_grid_identity_hash: str | None = None

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            expected_name=PHASE7_BINARY_PROBABILITY_MAP_SCHEMA_NAME,
            expected_version=PHASE7_BINARY_PROBABILITY_MAP_SCHEMA_VERSION,
            error_type=ProbabilityMapValidationError,
        )
        _require_sha256(
            self.probability_map_identity_hash,
            field_name="probability_map_identity_hash",
            error_type=ProbabilityMapValidationError,
        )
        foreground = _require_probability_volume(
            self.foreground_probability_map,
            field_name="foreground_probability_map",
        )
        background = _require_probability_volume(
            self.background_probability_map,
            field_name="background_probability_map",
        )
        if foreground.shape != background.shape:
            raise ProbabilityMapValidationError(
                "foreground and background probability maps must share shape [1,1,D,H,W]."
            )
        if self.probability_shape != cast(tuple[int, int, int, int, int], foreground.shape):
            raise ProbabilityMapValidationError(
                "probability_shape must match the foreground/background map shape."
            )
        if not math.isfinite(self.probability_sum_tolerance) or self.probability_sum_tolerance < 0:
            raise ProbabilityMapValidationError(
                "probability_sum_tolerance must be finite and nonnegative."
            )
        if not np.allclose(
            foreground + background,
            1.0,
            rtol=0.0,
            atol=self.probability_sum_tolerance,
        ):
            raise ProbabilityMapValidationError(
                "foreground/background probabilities must sum to one voxelwise."
            )
        _require_sha256(
            self.foreground_probability_content_hash,
            field_name="foreground_probability_content_hash",
            error_type=ProbabilityMapValidationError,
        )
        _require_sha256(
            self.background_probability_content_hash,
            field_name="background_probability_content_hash",
            error_type=ProbabilityMapValidationError,
        )
        if self.source_prediction_identity_hash is not None:
            _require_sha256(
                self.source_prediction_identity_hash,
                field_name="source_prediction_identity_hash",
                error_type=ProbabilityMapValidationError,
            )
        if self.common_grid_identity_hash is not None:
            _require_sha256(
                self.common_grid_identity_hash,
                field_name="common_grid_identity_hash",
                error_type=ProbabilityMapValidationError,
            )
        if self.foreground_probability_content_hash != array_content_sha256(foreground):
            raise ProbabilityMapValidationError(
                "foreground_probability_content_hash must match foreground probability content."
            )
        if self.background_probability_content_hash != array_content_sha256(background):
            raise ProbabilityMapValidationError(
                "background_probability_content_hash must match background probability content."
            )
        if self.probability_map_identity_hash != hash_binary_predictive_probability_map(self):
            raise UncertaintyIdentityMismatchError(
                "probability_map_identity_hash does not match deterministic probability content."
            )


@dataclass(frozen=True, slots=True)
class PredictiveEntropyInput:
    """Validated metadata input for a future predictive-entropy calculation."""

    schema_name: str
    schema_version: str
    entropy_input_identity_hash: str
    probability_map_identity_hash: str
    probability_shape: tuple[int, int, int, int, int]
    log_base: str
    zero_probability_policy: str

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            expected_name=PHASE7_ENTROPY_INPUT_SCHEMA_NAME,
            expected_version=PHASE7_ENTROPY_INPUT_SCHEMA_VERSION,
            error_type=ProbabilityMapValidationError,
        )
        _require_sha256(
            self.entropy_input_identity_hash,
            field_name="entropy_input_identity_hash",
            error_type=ProbabilityMapValidationError,
        )
        _require_sha256(
            self.probability_map_identity_hash,
            field_name="probability_map_identity_hash",
            error_type=ProbabilityMapValidationError,
        )
        _require_volume_shape_tuple(self.probability_shape, field_name="probability_shape")
        if self.log_base not in SUPPORTED_ENTROPY_LOG_BASES:
            raise ProbabilityMapValidationError(
                f"log_base must be one of {sorted(SUPPORTED_ENTROPY_LOG_BASES)!r}."
            )
        if self.zero_probability_policy != "zero_log_zero_is_zero":
            raise ProbabilityMapValidationError(
                "zero_probability_policy must be 'zero_log_zero_is_zero'."
            )
        if self.entropy_input_identity_hash != hash_predictive_entropy_input(self):
            raise UncertaintyIdentityMismatchError(
                "entropy_input_identity_hash does not match deterministic metadata."
            )


@dataclass(frozen=True, slots=True)
class PredictiveEntropyResultMetadata:
    """Metadata contract for a future entropy map result, without computing entropy."""

    schema_name: str
    schema_version: str
    entropy_result_metadata_hash: str
    entropy_input_identity_hash: str
    entropy_map_content_hash: str
    entropy_shape: tuple[int, int, int, int, int]
    log_base: str
    valid_voxel_count: int

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            expected_name=PHASE7_ENTROPY_RESULT_METADATA_SCHEMA_NAME,
            expected_version=PHASE7_ENTROPY_RESULT_METADATA_SCHEMA_VERSION,
            error_type=ProbabilityMapValidationError,
        )
        for field_name in (
            "entropy_result_metadata_hash",
            "entropy_input_identity_hash",
            "entropy_map_content_hash",
        ):
            _require_sha256(
                cast(str, getattr(self, field_name)),
                field_name=field_name,
                error_type=ProbabilityMapValidationError,
            )
        _require_volume_shape_tuple(self.entropy_shape, field_name="entropy_shape")
        if self.log_base not in SUPPORTED_ENTROPY_LOG_BASES:
            raise ProbabilityMapValidationError(
                f"log_base must be one of {sorted(SUPPORTED_ENTROPY_LOG_BASES)!r}."
            )
        _require_nonnegative_int(
            self.valid_voxel_count,
            field_name="valid_voxel_count",
            error_type=ProbabilityMapValidationError,
        )
        if self.entropy_result_metadata_hash != hash_predictive_entropy_result_metadata(self):
            raise UncertaintyIdentityMismatchError(
                "entropy_result_metadata_hash does not match deterministic metadata."
            )


@dataclass(frozen=True, slots=True)
class TTASampleRecord:
    """One deterministic probability-sample provenance record for TTA or ensembles."""

    schema_name: str
    schema_version: str
    sample_identity_hash: str
    sample_index: int
    sample_id: str
    probability_map_identity_hash: str
    transform_manifest_hash: str | None
    common_grid_identity_hash: str

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            expected_name=PHASE7_TTA_SAMPLE_RECORD_SCHEMA_NAME,
            expected_version=PHASE7_TTA_SAMPLE_RECORD_SCHEMA_VERSION,
            error_type=TTAContractValidationError,
        )
        _require_nonnegative_int(
            self.sample_index,
            field_name="sample_index",
            error_type=TTAContractValidationError,
        )
        _require_safe_identifier(
            self.sample_id,
            field_name="sample_id",
            error_type=TTAContractValidationError,
        )
        for field_name in (
            "sample_identity_hash",
            "probability_map_identity_hash",
            "common_grid_identity_hash",
        ):
            _require_sha256(
                cast(str, getattr(self, field_name)),
                field_name=field_name,
                error_type=TTAContractValidationError,
            )
        if self.transform_manifest_hash is not None:
            _require_sha256(
                self.transform_manifest_hash,
                field_name="transform_manifest_hash",
                error_type=TTAContractValidationError,
            )
        if self.sample_identity_hash != hash_tta_sample_record(self):
            raise UncertaintyIdentityMismatchError(
                "sample_identity_hash does not match deterministic sample metadata."
            )


@dataclass(frozen=True, slots=True)
class TTASampleManifest:
    """Ordered deterministic manifest for TTA or ensemble probability samples."""

    schema_name: str
    schema_version: str
    sample_manifest_identity_hash: str
    sample_records: tuple[TTASampleRecord, ...]
    sample_count: int
    aggregation_scope: str

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            expected_name=PHASE7_TTA_SAMPLE_MANIFEST_SCHEMA_NAME,
            expected_version=PHASE7_TTA_SAMPLE_MANIFEST_SCHEMA_VERSION,
            error_type=TTAContractValidationError,
        )
        _require_sha256(
            self.sample_manifest_identity_hash,
            field_name="sample_manifest_identity_hash",
            error_type=TTAContractValidationError,
        )
        if self.aggregation_scope != "common_grid_binary_foreground_probability":
            raise TTAContractValidationError(
                "aggregation_scope must be 'common_grid_binary_foreground_probability'."
            )
        if self.sample_count != len(self.sample_records):
            raise TTAContractValidationError("sample_count must equal len(sample_records).")
        if self.sample_count <= 0:
            raise TTAContractValidationError("sample_count must be positive.")
        expected_indices = tuple(range(self.sample_count))
        actual_indices = tuple(record.sample_index for record in self.sample_records)
        if actual_indices != expected_indices:
            raise TTAContractValidationError(
                "sample_records must be zero-based, contiguous, and deterministically ordered."
            )
        sample_ids = tuple(record.sample_id for record in self.sample_records)
        if len(set(sample_ids)) != len(sample_ids):
            raise TTAContractValidationError("sample_records must not duplicate sample_id values.")
        probability_hashes = tuple(
            record.probability_map_identity_hash for record in self.sample_records
        )
        if len(set(probability_hashes)) != len(probability_hashes):
            raise TTAContractValidationError(
                "sample_records must not duplicate probability_map_identity_hash values."
            )
        grid_hashes = {record.common_grid_identity_hash for record in self.sample_records}
        if len(grid_hashes) != 1:
            raise TTAContractValidationError(
                "all TTA samples must be restored to the same common grid before aggregation."
            )
        if self.sample_manifest_identity_hash != hash_tta_sample_manifest(self):
            raise UncertaintyIdentityMismatchError(
                "sample_manifest_identity_hash does not match deterministic manifest metadata."
            )


@dataclass(frozen=True, slots=True)
class TTAProbabilitySamples:
    """Aligned probability maps eligible for future deterministic TTA variance."""

    schema_name: str
    schema_version: str
    tta_probability_samples_identity_hash: str
    sample_manifest: TTASampleManifest
    probability_maps: tuple[BinaryPredictiveProbabilityMap, ...]
    sample_count: int
    variance_population_policy: str

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            expected_name=PHASE7_TTA_PROBABILITY_SAMPLES_SCHEMA_NAME,
            expected_version=PHASE7_TTA_PROBABILITY_SAMPLES_SCHEMA_VERSION,
            error_type=TTAContractValidationError,
        )
        _require_sha256(
            self.tta_probability_samples_identity_hash,
            field_name="tta_probability_samples_identity_hash",
            error_type=TTAContractValidationError,
        )
        if self.sample_count != len(self.probability_maps):
            raise TTAContractValidationError("sample_count must equal len(probability_maps).")
        if self.sample_count != self.sample_manifest.sample_count:
            raise TTAContractValidationError(
                "sample_count must match sample_manifest.sample_count."
            )
        if self.sample_count < 2:
            raise TTAContractValidationError("TTA variance requires at least two samples.")
        if self.variance_population_policy != "population_variance":
            raise TTAContractValidationError(
                "variance_population_policy must be 'population_variance'."
            )
        manifest_hashes = tuple(
            record.probability_map_identity_hash for record in self.sample_manifest.sample_records
        )
        map_hashes = tuple(
            probability_map.probability_map_identity_hash
            for probability_map in self.probability_maps
        )
        if map_hashes != manifest_hashes:
            raise TTAContractValidationError(
                "probability_maps must follow the manifest order exactly."
            )
        shapes = {probability_map.probability_shape for probability_map in self.probability_maps}
        if len(shapes) != 1:
            raise TTAContractValidationError(
                "all TTA probability samples must share one [1,1,D,H,W] shape."
            )
        grid_hashes = {
            probability_map.common_grid_identity_hash for probability_map in self.probability_maps
        }
        if len(grid_hashes) != 1 or None in grid_hashes:
            raise TTAContractValidationError(
                "all TTA probability samples must carry the same common_grid_identity_hash."
            )
        if self.tta_probability_samples_identity_hash != hash_tta_probability_samples(self):
            raise UncertaintyIdentityMismatchError(
                "tta_probability_samples_identity_hash does not match deterministic samples."
            )


@dataclass(frozen=True, slots=True)
class UncertaintySummaryInput:
    """Validated input metadata for future uncertainty summaries."""

    schema_name: str
    schema_version: str
    uncertainty_summary_input_hash: str
    uncertainty_source_type: str
    uncertainty_map_content_hash: str
    uncertainty_shape: tuple[int, int, int, int, int]
    probability_map_identity_hash: str | None
    sample_manifest_identity_hash: str | None
    valid_voxel_count: int

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            expected_name=PHASE7_UNCERTAINTY_SUMMARY_INPUT_SCHEMA_NAME,
            expected_version=PHASE7_UNCERTAINTY_SUMMARY_INPUT_SCHEMA_VERSION,
            error_type=UncertaintyEligibilityValidationError,
        )
        _require_sha256(
            self.uncertainty_summary_input_hash,
            field_name="uncertainty_summary_input_hash",
            error_type=UncertaintyEligibilityValidationError,
        )
        if self.uncertainty_source_type not in SUPPORTED_UNCERTAINTY_SOURCE_TYPES:
            raise UncertaintyEligibilityValidationError(
                "uncertainty_source_type must be one of "
                f"{sorted(SUPPORTED_UNCERTAINTY_SOURCE_TYPES)!r}."
            )
        _require_sha256(
            self.uncertainty_map_content_hash,
            field_name="uncertainty_map_content_hash",
            error_type=UncertaintyEligibilityValidationError,
        )
        _require_volume_shape_tuple(self.uncertainty_shape, field_name="uncertainty_shape")
        if self.probability_map_identity_hash is not None:
            _require_sha256(
                self.probability_map_identity_hash,
                field_name="probability_map_identity_hash",
                error_type=UncertaintyEligibilityValidationError,
            )
        if self.sample_manifest_identity_hash is not None:
            _require_sha256(
                self.sample_manifest_identity_hash,
                field_name="sample_manifest_identity_hash",
                error_type=UncertaintyEligibilityValidationError,
            )
        if (
            self.uncertainty_source_type == "predictive_entropy"
            and self.probability_map_identity_hash is None
        ):
            raise UncertaintyEligibilityValidationError(
                "predictive_entropy summaries require probability_map_identity_hash."
            )
        if self.uncertainty_source_type in {"tta_variance", "ensemble_variance"} and (
            self.sample_manifest_identity_hash is None
        ):
            raise UncertaintyEligibilityValidationError(
                "sample-variance summaries require sample_manifest_identity_hash."
            )
        _require_nonnegative_int(
            self.valid_voxel_count,
            field_name="valid_voxel_count",
            error_type=UncertaintyEligibilityValidationError,
        )
        if self.uncertainty_summary_input_hash != hash_uncertainty_summary_input(self):
            raise UncertaintyIdentityMismatchError(
                "uncertainty_summary_input_hash does not match deterministic summary input."
            )


@dataclass(frozen=True, slots=True)
class UncertaintyEvaluationEligibilityRecord:
    """Eligibility metadata for downstream uncertainty/evaluation calculations."""

    schema_name: str
    schema_version: str
    eligibility_record_hash: str
    evaluation_name: str
    status: EligibilityStatus
    valid_voxel_count: int
    sample_count: int | None
    reason_code: str | None
    reason_message: str | None

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name,
            self.schema_version,
            expected_name=PHASE7_EVALUATION_ELIGIBILITY_SCHEMA_NAME,
            expected_version=PHASE7_EVALUATION_ELIGIBILITY_SCHEMA_VERSION,
            error_type=UncertaintyEligibilityValidationError,
        )
        _require_sha256(
            self.eligibility_record_hash,
            field_name="eligibility_record_hash",
            error_type=UncertaintyEligibilityValidationError,
        )
        if self.evaluation_name not in SUPPORTED_EVALUATION_NAMES:
            raise UncertaintyEligibilityValidationError(
                f"evaluation_name must be one of {sorted(SUPPORTED_EVALUATION_NAMES)!r}."
            )
        if self.status not in SUPPORTED_ELIGIBILITY_STATUSES:
            raise UncertaintyEligibilityValidationError(
                f"status must be one of {sorted(SUPPORTED_ELIGIBILITY_STATUSES)!r}."
            )
        _require_nonnegative_int(
            self.valid_voxel_count,
            field_name="valid_voxel_count",
            error_type=UncertaintyEligibilityValidationError,
        )
        if self.sample_count is not None:
            _require_nonnegative_int(
                self.sample_count,
                field_name="sample_count",
                error_type=UncertaintyEligibilityValidationError,
            )
        if self.status == "eligible":
            if self.reason_code is not None or self.reason_message is not None:
                raise UncertaintyEligibilityValidationError(
                    "eligible records must not carry reason_code or reason_message."
                )
            if self.valid_voxel_count <= 0:
                raise UncertaintyEligibilityValidationError(
                    "eligible records require positive valid_voxel_count."
                )
            if self.evaluation_name == "tta_variance" and (
                self.sample_count is None or self.sample_count < 2
            ):
                raise UncertaintyEligibilityValidationError(
                    "eligible TTA variance requires sample_count >= 2."
                )
        else:
            if not self.reason_code or not self.reason_message:
                raise UncertaintyEligibilityValidationError(
                    "unavailable records require reason_code and reason_message."
                )
            _require_safe_identifier(
                self.reason_code,
                field_name="reason_code",
                error_type=UncertaintyEligibilityValidationError,
            )
        if self.eligibility_record_hash != hash_uncertainty_evaluation_eligibility_record(self):
            raise UncertaintyIdentityMismatchError(
                "eligibility_record_hash does not match deterministic eligibility metadata."
            )


def build_binary_predictive_probability_map(
    *,
    foreground_probability_map: np.ndarray,
    background_probability_map: np.ndarray,
    source_prediction_identity_hash: str | None = None,
    common_grid_identity_hash: str | None = None,
    probability_sum_tolerance: float = _PROBABILITY_SUM_TOLERANCE,
) -> BinaryPredictiveProbabilityMap:
    """Build one deterministic binary predictive-probability contract."""

    foreground = np.ascontiguousarray(foreground_probability_map.astype(_FLOAT_DTYPE, copy=False))
    background = np.ascontiguousarray(background_probability_map.astype(_FLOAT_DTYPE, copy=False))
    foreground_hash = array_content_sha256(foreground)
    background_hash = array_content_sha256(background)
    probability_shape = cast(tuple[int, int, int, int, int], tuple(foreground.shape))
    identity_hash = sha256_json(
        binary_predictive_probability_map_identity_payload_from_parts(
            foreground_probability_content_hash=foreground_hash,
            background_probability_content_hash=background_hash,
            probability_shape=probability_shape,
            probability_sum_tolerance=probability_sum_tolerance,
            source_prediction_identity_hash=source_prediction_identity_hash,
            common_grid_identity_hash=common_grid_identity_hash,
        )
    )
    return BinaryPredictiveProbabilityMap(
        schema_name=PHASE7_BINARY_PROBABILITY_MAP_SCHEMA_NAME,
        schema_version=PHASE7_BINARY_PROBABILITY_MAP_SCHEMA_VERSION,
        probability_map_identity_hash=identity_hash,
        foreground_probability_map=foreground,
        background_probability_map=background,
        foreground_probability_content_hash=foreground_hash,
        background_probability_content_hash=background_hash,
        probability_shape=probability_shape,
        probability_sum_tolerance=probability_sum_tolerance,
        source_prediction_identity_hash=source_prediction_identity_hash,
        common_grid_identity_hash=common_grid_identity_hash,
    )


def build_predictive_entropy_input(
    probability_map: BinaryPredictiveProbabilityMap,
    *,
    log_base: str = "natural",
) -> PredictiveEntropyInput:
    """Build entropy-input metadata without computing entropy."""

    payload = predictive_entropy_input_identity_payload_from_parts(
        probability_map_identity_hash=probability_map.probability_map_identity_hash,
        probability_shape=probability_map.probability_shape,
        log_base=log_base,
        zero_probability_policy="zero_log_zero_is_zero",
    )
    return PredictiveEntropyInput(
        schema_name=PHASE7_ENTROPY_INPUT_SCHEMA_NAME,
        schema_version=PHASE7_ENTROPY_INPUT_SCHEMA_VERSION,
        entropy_input_identity_hash=sha256_json(payload),
        probability_map_identity_hash=probability_map.probability_map_identity_hash,
        probability_shape=probability_map.probability_shape,
        log_base=log_base,
        zero_probability_policy="zero_log_zero_is_zero",
    )


def build_predictive_entropy_result_metadata(
    *,
    entropy_input: PredictiveEntropyInput,
    entropy_map_content_hash: str,
    valid_voxel_count: int,
) -> PredictiveEntropyResultMetadata:
    """Build entropy-result metadata without computing or carrying an entropy map."""

    payload = predictive_entropy_result_metadata_identity_payload_from_parts(
        entropy_input_identity_hash=entropy_input.entropy_input_identity_hash,
        entropy_map_content_hash=entropy_map_content_hash,
        entropy_shape=entropy_input.probability_shape,
        log_base=entropy_input.log_base,
        valid_voxel_count=valid_voxel_count,
    )
    return PredictiveEntropyResultMetadata(
        schema_name=PHASE7_ENTROPY_RESULT_METADATA_SCHEMA_NAME,
        schema_version=PHASE7_ENTROPY_RESULT_METADATA_SCHEMA_VERSION,
        entropy_result_metadata_hash=sha256_json(payload),
        entropy_input_identity_hash=entropy_input.entropy_input_identity_hash,
        entropy_map_content_hash=entropy_map_content_hash,
        entropy_shape=entropy_input.probability_shape,
        log_base=entropy_input.log_base,
        valid_voxel_count=valid_voxel_count,
    )


def build_tta_sample_record(
    *,
    sample_index: int,
    sample_id: str,
    probability_map_identity_hash: str,
    common_grid_identity_hash: str,
    transform_manifest_hash: str | None,
) -> TTASampleRecord:
    """Build one deterministic TTA sample record."""

    payload = tta_sample_record_identity_payload_from_parts(
        sample_index=sample_index,
        sample_id=sample_id,
        probability_map_identity_hash=probability_map_identity_hash,
        common_grid_identity_hash=common_grid_identity_hash,
        transform_manifest_hash=transform_manifest_hash,
    )
    return TTASampleRecord(
        schema_name=PHASE7_TTA_SAMPLE_RECORD_SCHEMA_NAME,
        schema_version=PHASE7_TTA_SAMPLE_RECORD_SCHEMA_VERSION,
        sample_identity_hash=sha256_json(payload),
        sample_index=sample_index,
        sample_id=sample_id,
        probability_map_identity_hash=probability_map_identity_hash,
        transform_manifest_hash=transform_manifest_hash,
        common_grid_identity_hash=common_grid_identity_hash,
    )


def build_tta_sample_manifest(
    sample_records: tuple[TTASampleRecord, ...],
) -> TTASampleManifest:
    """Build one deterministic TTA sample manifest."""

    payload = tta_sample_manifest_identity_payload_from_parts(
        sample_identity_hashes=tuple(record.sample_identity_hash for record in sample_records),
        sample_count=len(sample_records),
        aggregation_scope="common_grid_binary_foreground_probability",
    )
    return TTASampleManifest(
        schema_name=PHASE7_TTA_SAMPLE_MANIFEST_SCHEMA_NAME,
        schema_version=PHASE7_TTA_SAMPLE_MANIFEST_SCHEMA_VERSION,
        sample_manifest_identity_hash=sha256_json(payload),
        sample_records=sample_records,
        sample_count=len(sample_records),
        aggregation_scope="common_grid_binary_foreground_probability",
    )


def build_tta_probability_samples(
    *,
    sample_manifest: TTASampleManifest,
    probability_maps: tuple[BinaryPredictiveProbabilityMap, ...],
) -> TTAProbabilitySamples:
    """Build aligned probability-sample contracts for future population variance."""

    payload = tta_probability_samples_identity_payload_from_parts(
        sample_manifest_identity_hash=sample_manifest.sample_manifest_identity_hash,
        probability_map_identity_hashes=tuple(
            probability_map.probability_map_identity_hash for probability_map in probability_maps
        ),
        sample_count=len(probability_maps),
        variance_population_policy="population_variance",
    )
    return TTAProbabilitySamples(
        schema_name=PHASE7_TTA_PROBABILITY_SAMPLES_SCHEMA_NAME,
        schema_version=PHASE7_TTA_PROBABILITY_SAMPLES_SCHEMA_VERSION,
        tta_probability_samples_identity_hash=sha256_json(payload),
        sample_manifest=sample_manifest,
        probability_maps=probability_maps,
        sample_count=len(probability_maps),
        variance_population_policy="population_variance",
    )


def build_uncertainty_summary_input(
    *,
    uncertainty_source_type: str,
    uncertainty_map_content_hash: str,
    uncertainty_shape: tuple[int, int, int, int, int],
    valid_voxel_count: int,
    probability_map_identity_hash: str | None = None,
    sample_manifest_identity_hash: str | None = None,
) -> UncertaintySummaryInput:
    """Build future uncertainty-summary input metadata without computing summaries."""

    payload = uncertainty_summary_input_identity_payload_from_parts(
        uncertainty_source_type=uncertainty_source_type,
        uncertainty_map_content_hash=uncertainty_map_content_hash,
        uncertainty_shape=uncertainty_shape,
        probability_map_identity_hash=probability_map_identity_hash,
        sample_manifest_identity_hash=sample_manifest_identity_hash,
        valid_voxel_count=valid_voxel_count,
    )
    return UncertaintySummaryInput(
        schema_name=PHASE7_UNCERTAINTY_SUMMARY_INPUT_SCHEMA_NAME,
        schema_version=PHASE7_UNCERTAINTY_SUMMARY_INPUT_SCHEMA_VERSION,
        uncertainty_summary_input_hash=sha256_json(payload),
        uncertainty_source_type=uncertainty_source_type,
        uncertainty_map_content_hash=uncertainty_map_content_hash,
        uncertainty_shape=uncertainty_shape,
        probability_map_identity_hash=probability_map_identity_hash,
        sample_manifest_identity_hash=sample_manifest_identity_hash,
        valid_voxel_count=valid_voxel_count,
    )


def build_uncertainty_evaluation_eligibility_record(
    *,
    evaluation_name: str,
    status: EligibilityStatus,
    valid_voxel_count: int,
    sample_count: int | None = None,
    reason_code: str | None = None,
    reason_message: str | None = None,
) -> UncertaintyEvaluationEligibilityRecord:
    """Build one deterministic downstream evaluation eligibility record."""

    payload = uncertainty_evaluation_eligibility_identity_payload_from_parts(
        evaluation_name=evaluation_name,
        status=status,
        valid_voxel_count=valid_voxel_count,
        sample_count=sample_count,
        reason_code=reason_code,
        reason_message=reason_message,
    )
    return UncertaintyEvaluationEligibilityRecord(
        schema_name=PHASE7_EVALUATION_ELIGIBILITY_SCHEMA_NAME,
        schema_version=PHASE7_EVALUATION_ELIGIBILITY_SCHEMA_VERSION,
        eligibility_record_hash=sha256_json(payload),
        evaluation_name=evaluation_name,
        status=status,
        valid_voxel_count=valid_voxel_count,
        sample_count=sample_count,
        reason_code=reason_code,
        reason_message=reason_message,
    )


def binary_predictive_probability_map_identity_payload(
    probability_map: BinaryPredictiveProbabilityMap,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one binary probability map."""

    return binary_predictive_probability_map_identity_payload_from_parts(
        foreground_probability_content_hash=probability_map.foreground_probability_content_hash,
        background_probability_content_hash=probability_map.background_probability_content_hash,
        probability_shape=probability_map.probability_shape,
        probability_sum_tolerance=probability_map.probability_sum_tolerance,
        source_prediction_identity_hash=probability_map.source_prediction_identity_hash,
        common_grid_identity_hash=probability_map.common_grid_identity_hash,
    )


def binary_predictive_probability_map_identity_payload_from_parts(
    *,
    foreground_probability_content_hash: str,
    background_probability_content_hash: str,
    probability_shape: tuple[int, int, int, int, int],
    probability_sum_tolerance: float,
    source_prediction_identity_hash: str | None,
    common_grid_identity_hash: str | None,
) -> dict[str, JsonValue]:
    """Return the exact identity payload from probability-map fields."""

    return {
        "schema_name": PHASE7_BINARY_PROBABILITY_MAP_SCHEMA_NAME,
        "schema_version": PHASE7_BINARY_PROBABILITY_MAP_SCHEMA_VERSION,
        "foreground_probability_content_hash": foreground_probability_content_hash,
        "background_probability_content_hash": background_probability_content_hash,
        "probability_shape": list(probability_shape),
        "probability_sum_tolerance": float(probability_sum_tolerance),
        "source_prediction_identity_hash": source_prediction_identity_hash,
        "common_grid_identity_hash": common_grid_identity_hash,
    }


def predictive_entropy_input_identity_payload(
    entropy_input: PredictiveEntropyInput,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for entropy input metadata."""

    return predictive_entropy_input_identity_payload_from_parts(
        probability_map_identity_hash=entropy_input.probability_map_identity_hash,
        probability_shape=entropy_input.probability_shape,
        log_base=entropy_input.log_base,
        zero_probability_policy=entropy_input.zero_probability_policy,
    )


def predictive_entropy_input_identity_payload_from_parts(
    *,
    probability_map_identity_hash: str,
    probability_shape: tuple[int, int, int, int, int],
    log_base: str,
    zero_probability_policy: str,
) -> dict[str, JsonValue]:
    """Return the exact identity payload from entropy-input fields."""

    return {
        "schema_name": PHASE7_ENTROPY_INPUT_SCHEMA_NAME,
        "schema_version": PHASE7_ENTROPY_INPUT_SCHEMA_VERSION,
        "probability_map_identity_hash": probability_map_identity_hash,
        "probability_shape": list(probability_shape),
        "log_base": log_base,
        "zero_probability_policy": zero_probability_policy,
    }


def predictive_entropy_result_metadata_identity_payload(
    metadata: PredictiveEntropyResultMetadata,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for entropy-result metadata."""

    return predictive_entropy_result_metadata_identity_payload_from_parts(
        entropy_input_identity_hash=metadata.entropy_input_identity_hash,
        entropy_map_content_hash=metadata.entropy_map_content_hash,
        entropy_shape=metadata.entropy_shape,
        log_base=metadata.log_base,
        valid_voxel_count=metadata.valid_voxel_count,
    )


def predictive_entropy_result_metadata_identity_payload_from_parts(
    *,
    entropy_input_identity_hash: str,
    entropy_map_content_hash: str,
    entropy_shape: tuple[int, int, int, int, int],
    log_base: str,
    valid_voxel_count: int,
) -> dict[str, JsonValue]:
    """Return the exact identity payload from entropy-result metadata fields."""

    return {
        "schema_name": PHASE7_ENTROPY_RESULT_METADATA_SCHEMA_NAME,
        "schema_version": PHASE7_ENTROPY_RESULT_METADATA_SCHEMA_VERSION,
        "entropy_input_identity_hash": entropy_input_identity_hash,
        "entropy_map_content_hash": entropy_map_content_hash,
        "entropy_shape": list(entropy_shape),
        "log_base": log_base,
        "valid_voxel_count": valid_voxel_count,
    }


def tta_sample_record_identity_payload(record: TTASampleRecord) -> dict[str, JsonValue]:
    """Return the exact identity payload for one TTA sample record."""

    return tta_sample_record_identity_payload_from_parts(
        sample_index=record.sample_index,
        sample_id=record.sample_id,
        probability_map_identity_hash=record.probability_map_identity_hash,
        common_grid_identity_hash=record.common_grid_identity_hash,
        transform_manifest_hash=record.transform_manifest_hash,
    )


def tta_sample_record_identity_payload_from_parts(
    *,
    sample_index: int,
    sample_id: str,
    probability_map_identity_hash: str,
    common_grid_identity_hash: str,
    transform_manifest_hash: str | None,
) -> dict[str, JsonValue]:
    """Return the exact identity payload from TTA sample record fields."""

    return {
        "schema_name": PHASE7_TTA_SAMPLE_RECORD_SCHEMA_NAME,
        "schema_version": PHASE7_TTA_SAMPLE_RECORD_SCHEMA_VERSION,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "probability_map_identity_hash": probability_map_identity_hash,
        "common_grid_identity_hash": common_grid_identity_hash,
        "transform_manifest_hash": transform_manifest_hash,
    }


def tta_sample_manifest_identity_payload(
    manifest: TTASampleManifest,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one TTA sample manifest."""

    return tta_sample_manifest_identity_payload_from_parts(
        sample_identity_hashes=tuple(
            record.sample_identity_hash for record in manifest.sample_records
        ),
        sample_count=manifest.sample_count,
        aggregation_scope=manifest.aggregation_scope,
    )


def tta_sample_manifest_identity_payload_from_parts(
    *,
    sample_identity_hashes: tuple[str, ...],
    sample_count: int,
    aggregation_scope: str,
) -> dict[str, JsonValue]:
    """Return the exact identity payload from TTA sample manifest fields."""

    return {
        "schema_name": PHASE7_TTA_SAMPLE_MANIFEST_SCHEMA_NAME,
        "schema_version": PHASE7_TTA_SAMPLE_MANIFEST_SCHEMA_VERSION,
        "sample_identity_hashes": list(sample_identity_hashes),
        "sample_count": sample_count,
        "aggregation_scope": aggregation_scope,
    }


def tta_probability_samples_identity_payload(
    samples: TTAProbabilitySamples,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one TTA probability-sample set."""

    return tta_probability_samples_identity_payload_from_parts(
        sample_manifest_identity_hash=samples.sample_manifest.sample_manifest_identity_hash,
        probability_map_identity_hashes=tuple(
            probability_map.probability_map_identity_hash
            for probability_map in samples.probability_maps
        ),
        sample_count=samples.sample_count,
        variance_population_policy=samples.variance_population_policy,
    )


def tta_probability_samples_identity_payload_from_parts(
    *,
    sample_manifest_identity_hash: str,
    probability_map_identity_hashes: tuple[str, ...],
    sample_count: int,
    variance_population_policy: str,
) -> dict[str, JsonValue]:
    """Return the exact identity payload from TTA probability-sample fields."""

    return {
        "schema_name": PHASE7_TTA_PROBABILITY_SAMPLES_SCHEMA_NAME,
        "schema_version": PHASE7_TTA_PROBABILITY_SAMPLES_SCHEMA_VERSION,
        "sample_manifest_identity_hash": sample_manifest_identity_hash,
        "probability_map_identity_hashes": list(probability_map_identity_hashes),
        "sample_count": sample_count,
        "variance_population_policy": variance_population_policy,
    }


def uncertainty_summary_input_identity_payload(
    summary_input: UncertaintySummaryInput,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one uncertainty-summary input."""

    return uncertainty_summary_input_identity_payload_from_parts(
        uncertainty_source_type=summary_input.uncertainty_source_type,
        uncertainty_map_content_hash=summary_input.uncertainty_map_content_hash,
        uncertainty_shape=summary_input.uncertainty_shape,
        probability_map_identity_hash=summary_input.probability_map_identity_hash,
        sample_manifest_identity_hash=summary_input.sample_manifest_identity_hash,
        valid_voxel_count=summary_input.valid_voxel_count,
    )


def uncertainty_summary_input_identity_payload_from_parts(
    *,
    uncertainty_source_type: str,
    uncertainty_map_content_hash: str,
    uncertainty_shape: tuple[int, int, int, int, int],
    probability_map_identity_hash: str | None,
    sample_manifest_identity_hash: str | None,
    valid_voxel_count: int,
) -> dict[str, JsonValue]:
    """Return the exact identity payload from uncertainty-summary fields."""

    return {
        "schema_name": PHASE7_UNCERTAINTY_SUMMARY_INPUT_SCHEMA_NAME,
        "schema_version": PHASE7_UNCERTAINTY_SUMMARY_INPUT_SCHEMA_VERSION,
        "uncertainty_source_type": uncertainty_source_type,
        "uncertainty_map_content_hash": uncertainty_map_content_hash,
        "uncertainty_shape": list(uncertainty_shape),
        "probability_map_identity_hash": probability_map_identity_hash,
        "sample_manifest_identity_hash": sample_manifest_identity_hash,
        "valid_voxel_count": valid_voxel_count,
    }


def uncertainty_evaluation_eligibility_identity_payload(
    record: UncertaintyEvaluationEligibilityRecord,
) -> dict[str, JsonValue]:
    """Return the exact identity payload for one eligibility record."""

    return uncertainty_evaluation_eligibility_identity_payload_from_parts(
        evaluation_name=record.evaluation_name,
        status=record.status,
        valid_voxel_count=record.valid_voxel_count,
        sample_count=record.sample_count,
        reason_code=record.reason_code,
        reason_message=record.reason_message,
    )


def uncertainty_evaluation_eligibility_identity_payload_from_parts(
    *,
    evaluation_name: str,
    status: EligibilityStatus,
    valid_voxel_count: int,
    sample_count: int | None,
    reason_code: str | None,
    reason_message: str | None,
) -> dict[str, JsonValue]:
    """Return the exact identity payload from eligibility fields."""

    return {
        "schema_name": PHASE7_EVALUATION_ELIGIBILITY_SCHEMA_NAME,
        "schema_version": PHASE7_EVALUATION_ELIGIBILITY_SCHEMA_VERSION,
        "evaluation_name": evaluation_name,
        "status": status,
        "valid_voxel_count": valid_voxel_count,
        "sample_count": sample_count,
        "reason_code": reason_code,
        "reason_message": reason_message,
    }


def hash_binary_predictive_probability_map(
    probability_map: BinaryPredictiveProbabilityMap,
) -> str:
    """Return the deterministic probability-map identity hash."""

    return sha256_json(binary_predictive_probability_map_identity_payload(probability_map))


def hash_predictive_entropy_input(entropy_input: PredictiveEntropyInput) -> str:
    """Return the deterministic entropy-input identity hash."""

    return sha256_json(predictive_entropy_input_identity_payload(entropy_input))


def hash_predictive_entropy_result_metadata(metadata: PredictiveEntropyResultMetadata) -> str:
    """Return the deterministic entropy-result metadata hash."""

    return sha256_json(predictive_entropy_result_metadata_identity_payload(metadata))


def hash_tta_sample_record(record: TTASampleRecord) -> str:
    """Return the deterministic TTA sample-record hash."""

    return sha256_json(tta_sample_record_identity_payload(record))


def hash_tta_sample_manifest(manifest: TTASampleManifest) -> str:
    """Return the deterministic TTA sample-manifest hash."""

    return sha256_json(tta_sample_manifest_identity_payload(manifest))


def hash_tta_probability_samples(samples: TTAProbabilitySamples) -> str:
    """Return the deterministic TTA probability-samples hash."""

    return sha256_json(tta_probability_samples_identity_payload(samples))


def hash_uncertainty_summary_input(summary_input: UncertaintySummaryInput) -> str:
    """Return the deterministic uncertainty-summary input hash."""

    return sha256_json(uncertainty_summary_input_identity_payload(summary_input))


def hash_uncertainty_evaluation_eligibility_record(
    record: UncertaintyEvaluationEligibilityRecord,
) -> str:
    """Return the deterministic eligibility-record hash."""

    return sha256_json(uncertainty_evaluation_eligibility_identity_payload(record))


def array_content_sha256(array: np.ndarray) -> str:
    """Return a deterministic content hash independent of input memory layout."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def query_label_not_part_of_uncertainty_contracts_api() -> bool:
    """Return True because prediction/uncertainty contracts accept no query labels or masks."""

    return True


def _require_probability_volume(array: np.ndarray, *, field_name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 1:
        raise ProbabilityMapValidationError(f"{field_name} must have shape [1,1,D,H,W].")
    if value.shape[2] <= 0 or value.shape[3] <= 0 or value.shape[4] <= 0:
        raise ProbabilityMapValidationError(f"{field_name} spatial dimensions must be positive.")
    if value.dtype.kind != "f":
        raise ProbabilityMapValidationError(f"{field_name} must be a floating-point array.")
    contiguous = np.ascontiguousarray(value.astype(_FLOAT_DTYPE, copy=False))
    if not np.all(np.isfinite(contiguous)):
        raise ProbabilityMapValidationError(f"{field_name} must contain only finite values.")
    if np.any(contiguous < 0.0) or np.any(contiguous > 1.0):
        raise ProbabilityMapValidationError(f"{field_name} probabilities must lie in [0,1].")
    return contiguous


def _require_volume_shape_tuple(
    shape: tuple[int, int, int, int, int],
    *,
    field_name: str,
) -> None:
    if not isinstance(shape, tuple) or len(shape) != 5:
        raise ProbabilityMapValidationError(f"{field_name} must be a 5-item tuple.")
    if shape[0] != 1 or shape[1] != 1 or shape[2] <= 0 or shape[3] <= 0 or shape[4] <= 0:
        raise ProbabilityMapValidationError(f"{field_name} must describe [1,1,D,H,W].")


def _require_schema(
    schema_name: str,
    schema_version: str,
    *,
    expected_name: str,
    expected_version: str,
    error_type: type[Phase7UncertaintyContractError],
) -> None:
    if schema_name != expected_name:
        raise error_type(f"schema_name must be {expected_name!r}.")
    if schema_version != expected_version:
        raise error_type(f"schema_version must be {expected_version!r}.")


def _require_sha256(
    value: str,
    *,
    field_name: str,
    error_type: type[Phase7UncertaintyContractError],
) -> None:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise error_type(f"{field_name} must be a lowercase SHA-256 hex digest.")


def _require_safe_identifier(
    value: str,
    *,
    field_name: str,
    error_type: type[Phase7UncertaintyContractError],
) -> None:
    if not _SAFE_IDENTIFIER_RE.fullmatch(value):
        raise error_type(f"{field_name} must be a conservative identifier.")


def _require_nonnegative_int(
    value: int,
    *,
    field_name: str,
    error_type: type[Phase7UncertaintyContractError],
) -> None:
    if not isinstance(value, int) or value < 0:
        raise error_type(f"{field_name} must be a nonnegative integer.")
