"""Deterministic Gaussian-noise robustness transform for Phase 7."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.robustness.artifacts import (
    PHASE7_TRANSFORM_RESULT_SCHEMA_NAME,
    PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION,
    Phase7CorruptionSpecification,
    Phase7TransformResult,
)
from protoem_ct.robustness.geometry import BinaryMaskValidationError, validate_binary_mask

FloatArray: TypeAlias = npt.NDArray[np.float64]
MaskArray: TypeAlias = npt.NDArray[np.generic]

GAUSSIAN_NOISE_CORRUPTION_NAME: Final[str] = "gaussian_noise"
GAUSSIAN_NOISE_SEVERITY_PARAMETERS: Final[dict[str, dict[str, float]]] = {
    "none": {"sigma_hu": 0.0},
    "low": {"sigma_hu": 5.0},
    "medium": {"sigma_hu": 15.0},
    "high": {"sigma_hu": 30.0},
}
_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)


class Phase7TransformExecutionError(ValueError):
    """Base error for deterministic Phase 7 transform execution."""


class InvalidTransformInputError(Phase7TransformExecutionError):
    """Raised when a CT image or optional mask is malformed."""


class InvalidCorruptionSpecificationError(Phase7TransformExecutionError):
    """Raised when a corruption specification is incompatible with a transform."""


class NonFiniteTransformOutputError(Phase7TransformExecutionError):
    """Raised when a transform would produce NaN or Infinity."""


@dataclass(frozen=True, slots=True)
class Phase7ImageTransformOutput:
    """In-memory output plus immutable metadata for one image-only Phase 7 transform."""

    transformed_image: FloatArray
    transformed_mask: MaskArray | None
    transform_result: Phase7TransformResult


def gaussian_noise_parameters_for_severity(severity: str) -> dict[str, JsonValue]:
    """Return the exact persisted Gaussian-noise parameters for one severity level."""

    if severity not in GAUSSIAN_NOISE_SEVERITY_PARAMETERS:
        raise InvalidCorruptionSpecificationError(
            f"gaussian_noise severity must be one of "
            f"{sorted(GAUSSIAN_NOISE_SEVERITY_PARAMETERS)!r}."
        )
    return dict(GAUSSIAN_NOISE_SEVERITY_PARAMETERS[severity])


def apply_gaussian_noise(
    image: npt.ArrayLike,
    specification: Phase7CorruptionSpecification,
    *,
    mask: npt.ArrayLike | None = None,
) -> Phase7ImageTransformOutput:
    """Apply deterministic local-RNG Gaussian noise to a finite 3D CT image only.

    The noise sample is generated with ``numpy.random.default_rng`` from the
    specification's explicit deterministic seed. The NumPy global RNG state is
    not read or mutated.
    """

    _require_corruption_specification(
        specification,
        corruption_name=GAUSSIAN_NOISE_CORRUPTION_NAME,
        expected_parameters=gaussian_noise_parameters_for_severity(specification.severity),
        seed_required=True,
    )
    image_array = _require_ct_image(image)
    mask_array = _copy_validated_mask(mask, expected_shape=image_array.shape)
    input_image_hash = _array_content_sha256(image_array)
    input_mask_hash = _mask_content_sha256(mask_array) if mask_array is not None else None

    sigma_hu = _finite_nonnegative_parameter(
        specification.parameters["sigma_hu"], field_name="sigma_hu"
    )
    if sigma_hu == 0.0:
        transformed = np.ascontiguousarray(image_array.copy())
    else:
        seed = _require_seed(specification.deterministic_seed)
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=sigma_hu, size=image_array.shape)
        transformed = np.ascontiguousarray(image_array + noise.astype(_FLOAT_DTYPE, copy=False))
    if not bool(np.isfinite(transformed).all()):
        raise NonFiniteTransformOutputError("gaussian_noise produced non-finite output.")

    output_mask_hash = _mask_content_sha256(mask_array) if mask_array is not None else None
    return Phase7ImageTransformOutput(
        transformed_image=transformed,
        transformed_mask=None if mask_array is None else np.ascontiguousarray(mask_array.copy()),
        transform_result=_build_completed_transform_result(
            specification=specification,
            input_image_hash=input_image_hash,
            input_mask_hash=input_mask_hash,
            output_image_hash=_array_content_sha256(transformed),
            output_mask_hash=output_mask_hash,
            mask_present=mask_array is not None,
        ),
    )


def _require_corruption_specification(
    specification: Phase7CorruptionSpecification,
    *,
    corruption_name: str,
    expected_parameters: Mapping[str, JsonValue],
    seed_required: bool,
) -> None:
    if specification.corruption_name != corruption_name:
        raise InvalidCorruptionSpecificationError(
            f"expected {corruption_name!r} specification, got {specification.corruption_name!r}."
        )
    if specification.changes_geometry:
        raise InvalidCorruptionSpecificationError(f"{corruption_name} must not change geometry.")
    if specification.common_grid_restoration_required:
        raise InvalidCorruptionSpecificationError(
            f"{corruption_name} must not require common-grid restoration."
        )
    if seed_required:
        _require_seed(specification.deterministic_seed)
    elif specification.deterministic_seed is not None:
        raise InvalidCorruptionSpecificationError(
            f"{corruption_name} is deterministic without a seed; deterministic_seed must be null."
        )
    _require_exact_parameters(
        specification.parameters,
        expected_parameters=expected_parameters,
        corruption_name=corruption_name,
    )


def _require_ct_image(image: npt.ArrayLike) -> FloatArray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise InvalidTransformInputError("CT image must be a 3D array.")
    if np.iscomplexobj(array):
        raise InvalidTransformInputError("CT image must contain real values.")
    try:
        image64 = np.ascontiguousarray(array.astype(_FLOAT_DTYPE, copy=False))
    except (TypeError, ValueError) as exc:
        raise InvalidTransformInputError("CT image must be numeric.") from exc
    if not bool(np.isfinite(image64).all()):
        raise InvalidTransformInputError("CT image must contain only finite values.")
    return image64


def _copy_validated_mask(
    mask: npt.ArrayLike | None,
    *,
    expected_shape: tuple[int, int, int],
) -> MaskArray | None:
    if mask is None:
        return None
    if len(expected_shape) != 3:
        raise InvalidTransformInputError("expected mask shape must be 3D.")
    mask_array = np.asarray(mask)
    try:
        validate_binary_mask(mask_array, expected_shape=expected_shape)
    except BinaryMaskValidationError as exc:
        raise InvalidTransformInputError(
            "mask must be finite 3D binary and aligned by shape."
        ) from exc
    return np.ascontiguousarray(mask_array.copy())


def _require_exact_parameters(
    parameters: Mapping[str, JsonValue],
    *,
    expected_parameters: Mapping[str, JsonValue],
    corruption_name: str,
) -> None:
    if set(parameters) != set(expected_parameters):
        raise InvalidCorruptionSpecificationError(
            f"{corruption_name} parameters must exactly equal {sorted(expected_parameters)!r}."
        )
    for key, expected_value in expected_parameters.items():
        actual_value = parameters[key]
        if isinstance(expected_value, float):
            actual = _finite_nonnegative_parameter(actual_value, field_name=key)
            if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=0.0):
                raise InvalidCorruptionSpecificationError(
                    f"{corruption_name} parameter {key!r} must equal {expected_value!r}."
                )
        elif actual_value != expected_value:
            raise InvalidCorruptionSpecificationError(
                f"{corruption_name} parameter {key!r} must equal {expected_value!r}."
            )


def _finite_nonnegative_parameter(value: JsonValue, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidCorruptionSpecificationError(f"{field_name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise InvalidCorruptionSpecificationError(f"{field_name} must be finite and nonnegative.")
    return numeric


def _require_seed(value: int | None) -> int:
    if value is None:
        raise InvalidCorruptionSpecificationError(
            "gaussian_noise requires an explicit deterministic_seed."
        )
    return value


def _build_completed_transform_result(
    *,
    specification: Phase7CorruptionSpecification,
    input_image_hash: str,
    input_mask_hash: str | None,
    output_image_hash: str,
    output_mask_hash: str | None,
    mask_present: bool,
) -> Phase7TransformResult:
    payload: dict[str, JsonValue] = {
        "schema_name": PHASE7_TRANSFORM_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION,
        "corruption_specification_hash": specification.corruption_specification_hash,
        "input_image_content_hash": input_image_hash,
        "input_mask_content_hash": input_mask_hash,
        "output_image_content_hash": output_image_hash,
        "output_mask_content_hash": output_mask_hash,
        "geometry_record_hash": None,
        "finite_output": True,
        "mask_binary_preserved": True if mask_present else None,
        "common_grid_restored": False,
        "execution_status": "completed",
        "failure_code": None,
        "failure_message": None,
    }
    return Phase7TransformResult(
        transform_result_hash=sha256_json(payload),
        schema_name=PHASE7_TRANSFORM_RESULT_SCHEMA_NAME,
        schema_version=PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION,
        corruption_specification_hash=specification.corruption_specification_hash,
        input_image_content_hash=input_image_hash,
        input_mask_content_hash=input_mask_hash,
        output_image_content_hash=output_image_hash,
        output_mask_content_hash=output_mask_hash,
        geometry_record_hash=None,
        finite_output=True,
        mask_binary_preserved=True if mask_present else None,
        common_grid_restored=False,
        execution_status="completed",
        failure_code=None,
        failure_message=None,
    )


def _array_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array.astype(_FLOAT_DTYPE, copy=False))
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(int(item) for item in contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _mask_content_sha256(mask: np.ndarray) -> str:
    canonical = np.ascontiguousarray(mask.astype(np.uint8, copy=False))
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode("utf-8"))
    digest.update(str(tuple(int(item) for item in canonical.shape)).encode("utf-8"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "GAUSSIAN_NOISE_CORRUPTION_NAME",
    "GAUSSIAN_NOISE_SEVERITY_PARAMETERS",
    "InvalidCorruptionSpecificationError",
    "InvalidTransformInputError",
    "NonFiniteTransformOutputError",
    "Phase7ImageTransformOutput",
    "Phase7TransformExecutionError",
    "apply_gaussian_noise",
    "gaussian_noise_parameters_for_severity",
]
