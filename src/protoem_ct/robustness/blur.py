"""Deterministic Gaussian-blur robustness transform for Phase 7."""

from __future__ import annotations

import math
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from protoem_ct.artifacts.hashing import JsonValue
from protoem_ct.robustness.artifacts import Phase7CorruptionSpecification
from protoem_ct.robustness.noise import (
    InvalidCorruptionSpecificationError,
    NonFiniteTransformOutputError,
    Phase7ImageTransformOutput,
    _array_content_sha256,
    _build_completed_transform_result,
    _copy_validated_mask,
    _mask_content_sha256,
    _require_corruption_specification,
    _require_ct_image,
)

GAUSSIAN_BLUR_CORRUPTION_NAME: Final[str] = "gaussian_blur"
GAUSSIAN_BLUR_SEVERITY_PARAMETERS: Final[dict[str, dict[str, JsonValue]]] = {
    "none": {"sigma_voxels": 0.0, "truncate": 0.0, "boundary_mode": "edge"},
    "low": {"sigma_voxels": 0.5, "truncate": 2.0, "boundary_mode": "edge"},
    "medium": {"sigma_voxels": 1.0, "truncate": 3.0, "boundary_mode": "edge"},
    "high": {"sigma_voxels": 1.5, "truncate": 3.0, "boundary_mode": "edge"},
}
_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_SUPPORTED_BOUNDARY_MODES: Final[frozenset[str]] = frozenset({"edge"})


def gaussian_blur_parameters_for_severity(severity: str) -> dict[str, JsonValue]:
    """Return the exact persisted Gaussian-blur parameters for one severity level."""

    if severity not in GAUSSIAN_BLUR_SEVERITY_PARAMETERS:
        raise InvalidCorruptionSpecificationError(
            f"gaussian_blur severity must be one of {sorted(GAUSSIAN_BLUR_SEVERITY_PARAMETERS)!r}."
        )
    return dict(GAUSSIAN_BLUR_SEVERITY_PARAMETERS[severity])


def apply_gaussian_blur(
    image: npt.ArrayLike,
    specification: Phase7CorruptionSpecification,
    *,
    mask: npt.ArrayLike | None = None,
) -> Phase7ImageTransformOutput:
    """Apply deterministic image-only separable Gaussian blur to a finite 3D CT image."""

    _require_corruption_specification(
        specification,
        corruption_name=GAUSSIAN_BLUR_CORRUPTION_NAME,
        expected_parameters=gaussian_blur_parameters_for_severity(specification.severity),
        seed_required=False,
    )
    image_array = _require_ct_image(image)
    mask_array = _copy_validated_mask(mask, expected_shape=image_array.shape)
    input_image_hash = _array_content_sha256(image_array)
    input_mask_hash = _mask_content_sha256(mask_array) if mask_array is not None else None

    sigma_voxels = _expect_float_parameter(
        specification.parameters["sigma_voxels"],
        field_name="sigma_voxels",
    )
    truncate = _expect_float_parameter(
        specification.parameters["truncate"],
        field_name="truncate",
    )
    boundary_mode = _expect_boundary_mode(specification.parameters["boundary_mode"])
    if sigma_voxels == 0.0:
        transformed = np.ascontiguousarray(image_array.copy())
    else:
        transformed = _apply_separable_gaussian(
            image_array,
            sigma=sigma_voxels,
            truncate=truncate,
            boundary_mode=boundary_mode,
        )
    if not bool(np.isfinite(transformed).all()):
        raise NonFiniteTransformOutputError("gaussian_blur produced non-finite output.")

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


def gaussian_kernel_1d(*, sigma: float, truncate: float) -> np.ndarray:
    """Return the finite normalized 1D Gaussian kernel used by Phase 7 blur."""

    if not math.isfinite(sigma) or sigma < 0.0:
        raise InvalidCorruptionSpecificationError("sigma must be finite and nonnegative.")
    if not math.isfinite(truncate) or truncate < 0.0:
        raise InvalidCorruptionSpecificationError("truncate must be finite and nonnegative.")
    if sigma == 0.0:
        return np.asarray([1.0], dtype=_FLOAT_DTYPE)
    radius = int(math.ceil(truncate * sigma))
    if radius < 1:
        radius = 1
    offsets = np.arange(-radius, radius + 1, dtype=_FLOAT_DTYPE)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel_sum = float(np.sum(kernel))
    if not math.isfinite(kernel_sum) or kernel_sum <= 0.0:
        raise InvalidCorruptionSpecificationError("Gaussian kernel has invalid finite mass.")
    return np.ascontiguousarray(kernel / kernel_sum)


def _apply_separable_gaussian(
    image: np.ndarray,
    *,
    sigma: float,
    truncate: float,
    boundary_mode: Literal["edge"],
) -> np.ndarray:
    kernel = gaussian_kernel_1d(sigma=sigma, truncate=truncate)
    result = np.ascontiguousarray(image.astype(_FLOAT_DTYPE, copy=True))
    for axis in range(3):
        result = _convolve_along_axis(result, kernel=kernel, axis=axis, boundary_mode=boundary_mode)
    return np.ascontiguousarray(result)


def _convolve_along_axis(
    image: np.ndarray,
    *,
    kernel: np.ndarray,
    axis: int,
    boundary_mode: Literal["edge"],
) -> np.ndarray:
    radius = int((kernel.size - 1) // 2)
    if radius == 0:
        return np.ascontiguousarray(image.copy())
    pad_width = [(0, 0), (0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    pad_spec: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = (
        pad_width[0],
        pad_width[1],
        pad_width[2],
    )
    padded = np.pad(image, pad_spec, mode=boundary_mode)

    def _convolve(line: np.ndarray) -> np.ndarray:
        return np.convolve(line, kernel, mode="valid")

    convolved = np.apply_along_axis(_convolve, axis, padded)
    return np.ascontiguousarray(convolved.astype(_FLOAT_DTYPE, copy=False))


def _expect_float_parameter(value: JsonValue, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidCorruptionSpecificationError(f"{field_name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise InvalidCorruptionSpecificationError(f"{field_name} must be finite and nonnegative.")
    return numeric


def _expect_boundary_mode(value: JsonValue) -> Literal["edge"]:
    if not isinstance(value, str) or value not in _SUPPORTED_BOUNDARY_MODES:
        raise InvalidCorruptionSpecificationError(
            f"boundary_mode must be one of {sorted(_SUPPORTED_BOUNDARY_MODES)!r}."
        )
    return "edge"


__all__ = [
    "GAUSSIAN_BLUR_CORRUPTION_NAME",
    "GAUSSIAN_BLUR_SEVERITY_PARAMETERS",
    "apply_gaussian_blur",
    "gaussian_blur_parameters_for_severity",
    "gaussian_kernel_1d",
]
