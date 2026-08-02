"""Deterministic Phase 7 CT intensity robustness transforms."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.robustness.artifacts import (
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
    PHASE7_GEOMETRY_RECORD_SCHEMA_NAME,
    PHASE7_GEOMETRY_RECORD_SCHEMA_VERSION,
    PHASE7_TRANSFORM_RESULT_SCHEMA_NAME,
    PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION,
    Phase7CorruptionSpecification,
    Phase7GeometryRecord,
    Phase7TransformResult,
    phase7_corruption_specification_identity_payload,
)
from protoem_ct.robustness.geometry import (
    MASK_INTERPOLATION_NEAREST_NEIGHBOR,
    SpatialGrid,
    validate_binary_mask,
)

Array = npt.NDArray[np.generic]
FloatArray = npt.NDArray[np.float64]

INTENSITY_CORRUPTION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "contrast_shift",
        "hu_window_shift",
        "intensity_offset",
        "intensity_scale",
    }
)
INTENSITY_SEVERITY_PARAMETERS: Final[dict[str, dict[str, dict[str, float]]]] = {
    "hu_window_shift": {
        "none": {
            "window_center_shift_hu": 0.0,
            "window_min_hu": -200.0,
            "window_max_hu": 250.0,
        },
        "low": {
            "window_center_shift_hu": 25.0,
            "window_min_hu": -200.0,
            "window_max_hu": 250.0,
        },
        "medium": {
            "window_center_shift_hu": 50.0,
            "window_min_hu": -200.0,
            "window_max_hu": 250.0,
        },
        "high": {
            "window_center_shift_hu": 100.0,
            "window_min_hu": -200.0,
            "window_max_hu": 250.0,
        },
    },
    "intensity_scale": {
        "none": {"scale_factor": 1.0},
        "low": {"scale_factor": 0.95},
        "medium": {"scale_factor": 1.10},
        "high": {"scale_factor": 1.25},
    },
    "intensity_offset": {
        "none": {"offset_hu": 0.0},
        "low": {"offset_hu": 10.0},
        "medium": {"offset_hu": 25.0},
        "high": {"offset_hu": 50.0},
    },
    "contrast_shift": {
        "none": {"contrast_center_hu": 0.0, "contrast_factor": 1.0},
        "low": {"contrast_center_hu": 0.0, "contrast_factor": 0.90},
        "medium": {"contrast_center_hu": 0.0, "contrast_factor": 1.10},
        "high": {"contrast_center_hu": 0.0, "contrast_factor": 1.25},
    },
}

_FLOAT_DTYPE: Final[np.dtype[np.float64]] = np.dtype(np.float64)
_IMAGE_INTERPOLATION_FOR_NO_GEOMETRY_CHANGE: Final[str] = "linear"

JsonMapping: TypeAlias = dict[str, JsonValue]


class Phase7IntensityTransformError(ValueError):
    """Base error for deterministic Phase 7 intensity transforms."""


class InvalidIntensitySpecificationError(Phase7IntensityTransformError):
    """Raised when a corruption specification is not a supported intensity transform."""


class NonFiniteIntensityInputError(Phase7IntensityTransformError):
    """Raised when an input image contains NaN or Infinity."""


class NonFiniteIntensityOutputError(Phase7IntensityTransformError):
    """Raised when a transform produces a non-finite image."""


@dataclass(frozen=True, slots=True)
class IntensityTransformOutput:
    """Image-only intensity transform output with deterministic artifact records."""

    image: FloatArray
    mask: Array | None
    specification: Phase7CorruptionSpecification
    transform_result: Phase7TransformResult
    geometry_record: Phase7GeometryRecord | None


def intensity_severity_parameters(
    corruption_name: str,
    severity: str,
) -> JsonMapping:
    """Return exact deterministic parameters for one intensity corruption severity."""

    if corruption_name not in INTENSITY_SEVERITY_PARAMETERS:
        raise InvalidIntensitySpecificationError(
            f"unsupported intensity corruption: {corruption_name!r}."
        )
    severity_table = INTENSITY_SEVERITY_PARAMETERS[corruption_name]
    if severity not in severity_table:
        raise InvalidIntensitySpecificationError(f"unsupported intensity severity: {severity!r}.")
    return {key: float(value) for key, value in severity_table[severity].items()}


def build_intensity_corruption_specification(
    *,
    corruption_name: str,
    severity: str,
    deterministic_seed: int | None = None,
) -> Phase7CorruptionSpecification:
    """Build a self-hashed specification for a deterministic intensity transform."""

    payload: JsonMapping = {
        "changes_geometry": False,
        "common_grid_restoration_required": False,
        "corruption_name": corruption_name,
        "deterministic_seed": deterministic_seed,
        "image_interpolation": _IMAGE_INTERPOLATION_FOR_NO_GEOMETRY_CHANGE,
        "mask_interpolation": MASK_INTERPOLATION_NEAREST_NEIGHBOR,
        "parameters": intensity_severity_parameters(corruption_name, severity),
        "schema_name": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
        "schema_version": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
        "severity": severity,
    }
    return Phase7CorruptionSpecification(
        schema_name=PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
        schema_version=PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
        corruption_specification_hash=sha256_json(payload),
        corruption_name=corruption_name,
        severity=severity,
        parameters=cast(JsonMapping, payload["parameters"]),
        deterministic_seed=deterministic_seed,
        changes_geometry=False,
        image_interpolation=_IMAGE_INTERPOLATION_FOR_NO_GEOMETRY_CHANGE,
        mask_interpolation=MASK_INTERPOLATION_NEAREST_NEIGHBOR,
        common_grid_restoration_required=False,
    )


def apply_intensity_transform(
    image: Array,
    specification: Phase7CorruptionSpecification,
    *,
    mask: Array | None = None,
    spatial_grid: SpatialGrid | None = None,
) -> IntensityTransformOutput:
    """Apply one deterministic CT image-only intensity transform.

    Query labels and reference masks are intentionally absent from this API.
    """

    _validate_intensity_specification(specification)
    image_array = _require_finite_3d_image(image)
    mask_array = _prepare_mask(mask, expected_shape=cast(tuple[int, int, int], image_array.shape))
    if spatial_grid is not None and spatial_grid.shape != tuple(
        int(item) for item in image_array.shape
    ):
        raise InvalidIntensitySpecificationError("spatial_grid shape must match image shape.")

    with np.errstate(over="ignore", invalid="ignore"):
        transformed = _apply_transform(image_array, specification)
    if not bool(np.isfinite(transformed).all()):
        raise NonFiniteIntensityOutputError("transformed intensity image must be finite.")
    output_image = np.ascontiguousarray(transformed.astype(_FLOAT_DTYPE, copy=False))
    output_mask = None if mask_array is None else np.ascontiguousarray(mask_array.copy())

    geometry_record = _build_no_geometry_change_record(
        transform_name=specification.corruption_name,
        spatial_grid=spatial_grid,
        binary_mask_preserved=output_mask is not None or mask is None,
    )
    result = _build_transform_result(
        specification=specification,
        input_image=image_array,
        output_image=output_image,
        input_mask=mask_array,
        output_mask=output_mask,
        geometry_record=geometry_record,
    )
    return IntensityTransformOutput(
        image=output_image,
        mask=output_mask,
        specification=specification,
        transform_result=result,
        geometry_record=geometry_record,
    )


def query_label_not_part_of_intensity_transform_api() -> bool:
    """Return True because intensity transforms accept no query labels or reference masks."""

    return True


def array_content_sha256(array: Array) -> str:
    """Return a deterministic array content hash independent of memory layout."""

    contiguous = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(int(item) for item in contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _validate_intensity_specification(specification: Phase7CorruptionSpecification) -> None:
    if specification.corruption_name not in INTENSITY_CORRUPTION_NAMES:
        raise InvalidIntensitySpecificationError(
            f"unsupported intensity corruption: {specification.corruption_name!r}."
        )
    expected_parameters = intensity_severity_parameters(
        specification.corruption_name,
        specification.severity,
    )
    if dict(specification.parameters) != expected_parameters:
        raise InvalidIntensitySpecificationError(
            "intensity specification parameters must match the deterministic severity table."
        )
    if specification.changes_geometry:
        raise InvalidIntensitySpecificationError("intensity transforms must not change geometry.")
    if specification.common_grid_restoration_required:
        raise InvalidIntensitySpecificationError(
            "intensity transforms must not require fabricated common-grid restoration."
        )
    if specification.mask_interpolation != MASK_INTERPOLATION_NEAREST_NEIGHBOR:
        raise InvalidIntensitySpecificationError("mask interpolation must be nearest_neighbor.")
    if specification.corruption_specification_hash != sha256_json(
        phase7_corruption_specification_identity_payload(specification)
    ):
        raise InvalidIntensitySpecificationError("corruption specification self-hash is invalid.")


def _require_finite_3d_image(image: Array) -> FloatArray:
    value = np.asarray(image)
    if value.ndim != 3:
        raise NonFiniteIntensityInputError("intensity image must be 3D.")
    if value.dtype.kind not in {"f", "i", "u"}:
        raise NonFiniteIntensityInputError("intensity image must be numeric.")
    array = np.ascontiguousarray(value.astype(_FLOAT_DTYPE, copy=False))
    if not bool(np.isfinite(array).all()):
        raise NonFiniteIntensityInputError("intensity image must contain only finite values.")
    return array


def _prepare_mask(mask: Array | None, *, expected_shape: tuple[int, int, int]) -> Array | None:
    if mask is None:
        return None
    validate_binary_mask(mask, expected_shape=expected_shape)
    return np.ascontiguousarray(np.asarray(mask).copy())


def _apply_transform(
    image: FloatArray,
    specification: Phase7CorruptionSpecification,
) -> FloatArray:
    parameters = specification.parameters
    if specification.corruption_name == "hu_window_shift":
        shifted = image + _finite_parameter(parameters, "window_center_shift_hu")
        return np.clip(
            shifted,
            _finite_parameter(parameters, "window_min_hu"),
            _finite_parameter(parameters, "window_max_hu"),
        )
    if specification.corruption_name == "intensity_scale":
        scale_factor = _finite_parameter(parameters, "scale_factor")
        if scale_factor <= 0.0:
            raise InvalidIntensitySpecificationError("scale_factor must be positive.")
        return image * scale_factor
    if specification.corruption_name == "intensity_offset":
        return image + _finite_parameter(parameters, "offset_hu")
    if specification.corruption_name == "contrast_shift":
        center = _finite_parameter(parameters, "contrast_center_hu")
        factor = _finite_parameter(parameters, "contrast_factor")
        if factor <= 0.0:
            raise InvalidIntensitySpecificationError("contrast_factor must be positive.")
        return center + factor * (image - center)
    raise InvalidIntensitySpecificationError(
        f"unsupported intensity corruption: {specification.corruption_name!r}."
    )


def _finite_parameter(parameters: dict[str, JsonValue] | object, key: str) -> float:
    if not isinstance(parameters, dict) or key not in parameters:
        raise InvalidIntensitySpecificationError(f"missing required parameter: {key}.")
    value = parameters[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidIntensitySpecificationError(f"{key} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise InvalidIntensitySpecificationError(f"{key} must be finite.")
    return numeric


def _build_no_geometry_change_record(
    *,
    transform_name: str,
    spatial_grid: SpatialGrid | None,
    binary_mask_preserved: bool,
) -> Phase7GeometryRecord | None:
    if spatial_grid is None:
        return None
    orientation = "".join(spatial_grid.orientation)
    affine_hash = array_content_sha256(np.asarray(spatial_grid.affine, dtype=np.float64))
    payload: JsonMapping = {
        "binary_mask_preserved": binary_mask_preserved,
        "common_grid_restored": False,
        "geometry_change_type": "none",
        "image_interpolation": _IMAGE_INTERPOLATION_FOR_NO_GEOMETRY_CHANGE,
        "input_affine_hash": affine_hash,
        "input_orientation": orientation,
        "input_shape": list(spatial_grid.shape),
        "input_spacing": list(spatial_grid.spacing),
        "mask_interpolation": MASK_INTERPOLATION_NEAREST_NEIGHBOR,
        "output_affine_hash": affine_hash,
        "output_orientation": orientation,
        "output_shape": list(spatial_grid.shape),
        "output_spacing": list(spatial_grid.spacing),
        "restored_affine_hash": None,
        "restored_orientation": None,
        "restored_shape": None,
        "restored_spacing": None,
        "schema_name": PHASE7_GEOMETRY_RECORD_SCHEMA_NAME,
        "schema_version": PHASE7_GEOMETRY_RECORD_SCHEMA_VERSION,
        "transform_name": transform_name,
    }
    return Phase7GeometryRecord(
        schema_name=PHASE7_GEOMETRY_RECORD_SCHEMA_NAME,
        schema_version=PHASE7_GEOMETRY_RECORD_SCHEMA_VERSION,
        geometry_record_hash=sha256_json(payload),
        transform_name=transform_name,
        geometry_change_type="none",
        input_shape=spatial_grid.shape,
        output_shape=spatial_grid.shape,
        restored_shape=None,
        input_spacing=spatial_grid.spacing,
        output_spacing=spatial_grid.spacing,
        restored_spacing=None,
        input_orientation=orientation,
        output_orientation=orientation,
        restored_orientation=None,
        input_affine_hash=affine_hash,
        output_affine_hash=affine_hash,
        restored_affine_hash=None,
        image_interpolation=_IMAGE_INTERPOLATION_FOR_NO_GEOMETRY_CHANGE,
        mask_interpolation=MASK_INTERPOLATION_NEAREST_NEIGHBOR,
        binary_mask_preserved=binary_mask_preserved,
        common_grid_restored=False,
    )


def _build_transform_result(
    *,
    specification: Phase7CorruptionSpecification,
    input_image: FloatArray,
    output_image: FloatArray,
    input_mask: Array | None,
    output_mask: Array | None,
    geometry_record: Phase7GeometryRecord | None,
) -> Phase7TransformResult:
    input_mask_hash = None if input_mask is None else array_content_sha256(input_mask)
    output_mask_hash = None if output_mask is None else array_content_sha256(output_mask)
    payload: JsonMapping = {
        "common_grid_restored": False,
        "corruption_specification_hash": specification.corruption_specification_hash,
        "execution_status": "completed",
        "failure_code": None,
        "failure_message": None,
        "finite_output": True,
        "geometry_record_hash": None
        if geometry_record is None
        else geometry_record.geometry_record_hash,
        "input_image_content_hash": array_content_sha256(input_image),
        "input_mask_content_hash": input_mask_hash,
        "mask_binary_preserved": None if input_mask is None else True,
        "output_image_content_hash": array_content_sha256(output_image),
        "output_mask_content_hash": output_mask_hash,
        "schema_name": PHASE7_TRANSFORM_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION,
    }
    return Phase7TransformResult(
        schema_name=PHASE7_TRANSFORM_RESULT_SCHEMA_NAME,
        schema_version=PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION,
        transform_result_hash=sha256_json(payload),
        corruption_specification_hash=specification.corruption_specification_hash,
        input_image_content_hash=cast(str, payload["input_image_content_hash"]),
        input_mask_content_hash=input_mask_hash,
        output_image_content_hash=cast(str, payload["output_image_content_hash"]),
        output_mask_content_hash=output_mask_hash,
        geometry_record_hash=cast(str | None, payload["geometry_record_hash"]),
        finite_output=True,
        mask_binary_preserved=cast(bool | None, payload["mask_binary_preserved"]),
        common_grid_restored=False,
        execution_status="completed",
        failure_code=None,
        failure_message=None,
    )


__all__ = [
    "INTENSITY_CORRUPTION_NAMES",
    "INTENSITY_SEVERITY_PARAMETERS",
    "IntensityTransformOutput",
    "InvalidIntensitySpecificationError",
    "NonFiniteIntensityInputError",
    "NonFiniteIntensityOutputError",
    "Phase7IntensityTransformError",
    "apply_intensity_transform",
    "array_content_sha256",
    "build_intensity_corruption_specification",
    "intensity_severity_parameters",
    "query_label_not_part_of_intensity_transform_api",
]
