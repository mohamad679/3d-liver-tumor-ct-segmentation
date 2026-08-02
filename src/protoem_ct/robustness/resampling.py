"""Deterministic Phase 7 resampling robustness transforms.

This module implements only in-memory CPU resampling for Phase 7 robustness
assessment. It does not read or write files, execute models, compute metrics,
or accept query labels/reference masks for prediction or adaptation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage  # type: ignore[import-untyped]

from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.robustness.artifacts import (
    PHASE7_GEOMETRY_RECORD_SCHEMA_NAME,
    PHASE7_GEOMETRY_RECORD_SCHEMA_VERSION,
    PHASE7_TRANSFORM_RESULT_SCHEMA_NAME,
    PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION,
    Phase7CorruptionSpecification,
    Phase7GeometryRecord,
    Phase7TransformResult,
    hash_phase7_geometry_record,
    hash_phase7_transform_result,
)
from protoem_ct.robustness.geometry import (
    MASK_INTERPOLATION_NEAREST_NEIGHBOR,
    SpatialGrid,
    build_common_grid_restoration_record,
    build_geometry_change_record,
    build_spatial_grid,
    validate_binary_mask,
    validate_image_mask_alignment,
    validate_interpolation_policy,
)

FloatArray = npt.NDArray[np.float64]
MaskArray = npt.NDArray[np.uint8]
AffineTuple = tuple[tuple[float, float, float, float], ...]

RESAMPLING_RESULT_SCHEMA_NAME: Final[str] = "phase7_resampling_result"
RESAMPLING_RESULT_SCHEMA_VERSION: Final[str] = "v1"
SLICE_THICKNESS_CORRUPTION_NAME: Final[str] = "slice_thickness"
ANISOTROPIC_DOWNSAMPLING_CORRUPTION_NAME: Final[str] = "anisotropic_downsampling"
_SUPPORTED_CORRUPTIONS: Final[frozenset[str]] = frozenset(
    {SLICE_THICKNESS_CORRUPTION_NAME, ANISOTROPIC_DOWNSAMPLING_CORRUPTION_NAME}
)
_IMAGE_INTERPOLATION_ORDER: Final[dict[str, int]] = {"linear": 1, "bspline": 3}
_ZERO_SHA256: Final[str] = "0" * 64
_DEFAULT_ORIENTATION_STRING: Final[str] = "RAS"


class Phase7ResamplingError(ValueError):
    """Base error for deterministic Phase 7 resampling failures."""


class InvalidResamplingInputError(Phase7ResamplingError):
    """Raised when image, mask, grid, or specification inputs are malformed."""


class InvalidResamplingSpecificationError(Phase7ResamplingError):
    """Raised when a resampling corruption specification is unsupported."""


class ResamplingGeometryError(Phase7ResamplingError):
    """Raised when geometry records or common-grid restoration are inconsistent."""


@dataclass(frozen=True, slots=True)
class Phase7ResamplingResult:
    """In-memory deterministic result for one geometry-changing resampling transform."""

    schema_name: str
    schema_version: str
    restored_image: FloatArray
    restored_mask: MaskArray | None
    changed_image: FloatArray
    changed_mask: MaskArray | None
    source_grid: SpatialGrid
    changed_grid: SpatialGrid
    restored_grid: SpatialGrid
    geometry_record: Phase7GeometryRecord
    transform_result: Phase7TransformResult
    resampling_identity_hash: str

    def __post_init__(self) -> None:
        if self.schema_name != RESAMPLING_RESULT_SCHEMA_NAME:
            raise ResamplingGeometryError("resampling result schema_name is invalid.")
        if self.schema_version != RESAMPLING_RESULT_SCHEMA_VERSION:
            raise ResamplingGeometryError("resampling result schema_version is invalid.")
        restored = _require_float_image(self.restored_image, field_name="restored_image")
        changed = _require_float_image(self.changed_image, field_name="changed_image")
        if restored.shape != self.restored_grid.shape:
            raise ResamplingGeometryError("restored image shape must match restored_grid.")
        if changed.shape != self.changed_grid.shape:
            raise ResamplingGeometryError("changed image shape must match changed_grid.")
        if self.restored_mask is not None:
            validate_binary_mask(self.restored_mask, expected_shape=self.restored_grid.shape)
        if self.changed_mask is not None:
            validate_binary_mask(self.changed_mask, expected_shape=self.changed_grid.shape)
        if self.geometry_record.geometry_record_hash != hash_phase7_geometry_record(
            self.geometry_record
        ):
            raise ResamplingGeometryError("geometry record self-hash is invalid.")
        if self.transform_result.transform_result_hash != hash_phase7_transform_result(
            self.transform_result
        ):
            raise ResamplingGeometryError("transform result self-hash is invalid.")
        expected_hash = hash_resampling_result(self)
        if self.resampling_identity_hash not in {_ZERO_SHA256, expected_hash}:
            raise ResamplingGeometryError("resampling result identity hash mismatch.")
        if self.resampling_identity_hash == _ZERO_SHA256:
            object.__setattr__(self, "resampling_identity_hash", expected_hash)


def apply_slice_thickness_simulation(
    *,
    image: npt.ArrayLike,
    image_grid: SpatialGrid,
    specification: Phase7CorruptionSpecification,
    mask: npt.ArrayLike | None = None,
    mask_grid: SpatialGrid | None = None,
    target_common_grid: SpatialGrid | None = None,
) -> Phase7ResamplingResult:
    """Apply deterministic slice-thickness simulation and restore to the common grid.

    The slice axis is array axis 0, matching the repository's `[D, H, W]`
    convention. The changed grid uses `simulated_spacing` from the specification.
    """

    if specification.corruption_name != SLICE_THICKNESS_CORRUPTION_NAME:
        raise InvalidResamplingSpecificationError(
            "slice-thickness simulation requires a slice_thickness specification."
        )
    simulated_spacing = _require_spacing_parameter(
        specification.parameters.get("simulated_spacing"),
        field_name="simulated_spacing",
    )
    if not math.isclose(simulated_spacing[1], image_grid.spacing[1], rel_tol=0.0, abs_tol=1e-12):
        raise InvalidResamplingSpecificationError(
            "slice_thickness simulated_spacing may change only axis 0 spacing."
        )
    if not math.isclose(simulated_spacing[2], image_grid.spacing[2], rel_tol=0.0, abs_tol=1e-12):
        raise InvalidResamplingSpecificationError(
            "slice_thickness simulated_spacing may change only axis 0 spacing."
        )
    return apply_resampling_transform(
        image=image,
        image_grid=image_grid,
        specification=specification,
        output_spacing=simulated_spacing,
        mask=mask,
        mask_grid=mask_grid,
        target_common_grid=target_common_grid,
    )


def apply_anisotropic_downsampling(
    *,
    image: npt.ArrayLike,
    image_grid: SpatialGrid,
    specification: Phase7CorruptionSpecification,
    mask: npt.ArrayLike | None = None,
    mask_grid: SpatialGrid | None = None,
    target_common_grid: SpatialGrid | None = None,
) -> Phase7ResamplingResult:
    """Apply deterministic anisotropic downsampling and restore to the common grid."""

    if specification.corruption_name != ANISOTROPIC_DOWNSAMPLING_CORRUPTION_NAME:
        raise InvalidResamplingSpecificationError(
            "anisotropic downsampling requires an anisotropic_downsampling specification."
        )
    factors = _require_spacing_parameter(
        specification.parameters.get("downsampling_factors"),
        field_name="downsampling_factors",
    )
    if all(math.isclose(item, 1.0, rel_tol=0.0, abs_tol=1e-12) for item in factors):
        raise InvalidResamplingSpecificationError(
            "anisotropic_downsampling requires at least one non-identity factor."
        )
    if any(item < 1.0 for item in factors):
        raise InvalidResamplingSpecificationError(
            "downsampling_factors must be greater than or equal to 1.0."
        )
    output_spacing = tuple(
        float(spacing * factor) for spacing, factor in zip(image_grid.spacing, factors, strict=True)
    )
    return apply_resampling_transform(
        image=image,
        image_grid=image_grid,
        specification=specification,
        output_spacing=cast(tuple[float, float, float], output_spacing),
        mask=mask,
        mask_grid=mask_grid,
        target_common_grid=target_common_grid,
    )


def apply_resampling_transform(
    *,
    image: npt.ArrayLike,
    image_grid: SpatialGrid,
    specification: Phase7CorruptionSpecification,
    output_spacing: tuple[float, float, float],
    mask: npt.ArrayLike | None = None,
    mask_grid: SpatialGrid | None = None,
    target_common_grid: SpatialGrid | None = None,
) -> Phase7ResamplingResult:
    """Apply one supported resampling transform and restore image/mask to a common grid."""

    if specification.corruption_name not in _SUPPORTED_CORRUPTIONS:
        raise InvalidResamplingSpecificationError("unsupported resampling corruption.")
    if not specification.changes_geometry or not specification.common_grid_restoration_required:
        raise InvalidResamplingSpecificationError(
            "resampling specifications must change geometry and require common-grid restoration."
        )
    validate_interpolation_policy(
        image_interpolation=specification.image_interpolation,
        mask_interpolation=specification.mask_interpolation,
    )
    image_array = _require_float_image(image, field_name="image")
    if image_array.shape != image_grid.shape:
        raise InvalidResamplingInputError("image shape must match image_grid shape.")
    mask_array: MaskArray | None = None
    if (mask is None) != (mask_grid is None):
        raise InvalidResamplingInputError("mask and mask_grid must be supplied together.")
    if mask is not None and mask_grid is not None:
        validate_image_mask_alignment(image_grid, mask_grid)
        mask_candidate = np.asarray(mask)
        validate_binary_mask(mask_candidate, expected_shape=image_grid.shape)
        mask_array = np.ascontiguousarray(mask_candidate.astype(np.uint8, copy=False))

    target_grid = image_grid if target_common_grid is None else target_common_grid
    _validate_output_spacing(output_spacing)
    changed_shape = _shape_for_spacing(image_grid.shape, image_grid.spacing, output_spacing)
    changed_affine = _affine_with_spacing(image_grid.affine, output_spacing)
    changed_grid = build_spatial_grid(
        shape=changed_shape,
        affine=changed_affine,
        orientation=image_grid.orientation,
    )

    changed_image = _zoom_image(
        image_array,
        output_shape=changed_shape,
        image_interpolation=specification.image_interpolation,
    )
    changed_mask = (
        None if mask_array is None else _zoom_mask(mask_array, output_shape=changed_shape)
    )
    restored_image = _zoom_image(
        changed_image,
        output_shape=target_grid.shape,
        image_interpolation=specification.image_interpolation,
    )
    restored_mask = (
        None if changed_mask is None else _zoom_mask(changed_mask, output_shape=target_grid.shape)
    )
    if restored_mask is not None:
        validate_binary_mask(restored_mask, expected_shape=target_grid.shape)

    restored_grid = build_spatial_grid(
        shape=target_grid.shape,
        affine=target_grid.affine,
        orientation=target_grid.orientation,
    )
    build_geometry_change_record(
        source_grid=image_grid,
        changed_grid=changed_grid,
        permitted_changes=("affine", "field_of_view", "shape", "spacing"),
        image_interpolation=specification.image_interpolation,
        mask_interpolation=specification.mask_interpolation,
        common_grid_required=True,
    )
    build_common_grid_restoration_record(
        source_grid=image_grid,
        changed_grid=changed_grid,
        restored_grid=restored_grid,
        target_common_grid=target_grid,
        image_interpolation=specification.image_interpolation,
        mask_interpolation=specification.mask_interpolation,
    )
    if restored_mask is not None:
        validate_image_mask_alignment(restored_grid, target_grid)

    geometry_record = _build_phase7_geometry_record(
        transform_name=specification.corruption_name,
        source_grid=image_grid,
        changed_grid=changed_grid,
        restored_grid=restored_grid,
        image_interpolation=specification.image_interpolation,
    )
    transform_result = _build_transform_result(
        specification=specification,
        input_image=image_array,
        input_mask=mask_array,
        output_image=restored_image,
        output_mask=restored_mask,
        geometry_record=geometry_record,
    )
    return Phase7ResamplingResult(
        schema_name=RESAMPLING_RESULT_SCHEMA_NAME,
        schema_version=RESAMPLING_RESULT_SCHEMA_VERSION,
        restored_image=restored_image,
        restored_mask=restored_mask,
        changed_image=changed_image,
        changed_mask=changed_mask,
        source_grid=image_grid,
        changed_grid=changed_grid,
        restored_grid=restored_grid,
        geometry_record=geometry_record,
        transform_result=transform_result,
        resampling_identity_hash=_ZERO_SHA256,
    )


def resampling_result_identity_payload(result: Phase7ResamplingResult) -> dict[str, object]:
    """Return the deterministic identity payload for a resampling result."""

    return {
        "changed_grid_hash": result.changed_grid.grid_identity_hash,
        "changed_image_content_hash": _array_content_sha256(result.changed_image),
        "changed_mask_content_hash": None
        if result.changed_mask is None
        else _array_content_sha256(result.changed_mask),
        "geometry_record_hash": result.geometry_record.geometry_record_hash,
        "restored_grid_hash": result.restored_grid.grid_identity_hash,
        "restored_image_content_hash": _array_content_sha256(result.restored_image),
        "restored_mask_content_hash": None
        if result.restored_mask is None
        else _array_content_sha256(result.restored_mask),
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "source_grid_hash": result.source_grid.grid_identity_hash,
        "transform_result_hash": result.transform_result.transform_result_hash,
    }


def hash_resampling_result(result: Phase7ResamplingResult) -> str:
    """Return the canonical hash for a resampling result identity payload."""

    return sha256_json(resampling_result_identity_payload(result))


def query_label_not_part_of_resampling_api() -> bool:
    """Return true because resampling APIs do not accept query labels or reference masks."""

    return True


def _build_phase7_geometry_record(
    *,
    transform_name: str,
    source_grid: SpatialGrid,
    changed_grid: SpatialGrid,
    restored_grid: SpatialGrid,
    image_interpolation: str,
) -> Phase7GeometryRecord:
    payload = {
        "schema_name": PHASE7_GEOMETRY_RECORD_SCHEMA_NAME,
        "schema_version": PHASE7_GEOMETRY_RECORD_SCHEMA_VERSION,
        "transform_name": transform_name,
        "geometry_change_type": "resampling",
        "input_shape": list(source_grid.shape),
        "output_shape": list(changed_grid.shape),
        "restored_shape": list(restored_grid.shape),
        "input_spacing": list(source_grid.spacing),
        "output_spacing": list(changed_grid.spacing),
        "restored_spacing": list(restored_grid.spacing),
        "input_orientation": "".join(source_grid.orientation),
        "output_orientation": "".join(changed_grid.orientation),
        "restored_orientation": "".join(restored_grid.orientation),
        "input_affine_hash": _affine_hash(source_grid.affine),
        "output_affine_hash": _affine_hash(changed_grid.affine),
        "restored_affine_hash": _affine_hash(restored_grid.affine),
        "image_interpolation": image_interpolation,
        "mask_interpolation": MASK_INTERPOLATION_NEAREST_NEIGHBOR,
        "binary_mask_preserved": True,
        "common_grid_restored": True,
    }
    return Phase7GeometryRecord(
        schema_name=cast(str, payload["schema_name"]),
        schema_version=cast(str, payload["schema_version"]),
        geometry_record_hash=sha256_json(payload),
        transform_name=cast(str, payload["transform_name"]),
        geometry_change_type=cast(str, payload["geometry_change_type"]),
        input_shape=source_grid.shape,
        output_shape=changed_grid.shape,
        restored_shape=restored_grid.shape,
        input_spacing=source_grid.spacing,
        output_spacing=changed_grid.spacing,
        restored_spacing=restored_grid.spacing,
        input_orientation=cast(str, payload["input_orientation"]),
        output_orientation=cast(str, payload["output_orientation"]),
        restored_orientation=cast(str, payload["restored_orientation"]),
        input_affine_hash=cast(str, payload["input_affine_hash"]),
        output_affine_hash=cast(str, payload["output_affine_hash"]),
        restored_affine_hash=cast(str, payload["restored_affine_hash"]),
        image_interpolation=cast(str, payload["image_interpolation"]),
        mask_interpolation=cast(str, payload["mask_interpolation"]),
        binary_mask_preserved=cast(bool, payload["binary_mask_preserved"]),
        common_grid_restored=cast(bool, payload["common_grid_restored"]),
    )


def _build_transform_result(
    *,
    specification: Phase7CorruptionSpecification,
    input_image: FloatArray,
    input_mask: MaskArray | None,
    output_image: FloatArray,
    output_mask: MaskArray | None,
    geometry_record: Phase7GeometryRecord,
) -> Phase7TransformResult:
    payload = {
        "schema_name": PHASE7_TRANSFORM_RESULT_SCHEMA_NAME,
        "schema_version": PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION,
        "corruption_specification_hash": specification.corruption_specification_hash,
        "input_image_content_hash": _array_content_sha256(input_image),
        "input_mask_content_hash": None
        if input_mask is None
        else _array_content_sha256(input_mask),
        "output_image_content_hash": _array_content_sha256(output_image),
        "output_mask_content_hash": None
        if output_mask is None
        else _array_content_sha256(output_mask),
        "geometry_record_hash": geometry_record.geometry_record_hash,
        "finite_output": True,
        "mask_binary_preserved": True if input_mask is not None else None,
        "common_grid_restored": True,
        "execution_status": "completed",
        "failure_code": None,
        "failure_message": None,
    }
    return Phase7TransformResult(
        schema_name=cast(str, payload["schema_name"]),
        schema_version=cast(str, payload["schema_version"]),
        transform_result_hash=sha256_json(payload),
        corruption_specification_hash=cast(str, payload["corruption_specification_hash"]),
        input_image_content_hash=cast(str, payload["input_image_content_hash"]),
        input_mask_content_hash=cast(str | None, payload["input_mask_content_hash"]),
        output_image_content_hash=cast(str, payload["output_image_content_hash"]),
        output_mask_content_hash=cast(str | None, payload["output_mask_content_hash"]),
        geometry_record_hash=cast(str, payload["geometry_record_hash"]),
        finite_output=cast(bool, payload["finite_output"]),
        mask_binary_preserved=cast(bool | None, payload["mask_binary_preserved"]),
        common_grid_restored=cast(bool, payload["common_grid_restored"]),
        execution_status=cast(str, payload["execution_status"]),
        failure_code=cast(str | None, payload["failure_code"]),
        failure_message=cast(str | None, payload["failure_message"]),
    )


def _require_float_image(value: npt.ArrayLike, *, field_name: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidResamplingInputError(f"{field_name} must be numeric.") from exc
    if array.ndim != 3:
        raise InvalidResamplingInputError(f"{field_name} must be a 3D image.")
    if not bool(np.isfinite(array).all()):
        raise InvalidResamplingInputError(f"{field_name} must contain only finite values.")
    return np.ascontiguousarray(array)


def _require_spacing_parameter(value: object, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise InvalidResamplingSpecificationError(f"{field_name} must contain three values.")
    values: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise InvalidResamplingSpecificationError(f"{field_name}[{index}] must be numeric.")
        numeric = float(item)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise InvalidResamplingSpecificationError(
                f"{field_name}[{index}] must be positive and finite."
            )
        values.append(numeric)
    return cast(tuple[float, float, float], tuple(values))


def _validate_output_spacing(value: tuple[float, float, float]) -> None:
    if len(value) != 3:
        raise InvalidResamplingSpecificationError("output_spacing must contain three values.")
    for item in value:
        if not math.isfinite(float(item)) or float(item) <= 0.0:
            raise InvalidResamplingSpecificationError(
                "output_spacing values must be positive and finite."
            )


def _shape_for_spacing(
    input_shape: tuple[int, int, int],
    input_spacing: tuple[float, float, float],
    output_spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    return cast(
        tuple[int, int, int],
        tuple(
            max(1, int(round(size * in_spacing / out_spacing)))
            for size, in_spacing, out_spacing in zip(
                input_shape, input_spacing, output_spacing, strict=True
            )
        ),
    )


def _zoom_image(
    image: FloatArray,
    *,
    output_shape: tuple[int, int, int],
    image_interpolation: str,
) -> FloatArray:
    order = _IMAGE_INTERPOLATION_ORDER[image_interpolation]
    zoom_factors = _zoom_factors(image.shape, output_shape)
    output = cast(
        np.ndarray,
        ndimage.zoom(image, zoom=zoom_factors, order=order, mode="nearest", prefilter=order > 1),
    )
    output = np.ascontiguousarray(output.astype(np.float64, copy=False))
    if output.shape != output_shape:
        output = _coerce_shape(output, output_shape).astype(np.float64, copy=False)
    if not bool(np.isfinite(output).all()):
        raise InvalidResamplingInputError("resampled image contains non-finite values.")
    return output


def _zoom_mask(mask: MaskArray, *, output_shape: tuple[int, int, int]) -> MaskArray:
    zoom_factors = _zoom_factors(mask.shape, output_shape)
    output = cast(
        np.ndarray,
        ndimage.zoom(mask, zoom=zoom_factors, order=0, mode="nearest", prefilter=False),
    )
    output = np.ascontiguousarray(output.astype(np.uint8, copy=False))
    if output.shape != output_shape:
        output = _coerce_shape(output, output_shape).astype(np.uint8, copy=False)
    validate_binary_mask(output, expected_shape=output_shape)
    return np.ascontiguousarray(output)


def _zoom_factors(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    return cast(
        tuple[float, float, float],
        tuple(
            out_size / in_size for in_size, out_size in zip(input_shape, output_shape, strict=True)
        ),
    )


def _coerce_shape(array: FloatArray | MaskArray, output_shape: tuple[int, int, int]) -> np.ndarray:
    coerced = np.zeros(output_shape, dtype=array.dtype)
    slices = tuple(
        slice(0, min(found, expected))
        for found, expected in zip(array.shape, output_shape, strict=True)
    )
    coerced[slices] = array[slices]
    return np.ascontiguousarray(coerced)


def _affine_with_spacing(
    affine: AffineTuple,
    spacing: tuple[float, float, float],
) -> AffineTuple:
    array = np.asarray(affine, dtype=np.float64).copy()
    for axis, target_spacing in enumerate(spacing):
        column = array[:3, axis]
        norm = float(np.linalg.norm(column))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ResamplingGeometryError("affine columns must have positive finite norms.")
        array[:3, axis] = column / norm * target_spacing
    return cast(AffineTuple, tuple(tuple(float(item) for item in row) for row in array.tolist()))


def _affine_hash(affine: AffineTuple) -> str:
    return sha256_json({"affine": [[float(item) for item in row] for row in affine]})


def _array_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


__all__ = [
    "ANISOTROPIC_DOWNSAMPLING_CORRUPTION_NAME",
    "RESAMPLING_RESULT_SCHEMA_NAME",
    "RESAMPLING_RESULT_SCHEMA_VERSION",
    "SLICE_THICKNESS_CORRUPTION_NAME",
    "InvalidResamplingInputError",
    "InvalidResamplingSpecificationError",
    "Phase7ResamplingError",
    "Phase7ResamplingResult",
    "ResamplingGeometryError",
    "apply_anisotropic_downsampling",
    "apply_resampling_transform",
    "apply_slice_thickness_simulation",
    "hash_resampling_result",
    "query_label_not_part_of_resampling_api",
    "resampling_result_identity_payload",
]
