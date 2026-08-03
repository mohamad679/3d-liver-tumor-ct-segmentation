"""Deterministic crop/FOV robustness perturbation for Phase 7.

The public API in this module accepts CT image arrays and optional evaluation
masks only. It does not accept query labels or reference masks for prediction or
adaptation, does not execute models, and does not read or write files.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
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

Array = npt.NDArray[np.generic]
FloatArray = npt.NDArray[np.float64]
MaskArray = npt.NDArray[np.uint8]
CropBounds: TypeAlias = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

CROP_FOV_METADATA_SCHEMA_NAME: Final[str] = "phase7_crop_fov_metadata"
CROP_FOV_METADATA_SCHEMA_VERSION: Final[str] = "v1"
_ZERO_SHA256: Final[str] = "0" * 64
_ALLOWED_PARAMETER_KEYS: Final[frozenset[str]] = frozenset(
    {"crop_start", "crop_stop", "crop_margins", "crop_fraction"}
)
_NO_LABEL_FIELDS: Final[frozenset[str]] = frozenset(
    {"query_label", "query_labels", "label_map", "reference_mask", "query_reference_mask"}
)


class Phase7CropFovError(ValueError):
    """Base exception for deterministic Phase 7 crop/FOV failures."""


class InvalidCropSpecificationError(Phase7CropFovError):
    """Raised when crop/FOV parameters are invalid or outside the input volume."""


class CropFovInputError(Phase7CropFovError):
    """Raised when image, mask, or grid inputs violate the crop contract."""


class CropFovIdentityError(Phase7CropFovError):
    """Raised when self-hashed crop metadata is inconsistent."""


@dataclass(frozen=True, slots=True)
class CropFovMetadata:
    """Self-hashed metadata for one deterministic crop and common-grid restoration."""

    schema_name: str
    schema_version: str
    severity: str
    input_shape: tuple[int, int, int]
    crop_start: tuple[int, int, int]
    crop_stop: tuple[int, int, int]
    cropped_shape: tuple[int, int, int]
    pad_before: tuple[int, int, int]
    pad_after: tuple[int, int, int]
    restored_shape: tuple[int, int, int]
    mask_was_provided: bool
    input_mask_foreground_count: int | None
    cropped_mask_foreground_count: int | None
    restored_mask_foreground_count: int | None
    empty_lesion_before: bool | None
    empty_lesion_after_crop: bool | None
    lesion_removed: bool | None
    metadata_identity_hash: str

    def __post_init__(self) -> None:
        if self.schema_name != CROP_FOV_METADATA_SCHEMA_NAME:
            raise CropFovIdentityError("crop metadata schema_name is invalid.")
        if self.schema_version != CROP_FOV_METADATA_SCHEMA_VERSION:
            raise CropFovIdentityError("crop metadata schema_version is invalid.")
        _validate_shape(self.input_shape, field_name="input_shape")
        _validate_index_triplet(self.crop_start, field_name="crop_start")
        _validate_index_triplet(self.crop_stop, field_name="crop_stop")
        _validate_shape(self.cropped_shape, field_name="cropped_shape")
        _validate_index_triplet(self.pad_before, field_name="pad_before")
        _validate_index_triplet(self.pad_after, field_name="pad_after")
        _validate_shape(self.restored_shape, field_name="restored_shape")
        for axis, (start, stop, in_size, crop_size, pad_before, pad_after) in enumerate(
            zip(
                self.crop_start,
                self.crop_stop,
                self.input_shape,
                self.cropped_shape,
                self.pad_before,
                self.pad_after,
                strict=True,
            )
        ):
            if not 0 <= start < stop <= in_size:
                raise CropFovIdentityError(f"crop bounds are invalid on axis {axis}.")
            if stop - start != crop_size:
                raise CropFovIdentityError(f"cropped_shape does not match bounds on axis {axis}.")
            if pad_before != start or pad_after != in_size - stop:
                raise CropFovIdentityError(f"padding does not restore axis {axis}.")
        if self.restored_shape != self.input_shape:
            raise CropFovIdentityError("restored_shape must equal input_shape.")
        if not self.mask_was_provided:
            optional_fields = (
                self.input_mask_foreground_count,
                self.cropped_mask_foreground_count,
                self.restored_mask_foreground_count,
                self.empty_lesion_before,
                self.empty_lesion_after_crop,
                self.lesion_removed,
            )
            if any(item is not None for item in optional_fields):
                raise CropFovIdentityError("mask metadata must be null when no mask is provided.")
        else:
            _validate_optional_count(
                self.input_mask_foreground_count,
                field_name="input_mask_foreground_count",
            )
            _validate_optional_count(
                self.cropped_mask_foreground_count,
                field_name="cropped_mask_foreground_count",
            )
            _validate_optional_count(
                self.restored_mask_foreground_count,
                field_name="restored_mask_foreground_count",
            )
            if not all(
                isinstance(item, bool)
                for item in (
                    self.empty_lesion_before,
                    self.empty_lesion_after_crop,
                    self.lesion_removed,
                )
            ):
                raise CropFovIdentityError("mask empty-lesion fields must be booleans.")
        expected_hash = hash_crop_fov_metadata(self)
        if self.metadata_identity_hash not in {_ZERO_SHA256, expected_hash}:
            raise CropFovIdentityError("crop metadata identity hash mismatch.")
        if self.metadata_identity_hash == _ZERO_SHA256:
            object.__setattr__(self, "metadata_identity_hash", expected_hash)


@dataclass(frozen=True, slots=True)
class CropFovResult:
    """Immutable result for one crop/FOV perturbation restored to a common grid."""

    cropped_image: FloatArray
    restored_image: FloatArray
    cropped_mask: MaskArray | None
    restored_mask: MaskArray | None
    changed_grid: SpatialGrid
    target_common_grid: SpatialGrid
    metadata: CropFovMetadata
    geometry_change_record_hash: str
    common_grid_restoration_record_hash: str
    geometry_record: Phase7GeometryRecord
    transform_result: Phase7TransformResult
    result_identity_hash: str

    def __post_init__(self) -> None:
        expected_hash = hash_crop_fov_result(self)
        if self.result_identity_hash not in {_ZERO_SHA256, expected_hash}:
            raise CropFovIdentityError("crop/FOV result identity hash mismatch.")
        if self.result_identity_hash == _ZERO_SHA256:
            object.__setattr__(self, "result_identity_hash", expected_hash)


def apply_crop_fov_perturbation(
    *,
    image: Array,
    image_grid: SpatialGrid,
    corruption_specification: Phase7CorruptionSpecification,
    evaluation_mask: Array | None = None,
    mask_grid: SpatialGrid | None = None,
) -> CropFovResult:
    """Apply a deterministic crop/FOV perturbation and restore to the input grid.

    ``evaluation_mask`` is optional metadata for evaluation-only flows. The mask
    is cropped and padded with nearest-neighbor semantics, but it is not used to
    choose crop bounds or modify the image transform.
    """

    _validate_crop_specification(corruption_specification)
    image_array = _require_image(image, expected_shape=image_grid.shape)
    mask_array = _require_optional_mask(
        evaluation_mask,
        mask_grid=mask_grid,
        image_grid=image_grid,
    )
    crop_bounds = crop_bounds_from_specification(
        corruption_specification=corruption_specification,
        input_shape=image_grid.shape,
    )
    cropped_image = _crop_array(image_array, crop_bounds)
    restored_image = _restore_array(cropped_image, crop_bounds, image_grid.shape).astype(
        np.float64,
        copy=False,
    )
    if not bool(np.isfinite(restored_image).all()):
        raise CropFovInputError("restored image contains non-finite values.")

    cropped_mask: MaskArray | None = None
    restored_mask: MaskArray | None = None
    if mask_array is not None:
        cropped_mask = _crop_array(mask_array, crop_bounds).astype(np.uint8, copy=False)
        restored_mask = _restore_array(cropped_mask, crop_bounds, image_grid.shape).astype(
            np.uint8,
            copy=False,
        )
        validate_binary_mask(cropped_mask, expected_shape=cropped_mask.shape)
        validate_binary_mask(restored_mask, expected_shape=image_grid.shape)

    changed_grid = _build_changed_grid(image_grid=image_grid, crop_bounds=crop_bounds)
    geometry_change_record = build_geometry_change_record(
        source_grid=image_grid,
        changed_grid=changed_grid,
        permitted_changes=("affine", "field_of_view", "shape"),
        image_interpolation=corruption_specification.image_interpolation,
        mask_interpolation=MASK_INTERPOLATION_NEAREST_NEIGHBOR,
        common_grid_required=True,
    )
    common_grid_record = build_common_grid_restoration_record(
        source_grid=image_grid,
        changed_grid=changed_grid,
        restored_grid=image_grid,
        target_common_grid=image_grid,
        image_interpolation=corruption_specification.image_interpolation,
        mask_interpolation=MASK_INTERPOLATION_NEAREST_NEIGHBOR,
    )
    metadata = _build_metadata(
        severity=corruption_specification.severity,
        input_shape=image_grid.shape,
        crop_bounds=crop_bounds,
        cropped_mask=cropped_mask,
        restored_mask=restored_mask,
        input_mask=mask_array,
    )
    geometry_record = _build_phase7_geometry_record(
        image_grid=image_grid,
        changed_grid=changed_grid,
        image_interpolation=corruption_specification.image_interpolation,
        mask_was_provided=mask_array is not None,
    )
    transform_result = _build_transform_result(
        corruption_specification=corruption_specification,
        input_image=image_array,
        output_image=restored_image,
        input_mask=mask_array,
        output_mask=restored_mask,
        geometry_record=geometry_record,
    )
    result = CropFovResult(
        cropped_image=np.ascontiguousarray(cropped_image),
        restored_image=np.ascontiguousarray(restored_image),
        cropped_mask=None if cropped_mask is None else np.ascontiguousarray(cropped_mask),
        restored_mask=None if restored_mask is None else np.ascontiguousarray(restored_mask),
        changed_grid=changed_grid,
        target_common_grid=image_grid,
        metadata=metadata,
        geometry_change_record_hash=geometry_change_record.geometry_change_hash,
        common_grid_restoration_record_hash=common_grid_record.restoration_identity_hash,
        geometry_record=geometry_record,
        transform_result=transform_result,
        result_identity_hash=_ZERO_SHA256,
    )
    return type(result)(
        cropped_image=result.cropped_image,
        restored_image=result.restored_image,
        cropped_mask=result.cropped_mask,
        restored_mask=result.restored_mask,
        changed_grid=result.changed_grid,
        target_common_grid=result.target_common_grid,
        metadata=result.metadata,
        geometry_change_record_hash=result.geometry_change_record_hash,
        common_grid_restoration_record_hash=result.common_grid_restoration_record_hash,
        geometry_record=result.geometry_record,
        transform_result=result.transform_result,
        result_identity_hash=hash_crop_fov_result(result),
    )


def crop_bounds_from_specification(
    *,
    corruption_specification: Phase7CorruptionSpecification,
    input_shape: tuple[int, int, int],
) -> CropBounds:
    """Return inclusive-exclusive crop bounds from explicit spec parameters."""

    _validate_crop_specification(corruption_specification)
    _validate_shape(input_shape, field_name="input_shape")
    parameters = corruption_specification.parameters
    parameter_keys = set(parameters)
    if not parameter_keys.issubset(_ALLOWED_PARAMETER_KEYS):
        unsupported = sorted(parameter_keys - _ALLOWED_PARAMETER_KEYS)
        raise InvalidCropSpecificationError(
            f"unsupported crop/FOV parameter keys: {unsupported!r}."
        )
    if "crop_start" in parameters or "crop_stop" in parameters:
        if parameter_keys != {"crop_start", "crop_stop"}:
            raise InvalidCropSpecificationError(
                "crop_start/crop_stop cannot be mixed with margin or fraction parameters."
            )
        start = _expect_int_triplet(parameters["crop_start"], field_name="crop_start")
        stop = _expect_int_triplet(parameters["crop_stop"], field_name="crop_stop")
    elif "crop_margins" in parameters:
        if parameter_keys != {"crop_margins"}:
            raise InvalidCropSpecificationError(
                "crop_margins cannot be mixed with other crop parameters."
            )
        margins = _expect_margin_triplets(parameters["crop_margins"])
        start = cast(tuple[int, int, int], tuple(item[0] for item in margins))
        stop = cast(
            tuple[int, int, int],
            tuple(size - item[1] for size, item in zip(input_shape, margins, strict=True)),
        )
    elif "crop_fraction" in parameters:
        if parameter_keys != {"crop_fraction"}:
            raise InvalidCropSpecificationError(
                "crop_fraction cannot be mixed with other crop parameters."
            )
        fraction = _expect_fraction(parameters["crop_fraction"], field_name="crop_fraction")
        margin_counts = cast(
            tuple[int, int, int],
            tuple(int(math.floor(float(size) * fraction)) for size in input_shape),
        )
        start = margin_counts
        stop = cast(
            tuple[int, int, int],
            tuple(size - margin for size, margin in zip(input_shape, margin_counts, strict=True)),
        )
    else:
        raise InvalidCropSpecificationError("crop/FOV specification requires explicit crop bounds.")

    crop_bounds = tuple(
        (int(axis_start), int(axis_stop)) for axis_start, axis_stop in zip(start, stop, strict=True)
    )
    _validate_crop_bounds(cast(CropBounds, crop_bounds), input_shape=input_shape)
    return cast(CropBounds, crop_bounds)


def crop_fov_metadata_to_dict(metadata: CropFovMetadata) -> dict[str, JsonValue]:
    """Return a canonical JSON-compatible mapping for crop/FOV metadata."""

    return {
        "schema_name": metadata.schema_name,
        "schema_version": metadata.schema_version,
        "severity": metadata.severity,
        "input_shape": list(metadata.input_shape),
        "crop_start": list(metadata.crop_start),
        "crop_stop": list(metadata.crop_stop),
        "cropped_shape": list(metadata.cropped_shape),
        "pad_before": list(metadata.pad_before),
        "pad_after": list(metadata.pad_after),
        "restored_shape": list(metadata.restored_shape),
        "mask_was_provided": metadata.mask_was_provided,
        "input_mask_foreground_count": metadata.input_mask_foreground_count,
        "cropped_mask_foreground_count": metadata.cropped_mask_foreground_count,
        "restored_mask_foreground_count": metadata.restored_mask_foreground_count,
        "empty_lesion_before": metadata.empty_lesion_before,
        "empty_lesion_after_crop": metadata.empty_lesion_after_crop,
        "lesion_removed": metadata.lesion_removed,
        "metadata_identity_hash": metadata.metadata_identity_hash,
    }


def crop_fov_metadata_identity_payload(metadata: CropFovMetadata) -> dict[str, JsonValue]:
    """Return the deterministic identity payload for crop/FOV metadata."""

    payload = crop_fov_metadata_to_dict(metadata)
    payload.pop("metadata_identity_hash")
    return payload


def hash_crop_fov_metadata(metadata: CropFovMetadata) -> str:
    """Return the canonical hash for crop/FOV metadata."""

    return sha256_json(crop_fov_metadata_identity_payload(metadata))


def crop_fov_result_identity_payload(result: CropFovResult) -> dict[str, JsonValue]:
    """Return the deterministic identity payload for a crop/FOV result."""

    return {
        "common_grid_restoration_record_hash": result.common_grid_restoration_record_hash,
        "cropped_image_content_hash": _array_content_hash(result.cropped_image),
        "cropped_mask_content_hash": None
        if result.cropped_mask is None
        else _array_content_hash(result.cropped_mask),
        "geometry_change_record_hash": result.geometry_change_record_hash,
        "geometry_record_hash": result.geometry_record.geometry_record_hash,
        "metadata_identity_hash": result.metadata.metadata_identity_hash,
        "restored_image_content_hash": _array_content_hash(result.restored_image),
        "restored_mask_content_hash": None
        if result.restored_mask is None
        else _array_content_hash(result.restored_mask),
        "target_common_grid_hash": result.target_common_grid.grid_identity_hash,
        "transform_result_hash": result.transform_result.transform_result_hash,
    }


def hash_crop_fov_result(result: CropFovResult) -> str:
    """Return the canonical identity hash for a crop/FOV result."""

    return sha256_json(crop_fov_result_identity_payload(result))


def crop_fov_public_api_excludes_query_labels() -> bool:
    """Return true when the public crop API exposes no query-label/reference-mask fields."""

    fields = set(apply_crop_fov_perturbation.__annotations__)
    return fields.isdisjoint(_NO_LABEL_FIELDS)


def _validate_crop_specification(specification: Phase7CorruptionSpecification) -> None:
    if specification.corruption_name != "crop_fov":
        raise InvalidCropSpecificationError("corruption_name must be crop_fov.")
    if not specification.changes_geometry:
        raise InvalidCropSpecificationError("crop_fov must be marked as geometry-changing.")
    if not specification.common_grid_restoration_required:
        raise InvalidCropSpecificationError("crop_fov must require common-grid restoration.")
    validate_interpolation_policy(
        image_interpolation=specification.image_interpolation,
        mask_interpolation=specification.mask_interpolation,
    )


def _require_image(image: Array, *, expected_shape: tuple[int, int, int]) -> FloatArray:
    array = np.asarray(image)
    if array.shape != expected_shape:
        raise CropFovInputError("image shape must match image_grid shape.")
    if array.ndim != 3:
        raise CropFovInputError("image must be a 3D array.")
    if np.iscomplexobj(array):
        raise CropFovInputError("image must contain real values.")
    image64 = np.ascontiguousarray(array.astype(np.float64, copy=False))
    if not bool(np.isfinite(image64).all()):
        raise CropFovInputError("image values must be finite.")
    return image64


def _require_optional_mask(
    evaluation_mask: Array | None,
    *,
    mask_grid: SpatialGrid | None,
    image_grid: SpatialGrid,
) -> MaskArray | None:
    if evaluation_mask is None:
        if mask_grid is not None:
            raise CropFovInputError("mask_grid cannot be provided without evaluation_mask.")
        return None
    if mask_grid is None:
        raise CropFovInputError("mask_grid is required when evaluation_mask is provided.")
    validate_image_mask_alignment(image_grid, mask_grid)
    validate_binary_mask(evaluation_mask, expected_shape=image_grid.shape)
    return np.ascontiguousarray(np.asarray(evaluation_mask).astype(np.uint8, copy=False))


def _crop_array(array: npt.NDArray[np.generic], crop_bounds: CropBounds) -> npt.NDArray[np.generic]:
    return np.ascontiguousarray(
        array[
            crop_bounds[0][0] : crop_bounds[0][1],
            crop_bounds[1][0] : crop_bounds[1][1],
            crop_bounds[2][0] : crop_bounds[2][1],
        ]
    )


def _restore_array(
    cropped: npt.NDArray[np.generic],
    crop_bounds: CropBounds,
    restored_shape: tuple[int, int, int],
) -> npt.NDArray[np.generic]:
    output = np.zeros(restored_shape, dtype=cropped.dtype)
    output[
        crop_bounds[0][0] : crop_bounds[0][1],
        crop_bounds[1][0] : crop_bounds[1][1],
        crop_bounds[2][0] : crop_bounds[2][1],
    ] = cropped
    return np.ascontiguousarray(output)


def _build_changed_grid(*, image_grid: SpatialGrid, crop_bounds: CropBounds) -> SpatialGrid:
    crop_start = tuple(start for start, _stop in crop_bounds)
    cropped_shape = tuple(stop - start for start, stop in crop_bounds)
    affine = np.asarray(image_grid.affine, dtype=np.float64).copy()
    linear = affine[:3, :3]
    affine[:3, 3] = affine[:3, 3] + linear @ np.asarray(crop_start, dtype=np.float64)
    return build_spatial_grid(
        shape=cast(tuple[int, int, int], cropped_shape),
        affine=tuple(tuple(float(item) for item in row) for row in affine.tolist()),
        orientation=image_grid.orientation,
    )


def _build_metadata(
    *,
    severity: str,
    input_shape: tuple[int, int, int],
    crop_bounds: CropBounds,
    input_mask: MaskArray | None,
    cropped_mask: MaskArray | None,
    restored_mask: MaskArray | None,
) -> CropFovMetadata:
    crop_start = tuple(start for start, _stop in crop_bounds)
    crop_stop = tuple(stop for _start, stop in crop_bounds)
    cropped_shape = tuple(stop - start for start, stop in crop_bounds)
    pad_after = tuple(size - stop for size, stop in zip(input_shape, crop_stop, strict=True))
    if input_mask is None:
        metadata = CropFovMetadata(
            schema_name=CROP_FOV_METADATA_SCHEMA_NAME,
            schema_version=CROP_FOV_METADATA_SCHEMA_VERSION,
            severity=severity,
            input_shape=input_shape,
            crop_start=cast(tuple[int, int, int], crop_start),
            crop_stop=cast(tuple[int, int, int], crop_stop),
            cropped_shape=cast(tuple[int, int, int], cropped_shape),
            pad_before=cast(tuple[int, int, int], crop_start),
            pad_after=cast(tuple[int, int, int], pad_after),
            restored_shape=input_shape,
            mask_was_provided=False,
            input_mask_foreground_count=None,
            cropped_mask_foreground_count=None,
            restored_mask_foreground_count=None,
            empty_lesion_before=None,
            empty_lesion_after_crop=None,
            lesion_removed=None,
            metadata_identity_hash=_ZERO_SHA256,
        )
        return metadata

    assert cropped_mask is not None
    assert restored_mask is not None
    input_count = _foreground_count(input_mask)
    cropped_count = _foreground_count(cropped_mask)
    restored_count = _foreground_count(restored_mask)
    metadata = CropFovMetadata(
        schema_name=CROP_FOV_METADATA_SCHEMA_NAME,
        schema_version=CROP_FOV_METADATA_SCHEMA_VERSION,
        severity=severity,
        input_shape=input_shape,
        crop_start=cast(tuple[int, int, int], crop_start),
        crop_stop=cast(tuple[int, int, int], crop_stop),
        cropped_shape=cast(tuple[int, int, int], cropped_shape),
        pad_before=cast(tuple[int, int, int], crop_start),
        pad_after=cast(tuple[int, int, int], pad_after),
        restored_shape=input_shape,
        mask_was_provided=True,
        input_mask_foreground_count=input_count,
        cropped_mask_foreground_count=cropped_count,
        restored_mask_foreground_count=restored_count,
        empty_lesion_before=input_count == 0,
        empty_lesion_after_crop=cropped_count == 0,
        lesion_removed=input_count > 0 and cropped_count == 0,
        metadata_identity_hash=_ZERO_SHA256,
    )
    return metadata


def _build_phase7_geometry_record(
    *,
    image_grid: SpatialGrid,
    changed_grid: SpatialGrid,
    image_interpolation: str,
    mask_was_provided: bool,
) -> Phase7GeometryRecord:
    schema_name = PHASE7_GEOMETRY_RECORD_SCHEMA_NAME
    schema_version = PHASE7_GEOMETRY_RECORD_SCHEMA_VERSION
    transform_name = "crop_fov"
    geometry_change_type = "crop_fov"
    input_shape = image_grid.shape
    output_shape = changed_grid.shape
    restored_shape = image_grid.shape
    input_spacing = image_grid.spacing
    output_spacing = changed_grid.spacing
    restored_spacing = image_grid.spacing
    input_orientation = "".join(image_grid.orientation)
    output_orientation = "".join(changed_grid.orientation)
    restored_orientation = "".join(image_grid.orientation)
    input_affine_hash = _affine_content_hash(image_grid.affine)
    output_affine_hash = _affine_content_hash(changed_grid.affine)
    restored_affine_hash = _affine_content_hash(image_grid.affine)
    mask_interpolation = MASK_INTERPOLATION_NEAREST_NEIGHBOR
    binary_mask_preserved = True if mask_was_provided else True
    common_grid_restored = True
    identity_payload: dict[str, JsonValue] = {
        "binary_mask_preserved": binary_mask_preserved,
        "common_grid_restored": common_grid_restored,
        "geometry_change_type": geometry_change_type,
        "image_interpolation": image_interpolation,
        "input_affine_hash": input_affine_hash,
        "input_orientation": input_orientation,
        "input_shape": list(input_shape),
        "input_spacing": list(input_spacing),
        "mask_interpolation": mask_interpolation,
        "output_affine_hash": output_affine_hash,
        "output_orientation": output_orientation,
        "output_shape": list(output_shape),
        "output_spacing": list(output_spacing),
        "restored_affine_hash": restored_affine_hash,
        "restored_orientation": restored_orientation,
        "restored_shape": list(restored_shape),
        "restored_spacing": list(restored_spacing),
        "schema_name": schema_name,
        "schema_version": schema_version,
        "transform_name": transform_name,
    }
    record = Phase7GeometryRecord(
        schema_name=schema_name,
        schema_version=schema_version,
        geometry_record_hash=sha256_json(identity_payload),
        transform_name=transform_name,
        geometry_change_type=geometry_change_type,
        input_shape=input_shape,
        output_shape=output_shape,
        restored_shape=restored_shape,
        input_spacing=input_spacing,
        output_spacing=output_spacing,
        restored_spacing=restored_spacing,
        input_orientation=input_orientation,
        output_orientation=output_orientation,
        restored_orientation=restored_orientation,
        input_affine_hash=input_affine_hash,
        output_affine_hash=output_affine_hash,
        restored_affine_hash=restored_affine_hash,
        image_interpolation=image_interpolation,
        mask_interpolation=mask_interpolation,
        binary_mask_preserved=binary_mask_preserved,
        common_grid_restored=common_grid_restored,
    )
    if record.geometry_record_hash != hash_phase7_geometry_record(record):
        raise CropFovIdentityError("Phase 7 geometry record hash mismatch.")
    return record


def _build_transform_result(
    *,
    corruption_specification: Phase7CorruptionSpecification,
    input_image: FloatArray,
    output_image: FloatArray,
    input_mask: MaskArray | None,
    output_mask: MaskArray | None,
    geometry_record: Phase7GeometryRecord,
) -> Phase7TransformResult:
    schema_name = PHASE7_TRANSFORM_RESULT_SCHEMA_NAME
    schema_version = PHASE7_TRANSFORM_RESULT_SCHEMA_VERSION
    corruption_specification_hash = corruption_specification.corruption_specification_hash
    input_image_content_hash = _array_content_hash(input_image)
    input_mask_content_hash = None if input_mask is None else _array_content_hash(input_mask)
    output_image_content_hash = _array_content_hash(output_image)
    output_mask_content_hash = None if output_mask is None else _array_content_hash(output_mask)
    geometry_record_hash = geometry_record.geometry_record_hash
    finite_output = True
    mask_binary_preserved = None if input_mask is None else True
    common_grid_restored = True
    execution_status = "completed"
    failure_code = None
    failure_message = None
    identity_payload: dict[str, JsonValue] = {
        "common_grid_restored": common_grid_restored,
        "corruption_specification_hash": corruption_specification_hash,
        "execution_status": execution_status,
        "failure_code": failure_code,
        "failure_message": failure_message,
        "finite_output": finite_output,
        "geometry_record_hash": geometry_record_hash,
        "input_image_content_hash": input_image_content_hash,
        "input_mask_content_hash": input_mask_content_hash,
        "mask_binary_preserved": mask_binary_preserved,
        "output_image_content_hash": output_image_content_hash,
        "output_mask_content_hash": output_mask_content_hash,
        "schema_name": schema_name,
        "schema_version": schema_version,
    }
    result = Phase7TransformResult(
        schema_name=schema_name,
        schema_version=schema_version,
        transform_result_hash=sha256_json(identity_payload),
        corruption_specification_hash=corruption_specification_hash,
        input_image_content_hash=input_image_content_hash,
        input_mask_content_hash=input_mask_content_hash,
        output_image_content_hash=output_image_content_hash,
        output_mask_content_hash=output_mask_content_hash,
        geometry_record_hash=geometry_record_hash,
        finite_output=finite_output,
        mask_binary_preserved=mask_binary_preserved,
        common_grid_restored=common_grid_restored,
        execution_status=execution_status,
        failure_code=failure_code,
        failure_message=failure_message,
    )
    if result.transform_result_hash != hash_phase7_transform_result(result):
        raise CropFovIdentityError("Phase 7 transform result hash mismatch.")
    return result


def _validate_crop_bounds(crop_bounds: CropBounds, *, input_shape: tuple[int, int, int]) -> None:
    if len(crop_bounds) != 3:
        raise InvalidCropSpecificationError("crop bounds must contain three axes.")
    for axis, ((start, stop), size) in enumerate(zip(crop_bounds, input_shape, strict=True)):
        if isinstance(start, bool) or isinstance(stop, bool):
            raise InvalidCropSpecificationError("crop bounds must be integers.")
        if not 0 <= start < stop <= size:
            raise InvalidCropSpecificationError(f"invalid crop bounds on axis {axis}.")


def _expect_int_triplet(value: object, *, field_name: str) -> tuple[int, int, int]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise InvalidCropSpecificationError(f"{field_name} must contain three integers.")
    output: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise InvalidCropSpecificationError(f"{field_name}[{index}] must be an integer.")
        output.append(int(item))
    return cast(tuple[int, int, int], tuple(output))


def _expect_margin_triplets(
    value: object,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise InvalidCropSpecificationError(
            "crop_margins must contain three [before, after] pairs."
        )
    margins: list[tuple[int, int]] = []
    for axis, item in enumerate(value):
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise InvalidCropSpecificationError(f"crop_margins[{axis}] must contain two integers.")
        before, after = item
        if (
            isinstance(before, bool)
            or isinstance(after, bool)
            or not isinstance(before, int)
            or not isinstance(after, int)
            or before < 0
            or after < 0
        ):
            raise InvalidCropSpecificationError(
                f"crop_margins[{axis}] values must be nonnegative integers."
            )
        margins.append((int(before), int(after)))
    return cast(tuple[tuple[int, int], tuple[int, int], tuple[int, int]], tuple(margins))


def _expect_fraction(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidCropSpecificationError(f"{field_name} must be a finite number.")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction < 0.5:
        raise InvalidCropSpecificationError(f"{field_name} must be in [0, 0.5).")
    return fraction


def _validate_shape(value: tuple[int, int, int], *, field_name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise CropFovIdentityError(f"{field_name} must contain three positive integers.")


def _validate_index_triplet(value: tuple[int, int, int], *, field_name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise CropFovIdentityError(f"{field_name} must contain three nonnegative integers.")


def _validate_optional_count(value: int | None, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CropFovIdentityError(f"{field_name} must be a nonnegative integer.")


def _foreground_count(mask: MaskArray) -> int:
    return int(np.count_nonzero(mask))


def _array_content_hash(value: npt.NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(value)
    payload = (
        str(contiguous.dtype).encode("utf-8")
        + b"|"
        + repr(tuple(int(item) for item in contiguous.shape)).encode("utf-8")
        + b"|"
        + contiguous.tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()


def _affine_content_hash(value: object) -> str:
    affine = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return _array_content_hash(affine)


__all__ = [
    "CROP_FOV_METADATA_SCHEMA_NAME",
    "CROP_FOV_METADATA_SCHEMA_VERSION",
    "CropBounds",
    "CropFovIdentityError",
    "CropFovInputError",
    "CropFovMetadata",
    "CropFovResult",
    "InvalidCropSpecificationError",
    "Phase7CropFovError",
    "apply_crop_fov_perturbation",
    "crop_bounds_from_specification",
    "crop_fov_metadata_identity_payload",
    "crop_fov_metadata_to_dict",
    "crop_fov_public_api_excludes_query_labels",
    "crop_fov_result_identity_payload",
    "hash_crop_fov_metadata",
    "hash_crop_fov_result",
]
