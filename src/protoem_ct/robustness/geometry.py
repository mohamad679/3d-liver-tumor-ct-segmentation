"""Deterministic Phase 7 geometry contracts and validators.

This module defines metadata-only geometry contracts. It does not resample,
crop, corrupt, evaluate, publish, or read/write files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from protoem_ct.artifacts.hashing import JsonValue, sha256_json

PHASE7_GEOMETRY_SCHEMA_VERSION: Final[str] = "v1"
SPATIAL_GRID_SCHEMA_NAME: Final[str] = "phase7_spatial_grid"
GEOMETRY_CHANGE_RECORD_SCHEMA_NAME: Final[str] = "phase7_geometry_change_record"
COMMON_GRID_RESTORATION_RECORD_SCHEMA_NAME: Final[str] = "phase7_common_grid_restoration_record"

ALLOWED_IMAGE_INTERPOLATIONS: Final[frozenset[str]] = frozenset({"linear", "bspline"})
MASK_INTERPOLATION_NEAREST_NEIGHBOR: Final[str] = "nearest_neighbor"
ALLOWED_GEOMETRY_CHANGE_FIELDS: Final[frozenset[str]] = frozenset(
    {"shape", "affine", "spacing", "orientation", "field_of_view"}
)
_AXIS_FAMILIES: Final[dict[str, str]] = {
    "R": "x",
    "L": "x",
    "A": "y",
    "P": "y",
    "S": "z",
    "I": "z",
}
_ZERO_SHA256: Final[str] = "0" * 64

Array = npt.NDArray[np.generic]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
AffineTuple = tuple[tuple[float, float, float, float], ...]


class Phase7GeometryError(ValueError):
    """Base exception for Phase 7 geometry contract failures."""


class InvalidSpatialGridError(Phase7GeometryError):
    """Raised when grid shape, affine, spacing, or orientation metadata is invalid."""


class GeometryAlignmentError(Phase7GeometryError):
    """Raised when paired image/mask or prediction/reference grids are misaligned."""


class InterpolationPolicyError(Phase7GeometryError):
    """Raised when image or mask interpolation violates the Phase 7 policy."""


class BinaryMaskValidationError(Phase7GeometryError):
    """Raised when a mask is non-finite, non-real, non-binary, or has invalid shape."""


class GeometryChangeValidationError(Phase7GeometryError):
    """Raised when geometry changes are implicit, unpermitted, or inconsistent."""


class CommonGridRestorationError(Phase7GeometryError):
    """Raised when common-grid restoration metadata is invalid or misaligned."""


@dataclass(frozen=True, slots=True)
class SpatialGrid:
    """Immutable spatial grid metadata for one 3D image, mask, or prediction."""

    schema_name: str
    schema_version: str
    shape: tuple[int, int, int]
    affine: AffineTuple
    spacing: tuple[float, float, float]
    orientation: tuple[str, str, str]
    grid_identity_hash: str

    def __post_init__(self) -> None:
        if self.schema_name != SPATIAL_GRID_SCHEMA_NAME:
            raise InvalidSpatialGridError("spatial grid schema_name is invalid.")
        if self.schema_version != PHASE7_GEOMETRY_SCHEMA_VERSION:
            raise InvalidSpatialGridError("spatial grid schema_version is invalid.")
        _validate_shape(self.shape, field_name="shape")
        affine = validate_affine(self.affine, field_name="affine")
        spacing = _validate_spacing(self.spacing, field_name="spacing")
        derived_spacing = spacing_from_affine(affine)
        if not _allclose_tuple(spacing, derived_spacing, tolerance=1e-9):
            raise InvalidSpatialGridError("spacing must match affine-derived voxel spacing.")
        _validate_orientation(self.orientation, field_name="orientation")
        expected_hash = _hash_spatial_grid_payload(self, identity_hash=_ZERO_SHA256)
        if self.grid_identity_hash not in {_ZERO_SHA256, expected_hash}:
            raise InvalidSpatialGridError("spatial grid identity hash mismatch.")
        if self.grid_identity_hash == _ZERO_SHA256:
            object.__setattr__(self, "grid_identity_hash", expected_hash)


@dataclass(frozen=True, slots=True)
class GeometryChangeRecord:
    """Metadata record for an explicitly permitted geometry-changing operation."""

    schema_name: str
    schema_version: str
    source_grid_hash: str
    changed_grid_hash: str
    permitted_changes: tuple[str, ...]
    observed_changes: tuple[str, ...]
    image_interpolation: str
    mask_interpolation: str
    common_grid_required: bool
    geometry_change_hash: str

    def __post_init__(self) -> None:
        if self.schema_name != GEOMETRY_CHANGE_RECORD_SCHEMA_NAME:
            raise GeometryChangeValidationError("geometry change schema_name is invalid.")
        if self.schema_version != PHASE7_GEOMETRY_SCHEMA_VERSION:
            raise GeometryChangeValidationError("geometry change schema_version is invalid.")
        _require_sha256(self.source_grid_hash, field_name="source_grid_hash")
        _require_sha256(self.changed_grid_hash, field_name="changed_grid_hash")
        _validate_change_names(self.permitted_changes, field_name="permitted_changes")
        _validate_change_names(self.observed_changes, field_name="observed_changes")
        if not set(self.observed_changes).issubset(set(self.permitted_changes)):
            raise GeometryChangeValidationError(
                "observed geometry changes must be explicitly permitted."
            )
        validate_interpolation_policy(
            image_interpolation=self.image_interpolation,
            mask_interpolation=self.mask_interpolation,
        )
        expected_hash = _hash_geometry_change_record_payload(
            self,
            identity_hash=_ZERO_SHA256,
        )
        if self.geometry_change_hash not in {_ZERO_SHA256, expected_hash}:
            raise GeometryChangeValidationError("geometry change identity hash mismatch.")
        if self.geometry_change_hash == _ZERO_SHA256:
            object.__setattr__(self, "geometry_change_hash", expected_hash)


@dataclass(frozen=True, slots=True)
class CommonGridRestorationRecord:
    """Metadata proving a changed result was restored to the evaluation grid."""

    schema_name: str
    schema_version: str
    source_grid_hash: str
    changed_grid_hash: str
    target_common_grid_hash: str
    restored_grid_hash: str
    image_interpolation: str
    mask_interpolation: str
    restored_to_common_grid: bool
    restoration_identity_hash: str

    def __post_init__(self) -> None:
        if self.schema_name != COMMON_GRID_RESTORATION_RECORD_SCHEMA_NAME:
            raise CommonGridRestorationError("common-grid restoration schema_name is invalid.")
        if self.schema_version != PHASE7_GEOMETRY_SCHEMA_VERSION:
            raise CommonGridRestorationError("common-grid restoration schema_version is invalid.")
        for field_name, value in {
            "source_grid_hash": self.source_grid_hash,
            "changed_grid_hash": self.changed_grid_hash,
            "target_common_grid_hash": self.target_common_grid_hash,
            "restored_grid_hash": self.restored_grid_hash,
        }.items():
            _require_sha256(value, field_name=field_name)
        if not self.restored_to_common_grid:
            raise CommonGridRestorationError("restored_to_common_grid must be true.")
        if self.restored_grid_hash != self.target_common_grid_hash:
            raise CommonGridRestorationError(
                "restored_grid_hash must equal target_common_grid_hash."
            )
        validate_interpolation_policy(
            image_interpolation=self.image_interpolation,
            mask_interpolation=self.mask_interpolation,
        )
        expected_hash = _hash_common_grid_restoration_record_payload(
            self,
            identity_hash=_ZERO_SHA256,
        )
        if self.restoration_identity_hash not in {_ZERO_SHA256, expected_hash}:
            raise CommonGridRestorationError("common-grid restoration identity hash mismatch.")
        if self.restoration_identity_hash == _ZERO_SHA256:
            object.__setattr__(self, "restoration_identity_hash", expected_hash)


def build_spatial_grid(
    *,
    shape: tuple[int, int, int],
    affine: object,
    orientation: tuple[str, str, str],
) -> SpatialGrid:
    """Build a self-hashed spatial grid from explicit shape, affine, and orientation."""

    normalized_affine = validate_affine(affine, field_name="affine")
    return SpatialGrid(
        schema_name=SPATIAL_GRID_SCHEMA_NAME,
        schema_version=PHASE7_GEOMETRY_SCHEMA_VERSION,
        shape=shape,
        affine=normalized_affine,
        spacing=spacing_from_affine(normalized_affine),
        orientation=orientation,
        grid_identity_hash=_ZERO_SHA256,
    )


def validate_affine(value: object, *, field_name: str = "affine") -> AffineTuple:
    """Return a normalized finite nonsingular 4x4 affine tuple."""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidSpatialGridError(f"{field_name} must be numeric.") from exc
    if array.shape != (4, 4):
        raise InvalidSpatialGridError(f"{field_name} must be a 4x4 affine.")
    if not bool(np.isfinite(array).all()):
        raise InvalidSpatialGridError(f"{field_name} must be finite.")
    if not bool(np.allclose(array[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-9, rtol=0.0)):
        raise InvalidSpatialGridError(f"{field_name} final row must equal [0, 0, 0, 1].")
    determinant = float(np.linalg.det(array[:3, :3]))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise InvalidSpatialGridError(f"{field_name} linear part must be nonsingular.")
    rows = tuple(tuple(float(item) for item in row) for row in array.tolist())
    return cast(AffineTuple, rows)


def spacing_from_affine(affine: AffineTuple) -> tuple[float, float, float]:
    """Compute positive voxel spacing from the columns of a validated affine."""

    array = np.asarray(validate_affine(affine), dtype=np.float64)
    spacing = np.sqrt(np.sum(array[:3, :3] * array[:3, :3], axis=0))
    if spacing.shape != (3,) or not bool(np.isfinite(spacing).all()) or bool(np.any(spacing <= 0)):
        raise InvalidSpatialGridError("affine-derived spacing must be positive and finite.")
    return cast(tuple[float, float, float], tuple(float(item) for item in spacing.tolist()))


def validate_image_mask_alignment(
    image_grid: SpatialGrid,
    mask_grid: SpatialGrid,
    *,
    affine_tolerance: float = 1e-5,
    spacing_tolerance: float = 1e-5,
) -> None:
    """Require image and mask to share shape, affine, spacing, and orientation."""

    _validate_tolerance(affine_tolerance, field_name="affine_tolerance")
    _validate_tolerance(spacing_tolerance, field_name="spacing_tolerance")
    if image_grid.shape != mask_grid.shape:
        raise GeometryAlignmentError("image and mask shapes differ.")
    if not _allclose_affine(image_grid.affine, mask_grid.affine, tolerance=affine_tolerance):
        raise GeometryAlignmentError("image and mask affines differ.")
    if not _allclose_tuple(image_grid.spacing, mask_grid.spacing, tolerance=spacing_tolerance):
        raise GeometryAlignmentError("image and mask spacing differs.")
    if image_grid.orientation != mask_grid.orientation:
        raise GeometryAlignmentError("image and mask orientation differs.")


def validate_binary_mask(
    mask: Array, *, expected_shape: tuple[int, int, int] | None = None
) -> None:
    """Require a finite real binary mask without repairing values."""

    array = np.asarray(mask)
    if expected_shape is not None and tuple(int(item) for item in array.shape) != expected_shape:
        raise BinaryMaskValidationError("mask shape does not match the expected grid shape.")
    if array.ndim != 3:
        raise BinaryMaskValidationError("mask must be 3D.")
    if np.iscomplexobj(array):
        raise BinaryMaskValidationError("mask must contain real values.")
    try:
        finite = bool(np.isfinite(array).all())
    except TypeError as exc:
        raise BinaryMaskValidationError("mask values cannot be checked for finiteness.") from exc
    if not finite:
        raise BinaryMaskValidationError("mask must contain only finite values.")
    invalid_values = np.unique(array)[~np.isin(np.unique(array), [0, 1, False, True])]
    if invalid_values.size:
        raise BinaryMaskValidationError("mask must remain binary with values from {0, 1}.")


def validate_interpolation_policy(*, image_interpolation: str, mask_interpolation: str) -> None:
    """Validate Phase 7 image and mask interpolation policy names."""

    if image_interpolation not in ALLOWED_IMAGE_INTERPOLATIONS:
        raise InterpolationPolicyError(
            f"image interpolation must be one of: {sorted(ALLOWED_IMAGE_INTERPOLATIONS)!r}."
        )
    if mask_interpolation != MASK_INTERPOLATION_NEAREST_NEIGHBOR:
        raise InterpolationPolicyError("mask interpolation must be nearest_neighbor only.")


def observed_geometry_changes(
    source_grid: SpatialGrid, changed_grid: SpatialGrid
) -> tuple[str, ...]:
    """Return deterministic geometry metadata fields changed between two grids."""

    changes: list[str] = []
    if source_grid.shape != changed_grid.shape:
        changes.append("shape")
        changes.append("field_of_view")
    if not _allclose_affine(source_grid.affine, changed_grid.affine, tolerance=1e-9):
        changes.append("affine")
    if not _allclose_tuple(source_grid.spacing, changed_grid.spacing, tolerance=1e-9):
        changes.append("spacing")
    if source_grid.orientation != changed_grid.orientation:
        changes.append("orientation")
    return tuple(
        change for change in sorted(set(changes)) if change in ALLOWED_GEOMETRY_CHANGE_FIELDS
    )


def build_geometry_change_record(
    *,
    source_grid: SpatialGrid,
    changed_grid: SpatialGrid,
    permitted_changes: tuple[str, ...],
    image_interpolation: str,
    mask_interpolation: str,
    common_grid_required: bool = True,
) -> GeometryChangeRecord:
    """Build a self-hashed record for explicitly permitted geometry changes."""

    return GeometryChangeRecord(
        schema_name=GEOMETRY_CHANGE_RECORD_SCHEMA_NAME,
        schema_version=PHASE7_GEOMETRY_SCHEMA_VERSION,
        source_grid_hash=source_grid.grid_identity_hash,
        changed_grid_hash=changed_grid.grid_identity_hash,
        permitted_changes=tuple(sorted(permitted_changes)),
        observed_changes=observed_geometry_changes(source_grid, changed_grid),
        image_interpolation=image_interpolation,
        mask_interpolation=mask_interpolation,
        common_grid_required=common_grid_required,
        geometry_change_hash=_ZERO_SHA256,
    )


def build_common_grid_restoration_record(
    *,
    source_grid: SpatialGrid,
    changed_grid: SpatialGrid,
    restored_grid: SpatialGrid,
    target_common_grid: SpatialGrid,
    image_interpolation: str,
    mask_interpolation: str,
) -> CommonGridRestorationRecord:
    """Build a self-hashed proof that a result returned to the common grid."""

    validate_image_mask_alignment(restored_grid, target_common_grid)
    return CommonGridRestorationRecord(
        schema_name=COMMON_GRID_RESTORATION_RECORD_SCHEMA_NAME,
        schema_version=PHASE7_GEOMETRY_SCHEMA_VERSION,
        source_grid_hash=source_grid.grid_identity_hash,
        changed_grid_hash=changed_grid.grid_identity_hash,
        target_common_grid_hash=target_common_grid.grid_identity_hash,
        restored_grid_hash=restored_grid.grid_identity_hash,
        image_interpolation=image_interpolation,
        mask_interpolation=mask_interpolation,
        restored_to_common_grid=True,
        restoration_identity_hash=_ZERO_SHA256,
    )


def spatial_grid_to_dict(grid: SpatialGrid) -> dict[str, JsonValue]:
    """Return a canonical JSON-compatible mapping for a spatial grid."""

    return {
        "schema_name": grid.schema_name,
        "schema_version": grid.schema_version,
        "shape": list(grid.shape),
        "affine": [[float(item) for item in row] for row in grid.affine],
        "spacing": [float(item) for item in grid.spacing],
        "orientation": list(grid.orientation),
        "grid_identity_hash": grid.grid_identity_hash,
    }


def geometry_change_record_to_dict(record: GeometryChangeRecord) -> dict[str, JsonValue]:
    """Return a canonical JSON-compatible mapping for a geometry-change record."""

    return {
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "source_grid_hash": record.source_grid_hash,
        "changed_grid_hash": record.changed_grid_hash,
        "permitted_changes": list(record.permitted_changes),
        "observed_changes": list(record.observed_changes),
        "image_interpolation": record.image_interpolation,
        "mask_interpolation": record.mask_interpolation,
        "common_grid_required": record.common_grid_required,
        "geometry_change_hash": record.geometry_change_hash,
    }


def common_grid_restoration_record_to_dict(
    record: CommonGridRestorationRecord,
) -> dict[str, JsonValue]:
    """Return a canonical JSON-compatible mapping for a common-grid restoration record."""

    return {
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "source_grid_hash": record.source_grid_hash,
        "changed_grid_hash": record.changed_grid_hash,
        "target_common_grid_hash": record.target_common_grid_hash,
        "restored_grid_hash": record.restored_grid_hash,
        "image_interpolation": record.image_interpolation,
        "mask_interpolation": record.mask_interpolation,
        "restored_to_common_grid": record.restored_to_common_grid,
        "restoration_identity_hash": record.restoration_identity_hash,
    }


def hash_spatial_grid(grid: SpatialGrid) -> str:
    """Return the canonical identity hash for spatial-grid metadata."""

    return _hash_spatial_grid_payload(grid, identity_hash=_ZERO_SHA256)


def hash_geometry_change_record(record: GeometryChangeRecord) -> str:
    """Return the canonical identity hash for a geometry-change record."""

    return _hash_geometry_change_record_payload(record, identity_hash=_ZERO_SHA256)


def hash_common_grid_restoration_record(record: CommonGridRestorationRecord) -> str:
    """Return the canonical identity hash for a common-grid restoration record."""

    return _hash_common_grid_restoration_record_payload(record, identity_hash=_ZERO_SHA256)


def _hash_spatial_grid_payload(grid: SpatialGrid, *, identity_hash: str) -> str:
    payload = spatial_grid_to_dict(grid)
    payload["grid_identity_hash"] = identity_hash
    return sha256_json(payload)


def _hash_geometry_change_record_payload(
    record: GeometryChangeRecord,
    *,
    identity_hash: str,
) -> str:
    payload = geometry_change_record_to_dict(record)
    payload["geometry_change_hash"] = identity_hash
    return sha256_json(payload)


def _hash_common_grid_restoration_record_payload(
    record: CommonGridRestorationRecord,
    *,
    identity_hash: str,
) -> str:
    payload = common_grid_restoration_record_to_dict(record)
    payload["restoration_identity_hash"] = identity_hash
    return sha256_json(payload)


def _validate_shape(value: tuple[int, int, int], *, field_name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise InvalidSpatialGridError(f"{field_name} must contain three positive integers.")


def _validate_spacing(
    value: tuple[float, float, float],
    *,
    field_name: str,
) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise InvalidSpatialGridError(f"{field_name} must contain three values.")
    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise InvalidSpatialGridError(f"{field_name} values must be numeric.")
        numeric = float(item)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise InvalidSpatialGridError(f"{field_name} values must be positive and finite.")
        normalized.append(numeric)
    return cast(tuple[float, float, float], tuple(normalized))


def _validate_orientation(value: tuple[str, str, str], *, field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise InvalidSpatialGridError(f"{field_name} must contain three axis codes.")
    families: list[str] = []
    for code in value:
        if not isinstance(code, str) or code not in _AXIS_FAMILIES:
            raise InvalidSpatialGridError(f"{field_name} contains an invalid axis code.")
        families.append(_AXIS_FAMILIES[code])
    if len(set(families)) != 3:
        raise InvalidSpatialGridError(f"{field_name} must contain one axis from each family.")


def _validate_tolerance(value: float, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSpatialGridError(f"{field_name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise InvalidSpatialGridError(f"{field_name} must be finite and nonnegative.")


def _validate_change_names(value: tuple[str, ...], *, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise GeometryChangeValidationError(f"{field_name} must be a tuple.")
    if tuple(sorted(set(value))) != value:
        raise GeometryChangeValidationError(f"{field_name} must be sorted and unique.")
    invalid = [item for item in value if item not in ALLOWED_GEOMETRY_CHANGE_FIELDS]
    if invalid:
        raise GeometryChangeValidationError(f"{field_name} contains invalid changes: {invalid!r}.")


def _allclose_affine(first: AffineTuple, second: AffineTuple, *, tolerance: float) -> bool:
    return bool(
        np.allclose(
            np.asarray(first, dtype=np.float64),
            np.asarray(second, dtype=np.float64),
            atol=tolerance,
            rtol=0.0,
            equal_nan=False,
        )
    )


def _allclose_tuple(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    *,
    tolerance: float,
) -> bool:
    return bool(
        np.allclose(
            np.asarray(first, dtype=np.float64),
            np.asarray(second, dtype=np.float64),
            atol=tolerance,
            rtol=0.0,
            equal_nan=False,
        )
    )


def _require_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise GeometryChangeValidationError(f"{field_name} must be a SHA-256 hex digest.")
    if any(character not in "0123456789abcdef" for character in value):
        raise GeometryChangeValidationError(f"{field_name} must be lowercase SHA-256 hex.")


__all__ = [
    "ALLOWED_GEOMETRY_CHANGE_FIELDS",
    "ALLOWED_IMAGE_INTERPOLATIONS",
    "COMMON_GRID_RESTORATION_RECORD_SCHEMA_NAME",
    "GEOMETRY_CHANGE_RECORD_SCHEMA_NAME",
    "MASK_INTERPOLATION_NEAREST_NEIGHBOR",
    "PHASE7_GEOMETRY_SCHEMA_VERSION",
    "SPATIAL_GRID_SCHEMA_NAME",
    "BinaryMaskValidationError",
    "CommonGridRestorationError",
    "CommonGridRestorationRecord",
    "GeometryAlignmentError",
    "GeometryChangeRecord",
    "GeometryChangeValidationError",
    "InterpolationPolicyError",
    "InvalidSpatialGridError",
    "Phase7GeometryError",
    "SpatialGrid",
    "build_common_grid_restoration_record",
    "build_geometry_change_record",
    "build_spatial_grid",
    "common_grid_restoration_record_to_dict",
    "geometry_change_record_to_dict",
    "hash_common_grid_restoration_record",
    "hash_geometry_change_record",
    "hash_spatial_grid",
    "observed_geometry_changes",
    "spacing_from_affine",
    "spatial_grid_to_dict",
    "validate_affine",
    "validate_binary_mask",
    "validate_image_mask_alignment",
    "validate_interpolation_policy",
]
