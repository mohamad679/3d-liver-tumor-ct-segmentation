from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes
from protoem_ct.robustness.geometry import (
    COMMON_GRID_RESTORATION_RECORD_SCHEMA_NAME,
    PHASE7_GEOMETRY_SCHEMA_VERSION,
    SPATIAL_GRID_SCHEMA_NAME,
    BinaryMaskValidationError,
    CommonGridRestorationError,
    GeometryAlignmentError,
    GeometryChangeValidationError,
    InterpolationPolicyError,
    InvalidSpatialGridError,
    SpatialGrid,
    build_common_grid_restoration_record,
    build_geometry_change_record,
    build_spatial_grid,
    common_grid_restoration_record_to_dict,
    hash_spatial_grid,
    spatial_grid_to_dict,
    validate_affine,
    validate_binary_mask,
    validate_image_mask_alignment,
    validate_interpolation_policy,
)


def _affine(
    *,
    spacing: tuple[float, float, float] = (1.0, 1.5, 2.0),
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (spacing[0], 0.0, 0.0, translation[0]),
        (0.0, spacing[1], 0.0, translation[1]),
        (0.0, 0.0, spacing[2], translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _grid(
    *,
    shape: tuple[int, int, int] = (4, 5, 6),
    affine: tuple[tuple[float, float, float, float], ...] | None = None,
    orientation: tuple[str, str, str] = ("R", "A", "S"),
) -> SpatialGrid:
    return build_spatial_grid(
        shape=shape,
        affine=_affine() if affine is None else affine,
        orientation=orientation,
    )


def test_valid_image_mask_alignment() -> None:
    image_grid = _grid()
    mask_grid = _grid()

    validate_image_mask_alignment(image_grid, mask_grid)

    assert image_grid.grid_identity_hash == mask_grid.grid_identity_hash
    assert hash_spatial_grid(image_grid) == image_grid.grid_identity_hash


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(GeometryAlignmentError, match="shapes differ"):
        validate_image_mask_alignment(_grid(), _grid(shape=(4, 5, 7)))


def test_affine_mismatch_rejected() -> None:
    with pytest.raises(GeometryAlignmentError, match="affines differ"):
        validate_image_mask_alignment(
            _grid(),
            _grid(affine=_affine(translation=(0.01, 0.0, 0.0))),
            affine_tolerance=1e-5,
        )


def test_spacing_mismatch_rejected() -> None:
    with pytest.raises(GeometryAlignmentError):
        validate_image_mask_alignment(
            _grid(),
            _grid(affine=_affine(spacing=(1.0, 1.5, 2.1))),
            spacing_tolerance=1e-5,
        )


def test_orientation_mismatch_rejected() -> None:
    with pytest.raises(GeometryAlignmentError, match="orientation differs"):
        validate_image_mask_alignment(_grid(), _grid(orientation=("L", "A", "S")))


def test_invalid_and_nonfinite_affine_rejected() -> None:
    with pytest.raises(InvalidSpatialGridError, match="4x4"):
        validate_affine(np.eye(3))

    invalid = np.eye(4)
    invalid[0, 0] = np.nan
    with pytest.raises(InvalidSpatialGridError, match="finite"):
        validate_affine(invalid)

    singular = np.eye(4)
    singular[2, 2] = 0.0
    with pytest.raises(InvalidSpatialGridError, match="nonsingular"):
        validate_affine(singular)


def test_nonbinary_mask_rejection_and_binary_preservation_check() -> None:
    validate_binary_mask(np.asarray([[[0, 1], [1, 0]]], dtype=np.uint8))

    with pytest.raises(BinaryMaskValidationError, match="binary"):
        validate_binary_mask(np.asarray([[[0.0, 0.5], [1.0, 0.0]]], dtype=np.float64))

    with pytest.raises(BinaryMaskValidationError, match="finite"):
        validate_binary_mask(np.asarray([[[0.0, np.inf]]], dtype=np.float64))


def test_mask_interpolation_policy_rejection() -> None:
    validate_interpolation_policy(
        image_interpolation="linear",
        mask_interpolation="nearest_neighbor",
    )

    with pytest.raises(InterpolationPolicyError, match="mask interpolation"):
        validate_interpolation_policy(
            image_interpolation="linear",
            mask_interpolation="linear",
        )

    with pytest.raises(InterpolationPolicyError, match="image interpolation"):
        validate_interpolation_policy(
            image_interpolation="cubic",
            mask_interpolation="nearest_neighbor",
        )


def test_common_grid_restoration_validation() -> None:
    source = _grid()
    changed = _grid(shape=(4, 5, 5))
    record = build_common_grid_restoration_record(
        source_grid=source,
        changed_grid=changed,
        restored_grid=source,
        target_common_grid=source,
        image_interpolation="linear",
        mask_interpolation="nearest_neighbor",
    )

    assert record.restored_grid_hash == source.grid_identity_hash
    assert record.restored_to_common_grid is True
    assert common_grid_restoration_record_to_dict(record)["schema_name"] == (
        COMMON_GRID_RESTORATION_RECORD_SCHEMA_NAME
    )

    with pytest.raises(CommonGridRestorationError, match="restored_grid_hash"):
        type(record)(
            schema_name=COMMON_GRID_RESTORATION_RECORD_SCHEMA_NAME,
            schema_version=PHASE7_GEOMETRY_SCHEMA_VERSION,
            source_grid_hash=source.grid_identity_hash,
            changed_grid_hash=changed.grid_identity_hash,
            target_common_grid_hash=source.grid_identity_hash,
            restored_grid_hash=changed.grid_identity_hash,
            image_interpolation="linear",
            mask_interpolation="nearest_neighbor",
            restored_to_common_grid=True,
            restoration_identity_hash="0" * 64,
        )


def test_explicit_permitted_geometry_changes_are_required() -> None:
    source = _grid()
    changed = _grid(shape=(4, 5, 5))

    record = build_geometry_change_record(
        source_grid=source,
        changed_grid=changed,
        permitted_changes=("field_of_view", "shape"),
        image_interpolation="linear",
        mask_interpolation="nearest_neighbor",
    )

    assert record.observed_changes == ("field_of_view", "shape")

    with pytest.raises(GeometryChangeValidationError, match="explicitly permitted"):
        build_geometry_change_record(
            source_grid=source,
            changed_grid=changed,
            permitted_changes=("affine",),
            image_interpolation="linear",
            mask_interpolation="nearest_neighbor",
        )


def test_spatial_grid_hash_and_canonical_mapping_are_deterministic() -> None:
    first = _grid()
    second = _grid()
    different = _grid(shape=(4, 5, 7))

    assert first.grid_identity_hash == second.grid_identity_hash
    assert first.grid_identity_hash != different.grid_identity_hash
    assert canonical_json_bytes(spatial_grid_to_dict(first)) == canonical_json_bytes(
        spatial_grid_to_dict(second)
    )


def test_spatial_grid_self_hash_mismatch_rejected() -> None:
    grid = _grid()
    with pytest.raises(InvalidSpatialGridError, match="hash mismatch"):
        SpatialGrid(
            schema_name=SPATIAL_GRID_SCHEMA_NAME,
            schema_version=PHASE7_GEOMETRY_SCHEMA_VERSION,
            shape=grid.shape,
            affine=grid.affine,
            spacing=grid.spacing,
            orientation=grid.orientation,
            grid_identity_hash="1" * 64,
        )


def test_non_contiguous_mask_shape_validation_does_not_mutate_input() -> None:
    mask = np.asarray(np.arange(27).reshape(3, 3, 3) % 2, dtype=np.uint8)[:, ::-1, :]
    before = mask.copy()

    validate_binary_mask(mask, expected_shape=(3, 3, 3))

    np.testing.assert_array_equal(mask, before)


def test_spacing_and_orientation_metadata_validation() -> None:
    grid = _grid()

    with pytest.raises(InvalidSpatialGridError, match="spacing"):
        SpatialGrid(
            schema_name=SPATIAL_GRID_SCHEMA_NAME,
            schema_version=PHASE7_GEOMETRY_SCHEMA_VERSION,
            shape=grid.shape,
            affine=grid.affine,
            spacing=(1.0, 1.5, 9.0),
            orientation=grid.orientation,
            grid_identity_hash="0" * 64,
        )

    with pytest.raises(InvalidSpatialGridError, match="orientation"):
        SpatialGrid(
            schema_name=SPATIAL_GRID_SCHEMA_NAME,
            schema_version=PHASE7_GEOMETRY_SCHEMA_VERSION,
            shape=grid.shape,
            affine=grid.affine,
            spacing=grid.spacing,
            orientation=("R", "L", "S"),
            grid_identity_hash="0" * 64,
        )


def test_record_hash_changes_when_geometry_policy_changes() -> None:
    source = _grid()
    changed = _grid(shape=(4, 5, 5))
    record = build_geometry_change_record(
        source_grid=source,
        changed_grid=changed,
        permitted_changes=("field_of_view", "shape"),
        image_interpolation="linear",
        mask_interpolation="nearest_neighbor",
    )
    modified = replace(
        record,
        image_interpolation="bspline",
        geometry_change_hash="0" * 64,
    )

    assert modified.geometry_change_hash != record.geometry_change_hash
