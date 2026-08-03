from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.robustness.artifacts import (
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
    Phase7CorruptionSpecification,
    phase7_corruption_specification_from_json,
)
from protoem_ct.robustness.crop import (
    CropFovInputError,
    InvalidCropSpecificationError,
    apply_crop_fov_perturbation,
    crop_bounds_from_specification,
    crop_fov_public_api_excludes_query_labels,
    hash_crop_fov_metadata,
    hash_crop_fov_result,
)
from protoem_ct.robustness.geometry import SpatialGrid, build_spatial_grid


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


def _grid(shape: tuple[int, int, int] = (4, 5, 6)) -> SpatialGrid:
    return build_spatial_grid(shape=shape, affine=_affine(), orientation=("R", "A", "S"))


def _spec(**overrides: object) -> Phase7CorruptionSpecification:
    payload: dict[str, object] = {
        "schema_name": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
        "schema_version": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
        "corruption_name": "crop_fov",
        "severity": "medium",
        "parameters": {"crop_margins": [[1, 1], [1, 0], [2, 1]]},
        "deterministic_seed": None,
        "changes_geometry": True,
        "image_interpolation": "linear",
        "mask_interpolation": "nearest_neighbor",
        "common_grid_restoration_required": True,
    }
    payload.update(overrides)
    return phase7_corruption_specification_from_json(
        canonical_json_bytes({"corruption_specification_hash": sha256_json(payload), **payload})
    )


def test_exact_crop_box_and_padding_from_margins() -> None:
    image = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    mask = np.zeros((4, 5, 6), dtype=np.uint8)
    mask[1:3, 1:5, 2:5] = 1
    grid = _grid()

    result = apply_crop_fov_perturbation(
        image=image,
        image_grid=grid,
        evaluation_mask=mask,
        mask_grid=grid,
        corruption_specification=_spec(),
    )

    assert crop_bounds_from_specification(
        corruption_specification=_spec(), input_shape=grid.shape
    ) == ((1, 3), (1, 5), (2, 5))
    assert result.metadata.crop_start == (1, 1, 2)
    assert result.metadata.crop_stop == (3, 5, 5)
    assert result.metadata.pad_before == (1, 1, 2)
    assert result.metadata.pad_after == (1, 0, 1)
    assert result.cropped_image.shape == (2, 4, 3)
    np.testing.assert_array_equal(result.cropped_image, image[1:3, 1:5, 2:5])


def test_restored_output_shape_and_grid_match_common_grid() -> None:
    image = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    grid = _grid()

    result = apply_crop_fov_perturbation(
        image=image,
        image_grid=grid,
        corruption_specification=_spec(),
    )

    assert result.restored_image.shape == image.shape
    assert result.target_common_grid.grid_identity_hash == grid.grid_identity_hash
    assert result.geometry_record.input_shape == grid.shape
    assert result.geometry_record.output_shape == (2, 4, 3)
    assert result.geometry_record.restored_shape == grid.shape
    assert result.geometry_record.common_grid_restored is True
    assert result.transform_result.common_grid_restored is True


def test_mask_remains_binary_and_aligned_after_restoration() -> None:
    image = np.ones((4, 5, 6), dtype=np.float64)
    mask = np.zeros((4, 5, 6), dtype=np.uint8)
    mask[1:3, 1:5, 2:5] = 1
    grid = _grid()

    result = apply_crop_fov_perturbation(
        image=image,
        image_grid=grid,
        evaluation_mask=mask,
        mask_grid=grid,
        corruption_specification=_spec(),
    )

    assert result.restored_mask is not None
    assert set(np.unique(result.restored_mask).tolist()) <= {0, 1}
    assert result.restored_mask.shape == image.shape
    assert result.transform_result.mask_binary_preserved is True
    np.testing.assert_array_equal(result.restored_mask[1:3, 1:5, 2:5], 1)


def test_empty_lesion_and_lesion_removed_metadata_are_explicit() -> None:
    image = np.ones((4, 5, 6), dtype=np.float64)
    grid = _grid()
    outside_crop_mask = np.zeros((4, 5, 6), dtype=np.uint8)
    outside_crop_mask[0, 0, 0] = 1
    empty_mask = np.zeros((4, 5, 6), dtype=np.uint8)

    removed = apply_crop_fov_perturbation(
        image=image,
        image_grid=grid,
        evaluation_mask=outside_crop_mask,
        mask_grid=grid,
        corruption_specification=_spec(),
    )
    empty = apply_crop_fov_perturbation(
        image=image,
        image_grid=grid,
        evaluation_mask=empty_mask,
        mask_grid=grid,
        corruption_specification=_spec(),
    )

    assert removed.metadata.input_mask_foreground_count == 1
    assert removed.metadata.cropped_mask_foreground_count == 0
    assert removed.metadata.empty_lesion_before is False
    assert removed.metadata.empty_lesion_after_crop is True
    assert removed.metadata.lesion_removed is True
    assert empty.metadata.empty_lesion_before is True
    assert empty.metadata.lesion_removed is False


def test_geometry_records_self_hash_valid() -> None:
    image = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    result = apply_crop_fov_perturbation(
        image=image,
        image_grid=_grid(),
        corruption_specification=_spec(),
    )

    assert result.metadata.metadata_identity_hash == hash_crop_fov_metadata(result.metadata)
    assert result.result_identity_hash == hash_crop_fov_result(result)
    assert (
        result.geometry_record.geometry_record_hash == result.transform_result.geometry_record_hash
    )


def test_invalid_crop_bounds_rejected() -> None:
    grid = _grid()

    with pytest.raises(InvalidCropSpecificationError, match="invalid crop bounds"):
        crop_bounds_from_specification(
            corruption_specification=_spec(
                parameters={"crop_start": [0, 0, 0], "crop_stop": [4, 5, 7]}
            ),
            input_shape=grid.shape,
        )

    with pytest.raises(InvalidCropSpecificationError, match="unsupported"):
        crop_bounds_from_specification(
            corruption_specification=_spec(
                parameters={"crop_margins": [[0, 0], [0, 0], [0, 0]], "extra": 1}
            ),
            input_shape=grid.shape,
        )


def test_non_finite_image_rejected() -> None:
    image = np.ones((4, 5, 6), dtype=np.float64)
    image[0, 0, 0] = np.nan

    with pytest.raises(CropFovInputError, match="finite"):
        apply_crop_fov_perturbation(
            image=image,
            image_grid=_grid(),
            corruption_specification=_spec(),
        )


def test_input_arrays_unchanged() -> None:
    image = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    mask = (image % 2 == 0).astype(np.uint8)
    image_before = image.copy()
    mask_before = mask.copy()

    apply_crop_fov_perturbation(
        image=image,
        image_grid=_grid(),
        evaluation_mask=mask,
        mask_grid=_grid(),
        corruption_specification=_spec(),
    )

    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_repeated_execution_is_deterministic() -> None:
    image = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    grid = _grid()
    spec = _spec()

    first = apply_crop_fov_perturbation(
        image=image,
        image_grid=grid,
        corruption_specification=spec,
    )
    second = apply_crop_fov_perturbation(
        image=image,
        image_grid=grid,
        corruption_specification=spec,
    )

    np.testing.assert_array_equal(first.restored_image, second.restored_image)
    assert first.result_identity_hash == second.result_identity_hash


def test_non_contiguous_input_has_same_identity_and_output() -> None:
    contiguous = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    non_contiguous = np.asfortranarray(contiguous)
    assert not non_contiguous.flags.c_contiguous
    grid = _grid()
    spec = _spec()

    first = apply_crop_fov_perturbation(
        image=contiguous,
        image_grid=grid,
        corruption_specification=spec,
    )
    second = apply_crop_fov_perturbation(
        image=non_contiguous,
        image_grid=grid,
        corruption_specification=spec,
    )

    np.testing.assert_array_equal(first.restored_image, second.restored_image)
    assert first.result_identity_hash == second.result_identity_hash


def test_no_query_label_or_reference_mask_in_public_prediction_api() -> None:
    assert crop_fov_public_api_excludes_query_labels()
    public_names = set(apply_crop_fov_perturbation.__annotations__)
    public_names.update(
        field.name
        for field in fields(
            type(
                apply_crop_fov_perturbation(
                    image=np.ones((4, 5, 6), dtype=np.float64),
                    image_grid=_grid(),
                    corruption_specification=_spec(),
                )
            )
        )
    )

    forbidden = {
        "query_label",
        "query_labels",
        "label_map",
        "reference_mask",
        "query_reference_mask",
    }
    assert public_names.isdisjoint(forbidden)
