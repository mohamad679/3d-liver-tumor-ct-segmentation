from __future__ import annotations

import inspect

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import canonical_json_bytes, sha256_json
from protoem_ct.robustness.artifacts import (
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
    Phase7CorruptionSpecification,
    hash_phase7_geometry_record,
    hash_phase7_transform_result,
    phase7_corruption_specification_from_json,
)
from protoem_ct.robustness.geometry import SpatialGrid, build_spatial_grid
from protoem_ct.robustness.resampling import (
    InvalidResamplingInputError,
    InvalidResamplingSpecificationError,
    apply_anisotropic_downsampling,
    apply_slice_thickness_simulation,
    hash_resampling_result,
    query_label_not_part_of_resampling_api,
)


def _affine(
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (spacing[0], 0.0, 0.0, 0.0),
        (0.0, spacing[1], 0.0, 0.0),
        (0.0, 0.0, spacing[2], 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _grid(
    *,
    shape: tuple[int, int, int] = (4, 3, 2),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> SpatialGrid:
    return build_spatial_grid(
        shape=shape,
        affine=_affine(spacing=spacing),
        orientation=("R", "A", "S"),
    )


def _spec(
    *,
    corruption_name: str = "slice_thickness",
    parameters: dict[str, object] | None = None,
    image_interpolation: str = "linear",
    mask_interpolation: str = "nearest_neighbor",
) -> Phase7CorruptionSpecification:
    payload: dict[str, object] = {
        "schema_name": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
        "schema_version": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
        "corruption_name": corruption_name,
        "severity": "medium",
        "parameters": (
            {"simulated_spacing": [2.0, 1.0, 1.0]} if parameters is None else parameters
        ),
        "deterministic_seed": None,
        "changes_geometry": True,
        "image_interpolation": image_interpolation,
        "mask_interpolation": mask_interpolation,
        "common_grid_restoration_required": True,
    }
    return phase7_corruption_specification_from_json(
        canonical_json_bytes({"corruption_specification_hash": sha256_json(payload), **payload})
    )


def _image() -> np.ndarray:
    return np.arange(24, dtype=np.float64).reshape(4, 3, 2)


def _mask() -> np.ndarray:
    return (np.arange(24).reshape(4, 3, 2) % 2).astype(np.uint8)


def test_slice_thickness_spacing_and_shape_change() -> None:
    grid = _grid()

    result = apply_slice_thickness_simulation(
        image=_image(),
        image_grid=grid,
        mask=_mask(),
        mask_grid=grid,
        specification=_spec(parameters={"simulated_spacing": [2.0, 1.0, 1.0]}),
    )

    assert result.changed_grid.shape == (2, 3, 2)
    assert result.changed_grid.spacing == (2.0, 1.0, 1.0)
    assert result.restored_grid.shape == grid.shape
    assert result.restored_grid.grid_identity_hash == grid.grid_identity_hash
    assert result.geometry_record.transform_name == "slice_thickness"
    assert result.geometry_record.output_shape == (2, 3, 2)
    assert result.geometry_record.output_spacing == (2.0, 1.0, 1.0)
    assert result.geometry_record.restored_shape == grid.shape
    assert result.transform_result.common_grid_restored is True


def test_anisotropic_downsampling_spacing_and_shape_change() -> None:
    grid = _grid(shape=(6, 4, 2))
    spec = _spec(
        corruption_name="anisotropic_downsampling",
        parameters={"downsampling_factors": [3.0, 2.0, 1.0]},
    )

    result = apply_anisotropic_downsampling(
        image=np.arange(48, dtype=np.float64).reshape(6, 4, 2),
        image_grid=grid,
        mask=(np.arange(48).reshape(6, 4, 2) % 2).astype(np.uint8),
        mask_grid=grid,
        specification=spec,
    )

    assert result.changed_grid.shape == (2, 2, 2)
    assert result.changed_grid.spacing == (3.0, 2.0, 1.0)
    assert result.geometry_record.transform_name == "anisotropic_downsampling"
    assert result.geometry_record.geometry_change_type == "resampling"


def test_image_interpolation_is_deterministic_on_known_small_array() -> None:
    grid = _grid(shape=(4, 1, 1))
    image = np.arange(4, dtype=np.float64).reshape(4, 1, 1)

    result = apply_slice_thickness_simulation(
        image=image,
        image_grid=grid,
        specification=_spec(parameters={"simulated_spacing": [2.0, 1.0, 1.0]}),
    )

    np.testing.assert_allclose(result.changed_image[:, 0, 0], np.asarray([0.0, 3.0]))
    second = apply_slice_thickness_simulation(
        image=image,
        image_grid=grid,
        specification=_spec(parameters={"simulated_spacing": [2.0, 1.0, 1.0]}),
    )
    np.testing.assert_array_equal(result.changed_image, second.changed_image)
    assert result.resampling_identity_hash == second.resampling_identity_hash


def test_mask_uses_nearest_neighbor_and_remains_binary() -> None:
    grid = _grid(shape=(4, 1, 1))
    mask = np.asarray([0, 1, 1, 0], dtype=np.uint8).reshape(4, 1, 1)

    result = apply_slice_thickness_simulation(
        image=np.arange(4, dtype=np.float64).reshape(4, 1, 1),
        image_grid=grid,
        mask=mask,
        mask_grid=grid,
        specification=_spec(parameters={"simulated_spacing": [2.0, 1.0, 1.0]}),
    )

    assert result.changed_mask is not None
    assert result.restored_mask is not None
    assert set(np.unique(result.changed_mask).tolist()).issubset({0, 1})
    assert set(np.unique(result.restored_mask).tolist()).issubset({0, 1})

    with pytest.raises(ValueError, match="nearest_neighbor"):
        _spec(mask_interpolation="linear")


def test_common_grid_restoration_returns_target_grid_and_validates_alignment() -> None:
    grid = _grid()
    target = _grid(shape=(2, 3, 2), spacing=(2.0, 1.0, 1.0))

    result = apply_slice_thickness_simulation(
        image=_image(),
        image_grid=grid,
        mask=_mask(),
        mask_grid=grid,
        target_common_grid=target,
        specification=_spec(parameters={"simulated_spacing": [2.0, 1.0, 1.0]}),
    )

    assert result.restored_grid.grid_identity_hash == target.grid_identity_hash
    assert result.restored_image.shape == target.shape
    assert result.restored_mask is not None
    assert result.restored_mask.shape == target.shape


def test_geometry_and_transform_records_are_self_hash_valid() -> None:
    grid = _grid()

    result = apply_slice_thickness_simulation(
        image=_image(),
        image_grid=grid,
        mask=_mask(),
        mask_grid=grid,
        specification=_spec(),
    )

    assert result.geometry_record.geometry_record_hash == hash_phase7_geometry_record(
        result.geometry_record
    )
    assert result.transform_result.transform_result_hash == hash_phase7_transform_result(
        result.transform_result
    )
    assert result.resampling_identity_hash == hash_resampling_result(result)


def test_non_finite_input_rejected() -> None:
    image = _image()
    image[0, 0, 0] = np.nan

    with pytest.raises(InvalidResamplingInputError, match="finite"):
        apply_slice_thickness_simulation(
            image=image,
            image_grid=_grid(),
            specification=_spec(),
        )


def test_malformed_interpolation_and_spec_rejected() -> None:
    with pytest.raises(InvalidResamplingSpecificationError, match="only axis 0"):
        apply_slice_thickness_simulation(
            image=_image(),
            image_grid=_grid(),
            specification=_spec(parameters={"simulated_spacing": [2.0, 2.0, 1.0]}),
        )

    with pytest.raises(InvalidResamplingSpecificationError, match="greater than or equal"):
        apply_anisotropic_downsampling(
            image=_image(),
            image_grid=_grid(),
            specification=_spec(
                corruption_name="anisotropic_downsampling",
                parameters={"downsampling_factors": [1.0, 0.5, 1.0]},
            ),
        )

    with pytest.raises(ValueError, match="image interpolation"):
        apply_slice_thickness_simulation(
            image=_image(),
            image_grid=_grid(),
            specification=_spec(image_interpolation="cubic"),
        )


def test_inputs_remain_unchanged() -> None:
    image = _image()
    mask = _mask()
    image_before = image.copy()
    mask_before = mask.copy()
    grid = _grid()

    apply_slice_thickness_simulation(
        image=image,
        image_grid=grid,
        mask=mask,
        mask_grid=grid,
        specification=_spec(),
    )

    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_non_contiguous_input_has_same_output_and_identity() -> None:
    base = np.arange(48, dtype=np.float64).reshape(4, 6, 2)
    mask_base = (np.arange(48).reshape(4, 6, 2) % 2).astype(np.uint8)
    image = base[:, ::2, :]
    mask = mask_base[:, ::2, :]
    contiguous_image = np.ascontiguousarray(image)
    contiguous_mask = np.ascontiguousarray(mask)
    grid = _grid(shape=image.shape)

    first = apply_slice_thickness_simulation(
        image=image,
        image_grid=grid,
        mask=mask,
        mask_grid=grid,
        specification=_spec(),
    )
    second = apply_slice_thickness_simulation(
        image=contiguous_image,
        image_grid=grid,
        mask=contiguous_mask,
        mask_grid=grid,
        specification=_spec(),
    )

    np.testing.assert_array_equal(first.restored_image, second.restored_image)
    np.testing.assert_array_equal(first.restored_mask, second.restored_mask)
    assert first.resampling_identity_hash == second.resampling_identity_hash


def test_query_label_and_reference_mask_absent_from_public_api() -> None:
    banned = {
        "query_label",
        "query_labels",
        "label",
        "labels",
        "reference_mask",
        "query_reference_mask",
    }

    for function in (
        apply_slice_thickness_simulation,
        apply_anisotropic_downsampling,
    ):
        assert banned.isdisjoint(inspect.signature(function).parameters)
    assert query_label_not_part_of_resampling_api() is True
