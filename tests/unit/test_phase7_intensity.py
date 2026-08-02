from __future__ import annotations

import inspect

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.robustness.artifacts import (
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
    Phase7CorruptionSpecification,
    phase7_corruption_specification_identity_payload,
    phase7_transform_result_to_json,
)
from protoem_ct.robustness.geometry import (
    BinaryMaskValidationError,
    SpatialGrid,
    build_spatial_grid,
)
from protoem_ct.robustness.intensity import (
    INTENSITY_SEVERITY_PARAMETERS,
    InvalidIntensitySpecificationError,
    NonFiniteIntensityInputError,
    NonFiniteIntensityOutputError,
    apply_intensity_transform,
    array_content_sha256,
    build_intensity_corruption_specification,
    intensity_severity_parameters,
    query_label_not_part_of_intensity_transform_api,
)


def _image() -> np.ndarray:
    return np.asarray(
        [
            [[-300.0, -225.0], [0.0, 200.0]],
            [[50.0, 100.0], [-50.0, 25.0]],
        ],
        dtype=np.float64,
    )


def _mask() -> np.ndarray:
    return np.asarray([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=np.uint8)


def _grid() -> SpatialGrid:
    return build_spatial_grid(
        shape=(2, 2, 2),
        affine=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        orientation=("R", "A", "S"),
    )


def _artifact_spec(
    *,
    corruption_name: str,
    severity: str,
    parameters: dict[str, JsonValue],
    changes_geometry: bool = False,
) -> Phase7CorruptionSpecification:
    payload: dict[str, JsonValue] = {
        "schema_name": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
        "schema_version": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
        "corruption_name": corruption_name,
        "severity": severity,
        "parameters": parameters,
        "deterministic_seed": None,
        "changes_geometry": changes_geometry,
        "image_interpolation": "linear",
        "mask_interpolation": "nearest_neighbor",
        "common_grid_restoration_required": changes_geometry,
    }
    return Phase7CorruptionSpecification(
        schema_name=PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
        schema_version=PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
        corruption_specification_hash=sha256_json(payload),
        corruption_name=corruption_name,
        severity=severity,
        parameters=parameters,
        deterministic_seed=None,
        changes_geometry=changes_geometry,
        image_interpolation="linear",
        mask_interpolation="nearest_neighbor",
        common_grid_restoration_required=changes_geometry,
    )


def test_severity_parameter_mapping_is_exact_for_all_intensity_transforms() -> None:
    assert set(INTENSITY_SEVERITY_PARAMETERS) == {
        "contrast_shift",
        "hu_window_shift",
        "intensity_offset",
        "intensity_scale",
    }
    assert intensity_severity_parameters("hu_window_shift", "none") == {
        "window_center_shift_hu": 0.0,
        "window_min_hu": -200.0,
        "window_max_hu": 250.0,
    }
    assert intensity_severity_parameters("hu_window_shift", "low") == {
        "window_center_shift_hu": 25.0,
        "window_min_hu": -200.0,
        "window_max_hu": 250.0,
    }
    assert intensity_severity_parameters("hu_window_shift", "medium") == {
        "window_center_shift_hu": 50.0,
        "window_min_hu": -200.0,
        "window_max_hu": 250.0,
    }
    assert intensity_severity_parameters("hu_window_shift", "high") == {
        "window_center_shift_hu": 100.0,
        "window_min_hu": -200.0,
        "window_max_hu": 250.0,
    }
    assert intensity_severity_parameters("intensity_scale", "none") == {"scale_factor": 1.0}
    assert intensity_severity_parameters("intensity_scale", "low") == {"scale_factor": 0.95}
    assert intensity_severity_parameters("intensity_scale", "medium") == {"scale_factor": 1.1}
    assert intensity_severity_parameters("intensity_scale", "high") == {"scale_factor": 1.25}
    assert intensity_severity_parameters("intensity_offset", "none") == {"offset_hu": 0.0}
    assert intensity_severity_parameters("intensity_offset", "low") == {"offset_hu": 10.0}
    assert intensity_severity_parameters("intensity_offset", "medium") == {"offset_hu": 25.0}
    assert intensity_severity_parameters("intensity_offset", "high") == {"offset_hu": 50.0}
    assert intensity_severity_parameters("contrast_shift", "none") == {
        "contrast_center_hu": 0.0,
        "contrast_factor": 1.0,
    }
    assert intensity_severity_parameters("contrast_shift", "low") == {
        "contrast_center_hu": 0.0,
        "contrast_factor": 0.9,
    }
    assert intensity_severity_parameters("contrast_shift", "medium") == {
        "contrast_center_hu": 0.0,
        "contrast_factor": 1.1,
    }
    assert intensity_severity_parameters("contrast_shift", "high") == {
        "contrast_center_hu": 0.0,
        "contrast_factor": 1.25,
    }


@pytest.mark.parametrize(
    ("corruption_name", "severity", "expected"),
    [
        (
            "hu_window_shift",
            "low",
            np.asarray(
                [[[-200.0, -200.0], [25.0, 225.0]], [[75.0, 125.0], [-25.0, 50.0]]],
                dtype=np.float64,
            ),
        ),
        ("intensity_scale", "high", _image() * 1.25),
        ("intensity_offset", "medium", _image() + 25.0),
        ("contrast_shift", "medium", 0.0 + 1.1 * (_image() - 0.0)),
    ],
)
def test_intensity_transforms_modify_image_and_leave_mask_unchanged(
    corruption_name: str,
    severity: str,
    expected: np.ndarray,
) -> None:
    image = _image()
    mask = _mask()
    output = apply_intensity_transform(
        image,
        build_intensity_corruption_specification(
            corruption_name=corruption_name,
            severity=severity,
        ),
        mask=mask,
        spatial_grid=_grid(),
    )

    np.testing.assert_allclose(output.image, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(output.mask, mask)
    assert output.geometry_record is not None
    assert output.geometry_record.geometry_change_type == "none"
    assert output.geometry_record.common_grid_restored is False
    assert output.transform_result.common_grid_restored is False
    assert output.transform_result.mask_binary_preserved is True
    assert output.transform_result.output_mask_content_hash == array_content_sha256(mask)
    assert (
        output.transform_result.geometry_record_hash == output.geometry_record.geometry_record_hash
    )


def test_none_severity_is_identity_with_valid_transform_record() -> None:
    image = _image()
    output = apply_intensity_transform(
        image,
        build_intensity_corruption_specification(
            corruption_name="intensity_offset",
            severity="none",
        ),
    )

    np.testing.assert_array_equal(output.image, image)
    assert output.geometry_record is None
    assert output.transform_result.execution_status == "completed"
    assert output.transform_result.output_image_content_hash == array_content_sha256(output.image)


def test_nonfinite_input_and_output_are_rejected() -> None:
    image = _image()
    image[0, 0, 0] = np.nan
    spec = build_intensity_corruption_specification(
        corruption_name="intensity_scale",
        severity="high",
    )

    with pytest.raises(NonFiniteIntensityInputError, match="finite"):
        apply_intensity_transform(image, spec)

    huge = np.full((2, 2, 2), np.finfo(np.float64).max, dtype=np.float64)
    with pytest.raises(NonFiniteIntensityOutputError, match="finite"):
        apply_intensity_transform(huge, spec)


def test_binary_mask_validation_rejects_nonbinary_masks() -> None:
    spec = build_intensity_corruption_specification(
        corruption_name="intensity_offset",
        severity="low",
    )

    with pytest.raises(BinaryMaskValidationError, match="binary"):
        apply_intensity_transform(_image(), spec, mask=np.full((2, 2, 2), 0.5))


def test_specification_and_result_hashes_are_deterministic() -> None:
    spec_a = build_intensity_corruption_specification(
        corruption_name="contrast_shift",
        severity="high",
    )
    spec_b = build_intensity_corruption_specification(
        corruption_name="contrast_shift",
        severity="high",
    )
    first = apply_intensity_transform(_image()[:, ::-1, :], spec_a, mask=_mask()[:, ::-1, :])
    second = apply_intensity_transform(_image()[:, ::-1, :], spec_b, mask=_mask()[:, ::-1, :])

    assert spec_a.corruption_specification_hash == spec_b.corruption_specification_hash
    assert spec_a.corruption_specification_hash == sha256_json(
        phase7_corruption_specification_identity_payload(spec_a)
    )
    np.testing.assert_array_equal(first.image, second.image)
    assert (
        first.transform_result.transform_result_hash
        == second.transform_result.transform_result_hash
    )
    assert phase7_transform_result_to_json(
        first.transform_result
    ) == phase7_transform_result_to_json(second.transform_result)


def test_same_logical_non_contiguous_input_produces_same_content_hashes() -> None:
    spec = build_intensity_corruption_specification(
        corruption_name="intensity_scale",
        severity="low",
    )
    contiguous = _image()
    non_contiguous = np.asfortranarray(_image())

    first = apply_intensity_transform(contiguous, spec, mask=_mask())
    second = apply_intensity_transform(non_contiguous, spec, mask=np.asfortranarray(_mask()))

    np.testing.assert_array_equal(first.image, second.image)
    assert first.transform_result.input_image_content_hash == (
        second.transform_result.input_image_content_hash
    )
    assert (
        first.transform_result.transform_result_hash
        == second.transform_result.transform_result_hash
    )


def test_invalid_corruption_name_and_malformed_specification_are_rejected() -> None:
    with pytest.raises(InvalidIntensitySpecificationError, match="unsupported"):
        build_intensity_corruption_specification(
            corruption_name="gaussian_noise",
            severity="low",
        )

    artifact_spec = _artifact_spec(
        corruption_name="gaussian_noise",
        severity="low",
        parameters={"sigma_hu": 5.0},
    )
    with pytest.raises(InvalidIntensitySpecificationError, match="unsupported"):
        apply_intensity_transform(_image(), artifact_spec)

    wrong_parameters = _artifact_spec(
        corruption_name="intensity_offset",
        severity="low",
        parameters={"offset_hu": 11.0},
    )
    with pytest.raises(InvalidIntensitySpecificationError, match="severity table"):
        apply_intensity_transform(_image(), wrong_parameters)


def test_query_labels_and_reference_masks_are_absent_from_public_api() -> None:
    forbidden = {
        "query_label",
        "query_labels",
        "label_map",
        "reference_mask",
        "query_reference_mask",
    }

    assert query_label_not_part_of_intensity_transform_api() is True
    assert forbidden.isdisjoint(inspect.signature(apply_intensity_transform).parameters)
    assert forbidden.isdisjoint(
        inspect.signature(build_intensity_corruption_specification).parameters
    )
