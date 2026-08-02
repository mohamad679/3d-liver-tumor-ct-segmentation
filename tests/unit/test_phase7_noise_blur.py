from __future__ import annotations

import inspect

import numpy as np
import pytest

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_json
from protoem_ct.robustness.artifacts import (
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
    PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
    Phase7CorruptionSpecification,
    phase7_corruption_specification_from_json,
    phase7_transform_result_to_json,
)
from protoem_ct.robustness.blur import (
    apply_gaussian_blur,
    gaussian_blur_parameters_for_severity,
    gaussian_kernel_1d,
)
from protoem_ct.robustness.noise import (
    InvalidCorruptionSpecificationError,
    InvalidTransformInputError,
    apply_gaussian_noise,
    gaussian_noise_parameters_for_severity,
)


def _spec(
    *,
    corruption_name: str,
    severity: str = "low",
    parameters: dict[str, JsonValue] | None = None,
    deterministic_seed: int | None = 1729,
) -> Phase7CorruptionSpecification:
    if parameters is None and corruption_name == "gaussian_noise":
        parameters = gaussian_noise_parameters_for_severity(severity)
    if parameters is None and corruption_name == "gaussian_blur":
        parameters = gaussian_blur_parameters_for_severity(severity)
    payload: dict[str, object] = {
        "schema_name": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_NAME,
        "schema_version": PHASE7_CORRUPTION_SPECIFICATION_SCHEMA_VERSION,
        "corruption_name": corruption_name,
        "severity": severity,
        "parameters": parameters,
        "deterministic_seed": deterministic_seed,
        "changes_geometry": False,
        "image_interpolation": "linear",
        "mask_interpolation": "nearest_neighbor",
        "common_grid_restoration_required": False,
    }
    return phase7_corruption_specification_from_json(
        canonical_json_bytes({"corruption_specification_hash": sha256_json(payload), **payload})
    )


def test_gaussian_noise_is_deterministic_with_same_seed() -> None:
    image = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    spec = _spec(corruption_name="gaussian_noise", deterministic_seed=123)

    first = apply_gaussian_noise(image, spec)
    second = apply_gaussian_noise(np.asfortranarray(image), spec)

    np.testing.assert_allclose(first.transformed_image, second.transformed_image)
    assert (
        first.transform_result.output_image_content_hash
        == second.transform_result.output_image_content_hash
    )
    assert (
        first.transform_result.transform_result_hash
        == second.transform_result.transform_result_hash
    )


def test_gaussian_noise_seed_changes_output_and_hash() -> None:
    image = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    first = apply_gaussian_noise(
        image, _spec(corruption_name="gaussian_noise", deterministic_seed=1)
    )
    second = apply_gaussian_noise(
        image,
        _spec(corruption_name="gaussian_noise", deterministic_seed=2),
    )

    assert not np.array_equal(first.transformed_image, second.transformed_image)
    assert (
        first.transform_result.output_image_content_hash
        != second.transform_result.output_image_content_hash
    )


def test_gaussian_noise_does_not_mutate_global_rng_state() -> None:
    image = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    np.random.seed(2027)
    expected_next_values = np.random.random(5)
    np.random.seed(2027)

    apply_gaussian_noise(image, _spec(corruption_name="gaussian_noise", deterministic_seed=99))

    np.testing.assert_array_equal(np.random.random(5), expected_next_values)


def test_gaussian_noise_matches_known_local_rng_values() -> None:
    image = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    spec = _spec(corruption_name="gaussian_noise", deterministic_seed=44)

    result = apply_gaussian_noise(image, spec)
    expected = image + np.random.default_rng(44).normal(
        loc=0.0,
        scale=5.0,
        size=image.shape,
    )

    np.testing.assert_allclose(result.transformed_image, expected)


def test_gaussian_blur_known_small_impulse() -> None:
    image = np.asarray([[[0.0, 1.0, 0.0]]], dtype=np.float64)
    spec = _spec(corruption_name="gaussian_blur", deterministic_seed=None)
    kernel = gaussian_kernel_1d(sigma=0.5, truncate=2.0)

    result = apply_gaussian_blur(image, spec)

    expected = np.asarray([[[kernel[0], kernel[1], kernel[2]]]], dtype=np.float64)
    np.testing.assert_allclose(result.transformed_image, expected, atol=1e-15, rtol=0.0)


def test_severity_parameter_mappings_are_exact_and_persisted() -> None:
    assert gaussian_noise_parameters_for_severity("none") == {"sigma_hu": 0.0}
    assert gaussian_noise_parameters_for_severity("medium") == {"sigma_hu": 15.0}
    assert gaussian_blur_parameters_for_severity("high") == {
        "sigma_voxels": 1.5,
        "truncate": 3.0,
        "boundary_mode": "edge",
    }

    with pytest.raises(InvalidCorruptionSpecificationError):
        gaussian_noise_parameters_for_severity("extreme")
    with pytest.raises(InvalidCorruptionSpecificationError):
        gaussian_blur_parameters_for_severity("extreme")


def test_masks_are_unchanged_and_binary_validated() -> None:
    image = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    mask = np.asarray([[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=np.uint8)

    noise_result = apply_gaussian_noise(
        image,
        _spec(corruption_name="gaussian_noise"),
        mask=mask,
    )
    blur_result = apply_gaussian_blur(
        image,
        _spec(corruption_name="gaussian_blur", deterministic_seed=None),
        mask=np.asfortranarray(mask),
    )

    np.testing.assert_array_equal(noise_result.transformed_mask, mask)
    np.testing.assert_array_equal(blur_result.transformed_mask, mask)
    assert noise_result.transform_result.mask_binary_preserved is True
    assert blur_result.transform_result.mask_binary_preserved is True

    bad_mask = mask.astype(np.float64)
    bad_mask[0, 0, 0] = 0.5
    with pytest.raises(InvalidTransformInputError, match="mask"):
        apply_gaussian_noise(image, _spec(corruption_name="gaussian_noise"), mask=bad_mask)


def test_non_finite_input_rejected() -> None:
    image = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    image[0, 0, 0] = np.nan

    with pytest.raises(InvalidTransformInputError, match="finite"):
        apply_gaussian_noise(image, _spec(corruption_name="gaussian_noise"))
    with pytest.raises(InvalidTransformInputError, match="finite"):
        apply_gaussian_blur(
            image,
            _spec(corruption_name="gaussian_blur", deterministic_seed=None),
        )


def test_invalid_or_missing_seed_for_noise_is_rejected() -> None:
    image = np.zeros((2, 2, 2), dtype=np.float64)

    with pytest.raises(InvalidCorruptionSpecificationError, match="deterministic_seed"):
        apply_gaussian_noise(
            image,
            _spec(corruption_name="gaussian_noise", deterministic_seed=None),
        )


def test_malformed_specifications_are_rejected() -> None:
    image = np.zeros((2, 2, 2), dtype=np.float64)
    bad_noise = _spec(
        corruption_name="gaussian_noise",
        parameters={"sigma_hu": 6.0},
        deterministic_seed=1,
    )
    with pytest.raises(InvalidCorruptionSpecificationError, match="sigma_hu"):
        apply_gaussian_noise(image, bad_noise)

    bad_blur = _spec(
        corruption_name="gaussian_blur",
        parameters={"sigma_voxels": 0.5, "truncate": 2.0, "boundary_mode": "reflect"},
        deterministic_seed=None,
    )
    with pytest.raises(InvalidCorruptionSpecificationError, match="boundary_mode"):
        apply_gaussian_blur(image, bad_blur)


def test_transform_result_hashes_are_deterministic_and_canonical() -> None:
    image = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    first = apply_gaussian_blur(
        image,
        _spec(corruption_name="gaussian_blur", deterministic_seed=None),
    )
    second = apply_gaussian_blur(
        np.asfortranarray(image),
        _spec(corruption_name="gaussian_blur", deterministic_seed=None),
    )

    assert (
        first.transform_result.transform_result_hash
        == second.transform_result.transform_result_hash
    )
    assert phase7_transform_result_to_json(
        first.transform_result
    ) == phase7_transform_result_to_json(second.transform_result)


def test_none_severity_is_identity_but_still_records_completed_transform() -> None:
    image = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    noise = apply_gaussian_noise(
        image,
        _spec(corruption_name="gaussian_noise", severity="none", deterministic_seed=7),
    )
    blur = apply_gaussian_blur(
        image,
        _spec(corruption_name="gaussian_blur", severity="none", deterministic_seed=None),
    )

    np.testing.assert_array_equal(noise.transformed_image, image)
    np.testing.assert_array_equal(blur.transformed_image, image)
    assert noise.transform_result.execution_status == "completed"
    assert blur.transform_result.execution_status == "completed"


def test_no_query_label_or_reference_mask_public_api() -> None:
    forbidden = {
        "query_label",
        "query_labels",
        "label_map",
        "reference_mask",
        "query_reference_mask",
    }
    public_callables = (apply_gaussian_noise, apply_gaussian_blur)

    for function in public_callables:
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
