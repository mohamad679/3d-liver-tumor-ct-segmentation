from __future__ import annotations

import numpy as np
import pytest

from protoem_ct.research_v2.r2_preflight import (
    R2_PATCH_SIZE,
    choose_positive_center,
    extract_centered_patch,
    normalize_r2_ct,
    r2_binary_tumor_target,
)


def test_r2_binary_tumor_target_maps_only_raw_label_two() -> None:
    label = np.array(
        [
            [[0, 1], [2, 0]],
            [[1, 2], [0, 2]],
        ],
        dtype=np.int16,
    )
    target = r2_binary_tumor_target(label)
    expected = label == 2
    assert target.dtype == np.bool_
    assert np.array_equal(target, expected)
    assert int(np.count_nonzero(target)) == 3


def test_r2_binary_tumor_target_rejects_invalid_raw_label() -> None:
    label = np.zeros((3, 3, 3), dtype=np.int16)
    label[1, 1, 1] = 7
    with pytest.raises(ValueError, match="outside"):
        r2_binary_tumor_target(label)


def test_normalize_r2_ct_uses_preregistered_clip_and_scale() -> None:
    image = np.array([[[-300.0, -200.0, 50.0, 300.0, 500.0]]], dtype=np.float32)
    normalized = normalize_r2_ct(image)
    assert normalized.dtype == np.float32
    assert normalized.shape == image.shape
    assert np.allclose(
        normalized,
        np.array([[[-1.0, -1.0, 0.0, 1.0, 1.0]]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_positive_center_is_deterministic_and_on_tumor() -> None:
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[2:5, 6:9, 3:7] = True
    first = choose_positive_center(mask)
    second = choose_positive_center(mask)
    assert first == second
    assert bool(mask[first])


def test_extract_centered_patch_preserves_positive_voxel_and_exact_shape() -> None:
    mask = np.zeros((20, 18, 12), dtype=np.uint8)
    center = (0, 17, 11)
    mask[center] = 1
    patch = extract_centered_patch(
        mask,
        center=center,
        patch_size=R2_PATCH_SIZE,
        pad_value=0,
    )
    assert patch.shape == R2_PATCH_SIZE
    assert int(np.count_nonzero(patch)) == 1
