"""Pure helpers for the Research-v2 R2 real-train GPU preflight."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from protoem_ct.research_v2.r1_audit import validate_raw_lits_label_array

R2_SEED: Final[int] = 1729
R2_PATCH_SIZE: Final[tuple[int, int, int]] = (96, 96, 64)
R2_TARGET_SPACING_MM: Final[tuple[float, float, float]] = (1.5, 1.5, 1.5)
R2_INTENSITY_CLIP_HU: Final[tuple[float, float]] = (-200.0, 300.0)
R2_ALLOCATION_PROBE_BYTES: Final[int] = 256 * 1024 * 1024

FloatArray = npt.NDArray[np.floating]
BoolArray = npt.NDArray[np.bool_]


def r2_binary_tumor_target(label: npt.NDArray[np.generic]) -> BoolArray:
    """Map the locked raw tumor label value 2 to the R2 binary training target."""

    return validate_raw_lits_label_array(label)


def normalize_r2_ct(image_hu: npt.NDArray[np.generic]) -> FloatArray:
    """Clip to the preregistered HU window and scale linearly to [-1, 1]."""

    array = np.asarray(image_hu, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("R2 CT normalization requires an exactly 3D image")
    if not bool(np.isfinite(array).all()):
        raise ValueError("R2 CT image contains NaN or Inf")
    low, high = R2_INTENSITY_CLIP_HU
    clipped = np.clip(array, low, high)
    scaled = 2.0 * ((clipped - low) / (high - low)) - 1.0
    return cast(FloatArray, np.asarray(scaled, dtype=np.float32))


def choose_positive_center(mask: BoolArray, *, seed: int = R2_SEED) -> tuple[int, int, int]:
    """Choose one tumor voxel deterministically for the preflight positive patch."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 3:
        raise ValueError("R2 positive-center mask must be exactly 3D")
    coordinates = np.argwhere(array)
    if coordinates.size == 0:
        raise ValueError("R2 positive-center selection requires at least one tumor voxel")
    rng = np.random.default_rng(seed)
    selected = coordinates[int(rng.integers(0, len(coordinates)))]
    return (int(selected[0]), int(selected[1]), int(selected[2]))


def extract_centered_patch(
    array: npt.NDArray[np.generic],
    *,
    center: Sequence[int],
    patch_size: Sequence[int] = R2_PATCH_SIZE,
    pad_value: float | int | bool = 0,
) -> npt.NDArray[np.generic]:
    """Extract an exact-size 3D patch around a center, padding only beyond volume edges."""

    data = np.asarray(array)
    if data.ndim != 3:
        raise ValueError("R2 patch extraction requires an exactly 3D array")
    if len(center) != 3 or len(patch_size) != 3:
        raise ValueError("R2 patch center and size must each contain exactly three values")
    center3 = tuple(int(value) for value in center)
    size3 = tuple(int(value) for value in patch_size)
    if any(value <= 0 for value in size3):
        raise ValueError("R2 patch dimensions must be positive")

    source_slices: list[slice] = []
    destination_slices: list[slice] = []
    for axis in range(3):
        start = center3[axis] - size3[axis] // 2
        stop = start + size3[axis]
        source_start = max(0, start)
        source_stop = min(int(data.shape[axis]), stop)
        destination_start = source_start - start
        destination_stop = destination_start + (source_stop - source_start)
        source_slices.append(slice(source_start, source_stop))
        destination_slices.append(slice(destination_start, destination_stop))

    patch = np.full(size3, pad_value, dtype=data.dtype)
    patch[tuple(destination_slices)] = data[tuple(source_slices)]
    return np.asarray(patch)


__all__ = [
    "R2_ALLOCATION_PROBE_BYTES",
    "R2_INTENSITY_CLIP_HU",
    "R2_PATCH_SIZE",
    "R2_SEED",
    "R2_TARGET_SPACING_MM",
    "choose_positive_center",
    "extract_centered_patch",
    "normalize_r2_ct",
    "r2_binary_tumor_target",
]
