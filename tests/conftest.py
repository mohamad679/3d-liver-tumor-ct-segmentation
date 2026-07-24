"""Shared synthetic NIfTI fixtures for validation tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

NiftiArray = npt.NDArray[np.generic]
AffineArray = npt.NDArray[np.float64]
WriteNiftiFile = Callable[[Path, NiftiArray, AffineArray], Path]


@pytest.fixture
def matching_affine() -> AffineArray:
    """Return a deterministic affine shared by synthetic image-label pairs."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 1.5, 0.0, 5.0],
            [0.0, 0.0, 2.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def valid_ct_image() -> npt.NDArray[np.float32]:
    """Return a tiny deterministic synthetic 3D image array."""
    return (np.arange(24, dtype=np.float32).reshape((2, 3, 4)) / 10.0) - 1.0


@pytest.fixture
def valid_binary_tumor_mask() -> npt.NDArray[np.uint8]:
    """Return a tiny deterministic synthetic binary mask array."""
    mask = np.zeros((2, 3, 4), dtype=np.uint8)
    mask[0, 1, 2] = 1
    mask[1, 2, 3] = 1
    return mask


@pytest.fixture
def write_nifti_file() -> WriteNiftiFile:
    """Return a helper that writes test-only NIfTI files into pytest temp paths."""

    def _write_nifti_file(path: Path, data: NiftiArray, affine: AffineArray) -> Path:
        image = nib.Nifti1Image(data, affine)  # type: ignore[no-untyped-call]
        nib.save(image, str(path))
        return path

    return _write_nifti_file
