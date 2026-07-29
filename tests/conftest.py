"""Shared synthetic NIfTI fixtures for validation tests."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

NiftiArray = npt.NDArray[np.generic]
AffineArray = npt.NDArray[np.float64]
WriteNiftiFile = Callable[[Path, NiftiArray, AffineArray], Path]
ImportGuardRunner = Callable[[tuple[str, ...], tuple[str, ...]], None]
_MISSING_MODULE_ATTRIBUTE = object()


def _snapshot_module_state(
    prefixes: tuple[str, ...],
) -> tuple[dict[str, ModuleType], dict[tuple[str, str], object]]:
    """Capture affected module objects and their immediate parent-package attributes."""

    modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    attributes: dict[tuple[str, str], object] = {}
    for name in sorted(set(modules) | set(prefixes)):
        parent_name, separator, attribute_name = name.rpartition(".")
        if not separator:
            continue
        parent = sys.modules.get(parent_name)
        if parent is not None:
            attributes[(parent_name, attribute_name)] = getattr(
                parent,
                attribute_name,
                _MISSING_MODULE_ATTRIBUTE,
            )
    return modules, attributes


@pytest.fixture
def run_import_guard() -> ImportGuardRunner:
    """Run an import guard in a subprocess and require exact parent-state preservation."""

    def _run_import_guard(
        import_names: tuple[str, ...],
        forbidden_names: tuple[str, ...],
    ) -> None:
        prefixes = tuple(sorted(set(import_names) | set(forbidden_names)))
        modules_before, attributes_before = _snapshot_module_state(prefixes)
        script = "\n".join(
            (
                "import importlib",
                "import sys",
                f"for module_name in {import_names!r}:",
                "    importlib.import_module(module_name)",
                f"for module_name in {forbidden_names!r}:",
                "    assert module_name not in sys.modules, module_name",
            )
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        modules_after, attributes_after = _snapshot_module_state(prefixes)

        assert result.returncode == 0, result.stderr
        assert modules_after.keys() == modules_before.keys()
        assert all(modules_after[name] is module for name, module in modules_before.items())
        assert attributes_after.keys() == attributes_before.keys()
        assert all(attributes_after[key] is value for key, value in attributes_before.items())

    return _run_import_guard


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
