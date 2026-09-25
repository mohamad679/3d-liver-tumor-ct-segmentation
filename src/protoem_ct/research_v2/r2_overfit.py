"""Deterministic helpers for the Research-v2 R2 few-case real-data overfit run."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from protoem_ct.research_v2.r1_audit import resample_array_to_grid
from protoem_ct.research_v2.r2_preflight import R2_PATCH_SIZE

R2_THRESHOLD: Final[float] = 0.5
R2_POSITIVE_PATCH_FRACTION: Final[float] = 0.75
R2_FULL_VOLUME_EVAL_EVERY_STEPS: Final[int] = 250
R2_PATCH_LOG_EVERY_STEPS: Final[int] = 25
R2_CALIBRATION_MAX_STEPS: Final[int] = 20
R2_PRIMARY_MAX_STEPS: Final[int] = 2500
R2_PRIMARY_MAX_GPU_WALL_HOURS: Final[float] = 3.0
R2_RELOAD_PROBABILITY_ATOL: Final[float] = 1e-6

FloatArray = npt.NDArray[np.floating]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class R2SelectedCaseSpec:
    anonymous_case_id: str
    role: str
    tumor_voxel_count: int
    image_sha256: str
    label_sha256: str


R2_SELECTED_CASES: Final[tuple[R2SelectedCaseSpec, ...]] = (
    R2SelectedCaseSpec(
        anonymous_case_id="case_65917a1f3a47339a558042dac093bf1a",
        role="positive_q20",
        tumor_voxel_count=2261,
        image_sha256="fb2c353bc4cc35aedd9c8070883791c2241beb36bf7e5f884cd8a0f418226b9d",
        label_sha256="56282d27ae73a4ea808311dc54363fa344eee473c9af184f069ab94e97a70dae",
    ),
    R2SelectedCaseSpec(
        anonymous_case_id="case_f232be7b649d2166e5158aa636b38766",
        role="positive_q50",
        tumor_voxel_count=12265,
        image_sha256="7efcb4f28d9cc19b2c306260b5b75f13a8ab772d309a211dbdb4a8ef8b30a124",
        label_sha256="b4946039c62db1b69c4d0bc810e4a3a51effdf4fe380b7b53def160cb3c6bc18",
    ),
    R2SelectedCaseSpec(
        anonymous_case_id="case_6e4782e999691662f69fdf25f0e2e423",
        role="positive_q80",
        tumor_voxel_count=179093,
        image_sha256="7489d81f32ad48ce33e086cbc0105d65d1db4c9122c97af8d3ceef45eb9cfa47",
        label_sha256="3b12ff9755d6719311bcc74b825a45c6ce5ee56ef7a33e5c294289043dd58dbf",
    ),
    R2SelectedCaseSpec(
        anonymous_case_id="case_25928d2451a7d948a79e60084e3c5d32",
        role="empty_lexicographic_first",
        tumor_voxel_count=0,
        image_sha256="9f0fac0d26bfab882941e840a7a4eef6dab9d35d19bc64e1e79e7ec5738503bc",
        label_sha256="e5ec79c42475d308595a6108b3f48d231bfcba2d3692bbe773584f2d458d2fad",
    ),
)


def training_role_for_step(step: int) -> str:
    """Return the locked 3-positive/1-empty schedule for a one-indexed training step."""

    if step <= 0:
        raise ValueError("R2 training step must be positive")
    return "positive" if (step - 1) % 4 < 3 else "empty"


def positive_case_index_for_step(step: int) -> int:
    """Cycle q20/q50/q80 across positive training steps."""

    if training_role_for_step(step) != "positive":
        raise ValueError("positive-case index requested for a non-positive R2 step")
    positive_steps_before = (step - 1) - ((step - 1) // 4)
    return positive_steps_before % 3


def choose_random_positive_center(mask: BoolArray, *, rng: np.random.Generator) -> tuple[int, int, int]:
    """Sample a tumor voxel from one positive resampled mask."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 3:
        raise ValueError("R2 positive sampling mask must be exactly 3D")
    coordinates = np.argwhere(array)
    if len(coordinates) == 0:
        raise ValueError("R2 positive sampling requires at least one tumor voxel")
    selected = coordinates[int(rng.integers(0, len(coordinates)))]
    return cast(tuple[int, int, int], tuple(int(value) for value in selected))


def choose_random_center(shape: Sequence[int], *, rng: np.random.Generator) -> tuple[int, int, int]:
    """Sample one deterministic RNG-driven center from a 3D volume shape."""

    if len(shape) != 3:
        raise ValueError("R2 random-center shape must contain exactly three dimensions")
    shape3 = tuple(int(value) for value in shape)
    if any(value <= 0 for value in shape3):
        raise ValueError("R2 random-center dimensions must be positive")
    return cast(
        tuple[int, int, int],
        tuple(int(rng.integers(0, value)) for value in shape3),
    )


def binary_dice(ground_truth: BoolArray, prediction: BoolArray) -> float:
    """Compute the locked binary Dice convention, including both-empty = 1."""

    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError("R2 binary Dice requires same-shape 3D masks")
    gt_count = int(np.count_nonzero(gt))
    pred_count = int(np.count_nonzero(pred))
    if gt_count + pred_count == 0:
        return 1.0
    intersection = int(np.count_nonzero(gt & pred))
    return float(2.0 * intersection / (gt_count + pred_count))


def threshold_probability(probability: FloatArray, *, threshold: float = R2_THRESHOLD) -> BoolArray:
    """Validate and threshold an R2 tumor probability map at the fixed preregistered threshold."""

    if threshold != R2_THRESHOLD:
        raise ValueError("R2 threshold is locked at exactly 0.5")
    array = np.asarray(probability, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("R2 probability must be exactly 3D")
    if not bool(np.isfinite(array).all()):
        raise ValueError("R2 probability contains NaN or Inf")
    if bool(np.any(array < 0.0)) or bool(np.any(array > 1.0)):
        raise ValueError("R2 probability must lie in [0, 1]")
    return cast(BoolArray, array >= threshold)


def restore_probability_to_native(
    probability_resampled: FloatArray,
    *,
    resampled_affine: npt.NDArray[np.generic],
    native_shape: Sequence[int],
    native_affine: npt.NDArray[np.generic],
) -> FloatArray:
    """Restore continuous tumor probability to the native metric grid before thresholding."""

    restored = resample_array_to_grid(
        probability_resampled,
        source_affine=resampled_affine,
        target_shape=native_shape,
        target_affine=native_affine,
        is_label_or_mask=False,
    )
    array = np.asarray(restored, dtype=np.float32)
    if not bool(np.isfinite(array).all()):
        raise ValueError("R2 restored native probability contains NaN or Inf")
    if bool(np.any(array < -1e-6)) or bool(np.any(array > 1.0 + 1e-6)):
        raise ValueError("R2 restored native probability lies outside numerical [0, 1] tolerance")
    return cast(FloatArray, np.clip(array, 0.0, 1.0))


def patch_size() -> tuple[int, int, int]:
    return R2_PATCH_SIZE


__all__ = [
    "R2_CALIBRATION_MAX_STEPS",
    "R2_FULL_VOLUME_EVAL_EVERY_STEPS",
    "R2_PATCH_LOG_EVERY_STEPS",
    "R2_POSITIVE_PATCH_FRACTION",
    "R2_PRIMARY_MAX_GPU_WALL_HOURS",
    "R2_PRIMARY_MAX_STEPS",
    "R2_RELOAD_PROBABILITY_ATOL",
    "R2_SELECTED_CASES",
    "R2_THRESHOLD",
    "R2SelectedCaseSpec",
    "binary_dice",
    "choose_random_center",
    "choose_random_positive_center",
    "patch_size",
    "positive_case_index_for_step",
    "restore_probability_to_native",
    "threshold_probability",
    "training_role_for_step",
]
