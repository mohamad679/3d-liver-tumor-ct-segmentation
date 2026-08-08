"""Phase 8 definitive development pipeline (Package A: implementation only).

This module implements the approved, LOCKED configuration design identified
as ``PHASE8-DEFINITIVE-CONFIG-DESIGN-V1`` (see
``docs/phase8/DEFINITIVE_CONFIG_AUDIT.md``): a general-purpose, arbitrary
case-count preprocessing / training / validation pipeline for the Phase 8
definitive development candidate.

This is **implementation only**. It performs no filesystem I/O, no dataset
access, and no ``/Volumes`` access of any kind -- every function operates on
already-in-memory NumPy arrays and (for the training/validation runners)
in-memory ``torch``/``monai`` tensors and models supplied by the caller. It
must never be pointed at a real dataset from this module alone.

Locked design values (do not change without a new, separately approved
design identifier):

* Preprocessing: orientation -> RAS; spacing -> componentwise median of the
  development TRAIN partition (derivation function only, no fitting here);
  image interpolation -> trilinear; label interpolation -> nearest-neighbor;
  HU clip -> ``[-1000, 1000]``; intensity scaling -> linear to ``[-1, 1]``;
  tumor label -> raw label ``== 2``.
* Training candidate: MONAI SegResNet; patch size ``[64, 64, 32]``;
  foreground-biased random sampling at a 1:1 positive:negative ratio;
  AdamW, lr ``1e-4``, weight decay ``1e-5``; DiceCE loss; CPU; AMP disabled;
  no augmentation; seed ``1729``.
* Validation candidate: full-volume sliding-window inference, ROI
  ``[96, 96, 64]``, overlap ``0.25``, sliding-window batch size ``1``;
  checkpoint selection metric -> tumor Dice; inference threshold ``0.5``;
  neither the internal test partition nor external data is ever used for
  selection.

Explicitly out of scope for this module (and for
:class:`Phase8DefinitiveConfig`, which therefore has no fields for them):
epoch/step budget, validation cadence, checkpoint cadence, early stopping,
resume policy, augmentation experiments, hyperparameter search, and
multi-candidate comparison. Callers supply ``max_steps`` and case lists as
plain arguments to the pure training/validation functions below.

``torch`` and ``monai`` are imported lazily inside function bodies (mirroring
:mod:`protoem_ct.external.definitive_training_pilot`) so importing this
module never requires the heavy baseline runtime dependencies. This module
reads that pilot module only as an engineering reference for pattern reuse
(RAS-orientation checks, HU clip/scale, checkpoint metadata wrapping,
fail-closed nonfinite-loss/gradient handling, SegResNet construction, and
the sliding-window inference call); it does not import from it and does not
reuse its hard-locked, pilot-only, non-definitive dataclasses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.external.internal_evidence import (
    PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME,
    PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
    ArtifactReference,
    Phase8CheckpointMetadata,
    hash_phase8_checkpoint_metadata,
)

# ---------------------------------------------------------------------------
# Design identity
# ---------------------------------------------------------------------------

PHASE8_DEFINITIVE_CONFIG_DESIGN_IDENTIFIER: Final[str] = "PHASE8-DEFINITIVE-CONFIG-DESIGN-V1"

# Locked preprocessing values.
REQUIRED_ORIENTATION_POLICY: Final[str] = "ras"
REQUIRED_IMAGE_INTERPOLATION_POLICY: Final[str] = "trilinear"
REQUIRED_LABEL_INTERPOLATION_POLICY: Final[str] = "nearest"
REQUIRED_HU_CLIP_MIN: Final[float] = -1000.0
REQUIRED_HU_CLIP_MAX: Final[float] = 1000.0
REQUIRED_INTENSITY_SCALE_MIN: Final[float] = -1.0
REQUIRED_INTENSITY_SCALE_MAX: Final[float] = 1.0
REQUIRED_TUMOR_RAW_LABEL_VALUE: Final[int] = 2

# Locked training-candidate values.
REQUIRED_MODEL_FAMILY: Final[str] = "segresnet"
REQUIRED_PATCH_SIZE: Final[tuple[int, int, int]] = (64, 64, 32)
REQUIRED_SAMPLING_POLICY: Final[str] = "foreground_biased_random"
REQUIRED_POSITIVE_NEGATIVE_RATIO: Final[tuple[int, int]] = (1, 1)
REQUIRED_OPTIMIZER_NAME: Final[str] = "adamw"
REQUIRED_LEARNING_RATE: Final[float] = 1e-4
REQUIRED_WEIGHT_DECAY: Final[float] = 1e-5
REQUIRED_LOSS_NAME: Final[str] = "dice_ce"
REQUIRED_DEVICE_TYPE: Final[str] = "cpu"
REQUIRED_AMP_ENABLED: Final[bool] = False
REQUIRED_AUGMENTATION_POLICY: Final[str] = "none"
REQUIRED_SEED: Final[int] = 1729

# Locked validation-candidate values.
REQUIRED_VALIDATION_METHOD: Final[str] = "full_volume_sliding_window"
REQUIRED_SLIDING_WINDOW_ROI_SIZE: Final[tuple[int, int, int]] = (96, 96, 64)
REQUIRED_SLIDING_WINDOW_OVERLAP: Final[float] = 0.25
REQUIRED_SLIDING_WINDOW_BATCH_SIZE: Final[int] = 1
REQUIRED_CHECKPOINT_SELECTION_METRIC: Final[str] = "tumor_dice"
REQUIRED_INFERENCE_THRESHOLD: Final[float] = 0.5
REQUIRED_INTERNAL_TEST_USED_FOR_SELECTION: Final[bool] = False
REQUIRED_EXTERNAL_DATA_USED_FOR_SELECTION: Final[bool] = False

_MAX_NEGATIVE_PATCH_ATTEMPTS: Final[int] = 50


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class Phase8DefinitivePipelineError(ValueError):
    """Base error for the Phase 8 definitive development pipeline."""


class Phase8DefinitivePipelineConfigError(Phase8DefinitivePipelineError):
    """Raised when the locked definitive config is tampered with in any way."""


class Phase8DefinitivePipelineSamplingError(Phase8DefinitivePipelineError):
    """Raised when foreground-aware patch sampling cannot proceed fail-closed."""


class Phase8DefinitivePipelineRuntimeError(Phase8DefinitivePipelineError):
    """Raised when a torch/monai training or validation step fails closed."""


class Phase8DefinitivePipelineCheckpointError(Phase8DefinitivePipelineError):
    """Raised when checkpoint metadata construction violates a hard invariant."""


# ---------------------------------------------------------------------------
# A. Locked configuration
# ---------------------------------------------------------------------------


def _config_payload(
    *,
    design_identifier: str,
    orientation_policy: str,
    image_interpolation_policy: str,
    label_interpolation_policy: str,
    hu_clip_min: float,
    hu_clip_max: float,
    intensity_scale_min: float,
    intensity_scale_max: float,
    tumor_raw_label_value: int,
    model_family: str,
    patch_size: Sequence[int],
    sampling_policy: str,
    positive_negative_ratio: Sequence[int],
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    device_type: str,
    amp_enabled: bool,
    augmentation_policy: str,
    seed: int,
    validation_method: str,
    sliding_window_roi_size: Sequence[int],
    sliding_window_overlap: float,
    sliding_window_batch_size: int,
    checkpoint_selection_metric: str,
    inference_threshold: float,
    internal_test_used_for_selection: bool,
    external_data_used_for_selection: bool,
) -> dict[str, JsonValue]:
    return {
        "amp_enabled": amp_enabled,
        "augmentation_policy": augmentation_policy,
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "design_identifier": design_identifier,
        "device_type": device_type,
        "external_data_used_for_selection": external_data_used_for_selection,
        "hu_clip_max": hu_clip_max,
        "hu_clip_min": hu_clip_min,
        "image_interpolation_policy": image_interpolation_policy,
        "inference_threshold": inference_threshold,
        "intensity_scale_max": intensity_scale_max,
        "intensity_scale_min": intensity_scale_min,
        "internal_test_used_for_selection": internal_test_used_for_selection,
        "label_interpolation_policy": label_interpolation_policy,
        "learning_rate": learning_rate,
        "loss_name": loss_name,
        "model_family": model_family,
        "optimizer_name": optimizer_name,
        "orientation_policy": orientation_policy,
        "patch_size": list(patch_size),
        "positive_negative_ratio": list(positive_negative_ratio),
        "sampling_policy": sampling_policy,
        "seed": seed,
        "sliding_window_batch_size": sliding_window_batch_size,
        "sliding_window_overlap": sliding_window_overlap,
        "sliding_window_roi_size": list(sliding_window_roi_size),
        "tumor_raw_label_value": tumor_raw_label_value,
        "validation_method": validation_method,
        "weight_decay": weight_decay,
    }


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveConfig:
    """Immutable, LOCKED definitive-development configuration.

    Every field is hard-locked to the value approved under
    ``PHASE8-DEFINITIVE-CONFIG-DESIGN-V1``. This dataclass is not
    caller-tunable: :meth:`__post_init__` fails closed with
    :class:`Phase8DefinitivePipelineConfigError` if any field differs from
    its required value. The only public constructor is
    :func:`build_definitive_config_v1`.
    """

    design_identifier: str
    orientation_policy: str
    image_interpolation_policy: str
    label_interpolation_policy: str
    hu_clip_min: float
    hu_clip_max: float
    intensity_scale_min: float
    intensity_scale_max: float
    tumor_raw_label_value: int
    model_family: str
    patch_size: tuple[int, int, int]
    sampling_policy: str
    positive_negative_ratio: tuple[int, int]
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    loss_name: str
    device_type: str
    amp_enabled: bool
    augmentation_policy: str
    seed: int
    validation_method: str
    sliding_window_roi_size: tuple[int, int, int]
    sliding_window_overlap: float
    sliding_window_batch_size: int
    checkpoint_selection_metric: str
    inference_threshold: float
    internal_test_used_for_selection: bool
    external_data_used_for_selection: bool
    config_hash: str

    def __post_init__(self) -> None:
        if self.design_identifier != PHASE8_DEFINITIVE_CONFIG_DESIGN_IDENTIFIER:
            raise Phase8DefinitivePipelineConfigError(
                f"design_identifier must equal {PHASE8_DEFINITIVE_CONFIG_DESIGN_IDENTIFIER!r}."
            )
        if self.orientation_policy != REQUIRED_ORIENTATION_POLICY:
            raise Phase8DefinitivePipelineConfigError(
                f"orientation_policy must equal {REQUIRED_ORIENTATION_POLICY!r}."
            )
        if self.image_interpolation_policy != REQUIRED_IMAGE_INTERPOLATION_POLICY:
            raise Phase8DefinitivePipelineConfigError(
                f"image_interpolation_policy must equal {REQUIRED_IMAGE_INTERPOLATION_POLICY!r}."
            )
        if self.label_interpolation_policy != REQUIRED_LABEL_INTERPOLATION_POLICY:
            raise Phase8DefinitivePipelineConfigError(
                f"label_interpolation_policy must equal {REQUIRED_LABEL_INTERPOLATION_POLICY!r}."
            )
        if self.hu_clip_min != REQUIRED_HU_CLIP_MIN or self.hu_clip_max != REQUIRED_HU_CLIP_MAX:
            raise Phase8DefinitivePipelineConfigError(
                f"HU clip must equal [{REQUIRED_HU_CLIP_MIN}, {REQUIRED_HU_CLIP_MAX}]."
            )
        if (
            self.intensity_scale_min != REQUIRED_INTENSITY_SCALE_MIN
            or self.intensity_scale_max != REQUIRED_INTENSITY_SCALE_MAX
        ):
            raise Phase8DefinitivePipelineConfigError(
                "intensity scale must equal "
                f"[{REQUIRED_INTENSITY_SCALE_MIN}, {REQUIRED_INTENSITY_SCALE_MAX}]."
            )
        if self.tumor_raw_label_value != REQUIRED_TUMOR_RAW_LABEL_VALUE:
            raise Phase8DefinitivePipelineConfigError(
                f"tumor_raw_label_value must equal {REQUIRED_TUMOR_RAW_LABEL_VALUE!r}."
            )
        if self.model_family != REQUIRED_MODEL_FAMILY:
            raise Phase8DefinitivePipelineConfigError(
                f"model_family must equal {REQUIRED_MODEL_FAMILY!r}."
            )
        object.__setattr__(self, "patch_size", tuple(int(v) for v in self.patch_size))
        if self.patch_size != REQUIRED_PATCH_SIZE:
            raise Phase8DefinitivePipelineConfigError(
                f"patch_size must equal {REQUIRED_PATCH_SIZE!r}, got {self.patch_size!r}."
            )
        if self.sampling_policy != REQUIRED_SAMPLING_POLICY:
            raise Phase8DefinitivePipelineConfigError(
                f"sampling_policy must equal {REQUIRED_SAMPLING_POLICY!r}."
            )
        object.__setattr__(
            self,
            "positive_negative_ratio",
            tuple(int(v) for v in self.positive_negative_ratio),
        )
        if self.positive_negative_ratio != REQUIRED_POSITIVE_NEGATIVE_RATIO:
            raise Phase8DefinitivePipelineConfigError(
                f"positive_negative_ratio must equal {REQUIRED_POSITIVE_NEGATIVE_RATIO!r}."
            )
        if self.optimizer_name != REQUIRED_OPTIMIZER_NAME:
            raise Phase8DefinitivePipelineConfigError(
                f"optimizer_name must equal {REQUIRED_OPTIMIZER_NAME!r}."
            )
        if self.learning_rate != REQUIRED_LEARNING_RATE:
            raise Phase8DefinitivePipelineConfigError(
                f"learning_rate must equal {REQUIRED_LEARNING_RATE!r}."
            )
        if self.weight_decay != REQUIRED_WEIGHT_DECAY:
            raise Phase8DefinitivePipelineConfigError(
                f"weight_decay must equal {REQUIRED_WEIGHT_DECAY!r}."
            )
        if self.loss_name != REQUIRED_LOSS_NAME:
            raise Phase8DefinitivePipelineConfigError(
                f"loss_name must equal {REQUIRED_LOSS_NAME!r}."
            )
        if self.device_type != REQUIRED_DEVICE_TYPE:
            raise Phase8DefinitivePipelineConfigError(
                f"device_type must equal {REQUIRED_DEVICE_TYPE!r}."
            )
        if self.amp_enabled is not REQUIRED_AMP_ENABLED:
            raise Phase8DefinitivePipelineConfigError(
                f"amp_enabled must be {REQUIRED_AMP_ENABLED!r}."
            )
        if self.augmentation_policy != REQUIRED_AUGMENTATION_POLICY:
            raise Phase8DefinitivePipelineConfigError(
                f"augmentation_policy must equal {REQUIRED_AUGMENTATION_POLICY!r}."
            )
        if self.seed != REQUIRED_SEED:
            raise Phase8DefinitivePipelineConfigError(f"seed must equal {REQUIRED_SEED!r}.")
        if self.validation_method != REQUIRED_VALIDATION_METHOD:
            raise Phase8DefinitivePipelineConfigError(
                f"validation_method must equal {REQUIRED_VALIDATION_METHOD!r}."
            )
        object.__setattr__(
            self,
            "sliding_window_roi_size",
            tuple(int(v) for v in self.sliding_window_roi_size),
        )
        if self.sliding_window_roi_size != REQUIRED_SLIDING_WINDOW_ROI_SIZE:
            raise Phase8DefinitivePipelineConfigError(
                f"sliding_window_roi_size must equal {REQUIRED_SLIDING_WINDOW_ROI_SIZE!r}."
            )
        if self.sliding_window_overlap != REQUIRED_SLIDING_WINDOW_OVERLAP:
            raise Phase8DefinitivePipelineConfigError(
                f"sliding_window_overlap must equal {REQUIRED_SLIDING_WINDOW_OVERLAP!r}."
            )
        if self.sliding_window_batch_size != REQUIRED_SLIDING_WINDOW_BATCH_SIZE:
            raise Phase8DefinitivePipelineConfigError(
                f"sliding_window_batch_size must equal {REQUIRED_SLIDING_WINDOW_BATCH_SIZE!r}."
            )
        if self.checkpoint_selection_metric != REQUIRED_CHECKPOINT_SELECTION_METRIC:
            raise Phase8DefinitivePipelineConfigError(
                f"checkpoint_selection_metric must equal {REQUIRED_CHECKPOINT_SELECTION_METRIC!r}."
            )
        if self.inference_threshold != REQUIRED_INFERENCE_THRESHOLD:
            raise Phase8DefinitivePipelineConfigError(
                f"inference_threshold must equal {REQUIRED_INFERENCE_THRESHOLD!r}."
            )
        if self.internal_test_used_for_selection is not REQUIRED_INTERNAL_TEST_USED_FOR_SELECTION:
            raise Phase8DefinitivePipelineConfigError(
                "internal_test_used_for_selection must be "
                f"{REQUIRED_INTERNAL_TEST_USED_FOR_SELECTION!r}."
            )
        if self.external_data_used_for_selection is not REQUIRED_EXTERNAL_DATA_USED_FOR_SELECTION:
            raise Phase8DefinitivePipelineConfigError(
                "external_data_used_for_selection must be "
                f"{REQUIRED_EXTERNAL_DATA_USED_FOR_SELECTION!r}."
            )
        _require_self_hash(
            self.config_hash,
            _config_payload(
                design_identifier=self.design_identifier,
                orientation_policy=self.orientation_policy,
                image_interpolation_policy=self.image_interpolation_policy,
                label_interpolation_policy=self.label_interpolation_policy,
                hu_clip_min=self.hu_clip_min,
                hu_clip_max=self.hu_clip_max,
                intensity_scale_min=self.intensity_scale_min,
                intensity_scale_max=self.intensity_scale_max,
                tumor_raw_label_value=self.tumor_raw_label_value,
                model_family=self.model_family,
                patch_size=self.patch_size,
                sampling_policy=self.sampling_policy,
                positive_negative_ratio=self.positive_negative_ratio,
                optimizer_name=self.optimizer_name,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                loss_name=self.loss_name,
                device_type=self.device_type,
                amp_enabled=self.amp_enabled,
                augmentation_policy=self.augmentation_policy,
                seed=self.seed,
                validation_method=self.validation_method,
                sliding_window_roi_size=self.sliding_window_roi_size,
                sliding_window_overlap=self.sliding_window_overlap,
                sliding_window_batch_size=self.sliding_window_batch_size,
                checkpoint_selection_metric=self.checkpoint_selection_metric,
                inference_threshold=self.inference_threshold,
                internal_test_used_for_selection=self.internal_test_used_for_selection,
                external_data_used_for_selection=self.external_data_used_for_selection,
            ),
        )


def _require_self_hash(value: str, payload: dict[str, JsonValue]) -> None:
    expected = sha256_json(payload)
    if value != expected:
        raise Phase8DefinitivePipelineConfigError(
            "config_hash does not match the canonical locked-config identity."
        )


def build_definitive_config_v1() -> Phase8DefinitiveConfig:
    """Return the single approved ``PHASE8-DEFINITIVE-CONFIG-DESIGN-V1`` instance."""

    payload = _config_payload(
        design_identifier=PHASE8_DEFINITIVE_CONFIG_DESIGN_IDENTIFIER,
        orientation_policy=REQUIRED_ORIENTATION_POLICY,
        image_interpolation_policy=REQUIRED_IMAGE_INTERPOLATION_POLICY,
        label_interpolation_policy=REQUIRED_LABEL_INTERPOLATION_POLICY,
        hu_clip_min=REQUIRED_HU_CLIP_MIN,
        hu_clip_max=REQUIRED_HU_CLIP_MAX,
        intensity_scale_min=REQUIRED_INTENSITY_SCALE_MIN,
        intensity_scale_max=REQUIRED_INTENSITY_SCALE_MAX,
        tumor_raw_label_value=REQUIRED_TUMOR_RAW_LABEL_VALUE,
        model_family=REQUIRED_MODEL_FAMILY,
        patch_size=REQUIRED_PATCH_SIZE,
        sampling_policy=REQUIRED_SAMPLING_POLICY,
        positive_negative_ratio=REQUIRED_POSITIVE_NEGATIVE_RATIO,
        optimizer_name=REQUIRED_OPTIMIZER_NAME,
        learning_rate=REQUIRED_LEARNING_RATE,
        weight_decay=REQUIRED_WEIGHT_DECAY,
        loss_name=REQUIRED_LOSS_NAME,
        device_type=REQUIRED_DEVICE_TYPE,
        amp_enabled=REQUIRED_AMP_ENABLED,
        augmentation_policy=REQUIRED_AUGMENTATION_POLICY,
        seed=REQUIRED_SEED,
        validation_method=REQUIRED_VALIDATION_METHOD,
        sliding_window_roi_size=REQUIRED_SLIDING_WINDOW_ROI_SIZE,
        sliding_window_overlap=REQUIRED_SLIDING_WINDOW_OVERLAP,
        sliding_window_batch_size=REQUIRED_SLIDING_WINDOW_BATCH_SIZE,
        checkpoint_selection_metric=REQUIRED_CHECKPOINT_SELECTION_METRIC,
        inference_threshold=REQUIRED_INFERENCE_THRESHOLD,
        internal_test_used_for_selection=REQUIRED_INTERNAL_TEST_USED_FOR_SELECTION,
        external_data_used_for_selection=REQUIRED_EXTERNAL_DATA_USED_FOR_SELECTION,
    )
    return Phase8DefinitiveConfig(
        design_identifier=PHASE8_DEFINITIVE_CONFIG_DESIGN_IDENTIFIER,
        orientation_policy=REQUIRED_ORIENTATION_POLICY,
        image_interpolation_policy=REQUIRED_IMAGE_INTERPOLATION_POLICY,
        label_interpolation_policy=REQUIRED_LABEL_INTERPOLATION_POLICY,
        hu_clip_min=REQUIRED_HU_CLIP_MIN,
        hu_clip_max=REQUIRED_HU_CLIP_MAX,
        intensity_scale_min=REQUIRED_INTENSITY_SCALE_MIN,
        intensity_scale_max=REQUIRED_INTENSITY_SCALE_MAX,
        tumor_raw_label_value=REQUIRED_TUMOR_RAW_LABEL_VALUE,
        model_family=REQUIRED_MODEL_FAMILY,
        patch_size=REQUIRED_PATCH_SIZE,
        sampling_policy=REQUIRED_SAMPLING_POLICY,
        positive_negative_ratio=REQUIRED_POSITIVE_NEGATIVE_RATIO,
        optimizer_name=REQUIRED_OPTIMIZER_NAME,
        learning_rate=REQUIRED_LEARNING_RATE,
        weight_decay=REQUIRED_WEIGHT_DECAY,
        loss_name=REQUIRED_LOSS_NAME,
        device_type=REQUIRED_DEVICE_TYPE,
        amp_enabled=REQUIRED_AMP_ENABLED,
        augmentation_policy=REQUIRED_AUGMENTATION_POLICY,
        seed=REQUIRED_SEED,
        validation_method=REQUIRED_VALIDATION_METHOD,
        sliding_window_roi_size=REQUIRED_SLIDING_WINDOW_ROI_SIZE,
        sliding_window_overlap=REQUIRED_SLIDING_WINDOW_OVERLAP,
        sliding_window_batch_size=REQUIRED_SLIDING_WINDOW_BATCH_SIZE,
        checkpoint_selection_metric=REQUIRED_CHECKPOINT_SELECTION_METRIC,
        inference_threshold=REQUIRED_INFERENCE_THRESHOLD,
        internal_test_used_for_selection=REQUIRED_INTERNAL_TEST_USED_FOR_SELECTION,
        external_data_used_for_selection=REQUIRED_EXTERNAL_DATA_USED_FOR_SELECTION,
        config_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# B. Preprocessing (pure NumPy / nibabel-orientation functions)
# ---------------------------------------------------------------------------


def reorient_volume_to_ras(volume: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorient ``volume``/``affine`` to RAS and return the transformed pair.

    Performs a true reorientation (axis transpose/flip), not merely an
    orientation assertion, using ``nibabel.orientations``.
    """

    from nibabel.orientations import (
        apply_orientation,
        axcodes2ornt,
        io_orientation,
        ornt_transform,
    )

    affine_array = np.asarray(affine, dtype=np.float64)
    current_ornt = cast(Any, io_orientation)(affine_array)
    target_ornt = cast(Any, axcodes2ornt)(("R", "A", "S"))
    transform = cast(Any, ornt_transform)(current_ornt, target_ornt)
    reoriented_volume = cast(np.ndarray, cast(Any, apply_orientation)(volume, transform))

    # Compose the affine update implied by the orientation transform: for
    # each output axis, look up which source axis it came from and whether
    # it was flipped, then adjust the affine's columns and translation to
    # match, per the standard nibabel orientation-transform convention.
    reoriented_affine = affine_array.copy()
    axis_transform = np.asarray(transform, dtype=np.float64)
    new_affine = np.zeros_like(affine_array)
    for output_axis in range(3):
        source_axis = int(axis_transform[output_axis, 0])
        flip = axis_transform[output_axis, 1]
        new_affine[:, output_axis] = affine_array[:, source_axis] * flip
    new_affine[:, 3] = affine_array[:, 3]
    for output_axis in range(3):
        source_axis = int(axis_transform[output_axis, 0])
        flip = axis_transform[output_axis, 1]
        if flip < 0:
            dim_size = volume.shape[source_axis]
            new_affine[:, 3] += affine_array[:, source_axis] * (dim_size - 1)
    reoriented_affine = new_affine
    return reoriented_volume, reoriented_affine


def compute_median_train_spacing(
    spacings: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Return the componentwise median spacing over the development TRAIN partition.

    Pure function over caller-supplied spacing tuples; performs no dataset
    access and does not decide which cases belong to the train partition.
    """

    if not spacings:
        raise Phase8DefinitivePipelineConfigError(
            "compute_median_train_spacing requires at least one spacing tuple."
        )
    array = np.asarray(spacings, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise Phase8DefinitivePipelineConfigError(
            "each spacing must be a 3-tuple of (x, y, z) voxel spacing."
        )
    median = np.median(array, axis=0)
    return (float(median[0]), float(median[1]), float(median[2]))


def resample_volume_to_spacing(
    volume: np.ndarray,
    *,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    is_label: bool,
) -> np.ndarray:
    """Resample ``volume`` from ``source_spacing`` to ``target_spacing``.

    Uses ``scipy.ndimage.zoom`` with ``order=0`` (nearest-neighbor) for
    labels and ``order=1`` (trilinear-equivalent) for images, matching the
    locked interpolation policies.
    """

    from scipy.ndimage import zoom  # type: ignore[import-untyped]

    zoom_factors = tuple(
        float(source_spacing[axis]) / float(target_spacing[axis]) for axis in range(3)
    )
    order = 0 if is_label else 1
    resampled = cast(Any, zoom)(
        volume, zoom=zoom_factors, order=order, mode="nearest", prefilter=False
    )
    return cast(np.ndarray, resampled)


def clip_and_scale_intensity(volume: np.ndarray, *, config: Phase8DefinitiveConfig) -> np.ndarray:
    """Clip to the locked HU window then linearly rescale to the locked intensity range."""

    clipped = np.clip(volume.astype(np.float32, copy=True), config.hu_clip_min, config.hu_clip_max)
    hu_range = config.hu_clip_max - config.hu_clip_min
    scale_range = config.intensity_scale_max - config.intensity_scale_min
    scaled = (clipped - config.hu_clip_min) / hu_range * scale_range + config.intensity_scale_min
    return cast(np.ndarray, scaled)


def binarize_tumor_label(label_array: np.ndarray, *, config: Phase8DefinitiveConfig) -> np.ndarray:
    """Return the boolean tumor mask ``raw_label == config.tumor_raw_label_value``."""

    return cast(np.ndarray, label_array == config.tumor_raw_label_value)


# ---------------------------------------------------------------------------
# C. Foreground-aware patch sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8DefinitivePatch:
    """One sampled training patch and whether it was foreground-centered."""

    image_patch: np.ndarray
    label_patch: np.ndarray
    is_positive: bool


def _crop_patch(
    volume: np.ndarray, *, center: tuple[int, int, int], patch_size: tuple[int, int, int]
) -> np.ndarray:
    starts: list[int] = []
    for axis in range(3):
        half = patch_size[axis] // 2
        raw_start = center[axis] - half
        clamped_start = max(0, min(raw_start, volume.shape[axis] - patch_size[axis]))
        starts.append(clamped_start)
    return volume[
        starts[0] : starts[0] + patch_size[0],
        starts[1] : starts[1] + patch_size[1],
        starts[2] : starts[2] + patch_size[2],
    ]


def sample_foreground_aware_patches(
    image: np.ndarray,
    label_binary: np.ndarray,
    *,
    config: Phase8DefinitiveConfig,
    positive_count: int,
    negative_count: int,
    rng: np.random.Generator,
) -> list[Phase8DefinitivePatch]:
    """Sample foreground-biased positive/negative patches at the locked patch size.

    Positive patches are centered on a randomly chosen foreground voxel and
    are verified to contain at least one foreground voxel. Negative patches
    are centered on a randomly chosen background voxel and are retried (up
    to a small fixed bound) until an all-background crop is found, since a
    background-centered crop can still overlap nearby foreground. No padding
    is performed: ``image.shape`` must be at least ``config.patch_size`` in
    every dimension.
    """

    patch_size = config.patch_size
    if image.shape != label_binary.shape:
        raise Phase8DefinitivePipelineSamplingError(
            f"image shape {image.shape} does not match label shape {label_binary.shape}."
        )
    for axis in range(3):
        if image.shape[axis] < patch_size[axis]:
            raise Phase8DefinitivePipelineSamplingError(
                f"volume shape {image.shape} is smaller than patch_size {patch_size} "
                f"along axis {axis}."
            )

    patches: list[Phase8DefinitivePatch] = []

    if positive_count > 0:
        foreground_coordinates = np.argwhere(label_binary)
        if foreground_coordinates.shape[0] == 0:
            raise Phase8DefinitivePipelineSamplingError(
                "no foreground voxels are present in label_binary; cannot sample positive patches."
            )
        for _ in range(positive_count):
            index = int(rng.integers(0, foreground_coordinates.shape[0]))
            center = cast(
                tuple[int, int, int], tuple(int(v) for v in foreground_coordinates[index])
            )
            image_patch = _crop_patch(image, center=center, patch_size=patch_size)
            label_patch = _crop_patch(label_binary, center=center, patch_size=patch_size)
            if not bool(np.any(label_patch)):
                raise Phase8DefinitivePipelineSamplingError(
                    "positive patch centered on a foreground voxel unexpectedly "
                    "contains no foreground; this should be unreachable."
                )
            patches.append(
                Phase8DefinitivePatch(
                    image_patch=image_patch, label_patch=label_patch, is_positive=True
                )
            )

    if negative_count > 0:
        background_coordinates = np.argwhere(~label_binary)
        if background_coordinates.shape[0] == 0:
            raise Phase8DefinitivePipelineSamplingError(
                "no background voxels are present in label_binary; cannot sample negative patches."
            )
        for _ in range(negative_count):
            found = False
            for _attempt in range(_MAX_NEGATIVE_PATCH_ATTEMPTS):
                index = int(rng.integers(0, background_coordinates.shape[0]))
                center = cast(
                    tuple[int, int, int], tuple(int(v) for v in background_coordinates[index])
                )
                image_patch = _crop_patch(image, center=center, patch_size=patch_size)
                label_patch = _crop_patch(label_binary, center=center, patch_size=patch_size)
                if not bool(np.any(label_patch)):
                    patches.append(
                        Phase8DefinitivePatch(
                            image_patch=image_patch, label_patch=label_patch, is_positive=False
                        )
                    )
                    found = True
                    break
            if not found:
                raise Phase8DefinitivePipelineSamplingError(
                    "could not find an all-background patch within "
                    f"{_MAX_NEGATIVE_PATCH_ATTEMPTS} attempts; failing closed."
                )

    return patches


# ---------------------------------------------------------------------------
# D. Training runner
# ---------------------------------------------------------------------------


def _import_torch() -> Any:
    import torch  # type: ignore[import-not-found]

    return torch


def _import_monai() -> Any:
    import monai  # type: ignore[import-not-found]

    return monai


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveTrainCase:
    """One preprocessed training case in memory."""

    case_id: str
    image: np.ndarray
    label_binary: np.ndarray


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveTrainingStepResult:
    """Per-step training diagnostics."""

    step_index: int
    loss_value: float
    loss_finite: bool
    gradient_finite: bool


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveTrainingRunResult:
    """Result of a (possibly early-stopped, fail-closed) training run."""

    step_results: tuple[Phase8DefinitiveTrainingStepResult, ...]
    all_steps_finite: bool
    model_state_dict: Any


def _build_definitive_segresnet_model(*, torch: Any, monai: Any) -> Any:
    # Architecture hyperparameters below (init_filters, blocks_down/up,
    # upsample_mode) are not scientifically locked by
    # PHASE8-DEFINITIVE-CONFIG-DESIGN-V1; they are reused as engineering
    # reference from the bounded, non-definitive pilot
    # (definitive_training_pilot.py's ``_build_pilot_segresnet_model``), not
    # a new scientific decision.
    return monai.networks.nets.SegResNet(
        spatial_dims=3,
        init_filters=8,
        in_channels=1,
        out_channels=2,
        dropout_prob=None,
        blocks_down=(1, 1, 1),
        blocks_up=(1, 1),
        upsample_mode="deconv",
    ).to(torch.device("cpu"))


def _patch_to_tensors(patch: Phase8DefinitivePatch, *, torch: Any) -> tuple[Any, Any]:
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(patch.image_patch.astype(np.float32))[None, None, ...]
    ).to(dtype=torch.float32)
    label_tensor = torch.from_numpy(
        np.ascontiguousarray(patch.label_patch.astype(np.int64))[None, None, ...]
    ).to(dtype=torch.long)
    return image_tensor, label_tensor


def run_definitive_training(
    train_cases: Sequence[Phase8DefinitiveTrainCase],
    *,
    config: Phase8DefinitiveConfig,
    max_steps: int,
    rng_seed: int | None = None,
) -> Phase8DefinitiveTrainingRunResult:
    """Run up to ``max_steps`` fail-closed AdamW/DiceCE optimizer steps on CPU.

    At each step, one positive patch (from a case that contains tumor
    foreground) and one negative patch (from any case) are sampled at the
    locked 1:1 ratio and stacked into a batch. If the loss or any gradient
    is non-finite, the loop stops immediately without applying that
    optimizer step, and the returned result reports
    ``all_steps_finite=False``. ``max_steps`` and the case list are plain
    caller-supplied arguments; no training budget is baked into this
    function or into :class:`Phase8DefinitiveConfig`.
    """

    if not train_cases:
        raise Phase8DefinitivePipelineRuntimeError(
            "run_definitive_training requires at least one train case."
        )
    if max_steps < 1:
        raise Phase8DefinitivePipelineRuntimeError("max_steps must be at least 1.")

    positive_cases = [case for case in train_cases if bool(np.any(case.label_binary))]
    if not positive_cases:
        raise Phase8DefinitivePipelineRuntimeError(
            "no train case contains tumor foreground; at least one is required."
        )

    torch = _import_torch()
    monai = _import_monai()

    seed = config.seed if rng_seed is None else rng_seed
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    sampling_rng = np.random.default_rng(seed)

    model = _build_definitive_segresnet_model(torch=torch, monai=monai)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)

    step_results: list[Phase8DefinitiveTrainingStepResult] = []
    all_finite = True

    for step_index in range(max_steps):
        positive_case = positive_cases[step_index % len(positive_cases)]
        negative_case_index = int(sampling_rng.integers(0, len(train_cases)))
        negative_case = train_cases[negative_case_index]

        positive_patches = sample_foreground_aware_patches(
            positive_case.image,
            positive_case.label_binary,
            config=config,
            positive_count=1,
            negative_count=0,
            rng=sampling_rng,
        )
        negative_patches = sample_foreground_aware_patches(
            negative_case.image,
            negative_case.label_binary,
            config=config,
            positive_count=0,
            negative_count=1,
            rng=sampling_rng,
        )

        image_tensors = []
        label_tensors = []
        for patch in (*positive_patches, *negative_patches):
            image_tensor, label_tensor = _patch_to_tensors(patch, torch=torch)
            image_tensors.append(image_tensor)
            label_tensors.append(label_tensor)
        batch_image = torch.cat(image_tensors, dim=0)
        batch_label = torch.cat(label_tensors, dim=0)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_image)
        loss = loss_function(logits, batch_label)
        loss_value = float(loss.item())
        loss_finite = math.isfinite(loss_value)
        if not loss_finite:
            step_results.append(
                Phase8DefinitiveTrainingStepResult(
                    step_index=step_index,
                    loss_value=loss_value,
                    loss_finite=False,
                    gradient_finite=False,
                )
            )
            all_finite = False
            break

        loss.backward()
        gradient_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        if not gradient_finite:
            step_results.append(
                Phase8DefinitiveTrainingStepResult(
                    step_index=step_index,
                    loss_value=loss_value,
                    loss_finite=True,
                    gradient_finite=False,
                )
            )
            all_finite = False
            break

        optimizer.step()
        step_results.append(
            Phase8DefinitiveTrainingStepResult(
                step_index=step_index,
                loss_value=loss_value,
                loss_finite=True,
                gradient_finite=True,
            )
        )

    return Phase8DefinitiveTrainingRunResult(
        step_results=tuple(step_results),
        all_steps_finite=all_finite,
        model_state_dict=model.state_dict(),
    )


# ---------------------------------------------------------------------------
# E. Validation runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveValidationCase:
    """One preprocessed validation case in memory."""

    case_id: str
    image: np.ndarray
    label_binary: np.ndarray


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveValidationRunResult:
    """Tumor Dice/IoU for one full-volume sliding-window validation pass."""

    case_id: str
    tumor_dice: float
    tumor_iou: float
    prediction_finite: bool


def run_definitive_validation(
    model: Any,
    case: Phase8DefinitiveValidationCase,
    *,
    config: Phase8DefinitiveConfig,
) -> Phase8DefinitiveValidationRunResult:
    """Run full-volume sliding-window inference and compute tumor Dice/IoU.

    If the model's output is not finite, metrics are not computed on
    garbage: the result reports ``prediction_finite=False`` and
    ``tumor_dice``/``tumor_iou`` as ``float("nan")``.
    """

    from protoem_ct.evaluation.metrics import (
        compute_binary_confusion_counts,
        dice_from_counts,
        iou_from_counts,
    )

    torch = _import_torch()
    monai = _import_monai()

    image_tensor = torch.from_numpy(
        np.ascontiguousarray(case.image.astype(np.float32))[None, None, ...]
    ).to(dtype=torch.float32, device=torch.device("cpu"))

    model.eval()
    with torch.no_grad():
        logits = monai.inferers.sliding_window_inference(
            image_tensor,
            roi_size=config.sliding_window_roi_size,
            sw_batch_size=config.sliding_window_batch_size,
            predictor=model,
            overlap=config.sliding_window_overlap,
            mode="constant",
            progress=False,
            sw_device=torch.device("cpu"),
            device=torch.device("cpu"),
        )
        probabilities = torch.softmax(logits, dim=1)

    prediction_finite = bool(torch.isfinite(probabilities).all().item())
    if not prediction_finite:
        return Phase8DefinitiveValidationRunResult(
            case_id=case.case_id,
            tumor_dice=float("nan"),
            tumor_iou=float("nan"),
            prediction_finite=False,
        )

    tumor_probability = probabilities[0, 1, ...]
    prediction_mask = (tumor_probability >= config.inference_threshold).cpu().numpy().astype(bool)

    counts = compute_binary_confusion_counts(case.label_binary, prediction_mask)
    dice = dice_from_counts(counts)
    iou = iou_from_counts(counts)

    return Phase8DefinitiveValidationRunResult(
        case_id=case.case_id,
        tumor_dice=dice,
        tumor_iou=iou,
        prediction_finite=True,
    )


# ---------------------------------------------------------------------------
# F. Checkpoint metadata
# ---------------------------------------------------------------------------


def build_definitive_checkpoint_metadata(
    *,
    checkpoint_sha256: str,
    checkpoint_byte_size: int,
    config: Phase8DefinitiveConfig,
    development_manifest_hash: str,
    development_split_hash: str,
    originating_git_commit: str,
    package_environment_reference: ArtifactReference,
    completion_status: str,
    freeze_eligible: bool = False,
    preprocessing_evidence_hash: str,
    seed: int | None = None,
    training_adaptation_mode: str = "baseline_training",
    failure_codes: tuple[str, ...] = (),
    validation_metric_artifact_reference: ArtifactReference | None = None,
) -> Phase8CheckpointMetadata:
    """Build :class:`Phase8CheckpointMetadata` from a Package-A training run.

    This wraps the existing, unmodified
    :mod:`protoem_ct.external.internal_evidence` checkpoint schema; it does
    not add a new checkpoint schema. Package A produces only
    ``freeze_eligible=False`` checkpoints -- a caller passing
    ``freeze_eligible=True`` is rejected fail-closed, because a Package-A
    run is implementation-only and has no accompanying real training run.
    """

    if freeze_eligible is not False:
        raise Phase8DefinitivePipelineCheckpointError(
            "build_definitive_checkpoint_metadata never produces freeze_eligible=True "
            "checkpoints; Package A is implementation-only."
        )

    resolved_seed = config.seed if seed is None else seed

    # Phase8CheckpointMetadata is frozen and self-validates its own hash at
    # construction time, so the identity payload must be hashed before the
    # real instance exists. Build a plain attribute holder exposing exactly
    # the fields ``hash_phase8_checkpoint_metadata`` reads, reusing that
    # existing canonical hashing function rather than re-deriving the
    # identity payload by hand (avoids drift from internal_evidence.py).
    class _MetadataIdentityStandIn:
        pass

    stand_in = _MetadataIdentityStandIn()
    stand_in.schema_name = PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME  # type: ignore[attr-defined]
    stand_in.schema_version = PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION  # type: ignore[attr-defined]
    stand_in.checkpoint_sha256 = checkpoint_sha256  # type: ignore[attr-defined]
    stand_in.checkpoint_byte_size = checkpoint_byte_size  # type: ignore[attr-defined]
    stand_in.serialization_format = "torch_state_dict"  # type: ignore[attr-defined]
    stand_in.model_family = config.model_family  # type: ignore[attr-defined]
    stand_in.candidate_id = "phase8_definitive_segresnet"  # type: ignore[attr-defined]
    stand_in.seed = resolved_seed  # type: ignore[attr-defined]
    stand_in.training_adaptation_mode = training_adaptation_mode  # type: ignore[attr-defined]
    stand_in.originating_git_commit = originating_git_commit  # type: ignore[attr-defined]
    stand_in.package_environment_reference = package_environment_reference  # type: ignore[attr-defined]
    stand_in.training_config_hash = config.config_hash  # type: ignore[attr-defined]
    stand_in.preprocessing_evidence_hash = preprocessing_evidence_hash  # type: ignore[attr-defined]
    stand_in.development_manifest_hash = development_manifest_hash  # type: ignore[attr-defined]
    stand_in.development_split_hash = development_split_hash  # type: ignore[attr-defined]
    stand_in.validation_metric_artifact_reference = (  # type: ignore[attr-defined]
        validation_metric_artifact_reference
    )
    stand_in.completion_status = completion_status  # type: ignore[attr-defined]
    stand_in.failure_codes = failure_codes  # type: ignore[attr-defined]
    stand_in.no_external_data = True  # type: ignore[attr-defined]
    stand_in.freeze_eligible = False  # type: ignore[attr-defined]
    checkpoint_metadata_hash = hash_phase8_checkpoint_metadata(cast(Any, stand_in))

    return Phase8CheckpointMetadata(
        schema_name=PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME,
        schema_version=PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_size=checkpoint_byte_size,
        serialization_format="torch_state_dict",
        model_family=config.model_family,
        candidate_id="phase8_definitive_segresnet",
        seed=resolved_seed,
        training_adaptation_mode=training_adaptation_mode,
        originating_git_commit=originating_git_commit,
        package_environment_reference=package_environment_reference,
        training_config_hash=config.config_hash,
        preprocessing_evidence_hash=preprocessing_evidence_hash,
        development_manifest_hash=development_manifest_hash,
        development_split_hash=development_split_hash,
        validation_metric_artifact_reference=validation_metric_artifact_reference,
        device_type=config.device_type,
        amp_state="disabled",
        completion_status=completion_status,
        failure_codes=failure_codes,
        no_external_data=True,
        freeze_eligible=False,
        checkpoint_metadata_hash=checkpoint_metadata_hash,
    )
