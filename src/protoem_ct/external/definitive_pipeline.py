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

import copy
import hashlib
import io
import math
import multiprocessing
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, Final, Protocol, cast

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.data.phase2_paths import validate_explicit_external_output_root
from protoem_ct.external.internal_evidence import (
    PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME,
    PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
    ArtifactReference,
    Phase8CheckpointMetadata,
    hash_phase8_checkpoint_metadata,
)

_SHA256_HEX_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

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

# Locked SegResNet architecture values (approved as part of
# PHASE8-DEFINITIVE-CONFIG-DESIGN-V1; hash-bound via Phase8DefinitiveConfig).
REQUIRED_SPATIAL_DIMS: Final[int] = 3
REQUIRED_IN_CHANNELS: Final[int] = 1
REQUIRED_OUT_CHANNELS: Final[int] = 2
REQUIRED_INIT_FILTERS: Final[int] = 8
REQUIRED_BLOCKS_DOWN: Final[tuple[int, ...]] = (1, 1, 1)
REQUIRED_BLOCKS_UP: Final[tuple[int, ...]] = (1, 1)
REQUIRED_DROPOUT_PROB: Final[float | None] = None
REQUIRED_UPSAMPLE_MODE: Final[str] = "deconv"

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


class Phase8DefinitiveCheckpointPublicationError(Phase8DefinitivePipelineError):
    """Raised when publishing a checkpoint snapshot violates a hard invariant."""


class Phase8DefinitivePipelineValidationError(Phase8DefinitivePipelineError):
    """Raised when full development-validation orchestration violates a hard invariant."""


class Phase8DefinitivePipelineSelectionError(Phase8DefinitivePipelineError):
    """Raised when checkpoint selection violates a hard invariant; fails closed."""


class Phase8DefinitiveTrainingWatchdogTimeoutError(Phase8DefinitivePipelineRuntimeError):
    """Raised when the process-level training watchdog kills a blocked child process.

    No partial success is ever surfaced through this error: the caller must
    make a fresh call to retry, and no checkpoint selection is ever performed
    for the call that timed out.
    """


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
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    init_filters: int,
    blocks_down: Sequence[int],
    blocks_up: Sequence[int],
    dropout_prob: float | None,
    upsample_mode: str,
) -> dict[str, JsonValue]:
    return {
        "amp_enabled": amp_enabled,
        "augmentation_policy": augmentation_policy,
        "blocks_down": list(blocks_down),
        "blocks_up": list(blocks_up),
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "design_identifier": design_identifier,
        "device_type": device_type,
        "dropout_prob": dropout_prob,
        "external_data_used_for_selection": external_data_used_for_selection,
        "hu_clip_max": hu_clip_max,
        "hu_clip_min": hu_clip_min,
        "image_interpolation_policy": image_interpolation_policy,
        "in_channels": in_channels,
        "inference_threshold": inference_threshold,
        "init_filters": init_filters,
        "intensity_scale_max": intensity_scale_max,
        "intensity_scale_min": intensity_scale_min,
        "internal_test_used_for_selection": internal_test_used_for_selection,
        "label_interpolation_policy": label_interpolation_policy,
        "learning_rate": learning_rate,
        "loss_name": loss_name,
        "model_family": model_family,
        "optimizer_name": optimizer_name,
        "orientation_policy": orientation_policy,
        "out_channels": out_channels,
        "patch_size": list(patch_size),
        "positive_negative_ratio": list(positive_negative_ratio),
        "sampling_policy": sampling_policy,
        "seed": seed,
        "sliding_window_batch_size": sliding_window_batch_size,
        "sliding_window_overlap": sliding_window_overlap,
        "sliding_window_roi_size": list(sliding_window_roi_size),
        "spatial_dims": spatial_dims,
        "tumor_raw_label_value": tumor_raw_label_value,
        "upsample_mode": upsample_mode,
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
    spatial_dims: int
    in_channels: int
    out_channels: int
    init_filters: int
    blocks_down: tuple[int, ...]
    blocks_up: tuple[int, ...]
    dropout_prob: float | None
    upsample_mode: str
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
        if self.spatial_dims != REQUIRED_SPATIAL_DIMS:
            raise Phase8DefinitivePipelineConfigError(
                f"spatial_dims must equal {REQUIRED_SPATIAL_DIMS!r}."
            )
        if self.in_channels != REQUIRED_IN_CHANNELS:
            raise Phase8DefinitivePipelineConfigError(
                f"in_channels must equal {REQUIRED_IN_CHANNELS!r}."
            )
        if self.out_channels != REQUIRED_OUT_CHANNELS:
            raise Phase8DefinitivePipelineConfigError(
                f"out_channels must equal {REQUIRED_OUT_CHANNELS!r}."
            )
        if self.init_filters != REQUIRED_INIT_FILTERS:
            raise Phase8DefinitivePipelineConfigError(
                f"init_filters must equal {REQUIRED_INIT_FILTERS!r}."
            )
        object.__setattr__(self, "blocks_down", tuple(int(v) for v in self.blocks_down))
        if self.blocks_down != REQUIRED_BLOCKS_DOWN:
            raise Phase8DefinitivePipelineConfigError(
                f"blocks_down must equal {REQUIRED_BLOCKS_DOWN!r}, got {self.blocks_down!r}."
            )
        object.__setattr__(self, "blocks_up", tuple(int(v) for v in self.blocks_up))
        if self.blocks_up != REQUIRED_BLOCKS_UP:
            raise Phase8DefinitivePipelineConfigError(
                f"blocks_up must equal {REQUIRED_BLOCKS_UP!r}, got {self.blocks_up!r}."
            )
        if self.dropout_prob != REQUIRED_DROPOUT_PROB:
            raise Phase8DefinitivePipelineConfigError(
                f"dropout_prob must equal {REQUIRED_DROPOUT_PROB!r}."
            )
        if self.upsample_mode != REQUIRED_UPSAMPLE_MODE:
            raise Phase8DefinitivePipelineConfigError(
                f"upsample_mode must equal {REQUIRED_UPSAMPLE_MODE!r}."
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
                spatial_dims=self.spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                init_filters=self.init_filters,
                blocks_down=self.blocks_down,
                blocks_up=self.blocks_up,
                dropout_prob=self.dropout_prob,
                upsample_mode=self.upsample_mode,
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
        spatial_dims=REQUIRED_SPATIAL_DIMS,
        in_channels=REQUIRED_IN_CHANNELS,
        out_channels=REQUIRED_OUT_CHANNELS,
        init_filters=REQUIRED_INIT_FILTERS,
        blocks_down=REQUIRED_BLOCKS_DOWN,
        blocks_up=REQUIRED_BLOCKS_UP,
        dropout_prob=REQUIRED_DROPOUT_PROB,
        upsample_mode=REQUIRED_UPSAMPLE_MODE,
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
        spatial_dims=REQUIRED_SPATIAL_DIMS,
        in_channels=REQUIRED_IN_CHANNELS,
        out_channels=REQUIRED_OUT_CHANNELS,
        init_filters=REQUIRED_INIT_FILTERS,
        blocks_down=REQUIRED_BLOCKS_DOWN,
        blocks_up=REQUIRED_BLOCKS_UP,
        dropout_prob=REQUIRED_DROPOUT_PROB,
        upsample_mode=REQUIRED_UPSAMPLE_MODE,
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
class Phase8DefinitiveTrainCaseReference:
    """A lightweight, non-medical reference to one training case.

    Holds only the anonymous ``case_id`` (mirroring the ``case_<hex>``-style
    anonymous identifiers used elsewhere in this package); it never embeds
    image/label arrays and never holds a raw patient identifier or a
    filesystem path. Used by :func:`run_definitive_training_from_references`
    so a caller never has to materialize every case's full-volume arrays in
    memory at once -- only ``case_id`` strings are held for the whole case
    pool, and the actual arrays are fetched one case at a time, on demand,
    via a caller-supplied :class:`Phase8DefinitiveCaseLoader`.
    """

    case_id: str


class Phase8DefinitiveCaseLoader(Protocol):
    """Loads exactly one fully preprocessed training case, given its case_id.

    Implementations are supplied by the caller and are responsible for any
    dataset discovery and filesystem/dataset access; this module never
    performs such access itself and never invokes a loader with anything
    other than a ``case_id`` string.
    """

    def __call__(self, case_id: str) -> Phase8DefinitiveTrainCase: ...


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


def _build_definitive_segresnet_model(
    *, torch: Any, monai: Any, config: Phase8DefinitiveConfig
) -> Any:
    # Architecture hyperparameters below (spatial_dims, in/out_channels,
    # init_filters, blocks_down/up, dropout_prob, upsample_mode) are
    # scientifically locked by PHASE8-DEFINITIVE-CONFIG-DESIGN-V1 and are
    # part of ``config.config_hash``; they are read from ``config`` rather
    # than hard-coded so tampering with any of them is caught fail-closed by
    # :meth:`Phase8DefinitiveConfig.__post_init__` before this function ever
    # runs.
    return monai.networks.nets.SegResNet(
        spatial_dims=config.spatial_dims,
        init_filters=config.init_filters,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dropout_prob=config.dropout_prob,
        blocks_down=config.blocks_down,
        blocks_up=config.blocks_up,
        upsample_mode=config.upsample_mode,
    ).to(torch.device("cpu"))


def _patch_to_tensors(patch: Phase8DefinitivePatch, *, torch: Any) -> tuple[Any, Any]:
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(patch.image_patch.astype(np.float32))[None, None, ...]
    ).to(dtype=torch.float32)
    label_tensor = torch.from_numpy(
        np.ascontiguousarray(patch.label_patch.astype(np.int64))[None, None, ...]
    ).to(dtype=torch.long)
    return image_tensor, label_tensor


def _run_definitive_training_step(
    *,
    step_index: int,
    positive_case: Phase8DefinitiveTrainCase,
    negative_case: Phase8DefinitiveTrainCase,
    config: Phase8DefinitiveConfig,
    model: Any,
    optimizer: Any,
    loss_function: Any,
    torch: Any,
    sampling_rng: np.random.Generator,
) -> Phase8DefinitiveTrainingStepResult:
    """Run exactly one fail-closed AdamW/DiceCE optimizer step on CPU.

    This is the single shared training-step implementation: it samples one
    positive patch from ``positive_case`` and one negative patch from
    ``negative_case`` at the locked 1:1 ratio, runs the forward/backward
    pass, and applies the optimizer step unless the loss or any gradient is
    non-finite. Both :func:`run_definitive_training` (small in-memory case
    list) and :func:`run_definitive_training_from_references` (memory-safe,
    loader-based case-by-case path) call this helper so there is exactly one
    training-step implementation, not two parallel copies. The caller is
    responsible for choosing which cases to pass in for this step and for
    not retaining them beyond the call.
    """

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
        return Phase8DefinitiveTrainingStepResult(
            step_index=step_index,
            loss_value=loss_value,
            loss_finite=False,
            gradient_finite=False,
        )

    loss.backward()
    gradient_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    if not gradient_finite:
        return Phase8DefinitiveTrainingStepResult(
            step_index=step_index,
            loss_value=loss_value,
            loss_finite=True,
            gradient_finite=False,
        )

    optimizer.step()
    return Phase8DefinitiveTrainingStepResult(
        step_index=step_index,
        loss_value=loss_value,
        loss_finite=True,
        gradient_finite=True,
    )


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

    model = _build_definitive_segresnet_model(torch=torch, monai=monai, config=config)
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

        step_result = _run_definitive_training_step(
            step_index=step_index,
            positive_case=positive_case,
            negative_case=negative_case,
            config=config,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            torch=torch,
            sampling_rng=sampling_rng,
        )
        step_results.append(step_result)
        if not (step_result.loss_finite and step_result.gradient_finite):
            all_finite = False
            break

    return Phase8DefinitiveTrainingRunResult(
        step_results=tuple(step_results),
        all_steps_finite=all_finite,
        model_state_dict=model.state_dict(),
    )


def run_definitive_training_from_references(
    case_references: Sequence[Phase8DefinitiveTrainCaseReference],
    *,
    positive_case_ids: frozenset[str] | set[str],
    loader: Phase8DefinitiveCaseLoader | Callable[[str], Phase8DefinitiveTrainCase],
    config: Phase8DefinitiveConfig,
    max_steps: int,
    rng_seed: int | None = None,
) -> Phase8DefinitiveTrainingRunResult:
    """Memory-safe variant of :func:`run_definitive_training` for large case pools.

    Unlike :func:`run_definitive_training`, the caller never has to
    materialize every case's full-volume arrays in memory at once: only the
    lightweight ``case_id``-only ``case_references`` are held for the whole
    pool (e.g. all 91 development TRAIN cases), and ``positive_case_ids`` is
    a caller-supplied set of case IDs known (from existing Phase 2
    lesion-summary artifacts) to contain tumor foreground -- this function
    never inspects array content to decide positivity, since it never loads
    a case it does not need.

    For each optimizer step, the positive and negative case IDs are chosen
    with the exact same deterministic scheme as
    :func:`run_definitive_training` (``positive_refs[step_index %
    len(positive_refs)]`` and ``sampling_rng.integers(0,
    len(case_references))``), and only the case(s) required for that step
    are fetched via ``loader``. If the chosen positive and negative case IDs
    are identical for a step, ``loader`` is invoked exactly once for that
    case that step. The loaded full case(s) are held only in local
    variables scoped to that single iteration -- there is no persistent,
    cross-step cache, so at most two full preprocessed cases (one if
    positive and negative coincide) are logically alive in memory at any
    point during a step. This function reuses
    :func:`_run_definitive_training_step` (the same shared step
    implementation used by :func:`run_definitive_training`); it does not
    duplicate the AdamW/DiceCE/backward/step logic.

    This function performs no dataset discovery and no filesystem access of
    its own; ``loader`` is entirely caller-supplied.
    """

    if not case_references:
        raise Phase8DefinitivePipelineRuntimeError(
            "run_definitive_training_from_references requires at least one case reference."
        )
    if max_steps < 1:
        raise Phase8DefinitivePipelineRuntimeError("max_steps must be at least 1.")

    known_case_ids = {reference.case_id for reference in case_references}
    for positive_case_id in positive_case_ids:
        if positive_case_id not in known_case_ids:
            raise Phase8DefinitivePipelineRuntimeError(
                f"positive_case_ids contains {positive_case_id!r}, which is not present in "
                "case_references."
            )
    positive_refs = [
        reference for reference in case_references if reference.case_id in positive_case_ids
    ]
    if not positive_refs:
        raise Phase8DefinitivePipelineRuntimeError(
            "no case_references entry is marked in positive_case_ids; at least one is required."
        )

    torch = _import_torch()
    monai = _import_monai()

    seed = config.seed if rng_seed is None else rng_seed
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    sampling_rng = np.random.default_rng(seed)

    model = _build_definitive_segresnet_model(torch=torch, monai=monai, config=config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)

    step_results: list[Phase8DefinitiveTrainingStepResult] = []
    all_finite = True

    for step_index in range(max_steps):
        positive_case_id = positive_refs[step_index % len(positive_refs)].case_id
        negative_case_index = int(sampling_rng.integers(0, len(case_references)))
        negative_case_id = case_references[negative_case_index].case_id

        # Load only what this single step needs. If the two IDs coincide,
        # the loader is invoked once and the same in-memory case object is
        # reused as both positive and negative source for this step; either
        # way, nothing loaded here is retained past this iteration.
        positive_case = loader(positive_case_id)
        negative_case = (
            positive_case if negative_case_id == positive_case_id else loader(negative_case_id)
        )

        step_result = _run_definitive_training_step(
            step_index=step_index,
            positive_case=positive_case,
            negative_case=negative_case,
            config=config,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            torch=torch,
            sampling_rng=sampling_rng,
        )
        step_results.append(step_result)
        del positive_case, negative_case
        if not (step_result.loss_finite and step_result.gradient_finite):
            all_finite = False
            break

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


# ---------------------------------------------------------------------------
# G. Training execution policy (locked, hash-bound)
# ---------------------------------------------------------------------------
#
# This policy governs the *engineering* orchestration of a real definitive
# training run -- how many optimizer steps, which steps are candidate
# checkpoints, and how a checkpoint is selected -- as distinct from the
# *scientific* config above (preprocessing, architecture, optimizer
# hyperparameters). Like ``Phase8DefinitiveConfig``, every field is
# hard-locked: this dataclass is not caller-tunable and fails closed on any
# deviation from the single approved ``PHASE8-DEFINITIVE-TRAINING-POLICY-V1``
# design.

PHASE8_DEFINITIVE_TRAINING_POLICY_IDENTIFIER: Final[str] = "PHASE8-DEFINITIVE-TRAINING-POLICY-V1"

_POLICY_TOTAL_OPTIMIZER_STEPS: Final[int] = 500
_POLICY_CANDIDATE_CHECKPOINT_STEPS: Final[tuple[int, int]] = (250, 500)
_POLICY_EARLY_STOPPING: Final[bool] = False
_POLICY_RESUME_POLICY: Final[str] = "none"
_POLICY_MAXIMUM_WALL_CLOCK_SECONDS: Final[float] = 36000.0
_POLICY_VALIDATION_EXPECTED_CASE_COUNT: Final[int] = 20
_POLICY_FULL_VALIDATION_REQUIRED_FOR_EVERY_CANDIDATE: Final[bool] = True
_POLICY_SUBSET_VALIDATION_FOR_CHECKPOINT_SELECTION: Final[bool] = False
_POLICY_CHECKPOINT_SELECTION_METRIC: Final[str] = "mean_tumor_dice"
_POLICY_TIE_BREAK_POLICY: Final[str] = "earliest_checkpoint_step"
_POLICY_INTERNAL_TEST_USED_FOR_SELECTION: Final[bool] = False
_POLICY_EXTERNAL_DATA_USED_FOR_SELECTION: Final[bool] = False
_POLICY_EXTERNAL_LABELS_USED_FOR_SELECTION: Final[bool] = False


def _training_policy_payload(
    *,
    policy_identifier: str,
    total_optimizer_steps: int,
    candidate_checkpoint_steps: Sequence[int],
    early_stopping: bool,
    resume_policy: str,
    maximum_wall_clock_seconds: float,
    validation_expected_case_count: int,
    full_validation_required_for_every_candidate: bool,
    subset_validation_for_checkpoint_selection: bool,
    checkpoint_selection_metric: str,
    tie_break_policy: str,
    internal_test_used_for_selection: bool,
    external_data_used_for_selection: bool,
    external_labels_used_for_selection: bool,
) -> dict[str, JsonValue]:
    return {
        "candidate_checkpoint_steps": list(candidate_checkpoint_steps),
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "early_stopping": early_stopping,
        "external_data_used_for_selection": external_data_used_for_selection,
        "external_labels_used_for_selection": external_labels_used_for_selection,
        "full_validation_required_for_every_candidate": (
            full_validation_required_for_every_candidate
        ),
        "internal_test_used_for_selection": internal_test_used_for_selection,
        "maximum_wall_clock_seconds": maximum_wall_clock_seconds,
        "policy_identifier": policy_identifier,
        "resume_policy": resume_policy,
        "subset_validation_for_checkpoint_selection": (subset_validation_for_checkpoint_selection),
        "tie_break_policy": tie_break_policy,
        "total_optimizer_steps": total_optimizer_steps,
        "validation_expected_case_count": validation_expected_case_count,
    }


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveTrainingExecutionPolicy:
    """Immutable, LOCKED training-execution/checkpoint-selection policy.

    Every field is hard-locked to the value approved under
    ``PHASE8-DEFINITIVE-TRAINING-POLICY-V1``. This dataclass is not
    caller-tunable: :meth:`__post_init__` fails closed with
    :class:`Phase8DefinitivePipelineConfigError` if any field differs from
    its required value. The only public constructor is
    :func:`build_definitive_training_execution_policy_v1`.
    """

    policy_identifier: str
    total_optimizer_steps: int
    candidate_checkpoint_steps: tuple[int, int]
    early_stopping: bool
    resume_policy: str
    maximum_wall_clock_seconds: float
    validation_expected_case_count: int
    full_validation_required_for_every_candidate: bool
    subset_validation_for_checkpoint_selection: bool
    checkpoint_selection_metric: str
    tie_break_policy: str
    internal_test_used_for_selection: bool
    external_data_used_for_selection: bool
    external_labels_used_for_selection: bool
    policy_hash: str

    def __post_init__(self) -> None:
        if self.policy_identifier != PHASE8_DEFINITIVE_TRAINING_POLICY_IDENTIFIER:
            raise Phase8DefinitivePipelineConfigError(
                f"policy_identifier must equal {PHASE8_DEFINITIVE_TRAINING_POLICY_IDENTIFIER!r}."
            )
        if self.total_optimizer_steps != _POLICY_TOTAL_OPTIMIZER_STEPS:
            raise Phase8DefinitivePipelineConfigError(
                f"total_optimizer_steps must equal {_POLICY_TOTAL_OPTIMIZER_STEPS!r}."
            )
        object.__setattr__(
            self,
            "candidate_checkpoint_steps",
            tuple(int(v) for v in self.candidate_checkpoint_steps),
        )
        if self.candidate_checkpoint_steps != _POLICY_CANDIDATE_CHECKPOINT_STEPS:
            raise Phase8DefinitivePipelineConfigError(
                f"candidate_checkpoint_steps must equal "
                f"{_POLICY_CANDIDATE_CHECKPOINT_STEPS!r}, got "
                f"{self.candidate_checkpoint_steps!r}."
            )
        if self.early_stopping is not _POLICY_EARLY_STOPPING:
            raise Phase8DefinitivePipelineConfigError(
                f"early_stopping must be {_POLICY_EARLY_STOPPING!r}."
            )
        if self.resume_policy != _POLICY_RESUME_POLICY:
            raise Phase8DefinitivePipelineConfigError(
                f"resume_policy must equal {_POLICY_RESUME_POLICY!r}."
            )
        if self.maximum_wall_clock_seconds != _POLICY_MAXIMUM_WALL_CLOCK_SECONDS:
            raise Phase8DefinitivePipelineConfigError(
                f"maximum_wall_clock_seconds must equal {_POLICY_MAXIMUM_WALL_CLOCK_SECONDS!r}."
            )
        if self.validation_expected_case_count != _POLICY_VALIDATION_EXPECTED_CASE_COUNT:
            raise Phase8DefinitivePipelineConfigError(
                f"validation_expected_case_count must equal "
                f"{_POLICY_VALIDATION_EXPECTED_CASE_COUNT!r}."
            )
        if (
            self.full_validation_required_for_every_candidate
            is not _POLICY_FULL_VALIDATION_REQUIRED_FOR_EVERY_CANDIDATE
        ):
            raise Phase8DefinitivePipelineConfigError(
                "full_validation_required_for_every_candidate must be "
                f"{_POLICY_FULL_VALIDATION_REQUIRED_FOR_EVERY_CANDIDATE!r}."
            )
        if (
            self.subset_validation_for_checkpoint_selection
            is not _POLICY_SUBSET_VALIDATION_FOR_CHECKPOINT_SELECTION
        ):
            raise Phase8DefinitivePipelineConfigError(
                "subset_validation_for_checkpoint_selection must be "
                f"{_POLICY_SUBSET_VALIDATION_FOR_CHECKPOINT_SELECTION!r}."
            )
        if self.checkpoint_selection_metric != _POLICY_CHECKPOINT_SELECTION_METRIC:
            raise Phase8DefinitivePipelineConfigError(
                f"checkpoint_selection_metric must equal {_POLICY_CHECKPOINT_SELECTION_METRIC!r}."
            )
        if self.tie_break_policy != _POLICY_TIE_BREAK_POLICY:
            raise Phase8DefinitivePipelineConfigError(
                f"tie_break_policy must equal {_POLICY_TIE_BREAK_POLICY!r}."
            )
        if self.internal_test_used_for_selection is not _POLICY_INTERNAL_TEST_USED_FOR_SELECTION:
            raise Phase8DefinitivePipelineConfigError(
                "internal_test_used_for_selection must be "
                f"{_POLICY_INTERNAL_TEST_USED_FOR_SELECTION!r}."
            )
        if self.external_data_used_for_selection is not _POLICY_EXTERNAL_DATA_USED_FOR_SELECTION:
            raise Phase8DefinitivePipelineConfigError(
                "external_data_used_for_selection must be "
                f"{_POLICY_EXTERNAL_DATA_USED_FOR_SELECTION!r}."
            )
        if (
            self.external_labels_used_for_selection
            is not _POLICY_EXTERNAL_LABELS_USED_FOR_SELECTION
        ):
            raise Phase8DefinitivePipelineConfigError(
                "external_labels_used_for_selection must be "
                f"{_POLICY_EXTERNAL_LABELS_USED_FOR_SELECTION!r}."
            )
        _require_self_hash(
            self.policy_hash,
            _training_policy_payload(
                policy_identifier=self.policy_identifier,
                total_optimizer_steps=self.total_optimizer_steps,
                candidate_checkpoint_steps=self.candidate_checkpoint_steps,
                early_stopping=self.early_stopping,
                resume_policy=self.resume_policy,
                maximum_wall_clock_seconds=self.maximum_wall_clock_seconds,
                validation_expected_case_count=self.validation_expected_case_count,
                full_validation_required_for_every_candidate=(
                    self.full_validation_required_for_every_candidate
                ),
                subset_validation_for_checkpoint_selection=(
                    self.subset_validation_for_checkpoint_selection
                ),
                checkpoint_selection_metric=self.checkpoint_selection_metric,
                tie_break_policy=self.tie_break_policy,
                internal_test_used_for_selection=self.internal_test_used_for_selection,
                external_data_used_for_selection=self.external_data_used_for_selection,
                external_labels_used_for_selection=self.external_labels_used_for_selection,
            ),
        )


def build_definitive_training_execution_policy_v1() -> Phase8DefinitiveTrainingExecutionPolicy:
    """Return the single approved ``PHASE8-DEFINITIVE-TRAINING-POLICY-V1`` instance."""

    payload = _training_policy_payload(
        policy_identifier=PHASE8_DEFINITIVE_TRAINING_POLICY_IDENTIFIER,
        total_optimizer_steps=_POLICY_TOTAL_OPTIMIZER_STEPS,
        candidate_checkpoint_steps=_POLICY_CANDIDATE_CHECKPOINT_STEPS,
        early_stopping=_POLICY_EARLY_STOPPING,
        resume_policy=_POLICY_RESUME_POLICY,
        maximum_wall_clock_seconds=_POLICY_MAXIMUM_WALL_CLOCK_SECONDS,
        validation_expected_case_count=_POLICY_VALIDATION_EXPECTED_CASE_COUNT,
        full_validation_required_for_every_candidate=(
            _POLICY_FULL_VALIDATION_REQUIRED_FOR_EVERY_CANDIDATE
        ),
        subset_validation_for_checkpoint_selection=(
            _POLICY_SUBSET_VALIDATION_FOR_CHECKPOINT_SELECTION
        ),
        checkpoint_selection_metric=_POLICY_CHECKPOINT_SELECTION_METRIC,
        tie_break_policy=_POLICY_TIE_BREAK_POLICY,
        internal_test_used_for_selection=_POLICY_INTERNAL_TEST_USED_FOR_SELECTION,
        external_data_used_for_selection=_POLICY_EXTERNAL_DATA_USED_FOR_SELECTION,
        external_labels_used_for_selection=_POLICY_EXTERNAL_LABELS_USED_FOR_SELECTION,
    )
    return Phase8DefinitiveTrainingExecutionPolicy(
        policy_identifier=PHASE8_DEFINITIVE_TRAINING_POLICY_IDENTIFIER,
        total_optimizer_steps=_POLICY_TOTAL_OPTIMIZER_STEPS,
        candidate_checkpoint_steps=_POLICY_CANDIDATE_CHECKPOINT_STEPS,
        early_stopping=_POLICY_EARLY_STOPPING,
        resume_policy=_POLICY_RESUME_POLICY,
        maximum_wall_clock_seconds=_POLICY_MAXIMUM_WALL_CLOCK_SECONDS,
        validation_expected_case_count=_POLICY_VALIDATION_EXPECTED_CASE_COUNT,
        full_validation_required_for_every_candidate=(
            _POLICY_FULL_VALIDATION_REQUIRED_FOR_EVERY_CANDIDATE
        ),
        subset_validation_for_checkpoint_selection=(
            _POLICY_SUBSET_VALIDATION_FOR_CHECKPOINT_SELECTION
        ),
        checkpoint_selection_metric=_POLICY_CHECKPOINT_SELECTION_METRIC,
        tie_break_policy=_POLICY_TIE_BREAK_POLICY,
        internal_test_used_for_selection=_POLICY_INTERNAL_TEST_USED_FOR_SELECTION,
        external_data_used_for_selection=_POLICY_EXTERNAL_DATA_USED_FOR_SELECTION,
        external_labels_used_for_selection=_POLICY_EXTERNAL_LABELS_USED_FOR_SELECTION,
        policy_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# H. Continuous training with checkpoint snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveContinuousTrainingRunResult:
    """Result of one continuous, non-reinitialized training run with snapshots.

    ``checkpoint_snapshots`` maps ``optimizer_steps_completed`` (e.g. ``250``
    or ``500``) to a deep-copied ``model.state_dict()`` taken immediately
    after that many steps completed; a step whose training was not reached
    (e.g. because an earlier step went non-finite) has no entry. Each
    snapshot is an independent deep copy (see :func:`copy.deepcopy`), so
    later training never mutates an earlier snapshot in place.
    """

    step_results: tuple[Phase8DefinitiveTrainingStepResult, ...]
    all_steps_finite: bool
    checkpoint_snapshots: dict[int, Any]


def _run_definitive_continuous_training_with_snapshots_core(
    case_references: Sequence[Phase8DefinitiveTrainCaseReference],
    *,
    positive_case_ids: frozenset[str] | set[str],
    loader: Phase8DefinitiveCaseLoader | Callable[[str], Phase8DefinitiveTrainCase],
    config: Phase8DefinitiveConfig,
    total_optimizer_steps: int,
    candidate_checkpoint_steps: tuple[int, ...],
    rng_seed: int | None = None,
) -> Phase8DefinitiveContinuousTrainingRunResult:
    """Shared implementation behind the locked, policy-driven public entrypoint.

    This private helper is parametrized on step counts purely so unit tests
    can exercise the snapshot/continuity mechanics with a tiny step budget
    without waiting for 500 real optimizer steps; it is not itself a
    training-budget policy and is never called directly by production code
    with anything other than the locked policy values (see
    :func:`run_definitive_continuous_training_with_snapshots`, the only
    public entrypoint).
    """

    if not case_references:
        raise Phase8DefinitivePipelineRuntimeError(
            "run_definitive_continuous_training_with_snapshots requires at least one "
            "case reference."
        )
    if total_optimizer_steps < 1:
        raise Phase8DefinitivePipelineRuntimeError("total_optimizer_steps must be at least 1.")
    if not candidate_checkpoint_steps:
        raise Phase8DefinitivePipelineRuntimeError("candidate_checkpoint_steps must be nonempty.")

    known_case_ids = {reference.case_id for reference in case_references}
    for positive_case_id in positive_case_ids:
        if positive_case_id not in known_case_ids:
            raise Phase8DefinitivePipelineRuntimeError(
                f"positive_case_ids contains {positive_case_id!r}, which is not present in "
                "case_references."
            )
    positive_refs = [
        reference for reference in case_references if reference.case_id in positive_case_ids
    ]
    if not positive_refs:
        raise Phase8DefinitivePipelineRuntimeError(
            "no case_references entry is marked in positive_case_ids; at least one is required."
        )

    torch = _import_torch()
    monai = _import_monai()

    seed = config.seed if rng_seed is None else rng_seed
    # The model/optimizer are seeded and constructed exactly once for the
    # entire continuous run; there is no re-seeding or re-initialization at
    # any candidate checkpoint step.
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    sampling_rng = np.random.default_rng(seed)

    model = _build_definitive_segresnet_model(torch=torch, monai=monai, config=config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)

    candidate_steps_remaining = set(candidate_checkpoint_steps)
    step_results: list[Phase8DefinitiveTrainingStepResult] = []
    snapshots: dict[int, Any] = {}
    all_finite = True

    for step_index in range(total_optimizer_steps):
        positive_case_id = positive_refs[step_index % len(positive_refs)].case_id
        negative_case_index = int(sampling_rng.integers(0, len(case_references)))
        negative_case_id = case_references[negative_case_index].case_id

        positive_case = loader(positive_case_id)
        negative_case = (
            positive_case if negative_case_id == positive_case_id else loader(negative_case_id)
        )

        step_result = _run_definitive_training_step(
            step_index=step_index,
            positive_case=positive_case,
            negative_case=negative_case,
            config=config,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            torch=torch,
            sampling_rng=sampling_rng,
        )
        step_results.append(step_result)
        del positive_case, negative_case

        if not (step_result.loss_finite and step_result.gradient_finite):
            all_finite = False
            break

        steps_completed = step_index + 1
        if steps_completed in candidate_steps_remaining:
            # A deep copy is required (not a bare reference to
            # ``model.state_dict()``, whose tensors would otherwise continue
            # to be mutated in place by subsequent optimizer steps): each
            # snapshot must be an independent, frozen-in-time copy.
            snapshots[steps_completed] = copy.deepcopy(model.state_dict())

    return Phase8DefinitiveContinuousTrainingRunResult(
        step_results=tuple(step_results),
        all_steps_finite=all_finite,
        checkpoint_snapshots=snapshots,
    )


def run_definitive_continuous_training_with_snapshots(
    case_references: Sequence[Phase8DefinitiveTrainCaseReference],
    *,
    positive_case_ids: frozenset[str] | set[str],
    loader: Phase8DefinitiveCaseLoader | Callable[[str], Phase8DefinitiveTrainCase],
    config: Phase8DefinitiveConfig,
    policy: Phase8DefinitiveTrainingExecutionPolicy,
    rng_seed: int | None = None,
) -> Phase8DefinitiveContinuousTrainingRunResult:
    """Run one continuous ``policy.total_optimizer_steps``-step training run.

    The model and optimizer are constructed and seeded exactly once (not
    re-seeded at step 250), then trained continuously by repeated calls to
    the single shared :func:`_run_definitive_training_step` helper -- the
    same step implementation used by :func:`run_definitive_training` and
    :func:`run_definitive_training_from_references`. A deep-copied
    ``model.state_dict()`` snapshot is captured immediately after each step
    count in ``policy.candidate_checkpoint_steps`` (250 and 500) completes.
    If loss or any gradient goes non-finite at any step, training stops
    immediately (fail-closed, matching the existing ``all_steps_finite``
    behavior elsewhere in this module) and no snapshot is taken for any
    un-reached candidate step.

    This function fails closed if ``policy.total_optimizer_steps`` or
    ``policy.candidate_checkpoint_steps`` were tampered with -- defense in
    depth, even though :class:`Phase8DefinitiveTrainingExecutionPolicy`
    already self-validates both fields at construction time.
    """

    if policy.total_optimizer_steps != _POLICY_TOTAL_OPTIMIZER_STEPS:
        raise Phase8DefinitivePipelineConfigError(
            f"policy.total_optimizer_steps must equal {_POLICY_TOTAL_OPTIMIZER_STEPS!r}."
        )
    if policy.candidate_checkpoint_steps != _POLICY_CANDIDATE_CHECKPOINT_STEPS:
        raise Phase8DefinitivePipelineConfigError(
            f"policy.candidate_checkpoint_steps must equal {_POLICY_CANDIDATE_CHECKPOINT_STEPS!r}."
        )

    return _run_definitive_continuous_training_with_snapshots_core(
        case_references,
        positive_case_ids=positive_case_ids,
        loader=loader,
        config=config,
        total_optimizer_steps=policy.total_optimizer_steps,
        candidate_checkpoint_steps=policy.candidate_checkpoint_steps,
        rng_seed=rng_seed,
    )


# ---------------------------------------------------------------------------
# I. Checkpoint snapshot serialization and no-overwrite publication
# ---------------------------------------------------------------------------


def serialize_checkpoint_state_dict(state_dict: Any) -> tuple[bytes, str, int]:
    """Serialize a ``state_dict`` deterministically enough to hash and publish.

    Returns ``(payload_bytes, sha256_hex, byte_size)``.
    """

    torch = _import_torch()
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest(), len(payload)


def _publish_checkpoint_bytes_no_overwrite(*, payload: bytes, output_path: Path) -> None:
    """Write ``payload`` to ``output_path`` without ever overwriting an existing file.

    Mirrors :func:`protoem_ct.data._phase2_publication.publish_text_no_overwrite`'s
    hardlink-then-fallback-rename pattern, adapted for raw bytes rather than
    UTF-8 text.
    """

    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    created_temp = False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if temp_path.exists():
            raise Phase8DefinitiveCheckpointPublicationError(
                f"temporary checkpoint publication path already exists: {temp_path}"
            )
        temp_path.write_bytes(payload)
        created_temp = True
        if output_path.exists():
            raise Phase8DefinitiveCheckpointPublicationError(
                f"checkpoint publication target already exists: {output_path}"
            )
        try:
            os.link(temp_path, output_path)
        except OSError as link_exc:
            if output_path.exists():
                raise Phase8DefinitiveCheckpointPublicationError(
                    f"checkpoint publication target already exists: {output_path}"
                ) from link_exc
            os.rename(temp_path, output_path)
        else:
            temp_path.unlink()
        created_temp = False
    except Phase8DefinitiveCheckpointPublicationError:
        raise
    except OSError as exc:
        raise Phase8DefinitivePipelineRuntimeError(
            "failed to publish Phase 8 definitive checkpoint bytes"
        ) from exc
    finally:
        if created_temp and temp_path.exists():
            temp_path.unlink()


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveCheckpointPublicationResult:
    """One published checkpoint snapshot: its file, hash, and metadata."""

    optimizer_step: int
    output_path: Path
    checkpoint_sha256: str
    checkpoint_byte_size: int
    checkpoint_metadata: Phase8CheckpointMetadata


def publish_definitive_checkpoint_snapshot(
    *,
    state_dict: Any,
    optimizer_step: int,
    output_root: Path,
    config: Phase8DefinitiveConfig,
    development_manifest_hash: str,
    development_split_hash: str,
    originating_git_commit: str,
    package_environment_reference: ArtifactReference,
    preprocessing_evidence_hash: str,
    completion_status: str = "completed",
    seed: int | None = None,
) -> Phase8DefinitiveCheckpointPublicationResult:
    """Serialize, hash, and no-overwrite-publish one candidate checkpoint snapshot.

    ``output_root`` is validated with
    :func:`protoem_ct.data.phase2_paths.validate_explicit_external_output_root`
    before anything is written. ``optimizer_step`` must be one of the locked
    candidate checkpoint steps (250 or 500). Checkpoint metadata is built on
    top of the existing, unmodified :func:`build_definitive_checkpoint_metadata`
    (never freeze-eligible); this function does not fork
    :class:`Phase8CheckpointMetadata`. Publishing twice to the same
    ``output_root``/``optimizer_step`` fails closed.
    """

    if optimizer_step not in _POLICY_CANDIDATE_CHECKPOINT_STEPS:
        raise Phase8DefinitiveCheckpointPublicationError(
            f"optimizer_step must be one of {_POLICY_CANDIDATE_CHECKPOINT_STEPS!r}, got "
            f"{optimizer_step!r}."
        )

    validated_root = validate_explicit_external_output_root(output_root)
    payload, checkpoint_sha256, checkpoint_byte_size = serialize_checkpoint_state_dict(state_dict)
    output_path = validated_root / f"phase8_definitive_checkpoint_step_{optimizer_step}.pt"
    _publish_checkpoint_bytes_no_overwrite(payload=payload, output_path=output_path)

    checkpoint_metadata = build_definitive_checkpoint_metadata(
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_size=checkpoint_byte_size,
        config=config,
        development_manifest_hash=development_manifest_hash,
        development_split_hash=development_split_hash,
        originating_git_commit=originating_git_commit,
        package_environment_reference=package_environment_reference,
        completion_status=completion_status,
        preprocessing_evidence_hash=preprocessing_evidence_hash,
        seed=seed,
    )

    return Phase8DefinitiveCheckpointPublicationResult(
        optimizer_step=optimizer_step,
        output_path=output_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_size=checkpoint_byte_size,
        checkpoint_metadata=checkpoint_metadata,
    )


# ---------------------------------------------------------------------------
# J. Full development-validation orchestration (exactly 20 unique cases)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveValidationCaseReference:
    """A lightweight, non-medical reference to one validation case.

    Mirrors :class:`Phase8DefinitiveTrainCaseReference`: holds only the
    anonymous ``case_id`` and never embeds image/label arrays.
    """

    case_id: str


class Phase8DefinitiveValidationCaseLoader(Protocol):
    """Loads exactly one fully preprocessed validation case, given its case_id."""

    def __call__(self, case_id: str) -> Phase8DefinitiveValidationCase: ...


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveFullValidationResult:
    """Result of a full, one-at-a-time, 20-case development-validation pass."""

    candidate_step: int
    case_results: tuple[Phase8DefinitiveValidationRunResult, ...]
    case_ids: tuple[str, ...]
    mean_tumor_dice: float


def run_definitive_full_validation(
    case_references: Sequence[Phase8DefinitiveValidationCaseReference],
    *,
    loader: (
        Phase8DefinitiveValidationCaseLoader | Callable[[str], Phase8DefinitiveValidationCase]
    ),
    model: Any,
    config: Phase8DefinitiveConfig,
    policy: Phase8DefinitiveTrainingExecutionPolicy,
    candidate_step: int,
) -> Phase8DefinitiveFullValidationResult:
    """Run the locked full 20-case development-validation pass for one candidate.

    Fails closed unless ``case_references`` contains exactly
    ``policy.validation_expected_case_count`` (20) *unique* case IDs -- 19,
    21, and 20-with-a-duplicate are all rejected. Cases are loaded and
    evaluated strictly one at a time via ``loader`` (each case delegates to
    the existing :func:`run_definitive_validation`); no full-volume array is
    retained past the iteration that produced its result, so there is no
    persistent cross-case cache. ``mean_tumor_dice`` is computed only from
    tumor Dice (never IoU, never loss); if any case's prediction is
    non-finite, this function raises rather than silently excluding that
    case from the mean.
    """

    case_ids = [reference.case_id for reference in case_references]
    if len(case_ids) != policy.validation_expected_case_count:
        raise Phase8DefinitivePipelineValidationError(
            f"expected exactly {policy.validation_expected_case_count} validation case "
            f"references, got {len(case_ids)}."
        )
    if len(set(case_ids)) != len(case_ids):
        raise Phase8DefinitivePipelineValidationError(
            "validation case references contain one or more duplicate case_id values."
        )

    case_results: list[Phase8DefinitiveValidationRunResult] = []
    for case_id in case_ids:
        case = loader(case_id)
        result = run_definitive_validation(model, case, config=config)
        case_results.append(result)
        del case

    for result in case_results:
        if not result.prediction_finite or not math.isfinite(result.tumor_dice):
            raise Phase8DefinitivePipelineValidationError(
                f"validation case {result.case_id!r} produced a non-finite prediction; "
                "cannot compute mean_tumor_dice from an invalid run."
            )

    mean_dice = sum(result.tumor_dice for result in case_results) / len(case_results)

    return Phase8DefinitiveFullValidationResult(
        candidate_step=candidate_step,
        case_results=tuple(case_results),
        case_ids=tuple(sorted(case_ids)),
        mean_tumor_dice=mean_dice,
    )


# ---------------------------------------------------------------------------
# K. Checkpoint selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveCheckpointSelectionResult:
    """The outcome of comparing the two candidate checkpoints' validation results."""

    candidate_step_250: int
    candidate_step_500: int
    mean_tumor_dice_step_250: float
    mean_tumor_dice_step_500: float
    selected_step: int
    case_ids: tuple[str, ...]
    policy: Phase8DefinitiveTrainingExecutionPolicy
    config: Phase8DefinitiveConfig


def select_definitive_checkpoint(
    *,
    result_step_250: Phase8DefinitiveFullValidationResult | None,
    result_step_500: Phase8DefinitiveFullValidationResult | None,
    policy: Phase8DefinitiveTrainingExecutionPolicy,
    config: Phase8DefinitiveConfig,
) -> Phase8DefinitiveCheckpointSelectionResult:
    """Select the higher-mean-tumor-Dice candidate checkpoint; fails closed otherwise.

    Selection uses ``mean_tumor_dice`` exclusively -- there is no ``loss``
    parameter anywhere in this signature, and IoU is never read. On an exact
    ``==`` tie, step 250 is selected (``policy.tie_break_policy ==
    "earliest_checkpoint_step"``). Fails closed (raising
    :class:`Phase8DefinitivePipelineSelectionError`) if either candidate
    result is missing, either candidate's step is not 250/500, either
    candidate's per-case validation count is not exactly
    ``policy.validation_expected_case_count`` unique case IDs, any candidate
    contains a non-finite tumor Dice, or the two candidates were validated on
    different case sets.
    """

    if result_step_250 is None or result_step_500 is None:
        raise Phase8DefinitivePipelineSelectionError(
            "both step-250 and step-500 full-validation results are required for selection."
        )

    candidates = {250: result_step_250, 500: result_step_500}
    for expected_step, result in candidates.items():
        if result.candidate_step != expected_step:
            raise Phase8DefinitivePipelineSelectionError(
                f"candidate result for step {expected_step} has candidate_step="
                f"{result.candidate_step!r}."
            )
        if len(result.case_results) != policy.validation_expected_case_count:
            raise Phase8DefinitivePipelineSelectionError(
                f"candidate step {expected_step} has {len(result.case_results)} validation "
                f"case results, expected {policy.validation_expected_case_count}."
            )
        result_case_ids = [case_result.case_id for case_result in result.case_results]
        if len(set(result_case_ids)) != len(result_case_ids):
            raise Phase8DefinitivePipelineSelectionError(
                f"candidate step {expected_step} has duplicate validation case IDs."
            )
        for case_result in result.case_results:
            if not case_result.prediction_finite or not math.isfinite(case_result.tumor_dice):
                raise Phase8DefinitivePipelineSelectionError(
                    f"candidate step {expected_step} has a non-finite tumor_dice for case "
                    f"{case_result.case_id!r}; selection fails closed."
                )

    if result_step_250.case_ids != result_step_500.case_ids:
        raise Phase8DefinitivePipelineSelectionError(
            "step-250 and step-500 candidates were validated on different case sets."
        )

    mean_250 = result_step_250.mean_tumor_dice
    mean_500 = result_step_500.mean_tumor_dice

    if mean_500 > mean_250:
        selected_step = 500
    elif mean_250 > mean_500:
        selected_step = 250
    else:
        # Exact tie: policy.tie_break_policy == "earliest_checkpoint_step".
        selected_step = 250

    return Phase8DefinitiveCheckpointSelectionResult(
        candidate_step_250=250,
        candidate_step_500=500,
        mean_tumor_dice_step_250=mean_250,
        mean_tumor_dice_step_500=mean_500,
        selected_step=selected_step,
        case_ids=result_step_250.case_ids,
        policy=policy,
        config=config,
    )


# ---------------------------------------------------------------------------
# L. Machine-readable checkpoint-selection evidence artifact
# ---------------------------------------------------------------------------

PHASE8_DEFINITIVE_CHECKPOINT_SELECTION_EVIDENCE_SCHEMA_NAME: Final[str] = (
    "phase8_definitive_checkpoint_selection_evidence"
)
PHASE8_DEFINITIVE_CHECKPOINT_SELECTION_EVIDENCE_SCHEMA_VERSION: Final[str] = "v1"


def _checkpoint_selection_evidence_payload(
    *,
    schema_name: str,
    schema_version: str,
    policy_hash: str,
    candidate_step_250: int,
    candidate_step_500: int,
    candidate_checkpoint_hash_step_250: str,
    candidate_checkpoint_hash_step_500: str,
    validation_case_set_identity_hash: str,
    mean_tumor_dice_step_250: float,
    mean_tumor_dice_step_500: float,
    selected_checkpoint_step: int,
    selected_checkpoint_hash: str,
    selection_metric: str,
    tie_break_policy: str,
    internal_test_used: bool,
    external_data_used: bool,
    external_labels_used: bool,
    selection_completed: bool,
) -> dict[str, JsonValue]:
    return {
        "candidate_checkpoint_hash_step_250": candidate_checkpoint_hash_step_250,
        "candidate_checkpoint_hash_step_500": candidate_checkpoint_hash_step_500,
        "candidate_step_250": candidate_step_250,
        "candidate_step_500": candidate_step_500,
        "external_data_used": external_data_used,
        "external_labels_used": external_labels_used,
        "internal_test_used": internal_test_used,
        "mean_tumor_dice_step_250": mean_tumor_dice_step_250,
        "mean_tumor_dice_step_500": mean_tumor_dice_step_500,
        "policy_hash": policy_hash,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "selected_checkpoint_hash": selected_checkpoint_hash,
        "selected_checkpoint_step": selected_checkpoint_step,
        "selection_completed": selection_completed,
        "selection_metric": selection_metric,
        "tie_break_policy": tie_break_policy,
        "validation_case_set_identity_hash": validation_case_set_identity_hash,
    }


def _require_sha256_hex(value: str, *, field_name: str) -> None:
    if not _SHA256_HEX_RE.fullmatch(value):
        raise Phase8DefinitivePipelineValidationError(
            f"{field_name} must be a lowercase SHA-256 hex digest."
        )


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveCheckpointSelectionEvidence:
    """Machine-readable evidence of a Package-A checkpoint selection decision.

    This is evidence, not a freeze artifact: it records what was selected and
    why, but it carries no freeze-eligibility flag and must never be treated
    as one.
    """

    schema_name: str
    schema_version: str
    policy_hash: str
    candidate_step_250: int
    candidate_step_500: int
    candidate_checkpoint_hash_step_250: str
    candidate_checkpoint_hash_step_500: str
    validation_case_set_identity_hash: str
    mean_tumor_dice_step_250: float
    mean_tumor_dice_step_500: float
    selected_checkpoint_step: int
    selected_checkpoint_hash: str
    selection_metric: str
    tie_break_policy: str
    internal_test_used: bool
    external_data_used: bool
    external_labels_used: bool
    selection_completed: bool
    evidence_hash: str

    def __post_init__(self) -> None:
        if self.schema_name != PHASE8_DEFINITIVE_CHECKPOINT_SELECTION_EVIDENCE_SCHEMA_NAME:
            raise Phase8DefinitivePipelineValidationError(
                f"schema_name must equal "
                f"{PHASE8_DEFINITIVE_CHECKPOINT_SELECTION_EVIDENCE_SCHEMA_NAME!r}."
            )
        if self.schema_version != PHASE8_DEFINITIVE_CHECKPOINT_SELECTION_EVIDENCE_SCHEMA_VERSION:
            raise Phase8DefinitivePipelineValidationError(
                f"schema_version must equal "
                f"{PHASE8_DEFINITIVE_CHECKPOINT_SELECTION_EVIDENCE_SCHEMA_VERSION!r}."
            )
        if self.candidate_step_250 != 250 or self.candidate_step_500 != 500:
            raise Phase8DefinitivePipelineValidationError(
                "candidate_step_250 must equal 250 and candidate_step_500 must equal 500."
            )
        if self.selected_checkpoint_step not in (250, 500):
            raise Phase8DefinitivePipelineValidationError(
                "selected_checkpoint_step must be 250 or 500."
            )
        if self.selection_metric != _POLICY_CHECKPOINT_SELECTION_METRIC:
            raise Phase8DefinitivePipelineValidationError(
                f"selection_metric must equal {_POLICY_CHECKPOINT_SELECTION_METRIC!r}."
            )
        if self.tie_break_policy != _POLICY_TIE_BREAK_POLICY:
            raise Phase8DefinitivePipelineValidationError(
                f"tie_break_policy must equal {_POLICY_TIE_BREAK_POLICY!r}."
            )
        if self.internal_test_used is not False:
            raise Phase8DefinitivePipelineValidationError("internal_test_used must be False.")
        if self.external_data_used is not False:
            raise Phase8DefinitivePipelineValidationError("external_data_used must be False.")
        if self.external_labels_used is not False:
            raise Phase8DefinitivePipelineValidationError("external_labels_used must be False.")
        _require_sha256_hex(self.policy_hash, field_name="policy_hash")
        _require_sha256_hex(
            self.candidate_checkpoint_hash_step_250,
            field_name="candidate_checkpoint_hash_step_250",
        )
        _require_sha256_hex(
            self.candidate_checkpoint_hash_step_500,
            field_name="candidate_checkpoint_hash_step_500",
        )
        _require_sha256_hex(
            self.validation_case_set_identity_hash, field_name="validation_case_set_identity_hash"
        )
        _require_sha256_hex(self.selected_checkpoint_hash, field_name="selected_checkpoint_hash")

        expected_hash = sha256_json(
            _checkpoint_selection_evidence_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                policy_hash=self.policy_hash,
                candidate_step_250=self.candidate_step_250,
                candidate_step_500=self.candidate_step_500,
                candidate_checkpoint_hash_step_250=self.candidate_checkpoint_hash_step_250,
                candidate_checkpoint_hash_step_500=self.candidate_checkpoint_hash_step_500,
                validation_case_set_identity_hash=self.validation_case_set_identity_hash,
                mean_tumor_dice_step_250=self.mean_tumor_dice_step_250,
                mean_tumor_dice_step_500=self.mean_tumor_dice_step_500,
                selected_checkpoint_step=self.selected_checkpoint_step,
                selected_checkpoint_hash=self.selected_checkpoint_hash,
                selection_metric=self.selection_metric,
                tie_break_policy=self.tie_break_policy,
                internal_test_used=self.internal_test_used,
                external_data_used=self.external_data_used,
                external_labels_used=self.external_labels_used,
                selection_completed=self.selection_completed,
            )
        )
        if self.evidence_hash != expected_hash:
            raise Phase8DefinitivePipelineValidationError(
                "evidence_hash does not match the canonical selection-evidence identity."
            )


def build_definitive_checkpoint_selection_evidence(
    *,
    selection: Phase8DefinitiveCheckpointSelectionResult,
    checkpoint_hash_step_250: str,
    checkpoint_hash_step_500: str,
) -> Phase8DefinitiveCheckpointSelectionEvidence:
    """Build the machine-readable selection-evidence artifact from a selection result."""

    selected_checkpoint_hash = (
        checkpoint_hash_step_250 if selection.selected_step == 250 else checkpoint_hash_step_500
    )
    validation_case_set_identity_hash = sha256_json(list(selection.case_ids))

    payload = _checkpoint_selection_evidence_payload(
        schema_name=PHASE8_DEFINITIVE_CHECKPOINT_SELECTION_EVIDENCE_SCHEMA_NAME,
        schema_version=PHASE8_DEFINITIVE_CHECKPOINT_SELECTION_EVIDENCE_SCHEMA_VERSION,
        policy_hash=selection.policy.policy_hash,
        candidate_step_250=selection.candidate_step_250,
        candidate_step_500=selection.candidate_step_500,
        candidate_checkpoint_hash_step_250=checkpoint_hash_step_250,
        candidate_checkpoint_hash_step_500=checkpoint_hash_step_500,
        validation_case_set_identity_hash=validation_case_set_identity_hash,
        mean_tumor_dice_step_250=selection.mean_tumor_dice_step_250,
        mean_tumor_dice_step_500=selection.mean_tumor_dice_step_500,
        selected_checkpoint_step=selection.selected_step,
        selected_checkpoint_hash=selected_checkpoint_hash,
        selection_metric=_POLICY_CHECKPOINT_SELECTION_METRIC,
        tie_break_policy=_POLICY_TIE_BREAK_POLICY,
        internal_test_used=False,
        external_data_used=False,
        external_labels_used=False,
        selection_completed=True,
    )

    return Phase8DefinitiveCheckpointSelectionEvidence(
        schema_name=PHASE8_DEFINITIVE_CHECKPOINT_SELECTION_EVIDENCE_SCHEMA_NAME,
        schema_version=PHASE8_DEFINITIVE_CHECKPOINT_SELECTION_EVIDENCE_SCHEMA_VERSION,
        policy_hash=selection.policy.policy_hash,
        candidate_step_250=selection.candidate_step_250,
        candidate_step_500=selection.candidate_step_500,
        candidate_checkpoint_hash_step_250=checkpoint_hash_step_250,
        candidate_checkpoint_hash_step_500=checkpoint_hash_step_500,
        validation_case_set_identity_hash=validation_case_set_identity_hash,
        mean_tumor_dice_step_250=selection.mean_tumor_dice_step_250,
        mean_tumor_dice_step_500=selection.mean_tumor_dice_step_500,
        selected_checkpoint_step=selection.selected_step,
        selected_checkpoint_hash=selected_checkpoint_hash,
        selection_metric=_POLICY_CHECKPOINT_SELECTION_METRIC,
        tie_break_policy=_POLICY_TIE_BREAK_POLICY,
        internal_test_used=False,
        external_data_used=False,
        external_labels_used=False,
        selection_completed=True,
        evidence_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# M. Process-level watchdog for continuous training
# ---------------------------------------------------------------------------
#
# Mirrors ``run_phase8_bounded_pilot_with_watchdog`` /
# ``_phase8_bounded_pilot_child_entrypoint`` in
# :mod:`protoem_ct.external.definitive_training_pilot`: a parent process
# supervises the entire continuous training run inside an isolated,
# killable child process using ``multiprocessing.get_context("spawn")`` and
# a ``Pipe``. On timeout the child is terminated (SIGTERM, escalating to
# ``.kill()``) and joined in a ``finally`` block on every code path, no
# partial success is ever surfaced, and there is no automatic retry.

_DEFAULT_DEFINITIVE_TRAINING_CHILD_TERMINATION_GRACE_SECONDS: Final[float] = 5.0


def _definitive_training_child_entrypoint(
    conn: Connection,
    target: Callable[..., Phase8DefinitiveContinuousTrainingRunResult],
    kwargs: Mapping[str, Any],
) -> None:
    """Run ``target(**kwargs)`` inside the child process and send back the outcome."""

    try:
        result = target(**kwargs)
    except BaseException as exc:  # noqa: BLE001 -- propagated to the parent unchanged.
        try:
            conn.send(("error", exc))
        except Exception:  # noqa: BLE001 -- exc itself failed to pickle; send a substitute.
            conn.send(
                (
                    "error",
                    Phase8DefinitivePipelineRuntimeError(
                        f"child training process raised an unpicklable exception: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
        finally:
            conn.close()
        return
    conn.send(("ok", result))
    conn.close()


def run_definitive_continuous_training_with_snapshots_and_watchdog(
    case_references: Sequence[Phase8DefinitiveTrainCaseReference],
    *,
    positive_case_ids: frozenset[str] | set[str],
    loader: Phase8DefinitiveCaseLoader | Callable[[str], Phase8DefinitiveTrainCase],
    config: Phase8DefinitiveConfig,
    policy: Phase8DefinitiveTrainingExecutionPolicy,
    rng_seed: int | None = None,
    wall_clock_limit_seconds: float | None = None,
    _target: Callable[
        ..., Phase8DefinitiveContinuousTrainingRunResult
    ] = run_definitive_continuous_training_with_snapshots,
    _multiprocessing_context: BaseContext | None = None,
    _termination_grace_seconds: float = (
        _DEFAULT_DEFINITIVE_TRAINING_CHILD_TERMINATION_GRACE_SECONDS
    ),
) -> Phase8DefinitiveContinuousTrainingRunResult:
    """Run continuous training inside a supervised, killable child process.

    ``wall_clock_limit_seconds`` defaults to the locked
    ``policy.maximum_wall_clock_seconds`` (36000.0 seconds / 10 hours); it is
    an explicit override on *this call's* enforcement only -- it never
    mutates or bypasses the locked policy object itself, so passing a tiny
    value in a test does not weaken
    :class:`Phase8DefinitiveTrainingExecutionPolicy`'s own hard-locked
    ``maximum_wall_clock_seconds`` field.

    On timeout, the child is terminated at the operating-system level and
    this function raises :class:`Phase8DefinitiveTrainingWatchdogTimeoutError`
    -- no partial training result, snapshot, or checkpoint selection is ever
    surfaced for that call, and there is no automatic retry; the caller must
    make a fresh call.

    ``_target`` and ``_multiprocessing_context`` are testing-only seams (not
    part of the public contract); production callers must not pass them.
    Production always uses ``multiprocessing.get_context("spawn")``.
    """

    supervisor_start_time = time.monotonic()
    deadline_seconds = (
        policy.maximum_wall_clock_seconds
        if wall_clock_limit_seconds is None
        else wall_clock_limit_seconds
    )

    kwargs: dict[str, Any] = {
        "case_references": case_references,
        "positive_case_ids": positive_case_ids,
        "loader": loader,
        "config": config,
        "policy": policy,
        "rng_seed": rng_seed,
    }

    ctx = _multiprocessing_context or multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    # BaseContext.Process is dynamically bound per concrete context subclass;
    # mypy's typeshed stub does not expose it on the base class.
    child = ctx.Process(  # type: ignore[attr-defined]
        target=_definitive_training_child_entrypoint,
        kwargs={"conn": child_conn, "target": _target, "kwargs": kwargs},
        daemon=False,
    )

    def _terminate_and_join_child() -> None:
        if not child.is_alive():
            return
        child.terminate()
        child.join(timeout=_termination_grace_seconds)
        if child.is_alive():
            child.kill()
            child.join(timeout=_termination_grace_seconds)

    try:
        child.start()
        # The child now owns its end of the pipe; the parent must close its
        # copy of the child's end so EOF is detected correctly.
        child_conn.close()

        elapsed_before_join = time.monotonic() - supervisor_start_time
        join_timeout = max(0.0, deadline_seconds - elapsed_before_join)
        child.join(timeout=join_timeout)

        if child.is_alive():
            raise Phase8DefinitiveTrainingWatchdogTimeoutError(
                f"process-level watchdog killed the child training process after "
                f"{deadline_seconds:.3f}s with no checkpoint selection; the child was "
                f"terminated at the operating-system process level, not by an in-process "
                f"elapsed-time check."
            )

        if parent_conn.poll():
            try:
                status, payload = parent_conn.recv()
            except EOFError as exc:
                raise Phase8DefinitivePipelineRuntimeError(
                    f"child training process exited (exitcode={child.exitcode}) without "
                    f"returning a result."
                ) from exc
        else:
            raise Phase8DefinitivePipelineRuntimeError(
                f"child training process exited (exitcode={child.exitcode}) without "
                f"returning a result."
            )

        if status == "error":
            raise cast(BaseException, payload)
        return cast(Phase8DefinitiveContinuousTrainingRunResult, payload)
    finally:
        _terminate_and_join_child()
        parent_conn.close()
