"""Phase 8 bounded, verification-scale real-data training pilot (Substage 4B).

This module implements a deliberately bounded, explicitly user-approved,
CPU-only, verification-scale real-data training *pilot*. It is **not**
definitive training and produces **no** freeze-eligible, selection-eligible,
or scientific-metric-eligible artifacts anywhere.

It is the direct successor of
:mod:`protoem_ct.external.tiny_real_verification` (Substage 3), which opened
exactly one train and one validation NIfTI pair for a two-step dry/overfit
check. This pilot is a distinct, slightly larger step:

* Exactly two deterministic train cases (one tumor-positive, one
  empty-target) and one deterministic validation case, selected by pure
  metadata (manifest + development split + lesion-component counts), never
  by pixel content, image shape, or spacing.
* A fixed, exact 10:10 foreground-aware patch sampling schedule across
  20 optimizer steps: 10 positive-centered patches from the tumor-positive
  train case, 5 negative-centered patches from the tumor-positive train
  case, and 5 negative-centered patches from the empty-target train case.
* A single full-volume sliding-window validation forward pass, with no
  scientific segmentation metric computed anywhere (no Dice/IoU/HD95/NSD,
  no lesion recall/precision/F1, no prediction-file publication).
* A hard 45-minute wall-clock watchdog with fail-closed, no-partial-success
  semantics: every artifact is written atomically, in one commit, only after
  the entire bounded pipeline (including the sliding-window validation pass)
  completes successfully.
* Every published checkpoint is hard-coded ``pilot_only=True``,
  ``tiny_verification_only=False``, ``freeze_eligible=False``,
  ``selection_eligible=False``, ``definitive_training=False``, and
  ``scientific_metric_eligible=False`` -- there is no parameter capable of
  setting any of these to a more permissive value.

Before any medical-file access, this module strictly reconstructs and
verifies the Substage 4B prerequisite artifacts (the real-development input
binding, the fixed candidate inventory, and the preprocessing decision) and
cross-checks their embedded manifest/split hashes against the actual
manifest and split files being used.

``torch``, ``monai``, and (where avoidable) ``nibabel`` model/tensor access
are imported lazily inside function bodies so that importing this module
(and, more importantly, importing ``protoem_ct.cli.main``) never pulls in
the heavy baseline runtime dependencies.
"""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import pickle
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import nibabel as nib
import numpy as np

from protoem_ct.artifacts import (
    DatasetManifest,
    DevelopmentSplitManifest,
    LesionComponentsArtifact,
    hash_dataset_manifest,
    hash_development_split,
    phase2_artifact_from_json,
)
from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.baselines._torch_runtime import configure_phase3_cpu_torch_runtime
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)
from protoem_ct.data.phase2_paths import (
    InvalidDatasetRootError,
    resolve_regular_file_beneath_root,
    validate_explicit_dataset_root,
    validate_explicit_external_output_root,
)
from protoem_ct.external.internal_evidence import (
    PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME,
    PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
    ArtifactReference,
    Phase8CheckpointMetadata,
    Phase8FixedCandidateInventory,
    Phase8PreprocessingDecision,
    artifact_reference_to_dict,
    phase8_checkpoint_metadata_to_dict,
    phase8_fixed_candidate_inventory_from_mapping,
    phase8_preprocessing_decision_from_mapping,
)
from protoem_ct.external.real_development_runner import (
    Phase8RealDevelopmentInputBinding,
    phase8_real_development_input_binding_from_mapping,
)

MappingLike: TypeAlias = Mapping[str, object]

# ---------------------------------------------------------------------------
# Schema identity
# ---------------------------------------------------------------------------

PHASE8_BOUNDED_PILOT_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_BOUNDED_PILOT_CONFIG_SCHEMA_NAME: Final[str] = "phase8_definitive_training_pilot_config"
PHASE8_BOUNDED_PILOT_ACCESS_LEDGER_SCHEMA_NAME: Final[str] = (
    "phase8_definitive_training_pilot_access_ledger"
)
PHASE8_BOUNDED_PILOT_CHECKPOINT_METADATA_SCHEMA_NAME: Final[str] = (
    "phase8_definitive_training_pilot_checkpoint_metadata"
)
PHASE8_BOUNDED_PILOT_SUMMARY_SCHEMA_NAME: Final[str] = "phase8_definitive_training_pilot_summary"

PHASE8_BOUNDED_PILOT_CONFIG_FILENAME: Final[str] = "phase8_definitive_training_pilot_config.json"
PHASE8_BOUNDED_PILOT_ACCESS_LEDGER_FILENAME: Final[str] = (
    "phase8_definitive_training_pilot_access_ledger.json"
)
PHASE8_BOUNDED_PILOT_CHECKPOINT_METADATA_FILENAME: Final[str] = (
    "phase8_definitive_training_pilot_checkpoint_metadata.json"
)
PHASE8_BOUNDED_PILOT_SUMMARY_FILENAME: Final[str] = "phase8_definitive_training_pilot_summary.json"
PHASE8_BOUNDED_PILOT_CHECKPOINT_SUBDIRECTORY_NAME: Final[str] = "checkpoints"
PHASE8_BOUNDED_PILOT_CHECKPOINT_FILENAME: Final[str] = (
    "phase8_definitive_training_pilot_checkpoint.pt"
)

NON_SCIENTIFIC_DISCLAIMER: Final[str] = (
    "pilot_only: this run performs no scientific evaluation; it computes no "
    "Dice/IoU/HD95/NSD, lesion recall/precision/F1, false-positive-lesions-per-scan, "
    "or volume-error metric, and its results must not be read as evidence of "
    "segmentation performance, convergence, generalization, baseline validity, "
    "or checkpoint quality. This is not definitive training and is not "
    "freeze-eligible or selection-eligible."
)

# ---------------------------------------------------------------------------
# Hard compute limits -- validated, never silently clamped or parameterized
# ---------------------------------------------------------------------------

REQUIRED_DEVICE_TYPE: Final[str] = "cpu"
REQUIRED_TRAIN_CASE_COUNT: Final[int] = 2
REQUIRED_VALIDATION_CASE_COUNT: Final[int] = 1
REQUIRED_BATCH_SIZE: Final[int] = 1
REQUIRED_GRADIENT_ACCUMULATION_STEPS: Final[int] = 1
REQUIRED_MAX_OPTIMIZER_STEPS: Final[int] = 20
REQUIRED_MAX_EPOCHS: Final[int] = 1
REQUIRED_NUM_WORKERS: Final[int] = 0
REQUIRED_PATCH_SIZE: Final[tuple[int, int, int]] = (64, 64, 32)
REQUIRED_SLIDING_WINDOW_ROI_SIZE: Final[tuple[int, int, int]] = (96, 96, 64)
REQUIRED_SLIDING_WINDOW_OVERLAP: Final[float] = 0.25
REQUIRED_SLIDING_WINDOW_BATCH_SIZE: Final[int] = 1
REQUIRED_OPTIMIZER_NAME: Final[str] = "adamw"
REQUIRED_LEARNING_RATE: Final[float] = 1e-4
REQUIRED_WEIGHT_DECAY: Final[float] = 1e-5
REQUIRED_LOSS_NAME: Final[str] = "dice_ce"
REQUIRED_CLASS_WEIGHTING: Final[str] = "none"
FIXED_HU_WINDOW_MIN: Final[float] = -1000.0
FIXED_HU_WINDOW_MAX: Final[float] = 1000.0
FIT_SCOPE_NO_DATA_DEPENDENT_FIT: Final[str] = "no_data_dependent_fit"

REQUIRED_POSITIVE_PATCHES_FROM_POSITIVE_CASE: Final[int] = 10
REQUIRED_NEGATIVE_PATCHES_FROM_POSITIVE_CASE: Final[int] = 5
REQUIRED_NEGATIVE_PATCHES_FROM_EMPTY_CASE: Final[int] = 5
REQUIRED_TOTAL_PATCHES: Final[int] = (
    REQUIRED_POSITIVE_PATCHES_FROM_POSITIVE_CASE
    + REQUIRED_NEGATIVE_PATCHES_FROM_POSITIVE_CASE
    + REQUIRED_NEGATIVE_PATCHES_FROM_EMPTY_CASE
)

DEFAULT_WALL_CLOCK_LIMIT_SECONDS: Final[float] = 45.0 * 60.0

APPROVED_CANDIDATE_ID: Final[str] = "monai_segresnet_baseline"
APPROVED_MODEL_FAMILY: Final[str] = "monai_segresnet"

# Raw label domain for the approved LiTS/MSD Task03 Liver development cohort:
# 0=background, 1=liver, 2=tumor. See tiny_real_verification.py for the same,
# authoritative documentation of this fixed, non-tunable convention.
RAW_LABEL_TUMOR_VALUE: Final[int] = 2
RAW_LABEL_ALLOWED_VALUES: Final[tuple[int, ...]] = (0, 1, 2)
CASE_PAIR_AFFINE_TOLERANCE_MM: Final[float] = 1e-4
REQUIRED_ORIENTATION_AXCODES: Final[tuple[str, str, str]] = ("R", "A", "S")
_PADDED_IMAGE_FILL_VALUE: Final[float] = -1.0
_PADDED_LABEL_FILL_VALUE: Final[float] = 0.0

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{7,64}$")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class Phase8BoundedPilotError(ValueError):
    """Base error for the Phase 8 bounded real-data training pilot."""


class Phase8BoundedPilotConfigError(Phase8BoundedPilotError):
    """Raised when the bounded pilot config violates a hard compute limit."""


class Phase8BoundedPilotOutputRootError(Phase8BoundedPilotError):
    """Raised when the external output root fails the path-safety contract."""


class Phase8BoundedPilotPrerequisitesError(Phase8BoundedPilotError):
    """Raised when Substage 4B prerequisite artifacts fail reconstruction/cross-check."""


class Phase8BoundedPilotCaseSelectionError(Phase8BoundedPilotError):
    """Raised when deterministic case selection cannot proceed from metadata alone."""


class Phase8BoundedPilotDataError(Phase8BoundedPilotError):
    """Raised when a selected case's data is missing, unreadable, or unsafe."""


class Phase8BoundedPilotLabelDomainError(Phase8BoundedPilotDataError):
    """Raised when a raw label's values fall outside the approved {0,1,2} domain."""


class Phase8BoundedPilotOrientationError(Phase8BoundedPilotDataError):
    """Raised when a raw image/label pair is not in RAS orientation."""


class Phase8BoundedPilotSamplingScheduleError(Phase8BoundedPilotError):
    """Raised when the exact 10:10 foreground-aware sampling schedule cannot be built."""


class Phase8BoundedPilotRuntimeError(Phase8BoundedPilotError):
    """Raised when a bounded model, training, or checkpoint operation fails closed."""


class Phase8BoundedPilotWatchdogTimeoutError(Phase8BoundedPilotRuntimeError):
    """Raised when the 45-minute wall-clock watchdog fires; aborts with no publication."""


class Phase8BoundedPilotProcessWatchdogTimeoutError(Phase8BoundedPilotWatchdogTimeoutError):
    """Raised by the process-level supervisor when it kills the child pilot process.

    This is distinct from the plain :class:`Phase8BoundedPilotWatchdogTimeoutError`
    raised by the in-process ``_check_watchdog`` elapsed-time check. That check is a
    synchronous elapsed-time comparison made *between* stages inside
    :func:`run_phase8_bounded_pilot`'s own execution; it cannot fire while a single
    blocking call (for example ``monai.inferers.sliding_window_inference``, the
    bounded training loop, or checkpoint I/O) is actively running.

    This error instead means the *parent* process observed that the child process
    running the entire bounded pilot (medical data loading through artifact
    publication) did not return within ``wall_clock_limit_seconds`` and was
    terminated at the operating-system process level (SIGTERM, escalating to
    SIGKILL/``.kill()`` if still alive after a short grace period). It carries no
    in-process checkpoint name, because a single blocking call can prevent any
    in-process checkpoint from ever being reached. No partial success is ever
    surfaced for this outcome, and the caller performs no automatic retry.
    """

    killed_at_process_level: bool = True


class Phase8BoundedPilotPublicationError(Phase8BoundedPilotError):
    """Raised when guarded artifact publication fails safely."""


# ---------------------------------------------------------------------------
# A. Compute-limit validated configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8BoundedPilotConfig:
    """Immutable, hashed configuration encoding every bounded-pilot hard limit."""

    schema_name: str
    schema_version: str
    device_type: str
    amp_enabled: bool
    train_case_count: int
    validation_case_count: int
    batch_size: int
    gradient_accumulation_steps: int
    max_optimizer_steps: int
    max_epochs: int
    num_workers: int
    patch_size: tuple[int, int, int]
    positive_patches_from_positive_case: int
    negative_patches_from_positive_case: int
    negative_patches_from_empty_case: int
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    loss_name: str
    class_weighting: str
    sliding_window_roi_size: tuple[int, int, int]
    sliding_window_overlap: float
    sliding_window_batch_size: int
    hu_window_min: float
    hu_window_max: float
    fit_scope: str
    resume_disabled: bool
    hyperparameter_search_prohibited: bool
    candidate_comparison_prohibited: bool
    augmentation_disabled: bool
    wall_clock_limit_seconds: float
    seed: int
    originating_git_commit: str
    package_versions: tuple[tuple[str, str], ...]
    config_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_BOUNDED_PILOT_CONFIG_SCHEMA_NAME)
        _require_schema_version(self.schema_version)
        if self.device_type != REQUIRED_DEVICE_TYPE:
            raise Phase8BoundedPilotConfigError(
                "device_type must be 'cpu'; no other device is permitted."
            )
        if self.amp_enabled is not False:
            raise Phase8BoundedPilotConfigError("amp_enabled must be False.")
        if self.train_case_count != REQUIRED_TRAIN_CASE_COUNT:
            raise Phase8BoundedPilotConfigError("train_case_count must equal 2.")
        if self.validation_case_count != REQUIRED_VALIDATION_CASE_COUNT:
            raise Phase8BoundedPilotConfigError("validation_case_count must equal 1.")
        if self.batch_size != REQUIRED_BATCH_SIZE:
            raise Phase8BoundedPilotConfigError("batch_size must equal 1.")
        if self.gradient_accumulation_steps != REQUIRED_GRADIENT_ACCUMULATION_STEPS:
            raise Phase8BoundedPilotConfigError("gradient_accumulation_steps must equal 1.")
        if self.max_optimizer_steps != REQUIRED_MAX_OPTIMIZER_STEPS:
            raise Phase8BoundedPilotConfigError("max_optimizer_steps must equal 20.")
        if self.max_epochs != REQUIRED_MAX_EPOCHS:
            raise Phase8BoundedPilotConfigError("max_epochs must equal 1.")
        if self.num_workers != REQUIRED_NUM_WORKERS:
            raise Phase8BoundedPilotConfigError("num_workers must equal 0.")
        object.__setattr__(self, "patch_size", tuple(int(v) for v in self.patch_size))
        if self.patch_size != REQUIRED_PATCH_SIZE:
            raise Phase8BoundedPilotConfigError(
                f"patch_size must equal {REQUIRED_PATCH_SIZE!r}, got {self.patch_size!r}."
            )
        if self.positive_patches_from_positive_case != REQUIRED_POSITIVE_PATCHES_FROM_POSITIVE_CASE:
            raise Phase8BoundedPilotConfigError(
                "positive_patches_from_positive_case must equal 10."
            )
        if self.negative_patches_from_positive_case != REQUIRED_NEGATIVE_PATCHES_FROM_POSITIVE_CASE:
            raise Phase8BoundedPilotConfigError("negative_patches_from_positive_case must equal 5.")
        if self.negative_patches_from_empty_case != REQUIRED_NEGATIVE_PATCHES_FROM_EMPTY_CASE:
            raise Phase8BoundedPilotConfigError("negative_patches_from_empty_case must equal 5.")
        if self.optimizer_name != REQUIRED_OPTIMIZER_NAME:
            raise Phase8BoundedPilotConfigError("optimizer_name must equal 'adamw'.")
        if self.learning_rate != REQUIRED_LEARNING_RATE:
            raise Phase8BoundedPilotConfigError("learning_rate must equal 1e-4.")
        if self.weight_decay != REQUIRED_WEIGHT_DECAY:
            raise Phase8BoundedPilotConfigError("weight_decay must equal 1e-5.")
        if self.loss_name != REQUIRED_LOSS_NAME:
            raise Phase8BoundedPilotConfigError("loss_name must equal 'dice_ce'.")
        if self.class_weighting != REQUIRED_CLASS_WEIGHTING:
            raise Phase8BoundedPilotConfigError("class_weighting must equal 'none'.")
        object.__setattr__(
            self,
            "sliding_window_roi_size",
            tuple(int(v) for v in self.sliding_window_roi_size),
        )
        if self.sliding_window_roi_size != REQUIRED_SLIDING_WINDOW_ROI_SIZE:
            raise Phase8BoundedPilotConfigError(
                f"sliding_window_roi_size must equal {REQUIRED_SLIDING_WINDOW_ROI_SIZE!r}."
            )
        if self.sliding_window_overlap != REQUIRED_SLIDING_WINDOW_OVERLAP:
            raise Phase8BoundedPilotConfigError("sliding_window_overlap must equal 0.25.")
        if self.sliding_window_batch_size != REQUIRED_SLIDING_WINDOW_BATCH_SIZE:
            raise Phase8BoundedPilotConfigError("sliding_window_batch_size must equal 1.")
        if self.hu_window_min != FIXED_HU_WINDOW_MIN or self.hu_window_max != FIXED_HU_WINDOW_MAX:
            raise Phase8BoundedPilotConfigError(
                "HU window must equal the fixed constant "
                f"[{FIXED_HU_WINDOW_MIN}, {FIXED_HU_WINDOW_MAX}]; it must not be data-fit."
            )
        if self.fit_scope != FIT_SCOPE_NO_DATA_DEPENDENT_FIT:
            raise Phase8BoundedPilotConfigError(
                f"fit_scope must equal {FIT_SCOPE_NO_DATA_DEPENDENT_FIT!r}."
            )
        if self.resume_disabled is not True:
            raise Phase8BoundedPilotConfigError("resume_disabled must be True.")
        if self.hyperparameter_search_prohibited is not True:
            raise Phase8BoundedPilotConfigError("hyperparameter_search_prohibited must be True.")
        if self.candidate_comparison_prohibited is not True:
            raise Phase8BoundedPilotConfigError("candidate_comparison_prohibited must be True.")
        if self.augmentation_disabled is not True:
            raise Phase8BoundedPilotConfigError("augmentation_disabled must be True.")
        if not (self.wall_clock_limit_seconds > 0.0):
            raise Phase8BoundedPilotConfigError("wall_clock_limit_seconds must be positive.")
        if self.seed < 0:
            raise Phase8BoundedPilotConfigError("seed must be non-negative.")
        if not _GIT_COMMIT_RE.fullmatch(self.originating_git_commit):
            raise Phase8BoundedPilotConfigError(
                "originating_git_commit must match ^[0-9a-f]{7,64}$."
            )
        object.__setattr__(
            self,
            "package_versions",
            tuple(sorted((str(name), str(version)) for name, version in self.package_versions)),
        )
        if not self.package_versions:
            raise Phase8BoundedPilotConfigError("package_versions must not be empty.")
        for name, version in self.package_versions:
            if not name or not version:
                raise Phase8BoundedPilotConfigError("package_versions entries must be nonempty.")
        _require_self_hash(
            self.config_hash,
            _config_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                device_type=self.device_type,
                amp_enabled=self.amp_enabled,
                train_case_count=self.train_case_count,
                validation_case_count=self.validation_case_count,
                batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                max_optimizer_steps=self.max_optimizer_steps,
                max_epochs=self.max_epochs,
                num_workers=self.num_workers,
                patch_size=self.patch_size,
                positive_patches_from_positive_case=self.positive_patches_from_positive_case,
                negative_patches_from_positive_case=self.negative_patches_from_positive_case,
                negative_patches_from_empty_case=self.negative_patches_from_empty_case,
                optimizer_name=self.optimizer_name,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                loss_name=self.loss_name,
                class_weighting=self.class_weighting,
                sliding_window_roi_size=self.sliding_window_roi_size,
                sliding_window_overlap=self.sliding_window_overlap,
                sliding_window_batch_size=self.sliding_window_batch_size,
                hu_window_min=self.hu_window_min,
                hu_window_max=self.hu_window_max,
                fit_scope=self.fit_scope,
                resume_disabled=self.resume_disabled,
                hyperparameter_search_prohibited=self.hyperparameter_search_prohibited,
                candidate_comparison_prohibited=self.candidate_comparison_prohibited,
                augmentation_disabled=self.augmentation_disabled,
                wall_clock_limit_seconds=self.wall_clock_limit_seconds,
                seed=self.seed,
                originating_git_commit=self.originating_git_commit,
                package_versions=self.package_versions,
            ),
            field_name="config_hash",
        )


def _config_payload(
    *,
    schema_name: str,
    schema_version: str,
    device_type: str,
    amp_enabled: bool,
    train_case_count: int,
    validation_case_count: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_optimizer_steps: int,
    max_epochs: int,
    num_workers: int,
    patch_size: Sequence[int],
    positive_patches_from_positive_case: int,
    negative_patches_from_positive_case: int,
    negative_patches_from_empty_case: int,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    class_weighting: str,
    sliding_window_roi_size: Sequence[int],
    sliding_window_overlap: float,
    sliding_window_batch_size: int,
    hu_window_min: float,
    hu_window_max: float,
    fit_scope: str,
    resume_disabled: bool,
    hyperparameter_search_prohibited: bool,
    candidate_comparison_prohibited: bool,
    augmentation_disabled: bool,
    wall_clock_limit_seconds: float,
    seed: int,
    originating_git_commit: str,
    package_versions: Sequence[tuple[str, str]],
) -> dict[str, JsonValue]:
    return {
        "amp_enabled": amp_enabled,
        "augmentation_disabled": augmentation_disabled,
        "batch_size": batch_size,
        "candidate_comparison_prohibited": candidate_comparison_prohibited,
        "class_weighting": class_weighting,
        "device_type": device_type,
        "fit_scope": fit_scope,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "hu_window_max": hu_window_max,
        "hu_window_min": hu_window_min,
        "hyperparameter_search_prohibited": hyperparameter_search_prohibited,
        "learning_rate": learning_rate,
        "loss_name": loss_name,
        "max_epochs": max_epochs,
        "max_optimizer_steps": max_optimizer_steps,
        "negative_patches_from_empty_case": negative_patches_from_empty_case,
        "negative_patches_from_positive_case": negative_patches_from_positive_case,
        "num_workers": num_workers,
        "optimizer_name": optimizer_name,
        "originating_git_commit": originating_git_commit,
        "package_versions": {name: version for name, version in package_versions},
        "patch_size": list(patch_size),
        "positive_patches_from_positive_case": positive_patches_from_positive_case,
        "resume_disabled": resume_disabled,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "seed": seed,
        "sliding_window_batch_size": sliding_window_batch_size,
        "sliding_window_overlap": sliding_window_overlap,
        "sliding_window_roi_size": list(sliding_window_roi_size),
        "train_case_count": train_case_count,
        "validation_case_count": validation_case_count,
        "wall_clock_limit_seconds": wall_clock_limit_seconds,
        "weight_decay": weight_decay,
    }


def phase8_bounded_pilot_config_to_dict(config: Phase8BoundedPilotConfig) -> dict[str, JsonValue]:
    """Convert the bounded pilot config to a canonical mapping."""

    payload = _config_payload(
        schema_name=config.schema_name,
        schema_version=config.schema_version,
        device_type=config.device_type,
        amp_enabled=config.amp_enabled,
        train_case_count=config.train_case_count,
        validation_case_count=config.validation_case_count,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_optimizer_steps=config.max_optimizer_steps,
        max_epochs=config.max_epochs,
        num_workers=config.num_workers,
        patch_size=config.patch_size,
        positive_patches_from_positive_case=config.positive_patches_from_positive_case,
        negative_patches_from_positive_case=config.negative_patches_from_positive_case,
        negative_patches_from_empty_case=config.negative_patches_from_empty_case,
        optimizer_name=config.optimizer_name,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        loss_name=config.loss_name,
        class_weighting=config.class_weighting,
        sliding_window_roi_size=config.sliding_window_roi_size,
        sliding_window_overlap=config.sliding_window_overlap,
        sliding_window_batch_size=config.sliding_window_batch_size,
        hu_window_min=config.hu_window_min,
        hu_window_max=config.hu_window_max,
        fit_scope=config.fit_scope,
        resume_disabled=config.resume_disabled,
        hyperparameter_search_prohibited=config.hyperparameter_search_prohibited,
        candidate_comparison_prohibited=config.candidate_comparison_prohibited,
        augmentation_disabled=config.augmentation_disabled,
        wall_clock_limit_seconds=config.wall_clock_limit_seconds,
        seed=config.seed,
        originating_git_commit=config.originating_git_commit,
        package_versions=config.package_versions,
    )
    payload["config_hash"] = config.config_hash
    return payload


def build_phase8_bounded_pilot_config(
    *,
    seed: int,
    originating_git_commit: str,
    package_versions: Mapping[str, str],
    wall_clock_limit_seconds: float = DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
) -> Phase8BoundedPilotConfig:
    """Build and validate the bounded pilot config from caller intent.

    Every hard limit in the approved Substage 4B specification is hard-coded
    to its required value; only ``seed``, provenance fields, and
    ``wall_clock_limit_seconds`` (needed so tests can pass a tiny synthetic
    limit to deterministically trigger the watchdog) are caller-controlled,
    and each is validated against its bound at construction time.
    """

    package_versions_tuple = tuple(sorted(package_versions.items()))
    payload = _config_payload(
        schema_name=PHASE8_BOUNDED_PILOT_CONFIG_SCHEMA_NAME,
        schema_version=PHASE8_BOUNDED_PILOT_SCHEMA_VERSION,
        device_type=REQUIRED_DEVICE_TYPE,
        amp_enabled=False,
        train_case_count=REQUIRED_TRAIN_CASE_COUNT,
        validation_case_count=REQUIRED_VALIDATION_CASE_COUNT,
        batch_size=REQUIRED_BATCH_SIZE,
        gradient_accumulation_steps=REQUIRED_GRADIENT_ACCUMULATION_STEPS,
        max_optimizer_steps=REQUIRED_MAX_OPTIMIZER_STEPS,
        max_epochs=REQUIRED_MAX_EPOCHS,
        num_workers=REQUIRED_NUM_WORKERS,
        patch_size=REQUIRED_PATCH_SIZE,
        positive_patches_from_positive_case=REQUIRED_POSITIVE_PATCHES_FROM_POSITIVE_CASE,
        negative_patches_from_positive_case=REQUIRED_NEGATIVE_PATCHES_FROM_POSITIVE_CASE,
        negative_patches_from_empty_case=REQUIRED_NEGATIVE_PATCHES_FROM_EMPTY_CASE,
        optimizer_name=REQUIRED_OPTIMIZER_NAME,
        learning_rate=REQUIRED_LEARNING_RATE,
        weight_decay=REQUIRED_WEIGHT_DECAY,
        loss_name=REQUIRED_LOSS_NAME,
        class_weighting=REQUIRED_CLASS_WEIGHTING,
        sliding_window_roi_size=REQUIRED_SLIDING_WINDOW_ROI_SIZE,
        sliding_window_overlap=REQUIRED_SLIDING_WINDOW_OVERLAP,
        sliding_window_batch_size=REQUIRED_SLIDING_WINDOW_BATCH_SIZE,
        hu_window_min=FIXED_HU_WINDOW_MIN,
        hu_window_max=FIXED_HU_WINDOW_MAX,
        fit_scope=FIT_SCOPE_NO_DATA_DEPENDENT_FIT,
        resume_disabled=True,
        hyperparameter_search_prohibited=True,
        candidate_comparison_prohibited=True,
        augmentation_disabled=True,
        wall_clock_limit_seconds=wall_clock_limit_seconds,
        seed=seed,
        originating_git_commit=originating_git_commit,
        package_versions=package_versions_tuple,
    )
    return Phase8BoundedPilotConfig(
        schema_name=PHASE8_BOUNDED_PILOT_CONFIG_SCHEMA_NAME,
        schema_version=PHASE8_BOUNDED_PILOT_SCHEMA_VERSION,
        device_type=REQUIRED_DEVICE_TYPE,
        amp_enabled=False,
        train_case_count=REQUIRED_TRAIN_CASE_COUNT,
        validation_case_count=REQUIRED_VALIDATION_CASE_COUNT,
        batch_size=REQUIRED_BATCH_SIZE,
        gradient_accumulation_steps=REQUIRED_GRADIENT_ACCUMULATION_STEPS,
        max_optimizer_steps=REQUIRED_MAX_OPTIMIZER_STEPS,
        max_epochs=REQUIRED_MAX_EPOCHS,
        num_workers=REQUIRED_NUM_WORKERS,
        patch_size=REQUIRED_PATCH_SIZE,
        positive_patches_from_positive_case=REQUIRED_POSITIVE_PATCHES_FROM_POSITIVE_CASE,
        negative_patches_from_positive_case=REQUIRED_NEGATIVE_PATCHES_FROM_POSITIVE_CASE,
        negative_patches_from_empty_case=REQUIRED_NEGATIVE_PATCHES_FROM_EMPTY_CASE,
        optimizer_name=REQUIRED_OPTIMIZER_NAME,
        learning_rate=REQUIRED_LEARNING_RATE,
        weight_decay=REQUIRED_WEIGHT_DECAY,
        loss_name=REQUIRED_LOSS_NAME,
        class_weighting=REQUIRED_CLASS_WEIGHTING,
        sliding_window_roi_size=REQUIRED_SLIDING_WINDOW_ROI_SIZE,
        sliding_window_overlap=REQUIRED_SLIDING_WINDOW_OVERLAP,
        sliding_window_batch_size=REQUIRED_SLIDING_WINDOW_BATCH_SIZE,
        hu_window_min=FIXED_HU_WINDOW_MIN,
        hu_window_max=FIXED_HU_WINDOW_MAX,
        fit_scope=FIT_SCOPE_NO_DATA_DEPENDENT_FIT,
        resume_disabled=True,
        hyperparameter_search_prohibited=True,
        candidate_comparison_prohibited=True,
        augmentation_disabled=True,
        wall_clock_limit_seconds=wall_clock_limit_seconds,
        seed=seed,
        originating_git_commit=originating_git_commit,
        package_versions=package_versions_tuple,
        config_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# B. Prerequisite artifact reconstruction and cross-check (metadata only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8BoundedPilotPrerequisites:
    """Reconstructed and cross-checked Substage 4B prerequisite artifacts."""

    input_binding: Phase8RealDevelopmentInputBinding
    candidate_inventory: Phase8FixedCandidateInventory
    preprocessing_decision: Phase8PreprocessingDecision


def _load_and_hash_check_json(path: Path, *, expected_sha256: str, description: str) -> MappingLike:
    resolved_path = path.resolve(strict=True)
    if not resolved_path.is_file():
        raise Phase8BoundedPilotPrerequisitesError(f"{description} is not a regular file.")
    observed_sha256 = sha256_file(resolved_path)
    if observed_sha256 != expected_sha256:
        raise Phase8BoundedPilotPrerequisitesError(
            f"{description} sha256 mismatch: expected {expected_sha256}, got {observed_sha256}."
        )
    try:
        decoded = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Phase8BoundedPilotPrerequisitesError(f"{description} is not valid JSON.") from exc
    if not isinstance(decoded, dict):
        raise Phase8BoundedPilotPrerequisitesError(f"{description} must decode to a JSON object.")
    return cast(MappingLike, decoded)


def verify_phase8_bounded_pilot_prerequisites(
    *,
    input_binding_path: Path,
    candidate_inventory_path: Path,
    preprocessing_decision_path: Path,
    expected_input_binding_sha256: str,
    expected_candidate_inventory_sha256: str,
    expected_preprocessing_decision_sha256: str,
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
) -> Phase8BoundedPilotPrerequisites:
    """Reconstruct and cross-check the three Substage 4B prerequisite artifacts.

    Every prerequisite file's own SHA-256 is verified against the
    caller-declared expectation before it is parsed. Each artifact's own
    internal self-hash is re-validated by its existing dataclass contract at
    construction time. Finally, the manifest/split hashes embedded in the
    input binding and the preprocessing decision are cross-checked against
    the actual manifest and split objects in use, and the approved
    ``monai_segresnet_baseline`` candidate's preprocessing evidence hash is
    cross-checked against the preprocessing decision. This function performs
    no image/label pixel access.
    """

    input_binding_mapping = _load_and_hash_check_json(
        input_binding_path,
        expected_sha256=expected_input_binding_sha256,
        description="real-development input binding",
    )
    candidate_inventory_mapping = _load_and_hash_check_json(
        candidate_inventory_path,
        expected_sha256=expected_candidate_inventory_sha256,
        description="fixed candidate inventory",
    )
    preprocessing_decision_mapping = _load_and_hash_check_json(
        preprocessing_decision_path,
        expected_sha256=expected_preprocessing_decision_sha256,
        description="preprocessing decision",
    )

    try:
        input_binding = phase8_real_development_input_binding_from_mapping(input_binding_mapping)
        candidate_inventory = phase8_fixed_candidate_inventory_from_mapping(
            candidate_inventory_mapping
        )
        preprocessing_decision = phase8_preprocessing_decision_from_mapping(
            preprocessing_decision_mapping
        )
    except Exception as exc:  # noqa: BLE001 -- reconstruction failure paths vary by artifact.
        raise Phase8BoundedPilotPrerequisitesError(
            f"failed to reconstruct a Substage 4B prerequisite artifact: {exc}"
        ) from exc

    manifest_hash = hash_dataset_manifest(manifest)
    split_hash = hash_development_split(split)

    if input_binding.manifest_reference.artifact_hash != manifest_hash:
        raise Phase8BoundedPilotPrerequisitesError(
            "input binding manifest_reference does not match the loaded manifest."
        )
    if input_binding.split_reference.artifact_hash != split_hash:
        raise Phase8BoundedPilotPrerequisitesError(
            "input binding split_reference does not match the loaded split."
        )
    if preprocessing_decision.development_manifest_hash != manifest_hash:
        raise Phase8BoundedPilotPrerequisitesError(
            "preprocessing decision development_manifest_hash does not match the loaded manifest."
        )
    if preprocessing_decision.development_split_hash != split_hash:
        raise Phase8BoundedPilotPrerequisitesError(
            "preprocessing decision development_split_hash does not match the loaded split."
        )

    approved_candidates = [
        candidate
        for candidate in candidate_inventory.candidates
        if candidate.candidate_id == APPROVED_CANDIDATE_ID
    ]
    if not approved_candidates:
        raise Phase8BoundedPilotPrerequisitesError(
            "candidate inventory does not contain the approved candidate "
            f"{APPROVED_CANDIDATE_ID!r}."
        )
    approved_candidate = approved_candidates[0]
    if (
        approved_candidate.preprocessing_evidence_hash
        != preprocessing_decision.preprocessing_decision_hash
    ):
        raise Phase8BoundedPilotPrerequisitesError(
            "approved candidate preprocessing_evidence_hash does not match the "
            "preprocessing decision."
        )
    if approved_candidate.model_family != APPROVED_MODEL_FAMILY:
        raise Phase8BoundedPilotPrerequisitesError(
            f"approved candidate model_family must equal {APPROVED_MODEL_FAMILY!r}."
        )
    if approved_candidate.uses_external_artifacts:
        raise Phase8BoundedPilotPrerequisitesError(
            "approved candidate must not use external artifacts."
        )

    return Phase8BoundedPilotPrerequisites(
        input_binding=input_binding,
        candidate_inventory=candidate_inventory,
        preprocessing_decision=preprocessing_decision,
    )


# ---------------------------------------------------------------------------
# C. Deterministic case selection (pure metadata, no pixel access)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8BoundedPilotSelectedCase:
    """One deterministically selected case, identified by anonymous IDs only."""

    anonymous_patient_id: str
    anonymous_case_id: str
    partition: str
    pilot_role: str
    relative_image_path: str
    relative_label_path: str


def select_phase8_bounded_pilot_cases(
    *,
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
    lesion_components: LesionComponentsArtifact,
) -> tuple[
    Phase8BoundedPilotSelectedCase,
    Phase8BoundedPilotSelectedCase,
    Phase8BoundedPilotSelectedCase,
]:
    """Deterministically select the positive-train, empty-train, and validation cases.

    Pure metadata-derived logic, mirroring
    :func:`protoem_ct.external.tiny_real_verification.select_phase8_tiny_real_development_cases`
    for the validation case, and extending it with a lesion-count lookup
    (from ``lesion_components``, itself metadata, never pixel bytes) for the
    two required train categories:

    1. The lexicographically first train case (by
       ``(anonymous_patient_id, anonymous_case_id)``) with ``lesion_count>0``.
    2. The lexicographically first train case with ``lesion_count==0``.
    3. The lexicographically first validation case.

    Never inspects image or label file contents, image shape, or spacing.
    Raises :class:`Phase8BoundedPilotCaseSelectionError` and never opens any
    NIfTI file if metadata cannot identify both required train categories.
    """

    train_candidates = sorted(
        (a for a in split.assignments if a.partition == "train"),
        key=lambda item: (item.anonymous_patient_id, item.anonymous_case_id),
    )
    validation_candidates = sorted(
        (a for a in split.assignments if a.partition == "validation"),
        key=lambda item: (item.anonymous_patient_id, item.anonymous_case_id),
    )
    if not train_candidates:
        raise Phase8BoundedPilotCaseSelectionError(
            "no train-partition assignments are available for selection."
        )
    if not validation_candidates:
        raise Phase8BoundedPilotCaseSelectionError(
            "no validation-partition assignments are available for selection."
        )

    lesion_count_by_case_id: dict[str, int] = {
        record.anonymous_case_id: record.lesion_count
        for record in lesion_components.case_records
        if record.analysis_performed and record.lesion_count is not None
    }

    positive_train = next(
        (
            candidate
            for candidate in train_candidates
            if lesion_count_by_case_id.get(candidate.anonymous_case_id, -1) > 0
        ),
        None,
    )
    empty_train = next(
        (
            candidate
            for candidate in train_candidates
            if lesion_count_by_case_id.get(candidate.anonymous_case_id, -1) == 0
        ),
        None,
    )
    if positive_train is None or empty_train is None:
        raise Phase8BoundedPilotCaseSelectionError(
            "metadata cannot identify both a tumor-positive and an empty-target train case; "
            "BLOCKED before any NIfTI access."
        )
    if positive_train.anonymous_case_id == empty_train.anonymous_case_id:
        raise Phase8BoundedPilotCaseSelectionError(
            "tumor-positive and empty-target train case selection collided on one case."
        )

    first_validation = validation_candidates[0]
    selected_case_ids = {
        positive_train.anonymous_case_id,
        empty_train.anonymous_case_id,
        first_validation.anonymous_case_id,
    }
    case_by_id = {
        record.anonymous_case_id: record
        for record in manifest.cases
        if record.anonymous_case_id in selected_case_ids
    }

    def _resolve(role: str, assignment: object, partition: str) -> Phase8BoundedPilotSelectedCase:
        case_id = cast(Any, assignment).anonymous_case_id
        patient_id = cast(Any, assignment).anonymous_patient_id
        record = case_by_id.get(case_id)
        if record is None:
            raise Phase8BoundedPilotCaseSelectionError(
                f"selected {role} case is not present in the dataset manifest."
            )
        return Phase8BoundedPilotSelectedCase(
            anonymous_patient_id=patient_id,
            anonymous_case_id=case_id,
            partition=partition,
            pilot_role=role,
            relative_image_path=record.relative_image_path,
            relative_label_path=record.relative_label_path,
        )

    return (
        _resolve("train_positive", positive_train, "train"),
        _resolve("train_empty", empty_train, "train"),
        _resolve("validation", first_validation, "validation"),
    )


# ---------------------------------------------------------------------------
# D. Output-root path safety (external, must-not-already-exist contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8BoundedPilotPaths:
    """Validated external locations for one bounded pilot run."""

    output_root: Path
    checkpoints_dir: Path


def validate_phase8_bounded_pilot_output_root(
    output_root: Path,
    *,
    repository_root: Path,
) -> Phase8BoundedPilotPaths:
    """Validate an external, nonexistent, non-symlinked output root.

    Mirrors
    :func:`protoem_ct.external.tiny_real_verification.validate_phase8_tiny_real_output_root`.
    """

    root_path = Path(output_root)
    if str(root_path) == "" or not root_path.is_absolute():
        raise Phase8BoundedPilotOutputRootError("output_root must be an explicit absolute path.")
    if root_path.is_symlink():
        raise Phase8BoundedPilotOutputRootError("output_root must not be a symlink.")
    if root_path.exists():
        raise Phase8BoundedPilotOutputRootError("output_root must not already exist.")

    canonical_repository_root = validate_explicit_dataset_root(repository_root)
    try:
        canonical_output_root = validate_explicit_external_output_root(
            root_path, forbidden_roots=(canonical_repository_root,)
        )
    except InvalidDatasetRootError as exc:
        raise Phase8BoundedPilotOutputRootError("invalid bounded pilot output root.") from exc

    return Phase8BoundedPilotPaths(
        output_root=canonical_output_root,
        checkpoints_dir=canonical_output_root / PHASE8_BOUNDED_PILOT_CHECKPOINT_SUBDIRECTORY_NAME,
    )


# ---------------------------------------------------------------------------
# E. Dataset access ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8BoundedPilotAccessRecord:
    """Per-case access record; anonymous IDs only, no filesystem identity."""

    anonymous_case_id: str
    partition: str
    pilot_role: str
    image_opened: bool
    label_opened: bool

    def __post_init__(self) -> None:
        if self.partition not in {"train", "validation"}:
            raise Phase8BoundedPilotError(
                "access record partition must be 'train' or 'validation'."
            )
        if self.pilot_role not in {"train_positive", "train_empty", "validation"}:
            raise Phase8BoundedPilotError(
                "pilot_role must be 'train_positive', 'train_empty', or 'validation'."
            )


def _access_record_to_dict(record: Phase8BoundedPilotAccessRecord) -> dict[str, JsonValue]:
    return {
        "anonymous_case_id": record.anonymous_case_id,
        "image_opened": record.image_opened,
        "label_opened": record.label_opened,
        "partition": record.partition,
        "pilot_role": record.pilot_role,
    }


@dataclass(frozen=True, slots=True)
class Phase8BoundedPilotAccessLedger:
    """Bounded-pilot dataset access ledger; anonymous IDs only."""

    schema_name: str
    schema_version: str
    records: tuple[Phase8BoundedPilotAccessRecord, ...]
    internal_test_opened: bool
    external_data_opened: bool
    prior_checkpoint_loaded: bool
    file_access_count: int
    ledger_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_BOUNDED_PILOT_ACCESS_LEDGER_SCHEMA_NAME)
        _require_schema_version(self.schema_version)
        object.__setattr__(self, "records", tuple(self.records))
        if self.internal_test_opened is not False:
            raise Phase8BoundedPilotError("internal_test_opened must always be False.")
        if self.external_data_opened is not False:
            raise Phase8BoundedPilotError("external_data_opened must always be False.")
        if self.prior_checkpoint_loaded is not False:
            raise Phase8BoundedPilotError("prior_checkpoint_loaded must always be False.")
        if len(self.records) != 3:
            raise Phase8BoundedPilotError("access ledger must record exactly 3 selected cases.")
        expected_count = sum(
            int(record.image_opened) + int(record.label_opened) for record in self.records
        )
        if self.file_access_count != expected_count:
            raise Phase8BoundedPilotError(
                "file_access_count does not match the sum of opened files in records."
            )
        _require_self_hash(
            self.ledger_hash,
            _access_ledger_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                records=self.records,
                internal_test_opened=self.internal_test_opened,
                external_data_opened=self.external_data_opened,
                prior_checkpoint_loaded=self.prior_checkpoint_loaded,
                file_access_count=self.file_access_count,
            ),
            field_name="ledger_hash",
        )


def _access_ledger_payload(
    *,
    schema_name: str,
    schema_version: str,
    records: Sequence[Phase8BoundedPilotAccessRecord],
    internal_test_opened: bool,
    external_data_opened: bool,
    prior_checkpoint_loaded: bool,
    file_access_count: int,
) -> dict[str, JsonValue]:
    return {
        "external_data_opened": external_data_opened,
        "file_access_count": file_access_count,
        "internal_test_opened": internal_test_opened,
        "prior_checkpoint_loaded": prior_checkpoint_loaded,
        "records": [_access_record_to_dict(record) for record in records],
        "schema_name": schema_name,
        "schema_version": schema_version,
    }


def phase8_bounded_pilot_access_ledger_to_dict(
    ledger: Phase8BoundedPilotAccessLedger,
) -> dict[str, JsonValue]:
    """Convert the access ledger to a canonical mapping."""

    payload = _access_ledger_payload(
        schema_name=ledger.schema_name,
        schema_version=ledger.schema_version,
        records=ledger.records,
        internal_test_opened=ledger.internal_test_opened,
        external_data_opened=ledger.external_data_opened,
        prior_checkpoint_loaded=ledger.prior_checkpoint_loaded,
        file_access_count=ledger.file_access_count,
    )
    payload["ledger_hash"] = ledger.ledger_hash
    return payload


def build_phase8_bounded_pilot_access_ledger(
    *,
    train_positive_case: Phase8BoundedPilotSelectedCase,
    train_empty_case: Phase8BoundedPilotSelectedCase,
    validation_case: Phase8BoundedPilotSelectedCase,
) -> Phase8BoundedPilotAccessLedger:
    """Build the bounded-pilot access ledger for the three accessed cases."""

    records = (
        Phase8BoundedPilotAccessRecord(
            anonymous_case_id=train_positive_case.anonymous_case_id,
            partition="train",
            pilot_role="train_positive",
            image_opened=True,
            label_opened=True,
        ),
        Phase8BoundedPilotAccessRecord(
            anonymous_case_id=train_empty_case.anonymous_case_id,
            partition="train",
            pilot_role="train_empty",
            image_opened=True,
            label_opened=True,
        ),
        Phase8BoundedPilotAccessRecord(
            anonymous_case_id=validation_case.anonymous_case_id,
            partition="validation",
            pilot_role="validation",
            image_opened=True,
            label_opened=True,
        ),
    )
    file_access_count = sum(
        int(record.image_opened) + int(record.label_opened) for record in records
    )
    payload = _access_ledger_payload(
        schema_name=PHASE8_BOUNDED_PILOT_ACCESS_LEDGER_SCHEMA_NAME,
        schema_version=PHASE8_BOUNDED_PILOT_SCHEMA_VERSION,
        records=records,
        internal_test_opened=False,
        external_data_opened=False,
        prior_checkpoint_loaded=False,
        file_access_count=file_access_count,
    )
    return Phase8BoundedPilotAccessLedger(
        schema_name=PHASE8_BOUNDED_PILOT_ACCESS_LEDGER_SCHEMA_NAME,
        schema_version=PHASE8_BOUNDED_PILOT_SCHEMA_VERSION,
        records=records,
        internal_test_opened=False,
        external_data_opened=False,
        prior_checkpoint_loaded=False,
        file_access_count=file_access_count,
        ledger_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# F. Watchdog
# ---------------------------------------------------------------------------


def _check_watchdog(*, start_time: float, wall_clock_limit_seconds: float, checkpoint: str) -> None:
    """Raise if the bounded pilot's elapsed wall-clock time exceeds its limit.

    Called at bounded checkpoints throughout the pilot's own execution loop
    (before training starts, after each optimizer step, before validation,
    and before final publication). This is a synchronous elapsed-time check,
    not an out-of-band process/thread killer; that matches the only timeout
    precedent in this codebase (``subprocess.run(..., timeout=...)`` in
    ``protoem_ct.baselines.nnunet``), scaled down to an in-process loop that
    has no subprocess boundary to attach a timeout to.
    """

    elapsed = time.monotonic() - start_time
    if elapsed > wall_clock_limit_seconds:
        raise Phase8BoundedPilotWatchdogTimeoutError(
            f"wall-clock watchdog fired at checkpoint {checkpoint!r}: "
            f"elapsed {elapsed:.3f}s exceeds limit {wall_clock_limit_seconds:.3f}s."
        )


# ---------------------------------------------------------------------------
# G. Foreground-aware sampling schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8BoundedPilotPatchSample:
    """One deterministically selected patch-center sample in the 20-step schedule."""

    pilot_role: str
    is_positive: bool
    center: tuple[int, int, int]


def _sorted_voxel_coordinates(mask: np.ndarray) -> np.ndarray:
    """Return voxel coordinates matching ``mask`` in deterministic lexicographic order."""

    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return coordinates
    order = np.lexsort(
        tuple(coordinates[:, axis] for axis in range(coordinates.shape[1] - 1, -1, -1))
    )
    return coordinates[order]


def build_phase8_bounded_pilot_sampling_schedule(
    *,
    positive_case_label: np.ndarray,
    empty_case_label: np.ndarray,
    seed: int,
) -> tuple[Phase8BoundedPilotPatchSample, ...]:
    """Build the exact, deterministic 10:10 foreground-aware sampling schedule.

    Deterministic voxel-selection rule (documented per the Substage 4B
    contract): eligible voxel coordinates for each of the three groups are
    first sorted into a canonical lexicographic order via
    :func:`_sorted_voxel_coordinates`, then a single
    ``numpy.random.default_rng(seed)`` draws, in this fixed order --
    10 positive centers from the tumor-positive case, then 5 negative
    centers from the tumor-positive case, then 5 negative centers from the
    empty-target case -- a *without-replacement* sample of indices into that
    sorted array via ``rng.choice(..., replace=False)``. Because the RNG is
    seeded and consumed in a fixed order over a canonically sorted
    coordinate array, the resulting schedule is fully reproducible.

    Positive centers are drawn from raw voxels equal to
    :data:`RAW_LABEL_TUMOR_VALUE`; negative centers are drawn from raw
    voxels not equal to that value (background/liver). Fails closed with
    :class:`Phase8BoundedPilotSamplingScheduleError` -- and never silently
    changes the ratio or case count -- if any group has fewer eligible
    voxels than required.
    """

    rng = np.random.default_rng(seed)

    positive_foreground = _sorted_voxel_coordinates(positive_case_label == RAW_LABEL_TUMOR_VALUE)
    positive_background = _sorted_voxel_coordinates(positive_case_label != RAW_LABEL_TUMOR_VALUE)
    empty_background = _sorted_voxel_coordinates(empty_case_label != RAW_LABEL_TUMOR_VALUE)

    groups: tuple[tuple[np.ndarray, int, str, bool], ...] = (
        (
            positive_foreground,
            REQUIRED_POSITIVE_PATCHES_FROM_POSITIVE_CASE,
            "train_positive",
            True,
        ),
        (
            positive_background,
            REQUIRED_NEGATIVE_PATCHES_FROM_POSITIVE_CASE,
            "train_positive",
            False,
        ),
        (
            empty_background,
            REQUIRED_NEGATIVE_PATCHES_FROM_EMPTY_CASE,
            "train_empty",
            False,
        ),
    )

    samples: list[Phase8BoundedPilotPatchSample] = []
    for coordinates, count, pilot_role, is_positive in groups:
        if coordinates.shape[0] < count:
            raise Phase8BoundedPilotSamplingScheduleError(
                f"cannot satisfy the exact bounded schedule: {pilot_role} "
                f"{'positive' if is_positive else 'negative'} group has "
                f"{coordinates.shape[0]} eligible voxels, needs {count}."
            )
        chosen_indices = rng.choice(coordinates.shape[0], size=count, replace=False)
        for index in chosen_indices:
            center = tuple(int(v) for v in coordinates[int(index)])
            samples.append(
                Phase8BoundedPilotPatchSample(
                    pilot_role=pilot_role, is_positive=is_positive, center=cast(Any, center)
                )
            )

    if len(samples) != REQUIRED_TOTAL_PATCHES:
        raise Phase8BoundedPilotSamplingScheduleError(
            f"bounded schedule must contain exactly {REQUIRED_TOTAL_PATCHES} samples, "
            f"got {len(samples)}."
        )
    positive_count = sum(1 for sample in samples if sample.is_positive)
    negative_count = len(samples) - positive_count
    if positive_count != REQUIRED_POSITIVE_PATCHES_FROM_POSITIVE_CASE or negative_count != (
        REQUIRED_NEGATIVE_PATCHES_FROM_POSITIVE_CASE + REQUIRED_NEGATIVE_PATCHES_FROM_EMPTY_CASE
    ):
        raise Phase8BoundedPilotSamplingScheduleError(
            "bounded schedule failed the exact 10:10 positive:negative ratio check."
        )
    return tuple(samples)


# ---------------------------------------------------------------------------
# H. Checkpoint safety metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8BoundedPilotCheckpointMetadata:
    """Bounded-pilot checkpoint metadata; never freeze/selection/scientific eligible."""

    schema_name: str
    schema_version: str
    checkpoint_metadata: Phase8CheckpointMetadata
    pilot_only: bool
    tiny_verification_only: bool
    selection_eligible: bool
    definitive_training: bool
    scientific_metric_eligible: bool
    wrapper_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_BOUNDED_PILOT_CHECKPOINT_METADATA_SCHEMA_NAME)
        _require_schema_version(self.schema_version)
        if self.pilot_only is not True:
            raise Phase8BoundedPilotError("pilot_only must be True.")
        if self.tiny_verification_only is not False:
            raise Phase8BoundedPilotError("tiny_verification_only must be False.")
        if self.selection_eligible is not False:
            raise Phase8BoundedPilotError("selection_eligible must be False.")
        if self.definitive_training is not False:
            raise Phase8BoundedPilotError("definitive_training must be False.")
        if self.scientific_metric_eligible is not False:
            raise Phase8BoundedPilotError("scientific_metric_eligible must be False.")
        if self.checkpoint_metadata.freeze_eligible is not False:
            raise Phase8BoundedPilotError(
                "wrapped checkpoint_metadata.freeze_eligible must be False."
            )
        if self.checkpoint_metadata.completion_status != "synthetic_smoke":
            raise Phase8BoundedPilotError(
                "wrapped checkpoint_metadata.completion_status must be 'synthetic_smoke'."
            )
        _require_self_hash(
            self.wrapper_hash,
            _checkpoint_metadata_wrapper_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                checkpoint_metadata=self.checkpoint_metadata,
                pilot_only=self.pilot_only,
                tiny_verification_only=self.tiny_verification_only,
                selection_eligible=self.selection_eligible,
                definitive_training=self.definitive_training,
                scientific_metric_eligible=self.scientific_metric_eligible,
            ),
            field_name="wrapper_hash",
        )


def _checkpoint_metadata_wrapper_payload(
    *,
    schema_name: str,
    schema_version: str,
    checkpoint_metadata: Phase8CheckpointMetadata,
    pilot_only: bool,
    tiny_verification_only: bool,
    selection_eligible: bool,
    definitive_training: bool,
    scientific_metric_eligible: bool,
) -> dict[str, JsonValue]:
    return {
        "checkpoint_metadata": phase8_checkpoint_metadata_to_dict(checkpoint_metadata),
        "definitive_training": definitive_training,
        "pilot_only": pilot_only,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "scientific_metric_eligible": scientific_metric_eligible,
        "selection_eligible": selection_eligible,
        "tiny_verification_only": tiny_verification_only,
    }


def phase8_bounded_pilot_checkpoint_metadata_to_dict(
    metadata: Phase8BoundedPilotCheckpointMetadata,
) -> dict[str, JsonValue]:
    """Convert the checkpoint-safety wrapper to a canonical mapping."""

    payload = _checkpoint_metadata_wrapper_payload(
        schema_name=metadata.schema_name,
        schema_version=metadata.schema_version,
        checkpoint_metadata=metadata.checkpoint_metadata,
        pilot_only=metadata.pilot_only,
        tiny_verification_only=metadata.tiny_verification_only,
        selection_eligible=metadata.selection_eligible,
        definitive_training=metadata.definitive_training,
        scientific_metric_eligible=metadata.scientific_metric_eligible,
    )
    payload["wrapper_hash"] = metadata.wrapper_hash
    return payload


def build_phase8_bounded_pilot_checkpoint_metadata(
    *,
    config: Phase8BoundedPilotConfig,
    development_manifest_hash: str,
    development_split_hash: str,
    checkpoint_sha256: str,
    checkpoint_byte_size: int,
) -> Phase8BoundedPilotCheckpointMetadata:
    """Build bounded-pilot checkpoint metadata that is never freeze eligible.

    ``pilot_only``, ``tiny_verification_only``, ``selection_eligible``,
    ``definitive_training``, ``scientific_metric_eligible``, and the wrapped
    contract's own ``freeze_eligible`` are all hard-coded here; no parameter
    of this function can move any of them to a more permissive value. Using
    ``completion_status="synthetic_smoke"`` (rather than ``"completed"``)
    means the existing :class:`Phase8CheckpointMetadata` contract itself
    structurally rejects ``freeze_eligible=True`` for this checkpoint, the
    same mechanism used by ``tiny_real_verification.py``.
    """

    package_environment_reference = ArtifactReference(
        schema_name=PHASE8_BOUNDED_PILOT_CONFIG_SCHEMA_NAME,
        schema_version=PHASE8_BOUNDED_PILOT_SCHEMA_VERSION,
        artifact_hash=config.config_hash,
        artifact_role="bounded_pilot_config",
    )
    checkpoint_metadata_identity_payload: dict[str, JsonValue] = {
        "candidate_id": APPROVED_CANDIDATE_ID,
        "checkpoint_byte_size": checkpoint_byte_size,
        "checkpoint_sha256": checkpoint_sha256,
        "completion_status": "synthetic_smoke",
        "development_manifest_hash": development_manifest_hash,
        "development_split_hash": development_split_hash,
        "failure_codes": [],
        "freeze_eligible": False,
        "model_family": APPROVED_MODEL_FAMILY,
        "no_external_data": True,
        "originating_git_commit": config.originating_git_commit,
        "package_environment_reference": artifact_reference_to_dict(package_environment_reference),
        "preprocessing_evidence_hash": config.config_hash,
        "schema_name": PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME,
        "schema_version": PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
        "seed": config.seed,
        "serialization_format": "torch_state_dict",
        "training_adaptation_mode": "baseline_training",
        "training_config_hash": config.config_hash,
        "validation_metric_artifact_reference": None,
    }
    checkpoint_metadata = Phase8CheckpointMetadata(
        schema_name=PHASE8_CHECKPOINT_METADATA_SCHEMA_NAME,
        schema_version=PHASE8_INTERNAL_EVIDENCE_SCHEMA_VERSION,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_size=checkpoint_byte_size,
        serialization_format="torch_state_dict",
        model_family=APPROVED_MODEL_FAMILY,
        candidate_id=APPROVED_CANDIDATE_ID,
        seed=config.seed,
        training_adaptation_mode="baseline_training",
        originating_git_commit=config.originating_git_commit,
        package_environment_reference=package_environment_reference,
        training_config_hash=config.config_hash,
        preprocessing_evidence_hash=config.config_hash,
        development_manifest_hash=development_manifest_hash,
        development_split_hash=development_split_hash,
        validation_metric_artifact_reference=None,
        device_type="cpu",
        amp_state="disabled",
        completion_status="synthetic_smoke",
        failure_codes=(),
        no_external_data=True,
        freeze_eligible=False,
        checkpoint_metadata_hash=sha256_json(checkpoint_metadata_identity_payload),
    )

    payload = _checkpoint_metadata_wrapper_payload(
        schema_name=PHASE8_BOUNDED_PILOT_CHECKPOINT_METADATA_SCHEMA_NAME,
        schema_version=PHASE8_BOUNDED_PILOT_SCHEMA_VERSION,
        checkpoint_metadata=checkpoint_metadata,
        pilot_only=True,
        tiny_verification_only=False,
        selection_eligible=False,
        definitive_training=False,
        scientific_metric_eligible=False,
    )
    return Phase8BoundedPilotCheckpointMetadata(
        schema_name=PHASE8_BOUNDED_PILOT_CHECKPOINT_METADATA_SCHEMA_NAME,
        schema_version=PHASE8_BOUNDED_PILOT_SCHEMA_VERSION,
        checkpoint_metadata=checkpoint_metadata,
        pilot_only=True,
        tiny_verification_only=False,
        selection_eligible=False,
        definitive_training=False,
        scientific_metric_eligible=False,
        wrapper_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# I. Pilot summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8BoundedPilotSummary:
    """Non-scientific, pilot-only engineering diagnostics summary."""

    schema_name: str
    schema_version: str
    train_positive_anonymous_case_id: str
    train_empty_anonymous_case_id: str
    validation_anonymous_case_id: str
    executed_step_count: int
    step_finite_status: tuple[bool, ...]
    positive_patch_count: int
    negative_patch_count: int
    train_forward_shape: tuple[int, ...]
    validation_output_shape: tuple[int, ...]
    validation_finite_status: bool
    sliding_window_count: int
    checkpoint_round_trip_verified: bool
    checkpoint_sha256: str
    checkpoint_byte_size: int
    elapsed_seconds: float
    wall_clock_limit_seconds: float
    completion_status: str
    blocker_reason_codes: tuple[str, ...]
    pilot_only: bool
    definitive_training: bool
    freeze_eligible: bool
    selection_eligible: bool
    scientific_metric_eligible: bool
    non_scientific_disclaimer: str
    config_hash: str
    access_ledger_hash: str
    checkpoint_metadata_hash: str
    summary_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_BOUNDED_PILOT_SUMMARY_SCHEMA_NAME)
        _require_schema_version(self.schema_version)
        if self.pilot_only is not True:
            raise Phase8BoundedPilotError("pilot_only must be True.")
        if self.definitive_training is not False:
            raise Phase8BoundedPilotError("definitive_training must be False.")
        if self.freeze_eligible is not False:
            raise Phase8BoundedPilotError("freeze_eligible must be False.")
        if self.selection_eligible is not False:
            raise Phase8BoundedPilotError("selection_eligible must be False.")
        if self.scientific_metric_eligible is not False:
            raise Phase8BoundedPilotError("scientific_metric_eligible must be False.")
        if not self.non_scientific_disclaimer:
            raise Phase8BoundedPilotError("non_scientific_disclaimer must be nonempty.")
        if self.completion_status not in {"completed", "blocked"}:
            raise Phase8BoundedPilotError("completion_status must be 'completed' or 'blocked'.")
        if self.completion_status == "blocked" and not self.blocker_reason_codes:
            raise Phase8BoundedPilotError(
                "blocked completion_status requires blocker_reason_codes."
            )
        if self.completion_status == "completed" and self.blocker_reason_codes:
            raise Phase8BoundedPilotError(
                "completed completion_status must not include blocker_reason_codes."
            )
        object.__setattr__(self, "step_finite_status", tuple(self.step_finite_status))
        object.__setattr__(self, "train_forward_shape", tuple(self.train_forward_shape))
        object.__setattr__(self, "validation_output_shape", tuple(self.validation_output_shape))
        object.__setattr__(self, "blocker_reason_codes", tuple(sorted(self.blocker_reason_codes)))
        _require_self_hash(
            self.summary_hash,
            _summary_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                train_positive_anonymous_case_id=self.train_positive_anonymous_case_id,
                train_empty_anonymous_case_id=self.train_empty_anonymous_case_id,
                validation_anonymous_case_id=self.validation_anonymous_case_id,
                executed_step_count=self.executed_step_count,
                step_finite_status=self.step_finite_status,
                positive_patch_count=self.positive_patch_count,
                negative_patch_count=self.negative_patch_count,
                train_forward_shape=self.train_forward_shape,
                validation_output_shape=self.validation_output_shape,
                validation_finite_status=self.validation_finite_status,
                sliding_window_count=self.sliding_window_count,
                checkpoint_round_trip_verified=self.checkpoint_round_trip_verified,
                checkpoint_sha256=self.checkpoint_sha256,
                checkpoint_byte_size=self.checkpoint_byte_size,
                elapsed_seconds=self.elapsed_seconds,
                wall_clock_limit_seconds=self.wall_clock_limit_seconds,
                completion_status=self.completion_status,
                blocker_reason_codes=self.blocker_reason_codes,
                pilot_only=self.pilot_only,
                definitive_training=self.definitive_training,
                freeze_eligible=self.freeze_eligible,
                selection_eligible=self.selection_eligible,
                scientific_metric_eligible=self.scientific_metric_eligible,
                non_scientific_disclaimer=self.non_scientific_disclaimer,
                config_hash=self.config_hash,
                access_ledger_hash=self.access_ledger_hash,
                checkpoint_metadata_hash=self.checkpoint_metadata_hash,
            ),
            field_name="summary_hash",
        )


def _summary_payload(
    *,
    schema_name: str,
    schema_version: str,
    train_positive_anonymous_case_id: str,
    train_empty_anonymous_case_id: str,
    validation_anonymous_case_id: str,
    executed_step_count: int,
    step_finite_status: Sequence[bool],
    positive_patch_count: int,
    negative_patch_count: int,
    train_forward_shape: Sequence[int],
    validation_output_shape: Sequence[int],
    validation_finite_status: bool,
    sliding_window_count: int,
    checkpoint_round_trip_verified: bool,
    checkpoint_sha256: str,
    checkpoint_byte_size: int,
    elapsed_seconds: float,
    wall_clock_limit_seconds: float,
    completion_status: str,
    blocker_reason_codes: Sequence[str],
    pilot_only: bool,
    definitive_training: bool,
    freeze_eligible: bool,
    selection_eligible: bool,
    scientific_metric_eligible: bool,
    non_scientific_disclaimer: str,
    config_hash: str,
    access_ledger_hash: str,
    checkpoint_metadata_hash: str,
) -> dict[str, JsonValue]:
    return {
        "access_ledger_hash": access_ledger_hash,
        "blocker_reason_codes": list(blocker_reason_codes),
        "checkpoint_byte_size": checkpoint_byte_size,
        "checkpoint_metadata_hash": checkpoint_metadata_hash,
        "checkpoint_round_trip_verified": checkpoint_round_trip_verified,
        "checkpoint_sha256": checkpoint_sha256,
        "completion_status": completion_status,
        "config_hash": config_hash,
        "definitive_training": definitive_training,
        "elapsed_seconds": elapsed_seconds,
        "executed_step_count": executed_step_count,
        "freeze_eligible": freeze_eligible,
        "negative_patch_count": negative_patch_count,
        "non_scientific_disclaimer": non_scientific_disclaimer,
        "pilot_only": pilot_only,
        "positive_patch_count": positive_patch_count,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "scientific_metric_eligible": scientific_metric_eligible,
        "selection_eligible": selection_eligible,
        "sliding_window_count": sliding_window_count,
        "step_finite_status": list(step_finite_status),
        "train_empty_anonymous_case_id": train_empty_anonymous_case_id,
        "train_forward_shape": list(train_forward_shape),
        "train_positive_anonymous_case_id": train_positive_anonymous_case_id,
        "validation_anonymous_case_id": validation_anonymous_case_id,
        "validation_finite_status": validation_finite_status,
        "validation_output_shape": list(validation_output_shape),
        "wall_clock_limit_seconds": wall_clock_limit_seconds,
    }


def phase8_bounded_pilot_summary_to_dict(
    summary: Phase8BoundedPilotSummary,
) -> dict[str, JsonValue]:
    """Convert the pilot summary to a canonical mapping."""

    payload = _summary_payload(
        schema_name=summary.schema_name,
        schema_version=summary.schema_version,
        train_positive_anonymous_case_id=summary.train_positive_anonymous_case_id,
        train_empty_anonymous_case_id=summary.train_empty_anonymous_case_id,
        validation_anonymous_case_id=summary.validation_anonymous_case_id,
        executed_step_count=summary.executed_step_count,
        step_finite_status=summary.step_finite_status,
        positive_patch_count=summary.positive_patch_count,
        negative_patch_count=summary.negative_patch_count,
        train_forward_shape=summary.train_forward_shape,
        validation_output_shape=summary.validation_output_shape,
        validation_finite_status=summary.validation_finite_status,
        sliding_window_count=summary.sliding_window_count,
        checkpoint_round_trip_verified=summary.checkpoint_round_trip_verified,
        checkpoint_sha256=summary.checkpoint_sha256,
        checkpoint_byte_size=summary.checkpoint_byte_size,
        elapsed_seconds=summary.elapsed_seconds,
        wall_clock_limit_seconds=summary.wall_clock_limit_seconds,
        completion_status=summary.completion_status,
        blocker_reason_codes=summary.blocker_reason_codes,
        pilot_only=summary.pilot_only,
        definitive_training=summary.definitive_training,
        freeze_eligible=summary.freeze_eligible,
        selection_eligible=summary.selection_eligible,
        scientific_metric_eligible=summary.scientific_metric_eligible,
        non_scientific_disclaimer=summary.non_scientific_disclaimer,
        config_hash=summary.config_hash,
        access_ledger_hash=summary.access_ledger_hash,
        checkpoint_metadata_hash=summary.checkpoint_metadata_hash,
    )
    payload["summary_hash"] = summary.summary_hash
    return payload


# ---------------------------------------------------------------------------
# Bounded model/training/checkpoint pipeline
# ---------------------------------------------------------------------------


def _import_torch() -> Any:
    import torch  # type: ignore[import-not-found]

    return torch


def _import_monai() -> Any:
    import monai  # type: ignore[import-not-found]

    return monai


def _configure_pilot_determinism(*, torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _build_pilot_segresnet_model(*, torch: Any, monai: Any) -> Any:
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


def _validate_raw_case_pair_geometry(image_path: Path, label_path: Path) -> None:
    """Validate shape/affine/finite geometry for one raw train/validation pair.

    Mirrors
    :func:`protoem_ct.external.tiny_real_verification._validate_raw_case_pair_geometry`;
    see that module's docstring for why the raw MSD/LiTS three-class label
    domain is validated here rather than a strict binary domain.
    """

    if not image_path.exists() or not image_path.is_file():
        raise Phase8BoundedPilotDataError(f"image path is not a regular file: {image_path}")
    if not label_path.exists() or not label_path.is_file():
        raise Phase8BoundedPilotDataError(f"label path is not a regular file: {label_path}")

    try:
        image = cast(Any, nib.load(str(image_path)))
        label = cast(Any, nib.load(str(label_path)))
        image_array = np.asanyarray(image.dataobj)
        label_array = np.asanyarray(label.dataobj)
        image_affine = np.asarray(image.affine, dtype=float)
        label_affine = np.asarray(label.affine, dtype=float)
    except (OSError, ValueError) as exc:
        raise Phase8BoundedPilotDataError(f"failed to read NIfTI pair: {exc}") from exc

    if image_array.shape != label_array.shape:
        raise Phase8BoundedPilotDataError(
            f"image and label shapes differ: {image_array.shape} vs {label_array.shape}"
        )
    if image_array.ndim != 3:
        raise Phase8BoundedPilotDataError("volume must be 3-dimensional.")
    if not np.allclose(
        image_affine, label_affine, atol=CASE_PAIR_AFFINE_TOLERANCE_MM, rtol=0.0, equal_nan=False
    ):
        raise Phase8BoundedPilotDataError(
            f"image and label affines differ beyond tolerance {CASE_PAIR_AFFINE_TOLERANCE_MM}"
        )

    # Approved preprocessing asserts RAS orientation only; no reorientation is
    # ever performed. A non-RAS pair is a fail-closed condition, not a
    # silently-corrected one.
    image_axcodes = cast(tuple[str, ...], cast(Any, nib).aff2axcodes(image_affine))
    label_axcodes = cast(tuple[str, ...], cast(Any, nib).aff2axcodes(label_affine))
    if (
        tuple(image_axcodes) != REQUIRED_ORIENTATION_AXCODES
        or tuple(label_axcodes) != REQUIRED_ORIENTATION_AXCODES
    ):
        raise Phase8BoundedPilotOrientationError(
            "image/label pair must be RAS-oriented "
            f"({REQUIRED_ORIENTATION_AXCODES}); got image={image_axcodes}, label={label_axcodes}."
        )

    if not bool(np.isfinite(image_array).all()):
        raise Phase8BoundedPilotDataError("image data contains NaN or infinite values.")
    if not bool(np.isfinite(label_array).all()):
        raise Phase8BoundedPilotDataError("label data contains NaN or infinite values.")

    unique_values = np.unique(label_array)
    invalid_values = unique_values[~np.isin(unique_values, RAW_LABEL_ALLOWED_VALUES)]
    if invalid_values.size:
        raise Phase8BoundedPilotLabelDomainError(
            f"label data must be drawn from the raw LiTS/MSD domain "
            f"{RAW_LABEL_ALLOWED_VALUES}, got {invalid_values}"
        )


def _normalize_hu(image_array: np.ndarray, *, config: Phase8BoundedPilotConfig) -> np.ndarray:
    clipped = np.clip(
        image_array.astype(np.float32, copy=True), config.hu_window_min, config.hu_window_max
    )
    return cast(
        np.ndarray,
        ((clipped - config.hu_window_min) / (config.hu_window_max - config.hu_window_min)) * 2.0
        - 1.0,
    )


def _extract_padded_patch(
    volume: np.ndarray,
    *,
    center: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    fill_value: float,
) -> np.ndarray:
    """Extract a fixed-size patch centered on ``center``, zero-padding at volume boundaries.

    The patch box is computed as ``[center - patch_size//2, start + patch_size)``
    per axis. Any portion of that box that falls outside ``volume`` is
    explicitly padded with ``fill_value`` via :func:`numpy.pad`, rather than
    shifted to stay in bounds, so this satisfies the Substage 4B requirement
    to "safely pad at volume boundaries."
    """

    starts: list[int] = []
    ends: list[int] = []
    pre_pads: list[int] = []
    post_pads: list[int] = []
    for axis in range(3):
        half = patch_size[axis] // 2
        raw_start = center[axis] - half
        raw_end = raw_start + patch_size[axis]
        dim_size = volume.shape[axis]
        pre_pad = max(0, -raw_start)
        post_pad = max(0, raw_end - dim_size)
        clipped_start = max(raw_start, 0)
        clipped_end = min(raw_end, dim_size)
        starts.append(clipped_start)
        ends.append(clipped_end)
        pre_pads.append(pre_pad)
        post_pads.append(post_pad)

    cropped = volume[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]]
    padded = np.pad(
        cropped,
        pad_width=tuple(zip(pre_pads, post_pads, strict=True)),
        mode="constant",
        constant_values=fill_value,
    )
    if padded.shape != patch_size:
        raise Phase8BoundedPilotDataError(
            f"padded patch shape {padded.shape} does not equal required {patch_size}."
        )
    return padded


def _load_pilot_training_patch(
    *,
    image_array: np.ndarray,
    label_array: np.ndarray,
    center: tuple[int, int, int],
    config: Phase8BoundedPilotConfig,
    torch: Any,
) -> tuple[Any, Any]:
    """Build one training patch tensor pair centered on ``center``.

    Applies the fixed (non-data-fit) HU clip/normalize window, then a
    deterministic, boundary-safe padded crop to ``config.patch_size``.
    Derives the binary tumor-vs-background target from the raw label by the
    fixed, non-data-dependent rule
    ``foreground = (raw_label == RAW_LABEL_TUMOR_VALUE)``.
    """

    normalized_image = _normalize_hu(image_array, config=config)
    image_patch = _extract_padded_patch(
        normalized_image,
        center=center,
        patch_size=config.patch_size,
        fill_value=_PADDED_IMAGE_FILL_VALUE,
    )
    label_patch_raw = _extract_padded_patch(
        label_array.astype(np.float32, copy=False),
        center=center,
        patch_size=config.patch_size,
        fill_value=_PADDED_LABEL_FILL_VALUE,
    )
    label_patch = (np.rint(label_patch_raw).astype(np.int64) == RAW_LABEL_TUMOR_VALUE).astype(
        np.int64
    )

    image_tensor = torch.from_numpy(np.ascontiguousarray(image_patch)[None, None, ...]).to(
        dtype=torch.float32
    )
    label_tensor = torch.from_numpy(np.ascontiguousarray(label_patch)[None, None, ...]).to(
        dtype=torch.long
    )
    return image_tensor, label_tensor


def _run_bounded_training_steps(
    *,
    model: Any,
    optimizer: Any,
    loss_function: Any,
    schedule: Sequence[Phase8BoundedPilotPatchSample],
    case_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    config: Phase8BoundedPilotConfig,
    torch: Any,
    start_time: float,
    wall_clock_limit_seconds: float,
) -> tuple[tuple[bool, ...], tuple[int, ...]]:
    finite_status: list[bool] = []
    train_forward_shape: tuple[int, ...] = ()
    for step_index, sample in enumerate(schedule):
        image_array, label_array = case_arrays[sample.pilot_role]
        image_tensor, label_tensor = _load_pilot_training_patch(
            image_array=image_array,
            label_array=label_array,
            center=sample.center,
            config=config,
            torch=torch,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        if step_index == 0:
            train_forward_shape = tuple(int(v) for v in logits.shape)
        loss = loss_function(logits, label_tensor)
        if not bool(torch.isfinite(loss).item()):
            raise Phase8BoundedPilotRuntimeError("nonfinite_loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                raise Phase8BoundedPilotRuntimeError("nonfinite_gradient")
        optimizer.step()
        finite_status.append(True)
        _check_watchdog(
            start_time=start_time,
            wall_clock_limit_seconds=wall_clock_limit_seconds,
            checkpoint=f"after_step_{step_index}",
        )
    return tuple(finite_status), train_forward_shape


def _save_pilot_checkpoint_to_bytes(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    seed: int,
    step_count: int,
) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "schema_name": "phase8_definitive_training_pilot_checkpoint",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "step_count": step_count,
            "pilot_only": True,
        },
        buffer,
    )
    return buffer.getvalue()


def _reload_and_verify_checkpoint_from_bytes(
    *,
    torch: Any,
    monai: Any,
    checkpoint_bytes: bytes,
) -> bool:
    try:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
        reloaded_model = _build_pilot_segresnet_model(torch=torch, monai=monai)
        reloaded_model.load_state_dict(payload["model_state_dict"])
    except (
        MemoryError,
        RuntimeError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
    ) as exc:
        raise Phase8BoundedPilotRuntimeError("checkpoint_reload_failed") from exc
    return True


def _count_sliding_windows(
    *, volume_shape: tuple[int, int, int], roi_size: tuple[int, int, int], overlap: float
) -> int:
    """Return a deterministic diagnostic count of sliding-window inference patches.

    This is a self-contained, documented restatement of MONAI's public
    sliding-window scan-interval behaviour (an interval of
    ``round(roi * (1 - overlap))``, clamped to at least 1, and 1 window along
    any axis where the volume is no larger than the ROI) used only to
    populate the permitted ``sliding_window_count`` diagnostic field; it does
    not influence model execution.
    """

    total = 1
    for axis in range(3):
        dim = volume_shape[axis]
        roi = roi_size[axis]
        if dim <= roi:
            total *= 1
            continue
        interval = max(1, int(round(roi * (1.0 - overlap))))
        total *= 1 + -(-(dim - roi) // interval)  # ceil division
    return total


def _run_pilot_validation_sliding_window(
    *,
    torch: Any,
    monai: Any,
    model: Any,
    image_array: np.ndarray,
    label_array: np.ndarray,
    config: Phase8BoundedPilotConfig,
) -> tuple[tuple[int, ...], bool, int]:
    """Run exactly one full-volume sliding-window validation forward pass.

    Returns only the permitted diagnostics: output shape, a finite-status
    flag, and a sliding-window count. Computes no Dice/IoU/HD95/NSD or any
    other scientific segmentation metric.
    """

    normalized_image = _normalize_hu(image_array, config=config)
    unique_label_values = np.unique(label_array)
    invalid_values = unique_label_values[~np.isin(unique_label_values, RAW_LABEL_ALLOWED_VALUES)]
    if invalid_values.size:
        raise Phase8BoundedPilotLabelDomainError(
            f"validation label data must be drawn from the raw LiTS/MSD domain "
            f"{RAW_LABEL_ALLOWED_VALUES}, got {invalid_values}"
        )

    image_tensor = torch.from_numpy(np.ascontiguousarray(normalized_image)[None, None, ...]).to(
        dtype=torch.float32
    )

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
    finite_logits = bool(torch.isfinite(logits).all().item())
    if not finite_logits:
        raise Phase8BoundedPilotRuntimeError("nonfinite_validation_logits")
    probabilities = torch.softmax(logits, dim=1)
    if not bool(torch.isfinite(probabilities).all().item()):
        raise Phase8BoundedPilotRuntimeError("nonfinite_validation_probabilities")

    sliding_window_count = _count_sliding_windows(
        volume_shape=cast(tuple[int, int, int], tuple(int(v) for v in image_array.shape)),
        roi_size=config.sliding_window_roi_size,
        overlap=config.sliding_window_overlap,
    )
    output_shape = tuple(int(v) for v in logits.shape)
    return output_shape, finite_logits, sliding_window_count


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8BoundedPilotResult:
    """Result of one bounded Phase 8 real-data training pilot run."""

    output_root: Path
    checkpoint_path: Path
    config_hash: str
    access_ledger_hash: str
    checkpoint_metadata_hash: str
    summary_hash: str
    artifact_hashes: dict[str, str]


def _publish_json(path: Path, data: bytes) -> None:
    try:
        publish_text_no_overwrite(
            text=data.decode("utf-8"),
            output_path=path,
            temporary_exists_message=("temporary Phase 8 bounded pilot output already exists"),
            final_exists_message=("Phase 8 bounded pilot output already exists"),
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8BoundedPilotPublicationError(
            "failed to publish bounded pilot artifact."
        ) from exc


def run_phase8_bounded_pilot(  # noqa: PLR0913 -- every parameter is an explicit, required safety input.
    *,
    manifest_path: Path,
    split_path: Path,
    lesion_components_path: Path,
    expected_manifest_sha256: str,
    expected_split_sha256: str,
    expected_lesion_components_sha256: str,
    input_binding_path: Path,
    candidate_inventory_path: Path,
    preprocessing_decision_path: Path,
    expected_input_binding_sha256: str,
    expected_candidate_inventory_sha256: str,
    expected_preprocessing_decision_sha256: str,
    dataset_root: Path,
    output_root: Path,
    repository_root: Path,
    git_commit: str,
    package_versions: Mapping[str, str],
    seed: int = 1729,
    wall_clock_limit_seconds: float = DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
) -> Phase8BoundedPilotResult:
    """Run the bounded, verification-scale Phase 8 real-data training pilot.

    Every pre-flight safety check (compute-limit config, output-root path
    safety, dataset-root path safety, prerequisite reconstruction/cross-
    check, deterministic case selection) runs before any NIfTI file is
    opened. All disk writes -- the checkpoint file and the four JSON
    artifacts -- happen in a single commit at the very end, only after the
    full bounded pipeline (training, checkpoint round-trip, sliding-window
    validation) completes successfully; if the wall-clock watchdog or any
    other fail-closed condition fires at any point, nothing is written and
    ``output_root`` is never created.
    """

    start_time = time.monotonic()

    # Pre-flight 1: compute-limit config (no filesystem access).
    config = build_phase8_bounded_pilot_config(
        seed=seed,
        originating_git_commit=git_commit,
        package_versions=package_versions,
        wall_clock_limit_seconds=wall_clock_limit_seconds,
    )
    _check_watchdog(
        start_time=start_time,
        wall_clock_limit_seconds=config.wall_clock_limit_seconds,
        checkpoint="pre_flight_config",
    )

    # Pre-flight 2: output-root path safety (must not already exist).
    canonical_repository_root = repository_root.resolve(strict=True)
    paths = validate_phase8_bounded_pilot_output_root(
        output_root, repository_root=canonical_repository_root
    )

    # Pre-flight 3: dataset-root path safety.
    canonical_dataset_root = validate_explicit_dataset_root(dataset_root)

    # Pre-flight 4: manifest/split hash-verified load (JSON only, no pixels).
    resolved_manifest_path = manifest_path.resolve(strict=True)
    resolved_split_path = split_path.resolve(strict=True)
    resolved_lesion_components_path = lesion_components_path.resolve(strict=True)
    if sha256_file(resolved_manifest_path) != expected_manifest_sha256:
        raise Phase8BoundedPilotPrerequisitesError("manifest file sha256 mismatch.")
    if sha256_file(resolved_split_path) != expected_split_sha256:
        raise Phase8BoundedPilotPrerequisitesError("split file sha256 mismatch.")
    if sha256_file(resolved_lesion_components_path) != expected_lesion_components_sha256:
        raise Phase8BoundedPilotPrerequisitesError("lesion components file sha256 mismatch.")
    manifest = phase2_artifact_from_json(
        resolved_manifest_path.read_text(encoding="utf-8"), DatasetManifest
    )
    split = phase2_artifact_from_json(
        resolved_split_path.read_text(encoding="utf-8"), DevelopmentSplitManifest
    )
    lesion_components = phase2_artifact_from_json(
        resolved_lesion_components_path.read_text(encoding="utf-8"), LesionComponentsArtifact
    )
    if lesion_components.manifest_hash != manifest.manifest_hash:
        raise Phase8BoundedPilotPrerequisitesError(
            "lesion components manifest_hash does not match the loaded manifest."
        )
    if lesion_components.split_hash != split.split_hash:
        raise Phase8BoundedPilotPrerequisitesError(
            "lesion components split_hash does not match the loaded split."
        )

    # Pre-flight 5: Substage 4B prerequisite reconstruction and cross-check.
    verify_phase8_bounded_pilot_prerequisites(
        input_binding_path=input_binding_path,
        candidate_inventory_path=candidate_inventory_path,
        preprocessing_decision_path=preprocessing_decision_path,
        expected_input_binding_sha256=expected_input_binding_sha256,
        expected_candidate_inventory_sha256=expected_candidate_inventory_sha256,
        expected_preprocessing_decision_sha256=expected_preprocessing_decision_sha256,
        manifest=manifest,
        split=split,
    )
    _check_watchdog(
        start_time=start_time,
        wall_clock_limit_seconds=config.wall_clock_limit_seconds,
        checkpoint="pre_flight_prerequisites",
    )

    # Pre-flight 6: deterministic metadata-only case selection.
    train_positive_case, train_empty_case, validation_case = select_phase8_bounded_pilot_cases(
        manifest=manifest, split=split, lesion_components=lesion_components
    )

    train_positive_image_path = resolve_regular_file_beneath_root(
        canonical_dataset_root, train_positive_case.relative_image_path
    )
    train_positive_label_path = resolve_regular_file_beneath_root(
        canonical_dataset_root, train_positive_case.relative_label_path
    )
    train_empty_image_path = resolve_regular_file_beneath_root(
        canonical_dataset_root, train_empty_case.relative_image_path
    )
    train_empty_label_path = resolve_regular_file_beneath_root(
        canonical_dataset_root, train_empty_case.relative_label_path
    )
    validation_image_path = resolve_regular_file_beneath_root(
        canonical_dataset_root, validation_case.relative_image_path
    )
    validation_label_path = resolve_regular_file_beneath_root(
        canonical_dataset_root, validation_case.relative_label_path
    )

    _check_watchdog(
        start_time=start_time,
        wall_clock_limit_seconds=config.wall_clock_limit_seconds,
        checkpoint="before_first_nifti_open",
    )

    # First real medical-file access: geometry/finite/domain validation.
    _validate_raw_case_pair_geometry(train_positive_image_path, train_positive_label_path)
    _validate_raw_case_pair_geometry(train_empty_image_path, train_empty_label_path)

    train_positive_image = cast(Any, nib.load(str(train_positive_image_path)))
    train_positive_label = cast(Any, nib.load(str(train_positive_label_path)))
    train_empty_image = cast(Any, nib.load(str(train_empty_image_path)))
    train_empty_label = cast(Any, nib.load(str(train_empty_label_path)))
    train_positive_image_array = np.asanyarray(train_positive_image.dataobj)
    train_positive_label_array = np.asanyarray(train_positive_label.dataobj)
    train_empty_image_array = np.asanyarray(train_empty_image.dataobj)
    train_empty_label_array = np.asanyarray(train_empty_label.dataobj)

    schedule = build_phase8_bounded_pilot_sampling_schedule(
        positive_case_label=train_positive_label_array,
        empty_case_label=train_empty_label_array,
        seed=config.seed,
    )
    positive_patch_count = sum(1 for sample in schedule if sample.is_positive)
    negative_patch_count = len(schedule) - positive_patch_count

    torch = _import_torch()
    monai = _import_monai()
    configure_phase3_cpu_torch_runtime(torch=torch)
    _configure_pilot_determinism(torch=torch, seed=config.seed)

    model = _build_pilot_segresnet_model(torch=torch, monai=monai)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)

    case_arrays = {
        "train_positive": (train_positive_image_array, train_positive_label_array),
        "train_empty": (train_empty_image_array, train_empty_label_array),
    }
    finite_status, train_forward_shape = _run_bounded_training_steps(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        schedule=schedule,
        case_arrays=case_arrays,
        config=config,
        torch=torch,
        start_time=start_time,
        wall_clock_limit_seconds=config.wall_clock_limit_seconds,
    )

    checkpoint_bytes = _save_pilot_checkpoint_to_bytes(
        torch=torch,
        model=model,
        optimizer=optimizer,
        seed=config.seed,
        step_count=len(finite_status),
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint_byte_size = len(checkpoint_bytes)

    checkpoint_round_trip_verified = _reload_and_verify_checkpoint_from_bytes(
        torch=torch, monai=monai, checkpoint_bytes=checkpoint_bytes
    )

    _check_watchdog(
        start_time=start_time,
        wall_clock_limit_seconds=config.wall_clock_limit_seconds,
        checkpoint="before_validation",
    )

    _validate_raw_case_pair_geometry(validation_image_path, validation_label_path)
    validation_image = cast(Any, nib.load(str(validation_image_path)))
    validation_label = cast(Any, nib.load(str(validation_label_path)))
    validation_image_array = np.asanyarray(validation_image.dataobj)
    validation_label_array = np.asanyarray(validation_label.dataobj)

    validation_output_shape, validation_finite_status, sliding_window_count = (
        _run_pilot_validation_sliding_window(
            torch=torch,
            monai=monai,
            model=model,
            image_array=validation_image_array,
            label_array=validation_label_array,
            config=config,
        )
    )

    elapsed_seconds = time.monotonic() - start_time
    _check_watchdog(
        start_time=start_time,
        wall_clock_limit_seconds=config.wall_clock_limit_seconds,
        checkpoint="before_publication",
    )

    ledger = build_phase8_bounded_pilot_access_ledger(
        train_positive_case=train_positive_case,
        train_empty_case=train_empty_case,
        validation_case=validation_case,
    )
    checkpoint_metadata = build_phase8_bounded_pilot_checkpoint_metadata(
        config=config,
        development_manifest_hash=manifest.manifest_hash,
        development_split_hash=split.split_hash,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_size=checkpoint_byte_size,
    )
    summary_payload = _summary_payload(
        schema_name=PHASE8_BOUNDED_PILOT_SUMMARY_SCHEMA_NAME,
        schema_version=PHASE8_BOUNDED_PILOT_SCHEMA_VERSION,
        train_positive_anonymous_case_id=train_positive_case.anonymous_case_id,
        train_empty_anonymous_case_id=train_empty_case.anonymous_case_id,
        validation_anonymous_case_id=validation_case.anonymous_case_id,
        executed_step_count=len(finite_status),
        step_finite_status=finite_status,
        positive_patch_count=positive_patch_count,
        negative_patch_count=negative_patch_count,
        train_forward_shape=train_forward_shape,
        validation_output_shape=validation_output_shape,
        validation_finite_status=validation_finite_status,
        sliding_window_count=sliding_window_count,
        checkpoint_round_trip_verified=checkpoint_round_trip_verified,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_size=checkpoint_byte_size,
        elapsed_seconds=elapsed_seconds,
        wall_clock_limit_seconds=config.wall_clock_limit_seconds,
        completion_status="completed",
        blocker_reason_codes=(),
        pilot_only=True,
        definitive_training=False,
        freeze_eligible=False,
        selection_eligible=False,
        scientific_metric_eligible=False,
        non_scientific_disclaimer=NON_SCIENTIFIC_DISCLAIMER,
        config_hash=config.config_hash,
        access_ledger_hash=ledger.ledger_hash,
        checkpoint_metadata_hash=checkpoint_metadata.wrapper_hash,
    )
    summary = Phase8BoundedPilotSummary(
        schema_name=PHASE8_BOUNDED_PILOT_SUMMARY_SCHEMA_NAME,
        schema_version=PHASE8_BOUNDED_PILOT_SCHEMA_VERSION,
        train_positive_anonymous_case_id=train_positive_case.anonymous_case_id,
        train_empty_anonymous_case_id=train_empty_case.anonymous_case_id,
        validation_anonymous_case_id=validation_case.anonymous_case_id,
        executed_step_count=len(finite_status),
        step_finite_status=finite_status,
        positive_patch_count=positive_patch_count,
        negative_patch_count=negative_patch_count,
        train_forward_shape=train_forward_shape,
        validation_output_shape=validation_output_shape,
        validation_finite_status=validation_finite_status,
        sliding_window_count=sliding_window_count,
        checkpoint_round_trip_verified=checkpoint_round_trip_verified,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_size=checkpoint_byte_size,
        elapsed_seconds=elapsed_seconds,
        wall_clock_limit_seconds=config.wall_clock_limit_seconds,
        completion_status="completed",
        blocker_reason_codes=(),
        pilot_only=True,
        definitive_training=False,
        freeze_eligible=False,
        selection_eligible=False,
        scientific_metric_eligible=False,
        non_scientific_disclaimer=NON_SCIENTIFIC_DISCLAIMER,
        config_hash=config.config_hash,
        access_ledger_hash=ledger.ledger_hash,
        checkpoint_metadata_hash=checkpoint_metadata.wrapper_hash,
        summary_hash=sha256_json(summary_payload),
    )

    # Single commit point: create output_root and write every artifact only now.
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = paths.checkpoints_dir / PHASE8_BOUNDED_PILOT_CHECKPOINT_FILENAME
    if checkpoint_path.exists():
        raise Phase8BoundedPilotRuntimeError("checkpoint_collision")
    checkpoint_path.write_bytes(checkpoint_bytes)

    _publish_json(
        paths.output_root / PHASE8_BOUNDED_PILOT_CONFIG_FILENAME,
        canonical_json_bytes(phase8_bounded_pilot_config_to_dict(config)) + b"\n",
    )
    _publish_json(
        paths.output_root / PHASE8_BOUNDED_PILOT_ACCESS_LEDGER_FILENAME,
        canonical_json_bytes(phase8_bounded_pilot_access_ledger_to_dict(ledger)) + b"\n",
    )
    _publish_json(
        paths.output_root / PHASE8_BOUNDED_PILOT_CHECKPOINT_METADATA_FILENAME,
        canonical_json_bytes(phase8_bounded_pilot_checkpoint_metadata_to_dict(checkpoint_metadata))
        + b"\n",
    )
    _publish_json(
        paths.output_root / PHASE8_BOUNDED_PILOT_SUMMARY_FILENAME,
        canonical_json_bytes(phase8_bounded_pilot_summary_to_dict(summary)) + b"\n",
    )

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(paths.output_root.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    return Phase8BoundedPilotResult(
        output_root=paths.output_root,
        checkpoint_path=checkpoint_path,
        config_hash=config.config_hash,
        access_ledger_hash=ledger.ledger_hash,
        checkpoint_metadata_hash=checkpoint_metadata.wrapper_hash,
        summary_hash=summary.summary_hash,
        artifact_hashes=artifact_hashes,
    )


# ---------------------------------------------------------------------------
# Process-level supervision (parent/child watchdog)
# ---------------------------------------------------------------------------
#
# ``_check_watchdog`` above is a synchronous, in-process elapsed-time check. It
# can only fire *between* stages of ``run_phase8_bounded_pilot``'s own
# execution; it cannot interrupt a single long blocking call (sliding-window
# inference, the bounded training loop, or checkpoint I/O) while that call is
# still running. The functions below add a second, independent layer: a
# parent process supervises the *entire* real pilot execution (medical data
# loading through artifact publication) running inside an isolated child
# process, and can terminate that child at the operating-system level if it
# does not return within the wall-clock deadline. The existing per-stage
# ``_check_watchdog`` calls inside ``run_phase8_bounded_pilot`` are kept as
# defense-in-depth and are not modified by any of this.

_DEFAULT_CHILD_TERMINATION_GRACE_SECONDS: Final[float] = 5.0


def _phase8_bounded_pilot_child_entrypoint(
    conn: Connection,
    target: Callable[..., Phase8BoundedPilotResult],
    kwargs: Mapping[str, Any],
) -> None:
    """Run ``target(**kwargs)`` inside the child process and send back the outcome.

    Uses a :class:`multiprocessing.connection.Connection` (a ``Pipe`` endpoint)
    rather than a ``Queue``: ``Connection.send`` is a direct, synchronous write
    with no background feeder thread, so there is no risk of the child process
    exiting before a queued result has actually been flushed to the parent.
    """

    try:
        result = target(**kwargs)
    except BaseException as exc:  # noqa: BLE001 -- propagated to the parent unchanged.
        try:
            conn.send(("error", exc))
        except Exception:  # noqa: BLE001 -- exc itself failed to pickle; send a safe substitute.
            conn.send(
                (
                    "error",
                    Phase8BoundedPilotRuntimeError(
                        f"child process raised an unpicklable exception: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
        finally:
            conn.close()
        return
    conn.send(("ok", result))
    conn.close()


def run_phase8_bounded_pilot_with_watchdog(  # noqa: PLR0913 -- mirrors run_phase8_bounded_pilot's explicit inputs.
    *,
    manifest_path: Path,
    split_path: Path,
    lesion_components_path: Path,
    expected_manifest_sha256: str,
    expected_split_sha256: str,
    expected_lesion_components_sha256: str,
    input_binding_path: Path,
    candidate_inventory_path: Path,
    preprocessing_decision_path: Path,
    expected_input_binding_sha256: str,
    expected_candidate_inventory_sha256: str,
    expected_preprocessing_decision_sha256: str,
    dataset_root: Path,
    output_root: Path,
    repository_root: Path,
    git_commit: str,
    package_versions: Mapping[str, str],
    seed: int = 1729,
    wall_clock_limit_seconds: float = DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
    _target: Callable[..., Phase8BoundedPilotResult] = run_phase8_bounded_pilot,
    _multiprocessing_context: BaseContext | None = None,
    _termination_grace_seconds: float = _DEFAULT_CHILD_TERMINATION_GRACE_SECONDS,
) -> Phase8BoundedPilotResult:
    """Run the bounded Phase 8 pilot inside a supervised, killable child process.

    This wraps :func:`run_phase8_bounded_pilot` (unchanged, including its own
    internal per-stage ``_check_watchdog`` calls) with a second, process-level
    watchdog that can interrupt it even if it is blocked inside a single long
    call. The entire real pilot execution -- medical data loading, the bounded
    training loop, checkpoint round-trip, sliding-window validation, and
    artifact publication -- runs inside the child process, so all of it is
    protected by the same deadline.

    Semantics:

    * If the child returns within ``wall_clock_limit_seconds`` (accounting for
      time already spent by this parent before and while starting the child),
      its result or exception is propagated to the caller unchanged.
    * If the child does not return in time, it is terminated (SIGTERM, then
      escalated to ``.kill()`` if still alive after a short grace period),
      joined to avoid leaving a zombie/orphan process, and this function
      raises :class:`Phase8BoundedPilotProcessWatchdogTimeoutError`. No
      partial success is ever surfaced for this outcome, and this function
      performs no automatic retry.
    * The child process is always terminated and joined before this function
      returns or raises, on every code path (success, child-side exception,
      or timeout), via ``try``/``finally``, so no orphaned child process can
      remain.

    Every scientific/compute-limit parameter (patch size, ROI, overlap,
    hyperparameters, model, seed, step count, case selection, dataset root
    handling) is defined entirely by ``run_phase8_bounded_pilot`` itself and
    is passed through unchanged; this wrapper adds only process supervision.

    ``_target`` and ``_multiprocessing_context`` are testing-only seams (not
    part of the public contract): production callers must not pass them.
    Production always uses ``multiprocessing.get_context("spawn")`` -- the
    macOS default since Python 3.8, and safe here since the child is CPU-only
    -- which requires ``target``/``kwargs`` to be picklable; every parameter
    accepted above (``Path``, ``str``, ``float``, ``int``,
    ``Mapping[str, str]``) already is.
    """

    supervisor_start_time = time.monotonic()

    kwargs: dict[str, Any] = {
        "manifest_path": manifest_path,
        "split_path": split_path,
        "lesion_components_path": lesion_components_path,
        "expected_manifest_sha256": expected_manifest_sha256,
        "expected_split_sha256": expected_split_sha256,
        "expected_lesion_components_sha256": expected_lesion_components_sha256,
        "input_binding_path": input_binding_path,
        "candidate_inventory_path": candidate_inventory_path,
        "preprocessing_decision_path": preprocessing_decision_path,
        "expected_input_binding_sha256": expected_input_binding_sha256,
        "expected_candidate_inventory_sha256": expected_candidate_inventory_sha256,
        "expected_preprocessing_decision_sha256": expected_preprocessing_decision_sha256,
        "dataset_root": dataset_root,
        "output_root": output_root,
        "repository_root": repository_root,
        "git_commit": git_commit,
        "package_versions": dict(package_versions),
        "seed": seed,
        "wall_clock_limit_seconds": wall_clock_limit_seconds,
    }

    ctx = _multiprocessing_context or multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    # BaseContext.Process is dynamically bound per concrete context subclass;
    # mypy's typeshed stub does not expose it on the base class.
    child = ctx.Process(  # type: ignore[attr-defined]
        target=_phase8_bounded_pilot_child_entrypoint,
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
        join_timeout = max(0.0, wall_clock_limit_seconds - elapsed_before_join)
        child.join(timeout=join_timeout)

        if child.is_alive():
            raise Phase8BoundedPilotProcessWatchdogTimeoutError(
                f"process-level watchdog killed the child pilot process after "
                f"{wall_clock_limit_seconds:.3f}s with no publication; "
                f"the child was terminated at the operating-system process "
                f"level, not by the in-process elapsed-time check."
            )

        if parent_conn.poll():
            try:
                status, payload = parent_conn.recv()
            except EOFError as exc:
                raise Phase8BoundedPilotRuntimeError(
                    f"child pilot process exited (exitcode={child.exitcode}) without "
                    f"returning a result."
                ) from exc
        else:
            raise Phase8BoundedPilotRuntimeError(
                f"child pilot process exited (exitcode={child.exitcode}) without "
                f"returning a result."
            )

        if status == "error":
            raise cast(BaseException, payload)
        return cast(Phase8BoundedPilotResult, payload)
    finally:
        _terminate_and_join_child()
        parent_conn.close()


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _require_schema(value: str, expected: str) -> None:
    if value != expected:
        raise Phase8BoundedPilotError(f"schema_name must equal {expected!r}, got {value!r}.")


def _require_schema_version(value: str) -> None:
    if value != PHASE8_BOUNDED_PILOT_SCHEMA_VERSION:
        raise Phase8BoundedPilotError(
            f"schema_version must equal {PHASE8_BOUNDED_PILOT_SCHEMA_VERSION!r}."
        )


def _require_self_hash(observed: str, payload: dict[str, JsonValue], *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(observed):
        raise Phase8BoundedPilotError(f"{field_name} must be a lowercase SHA-256 hex.")
    expected = sha256_json(payload)
    if observed != expected:
        raise Phase8BoundedPilotError(f"{field_name} does not match its canonical payload.")
