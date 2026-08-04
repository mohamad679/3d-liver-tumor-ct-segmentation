"""Phase 8 tiny real-development engineering verification (Substage 3).

This module implements a deliberately bounded, CPU-only, engineering-only
dry/overfit verification that opens exactly one train and one validation
NIfTI image/label pair from an approved real medical-imaging development
dataset. It is the first module in this remediation track authorized to open
real dataset pixel bytes, and it is scoped as narrowly as possible:

* Exactly one train case and one validation case, selected by pure
  metadata-derived ordering (never by pixel content, lesion size, image
  shape, or spacing).
* CPU only, no AMP, batch size 1, at most two optimizer steps, at most one
  epoch, a single bounded spatial patch per case, zero DataLoader workers.
* No internal-test access, no external-dataset access, no scientific metric
  computation (no Dice/IoU/HD95), and no claim of segmentation performance,
  convergence, or checkpoint quality anywhere in this module's outputs.
* Every published checkpoint is hard-coded ``tiny_verification_only=True``,
  ``freeze_eligible=False``, ``selection_eligible=False``, and
  ``definitive_training=False`` -- there is no parameter capable of setting
  any of the three ineligibility flags to a more permissive value.

Every generated JSON artifact carries a canonical self-hash produced the same
way as every other artifact in this codebase
(:func:`protoem_ct.artifacts.hashing.sha256_json` over the payload with the
hash field itself excluded) and is published with
:func:`protoem_ct.data._phase2_publication.publish_text_no_overwrite`, which
never intentionally overwrites a pre-existing artifact.

``torch``, ``monai``, and (where avoidable) ``nibabel`` model/tensor access
are imported lazily inside function bodies so that importing this module
(and, more importantly, importing ``protoem_ct.cli.main``) never pulls in the
heavy baseline runtime dependencies.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import nibabel as nib
import numpy as np

from protoem_ct.artifacts import (
    DatasetManifest,
    DevelopmentSplitManifest,
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
    artifact_reference_to_dict,
    phase8_checkpoint_metadata_to_dict,
)
from protoem_ct.external.real_development_runner import (
    build_phase8_real_development_input_binding,
)

MappingLike: TypeAlias = Mapping[str, object]

# ---------------------------------------------------------------------------
# Schema identity
# ---------------------------------------------------------------------------

PHASE8_TINY_REAL_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_TINY_REAL_CONFIG_SCHEMA_NAME: Final[str] = "phase8_tiny_real_verification_config"
PHASE8_TINY_REAL_ACCESS_LEDGER_SCHEMA_NAME: Final[str] = "phase8_tiny_real_access_ledger"
PHASE8_TINY_REAL_CHECKPOINT_METADATA_SCHEMA_NAME: Final[str] = (
    "phase8_tiny_real_checkpoint_metadata"
)
PHASE8_TINY_REAL_VERIFICATION_SUMMARY_SCHEMA_NAME: Final[str] = (
    "phase8_tiny_real_verification_summary"
)

PHASE8_TINY_REAL_CONFIG_FILENAME: Final[str] = "phase8_tiny_real_verification_config.json"
PHASE8_TINY_REAL_ACCESS_LEDGER_FILENAME: Final[str] = "phase8_tiny_real_access_ledger.json"
PHASE8_TINY_REAL_CHECKPOINT_METADATA_FILENAME: Final[str] = (
    "phase8_tiny_real_checkpoint_metadata.json"
)
PHASE8_TINY_REAL_VERIFICATION_SUMMARY_FILENAME: Final[str] = (
    "phase8_tiny_real_verification_summary.json"
)
PHASE8_TINY_REAL_CHECKPOINT_SUBDIRECTORY_NAME: Final[str] = "checkpoints"
PHASE8_TINY_REAL_CHECKPOINT_FILENAME: Final[str] = "phase8_tiny_real_verification_checkpoint.pt"

NON_SCIENTIFIC_DISCLAIMER: Final[str] = (
    "verification_only: this run performs no scientific evaluation; it computes "
    "no Dice/IoU/HD95 or any segmentation-quality metric, and its results must "
    "not be read as evidence of segmentation performance, convergence, "
    "overfitting success, baseline validity, or checkpoint quality."
)

# ---------------------------------------------------------------------------
# Hard compute limits (Section C) -- validated, never silently clamped
# ---------------------------------------------------------------------------

REQUIRED_DEVICE_TYPE: Final[str] = "cpu"
REQUIRED_TRAIN_CASE_COUNT: Final[int] = 1
REQUIRED_VALIDATION_CASE_COUNT: Final[int] = 1
REQUIRED_BATCH_SIZE: Final[int] = 1
MIN_OPTIMIZER_STEPS: Final[int] = 1
MAX_OPTIMIZER_STEPS: Final[int] = 2
REQUIRED_MAX_EPOCHS: Final[int] = 1
REQUIRED_TRAIN_PATCHES_PER_STEP: Final[int] = 1
REQUIRED_MAX_VALIDATION_PATCHES: Final[int] = 1
REQUIRED_NUM_WORKERS: Final[int] = 0
MAX_PATCH_DIMS: Final[tuple[int, int, int]] = (32, 64, 64)
FIXED_HU_WINDOW_MIN: Final[float] = -1000.0
FIXED_HU_WINDOW_MAX: Final[float] = 1000.0
FIT_SCOPE_NO_DATA_DEPENDENT_FIT: Final[str] = "no_data_dependent_fit"
DEFAULT_PATCH_SIZE: Final[tuple[int, int, int]] = (24, 24, 16)

# Raw label domain for the approved LiTS/MSD Task03 Liver development cohort:
# 0=background, 1=liver, 2=tumor. This is the standard published LiTS/MSD
# convention, not a fitted or tunable value. PROJECT_SPEC.md fixes the
# project's task as binary tumor-vs-background segmentation, so the bounded
# patch loader derives foreground strictly from the raw tumor label value
# below; liver (1) and background (0) both map to the background class.
RAW_LABEL_TUMOR_VALUE: Final[int] = 2
RAW_LABEL_ALLOWED_VALUES: Final[tuple[int, ...]] = (0, 1, 2)
CASE_PAIR_AFFINE_TOLERANCE_MM: Final[float] = 1e-4

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{7,64}$")
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")


# ---------------------------------------------------------------------------
# Exception hierarchy (Section H)
# ---------------------------------------------------------------------------


class Phase8TinyRealVerificationError(ValueError):
    """Base error for the Phase 8 tiny real-development verification module."""


class Phase8TinyRealVerificationConfigError(Phase8TinyRealVerificationError):
    """Raised when the bounded verification config violates a hard compute limit."""


class Phase8TinyRealVerificationScopeError(Phase8TinyRealVerificationError):
    """Raised when a caller attempts to broaden the bounded verification scope."""


class Phase8TinyRealVerificationOutputRootError(Phase8TinyRealVerificationError):
    """Raised when the external output root fails the path-safety contract."""


class Phase8TinyRealVerificationCaseSelectionError(Phase8TinyRealVerificationError):
    """Raised when deterministic train/validation case selection cannot proceed."""


class Phase8TinyRealVerificationDataError(Phase8TinyRealVerificationError):
    """Raised when a selected case's data is missing, unreadable, or unsafe."""


class Phase8TinyRealVerificationLabelDomainError(Phase8TinyRealVerificationDataError):
    """Raised when a raw label's values fall outside the approved {0,1,2} domain."""


class Phase8TinyRealVerificationRuntimeError(Phase8TinyRealVerificationError):
    """Raised when a bounded model, training, or checkpoint operation fails closed."""


class Phase8TinyRealVerificationPublicationError(Phase8TinyRealVerificationError):
    """Raised when guarded artifact publication fails safely."""


# ---------------------------------------------------------------------------
# A. Compute-limit validated configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8TinyRealVerificationConfig:
    """Immutable, hashed configuration encoding every Section C hard limit."""

    schema_name: str
    schema_version: str
    device_type: str
    amp_enabled: bool
    train_case_count: int
    validation_case_count: int
    batch_size: int
    max_optimizer_steps: int
    max_epochs: int
    train_patches_per_step: int
    max_validation_patches: int
    num_workers: int
    patch_size: tuple[int, int, int]
    hu_window_min: float
    hu_window_max: float
    fit_scope: str
    seed: int
    originating_git_commit: str
    package_versions: tuple[tuple[str, str], ...]
    config_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_TINY_REAL_CONFIG_SCHEMA_NAME)
        _require_schema_version(self.schema_version)
        if self.device_type != REQUIRED_DEVICE_TYPE:
            raise Phase8TinyRealVerificationConfigError(
                "device_type must be 'cpu'; no other device is permitted."
            )
        if self.amp_enabled is not False:
            raise Phase8TinyRealVerificationConfigError("amp_enabled must be False.")
        if self.train_case_count != REQUIRED_TRAIN_CASE_COUNT:
            raise Phase8TinyRealVerificationConfigError("train_case_count must equal 1.")
        if self.validation_case_count != REQUIRED_VALIDATION_CASE_COUNT:
            raise Phase8TinyRealVerificationConfigError("validation_case_count must equal 1.")
        if self.batch_size != REQUIRED_BATCH_SIZE:
            raise Phase8TinyRealVerificationConfigError("batch_size must equal 1.")
        if not (MIN_OPTIMIZER_STEPS <= self.max_optimizer_steps <= MAX_OPTIMIZER_STEPS):
            raise Phase8TinyRealVerificationConfigError(
                f"max_optimizer_steps must satisfy {MIN_OPTIMIZER_STEPS} <= steps <= "
                f"{MAX_OPTIMIZER_STEPS}, got {self.max_optimizer_steps!r}."
            )
        if self.max_epochs != REQUIRED_MAX_EPOCHS:
            raise Phase8TinyRealVerificationConfigError("max_epochs must equal 1.")
        if self.train_patches_per_step != REQUIRED_TRAIN_PATCHES_PER_STEP:
            raise Phase8TinyRealVerificationConfigError("train_patches_per_step must equal 1.")
        if self.max_validation_patches != REQUIRED_MAX_VALIDATION_PATCHES:
            raise Phase8TinyRealVerificationConfigError("max_validation_patches must equal 1.")
        if self.num_workers != REQUIRED_NUM_WORKERS:
            raise Phase8TinyRealVerificationConfigError("num_workers must equal 0.")
        object.__setattr__(self, "patch_size", tuple(int(v) for v in self.patch_size))
        if len(self.patch_size) != 3:
            raise Phase8TinyRealVerificationConfigError("patch_size must have exactly 3 dims.")
        for dim_index, (dim_value, dim_bound) in enumerate(
            zip(self.patch_size, MAX_PATCH_DIMS, strict=True)
        ):
            if dim_value < 1:
                raise Phase8TinyRealVerificationConfigError(
                    f"patch_size[{dim_index}] must be positive."
                )
            if dim_value > dim_bound:
                raise Phase8TinyRealVerificationConfigError(
                    f"patch_size[{dim_index}]={dim_value} exceeds the bounded maximum "
                    f"{dim_bound} (spatial patch must be at most "
                    f"{MAX_PATCH_DIMS[0]}x{MAX_PATCH_DIMS[1]}x{MAX_PATCH_DIMS[2]})."
                )
        if self.hu_window_min != FIXED_HU_WINDOW_MIN or self.hu_window_max != FIXED_HU_WINDOW_MAX:
            raise Phase8TinyRealVerificationConfigError(
                "HU window must equal the fixed constant "
                f"[{FIXED_HU_WINDOW_MIN}, {FIXED_HU_WINDOW_MAX}]; it must not be data-fit."
            )
        if self.fit_scope != FIT_SCOPE_NO_DATA_DEPENDENT_FIT:
            raise Phase8TinyRealVerificationConfigError(
                f"fit_scope must equal {FIT_SCOPE_NO_DATA_DEPENDENT_FIT!r}."
            )
        if self.seed < 0:
            raise Phase8TinyRealVerificationConfigError("seed must be non-negative.")
        if not _GIT_COMMIT_RE.fullmatch(self.originating_git_commit):
            raise Phase8TinyRealVerificationConfigError(
                "originating_git_commit must match ^[0-9a-f]{7,64}$."
            )
        object.__setattr__(
            self,
            "package_versions",
            tuple(sorted((str(name), str(version)) for name, version in self.package_versions)),
        )
        if not self.package_versions:
            raise Phase8TinyRealVerificationConfigError("package_versions must not be empty.")
        for name, version in self.package_versions:
            if not name or not version:
                raise Phase8TinyRealVerificationConfigError(
                    "package_versions entries must be nonempty."
                )
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
                max_optimizer_steps=self.max_optimizer_steps,
                max_epochs=self.max_epochs,
                train_patches_per_step=self.train_patches_per_step,
                max_validation_patches=self.max_validation_patches,
                num_workers=self.num_workers,
                patch_size=self.patch_size,
                hu_window_min=self.hu_window_min,
                hu_window_max=self.hu_window_max,
                fit_scope=self.fit_scope,
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
    max_optimizer_steps: int,
    max_epochs: int,
    train_patches_per_step: int,
    max_validation_patches: int,
    num_workers: int,
    patch_size: Sequence[int],
    hu_window_min: float,
    hu_window_max: float,
    fit_scope: str,
    seed: int,
    originating_git_commit: str,
    package_versions: Sequence[tuple[str, str]],
) -> dict[str, JsonValue]:
    return {
        "amp_enabled": amp_enabled,
        "batch_size": batch_size,
        "device_type": device_type,
        "fit_scope": fit_scope,
        "hu_window_max": hu_window_max,
        "hu_window_min": hu_window_min,
        "max_epochs": max_epochs,
        "max_optimizer_steps": max_optimizer_steps,
        "max_validation_patches": max_validation_patches,
        "num_workers": num_workers,
        "originating_git_commit": originating_git_commit,
        "package_versions": {name: version for name, version in package_versions},
        "patch_size": list(patch_size),
        "schema_name": schema_name,
        "schema_version": schema_version,
        "seed": seed,
        "train_case_count": train_case_count,
        "train_patches_per_step": train_patches_per_step,
        "validation_case_count": validation_case_count,
    }


def phase8_tiny_real_verification_config_to_dict(
    config: Phase8TinyRealVerificationConfig,
) -> dict[str, JsonValue]:
    """Convert the bounded verification config to a canonical mapping."""

    payload = _config_payload(
        schema_name=config.schema_name,
        schema_version=config.schema_version,
        device_type=config.device_type,
        amp_enabled=config.amp_enabled,
        train_case_count=config.train_case_count,
        validation_case_count=config.validation_case_count,
        batch_size=config.batch_size,
        max_optimizer_steps=config.max_optimizer_steps,
        max_epochs=config.max_epochs,
        train_patches_per_step=config.train_patches_per_step,
        max_validation_patches=config.max_validation_patches,
        num_workers=config.num_workers,
        patch_size=config.patch_size,
        hu_window_min=config.hu_window_min,
        hu_window_max=config.hu_window_max,
        fit_scope=config.fit_scope,
        seed=config.seed,
        originating_git_commit=config.originating_git_commit,
        package_versions=config.package_versions,
    )
    payload["config_hash"] = config.config_hash
    return payload


def build_phase8_tiny_real_verification_config(
    *,
    max_optimizer_steps: int,
    seed: int,
    originating_git_commit: str,
    package_versions: Mapping[str, str],
    patch_size: tuple[int, int, int] = DEFAULT_PATCH_SIZE,
) -> Phase8TinyRealVerificationConfig:
    """Build and validate the bounded verification config from caller intent.

    Every Section C hard limit not exposed here (device, AMP, case counts,
    batch size, epochs, patches-per-step, worker count, HU window, fit scope)
    is hard-coded to its required value; only ``max_optimizer_steps``,
    ``patch_size``, ``seed``, and provenance fields are caller-controlled, and
    each is validated against its bound at construction time.
    """

    package_versions_tuple = tuple(sorted(package_versions.items()))
    payload = _config_payload(
        schema_name=PHASE8_TINY_REAL_CONFIG_SCHEMA_NAME,
        schema_version=PHASE8_TINY_REAL_SCHEMA_VERSION,
        device_type=REQUIRED_DEVICE_TYPE,
        amp_enabled=False,
        train_case_count=REQUIRED_TRAIN_CASE_COUNT,
        validation_case_count=REQUIRED_VALIDATION_CASE_COUNT,
        batch_size=REQUIRED_BATCH_SIZE,
        max_optimizer_steps=max_optimizer_steps,
        max_epochs=REQUIRED_MAX_EPOCHS,
        train_patches_per_step=REQUIRED_TRAIN_PATCHES_PER_STEP,
        max_validation_patches=REQUIRED_MAX_VALIDATION_PATCHES,
        num_workers=REQUIRED_NUM_WORKERS,
        patch_size=patch_size,
        hu_window_min=FIXED_HU_WINDOW_MIN,
        hu_window_max=FIXED_HU_WINDOW_MAX,
        fit_scope=FIT_SCOPE_NO_DATA_DEPENDENT_FIT,
        seed=seed,
        originating_git_commit=originating_git_commit,
        package_versions=package_versions_tuple,
    )
    return Phase8TinyRealVerificationConfig(
        schema_name=PHASE8_TINY_REAL_CONFIG_SCHEMA_NAME,
        schema_version=PHASE8_TINY_REAL_SCHEMA_VERSION,
        device_type=REQUIRED_DEVICE_TYPE,
        amp_enabled=False,
        train_case_count=REQUIRED_TRAIN_CASE_COUNT,
        validation_case_count=REQUIRED_VALIDATION_CASE_COUNT,
        batch_size=REQUIRED_BATCH_SIZE,
        max_optimizer_steps=max_optimizer_steps,
        max_epochs=REQUIRED_MAX_EPOCHS,
        train_patches_per_step=REQUIRED_TRAIN_PATCHES_PER_STEP,
        max_validation_patches=REQUIRED_MAX_VALIDATION_PATCHES,
        num_workers=REQUIRED_NUM_WORKERS,
        patch_size=patch_size,
        hu_window_min=FIXED_HU_WINDOW_MIN,
        hu_window_max=FIXED_HU_WINDOW_MAX,
        fit_scope=FIT_SCOPE_NO_DATA_DEPENDENT_FIT,
        seed=seed,
        originating_git_commit=originating_git_commit,
        package_versions=package_versions_tuple,
        config_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# B. Deterministic case selection (pure metadata, no pixel access)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8TinyRealSelectedCase:
    """One deterministically selected case, identified by anonymous IDs only."""

    anonymous_patient_id: str
    anonymous_case_id: str
    partition: str
    relative_image_path: str
    relative_label_path: str


def select_phase8_tiny_real_development_cases(
    *,
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
) -> tuple[Phase8TinyRealSelectedCase, Phase8TinyRealSelectedCase]:
    """Deterministically select exactly one train and one validation case.

    Pure metadata-derived logic: sorts split assignments by partition, then
    by ``(anonymous_patient_id, anonymous_case_id)`` ascending, and takes the
    first train entry and the first validation entry. Never inspects image or
    label file contents, lesion size, image shape, or spacing, and never
    depends on the input ordering of ``split.assignments``.

    Internal-test assignments are filtered out before any case lookup; the
    manifest lookup this function performs is restricted to exactly the two
    selected case IDs, so an internal-test case's relative paths are never
    resolved here.
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
        raise Phase8TinyRealVerificationCaseSelectionError(
            "no train-partition assignments are available for selection."
        )
    if not validation_candidates:
        raise Phase8TinyRealVerificationCaseSelectionError(
            "no validation-partition assignments are available for selection."
        )

    first_train = train_candidates[0]
    first_validation = validation_candidates[0]
    selected_case_ids = {first_train.anonymous_case_id, first_validation.anonymous_case_id}
    case_by_id = {
        record.anonymous_case_id: record
        for record in manifest.cases
        if record.anonymous_case_id in selected_case_ids
    }

    train_record = case_by_id.get(first_train.anonymous_case_id)
    if train_record is None:
        raise Phase8TinyRealVerificationCaseSelectionError(
            "selected train case is not present in the dataset manifest."
        )
    validation_record = case_by_id.get(first_validation.anonymous_case_id)
    if validation_record is None:
        raise Phase8TinyRealVerificationCaseSelectionError(
            "selected validation case is not present in the dataset manifest."
        )

    return (
        Phase8TinyRealSelectedCase(
            anonymous_patient_id=first_train.anonymous_patient_id,
            anonymous_case_id=first_train.anonymous_case_id,
            partition="train",
            relative_image_path=train_record.relative_image_path,
            relative_label_path=train_record.relative_label_path,
        ),
        Phase8TinyRealSelectedCase(
            anonymous_patient_id=first_validation.anonymous_patient_id,
            anonymous_case_id=first_validation.anonymous_case_id,
            partition="validation",
            relative_image_path=validation_record.relative_image_path,
            relative_label_path=validation_record.relative_label_path,
        ),
    )


# ---------------------------------------------------------------------------
# C. Output-root path safety (external, must-not-already-exist contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8TinyRealVerificationPaths:
    """Validated external locations for one tiny real-development run."""

    output_root: Path
    checkpoints_dir: Path


def validate_phase8_tiny_real_output_root(
    output_root: Path,
    *,
    repository_root: Path,
) -> Phase8TinyRealVerificationPaths:
    """Validate an external, nonexistent, non-symlinked output root.

    Mirrors ``protoem_ct.baselines.paths.validate_baseline_run_root``: the
    output root must be an absolute path, must not be a symlink (even a
    dangling one), must not already exist in any form, and must resolve
    outside the repository root. This is stricter than merely requiring an
    empty directory.
    """

    root_path = Path(output_root)
    if str(root_path) == "" or not root_path.is_absolute():
        raise Phase8TinyRealVerificationOutputRootError(
            "output_root must be an explicit absolute path."
        )
    if root_path.is_symlink():
        raise Phase8TinyRealVerificationOutputRootError("output_root must not be a symlink.")
    if root_path.exists():
        raise Phase8TinyRealVerificationOutputRootError("output_root must not already exist.")

    canonical_repository_root = validate_explicit_dataset_root(repository_root)
    try:
        canonical_output_root = validate_explicit_external_output_root(
            root_path, forbidden_roots=(canonical_repository_root,)
        )
    except InvalidDatasetRootError as exc:
        raise Phase8TinyRealVerificationOutputRootError(
            "invalid tiny real-development verification output root."
        ) from exc

    return Phase8TinyRealVerificationPaths(
        output_root=canonical_output_root,
        checkpoints_dir=canonical_output_root / PHASE8_TINY_REAL_CHECKPOINT_SUBDIRECTORY_NAME,
    )


# ---------------------------------------------------------------------------
# D. Dataset access ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8TinyRealAccessRecord:
    """Per-case access record; anonymous IDs only, no filesystem identity."""

    anonymous_case_id: str
    partition: str
    access_role: str
    image_opened: bool
    label_opened: bool

    def __post_init__(self) -> None:
        if self.partition not in {"train", "validation"}:
            raise Phase8TinyRealVerificationError(
                "access record partition must be 'train' or 'validation'."
            )
        if self.access_role not in {"train_verification", "validation_verification"}:
            raise Phase8TinyRealVerificationError(
                "access_role must be 'train_verification' or 'validation_verification'."
            )


def _access_record_to_dict(record: Phase8TinyRealAccessRecord) -> dict[str, JsonValue]:
    return {
        "access_role": record.access_role,
        "anonymous_case_id": record.anonymous_case_id,
        "image_opened": record.image_opened,
        "label_opened": record.label_opened,
        "partition": record.partition,
    }


@dataclass(frozen=True, slots=True)
class Phase8TinyRealAccessLedger:
    """Verification-only dataset access ledger; anonymous IDs only."""

    schema_name: str
    schema_version: str
    records: tuple[Phase8TinyRealAccessRecord, ...]
    internal_test_opened: bool
    external_data_opened: bool
    file_access_count: int
    ledger_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_TINY_REAL_ACCESS_LEDGER_SCHEMA_NAME)
        _require_schema_version(self.schema_version)
        object.__setattr__(self, "records", tuple(self.records))
        if self.internal_test_opened is not False:
            raise Phase8TinyRealVerificationError("internal_test_opened must always be False.")
        if self.external_data_opened is not False:
            raise Phase8TinyRealVerificationError("external_data_opened must always be False.")
        expected_count = sum(
            int(record.image_opened) + int(record.label_opened) for record in self.records
        )
        if self.file_access_count != expected_count:
            raise Phase8TinyRealVerificationError(
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
                file_access_count=self.file_access_count,
            ),
            field_name="ledger_hash",
        )


def _access_ledger_payload(
    *,
    schema_name: str,
    schema_version: str,
    records: Sequence[Phase8TinyRealAccessRecord],
    internal_test_opened: bool,
    external_data_opened: bool,
    file_access_count: int,
) -> dict[str, JsonValue]:
    return {
        "external_data_opened": external_data_opened,
        "file_access_count": file_access_count,
        "internal_test_opened": internal_test_opened,
        "records": [_access_record_to_dict(record) for record in records],
        "schema_name": schema_name,
        "schema_version": schema_version,
    }


def phase8_tiny_real_access_ledger_to_dict(
    ledger: Phase8TinyRealAccessLedger,
) -> dict[str, JsonValue]:
    """Convert the access ledger to a canonical mapping."""

    payload = _access_ledger_payload(
        schema_name=ledger.schema_name,
        schema_version=ledger.schema_version,
        records=ledger.records,
        internal_test_opened=ledger.internal_test_opened,
        external_data_opened=ledger.external_data_opened,
        file_access_count=ledger.file_access_count,
    )
    payload["ledger_hash"] = ledger.ledger_hash
    return payload


def build_phase8_tiny_real_access_ledger(
    *,
    train_case: Phase8TinyRealSelectedCase,
    validation_case: Phase8TinyRealSelectedCase,
) -> Phase8TinyRealAccessLedger:
    """Build the verification-only access ledger for the two accessed cases."""

    records = (
        Phase8TinyRealAccessRecord(
            anonymous_case_id=train_case.anonymous_case_id,
            partition="train",
            access_role="train_verification",
            image_opened=True,
            label_opened=True,
        ),
        Phase8TinyRealAccessRecord(
            anonymous_case_id=validation_case.anonymous_case_id,
            partition="validation",
            access_role="validation_verification",
            image_opened=True,
            label_opened=True,
        ),
    )
    file_access_count = sum(
        int(record.image_opened) + int(record.label_opened) for record in records
    )
    payload = _access_ledger_payload(
        schema_name=PHASE8_TINY_REAL_ACCESS_LEDGER_SCHEMA_NAME,
        schema_version=PHASE8_TINY_REAL_SCHEMA_VERSION,
        records=records,
        internal_test_opened=False,
        external_data_opened=False,
        file_access_count=file_access_count,
    )
    return Phase8TinyRealAccessLedger(
        schema_name=PHASE8_TINY_REAL_ACCESS_LEDGER_SCHEMA_NAME,
        schema_version=PHASE8_TINY_REAL_SCHEMA_VERSION,
        records=records,
        internal_test_opened=False,
        external_data_opened=False,
        file_access_count=file_access_count,
        ledger_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# E. Checkpoint safety metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8TinyRealCheckpointMetadata:
    """Verification-only checkpoint metadata; never freeze/selection eligible."""

    schema_name: str
    schema_version: str
    checkpoint_metadata: Phase8CheckpointMetadata
    tiny_verification_only: bool
    selection_eligible: bool
    definitive_training: bool
    wrapper_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_TINY_REAL_CHECKPOINT_METADATA_SCHEMA_NAME)
        _require_schema_version(self.schema_version)
        if self.tiny_verification_only is not True:
            raise Phase8TinyRealVerificationError("tiny_verification_only must be True.")
        if self.selection_eligible is not False:
            raise Phase8TinyRealVerificationError("selection_eligible must be False.")
        if self.definitive_training is not False:
            raise Phase8TinyRealVerificationError("definitive_training must be False.")
        if self.checkpoint_metadata.freeze_eligible is not False:
            raise Phase8TinyRealVerificationError(
                "wrapped checkpoint_metadata.freeze_eligible must be False."
            )
        if self.checkpoint_metadata.completion_status != "synthetic_smoke":
            raise Phase8TinyRealVerificationError(
                "wrapped checkpoint_metadata.completion_status must be 'synthetic_smoke'."
            )
        _require_self_hash(
            self.wrapper_hash,
            _checkpoint_metadata_wrapper_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                checkpoint_metadata=self.checkpoint_metadata,
                tiny_verification_only=self.tiny_verification_only,
                selection_eligible=self.selection_eligible,
                definitive_training=self.definitive_training,
            ),
            field_name="wrapper_hash",
        )


def _checkpoint_metadata_wrapper_payload(
    *,
    schema_name: str,
    schema_version: str,
    checkpoint_metadata: Phase8CheckpointMetadata,
    tiny_verification_only: bool,
    selection_eligible: bool,
    definitive_training: bool,
) -> dict[str, JsonValue]:
    return {
        "checkpoint_metadata": phase8_checkpoint_metadata_to_dict(checkpoint_metadata),
        "definitive_training": definitive_training,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "selection_eligible": selection_eligible,
        "tiny_verification_only": tiny_verification_only,
    }


def phase8_tiny_real_checkpoint_metadata_to_dict(
    metadata: Phase8TinyRealCheckpointMetadata,
) -> dict[str, JsonValue]:
    """Convert the checkpoint-safety wrapper to a canonical mapping."""

    payload = _checkpoint_metadata_wrapper_payload(
        schema_name=metadata.schema_name,
        schema_version=metadata.schema_version,
        checkpoint_metadata=metadata.checkpoint_metadata,
        tiny_verification_only=metadata.tiny_verification_only,
        selection_eligible=metadata.selection_eligible,
        definitive_training=metadata.definitive_training,
    )
    payload["wrapper_hash"] = metadata.wrapper_hash
    return payload


def build_phase8_tiny_real_checkpoint_metadata(
    *,
    config: Phase8TinyRealVerificationConfig,
    development_manifest_hash: str,
    development_split_hash: str,
    checkpoint_sha256: str,
    checkpoint_byte_size: int,
    executed_step_count: int,
) -> Phase8TinyRealCheckpointMetadata:
    """Build verification-only checkpoint metadata that is never freeze eligible.

    ``tiny_verification_only``, ``selection_eligible``, ``definitive_training``,
    and the wrapped contract's own ``freeze_eligible`` are all hard-coded here;
    no parameter of this function can move any of them to a more permissive
    value.
    """

    package_environment_reference = ArtifactReference(
        schema_name=PHASE8_TINY_REAL_CONFIG_SCHEMA_NAME,
        schema_version=PHASE8_TINY_REAL_SCHEMA_VERSION,
        artifact_hash=config.config_hash,
        artifact_role="tiny_verification_config",
    )
    # Phase8CheckpointMetadata validates its own self-hash at construction
    # time, so the hash must be computed from the identity payload (mirroring
    # protoem_ct.external.internal_evidence.hash_phase8_checkpoint_metadata)
    # before the single, correctly hashed instance is constructed.
    checkpoint_metadata_identity_payload: dict[str, JsonValue] = {
        "candidate_id": "phase8_tiny_real_verification",
        "checkpoint_byte_size": checkpoint_byte_size,
        "checkpoint_sha256": checkpoint_sha256,
        "completion_status": "synthetic_smoke",
        "development_manifest_hash": development_manifest_hash,
        "development_split_hash": development_split_hash,
        "failure_codes": [],
        "freeze_eligible": False,
        "model_family": "monai_segresnet",
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
        model_family="monai_segresnet",
        candidate_id="phase8_tiny_real_verification",
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
        schema_name=PHASE8_TINY_REAL_CHECKPOINT_METADATA_SCHEMA_NAME,
        schema_version=PHASE8_TINY_REAL_SCHEMA_VERSION,
        checkpoint_metadata=checkpoint_metadata,
        tiny_verification_only=True,
        selection_eligible=False,
        definitive_training=False,
    )
    return Phase8TinyRealCheckpointMetadata(
        schema_name=PHASE8_TINY_REAL_CHECKPOINT_METADATA_SCHEMA_NAME,
        schema_version=PHASE8_TINY_REAL_SCHEMA_VERSION,
        checkpoint_metadata=checkpoint_metadata,
        tiny_verification_only=True,
        selection_eligible=False,
        definitive_training=False,
        wrapper_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# G. Verification summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8TinyRealVerificationSummary:
    """Non-scientific, verification-only engineering diagnostics summary."""

    schema_name: str
    schema_version: str
    train_anonymous_case_id: str
    validation_anonymous_case_id: str
    executed_step_count: int
    step_finite_status: tuple[bool, ...]
    train_forward_shape: tuple[int, ...]
    validation_forward_shape: tuple[int, ...]
    checkpoint_round_trip_verified: bool
    checkpoint_sha256: str
    checkpoint_byte_size: int
    elapsed_seconds: float
    completion_status: str
    blocker_reason_codes: tuple[str, ...]
    verification_only: bool
    non_scientific_disclaimer: str
    config_hash: str
    access_ledger_hash: str
    checkpoint_metadata_hash: str
    summary_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_TINY_REAL_VERIFICATION_SUMMARY_SCHEMA_NAME)
        _require_schema_version(self.schema_version)
        if self.verification_only is not True:
            raise Phase8TinyRealVerificationError("verification_only must be True.")
        if not self.non_scientific_disclaimer:
            raise Phase8TinyRealVerificationError("non_scientific_disclaimer must be nonempty.")
        if self.completion_status not in {"completed", "blocked"}:
            raise Phase8TinyRealVerificationError(
                "completion_status must be 'completed' or 'blocked'."
            )
        if self.completion_status == "blocked" and not self.blocker_reason_codes:
            raise Phase8TinyRealVerificationError(
                "blocked completion_status requires blocker_reason_codes."
            )
        if self.completion_status == "completed" and self.blocker_reason_codes:
            raise Phase8TinyRealVerificationError(
                "completed completion_status must not include blocker_reason_codes."
            )
        object.__setattr__(self, "step_finite_status", tuple(self.step_finite_status))
        object.__setattr__(self, "train_forward_shape", tuple(self.train_forward_shape))
        object.__setattr__(self, "validation_forward_shape", tuple(self.validation_forward_shape))
        object.__setattr__(self, "blocker_reason_codes", tuple(sorted(self.blocker_reason_codes)))
        _require_self_hash(
            self.summary_hash,
            _summary_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                train_anonymous_case_id=self.train_anonymous_case_id,
                validation_anonymous_case_id=self.validation_anonymous_case_id,
                executed_step_count=self.executed_step_count,
                step_finite_status=self.step_finite_status,
                train_forward_shape=self.train_forward_shape,
                validation_forward_shape=self.validation_forward_shape,
                checkpoint_round_trip_verified=self.checkpoint_round_trip_verified,
                checkpoint_sha256=self.checkpoint_sha256,
                checkpoint_byte_size=self.checkpoint_byte_size,
                elapsed_seconds=self.elapsed_seconds,
                completion_status=self.completion_status,
                blocker_reason_codes=self.blocker_reason_codes,
                verification_only=self.verification_only,
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
    train_anonymous_case_id: str,
    validation_anonymous_case_id: str,
    executed_step_count: int,
    step_finite_status: Sequence[bool],
    train_forward_shape: Sequence[int],
    validation_forward_shape: Sequence[int],
    checkpoint_round_trip_verified: bool,
    checkpoint_sha256: str,
    checkpoint_byte_size: int,
    elapsed_seconds: float,
    completion_status: str,
    blocker_reason_codes: Sequence[str],
    verification_only: bool,
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
        "elapsed_seconds": elapsed_seconds,
        "executed_step_count": executed_step_count,
        "non_scientific_disclaimer": non_scientific_disclaimer,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "step_finite_status": list(step_finite_status),
        "train_anonymous_case_id": train_anonymous_case_id,
        "train_forward_shape": list(train_forward_shape),
        "validation_anonymous_case_id": validation_anonymous_case_id,
        "validation_forward_shape": list(validation_forward_shape),
        "verification_only": verification_only,
    }


def phase8_tiny_real_verification_summary_to_dict(
    summary: Phase8TinyRealVerificationSummary,
) -> dict[str, JsonValue]:
    """Convert the verification summary to a canonical mapping."""

    payload = _summary_payload(
        schema_name=summary.schema_name,
        schema_version=summary.schema_version,
        train_anonymous_case_id=summary.train_anonymous_case_id,
        validation_anonymous_case_id=summary.validation_anonymous_case_id,
        executed_step_count=summary.executed_step_count,
        step_finite_status=summary.step_finite_status,
        train_forward_shape=summary.train_forward_shape,
        validation_forward_shape=summary.validation_forward_shape,
        checkpoint_round_trip_verified=summary.checkpoint_round_trip_verified,
        checkpoint_sha256=summary.checkpoint_sha256,
        checkpoint_byte_size=summary.checkpoint_byte_size,
        elapsed_seconds=summary.elapsed_seconds,
        completion_status=summary.completion_status,
        blocker_reason_codes=summary.blocker_reason_codes,
        verification_only=summary.verification_only,
        non_scientific_disclaimer=summary.non_scientific_disclaimer,
        config_hash=summary.config_hash,
        access_ledger_hash=summary.access_ledger_hash,
        checkpoint_metadata_hash=summary.checkpoint_metadata_hash,
    )
    payload["summary_hash"] = summary.summary_hash
    return payload


# ---------------------------------------------------------------------------
# Bounded model/training/checkpoint pipeline (Section D operations)
# ---------------------------------------------------------------------------


def _import_torch() -> Any:
    import torch  # type: ignore[import-not-found]

    return torch


def _import_monai() -> Any:
    import monai  # type: ignore[import-not-found]

    return monai


def _configure_tiny_determinism(*, torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _build_tiny_segresnet_model(*, torch: Any, monai: Any) -> Any:
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

    Mirrors :func:`protoem_ct.data.validation.validate_nifti_pair`'s shape,
    affine-tolerance, and finite-value checks, but validates label values
    against the raw MSD/LiTS three-class domain ``RAW_LABEL_ALLOWED_VALUES``
    (``{0=background, 1=liver, 2=tumor}``) instead of a strict binary
    domain. This module derives the binary tumor-vs-background target from
    the raw label immediately after this check (see ``_load_bounded_patch``),
    matching PROJECT_SPEC.md's binary tumor-segmentation task definition;
    ``validate_nifti_pair`` is not reused here because it hard-codes a strict
    ``{0, 1}`` label domain intended for already-binarized masks, which the
    real LiTS/MSD raw labels are not.
    """

    if not image_path.exists() or not image_path.is_file():
        raise Phase8TinyRealVerificationDataError(f"image path is not a regular file: {image_path}")
    if not label_path.exists() or not label_path.is_file():
        raise Phase8TinyRealVerificationDataError(f"label path is not a regular file: {label_path}")

    try:
        image = cast(Any, nib.load(str(image_path)))
        label = cast(Any, nib.load(str(label_path)))
        image_array = np.asanyarray(image.dataobj)
        label_array = np.asanyarray(label.dataobj)
        image_affine = np.asarray(image.affine, dtype=float)
        label_affine = np.asarray(label.affine, dtype=float)
    except (OSError, ValueError) as exc:
        raise Phase8TinyRealVerificationDataError(f"failed to read NIfTI pair: {exc}") from exc

    if image_array.shape != label_array.shape:
        raise Phase8TinyRealVerificationDataError(
            f"image and label shapes differ: {image_array.shape} vs {label_array.shape}"
        )
    if not np.allclose(
        image_affine, label_affine, atol=CASE_PAIR_AFFINE_TOLERANCE_MM, rtol=0.0, equal_nan=False
    ):
        raise Phase8TinyRealVerificationDataError(
            f"image and label affines differ beyond tolerance {CASE_PAIR_AFFINE_TOLERANCE_MM}"
        )
    if not bool(np.isfinite(image_array).all()):
        raise Phase8TinyRealVerificationDataError("image data contains NaN or infinite values.")
    if not bool(np.isfinite(label_array).all()):
        raise Phase8TinyRealVerificationDataError("label data contains NaN or infinite values.")

    unique_values = np.unique(label_array)
    invalid_values = unique_values[~np.isin(unique_values, RAW_LABEL_ALLOWED_VALUES)]
    if invalid_values.size:
        raise Phase8TinyRealVerificationLabelDomainError(
            f"label data must be drawn from the raw LiTS/MSD domain "
            f"{RAW_LABEL_ALLOWED_VALUES}, got {invalid_values}"
        )


def _load_bounded_patch(
    *,
    image_path: Path,
    label_path: Path,
    config: Phase8TinyRealVerificationConfig,
    torch: Any,
) -> tuple[Any, Any]:
    """Load a deterministic fixed-corner bounded patch for one case.

    Applies the fixed (non-data-fit) HU clip/normalize window and a
    deterministic fixed-corner crop to ``config.patch_size``. Never randomly
    samples the crop location. Derives the binary tumor-vs-background target
    from the raw label by the fixed, non-data-dependent rule
    ``foreground = (raw_label == RAW_LABEL_TUMOR_VALUE)``; background (0) and
    liver (1) both map to the background class, matching the project's
    binary tumor-segmentation task definition.
    """

    image = cast(Any, nib.load(str(image_path)))
    label = cast(Any, nib.load(str(label_path)))
    image_array = np.asanyarray(image.dataobj).astype(np.float32, copy=True)
    label_array = np.asanyarray(label.dataobj)

    patch = config.patch_size
    if len(image_array.shape) != 3:
        raise Phase8TinyRealVerificationDataError("volume must be 3-dimensional.")
    if any(dim_size < patch[axis] for axis, dim_size in enumerate(image_array.shape)):
        raise Phase8TinyRealVerificationDataError(
            "volume is smaller than the bounded verification patch."
        )
    crop = tuple(slice(0, patch[axis]) for axis in range(3))
    image_patch = np.ascontiguousarray(image_array[crop])
    label_patch = np.ascontiguousarray(label_array[crop])
    label_patch = (label_patch == RAW_LABEL_TUMOR_VALUE).astype(np.int64)

    image_patch = np.clip(image_patch, config.hu_window_min, config.hu_window_max)
    image_patch = (
        (image_patch - config.hu_window_min) / (config.hu_window_max - config.hu_window_min)
    ) * 2.0 - 1.0

    image_tensor = torch.from_numpy(image_patch[None, None, ...]).to(dtype=torch.float32)
    label_tensor = torch.from_numpy(label_patch[None, ...]).to(dtype=torch.long)
    return image_tensor, label_tensor


def _run_bounded_training_steps(
    *,
    model: Any,
    optimizer: Any,
    loss_function: Any,
    image_tensor: Any,
    label_tensor: Any,
    torch: Any,
    max_steps: int,
) -> tuple[tuple[bool, ...], tuple[int, ...]]:
    finite_status: list[bool] = []
    train_forward_shape: tuple[int, ...] = ()
    for step_index in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        if step_index == 0:
            train_forward_shape = tuple(int(v) for v in logits.shape)
        loss = loss_function(logits, label_tensor)
        if not bool(torch.isfinite(loss).item()):
            raise Phase8TinyRealVerificationRuntimeError("nonfinite_loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                raise Phase8TinyRealVerificationRuntimeError("nonfinite_gradient")
        optimizer.step()
        finite_status.append(True)
    return tuple(finite_status), train_forward_shape


def _save_tiny_checkpoint(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    seed: int,
    step_count: int,
    checkpoint_path: Path,
) -> None:
    if checkpoint_path.exists():
        raise Phase8TinyRealVerificationRuntimeError("checkpoint_collision")
    torch.save(
        {
            "schema_name": "phase8_tiny_real_verification_checkpoint",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "step_count": step_count,
            "tiny_verification_only": True,
        },
        checkpoint_path,
    )


def _reload_and_verify_checkpoint(
    *,
    torch: Any,
    monai: Any,
    checkpoint_path: Path,
) -> bool:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        reloaded_model = _build_tiny_segresnet_model(torch=torch, monai=monai)
        reloaded_model.load_state_dict(payload["model_state_dict"])
    except (MemoryError, RuntimeError, KeyError, OSError) as exc:
        raise Phase8TinyRealVerificationRuntimeError("checkpoint_reload_failed") from exc
    return True


def _run_bounded_validation_forward(
    *,
    torch: Any,
    model: Any,
    image_tensor: Any,
    label_tensor: Any,
) -> tuple[int, ...]:
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
    if not bool(torch.isfinite(logits).all().item()):
        raise Phase8TinyRealVerificationRuntimeError("nonfinite_validation_logits")
    probabilities = torch.softmax(logits, dim=1)
    if not bool(torch.isfinite(probabilities).all().item()):
        raise Phase8TinyRealVerificationRuntimeError("nonfinite_validation_probabilities")
    unique_label_values = torch.unique(label_tensor)
    if not bool(((unique_label_values == 0) | (unique_label_values == 1)).all().item()):
        raise Phase8TinyRealVerificationRuntimeError("invalid_validation_label_values")
    return tuple(int(v) for v in logits.shape)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8TinyRealVerificationResult:
    """Result of one bounded Phase 8 tiny real-development verification run."""

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
            temporary_exists_message=(
                "temporary Phase 8 tiny real-development verification output already exists"
            ),
            final_exists_message=(
                "Phase 8 tiny real-development verification output already exists"
            ),
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8TinyRealVerificationPublicationError(
            "failed to publish tiny real-development verification artifact."
        ) from exc


def run_phase8_tiny_real_development_verification(
    *,
    manifest_path: Path,
    split_path: Path,
    expected_manifest_sha256: str,
    expected_split_sha256: str,
    dataset_root: Path,
    output_root: Path,
    repository_root: Path,
    approved_development_artifact_set_identity: str,
    git_commit: str,
    package_versions: Mapping[str, str],
    max_steps: int,
    seed: int,
    patch_size: tuple[int, int, int] = DEFAULT_PATCH_SIZE,
) -> Phase8TinyRealVerificationResult:
    """Run the bounded, verification-only Phase 8 tiny real-development check.

    Performs every Section D operation, in order, and only those operations:
    load one train pair, validate geometry/finite/binary, deterministic
    bounded preprocessing, build the approved MONAI SegResNet model, run at
    most two optimizer steps with fail-closed finite checks, persist and
    reload one verification-only checkpoint, load and forward one bounded
    validation patch, then stop. No Dice/IoU/HD95 or other scientific metric
    is computed. All pre-flight safety checks (compute-limit config,
    output-root path safety, dataset-root path safety) run before any NIfTI
    file is opened; if any pre-flight check fails, no file is opened and no
    artifact is written.
    """

    start_time = time.monotonic()

    # Pre-flight 1: compute-limit config (no filesystem access).
    config = build_phase8_tiny_real_verification_config(
        max_optimizer_steps=max_steps,
        seed=seed,
        originating_git_commit=git_commit,
        package_versions=package_versions,
        patch_size=patch_size,
    )

    # Pre-flight 2: output-root path safety (must not already exist).
    canonical_repository_root = repository_root.resolve(strict=True)
    paths = validate_phase8_tiny_real_output_root(
        output_root, repository_root=canonical_repository_root
    )

    # Pre-flight 3: dataset-root path safety.
    canonical_dataset_root = validate_explicit_dataset_root(dataset_root)

    # Pre-flight 4: manifest/split hash-verified binding (JSON only, no pixels).
    build_phase8_real_development_input_binding(
        manifest_path=manifest_path,
        split_path=split_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_split_sha256=expected_split_sha256,
        approved_development_artifact_set_identity=approved_development_artifact_set_identity,
    )
    resolved_manifest_path = manifest_path.resolve(strict=True)
    resolved_split_path = split_path.resolve(strict=True)
    manifest = phase2_artifact_from_json(
        resolved_manifest_path.read_text(encoding="utf-8"), DatasetManifest
    )
    split = phase2_artifact_from_json(
        resolved_split_path.read_text(encoding="utf-8"), DevelopmentSplitManifest
    )

    # Pre-flight 5: deterministic metadata-only case selection.
    train_case, validation_case = select_phase8_tiny_real_development_cases(
        manifest=manifest, split=split
    )

    train_image_path = resolve_regular_file_beneath_root(
        canonical_dataset_root, train_case.relative_image_path
    )
    train_label_path = resolve_regular_file_beneath_root(
        canonical_dataset_root, train_case.relative_label_path
    )
    validation_image_path = resolve_regular_file_beneath_root(
        canonical_dataset_root, validation_case.relative_image_path
    )
    validation_label_path = resolve_regular_file_beneath_root(
        canonical_dataset_root, validation_case.relative_label_path
    )

    # Section D, steps 1-2: load + validate the train pair before any model code.
    _validate_raw_case_pair_geometry(train_image_path, train_label_path)

    torch = _import_torch()
    monai = _import_monai()
    configure_phase3_cpu_torch_runtime(torch=torch)
    _configure_tiny_determinism(torch=torch, seed=config.seed)

    # Section D, step 3-6: bounded preprocessing, model, <=2 optimizer steps.
    train_image_tensor, train_label_tensor = _load_bounded_patch(
        image_path=train_image_path,
        label_path=train_label_path,
        config=config,
        torch=torch,
    )
    model = _build_tiny_segresnet_model(torch=torch, monai=monai)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.CrossEntropyLoss()
    finite_status, train_forward_shape = _run_bounded_training_steps(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        image_tensor=train_image_tensor,
        label_tensor=train_label_tensor,
        torch=torch,
        max_steps=config.max_optimizer_steps,
    )

    # Section D, step 7: persist one verification-only checkpoint.
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = paths.checkpoints_dir / PHASE8_TINY_REAL_CHECKPOINT_FILENAME
    _save_tiny_checkpoint(
        torch=torch,
        model=model,
        optimizer=optimizer,
        seed=config.seed,
        step_count=len(finite_status),
        checkpoint_path=checkpoint_path,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint_byte_size = checkpoint_path.stat().st_size

    # Section D, step 8: reload and verify state-dict compatibility.
    checkpoint_round_trip_verified = _reload_and_verify_checkpoint(
        torch=torch, monai=monai, checkpoint_path=checkpoint_path
    )

    # Section D, steps 9-11: one bounded validation patch forward pass.
    _validate_raw_case_pair_geometry(validation_image_path, validation_label_path)
    validation_image_tensor, validation_label_tensor = _load_bounded_patch(
        image_path=validation_image_path,
        label_path=validation_label_path,
        config=config,
        torch=torch,
    )
    validation_forward_shape = _run_bounded_validation_forward(
        torch=torch,
        model=model,
        image_tensor=validation_image_tensor,
        label_tensor=validation_label_tensor,
    )

    elapsed_seconds = time.monotonic() - start_time

    ledger = build_phase8_tiny_real_access_ledger(
        train_case=train_case, validation_case=validation_case
    )
    checkpoint_metadata = build_phase8_tiny_real_checkpoint_metadata(
        config=config,
        development_manifest_hash=manifest.manifest_hash,
        development_split_hash=split.split_hash,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_size=checkpoint_byte_size,
        executed_step_count=len(finite_status),
    )
    summary_payload = _summary_payload(
        schema_name=PHASE8_TINY_REAL_VERIFICATION_SUMMARY_SCHEMA_NAME,
        schema_version=PHASE8_TINY_REAL_SCHEMA_VERSION,
        train_anonymous_case_id=train_case.anonymous_case_id,
        validation_anonymous_case_id=validation_case.anonymous_case_id,
        executed_step_count=len(finite_status),
        step_finite_status=finite_status,
        train_forward_shape=train_forward_shape,
        validation_forward_shape=validation_forward_shape,
        checkpoint_round_trip_verified=checkpoint_round_trip_verified,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_size=checkpoint_byte_size,
        elapsed_seconds=elapsed_seconds,
        completion_status="completed",
        blocker_reason_codes=(),
        verification_only=True,
        non_scientific_disclaimer=NON_SCIENTIFIC_DISCLAIMER,
        config_hash=config.config_hash,
        access_ledger_hash=ledger.ledger_hash,
        checkpoint_metadata_hash=checkpoint_metadata.wrapper_hash,
    )
    summary = Phase8TinyRealVerificationSummary(
        schema_name=PHASE8_TINY_REAL_VERIFICATION_SUMMARY_SCHEMA_NAME,
        schema_version=PHASE8_TINY_REAL_SCHEMA_VERSION,
        train_anonymous_case_id=train_case.anonymous_case_id,
        validation_anonymous_case_id=validation_case.anonymous_case_id,
        executed_step_count=len(finite_status),
        step_finite_status=finite_status,
        train_forward_shape=train_forward_shape,
        validation_forward_shape=validation_forward_shape,
        checkpoint_round_trip_verified=checkpoint_round_trip_verified,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_size=checkpoint_byte_size,
        elapsed_seconds=elapsed_seconds,
        completion_status="completed",
        blocker_reason_codes=(),
        verification_only=True,
        non_scientific_disclaimer=NON_SCIENTIFIC_DISCLAIMER,
        config_hash=config.config_hash,
        access_ledger_hash=ledger.ledger_hash,
        checkpoint_metadata_hash=checkpoint_metadata.wrapper_hash,
        summary_hash=sha256_json(summary_payload),
    )

    _publish_json(
        paths.output_root / PHASE8_TINY_REAL_CONFIG_FILENAME,
        canonical_json_bytes(phase8_tiny_real_verification_config_to_dict(config)) + b"\n",
    )
    _publish_json(
        paths.output_root / PHASE8_TINY_REAL_ACCESS_LEDGER_FILENAME,
        canonical_json_bytes(phase8_tiny_real_access_ledger_to_dict(ledger)) + b"\n",
    )
    _publish_json(
        paths.output_root / PHASE8_TINY_REAL_CHECKPOINT_METADATA_FILENAME,
        canonical_json_bytes(phase8_tiny_real_checkpoint_metadata_to_dict(checkpoint_metadata))
        + b"\n",
    )
    _publish_json(
        paths.output_root / PHASE8_TINY_REAL_VERIFICATION_SUMMARY_FILENAME,
        canonical_json_bytes(phase8_tiny_real_verification_summary_to_dict(summary)) + b"\n",
    )

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(paths.output_root.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    return Phase8TinyRealVerificationResult(
        output_root=paths.output_root,
        checkpoint_path=checkpoint_path,
        config_hash=config.config_hash,
        access_ledger_hash=ledger.ledger_hash,
        checkpoint_metadata_hash=checkpoint_metadata.wrapper_hash,
        summary_hash=summary.summary_hash,
        artifact_hashes=artifact_hashes,
    )


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _require_schema(value: str, expected: str) -> None:
    if value != expected:
        raise Phase8TinyRealVerificationError(
            f"schema_name must equal {expected!r}, got {value!r}."
        )


def _require_schema_version(value: str) -> None:
    if value != PHASE8_TINY_REAL_SCHEMA_VERSION:
        raise Phase8TinyRealVerificationError(
            f"schema_version must equal {PHASE8_TINY_REAL_SCHEMA_VERSION!r}."
        )


def _require_self_hash(observed: str, payload: dict[str, JsonValue], *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(observed):
        raise Phase8TinyRealVerificationError(f"{field_name} must be a lowercase SHA-256 hex.")
    expected = sha256_json(payload)
    if observed != expected:
        raise Phase8TinyRealVerificationError(f"{field_name} does not match its canonical payload.")
