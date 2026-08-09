"""Phase 8 definitive development training: real-data orchestration driver.

This module wires the tested, implementation-only primitives in
:mod:`protoem_ct.external.definitive_pipeline` (Package A) to real MSD
Task03 Liver / LiTS data on disk. It performs no scientific-design work of
its own: preprocessing, architecture, hyperparameters, patch schedule,
training-execution policy, and checkpoint-selection rule are all read
unchanged from :mod:`protoem_ct.external.definitive_pipeline`. This module
adds exactly what that module deliberately does not: dataset discovery,
manifest/split verification, real NIfTI loading, and process-level
orchestration/watchdog wiring -- mirroring the established pattern in
:mod:`protoem_ct.external.definitive_training_pilot` and
:mod:`protoem_ct.external.real_development_runner`.

Hard, non-negotiable contract for a real run (see
the Phase 8 closure provenance records and ``docs/DECISIONS.md`` entry
``PHASE8-DEFINITIVE-TRAIN-SPACING-V1``):

* The development manifest and development split JSON files must byte-hash
  to the two exact, hard-coded values below (``REQUIRED_MANIFEST_SHA256``,
  ``REQUIRED_SPLIT_SHA256``) -- this module refuses any other manifest/split
  pair before opening a single NIfTI file.
* Exactly 91 TRAIN cases and exactly 20 VALIDATION cases are required from
  the split; internal-test case IDs are read from split metadata only, for
  exclusion-checking, and are never used to resolve a file path or opened.
* Only ``imagesTr``/``labelsTr`` files referenced by the manifest for
  TRAIN/VALIDATION cases may ever be opened.
* Target spacing is the fixed, approved
  ``PHASE8-DEFINITIVE-TRAIN-SPACING-V1`` value; it is passed directly to
  :func:`protoem_ct.external.definitive_pipeline.resample_volume_to_spacing`
  and is never recomputed from data.
* The patch schedule, patch materialization, training, checkpoint
  publication, full validation, and checkpoint selection are all delegated
  unchanged to :mod:`protoem_ct.external.definitive_pipeline`.

``torch``, ``monai``, and ``nibabel`` array access are deferred to function
bodies / the pipeline module so importing this module (and, more
importantly, importing ``protoem_ct.cli.main``) stays cheap.
"""

from __future__ import annotations

import multiprocessing
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, Final, cast

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
from protoem_ct.external import definitive_pipeline as pipeline
from protoem_ct.external.internal_evidence import (
    ArtifactReference,
    phase8_checkpoint_metadata_to_dict,
)

# ---------------------------------------------------------------------------
# Hard-locked real-data contract
# ---------------------------------------------------------------------------

REQUIRED_MANIFEST_SHA256: Final[str] = (
    "0e21a7d555e20b6011091bc18e462bc150cc23a8f46e522dbd8c01b014a44ae3"
)
REQUIRED_SPLIT_SHA256: Final[str] = (
    "416ca83e85c8598fc4f7065316153193211bf6b57875fbda4cafc01c360445f7"
)
REQUIRED_MANIFEST_EMBEDDED_HASH: Final[str] = (
    "c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b"
)
REQUIRED_SPLIT_EMBEDDED_HASH: Final[str] = (
    "936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb"
)
REQUIRED_TRAIN_CASE_COUNT: Final[int] = 91
REQUIRED_VALIDATION_CASE_COUNT: Final[int] = 20

# PHASE8-DEFINITIVE-TRAIN-SPACING-V1 (docs/DECISIONS.md); passed directly to
# resample_volume_to_spacing -- never recomputed from data by this module.
DEFINITIVE_TRAIN_TARGET_SPACING: Final[tuple[float, float, float]] = (
    0.767578125,
    0.767578125,
    1.0,
)

RAW_LABEL_ALLOWED_VALUES: Final[tuple[int, ...]] = (0, 1, 2)
REQUIRED_ORIENTATION_AXCODES: Final[tuple[str, str, str]] = ("R", "A", "S")
CASE_PAIR_AFFINE_TOLERANCE_MM: Final[float] = 1e-4
_ALLOWED_RELATIVE_PATH_PREFIXES: Final[tuple[str, ...]] = ("imagesTr/", "labelsTr/")

_GIT_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{7,64}$")

PHASE8_DEFINITIVE_REAL_TRAINING_ACCESS_LEDGER_SCHEMA_NAME: Final[str] = (
    "phase8_definitive_real_training_access_ledger"
)
PHASE8_DEFINITIVE_REAL_TRAINING_ACCESS_LEDGER_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_DEFINITIVE_REAL_TRAINING_SELECTION_EVIDENCE_FILENAME: Final[str] = (
    "phase8_definitive_checkpoint_selection_evidence.json"
)
PHASE8_DEFINITIVE_REAL_TRAINING_ACCESS_LEDGER_FILENAME: Final[str] = (
    "phase8_definitive_real_training_access_ledger.json"
)
PHASE8_DEFINITIVE_REAL_TRAINING_CHECKPOINT_METADATA_FILENAME_TEMPLATE: Final[str] = (
    "phase8_definitive_checkpoint_metadata_step_{step}.json"
)
_CHECKPOINTS_SUBDIRECTORY_NAME: Final[str] = "checkpoints"
_PATCHES_SUBDIRECTORY_NAME: Final[str] = "patches"

DEFAULT_WALL_CLOCK_LIMIT_SECONDS: Final[float] = 36000.0


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class Phase8DefinitiveRealTrainingError(ValueError):
    """Base error for the Phase 8 definitive real-data training driver."""


class Phase8DefinitiveRealTrainingOutputRootError(Phase8DefinitiveRealTrainingError):
    """Raised when the external output root fails the path-safety contract."""


class Phase8DefinitiveRealTrainingPrerequisitesError(Phase8DefinitiveRealTrainingError):
    """Raised when manifest/split/lesion-component hash verification fails closed."""


class Phase8DefinitiveRealTrainingCaseSelectionError(Phase8DefinitiveRealTrainingError):
    """Raised when deterministic case selection cannot proceed from metadata alone."""


class Phase8DefinitiveRealTrainingDataError(Phase8DefinitiveRealTrainingError):
    """Raised when a selected case's data is missing, unreadable, or unsafe."""


class Phase8DefinitiveRealTrainingLabelDomainError(Phase8DefinitiveRealTrainingDataError):
    """Raised when a raw label's values fall outside the approved {0,1,2} domain."""


class Phase8DefinitiveRealTrainingOrientationError(Phase8DefinitiveRealTrainingDataError):
    """Raised when a raw image/label pair is not in RAS orientation."""


class Phase8DefinitiveRealTrainingHashMismatchError(Phase8DefinitiveRealTrainingDataError):
    """Raised when a raw file's content hash does not match the manifest record."""


class Phase8DefinitiveRealTrainingRuntimeError(Phase8DefinitiveRealTrainingError):
    """Raised when an orchestration step fails closed for a non-data reason."""


class Phase8DefinitiveRealTrainingWatchdogTimeoutError(Phase8DefinitiveRealTrainingRuntimeError):
    """Raised when the process-level watchdog kills a blocked child process.

    No partial success is ever surfaced through this error: the caller must
    make a fresh call to retry, and no checkpoint selection is ever performed
    for the call that timed out.
    """


class Phase8DefinitiveRealTrainingPublicationError(Phase8DefinitiveRealTrainingError):
    """Raised when guarded artifact publication fails safely."""


# ---------------------------------------------------------------------------
# A. Manifest / split / lesion-components verification (no pixel access)
# ---------------------------------------------------------------------------


def _load_and_hash_check_json(path: Path, *, expected_sha256: str, description: str) -> Any:
    resolved_path = path.resolve(strict=True)
    if not resolved_path.is_file():
        raise Phase8DefinitiveRealTrainingPrerequisitesError(
            f"{description} is not a regular file."
        )
    observed_sha256 = sha256_file(resolved_path)
    if observed_sha256 != expected_sha256:
        raise Phase8DefinitiveRealTrainingPrerequisitesError(
            f"{description} sha256 mismatch: expected {expected_sha256}, got {observed_sha256}."
        )
    return resolved_path.read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveRealTrainingMetadata:
    """Verified manifest/split/lesion-components artifacts for one real run."""

    manifest: DatasetManifest
    split: DevelopmentSplitManifest
    lesion_components: LesionComponentsArtifact
    train_case_references: tuple[pipeline.Phase8DefinitiveTrainCaseReference, ...]
    validation_case_references: tuple[pipeline.Phase8DefinitiveValidationCaseReference, ...]
    positive_case_ids: frozenset[str]
    internal_test_case_ids: frozenset[str]


def load_and_verify_definitive_real_training_metadata(
    *,
    manifest_path: Path,
    split_path: Path,
    lesion_components_path: Path,
    expected_lesion_components_sha256: str,
) -> Phase8DefinitiveRealTrainingMetadata:
    """Load and hash-verify manifest/split/lesion-components; select case IDs.

    This function opens no NIfTI file. It fails closed unless the manifest
    and split files byte-hash to exactly the two hard-locked required
    values, the lesion-components artifact's own recorded manifest/split
    hashes match, and the split contains exactly 91 TRAIN and exactly 20
    VALIDATION cases.
    """

    manifest_text = _load_and_hash_check_json(
        manifest_path, expected_sha256=REQUIRED_MANIFEST_SHA256, description="dataset manifest"
    )
    split_text = _load_and_hash_check_json(
        split_path, expected_sha256=REQUIRED_SPLIT_SHA256, description="development split"
    )
    lesion_text = _load_and_hash_check_json(
        lesion_components_path,
        expected_sha256=expected_lesion_components_sha256,
        description="lesion components",
    )

    try:
        manifest = phase2_artifact_from_json(manifest_text, DatasetManifest)
        split = phase2_artifact_from_json(split_text, DevelopmentSplitManifest)
        lesion_components = phase2_artifact_from_json(lesion_text, LesionComponentsArtifact)
    except Exception as exc:  # noqa: BLE001 -- reconstruction failure paths vary by artifact.
        raise Phase8DefinitiveRealTrainingPrerequisitesError(
            f"failed to reconstruct a prerequisite artifact: {exc}"
        ) from exc

    # Byte-level file hash (checked above via _load_and_hash_check_json)
    # pins the exact approved manifest/split file bytes. The embedded
    # ``manifest_hash``/``split_hash`` fields are a *semantic* hash of the
    # artifact's content (computed by hash_dataset_manifest/
    # hash_development_split) and are necessarily different from the raw
    # file byte hash for a self-referential JSON document -- so they are
    # checked separately here against the exact approved semantic-hash
    # values, in addition to being checked for internal self-consistency
    # (the field must actually match a fresh hash of its own content, so a
    # hand-edited ``manifest_hash``/``split_hash`` field is rejected even if
    # every other check were bypassed).
    if manifest.manifest_hash != hash_dataset_manifest(manifest):
        raise Phase8DefinitiveRealTrainingPrerequisitesError(
            "manifest.manifest_hash is not internally self-consistent with its own content."
        )
    if manifest.manifest_hash != REQUIRED_MANIFEST_EMBEDDED_HASH:
        raise Phase8DefinitiveRealTrainingPrerequisitesError(
            f"manifest.manifest_hash does not match the approved embedded hash "
            f"{REQUIRED_MANIFEST_EMBEDDED_HASH}."
        )
    if split.split_hash != hash_development_split(split):
        raise Phase8DefinitiveRealTrainingPrerequisitesError(
            "split.split_hash is not internally self-consistent with its own content."
        )
    if split.split_hash != REQUIRED_SPLIT_EMBEDDED_HASH:
        raise Phase8DefinitiveRealTrainingPrerequisitesError(
            f"split.split_hash does not match the approved embedded hash "
            f"{REQUIRED_SPLIT_EMBEDDED_HASH}."
        )
    if lesion_components.manifest_hash != manifest.manifest_hash:
        raise Phase8DefinitiveRealTrainingPrerequisitesError(
            "lesion components manifest_hash does not match the loaded manifest."
        )
    if lesion_components.split_hash != split.split_hash:
        raise Phase8DefinitiveRealTrainingPrerequisitesError(
            "lesion components split_hash does not match the loaded split."
        )

    train_assignments = sorted(
        (a for a in split.assignments if a.partition == "train"),
        key=lambda item: (item.anonymous_patient_id, item.anonymous_case_id),
    )
    validation_assignments = sorted(
        (a for a in split.assignments if a.partition == "validation"),
        key=lambda item: (item.anonymous_patient_id, item.anonymous_case_id),
    )
    internal_test_case_ids = frozenset(
        a.anonymous_case_id for a in split.assignments if a.partition == "internal_test"
    )

    if len(train_assignments) != REQUIRED_TRAIN_CASE_COUNT:
        raise Phase8DefinitiveRealTrainingCaseSelectionError(
            f"expected exactly {REQUIRED_TRAIN_CASE_COUNT} TRAIN cases, got "
            f"{len(train_assignments)}."
        )
    if len(validation_assignments) != REQUIRED_VALIDATION_CASE_COUNT:
        raise Phase8DefinitiveRealTrainingCaseSelectionError(
            f"expected exactly {REQUIRED_VALIDATION_CASE_COUNT} VALIDATION cases, got "
            f"{len(validation_assignments)}."
        )

    manifest_case_ids = {case.anonymous_case_id for case in manifest.cases}
    for assignment in (*train_assignments, *validation_assignments):
        if assignment.anonymous_case_id not in manifest_case_ids:
            raise Phase8DefinitiveRealTrainingCaseSelectionError(
                f"split case {assignment.anonymous_case_id!r} is not present in the manifest."
            )

    lesion_count_by_case_id: dict[str, int] = {
        record.anonymous_case_id: record.lesion_count
        for record in lesion_components.case_records
        if record.analysis_performed and record.lesion_count is not None
    }
    positive_case_ids = frozenset(
        assignment.anonymous_case_id
        for assignment in train_assignments
        if lesion_count_by_case_id.get(assignment.anonymous_case_id, 0) > 0
    )
    if not positive_case_ids:
        raise Phase8DefinitiveRealTrainingCaseSelectionError(
            "metadata does not identify any tumor-positive TRAIN case; BLOCKED before any "
            "NIfTI access."
        )

    train_case_references = tuple(
        pipeline.Phase8DefinitiveTrainCaseReference(case_id=assignment.anonymous_case_id)
        for assignment in train_assignments
    )
    validation_case_references = tuple(
        pipeline.Phase8DefinitiveValidationCaseReference(case_id=assignment.anonymous_case_id)
        for assignment in validation_assignments
    )

    return Phase8DefinitiveRealTrainingMetadata(
        manifest=manifest,
        split=split,
        lesion_components=lesion_components,
        train_case_references=train_case_references,
        validation_case_references=validation_case_references,
        positive_case_ids=positive_case_ids,
        internal_test_case_ids=internal_test_case_ids,
    )


# ---------------------------------------------------------------------------
# B. Output-root path safety (must not already exist)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveRealTrainingPaths:
    """Validated external locations for one real definitive training run."""

    output_root: Path
    checkpoints_dir: Path
    patches_dir: Path


def validate_definitive_real_training_output_root(
    output_root: Path, *, repository_root: Path
) -> Phase8DefinitiveRealTrainingPaths:
    """Validate an external, nonexistent, non-symlinked output root."""

    root_path = Path(output_root)
    if str(root_path) == "" or not root_path.is_absolute():
        raise Phase8DefinitiveRealTrainingOutputRootError(
            "output_root must be an explicit absolute path."
        )
    if root_path.is_symlink():
        raise Phase8DefinitiveRealTrainingOutputRootError("output_root must not be a symlink.")
    if root_path.exists():
        raise Phase8DefinitiveRealTrainingOutputRootError("output_root must not already exist.")

    canonical_repository_root = validate_explicit_dataset_root(repository_root)
    try:
        canonical_output_root = validate_explicit_external_output_root(
            root_path, forbidden_roots=(canonical_repository_root,)
        )
    except InvalidDatasetRootError as exc:
        raise Phase8DefinitiveRealTrainingOutputRootError(
            "invalid definitive real-training output root."
        ) from exc

    return Phase8DefinitiveRealTrainingPaths(
        output_root=canonical_output_root,
        checkpoints_dir=canonical_output_root / _CHECKPOINTS_SUBDIRECTORY_NAME,
        patches_dir=canonical_output_root / _PATCHES_SUBDIRECTORY_NAME,
    )


# ---------------------------------------------------------------------------
# C. Real geometry / hash validation and preprocessing (first pixel access)
# ---------------------------------------------------------------------------


def _validate_relative_path_prefix(relative_path: str, *, field_name: str) -> None:
    if not relative_path.startswith(_ALLOWED_RELATIVE_PATH_PREFIXES):
        raise Phase8DefinitiveRealTrainingDataError(
            f"{field_name} must be beneath one of {_ALLOWED_RELATIVE_PATH_PREFIXES!r}, got "
            f"{relative_path!r}."
        )


def _validate_raw_case_pair_geometry_and_hash(
    *,
    image_path: Path,
    label_path: Path,
    expected_image_sha256: str,
    expected_label_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate hash/shape/affine/orientation/finite/label-domain for one raw pair.

    Returns ``(image_array, label_array, affine)`` only after every safety
    check has passed. Mirrors
    :func:`protoem_ct.external.tiny_real_verification._validate_raw_case_pair_geometry`
    and
    :func:`protoem_ct.external.definitive_training_pilot._validate_raw_case_pair_geometry`,
    with an added source-file content-hash check against the manifest's
    recorded ``image_sha256``/``label_sha256`` before any array is read.
    """

    if not image_path.exists() or not image_path.is_file():
        raise Phase8DefinitiveRealTrainingDataError(
            f"image path is not a regular file: {image_path}"
        )
    if not label_path.exists() or not label_path.is_file():
        raise Phase8DefinitiveRealTrainingDataError(
            f"label path is not a regular file: {label_path}"
        )

    observed_image_sha256 = sha256_file(image_path)
    if observed_image_sha256 != expected_image_sha256:
        raise Phase8DefinitiveRealTrainingHashMismatchError(
            "image file content sha256 does not match the manifest record."
        )
    observed_label_sha256 = sha256_file(label_path)
    if observed_label_sha256 != expected_label_sha256:
        raise Phase8DefinitiveRealTrainingHashMismatchError(
            "label file content sha256 does not match the manifest record."
        )

    try:
        image = cast(Any, nib.load(str(image_path)))
        label = cast(Any, nib.load(str(label_path)))
        image_array = np.asanyarray(image.dataobj)
        label_array = np.asanyarray(label.dataobj)
        image_affine = np.asarray(image.affine, dtype=float)
        label_affine = np.asarray(label.affine, dtype=float)
    except (OSError, ValueError) as exc:
        raise Phase8DefinitiveRealTrainingDataError(f"failed to read NIfTI pair: {exc}") from exc

    if image_array.shape != label_array.shape:
        raise Phase8DefinitiveRealTrainingDataError(
            f"image and label shapes differ: {image_array.shape} vs {label_array.shape}"
        )
    if image_array.ndim != 3:
        raise Phase8DefinitiveRealTrainingDataError("volume must be 3-dimensional.")
    if not np.allclose(
        image_affine, label_affine, atol=CASE_PAIR_AFFINE_TOLERANCE_MM, rtol=0.0, equal_nan=False
    ):
        raise Phase8DefinitiveRealTrainingDataError(
            f"image and label affines differ beyond tolerance {CASE_PAIR_AFFINE_TOLERANCE_MM}"
        )

    image_axcodes = cast(tuple[str, ...], cast(Any, nib).aff2axcodes(image_affine))
    label_axcodes = cast(tuple[str, ...], cast(Any, nib).aff2axcodes(label_affine))
    if (
        tuple(image_axcodes) != REQUIRED_ORIENTATION_AXCODES
        or tuple(label_axcodes) != REQUIRED_ORIENTATION_AXCODES
    ):
        raise Phase8DefinitiveRealTrainingOrientationError(
            "image/label pair must be RAS-oriented "
            f"({REQUIRED_ORIENTATION_AXCODES}); got image={image_axcodes}, label={label_axcodes}."
        )

    if not bool(np.isfinite(image_array).all()):
        raise Phase8DefinitiveRealTrainingDataError("image data contains NaN or infinite values.")
    if not bool(np.isfinite(label_array).all()):
        raise Phase8DefinitiveRealTrainingDataError("label data contains NaN or infinite values.")

    unique_values = np.unique(label_array)
    invalid_values = unique_values[~np.isin(unique_values, RAW_LABEL_ALLOWED_VALUES)]
    if invalid_values.size:
        raise Phase8DefinitiveRealTrainingLabelDomainError(
            f"label data must be drawn from the raw LiTS/MSD domain "
            f"{RAW_LABEL_ALLOWED_VALUES}, got {invalid_values}"
        )

    return image_array, label_array, image_affine


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    return cast(
        tuple[float, float, float],
        tuple(float(v) for v in np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))),
    )


def _preprocess_real_case(
    *,
    image_array: np.ndarray,
    label_array: np.ndarray,
    affine: np.ndarray,
    config: pipeline.Phase8DefinitiveConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the locked RAS/resample/HU-clip-scale/binarize preprocessing chain."""

    reoriented_image, reoriented_affine = pipeline.reorient_volume_to_ras(image_array, affine)
    reoriented_label, _ = pipeline.reorient_volume_to_ras(label_array, affine)
    source_spacing = _spacing_from_affine(reoriented_affine)

    resampled_image = pipeline.resample_volume_to_spacing(
        reoriented_image.astype(np.float32),
        source_spacing=source_spacing,
        target_spacing=DEFINITIVE_TRAIN_TARGET_SPACING,
        is_label=False,
    )
    resampled_label = pipeline.resample_volume_to_spacing(
        reoriented_label.astype(np.float32),
        source_spacing=source_spacing,
        target_spacing=DEFINITIVE_TRAIN_TARGET_SPACING,
        is_label=True,
    )

    scaled_image = pipeline.clip_and_scale_intensity(resampled_image, config=config)
    binary_label = pipeline.binarize_tumor_label(resampled_label, config=config)
    return scaled_image, binary_label


# ---------------------------------------------------------------------------
# D. Access ledger (anonymous IDs only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveRealTrainingAccessRecord:
    """Per-case access record; anonymous IDs only, no filesystem identity."""

    anonymous_case_id: str
    partition: str
    image_opened: bool
    label_opened: bool


class _AccessLedgerRecorder:
    """Accumulates access records as real cases are opened during the run."""

    def __init__(self) -> None:
        self.records: list[Phase8DefinitiveRealTrainingAccessRecord] = []

    def record(self, *, case_id: str, partition: str) -> None:
        self.records.append(
            Phase8DefinitiveRealTrainingAccessRecord(
                anonymous_case_id=case_id,
                partition=partition,
                image_opened=True,
                label_opened=True,
            )
        )

    def to_dict(self) -> dict[str, JsonValue]:
        entries: list[JsonValue] = [
            {
                "anonymous_case_id": record.anonymous_case_id,
                "image_opened": record.image_opened,
                "label_opened": record.label_opened,
                "partition": record.partition,
            }
            for record in self.records
        ]
        payload: dict[str, JsonValue] = {
            "entries": entries,
            "schema_name": PHASE8_DEFINITIVE_REAL_TRAINING_ACCESS_LEDGER_SCHEMA_NAME,
            "schema_version": PHASE8_DEFINITIVE_REAL_TRAINING_ACCESS_LEDGER_SCHEMA_VERSION,
        }
        payload["ledger_hash"] = sha256_json(payload)
        return payload


# ---------------------------------------------------------------------------
# E. Real case loaders
# ---------------------------------------------------------------------------


def _build_real_train_case_loader(
    *,
    manifest: DatasetManifest,
    dataset_root: Path,
    config: pipeline.Phase8DefinitiveConfig,
    ledger: _AccessLedgerRecorder,
) -> Callable[[str], pipeline.Phase8DefinitiveTrainCase]:
    case_by_id = {case.anonymous_case_id: case for case in manifest.cases}

    def _loader(case_id: str) -> pipeline.Phase8DefinitiveTrainCase:
        record = case_by_id.get(case_id)
        if record is None:
            raise Phase8DefinitiveRealTrainingDataError(
                f"train case {case_id!r} is not present in the dataset manifest."
            )
        _validate_relative_path_prefix(record.relative_image_path, field_name="relative_image_path")
        _validate_relative_path_prefix(record.relative_label_path, field_name="relative_label_path")
        image_path = resolve_regular_file_beneath_root(dataset_root, record.relative_image_path)
        label_path = resolve_regular_file_beneath_root(dataset_root, record.relative_label_path)

        image_array, label_array, affine = _validate_raw_case_pair_geometry_and_hash(
            image_path=image_path,
            label_path=label_path,
            expected_image_sha256=record.image_sha256,
            expected_label_sha256=record.label_sha256,
        )
        ledger.record(case_id=case_id, partition="train")
        scaled_image, binary_label = _preprocess_real_case(
            image_array=image_array, label_array=label_array, affine=affine, config=config
        )
        return pipeline.Phase8DefinitiveTrainCase(
            case_id=case_id, image=scaled_image, label_binary=binary_label
        )

    return _loader


def _build_real_validation_case_loader(
    *,
    manifest: DatasetManifest,
    dataset_root: Path,
    config: pipeline.Phase8DefinitiveConfig,
    ledger: _AccessLedgerRecorder,
) -> Callable[[str], pipeline.Phase8DefinitiveValidationCase]:
    case_by_id = {case.anonymous_case_id: case for case in manifest.cases}

    def _loader(case_id: str) -> pipeline.Phase8DefinitiveValidationCase:
        record = case_by_id.get(case_id)
        if record is None:
            raise Phase8DefinitiveRealTrainingDataError(
                f"validation case {case_id!r} is not present in the dataset manifest."
            )
        _validate_relative_path_prefix(record.relative_image_path, field_name="relative_image_path")
        _validate_relative_path_prefix(record.relative_label_path, field_name="relative_label_path")
        image_path = resolve_regular_file_beneath_root(dataset_root, record.relative_image_path)
        label_path = resolve_regular_file_beneath_root(dataset_root, record.relative_label_path)

        image_array, label_array, affine = _validate_raw_case_pair_geometry_and_hash(
            image_path=image_path,
            label_path=label_path,
            expected_image_sha256=record.image_sha256,
            expected_label_sha256=record.label_sha256,
        )
        ledger.record(case_id=case_id, partition="validation")
        scaled_image, binary_label = _preprocess_real_case(
            image_array=image_array, label_array=label_array, affine=affine, config=config
        )
        return pipeline.Phase8DefinitiveValidationCase(
            case_id=case_id, image=scaled_image, label_binary=binary_label
        )

    return _loader


# ---------------------------------------------------------------------------
# F. Preprocessing-evidence / package-environment reference helpers
# ---------------------------------------------------------------------------


def _build_preprocessing_evidence_hash(config: pipeline.Phase8DefinitiveConfig) -> str:
    payload: dict[str, JsonValue] = {
        "config_hash": config.config_hash,
        "design_identifier": "PHASE8-DEFINITIVE-TRAIN-SPACING-V1",
        "target_spacing": list(DEFINITIVE_TRAIN_TARGET_SPACING),
    }
    return sha256_json(payload)


def _build_package_environment_reference(package_versions: Mapping[str, str]) -> ArtifactReference:
    normalized = {str(name): str(version) for name, version in package_versions.items()}
    return ArtifactReference(
        schema_name="phase8_definitive_real_training_package_environment",
        schema_version="v1",
        artifact_hash=sha256_json(normalized),
        artifact_role="package_environment",
    )


# ---------------------------------------------------------------------------
# G. Orchestration result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8DefinitiveRealTrainingResult:
    """Result of one real Phase 8 definitive-development training run."""

    output_root: Path
    checkpoint_path_step_250: Path
    checkpoint_path_step_500: Path
    checkpoint_sha256_step_250: str
    checkpoint_sha256_step_500: str
    schedule_hash: str
    config_hash: str
    policy_hash: str
    mean_tumor_dice_step_250: float
    mean_tumor_dice_step_500: float
    selected_checkpoint_step: int
    selection_evidence_hash: str
    access_ledger_hash: str
    elapsed_seconds: float


def _publish_json(path: Path, payload: dict[str, JsonValue]) -> None:
    try:
        publish_text_no_overwrite(
            text=canonical_json_bytes(payload).decode("utf-8") + "\n",
            output_path=path,
            temporary_exists_message="temporary Phase 8 definitive real-training output exists",
            final_exists_message="Phase 8 definitive real-training output already exists",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8DefinitiveRealTrainingPublicationError(
            "failed to publish Phase 8 definitive real-training artifact."
        ) from exc


def run_phase8_definitive_real_training(  # noqa: PLR0913 -- every parameter is explicit input.
    *,
    manifest_path: Path,
    split_path: Path,
    lesion_components_path: Path,
    expected_lesion_components_sha256: str,
    dataset_root: Path,
    output_root: Path,
    repository_root: Path,
    git_commit: str,
    package_versions: Mapping[str, str],
) -> Phase8DefinitiveRealTrainingResult:
    """Run the full real Phase 8 definitive-development training pipeline.

    Every pre-flight safety check (output-root path safety, dataset-root
    path safety, manifest/split/lesion-components hash verification,
    deterministic metadata-only case-ID selection) runs before any NIfTI
    file is opened. The 500-step patch schedule is built from case IDs only.
    Patches are materialized sequentially (one full preprocessed TRAIN
    volume live at a time), training runs as one continuous 500-step
    trajectory with snapshots at steps 250 and 500, both checkpoints are
    published with reject-on-overwrite, both are evaluated on the same
    20-case VALIDATION partition, and the higher-mean-tumor-Dice checkpoint
    is selected (tie -> step 250). Every published checkpoint remains
    ``freeze_eligible=False``.
    """

    start_time = time.monotonic()

    if not _GIT_COMMIT_RE.fullmatch(git_commit):
        raise Phase8DefinitiveRealTrainingRuntimeError("git_commit must match ^[0-9a-f]{7,64}$.")

    canonical_repository_root = repository_root.resolve(strict=True)
    paths = validate_definitive_real_training_output_root(
        output_root, repository_root=canonical_repository_root
    )
    canonical_dataset_root = validate_explicit_dataset_root(dataset_root)

    # validate_definitive_real_training_output_root only confirms output_root
    # does not yet exist; it does not create it. Downstream publishers
    # (materialize_definitive_training_patches, publish_definitive_checkpoint_snapshot)
    # each independently validate their own subdirectory via
    # validate_explicit_external_output_root, which requires that
    # subdirectory's *parent* to already exist -- so output_root itself must
    # be created here, once, before any of them run. No subdirectory
    # (checkpoints/, patches/) is created here; each publisher creates its
    # own leaf directory as part of its own reject-on-overwrite publication.
    paths.output_root.mkdir(parents=True, exist_ok=False)

    metadata = load_and_verify_definitive_real_training_metadata(
        manifest_path=manifest_path,
        split_path=split_path,
        lesion_components_path=lesion_components_path,
        expected_lesion_components_sha256=expected_lesion_components_sha256,
    )

    config = pipeline.build_definitive_config_v1()
    policy = pipeline.build_definitive_training_execution_policy_v1()

    schedule = pipeline.build_definitive_patch_request_schedule(
        metadata.train_case_references,
        positive_case_ids=metadata.positive_case_ids,
        config=config,
        policy=policy,
    )

    ledger = _AccessLedgerRecorder()
    train_loader = _build_real_train_case_loader(
        manifest=metadata.manifest,
        dataset_root=canonical_dataset_root,
        config=config,
        ledger=ledger,
    )

    pipeline.materialize_definitive_training_patches(
        schedule,
        case_references=metadata.train_case_references,
        positive_case_ids=metadata.positive_case_ids,
        loader=train_loader,
        patch_root=paths.patches_dir,
        config=config,
    )

    training_result = pipeline.run_definitive_training_from_materialized_patches(
        schedule, patch_root=paths.patches_dir, config=config, policy=policy
    )
    if not training_result.all_steps_finite:
        raise Phase8DefinitiveRealTrainingRuntimeError(
            "definitive training encountered a non-finite loss/gradient; refusing to publish "
            "any checkpoint or run validation."
        )
    if set(training_result.checkpoint_snapshots.keys()) != {250, 500}:
        raise Phase8DefinitiveRealTrainingRuntimeError(
            "definitive training did not produce both required checkpoint snapshots (250, 500)."
        )

    preprocessing_evidence_hash = _build_preprocessing_evidence_hash(config)
    package_environment_reference = _build_package_environment_reference(package_versions)

    checkpoint_250 = pipeline.publish_definitive_checkpoint_snapshot(
        state_dict=training_result.checkpoint_snapshots[250],
        optimizer_step=250,
        output_root=paths.checkpoints_dir,
        config=config,
        development_manifest_hash=metadata.manifest.manifest_hash,
        development_split_hash=metadata.split.split_hash,
        originating_git_commit=git_commit,
        package_environment_reference=package_environment_reference,
        preprocessing_evidence_hash=preprocessing_evidence_hash,
    )
    checkpoint_500 = pipeline.publish_definitive_checkpoint_snapshot(
        state_dict=training_result.checkpoint_snapshots[500],
        optimizer_step=500,
        output_root=paths.checkpoints_dir,
        config=config,
        development_manifest_hash=metadata.manifest.manifest_hash,
        development_split_hash=metadata.split.split_hash,
        originating_git_commit=git_commit,
        package_environment_reference=package_environment_reference,
        preprocessing_evidence_hash=preprocessing_evidence_hash,
    )

    torch = pipeline._import_torch()
    monai = pipeline._import_monai()

    validation_loader = _build_real_validation_case_loader(
        manifest=metadata.manifest,
        dataset_root=canonical_dataset_root,
        config=config,
        ledger=ledger,
    )

    def _model_from_state_dict(state_dict: Any) -> Any:
        model = pipeline._build_definitive_segresnet_model(torch=torch, monai=monai, config=config)
        model.load_state_dict(state_dict)
        return model

    model_250 = _model_from_state_dict(training_result.checkpoint_snapshots[250])
    result_250 = pipeline.run_definitive_full_validation(
        metadata.validation_case_references,
        loader=validation_loader,
        model=model_250,
        config=config,
        policy=policy,
        candidate_step=250,
    )
    del model_250

    model_500 = _model_from_state_dict(training_result.checkpoint_snapshots[500])
    result_500 = pipeline.run_definitive_full_validation(
        metadata.validation_case_references,
        loader=validation_loader,
        model=model_500,
        config=config,
        policy=policy,
        candidate_step=500,
    )
    del model_500

    selection = pipeline.select_definitive_checkpoint(
        result_step_250=result_250, result_step_500=result_500, policy=policy, config=config
    )
    evidence = pipeline.build_definitive_checkpoint_selection_evidence(
        selection=selection,
        checkpoint_hash_step_250=checkpoint_250.checkpoint_sha256,
        checkpoint_hash_step_500=checkpoint_500.checkpoint_sha256,
    )

    _publish_json(
        paths.output_root / PHASE8_DEFINITIVE_REAL_TRAINING_SELECTION_EVIDENCE_FILENAME,
        cast(
            dict[str, JsonValue],
            {
                "candidate_checkpoint_hash_step_250": evidence.candidate_checkpoint_hash_step_250,
                "candidate_checkpoint_hash_step_500": evidence.candidate_checkpoint_hash_step_500,
                "candidate_step_250": evidence.candidate_step_250,
                "candidate_step_500": evidence.candidate_step_500,
                "evidence_hash": evidence.evidence_hash,
                "external_data_used": evidence.external_data_used,
                "external_labels_used": evidence.external_labels_used,
                "internal_test_used": evidence.internal_test_used,
                "mean_tumor_dice_step_250": evidence.mean_tumor_dice_step_250,
                "mean_tumor_dice_step_500": evidence.mean_tumor_dice_step_500,
                "policy_hash": evidence.policy_hash,
                "schema_name": evidence.schema_name,
                "schema_version": evidence.schema_version,
                "selected_checkpoint_hash": evidence.selected_checkpoint_hash,
                "selected_checkpoint_step": evidence.selected_checkpoint_step,
                "selection_completed": evidence.selection_completed,
                "selection_metric": evidence.selection_metric,
                "tie_break_policy": evidence.tie_break_policy,
                "validation_case_set_identity_hash": evidence.validation_case_set_identity_hash,
            },
        ),
    )
    _publish_json(
        paths.output_root
        / PHASE8_DEFINITIVE_REAL_TRAINING_CHECKPOINT_METADATA_FILENAME_TEMPLATE.format(step=250),
        phase8_checkpoint_metadata_to_dict(checkpoint_250.checkpoint_metadata),
    )
    _publish_json(
        paths.output_root
        / PHASE8_DEFINITIVE_REAL_TRAINING_CHECKPOINT_METADATA_FILENAME_TEMPLATE.format(step=500),
        phase8_checkpoint_metadata_to_dict(checkpoint_500.checkpoint_metadata),
    )
    _publish_json(
        paths.output_root / PHASE8_DEFINITIVE_REAL_TRAINING_ACCESS_LEDGER_FILENAME,
        ledger.to_dict(),
    )

    elapsed_seconds = time.monotonic() - start_time

    return Phase8DefinitiveRealTrainingResult(
        output_root=paths.output_root,
        checkpoint_path_step_250=checkpoint_250.output_path,
        checkpoint_path_step_500=checkpoint_500.output_path,
        checkpoint_sha256_step_250=checkpoint_250.checkpoint_sha256,
        checkpoint_sha256_step_500=checkpoint_500.checkpoint_sha256,
        schedule_hash=schedule.schedule_hash,
        config_hash=config.config_hash,
        policy_hash=policy.policy_hash,
        mean_tumor_dice_step_250=result_250.mean_tumor_dice,
        mean_tumor_dice_step_500=result_500.mean_tumor_dice,
        selected_checkpoint_step=selection.selected_step,
        selection_evidence_hash=evidence.evidence_hash,
        access_ledger_hash=cast(str, ledger.to_dict()["ledger_hash"]),
        elapsed_seconds=elapsed_seconds,
    )


# ---------------------------------------------------------------------------
# H. Process-level watchdog wrapping the entire pipeline
# ---------------------------------------------------------------------------
#
# Mirrors `definitive_training_pilot.run_phase8_bounded_pilot_with_watchdog`:
# the entire pipeline above (manifest verification through selection-
# evidence publication) runs inside an isolated, killable child process, so
# a single long-blocking call (NIfTI I/O, training, sliding-window
# validation) cannot defeat the deadline.

_DEFAULT_CHILD_TERMINATION_GRACE_SECONDS: Final[float] = 5.0


def _definitive_real_training_child_entrypoint(
    conn: Connection,
    target: Callable[..., Phase8DefinitiveRealTrainingResult],
    kwargs: Mapping[str, Any],
) -> None:
    try:
        result = target(**kwargs)
    except BaseException as exc:  # noqa: BLE001 -- propagated to the parent unchanged.
        try:
            conn.send(("error", exc))
        except Exception:  # noqa: BLE001 -- exc itself failed to pickle; send a substitute.
            conn.send(
                (
                    "error",
                    Phase8DefinitiveRealTrainingRuntimeError(
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


def run_phase8_definitive_real_training_with_watchdog(  # noqa: PLR0913
    *,
    manifest_path: Path,
    split_path: Path,
    lesion_components_path: Path,
    expected_lesion_components_sha256: str,
    dataset_root: Path,
    output_root: Path,
    repository_root: Path,
    git_commit: str,
    package_versions: Mapping[str, str],
    wall_clock_limit_seconds: float = DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
    _target: Callable[
        ..., Phase8DefinitiveRealTrainingResult
    ] = run_phase8_definitive_real_training,
    _multiprocessing_context: BaseContext | None = None,
    _termination_grace_seconds: float = _DEFAULT_CHILD_TERMINATION_GRACE_SECONDS,
) -> Phase8DefinitiveRealTrainingResult:
    """Run the real definitive training pipeline inside a supervised child process.

    On timeout the child is terminated (SIGTERM, escalating to ``.kill()``
    after a grace period), joined in a ``finally`` block on every code path,
    and this function raises
    :class:`Phase8DefinitiveRealTrainingWatchdogTimeoutError`. No partial
    success (checkpoint, validation, or selection evidence) is ever surfaced
    for a timed-out call, and there is no automatic retry.

    ``_target`` and ``_multiprocessing_context`` are testing-only seams;
    production callers must not pass them. Production always uses
    ``multiprocessing.get_context("spawn")``.
    """

    supervisor_start_time = time.monotonic()

    kwargs: dict[str, Any] = {
        "manifest_path": manifest_path,
        "split_path": split_path,
        "lesion_components_path": lesion_components_path,
        "expected_lesion_components_sha256": expected_lesion_components_sha256,
        "dataset_root": dataset_root,
        "output_root": output_root,
        "repository_root": repository_root,
        "git_commit": git_commit,
        "package_versions": dict(package_versions),
    }

    ctx = _multiprocessing_context or multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    child = ctx.Process(  # type: ignore[attr-defined]
        target=_definitive_real_training_child_entrypoint,
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
        child_conn.close()

        elapsed_before_join = time.monotonic() - supervisor_start_time
        join_timeout = max(0.0, wall_clock_limit_seconds - elapsed_before_join)
        child.join(timeout=join_timeout)

        if child.is_alive():
            raise Phase8DefinitiveRealTrainingWatchdogTimeoutError(
                f"process-level watchdog killed the child training process after "
                f"{wall_clock_limit_seconds:.3f}s with no publication; the child was "
                f"terminated at the operating-system process level, not by an in-process "
                f"elapsed-time check."
            )

        if parent_conn.poll():
            try:
                status, payload = parent_conn.recv()
            except EOFError as exc:
                raise Phase8DefinitiveRealTrainingRuntimeError(
                    f"child process exited (exitcode={child.exitcode}) without returning a result."
                ) from exc
        else:
            raise Phase8DefinitiveRealTrainingRuntimeError(
                f"child process exited (exitcode={child.exitcode}) without returning a result."
            )

        if status == "error":
            raise cast(BaseException, payload)
        return cast(Phase8DefinitiveRealTrainingResult, payload)
    finally:
        _terminate_and_join_child()
        parent_conn.close()


__all__ = [
    "DEFAULT_WALL_CLOCK_LIMIT_SECONDS",
    "DEFINITIVE_TRAIN_TARGET_SPACING",
    "Phase8DefinitiveRealTrainingAccessRecord",
    "Phase8DefinitiveRealTrainingCaseSelectionError",
    "Phase8DefinitiveRealTrainingDataError",
    "Phase8DefinitiveRealTrainingError",
    "Phase8DefinitiveRealTrainingHashMismatchError",
    "Phase8DefinitiveRealTrainingLabelDomainError",
    "Phase8DefinitiveRealTrainingMetadata",
    "Phase8DefinitiveRealTrainingOrientationError",
    "Phase8DefinitiveRealTrainingOutputRootError",
    "Phase8DefinitiveRealTrainingPaths",
    "Phase8DefinitiveRealTrainingPrerequisitesError",
    "Phase8DefinitiveRealTrainingPublicationError",
    "Phase8DefinitiveRealTrainingResult",
    "Phase8DefinitiveRealTrainingRuntimeError",
    "Phase8DefinitiveRealTrainingWatchdogTimeoutError",
    "REQUIRED_MANIFEST_SHA256",
    "REQUIRED_SPLIT_SHA256",
    "REQUIRED_TRAIN_CASE_COUNT",
    "REQUIRED_VALIDATION_CASE_COUNT",
    "load_and_verify_definitive_real_training_metadata",
    "run_phase8_definitive_real_training",
    "run_phase8_definitive_real_training_with_watchdog",
    "validate_definitive_real_training_output_root",
]
