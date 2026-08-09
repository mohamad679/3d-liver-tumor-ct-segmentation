"""Phase 8 Package F: external image-only inference and prediction lock.

This module wires already-committed, unmodified Phase 8 contracts into a single, minimal driver
that runs frozen-checkpoint inference over the 3D-IRCADb-01 image-only cohort and publishes a
prediction lock -- without ever opening an external label, mask, or mesh file, and without
computing any label-dependent metric.

Every scientific value used here (preprocessing, architecture, threshold, support policy, sliding
window inference parameters) is read unchanged from
:mod:`protoem_ct.external.definitive_pipeline` (``build_definitive_config_v1``). This module adds
exactly what that module does not: real DICOM image loading (image-only, via
:mod:`protoem_ct.external.image_qa`), frozen-artifact identity verification, frozen-checkpoint
loading, prediction array publication, and prediction-lock assembly/publication.

``torch`` and ``monai`` are imported lazily inside function bodies so importing this module never
requires the heavy baseline runtime dependencies.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)
from protoem_ct.data.phase2_paths import (
    InvalidDatasetRootError,
    validate_explicit_external_output_root,
)
from protoem_ct.external import definitive_pipeline as pipeline
from protoem_ct.external.artifacts import phase8_decision_freeze_from_json
from protoem_ct.external.image_qa import (
    Phase8ImageVolumeLoadError,
    load_patient_dicom_zip_volume,
)
from protoem_ct.external.ircadb import (
    Phase8IrcadbInventory,
    discover_phase8_ircadb_image_layout,
)
from protoem_ct.external.preregistration import phase8_external_preregistration_from_json

# ---------------------------------------------------------------------------
# Hard-locked identities (PACKAGE F authorization; see docs/DECISIONS.md).
# ---------------------------------------------------------------------------

REQUIRED_FREEZE_ARTIFACT_SHA256: Final[str] = (
    "5f7579418996d93caa398266f5dfd1c5489b50f3720b23a6d6d5e49f08bb928b"
)
REQUIRED_PREREGISTRATION_HASH: Final[str] = (
    "75d287ade45d2a778153885838d8635606ba5ae739ed88dbf0580671258db291"
)
REQUIRED_CHECKPOINT_SHA256: Final[str] = (
    "2d7989fd134b1348e82cc52afbcf4738c0ce3c17e9c68df774431f577dede651"
)
REQUIRED_DEFINITIVE_CONFIG_HASH: Final[str] = (
    "4e0075e0d499080c4065031d42ea7ff25637cafe154b6757c05edbc14e8c0624"
)
REQUIRED_TARGET_SPACING_XYZ_MM: Final[tuple[float, float, float]] = (
    0.767578125,
    0.767578125,
    1.0,
)
REQUIRED_THRESHOLD: Final[float] = 0.5
REQUIRED_SUPPORT_POLICY: Final[str] = "no_support"
EXTERNAL_LABEL_ACCESS: Final[bool] = False

PHASE8_PREDICTION_RECORD_SCHEMA_NAME: Final[str] = "phase8_external_prediction_record"
PHASE8_PREDICTION_RECORD_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_PREDICTION_LOCK_SCHEMA_NAME: Final[str] = "phase8_external_prediction_lock"
PHASE8_PREDICTION_LOCK_SCHEMA_VERSION: Final[str] = "v1"

PHASE8_PREDICTIONS_SUBDIRECTORY: Final[str] = "predictions"
PHASE8_PREDICTION_LOCK_FILENAME: Final[str] = "phase8_external_prediction_lock.json"
PHASE8_INFERENCE_SUMMARY_FILENAME: Final[str] = "phase8_external_image_only_inference_summary.json"

_GIT_COMMIT_RE_LEN_MIN: Final[int] = 7
_GIT_COMMIT_RE_LEN_MAX: Final[int] = 64


class Phase8ExternalInferenceError(ValueError):
    """Base error for the Phase 8 external image-only inference driver."""


class Phase8ExternalInferenceIdentityError(Phase8ExternalInferenceError):
    """Raised when a frozen freeze/preregistration/checkpoint identity does not match."""


class Phase8ExternalInferenceOutputRootError(Phase8ExternalInferenceError):
    """Raised when the output root already exists or fails path safety."""


class Phase8ExternalInferenceCaseError(Phase8ExternalInferenceError):
    """Raised when one external image case cannot be safely loaded or inferred fail-closed."""


class Phase8ExternalInferencePublicationError(Phase8ExternalInferenceError):
    """Raised when a prediction artifact or the prediction lock cannot be published."""


# ---------------------------------------------------------------------------
# A. Frozen-identity verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8FrozenIdentities:
    """Verified identities for the freeze, preregistration, checkpoint, and config."""

    freeze_artifact_sha256: str
    preregistration_hash: str
    checkpoint_sha256: str
    definitive_config: pipeline.Phase8DefinitiveConfig


def verify_phase8_frozen_identities(
    *,
    freeze_path: Path,
    preregistration_path: Path,
    checkpoint_path: Path,
) -> Phase8FrozenIdentities:
    """Verify the freeze artifact, preregistration, checkpoint, and config identities.

    Fails closed with :class:`Phase8ExternalInferenceIdentityError` on any mismatch. Reads only the
    freeze and preregistration JSON files and the checkpoint's byte content for hashing; the
    checkpoint's tensor payload is not deserialized here.
    """

    for path, name in (
        (freeze_path, "freeze_path"),
        (preregistration_path, "preregistration_path"),
        (checkpoint_path, "checkpoint_path"),
    ):
        if not path.is_absolute():
            raise Phase8ExternalInferenceIdentityError(f"{name} must be an absolute path.")
        if path.is_symlink():
            raise Phase8ExternalInferenceIdentityError(f"{name} must not be a symlink.")
        if not path.is_file():
            raise Phase8ExternalInferenceIdentityError(f"{name} must be a regular file.")

    freeze_sha256 = sha256_file(freeze_path)
    if freeze_sha256 != REQUIRED_FREEZE_ARTIFACT_SHA256:
        raise Phase8ExternalInferenceIdentityError(
            "freeze artifact SHA-256 does not match the required frozen identity."
        )
    freeze = phase8_decision_freeze_from_json(freeze_path.read_bytes())
    if not freeze.frozen_before_external_evaluation or freeze.freeze_state != "frozen":
        raise Phase8ExternalInferenceIdentityError(
            "freeze artifact is not in a frozen, evaluation-ready state."
        )

    preregistration = phase8_external_preregistration_from_json(preregistration_path.read_bytes())
    if preregistration.preregistration_hash != REQUIRED_PREREGISTRATION_HASH:
        raise Phase8ExternalInferenceIdentityError(
            "preregistration_hash does not match the required frozen identity."
        )
    if preregistration.external_label_access_state != "unavailable_before_prediction_lock":
        raise Phase8ExternalInferenceIdentityError(
            "preregistration must declare labels unavailable before prediction lock."
        )
    if not preregistration.no_tuning_declaration:
        raise Phase8ExternalInferenceIdentityError(
            "preregistration must declare no_tuning_declaration=true."
        )

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != REQUIRED_CHECKPOINT_SHA256:
        raise Phase8ExternalInferenceIdentityError(
            "checkpoint SHA-256 does not match the required frozen identity."
        )

    config = pipeline.build_definitive_config_v1()
    if config.config_hash != REQUIRED_DEFINITIVE_CONFIG_HASH:
        raise Phase8ExternalInferenceIdentityError(
            "definitive config hash does not match the required frozen identity."
        )
    if config.inference_threshold != REQUIRED_THRESHOLD:
        raise Phase8ExternalInferenceIdentityError("inference_threshold is not the frozen value.")
    if config.sliding_window_batch_size != 1 or config.sliding_window_overlap != 0.25:
        raise Phase8ExternalInferenceIdentityError("sliding-window parameters are not frozen.")

    return Phase8FrozenIdentities(
        freeze_artifact_sha256=freeze_sha256,
        preregistration_hash=preregistration.preregistration_hash,
        checkpoint_sha256=checkpoint_sha256,
        definitive_config=config,
    )


# ---------------------------------------------------------------------------
# B. Image-only loading and frozen preprocessing (never touches label files)
# ---------------------------------------------------------------------------


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    return cast(
        tuple[float, float, float],
        tuple(float(v) for v in np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))),
    )


def _rescale_affine_to_spacing(
    affine: np.ndarray,
    *,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> np.ndarray:
    """Return the affine implied by resampling from ``source_spacing`` to ``target_spacing``.

    Direction cosines and the world origin are unchanged by
    :func:`protoem_ct.external.definitive_pipeline.resample_volume_to_spacing` (a pure
    componentwise zoom); only each column's magnitude changes to the new per-axis spacing.
    """

    rescaled = affine.copy()
    for axis in range(3):
        scale = target_spacing[axis] / source_spacing[axis]
        rescaled[:3, axis] = affine[:3, axis] * scale
    return rescaled


def _lps_affine_to_ras(affine_lps: np.ndarray) -> np.ndarray:
    """Convert a DICOM-convention (LPS) affine to a RAS+ world affine.

    Mirrors the standard DICOM-to-NIfTI world-convention flip (negate the x and y rows) so that
    :func:`protoem_ct.external.definitive_pipeline.reorient_volume_to_ras` -- which assumes an
    already RAS+ world affine, exactly as it is fed by NIfTI-derived affines elsewhere in this
    package -- receives a geometrically consistent input.
    """

    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    return cast(np.ndarray, flip @ affine_lps)


@dataclass(frozen=True, slots=True)
class Phase8ExternalPreprocessedCase:
    """One image-only case after frozen RAS reorientation, resampling, and intensity scaling."""

    anonymous_case_id: str
    source_case_ordinal: int
    image_archive_sha256: str
    original_volume_shape_zyx: tuple[int, int, int]
    original_voxel_spacing_row_col_slice_mm: tuple[float, float, float]
    ras_reoriented_shape: tuple[int, int, int]
    ras_reoriented_spacing_mm: tuple[float, float, float]
    ras_reoriented_affine_mm: np.ndarray
    resampled_shape: tuple[int, int, int]
    resampled_affine_mm: np.ndarray
    preprocessed_image: np.ndarray


def load_and_preprocess_external_case(
    *,
    patient_dicom_zip: Path,
    anonymous_case_id: str,
    source_case_ordinal: int,
    config: pipeline.Phase8DefinitiveConfig,
) -> Phase8ExternalPreprocessedCase:
    """Load one image-only external case and apply the frozen preprocessing contract.

    Reads only ``patient_dicom_zip``. Never discovers or opens sibling mask/label/mesh files, and
    never uses any label content to determine geometry or intensity normalization.
    """

    try:
        loaded = load_patient_dicom_zip_volume(patient_dicom_zip)
    except Phase8ImageVolumeLoadError as exc:
        raise Phase8ExternalInferenceCaseError(
            f"case {anonymous_case_id}: image could not be safely loaded: {exc}"
        ) from exc

    affine_ras = _lps_affine_to_ras(loaded.affine_lps_mm)
    reoriented_image, reoriented_affine = pipeline.reorient_volume_to_ras(
        loaded.volume_hu, affine_ras
    )
    reoriented_spacing = _spacing_from_affine(reoriented_affine)

    resampled_image = pipeline.resample_volume_to_spacing(
        reoriented_image,
        source_spacing=reoriented_spacing,
        target_spacing=REQUIRED_TARGET_SPACING_XYZ_MM,
        is_label=False,
    )
    scaled_image = pipeline.clip_and_scale_intensity(resampled_image, config=config)
    resampled_affine = _rescale_affine_to_spacing(
        reoriented_affine,
        source_spacing=reoriented_spacing,
        target_spacing=REQUIRED_TARGET_SPACING_XYZ_MM,
    )

    return Phase8ExternalPreprocessedCase(
        anonymous_case_id=anonymous_case_id,
        source_case_ordinal=source_case_ordinal,
        image_archive_sha256=loaded.image_archive_sha256,
        original_volume_shape_zyx=cast(
            tuple[int, int, int], tuple(int(v) for v in loaded.volume_hu.shape)
        ),
        original_voxel_spacing_row_col_slice_mm=loaded.voxel_spacing_row_col_slice_mm,
        ras_reoriented_shape=cast(
            tuple[int, int, int], tuple(int(v) for v in reoriented_image.shape)
        ),
        ras_reoriented_spacing_mm=reoriented_spacing,
        ras_reoriented_affine_mm=reoriented_affine,
        resampled_shape=cast(tuple[int, int, int], tuple(int(v) for v in scaled_image.shape)),
        resampled_affine_mm=resampled_affine,
        preprocessed_image=scaled_image,
    )


# ---------------------------------------------------------------------------
# C. Frozen-checkpoint model loading and sliding-window inference
# ---------------------------------------------------------------------------


def load_frozen_definitive_model(
    *,
    checkpoint_path: Path,
    config: pipeline.Phase8DefinitiveConfig,
) -> Any:
    """Load the frozen SegResNet checkpoint. Caller must have already verified its SHA-256."""

    torch = pipeline._import_torch()  # noqa: SLF001 -- reuse the shared lazy import helper.
    monai = pipeline._import_monai()  # noqa: SLF001
    model = pipeline._build_definitive_segresnet_model(  # noqa: SLF001
        torch=torch, monai=monai, config=config
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@dataclass(frozen=True, slots=True)
class Phase8ExternalInferenceResult:
    """One case's frozen-threshold binary tumor prediction."""

    anonymous_case_id: str
    prediction_mask: np.ndarray
    prediction_finite: bool


def run_frozen_image_only_inference(
    model: Any,
    preprocessed: Phase8ExternalPreprocessedCase,
    *,
    config: pipeline.Phase8DefinitiveConfig,
) -> Phase8ExternalInferenceResult:
    """Run frozen full-volume sliding-window inference and threshold at the frozen value.

    Computes no label-dependent metric. If the model output is not finite, the returned result
    reports ``prediction_finite=False`` and an all-``False`` mask; the caller must fail closed on
    this condition rather than publish the mask.
    """

    torch = pipeline._import_torch()  # noqa: SLF001
    monai = pipeline._import_monai()  # noqa: SLF001

    image_tensor = torch.from_numpy(
        np.ascontiguousarray(preprocessed.preprocessed_image.astype(np.float32))[None, None, ...]
    ).to(dtype=torch.float32, device=torch.device("cpu"))

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
        empty_mask = np.zeros(preprocessed.resampled_shape, dtype=np.uint8)
        return Phase8ExternalInferenceResult(
            anonymous_case_id=preprocessed.anonymous_case_id,
            prediction_mask=empty_mask,
            prediction_finite=False,
        )

    tumor_probability = probabilities[0, 1, ...]
    mask = (tumor_probability >= config.inference_threshold).cpu().numpy().astype(np.uint8)
    return Phase8ExternalInferenceResult(
        anonymous_case_id=preprocessed.anonymous_case_id,
        prediction_mask=mask,
        prediction_finite=True,
    )


# ---------------------------------------------------------------------------
# D. Prediction record contract (self-hashing, image-only provenance)
# ---------------------------------------------------------------------------


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise Phase8ExternalInferenceError(f"{field_name} must be a lowercase SHA-256 hex digest.")


def _require_git_commit(value: str) -> None:
    if not (_GIT_COMMIT_RE_LEN_MIN <= len(value) <= _GIT_COMMIT_RE_LEN_MAX) or any(
        ch not in "0123456789abcdef" for ch in value
    ):
        raise Phase8ExternalInferenceError("git_commit must be a lowercase hex commit identifier.")


def _rounded_affine_tuple(affine: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(round(float(value), 10) for value in row) for row in affine)


@dataclass(frozen=True, slots=True)
class Phase8ExternalPredictionRecord:
    """Immutable, self-hashing per-case prediction record (image-only provenance)."""

    schema_name: str
    schema_version: str
    anonymous_case_id: str
    source_case_ordinal: int
    prediction_file_relative_path: str
    prediction_sha256: str
    prediction_shape_zyx: tuple[int, int, int]
    prediction_dtype: str
    binary_semantics: str
    threshold: float
    source_image_archive_sha256: str
    original_volume_shape_zyx: tuple[int, int, int]
    original_voxel_spacing_row_col_slice_mm: tuple[float, float, float]
    ras_reoriented_shape_zyx: tuple[int, int, int]
    ras_reoriented_affine_mm: tuple[tuple[float, ...], ...]
    resampled_shape_zyx: tuple[int, int, int]
    resampled_affine_mm: tuple[tuple[float, ...], ...]
    target_spacing_xyz_mm: tuple[float, float, float]
    preprocessing_orientation: str
    preprocessing_image_interpolation: str
    checkpoint_sha256: str
    definitive_config_hash: str
    freeze_artifact_sha256: str
    preregistration_hash: str
    support_policy: str
    external_label_access: bool
    record_hash: str

    def __post_init__(self) -> None:
        if self.schema_name != PHASE8_PREDICTION_RECORD_SCHEMA_NAME:
            raise Phase8ExternalInferenceError("prediction record schema_name mismatch.")
        if self.schema_version != PHASE8_PREDICTION_RECORD_SCHEMA_VERSION:
            raise Phase8ExternalInferenceError("prediction record schema_version mismatch.")
        for value, name in (
            (self.prediction_sha256, "prediction_sha256"),
            (self.source_image_archive_sha256, "source_image_archive_sha256"),
            (self.checkpoint_sha256, "checkpoint_sha256"),
            (self.definitive_config_hash, "definitive_config_hash"),
            (self.freeze_artifact_sha256, "freeze_artifact_sha256"),
            (self.preregistration_hash, "preregistration_hash"),
        ):
            _require_sha256(value, field_name=name)
        if self.threshold != REQUIRED_THRESHOLD:
            raise Phase8ExternalInferenceError("threshold must equal the frozen value.")
        if self.support_policy != REQUIRED_SUPPORT_POLICY:
            raise Phase8ExternalInferenceError("support_policy must equal the frozen value.")
        if self.external_label_access is not False:
            raise Phase8ExternalInferenceError("external_label_access must be false.")
        if self.prediction_dtype != "uint8":
            raise Phase8ExternalInferenceError("prediction_dtype must be uint8.")
        expected_hash = hash_phase8_external_prediction_record(self)
        if self.record_hash != expected_hash:
            raise Phase8ExternalInferenceError(
                "record_hash does not match deterministic prediction-record content."
            )


def phase8_external_prediction_record_identity_payload(
    record: Phase8ExternalPredictionRecord,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one prediction record, excluding its own hash."""

    return {
        "anonymous_case_id": record.anonymous_case_id,
        "binary_semantics": record.binary_semantics,
        "checkpoint_sha256": record.checkpoint_sha256,
        "definitive_config_hash": record.definitive_config_hash,
        "external_label_access": record.external_label_access,
        "freeze_artifact_sha256": record.freeze_artifact_sha256,
        "original_volume_shape_zyx": list(record.original_volume_shape_zyx),
        "original_voxel_spacing_row_col_slice_mm": list(
            record.original_voxel_spacing_row_col_slice_mm
        ),
        "prediction_dtype": record.prediction_dtype,
        "prediction_file_relative_path": record.prediction_file_relative_path,
        "prediction_sha256": record.prediction_sha256,
        "prediction_shape_zyx": list(record.prediction_shape_zyx),
        "preprocessing_image_interpolation": record.preprocessing_image_interpolation,
        "preprocessing_orientation": record.preprocessing_orientation,
        "preregistration_hash": record.preregistration_hash,
        "ras_reoriented_affine_mm": [list(row) for row in record.ras_reoriented_affine_mm],
        "ras_reoriented_shape_zyx": list(record.ras_reoriented_shape_zyx),
        "resampled_affine_mm": [list(row) for row in record.resampled_affine_mm],
        "resampled_shape_zyx": list(record.resampled_shape_zyx),
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "source_case_ordinal": record.source_case_ordinal,
        "source_image_archive_sha256": record.source_image_archive_sha256,
        "support_policy": record.support_policy,
        "target_spacing_xyz_mm": list(record.target_spacing_xyz_mm),
        "threshold": record.threshold,
    }


def hash_phase8_external_prediction_record(record: Phase8ExternalPredictionRecord) -> str:
    """Return the lowercase SHA-256 identity hash for one prediction record."""

    return sha256_json(phase8_external_prediction_record_identity_payload(record))


def phase8_external_prediction_record_to_dict(
    record: Phase8ExternalPredictionRecord,
) -> dict[str, JsonValue]:
    """Convert one prediction record to a canonical mapping including ``record_hash``."""

    payload = phase8_external_prediction_record_identity_payload(record)
    payload["record_hash"] = record.record_hash
    return payload


def build_phase8_external_prediction_record(
    *,
    preprocessed: Phase8ExternalPreprocessedCase,
    prediction_file_relative_path: str,
    prediction_sha256: str,
    prediction_shape_zyx: tuple[int, int, int],
    identities: Phase8FrozenIdentities,
) -> Phase8ExternalPredictionRecord:
    """Build a self-hashed prediction record from a preprocessed case and inference result."""

    payload_without_hash: dict[str, JsonValue] = {
        "anonymous_case_id": preprocessed.anonymous_case_id,
        "binary_semantics": "1_if_tumor_probability_ge_threshold_else_0",
        "checkpoint_sha256": identities.checkpoint_sha256,
        "definitive_config_hash": identities.definitive_config.config_hash,
        "external_label_access": EXTERNAL_LABEL_ACCESS,
        "freeze_artifact_sha256": identities.freeze_artifact_sha256,
        "original_volume_shape_zyx": list(preprocessed.original_volume_shape_zyx),
        "original_voxel_spacing_row_col_slice_mm": list(
            preprocessed.original_voxel_spacing_row_col_slice_mm
        ),
        "prediction_dtype": "uint8",
        "prediction_file_relative_path": prediction_file_relative_path,
        "prediction_sha256": prediction_sha256,
        "prediction_shape_zyx": list(prediction_shape_zyx),
        "preprocessing_image_interpolation": (
            identities.definitive_config.image_interpolation_policy
        ),
        "preprocessing_orientation": identities.definitive_config.orientation_policy,
        "preregistration_hash": identities.preregistration_hash,
        "ras_reoriented_affine_mm": [
            [round(float(v), 10) for v in row] for row in preprocessed.ras_reoriented_affine_mm
        ],
        "ras_reoriented_shape_zyx": list(preprocessed.ras_reoriented_shape),
        "resampled_affine_mm": [
            [round(float(v), 10) for v in row] for row in preprocessed.resampled_affine_mm
        ],
        "resampled_shape_zyx": list(preprocessed.resampled_shape),
        "schema_name": PHASE8_PREDICTION_RECORD_SCHEMA_NAME,
        "schema_version": PHASE8_PREDICTION_RECORD_SCHEMA_VERSION,
        "source_case_ordinal": preprocessed.source_case_ordinal,
        "source_image_archive_sha256": preprocessed.image_archive_sha256,
        "support_policy": REQUIRED_SUPPORT_POLICY,
        "target_spacing_xyz_mm": list(REQUIRED_TARGET_SPACING_XYZ_MM),
        "threshold": REQUIRED_THRESHOLD,
    }
    record_hash = sha256_json(payload_without_hash)
    return Phase8ExternalPredictionRecord(
        schema_name=PHASE8_PREDICTION_RECORD_SCHEMA_NAME,
        schema_version=PHASE8_PREDICTION_RECORD_SCHEMA_VERSION,
        anonymous_case_id=preprocessed.anonymous_case_id,
        source_case_ordinal=preprocessed.source_case_ordinal,
        prediction_file_relative_path=prediction_file_relative_path,
        prediction_sha256=prediction_sha256,
        prediction_shape_zyx=prediction_shape_zyx,
        prediction_dtype="uint8",
        binary_semantics="1_if_tumor_probability_ge_threshold_else_0",
        threshold=REQUIRED_THRESHOLD,
        source_image_archive_sha256=preprocessed.image_archive_sha256,
        original_volume_shape_zyx=preprocessed.original_volume_shape_zyx,
        original_voxel_spacing_row_col_slice_mm=(
            preprocessed.original_voxel_spacing_row_col_slice_mm
        ),
        ras_reoriented_shape_zyx=preprocessed.ras_reoriented_shape,
        ras_reoriented_affine_mm=_rounded_affine_tuple(preprocessed.ras_reoriented_affine_mm),
        resampled_shape_zyx=preprocessed.resampled_shape,
        resampled_affine_mm=_rounded_affine_tuple(preprocessed.resampled_affine_mm),
        target_spacing_xyz_mm=REQUIRED_TARGET_SPACING_XYZ_MM,
        preprocessing_orientation=identities.definitive_config.orientation_policy,
        preprocessing_image_interpolation=identities.definitive_config.image_interpolation_policy,
        checkpoint_sha256=identities.checkpoint_sha256,
        definitive_config_hash=identities.definitive_config.config_hash,
        freeze_artifact_sha256=identities.freeze_artifact_sha256,
        preregistration_hash=identities.preregistration_hash,
        support_policy=REQUIRED_SUPPORT_POLICY,
        external_label_access=EXTERNAL_LABEL_ACCESS,
        record_hash=record_hash,
    )


# ---------------------------------------------------------------------------
# E. Prediction lock contract (self-hashing, complete-inventory-required)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8ExternalPredictionLock:
    """Immutable, self-hashing complete inventory of every locked external prediction."""

    schema_name: str
    schema_version: str
    cohort_identifier: str
    preregistration_hash: str
    freeze_artifact_sha256: str
    checkpoint_sha256: str
    definitive_config_hash: str
    frozen_threshold: float
    support_policy: str
    target_spacing_xyz_mm: tuple[float, float, float]
    observed_image_case_count: int
    ordered_case_ids: tuple[str, ...]
    case_record_hashes: tuple[str, ...]
    prediction_sha256_by_case: tuple[tuple[str, str], ...]
    inference_completion_state: str
    external_label_access: bool
    no_tuning: bool
    git_commit: str
    lock_hash: str

    def __post_init__(self) -> None:
        if self.schema_name != PHASE8_PREDICTION_LOCK_SCHEMA_NAME:
            raise Phase8ExternalInferenceError("prediction lock schema_name mismatch.")
        if self.schema_version != PHASE8_PREDICTION_LOCK_SCHEMA_VERSION:
            raise Phase8ExternalInferenceError("prediction lock schema_version mismatch.")
        for value, name in (
            (self.preregistration_hash, "preregistration_hash"),
            (self.freeze_artifact_sha256, "freeze_artifact_sha256"),
            (self.checkpoint_sha256, "checkpoint_sha256"),
            (self.definitive_config_hash, "definitive_config_hash"),
        ):
            _require_sha256(value, field_name=name)
        _require_git_commit(self.git_commit)
        if self.frozen_threshold != REQUIRED_THRESHOLD:
            raise Phase8ExternalInferenceError("frozen_threshold must equal the required value.")
        if self.support_policy != REQUIRED_SUPPORT_POLICY:
            raise Phase8ExternalInferenceError("support_policy must equal the required value.")
        if self.external_label_access is not False:
            raise Phase8ExternalInferenceError("external_label_access must be false.")
        if not self.no_tuning:
            raise Phase8ExternalInferenceError("no_tuning must be true.")
        if self.inference_completion_state != "completed":
            raise Phase8ExternalInferenceError(
                "inference_completion_state must be 'completed' for a published lock."
            )
        if len(self.ordered_case_ids) != len(set(self.ordered_case_ids)):
            raise Phase8ExternalInferenceError("ordered_case_ids must not contain duplicates.")
        if self.observed_image_case_count != len(self.ordered_case_ids):
            raise Phase8ExternalInferenceError(
                "observed_image_case_count must equal len(ordered_case_ids)."
            )
        if len(self.case_record_hashes) != len(self.ordered_case_ids):
            raise Phase8ExternalInferenceError(
                "case_record_hashes must have one entry per ordered case ID."
            )
        prediction_case_ids = tuple(case_id for case_id, _sha256 in self.prediction_sha256_by_case)
        if prediction_case_ids != self.ordered_case_ids:
            raise Phase8ExternalInferenceError(
                "prediction_sha256_by_case must cover exactly ordered_case_ids, in order."
            )
        for _case_id, prediction_sha256 in self.prediction_sha256_by_case:
            _require_sha256(prediction_sha256, field_name="prediction_sha256_by_case")
        expected_hash = hash_phase8_external_prediction_lock(self)
        if self.lock_hash != expected_hash:
            raise Phase8ExternalInferenceError(
                "lock_hash does not match deterministic prediction-lock content."
            )


def phase8_external_prediction_lock_identity_payload(
    lock: Phase8ExternalPredictionLock,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for the prediction lock, excluding ``lock_hash``."""

    return {
        "case_record_hashes": list(lock.case_record_hashes),
        "checkpoint_sha256": lock.checkpoint_sha256,
        "cohort_identifier": lock.cohort_identifier,
        "definitive_config_hash": lock.definitive_config_hash,
        "external_label_access": lock.external_label_access,
        "freeze_artifact_sha256": lock.freeze_artifact_sha256,
        "frozen_threshold": lock.frozen_threshold,
        "git_commit": lock.git_commit,
        "inference_completion_state": lock.inference_completion_state,
        "no_tuning": lock.no_tuning,
        "observed_image_case_count": lock.observed_image_case_count,
        "ordered_case_ids": list(lock.ordered_case_ids),
        "prediction_sha256_by_case": [list(item) for item in lock.prediction_sha256_by_case],
        "preregistration_hash": lock.preregistration_hash,
        "schema_name": lock.schema_name,
        "schema_version": lock.schema_version,
        "support_policy": lock.support_policy,
        "target_spacing_xyz_mm": list(lock.target_spacing_xyz_mm),
    }


def hash_phase8_external_prediction_lock(lock: Phase8ExternalPredictionLock) -> str:
    """Return the lowercase SHA-256 identity hash for the prediction lock."""

    return sha256_json(phase8_external_prediction_lock_identity_payload(lock))


def phase8_external_prediction_lock_to_dict(
    lock: Phase8ExternalPredictionLock,
) -> dict[str, JsonValue]:
    """Convert the prediction lock to a canonical mapping including ``lock_hash``."""

    payload = phase8_external_prediction_lock_identity_payload(lock)
    payload["lock_hash"] = lock.lock_hash
    return payload


def build_phase8_external_prediction_lock(
    *,
    records: tuple[Phase8ExternalPredictionRecord, ...],
    identities: Phase8FrozenIdentities,
    git_commit: str,
) -> Phase8ExternalPredictionLock:
    """Build a self-hashed prediction lock from a complete, ordered set of prediction records."""

    ordered = tuple(sorted(records, key=lambda item: item.source_case_ordinal))
    ordered_case_ids = tuple(item.anonymous_case_id for item in ordered)
    case_record_hashes = tuple(item.record_hash for item in ordered)
    prediction_sha256_by_case = tuple(
        (item.anonymous_case_id, item.prediction_sha256) for item in ordered
    )
    payload: dict[str, JsonValue] = {
        "case_record_hashes": list(case_record_hashes),
        "checkpoint_sha256": identities.checkpoint_sha256,
        "cohort_identifier": "3d_ircadb_01",
        "definitive_config_hash": identities.definitive_config.config_hash,
        "external_label_access": EXTERNAL_LABEL_ACCESS,
        "freeze_artifact_sha256": identities.freeze_artifact_sha256,
        "frozen_threshold": REQUIRED_THRESHOLD,
        "git_commit": git_commit,
        "inference_completion_state": "completed",
        "no_tuning": True,
        "observed_image_case_count": len(ordered_case_ids),
        "ordered_case_ids": list(ordered_case_ids),
        "prediction_sha256_by_case": [list(item) for item in prediction_sha256_by_case],
        "preregistration_hash": identities.preregistration_hash,
        "schema_name": PHASE8_PREDICTION_LOCK_SCHEMA_NAME,
        "schema_version": PHASE8_PREDICTION_LOCK_SCHEMA_VERSION,
        "support_policy": REQUIRED_SUPPORT_POLICY,
        "target_spacing_xyz_mm": list(REQUIRED_TARGET_SPACING_XYZ_MM),
    }
    lock_hash = sha256_json(payload)
    return Phase8ExternalPredictionLock(
        schema_name=PHASE8_PREDICTION_LOCK_SCHEMA_NAME,
        schema_version=PHASE8_PREDICTION_LOCK_SCHEMA_VERSION,
        cohort_identifier="3d_ircadb_01",
        preregistration_hash=identities.preregistration_hash,
        freeze_artifact_sha256=identities.freeze_artifact_sha256,
        checkpoint_sha256=identities.checkpoint_sha256,
        definitive_config_hash=identities.definitive_config.config_hash,
        frozen_threshold=REQUIRED_THRESHOLD,
        support_policy=REQUIRED_SUPPORT_POLICY,
        target_spacing_xyz_mm=REQUIRED_TARGET_SPACING_XYZ_MM,
        observed_image_case_count=len(ordered_case_ids),
        ordered_case_ids=ordered_case_ids,
        case_record_hashes=case_record_hashes,
        prediction_sha256_by_case=prediction_sha256_by_case,
        inference_completion_state="completed",
        external_label_access=EXTERNAL_LABEL_ACCESS,
        no_tuning=True,
        git_commit=git_commit,
        lock_hash=lock_hash,
    )


# ---------------------------------------------------------------------------
# F. Publication helpers (no-overwrite; mirrors the rest of Phase 8)
# ---------------------------------------------------------------------------


def _publish_json(path: Path, payload: dict[str, JsonValue]) -> None:
    try:
        publish_text_no_overwrite(
            text=(canonical_json_bytes(payload) + b"\n").decode("utf-8"),
            output_path=path,
            temporary_exists_message="temporary Phase 8 Package F artifact already exists.",
            final_exists_message="Phase 8 Package F artifact already exists.",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8ExternalInferencePublicationError(str(exc)) from exc


def _publish_prediction_array(path: Path, mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, mask, allow_pickle=False)
    payload = buffer.getvalue()
    temp_path = path.with_name(f".{path.name}.tmp")
    if temp_path.exists() or path.exists():
        raise Phase8ExternalInferencePublicationError(
            f"prediction artifact already exists or is mid-publication: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(payload)
    try:
        import os

        os.link(temp_path, path)
        temp_path.unlink()
    except OSError:
        temp_path.rename(path)
    return sha256_file(path)


def _validate_output_root(output_root: Path, *, dataset_root: Path, repository_root: Path) -> Path:
    if output_root.exists():
        raise Phase8ExternalInferenceOutputRootError(
            f"output_root already exists; refusing to reuse or silently version: {output_root}"
        )
    try:
        return validate_explicit_external_output_root(
            output_root,
            forbidden_roots=(dataset_root, repository_root),
        )
    except InvalidDatasetRootError as exc:
        raise Phase8ExternalInferenceOutputRootError(str(exc)) from exc


# ---------------------------------------------------------------------------
# G. Top-level driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8ExternalInferenceRunResult:
    """Aggregate result of one definitive external image-only inference run."""

    output_root: Path
    identities: Phase8FrozenIdentities
    ircadb_inventory: Phase8IrcadbInventory
    records: tuple[Phase8ExternalPredictionRecord, ...]
    lock: Phase8ExternalPredictionLock
    lock_hash: str
    per_case_elapsed_seconds: tuple[tuple[str, float], ...]
    total_elapsed_seconds: float


def run_phase8_external_image_only_inference(
    *,
    dataset_root: Path,
    freeze_path: Path,
    preregistration_path: Path,
    checkpoint_path: Path,
    output_root: Path,
    repository_root: Path,
    git_commit: str,
) -> Phase8ExternalInferenceRunResult:
    """Run the complete, once-only, image-only external inference and publish the prediction lock.

    Fails closed (raising before publishing a lock) on: any frozen-identity mismatch, an
    already-existing output root, any unreadable/inconsistent image case, or any non-finite model
    output. Never opens a mask/label/mesh file. Never computes a label-dependent metric.
    """

    start_time = time.monotonic()
    if output_root.exists():
        raise Phase8ExternalInferenceOutputRootError(
            f"output_root already exists; refusing to reuse or silently version: {output_root}"
        )
    identities = verify_phase8_frozen_identities(
        freeze_path=freeze_path,
        preregistration_path=preregistration_path,
        checkpoint_path=checkpoint_path,
    )
    canonical_output_root = _validate_output_root(
        output_root, dataset_root=dataset_root, repository_root=repository_root
    )
    inventory = discover_phase8_ircadb_image_layout(dataset_root)
    canonical_dataset_root = dataset_root.resolve(strict=True)

    model = load_frozen_definitive_model(
        checkpoint_path=checkpoint_path, config=identities.definitive_config
    )

    records: list[Phase8ExternalPredictionRecord] = []
    per_case_elapsed: list[tuple[str, float]] = []

    for case in inventory.cases:
        case_start = time.monotonic()
        patient_archive = canonical_dataset_root / case.patient_dicom_zip_relative_path
        preprocessed = load_and_preprocess_external_case(
            patient_dicom_zip=patient_archive,
            anonymous_case_id=case.anonymous_case_id,
            source_case_ordinal=case.source_case_ordinal,
            config=identities.definitive_config,
        )
        inference_result = run_frozen_image_only_inference(
            model, preprocessed, config=identities.definitive_config
        )
        if not inference_result.prediction_finite:
            raise Phase8ExternalInferenceCaseError(
                f"case {case.anonymous_case_id}: model output was not finite; failing closed."
            )
        prediction_relative_path = f"{PHASE8_PREDICTIONS_SUBDIRECTORY}/{case.anonymous_case_id}.npy"
        prediction_path = canonical_output_root / prediction_relative_path
        prediction_sha256 = _publish_prediction_array(
            prediction_path, inference_result.prediction_mask
        )
        record = build_phase8_external_prediction_record(
            preprocessed=preprocessed,
            prediction_file_relative_path=prediction_relative_path,
            prediction_sha256=prediction_sha256,
            prediction_shape_zyx=cast(
                tuple[int, int, int],
                tuple(int(v) for v in inference_result.prediction_mask.shape),
            ),
            identities=identities,
        )
        _publish_json(
            canonical_output_root
            / f"{PHASE8_PREDICTIONS_SUBDIRECTORY}/{case.anonymous_case_id}_prediction_record.json",
            phase8_external_prediction_record_to_dict(record),
        )
        records.append(record)
        per_case_elapsed.append((case.anonymous_case_id, time.monotonic() - case_start))

    if len(records) != inventory.expected_case_count:
        raise Phase8ExternalInferenceCaseError(
            "not every discovered external image case produced a prediction; failing closed."
        )

    lock = build_phase8_external_prediction_lock(
        records=tuple(records), identities=identities, git_commit=git_commit
    )
    _publish_json(
        canonical_output_root / PHASE8_PREDICTION_LOCK_FILENAME,
        phase8_external_prediction_lock_to_dict(lock),
    )

    total_elapsed = time.monotonic() - start_time
    summary: dict[str, JsonValue] = {
        "checkpoint_sha256": identities.checkpoint_sha256,
        "definitive_config_hash": identities.definitive_config.config_hash,
        "external_label_access": EXTERNAL_LABEL_ACCESS,
        "freeze_artifact_sha256": identities.freeze_artifact_sha256,
        "git_commit": git_commit,
        "lock_hash": lock.lock_hash,
        "no_metric_computed": True,
        "no_tuning": True,
        "observed_image_case_count": len(records),
        "per_case_elapsed_seconds": {case_id: elapsed for case_id, elapsed in per_case_elapsed},
        "preregistration_hash": identities.preregistration_hash,
        "schema_name": "phase8_external_image_only_inference_summary",
        "schema_version": "v1",
        "total_elapsed_seconds": total_elapsed,
    }
    _publish_json(canonical_output_root / PHASE8_INFERENCE_SUMMARY_FILENAME, summary)

    return Phase8ExternalInferenceRunResult(
        output_root=canonical_output_root,
        identities=identities,
        ircadb_inventory=inventory,
        records=tuple(records),
        lock=lock,
        lock_hash=lock.lock_hash,
        per_case_elapsed_seconds=tuple(per_case_elapsed),
        total_elapsed_seconds=total_elapsed,
    )


__all__ = [
    "EXTERNAL_LABEL_ACCESS",
    "PHASE8_INFERENCE_SUMMARY_FILENAME",
    "PHASE8_PREDICTIONS_SUBDIRECTORY",
    "PHASE8_PREDICTION_LOCK_FILENAME",
    "PHASE8_PREDICTION_LOCK_SCHEMA_NAME",
    "PHASE8_PREDICTION_LOCK_SCHEMA_VERSION",
    "PHASE8_PREDICTION_RECORD_SCHEMA_NAME",
    "PHASE8_PREDICTION_RECORD_SCHEMA_VERSION",
    "REQUIRED_CHECKPOINT_SHA256",
    "REQUIRED_DEFINITIVE_CONFIG_HASH",
    "REQUIRED_FREEZE_ARTIFACT_SHA256",
    "REQUIRED_PREREGISTRATION_HASH",
    "REQUIRED_SUPPORT_POLICY",
    "REQUIRED_TARGET_SPACING_XYZ_MM",
    "REQUIRED_THRESHOLD",
    "Phase8ExternalInferenceCaseError",
    "Phase8ExternalInferenceError",
    "Phase8ExternalInferenceIdentityError",
    "Phase8ExternalInferenceOutputRootError",
    "Phase8ExternalInferencePublicationError",
    "Phase8ExternalInferenceResult",
    "Phase8ExternalInferenceRunResult",
    "Phase8ExternalPredictionLock",
    "Phase8ExternalPredictionRecord",
    "Phase8ExternalPreprocessedCase",
    "Phase8FrozenIdentities",
    "build_phase8_external_prediction_lock",
    "build_phase8_external_prediction_record",
    "hash_phase8_external_prediction_lock",
    "hash_phase8_external_prediction_record",
    "load_and_preprocess_external_case",
    "load_frozen_definitive_model",
    "phase8_external_prediction_lock_identity_payload",
    "phase8_external_prediction_lock_to_dict",
    "phase8_external_prediction_record_identity_payload",
    "phase8_external_prediction_record_to_dict",
    "run_frozen_image_only_inference",
    "run_phase8_external_image_only_inference",
    "verify_phase8_frozen_identities",
]
