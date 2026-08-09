"""Phase 8 Package G: external label evaluation of already-frozen, already-locked predictions.

This is the FIRST module in this repository authorized to open real 3D-IRCADb-01 tumor labels.
It exists exclusively to *evaluate* predictions that were generated and locked without label access
(Package F, :mod:`protoem_ct.external.image_only_inference`). It must never regenerate, adjust, or
influence those predictions:

* It never imports or invokes model inference code (no ``torch``, no ``monai``, no checkpoint load).
* It never opens a locked prediction ``.npy`` file in write mode.
* It re-verifies every locked prediction's SHA-256 against the prediction lock at evaluation time
  (defense in depth beyond whatever verification produced the lock).
* It reuses the already-committed, unmodified label-mapping policy, eligibility contracts, metric
  formulas, and geometry-alignment primitives verbatim; it adds no new statistical method and no new
  threshold/preprocessing/support policy.
* Exclusions are driven only by the preregistered fail-closed contracts (missing tumor source,
  geometry incompatibility) -- never by a case's metric values.

Only 3D-IRCADb-01's ``MASKS_DICOM.zip`` archives are opened for label content.
``LABELLED_DICOM.zip``, ``MESHES_VTK.zip``, and ``liver_*.jpg`` are never opened. Within
``MASKS_DICOM.zip``, only role folders whose name case-insensitively starts with ``livertumor`` are
read; the ``liver`` (liver context) folder and any other organ folder are never opened.
"""

from __future__ import annotations

import contextlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np

from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.baselines.metrics import (
    BaselineCaseMetrics,
    BaselineMetricReport,
    baseline_case_metrics_to_json,
    baseline_metric_report_to_json,
    build_baseline_metric_report,
    compute_baseline_case_metrics,
)
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
from protoem_ct.external.domain_shift import (
    Phase8DomainShiftRecord,
    build_phase8_domain_shift_record,
    phase8_domain_shift_record_to_dict,
)
from protoem_ct.external.eligibility import (
    Phase8EligibilityAccounting,
    Phase8EligibilityCase,
    build_phase8_label_compatible_case,
    build_phase8_post_label_eligibility_accounting,
    phase8_eligibility_accounting_from_json,
    phase8_eligibility_accounting_to_dict,
)
from protoem_ct.external.image_only_inference import (
    REQUIRED_CHECKPOINT_SHA256,
    REQUIRED_DEFINITIVE_CONFIG_HASH,
    REQUIRED_FREEZE_ARTIFACT_SHA256,
    REQUIRED_PREREGISTRATION_HASH,
    REQUIRED_SUPPORT_POLICY,
    REQUIRED_TARGET_SPACING_XYZ_MM,
    REQUIRED_THRESHOLD,
    Phase8ExternalInferenceError,
    Phase8ExternalPredictionLock,
    Phase8ExternalPredictionRecord,
    _lps_affine_to_ras,  # noqa: SLF001 -- reuse the shared LPS->RAS convention helper.
    _spacing_from_affine,  # noqa: SLF001 -- reuse the shared affine-to-spacing helper.
)
from protoem_ct.external.image_qa import (
    Phase8DicomSlice,
    Phase8ImageQaError,
    Phase8ImageVolumeLoadError,
    _build_lps_affine,  # noqa: SLF001 -- reuse the shared LPS affine builder.
    _read_dicom_slice,  # noqa: SLF001 -- reuse the shared safe DICOM header parser.
    _slice_sort_key,  # noqa: SLF001 -- reuse the shared slice-ordering key.
    _voxel_spacing,  # noqa: SLF001 -- reuse the shared voxel-spacing estimator.
)
from protoem_ct.external.ircadb import (
    discover_phase8_ircadb_image_layout,
    is_ignored_macos_metadata_path,
    normalize_safe_ircadb_zip_member_path,
)
from protoem_ct.external.label_mapping import (
    Phase8LabelMappingValidationError,
    apply_phase8_label_mapping_policy,
    build_default_phase8_label_mapping_policy,
)
from protoem_ct.external.manifest import (
    PHASE8_IRCADB_CASE_COUNT,
    Phase8ExternalImageManifest,
    phase8_external_image_manifest_from_json,
)
from protoem_ct.external.preregistration import phase8_external_preregistration_from_json
from protoem_ct.external.statistical_policy import phase8_statistical_policy_component_hash

PHASE8_LABEL_EVALUATION_SCHEMA_NAME: Final[str] = "phase8_external_label_evaluation_summary"
PHASE8_LABEL_EVALUATION_SCHEMA_VERSION: Final[str] = "v1"

REQUIRED_LOCK_HASH: Final[str] = "190532a6abb7c3308de9abe7dc7455f4fafae464d51347ea2273874d13afdc94"
REQUIRED_COHORT_IDENTIFIER: Final[str] = "3d_ircadb_01"

_MASKS_ARCHIVE_NAME: Final[str] = "MASKS_DICOM.zip"
_LIVERTUMOR_PREFIX: Final[str] = "livertumor"
_BOOTSTRAP_SEED: Final[int] = 1729
_BOOTSTRAP_RESAMPLE_COUNT: Final[int] = 10000
_BOOTSTRAP_CONFIDENCE: Final[float] = 0.95

# Module-level (rather than inline) so tests can monkeypatch it without touching the real
# external drive. This path is read-only evidence produced by the already-frozen Package C
# checkpoint-selection step; it is never written by this module.
_INTERNAL_CHECKPOINT_EVIDENCE_PATH: Final[Path] = Path(
    "/Volumes/Lexar/ProtoEM-CT/runs/phase8_definitive_training_v1/"
    "phase8_definitive_checkpoint_selection_evidence.json"
)

_PHASE8_METRIC_SCALAR_EXTRACTORS: Final[
    dict[str, Any]
] = {}  # populated below, after BaselineCaseMetrics is imported.


class Phase8LabelEvaluationError(ValueError):
    """Base error for Phase 8 Package G external label evaluation."""


class Phase8LabelEvaluationIdentityError(Phase8LabelEvaluationError):
    """Raised when a frozen freeze/preregistration/lock identity does not match."""


class Phase8LabelEvaluationGeometryError(Phase8LabelEvaluationError):
    """Raised (and caught per-case) when one label's geometry cannot be safely aligned."""


class Phase8LabelEvaluationImplementationError(Phase8LabelEvaluationError):
    """Raised (never caught) when the geometry-alignment chain violates its own guarantees.

    This indicates a defect in this module's own driver code, not a data-quality issue -- it must
    halt the run rather than be silently coerced or treated as a per-case exclusion.
    """


class Phase8LabelEvaluationOutputRootError(Phase8LabelEvaluationError):
    """Raised when the output root already exists or fails path safety."""


class Phase8LabelEvaluationPublicationError(Phase8LabelEvaluationError):
    """Raised when a Package G artifact cannot be published without overwriting."""


# ---------------------------------------------------------------------------
# A. Frozen-identity verification (freeze, preregistration, prediction lock).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8LabelEvaluationIdentities:
    """Verified freeze/preregistration/prediction-lock identities for this evaluation run."""

    freeze_artifact_sha256: str
    preregistration_hash: str
    lock: Phase8ExternalPredictionLock


def verify_phase8_label_evaluation_identities(
    *,
    freeze_path: Path,
    preregistration_path: Path,
    prediction_lock_path: Path,
) -> Phase8LabelEvaluationIdentities:
    """Verify the freeze artifact, preregistration, prediction lock.

    Fails closed on any mismatch.
    """

    for path, name in (
        (freeze_path, "freeze_path"),
        (preregistration_path, "preregistration_path"),
        (prediction_lock_path, "prediction_lock_path"),
    ):
        if not path.is_absolute():
            raise Phase8LabelEvaluationIdentityError(f"{name} must be an absolute path.")
        if path.is_symlink():
            raise Phase8LabelEvaluationIdentityError(f"{name} must not be a symlink.")
        if not path.is_file():
            raise Phase8LabelEvaluationIdentityError(f"{name} must be a regular file.")

    freeze_sha256 = sha256_file(freeze_path)
    if freeze_sha256 != REQUIRED_FREEZE_ARTIFACT_SHA256:
        raise Phase8LabelEvaluationIdentityError(
            "freeze artifact SHA-256 does not match the required frozen identity."
        )
    freeze = phase8_decision_freeze_from_json(freeze_path.read_bytes())
    if not freeze.frozen_before_external_evaluation or freeze.freeze_state != "frozen":
        raise Phase8LabelEvaluationIdentityError(
            "freeze artifact is not in a frozen, evaluation-ready state."
        )

    preregistration = phase8_external_preregistration_from_json(preregistration_path.read_bytes())
    if preregistration.preregistration_hash != REQUIRED_PREREGISTRATION_HASH:
        raise Phase8LabelEvaluationIdentityError(
            "preregistration_hash does not match the required frozen identity."
        )
    if not preregistration.no_tuning_declaration:
        raise Phase8LabelEvaluationIdentityError(
            "preregistration must declare no_tuning_declaration=true."
        )

    lock = _load_and_validate_prediction_lock(prediction_lock_path)
    if lock.lock_hash != REQUIRED_LOCK_HASH:
        raise Phase8LabelEvaluationIdentityError(
            "prediction lock_hash does not match the required frozen identity."
        )
    if lock.freeze_artifact_sha256 != REQUIRED_FREEZE_ARTIFACT_SHA256:
        raise Phase8LabelEvaluationIdentityError("lock freeze_artifact_sha256 mismatch.")
    if lock.preregistration_hash != REQUIRED_PREREGISTRATION_HASH:
        raise Phase8LabelEvaluationIdentityError("lock preregistration_hash mismatch.")
    if lock.checkpoint_sha256 != REQUIRED_CHECKPOINT_SHA256:
        raise Phase8LabelEvaluationIdentityError("lock checkpoint_sha256 mismatch.")
    if lock.definitive_config_hash != REQUIRED_DEFINITIVE_CONFIG_HASH:
        raise Phase8LabelEvaluationIdentityError("lock definitive_config_hash mismatch.")
    if lock.frozen_threshold != REQUIRED_THRESHOLD:
        raise Phase8LabelEvaluationIdentityError("lock frozen_threshold mismatch.")
    if lock.support_policy != REQUIRED_SUPPORT_POLICY:
        raise Phase8LabelEvaluationIdentityError("lock support_policy mismatch.")
    if tuple(lock.target_spacing_xyz_mm) != REQUIRED_TARGET_SPACING_XYZ_MM:
        raise Phase8LabelEvaluationIdentityError("lock target_spacing_xyz_mm mismatch.")
    if lock.cohort_identifier != REQUIRED_COHORT_IDENTIFIER:
        raise Phase8LabelEvaluationIdentityError("lock cohort_identifier mismatch.")
    if lock.external_label_access is not False:
        raise Phase8LabelEvaluationIdentityError(
            "lock must record external_label_access=False (labels were not used to produce it)."
        )
    expected_ids = tuple(
        f"ext-ircadb-{ordinal:03d}" for ordinal in range(1, PHASE8_IRCADB_CASE_COUNT + 1)
    )
    if lock.ordered_case_ids != expected_ids:
        raise Phase8LabelEvaluationIdentityError("lock ordered_case_ids does not cover 20 cases.")

    return Phase8LabelEvaluationIdentities(
        freeze_artifact_sha256=freeze_sha256,
        preregistration_hash=preregistration.preregistration_hash,
        lock=lock,
    )


def _load_and_validate_prediction_lock(path: Path) -> Phase8ExternalPredictionLock:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return Phase8ExternalPredictionLock(
            schema_name=payload["schema_name"],
            schema_version=payload["schema_version"],
            cohort_identifier=payload["cohort_identifier"],
            preregistration_hash=payload["preregistration_hash"],
            freeze_artifact_sha256=payload["freeze_artifact_sha256"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            definitive_config_hash=payload["definitive_config_hash"],
            frozen_threshold=payload["frozen_threshold"],
            support_policy=payload["support_policy"],
            target_spacing_xyz_mm=tuple(payload["target_spacing_xyz_mm"]),
            observed_image_case_count=payload["observed_image_case_count"],
            ordered_case_ids=tuple(payload["ordered_case_ids"]),
            case_record_hashes=tuple(payload["case_record_hashes"]),
            prediction_sha256_by_case=tuple(
                tuple(item) for item in payload["prediction_sha256_by_case"]
            ),
            inference_completion_state=payload["inference_completion_state"],
            external_label_access=payload["external_label_access"],
            no_tuning=payload["no_tuning"],
            git_commit=payload["git_commit"],
            lock_hash=payload["lock_hash"],
        )
    except (KeyError, Phase8ExternalInferenceError) as exc:
        raise Phase8LabelEvaluationIdentityError(
            f"prediction lock at {path} failed hash/schema validation: {exc}"
        ) from exc


def _load_and_validate_prediction_record(path: Path) -> Phase8ExternalPredictionRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return Phase8ExternalPredictionRecord(
            schema_name=payload["schema_name"],
            schema_version=payload["schema_version"],
            anonymous_case_id=payload["anonymous_case_id"],
            source_case_ordinal=payload["source_case_ordinal"],
            prediction_file_relative_path=payload["prediction_file_relative_path"],
            prediction_sha256=payload["prediction_sha256"],
            prediction_shape_zyx=tuple(payload["prediction_shape_zyx"]),
            prediction_dtype=payload["prediction_dtype"],
            binary_semantics=payload["binary_semantics"],
            threshold=payload["threshold"],
            source_image_archive_sha256=payload["source_image_archive_sha256"],
            original_volume_shape_zyx=tuple(payload["original_volume_shape_zyx"]),
            original_voxel_spacing_row_col_slice_mm=tuple(
                payload["original_voxel_spacing_row_col_slice_mm"]
            ),
            ras_reoriented_shape_zyx=tuple(payload["ras_reoriented_shape_zyx"]),
            ras_reoriented_affine_mm=tuple(
                tuple(row) for row in payload["ras_reoriented_affine_mm"]
            ),
            resampled_shape_zyx=tuple(payload["resampled_shape_zyx"]),
            resampled_affine_mm=tuple(tuple(row) for row in payload["resampled_affine_mm"]),
            target_spacing_xyz_mm=tuple(payload["target_spacing_xyz_mm"]),
            preprocessing_orientation=payload["preprocessing_orientation"],
            preprocessing_image_interpolation=payload["preprocessing_image_interpolation"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            definitive_config_hash=payload["definitive_config_hash"],
            freeze_artifact_sha256=payload["freeze_artifact_sha256"],
            preregistration_hash=payload["preregistration_hash"],
            support_policy=payload["support_policy"],
            external_label_access=payload["external_label_access"],
            record_hash=payload["record_hash"],
        )
    except (KeyError, Phase8ExternalInferenceError) as exc:
        raise Phase8LabelEvaluationIdentityError(
            f"prediction record at {path} failed hash/schema validation: {exc}"
        ) from exc


def verify_locked_prediction_array(
    *,
    predictions_dir: Path,
    anonymous_case_id: str,
    lock: Phase8ExternalPredictionLock,
) -> tuple[np.ndarray, Phase8ExternalPredictionRecord]:
    """Re-verify one locked prediction's SHA-256 and load it read-only. Never writes to it."""

    prediction_path = predictions_dir / f"{anonymous_case_id}.npy"
    record_path = predictions_dir / f"{anonymous_case_id}_prediction_record.json"
    if prediction_path.is_symlink() or record_path.is_symlink():
        raise Phase8LabelEvaluationIdentityError(
            f"case {anonymous_case_id}: prediction artifacts must not be symlinks."
        )
    expected_sha256 = dict(lock.prediction_sha256_by_case).get(anonymous_case_id)
    if expected_sha256 is None:
        raise Phase8LabelEvaluationIdentityError(
            f"case {anonymous_case_id}: not present in the prediction lock."
        )
    observed_sha256 = sha256_file(prediction_path)
    if observed_sha256 != expected_sha256:
        raise Phase8LabelEvaluationIdentityError(
            f"case {anonymous_case_id}: prediction file SHA-256 no longer matches the lock "
            "(fail-closed; a locked prediction must never change)."
        )
    record = _load_and_validate_prediction_record(record_path)
    if record.prediction_sha256 != expected_sha256:
        raise Phase8LabelEvaluationIdentityError(
            f"case {anonymous_case_id}: prediction record sha256 does not match the lock."
        )
    prediction_array = np.load(prediction_path, allow_pickle=False)
    if prediction_array.dtype != np.uint8:
        raise Phase8LabelEvaluationIdentityError(
            f"case {anonymous_case_id}: locked prediction must be uint8."
        )
    return prediction_array, record


# ---------------------------------------------------------------------------
# B. Real-label geometry alignment (the only part of this module that opens labels).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _MaskFolderVolume:
    role_folder: str
    volume: np.ndarray
    rows: int
    columns: int
    slice_count: int
    voxel_spacing_row_col_slice_mm: tuple[float, float, float]
    affine_lps_mm: np.ndarray


def discover_livertumor_zip_members(masks_zip: Path) -> dict[str, list[str]]:
    """Return ``{role_folder_name: [member_paths]}`` for ``livertumor*``-prefixed folders only.

    Opens only the ZIP central directory (no member content) of ``masks_zip``. Any member whose
    role-folder name (after stripping an optional leading ``MASKS_DICOM`` path component) does not
    case-insensitively start with ``livertumor`` is ignored -- this specifically excludes the
    ``liver`` (context-only) folder and any other organ folder, per the preregistered policy's
    minimal label-access requirement.
    """

    groups: dict[str, list[str]] = {}
    with zipfile.ZipFile(masks_zip, "r") as archive:
        for info in archive.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            try:
                normalized = normalize_safe_ircadb_zip_member_path(name)
            except ValueError:
                continue
            if is_ignored_macos_metadata_path(normalized):
                continue
            parts = list(PurePosixPath(normalized).parts)
            if not parts:
                continue
            if parts[0].upper() == "MASKS_DICOM":
                parts = parts[1:]
            if not parts:
                continue
            role_folder = parts[0]
            if not role_folder.lower().startswith(_LIVERTUMOR_PREFIX):
                continue
            groups.setdefault(role_folder, []).append(normalized)
    return groups


def _mask_pixel_array(item: Phase8DicomSlice) -> np.ndarray:
    """Decode one 8-bit unsigned mask slice. Value != 0 is foreground (source_value_policy)."""

    if item.bits_allocated != 8:
        raise Phase8LabelEvaluationGeometryError(
            f"unsupported mask BitsAllocated={item.bits_allocated} (expected 8)."
        )
    expected_values = item.rows * item.columns
    if len(item.pixel_data) < expected_values:
        raise Phase8LabelEvaluationGeometryError("mask pixel data is shorter than rows*columns.")
    array = np.frombuffer(item.pixel_data[:expected_values], dtype=np.dtype("u1"))
    return array.reshape((item.rows, item.columns))


def _require_folder_header_consistency(slices: tuple[Phase8DicomSlice, ...]) -> None:
    if len({(item.rows, item.columns) for item in slices}) != 1:
        raise Phase8LabelEvaluationGeometryError("inconsistent mask rows/columns within a folder.")
    spacings = {tuple(round(v, 6) for v in item.pixel_spacing) for item in slices}
    if len(spacings) != 1:
        raise Phase8LabelEvaluationGeometryError("inconsistent mask pixel spacing within a folder.")
    orientations = {
        tuple(round(v, 6) for v in item.image_orientation_patient)
        for item in slices
        if item.image_orientation_patient is not None
    }
    if len(orientations) != 1:
        raise Phase8LabelEvaluationGeometryError(
            "inconsistent or missing mask orientation within a folder."
        )


def _build_mask_folder_volume(
    archive: zipfile.ZipFile,
    role_folder: str,
    member_paths: Sequence[str],
) -> _MaskFolderVolume:
    try:
        slices = tuple(_read_dicom_slice(archive.read(member), member) for member in member_paths)
    except Phase8ImageQaError as exc:
        raise Phase8LabelEvaluationGeometryError(
            f"mask folder {role_folder!r} could not be parsed: {exc}"
        ) from exc
    if not slices:
        raise Phase8LabelEvaluationGeometryError(f"mask folder {role_folder!r} has no members.")
    _require_folder_header_consistency(slices)
    ordered = tuple(sorted(slices, key=_slice_sort_key))
    voxel_spacing = _voxel_spacing(ordered)
    if voxel_spacing is None:
        raise Phase8LabelEvaluationGeometryError(
            f"mask folder {role_folder!r} slice spacing could not be determined."
        )
    try:
        affine = _build_lps_affine(ordered, voxel_spacing=voxel_spacing)
    except Phase8ImageVolumeLoadError as exc:
        raise Phase8LabelEvaluationGeometryError(
            f"mask folder {role_folder!r} affine could not be built: {exc}"
        ) from exc
    volume = np.stack([_mask_pixel_array(item) for item in ordered], axis=0)
    return _MaskFolderVolume(
        role_folder=role_folder,
        volume=volume,
        rows=ordered[0].rows,
        columns=ordered[0].columns,
        slice_count=volume.shape[0],
        voxel_spacing_row_col_slice_mm=voxel_spacing,
        affine_lps_mm=affine,
    )


def load_and_align_case_tumor_label(
    *,
    masks_zip: Path,
    prediction_record: Phase8ExternalPredictionRecord,
) -> tuple[str, tuple[str, ...], np.ndarray | None]:
    """Load, union, and align one case's real tumor label onto the locked prediction grid.

    Returns ``(status, reason_codes, aligned_boolean_label)``. ``aligned_boolean_label`` is
    ``None`` unless ``status == "compatible"``. Opens only ``masks_zip`` (``MASKS_DICOM.zip``),
    and only ``livertumor*``-prefixed folders within it.
    """

    if masks_zip.is_symlink():
        return "incompatible", ("label_unreadable",), None
    if not masks_zip.is_file():
        return "incompatible", ("label_archive_missing",), None

    try:
        groups = discover_livertumor_zip_members(masks_zip)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return "incompatible", ("label_unreadable",), None

    policy = build_default_phase8_label_mapping_policy()
    if not groups:
        with contextlib.suppress(Phase8LabelMappingValidationError):
            apply_phase8_label_mapping_policy(policy, {})
        return "incompatible", ("tumor_target_absent",), None

    try:
        with zipfile.ZipFile(masks_zip, "r") as archive:
            folder_volumes = tuple(
                _build_mask_folder_volume(archive, name, sorted(members))
                for name, members in sorted(groups.items())
            )
    except (Phase8LabelEvaluationGeometryError, OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return "incompatible", ("label_geometry_incompatible",), None

    shapes = {(fv.slice_count, fv.rows, fv.columns) for fv in folder_volumes}
    spacings = {
        tuple(round(v, 4) for v in fv.voxel_spacing_row_col_slice_mm) for fv in folder_volumes
    }
    affines = {tuple(np.round(fv.affine_lps_mm, 4).flatten().tolist()) for fv in folder_volumes}
    if len(shapes) != 1 or len(spacings) != 1 or len(affines) != 1:
        return "incompatible", ("label_geometry_incompatible",), None

    (only_shape,) = shapes
    if only_shape != tuple(prediction_record.original_volume_shape_zyx):
        return "incompatible", ("label_geometry_incompatible",), None
    (only_spacing,) = spacings
    expected_spacing = prediction_record.original_voxel_spacing_row_col_slice_mm
    if any(
        abs(observed - round(expected, 4)) > 1e-2
        for observed, expected in zip(only_spacing, expected_spacing, strict=True)
    ):
        return "incompatible", ("label_geometry_incompatible",), None

    union = folder_volumes[0].volume.astype(np.uint8, copy=True)
    for extra in folder_volumes[1:]:
        union = np.logical_or(union, extra.volume != 0).astype(np.uint8)

    try:
        tumor_mask_bool = apply_phase8_label_mapping_policy(policy, {"tumor_lesion_mask": union})
    except Phase8LabelMappingValidationError:
        return "incompatible", ("non_binary_target_after_mapping",), None

    affine_ras = _lps_affine_to_ras(folder_volumes[0].affine_lps_mm)
    reoriented, reoriented_affine = pipeline.reorient_volume_to_ras(
        tumor_mask_bool.astype(np.uint8), affine_ras
    )
    if tuple(int(v) for v in reoriented.shape) != tuple(prediction_record.ras_reoriented_shape_zyx):
        return "incompatible", ("label_geometry_incompatible",), None

    reoriented_spacing = _spacing_from_affine(reoriented_affine)
    resampled = pipeline.resample_volume_to_spacing(
        reoriented,
        source_spacing=reoriented_spacing,
        target_spacing=REQUIRED_TARGET_SPACING_XYZ_MM,
        is_label=True,
    )
    if tuple(int(v) for v in resampled.shape) != tuple(prediction_record.resampled_shape_zyx):
        raise Phase8LabelEvaluationImplementationError(
            "resampled label shape does not match the locked prediction record's "
            "resampled_shape_zyx; this indicates a geometry-chain defect in this module, "
            "not a data issue. Halting."
        )
    final_mask = resampled != 0
    return "compatible", ("compatible_binary_tumor_target",), final_mask


# ---------------------------------------------------------------------------
# C. Metric-scalar extraction and case-level bootstrap.
# ---------------------------------------------------------------------------


def _tumor_volume_error_signed(case: BaselineCaseMetrics) -> float | None:
    return case.signed_volume_error_ml


def _tumor_volume_error_absolute(case: BaselineCaseMetrics) -> float | None:
    return case.absolute_volume_error_ml


_PHASE8_METRIC_SCALAR_EXTRACTORS.update(
    {
        "tumor_dice": lambda case: case.tumor_dice,
        "tumor_iou": lambda case: case.tumor_iou,
        "tumor_hd95": lambda case: case.hd95_mm,
        "tumor_normalized_surface_dice": lambda case: case.normalized_surface_dice,
        "lesion_wise_recall": lambda case: case.lesion_recall,
        "lesion_wise_precision": lambda case: case.lesion_precision,
        "lesion_f1": lambda case: case.lesion_f1,
        "false_positive_lesions_per_scan": lambda case: float(case.false_positive_lesions_per_scan),
        "tumor_volume_error_signed_ml": _tumor_volume_error_signed,
        "tumor_volume_error_absolute_ml": _tumor_volume_error_absolute,
        "tumor_volume_error_relative": lambda case: case.relative_volume_error,
    }
)


def bootstrap_case_level_confidence_intervals(
    case_records: Sequence[BaselineCaseMetrics],
    *,
    seed: int = _BOOTSTRAP_SEED,
    resample_count: int = _BOOTSTRAP_RESAMPLE_COUNT,
    confidence_level: float = _BOOTSTRAP_CONFIDENCE,
) -> dict[str, dict[str, JsonValue]]:
    """Case-level (patient-level) percentile bootstrap over ``case_records``.

    Uses exactly one :func:`numpy.random.default_rng` stream, seeded once, shared across every
    metric (case-index resamples are drawn once and reused per metric). Undefined per-case metric
    values are excluded from a replicate's mean; a replicate with zero defined values contributes
    nothing, and if every replicate is undefined for a metric the CI is reported unavailable.
    """

    case_count = len(case_records)
    rng = np.random.default_rng(seed)
    resample_indices = (
        rng.integers(0, case_count, size=(resample_count, case_count)) if case_count > 0 else None
    )
    lower_pct = (1.0 - confidence_level) / 2.0 * 100.0
    upper_pct = (1.0 + confidence_level) / 2.0 * 100.0

    results: dict[str, dict[str, JsonValue]] = {}
    for metric_name, extractor in _PHASE8_METRIC_SCALAR_EXTRACTORS.items():
        raw_values = [extractor(case) for case in case_records]
        defined_count = sum(1 for value in raw_values if value is not None)
        if case_count == 0 or resample_indices is None:
            results[metric_name] = {
                "point_estimate": None,
                "defined_case_count": 0,
                "case_count": 0,
                "ci_available": False,
                "ci_low": None,
                "ci_high": None,
                "defined_replicate_count": 0,
                "resample_count": resample_count,
                "confidence_level": confidence_level,
            }
            continue
        array = np.array(
            [value if value is not None else np.nan for value in raw_values], dtype=np.float64
        )
        replicate_means: list[float] = []
        for row in resample_indices:
            sample = array[row]
            defined = sample[~np.isnan(sample)]
            if defined.size == 0:
                continue
            replicate_means.append(float(np.mean(defined)))
        defined_point = array[~np.isnan(array)]
        point_estimate = float(np.mean(defined_point)) if defined_point.size > 0 else None
        if replicate_means:
            ci_low = float(np.percentile(replicate_means, lower_pct))
            ci_high = float(np.percentile(replicate_means, upper_pct))
            ci_available = True
        else:
            ci_low = None
            ci_high = None
            ci_available = False
        results[metric_name] = {
            "point_estimate": point_estimate,
            "defined_case_count": defined_count,
            "case_count": case_count,
            "ci_available": ci_available,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "defined_replicate_count": len(replicate_means),
            "resample_count": resample_count,
            "confidence_level": confidence_level,
        }
    return results


# ---------------------------------------------------------------------------
# D. Publication helpers (no-overwrite).
# ---------------------------------------------------------------------------


def _publish_json(path: Path, payload: Mapping[str, JsonValue] | JsonValue) -> None:
    try:
        publish_text_no_overwrite(
            text=(canonical_json_bytes(payload) + b"\n").decode("utf-8"),
            output_path=path,
            temporary_exists_message="temporary Phase 8 Package G artifact already exists.",
            final_exists_message="Phase 8 Package G artifact already exists.",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8LabelEvaluationPublicationError(str(exc)) from exc


def _validate_output_root(output_root: Path, *, dataset_root: Path, repository_root: Path) -> Path:
    if output_root.exists():
        raise Phase8LabelEvaluationOutputRootError(
            f"output_root already exists; refusing to reuse or silently version: {output_root}"
        )
    try:
        return validate_explicit_external_output_root(
            output_root,
            forbidden_roots=(dataset_root, repository_root),
        )
    except InvalidDatasetRootError as exc:
        raise Phase8LabelEvaluationOutputRootError(str(exc)) from exc


# ---------------------------------------------------------------------------
# E. Top-level driver.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8LabelEvaluationRunResult:
    """Aggregate result of one definitive Package G external label evaluation run."""

    output_root: Path
    identities: Phase8LabelEvaluationIdentities
    eligibility_accounting: Phase8EligibilityAccounting
    metric_report: BaselineMetricReport
    bootstrap: dict[str, dict[str, JsonValue]]
    domain_shift_record: Phase8DomainShiftRecord
    excluded_case_reasons: tuple[tuple[str, tuple[str, ...]], ...]


def run_phase8_external_label_evaluation(
    *,
    dataset_root: Path,
    freeze_path: Path,
    preregistration_path: Path,
    prediction_lock_path: Path,
    predictions_dir: Path,
    image_manifest_path: Path,
    prelabel_accounting_path: Path,
    output_root: Path,
    repository_root: Path,
) -> Phase8LabelEvaluationRunResult:
    """Run the complete, once-only Package G external label evaluation and publish its artifacts.

    Opens real 3D-IRCADb-01 tumor labels (``MASKS_DICOM.zip`` / ``livertumor*`` folders only) to
    evaluate the already-frozen, already-locked predictions in ``predictions_dir``. Never reruns
    inference, never writes to ``predictions_dir``, and never excludes a case for a performance
    reason -- only via the preregistered fail-closed contracts (missing tumor source, geometry
    incompatibility).
    """

    if output_root.exists():
        raise Phase8LabelEvaluationOutputRootError(
            f"output_root already exists; refusing to reuse or silently version: {output_root}"
        )
    identities = verify_phase8_label_evaluation_identities(
        freeze_path=freeze_path,
        preregistration_path=preregistration_path,
        prediction_lock_path=prediction_lock_path,
    )
    canonical_output_root = _validate_output_root(
        output_root, dataset_root=dataset_root, repository_root=repository_root
    )

    image_manifest = phase8_external_image_manifest_from_json(image_manifest_path.read_bytes())
    prelabel_accounting = phase8_eligibility_accounting_from_json(
        prelabel_accounting_path.read_bytes()
    )
    if prelabel_accounting.accounting_state != "pre_label_access_pending":
        raise Phase8LabelEvaluationIdentityError(
            "pre-label eligibility accounting must be in pre_label_access_pending state."
        )
    if prelabel_accounting.external_image_manifest_hash != image_manifest.manifest_hash:
        raise Phase8LabelEvaluationIdentityError(
            "pre-label accounting does not reference the loaded image manifest."
        )

    inventory = discover_phase8_ircadb_image_layout(dataset_root)
    canonical_dataset_root = dataset_root.resolve(strict=True)
    prelabel_by_id = {case.anonymous_case_id: case for case in prelabel_accounting.cases}

    case_records: list[BaselineCaseMetrics] = []
    updated_eligibility_cases: list[Phase8EligibilityCase] = []
    label_ledger_entries: list[dict[str, JsonValue]] = []
    qualitative_entries: list[dict[str, JsonValue]] = []
    excluded_reasons: list[tuple[str, tuple[str, ...]]] = []

    for ircadb_case in inventory.cases:
        anonymous_case_id = ircadb_case.anonymous_case_id
        prelabel_case = prelabel_by_id[anonymous_case_id]

        if not prelabel_case.inference_eligible:
            updated_eligibility_cases.append(
                build_phase8_label_compatible_case(
                    prelabel_case,
                    compatible=False,
                    label_compatibility_reason_codes=("pending_label_access",),
                )
            )
            continue

        prediction_array, prediction_record = verify_locked_prediction_array(
            predictions_dir=predictions_dir,
            anonymous_case_id=anonymous_case_id,
            lock=identities.lock,
        )
        masks_zip = canonical_dataset_root / ircadb_case.source_case_directory / _MASKS_ARCHIVE_NAME
        status, reason_codes, aligned_label = load_and_align_case_tumor_label(
            masks_zip=masks_zip,
            prediction_record=prediction_record,
        )
        label_ledger_entries.append(
            {
                "anonymous_case_id": anonymous_case_id,
                "label_compatibility_status": status,
                "label_compatibility_reason_codes": list(reason_codes),
            }
        )
        updated_eligibility_cases.append(
            build_phase8_label_compatible_case(
                prelabel_case,
                compatible=(status == "compatible"),
                label_compatibility_reason_codes=reason_codes,
            )
        )
        if status != "compatible" or aligned_label is None:
            excluded_reasons.append((anonymous_case_id, reason_codes))
            continue
        if aligned_label.shape != prediction_array.shape:
            raise Phase8LabelEvaluationImplementationError(
                f"case {anonymous_case_id}: aligned label shape {aligned_label.shape} does not "
                f"match locked prediction shape {prediction_array.shape}."
            )
        case_metrics = compute_baseline_case_metrics(
            case_identifier=anonymous_case_id,
            ground_truth_mask=aligned_label,
            prediction_mask=prediction_array != 0,
            voxel_spacing_mm=REQUIRED_TARGET_SPACING_XYZ_MM,
            nsd_tolerance_mm=1.0,
        )
        case_records.append(case_metrics)
        qualitative_entries.append(
            {
                "anonymous_case_id": anonymous_case_id,
                "prediction_sha256": prediction_record.prediction_sha256,
                "aligned_label_provenance_hash": sha256_json(
                    {
                        "anonymous_case_id": anonymous_case_id,
                        "label_shape_zyx": list(int(v) for v in aligned_label.shape),
                        "prediction_record_hash": prediction_record.record_hash,
                    }
                ),
                "case_metrics_artifact_hash": case_metrics.artifact_hash,
            }
        )

    label_access_ledger_hash = sha256_json(
        sorted(label_ledger_entries, key=lambda item: cast(str, item["anonymous_case_id"]))
    )
    post_label_accounting = build_phase8_post_label_eligibility_accounting(
        previous_accounting=prelabel_accounting,
        label_access_ledger_hash=label_access_ledger_hash,
        compatibility_updates=tuple(updated_eligibility_cases),
    )

    case_records_sorted = tuple(sorted(case_records, key=lambda item: item.case_identifier))
    eligible_ids_sorted = tuple(sorted(item.case_identifier for item in case_records_sorted))
    development_split_sha256 = sha256_json(list(eligible_ids_sorted))
    metric_report = build_baseline_metric_report(
        baseline_family="monai_segresnet",
        run_identifier="phase8_external_evaluation_v1",
        dataset_manifest_sha256=image_manifest.manifest_hash,
        development_split_sha256=development_split_sha256,
        prediction_manifest_sha256=identities.lock.lock_hash,
        metric_config_sha256=phase8_statistical_policy_component_hash("metric_configuration"),
        nsd_tolerance_mm=1.0,
        case_records=case_records_sorted,
    )
    bootstrap = bootstrap_case_level_confidence_intervals(case_records_sorted)
    domain_shift_record = build_phase8_domain_shift_record(external_manifest=image_manifest)

    internal_comparison = _build_internal_external_comparison(bootstrap=bootstrap)

    _publish_json(
        canonical_output_root / "phase8_external_eligibility_accounting.json",
        phase8_eligibility_accounting_to_dict(post_label_accounting),
    )
    for case_metrics in case_records_sorted:
        _publish_json(
            canonical_output_root / "case_metrics" / f"{case_metrics.case_identifier}.json",
            json.loads(baseline_case_metrics_to_json(case_metrics).decode("utf-8")),
        )
    _publish_json(
        canonical_output_root / "phase8_external_case_metrics.json",
        {
            "schema_name": "phase8_external_case_metrics_collection",
            "schema_version": "v1",
            "case_count": len(case_records_sorted),
            "cases": [
                json.loads(baseline_case_metrics_to_json(case).decode("utf-8"))
                for case in case_records_sorted
            ],
        },
    )
    _publish_json(
        canonical_output_root / "phase8_external_metric_report.json",
        json.loads(baseline_metric_report_to_json(metric_report).decode("utf-8")),
    )
    _publish_json(
        canonical_output_root / "phase8_external_bootstrap_ci.json",
        {
            "schema_name": "phase8_external_bootstrap_ci",
            "schema_version": "v1",
            "random_seed": _BOOTSTRAP_SEED,
            "resample_count": _BOOTSTRAP_RESAMPLE_COUNT,
            "confidence_level": _BOOTSTRAP_CONFIDENCE,
            "interval_method": "percentile",
            "resampling_unit": "case_patient",
            "metrics": cast(JsonValue, bootstrap),
        },
    )
    _publish_json(
        canonical_output_root / "phase8_external_domain_shift_record.json",
        phase8_domain_shift_record_to_dict(domain_shift_record),
    )
    sorted_qualitative_entries: list[dict[str, JsonValue]] = sorted(
        qualitative_entries, key=lambda item: cast(str, item["anonymous_case_id"])
    )
    _publish_json(
        canonical_output_root / "phase8_external_qualitative_index.json",
        {
            "schema_name": "phase8_external_qualitative_index",
            "schema_version": "v1",
            "selection_rule": "include_all_evaluation_eligible_cases_when_count_at_most_20",
            "tie_breaking": "anonymous_case_id_lexicographic",
            "case_count": len(qualitative_entries),
            "cases": cast(JsonValue, sorted_qualitative_entries),
        },
    )
    _publish_json(
        canonical_output_root / "phase8_internal_external_comparison.json",
        internal_comparison,
    )
    summary = _build_summary(
        identities=identities,
        image_manifest=image_manifest,
        post_label_accounting=post_label_accounting,
        metric_report=metric_report,
        bootstrap=bootstrap,
        domain_shift_record=domain_shift_record,
        excluded_reasons=tuple(excluded_reasons),
    )
    _publish_json(canonical_output_root / "phase8_external_evaluation_summary.json", summary)

    return Phase8LabelEvaluationRunResult(
        output_root=canonical_output_root,
        identities=identities,
        eligibility_accounting=post_label_accounting,
        metric_report=metric_report,
        bootstrap=bootstrap,
        domain_shift_record=domain_shift_record,
        excluded_case_reasons=tuple(excluded_reasons),
    )


def _build_internal_external_comparison(
    *, bootstrap: Mapping[str, Mapping[str, JsonValue]]
) -> dict[str, JsonValue]:
    internal_evidence_path = _INTERNAL_CHECKPOINT_EVIDENCE_PATH
    internal_tumor_dice: float | None = None
    internal_checkpoint_sha256: str | None = None
    if internal_evidence_path.is_file():
        evidence = json.loads(internal_evidence_path.read_text(encoding="utf-8"))
        internal_tumor_dice = evidence.get("mean_tumor_dice_step_500")
        # `phase8_definitive_checkpoint_selection_evidence` (see
        # protoem_ct.external.definitive_pipeline / definitive_training) publishes the selected
        # checkpoint hash under the key "selected_checkpoint_hash", not "checkpoint_sha256". The
        # latter key does not exist in that schema, so the previous lookup always returned None.
        internal_checkpoint_sha256 = evidence.get("selected_checkpoint_hash")

    tumor_dice_bootstrap = bootstrap["tumor_dice"]
    comparisons: dict[str, JsonValue] = {}
    comparisons["tumor_dice"] = {
        "internal_value": internal_tumor_dice,
        "internal_evidence_status": (
            "single_frozen_scalar_no_ci" if internal_tumor_dice is not None else "not_available"
        ),
        "internal_ci_available": False,
        "internal_checkpoint_sha256": internal_checkpoint_sha256,
        "internal_checkpoint_matches_locked_checkpoint": (
            internal_checkpoint_sha256 == REQUIRED_CHECKPOINT_SHA256
        ),
        "internal_split": "pooled_internal_development_source_lits_and_msd_task03_liver",
        "external_point_estimate": tumor_dice_bootstrap["point_estimate"],
        "external_ci_low": tumor_dice_bootstrap["ci_low"],
        "external_ci_high": tumor_dice_bootstrap["ci_high"],
        "external_ci_available": tumor_dice_bootstrap["ci_available"],
    }
    for metric_name in (
        "tumor_iou",
        "tumor_hd95",
        "tumor_normalized_surface_dice",
        "lesion_wise_recall",
        "lesion_wise_precision",
        "lesion_f1",
        "false_positive_lesions_per_scan",
        "tumor_volume_error_signed_ml",
    ):
        stats = bootstrap[metric_name]
        comparisons[metric_name] = {
            "internal_value": None,
            "internal_evidence_status": "not_available",
            "internal_ci_available": False,
            "external_point_estimate": stats["point_estimate"],
            "external_ci_low": stats["ci_low"],
            "external_ci_high": stats["ci_high"],
            "external_ci_available": stats["ci_available"],
        }
    return {
        "schema_name": "phase8_internal_external_comparison",
        "schema_version": "v1",
        "cohort_relationship": "independent_unpaired",
        "claims_policy": "descriptive_only_no_superiority_or_generalization_claims",
        "comparisons": comparisons,
    }


def _build_summary(
    *,
    identities: Phase8LabelEvaluationIdentities,
    image_manifest: Phase8ExternalImageManifest,
    post_label_accounting: Phase8EligibilityAccounting,
    metric_report: BaselineMetricReport,
    bootstrap: Mapping[str, Mapping[str, JsonValue]],
    domain_shift_record: Phase8DomainShiftRecord,
    excluded_reasons: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, JsonValue]:
    return {
        "schema_name": PHASE8_LABEL_EVALUATION_SCHEMA_NAME,
        "schema_version": PHASE8_LABEL_EVALUATION_SCHEMA_VERSION,
        "freeze_artifact_sha256": identities.freeze_artifact_sha256,
        "preregistration_hash": identities.preregistration_hash,
        "prediction_lock_hash": identities.lock.lock_hash,
        "checkpoint_sha256": identities.lock.checkpoint_sha256,
        "definitive_config_hash": identities.lock.definitive_config_hash,
        "cohort_identifier": identities.lock.cohort_identifier,
        "image_manifest_hash": image_manifest.manifest_hash,
        "case_count": post_label_accounting.case_count,
        "evaluation_eligible_count": post_label_accounting.evaluation_eligible_count,
        "excluded_count": post_label_accounting.excluded_count,
        "excluded_case_reasons": [
            {"anonymous_case_id": case_id, "reason_codes": list(reasons)}
            for case_id, reasons in sorted(excluded_reasons, key=lambda item: item[0])
        ],
        "metric_report_artifact_hash": metric_report.artifact_hash,
        "aggregate": {
            "tumor_dice": {
                "pooled": metric_report.aggregate.pooled_dice if metric_report.aggregate else None,
                "macro": metric_report.aggregate.macro_dice if metric_report.aggregate else None,
                "bootstrap_ci_low": bootstrap["tumor_dice"]["ci_low"],
                "bootstrap_ci_high": bootstrap["tumor_dice"]["ci_high"],
            },
        },
        "bootstrap_configuration": {
            "random_seed": _BOOTSTRAP_SEED,
            "resample_count": _BOOTSTRAP_RESAMPLE_COUNT,
            "confidence_level": _BOOTSTRAP_CONFIDENCE,
            "interval_method": "percentile",
            "resampling_unit": "case_patient",
        },
        "domain_shift_record_hash": domain_shift_record.domain_shift_record_hash,
        "no_inference_rerun": True,
        "no_prediction_modified": True,
        "no_performance_based_exclusion": True,
    }


__all__ = [
    "PHASE8_LABEL_EVALUATION_SCHEMA_NAME",
    "PHASE8_LABEL_EVALUATION_SCHEMA_VERSION",
    "REQUIRED_LOCK_HASH",
    "Phase8LabelEvaluationError",
    "Phase8LabelEvaluationGeometryError",
    "Phase8LabelEvaluationIdentities",
    "Phase8LabelEvaluationIdentityError",
    "Phase8LabelEvaluationImplementationError",
    "Phase8LabelEvaluationOutputRootError",
    "Phase8LabelEvaluationPublicationError",
    "Phase8LabelEvaluationRunResult",
    "bootstrap_case_level_confidence_intervals",
    "discover_livertumor_zip_members",
    "load_and_align_case_tumor_label",
    "run_phase8_external_label_evaluation",
    "verify_locked_prediction_array",
    "verify_phase8_label_evaluation_identities",
]
