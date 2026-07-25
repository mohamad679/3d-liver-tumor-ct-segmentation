"""Deterministic Phase 1 evaluation metrics for persisted segmentation masks."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.filebasedimages import ImageFileError

from protoem_ct.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactError,
    ArtifactValidationError,
    EvaluationArtifact,
    EvaluationCaseMetrics,
    InferenceArtifact,
    PreprocessArtifact,
    artifact_from_json,
    artifact_to_json,
    sha256_file,
)
from protoem_ct.artifacts.hashing import HashFileError
from protoem_ct.evaluation.dummy_inference import DUMMY_INFERENCE_METHOD, DUMMY_INFERENCE_STAGE

EVALUATION_STAGE = "evaluate"
AFFINE_ABSOLUTE_TOLERANCE = 1e-5

Array = npt.NDArray[np.generic]
BoolArray = npt.NDArray[np.bool_]
AffineArray = npt.NDArray[np.float64]


class EvaluationError(ValueError):
    """Base error for Phase 1 deterministic evaluation failures."""


class EvaluationInputArtifactError(EvaluationError):
    """Raised when evaluation input artifacts are invalid or inconsistent."""


class EvaluationPathError(EvaluationError):
    """Raised when evaluation input paths are unsafe."""


class EvaluationNiftiInputError(EvaluationError):
    """Raised when a label or prediction NIfTI file is missing or invalid."""


class EvaluationGeometryMismatchError(EvaluationError):
    """Raised when label and prediction geometry differs."""


class EvaluationPredictionHashMismatchError(EvaluationError):
    """Raised when a prediction file hash does not match its inference artifact."""


class EvaluationOutputCollisionError(EvaluationError):
    """Raised when evaluation would overwrite an existing output artifact."""


class EvaluationFailureError(EvaluationError):
    """Raised when evaluation cannot complete after inputs have been accepted."""


@dataclass(frozen=True, slots=True)
class SegmentationConfusionCounts:
    """Integer confusion counts for one binary segmentation case."""

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    ground_truth_foreground_voxels: int
    predicted_foreground_voxels: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.true_positives, "true_positives")
        _require_nonnegative_int(self.false_positives, "false_positives")
        _require_nonnegative_int(self.false_negatives, "false_negatives")
        _require_nonnegative_int(self.true_negatives, "true_negatives")
        _require_nonnegative_int(
            self.ground_truth_foreground_voxels,
            "ground_truth_foreground_voxels",
        )
        _require_nonnegative_int(
            self.predicted_foreground_voxels,
            "predicted_foreground_voxels",
        )
        if self.ground_truth_foreground_voxels != self.true_positives + self.false_negatives:
            msg = "ground_truth_foreground_voxels must equal true_positives + false_negatives"
            raise EvaluationFailureError(msg)
        if self.predicted_foreground_voxels != self.true_positives + self.false_positives:
            msg = "predicted_foreground_voxels must equal true_positives + false_positives"
            raise EvaluationFailureError(msg)


@dataclass(frozen=True, slots=True)
class _EvaluationCasePaths:
    case_id: str
    label_relative_path: PurePosixPath
    prediction_relative_path: PurePosixPath
    label_path: Path
    prediction_path: Path
    expected_prediction_hash: str


def evaluate_predictions(
    preprocess_artifact_path: Path,
    inference_artifact_path: Path,
    *,
    preprocessed_root: Path,
    prediction_root: Path,
    artifact_output_path: Path,
    git_commit: str,
    created_at_utc: str,
) -> EvaluationArtifact:
    """Evaluate persisted dummy predictions against persisted preprocessed labels."""
    _require_nonempty_text(git_commit, "git_commit")
    _require_nonempty_text(created_at_utc, "created_at_utc")

    preprocess_artifact = _load_preprocess_artifact(preprocess_artifact_path)
    inference_artifact = _load_inference_artifact(inference_artifact_path)
    _verify_artifact_linkage(preprocess_artifact, inference_artifact)
    preprocessed_root_resolved = _resolve_existing_directory(
        preprocessed_root,
        "preprocessed_root",
    )
    prediction_root_resolved = _resolve_existing_directory(
        prediction_root,
        "prediction_root",
    )
    artifact_output = _prepare_artifact_output_path(artifact_output_path)

    artifact_temp_path: Path | None = None
    try:
        case_paths = _resolve_case_paths(
            preprocess_artifact,
            inference_artifact,
            preprocessed_root=preprocessed_root_resolved,
            prediction_root=prediction_root_resolved,
        )
        _verify_prediction_hashes(case_paths)
        case_metrics = _evaluate_cases(case_paths)
        artifact = _build_evaluation_artifact(
            preprocess_artifact,
            case_metrics=case_metrics,
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
        artifact_temp_path = _write_artifact_to_temp(artifact, artifact_output)
        _publish_artifact(artifact_temp_path, artifact_output)
        return artifact
    except EvaluationError:
        _cleanup_after_failure(artifact_temp_path)
        raise
    except Exception as exc:
        _cleanup_after_failure(artifact_temp_path)
        msg = f"evaluation failed: {exc}"
        raise EvaluationFailureError(msg) from exc


def compute_binary_confusion_counts(
    ground_truth_mask: BoolArray,
    prediction_mask: BoolArray,
) -> SegmentationConfusionCounts:
    """Compute integer binary segmentation confusion counts from boolean masks."""
    if ground_truth_mask.shape != prediction_mask.shape:
        msg = (
            "label and prediction shapes differ: "
            f"label shape={ground_truth_mask.shape}, prediction shape={prediction_mask.shape}"
        )
        raise EvaluationGeometryMismatchError(msg)

    true_positives = int(np.count_nonzero(np.logical_and(ground_truth_mask, prediction_mask)))
    false_positives = int(
        np.count_nonzero(np.logical_and(np.logical_not(ground_truth_mask), prediction_mask))
    )
    false_negatives = int(
        np.count_nonzero(np.logical_and(ground_truth_mask, np.logical_not(prediction_mask)))
    )
    true_negatives = int(
        np.count_nonzero(
            np.logical_and(np.logical_not(ground_truth_mask), np.logical_not(prediction_mask))
        )
    )
    return SegmentationConfusionCounts(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        ground_truth_foreground_voxels=int(np.count_nonzero(ground_truth_mask)),
        predicted_foreground_voxels=int(np.count_nonzero(prediction_mask)),
    )


def dice_from_counts(counts: SegmentationConfusionCounts) -> float:
    """Compute Dice as 2 * TP / (2 * TP + FP + FN), with empty masks equal to 1."""
    denominator = (2 * counts.true_positives) + counts.false_positives + counts.false_negatives
    if denominator == 0:
        return 1.0
    return (2 * counts.true_positives) / denominator


def iou_from_counts(counts: SegmentationConfusionCounts) -> float:
    """Compute IoU as TP / (TP + FP + FN), with empty masks equal to 1."""
    denominator = counts.true_positives + counts.false_positives + counts.false_negatives
    if denominator == 0:
        return 1.0
    return counts.true_positives / denominator


def compute_case_metrics(
    case_id: str,
    ground_truth_mask: BoolArray,
    prediction_mask: BoolArray,
) -> EvaluationCaseMetrics:
    """Compute deterministic Dice and IoU metrics for one case from boolean masks."""
    counts = compute_binary_confusion_counts(ground_truth_mask, prediction_mask)
    dice = dice_from_counts(counts)
    iou = iou_from_counts(counts)
    _require_unit_metric(dice, "dice")
    _require_unit_metric(iou, "iou")
    return EvaluationCaseMetrics(
        case_id=case_id,
        true_positives=counts.true_positives,
        false_positives=counts.false_positives,
        false_negatives=counts.false_negatives,
        true_negatives=counts.true_negatives,
        ground_truth_foreground_voxels=counts.ground_truth_foreground_voxels,
        predicted_foreground_voxels=counts.predicted_foreground_voxels,
        dice=dice,
        iou=iou,
    )


def _load_preprocess_artifact(preprocess_artifact_path: Path) -> PreprocessArtifact:
    try:
        text = preprocess_artifact_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = "failed to read preprocessing artifact"
        raise EvaluationInputArtifactError(msg) from exc

    try:
        return artifact_from_json(text, PreprocessArtifact)
    except ArtifactValidationError as exc:
        if "artifact path" in str(exc):
            raise _unsafe_artifact_path_error("preprocessing artifact", exc) from exc
        msg = f"invalid preprocessing artifact: {exc}"
        raise EvaluationInputArtifactError(msg) from exc
    except ArtifactError as exc:
        msg = f"invalid preprocessing artifact: {exc}"
        raise EvaluationInputArtifactError(msg) from exc


def _load_inference_artifact(inference_artifact_path: Path) -> InferenceArtifact:
    try:
        text = inference_artifact_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = "failed to read inference artifact"
        raise EvaluationInputArtifactError(msg) from exc

    try:
        return artifact_from_json(text, InferenceArtifact)
    except ArtifactValidationError as exc:
        if "artifact path" in str(exc):
            raise _unsafe_artifact_path_error("inference artifact", exc) from exc
        msg = f"invalid inference artifact: {exc}"
        raise EvaluationInputArtifactError(msg) from exc
    except ArtifactError as exc:
        msg = f"invalid inference artifact: {exc}"
        raise EvaluationInputArtifactError(msg) from exc


def _unsafe_artifact_path_error(role: str, exc: ArtifactValidationError) -> EvaluationPathError:
    text = str(exc)
    if "parent traversal" in text:
        detail = "parent traversal is not allowed"
    elif "relative" in text:
        detail = "paths must be relative and project-local"
    else:
        detail = "path values are unsafe"
    return EvaluationPathError(f"unsafe {role} path: {detail}")


def _verify_artifact_linkage(
    preprocess_artifact: PreprocessArtifact,
    inference_artifact: InferenceArtifact,
) -> None:
    if preprocess_artifact.schema_version != inference_artifact.schema_version:
        msg = "preprocessing and inference artifact schema versions are incompatible"
        raise EvaluationInputArtifactError(msg)
    if preprocess_artifact.stage != "preprocess":
        msg = "preprocessing artifact stage must be 'preprocess'"
        raise EvaluationInputArtifactError(msg)
    if inference_artifact.stage != DUMMY_INFERENCE_STAGE:
        msg = f"inference artifact stage must be {DUMMY_INFERENCE_STAGE!r}"
        raise EvaluationInputArtifactError(msg)
    if inference_artifact.method != DUMMY_INFERENCE_METHOD:
        msg = f"inference method must be fixed to {DUMMY_INFERENCE_METHOD!r}"
        raise EvaluationInputArtifactError(msg)
    if preprocess_artifact.config_hash != inference_artifact.config_hash:
        msg = "preprocessing and inference config hashes do not match"
        raise EvaluationInputArtifactError(msg)
    if preprocess_artifact.manifest_hash != inference_artifact.manifest_hash:
        msg = "preprocessing and inference manifest hashes do not match"
        raise EvaluationInputArtifactError(msg)
    if preprocess_artifact.output_case_ids != inference_artifact.prediction_case_ids:
        msg = "preprocessing and inference case IDs must match in identical order"
        raise EvaluationInputArtifactError(msg)
    case_count = len(preprocess_artifact.output_case_ids)
    if len(set(preprocess_artifact.output_case_ids)) != case_count:
        msg = "preprocessing artifact output_case_ids must be unique"
        raise EvaluationInputArtifactError(msg)
    if (
        len(preprocess_artifact.output_label_paths) != case_count
        or len(inference_artifact.prediction_paths) != case_count
        or len(inference_artifact.prediction_hashes) != case_count
    ):
        msg = (
            "case IDs, label paths, prediction paths, and prediction hashes must have equal lengths"
        )
        raise EvaluationInputArtifactError(msg)


def _resolve_existing_directory(path: Path, role: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        msg = f"{role} does not exist or cannot be resolved"
        raise EvaluationPathError(msg) from exc
    if not resolved.is_dir():
        msg = f"{role} is not a directory"
        raise EvaluationPathError(msg)
    return resolved


def _prepare_artifact_output_path(artifact_output_path: Path) -> Path:
    if not artifact_output_path.name:
        msg = "artifact_output_path must name a JSON output file"
        raise EvaluationPathError(msg)
    if _contains_parent_traversal(artifact_output_path):
        msg = "artifact_output_path must not contain parent traversal"
        raise EvaluationPathError(msg)
    if artifact_output_path.exists() or artifact_output_path.is_symlink():
        msg = "refusing to overwrite existing evaluation artifact"
        raise EvaluationOutputCollisionError(msg)
    artifact_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = artifact_output_path.parent.resolve(strict=True)
    except OSError as exc:
        msg = "failed to resolve evaluation artifact parent"
        raise EvaluationPathError(msg) from exc
    if not parent.is_dir():
        msg = "evaluation artifact parent is not a directory"
        raise EvaluationPathError(msg)
    return parent / artifact_output_path.name


def _resolve_case_paths(
    preprocess_artifact: PreprocessArtifact,
    inference_artifact: InferenceArtifact,
    *,
    preprocessed_root: Path,
    prediction_root: Path,
) -> tuple[_EvaluationCasePaths, ...]:
    seen_label_paths: set[Path] = set()
    seen_prediction_paths: set[Path] = set()
    case_paths: list[_EvaluationCasePaths] = []
    for case_id, label_path, prediction_path, prediction_hash in zip(
        preprocess_artifact.output_case_ids,
        preprocess_artifact.output_label_paths,
        inference_artifact.prediction_paths,
        inference_artifact.prediction_hashes,
        strict=True,
    ):
        label_relative_path = _relative_posix_path(label_path, "preprocessed label")
        prediction_relative_path = _relative_posix_path(prediction_path, "prediction")
        resolved_label_path = _resolve_nifti_input_path(
            label_relative_path,
            root=preprocessed_root,
            root_role="preprocessed_root",
            role="preprocessed label",
        )
        resolved_prediction_path = _resolve_nifti_input_path(
            prediction_relative_path,
            root=prediction_root,
            root_role="prediction_root",
            role="prediction",
        )
        if resolved_label_path in seen_label_paths:
            msg = "duplicate resolved preprocessed label path"
            raise EvaluationPathError(msg)
        if resolved_prediction_path in seen_prediction_paths:
            msg = "duplicate resolved prediction path"
            raise EvaluationPathError(msg)
        seen_label_paths.add(resolved_label_path)
        seen_prediction_paths.add(resolved_prediction_path)
        case_paths.append(
            _EvaluationCasePaths(
                case_id=case_id,
                label_relative_path=label_relative_path,
                prediction_relative_path=prediction_relative_path,
                label_path=resolved_label_path,
                prediction_path=resolved_prediction_path,
                expected_prediction_hash=prediction_hash,
            )
        )
    return tuple(case_paths)


def _relative_posix_path(value: str, role: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        msg = f"{role} path must be a nonempty relative path"
        raise EvaluationPathError(msg)
    if value.startswith("~"):
        msg = f"{role} path must not use a home-relative path"
        raise EvaluationPathError(msg)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute():
        msg = f"{role} path must be relative"
        raise EvaluationPathError(msg)
    if ".." in posix_path.parts or ".." in windows_path.parts:
        msg = f"{role} path must not contain parent traversal"
        raise EvaluationPathError(msg)
    return posix_path


def _resolve_nifti_input_path(
    relative_path: PurePosixPath,
    *,
    root: Path,
    root_role: str,
    role: str,
) -> Path:
    candidate_path = root.joinpath(*relative_path.parts)
    try:
        resolved_path = candidate_path.resolve(strict=True)
    except OSError as exc:
        msg = f"{role} is missing or cannot be resolved: {relative_path.as_posix()}"
        raise EvaluationNiftiInputError(msg) from exc

    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        msg = f"{role} path escapes {root_role}: {relative_path.as_posix()}"
        raise EvaluationPathError(msg) from exc

    if not resolved_path.is_file():
        msg = f"{role} path is not a regular file: {relative_path.as_posix()}"
        raise EvaluationNiftiInputError(msg)
    return resolved_path


def _verify_prediction_hashes(case_paths: tuple[_EvaluationCasePaths, ...]) -> None:
    for case_path in case_paths:
        try:
            recomputed_hash = sha256_file(case_path.prediction_path)
        except HashFileError as exc:
            msg = f"case {case_path.case_id} prediction hash could not be recomputed"
            raise EvaluationNiftiInputError(msg) from exc
        if recomputed_hash != case_path.expected_prediction_hash:
            msg = f"case {case_path.case_id} prediction hash mismatch"
            raise EvaluationPredictionHashMismatchError(msg)


def _evaluate_cases(
    case_paths: tuple[_EvaluationCasePaths, ...],
) -> tuple[EvaluationCaseMetrics, ...]:
    metrics: list[EvaluationCaseMetrics] = []
    for case_path in case_paths:
        metrics.append(_evaluate_case(case_path))
    return tuple(metrics)


def _evaluate_case(case_path: _EvaluationCasePaths) -> EvaluationCaseMetrics:
    label_image = _load_nifti(case_path.label_path, "preprocessed label", case_id=case_path.case_id)
    prediction_image = _load_nifti(
        case_path.prediction_path, "prediction", case_id=case_path.case_id
    )
    label_array = _as_array(label_image, "preprocessed label", case_id=case_path.case_id)
    prediction_array = _as_array(prediction_image, "prediction", case_id=case_path.case_id)
    _require_3d_array(label_array, "preprocessed label", case_id=case_path.case_id)
    _require_3d_array(prediction_array, "prediction", case_id=case_path.case_id)
    _require_real_finite_values(label_array, "preprocessed label", case_id=case_path.case_id)
    _require_real_finite_values(prediction_array, "prediction", case_id=case_path.case_id)
    _require_binary_values(label_array, "preprocessed label", case_id=case_path.case_id)
    _require_binary_values(prediction_array, "prediction", case_id=case_path.case_id)
    if label_array.shape != prediction_array.shape:
        msg = (
            f"case {case_path.case_id} label and prediction shapes differ: "
            f"label shape={label_array.shape}, prediction shape={prediction_array.shape}"
        )
        raise EvaluationGeometryMismatchError(msg)

    label_affine = _as_affine(label_image, "preprocessed label", case_id=case_path.case_id)
    prediction_affine = _as_affine(prediction_image, "prediction", case_id=case_path.case_id)
    if not np.allclose(
        label_affine,
        prediction_affine,
        atol=AFFINE_ABSOLUTE_TOLERANCE,
        rtol=0.0,
        equal_nan=False,
    ):
        msg = (
            f"case {case_path.case_id} label and prediction affines differ beyond "
            f"absolute tolerance {AFFINE_ABSOLUTE_TOLERANCE}"
        )
        raise EvaluationGeometryMismatchError(msg)

    label_mask = cast(BoolArray, np.asarray(label_array == 1, dtype=np.bool_))
    prediction_mask = cast(BoolArray, np.asarray(prediction_array == 1, dtype=np.bool_))
    return compute_case_metrics(case_path.case_id, label_mask, prediction_mask)


def _load_nifti(path: Path, role: str, *, case_id: str) -> Any:
    try:
        return cast(Any, nib.load(str(path)))
    except ImageFileError as exc:
        msg = f"case {case_id} {role} is not a readable NIfTI file"
        raise EvaluationNiftiInputError(msg) from exc
    except OSError as exc:
        msg = f"case {case_id} failed to read {role} NIfTI file"
        raise EvaluationNiftiInputError(msg) from exc
    except ValueError as exc:
        msg = f"case {case_id} has invalid {role} NIfTI data"
        raise EvaluationNiftiInputError(msg) from exc


def _as_array(image: Any, role: str, *, case_id: str) -> Array:
    try:
        return cast(Array, np.asanyarray(image.dataobj))
    except ValueError as exc:
        msg = f"case {case_id} has invalid {role} array data"
        raise EvaluationNiftiInputError(msg) from exc


def _as_affine(image: Any, role: str, *, case_id: str) -> AffineArray:
    affine = cast(AffineArray, np.asarray(image.affine, dtype=np.float64))
    if affine.shape != (4, 4):
        msg = f"case {case_id} {role} affine must be 4x4"
        raise EvaluationNiftiInputError(msg)
    if not np.isfinite(affine).all():
        msg = f"case {case_id} {role} affine contains NaN or Infinity"
        raise EvaluationNiftiInputError(msg)
    return affine


def _require_3d_array(array: Array, role: str, *, case_id: str) -> None:
    if array.ndim != 3:
        msg = f"case {case_id} {role} must be 3D"
        raise EvaluationNiftiInputError(msg)


def _require_real_finite_values(array: Array, role: str, *, case_id: str) -> None:
    try:
        has_only_finite_values = bool(np.isfinite(array).all())
    except TypeError as exc:
        msg = f"case {case_id} {role} values cannot be checked"
        raise EvaluationNiftiInputError(msg) from exc
    if not has_only_finite_values:
        msg = f"case {case_id} {role} contains NaN or infinite values"
        raise EvaluationNiftiInputError(msg)


def _require_binary_values(array: Array, role: str, *, case_id: str) -> None:
    if np.iscomplexobj(array):
        msg = f"case {case_id} {role} must contain only real binary values"
        raise EvaluationNiftiInputError(msg)
    unique_values = cast(Array, np.unique(array))
    invalid_values = unique_values[~np.isin(unique_values, [0, 1])]
    if invalid_values.size:
        msg = f"case {case_id} {role} must be binary with values from {{0, 1}}"
        raise EvaluationNiftiInputError(msg)


def _build_evaluation_artifact(
    preprocess_artifact: PreprocessArtifact,
    *,
    case_metrics: tuple[EvaluationCaseMetrics, ...],
    git_commit: str,
    created_at_utc: str,
) -> EvaluationArtifact:
    if not case_metrics:
        msg = "evaluation requires at least one case"
        raise EvaluationFailureError(msg)

    total_true_positives = sum(metric.true_positives for metric in case_metrics)
    total_false_positives = sum(metric.false_positives for metric in case_metrics)
    total_false_negatives = sum(metric.false_negatives for metric in case_metrics)
    total_true_negatives = sum(metric.true_negatives for metric in case_metrics)
    case_count = len(case_metrics)
    macro_mean_dice = sum(metric.dice for metric in case_metrics) / case_count
    macro_mean_iou = sum(metric.iou for metric in case_metrics) / case_count
    micro_counts = SegmentationConfusionCounts(
        true_positives=total_true_positives,
        false_positives=total_false_positives,
        false_negatives=total_false_negatives,
        true_negatives=total_true_negatives,
        ground_truth_foreground_voxels=sum(
            metric.ground_truth_foreground_voxels for metric in case_metrics
        ),
        predicted_foreground_voxels=sum(
            metric.predicted_foreground_voxels for metric in case_metrics
        ),
    )
    micro_dice = dice_from_counts(micro_counts)
    micro_iou = iou_from_counts(micro_counts)
    for field_name, value in {
        "macro_mean_dice": macro_mean_dice,
        "macro_mean_iou": macro_mean_iou,
        "micro_dice": micro_dice,
        "micro_iou": micro_iou,
    }.items():
        _require_unit_metric(value, field_name)

    try:
        return EvaluationArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            stage=EVALUATION_STAGE,
            created_at_utc=created_at_utc,
            git_commit=git_commit,
            config_hash=preprocess_artifact.config_hash,
            manifest_hash=preprocess_artifact.manifest_hash,
            method=DUMMY_INFERENCE_METHOD,
            case_ids=tuple(metric.case_id for metric in case_metrics),
            case_metrics=case_metrics,
            macro_mean_dice=macro_mean_dice,
            macro_mean_iou=macro_mean_iou,
            micro_dice=micro_dice,
            micro_iou=micro_iou,
            total_true_positives=total_true_positives,
            total_false_positives=total_false_positives,
            total_false_negatives=total_false_negatives,
            total_true_negatives=total_true_negatives,
            evaluated_case_count=case_count,
        )
    except ArtifactError as exc:
        msg = f"failed to construct evaluation artifact: {exc}"
        raise EvaluationFailureError(msg) from exc


def _write_artifact_to_temp(
    artifact: EvaluationArtifact,
    artifact_output_path: Path,
) -> Path:
    descriptor, temp_path_text = tempfile.mkstemp(
        prefix=f".{artifact_output_path.name}.",
        suffix=".tmp",
        dir=artifact_output_path.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_path_text)
    try:
        artifact_to_json(artifact, temp_path)
    except ArtifactError as exc:
        temp_path.unlink(missing_ok=True)
        msg = f"failed to write staged evaluation artifact: {exc}"
        raise EvaluationFailureError(msg) from exc
    return temp_path


def _publish_artifact(artifact_temp_path: Path, artifact_output_path: Path) -> None:
    if artifact_output_path.exists() or artifact_output_path.is_symlink():
        msg = "refusing to overwrite existing evaluation artifact"
        raise EvaluationOutputCollisionError(msg)
    artifact_temp_path.replace(artifact_output_path)


def _cleanup_after_failure(artifact_temp_path: Path | None) -> None:
    if artifact_temp_path is not None and artifact_temp_path.exists():
        artifact_temp_path.unlink(missing_ok=True)


def _contains_parent_traversal(path: Path) -> bool:
    value = str(path)
    return ".." in PurePosixPath(value).parts or ".." in PureWindowsPath(value).parts


def _require_nonempty_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a nonempty string"
        raise EvaluationInputArtifactError(msg)


def _require_nonnegative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{field_name} must be a nonnegative integer"
        raise EvaluationFailureError(msg)


def _require_unit_metric(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        msg = f"{field_name} must be finite and in [0, 1]"
        raise EvaluationFailureError(msg)
