"""Unit tests for deterministic Phase 1 evaluation metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest

from protoem_ct.artifacts import (
    EvaluationArtifact,
    InferenceArtifact,
    PreprocessArtifact,
    SyntheticManifest,
    ValidationArtifact,
    artifact_from_json,
    artifact_to_dict,
    sha256_file,
)
from protoem_ct.data.manifest_validation import validate_synthetic_manifest
from protoem_ct.data.preprocessing import preprocess_synthetic_dataset
from protoem_ct.data.synthetic import (
    SyntheticDataConfig,
    SyntheticDatasetResult,
    create_synthetic_dataset,
)
from protoem_ct.evaluation.dummy_inference import run_dummy_inference
from protoem_ct.evaluation.metrics import (
    EvaluationGeometryMismatchError,
    EvaluationInputArtifactError,
    EvaluationNiftiInputError,
    EvaluationOutputCollisionError,
    EvaluationPathError,
    EvaluationPredictionHashMismatchError,
    SegmentationConfusionCounts,
    compute_binary_confusion_counts,
    compute_case_metrics,
    dice_from_counts,
    evaluate_predictions,
    iou_from_counts,
)

GIT_COMMIT = "48ea1d6"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"
PREPROCESS_CREATED_AT_UTC = "2026-07-25T01:10:00Z"
INFERENCE_CREATED_AT_UTC = "2026-07-25T01:15:00Z"
EVALUATION_CREATED_AT_UTC = "2026-07-25T01:20:00Z"


def _synthetic_config(**overrides: object) -> SyntheticDataConfig:
    values: dict[str, object] = {
        "dataset_id": "phase1-synthetic",
        "seed": 1729,
        "case_count": 3,
        "shape": (8, 8, 6),
        "spacing": (1.5, 1.5, 2.0),
        "generated_data_root": PurePosixPath("generated/phase1/data"),
    }
    values.update(overrides)
    return SyntheticDataConfig(**values)  # type: ignore[arg-type]


def _dataset(tmp_path: Path) -> SyntheticDatasetResult:
    return create_synthetic_dataset(
        _synthetic_config(),
        output_root=tmp_path / "data-root",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )


def _validate(
    result: SyntheticDatasetResult,
    tmp_path: Path,
) -> tuple[Path, ValidationArtifact]:
    validation_path = tmp_path / "artifacts" / "validation.json"
    validation = validate_synthetic_manifest(
        result.manifest_path,
        data_root=tmp_path / "data-root",
        output_path=validation_path,
        git_commit=GIT_COMMIT,
        created_at_utc=VALIDATION_CREATED_AT_UTC,
    )
    return validation_path, validation


def _preprocess(
    result: SyntheticDatasetResult,
    validation_path: Path,
    tmp_path: Path,
) -> tuple[Path, PreprocessArtifact]:
    artifact_path = tmp_path / "artifacts" / "preprocess.json"
    artifact = preprocess_synthetic_dataset(
        result.manifest_path,
        validation_path,
        data_root=tmp_path / "data-root",
        output_root=tmp_path / "preprocessed",
        artifact_output_path=artifact_path,
        config_path=Path("configs/experiment/phase1.yaml"),
        git_commit=GIT_COMMIT,
        created_at_utc=PREPROCESS_CREATED_AT_UTC,
    )
    return artifact_path, artifact


def _run_dummy(
    preprocess_artifact_path: Path,
    tmp_path: Path,
) -> tuple[Path, InferenceArtifact]:
    artifact_path = tmp_path / "artifacts" / "inference.json"
    artifact = run_dummy_inference(
        preprocess_artifact_path,
        preprocessed_root=tmp_path / "preprocessed",
        output_root=tmp_path / "predictions",
        artifact_output_path=artifact_path,
        config_path=Path("configs/experiment/phase1.yaml"),
        git_commit=GIT_COMMIT,
        created_at_utc=INFERENCE_CREATED_AT_UTC,
    )
    return artifact_path, artifact


def _prepared(
    tmp_path: Path,
) -> tuple[SyntheticManifest, Path, PreprocessArtifact, Path, InferenceArtifact]:
    result = _dataset(tmp_path)
    validation_path, _validation = _validate(result, tmp_path)
    preprocess_artifact_path, preprocess_artifact = _preprocess(result, validation_path, tmp_path)
    inference_artifact_path, inference_artifact = _run_dummy(preprocess_artifact_path, tmp_path)
    return (
        result.manifest,
        preprocess_artifact_path,
        preprocess_artifact,
        inference_artifact_path,
        inference_artifact,
    )


def _run_evaluate(
    preprocess_artifact_path: Path,
    inference_artifact_path: Path,
    tmp_path: Path,
    *,
    artifact_output_path: Path | None = None,
) -> EvaluationArtifact:
    return evaluate_predictions(
        preprocess_artifact_path,
        inference_artifact_path,
        preprocessed_root=tmp_path / "preprocessed",
        prediction_root=tmp_path / "predictions",
        artifact_output_path=artifact_output_path or tmp_path / "artifacts" / "evaluation.json",
        git_commit=GIT_COMMIT,
        created_at_utc=EVALUATION_CREATED_AT_UTC,
    )


def _write_preprocess_payload(
    path: Path,
    artifact: PreprocessArtifact,
    **overrides: object,
) -> None:
    payload = cast(dict[str, Any], artifact_to_dict(artifact))
    payload.update(overrides)
    _write_json(path, payload)


def _write_inference_payload(
    path: Path,
    artifact: InferenceArtifact,
    **overrides: object,
) -> None:
    payload = cast(dict[str, Any], artifact_to_dict(artifact))
    payload.update(overrides)
    _write_json(path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_nifti(path: Path) -> Any:
    return cast(Any, nib.load(str(path)))


def _array(path: Path) -> np.ndarray[Any, Any]:
    return cast(np.ndarray[Any, Any], np.asanyarray(_load_nifti(path).dataobj))


def _affine(path: Path) -> np.ndarray[Any, Any]:
    return cast(np.ndarray[Any, Any], np.asarray(_load_nifti(path).affine, dtype=np.float64))


def _write_nifti(path: Path, data: np.ndarray[Any, Any], affine: np.ndarray[Any, Any]) -> None:
    image = nib.Nifti1Image(data, affine)  # type: ignore[no-untyped-call]
    nib.save(image, str(path))


def _label_path(tmp_path: Path, artifact: PreprocessArtifact, index: int = 0) -> Path:
    return tmp_path / "preprocessed" / artifact.output_label_paths[index]


def _prediction_path(tmp_path: Path, artifact: InferenceArtifact, index: int = 0) -> Path:
    return tmp_path / "predictions" / artifact.prediction_paths[index]


def _refresh_prediction_hash(
    inference_artifact_path: Path,
    inference_artifact: InferenceArtifact,
    prediction_path: Path,
    *,
    index: int = 0,
) -> None:
    prediction_hashes = list(inference_artifact.prediction_hashes)
    prediction_hashes[index] = sha256_file(prediction_path)
    _write_inference_payload(
        inference_artifact_path,
        inference_artifact,
        prediction_hashes=prediction_hashes,
    )


def _assert_metrics_are_unit_interval(artifact: EvaluationArtifact) -> None:
    values = [
        artifact.macro_mean_dice,
        artifact.macro_mean_iou,
        artifact.micro_dice,
        artifact.micro_iou,
        *[metric.dice for metric in artifact.case_metrics],
        *[metric.iou for metric in artifact.case_metrics],
    ]
    assert all(math.isfinite(value) for value in values)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_exact_known_confusion_counts_and_metrics() -> None:
    ground_truth = np.array(
        [[True, True, False], [False, True, False]],
        dtype=np.bool_,
    )
    prediction = np.array(
        [[True, False, True], [False, True, False]],
        dtype=np.bool_,
    )

    counts = compute_binary_confusion_counts(ground_truth, prediction)
    metrics = compute_case_metrics("known-case", ground_truth, prediction)

    assert counts == SegmentationConfusionCounts(
        true_positives=2,
        false_positives=1,
        false_negatives=1,
        true_negatives=2,
        ground_truth_foreground_voxels=3,
        predicted_foreground_voxels=3,
    )
    assert dice_from_counts(counts) == 2 / 3
    assert iou_from_counts(counts) == 0.5
    assert metrics.dice == 2 / 3
    assert metrics.iou == 0.5


def test_perfect_prediction_produces_unit_metrics() -> None:
    mask = np.array([[True, False], [False, True]], dtype=np.bool_)

    metrics = compute_case_metrics("perfect", mask, mask)

    assert metrics.dice == 1.0
    assert metrics.iou == 1.0


def test_completely_missed_foreground_produces_zero_metrics() -> None:
    ground_truth = np.array([[True, False], [True, False]], dtype=np.bool_)
    prediction = np.zeros_like(ground_truth)

    metrics = compute_case_metrics("missed", ground_truth, prediction)

    assert metrics.true_positives == 0
    assert metrics.false_negatives == 2
    assert metrics.dice == 0.0
    assert metrics.iou == 0.0


def test_both_masks_empty_produces_unit_metrics() -> None:
    ground_truth = np.zeros((2, 2), dtype=np.bool_)
    prediction = np.zeros((2, 2), dtype=np.bool_)

    metrics = compute_case_metrics("empty", ground_truth, prediction)

    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.dice == 1.0
    assert metrics.iou == 1.0


def test_valid_end_to_end_evaluation_succeeds(tmp_path: Path) -> None:
    _manifest, preprocess_path, preprocess_artifact, inference_path, inference_artifact = _prepared(
        tmp_path
    )

    artifact = _run_evaluate(preprocess_path, inference_path, tmp_path)
    persisted = artifact_from_json(
        (tmp_path / "artifacts" / "evaluation.json").read_text(encoding="utf-8"),
        EvaluationArtifact,
    )

    assert persisted == artifact
    assert artifact.stage == "evaluate"
    assert artifact.method == "dummy"
    assert artifact.case_ids == preprocess_artifact.output_case_ids
    assert artifact.case_ids == inference_artifact.prediction_case_ids
    assert artifact.config_hash == preprocess_artifact.config_hash
    assert artifact.config_hash == inference_artifact.config_hash
    assert artifact.manifest_hash == preprocess_artifact.manifest_hash
    assert artifact.manifest_hash == inference_artifact.manifest_hash
    assert artifact.evaluated_case_count == 3


def test_deterministic_ordered_per_case_metrics(tmp_path: Path) -> None:
    _manifest, preprocess_path, preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )

    artifact = _run_evaluate(preprocess_path, inference_path, tmp_path)

    assert artifact.case_ids == preprocess_artifact.output_case_ids
    assert tuple(metric.case_id for metric in artifact.case_metrics) == artifact.case_ids


def test_macro_micro_and_aggregate_metrics_are_correct(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )

    artifact = _run_evaluate(preprocess_path, inference_path, tmp_path)
    total_true_positives = sum(metric.true_positives for metric in artifact.case_metrics)
    total_false_positives = sum(metric.false_positives for metric in artifact.case_metrics)
    total_false_negatives = sum(metric.false_negatives for metric in artifact.case_metrics)
    total_true_negatives = sum(metric.true_negatives for metric in artifact.case_metrics)

    assert artifact.macro_mean_dice == sum(metric.dice for metric in artifact.case_metrics) / 3
    assert artifact.macro_mean_iou == sum(metric.iou for metric in artifact.case_metrics) / 3
    assert artifact.total_true_positives == total_true_positives
    assert artifact.total_false_positives == total_false_positives
    assert artifact.total_false_negatives == total_false_negatives
    assert artifact.total_true_negatives == total_true_negatives
    micro_counts = SegmentationConfusionCounts(
        true_positives=total_true_positives,
        false_positives=total_false_positives,
        false_negatives=total_false_negatives,
        true_negatives=total_true_negatives,
        ground_truth_foreground_voxels=sum(
            metric.ground_truth_foreground_voxels for metric in artifact.case_metrics
        ),
        predicted_foreground_voxels=sum(
            metric.predicted_foreground_voxels for metric in artifact.case_metrics
        ),
    )
    assert artifact.micro_dice == dice_from_counts(micro_counts)
    assert artifact.micro_iou == iou_from_counts(micro_counts)
    _assert_metrics_are_unit_interval(artifact)


def test_prediction_file_hashes_are_verified_on_success(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, inference_artifact = (
        _prepared(tmp_path)
    )

    _run_evaluate(preprocess_path, inference_path, tmp_path)

    for relative_path, prediction_hash in zip(
        inference_artifact.prediction_paths,
        inference_artifact.prediction_hashes,
        strict=True,
    ):
        assert sha256_file(tmp_path / "predictions" / relative_path) == prediction_hash


def test_altered_prediction_is_rejected_by_hash(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, inference_artifact = (
        _prepared(tmp_path)
    )
    prediction_path = _prediction_path(tmp_path, inference_artifact)
    prediction = _array(prediction_path).astype(np.uint8, copy=True)
    prediction[0, 0, 0] = 1 - prediction[0, 0, 0]
    _write_nifti(prediction_path, prediction, _affine(prediction_path))

    with pytest.raises(EvaluationPredictionHashMismatchError, match="hash mismatch"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_inconsistent_config_hash_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, inference_artifact = (
        _prepared(tmp_path)
    )
    _write_inference_payload(inference_path, inference_artifact, config_hash="1" * 64)

    with pytest.raises(EvaluationInputArtifactError, match="config hashes"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_inconsistent_manifest_hash_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, inference_artifact = (
        _prepared(tmp_path)
    )
    _write_inference_payload(inference_path, inference_artifact, manifest_hash="1" * 64)

    with pytest.raises(EvaluationInputArtifactError, match="manifest hashes"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_mismatched_case_order_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, inference_artifact = (
        _prepared(tmp_path)
    )
    _write_inference_payload(
        inference_path,
        inference_artifact,
        prediction_case_ids=list(reversed(inference_artifact.prediction_case_ids)),
    )

    with pytest.raises(EvaluationInputArtifactError, match="identical order"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_mismatched_case_path_hash_lengths_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, inference_artifact = (
        _prepared(tmp_path)
    )
    _write_inference_payload(
        inference_path,
        inference_artifact,
        prediction_hashes=list(inference_artifact.prediction_hashes[:-1]),
    )

    with pytest.raises(EvaluationInputArtifactError, match="equal lengths"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_duplicate_case_id_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, preprocess_artifact, inference_path, inference_artifact = _prepared(
        tmp_path
    )
    duplicate_case_ids = list(preprocess_artifact.output_case_ids)
    duplicate_case_ids[1] = duplicate_case_ids[0]
    _write_preprocess_payload(
        preprocess_path, preprocess_artifact, output_case_ids=duplicate_case_ids
    )
    _write_inference_payload(
        inference_path,
        inference_artifact,
        prediction_case_ids=duplicate_case_ids,
    )

    with pytest.raises(EvaluationInputArtifactError, match="unique"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_duplicate_label_path_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )
    duplicate_label_paths = list(preprocess_artifact.output_label_paths)
    duplicate_label_paths[1] = duplicate_label_paths[0]
    _write_preprocess_payload(
        preprocess_path,
        preprocess_artifact,
        output_label_paths=duplicate_label_paths,
    )

    with pytest.raises(EvaluationPathError, match="duplicate resolved preprocessed label"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_duplicate_prediction_path_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, inference_artifact = (
        _prepared(tmp_path)
    )
    duplicate_prediction_paths = list(inference_artifact.prediction_paths)
    duplicate_prediction_paths[1] = duplicate_prediction_paths[0]
    _write_inference_payload(
        inference_path,
        inference_artifact,
        prediction_paths=duplicate_prediction_paths,
    )

    with pytest.raises(EvaluationInputArtifactError, match="prediction_paths"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_missing_label_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )
    _label_path(tmp_path, preprocess_artifact).unlink()

    with pytest.raises(EvaluationNiftiInputError, match="missing"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_missing_prediction_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, inference_artifact = (
        _prepared(tmp_path)
    )
    _prediction_path(tmp_path, inference_artifact).unlink()

    with pytest.raises(EvaluationNiftiInputError, match="missing"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_absolute_path_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )
    label_paths = list(preprocess_artifact.output_label_paths)
    label_paths[0] = "/absolute/escape.nii"
    _write_preprocess_payload(preprocess_path, preprocess_artifact, output_label_paths=label_paths)

    with pytest.raises(EvaluationPathError, match="relative"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_parent_traversal_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )
    label_paths = list(preprocess_artifact.output_label_paths)
    label_paths[0] = "../escape.nii"
    _write_preprocess_payload(preprocess_path, preprocess_artifact, output_label_paths=label_paths)

    with pytest.raises(EvaluationPathError, match="parent traversal"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_symlink_escape_rejected_when_supported(tmp_path: Path) -> None:
    _manifest, preprocess_path, preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )
    outside_label = tmp_path / "outside-label.nii"
    outside_label.write_bytes(_label_path(tmp_path, preprocess_artifact).read_bytes())
    symlink_path = tmp_path / "preprocessed" / "labels" / "escape.nii"
    try:
        symlink_path.symlink_to(outside_label)
    except OSError as exc:
        pytest.skip(f"platform does not support this symlink test: {exc}")

    label_paths = list(preprocess_artifact.output_label_paths)
    label_paths[0] = "labels/escape.nii"
    _write_preprocess_payload(preprocess_path, preprocess_artifact, output_label_paths=label_paths)

    with pytest.raises(EvaluationPathError, match="escapes preprocessed_root"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_nonbinary_label_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )
    label_path = _label_path(tmp_path, preprocess_artifact)
    label = _array(label_path).astype(np.uint8, copy=True)
    label[0, 0, 0] = 2
    _write_nifti(label_path, label, _affine(label_path))

    with pytest.raises(EvaluationNiftiInputError, match="binary"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_nonbinary_prediction_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, inference_artifact = (
        _prepared(tmp_path)
    )
    prediction_path = _prediction_path(tmp_path, inference_artifact)
    prediction = _array(prediction_path).astype(np.uint8, copy=True)
    prediction[0, 0, 0] = 2
    _write_nifti(prediction_path, prediction, _affine(prediction_path))
    _refresh_prediction_hash(inference_path, inference_artifact, prediction_path)

    with pytest.raises(EvaluationNiftiInputError, match="binary"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf])
def test_nan_or_infinity_rejected(tmp_path: Path, bad_value: float) -> None:
    _manifest, preprocess_path, preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )
    label_path = _label_path(tmp_path, preprocess_artifact)
    label = _array(label_path).astype(np.float32, copy=True)
    label[0, 0, 0] = bad_value
    _write_nifti(label_path, label, _affine(label_path))

    with pytest.raises(EvaluationNiftiInputError, match="NaN|infinite"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_shape_mismatch_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, inference_artifact = (
        _prepared(tmp_path)
    )
    prediction_path = _prediction_path(tmp_path, inference_artifact)
    prediction = np.zeros((2, 2, 2), dtype=np.uint8)
    _write_nifti(prediction_path, prediction, _affine(prediction_path))
    _refresh_prediction_hash(inference_path, inference_artifact, prediction_path)

    with pytest.raises(EvaluationGeometryMismatchError, match="shapes differ"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_affine_mismatch_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, inference_artifact = (
        _prepared(tmp_path)
    )
    prediction_path = _prediction_path(tmp_path, inference_artifact)
    affine = _affine(prediction_path)
    affine[0, 3] += 0.01
    _write_nifti(prediction_path, _array(prediction_path).astype(np.uint8, copy=True), affine)
    _refresh_prediction_hash(inference_path, inference_artifact, prediction_path)

    with pytest.raises(EvaluationGeometryMismatchError, match="affines differ"):
        _run_evaluate(preprocess_path, inference_path, tmp_path)


def test_existing_artifact_output_rejected(tmp_path: Path) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )
    artifact_output_path = tmp_path / "artifacts" / "evaluation.json"
    artifact_output_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(EvaluationOutputCollisionError, match="overwrite"):
        _run_evaluate(
            preprocess_path,
            inference_path,
            tmp_path,
            artifact_output_path=artifact_output_path,
        )


def test_failed_evaluation_leaves_no_partial_artifact_or_temporary_file(tmp_path: Path) -> None:
    _manifest, preprocess_path, preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )
    artifact_output_path = tmp_path / "evaluation-output" / "evaluation.json"
    label_path = _label_path(tmp_path, preprocess_artifact)
    label = _array(label_path).astype(np.float32, copy=True)
    label[0, 0, 0] = np.nan
    _write_nifti(label_path, label, _affine(label_path))

    with pytest.raises(EvaluationNiftiInputError):
        _run_evaluate(
            preprocess_path,
            inference_path,
            tmp_path,
            artifact_output_path=artifact_output_path,
        )

    assert not artifact_output_path.exists()
    assert not any(
        path.name.startswith(".evaluation.json.") for path in artifact_output_path.parent.iterdir()
    )


def test_repeated_evaluation_with_identical_inputs_produces_byte_identical_json(
    tmp_path: Path,
) -> None:
    _manifest, preprocess_path, _preprocess_artifact, inference_path, _inference_artifact = (
        _prepared(tmp_path)
    )
    first_path = tmp_path / "first-artifacts" / "evaluation.json"
    second_path = tmp_path / "second-artifacts" / "evaluation.json"

    first = _run_evaluate(
        preprocess_path,
        inference_path,
        tmp_path,
        artifact_output_path=first_path,
    )
    second = _run_evaluate(
        preprocess_path,
        inference_path,
        tmp_path,
        artifact_output_path=second_path,
    )

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")
    assert second_path.read_bytes().endswith(b"\n")
