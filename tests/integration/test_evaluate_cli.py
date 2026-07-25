"""Integration tests for the evaluate CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
from typer.testing import CliRunner

from protoem_ct.artifacts import EvaluationArtifact, InferenceArtifact, artifact_from_json
from protoem_ct.cli.main import app
from protoem_ct.data.manifest_validation import validate_synthetic_manifest
from protoem_ct.data.preprocessing import preprocess_synthetic_dataset
from protoem_ct.data.synthetic import create_synthetic_dataset, load_synthetic_config
from protoem_ct.evaluation.dummy_inference import run_dummy_inference

GIT_COMMIT = "48ea1d6"
CREATED_AT_UTC = "2026-07-25T01:00:00Z"
VALIDATION_CREATED_AT_UTC = "2026-07-25T01:05:00Z"
PREPROCESS_CREATED_AT_UTC = "2026-07-25T01:10:00Z"
INFERENCE_CREATED_AT_UTC = "2026-07-25T01:15:00Z"
EVALUATION_CREATED_AT_UTC = "2026-07-25T01:20:00Z"


def _invoke_evaluate(*args: str) -> Any:
    """Invoke the public evaluate CLI command."""
    runner = CliRunner()
    return runner.invoke(app, ["evaluate", *args])


def _create_validate_preprocess_infer(tmp_path: Path) -> tuple[Any, InferenceArtifact]:
    config = load_synthetic_config(Path("configs/data/synthetic.yaml"))
    result = create_synthetic_dataset(
        config,
        output_root=tmp_path / "data-root",
        git_commit=GIT_COMMIT,
        created_at_utc=CREATED_AT_UTC,
    )
    validation_path = tmp_path / "artifacts" / "validation.json"
    validate_synthetic_manifest(
        result.manifest_path,
        data_root=tmp_path / "data-root",
        output_path=validation_path,
        git_commit=GIT_COMMIT,
        created_at_utc=VALIDATION_CREATED_AT_UTC,
    )
    preprocess_artifact = preprocess_synthetic_dataset(
        result.manifest_path,
        validation_path,
        data_root=tmp_path / "data-root",
        output_root=tmp_path / "preprocessed",
        artifact_output_path=tmp_path / "artifacts" / "preprocess.json",
        config_path=Path("configs/experiment/phase1.yaml"),
        git_commit=GIT_COMMIT,
        created_at_utc=PREPROCESS_CREATED_AT_UTC,
    )
    inference_artifact = run_dummy_inference(
        tmp_path / "artifacts" / "preprocess.json",
        preprocessed_root=tmp_path / "preprocessed",
        output_root=tmp_path / "predictions",
        artifact_output_path=tmp_path / "artifacts" / "inference.json",
        config_path=Path("configs/experiment/phase1.yaml"),
        git_commit=GIT_COMMIT,
        created_at_utc=INFERENCE_CREATED_AT_UTC,
    )
    return preprocess_artifact, inference_artifact


def _evaluate_args(artifact_output: str = "artifacts/evaluation.json") -> list[str]:
    return [
        "--preprocess-artifact",
        "artifacts/preprocess.json",
        "--inference-artifact",
        "artifacts/inference.json",
        "--preprocessed-root",
        "preprocessed",
        "--prediction-root",
        "predictions",
        "--artifact-output",
        artifact_output,
        "--git-commit",
        GIT_COMMIT,
        "--created-at-utc",
        EVALUATION_CREATED_AT_UTC,
    ]


def _error_output(result: Any) -> str:
    return cast(str, getattr(result, "stderr", "") or result.output)


def _assert_nonzero_without_traceback(result: Any, tmp_path: Path) -> str:
    error_output = _error_output(result)
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "Traceback" not in error_output
    assert str(tmp_path) not in result.output
    assert str(tmp_path) not in error_output
    return error_output


def _load_array(path: Path) -> np.ndarray[Any, Any]:
    image = cast(Any, nib.load(str(path)))
    return cast(np.ndarray[Any, Any], np.asanyarray(image.dataobj))


def _load_affine(path: Path) -> np.ndarray[Any, Any]:
    image = cast(Any, nib.load(str(path)))
    return cast(np.ndarray[Any, Any], np.asarray(image.affine, dtype=np.float64))


def _write_nifti(path: Path, data: np.ndarray[Any, Any]) -> None:
    image = nib.Nifti1Image(data, _load_affine(path))  # type: ignore[no-untyped-call]
    nib.save(image, str(path))


def _independent_expected_metrics(
    tmp_path: Path,
    artifact: EvaluationArtifact,
) -> dict[str, object]:
    true_positives: list[int] = []
    false_positives: list[int] = []
    false_negatives: list[int] = []
    true_negatives: list[int] = []
    dice_values: list[float] = []
    iou_values: list[float] = []
    for case_metric in artifact.case_metrics:
        label = _load_array(tmp_path / "preprocessed" / f"labels/{case_metric.case_id}.nii") == 1
        prediction = _load_array(
            tmp_path / "predictions" / f"predictions/{case_metric.case_id}.nii"
        )
        prediction_mask = prediction == 1
        tp = int(np.count_nonzero(np.logical_and(label, prediction_mask)))
        fp = int(np.count_nonzero(np.logical_and(np.logical_not(label), prediction_mask)))
        fn = int(np.count_nonzero(np.logical_and(label, np.logical_not(prediction_mask))))
        tn = int(
            np.count_nonzero(np.logical_and(np.logical_not(label), np.logical_not(prediction_mask)))
        )
        denominator = (2 * tp) + fp + fn
        union = tp + fp + fn
        true_positives.append(tp)
        false_positives.append(fp)
        false_negatives.append(fn)
        true_negatives.append(tn)
        dice_values.append(1.0 if denominator == 0 else (2 * tp) / denominator)
        iou_values.append(1.0 if union == 0 else tp / union)

    total_tp = sum(true_positives)
    total_fp = sum(false_positives)
    total_fn = sum(false_negatives)
    total_tn = sum(true_negatives)
    micro_denominator = (2 * total_tp) + total_fp + total_fn
    micro_union = total_tp + total_fp + total_fn
    return {
        "macro_mean_dice": sum(dice_values) / len(dice_values),
        "macro_mean_iou": sum(iou_values) / len(iou_values),
        "micro_dice": 1.0 if micro_denominator == 0 else (2 * total_tp) / micro_denominator,
        "micro_iou": 1.0 if micro_union == 0 else total_tp / micro_union,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "total_true_negatives": total_tn,
    }


def test_evaluate_cli_success_writes_readable_artifact_with_expected_metrics(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    preprocess_artifact, _inference_artifact = _create_validate_preprocess_infer(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = _invoke_evaluate(*_evaluate_args())

    assert result.exit_code == 0
    artifact_path = tmp_path / "artifacts" / "evaluation.json"
    assert artifact_path.is_file()
    artifact = artifact_from_json(artifact_path.read_text(encoding="utf-8"), EvaluationArtifact)
    assert result.output == (
        "evaluation success\n"
        "evaluated case count: 3\n"
        "evaluation artifact path: artifacts/evaluation.json\n"
        f"macro Dice: {artifact.macro_mean_dice}\n"
        f"macro IoU: {artifact.macro_mean_iou}\n"
        f"config hash: {preprocess_artifact.config_hash}\n"
        f"manifest hash: {preprocess_artifact.manifest_hash}\n"
        "method: dummy\n"
    )
    expected_metrics = _independent_expected_metrics(tmp_path, artifact)
    assert artifact.macro_mean_dice == expected_metrics["macro_mean_dice"]
    assert artifact.macro_mean_iou == expected_metrics["macro_mean_iou"]
    assert artifact.micro_dice == expected_metrics["micro_dice"]
    assert artifact.micro_iou == expected_metrics["micro_iou"]
    assert artifact.total_true_positives == expected_metrics["total_true_positives"]
    assert artifact.total_false_positives == expected_metrics["total_false_positives"]
    assert artifact.total_false_negatives == expected_metrics["total_false_negatives"]
    assert artifact.total_true_negatives == expected_metrics["total_true_negatives"]


def test_evaluate_cli_repeated_runs_produce_byte_identical_artifact_json(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _preprocess_artifact, _inference_artifact = _create_validate_preprocess_infer(tmp_path)
    monkeypatch.chdir(tmp_path)

    first = _invoke_evaluate(*_evaluate_args("first-artifacts/evaluation.json"))
    second = _invoke_evaluate(*_evaluate_args("second-artifacts/evaluation.json"))

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert (tmp_path / "first-artifacts" / "evaluation.json").read_bytes() == (
        tmp_path / "second-artifacts" / "evaluation.json"
    ).read_bytes()


def test_evaluate_cli_altered_prediction_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _preprocess_artifact, inference_artifact = _create_validate_preprocess_infer(tmp_path)
    prediction_path = tmp_path / "predictions" / inference_artifact.prediction_paths[0]
    prediction = _load_array(prediction_path).astype(np.uint8, copy=True)
    prediction[0, 0, 0] = 1 - prediction[0, 0, 0]
    _write_nifti(prediction_path, prediction)
    monkeypatch.chdir(tmp_path)

    result = _invoke_evaluate(*_evaluate_args())
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "hash mismatch" in error_output
    assert not (tmp_path / "artifacts" / "evaluation.json").exists()


def test_evaluate_cli_inconsistent_artifact_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _preprocess_artifact, _inference_artifact = _create_validate_preprocess_infer(tmp_path)
    inference_path = tmp_path / "artifacts" / "inference.json"
    payload = json.loads(inference_path.read_text(encoding="utf-8"))
    payload["config_hash"] = "1" * 64
    inference_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = _invoke_evaluate(*_evaluate_args())
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "config hashes" in error_output
    assert not (tmp_path / "artifacts" / "evaluation.json").exists()


def test_evaluate_cli_existing_artifact_output_exits_nonzero_without_traceback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _preprocess_artifact, _inference_artifact = _create_validate_preprocess_infer(tmp_path)
    (tmp_path / "artifacts" / "evaluation.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = _invoke_evaluate(*_evaluate_args())
    error_output = _assert_nonzero_without_traceback(result, tmp_path)

    assert "overwrite" in error_output
