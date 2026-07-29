from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest

from protoem_ct.baselines.evaluation import (
    BaselineEvaluationFailure,
    BaselineEvaluationSerializationError,
    BaselineEvaluationValidationError,
    import_saved_baseline_predictions,
    load_reference_label_mapping_artifact,
)
from protoem_ct.baselines.metrics import baseline_metric_report_from_json
from protoem_ct.baselines.predictions import baseline_prediction_manifest_from_json
from protoem_ct.baselines.synthetic import (
    BASELINE_SYNTHETIC_DATASET_NAME,
    generate_baseline_synthetic_fixture,
)


def _fixture_mapping(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    fixture_root = tmp_path / "fixture-root"
    generate_baseline_synthetic_fixture(fixture_root)
    dataset_root = fixture_root / BASELINE_SYNTHETIC_DATASET_NAME
    mapping = {
        "synthetic_test_001": dataset_root / "labelsTs" / "synthetic_test_001.nii.gz",
        "synthetic_test_002": dataset_root / "labelsTs" / "synthetic_test_002.nii.gz",
    }
    return mapping, dataset_root


def _copy_prediction(label_path: Path, prediction_path: Path, *, empty: bool = False) -> None:
    image = cast(Any, nib.load(str(label_path)))
    label = np.asanyarray(image.dataobj).astype(np.uint8, copy=True)
    if empty:
        label.fill(0)
    copied = nib.Nifti1Image(label, image.affine, image.header.copy())  # type: ignore[no-untyped-call]
    nib.save(copied, str(prediction_path))


def _result_paths(tmp_path: Path) -> tuple[Path, Path]:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return outputs / "prediction_manifest.json", outputs / "metric_report.json"


def test_perfect_copied_predictions_and_deterministic_outputs(tmp_path: Path) -> None:
    mapping, _dataset_root = _fixture_mapping(tmp_path)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    for case_identifier, label_path in mapping.items():
        _copy_prediction(label_path, prediction_dir / f"{case_identifier}.nii.gz")
    manifest_output, report_output = _result_paths(tmp_path)

    result = import_saved_baseline_predictions(
        baseline_family="nnunet_v2",
        run_identifier="run_001",
        prediction_directory=prediction_dir,
        label_paths_by_case=mapping,
        dataset_manifest_sha256="1" * 64,
        development_split_sha256="2" * 64,
        checkpoint_sha256="3" * 64,
        metric_config_sha256="4" * 64,
        nsd_tolerance_mm=1.0,
        affine_tolerance_mm=0.0001,
        prediction_manifest_output_path=manifest_output,
        metric_report_output_path=report_output,
    )

    assert result.metric_report.aggregate is not None
    assert result.metric_report.aggregate.case_count == 2
    assert manifest_output.read_bytes() == manifest_output.read_bytes()
    assert report_output.read_bytes() == report_output.read_bytes()
    assert (
        baseline_prediction_manifest_from_json(manifest_output.read_bytes()).artifact_hash
        == result.prediction_manifest.artifact_hash
    )
    assert (
        baseline_metric_report_from_json(report_output.read_bytes()).artifact_hash
        == result.metric_report.artifact_hash
    )
    serialized = report_output.read_text(encoding="utf-8")
    assert str(prediction_dir) not in serialized
    assert str(tmp_path) not in serialized


def test_partial_and_empty_predictions(tmp_path: Path) -> None:
    mapping, _dataset_root = _fixture_mapping(tmp_path)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _copy_prediction(mapping["synthetic_test_001"], prediction_dir / "synthetic_test_001.nii.gz")
    _copy_prediction(
        mapping["synthetic_test_002"],
        prediction_dir / "synthetic_test_002.nii.gz",
        empty=True,
    )
    manifest_output, report_output = _result_paths(tmp_path)
    result = import_saved_baseline_predictions(
        baseline_family="monai_segresnet",
        run_identifier="run_002",
        prediction_directory=prediction_dir,
        label_paths_by_case=mapping,
        dataset_manifest_sha256="1" * 64,
        development_split_sha256="2" * 64,
        checkpoint_sha256="3" * 64,
        metric_config_sha256="4" * 64,
        nsd_tolerance_mm=1.0,
        affine_tolerance_mm=0.0001,
        prediction_manifest_output_path=manifest_output,
        metric_report_output_path=report_output,
    )

    assert result.metric_report.aggregate is not None
    assert result.metric_report.aggregate.macro_dice < 1.0


def test_missing_prediction_rejected(tmp_path: Path) -> None:
    mapping, _dataset_root = _fixture_mapping(tmp_path)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _copy_prediction(mapping["synthetic_test_001"], prediction_dir / "synthetic_test_001.nii.gz")
    manifest_output, report_output = _result_paths(tmp_path)

    with pytest.raises(BaselineEvaluationFailure) as exc_info:
        import_saved_baseline_predictions(
            baseline_family="nnunet_v2",
            run_identifier="run_001",
            prediction_directory=prediction_dir,
            label_paths_by_case=mapping,
            dataset_manifest_sha256="1" * 64,
            development_split_sha256="2" * 64,
            checkpoint_sha256="3" * 64,
            metric_config_sha256="4" * 64,
            nsd_tolerance_mm=1.0,
            affine_tolerance_mm=0.0001,
            prediction_manifest_output_path=manifest_output,
            metric_report_output_path=report_output,
        )

    assert exc_info.value.failure_code == "missing_prediction"


def test_unexpected_prediction_rejected(tmp_path: Path) -> None:
    mapping, _dataset_root = _fixture_mapping(tmp_path)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    for case_identifier, label_path in mapping.items():
        _copy_prediction(label_path, prediction_dir / f"{case_identifier}.nii.gz")
    _copy_prediction(next(iter(mapping.values())), prediction_dir / "unexpected_case.nii.gz")
    manifest_output, report_output = _result_paths(tmp_path)

    with pytest.raises(BaselineEvaluationFailure) as exc_info:
        import_saved_baseline_predictions(
            baseline_family="nnunet_v2",
            run_identifier="run_001",
            prediction_directory=prediction_dir,
            label_paths_by_case=mapping,
            dataset_manifest_sha256="1" * 64,
            development_split_sha256="2" * 64,
            checkpoint_sha256="3" * 64,
            metric_config_sha256="4" * 64,
            nsd_tolerance_mm=1.0,
            affine_tolerance_mm=0.0001,
            prediction_manifest_output_path=manifest_output,
            metric_report_output_path=report_output,
        )

    assert exc_info.value.failure_code == "unexpected_prediction"


def test_invalid_values_shape_affine_and_spacing_rejected(tmp_path: Path) -> None:
    mapping, _dataset_root = _fixture_mapping(tmp_path)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    first_label = cast(Any, nib.load(str(mapping["synthetic_test_001"])))
    invalid = np.asanyarray(first_label.dataobj).astype(np.float32, copy=True)
    invalid[0, 0, 0] = 0.5
    invalid_image = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        invalid,
        first_label.affine,
        first_label.header.copy(),
    )
    nib.save(invalid_image, str(prediction_dir / "synthetic_test_001.nii.gz"))
    _copy_prediction(mapping["synthetic_test_002"], prediction_dir / "synthetic_test_002.nii.gz")
    manifest_output, report_output = _result_paths(tmp_path)
    with pytest.raises(BaselineEvaluationFailure) as exc_info:
        import_saved_baseline_predictions(
            baseline_family="nnunet_v2",
            run_identifier="run_001",
            prediction_directory=prediction_dir,
            label_paths_by_case=mapping,
            dataset_manifest_sha256="1" * 64,
            development_split_sha256="2" * 64,
            checkpoint_sha256="3" * 64,
            metric_config_sha256="4" * 64,
            nsd_tolerance_mm=1.0,
            affine_tolerance_mm=0.0001,
            prediction_manifest_output_path=manifest_output,
            metric_report_output_path=report_output,
        )
    assert exc_info.value.failure_code == "invalid_prediction_values"

    prediction_dir = tmp_path / "predictions-shape"
    prediction_dir.mkdir()
    smaller = np.zeros((4, 4, 4), dtype=np.uint8)
    smaller_image = nib.Nifti1Image(smaller, np.eye(4), None)  # type: ignore[no-untyped-call]
    nib.save(smaller_image, str(prediction_dir / "synthetic_test_001.nii.gz"))
    _copy_prediction(mapping["synthetic_test_002"], prediction_dir / "synthetic_test_002.nii.gz")
    shape_outputs = tmp_path / "outputs-shape"
    shape_outputs.mkdir()
    with pytest.raises(BaselineEvaluationFailure) as exc_info:
        import_saved_baseline_predictions(
            baseline_family="nnunet_v2",
            run_identifier="run_001",
            prediction_directory=prediction_dir,
            label_paths_by_case=mapping,
            dataset_manifest_sha256="1" * 64,
            development_split_sha256="2" * 64,
            checkpoint_sha256="3" * 64,
            metric_config_sha256="4" * 64,
            nsd_tolerance_mm=1.0,
            affine_tolerance_mm=0.0001,
            prediction_manifest_output_path=shape_outputs / "manifest.json",
            metric_report_output_path=shape_outputs / "report.json",
        )
    assert exc_info.value.failure_code == "shape_mismatch"

    prediction_dir = tmp_path / "predictions-affine"
    prediction_dir.mkdir()
    modified_affine = first_label.affine.copy()
    modified_affine[0, 3] += 10.0
    shifted_image = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.asanyarray(first_label.dataobj).astype(np.uint8, copy=True),
        modified_affine,
        first_label.header.copy(),
    )
    nib.save(shifted_image, str(prediction_dir / "synthetic_test_001.nii.gz"))
    _copy_prediction(mapping["synthetic_test_002"], prediction_dir / "synthetic_test_002.nii.gz")
    affine_tmp = tmp_path / "affine"
    affine_tmp.mkdir()
    manifest_output, report_output = _result_paths(affine_tmp)
    with pytest.raises(BaselineEvaluationFailure) as exc_info:
        import_saved_baseline_predictions(
            baseline_family="nnunet_v2",
            run_identifier="run_001",
            prediction_directory=prediction_dir,
            label_paths_by_case=mapping,
            dataset_manifest_sha256="1" * 64,
            development_split_sha256="2" * 64,
            checkpoint_sha256="3" * 64,
            metric_config_sha256="4" * 64,
            nsd_tolerance_mm=1.0,
            affine_tolerance_mm=0.0001,
            prediction_manifest_output_path=manifest_output,
            metric_report_output_path=report_output,
        )
    assert exc_info.value.failure_code == "affine_mismatch"


def test_no_overwrite_and_cleanup_without_damaging_sources(tmp_path: Path) -> None:
    mapping, _dataset_root = _fixture_mapping(tmp_path)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    for case_identifier, label_path in mapping.items():
        _copy_prediction(label_path, prediction_dir / f"{case_identifier}.nii.gz")
    manifest_output, report_output = _result_paths(tmp_path)
    manifest_output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(BaselineEvaluationValidationError):
        import_saved_baseline_predictions(
            baseline_family="nnunet_v2",
            run_identifier="run_001",
            prediction_directory=prediction_dir,
            label_paths_by_case=mapping,
            dataset_manifest_sha256="1" * 64,
            development_split_sha256="2" * 64,
            checkpoint_sha256="3" * 64,
            metric_config_sha256="4" * 64,
            nsd_tolerance_mm=1.0,
            affine_tolerance_mm=0.0001,
            prediction_manifest_output_path=manifest_output,
            metric_report_output_path=report_output,
        )

    assert prediction_dir.joinpath("synthetic_test_001.nii.gz").exists()
    assert manifest_output.read_text(encoding="utf-8") == "existing\n"


def test_reference_mapping_artifact_parsing_and_duplicate_key_rejection(tmp_path: Path) -> None:
    mapping, _dataset_root = _fixture_mapping(tmp_path)
    mapping_artifact = tmp_path / "mapping.json"
    mapping_artifact.write_text(
        json.dumps(
            {
                "label_paths_by_case": {
                    case_identifier: str(path) for case_identifier, path in mapping.items()
                }
            }
        ),
        encoding="utf-8",
    )

    parsed = load_reference_label_mapping_artifact(mapping_artifact)
    assert set(parsed) == set(mapping)

    duplicate_mapping = tmp_path / "duplicate_mapping.json"
    duplicate_mapping.write_text(
        '{"label_paths_by_case":{"synthetic_test_001":"/tmp/a","synthetic_test_001":"/tmp/b"}}',
        encoding="utf-8",
    )
    with pytest.raises(BaselineEvaluationSerializationError):
        load_reference_label_mapping_artifact(duplicate_mapping)
