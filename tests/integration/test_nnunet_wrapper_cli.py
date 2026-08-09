from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
from typer.testing import CliRunner

from protoem_ct.baselines import (
    BASELINE_SYNTHETIC_DATASET_NAME,
    BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS,
    baseline_metric_report_from_json,
    baseline_prediction_manifest_from_json,
    build_nnunet_v2_run_config,
    generate_baseline_synthetic_fixture,
    nnunet_v2_run_config_to_json,
)
from protoem_ct.cli.main import app

RUNNER = CliRunner()


def _copy_prediction(label_path: Path, prediction_path: Path) -> None:
    image = cast(Any, nib.load(str(label_path)))
    copied = nib.Nifti1Image(
        np.asanyarray(image.dataobj).astype("uint8", copy=True),
        image.affine,
        image.header.copy(),
    )  # type: ignore[no-untyped-call]
    nib.save(copied, str(prediction_path))


def test_inspect_nnunet_v2_baseline_plan_cli(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture-root"
    generate_baseline_synthetic_fixture(fixture_root)
    run_config = build_nnunet_v2_run_config(seed=17)
    run_config_path = tmp_path / "run-config.json"
    run_config_path.write_bytes(nnunet_v2_run_config_to_json(run_config))
    run_parent = tmp_path / "runs"
    run_parent.mkdir()
    run_root = run_parent / "run-001"

    result = RUNNER.invoke(
        app,
        [
            "inspect-nnunet-v2-baseline-plan",
            "--run-config",
            str(run_config_path),
            "--runtime-raw-root",
            str(fixture_root),
            "--runtime-run-root",
            str(run_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "contract version: nnunet_v2_run_config_v1" in result.output
    assert "planning executable: nnUNetv2_plan_and_preprocess" in result.output
    assert "training executable: nnUNetv2_train" in result.output
    assert "prediction executable: nnUNetv2_predict" in result.output
    assert str(fixture_root) not in result.output
    assert str(run_root) not in result.output


def test_import_baseline_predictions_cli_and_git_visibility(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture-root"
    generate_baseline_synthetic_fixture(fixture_root)
    dataset_root = fixture_root / BASELINE_SYNTHETIC_DATASET_NAME
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    for case_identifier in BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS:
        label_path = dataset_root / "labelsTs" / f"{case_identifier}.nii.gz"
        _copy_prediction(label_path, prediction_dir / f"{case_identifier}.nii.gz")
    mapping_artifact = tmp_path / "reference_mapping.json"
    mapping_artifact.write_text(
        json.dumps(
            {
                "label_paths_by_case": {
                    case_identifier: str(dataset_root / "labelsTs" / f"{case_identifier}.nii.gz")
                    for case_identifier in BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS
                }
            }
        ),
        encoding="utf-8",
    )
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    prediction_manifest_output = outputs / "prediction_manifest.json"
    metric_report_output = outputs / "metric_report.json"

    result = RUNNER.invoke(
        app,
        [
            "import-baseline-predictions",
            "--baseline-family",
            "nnunet_v2",
            "--run-identifier",
            "run_001",
            "--prediction-directory",
            str(prediction_dir),
            "--reference-mapping-artifact",
            str(mapping_artifact),
            "--dataset-manifest-sha256",
            "1" * 64,
            "--development-split-sha256",
            "2" * 64,
            "--checkpoint-sha256",
            "3" * 64,
            "--metric-config-sha256",
            "4" * 64,
            "--nsd-tolerance-mm",
            "1.0",
            "--affine-tolerance-mm",
            "0.0001",
            "--prediction-manifest-output",
            str(prediction_manifest_output),
            "--metric-report-output",
            str(metric_report_output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "manifest version: baseline_prediction_manifest_v1" in result.output
    manifest = baseline_prediction_manifest_from_json(prediction_manifest_output.read_bytes())
    report = baseline_metric_report_from_json(metric_report_output.read_bytes())
    assert manifest.artifact_hash
    assert report.artifact_hash
    git_status = subprocess.run(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            "*.nii",
            "*.nii.gz",
            "*.dcm",
            "*.json",
            "*.csv",
            "*.tsv",
            "*.key",
            "*.log",
            "*.ckpt",
            "*.pth",
            "*.pt",
            ":(exclude)reports/**",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert git_status.stdout.strip() == ""
    second_result = RUNNER.invoke(
        app,
        [
            "import-baseline-predictions",
            "--baseline-family",
            "nnunet_v2",
            "--run-identifier",
            "run_001",
            "--prediction-directory",
            str(prediction_dir),
            "--reference-mapping-artifact",
            str(mapping_artifact),
            "--dataset-manifest-sha256",
            "1" * 64,
            "--development-split-sha256",
            "2" * 64,
            "--checkpoint-sha256",
            "3" * 64,
            "--metric-config-sha256",
            "4" * 64,
            "--nsd-tolerance-mm",
            "1.0",
            "--affine-tolerance-mm",
            "0.0001",
            "--prediction-manifest-output",
            str(prediction_manifest_output),
            "--metric-report-output",
            str(metric_report_output),
        ],
    )
    assert second_result.exit_code != 0
