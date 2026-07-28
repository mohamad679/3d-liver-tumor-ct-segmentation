from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

import nibabel as nib
from typer.testing import CliRunner

from protoem_ct.baselines.synthetic import (
    BASELINE_SYNTHETIC_DATASET_NAME,
    BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS,
    BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS,
    baseline_synthetic_fixture_from_json,
)
from protoem_ct.cli.main import app


def _invoke(*args: str) -> Any:
    runner = CliRunner()
    return runner.invoke(app, ["generate-baseline-synthetic-fixture", *args])


def test_cli_generation_layout_and_aggregate_output(tmp_path: Path) -> None:
    output_root = tmp_path / "fixture-root"

    result = _invoke("--output-root", str(output_root))

    assert result.exit_code == 0
    assert "baseline synthetic fixture generation success" in result.output
    assert "contract version: baseline_synthetic_fixture_v1" in result.output
    assert f"dataset name: {BASELINE_SYNTHETIC_DATASET_NAME}" in result.output
    assert "training case count: 5" in result.output
    assert "test case count: 2" in result.output
    assert "total generated file count: 16" in result.output
    assert str(output_root) not in result.output

    dataset_root = output_root / BASELINE_SYNTHETIC_DATASET_NAME
    assert (dataset_root / "dataset.json").is_file()
    assert (output_root / "baseline_synthetic_fixture.json").is_file()
    for case_identifier in BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS:
        assert (dataset_root / "imagesTr" / f"{case_identifier}_0000.nii.gz").is_file()
        assert (dataset_root / "labelsTr" / f"{case_identifier}.nii.gz").is_file()
    for case_identifier in BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS:
        assert (dataset_root / "imagesTs" / f"{case_identifier}_0000.nii.gz").is_file()
        assert (dataset_root / "labelsTs" / f"{case_identifier}.nii.gz").is_file()


def test_second_invocation_same_destination_fails_without_modification(tmp_path: Path) -> None:
    output_root = tmp_path / "fixture-root"
    first = _invoke("--output-root", str(output_root))
    before_hash = (
        __import__("hashlib")
        .sha256((output_root / "baseline_synthetic_fixture.json").read_bytes())
        .hexdigest()
    )

    second = _invoke("--output-root", str(output_root))

    assert first.exit_code == 0
    assert second.exit_code != 0
    assert "Phase 3 baseline synthetic fixture error:" in (
        getattr(second, "stderr", "") or second.output
    )
    assert "Traceback" not in second.output
    after_hash = (
        __import__("hashlib")
        .sha256((output_root / "baseline_synthetic_fixture.json").read_bytes())
        .hexdigest()
    )
    assert before_hash == after_hash


def test_generated_dataset_json_exact_schema_and_nibabel_geometry(tmp_path: Path) -> None:
    output_root = tmp_path / "fixture-root"
    result = _invoke("--output-root", str(output_root))
    assert result.exit_code == 0

    dataset_json = json.loads(
        (output_root / BASELINE_SYNTHETIC_DATASET_NAME / "dataset.json").read_text(encoding="utf-8")
    )
    assert dataset_json == {
        "channel_names": {"0": "CT"},
        "file_ending": ".nii.gz",
        "labels": {"background": 0, "tumor": 1},
        "name": BASELINE_SYNTHETIC_DATASET_NAME,
        "numTest": 2,
        "numTraining": 5,
    }

    image_label_pairs = [
        (
            output_root
            / BASELINE_SYNTHETIC_DATASET_NAME
            / "imagesTr"
            / f"{case_identifier}_0000.nii.gz",
            output_root
            / BASELINE_SYNTHETIC_DATASET_NAME
            / "labelsTr"
            / f"{case_identifier}.nii.gz",
        )
        for case_identifier in BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS
    ] + [
        (
            output_root
            / BASELINE_SYNTHETIC_DATASET_NAME
            / "imagesTs"
            / f"{case_identifier}_0000.nii.gz",
            output_root
            / BASELINE_SYNTHETIC_DATASET_NAME
            / "labelsTs"
            / f"{case_identifier}.nii.gz",
        )
        for case_identifier in BASELINE_SYNTHETIC_TEST_CASE_IDENTIFIERS
    ]
    assert len(image_label_pairs) == 7

    for image_path, label_path in image_label_pairs:
        image = cast(Any, nib.load(str(image_path)))
        label = cast(Any, nib.load(str(label_path)))
        assert image.shape == label.shape == (24, 24, 16)
        assert tuple(float(value) for value in image.header.get_zooms()[:3]) == (
            1.0,
            1.0,
            2.5,
        )
        assert tuple(float(value) for value in label.header.get_zooms()[:3]) == (
            1.0,
            1.0,
            2.5,
        )
        assert image.get_qform(coded=True)[1] != 0
        assert label.get_qform(coded=True)[1] != 0
        assert image.get_sform(coded=True)[1] != 0
        assert label.get_sform(coded=True)[1] != 0


def test_recorded_hashes_match_files_and_artifact_parses(tmp_path: Path) -> None:
    output_root = tmp_path / "fixture-root"
    result = _invoke("--output-root", str(output_root))
    assert result.exit_code == 0

    artifact = baseline_synthetic_fixture_from_json(
        (output_root / "baseline_synthetic_fixture.json").read_bytes()
    )
    for file_record in artifact.file_records:
        path = output_root / file_record.relative_path
        assert path.is_file()
        assert __import__("hashlib").sha256(path.read_bytes()).hexdigest() == file_record.sha256
        assert path.stat().st_size == file_record.byte_size


def test_no_generated_file_appears_in_git_status(tmp_path: Path) -> None:
    output_root = tmp_path / "fixture-root"
    result = _invoke("--output-root", str(output_root))
    assert result.exit_code == 0

    git_status = subprocess.run(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            "*.nii",
            "*.nii.gz",
            "*.json",
            "*.csv",
            "*.tsv",
            "*.key",
            "*.log",
            "*.ckpt",
            "*.pth",
            "*.pt",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert git_status.returncode == 0
    assert git_status.stdout.strip() == ""
