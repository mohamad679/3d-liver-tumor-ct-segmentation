"""Integration tests for the Phase 7 robustness/uncertainty CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner, Result

from protoem_ct.cli.main import app
from protoem_ct.robustness.artifacts import PHASE7_CORRUPTION_NAMES
from protoem_ct.robustness.publication import PHASE7_PUBLICATION_FILENAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "phase7_robustness_uncertainty.yaml"


def _invoke(*args: str) -> Result:
    return CliRunner().invoke(app, [*args])


def test_cli_help_exposes_phase7_command() -> None:
    result = _invoke("--help")

    assert result.exit_code == 0
    assert "run-phase7-robustness-uncertainty" in result.output


def test_successful_bounded_synthetic_phase7_run(tmp_path: Path) -> None:
    output_root = tmp_path / "phase7-published"

    result = _invoke(
        "run-phase7-robustness-uncertainty",
        "--config",
        str(CONFIG_PATH),
        "--output-root",
        str(output_root),
    )

    assert result.exit_code == 0, result.output
    assert "Phase 7 robustness/uncertainty success" in result.output
    assert "mode: synthetic_only" in result.output
    assert "uncertainty type: tta_variance" in result.output
    assert {path.name for path in output_root.iterdir() if path.is_file()} == set(
        PHASE7_PUBLICATION_FILENAMES
    )


def test_all_required_corruption_families_are_represented(tmp_path: Path) -> None:
    output_root = tmp_path / "phase7-published"
    result = _invoke(
        "run-phase7-robustness-uncertainty",
        "--config",
        str(CONFIG_PATH),
        "--output-root",
        str(output_root),
    )
    assert result.exit_code == 0, result.output

    manifest = json.loads((output_root / "corruption_manifest.json").read_text("utf-8"))
    names = [item["corruption_name"] for item in manifest["specifications"]]
    assert names == [
        "hu_window_shift",
        "intensity_scale",
        "intensity_offset",
        "contrast_shift",
        "gaussian_noise",
        "gaussian_blur",
        "slice_thickness",
        "anisotropic_downsampling",
        "crop_fov",
    ]
    assert set(names) == PHASE7_CORRUPTION_NAMES


def test_json_and_markdown_are_deterministic_across_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "run-a"
    root_b = tmp_path / "run-b"
    for output_root in (root_a, root_b):
        result = _invoke(
            "run-phase7-robustness-uncertainty",
            "--config",
            str(CONFIG_PATH),
            "--output-root",
            str(output_root),
        )
        assert result.exit_code == 0, result.output

    for filename in PHASE7_PUBLICATION_FILENAMES:
        if filename.endswith((".json", ".md")):
            assert (root_a / filename).read_bytes() == (root_b / filename).read_bytes()
        elif filename.endswith(".png"):
            assert (root_a / filename).stat().st_size > 0
            assert (root_b / filename).stat().st_size > 0


def test_safe_idempotent_regeneration(tmp_path: Path) -> None:
    output_root = tmp_path / "phase7-published"
    first = _invoke(
        "run-phase7-robustness-uncertainty",
        "--config",
        str(CONFIG_PATH),
        "--output-root",
        str(output_root),
    )
    second = _invoke(
        "run-phase7-robustness-uncertainty",
        "--config",
        str(CONFIG_PATH),
        "--output-root",
        str(output_root),
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "reused existing output: true" in second.output


def test_query_label_and_reference_mask_are_isolated_to_evaluation(tmp_path: Path) -> None:
    output_root = tmp_path / "phase7-published"
    result = _invoke(
        "run-phase7-robustness-uncertainty",
        "--config",
        str(CONFIG_PATH),
        "--output-root",
        str(output_root),
    )

    assert result.exit_code == 0, result.output
    manifest = (output_root / "corruption_manifest.json").read_text("utf-8")
    uncertainty = (output_root / "uncertainty_result.json").read_text("utf-8")
    assert "query_label" not in manifest
    assert "reference_mask" not in manifest
    assert "query_label" not in uncertainty
    assert "reference_mask" not in uncertainty
    assert "reference_mask_content_hash" in (output_root / "calibration_result.json").read_text(
        "utf-8"
    )


def test_no_generated_phase7_artifacts_inside_repository(tmp_path: Path) -> None:
    result = _invoke(
        "run-phase7-robustness-uncertainty",
        "--config",
        str(CONFIG_PATH),
        "--output-root",
        str(tmp_path / "phase7-published"),
    )

    assert result.exit_code == 0, result.output
    for filename in PHASE7_PUBLICATION_FILENAMES:
        assert not (REPO_ROOT / filename).exists()
