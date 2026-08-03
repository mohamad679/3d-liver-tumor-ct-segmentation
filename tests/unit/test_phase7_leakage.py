"""Phase 7 leakage and scope-boundary tests."""

from __future__ import annotations

import inspect
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import typer
from typer.testing import CliRunner

from protoem_ct.cli.main import (
    app,
    load_phase7_robustness_uncertainty_settings,
    run_and_publish_phase7_robustness_uncertainty,
)
from protoem_ct.evaluation.calibration import compute_binary_calibration_ece
from protoem_ct.evaluation.degradation import query_label_not_part_of_phase7_degradation_api
from protoem_ct.evaluation.failure_detection import (
    compute_failure_detection_auroc,
    compute_uncertainty_error_correlation,
    failure_detection_public_prediction_api_has_no_labels,
    query_label_not_part_of_failure_detection_prediction_api,
)
from protoem_ct.evaluation.risk_coverage import (
    compute_risk_coverage,
    derive_hard_prediction_from_binary_probabilities,
    query_label_not_part_of_risk_coverage_prediction_api,
)
from protoem_ct.evaluation.subgroups import (
    LesionSubgroupCase,
    query_label_not_part_of_phase7_subgroup_prediction_api,
)
from protoem_ct.protoem.inference import (
    build_protoem_final_inference_result,
    query_label_not_part_of_final_inference_api,
)
from protoem_ct.protoem.objective import (
    compute_protoem_objective_terms,
    run_protoem_e_step,
    run_protoem_m_step,
)
from protoem_ct.protoem.optimize import run_protoem_optimization
from protoem_ct.robustness.artifacts import PHASE7_CORRUPTION_NAMES
from protoem_ct.robustness.blur import apply_gaussian_blur
from protoem_ct.robustness.crop import (
    apply_crop_fov_perturbation,
    crop_fov_public_api_excludes_query_labels,
)
from protoem_ct.robustness.intensity import (
    apply_intensity_transform,
    query_label_not_part_of_intensity_transform_api,
)
from protoem_ct.robustness.noise import apply_gaussian_noise
from protoem_ct.robustness.publication import PHASE7_PUBLICATION_FILENAMES
from protoem_ct.robustness.resampling import (
    apply_anisotropic_downsampling,
    apply_resampling_transform,
    apply_slice_thickness_simulation,
    query_label_not_part_of_resampling_api,
)
from protoem_ct.uncertainty.contracts import query_label_not_part_of_uncertainty_contracts_api
from protoem_ct.uncertainty.entropy import (
    compute_predictive_entropy,
    query_label_not_part_of_predictive_entropy_api,
)
from protoem_ct.uncertainty.tta import (
    compute_ensemble_probability_variance,
    compute_tta_probability_variance,
    query_label_not_part_of_tta_api,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "phase7_robustness_uncertainty.yaml"

_PHASE7_SOURCE_SURFACES = (
    CONFIG_PATH,
    REPO_ROOT / "src/protoem_ct/robustness/artifacts.py",
    REPO_ROOT / "src/protoem_ct/robustness/geometry.py",
    REPO_ROOT / "src/protoem_ct/robustness/intensity.py",
    REPO_ROOT / "src/protoem_ct/robustness/noise.py",
    REPO_ROOT / "src/protoem_ct/robustness/blur.py",
    REPO_ROOT / "src/protoem_ct/robustness/resampling.py",
    REPO_ROOT / "src/protoem_ct/robustness/crop.py",
    REPO_ROOT / "src/protoem_ct/robustness/publication.py",
    REPO_ROOT / "src/protoem_ct/uncertainty/artifacts.py",
    REPO_ROOT / "src/protoem_ct/uncertainty/contracts.py",
    REPO_ROOT / "src/protoem_ct/uncertainty/entropy.py",
    REPO_ROOT / "src/protoem_ct/uncertainty/tta.py",
    REPO_ROOT / "src/protoem_ct/uncertainty/publication.py",
    REPO_ROOT / "src/protoem_ct/evaluation/calibration.py",
    REPO_ROOT / "src/protoem_ct/evaluation/risk_coverage.py",
    REPO_ROOT / "src/protoem_ct/evaluation/failure_detection.py",
    REPO_ROOT / "src/protoem_ct/evaluation/degradation.py",
    REPO_ROOT / "src/protoem_ct/evaluation/subgroups.py",
)

_FORBIDDEN_PHASE7_SCOPE_TOKENS = (
    "3d-ircadb",
    "ircad",
    "phase 8",
    "phase8",
    "llm",
    "vlm",
    "checkpoint",
    "download",
    "private data",
    "identifiable",
)
_FORBIDDEN_PHASE7_SCOPE_PATTERNS = tuple(
    re.compile(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])")
    for token in _FORBIDDEN_PHASE7_SCOPE_TOKENS
)

_FORBIDDEN_PREDICTION_API_FIELDS = frozenset(
    {
        "query_label",
        "query_labels",
        "label",
        "labels",
        "label_map",
        "reference_mask",
        "query_reference_mask",
        "model",
        "model_callable",
        "predictor",
        "predict_fn",
        "checkpoint",
        "weights",
        "dataset_root",
        "download_root",
    }
)


def test_phase7_config_and_command_are_synthetic_only_without_external_phase8_surface() -> None:
    settings = load_phase7_robustness_uncertainty_settings(CONFIG_PATH)
    root = settings.effective_config_mapping["phase7_robustness_uncertainty"]

    assert isinstance(root, dict)
    assert root["synthetic_mode_only"] is True
    assert tuple(spec.corruption_name for spec in settings.corruption_specs) == (
        "hu_window_shift",
        "intensity_scale",
        "intensity_offset",
        "contrast_shift",
        "gaussian_noise",
        "gaussian_blur",
        "slice_thickness",
        "anisotropic_downsampling",
        "crop_fov",
    )
    assert {spec.corruption_name for spec in settings.corruption_specs} == PHASE7_CORRUPTION_NAMES

    help_result = CliRunner().invoke(app, ["run-phase7-robustness-uncertainty", "--help"])
    assert help_result.exit_code == 0, help_result.output
    normalized_help = help_result.output.lower()
    for forbidden in _FORBIDDEN_PHASE7_SCOPE_TOKENS:
        assert forbidden not in normalized_help
    assert "synthetic" in normalized_help
    _assert_phase7_command_exposes_required_options()


def test_phase7_help_rendering_remains_leakage_safe_with_color_and_narrow_width() -> None:
    for result in (
        CliRunner().invoke(
            app,
            ["run-phase7-robustness-uncertainty", "--help"],
            color=False,
            terminal_width=100,
        ),
        CliRunner().invoke(
            app,
            ["run-phase7-robustness-uncertainty", "--help"],
            color=True,
            terminal_width=100,
        ),
        CliRunner(env={"COLUMNS": "40"}).invoke(
            app,
            ["run-phase7-robustness-uncertainty", "--help"],
            color=True,
            terminal_width=40,
        ),
    ):
        assert result.exit_code == 0, repr(result.stdout)
        normalized_help = result.stdout.lower()
        assert "synthetic" in normalized_help
        for forbidden in _FORBIDDEN_PHASE7_SCOPE_TOKENS:
            assert forbidden not in normalized_help

    _assert_phase7_command_exposes_required_options()


def test_phase7_source_surfaces_do_not_expose_phase8_or_download_pathways() -> None:
    for path in _PHASE7_SOURCE_SURFACES:
        normalized = path.read_text("utf-8").lower()
        for pattern in _FORBIDDEN_PHASE7_SCOPE_PATTERNS:
            assert pattern.search(normalized) is None, f"{pattern.pattern!r} appeared in {path}"


def test_corruption_and_uncertainty_prediction_apis_have_no_labels_models_or_checkpoints() -> None:
    allowed_mask_companion_apis: tuple[Callable[..., object], ...] = (
        apply_intensity_transform,
        apply_gaussian_noise,
        apply_gaussian_blur,
        apply_slice_thickness_simulation,
        apply_anisotropic_downsampling,
        apply_resampling_transform,
        apply_crop_fov_perturbation,
    )
    for callable_item in allowed_mask_companion_apis:
        _assert_signature_excludes_forbidden_prediction_fields(
            callable_item,
            allowed_fields=frozenset({"mask", "mask_grid", "evaluation_mask", "spatial_grid"}),
        )

    uncertainty_prediction_apis: tuple[Callable[..., object], ...] = (
        compute_predictive_entropy,
        compute_tta_probability_variance,
        compute_ensemble_probability_variance,
    )
    for callable_item in uncertainty_prediction_apis:
        _assert_signature_excludes_forbidden_prediction_fields(callable_item)

    assert query_label_not_part_of_intensity_transform_api()
    assert query_label_not_part_of_resampling_api()
    assert crop_fov_public_api_excludes_query_labels()
    assert query_label_not_part_of_uncertainty_contracts_api()
    assert query_label_not_part_of_predictive_entropy_api()
    assert query_label_not_part_of_tta_api()


def test_phase6_optimization_and_phase7_final_prediction_surfaces_remain_label_free() -> None:
    for callable_item in (
        run_protoem_optimization,
        run_protoem_e_step,
        run_protoem_m_step,
        compute_protoem_objective_terms,
        build_protoem_final_inference_result,
        run_and_publish_phase7_robustness_uncertainty,
    ):
        _assert_signature_excludes_forbidden_prediction_fields(callable_item)

    assert query_label_not_part_of_final_inference_api()


def test_reference_masks_are_limited_to_post_prediction_evaluation_surfaces() -> None:
    assert "reference_mask" in inspect.signature(compute_binary_calibration_ece).parameters
    assert "reference_mask" in inspect.signature(compute_risk_coverage).parameters
    assert "reference_mask" in LesionSubgroupCase.__dataclass_fields__

    for callable_item in (
        derive_hard_prediction_from_binary_probabilities,
        compute_uncertainty_error_correlation,
        compute_failure_detection_auroc,
    ):
        _assert_signature_excludes_forbidden_prediction_fields(callable_item)

    assert query_label_not_part_of_risk_coverage_prediction_api()
    assert query_label_not_part_of_failure_detection_prediction_api()
    assert failure_detection_public_prediction_api_has_no_labels()
    assert query_label_not_part_of_phase7_degradation_api()
    assert query_label_not_part_of_phase7_subgroup_prediction_api()


def test_bounded_synthetic_cli_run_does_not_create_git_visible_artifacts(tmp_path: Path) -> None:
    before = _git_status_porcelain()
    output_root = tmp_path / "phase7-leakage-synthetic"

    result = CliRunner().invoke(
        app,
        [
            "run-phase7-robustness-uncertainty",
            "--config",
            str(CONFIG_PATH),
            "--output-root",
            str(output_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert {path.name for path in output_root.iterdir() if path.is_file()} == set(
        PHASE7_PUBLICATION_FILENAMES
    )
    assert _git_status_porcelain() == before
    for filename in PHASE7_PUBLICATION_FILENAMES:
        assert not (REPO_ROOT / filename).exists()


def _assert_signature_excludes_forbidden_prediction_fields(
    callable_item: Callable[..., object],
    *,
    allowed_fields: frozenset[str] = frozenset(),
) -> None:
    exposed_fields = set(inspect.signature(callable_item).parameters)
    forbidden = _FORBIDDEN_PREDICTION_API_FIELDS - allowed_fields
    assert exposed_fields.isdisjoint(forbidden), (
        f"{callable_item.__module__}.{callable_item.__name__} exposes "
        f"{sorted(exposed_fields & forbidden)!r}"
    )


def _assert_phase7_command_exposes_required_options() -> None:
    command = cast(Any, typer.main.get_command(app))
    commands = cast(dict[str, Any], command.commands)
    phase7_command = commands.get("run-phase7-robustness-uncertainty")
    assert phase7_command is not None
    options_by_name: dict[str, Any] = {}
    for parameter in cast(Sequence[Any], cast(Any, phase7_command).params):
        if hasattr(parameter, "opts") and hasattr(parameter, "required"):
            options_by_name[cast(str, parameter.name)] = parameter
    assert "--config" in cast(Sequence[str], options_by_name["config_path"].opts)
    assert "--output-root" in cast(Sequence[str], options_by_name["output_root"].opts)
    assert options_by_name["output_root"].required is True


def _git_status_porcelain() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line)
