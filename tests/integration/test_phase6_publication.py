"""Integration tests for deterministic Phase 6 publication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from protoem_ct.protoem import (
    PHASE6_ABLATION_PLAN_NAME,
    PHASE6_CONVERGENCE_PLOT_NAME,
    PHASE6_EFFECTIVE_CONFIG_NAME,
    PHASE6_FINAL_INFERENCE_NAME,
    PHASE6_OBJECTIVE_TRACE_NAME,
    PHASE6_RUN_SUMMARY_NAME,
    PHASE6_SUMMARY_MARKDOWN_NAME,
    Phase6PublicationCollisionError,
    Phase6PublicationPathError,
    build_phase6_convergence_plot_png,
    load_phase6_protoem_settings,
    run_and_publish_phase6_protoem,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_markdown_and_json_are_deterministic_across_two_output_roots(tmp_path: Path) -> None:
    settings = load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = run_and_publish_phase6_protoem(output_root=first_root, settings=settings)
    second = run_and_publish_phase6_protoem(output_root=second_root, settings=settings)

    assert first.reused_existing_output is False
    assert second.reused_existing_output is False
    first_files = {
        path.relative_to(first_root).as_posix(): path.read_bytes()
        for path in first_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".md"}
    }
    second_files = {
        path.relative_to(second_root).as_posix(): path.read_bytes()
        for path in second_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".md"}
    }
    assert first_files == second_files
    assert (first_root / PHASE6_CONVERGENCE_PLOT_NAME).exists()
    assert (second_root / PHASE6_CONVERGENCE_PLOT_NAME).exists()


def test_markdown_is_derived_from_json(tmp_path: Path) -> None:
    settings = load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")
    output_root = tmp_path / "published"

    _ = run_and_publish_phase6_protoem(output_root=output_root, settings=settings)

    run_summary = json.loads((output_root / PHASE6_RUN_SUMMARY_NAME).read_text(encoding="utf-8"))
    markdown = (output_root / PHASE6_SUMMARY_MARKDOWN_NAME).read_text(encoding="utf-8")
    assert run_summary["config_hash"] in markdown
    assert run_summary["execution_status"] in markdown


def test_plot_is_derived_from_trace_json(tmp_path: Path) -> None:
    settings = load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")
    output_root = tmp_path / "published"

    _ = run_and_publish_phase6_protoem(output_root=output_root, settings=settings)

    trace_bytes = (output_root / PHASE6_OBJECTIVE_TRACE_NAME).read_bytes()
    regenerated = build_phase6_convergence_plot_png(trace_bytes)
    assert regenerated == (output_root / PHASE6_CONVERGENCE_PLOT_NAME).read_bytes()


def test_safe_idempotent_regeneration(tmp_path: Path) -> None:
    settings = load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")
    output_root = tmp_path / "published"

    first = run_and_publish_phase6_protoem(output_root=output_root, settings=settings)
    second = run_and_publish_phase6_protoem(output_root=output_root, settings=settings)

    assert first.reused_existing_output is False
    assert second.reused_existing_output is True


def test_incompatible_overwrite_rejection(tmp_path: Path) -> None:
    settings = load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")
    output_root = tmp_path / "published"
    output_root.mkdir()
    (output_root / PHASE6_EFFECTIVE_CONFIG_NAME).write_text("not canonical\n", encoding="utf-8")

    with pytest.raises(Phase6PublicationCollisionError):
        run_and_publish_phase6_protoem(output_root=output_root, settings=settings)


def test_repository_internal_root_rejection() -> None:
    settings = load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")

    with pytest.raises(Phase6PublicationPathError):
        run_and_publish_phase6_protoem(
            output_root=REPO_ROOT / "src" / "phase6-forbidden-output",
            settings=settings,
        )


def test_parent_traversal_output_root_rejection(tmp_path: Path) -> None:
    settings = load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")
    traversal_root = tmp_path / ".." / "escape"

    with pytest.raises(Phase6PublicationPathError):
        run_and_publish_phase6_protoem(output_root=traversal_root, settings=settings)


def test_symlink_escape_rejection(tmp_path: Path) -> None:
    settings = load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")
    external_target = tmp_path / "external"
    external_target.mkdir()
    symlink_root = tmp_path / "symlink-root"
    symlink_root.symlink_to(external_target, target_is_directory=True)

    with pytest.raises(Phase6PublicationPathError):
        run_and_publish_phase6_protoem(output_root=symlink_root / "nested", settings=settings)


def test_ablation_plan_and_final_inference_files_exist(tmp_path: Path) -> None:
    settings = load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")
    output_root = tmp_path / "published"

    _ = run_and_publish_phase6_protoem(output_root=output_root, settings=settings)

    assert (output_root / PHASE6_ABLATION_PLAN_NAME).exists()
    assert (output_root / PHASE6_FINAL_INFERENCE_NAME).exists()
