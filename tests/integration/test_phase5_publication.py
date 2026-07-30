"""Integration tests for deterministic Phase 5 publication."""

from __future__ import annotations

import json
from pathlib import Path

from protoem_ct.retrieval import (
    PHASE5_COMPARISON_JSON_NAME,
    PHASE5_COMPARISON_MARKDOWN_NAME,
    PHASE5_EFFECTIVE_CONFIG_JSON_NAME,
    PHASE5_RUN_SUMMARY_JSON_NAME,
    load_phase5_foundation_retrieval_settings,
    phase5_comparison_table_from_json,
    run_and_publish_phase5_retrieval,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_deterministic_publication_across_two_output_roots(tmp_path: Path) -> None:
    settings = load_phase5_foundation_retrieval_settings(
        REPO_ROOT / "configs" / "phase5_foundation_retrieval.yaml"
    )
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    tracked_before = {
        path.relative_to(REPO_ROOT).as_posix(): path.read_bytes()
        for directory_name in ("data", "reports", "src")
        for path in (REPO_ROOT / directory_name).rglob("*")
        if path.is_file() and path.suffix in {".json", ".md", ".py", ".yaml"}
    }

    first = run_and_publish_phase5_retrieval(output_root=first_root, settings=settings)
    second = run_and_publish_phase5_retrieval(output_root=second_root, settings=settings)

    assert first.reused_existing_output is False
    assert second.reused_existing_output is False
    first_table = phase5_comparison_table_from_json(
        (first_root / PHASE5_COMPARISON_JSON_NAME).read_bytes()
    )
    second_table = phase5_comparison_table_from_json(
        (second_root / PHASE5_COMPARISON_JSON_NAME).read_bytes()
    )
    assert len(first_table.records) == 2
    assert len(second_table.records) == 2
    assert (
        sum(
            1
            for line in (first_root / PHASE5_COMPARISON_MARKDOWN_NAME)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.startswith("| ")
        )
        - 2
    ) == 2
    assert json.loads(
        (first_root / PHASE5_RUN_SUMMARY_JSON_NAME).read_text(encoding="utf-8")
    ) == json.loads((second_root / PHASE5_RUN_SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
    assert {
        path.relative_to(first_root).as_posix(): path.read_bytes()
        for path in first_root.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second_root).as_posix(): path.read_bytes()
        for path in second_root.rglob("*")
        if path.is_file()
    }
    assert (first_root / PHASE5_EFFECTIVE_CONFIG_JSON_NAME).exists()

    tracked_after = {
        path.relative_to(REPO_ROOT).as_posix(): path.read_bytes()
        for directory_name in ("data", "reports", "src")
        for path in (REPO_ROOT / directory_name).rglob("*")
        if path.is_file() and path.suffix in {".json", ".md", ".py", ".yaml"}
    }
    assert tracked_before == tracked_after
