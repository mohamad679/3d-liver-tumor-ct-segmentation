"""Integration tests for deterministic Phase 6 ablation execution and publication flow."""

from __future__ import annotations

import json
from pathlib import Path

from protoem_ct.protoem import (
    PHASE6_ABLATION_COMPARISON_JSON_NAME,
    PHASE6_ABLATION_COMPARISON_MARKDOWN_NAME,
    PHASE6_ABLATION_RUN_INVENTORY_JSON_NAME,
    PROTOEM_ABLATION_VARIANT_ORDER,
    load_phase6_protoem_settings,
    run_and_publish_phase6_protoem,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_phase6_ablation_outputs_use_canonical_order_and_json_derived_markdown(
    tmp_path: Path,
) -> None:
    settings = load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")
    output_root = tmp_path / "published"

    _ = run_and_publish_phase6_protoem(output_root=output_root, settings=settings)

    comparison = json.loads(
        (output_root / PHASE6_ABLATION_COMPARISON_JSON_NAME).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (output_root / PHASE6_ABLATION_RUN_INVENTORY_JSON_NAME).read_text(encoding="utf-8")
    )
    markdown = (output_root / PHASE6_ABLATION_COMPARISON_MARKDOWN_NAME).read_text(encoding="utf-8")

    assert [record["variant_name"] for record in comparison["records"]] == list(
        PROTOEM_ABLATION_VARIANT_ORDER
    )
    assert [record["variant_name"] for record in inventory["records"]] == list(
        PROTOEM_ABLATION_VARIANT_ORDER
    )
    assert comparison["records"][0]["variant_name"] in markdown
    assert comparison["records"][0]["execution_status"] in markdown


def test_repeated_publication_is_byte_identical_for_comparison_outputs(tmp_path: Path) -> None:
    settings = load_phase6_protoem_settings(REPO_ROOT / "configs" / "phase6_protoem_ct.yaml")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    _ = run_and_publish_phase6_protoem(output_root=first_root, settings=settings)
    _ = run_and_publish_phase6_protoem(output_root=second_root, settings=settings)

    for filename in (
        PHASE6_ABLATION_COMPARISON_JSON_NAME,
        PHASE6_ABLATION_COMPARISON_MARKDOWN_NAME,
        PHASE6_ABLATION_RUN_INVENTORY_JSON_NAME,
    ):
        assert (first_root / filename).read_bytes() == (second_root / filename).read_bytes()
