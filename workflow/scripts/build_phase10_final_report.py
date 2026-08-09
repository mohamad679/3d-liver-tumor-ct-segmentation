# ruff: noqa: E501
"""Render the Phase 10 final report and reproduction README from artifacts.

Inputs are the P10-A inventory and P10-B table/figure/statistics manifest.
The renderer reads machine-readable JSON/CSV/SVG outputs only; it does not
open medical images, predictions, checkpoints, or raw labels.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from build_phase10_artifact_inventory import sha256_file, sha256_json

SCHEMA_VERSION = "v1"


class Phase10ReportError(RuntimeError):
    """Raised when required report source artifacts are missing or inconsistent."""


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    if not path.exists():
        raise Phase10ReportError(f"Missing report source JSON: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise Phase10ReportError(f"Expected JSON object at {path}")
    return loaded


def write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write deterministic JSON with self-hash and return that hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    full_payload = dict(payload)
    full_payload["self_hash"] = sha256_json(full_payload)
    path.write_text(json.dumps(full_payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return str(full_payload["self_hash"])


def table_by_id(manifest: dict[str, Any], table_id: str) -> dict[str, Any]:
    """Load a P10-B table JSON by table id."""
    for table in manifest.get("tables", []):
        if table.get("id") == table_id:
            return load_json(Path(str(table["json_path"])))
    raise Phase10ReportError(f"Missing table in P10-B manifest: {table_id}")


def figure_by_id(manifest: dict[str, Any], figure_id: str) -> dict[str, Any]:
    """Return a P10-B figure manifest entry by id."""
    for figure in manifest.get("figures", []):
        if figure.get("id") == figure_id:
            return figure
    raise Phase10ReportError(f"Missing figure in P10-B manifest: {figure_id}")


def statistic_by_id(manifest: dict[str, Any], statistic_id: str) -> dict[str, Any]:
    """Load a P10-B statistics JSON by id."""
    for statistic in manifest.get("statistics", []):
        if statistic.get("id") == statistic_id:
            return load_json(Path(str(statistic["json_path"])))
    raise Phase10ReportError(f"Missing statistics artifact in P10-B manifest: {statistic_id}")


def row_by_label(table: dict[str, Any], label: str) -> dict[str, Any]:
    """Find one table row by label."""
    for row in table.get("rows", []):
        if row.get("label") == label:
            return row
    raise Phase10ReportError(f"Missing row {label!r} in table {table.get('table_id')}")


def row_value(table: dict[str, Any], label: str, field: str = "value") -> str:
    """Return a table row field as a string."""
    return str(row_by_label(table, label)[field])


def render_table_markdown(table: dict[str, Any], *, max_rows: int | None = None) -> str:
    """Render one table artifact to Markdown."""
    columns = [str(column) for column in table["columns"]]
    rows = table["rows"][:max_rows] if max_rows is not None else table["rows"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def render_table_html(table: dict[str, Any], *, max_rows: int | None = None) -> str:
    """Render one table artifact to HTML."""
    columns = [str(column) for column in table["columns"]]
    rows = table["rows"][:max_rows] if max_rows is not None else table["rows"]
    parts = ["<table><thead><tr>"]
    for column in columns:
        parts.append(f"<th>{html.escape(column)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for column in columns:
            parts.append(f"<td>{html.escape(str(row.get(column, '')))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_report_markdown(
    inventory: dict[str, Any], b_manifest: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Render final technical report Markdown and return numeric provenance."""
    cohort = table_by_id(b_manifest, "phase10_cohort_data_summary")
    external = table_by_id(b_manifest, "phase10_external_validation_metrics")
    comparison = table_by_id(b_manifest, "phase10_internal_external_descriptive_comparison")
    unavailable = table_by_id(b_manifest, "phase10_unavailable_results_and_limitations")
    policy = statistic_by_id(b_manifest, "phase10_statistical_policy")

    numeric_sources: list[dict[str, Any]] = []
    for table in (cohort, external, comparison):
        for row in table.get("rows", []):
            numeric_sources.append(
                {
                    "table_id": table["table_id"],
                    "row": row,
                    "source_inventory_hash": table["source_inventory_hash"],
                }
            )
    for field_path in (
        "bootstrap_policy.random_seed",
        "bootstrap_policy.resample_count",
        "bootstrap_policy.confidence_level",
    ):
        current: Any = policy
        for part in field_path.split("."):
            current = current[part]
        numeric_sources.append(
            {
                "statistics_id": "phase10_statistical_policy",
                "field": field_path,
                "value": current,
                "source_artifact": "reports/statistics/phase10_statistical_policy.json",
                "source_inventory_hash": policy["source_inventory_hash"],
            }
        )

    lines = [
        "# ProtoEM-CT Final Scientific Package",
        "",
        "Status: Phase 10 final scientific package. Phase 9 was skipped by user decision and was not executed.",
        "",
        "## Artifact Provenance",
        "",
        f"- P10-A inventory hash: `{inventory['self_hash']}`",
        f"- P10-B outputs manifest hash: `{b_manifest['self_hash']}`",
        "- Report values are rendered from machine-readable Phase 10 JSON/CSV artifacts.",
        "- No model training, inference, prediction regeneration, external tuning, or checkpoint reselection is part of this report build.",
        "",
        "## Scientific Question",
        "",
        "ProtoEM-CT studies robust transductive few-shot adaptation for 3D CT liver-tumor segmentation under leakage-safe development and external-validation constraints.",
        "",
        "## Cohorts and Dataset Policy",
        "",
        f"The development source contains {row_value(cohort, 'Development cases')} LiTS/MSD Task03 Liver cases and is treated as one LiTS-derived source, not independent cohorts.",
        f"The patient-level development split contains {row_value(cohort, 'Development train cases')} train, {row_value(cohort, 'Development validation cases')} validation, and {row_value(cohort, 'Immutable internal-test cases')} immutable internal-test cases.",
        f"The external 3D-IRCADb-01 accounting records {row_value(cohort, 'External discovered cases')} discovered cases, {row_value(cohort, 'External evaluation-eligible cases')} evaluation-eligible cases, and {row_value(cohort, 'External excluded cases')} excluded cases.",
        "",
        render_table_markdown(cohort),
        "",
        "## Leakage Prevention",
        "",
        "The project preserves patient-level development splits, treats LiTS/MSD as a single development source, and records that external labels were not used for tuning, model selection, threshold selection, preprocessing decisions, support policy, or protocol iteration.",
        "",
        "## Preprocessing and Freeze",
        "",
        "The Phase 8 freeze used a fixed development-derived target spacing and fixed threshold/support policies recorded in saved artifacts. These are reported as provenance, not reopened or optimized in Phase 10.",
        "",
        "## Baselines, Few-Shot Protocol, Retrieval, ProtoEM-CT, Ablations, Robustness, and Uncertainty",
        "",
        "Durable machine-readable performance artifacts for several earlier planned reporting surfaces were not available in the P10-A inventory. Their numeric results are intentionally not reported.",
        "",
        render_table_markdown(unavailable, max_rows=10),
        "",
        "## External Validation",
        "",
        f"External tumor Dice was {row_by_label(external, 'Tumor Dice (macro)')['point_estimate']} with 95% CI [{row_by_label(external, 'Tumor Dice (macro)')['ci_low']}, {row_by_label(external, 'Tumor Dice (macro)')['ci_high']}].",
        f"External tumor IoU was {row_by_label(external, 'Tumor IoU (macro)')['point_estimate']} with 95% CI [{row_by_label(external, 'Tumor IoU (macro)')['ci_low']}, {row_by_label(external, 'Tumor IoU (macro)')['ci_high']}].",
        f"External lesion F1 was {row_by_label(external, 'Lesion F1')['point_estimate']} with 95% CI [{row_by_label(external, 'Lesion F1')['ci_low']}, {row_by_label(external, 'Lesion F1')['ci_high']}].",
        f"False-positive lesions per scan were {row_by_label(external, 'False-positive lesions/scan')['point_estimate']} with 95% CI [{row_by_label(external, 'False-positive lesions/scan')['ci_low']}, {row_by_label(external, 'False-positive lesions/scan')['ci_high']}].",
        "",
        "These are poor external results and do not support a claim of strong external generalization.",
        "",
        render_table_markdown(external),
        "",
        "## Internal-vs-External Descriptive Comparison",
        "",
        "The internal-vs-external comparison is descriptive only. It does not report superiority, non-inferiority, deployment readiness, or broad generalization.",
        "",
        render_table_markdown(comparison),
        "",
        "## Statistical Comparisons",
        "",
        f"The saved statistical policy uses case/patient-level bootstrap resampling with seed `{policy['bootstrap_policy']['random_seed']}`, `{policy['bootstrap_policy']['resample_count']}` resamples, `{policy['bootstrap_policy']['confidence_level']}` confidence level, and `{policy['bootstrap_policy']['interval_method']}` intervals.",
        "Phase 10 added no new significance tests and reports no new p-values.",
        "",
        "## Limitations",
        "",
        "- The evaluated external result is negative and must not be reframed as a partial success.",
        "- External evaluation is limited to the artifact-recorded eligible case count.",
        "- Unsupported Phase 3-7 numeric results are unavailable in durable Phase-10 inventory and are intentionally omitted.",
        "- This repository does not commit raw medical data, checkpoints, predictions, model weights, credentials, or secrets.",
        "",
        "## Reproducibility",
        "",
        "The report is generated from the P10-A inventory, P10-B tables, figures, statistics JSON, and saved Phase 8 comparison artifact. The required Gate 8 command is:",
        "",
        "```bash",
        "uv run snakemake --cores 4 --rerun-incomplete reports/final_report.html",
        "```",
        "",
        "## Final Conclusion",
        "",
        "Phase 10 packages the completed evidence without changing the scientific result. External validation remains an honest negative result. The core project is complete only if Gate 8 reproduction and final review pass.",
        "",
    ]
    return "\n".join(lines), {"numeric_statement_sources": numeric_sources}


def markdown_to_html(markdown: str, b_manifest: dict[str, Any]) -> str:
    """Render a simple deterministic HTML report from Markdown and generated SVGs."""
    escaped = html.escape(markdown)
    body = escaped.replace("\n", "<br>\n")
    figure_html = []
    for figure_id in (
        "phase10_cohort_counts",
        "phase10_external_validation_ci",
        "phase10_checkpoint_selection",
        "phase10_internal_external_tumor_dice",
    ):
        figure = figure_by_id(b_manifest, figure_id)
        svg = Path(str(figure["svg_path"])).read_text(encoding="utf-8")
        figure_html.append(f"<section>{svg}</section>")
    return (
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8"><title>ProtoEM-CT Final Scientific Package</title>'
        "<style>body{font-family:Arial,sans-serif;line-height:1.45;margin:32px;max-width:1180px}"
        "table{border-collapse:collapse;margin:16px 0;font-size:12px}td,th{border:1px solid #cbd5e1;padding:4px 6px;vertical-align:top}"
        "code{background:#eef2f7;padding:1px 3px}</style></head><body>"
        f"<main>{body}</main><hr><h2>Figures</h2>{''.join(figure_html)}</body></html>\n"
    )


def write_reproduction_readme(
    path: Path, inventory: dict[str, Any], b_manifest: dict[str, Any]
) -> None:
    """Write the Phase 10 reproduction README."""
    text = f"""# Phase 10 Reproduction README

This document describes artifact-only reproduction of the final scientific package.

## Environment

- Python 3.11
- `uv`
- Snakemake
- Repository branch: `phase/10-final-scientific-package`

## Data and Artifact Expectations

Raw medical datasets, DICOM/NIfTI arrays, predictions, checkpoints, model weights, credentials, and secrets are not committed to Git.

The default local Phase 10 inventory command reads saved non-medical JSON/provenance artifacts from the configured run-artifact root. On this workstation the recovered inventory used:

- Inventory hash: `{inventory["self_hash"]}`
- P10-B manifest hash: `{b_manifest["self_hash"]}`

Use `--drive-root` on `workflow/scripts/build_phase10_artifact_inventory.py` if your saved artifact root differs.

## Rebuild Commands

```bash
uv run python workflow/scripts/build_phase10_artifact_inventory.py --repo-root "$PWD" --drive-root /path/to/ProtoEM-CT/runs --output-json reports/phase10/artifact_inventory.json --output-markdown reports/phase10/ARTIFACT_INVENTORY.md
uv run python workflow/scripts/build_phase10_scientific_outputs.py --inventory reports/phase10/artifact_inventory.json --reports-root reports
uv run python workflow/scripts/build_phase10_final_report.py --inventory reports/phase10/artifact_inventory.json --b-manifest reports/phase10/phase10_b_outputs_manifest.json --report-md reports/final_report.md --report-html reports/final_report.html --report-manifest reports/phase10/final_report_manifest.json --reproduction-readme docs/PHASE10_REPRODUCTION.md
```

Gate 8 uses the repository DAG:

```bash
uv run snakemake --cores 4 --rerun-incomplete reports/final_report.html
```

The Phase 10 DAG path must not trigger training, inference, external-data experiments, downloads, or prediction regeneration.

## Outputs

- Final Markdown report: `reports/final_report.md`
- Final HTML report: `reports/final_report.html`
- Report manifest: `reports/phase10/final_report_manifest.json`
- Tables: `reports/tables/phase10_*.json` and `reports/tables/phase10_*.csv`
- Figures: `reports/figures/phase10_*.json` and `reports/figures/phase10_*.svg`
- Statistics: `reports/statistics/phase10_*.json`

## Scope

Phase 9 was skipped by user decision and is not executed. Phase 10 is the final core-project reporting and packaging phase.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_report(
    inventory_path: Path,
    b_manifest_path: Path,
    report_md: Path,
    report_html: Path,
    report_manifest: Path,
    reproduction_readme: Path,
) -> dict[str, Any]:
    """Write final report, HTML, reproduction README, and report manifest."""
    inventory = load_json(inventory_path)
    b_manifest = load_json(b_manifest_path)
    markdown, numeric_sources = render_report_markdown(inventory, b_manifest)
    html_text = markdown_to_html(markdown, b_manifest)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_html.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(markdown, encoding="utf-8")
    report_html.write_text(html_text, encoding="utf-8")
    write_reproduction_readme(reproduction_readme, inventory, b_manifest)
    manifest = {
        "schema_name": "phase10_final_report_manifest",
        "schema_version": SCHEMA_VERSION,
        "source_inventory_path": str(inventory_path),
        "source_inventory_hash": inventory["self_hash"],
        "source_b_manifest_path": str(b_manifest_path),
        "source_b_manifest_hash": b_manifest["self_hash"],
        "report_markdown_path": str(report_md),
        "report_markdown_sha256": sha256_file(report_md),
        "report_html_path": str(report_html),
        "report_html_sha256": sha256_file(report_html),
        "reproduction_readme_path": str(reproduction_readme),
        "reproduction_readme_sha256": sha256_file(reproduction_readme),
        "numeric_statement_source_count": len(numeric_sources["numeric_statement_sources"]),
        **numeric_sources,
        "phase9_status": "skipped_by_user_decision_not_executed",
        "training_rerun": False,
        "inference_rerun": False,
        "external_label_tuning": False,
    }
    manifest_hash = write_json(report_manifest, manifest)
    manifest["self_hash"] = manifest_hash
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory", type=Path, default=Path("reports/phase10/artifact_inventory.json")
    )
    parser.add_argument(
        "--b-manifest", type=Path, default=Path("reports/phase10/phase10_b_outputs_manifest.json")
    )
    parser.add_argument("--report-md", type=Path, default=Path("reports/final_report.md"))
    parser.add_argument("--report-html", type=Path, default=Path("reports/final_report.html"))
    parser.add_argument(
        "--report-manifest", type=Path, default=Path("reports/phase10/final_report_manifest.json")
    )
    parser.add_argument(
        "--reproduction-readme", type=Path, default=Path("docs/PHASE10_REPRODUCTION.md")
    )
    args = parser.parse_args()
    manifest = write_report(
        args.inventory,
        args.b_manifest,
        args.report_md,
        args.report_html,
        args.report_manifest,
        args.reproduction_readme,
    )
    print(
        f"Wrote {args.report_html} "
        f"(sha256={manifest['report_html_sha256']}, manifest_hash={manifest['self_hash']})"
    )


if __name__ == "__main__":
    main()
