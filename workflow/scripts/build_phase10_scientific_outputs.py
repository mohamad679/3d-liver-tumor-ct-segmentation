# ruff: noqa: E501
"""Build Phase 10 artifact-backed tables, figures, and statistics.

This script consumes the validated P10-A inventory plus selected saved Phase 8
machine-readable artifacts. It performs no training, inference, prediction
generation, thresholding, or medical-metric recomputation.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build_phase10_artifact_inventory import sha256_file, sha256_json

SCHEMA_VERSION = "v1"


class Phase10OutputError(RuntimeError):
    """Raised when required saved Phase 10 inputs are missing or malformed."""


@dataclass(frozen=True, slots=True)
class MetricLookup:
    """Lookup table for P10-A metric entries."""

    metrics: dict[str, dict[str, Any]]

    @classmethod
    def from_inventory(cls, inventory: dict[str, Any]) -> MetricLookup:
        metrics: dict[str, dict[str, Any]] = {}
        for metric in inventory.get("metrics", []):
            name = metric.get("name")
            if not isinstance(name, str):
                raise Phase10OutputError("Inventory metric missing string name")
            if name in metrics:
                raise Phase10OutputError(f"Duplicate inventory metric: {name}")
            metrics[name] = metric
        return cls(metrics)

    def require(self, metric_name: str) -> dict[str, Any]:
        """Return a metric entry or fail closed."""
        try:
            return self.metrics[metric_name]
        except KeyError as exc:
            raise Phase10OutputError(f"Missing required metric: {metric_name}") from exc

    def value(self, metric_name: str) -> Any:
        """Return a metric value."""
        return self.require(metric_name)["value"]

    def source(self, metric_name: str) -> dict[str, str]:
        """Return source provenance for a metric."""
        metric = self.require(metric_name)
        return {
            "source_metric": metric_name,
            "source_artifact": str(metric["source_artifact"]),
            "source_field": str(metric["source_field_path"]),
            "value_kind": str(metric["value_kind"]),
        }


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.exists():
        raise Phase10OutputError(f"Required JSON input is missing: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise Phase10OutputError(f"Expected JSON object at {path}")
    return loaded


def load_existing_phase10_comparison(reports_root: Path) -> dict[str, Any]:
    """Load the saved Phase 10 comparison when the drive artifact is unavailable."""
    fallback = reports_root / "statistics" / "phase10_internal_external_descriptive_comparison.json"
    return load_json(fallback)


def write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write deterministic JSON with a self-hash and return that hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    full_payload = dict(payload)
    full_payload["self_hash"] = sha256_json(full_payload)
    path.write_text(json.dumps(full_payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return str(full_payload["self_hash"])


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write deterministic CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any, digits: int = 6) -> str:
    """Format values for compact tables and SVG labels."""
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(v, digits=digits) for v in value) + "]"
    return str(value)


def metric_row(
    lookup: MetricLookup,
    *,
    label: str,
    metric_name: str,
    unit: str = "",
    valid_case_metric: str | None = None,
) -> dict[str, Any]:
    """Create a table row from one inventory metric."""
    source = lookup.source(metric_name)
    return {
        "label": label,
        "value": fmt(lookup.value(metric_name)),
        "unit": unit,
        "valid_case_count": fmt(lookup.value(valid_case_metric)) if valid_case_metric else "",
        **source,
    }


def ci_row(
    lookup: MetricLookup,
    *,
    label: str,
    point_metric: str,
    low_metric: str,
    high_metric: str,
    unit: str = "",
) -> dict[str, Any]:
    """Create a point-estimate plus confidence-interval row."""
    source = lookup.source(point_metric)
    low_source = lookup.source(low_metric)
    high_source = lookup.source(high_metric)
    return {
        "label": label,
        "point_estimate": fmt(lookup.value(point_metric)),
        "ci_low": fmt(lookup.value(low_metric)),
        "ci_high": fmt(lookup.value(high_metric)),
        "unit": unit,
        "valid_case_count": fmt(lookup.value("case_count")),
        **source,
        "ci_low_source_artifact": low_source["source_artifact"],
        "ci_low_source_field": low_source["source_field"],
        "ci_high_source_artifact": high_source["source_artifact"],
        "ci_high_source_field": high_source["source_field"],
    }


def point_without_ci_row(
    lookup: MetricLookup, *, label: str, point_metric: str, unit: str = ""
) -> dict[str, Any]:
    """Create a point-estimate row with explicitly unavailable CI fields."""
    source = lookup.source(point_metric)
    return {
        "label": label,
        "point_estimate": fmt(lookup.value(point_metric)),
        "ci_low": "n/a",
        "ci_high": "n/a",
        "unit": unit,
        "valid_case_count": fmt(lookup.value("case_count")),
        **source,
        "ci_low_source_artifact": "n/a",
        "ci_low_source_field": "n/a",
        "ci_high_source_artifact": "n/a",
        "ci_high_source_field": "n/a",
    }


def build_tables(
    inventory: dict[str, Any], comparison: dict[str, Any]
) -> list[tuple[str, list[str], list[dict[str, Any]], dict[str, Any]]]:
    """Build P10-B table rows and metadata."""
    lookup = MetricLookup.from_inventory(inventory)
    unavailable_rows = [
        {
            "scope": item["referenced_in"],
            "description": item["description"],
            "reason": item["reason"],
            "value_kind": "unavailable",
        }
        for item in inventory.get("unavailable", [])
    ]
    comparison_rows = []
    comparison_items = comparison.get("comparisons", {})
    if not isinstance(comparison_items, dict):
        raise Phase10OutputError("Internal/external comparison artifact has invalid comparisons")
    for metric_name, comparison_entry in comparison_items.items():
        comparison_rows.append(
            {
                "metric": metric_name,
                "internal_value": fmt(comparison_entry.get("internal_value")),
                "external_value": fmt(comparison_entry.get("external_point_estimate")),
                "external_ci_low": fmt(comparison_entry.get("external_ci_low")),
                "external_ci_high": fmt(comparison_entry.get("external_ci_high")),
                "difference_external_minus_internal": "not_materialized_in_source_artifact",
                "claims_policy": comparison.get("claims_policy", "descriptive_only"),
                "source_artifact": "phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json",
                "value_kind": "descriptive",
            }
        )

    return [
        (
            "phase10_cohort_data_summary",
            [
                "label",
                "value",
                "unit",
                "valid_case_count",
                "source_metric",
                "source_artifact",
                "source_field",
                "value_kind",
            ],
            [
                metric_row(
                    lookup,
                    label="Development cases",
                    metric_name="development_case_count",
                    unit="cases",
                ),
                metric_row(
                    lookup,
                    label="Development train cases",
                    metric_name="train_case_count",
                    unit="cases",
                ),
                metric_row(
                    lookup,
                    label="Development validation cases",
                    metric_name="validation_case_count",
                    unit="cases",
                ),
                metric_row(
                    lookup,
                    label="Immutable internal-test cases",
                    metric_name="internal_test_case_count",
                    unit="cases",
                ),
                metric_row(
                    lookup,
                    label="Geometry QA passed cases",
                    metric_name="geometry_qa_passed_case_count",
                    unit="cases",
                ),
                metric_row(
                    lookup,
                    label="Geometry QA failed cases",
                    metric_name="geometry_qa_failed_case_count",
                    unit="cases",
                ),
                metric_row(
                    lookup,
                    label="Total development lesions",
                    metric_name="total_lesion_count",
                    unit="lesions",
                ),
                metric_row(
                    lookup,
                    label="External discovered cases",
                    metric_name="external_discovered_count",
                    unit="cases",
                ),
                metric_row(
                    lookup,
                    label="External inference-eligible cases",
                    metric_name="external_inference_eligible_count",
                    unit="cases",
                ),
                metric_row(
                    lookup,
                    label="External evaluation-eligible cases",
                    metric_name="external_evaluation_eligible_count",
                    unit="cases",
                ),
                metric_row(
                    lookup,
                    label="External excluded cases",
                    metric_name="external_excluded_count",
                    unit="cases",
                ),
            ],
            {
                "title": "Phase 10 Cohort Data Summary",
                "notes": [
                    "LiTS/MSD Task03 Liver is one LiTS-derived development source.",
                    "External cohort is 3D-IRCADb-01.",
                ],
            },
        ),
        (
            "phase10_external_validation_metrics",
            [
                "label",
                "point_estimate",
                "ci_low",
                "ci_high",
                "unit",
                "valid_case_count",
                "source_metric",
                "source_artifact",
                "source_field",
                "value_kind",
                "ci_low_source_artifact",
                "ci_low_source_field",
                "ci_high_source_artifact",
                "ci_high_source_field",
            ],
            [
                ci_row(
                    lookup,
                    label="Tumor Dice (macro)",
                    point_metric="macro_dice",
                    low_metric="tumor_dice_ci_low",
                    high_metric="tumor_dice_ci_high",
                ),
                point_without_ci_row(
                    lookup, label="Tumor Dice (pooled)", point_metric="pooled_dice"
                ),
                ci_row(
                    lookup,
                    label="Tumor IoU (macro)",
                    point_metric="macro_iou",
                    low_metric="tumor_iou_ci_low",
                    high_metric="tumor_iou_ci_high",
                ),
                ci_row(
                    lookup,
                    label="Tumor HD95 (macro)",
                    point_metric="macro_hd95_mm",
                    low_metric="tumor_hd95_ci_low",
                    high_metric="tumor_hd95_ci_high",
                    unit="mm",
                ),
                ci_row(
                    lookup,
                    label="Tumor NSD",
                    point_metric="macro_normalized_surface_dice",
                    low_metric="tumor_normalized_surface_dice_ci_low",
                    high_metric="tumor_normalized_surface_dice_ci_high",
                ),
                ci_row(
                    lookup,
                    label="Lesion recall",
                    point_metric="macro_lesion_recall",
                    low_metric="lesion_wise_recall_ci_low",
                    high_metric="lesion_wise_recall_ci_high",
                ),
                ci_row(
                    lookup,
                    label="Lesion precision",
                    point_metric="macro_lesion_precision",
                    low_metric="lesion_wise_precision_ci_low",
                    high_metric="lesion_wise_precision_ci_high",
                ),
                ci_row(
                    lookup,
                    label="Lesion F1",
                    point_metric="macro_lesion_f1",
                    low_metric="lesion_f1_ci_low",
                    high_metric="lesion_f1_ci_high",
                ),
                ci_row(
                    lookup,
                    label="False-positive lesions/scan",
                    point_metric="mean_false_positive_lesions_per_scan",
                    low_metric="false_positive_lesions_per_scan_ci_low",
                    high_metric="false_positive_lesions_per_scan_ci_high",
                    unit="lesions/scan",
                ),
                ci_row(
                    lookup,
                    label="Signed tumor volume error",
                    point_metric="mean_signed_volume_error_ml",
                    low_metric="tumor_volume_error_signed_ml_ci_low",
                    high_metric="tumor_volume_error_signed_ml_ci_high",
                    unit="mL",
                ),
                ci_row(
                    lookup,
                    label="Absolute tumor volume error",
                    point_metric="mean_absolute_volume_error_ml",
                    low_metric="tumor_volume_error_absolute_ml_ci_low",
                    high_metric="tumor_volume_error_absolute_ml_ci_high",
                    unit="mL",
                ),
                ci_row(
                    lookup,
                    label="Relative tumor volume error",
                    point_metric="macro_relative_volume_error",
                    low_metric="tumor_volume_error_relative_ci_low",
                    high_metric="tumor_volume_error_relative_ci_high",
                ),
            ],
            {
                "title": "Phase 10 External Validation Metrics",
                "notes": ["Confirmatory Phase 8 metrics with saved bootstrap CIs."],
            },
        ),
        (
            "phase10_internal_external_descriptive_comparison",
            [
                "metric",
                "internal_value",
                "external_value",
                "external_ci_low",
                "external_ci_high",
                "difference_external_minus_internal",
                "claims_policy",
                "source_artifact",
                "value_kind",
            ],
            comparison_rows,
            {
                "title": "Phase 10 Internal-vs-External Descriptive Comparison",
                "notes": ["Descriptive only; no superiority or generalization claim."],
            },
        ),
        (
            "phase10_unavailable_results_and_limitations",
            ["scope", "description", "reason", "value_kind"],
            unavailable_rows,
            {
                "title": "Phase 10 Unavailable Results and Limitations",
                "notes": ["Unavailable metrics are intentionally not reported."],
            },
        ),
    ]


def write_table_outputs(
    inventory: dict[str, Any], comparison: dict[str, Any], reports_root: Path
) -> list[dict[str, Any]]:
    """Write table CSV/JSON files."""
    records: list[dict[str, Any]] = []
    for table_id, columns, rows, metadata in build_tables(inventory, comparison):
        json_path = reports_root / "tables" / f"{table_id}.json"
        csv_path = reports_root / "tables" / f"{table_id}.csv"
        payload = {
            "schema_name": "phase10_table",
            "schema_version": SCHEMA_VERSION,
            "table_id": table_id,
            "source_inventory_hash": inventory["self_hash"],
            "columns": columns,
            "rows": rows,
            **metadata,
        }
        table_hash = write_json(json_path, payload)
        write_csv(csv_path, columns, rows)
        records.append(
            {
                "id": table_id,
                "json_path": str(json_path),
                "json_sha256": sha256_file(json_path),
                "csv_path": str(csv_path),
                "csv_sha256": sha256_file(csv_path),
                "self_hash": table_hash,
                "source_inventory_hash": inventory["self_hash"],
            }
        )
    return records


def point(lookup: MetricLookup, label: str, metric_name: str) -> dict[str, Any]:
    """Create a figure data point from one metric."""
    source = lookup.source(metric_name)
    return {"label": label, "value": lookup.value(metric_name), **source}


def ci_point_source(lookup: MetricLookup, low_metric: str, high_metric: str) -> dict[str, Any]:
    """Return CI values and provenance for a figure data point."""
    low_source = lookup.source(low_metric)
    high_source = lookup.source(high_metric)
    return {
        "ci_low": lookup.value(low_metric),
        "ci_high": lookup.value(high_metric),
        "ci_low_source_artifact": low_source["source_artifact"],
        "ci_low_source_field": low_source["source_field"],
        "ci_high_source_artifact": high_source["source_artifact"],
        "ci_high_source_field": high_source["source_field"],
    }


def build_figures(inventory: dict[str, Any], comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Build figure specs."""
    lookup = MetricLookup.from_inventory(inventory)
    internal_external_points = []
    comparison_items = comparison.get("comparisons", {})
    if isinstance(comparison_items, dict):
        item = comparison_items.get("tumor_dice")
        if isinstance(item, dict):
            internal_external_points = [
                {
                    "label": "Internal validation Dice",
                    "value": item.get("internal_value"),
                    "ci_low": None,
                    "ci_high": None,
                    "source_metric": "tumor_dice_internal_scalar",
                    "source_artifact": "phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json",
                    "source_field": "comparisons.tumor_dice.internal_value",
                    "value_kind": "descriptive",
                },
                {
                    "label": "External macro Dice",
                    "value": item.get("external_point_estimate"),
                    "ci_low": item.get("external_ci_low"),
                    "ci_high": item.get("external_ci_high"),
                    "source_metric": "tumor_dice_external_macro",
                    "source_artifact": "phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json",
                    "source_field": "comparisons.tumor_dice.external_point_estimate",
                    "value_kind": "descriptive",
                },
            ]
    return [
        {
            "figure_id": "phase10_cohort_counts",
            "title": "Phase 10 Cohort Counts",
            "points": [
                point(lookup, "Train", "train_case_count"),
                point(lookup, "Validation", "validation_case_count"),
                point(lookup, "Internal test", "internal_test_case_count"),
                point(lookup, "External discovered", "external_discovered_count"),
                point(lookup, "External eligible", "external_evaluation_eligible_count"),
                point(lookup, "External excluded", "external_excluded_count"),
            ],
        },
        {
            "figure_id": "phase10_external_validation_ci",
            "title": "Phase 10 External Validation Bootstrap CIs",
            "points": [
                point(lookup, "Dice", "macro_dice")
                | ci_point_source(lookup, "tumor_dice_ci_low", "tumor_dice_ci_high"),
                point(lookup, "IoU", "macro_iou")
                | ci_point_source(lookup, "tumor_iou_ci_low", "tumor_iou_ci_high"),
                point(lookup, "NSD", "macro_normalized_surface_dice")
                | ci_point_source(
                    lookup,
                    "tumor_normalized_surface_dice_ci_low",
                    "tumor_normalized_surface_dice_ci_high",
                ),
                point(lookup, "Lesion F1", "macro_lesion_f1")
                | ci_point_source(lookup, "lesion_f1_ci_low", "lesion_f1_ci_high"),
            ],
        },
        {
            "figure_id": "phase10_checkpoint_selection",
            "title": "Phase 10 Checkpoint Selection Evidence",
            "points": [
                point(lookup, "Step 250", "mean_tumor_dice_step_250"),
                point(lookup, "Step 500", "mean_tumor_dice_step_500"),
            ],
        },
        {
            "figure_id": "phase10_internal_external_tumor_dice",
            "title": "Phase 10 Internal vs External Tumor Dice",
            "points": internal_external_points,
        },
    ]


def render_svg(spec: dict[str, Any]) -> str:
    """Render a deterministic SVG bar/error plot from a figure spec."""
    points = spec["points"]
    values = [float(p["value"]) for p in points if p.get("value") is not None]
    lows = [float(p["ci_low"]) for p in points if p.get("ci_low") is not None]
    highs = [float(p["ci_high"]) for p in points if p.get("ci_high") is not None]
    domain_values = values + lows + highs + [0.0]
    min_value = min(domain_values)
    max_value = max(domain_values)
    span = max(max_value - min_value, 1e-12)
    width = 780
    height = 88 + len(points) * 46
    left = 220
    bar_width = 430

    def scale(value: float) -> float:
        return left + ((value - min_value) / span) * bar_width

    zero_x = scale(0.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" aria-label="{html.escape(spec["title"])}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#1f2933}.title{font-size:18px;font-weight:700}.bar{fill:#357a7f}.neg{fill:#b85c38}.axis{stroke:#6b7280;stroke-width:1}.ci{stroke:#243b53;stroke-width:2}</style>",
        f'<text class="title" x="20" y="30">{html.escape(spec["title"])}</text>',
        f'<line class="axis" x1="{zero_x:.2f}" y1="52" x2="{zero_x:.2f}" y2="{height - 26}" />',
    ]
    for index, point_item in enumerate(points):
        y = 64 + index * 46
        value = float(point_item["value"])
        value_x = scale(value)
        x = min(zero_x, value_x)
        w = max(abs(value_x - zero_x), 1.0)
        css_class = "bar neg" if value < 0 else "bar"
        lines.append(f'<text x="20" y="{y + 16}">{html.escape(str(point_item["label"]))}</text>')
        if point_item.get("ci_low") is not None and point_item.get("ci_high") is not None:
            low_x = scale(float(point_item["ci_low"]))
            high_x = scale(float(point_item["ci_high"]))
            lines.append(
                f'<line class="ci" x1="{low_x:.2f}" y1="{y + 11}" x2="{high_x:.2f}" y2="{y + 11}" />'
            )
        lines.append(
            f'<rect class="{css_class}" x="{x:.2f}" y="{y}" width="{w:.2f}" height="22" />'
        )
        lines.append(f'<text x="670" y="{y + 16}">{html.escape(fmt(value))}</text>')
    lines.append("</svg>\n")
    return "\n".join(lines)


def write_figure_outputs(
    inventory: dict[str, Any], comparison: dict[str, Any], reports_root: Path
) -> list[dict[str, Any]]:
    """Write figure JSON specs and SVG renderings."""
    records: list[dict[str, Any]] = []
    for figure in build_figures(inventory, comparison):
        payload = {
            "schema_name": "phase10_figure",
            "schema_version": SCHEMA_VERSION,
            "source_inventory_hash": inventory["self_hash"],
            **figure,
        }
        json_path = reports_root / "figures" / f"{figure['figure_id']}.json"
        svg_path = reports_root / "figures" / f"{figure['figure_id']}.svg"
        figure_hash = write_json(json_path, payload)
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(render_svg(payload), encoding="utf-8")
        records.append(
            {
                "id": figure["figure_id"],
                "json_path": str(json_path),
                "json_sha256": sha256_file(json_path),
                "svg_path": str(svg_path),
                "svg_sha256": sha256_file(svg_path),
                "self_hash": figure_hash,
                "source_inventory_hash": inventory["self_hash"],
            }
        )
    return records


def write_statistics_outputs(
    inventory: dict[str, Any], comparison: dict[str, Any], reports_root: Path
) -> list[dict[str, Any]]:
    """Write P10-B statistical summary files."""
    lookup = MetricLookup.from_inventory(inventory)
    policy = {
        "schema_name": "phase10_statistical_policy",
        "schema_version": SCHEMA_VERSION,
        "source_inventory_hash": inventory["self_hash"],
        "bootstrap_policy": {
            "resampling_unit": "case_patient",
            "resample_count": 10000,
            "random_seed": 1729,
            "confidence_level": 0.95,
            "interval_method": "percentile",
        },
        "internal_external_policy": comparison.get("claims_policy", "descriptive_only"),
        "new_phase10_significance_tests": 0,
    }
    bootstrap = {
        "schema_name": "phase10_external_bootstrap_ci",
        "schema_version": SCHEMA_VERSION,
        "source_inventory_hash": inventory["self_hash"],
        "rows": [
            ci_row(
                lookup,
                label="Tumor Dice",
                point_metric="macro_dice",
                low_metric="tumor_dice_ci_low",
                high_metric="tumor_dice_ci_high",
            ),
            ci_row(
                lookup,
                label="Tumor IoU",
                point_metric="macro_iou",
                low_metric="tumor_iou_ci_low",
                high_metric="tumor_iou_ci_high",
            ),
            ci_row(
                lookup,
                label="Tumor HD95",
                point_metric="macro_hd95_mm",
                low_metric="tumor_hd95_ci_low",
                high_metric="tumor_hd95_ci_high",
                unit="mm",
            ),
            ci_row(
                lookup,
                label="Tumor NSD",
                point_metric="macro_normalized_surface_dice",
                low_metric="tumor_normalized_surface_dice_ci_low",
                high_metric="tumor_normalized_surface_dice_ci_high",
            ),
            ci_row(
                lookup,
                label="Lesion recall",
                point_metric="macro_lesion_recall",
                low_metric="lesion_wise_recall_ci_low",
                high_metric="lesion_wise_recall_ci_high",
            ),
            ci_row(
                lookup,
                label="Lesion precision",
                point_metric="macro_lesion_precision",
                low_metric="lesion_wise_precision_ci_low",
                high_metric="lesion_wise_precision_ci_high",
            ),
            ci_row(
                lookup,
                label="Lesion F1",
                point_metric="macro_lesion_f1",
                low_metric="lesion_f1_ci_low",
                high_metric="lesion_f1_ci_high",
            ),
            ci_row(
                lookup,
                label="False-positive lesions/scan",
                point_metric="mean_false_positive_lesions_per_scan",
                low_metric="false_positive_lesions_per_scan_ci_low",
                high_metric="false_positive_lesions_per_scan_ci_high",
                unit="lesions/scan",
            ),
            ci_row(
                lookup,
                label="Signed tumor volume error",
                point_metric="mean_signed_volume_error_ml",
                low_metric="tumor_volume_error_signed_ml_ci_low",
                high_metric="tumor_volume_error_signed_ml_ci_high",
                unit="mL",
            ),
            ci_row(
                lookup,
                label="Absolute tumor volume error",
                point_metric="mean_absolute_volume_error_ml",
                low_metric="tumor_volume_error_absolute_ml_ci_low",
                high_metric="tumor_volume_error_absolute_ml_ci_high",
                unit="mL",
            ),
            ci_row(
                lookup,
                label="Relative tumor volume error",
                point_metric="macro_relative_volume_error",
                low_metric="tumor_volume_error_relative_ci_low",
                high_metric="tumor_volume_error_relative_ci_high",
            ),
        ],
    }
    comparison_payload = {
        "schema_name": "phase10_internal_external_descriptive_comparison",
        "schema_version": SCHEMA_VERSION,
        "source_inventory_hash": inventory["self_hash"],
        "source_artifact": "phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json",
        "claims_policy": comparison.get("claims_policy", "descriptive_only"),
        "cohort_relationship": comparison.get("cohort_relationship", "independent_unpaired"),
        "comparisons": comparison.get("comparisons", {}),
    }
    outputs = [
        ("phase10_statistical_policy", policy),
        ("phase10_external_bootstrap_ci", bootstrap),
        ("phase10_internal_external_descriptive_comparison", comparison_payload),
    ]
    records: list[dict[str, Any]] = []
    for stem, payload in outputs:
        path = reports_root / "statistics" / f"{stem}.json"
        self_hash = write_json(path, payload)
        records.append(
            {
                "id": stem,
                "json_path": str(path),
                "json_sha256": sha256_file(path),
                "self_hash": self_hash,
                "source_inventory_hash": inventory["self_hash"],
            }
        )
    return records


def write_outputs(inventory_path: Path, reports_root: Path) -> dict[str, Any]:
    """Build all P10-B outputs and return the manifest payload."""
    inventory = load_json(inventory_path)
    drive_root = Path(str(inventory["drive_root"]))
    comparison_path = (
        drive_root
        / "phase8_external_evaluation_v1"
        / "phase8_internal_external_comparison_corrected_v1.json"
    )
    if comparison_path.exists():
        comparison = load_json(comparison_path)
        comparison_source = str(comparison_path)
        comparison_sha256 = sha256_file(comparison_path)
    else:
        comparison = load_existing_phase10_comparison(reports_root)
        comparison_source = str(
            reports_root / "statistics" / "phase10_internal_external_descriptive_comparison.json"
        )
        comparison_sha256 = sha256_file(
            reports_root / "statistics" / "phase10_internal_external_descriptive_comparison.json"
        )
    tables = write_table_outputs(inventory, comparison, reports_root)
    figures = write_figure_outputs(inventory, comparison, reports_root)
    statistics = write_statistics_outputs(inventory, comparison, reports_root)
    manifest = {
        "schema_name": "phase10_b_outputs_manifest",
        "schema_version": SCHEMA_VERSION,
        "source_inventory_path": str(inventory_path),
        "source_inventory_hash": inventory["self_hash"],
        "source_comparison_artifact": comparison_source,
        "source_comparison_artifact_sha256": comparison_sha256,
        "tables": tables,
        "figures": figures,
        "statistics": statistics,
    }
    manifest_path = reports_root / "phase10" / "phase10_b_outputs_manifest.json"
    manifest_hash = write_json(manifest_path, manifest)
    manifest["self_hash"] = manifest_hash
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory", type=Path, default=Path("reports/phase10/artifact_inventory.json")
    )
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    args = parser.parse_args()
    manifest = write_outputs(args.inventory, args.reports_root)
    print(
        f"Wrote reports/phase10/phase10_b_outputs_manifest.json (self_hash={manifest['self_hash']})"
    )


if __name__ == "__main__":
    main()
