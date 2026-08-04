"""Tests for Phase 8 statistical preregistration policy helpers."""

from __future__ import annotations

from typing import cast

from protoem_ct.artifacts.hashing import JsonValue, sha256_json
from protoem_ct.external.statistical_policy import (
    PHASE8_SEGMENTATION_METRICS,
    build_phase8_statistical_policy_bundle,
    phase8_statistical_policy_component_hash,
)


def test_phase8_statistical_policy_freezes_required_metrics_and_bootstrap() -> None:
    bundle = build_phase8_statistical_policy_bundle()
    metric_configuration = cast(dict[str, JsonValue], bundle["metric_configuration"])
    bootstrap_configuration = cast(dict[str, JsonValue], bundle["bootstrap_configuration"])
    comparison_configuration = cast(dict[str, JsonValue], bundle["comparison_configuration"])
    qualitative_output_policy = cast(dict[str, JsonValue], bundle["qualitative_output_policy"])
    robustness_uncertainty = cast(dict[str, JsonValue], bundle["robustness_uncertainty_inclusion"])

    assert bundle["policy_bundle_hash"] == sha256_json(
        {key: value for key, value in bundle.items() if key != "policy_bundle_hash"}
    )
    assert metric_configuration["metrics"] == list(PHASE8_SEGMENTATION_METRICS)
    assert bootstrap_configuration["resampling_unit"] == "case_patient"
    assert bootstrap_configuration["random_seed"] == 1729
    assert bootstrap_configuration["resample_count"] == 10000
    assert bootstrap_configuration["interval_method"] == "percentile"
    assert comparison_configuration["cohort_relationship"] == "independent_unpaired"
    assert qualitative_output_policy["selection_rule"] == (
        "include_all_evaluation_eligible_cases_when_count_at_most_20"
    )
    assert robustness_uncertainty["inclusion_state"] == "not_included"


def test_phase8_statistical_policy_component_hashes_are_stable() -> None:
    first = phase8_statistical_policy_component_hash("bootstrap_configuration")
    second = phase8_statistical_policy_component_hash("bootstrap_configuration")

    assert first == second
    assert first != phase8_statistical_policy_component_hash("metric_configuration")
