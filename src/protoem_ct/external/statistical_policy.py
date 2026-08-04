"""Phase 8 preregistered statistical and publication policy helpers."""

from __future__ import annotations

from typing import Final

from protoem_ct.artifacts.hashing import JsonValue, sha256_json

PHASE8_STATISTICAL_POLICY_SCHEMA_NAME: Final[str] = "phase8_statistical_policy_bundle"
PHASE8_STATISTICAL_POLICY_SCHEMA_VERSION: Final[str] = "v1"

PHASE8_SEGMENTATION_METRICS: Final[tuple[str, ...]] = (
    "tumor_dice",
    "tumor_iou",
    "tumor_hd95",
    "tumor_normalized_surface_dice",
    "lesion_wise_recall",
    "lesion_wise_precision",
    "lesion_f1",
    "false_positive_lesions_per_scan",
    "tumor_volume_error",
)


def build_phase8_statistical_policy_bundle() -> dict[str, JsonValue]:
    """Return the pre-result Phase 8 metric, bootstrap, and publication policy bundle."""

    metric_configuration: dict[str, JsonValue] = {
        "schema_name": "phase8_metric_configuration",
        "schema_version": "v1",
        "metrics": list(PHASE8_SEGMENTATION_METRICS),
        "aggregate_policy": {
            "tumor_dice": "pooled_and_macro_reported",
            "tumor_iou": "pooled_and_macro_reported",
            "tumor_hd95": "macro_defined_cases",
            "tumor_normalized_surface_dice": "macro",
            "lesion_wise_recall": "micro_and_macro_defined_cases",
            "lesion_wise_precision": "micro_and_macro_defined_cases",
            "lesion_f1": "micro_and_macro",
            "false_positive_lesions_per_scan": "case_mean",
            "tumor_volume_error": "signed_absolute_and_relative_reported",
        },
        "empty_mask_handling": "phase3_binary_tumor_metric_decision_2026_07_28",
        "surface_definition": "deterministic_3d_face_connectivity",
        "lesion_connectivity": "26_connected_components",
        "lesion_matching": "deterministic_maximum_overlap_one_to_one_shared_voxel_required",
        "undefined_metric_encoding": "json_null",
        "valid_case_accounting": "report_metric_specific_defined_case_counts",
    }
    bootstrap_configuration: dict[str, JsonValue] = {
        "schema_name": "phase8_bootstrap_configuration",
        "schema_version": "v1",
        "resampling_unit": "case_patient",
        "resampling_design": "external_cases_sampled_with_replacement",
        "random_seed": 1729,
        "resample_count": 10000,
        "confidence_level": 0.95,
        "interval_method": "percentile",
        "undefined_metric_handling": "exclude_undefined_values_with_defined_case_counts_reported",
        "minimum_valid_case_reporting": "report_all_counts_and_mark_ci_unavailable_when_zero",
        "pseudoreplication_policy": "no_lesion_level_or_voxel_level_resampling",
    }
    comparison_configuration: dict[str, JsonValue] = {
        "schema_name": "phase8_internal_external_comparison_policy",
        "schema_version": "v1",
        "cohort_relationship": "independent_unpaired",
        "claims_policy": "descriptive_only_no_superiority_or_generalization_claims",
        "reported_values": (
            "point_estimates_confidence_intervals_and_external_minus_internal_differences"
        ),
        "post_result_decision_policy": "no_model_threshold_support_or_preprocessing_changes",
    }
    qualitative_output_policy: dict[str, JsonValue] = {
        "schema_name": "phase8_qualitative_output_policy",
        "schema_version": "v1",
        "selection_rule": "include_all_evaluation_eligible_cases_when_count_at_most_20",
        "fallback_rule": "if_eligible_count_exceeds_20_use_hash_ranked_sample_with_seed_1729",
        "tie_breaking": "anonymous_case_id_lexicographic",
        "requires_labels_or_predictions_before_selection": False,
    }
    publication_configuration: dict[str, JsonValue] = {
        "schema_name": "phase8_publication_configuration",
        "schema_version": "v1",
        "source_policy": "json_first_from_persisted_preregistered_artifacts",
        "absolute_path_policy": "exclude_from_scientific_identities",
        "external_label_policy": "evaluation_only_after_prediction_lock",
        "mandatory_limitation": "small_external_sample_size",
    }
    robustness_uncertainty: dict[str, JsonValue] = {
        "schema_name": "phase8_robustness_uncertainty_inclusion",
        "schema_version": "v1",
        "inclusion_state": "not_included",
        "policy_reference": None,
        "rationale": "no_approved_phase8_external_robustness_uncertainty_artifacts",
    }
    payload: dict[str, JsonValue] = {
        "bootstrap_configuration": bootstrap_configuration,
        "comparison_configuration": comparison_configuration,
        "metric_configuration": metric_configuration,
        "publication_configuration": publication_configuration,
        "qualitative_output_policy": qualitative_output_policy,
        "robustness_uncertainty_inclusion": robustness_uncertainty,
        "schema_name": PHASE8_STATISTICAL_POLICY_SCHEMA_NAME,
        "schema_version": PHASE8_STATISTICAL_POLICY_SCHEMA_VERSION,
    }
    payload["policy_bundle_hash"] = sha256_json(payload)
    return payload


def phase8_statistical_policy_component_hash(component_name: str) -> str:
    """Return the canonical SHA-256 hash for one named policy component."""

    bundle = build_phase8_statistical_policy_bundle()
    component = bundle.get(component_name)
    if not isinstance(component, dict):
        raise ValueError(f"unknown Phase 8 statistical policy component: {component_name}")
    return sha256_json(component)


__all__ = [
    "PHASE8_SEGMENTATION_METRICS",
    "PHASE8_STATISTICAL_POLICY_SCHEMA_NAME",
    "PHASE8_STATISTICAL_POLICY_SCHEMA_VERSION",
    "build_phase8_statistical_policy_bundle",
    "phase8_statistical_policy_component_hash",
]
