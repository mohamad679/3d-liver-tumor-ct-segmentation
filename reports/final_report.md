# ProtoEM-CT Final Scientific Package

Status: Phase 10 final scientific package. Phase 9 was skipped by user decision and was not executed.

## Artifact Provenance

- P10-A inventory hash: `67da0cadfd3d8127f3bfac55f8c3984bd46e6888aa3d6720ad23e8dcf252dea9`
- P10-B outputs manifest hash: `bce7894adbb3d059e6ee6367e7f90f49a30f32652e220d3aeb5b6298f2408035`
- Report values are rendered from machine-readable Phase 10 JSON/CSV artifacts.
- No model training, inference, prediction regeneration, external tuning, or checkpoint reselection is part of this report build.

## Scientific Question

ProtoEM-CT studies robust transductive few-shot adaptation for 3D CT liver-tumor segmentation under leakage-safe development and external-validation constraints.

## Cohorts and Dataset Policy

The development source contains 131 LiTS/MSD Task03 Liver cases and is treated as one LiTS-derived source, not independent cohorts.
The patient-level development split contains 91 train, 20 validation, and 20 immutable internal-test cases.
The external 3D-IRCADb-01 accounting records 20 discovered cases, 15 evaluation-eligible cases, and 5 excluded cases.

| label | value | unit | valid_case_count | source_metric | source_artifact | source_field | value_kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Development cases | 131 | cases |  | development_case_count | phase2_real_lits_v2/manifest.json | case_count | descriptive |
| Development train cases | 91 | cases |  | train_case_count | phase2_real_lits_v2/split.json | train_case_count | descriptive |
| Development validation cases | 20 | cases |  | validation_case_count | phase2_real_lits_v2/split.json | validation_case_count | descriptive |
| Immutable internal-test cases | 20 | cases |  | internal_test_case_count | phase2_real_lits_v2/split.json | internal_test_case_count | descriptive |
| Geometry QA passed cases | 131 | cases |  | geometry_qa_passed_case_count | phase2_real_lits_v2/geometry_qa.json | passed_case_count | descriptive |
| Geometry QA failed cases | 0 | cases |  | geometry_qa_failed_case_count | phase2_real_lits_v2/geometry_qa.json | failed_case_count | descriptive |
| Total development lesions | 908 | lesions |  | total_lesion_count | phase2_real_lits_v2/lesion_components.json | sum(case_records[*].lesion_count) | descriptive |
| External discovered cases | 20 | cases |  | external_discovered_count | phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json | discovered_count | descriptive |
| External inference-eligible cases | 20 | cases |  | external_inference_eligible_count | phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json | inference_eligible_count | descriptive |
| External evaluation-eligible cases | 15 | cases |  | external_evaluation_eligible_count | phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json | evaluation_eligible_count | descriptive |
| External excluded cases | 5 | cases |  | external_excluded_count | phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json | excluded_count | descriptive |

## Leakage Prevention

The project preserves patient-level development splits, treats LiTS/MSD as a single development source, and records that external labels were not used for tuning, model selection, threshold selection, preprocessing decisions, support policy, or protocol iteration.

## Preprocessing and Freeze

The Phase 8 freeze used a fixed development-derived target spacing and fixed threshold/support policies recorded in saved artifacts. These are reported as provenance, not reopened or optimized in Phase 10.

## Baselines, Few-Shot Protocol, Retrieval, ProtoEM-CT, Ablations, Robustness, and Uncertainty

Durable machine-readable performance artifacts for several earlier planned reporting surfaces were not available in the P10-A inventory. Their numeric results are intentionally not reported.

| scope | description | reason | value_kind |
| --- | --- | --- | --- |
| docs/phase8/SUPERVISOR_HANDOFF.md (Substage 3, blocked v1 run) | Checkpoint at phase8_substage3_tiny_real_v1/checkpoints/phase8_tiny_real_verification_checkpoint.pt reflects pre-fix buggy liver-vs-background training and has no accompanying JSON artifact (run never reached the publication step). | No JSON summary/config/metadata artifact was ever published for this run; only the stray checkpoint file exists. Must not be used or reported as a scientific result. | unavailable |
| Gate 3 (ACCEPTANCE_CHECKLIST.md) | Gate 3 checkboxes for CPU shape smoke tests, tiny-subset overfit tests, deterministic metric JSON from saved predictions, Git-visible-artifact absence check, full local quality checks, GitHub-hosted CI, and Gate 3 close-out documentation are unchecked. | No Gate 3 close-out evidence block or backing artifact exists in the repository or on the external drive for these specific items; the checklist itself records them as not yet done (pre-existing repo state, not a Phase 10 blocker). | unavailable |
| Phase 10 baseline-performance table scope | Phase 3 nnU-Net v2 and MONAI SegResNet baseline-performance metric rows. | No persistent machine-readable baseline-performance artifact was located in the repository or under /Volumes/Lexar/ProtoEM-CT/runs; Gate 3 remains partly unchecked and must not be reported as completed scientific baseline performance. | unavailable |
| ACCEPTANCE_CHECKLIST.md Gate 5 | Gate 5 status/criteria. | Gate 5 is recorded as 'Pending definition' with no criteria and no backing artifact. | unavailable |
| Phase 10 retrieval/prototype/foundation-baseline scope | Phase 5 retrieval, prototype, and foundation-baseline performance rows. | No durable Phase 5 publication artifact directory or Gate 5 acceptance evidence was located; Gate 5 is pending definition. These values are therefore unavailable and must not be inferred from code or prose. | unavailable |
| ACCEPTANCE_CHECKLIST.md Gate 7 | Gate 7 status/criteria. | Gate 7 is recorded as 'Pending definition' with no criteria and no backing artifact, despite docs/LEAKAGE_AUDIT.md describing a Phase 7 leakage/scope verification pass. | unavailable |
| docs/PHASE_PLAN.md / ACCEPTANCE_CHECKLIST.md Gate 4 | Phase 4 few-shot protocol support manifests, adaptation configs, and protocol-table artifacts (K x replicate x adaptation-mode). | Gate 4 evidence was produced from a synthetic publication run using test fixtures into an ephemeral external temporary output root; no persistent Phase 4 artifact directory was found under /Volumes/Lexar/ProtoEM-CT/runs. Only the counts/hashes recorded in ACCEPTANCE_CHECKLIST.md and docs/LEAKAGE_AUDIT.md are available as provenance, not independently re-verified raw files this session. | unavailable |
| ACCEPTANCE_CHECKLIST.md Gate 6 | Phase 6 / ProtoEM-CT ablation-comparison, objective-trace, and publication artifacts (ablation_comparison.json, objective_trace.json, run_summary.json, etc.). | Gate 6 evidence was produced from two independent synthetic runs into mktemp directories (ROOT_A/ROOT_B) that were not persisted; no Phase 6 output directory was found under /Volumes/Lexar/ProtoEM-CT/runs. Only the values quoted in ACCEPTANCE_CHECKLIST.md are available. | unavailable |
| docs/LEAKAGE_AUDIT.md Phase 7 section | Phase 7 robustness/uncertainty evaluation artifacts (calibration, risk-coverage, degradation, lesion-subgroup reports). | Phase 7 CLI is documented as 'synthetic_mode_only: true' with no real-data execution; no Phase 7 output directory exists under /Volumes/Lexar/ProtoEM-CT/runs. Robustness/uncertainty is explicitly 'not_included' in the Wave 4 statistical policy for Phase 8 as well. | unavailable |
| docs/phase8/FINAL_REPORT.md section Q | Git commit 51638ca662510bf4cb031b2aec29778bf26bba26 (comparison provenance fix) and f1d4e1be7b6f7aa3657792cae9125c825affdded / f264ce79dd68039256d140f03c468baed8ecea07 (Package C execution/capture). | Not re-verified in this P10-A session (out of this agent's Git-log scope); FINAL_REPORT.md section S already records these as independently re-verified in the Phase 8 closure session. Flagged here as inherited, not re-checked, provenance. | unavailable |

## External Validation

External tumor Dice was 0.0141157 with 95% CI [0.00434155, 0.025013].
External tumor IoU was 0.00721538 with 95% CI [0.00220828, 0.0127937].
External lesion F1 was 0.000829379 with 95% CI [0.000388752, 0.00130496].
False-positive lesions per scan were 2358.4 with 95% CI [1893.32, 2872.07].

These are poor external results and do not support a claim of strong external generalization.

| label | point_estimate | ci_low | ci_high | unit | valid_case_count | source_metric | source_artifact | source_field | value_kind | ci_low_source_artifact | ci_low_source_field | ci_high_source_artifact | ci_high_source_field |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tumor Dice (macro) | 0.0141157 | 0.00434155 | 0.025013 |  | 15 | macro_dice | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.macro_dice | confirmatory | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_dice.ci_low | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_dice.ci_high |
| Tumor Dice (pooled) | 0.0219557 | n/a | n/a |  | 15 | pooled_dice | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.pooled_dice | confirmatory | n/a | n/a | n/a | n/a |
| Tumor IoU (macro) | 0.00721538 | 0.00220828 | 0.0127937 |  | 15 | macro_iou | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.macro_iou | confirmatory | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_iou.ci_low | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_iou.ci_high |
| Tumor HD95 (macro) | 176.626 | 159.324 | 194.302 | mm | 15 | macro_hd95_mm | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.macro_hd95_mm | confirmatory | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_hd95.ci_low | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_hd95.ci_high |
| Tumor NSD | 0.00330141 | 0.000869538 | 0.00616264 |  | 15 | macro_normalized_surface_dice | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.macro_normalized_surface_dice | confirmatory | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_normalized_surface_dice.ci_low | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_normalized_surface_dice.ci_high |
| Lesion recall | 0.158868 | 0.0481536 | 0.310257 |  | 15 | macro_lesion_recall | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.macro_lesion_recall | confirmatory | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.lesion_wise_recall.ci_low | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.lesion_wise_recall.ci_high |
| Lesion precision | 0.000417023 | 0.000195357 | 0.000656226 |  | 15 | macro_lesion_precision | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.macro_lesion_precision | confirmatory | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.lesion_wise_precision.ci_low | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.lesion_wise_precision.ci_high |
| Lesion F1 | 0.000829379 | 0.000388752 | 0.00130496 |  | 15 | macro_lesion_f1 | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.macro_lesion_f1 | confirmatory | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.lesion_f1.ci_low | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.lesion_f1.ci_high |
| False-positive lesions/scan | 2358.4 | 1893.32 | 2872.07 | lesions/scan | 15 | mean_false_positive_lesions_per_scan | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.mean_false_positive_lesions_per_scan | confirmatory | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.false_positive_lesions_per_scan.ci_low | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.false_positive_lesions_per_scan.ci_high |
| Signed tumor volume error | -30.979 | -96.5135 | 18.2676 | mL | 15 | mean_signed_volume_error_ml | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.mean_signed_volume_error_ml | confirmatory | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_volume_error_signed_ml.ci_low | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_volume_error_signed_ml.ci_high |
| Absolute tumor volume error | 73.1784 | 36.1856 | 126.893 | mL | 15 | mean_absolute_volume_error_ml | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.mean_absolute_volume_error_ml | confirmatory | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_volume_error_absolute_ml.ci_low | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_volume_error_absolute_ml.ci_high |
| Relative tumor volume error | 2.83389 | 0.754696 | 5.48452 |  | 15 | macro_relative_volume_error | phase8_external_evaluation_v1/phase8_external_metric_report.json | aggregate.macro_relative_volume_error | confirmatory | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_volume_error_relative.ci_low | phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json | metrics.tumor_volume_error_relative.ci_high |

## Internal-vs-External Descriptive Comparison

The internal-vs-external comparison is descriptive only. It does not report superiority, non-inferiority, deployment readiness, or broad generalization.

| metric | internal_value | external_value | external_ci_low | external_ci_high | difference_external_minus_internal | claims_policy | source_artifact | value_kind |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| false_positive_lesions_per_scan | unavailable | 2358.4 | 1893.32 | 2872.07 | not_materialized_in_source_artifact | descriptive_only_no_superiority_or_generalization_claims | phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json | descriptive |
| lesion_f1 | unavailable | 0.000829379 | 0.000388752 | 0.00130496 | not_materialized_in_source_artifact | descriptive_only_no_superiority_or_generalization_claims | phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json | descriptive |
| lesion_wise_precision | unavailable | 0.000417023 | 0.000195357 | 0.000656226 | not_materialized_in_source_artifact | descriptive_only_no_superiority_or_generalization_claims | phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json | descriptive |
| lesion_wise_recall | unavailable | 0.158868 | 0.0481536 | 0.310257 | not_materialized_in_source_artifact | descriptive_only_no_superiority_or_generalization_claims | phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json | descriptive |
| tumor_dice | 0.015793 | 0.0141157 | 0.00434155 | 0.025013 | not_materialized_in_source_artifact | descriptive_only_no_superiority_or_generalization_claims | phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json | descriptive |
| tumor_hd95 | unavailable | 176.626 | 159.324 | 194.302 | not_materialized_in_source_artifact | descriptive_only_no_superiority_or_generalization_claims | phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json | descriptive |
| tumor_iou | unavailable | 0.00721538 | 0.00220828 | 0.0127937 | not_materialized_in_source_artifact | descriptive_only_no_superiority_or_generalization_claims | phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json | descriptive |
| tumor_normalized_surface_dice | unavailable | 0.00330141 | 0.000869538 | 0.00616264 | not_materialized_in_source_artifact | descriptive_only_no_superiority_or_generalization_claims | phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json | descriptive |
| tumor_volume_error_signed_ml | unavailable | -30.979 | -96.5135 | 18.2676 | not_materialized_in_source_artifact | descriptive_only_no_superiority_or_generalization_claims | phase8_external_evaluation_v1/phase8_internal_external_comparison_corrected_v1.json | descriptive |

## Statistical Comparisons

The saved statistical policy uses case/patient-level bootstrap resampling with seed `1729`, `10000` resamples, `0.95` confidence level, and `percentile` intervals.
Phase 10 added no new significance tests and reports no new p-values.

## Limitations

- The evaluated external result is negative and must not be reframed as a partial success.
- External evaluation is limited to the artifact-recorded eligible case count.
- Unsupported Phase 3-7 numeric results are unavailable in durable Phase-10 inventory and are intentionally omitted.
- This repository does not commit raw medical data, checkpoints, predictions, model weights, credentials, or secrets.

## Reproducibility

The report is generated from the P10-A inventory, P10-B tables, figures, statistics JSON, and saved Phase 8 comparison artifact. The required Gate 8 command is:

```bash
uv run snakemake --cores 4 --rerun-incomplete reports/final_report.html
```

## Final Conclusion

Phase 10 packages the completed evidence without changing the scientific result. External validation remains an honest negative result. The core project is complete only if Gate 8 reproduction and final review pass.
