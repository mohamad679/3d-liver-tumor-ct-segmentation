# Phase 10 Artifact Inventory (P10-A)

Machine-readable provenance inventory built from already-saved artifacts only. No training, inference, or medical-metric recomputation occurred while building this inventory.

- Drive root: `/Volumes/Lexar/ProtoEM-CT/runs`
- Drive available this session: `True`
- Inventory self-hash: `67da0cadfd3d8127f3bfac55f8c3984bd46e6888aa3d6720ad23e8dcf252dea9`
- Artifacts inventoried: `23`
- Metric values inventoried: `64`

## Hash verification results

| Artifact | Field | Status |
| --- | --- | --- |
| `phase2_real_lits_v2/manifest.json` | `manifest_hash` | **match** |
| `phase2_real_lits_v2/split.json` | `split_hash` | **match** |
| `phase2_real_lits_v2/leakage_audit.json` | `audit_hash` | **match** |
| `phase2_real_lits_v2/development_summary.json` | `summary_artifact_hash` | **match** |
| `phase2_real_lits_v2/development_qa_report.json` | `qa_artifact_hash` | **match** |
| `phase2_real_lits_v2/manifest.json` | `raw_file_sha256` | **match** |
| `phase8_definitive_training_v1/checkpoints/phase8_definitive_checkpoint_step_500.pt` | `raw_file_sha256` | **match** |
| `phase8_definitive_freeze_v1/phase8_decision_freeze.json` | `raw_file_sha256` | **match** |
| `phase8_definitive_freeze_v1/phase8_decision_freeze.json` | `freeze_inventory_hash` | **match** |
| `phase8_external_preregistration_v1/phase8_external_preregistration.json` | `preregistration_hash` | **match** |
| `phase8_external_image_inference_v1/phase8_external_prediction_lock.json` | `lock_hash` | **match** |
| `phase8_external_evaluation_v1/phase8_external_metric_report.json` | `artifact_hash` | **match** |
| `phase8_external_evaluation_v1/phase8_external_domain_shift_record.json` | `domain_shift_record_hash` | **match** |
| `phase8_external_evaluation_v1/phase8_internal_external_comparison.json (original, unchanged)` | `raw_file_sha256` | **match** |
| `phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json` | `accounting_hash` | **match** |
| `phase8_external_evaluation_v1/phase8_final_closure_reproduction_record.json` | `closure_record_hash` | **match** |

## Metrics by phase

### phase2_data_qa (18 values)

| Name | Value | Kind | Source |
| --- | --- | --- | --- |
| `development_case_count` | `131` | descriptive | `phase2_real_lits_v2/manifest.json#case_count` |
| `train_patient_count` | `91` | descriptive | `phase2_real_lits_v2/split.json#train_patient_count` |
| `validation_patient_count` | `20` | descriptive | `phase2_real_lits_v2/split.json#validation_patient_count` |
| `internal_test_patient_count` | `20` | descriptive | `phase2_real_lits_v2/split.json#internal_test_patient_count` |
| `train_case_count` | `91` | descriptive | `phase2_real_lits_v2/split.json#train_case_count` |
| `validation_case_count` | `20` | descriptive | `phase2_real_lits_v2/split.json#validation_case_count` |
| `internal_test_case_count` | `20` | descriptive | `phase2_real_lits_v2/split.json#internal_test_case_count` |
| `geometry_qa_passed_case_count` | `131` | descriptive | `phase2_real_lits_v2/geometry_qa.json#passed_case_count` |
| `geometry_qa_failed_case_count` | `0` | descriptive | `phase2_real_lits_v2/geometry_qa.json#failed_case_count` |
| `total_lesion_count` | `908` | descriptive | `phase2_real_lits_v2/lesion_components.json#sum(case_records[*].lesion_count)` |
| `aggregate_intensity_mean` | `-543.7532124012079` | descriptive | `phase2_real_lits_v2/development_summary.json#aggregate_intensity_mean` |
| `aggregate_intensity_std` | `530.4038308097652` | descriptive | `phase2_real_lits_v2/development_summary.json#aggregate_intensity_std` |
| `aggregate_intensity_min` | `-2048.0` | descriptive | `phase2_real_lits_v2/development_summary.json#aggregate_intensity_min` |
| `aggregate_intensity_max` | `27572.0` | descriptive | `phase2_real_lits_v2/development_summary.json#aggregate_intensity_max` |
| `lesion_volume_mean_mm3` | `11249.171292937193` | descriptive | `phase2_real_lits_v2/development_summary.json#lesion_volume_mean_mm3` |
| `lesion_volume_median_mm3` | `539.368090150422` | descriptive | `phase2_real_lits_v2/development_summary.json#lesion_volume_median_mm3` |
| `lesion_volume_min_mm3` | `0.347994607721148` | descriptive | `phase2_real_lits_v2/development_summary.json#lesion_volume_min_mm3` |
| `lesion_volume_max_mm3` | `968636.2090784109` | descriptive | `phase2_real_lits_v2/development_summary.json#lesion_volume_max_mm3` |

### phase8_external_validation (46 values)

| Name | Value | Kind | Source |
| --- | --- | --- | --- |
| `macro_dice` | `0.014115733440605011` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.macro_dice` |
| `pooled_dice` | `0.021955736330271848` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.pooled_dice` |
| `macro_iou` | `0.007215381213910843` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.macro_iou` |
| `macro_hd95_mm` | `176.6259570403819` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.macro_hd95_mm` |
| `macro_normalized_surface_dice` | `0.0033014104480338863` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.macro_normalized_surface_dice` |
| `macro_lesion_recall` | `0.15886788048552752` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.macro_lesion_recall` |
| `macro_lesion_precision` | `0.0004170230872244496` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.macro_lesion_precision` |
| `macro_lesion_f1` | `0.0008293785379380446` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.macro_lesion_f1` |
| `mean_false_positive_lesions_per_scan` | `2358.4` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.mean_false_positive_lesions_per_scan` |
| `mean_signed_volume_error_ml` | `-30.97904055175781` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.mean_signed_volume_error_ml` |
| `mean_absolute_volume_error_ml` | `73.1784307937622` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.mean_absolute_volume_error_ml` |
| `macro_relative_volume_error` | `2.833886301727687` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.macro_relative_volume_error` |
| `case_count` | `15` | confirmatory | `phase8_external_evaluation_v1/phase8_external_metric_report.json#aggregate.case_count` |
| `false_positive_lesions_per_scan_ci_low` | `1893.32` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.false_positive_lesions_per_scan.ci_low` |
| `false_positive_lesions_per_scan_ci_high` | `2872.075` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.false_positive_lesions_per_scan.ci_high` |
| `lesion_f1_ci_low` | `0.00038875182666259906` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.lesion_f1.ci_low` |
| `lesion_f1_ci_high` | `0.001304960931868939` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.lesion_f1.ci_high` |
| `lesion_wise_precision_ci_low` | `0.00019535650440340803` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.lesion_wise_precision.ci_low` |
| `lesion_wise_precision_ci_high` | `0.0006562260559801865` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.lesion_wise_precision.ci_high` |
| `lesion_wise_recall_ci_low` | `0.04815359477124183` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.lesion_wise_recall.ci_low` |
| `lesion_wise_recall_ci_high` | `0.31025676937441643` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.lesion_wise_recall.ci_high` |
| `tumor_dice_ci_low` | `0.004341547521318757` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_dice.ci_low` |
| `tumor_dice_ci_high` | `0.02501302650455788` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_dice.ci_high` |
| `tumor_hd95_ci_low` | `159.32400767790955` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_hd95.ci_low` |
| `tumor_hd95_ci_high` | `194.301561440465` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_hd95.ci_high` |
| `tumor_iou_ci_low` | `0.002208283278991875` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_iou.ci_low` |
| `tumor_iou_ci_high` | `0.012793720568727924` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_iou.ci_high` |
| `tumor_normalized_surface_dice_ci_low` | `0.0008695378349168014` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_normalized_surface_dice.ci_low` |
| `tumor_normalized_surface_dice_ci_high` | `0.0061626388043957` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_normalized_surface_dice.ci_high` |
| `tumor_volume_error_absolute_ml_ci_low` | `36.18556097774507` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_volume_error_absolute_ml.ci_low` |
| `tumor_volume_error_absolute_ml_ci_high` | `126.89318989557269` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_volume_error_absolute_ml.ci_high` |
| `tumor_volume_error_relative_ci_low` | `0.7546956418899935` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_volume_error_relative.ci_low` |
| `tumor_volume_error_relative_ci_high` | `5.484524303515011` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_volume_error_relative.ci_high` |
| `tumor_volume_error_signed_ml_ci_low` | `-96.51345051441191` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_volume_error_signed_ml.ci_low` |
| `tumor_volume_error_signed_ml_ci_high` | `18.267631285171493` | confirmatory | `phase8_external_evaluation_v1/phase8_external_bootstrap_ci.json#metrics.tumor_volume_error_signed_ml.ci_high` |
| `external_discovered_count` | `20` | descriptive | `phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json#discovered_count` |
| `external_image_readable_count` | `20` | descriptive | `phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json#image_readable_count` |
| `external_image_qa_eligible_count` | `20` | descriptive | `phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json#image_qa_eligible_count` |
| `external_inference_eligible_count` | `20` | descriptive | `phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json#inference_eligible_count` |
| `external_evaluation_eligible_count` | `15` | descriptive | `phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json#evaluation_eligible_count` |
| `external_excluded_count` | `5` | descriptive | `phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json#excluded_count` |
| `evaluation_eligible_case_count` | `15` | descriptive | `phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json#recomputed(cases[*].excluded==false)` |
| `excluded_case_count` | `5` | descriptive | `phase8_external_evaluation_v1/phase8_external_eligibility_accounting.json#recomputed(cases[*].excluded==true)` |
| `mean_tumor_dice_step_250` | `0.0016380082094079678` | descriptive | `phase8_definitive_training_v1/phase8_definitive_checkpoint_selection_evidence.json#mean_tumor_dice_step_250` |
| `mean_tumor_dice_step_500` | `0.01579295321113191` | descriptive | `phase8_definitive_training_v1/phase8_definitive_checkpoint_selection_evidence.json#mean_tumor_dice_step_500` |
| `derived_median_spacing_xyz_mm` | `[0.767578125, 0.767578125, 1.0]` | descriptive | `phase8_package_b_engineering_verification_v1/phase8_package_b_engineering_report.json#derived_median_spacing` |

## Unavailable (referenced in prose, no backing artifact located)

- **docs/phase8/SUPERVISOR_HANDOFF.md (Substage 3, blocked v1 run)**: Checkpoint at phase8_substage3_tiny_real_v1/checkpoints/phase8_tiny_real_verification_checkpoint.pt reflects pre-fix buggy liver-vs-background training and has no accompanying JSON artifact (run never reached the publication step). — _No JSON summary/config/metadata artifact was ever published for this run; only the stray checkpoint file exists. Must not be used or reported as a scientific result._
- **Gate 3 (ACCEPTANCE_CHECKLIST.md)**: Gate 3 checkboxes for CPU shape smoke tests, tiny-subset overfit tests, deterministic metric JSON from saved predictions, Git-visible-artifact absence check, full local quality checks, GitHub-hosted CI, and Gate 3 close-out documentation are unchecked. — _No Gate 3 close-out evidence block or backing artifact exists in the repository or on the external drive for these specific items; the checklist itself records them as not yet done (pre-existing repo state, not a Phase 10 blocker)._
- **Phase 10 baseline-performance table scope**: Phase 3 nnU-Net v2 and MONAI SegResNet baseline-performance metric rows. — _No persistent machine-readable baseline-performance artifact was located in the repository or under /Volumes/Lexar/ProtoEM-CT/runs; Gate 3 remains partly unchecked and must not be reported as completed scientific baseline performance._
- **ACCEPTANCE_CHECKLIST.md Gate 5**: Gate 5 status/criteria. — _Gate 5 is recorded as 'Pending definition' with no criteria and no backing artifact._
- **Phase 10 retrieval/prototype/foundation-baseline scope**: Phase 5 retrieval, prototype, and foundation-baseline performance rows. — _No durable Phase 5 publication artifact directory or Gate 5 acceptance evidence was located; Gate 5 is pending definition. These values are therefore unavailable and must not be inferred from code or prose._
- **ACCEPTANCE_CHECKLIST.md Gate 7**: Gate 7 status/criteria. — _Gate 7 is recorded as 'Pending definition' with no criteria and no backing artifact, despite docs/LEAKAGE_AUDIT.md describing a Phase 7 leakage/scope verification pass._
- **docs/PHASE_PLAN.md / ACCEPTANCE_CHECKLIST.md Gate 4**: Phase 4 few-shot protocol support manifests, adaptation configs, and protocol-table artifacts (K x replicate x adaptation-mode). — _Gate 4 evidence was produced from a synthetic publication run using test fixtures into an ephemeral external temporary output root; no persistent Phase 4 artifact directory was found under /Volumes/Lexar/ProtoEM-CT/runs. Only the counts/hashes recorded in ACCEPTANCE_CHECKLIST.md and docs/LEAKAGE_AUDIT.md are available as provenance, not independently re-verified raw files this session._
- **ACCEPTANCE_CHECKLIST.md Gate 6**: Phase 6 / ProtoEM-CT ablation-comparison, objective-trace, and publication artifacts (ablation_comparison.json, objective_trace.json, run_summary.json, etc.). — _Gate 6 evidence was produced from two independent synthetic runs into mktemp directories (ROOT_A/ROOT_B) that were not persisted; no Phase 6 output directory was found under /Volumes/Lexar/ProtoEM-CT/runs. Only the values quoted in ACCEPTANCE_CHECKLIST.md are available._
- **docs/LEAKAGE_AUDIT.md Phase 7 section**: Phase 7 robustness/uncertainty evaluation artifacts (calibration, risk-coverage, degradation, lesion-subgroup reports). — _Phase 7 CLI is documented as 'synthetic_mode_only: true' with no real-data execution; no Phase 7 output directory exists under /Volumes/Lexar/ProtoEM-CT/runs. Robustness/uncertainty is explicitly 'not_included' in the Wave 4 statistical policy for Phase 8 as well._
- **docs/phase8/FINAL_REPORT.md section Q**: Git commit 51638ca662510bf4cb031b2aec29778bf26bba26 (comparison provenance fix) and f1d4e1be7b6f7aa3657792cae9125c825affdded / f264ce79dd68039256d140f03c468baed8ecea07 (Package C execution/capture). — _Not re-verified in this P10-A session (out of this agent's Git-log scope); FINAL_REPORT.md section S already records these as independently re-verified in the Phase 8 closure session. Flagged here as inherited, not re-checked, provenance._

## Conflicts between prose and artifacts

None found this session.

