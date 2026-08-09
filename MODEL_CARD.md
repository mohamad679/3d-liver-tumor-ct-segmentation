# Model Card — ProtoEM-CT Phase 10

## Model And Artifact Scope

ProtoEM-CT is a research project for binary liver-tumor segmentation in 3D CT. The Phase 10 package reports the frozen Phase 8 external-validation artifact chain and does not introduce a new model, checkpoint, threshold, support policy, or experiment.

The evaluated Phase 8 candidate is recorded as a MONAI SegResNet-style baseline checkpoint selected using development-validation evidence before external label access. The frozen support policy was `no_support`, and the fixed threshold was `0.5`.

## Intended Use

- Research documentation of a leakage-controlled CT tumor-segmentation workflow.
- Reproduction of final tables, figures, statistics, and report from saved machine-readable artifacts.
- Evidence for engineering and scientific-process capability, including negative-result reporting.

## Non-Intended Use

- Clinical diagnosis, treatment planning, triage, or deployment.
- Patient-specific decision support.
- Claims of robust external generalization or state-of-the-art performance.
- Further model selection, threshold tuning, support-policy tuning, or preprocessing changes using external labels.

## Evaluation Summary

The external cohort is 3D-IRCADb-01. The Phase 10 report records 20 discovered external cases, 15 evaluation-eligible cases, and 5 excluded cases under the preregistered eligibility rule.

External validation was negative. Artifact-backed Phase 8 results include macro tumor Dice `0.0141157` with 95% CI `[0.00434155, 0.025013]`, macro tumor IoU `0.00721538` with 95% CI `[0.00220828, 0.0127937]`, lesion F1 `0.000829379` with 95% CI `[0.000388752, 0.00130496]`, and false-positive lesions per scan `2358.4` with 95% CI `[1893.32, 2872.07]`.

These results do not support clinical readiness or strong external generalization.

## Limitations And Failure Modes

- Very poor external tumor-overlap and lesion-level performance.
- Very high false-positive lesion burden.
- Small external evaluation set after eligibility accounting.
- Internal-vs-external comparison is descriptive only.
- Durable machine-readable Phase 3-7 performance artifacts were unavailable for several planned final-package surfaces, so those results are intentionally omitted rather than inferred.

## Reproducibility And Provenance

Primary Phase 10 provenance files:

- `reports/phase10/artifact_inventory.json`
- `reports/phase10/phase10_b_outputs_manifest.json`
- `reports/phase10/final_report_manifest.json`
- `docs/PHASE10_REPRODUCTION.md`

The final report is generated from machine-readable artifacts. Phase 10 did not rerun training, inference, prediction generation, checkpoint selection, threshold tuning, preprocessing changes, or external-label tuning.
