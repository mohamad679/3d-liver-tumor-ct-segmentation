# Phase Plan

## Current Status

Active phase: none.

The core ProtoEM-CT research package is complete. Phase 8 external validation is closed with a
negative external-validation result, Phase 9 optional LLM/VLM work was not executed, and Phase 10
final scientific packaging is closed. Gate 8 passed for artifact-only final-report reproduction.

## Scientific Scope

ProtoEM-CT studies binary liver-tumor segmentation in 3D CT. Liver context may be used for
preprocessing, masking, quality control, or analysis, but the primary segmentation target is tumor
foreground versus background.

The central research question is whether a strictly leakage-controlled 3D CT segmentation workflow,
including limited-data and few-shot protocol components, can produce reproducible evidence about
generalization under domain shift.

## Cohort Boundaries

The development source is LiTS / MSD Task03 Liver treated as one LiTS-derived development source.
LiTS and MSD Task03 Liver must not be interpreted as independent cohorts.

The external validation cohort is 3D-IRCADb-01. External labels must not influence preprocessing,
checkpoint selection, threshold selection, support policy, model selection, postprocessing, report
selection, or protocol iteration.

No medical data, NIfTI/DICOM files, predictions, model weights, checkpoints, private dataset paths,
credentials, or large generated artifacts may be committed to Git.

## Phase Summary

### Phase 0: Repository and Validation Foundation

Completed. The project established Python 3.11 packaging, uv, Ruff, mypy, pytest, pre-commit,
GitHub Actions, synthetic NIfTI fixtures, and the `protoem-ct validate-pair` validation surface.

### Phase 1: Synthetic Reproducible DAG

Completed. A Snakemake DAG verifies a JSON-first synthetic workflow:
`create_data -> validate -> preprocess -> infer_dummy -> evaluate -> report`. Phase 1 contains no
scientific model result.

### Phase 2: Development-Cohort QA and Leakage Controls

Completed. The LiTS-derived development cohort has an anonymous manifest, patient-level
train/validation/internal-test split, geometry and label QA, lesion summaries, histogram summaries,
and leakage-audit evidence. The verified development split is 91 train, 20 validation, and 20
immutable internal-test cases/patients.

### Phase 3: Baseline Infrastructure

Completed for later-phase planning and integration. The repository includes nnU-Net v2 and MONAI
SegResNet baseline infrastructure, metric contracts, path safety, checkpoint/provenance contracts,
and synthetic or bounded verification surfaces. It does not claim strong baseline efficacy.

### Phase 4: Few-Shot Protocol

Completed. The project defines deterministic support manifests for `K = 1, 2, 5, 10, 20`, fixed
replicates, patient/case leakage checks, adaptation-mode contracts, and protocol-table artifacts.
Gate 4 evidence remains protocol evidence, not a claim of adaptation performance.

### Phase 5: Retrieval and Prototype Foundation

Completed for the non-transductive retrieval/prototype foundation scope. The phase defines typed
feature-encoder and prototype-conditioned segmentation interfaces, deterministic embedding/prototype contracts,
cosine retrieval, prototype-only inference, and synthetic verification. It does not claim external
validation or clinical performance.

### Phase 6: ProtoEM-CT Adaptation Contracts

Completed. Phase 6 implements bounded ProtoEM-CT objective-driven transductive adaptation contracts
with persisted objective traces, stopping records, failure handling, leakage tests, and synthetic
publication. It does not claim theoretical convergence, GPU execution, external validation, or
clinical validity.

### Phase 7: Robustness and Uncertainty Contracts

Completed for internal workflow contract coverage. The repository contains deterministic
robustness, uncertainty, calibration, risk-coverage, failure-analysis, degradation, subgroup, and
publication contracts with synthetic verification. Phase 7 results are not used to tune external
validation.

### Phase 8: External Validation

Closed on 2026-08-09 with a negative result. The definitive checkpoint, preprocessing, support
policy, threshold, metric configuration, bootstrap configuration, and publication policy were frozen
before external evaluation. Predictions were locked before external labels were opened. External
metrics were computed only after the frozen prediction lock.

Development-validation tumor Dice for the selected checkpoint was approximately `0.0158`. External
macro tumor Dice on evaluation-eligible 3D-IRCADb-01 cases was approximately `0.0141`, with the
recorded 95% confidence interval preserved in the Phase 8 and Phase 10 artifacts. No strong
generalization claim is supported.

### Phase 9: Optional LLM/VLM Track

Skipped by user decision and not executed. No LLM/VLM/RAG result is claimed in the final package.

### Phase 10: Final Scientific Package

Closed. Phase 10 generates the final report, tables, figures, statistics, model card, data card,
release checklist, and release-candidate metadata from saved machine-readable artifacts. It does
not rerun training, inference, prediction generation, checkpoint selection, threshold selection,
preprocessing changes, or external-label tuning.

## Gate 8 Evidence

Gate 8 passed for final scientific packaging and artifact-only report regeneration. Current Phase 10
outputs include:

- `reports/phase10/artifact_inventory.json`
- `reports/phase10/phase10_b_outputs_manifest.json`
- `reports/phase10/final_report_manifest.json`
- `reports/final_report.html`
- `reports/phase10/release_candidate.json`

Validated final-report regeneration command:

```bash
uv run snakemake --cores 4 --rerun-incomplete reports/final_report.html
```

## Final Gate Constraints

- Treat LiTS / MSD Task03 Liver as one development source.
- Preserve external-label isolation and prediction-lock ordering.
- Preserve the negative external-validation result without reframing it as success.
- Preserve machine-readable artifacts as the source of reported values.
- Do not commit medical data, predictions, checkpoints, model weights, private dataset paths,
  credentials, or large generated artifacts.
