# Leakage Audit

## Phase 2 Audit Status

Status: PASSED for the verified v2 development-cohort leakage audit.

This document records aggregate leakage evidence from the approved Phase 2 v2 read-only
development-cohort execution. The evidence is derived from saved machine-readable JSON artifacts and
their verified hashes. It is dataset-QA and leakage evidence only; it is not model-performance,
external-validation, clinical-validity, or Phase 3 evidence.

MSD Task03 Liver is treated as the LiTS-derived development cohort and not as an independent second
cohort. Real 3D-IRCADb-01 remains untouched for later Phase 8 external validation.

## Reproducibility Evidence

- Source development manifest hash:
  `c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b`
- Split-policy name and version: `patient_hash_rank_v1`
- Split seed: `1729`
- Split artifact hash: `936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb`
- Geometry QA artifact hash:
  `b8e65c558d5f574a1b40f9a5b15a033efbfbd6952006324fae35173940f94ee2`
- Lesion-components artifact hash:
  `f7ce4874801cc8d5387bf93fd1480d376db4028026a90723dda2c5e2f48fbc60`
- Development-summary artifact hash:
  `9b721e10232a08c9a5dd5361a11d7595abab04c071058d584e5895087b91951c`
- Final development-QA artifact hash:
  `3f816db6227a1d2a8b349e248a9654eb4eed1e57760b48d32a6a275e1deca4f1`
- Leakage-audit artifact hash:
  `a32da3ac9d5689ec055c3875076ddc2075ca585219160c2d4200ca36484bf199`
- Audit generation timestamp supplied explicitly: `2026-07-27T01:00:00Z`

All stored artifact hashes recomputed successfully, and all cross-artifact links were exact.

## Partition Counts

| Partition | Patient count | Case count |
| --- | ---: | ---: |
| Train | 91 | 91 |
| Validation | 20 | 20 |
| Immutable internal test | 20 | 20 |
| Total unique | 131 | 131 |

Assignment completeness passed: every manifest case appears exactly once in the split, every split
case exists in the manifest, and train, validation, and immutable internal-test partitions are
nonempty.

## Pairwise Overlap Evidence

| Partition pair | Patient overlap count | Case overlap count |
| --- | ---: | ---: |
| Train / validation | 0 | 0 |
| Train / immutable internal test | 0 | 0 |
| Validation / immutable internal test | 0 | 0 |

## Duplicate and Near-Duplicate Evidence

- Cross-partition image SHA-256 overlap count: 0
- Cross-partition label SHA-256 overlap count: 0
- Cross-partition complete image/label hash-pair overlap count: 0
- Leakage finding codes: none
- Cross-partition near-duplicate policy and method: not executed in Phase 2; no Phase 2 external
  cohort comparison is authorized.

## Cohort and Processing Confirmations

- LiTS/MSD equivalence warning retained: LiTS labeled training data is the development cohort, and
  MSD Task03 Liver is derived from LiTS rather than an independent cohort.
- Real 3D-IRCADb-01 external data was not accessed, scanned, inventoried, QA'd, split, tuned on, or
  summarized during Phase 2: confirmed
- External labels did not influence development decisions: confirmed
- External-data access flag in the leakage artifact: false
- Preprocessing-on-nontraining-data flag in the leakage artifact: false
- No preprocessing was fitted on validation, immutable internal-test, or external data: confirmed
- Split creation occurred before any data-dependent preprocessing fit: confirmed
- Lesion-aware stratification was not used for this split: confirmed
- Every development patient and case was assigned exactly once: confirmed
- Real-data inputs were supplied through an explicit lawful local root and outputs stayed external
  to Git: confirmed

## Unresolved Findings

| Severity | Finding | Evidence reference | Required resolution | Status |
| --- | --- | --- | --- | --- |
| None | No leakage finding codes were recorded | Verified v2 leakage-audit artifact | None | Closed |

Gate 2 requires no unresolved critical finding. The verified v2 leakage audit has no unresolved
critical findings.

## Review and Sign-Off

- Evidence reviewed: manifest, split, geometry QA, lesion components, development summary, final QA,
  and leakage-audit JSON artifacts parsed through the Phase 2 serializer
- Unresolved critical findings: 0
- Gate 2 leakage-audit recommendation: pass
- Reviewer sign-off: pending repository review and commit

## Phase 4 Support-to-Internal-Test Verification

Status: PASSED for deterministic Phase 4 support-manifest publication verification on 2026-07-30.

This Phase 4 note records support/internal-test separation evidence only. It is not adaptation
training, inference, model-performance, duration, memory, external-validation, or clinical-validity
evidence.

- Verification surface: deterministic Phase 4 support-manifest generation, deterministic protocol
  publication CLI, and generation-summary overlap counts from one synthetic publication using the
  existing test-fixture constructors
- Verified support-manifest count: `15`
- Verified adaptation-config count: `45`
- Verified protocol-table row count: `45`
- Verified protocol Markdown data-row count: `45`
- Support/internal-test patient overlap count: `0`
- Support/internal-test case overlap count: `0`
- Leakage check passed: `true`
- Deterministic regeneration across two external output roots: `true`

Phase 4 retained the immutable internal-test cohort boundary and published no tracked support
artifacts, checkpoints, predictions, or medical data into the repository.

## Phase 6 Query-Label and Evaluation-Mask Isolation

Status: PASSED for the verified local Gate 6 synthetic close-out on 2026-08-02.

This Phase 6 note records leakage-boundary evidence only. It is not model-performance,
external-validation, robustness, uncertainty, calibration, GPU, or clinical-validity evidence.

- Verification surface:
  - repository-wide `uv run pytest -q`: PASS, `1115 passed, 3 skipped`
  - repository-wide `uv run mypy src`: PASS
  - two independent synthetic `uv run protoem-ct run-phase6-protoem` executions: PASS
- Phase 6 query-label leakage result: no query labels enter the initialization, E-step, M-step,
  optimization, stopping, final-inference, or ablation execution APIs
- Public-API enforcement evidence is covered by the Phase 6 unit and integration tests for
  initialization, objective, state, optimization, stopping, inference, learned schedule, ablation,
  and transductive flow
- Synthetic reference-mask isolation result: synthetic reference masks are used only after final
  prediction for deterministic Dice/IoU comparison fields. They do not alter initialization
  identity, objective traces, stopping records, inference identities, or prediction hashes
- Support/query patient and case overlap remained rejected by the completed Phase 5 boundary and
  Phase 6 initialization contract
- Real 3D-IRCADb-01 access remained absent: confirmed
- No external labels influenced Phase 6 configuration, stopping, threshold selection, ablation
  execution, or publication: confirmed

Gate 6 leakage recommendation: pass.

## Phase 7 Robustness/Uncertainty Leakage and Scope Verification

Status: PASSED for the focused P7-LEAKAGE-SAFETY verification on 2026-08-02.

This Phase 7 note records robustness/uncertainty leakage-boundary evidence only. It is not
external-validation, real-data, GPU, clinical-validity, theoretical robustness, or model-performance
evidence.

- Verification surface:
  - `uv run ruff format tests/unit/test_phase7_leakage.py`: PASS, `1 file left unchanged`
  - `uv run ruff check tests/unit/test_phase7_leakage.py`: PASS
  - `uv run mypy tests/unit/test_phase7_leakage.py`: PASS
  - `uv run pytest tests/unit/test_phase7_leakage.py`: PASS, `6 passed`
- Phase 7 config and CLI mode: `configs/phase7_robustness_uncertainty.yaml` is validated as
  `synthetic_mode_only: true`, and the Phase 7 command surface exposes only
  `run-phase7-robustness-uncertainty --config --output-root`.
- External-cohort boundary: focused tests found no Phase 8, 3D-IRCADb-01, LLM/VLM, checkpoint,
  download, private-data, or identifiable-data pathway in the Phase 7 config or Phase 7
  robustness/uncertainty/evaluation/publication source surfaces.
- Label/reference-mask isolation:
  - corruption and uncertainty prediction APIs expose no query-label, reference-mask, model,
    checkpoint, download, or dataset-root prediction parameters;
  - optional transform masks are limited to binary geometry/evaluation companions and are not used
    to choose corruption parameters or prediction behavior;
  - calibration, risk-coverage, failure-detection, degradation, and lesion-subgroup functions are
    post-prediction evaluation surfaces;
  - Phase 6 optimization and final-inference APIs remain query-label/reference-mask free.
- Synthetic reference masks in the bounded Phase 7 CLI are generated in memory and used only after
  prediction/probability construction for evaluation artifacts: calibration, risk coverage,
  uncertainty-error/failure analysis, degradation, and lesion subgroup reporting.
- Generated-artifact hygiene: the focused test executes the bounded synthetic Phase 7 CLI into a
  pytest temporary directory, compares `git status --porcelain --untracked-files=all` before and
  after the run, and verifies that no Phase 7 publication filename appears at the repository root.

Gate 7 leakage recommendation: pass for the tested Phase 7 leakage/scope surface, pending final
Gate 7 repository-wide verification.
