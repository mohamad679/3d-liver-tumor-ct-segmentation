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
