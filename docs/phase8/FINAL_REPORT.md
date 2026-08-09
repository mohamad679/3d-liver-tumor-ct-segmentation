# Phase 8 Final Report — External Validation (3D-IRCADb-01)

Status: Phase 8 is **CLOSED** with a **negative external-validation result**. This report is the
authoritative Phase 8 closure document. It contains no medical data, medical arrays, DICOM/NIfTI
content, or PHI; it references only machine-readable, non-medical provenance and metric artifacts
recorded during Packages C through H.

## A. Phase 8 objective

Phase 8 executed the single, preregistered external-validation protocol for the frozen ProtoEM-CT
Phase 8 baseline candidate against the real, previously untouched 3D-IRCADb-01 cohort, without any
external-data-driven tuning, model selection, threshold selection, or protocol iteration, and
reported the result — positive or negative — exactly as obtained.

## B. Frozen model/configuration

The evaluated model is the frozen Package C definitive-training checkpoint at step 500, executed
under `Phase8DefinitiveConfig` hash
`4e0075e0d499080c4065031d42ea7ff25637cafe154b6757c05edbc14e8c0624`. Every configuration field
relevant to external inference (preprocessing, spacing, support policy, threshold) is bound to this
hash and is fail-closed verified against it at inference and evaluation time (Packages F and G).

## C. Preprocessing freeze

Preprocessing (orientation to RAS, resampling to the frozen target spacing, intensity handling) is
fixed by the Package C/D freeze and re-verified, not re-derived, at external inference time. No
preprocessing parameter was changed after the internal development-validation checkpoint selection
in Package C, and none was changed based on any external result.

## D. Target spacing provenance

Target spacing `(0.767578125, 0.767578125, 1.0)` mm (x, y, z) was derived exclusively from the 91
LiTS/MSD-Task03-Liver **train**-partition cases (`derived_median_spacing` in the Package B
engineering report, preprocessing/config identity
`286b4a200c147718cfe885850e670254a52e8e1ec5d9683d78fd36af4f293d37`). No development-validation or
internal-test volume, and no external volume or label, contributed to this derivation
(`docs/DECISIONS.md`, "2026-08-09: Approve Target Spacing Provenance for Package C").

## E. Support policy

`no_support` (locked baseline candidate; no few-shot support set was used for the evaluated
checkpoint).

## F. Threshold

Fixed constant threshold `0.5`, frozen as part of `Phase8DefinitiveConfig` and explicitly recorded
as "not tuned on external data" in the Package D decision-freeze rationale
(`fixed-constant-threshold-not-tuned-on-external-data`).

## G. Selected checkpoint identity/hash

Checkpoint step `500`, SHA-256 `2d7989fd134b1348e82cc52afbcf4738c0ce3c17e9c68df774431f577dede651`
(`REQUIRED_CHECKPOINT_SHA256` in `src/protoem_ct/external/image_only_inference.py`, re-verified
against `/Volumes/Lexar/ProtoEM-CT/runs/phase8_definitive_training_v1/checkpoints/phase8_definitive_checkpoint_step_500.pt`
this closure session — see Reproduction Check, section S).

## H. Development model-selection evidence

Package C compared exactly two candidates on the identical, complete 20-case development-validation
set, before any external evaluation:

- step 250: mean tumor Dice `0.0016380082094079678`
- step 500: mean tumor Dice `0.01579295321113191`

Step 500 was selected because `0.01579295321113191 > 0.0016380082094079678`, per the locked
`mean_tumor_dice` selection metric with earliest-step tie-break policy (not a tie in this
comparison). `internal_test_used=false`, `external_data_used=false`, `external_labels_used=false`
for this selection. The selected checkpoint's development-validation tumor Dice is itself very low
in absolute terms (see Limitations, section R).

## I. Preregistration identity

`phase8_external_preregistration.json`, `preregistration_hash`
`75d287ade45d2a778153885838d8635606ba5ae739ed88dbf0580671258db291`, published before any external
label was accessed.

## J. Prediction-lock identity

`phase8_external_prediction_lock.json`, `lock_hash`
`190532a6abb7c3308de9abe7dc7455f4fafae464d51347ea2273874d13afdc94`, produced by Package F
(image-only inference, no label access) and locked before Package G ever opened an external label.

## K. External cohort and eligibility

Cohort: 3D-IRCADb-01, 20 discovered cases. Under the preregistered post-label eligibility accounting
(`phase8_external_eligibility_accounting.json`, `accounting_hash`
`670e9dc3e128a538c784804f4d23ebc1d4c7459943f08e4f2af5020c474a0af2`):

- discovered: 20
- image-readable / image-QA-eligible / inference-eligible: 20 / 20 / 20
- evaluation-eligible: **15**
- excluded (not evaluated): **5** — `ext-ircadb-005`, `ext-ircadb-007`, `ext-ircadb-011`,
  `ext-ircadb-014`, `ext-ircadb-020`, each excluded for reason `tumor_target_absent` under the
  preregistered missing-source/eligibility rule. This is a fail-closed geometry/label-source
  exclusion, decided by the preregistered rule alone — **it is not a performance-based exclusion**
  and must not be reinterpreted as one.

## L. Label mapping

External tumor labels were sourced exclusively from `MASKS_DICOM.zip` role folders whose name
case-insensitively starts with `livertumor`; the `liver` (liver-context) folder and all other organ
folders, `LABELLED_DICOM.zip`, `MESHES_VTK.zip`, and `liver_*.jpg` files were never opened. The
mapping/eligibility policy is the same, unmodified Wave 3 policy frozen in Package D
(`label_mapping_policy_reference` artifact hash
`2781c3d7eadfafc1b384d34dab0e077528816e0c47ebeda3016c028f5ed00ced`).

## M. External metrics (all 9, as computed and published in Package G — not recomputed here)

Bootstrap policy for all metrics: 10,000 case-level (patient/case) resamples, seed `1729`, 95%
percentile intervals (`phase8_external_bootstrap_ci.json`).

| Metric | Point estimate | 95% CI |
|---|---|---|
| Tumor Dice (macro) | 0.01412 | [0.00434, 0.02501] |
| Tumor Dice (pooled) | 0.02196 | n/a |
| Tumor IoU (macro) | 0.00722 | [0.00221, 0.01279] |
| Tumor HD95 (macro, mm; 15/15 defined) | 176.63 | [159.32, 194.30] |
| Tumor NSD (macro) | 0.00330 | [0.00087, 0.00616] |
| Lesion recall (macro) | 0.1589 | [0.0482, 0.3103] |
| Lesion precision (macro) | 0.000417 | [0.000195, 0.000656] |
| Lesion F1 (macro) | 0.000829 | [0.000389, 0.001305] |
| False-positive lesions/scan (mean) | 2358.4 | [1893.3, 2872.1] |
| Tumor volume error, signed (mL, mean) | -30.98 | [-96.51, 18.27] |
| Tumor volume error, absolute (mL, mean) | 73.18 | [36.19, 126.89] |
| Tumor volume error, relative (macro) | 2.834 | [0.755, 5.485] |

Source: `phase8_external_metric_report.json` (`artifact_hash`
`a6dbdd3998e2c55d82e725cba20fbfebf685378f0e719547df5d008eeae0b2da`) and
`phase8_external_bootstrap_ci.json`, both under
`/Volumes/Lexar/ProtoEM-CT/runs/phase8_external_evaluation_v1`. These values were **not
recomputed** during Phase 8 closure; they are quoted verbatim from the already-committed Package G
artifacts (cross-checked against the mounted external drive this session — see section S).

## N. Bootstrap CI policy and results

See section M. `interval_method: percentile`, `resampling_unit: case_patient`, `resample_count:
10000`, `random_seed: 1729`, `confidence_level: 0.95`, applied uniformly across all 9 evaluated
metric families with no metric-specific deviation.

## O. Domain-shift summary

`phase8_external_domain_shift_record.json` (`domain_shift_record_hash`
`04665b449d8b3d7838a28e3045cdd57b67c9fccf6e1f091b7f5f1abd5963da0b`) records **aggregate-only**
external-cohort characteristics across 11 dimensions (source identity, acquisition representation,
image matrix, slice counts, voxel spacing, anisotropy, orientation, HU readiness, modality, image-QA
compatibility, and differences vs. the frozen internal preprocessing contract). Every dimension's
`comparison_status` field is `"not_compared"`: the artifact reports external aggregate statistics
only and does **not** perform or claim a matched statistical comparison against an internal
distribution. Consistent with the task boundary, **this report does not claim that any observed or
unobserved domain shift caused the external performance result** — the committed artifact does not
support a causal claim, only descriptive aggregate characterization.

## P. Anonymous qualitative-output reference

`phase8_external_qualitative_index.json` indexes all 15 evaluation-eligible cases (selection rule:
`include_all_evaluation_eligible_cases_when_count_at_most_20`, tie-break:
`anonymous_case_id_lexicographic`). No case was cherry-picked, and no qualitative case was
regenerated during this closure. No medical data from that index is reproduced in this report.

## Q. Corrected internal-vs-external comparison reference

The original Package G artifact, `phase8_internal_external_comparison.json`
(SHA-256 of the raw published file: `a082e2588b6aa9f10d64dcd90d89e1f1c8dd9bb855b71c94f3994cbca13cc8dc`),
contained a provenance-field defect: `internal_checkpoint_sha256=null` and
`internal_checkpoint_matches_locked_checkpoint=false`, caused by a wrong dict-key lookup
(`"checkpoint_sha256"` instead of the schema's actual `"selected_checkpoint_hash"` key) in
`_build_internal_external_comparison` (`src/protoem_ct/external/label_evaluation.py`). This defect
was **non-scientific**: it never touched predictions, external labels, eligibility, per-case
metrics, aggregate metrics, bootstrap CIs, checkpoint selection, or any metric value.

The bug was fixed minimally (commit `51638ca662510bf4cb031b2aec29778bf26bba26`, "fix(phase8):
correct comparison checkpoint provenance"), covered by a focused regression test, and used to
publish a new, additive artifact — **without overwriting or deleting the original** —
`phase8_internal_external_comparison_corrected_v1.json` under the same Package G output root
(`/Volumes/Lexar/ProtoEM-CT/runs/phase8_external_evaluation_v1/`). The corrected artifact was
verified field-by-field against the original: every field is byte/value-identical except the two
defective provenance fields, which now correctly read
`internal_checkpoint_sha256=2d7989fd134b1348e82cc52afbcf4738c0ce3c17e9c68df774431f577dede651` and
`internal_checkpoint_matches_locked_checkpoint=true`. The corrected artifact embeds a
`correction_provenance` block recording the correction reason, the corrected field paths, the fix
commit, and a reference (relative path + raw-file SHA-256) back to the preserved original. **For
final reporting, the corrected artifact supersedes the original; the original remains the immutable
historical Package G record and was not modified.**

The internal-vs-external comparison itself remains descriptive-only
(`claims_policy: descriptive_only_no_superiority_or_generalization_claims`,
`cohort_relationship: independent_unpaired`) — a single frozen internal scalar with no confidence
interval compared against a bootstrapped external distribution. It supports no superiority or
generalization claim in either direction.

## R. Limitations

- The selected checkpoint (step 500) has a very low development-validation tumor Dice
  (`0.01579295321113191`) — roughly 1.6% — even before any external evaluation.
- External tumor Dice, IoU, and NSD are very poor in absolute terms (macro Dice `0.01412`, IoU
  `0.00722`, NSD `0.00330`), and lesion-level precision/F1 are extremely poor (precision
  `0.000417`, F1 `0.000829`).
- False-positive lesion burden is very high (mean `2358.4` false-positive lesions per scan).
- Five of the 20 discovered IRCADb cases (`ext-ircadb-005`, `ext-ircadb-007`, `ext-ircadb-011`,
  `ext-ircadb-014`, `ext-ircadb-020`) lacked the preregistered tumor-target label source
  (`livertumor*` folder) and were not evaluable under the locked fail-closed eligibility rule; this
  is a geometry/label-source exclusion, not a performance-based one.
- External evaluation contains **15** eligible cases; all reported external statistics are over
  this 15-case cohort.
- The internal-vs-external comparison is descriptive-only: one frozen internal point estimate
  without a confidence interval versus a bootstrapped external distribution; it makes no
  superiority or generalization claim.
- No external tuning of any kind (hyperparameter, threshold, preprocessing, model selection, or
  protocol) was allowed or performed at any point in Phase 8.
- LiTS and MSD Task03 Liver are the same underlying cohort and are treated throughout as one
  development source, never as two independent cohorts.
- Package C's real execution ran from base HEAD `f1d4e1be7b6f7aa3657792cae9125c825affdded` in an
  intentionally uncommitted working-tree state; the exact executed file bytes were subsequently
  captured, unmodified, in a later, separate commit `f264ce79dd68039256d140f03c468baed8ecea07`
  ("feat(phase8): add definitive real training driver"). The capture commit is not the execution
  base HEAD and must not be represented as such (`docs/DECISIONS.md`, "2026-08-09: Record Package C
  Execution Provenance"; both commit hashes independently re-verified present in this repository's
  Git history during this closure).
- An earlier, separate incident (accidental loss of an untracked pilot test file,
  `tests/unit/test_phase8_definitive_training_pilot.py`) was addressed by reconstruction and a
  documented remediation/recovery audit (pre-remediation 41 COVERED / 16 WEAK / 36 MISSING of 93
  requirements; reduced to a small residual afterward), plus a separately fixed checkpoint-reload
  defect. This incident is historical, remains documented in `docs/phase8/SUPERVISOR_HANDOFF.md`,
  and is **not** described here as fully or perfectly recovered — only as remediated to a
  documented, audited residual state.
- The original Package G comparison artifact (`phase8_internal_external_comparison.json`) contains
  the known provenance-field defect described in section Q and is superseded for final reporting by
  the explicitly corrected artifact; the original is preserved unmodified as the historical record.

No limitation beyond what is directly supported by the artifacts and decisions cited above is
asserted in this report.

## S. Reproducibility / provenance chain

This closure performed a **non-scientific** reproduction/provenance check only: no training,
inference, or metric computation was rerun. The following chain was verified this session by
loading and/or hash-validating existing artifacts (external drive `/Volumes/Lexar` was mounted and
accessible this session, so these are freshly re-verified, not merely quoted):

| Artifact | Field | Value | Re-verified this session |
|---|---|---|---|
| `phase8_decision_freeze.json` | raw-file SHA-256 | `5f7579418996d93caa398266f5dfd1c5489b50f3720b23a6d6d5e49f08bb928b` | yes |
| checkpoint step 500 `.pt` | SHA-256 | `2d7989fd134b1348e82cc52afbcf4738c0ce3c17e9c68df774431f577dede651` | yes |
| `phase8_external_preregistration.json` | `preregistration_hash` field | `75d287ade45d2a778153885838d8635606ba5ae739ed88dbf0580671258db291` | yes |
| `phase8_external_prediction_lock.json` | `lock_hash` field | `190532a6abb7c3308de9abe7dc7455f4fafae464d51347ea2273874d13afdc94` | yes |
| `phase8_external_metric_report.json` | `artifact_hash` field | `a6dbdd3998e2c55d82e725cba20fbfebf685378f0e719547df5d008eeae0b2da` | yes |
| `phase8_external_domain_shift_record.json` | `domain_shift_record_hash` field | `04665b449d8b3d7838a28e3045cdd57b67c9fccf6e1f091b7f5f1abd5963da0b` | yes |
| `phase8_internal_external_comparison.json` | raw-file SHA-256 (unchanged, pre- and post-closure) | `a082e2588b6aa9f10d64dcd90d89e1f1c8dd9bb855b71c94f3994cbca13cc8dc` | yes |
| `phase8_internal_external_comparison_corrected_v1.json` | published this closure; scientifically-relevant fields verified field-by-field equal to the original | n/a (new artifact) | yes |
| fix commit | Git SHA | `51638ca662510bf4cb031b2aec29778bf26bba26` | yes (local Git) |
| Package C execution-base / capture commits | Git SHA | `f1d4e1be7b6f7aa3657792cae9125c825affdded` / `f264ce79dd68039256d140f03c468baed8ecea07` | yes (local Git) |

Note on the self-hash fields above: these are the artifacts' own canonical-JSON self-hash fields
(e.g. `preregistration_hash`, `lock_hash`, `artifact_hash`), not raw-file `shasum` of the
pretty-printed JSON on disk (which differs due to whitespace/formatting and is not the artifact's
identity contract). Both the freeze artifact and the checkpoint binary have no such internal
self-hash field, so their raw-file/binary SHA-256 is the correct comparison and was verified
directly.

A minimal machine-readable closure/reproduction record for this chain is published at
`phase8_final_closure_reproduction_record.json` under
`/Volumes/Lexar/ProtoEM-CT/runs/phase8_external_evaluation_v1/` (see
`src/protoem_ct/external/closure.py`), containing only these non-medical identities and no medical
array or PHI.

## T. Explicit no-tuning statement

No hyperparameter, threshold, preprocessing, checkpoint-selection, or protocol decision was changed
after any external label was accessed. Checkpoint selection (Package C, section H) completed before
Package E's preregistration was published, which itself completed before Package F's image-only
inference and prediction lock, which itself completed before Package G ever opened an external
label. External labels were used only to compute the metrics reported in section M; they influenced
no design, tuning, or selection decision anywhere in Phase 8.

## U. Explicit negative-result interpretation

External performance is very poor across every one of the 9 preregistered metric families. This is
an honest negative result, not a partial success and not a result awaiting reinterpretation. The
external result must not be reframed as successful generalization. The evaluated checkpoint also had
very low development-validation tumor Dice before any external data was touched (section H),
so the poor external result is consistent with — not contradicted by — the internal evidence
available at selection time. Taken together, the evidence gathered in Phase 8 does **not** support a
claim of strong tumor-segmentation performance or robust external generalization for this frozen
baseline candidate.

## V. Phase 8 closure status

**Phase 8 is CLOSED.** The locked, preregistered external-validation protocol was executed exactly
as designed and specified — preregistration before label access, image-only inference and prediction
locking before label access, label-only evaluation after locking, no external tuning at any stage —
and its result, however poor, was reported completely and transparently. Phase 8 closure does not
depend on the external result being favorable; it depends on the protocol having been followed
without deviation and reported honestly, which is the case here. No Phase 9, LLM/VLM track work, or
any phase beyond Phase 8 has begun.
