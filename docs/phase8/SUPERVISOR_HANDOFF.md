# Phase 8 Supervisor Handoff

## Current State

- Branch: `phase/8-external-validation`
- Wave: `4`
- Status: Wave 4 guarded internal-evidence readiness completed with scientifically valid
  `BLOCKED` result. No decision freeze, evaluation-ready preregistration, external-validation gate,
  or Phase 8 completion is claimed.
- Current HEAD at Wave 2 start: `7809d01`
- External drive: connected only through explicit user-provided roots.
- External dataset access: image-only `PATIENT_DICOM.zip` access for 3D-IRCADb-01 Wave 2.
  Wave 3 consumed only corrected Wave 2 JSON artifacts and did not access the raw dataset root.
- External label access: none.
- Phase scope: Phase 8 external validation on `3D-IRCADb-01` only.
- Phase 9 and Phase 10: not started.

## Approved Commits

- No Phase 8 commits have been made in Wave 1.
- Future work must use small reviewable commits after approved substages on
  `phase/8-external-validation`.

## Completed Wave 0 Agents

All Wave 0 planning agents used the real delegated-agent capability and were read-only:

- `P8-SCIENTIFIC-PLAN`: complete.
- `P8-DATA-BOUNDARY-PLAN`: complete.
- `P8-ENGINEERING-PLAN`: complete.
- `P8-LEAKAGE-PLAN`: complete.

## Completed Wave 1 Agents

All Wave 1 implementation agents used the real delegated-agent capability and were reviewed by the
Supervisor:

- `P8-FREEZE-CONTRACTS`: complete and approved after integration review.
- `P8-PREREGISTRATION-CONTRACTS`: complete and approved after integration review.
- `P8-LEAKAGE-CONTRACTS`: complete and approved after integration review.

## Completed Wave 2 Agents

All Wave 2 agents used the real delegated-agent capability and were reviewed by the Supervisor:

- `P8-IRCADB-ADAPTER`: complete and approved after integration review.
- `P8-ANONYMOUS-MANIFEST`: complete and approved after integration review.
- `P8-IMAGE-QA`: complete and approved after integration review.
- `P8-WAVE2-LEAKAGE-REVIEW`: initial review found and blocked a UID-derived
  `image_series_identity`; follow-up review passed after correction.

## Completed Wave 3 Agents

All Wave 3 agents used the real delegated-agent capability and were reviewed by the Supervisor:

- `P8-LABEL-MAPPING-POLICY`: complete and approved after integration review.
- `P8-DOMAIN-SHIFT`: complete and approved after integration review.
- `P8-ELIGIBILITY-ACCOUNTING`: complete and approved after integration review.
- `P8-WAVE3-REVIEW`: independent read-only review passed with no blockers.

## Completed Wave 4 Agents

All Wave 4 agents used the real delegated-agent capability and were reviewed by the Supervisor:

- `P8-INTERNAL-EVIDENCE-AUDIT`: complete, read-only, found Wave 4 freeze readiness blocked.
- `P8-STATISTICAL-PREREGISTRATION`: complete, read-only, recommended exact metric, bootstrap,
  comparison, qualitative-output, publication, limitations, and robustness/uncertainty policies.
- `P8-FREEZE-AND-PREREGISTRATION`: complete, read-only review, approved only the guarded
  `BLOCKED` readiness path.
- `P8-WAVE4-INDEPENDENT-REVIEW`: complete, read-only, passed with `BLOCKED` Wave 4 state
  confirmed and no unresolved review blocker.

## Blocked Agents

- No Wave 4 subagent failed to run.
- Wave 4 freeze/preregistration generation is blocked by missing real internal-development
  provenance.

## Completed Wave 4 Blocker-Remediation Planning Agents

All remediation-planning agents used the real delegated-agent capability and were read-only:

- `P8R-DEVELOPMENT-DATA-AUDIT`: complete; approved the Phase 2 v2 manifest, split, QA,
  lesion-summary, development-summary, final-QA, and leakage-audit artifacts as the current
  development evidence set for future remediation planning.
- `P8R-TRAINING-READINESS-AUDIT`: complete; found no existing end-to-end real-development pathway
  capable of producing a freeze-eligible checkpoint package.
- `P8R-SELECTION-FREEZE-PLAN`: complete; defined the development-only selection/freeze procedure
  and confirmed that no current candidate set is freezeable.
- `P8R-COMPUTE-EXECUTION-PLAN`: complete; found no CUDA and no available MPS in the inspected
  Phase 3 environment, and identified that tiny real-pixel verification and definitive training
  require implementation plus user approval.

Planning reconciliation is recorded in `docs/phase8/INTERNAL_EVIDENCE_REMEDIATION_PLAN.md`.
No remediation was implemented, no model was trained, no checkpoint was created, no readiness rerun
was executed, and Wave 5 remains blocked.

## Internal Evidence Remediation Substage 1

Status: completed as contract-only source implementation; not committed, not pushed, and not a
Wave 4 readiness release. Gate 8 is not claimed.

Real delegated agents used:

- `P8R-EVIDENCE-CONTRACTS`: complete; implemented the bounded schema/test work in
  `src/protoem_ct/external/internal_evidence.py` and
  `tests/unit/test_phase8_internal_evidence.py`.
- `P8R-EVIDENCE-CONTRACT-REVIEW`: complete; independent read-only review initially returned
  `BLOCKED` because aggregate validation did not bind selected multi-candidate workflow fields
  tightly enough. The Supervisor remediated those findings by requiring selected/checkpoint
  candidate consistency for preprocessing hash, training config hash, model family, adaptation
  mode, fixed seed membership, and primary validation metric name. No third subagent was spawned
  because the task was capped at two real subagents.

Approved Substage 1 schema surfaces, all `v1`:

- `phase8_preprocessing_decision`
- `phase8_fixed_candidate_inventory`
- `phase8_checkpoint_metadata`
- `phase8_validation_evidence_reference`
- `phase8_model_selection_decision`
- `phase8_threshold_decision`
- `phase8_support_policy`
- `phase8_internal_evidence_package`

Approved Substage 1 files:

- `src/protoem_ct/external/internal_evidence.py`
- `src/protoem_ct/external/__init__.py`
- `tests/unit/test_phase8_internal_evidence.py`
- `docs/phase8/SUPERVISOR_HANDOFF.md`

Targeted local test results:

- `uv run ruff format --check src/protoem_ct/external/internal_evidence.py src/protoem_ct/external/__init__.py tests/unit/test_phase8_internal_evidence.py`: PASS, `3 files already formatted`
- `uv run ruff check src/protoem_ct/external/internal_evidence.py src/protoem_ct/external/__init__.py tests/unit/test_phase8_internal_evidence.py`: PASS, `All checks passed!`
- `uv run mypy src/protoem_ct/external/internal_evidence.py src/protoem_ct/external/__init__.py tests/unit/test_phase8_internal_evidence.py`: PASS, `Success: no issues found in 3 source files`
- `uv run pytest -q tests/unit/test_phase8_internal_evidence.py`: PASS, `13 passed in 1.22s`
- `git diff --check`: PASS, no output

Boundary confirmation:

- `/Volumes` was not accessed.
- No development or external dataset was accessed.
- No NIfTI, DICOM, ZIP, mask, label, prediction, checkpoint, model weight, or generated run artifact
  was opened or modified.
- No training, inference, GPU, MPS, long CPU execution, real checkpoint creation, model selection,
  threshold selection, or support-set construction occurred.
- Synthetic values were used only as in-memory unit-test mappings.

Readiness states and blocker reason codes now represented by the contracts:

- Evidence/package readiness states: `READY`, `BLOCKED`, `freeze_ready`, `resolved`,
  `unresolved`, `completed`, `failed`, and `synthetic_smoke` where component-specific.
- Aggregate blocker reason codes are computed from cross-contract invariants, including candidate,
  checkpoint, preprocessing, manifest, split, metric, threshold, support, external-evidence,
  internal-test, synthetic-only, failed, incompatible, and unresolved states.

Unresolved scientific decisions remain unchanged:

- No real candidate inventory is frozen.
- No real checkpoint is selected or freeze eligible.
- No real threshold policy is selected.
- No real support/no-support policy is selected.
- No preprocessing decision is frozen for a selected real workflow.

Exact Substage 2 action:

- Implement the real MONAI SegResNet development-runner scaffold only after explicit authorization
  for Substage 2 scope, while preserving the data/compute approval boundary for any raw
  development-pixel access.

Wave 4 remains `BLOCKED`. Wave 5 remains blocked and unreleased.

## Approved Wave 1 Files

- `src/protoem_ct/external/__init__.py`
- `src/protoem_ct/external/artifacts.py`
- `src/protoem_ct/external/preregistration.py`
- `src/protoem_ct/external/leakage.py`
- `tests/unit/test_phase8_artifacts.py`
- `tests/unit/test_phase8_preregistration.py`
- `tests/unit/test_phase8_leakage.py`

## Approved Wave 2 Files

- `src/protoem_ct/external/ircadb.py`
- `src/protoem_ct/external/manifest.py`
- `src/protoem_ct/external/image_qa.py`
- `src/protoem_ct/external/wave2.py`
- `src/protoem_ct/external/__init__.py`
- `src/protoem_ct/cli/main.py`
- `tests/unit/test_phase8_ircadb.py`
- `tests/unit/test_phase8_external_manifest.py`
- `tests/unit/test_phase8_image_qa.py`
- `tests/unit/test_phase8_wave2.py`

## Approved Wave 3 Files

- `src/protoem_ct/external/label_mapping.py`
- `src/protoem_ct/external/domain_shift.py`
- `src/protoem_ct/external/eligibility.py`
- `src/protoem_ct/external/wave3.py`
- `src/protoem_ct/external/__init__.py`
- `src/protoem_ct/cli/main.py`
- `tests/unit/test_phase8_label_mapping.py`
- `tests/unit/test_phase8_domain_shift.py`
- `tests/unit/test_phase8_eligibility.py`
- `tests/unit/test_phase8_wave3.py`

## Wave 4 Repository Files Pending Review

- `docs/DECISIONS.md`
- `docs/phase8/SUPERVISOR_HANDOFF.md`
- `src/protoem_ct/external/statistical_policy.py`
- `src/protoem_ct/external/freeze.py`
- `src/protoem_ct/external/wave4.py`
- `src/protoem_ct/external/__init__.py`
- `src/protoem_ct/cli/main.py`
- `tests/unit/test_phase8_statistical_policy.py`
- `tests/unit/test_phase8_wave4.py`

## Wave 4 Real Execution Evidence

- Wave 4 output root relative name: `phase8_wave4_freeze_v1`
- Wave 4 final state: `BLOCKED`
- Generated artifact relative names and SHA-256 hashes:
  - `phase8_internal_evidence_readiness.json`:
    `24a7fd5fd90f347292fce4012a416577dcf8e1e7341aaa298ede7ac8c9269315`
  - `phase8_wave4_generation_summary.json`:
    `ffeed2848e5ffa821ce46bd253b127c4da965e6104696e2fbdbb65bedaed2796`
- `phase8_decision_freeze.json`: not generated.
- `phase8_external_preregistration.json`: not generated.
- Deterministic rerun output root:
  `/private/tmp/protoem-ct-phase8-wave4-rerun-20260804-v2`
- Deterministic rerun result: both Wave 4 JSON artifacts were byte-identical to the real Wave 4
  output root.
- Generated-artifact scans for absolute paths, PHI-like tokens, prohibited archive names,
  placeholder/zero hashes, and synthetic-checkpoint acceptance language: PASS, no matches.
- Explicit confirmation: no raw external data, external labels, external predictions, raw
  development medical images, external inference, external metrics, bootstrap execution, or Wave 5
  work occurred.
- Explicit confirmation: Wave 2 and Wave 3 artifacts were not modified.

### Wave 4 Readiness by Category

| Category | Present | Freezeable now | Result |
| --- | --- | --- | --- |
| `preprocessing_decision` | Yes, synthetic Phase 6 config only | No | Blocked: real preprocessing decision linked to selected model is missing |
| `support_policy` | No | No | Blocked: immutable external-label-free support or no-support policy is missing |
| `threshold_decision` | Yes, synthetic Phase 6 config only | No | Blocked: real fixed threshold decision with development-only provenance is missing |
| `checkpoint_metadata` | No | No | Blocked: real trained checkpoint metadata and selection provenance are missing |
| `model_selection_decision` | No | No | Blocked: real development-only model-selection artifact is missing |
| `label_mapping_policy` | Yes | Yes, policy only | Predeclared Wave 3 policy remains empirically unverified until label ledger |
| `metric_configuration` | Yes, newly recorded policy | No | Blocked pending complete freeze package |
| `bootstrap_configuration` | Yes, newly recorded policy | No | Blocked pending complete freeze package |
| `publication_configuration` | Yes, newly recorded policy | No | Blocked pending complete freeze package |

### Wave 4 Statistical Policy Decisions

- Metrics: tumor Dice, tumor IoU, tumor HD95, tumor normalized surface Dice, lesion-wise recall,
  lesion-wise precision, lesion F1, false-positive lesions per scan, and tumor volume error, using
  the accepted Phase 3 binary-tumor metric definitions and empty-mask behavior.
- Bootstrap: anonymous case/patient unit, seed `1729`, `10000` resamples, `95%` percentile
  intervals, metric-specific valid-case accounting, undefined values excluded with unavailable CIs
  reported, and no lesion-level or voxel-level pseudoreplication.
- Internal-versus-external comparison: independent unpaired descriptive estimates only; point
  estimates, confidence intervals, and external-minus-internal differences; no superiority,
  generalization, clinical-validity, deployment, or post-result decision changes.
- Qualitative output: include all evaluation-eligible anonymous cases when the eligible count is at
  most `20`, ordered by anonymous ID; if future eligibility exceeds `20`, use a fixed hash-ranked
  sample with seed `1729`.
- Robustness/uncertainty: `not_included`.
- Mandatory limitation: small external sample size.

### Wave 4 Blockers

- Real trained checkpoint metadata with development manifest/config hashes is missing.
- Selection rule and selected candidate are not recorded.
- Real development-only model-selection artifact is missing.
- Immutable external-label-free support or no-support policy is missing.
- Real fixed threshold decision with development-only provenance is missing.
- Real preprocessing decision linked to the selected model is missing.
- Metric, bootstrap, and publication policies are newly recorded in Wave 4 and cannot be frozen
  without the complete decision package.

## Wave 3 Real-Data Evidence

- Corrected Wave 2 artifact root consumed:
  `phase8_wave2_inventory_v1_uidfree`
- Wave 3 external output root relative name:
  `phase8_wave3_policy_v1`
- External image manifest identity:
  `23a0fa16247b515a966be29b272c9803f12dd4ef7eeebdaa0096cc7c04972c97`
- Label-mapping policy hash:
  `078bd5e84fcbfe18efea2a44275ed9ce91047a4a2890220e2a61fa61e3562304`
- Domain-shift record hash:
  `04665b449d8b3d7838a28e3045cdd57b67c9fccf6e1f091b7f5f1abd5963da0b`
- Eligibility policy hash:
  `54de20228c8baa25cd27e3d647d7c1b7b534fa5c65005bf000e2044a6b03d7ee`
- Cohort-accounting hash:
  `5f35e6851d039dda0ae5be1591e79877249a0fd150d28e733e0d33bc72410156`
- Generated artifact relative names and SHA-256 hashes:
  - `phase8_external_label_mapping_policy.json`:
    `2781c3d7eadfafc1b384d34dab0e077528816e0c47ebeda3016c028f5ed00ced`
  - `phase8_external_domain_shift_record.json`:
    `d31242dfee473be16d9eba23a7cbb63b28e1a8f1505cac6ff5bf6edbb177914d`
  - `phase8_external_eligibility_policy.json`:
    `686b2b8b3d86be905a0d458545b866973ef8307cda6fa956d61684f3e0bb4665`
  - `phase8_external_cohort_accounting.json`:
    `321bc3a8e46d99989eccc4ebbacc33afeaf7ecb60c9f37256f2e641c6c5ea553`
  - `phase8_wave3_generation_summary.json`:
    `8b734a3d721387ab5fda0cedddbb389bd294800dab61f1e4e1b04d670bf76374`
- Deterministic rerun output root:
  `/private/tmp/protoem-ct-phase8-wave3-rerun-20260804`
- Deterministic rerun result: all five Wave 3 JSON artifacts were byte-identical to the real
  Wave 3 output root.

### Wave 3 Label-Mapping Verification State

- Schema: `phase8_label_mapping_policy` / `v1`
- Verification state: `expected_documented_not_empirically_verified`
- Explicitly unverified until authorized label-ledger QA:
  - actual external label archive member names;
  - actual mask file layout;
  - actual source label values;
  - actual source-label availability;
  - actual source-label geometry compatibility.
- Expected/documented policy only:
  - target task is binary liver-tumor segmentation;
  - target classes are `background` and `tumor_foreground`;
  - permitted abstract roles are `liver_context_mask` and `tumor_lesion_mask`;
  - tumor lesion nonzero voxels map to tumor foreground;
  - liver/context masks are excluded from target foreground;
  - all permitted tumor lesion sources are aggregated by union;
  - labels require same grid shape, affine, spacing, and orientation;
  - label interpolation policy is nearest neighbor if later resampling is authorized elsewhere.
- No external case is excluded based on unseen label content.

### Wave 3 Domain-Shift Status

- Schema: `phase8_domain_shift_record` / `v1`
- Supported aggregate image-only dimensions:
  `source_identity`, `acquisition_representation`, `image_matrix`, `slice_counts`,
  `voxel_spacing`, `anisotropy`, `orientation`, `hu_readiness`, `modality`, and
  `image_qa_compatibility`.
- Unavailable dimensions:
  - `differences_vs_frozen_internal_preprocessing_contract`:
    `approved_internal_aggregate_reference_absent`
- Scanner/vendor/site values were not fabricated or recorded.
- No raw internal MSD/LiTS images were read or reconstructed for comparison.

### Wave 3 Eligibility and Accounting Totals

- Schema names:
  - `phase8_eligibility_policy` / `v1`
  - `phase8_eligibility_case` / `v1`
  - `phase8_eligibility_accounting` / `v1`
- Discovered cohort: `20`
- Image-readable cohort: `20`
- Image-QA eligible cohort: `20`
- Inference-eligible cohort: `20`
- Label-compatibility-pending cohort: `20`
- Evaluation-eligible cohort: `0`
- Excluded cases: `0`
- Deferred cases: `20`
- Deferred anonymous IDs: `ext-ircadb-001` through `ext-ircadb-020`
- Deferred reason codes:
  - eligibility reason: `label_compatibility_pending`
  - label-compatibility reason: `pending_label_access`
- Final evaluation eligibility remains pending label compatibility after authorized label access.

### Wave 3 Boundary Confirmation

- Real subagents were used.
- Wave 3 consumed only the corrected Wave 2 manifest, QA collection, layout artifact, and generation
  summary.
- Wave 3 did not access the raw external dataset root.
- `MASKS_DICOM.zip`, `LABELLED_DICOM.zip`, `MESHES_VTK.zip`, and JPG files were not opened, listed
  internally, extracted, read, or hashed.
- External labels were not accessed.
- No label-derived class observations were made.
- No inference, tuning, support selection, checkpoint selection, threshold selection, segmentation
  metrics, bootstrap, montage, post-result analysis, or Wave 4 work occurred.
- Raw dataset and Wave 2 artifacts were not modified by the Wave 3 code path.
- Generated Wave 3 artifacts remain outside Git.

## Wave 2 Real-Data Evidence

- Dataset archive SHA-256 reference:
  `47e7d155fdfd1763bd18a642829aee19697c9ce394ecebed40925b0e182820d9`
- Dataset archive size: `820270584` bytes
- Corrected external output root relative name: `phase8_wave2_inventory_v1_uidfree`
- Superseded external output root relative name: `phase8_wave2_inventory_v1`
  - Reason: independent leakage review found that the first run's image-series identity was derived
    from a DICOM `SeriesInstanceUID` hash.
  - Action: artifacts were left untouched; no external file was deleted or overwritten.
- Corrected manifest hash:
  `23a0fa16247b515a966be29b272c9803f12dd4ef7eeebdaa0096cc7c04972c97`
- Aggregate discovered case count: `20`
- Aggregate image-QA pass/fail count: `20 passed`, `0 failed`
- Anonymous IDs:
  `ext-ircadb-001`, `ext-ircadb-002`, `ext-ircadb-003`, `ext-ircadb-004`,
  `ext-ircadb-005`, `ext-ircadb-006`, `ext-ircadb-007`, `ext-ircadb-008`,
  `ext-ircadb-009`, `ext-ircadb-010`, `ext-ircadb-011`, `ext-ircadb-012`,
  `ext-ircadb-013`, `ext-ircadb-014`, `ext-ircadb-015`, `ext-ircadb-016`,
  `ext-ircadb-017`, `ext-ircadb-018`, `ext-ircadb-019`, `ext-ircadb-020`
- Failed-case reason codes: none.
- Corrected generated artifact relative names and SHA-256 hashes:
  - `phase8_external_image_manifest.json`:
    `470be686108084c7e37b8090299f470754216c1f10a23294840f14a1bc1af99e`
  - `phase8_wave2_generation_summary.json`:
    `34a7f19f47d87e467394c6e1435cb38b465407a50b8173233f135ac785287901`
  - `phase8_wave2_image_qa_results.json`:
    `10fa322d621dad2f92b8661b7a5c3d3fb281fe738662adce80e0a845b3c1986f`
  - `phase8_wave2_ircadb_layout.json`:
    `2639c07ab2ae2a75f00bdd2fe2eda878a65dfb53c458b8a71d4f99afed1a1742`
- Explicit confirmation: labels were not accessed.
- Explicit confirmation: raw data was not modified by the Wave 2 code path.
- Explicit confirmation: `MASKS_DICOM.zip`, `LABELLED_DICOM.zip`, `MESHES_VTK.zip`, and JPG files
  were not opened by the Wave 2 code path.
- Explicit confirmation: no inference, tuning, segmentation metrics, threshold selection,
  checkpoint selection, support-policy selection, or label mapping implementation occurred.

## Frozen Artifact Hashes

- No real Phase 8 frozen scientific artifact has been generated yet.
- Wave 1 defines typed contracts for:
  - `phase8_decision_freeze` / `v1`
  - `phase8_frozen_decision_reference` / `v1`
  - `phase8_external_preregistration` / `v1`
  - `phase8_artifact_reference` / `v1`
  - `phase8_robustness_uncertainty_inclusion` / `v1`
  - `phase8_frozen_decisions` / `v1`
  - `phase8_external_prediction_lock_record` / `v1`
  - `phase8_external_evaluation_access_event` / `v1`
  - `phase8_deviation_record` / `v1`
  - `external_label_access_ledger` / `v1`
  - `phase8_post_result_decision_lock` / `v1`
- Required future frozen artifacts include:
  - `phase8_decision_freeze_v1`
  - `phase8_external_preregistration_v1`
  - `phase8_label_mapping_policy_v1`
  - `phase8_external_image_manifest_v1`
  - `phase8_external_prediction_manifest_v1`
  - `external_label_access_ledger_v1`
  - `phase8_external_metric_report_v1`
  - `phase8_bootstrap_ci_v1`
  - `phase8_internal_external_comparison_v1`
  - `phase8_publication_summary_v1`

## Unresolved Blockers and Scientific Conflicts

- Bootstrap CI policy is not yet frozen or implemented.
- External few-shot support-label use is unresolved. The Wave 0 plan defaults to evaluation-only
  external labels unless a later preregistration explicitly freezes support/query rules before any
  labels are read.
- Real 3D-IRCADb-01 layout is unknown from repository context.
- Existing `IrcadbStyleAdapter` is synthetic-tested and Phase 2-only; Phase 8 must authorize or
  replace it before real external discovery.
- Existing Phase 2 manifest schema is development-oriented and label-bearing; Phase 8 needs
  external image and label-gated manifest schemas.
- External robustness/uncertainty reporting is optional and must be preregistered before evaluation
  if included.
- Phase 8 Gate 8 evidence is not yet generated.
- Wave 1 made no concrete scientific parameter choices and did not select checkpoints, thresholds,
  support policies, metrics, bootstrap settings, qualitative-output settings, or publication
  settings.

## Boundary Ledger

- First point external drive is required: Wave 2, `P8-IRCADB-DRYRUN`, after explicit user approval
  and only with one explicit absolute dataset root and one explicit absolute output root.
- First point external labels may be read: Wave 7, `P8-LABEL-LEDGER-QA`, after prediction lock,
  freeze, preregistration, and label-access ledger checks.
- Wave 0, Wave 1, Wave 3 label-policy work, Wave 4 freeze, and Wave 5 synthetic pipeline must not
  read real external labels.
- No agent may search `/Volumes` or infer dataset locations. Use only explicit user-provided roots.

## Exact Next Action

Stop after Wave 4 blocker-remediation planning. The exact next implementation action for a future
approved remediation session is to define real-development evidence schemas and readiness-input
contracts for preprocessing decision, candidate inventory, checkpoint metadata, validation metric
links, model-selection decision, threshold decision, and support/no-support policy.

Wave 5 remains blocked pending Wave 4 review and commit. Until released, no agent may access
external labels, predictions, inference, metrics execution, bootstrap execution, montage
generation, publication execution, or Wave 5 work. A future freeze/evaluation-ready
preregistration requires real internal-development provenance for the unresolved Wave 4 blockers.

## Wave 3 Verification Commands

- `uv run ruff format src/protoem_ct/external/label_mapping.py src/protoem_ct/external/domain_shift.py src/protoem_ct/external/eligibility.py src/protoem_ct/external/wave3.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py tests/unit/test_phase8_label_mapping.py tests/unit/test_phase8_domain_shift.py tests/unit/test_phase8_eligibility.py tests/unit/test_phase8_wave3.py`: PASS, `3 files reformatted` during integration and later files left unchanged.
- `uv run ruff format --check src/protoem_ct/external/label_mapping.py src/protoem_ct/external/domain_shift.py src/protoem_ct/external/eligibility.py src/protoem_ct/external/wave3.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py tests/unit/test_phase8_label_mapping.py tests/unit/test_phase8_domain_shift.py tests/unit/test_phase8_eligibility.py tests/unit/test_phase8_wave3.py`: PASS, `10 files already formatted`.
- `uv run ruff check src/protoem_ct/external/label_mapping.py src/protoem_ct/external/domain_shift.py src/protoem_ct/external/eligibility.py src/protoem_ct/external/wave3.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py tests/unit/test_phase8_label_mapping.py tests/unit/test_phase8_domain_shift.py tests/unit/test_phase8_eligibility.py tests/unit/test_phase8_wave3.py`: PASS, `All checks passed!`
- `uv run mypy src/protoem_ct/external/label_mapping.py src/protoem_ct/external/domain_shift.py src/protoem_ct/external/eligibility.py src/protoem_ct/external/wave3.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py tests/unit/test_phase8_label_mapping.py tests/unit/test_phase8_domain_shift.py tests/unit/test_phase8_eligibility.py tests/unit/test_phase8_wave3.py`: PASS.
- `uv run pytest -q tests/unit/test_phase8_label_mapping.py tests/unit/test_phase8_domain_shift.py tests/unit/test_phase8_eligibility.py tests/unit/test_phase8_wave3.py`: PASS, `24 passed`.
- `uv run pytest -q tests/unit/test_phase8_artifacts.py tests/unit/test_phase8_preregistration.py tests/unit/test_phase8_leakage.py tests/unit/test_phase8_ircadb.py tests/unit/test_phase8_external_manifest.py tests/unit/test_phase8_image_qa.py tests/unit/test_phase8_wave2.py tests/unit/test_phase8_label_mapping.py tests/unit/test_phase8_domain_shift.py tests/unit/test_phase8_eligibility.py tests/unit/test_phase8_wave3.py`: PASS, `85 passed`.
- `uv run pytest -q tests/unit/test_phase2_paths.py tests/unit/test_phase2_publication.py tests/unit/test_adapter_protocol.py tests/unit/test_ircadb_adapter.py tests/unit/test_manifest_builder.py tests/unit/test_manifest_validation.py`: PASS, `103 passed`.
- `uv run protoem-ct run-phase8-wave3-policy --help`: PASS.
- `uv run protoem-ct run-phase8-wave3-policy --wave2-artifact-root <corrected Wave 2 root> --output-root <Wave 3 external output root> --repository-root <repository root>`: PASS.
- Deterministic rerun into `/private/tmp/protoem-ct-phase8-wave3-rerun-20260804`: PASS.
- Byte comparison for five Wave 3 JSON artifacts between real and rerun roots: PASS.
- Generated-artifact scan for absolute paths, PHI tokens, raw UID-like values, prohibited archive
  member names, and JPG references: PASS, no matches.
- Supervisor source scan for Wave 3 ZIP-opening and prohibited archive access patterns: PASS, no
  Wave 3 source path opens or lists prohibited archives.
- Independent `P8-WAVE3-REVIEW`: PASS, no blockers.

## Wave 2 Verification Commands

- `uv run ruff format src/protoem_ct/external/ircadb.py src/protoem_ct/external/manifest.py src/protoem_ct/external/image_qa.py src/protoem_ct/external/wave2.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py tests/unit/test_phase8_ircadb.py tests/unit/test_phase8_external_manifest.py tests/unit/test_phase8_image_qa.py tests/unit/test_phase8_wave2.py`: PASS, `1 file reformatted` during integration and later files left unchanged.
- `uv run ruff format --check src/protoem_ct/external/ircadb.py src/protoem_ct/external/manifest.py src/protoem_ct/external/image_qa.py src/protoem_ct/external/wave2.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py tests/unit/test_phase8_ircadb.py tests/unit/test_phase8_external_manifest.py tests/unit/test_phase8_image_qa.py tests/unit/test_phase8_wave2.py`: PASS.
- `uv run ruff check src/protoem_ct/external/ircadb.py src/protoem_ct/external/manifest.py src/protoem_ct/external/image_qa.py src/protoem_ct/external/wave2.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py tests/unit/test_phase8_ircadb.py tests/unit/test_phase8_external_manifest.py tests/unit/test_phase8_image_qa.py tests/unit/test_phase8_wave2.py`: PASS, `All checks passed!`
- `uv run mypy src/protoem_ct/external/ircadb.py src/protoem_ct/external/manifest.py src/protoem_ct/external/image_qa.py src/protoem_ct/external/wave2.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py tests/unit/test_phase8_ircadb.py tests/unit/test_phase8_external_manifest.py tests/unit/test_phase8_image_qa.py tests/unit/test_phase8_wave2.py`: PASS.
- `uv run pytest -q tests/unit/test_phase8_ircadb.py tests/unit/test_phase8_external_manifest.py tests/unit/test_phase8_image_qa.py tests/unit/test_phase8_wave2.py`: PASS, `25 passed`.
- `uv run pytest -q tests/unit/test_phase2_paths.py tests/unit/test_phase2_publication.py tests/unit/test_adapter_protocol.py tests/unit/test_ircadb_adapter.py tests/unit/test_manifest_builder.py tests/unit/test_manifest_validation.py`: PASS, `103 passed`.
- `uv run protoem-ct run-phase8-wave2-image-inventory --help`: PASS.
- `uv run protoem-ct run-phase8-wave2-image-inventory --dataset-root <explicit 3Dircadb1 root> --dataset-archive <explicit outer archive> --output-root <external corrected Wave 2 root> --repository-root <repository root>`: PASS.
- `rg` artifact scan for absolute paths, PHI tokens, raw UID patterns, and prohibited archive
  internal paths under the corrected output root: PASS, no matches.

## Wave 2 Boundary Confirmation

- Real subagents were used.
- Only the explicit user-provided dataset root and archive were accessed.
- The real dataset root was treated read-only.
- `PATIENT_DICOM.zip` archives were opened for image-only DICOM QA.
- `MASKS_DICOM.zip`, `LABELLED_DICOM.zip`, `MESHES_VTK.zip`, and JPG files were not opened,
  listed internally, extracted, or hashed.
- External labels were not read.
- No PHI-bearing DICOM fields were printed or persisted.
- No DICOM `SeriesInstanceUID` or `SOPInstanceUID` value or UID-derived public identity is
  persisted in corrected Wave 2 artifacts.
- No absolute path appears in corrected Wave 2 canonical artifacts.
- No inference, tuning, segmentation metrics, label mapping, checkpoint selection, threshold
  selection, support-policy selection, bootstrap, montage, or Wave 3 work occurred.

## Wave 1 Verification Commands

- `uv run ruff format src/protoem_ct/external/__init__.py src/protoem_ct/external/artifacts.py src/protoem_ct/external/preregistration.py src/protoem_ct/external/leakage.py tests/unit/test_phase8_artifacts.py tests/unit/test_phase8_preregistration.py tests/unit/test_phase8_leakage.py`: PASS, `7 files left unchanged`
- `uv run ruff format --check src/protoem_ct/external/__init__.py src/protoem_ct/external/artifacts.py src/protoem_ct/external/preregistration.py src/protoem_ct/external/leakage.py tests/unit/test_phase8_artifacts.py tests/unit/test_phase8_preregistration.py tests/unit/test_phase8_leakage.py`: PASS, `7 files already formatted`
- `uv run ruff check src/protoem_ct/external/__init__.py src/protoem_ct/external/artifacts.py src/protoem_ct/external/preregistration.py src/protoem_ct/external/leakage.py tests/unit/test_phase8_artifacts.py tests/unit/test_phase8_preregistration.py tests/unit/test_phase8_leakage.py`: PASS, `All checks passed!`
- `uv run mypy src/protoem_ct/external/__init__.py src/protoem_ct/external/artifacts.py src/protoem_ct/external/preregistration.py src/protoem_ct/external/leakage.py tests/unit/test_phase8_artifacts.py tests/unit/test_phase8_preregistration.py tests/unit/test_phase8_leakage.py`: PASS, `Success: no issues found in 7 source files`
- `uv run pytest -q tests/unit/test_phase8_artifacts.py tests/unit/test_phase8_preregistration.py tests/unit/test_phase8_leakage.py`: PASS, `36 passed in 0.84s`

## Wave 1 Boundary Confirmation

- `/Volumes` was not accessed.
- No dataset path was accessed.
- The external drive remained disconnected.
- Real 3D-IRCADb-01 was not inspected.
- External labels were not read.
- No inference, metrics, bootstrap, montage, or publication execution was implemented or run.
- No concrete decision values, model-selection results, threshold choices, checkpoint choices, or
  scientific result values were fabricated.

## Wave 0 Verification Commands

Required after Wave 0 docs edits:

- `git diff --check`
- `git diff --stat`
- `git status --short --branch`
