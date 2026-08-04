# Phase 8 Supervisor Handoff

## Current State

- Branch: `phase/8-external-validation`
- Wave: `2`
- Status: Wave 2 review complete for real external-drive discovery, anonymous image manifest, and
  image-only QA. Gate 8 is not claimed.
- Current HEAD at Wave 2 start: `7809d01`
- External drive: connected only through explicit user-provided roots.
- External dataset access: image-only `PATIENT_DICOM.zip` access for 3D-IRCADb-01 Wave 2.
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

## Blocked Agents

- No Wave 2 implementation agents are blocked.
- Wave 3 remains blocked pending repository review and a Wave 2 commit.

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

Stop after Wave 2. The exact next action is repository review, then a Wave 2 commit if approved.

Wave 3 remains blocked pending review and commit. Until released, no agent may access external
labels, predictions, checkpoints, inference, metrics execution, bootstrap execution, montage
generation, publication execution, or label-mapping implementation.

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
