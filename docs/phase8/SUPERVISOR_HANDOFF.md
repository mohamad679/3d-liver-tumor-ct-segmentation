# Phase 8 Supervisor Handoff

## Current State

- Branch: `phase/8-external-validation`
- Wave: `1`
- Status: Wave 1 review complete for frozen-decision, preregistration, and leakage contracts.
- Current HEAD at Wave 1 start: `e57b003`
- External drive: intentionally disconnected for Wave 1.
- External dataset access: none.
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

## Blocked Agents

- No Wave 1 agents are blocked.
- No Wave 2 agent may start until the user explicitly instructs that the external drive may be
  connected and supplies the required roots.

## Approved Wave 1 Files

- `src/protoem_ct/external/__init__.py`
- `src/protoem_ct/external/artifacts.py`
- `src/protoem_ct/external/preregistration.py`
- `src/protoem_ct/external/leakage.py`
- `tests/unit/test_phase8_artifacts.py`
- `tests/unit/test_phase8_preregistration.py`
- `tests/unit/test_phase8_leakage.py`

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

Stop after Wave 1. The exact next action is Wave 2 external dry-run preparation only after explicit
user instruction to connect the external drive and provide one dataset root and one output root.

Wave 2 remains blocked. Until released, no agent may access `/Volumes`, real 3D-IRCADb-01, external
labels, predictions, checkpoints, generated medical artifacts, inference, metrics execution,
bootstrap execution, montage generation, or publication execution.

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
