# Phase 8 Supervisor Handoff

## Current State

- Branch: `phase/8-external-validation`
- Wave: `0`
- Status: Wave 0 planning complete; Wave 1 is not released in this session.
- Current HEAD at Wave 0 start: `f19b183`
- External drive: intentionally disconnected for Wave 0.
- External dataset access: none.
- External label access: none.
- Phase scope: Phase 8 external validation on `3D-IRCADb-01` only.
- Phase 9 and Phase 10: not started.

## Approved Commits

- No Phase 8 commits have been made in Wave 0.
- Future work must use small reviewable commits after approved substages on
  `phase/8-external-validation`.

## Completed Wave 0 Agents

All Wave 0 planning agents used the real delegated-agent capability and were read-only:

- `P8-SCIENTIFIC-PLAN`: complete.
- `P8-DATA-BOUNDARY-PLAN`: complete.
- `P8-ENGINEERING-PLAN`: complete.
- `P8-LEAKAGE-PLAN`: complete.

## Blocked Agents

- No Wave 0 agents are blocked.
- No downstream implementation agent may start from this Wave 0 session.

## Frozen Artifact Hashes

- No Phase 8 frozen artifact has been generated yet.
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

## Boundary Ledger

- First point external drive is required: Wave 2, `P8-IRCADB-DRYRUN`, after explicit user approval
  and only with one explicit absolute dataset root and one explicit absolute output root.
- First point external labels may be read: Wave 7, `P8-LABEL-LEDGER-QA`, after prediction lock,
  freeze, preregistration, and label-access ledger checks.
- Wave 0, Wave 1, Wave 3 label-policy work, Wave 4 freeze, and Wave 5 synthetic pipeline must not
  read real external labels.
- No agent may search `/Volumes` or infer dataset locations. Use only explicit user-provided roots.

## Exact Next Action

Start Wave 1 only after user approval. The first Wave 1 action is to create the Phase 8 freeze,
preregistration, and leakage schema/test plan in source and tests using synthetic fixtures only:

1. `P8-FREEZE-INVENTORY`
2. `P8-PREREGISTRATION-SCHEMA`
3. `P8-LEAKAGE-CONTRACTS`

Wave 1 must not access `/Volumes`, real 3D-IRCADb-01, external labels, predictions, checkpoints, or
generated medical artifacts.

## Wave 0 Verification Commands

Required after Wave 0 docs edits:

- `git diff --check`
- `git diff --stat`
- `git status --short --branch`
