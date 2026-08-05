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

## Internal Evidence Remediation Substage 2

Status: completed as contract-only scaffold/orchestration-planning source implementation; not
committed, not pushed, and not a Wave 4 readiness release. Gate 8 is not claimed. This substage
built a metadata-only, manifest/split-driven planning scaffold for a future real MONAI SegResNet
development run. It did not execute training, did not open any dataset or pixel file, and did not
create any checkpoint, prediction, or real metric artifact.

Real delegated agents used (exactly two, sequential, no concurrent edits):

- `P8R-RUNNER-SCAFFOLD`: complete; implemented the scaffold and targeted tests in
  `src/protoem_ct/external/real_development_runner.py` and
  `tests/unit/test_phase8_real_development_runner.py`, plus the CLI command in
  `src/protoem_ct/cli/main.py` and re-exports in `src/protoem_ct/external/__init__.py`.
- `P8R-RUNNER-SCAFFOLD-REVIEW`: complete; independent read-only-plus-remediation review of the
  complete diff and tests against 12 boundary/behavior checks. Verdict: PASS. No genuine defects
  were found; no fixes were required.

Approved Substage 2 schema surfaces, all `v1`:

- `phase8_real_development_input_binding`
- `phase8_real_development_run_plan`
- `phase8_real_development_case_binding`
- `phase8_real_development_case_binding_collection`
- `phase8_real_development_plan_summary`

Approved Substage 2 files:

- `src/protoem_ct/external/real_development_runner.py` (new)
- `tests/unit/test_phase8_real_development_runner.py` (new)
- `src/protoem_ct/external/__init__.py` (re-exports added)
- `src/protoem_ct/cli/main.py` (new `plan-phase8-real-development-run` command)

CLI command: `protoem-ct plan-phase8-real-development-run`. The name contains no "train",
"run-development", or "execute" language. `--help` and echoed run output explicitly state
`scaffold_only: true`, `pixel_access_not_started: true`, `training_executed: false`,
`checkpoint_created: false`, and `real_metrics_computed: false`.

Exact execution boundary:

- `pixel_access_state` is hard-locked to `"not_started"` and `execution_release_state` to
  `"scaffold_only"` for every object the public API can produce; no builder exposes a parameter to
  set another value.
- `execute_phase8_real_development_run(...)` unconditionally raises
  `Phase8RealDevelopmentExecutionNotReleasedError` regardless of arguments, including when a fake
  executor double is supplied.
- `Phase8RealDevelopmentExecutor` is a typed `Protocol` only; every method body is
  `raise NotImplementedError`. No MONAI dataset construction, Torch training loop, or checkpoint
  I/O exists anywhere in the module.
- No code path opens NIfTI/DICOM image or label bytes; only two explicit JSON files (manifest,
  split) are opened, and image/label relative paths are validated as path-safe strings only
  (`normalize_safe_relative_posix_path`), never filesystem-checked or opened.
- Internal-test cases can never enter a training/validation case binding; the run-plan builder
  cross-checks intended train/validation IDs against the input binding's internal-test set and
  rejects any overlap.
- Every schema enforces `no_external_data=True`; no field can reference external (3D-IRCADb) data.
- Output-root publication reuses `validate_explicit_external_output_root` with the repository root
  as a forbidden root, rejects symlinks, and requires the output root to be empty or absent
  (no-overwrite `publish_text_no_overwrite`).
- No absolute local path, timestamp, hostname, or username can enter any hashed/canonical payload.

Targeted local test results (Supervisor-run, independent of both subagents):

- `uv run ruff format --check src/protoem_ct/external/real_development_runner.py tests/unit/test_phase8_real_development_runner.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py`: PASS, `4 files already formatted`
- `uv run ruff check src/protoem_ct/external/real_development_runner.py tests/unit/test_phase8_real_development_runner.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py`: PASS, `All checks passed!`
- `uv run mypy src/protoem_ct/external/real_development_runner.py src/protoem_ct/cli/main.py tests/unit/test_phase8_real_development_runner.py`: PASS, `Success: no issues found in 3 source files`
- `uv run pytest -q tests/unit/test_phase8_real_development_runner.py`: PASS, `23 passed`
- `uv run pytest -q tests/unit/test_phase8_internal_evidence.py tests/unit/test_phase8_wave3.py tests/unit/test_phase8_wave4.py`: PASS, `19 passed` (unmodified regression surface)
- `uv run protoem-ct plan-phase8-real-development-run --help`: PASS
- Deterministic two-output-root publication byte-identity test: PASS (`test_publication_is_byte_identical_across_two_output_roots`)
- `git diff --check`: PASS, no output
- Git hygiene scan: only `src/protoem_ct/cli/main.py`, `src/protoem_ct/external/__init__.py`,
  `src/protoem_ct/external/real_development_runner.py`, and
  `tests/unit/test_phase8_real_development_runner.py` appear in `git status`; no generated
  medical/checkpoint/prediction artifact is present.

Boundary confirmation:

- `/Volumes` was not accessed.
- No development or external dataset was accessed.
- No NIfTI, DICOM, ZIP, mask, label, prediction, or checkpoint file was opened or modified.
- No `torch`, `monai`, `nibabel`, or `SimpleITK` real invocation occurred; the module never imports
  or invokes them (grep-verified and enforced by an automated static-scan unit test).
- No training, inference, GPU, MPS, checkpoint creation, real metric computation, candidate
  selection, threshold selection, or support-set construction occurred.
- Only synthetic in-memory/`tmp_path` JSON fixtures and test doubles were used in tests.

Unresolved scientific decisions remain unchanged (same as Substage 1):

- No real candidate inventory is frozen.
- No real checkpoint is selected or freeze eligible.
- No real threshold policy is selected.
- No real support/no-support policy is selected.
- No preprocessing decision is frozen for a selected real workflow.

Exact Substage 3 action (completed, see below):

- Implement the tiny real-development dry/overfit verification using a bounded tiny train/validation
  subset and a fresh external output root. This was the first point raw development NIfTI pixel
  arrays were opened, and it required explicit user approval before any raw development-pixel
  access occurred. See "Internal Evidence Remediation Substage 3" below for the completed outcome.

Wave 4 remains `BLOCKED`. Wave 5 remains blocked and unreleased.

## Internal Evidence Remediation Substage 3

Status: **PASS**. Implementation, code-level review, defect fix, one approved post-fix real rerun
(v2), and independent read-only rerun-artifact review are all complete. Not committed, not pushed.
Gate 8 is not claimed. Wave 4 remains `BLOCKED`. Wave 5 remains blocked.

User approval scope: the user explicitly approved one bounded, CPU-only, verification-only real
NIfTI pixel access (exactly one train case, one validation case, at most two optimizer steps) from
the approved Phase 2 v2 development artifact set. No definitive training, no model/threshold/support
selection, no internal-test or external-data access was authorized.

Real delegated agents used (exactly two, sequential, no concurrent edits):

- `P8R-TINY-REAL-VERIFICATION`: complete; implemented
  `src/protoem_ct/external/tiny_real_verification.py`,
  `tests/unit/test_phase8_tiny_real_verification.py`, and the
  `run-phase8-tiny-real-development-verification` CLI command in `src/protoem_ct/cli/main.py`. Its
  synthetic test suite passed (30 passed / 3 skipped in the ambient environment without torch/monai).
- `P8R-TINY-REAL-VERIFICATION-REVIEW`: complete; independent review of the complete diff (including
  the Supervisor's post-run defect fix, see below). Verdict: **PASS**, no defects found across all
  12 reviewed boundary/behavior checks (approval gating, deterministic metadata-only case selection,
  internal-test exclusion, external-data exclusion, compute-limit enforcement, the label-domain fix
  correctness, checkpoint non-freeze-eligibility, absence of scientific-metric claims, no dataset
  mutation, outputs outside Git, no PHI/absolute-path leakage, Wave 4/5 untouched).

Approved Substage 3 schema surfaces, all `v1`:

- `phase8_tiny_real_verification_config`
- `phase8_tiny_real_access_ledger`
- `phase8_tiny_real_checkpoint_metadata` (wraps the existing `phase8_checkpoint_metadata` contract
  with `completion_status="synthetic_smoke"`, hard-coded `freeze_eligible=false`,
  `selection_eligible=false`, `definitive_training=false`, `tiny_verification_only=true`)
- `phase8_tiny_real_verification_summary`

Approved Substage 3 files (uncommitted working-tree changes only):

- `src/protoem_ct/external/tiny_real_verification.py` (new)
- `tests/unit/test_phase8_tiny_real_verification.py` (new)
- `src/protoem_ct/cli/main.py` (new `run-phase8-tiny-real-development-verification` command)

CLI command: `protoem-ct run-phase8-tiny-real-development-verification`, gated by a required
`--approve-tiny-real-verification` flag checked before any file access. `--help` states the run is
bounded, verification-only, non-scientific, and requires explicit approval.

### Pre-flight verification (passed)

- Branch `phase/8-external-validation`, HEAD `4204757`, worktree clean before editing: confirmed.
- Manifest byte-level SHA-256 `0e21a7d555e20b6011091bc18e462bc150cc23a8f46e522dbd8c01b014a44ae3` and
  embedded `manifest_hash` `c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b`: verified
  match.
- Split byte-level SHA-256 `416ca83e85c8598fc4f7065316153193211bf6b57875fbda4cafc01c360445f7` and
  embedded `split_hash` `936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb`: verified
  match.
- `schema_version == 2` for both artifacts: confirmed. Counts 91/20/20, patient-disjoint across
  partitions: confirmed. Output root
  `/Volumes/Lexar/ProtoEM-CT/runs/phase8_substage3_tiny_real_v1` did not exist and was not a
  symlink before execution: confirmed.
- Deterministic selected cases (first-in-canonical-order): train = `case_25928d2451a7d948a79e60084e3c5d32`
  (patient `pat_00279fb16b44fe5112f954f7f68f0a8c`), validation = `case_c20af8bddfe32cd26e29c12d372b291c`
  (patient `pat_01e021d325489a6e918ff83bb101c2e8`).

### Real execution result: BLOCKED (defect found, session ended without a successful rerun)

The one approved real execution completed pre-flight, opened the real train pair, ran the full
2-step bounded training and checkpoint save/reload, then failed with `InvalidLabelError` while
loading the validation pair, before any JSON artifact was published.

Root cause: the original implementation reused `protoem_ct.data.validation.validate_nifti_pair`,
which enforces a strict binary `{0,1}` label domain. The real MSD Task03_Liver raw labels are
three-valued (`0=background, 1=liver, 2=tumor`), not binary. The selected training case's raw label
happened to contain only `{0,1}` (no imaged tumor in that scan), so it passed the naive binary check
by coincidence — but voxel value `1` there means "liver," not "tumor," so the completed training
step was semantically training on liver-vs-background, not tumor-vs-background, contradicting
`PROJECT_SPEC.md`'s binary tumor-segmentation task definition. The validation case correctly tripped
the same check because it has real tumor voxels.

The user was informed and explicitly chose: fix the defect now, but do **not** rerun against the
real dataset this session. The Supervisor applied one narrow fix:

- Replaced both `validate_nifti_pair(...)` call sites with a new local `_validate_raw_case_pair_geometry`
  that performs the same shape/affine/finite checks but validates the raw label domain against
  `{0, 1, 2}` instead of strict binary, raising `Phase8TinyRealVerificationLabelDomainError` for
  out-of-domain values.
- `_load_bounded_patch` now binarizes the cropped label patch via the fixed, non-data-dependent rule
  `foreground = (raw_label == 2)` (module constant `RAW_LABEL_TUMOR_VALUE = 2`, the standard
  published LiTS/MSD convention), applied identically to both the train and validation patch.
- Added two regression tests: an out-of-domain label value (`9`) is rejected fail-closed with no
  output root created; a synthetic three-class label proves a liver voxel becomes background and a
  tumor voxel becomes foreground after `_load_bounded_patch`.
- Updated one pre-existing test to expect the new exception type in place of the old
  `NiftiValidationError` import.

All synthetic tests (35 passed, 0 skipped) pass under the isolated Phase 3 `torch`/`monai` baseline
environment (`environments/phase3-baselines/intel-macos-cpu`) after the fix. The independent review
(`P8R-TINY-REAL-VERIFICATION-REVIEW`) confirmed the fix is correct and complete.

**No real-dataset rerun occurred this session.** A fresh real-dataset execution is deferred to a
future explicitly-approved session.

### Stray artifact from the blocked real run

- One checkpoint file exists at
  `/Volumes/Lexar/ProtoEM-CT/runs/phase8_substage3_tiny_real_v1/checkpoints/phase8_tiny_real_verification_checkpoint.pt`,
  SHA-256 `6bc5957167fed8d45bc802ed28effec548d551a2d3ae74a7a3ccb73609b0828e`, 1,390,026 bytes. It
  reflects the **pre-fix, buggy** liver-vs-background training and must not be used for anything. It
  was deliberately **not deleted** (no-delete-existing-run policy) and no JSON artifacts were
  published alongside it — the run never reached the publication step, so no
  `phase8_tiny_real_verification_config.json`, `phase8_tiny_real_access_ledger.json`,
  `phase8_tiny_real_checkpoint_metadata.json`, or `phase8_tiny_real_verification_summary.json` exists
  for this run.
- Elapsed time and step count for the blocked run: 2 optimizer steps executed and completed
  (finite loss/gradients verified, checkpoint saved and reloaded successfully) before the run halted
  at validation-pair loading; no summary artifact was published to record elapsed wall-clock time.

### Boundary confirmation

- Internal-test cohort: never accessed. Selection filters internal-test assignments out before any
  manifest path lookup; a loader-spy test proves no internal-test file is ever opened.
- External (3D-IRCADb) data: never accessed. Static source-literal scan and manual grep both confirm
  no external-dataset path/literal exists in the new module.
- Real dataset files: only read (`nib.load`), never modified. All writes are scoped to `output_root`.
- No scientific metric (Dice/IoU/HD95) was computed or claimed; the terms appear only inside the
  fixed non-scientific disclaimer string.
- No absolute path, raw filename, or PHI was persisted (no JSON was published this run, and the
  artifact builders that would have produced it are covered by dedicated leakage-scan tests).
- No code was committed or pushed.
- Wave 4 and Wave 5 remain `BLOCKED`.

### Targeted validation commands run and results

- `uv run ruff format --check` / `uv run ruff check` on the three changed files: PASS.
- `uv run mypy` on the three changed files: PASS, no issues found.
- `uv run pytest -q tests/unit/test_phase8_tiny_real_verification.py tests/unit/test_phase8_internal_evidence.py tests/unit/test_phase8_real_development_runner.py`: PASS, `67 passed, 4 skipped`
  (skips are the torch/monai-gated tests, legitimately skipped in the ambient core environment).
- `env -u VIRTUAL_ENV uv run --project environments/phase3-baselines/intel-macos-cpu --locked --no-sync python -m pytest -q tests/unit/test_phase8_tiny_real_verification.py`: PASS, `35 passed`, 0 skipped
  (isolated Phase 3 CPU baseline environment with real `torch==2.2.2`/`monai==1.4.0`/`nibabel==5.4.2`).
- `uv run protoem-ct run-phase8-tiny-real-development-verification --help`: PASS.
- `git diff --check`: PASS, no output.
- `git status --short --branch`: `## phase/8-external-validation` with exactly
  `M src/protoem_ct/cli/main.py`, `?? src/protoem_ct/external/tiny_real_verification.py`,
  `?? tests/unit/test_phase8_tiny_real_verification.py` — no other file changed, no generated
  artifact is Git-visible.
- Independent review (`P8R-TINY-REAL-VERIFICATION-REVIEW`): **PASS**, no defects found.

### Post-fix rerun (v2) — one-time approved, completed, PASS

User approval scope for this rerun: exactly one additional bounded real-development verification
run using the already-selected train/validation cases and the now-fixed code, writing to a fresh
output root `/Volumes/Lexar/ProtoEM-CT/runs/phase8_substage3_tiny_real_v2`. The prior invalid v1
output root and checkpoint were not to be deleted, overwritten, or reused.

Pre-run checks (all passed before any NIfTI file was opened): branch `phase/8-external-validation`
HEAD `4204757` confirmed; worktree contained only the expected uncommitted Substage 3 files; manifest
byte-SHA-256 and embedded `manifest_hash` verified match; split byte-SHA-256 and embedded
`split_hash`/`source_manifest_hash` verified match; `schema_version==2` for both; counts 91/20/20
confirmed; both approved case IDs confirmed in their expected partitions (`case_25928d2451a7d948a79e60084e3c5d32`
→ train, `case_c20af8bddfe32cd26e29c12d372b291c` → validation); new output root confirmed absent and
not a symlink; prior v1 checkpoint hash reconfirmed unchanged (`6bc5957167fed8d45bc802ed28effec548d551a2d3ae74a7a3ccb73609b0828e`)
before the rerun, proving it was never touched by this workflow.

Execution: `protoem-ct run-phase8-tiny-real-development-verification` was invoked exactly once,
via the isolated Phase 3 `torch==2.2.2`/`monai==1.4.0`/`nibabel==5.4.2` environment, and completed
successfully. Anonymous train ID `case_25928d2451a7d948a79e60084e3c5d32`, anonymous validation ID
`case_c20af8bddfe32cd26e29c12d372b291c`. Actual limits used: `device=cpu`, `amp_enabled=false`,
`batch_size=1`, `max_optimizer_steps=2`, `max_epochs=1`, `num_workers=0`, `patch_size=[24,24,16]`
(within the `<=[32,64,64]` bound). `executed_step_count=2`, `step_finite_status=[true,true]`,
`checkpoint_round_trip_verified=true`, `elapsed_seconds≈21.73`. No scientific metric or performance
claim was produced.

New checkpoint: `checkpoints/phase8_tiny_real_verification_checkpoint.pt` (relative to the v2 output
root), SHA-256 `6bc5957167fed8d45bc802ed28effec548d551a2d3ae74a7a3ccb73609b0828e`, 1,390,026 bytes,
`tiny_verification_only=true`, `freeze_eligible=false`, `selection_eligible=false`,
`definitive_training=false`, nested `completion_status="synthetic_smoke"`.

**Note on the checkpoint hash matching v1's hash**: this is a verified coincidence, not reuse. The
v1 and v2 checkpoint files are distinct files on disk (different inodes, `cmp` confirms byte-identical
content). The training case's raw label (`liver_41.nii.gz`) has zero tumor voxels anywhere in the
full volume, and the fixed-corner crop region `[0:24,0:24,0:16]` used by the bounded patch loader is
entirely raw value `0` in that scan — so both the old buggy code (raw label used as-is) and the new
fixed code (`foreground = (raw_label == 2)`) produce an identical all-background training target for
this specific case and crop, yielding deterministically identical model weights after 2 steps. Both
the Supervisor and the independent reviewer verified this numerically against the real label file.

Generated JSON artifacts (relative to the v2 output root) and SHA-256 hashes:

- `phase8_tiny_real_verification_config.json`: `04f01bfeb71b44a04183e6b18066de0cb60d6261beb24954edbdcec34010713d`
- `phase8_tiny_real_access_ledger.json`: `166cde56b4905da333534c6a436c898305f21416a129927311a16ffe1b55231e`
- `phase8_tiny_real_checkpoint_metadata.json`: `d4b1a08180cf31801d9fa51d2bdbec89db9fec90b2e6133227b6f050ed4c6575`
- `phase8_tiny_real_verification_summary.json`: `6d2cfff9d1efbce6caea612bf5b0a6a4a6c838058897817c3adfcf86fc46fdeb`

Boundary confirmation for the rerun: access ledger records exactly the two approved cases
(`train_verification`/`validation_verification`), `internal_test_opened=false`,
`external_data_opened=false`, `file_access_count=4`. No PHI, absolute path, or raw filename appears
in any of the four generated JSON files (leakage scan: no matches). The two real dataset files for
each case (image + label) were rehashed after the run and match the manifest's recorded
`image_sha256`/`label_sha256` exactly — no dataset file was modified. The prior v1 output root still
contains only its original single checkpoint file — untouched. No code was committed or pushed;
`git status --short --branch` shows only the same 4 working-tree changes as before the rerun.

Independent read-only reviewer `P8R-TINY-REAL-RERUN-REVIEW`: **PASS**, all 12 checklist items
confirmed by direct, independent re-derivation (including independently recomputing checkpoint and
dataset-file hashes and numerically re-verifying the checkpoint-coincidence explanation). The
reviewer also flagged a minor documentation-only slip in the Supervisor's earlier ad-hoc diagnostic:
during initial defect discovery, the Supervisor manually inspected `liver_43.nii.gz` as an
illustrative stand-in for "the validation case," but the validation case's actual file (per the
manifest) is `liver_51.nii.gz`. This did not affect the real CLI runs (which always resolve paths
correctly by case ID) or the defect diagnosis/fix, only one earlier exploratory shell command's
choice of example file.

### Exact next action

Substage 3 is PASS. Substage 4 (definitive development training and validation-only selection)
requires its own separate explicit user approval before any work begins — it is not authorized by
this rerun's approval. Wave 4 and Wave 5 remain `BLOCKED` and must stay blocked until a Substage 4/5/6
remediation path is separately approved and completed.

Wave 4 remains `BLOCKED`. Wave 5 remains blocked and unreleased.

## Internal Evidence Remediation Substage 4B (Bounded Real-Data Pilot)

Status: **PILOT_PASS**. Implementation, Supervisor code review with one defect fix, the single
approved real execution, and independent read-only review are all complete. Not committed, not
pushed. Gate 8 is not claimed. This is a bounded, non-scientific, verification-scale pilot — it is
**not** definitive training, not freeze eligible, not selection eligible, and does not release
Substage 4A's `Phase8DefinitiveExecutionRelease` gate. Wave 4 remains `BLOCKED`. Wave 5 remains
blocked and unreleased.

### User approval boundary

The user explicitly approved exactly one bounded, CPU-only, verification-scale real pilot
execution with a fixed boundary: read-only dataset root
`/Volumes/Lexar/ProtoEM-CT/datasets/msd/Task03_Liver`; approved metadata root
`/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2` (manifest hash
`c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b`, split hash
`936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb`); approved prerequisites root
`/Volumes/Lexar/ProtoEM-CT/runs/phase8_substage4b_prerequisites_v1`; fresh, previously-nonexistent
output root `/Volumes/Lexar/ProtoEM-CT/runs/phase8_substage4b_bounded_pilot_v1`; exactly 2 train
cases + 1 validation case (6 real medical files total); deterministic metadata-only case selection;
fixed 20-optimizer-step MONAI SegResNet baseline training with an exact 10:10 foreground-aware
patch sampling schedule; one full-volume sliding-window validation forward pass with no scientific
metric computation; a 45-minute wall-clock watchdog; and no Wave 4/5 work. No definitive training,
no model/threshold/support selection, and no internal-test or external-data access was authorized.

### Real subagents used (exactly two, sequential, no concurrent edits)

- `P8R-BOUNDED-PILOT`: complete; implemented `src/protoem_ct/external/definitive_training_pilot.py`
  (2436 lines), `tests/unit/test_phase8_definitive_training_pilot.py`, and the
  `run-phase8-bounded-real-development-pilot` CLI command in `src/protoem_ct/cli/main.py`. Ran all
  synthetic tests only (no real-data access) and reported 24 passed/4 skipped (ambient) and 28
  passed (isolated Phase 3 CPU baseline environment).
- Supervisor code review (not a subagent): read the full implementation and found one real gap —
  no RAS-orientation check existed anywhere, despite the approved spec requiring
  `orientation: assert RAS only` and listing `non-RAS orientation` as a fail-closed condition. Added
  `Phase8BoundedPilotOrientationError`, a `REQUIRED_ORIENTATION_AXCODES = ("R","A","S")` check via
  `nib.aff2axcodes(...)` inside `_validate_raw_case_pair_geometry` (used for all three real pairs:
  train-positive, train-empty, and validation), and two regression tests
  (`test_geometry_validation_accepts_ras_oriented_pair`,
  `test_geometry_validation_rejects_non_ras_orientation`). Reran the full targeted test surface
  after the fix: 26 passed/4 skipped (ambient), 30 passed (isolated Phase 3 CPU baseline
  environment).
- `P8R-BOUNDED-PILOT-REVIEW`: complete; independent read-only review after the one real execution.
  Independently re-derived case selection from the real metadata, independently recomputed all 6
  dataset-file hashes and the checkpoint hash, independently re-read the source for every boundary
  claim (case selection, 10:10 schedule enforcement, watchdog placement, no-prior-checkpoint-load,
  checkpoint ineligibility flags and their structural enforcement in `internal_evidence.py`, no
  scientific-metric/prediction output, orientation-fix wiring). Verdict: **PASS**. Flagged one
  non-blocking residual observation: no watchdog check exists *during* the single sliding-window
  validation forward pass itself (only immediately before and after it) — assessed as low severity
  because it is one bounded, fixed-parameter inference call, not an unbounded loop, and the total
  run stayed at ~340s against a 2700s limit.

### Real execution result: PASS

Reverified before real execution: branch `phase/8-external-validation`, HEAD
`939ab4f356d84e590abdc7dcd79a1c2334334afa`, worktree contained only the three expected pilot
files (`M src/protoem_ct/cli/main.py`, `?? src/protoem_ct/external/definitive_training_pilot.py`,
`?? tests/unit/test_phase8_definitive_training_pilot.py`). Manifest/split/lesion-components file
byte-SHA-256 and prerequisite-artifact byte-SHA-256 were verified before use. Output root confirmed
absent and not a symlink. The 6 real medical files were hashed before the run.

Selected anonymous case IDs (deterministic, metadata-only, independently re-derived by the
reviewer): train-positive `case_19cd5c484db56de3a85635b4728f910c` (patient
`pat_1900b8a02daa2d54840b917175ddbe95`-lineage, relative path `imagesTr/liver_43.nii.gz` —
relative path never persisted in any artifact); train-empty
`case_25928d2451a7d948a79e60084e3c5d32` (`imagesTr/liver_41.nii.gz`); validation
`case_c20af8bddfe32cd26e29c12d372b291c` (`imagesTr/liver_51.nii.gz`).

The CLI command `run-phase8-bounded-real-development-pilot` was invoked exactly once, via the
isolated Phase 3 `torch==2.2.2`/`monai==1.4.0`/`nibabel==5.4.2` environment, and completed
successfully (exit code 0). No retry occurred and none is planned.

Actual configuration (all hard-coded, not caller-tunable — verified field-by-field against the
published `phase8_definitive_training_pilot_config.json`): `device=cpu`, `amp_enabled=false`,
`seed=1729`, `train_case_count=2`, `validation_case_count=1`, `patch_size=[64,64,32]`,
`batch_size=1`, `gradient_accumulation_steps=1`, `optimizer=adamw`, `learning_rate=1e-4`,
`weight_decay=1e-5`, `loss=dice_ce`, `class_weighting=none`, `max_optimizer_steps=20`,
`max_epochs=1`, `num_workers=0`, `resume_disabled=true`, `hyperparameter_search_prohibited=true`,
`candidate_comparison_prohibited=true`, `augmentation_disabled=true`,
`sliding_window_roi_size=[96,96,64]`, `sliding_window_overlap=0.25`,
`sliding_window_batch_size=1`, `hu_window=[-1000.0,1000.0]`, `fit_scope=no_data_dependent_fit`,
`wall_clock_limit_seconds=2700.0`.

Actual step/sampling counts: `executed_step_count=20`, all 20 `step_finite_status=true`,
`positive_patch_count=10`, `negative_patch_count=10` (exact approved 10:10 ratio),
`train_forward_shape=[1,2,64,64,32]`.

Elapsed time: `340.38` seconds (~5.7 minutes), well under the 2700-second (45-minute) watchdog
limit. Peak memory was not measured (not claimed as available).

Sliding-window validation result: `sliding_window_count=245`, `validation_output_shape=
[1,2,512,512,227]`, `validation_finite_status=true`, ROI `[96,96,64]`, overlap `0.25`, batch size
`1`, device `cpu`, AMP `false`. No Dice/IoU/HD95/NSD/lesion-recall/precision/F1/false-positive-
lesions-per-scan/volume-error/threshold-comparison/prediction file was computed or published;
confirmed by both the Supervisor's and the independent reviewer's grep scans of all four published
JSON files (the only hits were inside the fixed `non_scientific_disclaimer` sentence explaining
what was *not* computed).

Checkpoint: relative filename `checkpoints/phase8_definitive_training_pilot_checkpoint.pt`,
SHA-256 `c22dd78aa9c9317a4021a9d4533e95a07747321ebf2914fa9a7addba6df6dfc2` (recomputed
independently by both the Supervisor and the reviewer, matches the published metadata exactly),
byte size `1381530`, `checkpoint_round_trip_verified=true`. Flags: `pilot_only=true`,
`tiny_verification_only=false`, `freeze_eligible=false`, `selection_eligible=false`,
`definitive_training=false`, `scientific_metric_eligible=false`, nested
`checkpoint_metadata.completion_status="synthetic_smoke"`. The existing
`Phase8CheckpointMetadata.__post_init__` validator in `internal_evidence.py` structurally rejects
`freeze_eligible=True` whenever `completion_status != "completed"` (lines 346-349), which
mechanically forbids this checkpoint from ever being marked freeze eligible.

Generated JSON artifacts (relative to the pilot output root) and SHA-256 hashes:

- `phase8_definitive_training_pilot_config.json`:
  `130fb83a8dea8d70c50cbe1db8ad7d5a1a80103760b5581bfe7577a19fcc7275`
- `phase8_definitive_training_pilot_access_ledger.json`:
  `672f1ff8dafa70172cb79fd41625d1dea335fdd64f79ca309f8886b9aeff905f`
- `phase8_definitive_training_pilot_checkpoint_metadata.json`:
  `ca221222cdf39be3fe7ea76e4189e74fe8e0dd82932f5ff54d27aaa15dc42cec`
- `phase8_definitive_training_pilot_summary.json`:
  `557a6be2c68158a4ef3f66d4d7d7ecd22880c4033a61e5ecb6b3ed5e12588198`

Access ledger: exactly the three approved anonymous case IDs, `file_access_count=6`,
`internal_test_opened=false`, `external_data_opened=false`, `prior_checkpoint_loaded=false`. No
absolute path, raw filename (e.g. `liver_43.nii.gz`), PHI, NIfTI header, voxel array, or prediction
appears in any of the four generated JSON files — confirmed by grep scans from both the Supervisor
and the independent reviewer.

Raw-data modification result: all 6 real dataset files were rehashed after the run by both the
Supervisor and the independent reviewer; every hash matched the pre-run hash exactly. No dataset
file was modified.

Independent review (`P8R-BOUNDED-PILOT-REVIEW`): **PASS**, all 14 checklist items plus the
orientation-fix verification confirmed via independent re-derivation (not by trusting the
Supervisor's claims). One non-blocking residual observation recorded above (no watchdog check
during the single sliding-window pass itself).

### Boundary confirmation

- Only the three approved anonymous cases were opened; internal-test and external (3D-IRCADb) data
  were never accessed (`internal_test_opened=false`, `external_data_opened=false`, confirmed by
  source-level review of the case-selection filter).
- No prior checkpoint was loaded as initialization (`prior_checkpoint_loaded=false`; the only
  `torch.load` call in the module deserializes the checkpoint bytes just written in the same run,
  as a save/reload self-verification, not a disk read of a pre-existing file).
- No definitive-training release, model selection, threshold tuning, Wave 4, or Wave 5 work
  occurred; `docs/phase8/SUPERVISOR_HANDOFF.md` was not modified until this post-review update, and
  the pilot source module contains no reference to `Phase8DefinitiveExecutionRelease`,
  `release_state="released"`, model-selection, or threshold-selection logic.
- Generated outputs remained entirely outside Git; `git status --short --branch` before and after
  the run showed only the three expected pilot source/test files.

### Exact next action

Substage 4B is `PILOT_PASS` and stops here per its approval scope. Substage 4A's definitive-training
release gate remains unreleased (`release_state="not_released"` is still the only reachable state).
Any future definitive-training execution, model/threshold/support selection, or Wave 4 rerun
requires its own separate explicit user approval; this pilot's approval does not extend to it.

Wave 4 remains `BLOCKED`. Wave 5 remains blocked and unreleased.

## Internal Evidence Remediation Substage 4A

Status: completed as contract-and-boundary-only source implementation; not committed, not pushed,
and not a Wave 4 readiness release. Gate 8 is not claimed. Wave 4 remains `BLOCKED`. Wave 5 remains
blocked and unreleased. This substage is the first half of Substage 4 ("Definitive development
training and validation-only selection"): it built every contract, executor boundary, and CLI
surface needed for a future definitive MONAI SegResNet training run, gated behind an explicit,
separately-versioned release artifact that does not exist yet and cannot be produced in this
session. No real dataset, checkpoint, model, or training touch occurred; no NIfTI/DICOM/ZIP/label/
prediction/checkpoint file was opened; `/Volumes` was not accessed; 3D-IRCADb-01 was not accessed;
the immutable internal test cohort was not accessed; Wave 4 was not rerun; Wave 5 was not begun.

Real delegated agent used: `P8R-DEFINITIVE-RUNNER` (this agent), implementing
`src/protoem_ct/external/definitive_training.py`,
`tests/unit/test_phase8_definitive_training.py`, the two new CLI commands in
`src/protoem_ct/cli/main.py`, and additive re-exports in `src/protoem_ct/external/__init__.py`.

Approved Substage 4A schema surfaces, both `v1`:

- `phase8_definitive_training_config`
- `phase8_definitive_execution_release`

New contracts reuse the Substage 1 schemas unchanged (`phase8_preprocessing_decision`,
`phase8_checkpoint_metadata`, `phase8_validation_evidence_reference`,
`phase8_model_selection_decision`, `phase8_threshold_decision`, `phase8_support_policy`) rather than
forking them.

Scientific policy encoded (contract defaults/validators, not a frozen decision record):

- Single preregistered candidate: `monai_segresnet_baseline`; `selection_status` is always
  `single_candidate_preregistered`.
- Primary validation metric: tumor Dice.
- Fixed deterministic tie-break order (this module's own constant, distinct in field order from the
  general multi-candidate order in `docs/phase8/INTERNAL_EVIDENCE_REMEDIATION_PLAN.md`): lesion F1
  descending, then HD95 ascending when defined, then false-positive lesions per scan ascending, then
  trainable-parameter count ascending, then candidate ID lexicographic.
- Threshold policy: `fixed_constant` at `0.5` (no threshold-search parameter exists anywhere in the
  public API).
- Support policy: `no_support` (no support-manifest parameter exists anywhere in the public API).
- `device_type` fixed to `"cpu"`, `amp_enabled` fixed to `False`, `no_external_data` and
  `internal_test_excluded` fixed to `True` for every object the public builder can produce.

CLI commands:

- `plan-phase8-definitive-development-training`: plans (never executes) a future definitive
  training run; publishes a `Phase8DefinitiveTrainingConfig` whose `execution_release_state` is
  always `"awaiting_explicit_user_approval"` alongside a `Phase8DefinitiveExecutionRelease` whose
  `release_state` is always `"not_released"`. Never opens an image, label, prediction, or checkpoint
  file.
- `run-phase8-definitive-development-training`: the honest "run" command. Loads a config and release
  from explicit paths, verifies the release binds to that config, and fails closed with
  `Phase8DefinitiveTrainingNotReleasedError` before opening any manifest, split, or medical file
  whenever the release is absent, invalid, mismatched, or `not_released` -- which is every real
  invocation in this environment today, since no function reachable from this codebase's own
  tooling can produce a `released` release.

Exact execution/publication boundary:

- `Phase8DefinitiveExecutionRelease.release_state` can only be `"not_released"` for every object
  produced by `build_unreleased_definitive_execution_release(...)`, the only release-builder
  reachable from the CLI or any public function in this module. A `released` release can only be
  constructed via direct, out-of-band dataclass construction, exercised solely inside this
  substage's own tests, to prove the contract's shape without ever emitting a real-release code
  path.
- `execute_phase8_definitive_training(...)` and all six `publish_phase8_definitive_*` functions
  validate the release's `bound_config_hash` against the config's `config_hash` and the release's
  `release_state` before opening any file path, reading any manifest/split, or importing
  torch/monai/nibabel. Test-verified with a `builtins.open` spy: zero file-open calls occur before
  the not-released/mismatch error is raised.
- `Phase8DefinitiveTrainingExecutor` is a typed `Protocol` only; every method body is
  `raise NotImplementedError`. No MONAI dataset construction, Torch training loop, or checkpoint I/O
  exists anywhere in the module.
- No unconditional `import torch`/`import monai`/`import nibabel` statement exists anywhere in the
  module (regex- and AST-verified by an automated static-scan unit test); the module never imports
  or invokes any of them.

Targeted local verification (Supervisor-run):

- `uv run ruff format --check src/protoem_ct/external/definitive_training.py tests/unit/test_phase8_definitive_training.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py`: PASS, `4 files already formatted`
- `uv run ruff check src/protoem_ct/external/definitive_training.py tests/unit/test_phase8_definitive_training.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py`: PASS, `All checks passed!`
- `uv run mypy src/protoem_ct/external/definitive_training.py tests/unit/test_phase8_definitive_training.py src/protoem_ct/external/__init__.py src/protoem_ct/cli/main.py`: PASS, `Success: no issues found in 4 source files`
- `uv run pytest -q tests/unit/test_phase8_definitive_training.py`: PASS, `53 passed`
- `uv run pytest -q tests/unit/test_phase8_internal_evidence.py tests/unit/test_phase8_real_development_runner.py tests/unit/test_phase8_tiny_real_verification.py`: PASS, `67 passed, 4 skipped` (unmodified regression surface)
- `uv run protoem-ct plan-phase8-definitive-development-training --help`: PASS
- `uv run protoem-ct run-phase8-definitive-development-training --help`: PASS
- Synthetic end-to-end plan-only CLI run into a `mktemp -d` output root: PASS; the immediately
  following `run-phase8-definitive-development-training` invocation against the published
  not-released artifacts correctly failed closed with `Phase8DefinitiveTrainingNotReleasedError`
  (exit code 1).
- `git diff --check`: PASS, no output.

Boundary confirmation:

- `/Volumes` was not accessed.
- No development or external dataset was accessed.
- No NIfTI, DICOM, ZIP, mask, label, prediction, or checkpoint file was opened or modified.
- No real training, inference, GPU/MPS use, checkpoint creation, or real metric computation
  occurred.
- The immutable internal test cohort was not accessed.
- Wave 4 was not rerun; Wave 5 was not begun.
- Only synthetic in-memory/`tmp_path`/`mktemp -d` JSON fixtures were used.

Exact next action: Substage 4B (or a combined re-scoped continuation) requires its own separate
explicit user approval to produce and bind a real `released` execution release before any real
definitive training compute may begin. Wave 4 and Wave 5 remain `BLOCKED` and must stay blocked
until a Substage 4B/5/6 remediation path is separately approved and completed.

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

## Post-Pilot Watchdog Remediation

Status: completed, source-and-test-only, not committed, not pushed.

The single approved real bounded pilot execution described above already completed successfully
before this remediation began. This remediation did **not** repeat that execution, did not access
`/Volumes` or any dataset, did not open the existing pilot output root, and did not change any
generated pilot artifact (config, access ledger, checkpoint metadata, summary, or checkpoint file).
The repaired code did not generate the existing run; the existing run's hashes recorded above are
unaffected by this remediation.

An independent reviewer had identified one residual gap in the previously-approved watchdog:
`_check_watchdog` in `definitive_training_pilot.py` only checks elapsed time between stages, so it
could not interrupt the single blocking `sliding_window_inference` call (or the training loop or
checkpoint I/O) while it was actively running.

Fix: a new parent-process supervisor, `run_phase8_bounded_pilot_with_watchdog`, runs the entire
existing `run_phase8_bounded_pilot` pipeline (medical data loading through artifact publication) in
an isolated `multiprocessing` child process (`spawn` context, macOS/CPU-valid), joined against the
same single 45-minute (`DEFAULT_WALL_CLOCK_LIMIT_SECONDS`) deadline, minus time already spent
starting the child. If the child does not finish in time, it is terminated (SIGTERM, escalating to
`.kill()`) and joined, a new `Phase8BoundedPilotProcessWatchdogTimeoutError` is raised (distinct
from the existing in-process timeout), no summary is published, and there is no automatic retry.
Cleanup runs in a `finally` on every path (success, error, timeout), so no orphan child process can
remain. The CLI command `run-phase8-bounded-real-development-pilot` now calls this supervised entry
point instead of the raw one; the existing `--approve-bounded-pilot` gate still runs first and still
blocks any file access without it. No scientific config (patch size, ROI, overlap, hyperparameters,
model, seed, step count, case selection) was changed.

Synthetic timeout test result (`test_supervised_pilot_kills_blocking_child_before_it_returns`): a
deliberately blocking synthetic target (`time.sleep(5.0)`) under a 0.2-second test deadline is
actively killed before it returns; `multiprocessing.active_children()` is empty afterward (real
process death, not a cooperative check); the output root is never created (no partial-success
publication); and the blocking target's call counter is exactly `1` (no retry). A companion test
(`test_supervised_pilot_deadline_covers_publication_not_just_earlier_stages`) proves the same single
deadline also covers artifact publication, not just the earlier stages.

Independent review (`P8R-PILOT-WATCHDOG-REVIEW`): **APPROVED**. All requirements (deadline coverage
of the full pipeline including publication, real process-level termination, dedicated error type, no
partial success, no retry, macOS/CPU-valid spawn mechanism, cleanup with no orphans, approval gate
still precedes the supervised call, no scientific config changed, single source of truth for the
45-minute default) were independently verified by reading the diff and rerunning the checks below.
One non-blocking style nit was found and fixed (the CLI's `--wall-clock-limit-seconds` default now
imports `DEFAULT_WALL_CLOCK_LIMIT_SECONDS` instead of duplicating the literal `2700.0`).

**Incident disclosure:** while implementing this fix, the P8R-PILOT-WATCHDOG-FIX subagent
accidentally overwrote the pre-existing, untracked, never-committed test file
`tests/unit/test_phase8_definitive_training_pilot.py` and could not recover the original content (no
git blob, no editor local history, no Time Machine backup existed for an untracked file). It was
reconstructed from scratch against the current source module. The independent reviewer found the
reconstruction's coverage solid but noted one concrete gap versus what this handoff's own prior
entries describe: `test_geometry_validation_rejects_non_ras_orientation` was missing. That test was
re-added directly by the Supervisor after review (LPS-affine input asserted to raise
`Phase8BoundedPilotOrientationError`) and now passes.

Verification commands run (pilot files only, not the full suite):

- `ruff format --check src/protoem_ct/external/definitive_training_pilot.py src/protoem_ct/cli/main.py tests/unit/test_phase8_definitive_training_pilot.py`: PASS, `3 files already formatted`
- `ruff check` (same files): PASS, `All checks passed!`
- `mypy src/protoem_ct/external/definitive_training_pilot.py src/protoem_ct/cli/main.py`: PASS, `Success: no issues found in 2 source files`
- `pytest tests/unit/test_phase8_definitive_training_pilot.py -q`: PASS, `30 passed, 6 skipped` (skips are the pre-existing torch/monai ambient-environment gate)
- `pytest tests/unit/test_phase8_internal_evidence.py tests/unit/test_phase8_real_development_runner.py -q`: PASS, `36 passed`
- `protoem-ct --help`: PASS, lists `run-phase8-bounded-real-development-pilot`
- `git diff --check`: PASS, clean
- `git diff --stat`: `docs/phase8/SUPERVISOR_HANDOFF.md` and `src/protoem_ct/cli/main.py` only (395 insertions, 0 deletions); untracked `src/protoem_ct/external/definitive_training_pilot.py` and `tests/unit/test_phase8_definitive_training_pilot.py` confirmed via `git status --porcelain`
- `git status --short --branch`: only the four expected files, nothing else touched

Wave 4 remains `BLOCKED`. Wave 5 remains blocked and unreleased.

## Post-Incident Test-Recovery Audit

Status: **BLOCKED**. A defect was found in the pilot source module while strengthening the
reconstructed test file. Per the audit's own constraint, the Supervisor did not fix it. Not
committed, not pushed. Gate 8 is not claimed. Wave 4 remains `BLOCKED`. Wave 5 remains blocked.

### Purpose

Prove whether the reconstructed `tests/unit/test_phase8_definitive_training_pilot.py` (see
"Incident disclosure" above) covers the complete approved Substage 4B pilot contract, not merely
the current implementation's behavior.

### Real subagents used (two, sequential)

- `P8R-PILOT-TEST-RECOVERY-AUDIT`: complete, read-only. Independently derived a 93-requirement
  coverage matrix (sections A-I of the approved pilot contract) against the reconstructed test file,
  `internal_evidence.py`, `definitive_training.py`, `definitive_training_pilot.py`, `cli/main.py`,
  and this handoff. Result before remediation: **41 COVERED, 16 WEAK, 36 MISSING** of 93. Flagged one
  vacuous assertion (`forbidden == ["anon-p0002"] or forbidden == []`, tautologically true) and one
  critical absent case (no test for `Phase8BoundedPilotLabelDomainError`, the exact defect class from
  the Substage 3 incident).
- `P8R-PILOT-TEST-RECOVERY-REVIEW`: **not started**. Per its own precondition ("start only after any
  missing tests are added and all validation passes"), the independent review is deferred because
  remediation validation did not fully pass (see Defect below).

### Supervisor remediation applied

Added or strengthened tests only in `tests/unit/test_phase8_definitive_training_pilot.py`:

- Fixed the vacuous access-ledger assertion; replaced with a real forbidden-raw-filename/path scan.
- Added `test_geometry_validation_rejects_out_of_domain_label_value` (raw label `9`), a
  nonfinite-image-value rejection test, and an affine-only (same-shape) mismatch rejection test.
- Added a direct `_normalize_hu` unit test (HU clip `[-1000,1000]` then linear rescale to
  `[-1,1]` at known sample points) and an `_extract_padded_patch` boundary test (shape and fill
  value at a volume edge).
- Added a sampling-schedule test proving positive centers index actual tumor voxels and negative
  centers do not (previously only counts were checked).
- Added a parametrized test exercising the config's `replace()`-reject path for every remaining
  hard-coded field (previously 3 of ~20 fields; now all ~20, plus `patch_size` and
  `sliding_window_roi_size`).
- Added three prerequisite cross-hash-mismatch rejection tests (`preprocessing_evidence_hash`,
  `model_family`, `uses_external_artifacts`), all confirmed to fail before any NIfTI access.
- Added self-hash tamper-rejection tests for `Phase8BoundedPilotConfig`,
  `Phase8BoundedPilotAccessLedger`, and the checkpoint-metadata wrapper.
- Added a static source-scan test proving the pilot module never imports the Substage 4A
  `definitive_training` release contract, so a `released` `Phase8DefinitiveExecutionRelease` can
  never substitute for `--approve-bounded-pilot`.
- Added torch/monai-gated tests (skipped in the ambient environment, run in the isolated Phase 3 CPU
  environment): label-binarization correctness on `_load_pilot_training_patch`, nonfinite-loss and
  nonfinite-gradient fail-closed paths in `_run_bounded_training_steps` (via a custom
  `torch.autograd.Function` injecting a NaN gradient), a sliding-window-inference call-parameter spy
  proving exactly one call with the required ROI/overlap/batch-size and nonfinite-logits fail-closed
  behavior, an optimizer/loss identity check on the real pipeline (`torch.optim.AdamW` with
  `lr=1e-4`/`weight_decay=1e-5`, exactly one `torch.load` call proving no prior checkpoint is ever
  read), and a checkpoint-reload-on-corrupted-bytes test.
- Extended the full-pipeline leakage scan to include lesion-recall/precision/F1,
  false-positive-lesions-per-scan, volume-error key names, raw dataset filenames, and the
  repository-root path string; changed two "file present" smoke assertions to exact
  published-file-set equality.
- Net: 34 tests before -> 66 non-skipped ambient tests / 78 tests total after (24 new test
  functions, one of them parametrized into 21 cases).

### Defect found (source unchanged, reported per audit boundary)

`_reload_and_verify_checkpoint_from_bytes` in `src/protoem_ct/external/definitive_training_pilot.py`
(lines 1967-1979) catches only `(MemoryError, RuntimeError, KeyError, OSError)` around
`torch.load(...)`. Malformed/corrupted checkpoint bytes raise `_pickle.UnpicklingError`, which is a
subclass of `pickle.PickleError` / `Exception`, not of any caught type, so it propagates as a raw,
undocumented exception instead of the module's own typed fail-closed contract
`Phase8BoundedPilotRuntimeError("checkpoint_reload_failed")` that every other failure path in this
module (nonfinite loss, nonfinite gradient, nonfinite validation logits/probabilities, prerequisite
mismatches) consistently uses. This was caught only in the isolated Phase 3 CPU environment (real
`torch`), not in the ambient environment, by the new
`test_checkpoint_reload_fails_closed_on_corrupted_bytes` test. In the current production flow this
path only ever reloads bytes the same run just wrote (never a pre-existing file), so it has not
caused a real incident; it is a latent fail-closed-contract gap, not a data-safety breach. Per the
task's explicit boundary, the Supervisor did **not** modify `definitive_training_pilot.py` to widen
the `except` clause. The new test remains in the file, failing in the isolated environment, as
accurate evidence of the gap rather than being removed, weakened, or marked `xfail`.

### Validation run

- `uv run ruff format --check tests/unit/test_phase8_definitive_training_pilot.py`: PASS, `1 file
  already formatted`.
- `uv run ruff check tests/unit/test_phase8_definitive_training_pilot.py`: PASS, `All checks
  passed!`.
- `uv run mypy tests/unit/test_phase8_definitive_training_pilot.py`: PASS, `Success: no issues found
  in 1 source file`.
- `uv run pytest -q tests/unit/test_phase8_definitive_training_pilot.py` (ambient, no torch/monai):
  PASS, `66 passed, 12 skipped`.
- `env -u VIRTUAL_ENV uv run --project environments/phase3-baselines/intel-macos-cpu --locked
  --no-sync python -m pytest -q tests/unit/test_phase8_definitive_training_pilot.py` (isolated Phase
  3 CPU environment, real `torch`/`monai`): **1 failed, 77 passed** — the corrupted-checkpoint-bytes
  defect above; every other new and pre-existing test passed.
- `git diff --check`: PASS, no output.
- `git diff --stat`: `docs/phase8/SUPERVISOR_HANDOFF.md` and `src/protoem_ct/cli/main.py` only
  (unchanged by this audit until this section); untracked
  `src/protoem_ct/external/definitive_training_pilot.py` (unchanged) and
  `tests/unit/test_phase8_definitive_training_pilot.py` (test-only diff).
- `git status --short --branch`: only the same four expected files.
- Relevant internal-evidence/definitive-training regression tests and the CLI `--help` path were not
  re-run in this audit turn beyond what is already recorded above in "Internal Evidence Remediation
  Substage 4B"; no source file they cover changed.

### Final verdict: BLOCKED

Missing/weak coverage was substantially remediated (36 MISSING and 16 WEAK reduced to a small
residual, see limitations below), but full validation does not pass: one new, non-vacuous test
correctly demonstrates a real fail-closed-contract gap in `_reload_and_verify_checkpoint_from_bytes`
that the audit is not authorized to fix. `P8R-PILOT-TEST-RECOVERY-REVIEW` was not started because its
own precondition (all validation passing) was not met.

### Remaining limitations

- The corrupted-checkpoint-bytes defect above requires a source-code fix (widen the `except` clause
  to also catch `pickle.UnpicklingError`, or catch a broader `Exception` subset consistent with the
  module's fail-closed philosophy) in a future, separately authorized change.
- A handful of lower-priority WEAK items from the original 93-requirement matrix were not
  individually re-tested this turn (e.g., a dedicated exception-propagation test for a
  `sliding_window_inference` call that itself raises rather than returning NaN, and construction-time
  self-hash tamper tests for `Phase8BoundedPilotSummary` specifically) — these remain lower-severity
  residual gaps, not blockers, and should be closed alongside the defect fix.
- `P8R-PILOT-TEST-RECOVERY-REVIEW` (independent read-only re-inspection of the final test file and
  requirement matrix) has not yet run and must run after the defect fix and a full green validation
  pass, per its stated precondition.

Wave 4 remains `BLOCKED`. Wave 5 remains blocked and unreleased.

## Checkpoint-Reload Defect Fix and Recovery-Audit Completion

Status: **TEST_RECOVERY_AUDIT_PASS**. This section closes out the "Post-Incident Test-Recovery
Audit" above: the previously-disclosed checkpoint-reload defect is fixed under a separately
authorized, narrowly bounded scope, all validation now passes in both environments, and the
deferred independent review has run and passed.

### Recap of what this closes

- The original overwrite incident (accidental loss of the untracked
  `tests/unit/test_phase8_definitive_training_pilot.py`, reconstruction from scratch, and the
  independent audit's 93-requirement coverage matrix) is documented above under "Incident
  disclosure" and "Post-Incident Test-Recovery Audit" and is unchanged by this entry.
- The recovery audit found: pre-remediation **41 COVERED / 16 WEAK / 36 MISSING** of 93; after
  Supervisor remediation, "36 MISSING and 16 WEAK reduced to a small residual" (see that section
  for the itemized list of tests added).
- The recovery audit also found the checkpoint-reload defect (below) and left it unfixed and the
  new red test in place, per its own boundary.

### Checkpoint-reload defect fix

`_reload_and_verify_checkpoint_from_bytes` in `src/protoem_ct/external/definitive_training_pilot.py`
caught only `(MemoryError, RuntimeError, KeyError, OSError)` around `torch.load(...)`, so corrupted
checkpoint bytes raised a raw, undocumented `_pickle.UnpicklingError` instead of the module's own
fail-closed contract. Exact source change (only lines touched in this file):

1. Added `import pickle` to the top-level imports (alongside the existing `hashlib`, `io`, `json`,
   `multiprocessing`, `re`, `time` imports).
2. In `_reload_and_verify_checkpoint_from_bytes`, widened the except tuple from
   `(MemoryError, RuntimeError, KeyError, OSError)` to
   `(MemoryError, RuntimeError, KeyError, OSError, pickle.UnpicklingError)`, still raising
   `Phase8BoundedPilotRuntimeError("checkpoint_reload_failed") from exc`.

No other line in `definitive_training_pilot.py` changed: no broadened checkpoint formats, no change
to hash/size/state-dict/metadata/reload-compatibility checks, no `except Exception`, and
`KeyboardInterrupt`/`SystemExit`/unrelated programming defects are still not caught.

Proof the raw exception is now normalized with cause preserved (isolated Phase 3 CPU environment,
real `torch`):

```
type: Phase8BoundedPilotRuntimeError
msg: checkpoint_reload_failed
cause type: UnpicklingError
```

### Validation run

- `uv run ruff format --check src/protoem_ct/external/definitive_training_pilot.py tests/unit/test_phase8_definitive_training_pilot.py`: PASS, `2 files already formatted`.
- `uv run ruff check` (same files): PASS, `All checks passed!`.
- `uv run mypy` (same files): PASS, `Success: no issues found in 2 source files`.
- `uv run pytest -q tests/unit/test_phase8_definitive_training_pilot.py` (ambient, no torch/monai):
  PASS, `66 passed, 12 skipped`.
- `env -u VIRTUAL_ENV uv run --project environments/phase3-baselines/intel-macos-cpu --locked
  --no-sync python -m pytest -q tests/unit/test_phase8_definitive_training_pilot.py` (isolated Phase
  3 CPU environment, real `torch==2.2.2`/`monai==1.4.0`): **PASS, 78 passed**, 0 failed, 0 skipped
  (previously 77 passed / 1 failed; the previously-failing
  `test_checkpoint_reload_fails_closed_on_corrupted_bytes` now passes).
- Regression: `uv run pytest -q tests/unit/test_phase8_internal_evidence.py tests/unit/test_phase8_definitive_training.py tests/unit/test_phase8_real_development_runner.py tests/unit/test_phase8_tiny_real_verification.py`:
  PASS, `120 passed, 4 skipped`.
- `git diff --check`: PASS, no output.
- `git diff --stat` (including untracked): `docs/phase8/SUPERVISOR_HANDOFF.md` (this entry) and
  `src/protoem_ct/cli/main.py` (byte-for-byte unchanged during this step — diffed and hashed before
  and after the fix, identical); untracked `src/protoem_ct/external/definitive_training_pilot.py`
  (the two-line fix above) and `tests/unit/test_phase8_definitive_training_pilot.py` (unchanged in
  this step; already contained the corrupted-bytes test from the prior audit turn).
- `git status --short --branch`: only the same four expected files.

### Independent review: `P8R-PILOT-TEST-RECOVERY-REVIEW`

Real, read-only subagent, run after all validation above passed (its stated precondition).
Verdict: **PASS**. Findings:

- The narrow fix is exactly the two changes described above; no other line in the file changed; no
  `except Exception`; `KeyboardInterrupt`/`SystemExit` not swallowed; `from exc` chaining intact so
  `__cause__` is the original `pickle.UnpicklingError`; no hash/size/state-dict/metadata/reload check
  weakened; `cli/main.py` confirmed to have received no new edits during this step.
- Coverage: deferred to the prior audit's baseline (41/16/36) plus the handoff's remediation list;
  spot-checked broadly (checkpoint reload, orientation/geometry, label-domain, hash-mismatch,
  self-hash tamper, forbidden-metric scan, sliding-window spy) and confirmed all claimed additions
  are present and passing.
- Two residual WEAK items remain, both independently assessed as **non-critical**:
  1. No dedicated test for `sliding_window_inference` raising an exception (vs. returning NaN) —
     the call site has no surrounding try/except, so any raised exception already propagates
     uncaught by default; no fail-closed contract is at risk.
  2. No construction-time self-hash tamper test specifically for `Phase8BoundedPilotSummary` — the
     same self-hash-verification pattern is already covered for `Phase8BoundedPilotConfig`,
     `Phase8BoundedPilotAccessLedger`, and the checkpoint-metadata wrapper; low incremental risk.
- Every CRITICAL requirement (fail-closed checkpoint reload, orientation rejection, label-domain
  rejection, hash-mismatch rejection, nonfinite-loss/gradient/logits rejection, no-Dice/IoU-leakage
  source scan, no-retry structure) has non-vacuous behavioral/structural test coverage: **YES**.
- Incident-disclosure accuracy re-confirmed: `test_geometry_validation_rejects_non_ras_orientation`
  exists, constructs an LPS affine, and asserts `Phase8BoundedPilotOrientationError`, matching the
  handoff's account exactly.
- Independently re-ran the ambient suite and observed the same `66 passed, 12 skipped`.

### Final 93-requirement coverage counts

- Before this recovery work (original audit, pre-remediation): **41 COVERED / 16 WEAK / 36
  MISSING**.
- After the recovery audit's test remediation and this turn's source fix: small residual of **2
  WEAK, non-critical** items (listed above); **0 MISSING critical items**; all other items COVERED.
  (Exact per-item re-derivation of all 93 was not repeated this turn; the independent reviewer
  spot-checked broadly across sections A-I rather than re-deriving line-by-line, consistent with the
  prior audit's own itemization.)

### No real execution or dataset access

Confirmed: no `/Volumes` path was accessed at any point in this step; no dataset, checkpoint file,
generated pilot artifact, or completed real pilot output was read, written, or modified; the real
pilot was not rerun; no real training or inference was executed; only the existing unit-test suite
(ambient and isolated-environment `pytest`) and static-analysis tools (`ruff`, `mypy`) ran.

### Final verdict: `TEST_RECOVERY_AUDIT_PASS`

The checkpoint-reload source defect is fixed under the strict narrow boundary, the isolated
environment has zero failures (78 passed, 0 failed), no critical requirement remains WEAK or
MISSING, and independent review passed.

Wave 4 remains `BLOCKED`. Wave 5 remains blocked and unreleased.
