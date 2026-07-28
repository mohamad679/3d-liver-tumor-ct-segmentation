# Decisions

## Decision Log Template

### YYYY-MM-DD: Decision Title

- Status: proposed | accepted | superseded
- Context:
- Decision:
- Consequences:

## Initial Decisions

### 2026-07-24: Use Python 3.11

- Status: accepted
- Context: The project needs a stable Python version compatible with the planned scientific and engineering stack.
- Decision: Use Python 3.11.
- Consequences: Tooling, type checking, packaging, and CI will target Python 3.11.

### 2026-07-24: Use uv with pyproject.toml

- Status: accepted
- Context: The project needs reproducible dependency management and standard Python project metadata.
- Decision: Use uv with `pyproject.toml`.
- Consequences: Dependency synchronization and command execution will use uv.

### 2026-07-24: Use src Layout

- Status: accepted
- Context: The project needs clear separation between importable package code and repository support files.
- Decision: Use a `src/` layout.
- Consequences: Package imports and type checking will be configured against `src`.

### 2026-07-24: Keep Smoke Tests CPU-Only

- Status: accepted
- Context: Phase 0 must verify the environment contract without requiring GPU access.
- Decision: Use CPU-only smoke tests.
- Consequences: CI can run basic validation without specialized hardware.

### 2026-07-24: Use Synthetic NIfTI Fixtures

- Status: accepted
- Context: Validation tests need NIfTI-like inputs without committing medical data.
- Decision: Use tiny synthetic NIfTI CT and tumor-mask fixtures.
- Consequences: Tests can cover image and mask validation without including real patient data.

### 2026-07-24: Exclude Training from Phase 0

- Status: accepted
- Context: Phase 0 is limited to environment and validation scaffolding.
- Decision: Do not implement training in Phase 0.
- Consequences: Model code, real-data ingestion, and training workflows are deferred.

### 2026-07-24: Keep Local Paths Out of Tracked Configuration

- Status: accepted
- Context: Local dataset paths and machine-specific locations must not leak into version control.
- Decision: Local paths are allowed only through environment variables or ignored configuration.
- Consequences: Tracked files must avoid user-specific absolute paths and private storage locations.

### 2026-07-24: Exclude Medical Data and Generated Predictions from Git

- Status: accepted
- Context: The repository must not contain protected or large generated artifacts.
- Decision: Medical data and generated predictions are excluded from Git.
- Consequences: Data, model weights, predictions, credentials, API keys, and large generated artifacts must remain untracked.

## Phase 0 Close-Out Decisions

### 2026-07-25: Accept Gate 0 Locally

- Status: accepted
- Context: The Phase 0 local audit verified Python 3.11.9, a valid uv lockfile, Ruff lint and format checks, mypy over `src` and over `src tests`, pre-commit configuration and hooks, the `protoem-ct validate-pair` CLI help path, 24 pytest tests, and 1 smoke test.
- Decision: Gate 0 is accepted locally after `24 passed` for the full test suite and `1 passed` for the smoke test.
- Consequences: Phase 0 close-out documentation may be prepared without claiming GitHub-hosted CI has passed.

### 2026-07-25: Keep Synthetic NIfTI Fixtures Temporary

- Status: accepted
- Context: Phase 0 validation requires tiny NIfTI-like image and label inputs without committing medical data or generated binary artifacts.
- Decision: Synthetic NIfTI fixtures remain temporary pytest artifacts created under pytest temporary directories and are not committed.
- Consequences: Tests remain reproducible without tracking NIfTI, DICOM, medical data, predictions, model weights, checkpoints, credentials, or generated binary artifacts.

### 2026-07-25: Do Not Claim Hosted CI Before Execution

- Status: accepted
- Context: CPU-only GitHub Actions workflow configuration exists, but hosted CI status can only be established after the branch is pushed and the workflow executes on GitHub.
- Decision: GitHub-hosted CI status must not be claimed before push and execution.
- Consequences: Local Gate 0 status and hosted CI status remain separate evidence categories.

### 2026-07-25: Require Phase 0 Integration Before Phase 1

- Status: accepted
- Context: Phase 1 depends on the completed Phase 0 environment contract and repository workflow.
- Decision: Phase 1 cannot begin until the Phase 0 branch is committed, pushed, reviewed, and integrated according to the repository workflow.
- Consequences: Phase 1 remains the next phase, but it must not start from unintegrated Phase 0 close-out work.

## Phase 1 Planning Decisions

### 2026-07-25: Use Snakemake for the Synthetic DAG

- Status: accepted
- Context: Phase 1 needs a reproducible synthetic end-to-end DAG with explicit stage inputs and outputs.
- Decision: Use Snakemake to orchestrate the Phase 1 synthetic DAG.
- Consequences: The implementation review must confirm Snakemake dependency selection and macOS x86_64 compatibility before adding dependencies.

### 2026-07-25: Use Local MLflow Tracking Only

- Status: accepted
- Context: Phase 1 needs experiment metadata capture without introducing remote infrastructure or credentials.
- Decision: Use local MLflow tracking only; do not use a remote tracking server in Phase 1.
- Consequences: Phase 1 must avoid remote tracking configuration and guard against leaking machine-specific local paths into tracked files or reports.

### 2026-07-25: Generate Reports from Machine-Readable Artifacts

- Status: accepted
- Context: Scientific claims and report values must be traceable to persisted artifacts rather than transient process state or edited prose.
- Decision: Generate Phase 1 reports only from validated machine-readable artifacts.
- Consequences: Report generation must read saved JSON artifacts and preserve traceability from report values back to artifact schemas.

### 2026-07-25: Use Deterministic Synthetic Inputs and Dummy Inference

- Status: accepted
- Context: Phase 1 validates pipeline mechanics without beginning model development or scientific evaluation.
- Decision: Use deterministic synthetic inputs and deterministic dummy inference.
- Consequences: The DAG can test reproducibility, hashing, evaluation, and reporting while deferring real-data and model work.

### 2026-07-25: Treat Generated Artifacts as Disposable

- Status: accepted
- Context: Generated artifacts can become stale, large, or machine-specific and must not be mistaken for source truth.
- Decision: Treat Phase 1 generated artifacts as disposable and excluded from Git.
- Consequences: Gate 1 requires clean rebuild testing from an empty generated-artifact directory.

### 2026-07-25: Require Clean Rebuild Testing Before Gate 1

- Status: accepted
- Context: Stale generated artifacts can cause false success in DAG and report checks.
- Decision: Require a clean rebuild test before accepting Gate 1.
- Consequences: Gate 1 cannot pass unless the final synthetic report is regenerated from an empty generated-artifact state.

### 2026-07-25: Defer Real-Data and Model Work

- Status: accepted
- Context: Phase 1 is limited to documentation planning and later synthetic DAG implementation.
- Decision: Do not begin real-data ingestion, neural-network training, model baselines, few-shot protocol work, robustness evaluation, external validation, or LLM/VLM work in Phase 1.
- Consequences: Phase 1 remains an engineering reproducibility milestone and cannot produce scientific model claims.

## Phase 1 Close-Out Decisions

### 2026-07-25: Base Gate 1 on an Explicit External Generated Root

- Status: accepted
- Context: Generated Phase 1 data, predictions, reports, JSON artifacts, Snakemake metadata, and MLflow runs must not become repository source files.
- Decision: Gate 1 local verification uses an explicit absolute `generated_root` supplied through Snakemake `--config`, preferably beneath a temporary directory outside the repository.
- Consequences: The Snakefile does not derive output roots, Git commits, or timestamps from the environment or current working directory. Local Gate 1 evidence depends on a clean external generated root.

### 2026-07-25: Keep Report Generation JSON-Only and Independent of MLflow

- Status: accepted
- Context: The Phase 1 report must be deterministic and traceable to persisted machine-readable artifacts without coupling the report to local MLflow run identifiers or tracking state.
- Decision: Generate the report only from persisted JSON artifacts and keep MLflow tracking as a separate post-report operation.
- Consequences: The Snakemake `report` rule does not read NIfTI files, query MLflow, recompute metrics, or depend on `protoem-ct track-run`.

### 2026-07-25: Treat Dummy-Inference Metrics as Pipeline Verification Values

- Status: accepted
- Context: Phase 1 uses synthetic data and deterministic dummy inference to verify artifact linkage, hashing, evaluation, report generation, and orchestration.
- Decision: Metrics produced by dummy inference are pipeline-verification values only, not scientific model results.
- Consequences: Phase 1 close-out must not claim model performance, clinical validity, generalization, or state-of-the-art behavior.

## Phase 2 Planning Decisions

### 2026-07-25: Separate Development and External Cohorts

- Status: accepted
- Context: Development QA and protocol decisions must not consume held-out external information.
- Decision: Use LiTS labeled training data as the development cohort and reserve real
  3D-IRCADb-01 exclusively for Phase 8 external validation.
- Consequences: Phase 2 must not scan, inventory, QA, split, tune on, or summarize real external
  data, and external labels cannot influence development decisions.

### 2026-07-25: Treat LiTS and MSD Task03 Liver as Non-Independent

- Status: accepted
- Context: MSD Task03 Liver is derived from LiTS and can represent the same cases in a different
  layout.
- Decision: Never count or analyze LiTS and MSD Task03 Liver as independent cohorts.
- Consequences: Alternate MSD-style layout support cannot create an additional cohort, validation
  set, or external test set; duplicate and overlap controls must enforce this boundary.

### 2026-07-25: Keep the External Cohort Untouched Until Phase 8

- Status: accepted
- Context: Even unlabeled inventory or QA statistics from the external cohort could influence
  development protocol choices.
- Decision: Permit a 3D-IRCADb-style adapter interface in Phase 2 only when exercised with synthetic
  fixtures; real 3D-IRCADb-01 remains untouched until Phase 8.
- Consequences: Phase 2 commands, tests, manifests, reports, and leakage checks cannot require,
  discover, or access the real external root.

### 2026-07-25: Use Anonymous Deterministic Manifest Identifiers

- Status: accepted
- Context: Reproducible manifests need stable case identity without retaining identifying source
  values.
- Decision: Use deterministic anonymous patient and case IDs containing no raw patient name,
  medical-record identifier, birth date, accession number, or original DICOM identifier. If stable
  source mapping is necessary, use a one-way deterministic project namespace with explicit collision
  checks and do not commit a reverse mapping.
- Consequences: Reversible pseudonymization is not prescribed; schemas and tests must reject unsafe
  identifiers, collisions, duplicate IDs, and source-identifier mappings in tracked artifacts.

### 2026-07-25: Require Explicit External Data and Generated Roots

- Status: accepted
- Context: Dataset discovery based on current working directory or machine-wide search can access the
  wrong data and leak machine-specific paths.
- Decision: Supply dataset roots only by explicit CLI argument or ignored local configuration, and
  write all outputs only beneath an explicit external generated root.
- Consequences: No absolute local path is committed; adapters cannot search the computer, infer a
  dataset location, write into source roots, or follow symlinks outside the explicit root.

### 2026-07-25: Split Development Data at Patient Level Only

- Status: accepted
- Context: Volume- or slice-level assignment can place observations from one patient into multiple
  partitions.
- Decision: Create train, validation, and immutable internal-test assignments at patient level using
  an explicit seed and versioned policy, deterministic ordering, complete assignment, and zero
  patient and case overlap.
- Consequences: Split artifacts link to the source manifest hash and must be byte-identical for
  identical approved inputs. Final ratios and stratification details require approval before real
  split generation.

### 2026-07-25: Keep QA Read-Only

- Status: accepted
- Context: Phase 2 QA is intended to characterize source data, not transform it.
- Decision: QA performs no resampling, reorientation, normalization, clipping, repair, label
  remapping, interpolation, or preprocessing fitting and never modifies source files.
- Consequences: QA artifacts describe the data as found, and any later preprocessing is a separate
  versioned phase and artifact chain.

### 2026-07-25: Use Fixed Histogram Bins and Explicit 3D Connectivity

- Status: accepted
- Context: Per-cohort fitted histograms and implicit component connectivity make summaries
  incomparable or ambiguous.
- Decision: Use histogram bins from tracked configuration shared across development QA and a
  documented explicit 3D connected-component connectivity.
- Consequences: Bins are never fitted per cohort. Proposed 26-connectivity, final histogram limits
  and bins, and the connectivity choice require explicit approval before real QA.

### 2026-07-25: Report Invalid Cases Without Auto-Repair

- Status: accepted
- Context: Silent correction or omission would hide data quality failures and make cohort counts
  irreproducible.
- Decision: Invalid cases fail QA and remain represented with stable explicit failure reasons; they
  are not silently corrected or skipped.
- Consequences: Dataset and Markdown reports must expose failures and distinguish unavailable fields
  from valid zero-valued measurements.

### 2026-07-25: Prevent Preprocessing Fit Leakage

- Status: accepted
- Context: Fitting preprocessing on held-out partitions or external data would leak information into
  development.
- Decision: Fit no preprocessing before or during split creation and never fit preprocessing on
  validation, immutable internal-test, or external data.
- Consequences: Any future fit uses the development training partition only and records its source
  split and manifest hashes; external labels never affect fitted parameters.

### 2026-07-25: Use Synthetic Fixtures for All Hosted-CI Tests

- Status: accepted
- Context: Hosted CI cannot lawfully or reliably depend on local medical datasets.
- Decision: Generate all adapter, manifest, split, QA, report, and leakage-test inputs synthetically
  under temporary test directories.
- Consequences: Hosted CI requires no real dataset, medical image, external label, machine-specific
  path, network data download, or committed generated volume.

### 2026-07-27: Approve Explicit Real-Run Affine Tolerance Rerun

- Status: accepted
- Context: The first approved real Task03 Liver development-cohort QA run used
  `affine_tolerance=0.00001 mm`. It produced 122 passing cases and 9 failing cases; every failure
  used only the `affine_mismatch` code. All 9 failing pairs had matching orientation. The observed
  numeric differences were small: maximum spacing delta was at most `5.96046447754e-08 mm`,
  maximum selected-sform affine delta was `3.0517578125e-05 mm`, maximum corner displacement was
  approximately `5.71685607815e-05 mm`, maximum voxel translation delta was approximately
  `4.73484848555e-05 voxels`, and linear voxel-transform deviations were below `1e-7`.
- Decision: Treat these differences as consistent with NIfTI header floating-point precision rather
  than meaningful image/label misregistration. The approved rerun tolerance is explicitly
  `0.0001 mm`.
- Consequences: This is a real-run configuration decision, not a library default. Source NIfTI files
  remain unchanged, and the original failed run remains preserved externally. No Gate 2 pass is
  claimed until the rerun and artifact verification succeed.

### 2026-07-27: Accept Verified Phase 2 v2 Real-Data QA Close-Out

- Status: accepted
- Context: The preserved first real-run diagnostic showed affine differences consistent with NIfTI
  header floating-point precision rather than meaningful image/label misregistration. The approved
  v2 rerun used explicit `affine_tolerance=0.0001 mm` as a real-run configuration value, not a
  library default.
- Decision: Accept the verified v2 Phase 2 development-cohort QA and leakage evidence for Gate 2
  documentation close-out. The v2 artifacts recorded the full Git commit used for execution and are
  identified by these hashes: manifest
  `c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b`, split
  `936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb`, geometry QA
  `b8e65c558d5f574a1b40f9a5b15a033efbfbd6952006324fae35173940f94ee2`, lesion components
  `f7ce4874801cc8d5387bf93fd1480d376db4028026a90723dda2c5e2f48fbc60`, development summary
  `9b721e10232a08c9a5dd5361a11d7595abab04c071058d584e5895087b91951c`, final QA report
  `3f816db6227a1d2a8b349e248a9654eb4eed1e57760b48d32a6a275e1deca4f1`, and leakage audit
  `a32da3ac9d5689ec055c3875076ddc2075ca585219160c2d4200ca36484bf199`.
- Consequences: The v2 result is 131/131 geometry-label QA cases passed, a deterministic
  patient-level 91/20/20 split, 908 total lesions, and a leakage audit passed with zero overlaps and
  zero findings. LiTS and MSD Task03 Liver remain the same development-cohort lineage and are not
  independent cohorts. Source files were unchanged, the v1 failed run remains preserved externally,
  and v2 artifacts remain external. This decision does not claim model performance, external
  validation, clinical validity, or Phase 3 progress.

### 2026-07-28: Kick Off Phase 3 Baseline Scope and Gate 3

- Status: accepted
- Context: Phase 2 documentation close-out established the approved development-cohort QA and
  leakage boundary. Phase 3 now needs a narrow baseline implementation scope that preserves the
  external-validation boundary, keeps real development data disconnected during initial
  implementation, and prevents later-phase work from starting prematurely.
- Decision: Phase 3 begins from the merged Phase 2 commit. The baseline scope is limited to
  `nnU-Net v2` and `MONAI SegResNet`. Synthetic or tiny-data execution is sufficient for Gate 3.
  Real development data remains disconnected during initial implementation, and full real-cohort
  training is deferred until smoke and overfit gates pass. Artifacts, checkpoints, predictions,
  MLflow runs, and model weights must remain outside Git. AMP is conditional on safe device
  support. No LLM/VLM work may begin before baseline, few-shot, robustness, and external validation
  are complete. Task03 Liver remains the LiTS-derived development cohort, not an independent
  cohort.
- Consequences: Gate 3 scope is restricted to shared baseline infrastructure, deterministic metric
  generation, synthetic or tiny-data end-to-end paths, CPU smoke tests, and tiny-subset overfit
  evidence for the two approved baseline families only. Later scientific work, real-data baseline
  execution, external validation, and LLM/VLM efforts require separate approvals and later gates.
