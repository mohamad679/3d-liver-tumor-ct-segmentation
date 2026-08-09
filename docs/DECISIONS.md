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

### 2026-08-09: Skip Phase 9

- Status: accepted
- Context: The user explicitly decided not to execute the Phase 9 LLM/VLM/RAG track.
- Decision: Phase 9 is skipped by user decision and not executed. Phase 10 proceeds as the final core-project phase.
- Consequences: No LLM, VLM, RAG, or language-model result may be claimed in the final package.

### 2026-08-09: Close Core Project Through Artifact-Driven Phase 10 Reporting

- Status: accepted
- Context: The final scientific package must report completed evidence without rerunning training, inference, prediction generation, checkpoint selection, threshold selection, preprocessing changes, or external-label tuning.
- Decision: Generate Phase 10 tables, figures, statistics, report, cards, application materials, release checklist, and release-candidate metadata from saved machine-readable artifacts.
- Consequences: Unavailable Phase 3-7 numeric results are explicitly omitted or marked unavailable. The negative Phase 8 external-validation result remains negative and must not be reframed as strong external generalization.

## Phase 0 Close-Out Decisions

### 2026-08-04: Preregister Phase 8 Statistical and Publication Policies Before Results

- Status: accepted
- Context: Phase 8 Wave 4 requires metric, bootstrap, internal-versus-external comparison,
  qualitative-output, robustness/uncertainty, limitations, and publication policies to be fixed
  before external labels, predictions, metrics, or bootstrap results are opened. Existing Phase 3
  decisions define binary tumor metric semantics, but Phase 8-specific bootstrap and comparison
  policies were not previously frozen.
- Decision: Phase 8 external evaluation inherits the Phase 3 binary-tumor metric definitions and
  empty-mask behavior. Bootstrap resampling is at the anonymous case/patient unit with seed `1729`,
  `10000` resamples, `95%` percentile intervals, metric-specific valid-case accounting, and no
  lesion-level or voxel-level pseudoreplication. Internal-versus-external comparisons are
  independent and descriptive only, reporting point estimates, confidence intervals, and
  external-minus-internal differences without superiority, non-inferiority, broad generalization,
  clinical-validity, deployment, state-of-the-art, or post-result model/threshold/support changes.
  Qualitative outputs include all evaluation-eligible anonymous cases when the eligible count is at
  most `20`, ordered by anonymous ID; if future eligibility exceeds `20`, a fixed hash-ranked sample
  with seed `1729` is used. External robustness/uncertainty reporting is not included unless a
  compatible frozen Phase 7 policy is approved before evaluation. Small external sample size is a
  mandatory limitation, and publication remains JSON-first from persisted validated artifacts.
- Consequences: These policies may be referenced in Wave 4 readiness and later preregistration
  artifacts, but they do not by themselves make Wave 4 evaluation-ready. A real checkpoint,
  model-selection decision, support policy, threshold decision, and preprocessing decision still
  require valid development-only provenance before a freeze or evaluation-ready preregistration can
  be generated.

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

### 2026-07-28: Isolate the Phase 3 Intel macOS CPU Baseline Environment

- Status: accepted
- Context: The root project environment currently serves the Phase 0-2 tooling contract and keeps
  its existing root lock, including NumPy 2.x. The Phase 3 Intel macOS CPU baseline stack requires
  a narrower pinned environment around `nnunetv2[intel_macos]`, `torch==2.2.2`, and
  `numpy==1.26.4`, and the candidate audit found only one compatible baseline set.
- Decision: Keep the root project environment unchanged and create an isolated Intel macOS CPU
  Phase 3 baseline environment. The direct versions are `numpy==1.26.4`, `torch==2.2.2`,
  `torchvision==0.17.2`, `monai==1.4.0`, `nnunetv2[intel_macos]==2.8.1`, `mlflow==3.14.0`,
  `simpleitk==2.5.5`, `scikit-image==0.26.0`, `scipy==1.17.1`, and `nibabel==5.4.2`, plus the
  local editable root project. Candidate A was the only resolved candidate. `monai==1.5.2` and
  `monai==1.6.0` are rejected because they require newer torch versions. `torchvision==0.17.2` is
  explicitly pinned to the official `torch==2.2.2` pairing. MLflow is declared inside the baseline
  environment. A possible `acvl-utils` source build during a later sync is an acknowledged
  installation risk. No installation or runtime validation is claimed yet. Later Linux and CUDA
  support will use a sibling environment rather than changing scientific code.
- Consequences: Phase 3 dependency resolution, sync, import verification, and CPU smoke execution
  can proceed within the isolated baseline environment without changing the root `pyproject.toml`
  or root `uv.lock`. The Gate 3 dependency and environment checklist item remains incomplete until a
  later approved sync and runtime verification step passes.

### 2026-07-28: Use Shared External Run-Path and Provenance Contracts for Phase 3

- Status: accepted
- Context: Phase 3 baseline execution needs one deterministic contract for external run roots and
  one deterministic provenance artifact shared by `nnU-Net v2` and `MONAI SegResNet`. The contract
  must keep runtime output paths outside Git, avoid persisting machine-specific absolute paths, and
  record failure state without leaking medical identifiers or free-text exception content.
- Decision: Baseline output roots are runtime-only and external to Git. Absolute paths are never
  persisted in baseline provenance artifacts. Both approved baseline families share one versioned
  provenance contract. Callers must provide environment facts and package versions explicitly
  instead of collecting them implicitly. No medical identifiers or free-text exception messages are
  permitted in the persisted provenance record. Failed runs store machine-readable failure codes.
  Directory creation is explicit, no-overwrite, and separate from validation.
- Consequences: Phase 3 baseline preparation, training, inference, and evaluation code must route
  checkpoints, predictions, metrics, MLflow state, logs, and temporary files through validated
  external run roots. Persisted provenance remains deterministic, hash-verified, and portable
  across machines because it excludes absolute paths and runtime-discovered free text.

### 2026-07-28: Fix the Shared Phase 3 Binary-Tumor Metric Definitions

- Status: accepted
- Context: Phase 3 needs one deterministic project-owned metric implementation shared by
  `nnU-Net v2` and `MONAI SegResNet`, with explicit empty-mask behavior, stable lesion matching,
  and machine-readable artifacts that never persist NaN values or model-framework-specific state.
- Decision: The exact empty-mask policy is fixed as follows: both-empty cases use Dice `1.0`, IoU
  `1.0`, HD95 `0.0`, NSD `1.0`, lesion recall `null`, lesion precision `null`, lesion F1 `1.0`,
  zero false-positive lesions, zero volume errors, and relative volume error `null`; ground-truth
  nonempty with empty prediction uses Dice `0.0`, IoU `0.0`, HD95 `null`, NSD `0.0`, lesion recall
  `0.0`, lesion precision `null`, and lesion F1 `0.0`; ground-truth empty with nonempty prediction
  uses Dice `0.0`, IoU `0.0`, HD95 `null`, NSD `0.0`, lesion recall `null`, lesion precision
  `0.0`, and lesion F1 `0.0`. Foreground surfaces use a deterministic 3D face-connectivity
  surface definition. HD95 is the 95th percentile of concatenated bidirectional surface distances
  in physical millimetres. NSD uses the explicit tolerance supplied to the metric call and counts
  bidirectional surface samples whose distance is less than or equal to that tolerance.
  Twenty-six-connected components define lesions. Lesion matches use deterministic maximum-overlap
  one-to-one bipartite assignment and require at least one shared voxel. Signed volume error is
  `predicted_volume_ml - ground_truth_volume_ml`. Undefined values are persisted as JSON `null`,
  never NaN. Both approved baselines use the same project metric implementation, and metric values
  are computed by project code rather than an LLM.
- Consequences: Phase 3 report artifacts can aggregate case metrics deterministically across both
  baseline families without importing training frameworks or relying on external metric services.
  Split-lesion, merged-lesion, empty-mask, and undefined-value cases now have one shared
  repository definition that later baseline wrappers must preserve exactly.

### 2026-07-28: Use One Fixed Synthetic Phase 3 Fixture Dataset

- Status: accepted
- Context: Phase 3 baseline preparation and metric-path testing require one deterministic,
  framework-independent 3D fixture dataset that exercises empty-mask, multi-lesion, boundary, and
  irregular-label cases without touching LiTS, MSD Task03 Liver, or 3D-IRCADb.
- Decision: Phase 3 uses one fixed synthetic dataset shared by both baselines. The dataset ID
  `901` is synthetic-only. The fixture contains five training cases and two test cases. The fixture
  is not medical data. Images are deterministic coordinate-generated `int16`, and labels are
  deterministic binary `uint8`. Spacing and affine are explicit. Compressed NIfTI bytes and the
  fixture manifest are deterministic. `labelsTs` is project evaluation ground truth and is not an
  nnU-Net training input. Generated fixtures remain outside Git. Real data remains disconnected.
- Consequences: Both baseline families can consume the same external synthetic dataset contract for
  smoke paths, overfit-path scaffolding, and metric validation without any random seed handling,
  path discovery, or real-data access. Repeated generation under different temporary roots remains
  byte-identical at the relative-file level.

### 2026-07-28: Keep nnU-Net v2 Wrapping Project-Owned and Runtime-External

- Status: accepted
- Context: Phase 3 needs an `nnU-Net v2` wrapper that matches the installed `nnunetv2==2.8.1`
  command-line interface exactly while preserving the repository boundary against real execution,
  shell injection, absolute-path persistence, and framework-owned metric logic.
- Decision: `nnU-Net v2` is locked to version `2.8.1` in the isolated baseline environment. CLI
  option spellings are derived from the locally installed `--help` output. Command execution never
  uses a shell. Runtime `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` paths remain
  external and are never persisted. Prediction import is baseline-neutral and owned by project
  code. Saved prediction files, not console text, feed the metric engine. Prediction geometry and
  binary values are validated without repair. Subprocess failures use machine-readable failure
  codes. No `nnU-Net` execution has occurred in this implementation step.
- Consequences: Phase 3 can inspect, validate, and later execute `nnU-Net v2` through deterministic
  command tuples, sanitized runtime environments, typed failure handling, and project-owned saved-
  prediction evaluation. Synthetic and tiny-data paths can be exercised later without changing the
  scientific metric implementation or persisting machine-specific runtime paths.

### 2026-07-28: Use CPU-Only Synthetic Gate 3 Baselines Before Final Evidence

- Status: accepted
- Context: Phase 3 needs a bounded reproducible software-path execution for both approved baseline
  families before any final committed Gate 3 evidence run. The implementation must remain on Intel
  macOS CPU, must not use AMP, and must keep all temporary artifacts outside Git.
- Decision: Use one fixed synthetic fixture for both baselines, run `MONAI SegResNet` as a small
  deterministic CPU-only `monai.networks.nets.SegResNet` with fixed CT clipping and scaling,
  `Adam`, cross-entropy loss, 16 bounded training steps, no dropout, `sliding_window_inference`
  with ROI `(24, 24, 16)` and overlap `0.0`, and deterministic seed `1729`. Use real nnU-Net v2
  planning and preprocessing first, then use the actual nnU-Net trainer/plans machinery to build
  the planned network and run a bounded project-owned CPU tiny overfit loop with no fallback. Save
  checkpoints externally with no overwrite, validate resume metadata explicitly, and log only
  non-sensitive MLflow metadata beneath the approved external run root. The temporary pre-commit
  self-test is engineering evidence only and is not final Gate 3 evidence; final Gate 3 evidence
  must be rerun after the implementation is committed.
- Consequences: Phase 3 can verify deterministic synthetic preparation, training-path loss
  reduction, saved-prediction import, metric JSON generation, and metadata logging for both
  approved baseline families without claiming scientific efficacy, full-cohort training,
  generalization, or clinical validity.

### 2026-08-02: Treat learned_positive_step as Explicit Parameterized Positive-Step Input

- Status: accepted
- Context: Phase 6 implements an optional positive-step schedule path, but Phase 6 does not include
  training of those step parameters, optimizer-driven fitting, or any claim that the schedule was
  learned from data.
- Decision: In Phase 6 close-out, `learned_positive_step` is documented scientifically as an
  explicit parameterized positive-step schedule whose raw step parameters are supplied by config or
  caller input. The implementation must not describe those parameters as trained or learned from
  data unless a later phase adds real schedule training evidence.
- Consequences: Gate 6 documentation, publication Markdown, ablation comparison, and run summaries
  state that positive-step parameters are supplied, not trained. Phase 6 can execute the optional
  parameterized schedule path without claiming schedule learning or Phase 7 functionality.

### 2026-08-08: Approve Definitive SegResNet Architecture and Require Memory-Safe Training Input

- Status: accepted
- Context: `PHASE8-DEFINITIVE-ARCHITECTURE-MEMORY-POLICY-V1`. The Phase 8 definitive-training
  pipeline (`src/protoem_ct/external/definitive_pipeline.py`) previously hard-coded MONAI SegResNet
  architecture kwargs (`spatial_dims=3, in_channels=1, out_channels=2, init_filters=8,
  blocks_down=(1,1,1), blocks_up=(1,1), dropout_prob=None, upsample_mode="deconv"`) as an
  undocumented engineering-reference value inherited from the bounded, non-scientific Substage 4B
  pilot, not as an approved scientific decision. Separately, `run_definitive_training` required the
  caller to supply a `Sequence[Phase8DefinitiveTrainCase]` holding complete preprocessed image/label
  NumPy arrays for every case, which would force all 91 development-TRAIN volumes to coexist in RAM
  for a real run.
- Decision: The exact SegResNet architecture values above are now explicitly approved independently
  of the pilot and are hash-bound fields on `Phase8DefinitiveConfig`, fail-closed validated in
  `__post_init__`, and read by `_build_definitive_segresnet_model` from the config rather than from
  hard-coded literals. This approval does not make the Substage 4B pilot scientific evidence; no
  architecture comparison or hyperparameter search was performed. A memory-safe training-input path,
  `run_definitive_training_from_references`, was added: callers supply lightweight
  `Phase8DefinitiveTrainCaseReference` objects (anonymous `case_id` only, no arrays) plus a
  caller-supplied per-case loader callable. Preprocessing is sequential and on-demand: at most two
  full preprocessed cases (one if the deterministically-selected positive and negative case IDs
  coincide, loaded once) are live in memory per optimizer step, and no persistent full-volume cache
  exists across steps. Case selection remains deterministic under seed `1729`, and the locked
  foreground-biased 1:1 positive:negative sampling policy is unchanged, reused via the existing
  `sample_foreground_aware_patches` function through one shared internal step helper used by both the
  existing small in-memory API and the new reference-based path. No definitive training was executed
  under this decision.
- Consequences: A future real 91-case definitive training run can use
  `run_definitive_training_from_references` without materializing all 91 preprocessed volumes in RAM
  simultaneously. Any future change to the locked architecture values requires a new, separately
  approved design identifier. This decision does not select a checkpoint, freeze preprocessing, or
  release Wave 4/5.

### 2026-08-08: Approve Definitive Training Orchestration Policy (Implementation Only)

- Status: accepted
- Context: `PHASE8-DEFINITIVE-TRAINING-POLICY-V1`. Package A (`definitive_pipeline.py`) implemented
  locked preprocessing, patch sampling, a single training-step helper, and single-case validation,
  but had no policy for how many optimizer steps a real definitive run uses, which checkpoints are
  compared, how full-cohort validation is enforced, or how a checkpoint is selected. This substage
  implements and synthetically tests that orchestration without executing any real definitive
  training.
- Decision: `Phase8DefinitiveTrainingExecutionPolicy` is a new immutable, hash-bound, LOCKED
  dataclass (single constructor `build_definitive_training_execution_policy_v1()`) fixing:
  `total_optimizer_steps=500`; `candidate_checkpoint_steps=(250, 500)`; `early_stopping=False`;
  `resume_policy="none"`; `maximum_wall_clock_seconds=36000.0`; `validation_expected_case_count=20`;
  `full_validation_required_for_every_candidate=True`;
  `subset_validation_for_checkpoint_selection=False`; `checkpoint_selection_metric="mean_tumor_dice"`;
  `tie_break_policy="earliest_checkpoint_step"`; `internal_test_used_for_selection=False`;
  `external_data_used_for_selection=False`; `external_labels_used_for_selection=False`. Any
  deviation fails closed in `__post_init__`.
  `run_definitive_continuous_training_with_snapshots_and_watchdog` trains the model and optimizer
  as one continuous 500-step trajectory (single seed/construction call, no reinitialization at step
  250), taking a deep-copied `state_dict` snapshot only after steps 250 and 500 complete, and is
  wrapped in a `multiprocessing` spawn/Pipe process-level watchdog (mirroring
  `definitive_training_pilot.py`'s `run_phase8_bounded_pilot_with_watchdog`) that fails closed with
  `Phase8DefinitiveTrainingWatchdogTimeoutError` on timeout with no partial checkpoint selection and
  no automatic resume. `run_definitive_full_validation` requires exactly 20 unique validation case
  IDs (fails closed on 19, 21, or any duplicate), loads and evaluates cases one at a time via a
  caller-supplied loader with no persistent full-volume cache, and computes `mean_tumor_dice` only
  from finite per-case tumor Dice values. `select_definitive_checkpoint` requires both candidates to
  share the identical 20-case set, rejects any non-finite Dice or step outside `{250, 500}`, selects
  the higher `mean_tumor_dice`, and on an exact tie selects step 250; its signature has no loss or
  IoU input, so neither can influence selection. `build_definitive_checkpoint_selection_evidence`
  produces a self-hashed `Phase8DefinitiveCheckpointSelectionEvidence` recording both candidates'
  checkpoint hashes, both mean-Dice values, the selected step/hash, the selection metric, the
  tie-break policy, and hard-locked `internal_test_used=False` / `external_data_used=False` /
  `external_labels_used=False` flags -- this is selection evidence, not a Phase 8 freeze artifact.
  Checkpoint publication reuses the existing `Phase8CheckpointMetadata` schema unchanged (via the
  existing `build_definitive_checkpoint_metadata`, still hard-locked to `freeze_eligible=False`) and
  rejects overwrite. No real dataset, `/Volumes` path, or prior real checkpoint hash appears in the
  new code; a static-scan unit test enforces this. No definitive training was executed and no real
  checkpoint was selected under this decision.
- Consequences: A future approved real definitive-training session can call this orchestration
  end-to-end against the real 91-case development TRAIN partition and the 20-case development
  VALIDATION partition to produce genuine selection evidence, but this decision by itself does not
  authorize that execution, does not freeze preprocessing/support/threshold/checkpoint decisions,
  and does not release Wave 4 or Wave 5.

### 2026-08-08: Materialize Definitive Training Patches Sequentially (Implementation Only)

- Status: accepted
- Context: `P8-DEFINITIVE-PATCH-MATERIALIZATION-V1`. Package A/B's memory-safe
  `run_definitive_training_from_references` and
  `run_definitive_continuous_training_with_snapshots` correctly avoid holding all 91 development
  TRAIN volumes in RAM at once, but a real filesystem loader performing load, RAS reorientation,
  resampling, normalization, and label conversion on every loader invocation would repreprocess a
  full CT volume on every one of the 500 optimizer steps a real run needs. Package B measured
  preprocessing of two real cases at approximately 32.92 seconds total against approximately 23.81
  seconds for two optimizer steps, so repeated full-volume preprocessing inside all 500 steps was
  judged operationally unacceptable against the approved wall-clock budget.
- Decision: The approved memory policy
  (`preprocess_cases_sequentially_and_materialize_only_required_training_patches`) is now
  implemented as a two-stage mechanism in `definitive_pipeline.py`, with no scientific policy
  change of any kind. `build_definitive_patch_request_schedule` generates a deterministic,
  self-hashed `Phase8DefinitivePatchSchedule` of exactly `policy.total_optimizer_steps` (500)
  entries, each fixing one positive and one negative anonymous case ID for one optimizer step, in
  exact step order, before any medical volume is loaded. Positive-case rotation is unchanged
  (`positive_refs[step_index % len(positive_refs)]`); negative-case selection uses the same
  `np.random.default_rng(seed).integers(0, len(case_references))` algorithm, but now on a
  dedicated, single-purpose RNG stream used only for case selection -- intentionally decoupled from
  the separate patch-voxel-position RNG used during materialization, because case selection must be
  resolvable before any real image/label array exists and therefore cannot share a stream with
  sampling draws whose count depends on real volume content. `case_references` order is preserved
  exactly as supplied, with no internal sorting or renumbering. `materialize_definitive_training_patches`
  then processes cases strictly sequentially in that same order: for each case the schedule actually
  requires, the caller-supplied loader is invoked exactly once, every patch any schedule entry needs
  from that case is sampled (via the existing, unchanged `sample_foreground_aware_patches`, using a
  deterministic per-`(seed, step_index, role)` RNG independent of case content) and published, and
  the loaded case is discarded before the next case is loaded -- at most one full preprocessed case
  is live at any point, and no persistent full-volume cache exists across cases or across separate
  materialization runs. Published patch artifacts are anonymous NPZ arrays plus self-hashed JSON
  metadata (step index, role, anonymous case ID, shape/dtype, deterministic SHA-256 content hash,
  the definitive config hash, and the schedule hash; no raw patient identifier and no filesystem
  source path), written beneath an explicit external output root with no-overwrite publication
  reusing the same hardlink-then-rename helper as checkpoint-snapshot publication.
  `run_definitive_training_from_materialized_patches` trains from these pre-materialized patches in
  exact step order, initializing the model, optimizer, and seed exactly once for the whole run and
  reusing the existing shared fail-closed forward/backward step helper (split into a
  patches-consuming core shared by both the in-memory and materialized-patch paths, so there is no
  second scientific training algorithm); checkpoint snapshots remain positioned at exactly steps 250
  and 500. No architecture, preprocessing, sampling ratio, optimizer, loss, device, AMP, seed, step
  budget, checkpoint-candidate, or selection policy value changed. No definitive training was
  executed and no real patch, checkpoint, or dataset access occurred under this decision.
- Consequences: A future approved real definitive-training run can build the 500-step schedule,
  materialize its patches from the real 91-case development TRAIN partition one case at a time, and
  train from those patches without ever repeating full-volume preprocessing inside the optimizer
  loop. This decision does not select a checkpoint, freeze preprocessing, or release Wave 4/5, and
  does not by itself authorize real definitive training execution.

### 2026-08-08: Lock Definitive Sampling RNG Streams (Implementation Only)

- Status: accepted
- Context: `P8-DEFINITIVE-SAMPLING-RNG-POLICY-V1`. The patch-request schedule necessarily separates
  case-selection RNG from within-case voxel/patch-sampling RNG (case selection must be resolvable
  before any medical volume loads), whereas the earlier in-memory training paths
  (`run_definitive_training_from_references`, `run_definitive_continuous_training_with_snapshots`)
  used one shared RNG stream for both. This is a decoupling of RNG streams, not a change to the
  approved sampling distribution, but it needed an explicit, hash-bound reproducibility contract
  instead of an implicit one, and the documentation needed to explicitly disclaim any false
  bit-for-bit equivalence to the old shared-stream realization.
- Decision: The RNG-stream policy is now explicit and hash-bound. `base_seed = config.seed`
  (1729). `case_selection_rng`: one `np.random.default_rng(base_seed)` stream used only for
  negative-case-reference selection, one `integers(0, len(case_references))` draw per optimizer
  step, over the approved ordered `case_references` sequence. `positive_case_selection`: no RNG --
  `positive_refs[step_index % len(positive_refs)]`, unchanged. `patch_sampling_rng`: one
  `np.random.default_rng([base_seed, step_index, 0 if role == "positive" else 1])` stream per patch
  request, used only for that request's voxel/patch sampling; retries for an all-background
  negative patch consume only that request's own stream and cannot alter any other request's
  stream or any case selection, because each request constructs a fresh, independent generator.
  `Phase8DefinitivePatchSchedule` now carries `rng_policy_identifier` /
  `rng_policy_version` (`PHASE8_DEFINITIVE_SAMPLING_RNG_POLICY_IDENTIFIER` /
  `_VERSION`, both `"P8-DEFINITIVE-SAMPLING-RNG-POLICY-V1"` / `"v1"`), fail-closed validated in
  `__post_init__` and included in the schedule's self-hash payload, so a schedule built under a
  different RNG-policy identifier/version has a different identity or fails validation. The module
  docstring and the `build_definitive_patch_request_schedule` docstring now explicitly state that
  the old shared-stream behavior was used only before definitive patch materialization existed,
  remains exactly as implemented there, and is not claimed to be bit-for-bit reproduced by the new
  decoupled streams -- only the approved sampling distribution (seed 1729, 1:1 ratio, positive
  rotation, deterministic-uniform negative draws over the approved case-reference sequence, patch
  size `[64, 64, 32]`, foreground/background semantics) is preserved. No model, preprocessing, or
  training hyperparameter changed; 500 steps, checkpoints 250/500, 20-case full validation, mean
  tumor Dice selection, and the 0.5 threshold are all unchanged. No real definitive training had
  occurred before this policy was fixed, so no scientific result is changed or invalidated. No real
  data was accessed and no definitive training was executed under this decision.
- Consequences: Future real definitive-training execution has an unambiguous, hash-verifiable RNG
  contract for the decoupled schedule/materialization design, and any future attempt to change the
  RNG derivation rules must introduce a new, separately approved RNG-policy identifier/version
  rather than silently reinterpreting the existing one.

### 2026-08-08: Record Definitive Train Spacing Provenance (PHASE8-DEFINITIVE-TRAIN-SPACING-V1)

- Status: accepted
- Context: `Phase8DefinitiveConfig` deliberately carries no spacing field -- the locked design leaves
  target spacing as a runtime derivation (`compute_median_train_spacing`) rather than a hard-coded
  constant, and `docs/phase8/DEFINITIVE_CONFIG_AUDIT.md` recorded voxel-spacing/resampling policy as
  an open, unapproved question at the time of that audit. The real derivation was subsequently
  performed by the Package B engineering-verification run
  (`/Volumes/Lexar/ProtoEM-CT/runs/phase8_package_b_engineering_verification_v1/`, already referenced
  in the `Materialize Definitive Training Patches Sequentially` decision above via its measured
  preprocessing/training timings) and explicitly approved for definitive development training with
  `train_target_spacing: [0.767578125, 0.767578125, 1.0]`. This entry records that
  already-approved provenance so it is
  discoverable from `docs/DECISIONS.md`; it does not perform, repeat, or authorize any new spacing
  computation.
- Decision: The approved definitive-training target spacing is `[0.767578125, 0.767578125, 1.0]`
  mm, componentwise median voxel spacing over all 91 DEVELOPMENT TRAIN cases only, computed after
  RAS reorientation. Source provenance: development manifest hash
  `c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b`
  (`/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/manifest.json`) and development split hash
  `936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb`
  (`/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/split.json`), both verified against the
  Package B engineering report
  (`phase8_package_b_engineering_verification_v1/phase8_package_b_engineering_report.json`), whose
  `derived_median_spacing` field records exactly `[0.767578125, 0.767578125, 1.0]` and whose
  `train_case_count_for_spacing` field records exactly `91`. The Package-B preprocessing/config
  identity containing the derived spacing is
  `286b4a200c147718cfe885850e670254a52e8e1ec5d9683d78fd36af4f293d37` (that report's
  `preprocessing_config_hash` field). Provenance boundary: TRAIN cases only; no development-
  validation spacing was used in the derivation; no internal-test volume was opened
  (`internal_test_case_count_present_but_excluded: 20` in the same report); no external data or
  external labels were used. Status: approved for definitive development preprocessing.
- Consequences: A real Package C definitive-training execution may use this target spacing without
  re-deriving it, citing this decision and the underlying Package B engineering report as
  provenance. This decision does not itself authorize training execution, does not change
  `Phase8DefinitiveConfig` (spacing remains a runtime-supplied value, not a config field), and does
  not modify any prior decision's content -- including the historical open-question language
  preserved as-written in `docs/phase8/DEFINITIVE_CONFIG_AUDIT.md`.

### 2026-08-09: Record Package C Execution Provenance

- Context: Package C (real Phase 8 definitive development training) executed successfully. Every
  boundary condition in `PHASE8-REAL-DEFINITIVE-DEVELOPMENT-TRAINING-V1` was independently verified
  before and after the run, including by a fresh independent read-only reviewer with no prior
  context on the run, who returned an overall **PASS** across all checked items (train-pool count,
  internal-test exclusion, approved spacing, RNG policy, single continuous trajectory, checkpoint
  identities, identical 20-case validation set for both candidates, correct Dice-based selection
  arithmetic, no external-data access, no post-result policy change, and `freeze_eligible=false`).
  This entry records that outcome's provenance transparently, including an intentional wrinkle in
  how the execution code reached Git: it does not perform, repeat, or authorize any rerun of
  training, validation, or metric computation, and it does not open any medical image, label, or
  materialized-patch array.
- Decision: Package C definitive development training is recorded as **complete**. The run executed
  with the real-data driver in an intentionally uncommitted working-tree state on top of execution
  base HEAD `f1d4e1be7b6f7aa3657792cae9125c825affdded`. The exact execution-code files, and their
  pre-commit SHA-256 hashes (computed from the working tree exactly as it stood when the run
  executed, before any commit was made), were:
  - `src/protoem_ct/cli/main.py`:
    `0a77c1d857f4ce015d8263bfc5a4168e3d167360263c0bdb73dd59751b2c1c2a`
  - `src/protoem_ct/external/definitive_real_training_driver.py`:
    `98de618b9e3aae7df7579a12c3f9b54f9fb6eeaf25310c5cc7dc40fcbcc30f1e`
  - `tests/unit/test_phase8_definitive_real_driver.py`:
    `895162a46b524e22345551914f2d8449a418dc23c3183264c086ca9f995fd910`

  These three exact file contents were subsequently captured, unmodified, in a **post-run
  code-capture commit** `f264ce79dd68039256d140f03c468baed8ecea07`
  (`feat(phase8): add definitive real training driver`), created strictly to preserve the exact
  bytes that produced the run. All three post-commit file hashes were reverified to be byte-identical
  to their pre-commit hashes above before this entry was written. **This capture commit must not be
  misrepresented as the Git HEAD the run started from.** The run's execution base HEAD remains
  `f1d4e1be7b6f7aa3657792cae9125c825affdded`; the capture commit is a later, separate commit that
  exists solely to make the already-executed code reviewable and citable in Git history. Where
  existing Package C checkpoint metadata (`phase8_definitive_checkpoint_metadata_step_{250,500}.json`)
  records `originating_git_commit: f1d4e1be1...`, that field is correct and unchanged: it documents
  the execution base HEAD, not the later code-capture commit, and no artifact was rewritten to
  obscure this distinction.

  Run output root: `/Volumes/Lexar/ProtoEM-CT/runs/phase8_definitive_training_v1`. Selected
  checkpoint: step `500`, SHA-256 `2d7989fd134b1348e82cc52afbcf4738c0ce3c17e9c68df774431f577dede651`
  (re-verified byte-identical against the artifact on disk as part of this closure). Selection
  evidence hash `81be6e4f7e6a56d21f07bc101b21d19ac18b0580c74548e9490bf295382c8bc8` (re-verified).
  Step-250 mean tumor Dice `0.0016380082094079678`; step-500 mean tumor Dice
  `0.01579295321113191`. Step 500 was selected because its complete 20-case development-validation
  mean tumor Dice was higher than step 250's, per the locked `mean_tumor_dice` selection metric with
  earliest-step tie-break (not a tie here). `internal_test_used=false`, `external_data_used=false`,
  `external_labels_used=false`; no hyperparameter, preprocessing, or threshold tuning occurred after
  results were known; the selected checkpoint remains `freeze_eligible=false`. The scientific run
  itself is not being repeated, rerun, or re-derived by this entry.
- Consequences: Package C's execution code is now committed and reviewable, closing the provenance
  gap between "what ran" and "what is in Git," without disturbing the already-completed scientific
  result. This entry does not begin Freeze, does not access 3D-IRCADb-01, and does not change
  `freeze_eligible` on any checkpoint. Phase 8 Freeze remains the next, separately authorized stage.

### 2026-08-09: Publish Package D Phase 8 Definitive Freeze

- Status: accepted
- Authorization: the project owner explicitly authorized execution of "PACKAGE D — PHASE 8
  DEFINITIVE FREEZE" against the completed, reviewed Package C result recorded in the immediately
  preceding entry above. This is the explicit, separate authorization that entry stated Freeze
  required before proceeding and is recorded here for traceability, consistent with this document's
  convention of documenting authorization after the corresponding real execution completes.
- Context: `Phase8CheckpointMetadata.freeze_eligible` on Package C's published checkpoint metadata
  (`phase8_definitive_checkpoint_metadata_step_{250,500}.json`) is `false` for both candidates. This
  is not a judgment that either checkpoint is scientifically deficient or ineligible to be frozen --
  it is a hard, unconditional structural property of the publishing function both Package A and
  Package C reuse unchanged (`build_definitive_checkpoint_metadata` in `definitive_pipeline.py`,
  lines 1382-1413): that function fails closed if a caller ever passes `freeze_eligible=True`,
  because "Package A is implementation-only" and was designed to never self-declare freeze
  eligibility. The authoritative freeze decision was always intended to live in a separate artifact,
  produced by a later, dedicated freeze step -- this entry and the artifact it records are that step.
  Consistent with the Package C task's own explicit instruction ("Do NOT rewrite historical
  checkpoint metadata merely to make `freeze_eligible=true`"), neither Package C checkpoint metadata
  file was modified by this work; both remain exactly as originally published, `freeze_eligible=false`
  included.
- Decision: A new, minimal wiring module `src/protoem_ct/external/definitive_freeze.py` (committed
  `3b2c5db60c8ae261d9bf8b7f9863720c1437afd6`, reviewed by one independent read-only reviewer, PASS)
  reuses the pre-existing, unmodified `Phase8DecisionFreeze`/`FrozenDecisionReference` schema
  (`artifacts.py`) and `Phase8SupportPolicy` schema (`internal_evidence.py`) to assemble and publish
  the authoritative freeze record, without touching Package C's checkpoint metadata. Support policy:
  `policy_type="no_support"`, reusing the already-committed, already-locked
  `REQUIRED_SUPPORT_POLICY_TYPE` constant from `definitive_training.py` (Substage 4A, commit
  `809edeb`, predating this work) -- confirmed the only support-policy type ever concretely
  constructed anywhere in this repository, bound to Package C's real
  `selected_candidate_id="phase8_definitive_segresnet"` and
  `selected_checkpoint_hash=2d7989fd134b1348e82cc52afbcf4738c0ce3c17e9c68df774431f577dede651`. The
  published `Phase8DecisionFreeze` inventory (`freeze_state="frozen"`,
  `frozen_before_external_evaluation=true`) contains exactly the nine required categories, each bound
  by hash to already-real Package C/Wave 3/Wave 4 evidence: `preprocessing_decision` and
  `threshold_decision` both cite the real definitive config hash
  `4e0075e0d499080c4065031d42ea7ff25637cafe154b6757c05edbc14e8c0624` (which encodes RAS orientation,
  trilinear/nearest interpolation, HU clip `[-1000,1000]`, intensity scale `[-1,1]`, tumor raw label
  `2`, and the fixed inference threshold `0.5`, unchanged); `preprocessing_decision` additionally
  cites spacing provenance `286b4a200c147718cfe885850e670254a52e8e1ec5d9683d78fd36af4f293d37`
  (`PHASE8-DEFINITIVE-TRAIN-SPACING-V1`); `model_selection_decision` cites the real selection
  evidence hash `81be6e4f7e6a56d21f07bc101b21d19ac18b0580c74548e9490bf295382c8bc8`;
  `checkpoint_metadata` cites the real step-500 checkpoint-metadata hash
  `acb46ab92d7dd5f3ce809fa21a47a1d836596d08f80676c32358321a4c50c134` and the checkpoint SHA-256;
  `label_mapping_policy` cites the existing Wave 3 hash
  `3fa990dcaf8922e31d9ceb8fe99ad87da53301c08469ec1a77729eef24143f3d`; `metric_configuration`,
  `bootstrap_configuration`, and `publication_configuration` cite the existing Wave 4 statistical
  -policy component hashes computed by `phase8_statistical_policy_component_hash(...)`, unchanged.
  Published, reject-on-overwrite, to
  `/Volumes/Lexar/ProtoEM-CT/runs/phase8_definitive_freeze_v1/phase8_decision_freeze.json` and
  `phase8_definitive_support_policy.json`. No medical image/label file, no `/Volumes/Lexar/ProtoEM-CT/
  datasets/external` path, and no internal-test artifact was accessed by this work. No training,
  validation, or metric was rerun or recomputed.
- Consequences: Phase 8 now has a single, self-validating, hash-bound freeze inventory citing every
  required decision category by real evidence hash, without ever rewriting Package C's own
  checkpoint metadata. This entry does not perform external-evaluation preregistration, does not
  access 3D-IRCADb-01, and does not perform external inference -- preregistration remains the next,
  separately authorized Phase 8 stage.

### 2026-08-09: Publish Package E Phase 8 External-Evaluation Preregistration

- Status: accepted
- Authorization: the project owner explicitly authorized execution of "PACKAGE E — EXTERNAL
  EVALUATION PREREGISTRATION" against the completed, reviewed Package D freeze recorded in the
  immediately preceding entry above, with an explicit, repeated instruction that no 3D-IRCADb-01
  image, label, or directory listing may be accessed at this stage.
- Context: `Phase8ExternalPreregistration` (`preregistration.py`) and its sibling contracts
  (`label_mapping.py`, `eligibility.py`, `domain_shift.py`, `statistical_policy.py`) were already
  fully implemented and unit-tested from earlier Phase 8 work, but no wiring existed to assemble a
  concrete preregistration instance from the real, now-frozen Package D decision inventory, or to
  publish one. That wiring was the one genuinely missing piece for this step.
- Decision: A new, minimal wiring module
  `src/protoem_ct/external/preregistration_publication.py` (reviewed by one independent read-only
  reviewer; two minor follow-up hardening fixes applied after review -- cross-checking the supplied
  support-policy artifact's content against the freeze's own recorded `support_policy` hash, and
  reusing `preregistration.py`'s own `PHASE8_ARTIFACT_REFERENCE_SCHEMA_NAME`/`_VERSION` constants
  instead of duplicating them -- both verified not to change the published preregistration's
  identity hash) builds `Phase8ExternalPreregistration` entirely from already-committed authority: it
  reads the two Package D artifacts named in this task (`phase8_decision_freeze.json`,
  `phase8_definitive_support_policy.json`) only after verifying each file's SHA-256 against the
  caller-supplied expected value (fail-closed on mismatch, symlink, non-absolute path, or
  non-regular-file), then cross-checks that `build_default_phase8_label_mapping_policy()` and the
  `bootstrap_configuration`/`metric_configuration`/`publication_configuration` components of
  `build_phase8_statistical_policy_bundle()` still hash to exactly what the freeze inventory already
  recorded for those categories, failing closed on any drift rather than trusting local
  recomputation. `frozen_decision_inventory_hash` is the freeze's own
  `freeze_inventory_hash=efca32a42d435db05b171ae5f064d38c36eeaf1352e6c86df76fa1a8d94ac145`. Because
  the anonymous external image manifest and domain-shift record do not exist yet -- no external data
  has been accessed -- `anonymous_manifest_reference` and `domain_shift_record_reference` are left
  `reference_state="unresolved"` (`artifact_hash=None`) and `lifecycle_state="draft"` is the only
  value the schema permits in that state (`evaluation_ready` requires every reference resolved). All
  nine `PHASE8_SEGMENTATION_METRICS` (`tumor_dice`, `tumor_iou`, `tumor_hd95`,
  `tumor_normalized_surface_dice`, `lesion_wise_recall`, `lesion_wise_precision`, `lesion_f1`,
  `false_positive_lesions_per_scan`, `tumor_volume_error`) are preregistered as descriptive metrics
  with no new primary/confirmatory hierarchy introduced, per the existing statistical-policy
  contract's own framing (`comparison_configuration.claims_policy="descriptive_only..."`). Bootstrap
  policy (`case_patient` resampling unit, seed `1729`, `10000` resamples, `95%` percentile CI),
  qualitative-montage selection (all eligible cases if <=20, else hash-ranked with the same seed), and
  eligibility fail-closed rules (`build_phase8_eligibility_policy()`) are all bound by reference to
  their existing, unmodified repository contracts -- none were redefined. `no_tuning_declaration`,
  `external_label_access_state="unavailable_before_prediction_lock"`, and
  `prediction_lock_requirement="required_before_label_access"` are hardcoded, not configurable.
  Published, reject-on-overwrite, to
  `/Volumes/Lexar/ProtoEM-CT/runs/phase8_external_preregistration_v1/phase8_external_preregistration.json`,
  self-validating hash `75d287ade45d2a778153885838d8635606ba5ae739ed88dbf0580671258db291`. The
  external cohort root `/Volumes/Lexar/ProtoEM-CT/datasets/external/3D-IRCADb-01` was recorded nowhere
  in code, hashed payload, or published artifact -- it was never stat'd, listed, opened, or otherwise
  observed by this work.
- Consequences: Phase 8 now has a committed, hash-bound, self-validating draft preregistration for
  the 3D-IRCADb-01 external evaluation, referencing only already-frozen Package D decisions and
  already-committed repository policy contracts, with no new scientific decision introduced. This
  entry does not access 3D-IRCADb-01, does not perform external inference, and does not compute any
  metric -- image-only external inference (with predictions locked/hashed before any external label
  is accessed) remains the next, separately authorized Phase 8 stage.

### 2026-08-09: Publish Package F Phase 8 External Image-Only Inference and Prediction Lock

- Status: accepted
- Authorization: the project owner explicitly authorized execution of "PACKAGE F -- EXTERNAL
  IMAGE-ONLY INFERENCE + PREDICTION LOCK" against the completed, reviewed Package E preregistration
  recorded in the immediately preceding entry above, with explicit image-only access approved and
  external label access explicitly not approved.
- Context: no real image-only inference driver existed yet -- `ircadb.py`, `image_qa.py`, and
  `manifest.py` covered image-only discovery/QA/manifest contracts, and `definitive_pipeline.py`
  covered frozen preprocessing/architecture/inference primitives, but nothing wired frozen-checkpoint
  inference to real external DICOM images or published a prediction lock.
- Decision: A new, minimal wiring module `src/protoem_ct/external/image_only_inference.py` (plus a
  small image-only DICOM volume+affine loader, `load_patient_dicom_zip_volume`, added to the existing
  `image_qa.py`) was implemented, synthetically tested (20 tests, including determinism, fail-closed
  identity/output-root/non-finite/incomplete-inventory checks, and a structural proof that garbage-byte
  `MASKS_DICOM.zip`/`LABELLED_DICOM.zip` sibling files are never opened), independently reviewed by one
  read-only reviewer (PASS), and committed (`9cde59f29986cd82c71e2ebb1429f6d5bc306da1`,
  `feat(phase8): add external image-only inference driver`) before any external image was opened for
  definitive inference. The driver reuses `definitive_pipeline.py`'s unmodified frozen preprocessing
  (RAS reorientation, trilinear resampling to `PHASE8-DEFINITIVE-TRAIN-SPACING-V1`
  `[0.767578125, 0.767578125, 1.0]` mm, HU clip `[-1000,1000]`, intensity scale `[-1,1]`), frozen
  SegResNet architecture, frozen sliding-window inference (ROI `[96,96,64]`, overlap `0.25`, batch `1`,
  CPU, AMP disabled), and frozen threshold `0.5` -- no scientific value was changed. It verifies the
  freeze artifact SHA-256, the preregistration's self-hash, the checkpoint SHA-256, and the definitive
  config hash before opening any external image, and never discovers, opens, or reads
  `MASKS_DICOM.zip`/`LABELLED_DICOM.zip`/`MESHES_VTK.zip`/`liver_*.jpg`.

  Run against dataset root `/Volumes/Lexar/ProtoEM-CT/datasets/external/3D-IRCADb-01/raw/3Dircadb1`
  once, definitively, using the isolated CPU torch/MONAI environment
  (`environments/phase3-baselines/intel-macos-cpu`), at Git HEAD
  `9cde59f29986cd82c71e2ebb1429f6d5bc306da1`. All 20 image-only cases (`ext-ircadb-001`..`020`)
  produced a finite, binary, uint8 prediction; none failed. Published, reject-on-overwrite, to
  `/Volumes/Lexar/ProtoEM-CT/runs/phase8_external_image_inference_v1`: one `.npy` prediction array and
  one self-hashed `_prediction_record.json` per case under `predictions/`, plus
  `phase8_external_prediction_lock.json` (`lock_hash`
  `190532a6abb7c3308de9abe7dc7455f4fafae464d51347ea2273874d13afdc94`) and a non-medical
  `phase8_external_image_only_inference_summary.json`. The lock binds `cohort_identifier`
  (`3d_ircadb_01`), the preregistration hash, freeze artifact SHA-256, checkpoint SHA-256, definitive
  config hash, frozen threshold, support policy (`no_support`), target spacing, the complete ordered
  20-case ID/prediction-hash inventory, `external_label_access=false`, `no_tuning=true`,
  `inference_completion_state="completed"`, and the executing Git commit. No label-dependent metric
  (Dice, IoU, HD95, NSD, lesion metrics) was computed anywhere in this work; only image-side technical
  diagnostics (finite output, prediction existence, shape/geometry consistency) were checked. One
  independent read-only reviewer re-verified all published artifacts against the frozen identities,
  recomputed every prediction file's SHA-256 against the lock, confirmed no label-path token appears in
  any published metadata, confirmed no prediction file was modified after lock publication, and
  returned PASS on all checklist items.
- Consequences: Phase 8 now has a complete, hash-locked, image-only external prediction inventory for
  all 20 3D-IRCADb-01 cases, published before any external label was ever accessed. This entry does not
  access 3D-IRCADb-01 labels, does not perform external evaluation, and does not compute any metric --
  external label access and evaluation against these already-locked predictions remain the next,
  separately authorized Phase 8 stage.

### 2026-08-09: Publish Package G Phase 8 External Label Evaluation

- Status: accepted
- Authorization: the project owner explicitly authorized execution of "PACKAGE G -- EXTERNAL LABEL
  EVALUATION" against the completed, reviewed Package F prediction lock recorded in the immediately
  preceding entry above, explicitly authorizing this module to open real 3D-IRCADb-01 tumor labels
  for the first time, strictly to evaluate the already-frozen, already-locked predictions -- never
  to regenerate, adjust, or influence them.
- Decision: A new module `src/protoem_ct/external/label_evaluation.py` was implemented and
  synthetically tested (17 tests in `tests/unit/test_phase8_label_evaluation.py`, covering
  prediction-lock/identity hash-mismatch fail-closed behavior, no mutation of locked predictions,
  missing-tumor-source and geometry-inconsistency fail-closed paths, nearest-neighbor resampling
  producing no fractional values, all 9 metrics coming from the existing
  `protoem_ct.baselines.metrics.compute_baseline_case_metrics` verbatim, bootstrap determinism/config,
  no performance-based exclusion, and output-root overwrite rejection), independently reviewed, and
  committed (`955dc78`, `feat(phase8): add external label evaluation driver (Package G wiring)`) before
  any real label was opened. It re-verifies the freeze artifact SHA-256, preregistration hash, and
  prediction lock (reconstructing `Phase8ExternalPredictionLock`/`Phase8ExternalPredictionRecord`
  directly so their own `__post_init__` hash checks apply), and re-recomputes every locked `.npy`
  file's SHA-256 against the lock at evaluation time (defense in depth). It opens only
  `MASKS_DICOM.zip` per case, and within it only role folders whose name case-insensitively starts with
  `livertumor` (never `liver`, `LABELLED_DICOM.zip`, `MESHES_VTK.zip`, or `liver_*.jpg`), aligns the
  union of those folders onto the locked prediction grid using only already-committed frozen primitives
  (`image_qa._build_lps_affine`, `image_only_inference._lps_affine_to_ras`,
  `definitive_pipeline.reorient_volume_to_ras`, `definitive_pipeline.resample_volume_to_spacing` with
  `is_label=True`, i.e. `scipy.ndimage.zoom(order=0)`), and applies
  `label_mapping.build_default_phase8_label_mapping_policy()` unmodified. Eligibility exclusions are
  driven only by `protoem_ct.external.eligibility`'s preregistered fail-closed reason codes (never by a
  case's metric values). No model/inference code (`torch`, `monai`, checkpoint loading) is imported.

  Run once, definitively, against dataset root
  `/Volumes/Lexar/ProtoEM-CT/datasets/external/3D-IRCADb-01/raw/3Dircadb1` and the Package F prediction
  lock, at Git HEAD `955dc78f95cfc85d19219dce59be95baf311e042` (Package G wiring commit). Of the 20
  cases, 5 had no
  `livertumor*` folder in `MASKS_DICOM.zip` at all (`ext-ircadb-005`, `ext-ircadb-007`,
  `ext-ircadb-011`, `ext-ircadb-014`, `ext-ircadb-020`) and were excluded fail-closed with
  `label_compatibility_status="incompatible"`, reason `tumor_target_absent` -- not for any metric
  reason. The remaining 15 cases were geometry-compatible and evaluated. Headline results (macro /
  pooled point estimates; case-level bootstrap 95% percentile CI, seed `1729`, `10000` resamples, over
  the 15 eligible cases): `tumor_dice` macro `0.014115733440605011` (CI `[0.004341547521318757,
  0.02501302650455788]`), pooled `0.021955736330271848`; `tumor_iou` macro `0.007215381213910843`
  (CI `[0.002208283278991875, 0.012793720568727924]`), pooled `0.011099719421616427`;
  `tumor_normalized_surface_dice` macro `0.0033014104480338863`
  (CI `[0.0008695378349168014, 0.0061626388043957]`); `tumor_hd95` macro `176.6259570403819` mm over
  15 defined cases (CI `[159.32400767790955, 194.301561440465]`); `lesion_wise_recall` macro
  `0.15886788048552752` (CI `[0.04815359477124183, 0.31025676937441643]`); `lesion_wise_precision`
  macro `0.0004170230872244496` (CI `[0.00019535650440340803, 0.0006562260559801865]`); `lesion_f1`
  macro `0.0008293785379380446` (CI `[0.00038875182666259906, 0.001304960931868939]`);
  `false_positive_lesions_per_scan` mean `2358.4` (CI `[1893.32, 2872.075]`); `tumor_volume_error`
  mean signed `-30.97904055175781` mL (CI `[-96.51345051441191, 18.267631285171493]`), mean absolute
  `73.1784307937622` mL (CI `[36.18556097774507, 126.89318989557269]`), macro relative
  `2.833886301727687` (CI `[0.7546956418899935, 5.484524303515011]`). This is consistent with (not
  contradicted by) the thin internal validation evidence
  (`phase8_definitive_training_v1/phase8_definitive_checkpoint_selection_evidence.json`:
  `mean_tumor_dice_step_500=0.01579295321113191`, single frozen scalar, no CI, `internal_test_used`
  and `external_data_used` both false) -- an honest, non-tuned, low-performing result on both cohorts,
  not a bug signature by itself.

  Published, reject-on-overwrite, to `/Volumes/Lexar/ProtoEM-CT/runs/phase8_external_evaluation_v1`:
  `phase8_external_eligibility_accounting.json`, `phase8_external_case_metrics.json` (plus one file per
  case under `case_metrics/`), `phase8_external_metric_report.json` (`artifact_hash`
  `a6dbdd3998e2c55d82e725cba20fbfebf685378f0e719547df5d008eeae0b2da`), `phase8_external_bootstrap_ci.json`,
  `phase8_external_domain_shift_record.json` (`domain_shift_record_hash`
  `04665b449d8b3d7838a28e3045cdd57b67c9fccf6e1f091b7f5f1abd5963da0b`, rebuilt fresh, image-only,
  `labels_accessed=false`, consistent with the pre-freeze Wave 3 artifact), a non-medical
  `phase8_external_qualitative_index.json` (all 15 eligible cases, anonymous IDs and hashes only, no
  pixel data), `phase8_internal_external_comparison.json`, and
  `phase8_external_evaluation_summary.json`.
- Independent review: one read-only self-review pass re-verified all 20 locked prediction files'
  SHA-256 against the lock (zero mismatches, zero byte/mtime changes), confirmed no `torch`/`monai`
  import or write-mode access to the predictions directory anywhere in `label_evaluation.py`, confirmed
  the label-mapping policy is used unmodified, confirmed the 5 exclusions match `tumor_target_absent`
  exactly and nothing else was excluded, confirmed the geometry chain uses only the listed reused frozen
  functions, confirmed exactly the 9 preregistered metrics appear (`tumor_volume_error` reported as
  signed/absolute/relative per policy) with no extra metric, confirmed the bootstrap configuration
  matches exactly, and confirmed the qualitative index includes literally all 15 eligible cases. It
  found one disclosed, non-scientific defect: `_build_internal_external_comparison` reads the internal
  checkpoint-selection evidence field under the wrong key (`checkpoint_sha256` instead of the evidence
  file's actual field `selected_checkpoint_hash`), so the published
  `phase8_internal_external_comparison.json`'s `internal_checkpoint_sha256` is `null` and
  `internal_checkpoint_matches_locked_checkpoint` is `false` instead of correctly resolving to
  `2d7989fd134b1348e82cc52afbcf4738c0ce3c17e9c68df774431f577dede651` / `true`. This affects only that
  one subsidiary provenance cross-check field in one comparison artifact; it does not affect any
  prediction, label, metric, aggregate, CI, or eligibility decision, and the reported
  `internal_value` (`mean_tumor_dice_step_500`) itself is correct. Per explicit project direction,
  the already-published output root was not touched or regenerated to fix this; it is recorded here as
  a known limitation of the current `phase8_internal_external_comparison.json` for a future,
  separately authorized correction pass.
- Consequences: Phase 8 now has a complete, hash-verified, once-only external label evaluation of the
  Package F predictions, with an honest low-performance result, a disclosed cosmetic defect in one
  comparison field, and zero changes to any locked prediction, checkpoint, threshold, preprocessing, or
  support-policy decision. No inference was rerun and no case was excluded for a performance reason.

### 2026-08-09: Package H — Final Report, Reproduction Check, and Phase 8 Closure

- Context: the project owner directed the final Phase 8 task, Package H, to fix the one disclosed
  non-scientific defect from the immediately preceding entry, publish a corrected comparison
  artifact, write the final Phase 8 report, perform a non-scientific reproduction/provenance check,
  update project tracking, and close Phase 8. This entry does not perform, repeat, or authorize any
  retraining, reinference, or metric recomputation.
- Decision: the wrong-dict-key defect in `_build_internal_external_comparison`
  (`src/protoem_ct/external/label_evaluation.py`) was fixed minimally: the lookup key
  `"checkpoint_sha256"` was corrected to the evidence file's actual field
  `"selected_checkpoint_hash"`, and the previously inline evidence-file path was lifted to a module
  constant (`_INTERNAL_CHECKPOINT_EVIDENCE_PATH`) so it can be monkeypatched in tests. No metric
  calculation, aggregation, bootstrap, label mapping, geometry, eligibility, internal value,
  external value, checkpoint selection, or prediction was touched. A focused regression test
  (`tests/unit/test_phase8_label_evaluation.py`) proves: the correct locked checkpoint hash is
  emitted; `internal_checkpoint_matches_locked_checkpoint=true` when identities match; a genuine
  mismatch still records `false`; and no metric field changes as a result of the correction.
  Committed as `51638ca662510bf4cb031b2aec29778bf26bba26`
  ("fix(phase8): correct comparison checkpoint provenance"), before any corrected artifact was
  published, per an independent skeptical re-read of the diff (PASS).
- The original `phase8_internal_external_comparison.json`
  (raw-file SHA-256 `a082e2588b6aa9f10d64dcd90d89e1f1c8dd9bb855b71c94f3994cbca13cc8dc`) was left
  untouched under `/Volumes/Lexar/ProtoEM-CT/runs/phase8_external_evaluation_v1`, as required. A
  separate, additive artifact, `phase8_internal_external_comparison_corrected_v1.json`, was
  published to the same output root by calling the now-fixed `_build_internal_external_comparison`
  with the real, already-committed bootstrap CI values (`phase8_external_bootstrap_ci.json`); every
  field was verified field-by-field equal to the original except the two defective provenance
  fields, which now read `internal_checkpoint_sha256 =
  2d7989fd134b1348e82cc52afbcf4738c0ce3c17e9c68df774431f577dede651` and
  `internal_checkpoint_matches_locked_checkpoint = true`. The corrected artifact embeds a
  `correction_provenance` block naming the reason, the corrected field paths, the fix commit, and a
  reference back to the preserved original.
- A new small, self-hashed, non-medical artifact contract,
  `Phase8ClosureReproductionRecord` (`src/protoem_ct/external/closure.py`, with tests in
  `tests/unit/test_phase8_closure.py`), was added to record the verified Phase 8 provenance chain
  (freeze/checkpoint/preregistration/prediction-lock/metric-report/domain-shift hashes, the
  corrected-comparison reference, and the fix commit) at closure. One instance was published,
  reject-on-overwrite, to
  `/Volumes/Lexar/ProtoEM-CT/runs/phase8_external_evaluation_v1/phase8_final_closure_reproduction_record.json`
  (`closure_record_hash` `b4b8964dd33d6be3c83ede37600c01fc97a1221b9f5b05f4a22adca18d63b816`).
- The external drive (`/Volumes/Lexar`) was mounted and accessible throughout this closure session.
  The freeze artifact's raw-file SHA-256, the checkpoint binary's SHA-256, and the self-hash fields
  of the preregistration, prediction-lock, metric-report, and domain-shift-record artifacts were all
  independently re-verified against the mounted artifacts and found to exactly match the values
  already recorded in this file and the Phase 8 closure provenance records. Both Package C
  provenance commit hashes (`f1d4e1be7b6f7aa3657792cae9125c825affdded` execution-base HEAD and
  `f264ce79dd68039256d140f03c468baed8ecea07` post-run code-capture commit) were independently
  re-verified present in this repository's Git history.
- The final closure report, `docs/phase8/FINAL_REPORT.md`, was written covering the frozen
  configuration, preregistration/lock/evaluation identities, all 9 external metrics with bootstrap
  CIs, the domain-shift artifact's aggregate-only, `not_compared` status (no causal domain-shift
  claim is made), the qualitative-output index reference, the corrected comparison, an explicit
  no-tuning statement, an explicit negative-result interpretation, and limitations (including the
  low development-validation Dice, the poor external metrics, the five `tumor_target_absent`
  exclusions, the LiTS/MSD-Task03-Liver single-cohort caveat, the Package C provenance-commit
  distinction, the historical pilot-test-recovery incident stated without an unsupported "fully
  recovered" claim, and the superseded original comparison artifact).
  `ACCEPTANCE_CHECKLIST.md` (Gate 8) and `docs/PHASE_PLAN.md` were updated to reflect this closure
  without claiming a result beyond what the artifacts support.
- A final independent, skeptical review pass (performed as a distinct step after all closure
  artifacts and docs were prepared) verified: no scientific computation was rerun during closure; the
  comparison provenance bug was corrected only in the two provenance fields; the original Package G
  comparison artifact is preserved unmodified; the corrected comparison preserves every other
  scientific value; the freeze/preregistration/prediction-lock/evaluation identities match; the
  external results in the final report exactly match the existing artifacts; no unsupported
  positive-generalization claim is made anywhere in the closure documents; the limitations honestly
  state the poor internal and external performance; LiTS/MSD Task03 Liver is never treated as two
  independent cohorts; no internal-test medical data was newly accessed; no external prediction was
  regenerated; and the acceptance/closure statements match the actual completed work. Verdict: PASS.
- Verified local outcomes for the Package H changes:
  `uv run pytest -q tests/unit/test_phase8_label_evaluation.py tests/unit/test_phase8_closure.py`:
  PASS, `24 passed`; `uv run ruff format --check` and `uv run ruff check` on changed files: PASS;
  `uv run mypy` on changed files: PASS; `git diff --check`: PASS.
- Decision: Phase 8 is recorded as **complete and closed** with a **negative external-validation
  result**. Closure is granted because the locked, preregistered protocol was executed exactly as
  designed and its result was reported completely and transparently -- not because the scientific
  result is favorable. External performance must not be reframed as successful generalization.
- Consequences: Phase 8 is closed. No Phase 9, LLM/VLM track, or any phase beyond Phase 8 has begun.
  Any future correction, re-evaluation, or extension of Phase 8 evidence requires a new, separately
  authorized decision entry; this entry does not itself authorize one.
