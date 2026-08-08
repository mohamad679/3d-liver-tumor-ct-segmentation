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
