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
