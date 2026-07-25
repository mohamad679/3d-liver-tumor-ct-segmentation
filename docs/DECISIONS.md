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
