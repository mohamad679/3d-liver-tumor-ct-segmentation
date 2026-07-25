# Phase Plan

## Active Phase

Phase 0 is completed.

Phase 1 is completed locally.

Active phase: Phase 2 planning is next but has not started.

## Phase 0 Scope

Phase 0 establishes the environment contract, repository structure, quality gates, fixture strategy, and minimal validation surface needed before any scientific implementation begins.

Phase 0 includes planning and, in the implementation step, repository scaffolding for Python 3.11 tooling, linting, typing, tests, pre-commit checks, CPU-only continuous integration, tiny synthetic NIfTI fixtures, and a `protoem-ct validate-pair` CLI smoke path.

Training code, real-data ingestion, model implementation, and Phase 1 work are excluded from Phase 0.

## Planned Files and Directories

- `AGENTS.md`
- `PROJECT_SPEC.md`
- `ACCEPTANCE_CHECKLIST.md`
- `docs/`
- `docs/PHASE_PLAN.md`
- `docs/DECISIONS.md`
- `pyproject.toml`
- `uv.lock`
- `Makefile`
- `.pre-commit-config.yaml`
- `.github/workflows/`
- `.github/workflows/ci.yml`
- `src/`
- `src/protoem_ct/`
- `tests/`
- `tests/fixtures/`

## Implemented Files and Components

- Repository governance and planning documents:
  - `AGENTS.md`
  - `PROJECT_SPEC.md`
  - `ACCEPTANCE_CHECKLIST.md`
  - `docs/PHASE_PLAN.md`
  - `docs/DECISIONS.md`
- Python 3.11 package configuration:
  - `pyproject.toml`
  - `uv.lock`
  - `src/protoem_ct/`
  - `src/protoem_ct/py.typed`
- Quality and automation:
  - `Makefile`
  - `.pre-commit-config.yaml`
  - `.github/workflows/ci.yml`
- Phase 0 validation surface:
  - `src/protoem_ct/data/validation.py`
  - `src/protoem_ct/cli/main.py`
  - `protoem-ct validate-pair` CLI
- Tests and temporary synthetic NIfTI fixture generation:
  - `tests/conftest.py`
  - `tests/unit/test_validation.py`
  - `tests/integration/test_validate_pair_cli.py`
  - `tests/smoke/test_validate_pair_smoke.py`

## Planned Checks

- `uv sync`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src`
- `uv run pytest -q`

## Risks

- Dependency compatibility on macOS x86_64
- Accidental inclusion of data or generated artifacts
- NIfTI affine comparison behavior
- CLI packaging errors

## Gate 0 Acceptance Criteria

Gate 0 is accepted when the standard repository structure exists, Python 3.11 and uv packaging are configured, ruff, mypy, pytest, hypothesis, pre-commit, Makefile, and CPU-only GitHub Actions are present, tiny synthetic NIfTI CT and tumor-mask fixtures exist, the `protoem-ct validate-pair` CLI is available, mismatch and invalid-data tests cover shape mismatch, affine mismatch, invalid labels, and NaNs, and linting, formatting check, typing, tests, and smoke test all pass.

## Phase 1 Scope

Phase 1 implements a reproducible synthetic end-to-end DAG only. The DAG stages are:

1. `create-data`
2. `validate-data`
3. `preprocess-data`
4. `infer-dummy`
5. `evaluate`
6. `report`

The final Snakemake DAG uses six explicit rules in this order: `create_data -> validate -> preprocess -> infer_dummy -> evaluate -> report`.

The Phase 1 report is generated only from saved machine-readable JSON artifacts. Report values are traceable to those persisted JSON files, not recomputed from in-memory state, transient logs, NIfTI files, MLflow, or human-edited report text.

Phase 1 includes separate local MLflow tracking only. MLflow is not a prerequisite for Gate 1 or for report generation. No remote MLflow tracking server is in scope.

Phase 1 includes config hashing and manifest hashing so generated outputs can be traced to the exact configuration and artifact manifest used to create them.

Phase 1 includes CLI commands for synthetic DAG operations, including creation of synthetic inputs, validation, preprocessing, dummy inference, evaluation, report generation, and separate local MLflow run tracking.

Phase 1 includes integration tests that rebuild the final synthetic report from an empty generated-artifact directory and from a nonexistent generated root.

Phase 1 must not include real-data ingestion, neural-network training, model baseline implementation, few-shot protocol work, robustness evaluation, external validation, or LLM/VLM work.

Phase 1 uses deterministic synthetic data and deterministic dummy inference for pipeline verification only. It contains no scientific model result.

## Phase 1 Planned Files and Directories

- `Snakefile`
- `configs/experiment/`
- `configs/data/`
- `src/protoem_ct/cli/`
- `src/protoem_ct/data/`
- `src/protoem_ct/evaluation/`
- `src/protoem_ct/reporting/`
- `tests/integration/`
- `tests/smoke/`
- `reports/templates/`

## Phase 1 Dependency Status

Phase 1 uses dependencies that are already present in the locked environment:

- `snakemake`
- `mlflow`

No dependencies were added or updated for Phase 1 close-out. `hydra-core` and `omegaconf` were not introduced for the Phase 1 DAG.

## Phase 1 Planned CLI Commands

- `protoem-ct create-data`
- `protoem-ct validate-data`
- `protoem-ct preprocess-data`
- `protoem-ct infer-dummy`
- `protoem-ct evaluate`
- `protoem-ct report`
- `protoem-ct track-run`
- `uv run snakemake --snakefile Snakefile --cores 1 --rerun-incomplete --config generated_root=/absolute/external/generated-root git_commit=<explicit-commit> created_at_utc=<explicit-utc-timestamp>`

The Snakemake DAG requires `generated_root`, `git_commit`, and `created_at_utc` through explicit `--config` values. `generated_root` must be an absolute path and is expected to be outside the repository for local Gate 1 verification.

## Phase 1 Planned Verification Commands

- `uv sync`
- `make lint`
- `make test`
- `make smoke`
- `uv run snakemake --snakefile Snakefile --cores 1 --rerun-incomplete --config generated_root=/absolute/external/generated-root git_commit=<explicit-commit> created_at_utc=<explicit-utc-timestamp>`
- A clean rebuild of the final synthetic report from an empty generated-artifact directory

## Phase 1 Risks

- Nondeterministic artifact content
- Unstable hashes
- Hidden dependency on working-directory state
- Stale generated artifacts causing false success
- MLflow local-path leakage
- Report values not traceable to JSON
- Snakemake compatibility on macOS x86_64

## Gate 1 Acceptance Criteria

Gate 1 is accepted when the complete synthetic DAG runs from a clean generated-artifact state, every stage has explicit inputs and outputs, artifacts use documented schemas, the report is generated only from persisted JSON, config, manifest, and Git metadata are recorded, repeated runs are deterministic where expected, the integration test passes, and lint, typing, full tests, and smoke tests pass.

## Implementation Notes and Command Results

Phase 0 implemented only the environment contract, repository structure, quality gates, temporary synthetic fixture strategy, validation helper, and `protoem-ct validate-pair` CLI smoke path.

No training, model implementation, real-data ingestion, or Phase 1 work was performed.

Verified local outcomes:

- Python: 3.11.9
- `uv sync --frozen --python 3.11 --all-groups`: PASS
- `uv lock --check`: PASS
- `make lint`: PASS, including Ruff lint, Ruff format check, and `mypy src`
- `make test`: PASS, `24 passed`
- `make smoke`: PASS, `1 passed`
- `uv run mypy src tests`: PASS
- `uv run pre-commit validate-config`: PASS
- `uv run pre-commit run --all-files`: PASS
- `uv run protoem-ct --help`: PASS
- `uv run protoem-ct validate-pair --help`: PASS
- Git worktree before documentation close-out editing: clean
- No tracked NIfTI, DICOM, model weights, checkpoints, predictions, credentials, or generated binary artifacts
- Synthetic NIfTI files are created dynamically under pytest temporary directories only

Operational environment note: some uv-backed commands initially failed inside the managed Codex sandbox because it could not access `$HOME/.cache/uv`. The exact commands passed after approved normal filesystem access. This is not a project defect.

Gate 0 passed locally.

GitHub push and hosted CI verification remain publication tasks, not Phase 0 implementation defects. GitHub-hosted CI status must not be claimed until the workflow runs on GitHub.

### Phase 1 Local Close-Out

Phase 1 implemented the final local Snakemake DAG and local close-out evidence. The DAG has exactly these rules: `all`, `create_data`, `validate`, `preprocess`, `infer_dummy`, `evaluate`, and `report`.

The rule dependency order is `create_data -> validate -> preprocess -> infer_dummy -> evaluate -> report`. Each rule invokes the existing `protoem-ct` CLI command for that stage and does not duplicate Python stage logic inside the Snakefile.

The required local command pattern is:

`uv run snakemake --snakefile Snakefile --cores 1 --rerun-incomplete --config generated_root=/absolute/external/generated-root git_commit=<explicit-commit> created_at_utc=<explicit-utc-timestamp>`

The fixed generated layout under the explicit generated root is:

- `data/`
- `artifacts/validation.json`
- `preprocessed/`
- `artifacts/preprocess.json`
- `predictions/`
- `artifacts/inference.json`
- `artifacts/evaluation.json`
- `report/phase1_report.md`
- `report/report_artifact.json`

Verified local outcomes:

- `uv lock --check`: PASS
- `uv run ruff check .`: PASS
- `uv run ruff format --check .`: PASS
- `uv run mypy src tests`: PASS
- `uv run pytest -q`: PASS, `285 passed`
- `make smoke`: PASS, `1 passed`
- `uv run pre-commit validate-config`: PASS
- `uv run pre-commit run --all-files`: PASS
- `uv run snakemake --snakefile Snakefile --lint --config generated_root=/tmp/protoem-ct-phase1-lint git_commit=phase1-lint created_at_utc=2026-01-01T00:00:00Z`: PASS
- `uv run snakemake --snakefile Snakefile --list-rules --config generated_root=/tmp/protoem-ct-phase1-list git_commit=phase1-list created_at_utc=2026-01-01T00:00:00Z`: PASS
- `uv run pytest -q tests/integration/test_phase1_snakemake_dag.py`: PASS, `9 passed`
- Additional temporary-directory real DAG execution outside the repository: PASS

The report generation stage remains JSON-only and independent of MLflow. Local MLflow tracking remains available as a separate `protoem-ct track-run` operation after report generation.

Gate 1 passed locally because the real empty-root DAG test passed and the verification checks above passed locally.

Phase 1 uses synthetic data and deterministic dummy inference for pipeline verification only. It contains no model training, no real-data execution, and no scientific model result.

Hosted CI, pull request review, branch merge, and hosted deployment success have not been claimed.
