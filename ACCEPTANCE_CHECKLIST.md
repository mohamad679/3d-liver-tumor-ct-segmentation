# Acceptance Checklist

## Gate 0

- [x] Standard repository structure
- [x] Python 3.11 configuration
- [x] uv and `pyproject.toml`
- [x] ruff
- [x] mypy
- [x] pytest
- [x] hypothesis
- [x] pre-commit
- [x] Makefile
- [x] CPU-only GitHub Actions
- [x] Tiny synthetic NIfTI CT and tumor-mask fixtures
- [x] `protoem-ct validate-pair` CLI
- [x] Tests for shape mismatch
- [x] Tests for affine mismatch
- [x] Tests for invalid labels
- [x] Tests for NaNs
- [x] Lint passes
- [x] Typing passes
- [x] Tests pass
- [x] Smoke test passes

### Gate 0 Evidence

Gate 0 status: PASSED locally

- Python: 3.11.9
- `uv sync --frozen --python 3.11 --all-groups`: PASS
- `uv lock --check`: PASS
- `make lint`: PASS
- `make test`: PASS, `24 passed`
- `make smoke`: PASS, `1 passed`
- `uv run mypy src tests`: PASS
- `uv run pre-commit validate-config`: PASS
- `uv run pre-commit run --all-files`: PASS
- `uv run protoem-ct --help`: PASS
- `uv run protoem-ct validate-pair --help`: PASS
- No tracked NIfTI, DICOM, model weights, checkpoints, predictions, credentials, or generated binary artifacts.
- Synthetic NIfTI files are created dynamically under pytest temporary directories only.
- GitHub-hosted CI is not claimed as passed until it runs on GitHub.

## Gate 1

- [x] Complete synthetic DAG runs from a clean generated-artifact state
- [x] Every stage has explicit inputs and outputs
- [x] Artifacts use documented schemas
- [x] Report is generated only from persisted JSON
- [x] Config, manifest, and Git metadata are recorded
- [x] Repeated runs are deterministic where expected
- [x] Integration test passes
- [x] Lint passes
- [x] Typing passes
- [x] Full tests pass
- [x] Smoke tests pass

### Gate 1 Evidence

Gate 1 status: PASSED locally

- Local Snakemake command pattern:
  `uv run snakemake --snakefile Snakefile --cores 1 --rerun-incomplete --config generated_root=/absolute/external/generated-root git_commit=<explicit-commit> created_at_utc=<explicit-utc-timestamp>`
- Real empty-root DAG test: PASS via `uv run pytest -q tests/integration/test_phase1_snakemake_dag.py`, `9 passed`
- Full test suite: PASS via `uv run pytest -q`, `285 passed`
- Smoke tests: PASS via `make smoke`, `1 passed`
- `uv lock --check`: PASS
- `uv run ruff check .`: PASS
- `uv run ruff format --check .`: PASS
- `uv run mypy src tests`: PASS
- `uv run pre-commit validate-config`: PASS
- `uv run pre-commit run --all-files`: PASS
- `uv run snakemake --snakefile Snakefile --lint --config generated_root=/tmp/protoem-ct-phase1-lint git_commit=phase1-lint created_at_utc=2026-01-01T00:00:00Z`: PASS
- `uv run snakemake --snakefile Snakefile --list-rules --config generated_root=/tmp/protoem-ct-phase1-list git_commit=phase1-list created_at_utc=2026-01-01T00:00:00Z`: PASS
- The complete DAG was verified from an empty existing generated root and a nonexistent generated root under pytest temporary directories.
- Deleting only the final report directory rebuilt a byte-identical report without modifying upstream artifact bytes or mtimes.
- Reports from independent generated roots with identical explicit Git commit and timestamp were byte-identical.
- The report is generated from persisted JSON artifacts and does not depend on MLflow.
- Local MLflow tracking remains a separate post-report operation and is not a Gate 1 prerequisite.
- Phase 1 uses synthetic data and deterministic dummy inference for pipeline verification only; it contains no scientific model result.
- GitHub-hosted CI, PR review, branch merge, and hosted deployment status are not claimed.

## Gate 2

- [ ] Real LiTS development-cohort manifest is produced from an explicit dataset root
- [ ] Deterministic patient-level train, validation, and immutable internal-test split
      artifacts are produced
- [ ] Pairwise patient-overlap and case-overlap tests are all zero
- [ ] Machine-readable per-case, patient-level, and development dataset-level QA artifacts
      are reproducible
- [ ] Development Markdown QA report is reproducibly generated from saved QA artifacts
- [ ] LiTS-style and 3D-IRCADb-style adapters have synthetic format and edge-case tests
- [ ] Geometry, label, finite-value, connected-component, histogram, empty-tumor, duplicate,
      missing, ambiguous, and unsafe-path edge cases are covered by synthetic tests
- [ ] No medical data, PHI-bearing manifest, prediction, derived volume, or machine-specific
      dataset path is tracked
- [ ] Real 3D-IRCADb-01 external data was not accessed, scanned, QA'd, split, tuned on, or
      summarized
- [ ] Lint passes
- [ ] Typing passes
- [ ] Full tests pass
- [ ] Smoke tests pass
- [ ] GitHub-hosted CI passes
- [ ] Leakage audit is completed with no unresolved critical finding

### Gate 2 Evidence

Gate 2 status: NOT EVALUATED

Evidence must be recorded only after the approved Phase 2 implementation and real LiTS
development execution. Do not enter estimated case counts, overlap counts, dataset availability,
command results, or hosted-CI results. Gate 2 cannot pass until the real LiTS manifest, split,
development QA artifacts, generated Markdown report, completed leakage audit, and all local and
hosted checks exist.

## Gate 3

- [ ] Pending definition.

## Gate 4

- [ ] Pending definition.

## Gate 5

- [ ] Pending definition.

## Gate 6

- [ ] Pending definition.

## Gate 7

- [ ] Pending definition.

## Gate 8

- [ ] Pending definition.
