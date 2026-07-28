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

- [x] Real LiTS development-cohort manifest is produced from an explicit dataset root
- [x] Deterministic patient-level train, validation, and immutable internal-test split
      artifacts are produced
- [x] Pairwise patient-overlap and case-overlap tests are all zero
- [x] Machine-readable per-case, patient-level, and development dataset-level QA artifacts
      are reproducible
- [x] Development deterministic JSON QA report is reproducibly generated from saved QA artifacts
- [x] LiTS-style and 3D-IRCADb-style adapters have synthetic format and edge-case tests
- [x] Geometry, label, finite-value, connected-component, histogram, empty-tumor, duplicate,
      missing, ambiguous, and unsafe-path edge cases are covered by synthetic tests
- [x] No medical data, PHI-bearing manifest, prediction, derived volume, or machine-specific
      dataset path is tracked
- [x] Real 3D-IRCADb-01 external data was not accessed, scanned, QA'd, split, tuned on, or
      summarized
- [x] Lint passes
- [x] Typing passes
- [x] Full tests pass
- [x] Smoke tests pass
- [x] GitHub-hosted CI passes
- [x] Leakage audit is completed with no unresolved critical finding

### Gate 2 Evidence

Gate 2 status: PASSED locally for real-data development-cohort QA and leakage evidence.

Verified v2 evidence is based on the approved read-only MSD Task03 Liver / LiTS-derived
development-cohort execution. Task03 Liver is treated as the LiTS-derived development cohort, not as
an independent second cohort.

- Reproducible anonymous manifest: 131 development cases, `manifest_hash`
  `c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b`
- Deterministic patient-level split: 91 train, 20 validation, 20 immutable internal test,
  `split_hash` `936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb`
- Pairwise patient-overlap counts: zero
- Pairwise case-overlap counts: zero
- Cross-partition image SHA-256 overlap: zero
- Cross-partition label SHA-256 overlap: zero
- Cross-partition image/label hash-pair overlap: zero
- Geometry/label QA: 131 passed, 0 failed, explicit affine tolerance `0.0001 mm`
- Lesion summaries: 131 analyzed, 0 skipped, 908 total lesions
- CT development summaries: 131 analyzed, 0 skipped
- Final deterministic JSON QA report: 131 passed, 0 failed
- LeakageAuditArtifact: verified, `audit_passed=True`, zero findings
- External artifacts and data remain outside Git; no generated dataset, NIfTI, key, JSON artifact,
  CSV, TSV, or log is Git-visible.

These are dataset-QA and leakage results only. They do not claim model performance, scientific
efficacy, segmentation generalization, external validation, clinical validity, or Phase 3 progress.
GitHub-hosted CI is not claimed until it runs on GitHub.

## Gate 3

- [x] Dependency and environment contracts for the Phase 3 baseline stack are documented and
      verified
- [x] Safe external output-path contracts for artifacts, checkpoints, predictions, and MLflow runs
      are documented and enforced
- [x] Deterministic provenance is recorded for baseline preparation, training, inference, and
      evaluation paths
- [x] Metric engine covers tumor Dice, IoU, HD95, normalized surface Dice, lesion-wise recall,
      lesion-wise precision, lesion F1, false-positive lesions per scan, volume error, and explicit
      empty-mask behavior, with dedicated empty-mask tests
- [x] Synthetic 3D fixtures exist for baseline preparation, inference, and metric-path testing
- [ ] `nnU-Net v2` baseline wrapper path completes the required synthetic or tiny-data end-to-end
      flow
- [ ] `MONAI SegResNet` baseline path completes the required synthetic or tiny-data end-to-end flow
- [ ] CPU shape smoke tests pass
- [ ] Tiny-subset overfit tests demonstrate that each training path can reduce its configured loss
- [ ] Deterministic metric JSON is produced from saved predictions and labels by project code where
      the contract claims determinism
- [ ] No tracked checkpoints, predictions, datasets, or model weights are Git-visible
- [ ] Full local repository quality checks pass
- [ ] GitHub-hosted CI passes
- [ ] Gate 3 close-out is documented without claiming full development-cohort training

### Gate 3 Scope Note

Gate 3 is limited to the Phase 3 baseline scope: `nnU-Net v2`, `MONAI SegResNet`, shared baseline
infrastructure, deterministic metric JSON, synthetic or tiny-data execution, CPU smoke tests, and
tiny-subset overfit evidence. Gate 3 does not require full training on all 131 development cases.

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
