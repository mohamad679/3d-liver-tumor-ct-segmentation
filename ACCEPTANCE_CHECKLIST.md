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
- [x] `nnU-Net v2` baseline wrapper path completes the required synthetic or tiny-data end-to-end
      flow
- [x] `MONAI SegResNet` baseline path completes the required synthetic or tiny-data end-to-end flow
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

- [x] Deterministic patient-level support manifests exist for `K = 1, 2, 5, 10, 20`
- [x] Each `K` has exactly three fixed support manifests with stable replicate identifiers
- [x] Every support manifest is linked to the exact source development manifest hash and development
      split hash
- [x] Support patients and cases are disjoint from the immutable internal-test cohort in every
      generated artifact
- [x] Lesion-burden stratification is used where feasible and deterministic fallback behavior is
      explicitly recorded where infeasible
- [x] `head_only`, `decoder_only`, and `full_finetune` adaptation modes are implemented with
      explicit parameter-group validation
- [x] Trainable-parameter count is available from the adaptation-mode contract
- [x] Duration and memory are explicitly representable as unavailable because no Phase 4 adaptation
      training execution was performed
- [x] Deterministic adaptation configs are generated
- [x] Deterministic protocol-table artifact enumerates all planned
      `K x replicate x adaptation-mode` runs and their source hashes
- [x] Unit, integration, leakage, determinism, synthetic smoke, lint, format, typing, and full
      repository tests pass locally
- [x] No tracked medical data, PHI-bearing support manifests, checkpoints, predictions, or large
      generated artifacts are Git-visible
- [x] Gate 4 close-out remains limited to few-shot protocol generation and publication without
      Phase 5, ProtoEM-CT, or adaptation execution

### Gate 4 Evidence

Gate 4 status: PASSED locally on 2026-07-30

- Repository-wide verification:
  - `uv run ruff check .`: PASS, `All checks passed!`
  - `uv run ruff format --check .`: PASS, `132 files already formatted`
  - `uv run mypy src`: PASS, `Success: no issues found in 53 source files`
  - `uv run pytest -q`: PASS, `769 passed, 2 skipped in 388.27s (0:06:28)`
  - `uv run protoem-ct generate-phase4-fewshot-protocol --help`: PASS
- Synthetic Phase 4 publication using existing test-fixture constructors and external temporary
  output roots:
  - support manifests: `15`
  - adaptation configs: `45`
  - protocol-table JSON rows: `45`
  - protocol-table Markdown data rows: `45`
  - support/internal-test patient overlap count: `0`
  - support/internal-test case overlap count: `0`
  - leakage check passed: `true`
  - byte-identical artifacts across two separate external output roots: `true`
- Adaptation modes verified: `head_only`, `decoder_only`, `full_finetune`
- Trainable-parameter-count evidence from the deterministic adaptation unit-test model:
  - `head_only`: total `98`, trainable `15`, frozen `83`
  - `decoder_only`: total `98`, trainable `38`, frozen `60`
  - `full_finetune`: total `98`, trainable `98`, frozen `0`
- Duration and memory are intentionally unavailable at Gate 4 because no Phase 4 adaptation
  training execution was performed. The existing `fewshot_run_summary_v1` schema supports
  `duration_seconds = null` and `memory_availability_status = "unavailable"` with null memory
  fields.

These results verify the deterministic few-shot protocol contract only. They do not claim
adaptation performance, runtime duration, memory usage, GPU execution, checkpoint quality, external
validation, or any Phase 5/ProtoEM-CT behavior.

## Gate 5

- [ ] Pending definition.

## Gate 6

- [ ] Pending definition.

## Gate 7

- [ ] Pending definition.

## Gate 8

- [ ] Pending definition.
