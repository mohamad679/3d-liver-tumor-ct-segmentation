# Phase Plan

## Active Phase

Phase 0 is completed.

Phase 1 is completed locally.

Phase 2 Gate 2 close-out is completed locally from the approved real-data development-cohort QA and
leakage artifacts.

Active phase: Phase 3 baselines. Phase 3 status is in progress from base commit
`907ea9d3f559ce959b92bc78c05c75c8187a5f32` on branch `phase/3-baselines`.

Phase 4 and later phases remain blocked until the Phase 3 Gate 3 baseline scope is completed,
reviewed, and merged.

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

## Phase 2 Scope

Phase 2 establishes the real LiTS development-cohort inventory and QA contract while preserving the
Phase 8 external-validation boundary. The normative details are defined in
`docs/PHASE2_DATA_CONTRACT.md`.

Phase 2 includes:

- a LiTS-style NIfTI development-cohort adapter;
- a 3D-IRCADb-style adapter interface with synthetic format tests only;
- anonymous machine-readable dataset manifests;
- deterministic patient-level development splits;
- orientation, spacing, affine, shape, label, and finite-value QA;
- connected-component lesion summaries;
- fixed-bin CT intensity histograms and lesion-size summaries;
- machine-readable development QA artifacts;
- a Markdown QA report generated from saved artifacts;
- a patient-overlap and leakage audit; and
- synthetic edge-case tests.

Phase 2 excludes model training, baseline implementation, preprocessing fitting, few-shot support
selection, robustness experiments, external evaluation, and LLM/VLM work.

## Phase 2 Cohort and Data Boundaries

LiTS labeled training data is the development cohort. LiTS and MSD Task03 Liver are representations
of the same underlying cohort and must never be treated as independent cohorts. All development
train, validation, and immutable internal-test assignments are patient-disjoint.

Real 3D-IRCADb-01 remains untouched until Phase 8 external validation. Phase 2 may implement its
adapter interface and exercise it with synthetic fixtures, but must not scan, inventory, QA, split,
tune on, or generate statistics from the real external cohort. External labels must not influence
any development decision.

Dataset roots enter only through explicit CLI arguments or ignored local configuration. Adapters
operate only beneath that explicit root, do not infer locations from the current working directory,
do not search the computer, do not follow symlinks outside the root, and never modify source data.
All outputs are written beneath an explicit external generated root. Tracked tests and examples use
synthetic data only; no medical data, PHI-bearing manifest, prediction, derived volume,
source-identifier mapping, or absolute machine path is committed.

## Phase 2 Planned Implementation Sequence

Each numbered substage is a small, reviewable unit. Work proceeds on one named substage at a time in
this order:

1. Schema and hashing contracts
2. Shared safe-path and adapter protocol
3. LiTS-style discovery adapter
4. Synthetic-tested 3D-IRCADb-style adapter
5. Anonymous manifest CLI
6. Patient-level split generation
7. Geometry and label QA
8. Lesion connected-component summaries
9. Fixed CT histogram and dataset summaries
10. QA artifact and JSON-only report
11. Leakage audit automation
12. Synthetic end-to-end Phase 2 DAG
13. Real LiTS dry-run inventory
14. Real LiTS manifest and QA execution after explicit user approval
15. Gate 2 close-out, hosted CI, and merge

Substages 1 through 12 must be completed and reviewed with synthetic fixtures before any real-data
execution. Substages 13 and 14 require the user to confirm the dataset's lawful local availability
and provide an explicit root path. Do not request that path during planning. Substage 14 additionally
requires explicit approval of the final split parameters and QA scientific conventions.

The Phase 2 report path is JSON-only in its data dependency: per-case, patient-level, and
dataset-level QA JSON artifacts are persisted first, and the Markdown report reads only those saved
artifacts. It must not open medical volumes or recompute QA statistics.

## Phase 2 Planned Artifacts

- Anonymous, versioned LiTS development dataset manifest linked to explicit Git and time metadata
- Versioned patient-level split JSON linked to the source manifest hash
- Per-case QA JSON retaining both valid and invalid cases with explicit failure reasons
- Patient-level and development dataset-level QA summary JSON
- Fixed-bin CT histogram and lesion-size summary JSON
- Leakage-audit JSON evidence and completed `docs/LEAKAGE_AUDIT.md`
- Development Markdown QA report generated only from saved JSON artifacts

Generated artifacts remain outside the repository beneath the explicit generated root.

## Phase 2 Planned Verification

- Unit tests for schemas, hashing, deterministic serialization, path safety, adapter layouts, split
  invariants, QA calculations, connected components, histograms, and failure behavior
- Integration tests for the manifest, split, QA, report, leakage-audit, and synthetic DAG CLI paths
- Synthetic edge-case tests for missing, ambiguous, duplicate, unsafe, invalid, empty-tumor, and
  connectivity-sensitive cases
- Byte-for-byte reproducibility checks for manifest-independent ordering, split artifacts, QA
  artifacts, and the generated Markdown report where explicit metadata is identical
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src tests`
- `uv run pytest -q`
- `make smoke`
- `uv run pre-commit run --all-files`
- Phase 2 Snakemake lint and clean synthetic DAG execution
- GitHub-hosted CI after publication

## Phase 2 Risks and Approval Points

- Local LiTS layout may differ from published conventions; adapter discovery therefore requires
  configurable, reviewed layout rules.
- Anonymous identifiers or root fingerprints could disclose source identifiers if their construction
  is poorly designed; implementation requires one-way project namespacing, collision checks, and no
  committed reverse mapping.
- Dataset duplication or LiTS/MSD relabeling could create cohort leakage.
- Split ratios, minimum partition sizes, deterministic rounding, optional lesion-aware
  stratification, fixed CT histogram range and bins, affine tolerance, 3D connected-component
  connectivity, lesion tie-breaking, and cross-partition near-duplicate policy remain explicit
  approval points before real execution.
- Invalid cases must remain visible in artifacts instead of being silently repaired or skipped.
- Filesystem symlinks and alternate spellings can escape or alias roots unless resolved and rejected.
- Real external data access would violate the Phase 8 boundary.

## Gate 2 Acceptance Criteria

Gate 2 requires a real LiTS development manifest produced from an explicit dataset root,
deterministic patient-level train/validation/immutable-internal-test split artifacts, zero pairwise
patient and case overlap, reproducible machine-readable QA artifacts and development Markdown QA
report, synthetic adapter and QA edge-case tests, no tracked medical data or machine-specific paths,
no access to real external data, passing lint, typing, full tests, smoke tests, and GitHub-hosted CI,
and a completed leakage audit with no unresolved critical finding.

Gate 2 passed locally for the Phase 2 real-data development-cohort QA and leakage-audit scope. The
verified development cohort is MSD Task03 Liver used as the LiTS-derived development cohort, with
131 cases and no independent-cohort interpretation. The deterministic patient-level split is 91
train, 20 validation, and 20 immutable internal-test patients. The final QA report recorded 131
passed cases and 0 failed cases. The leakage audit passed with zero patient overlap, zero case
overlap, zero cross-partition image-hash overlap, zero cross-partition label-hash overlap, zero
cross-partition image/label hash-pair overlap, and zero finding codes.

Real 3D-IRCADb-01 remains untouched for later Phase 8 external validation. These Phase 2 results are
dataset-QA and leakage results only; they do not claim model performance, scientific efficacy,
clinical validity, or external validation. GitHub-hosted CI status must still be established after
publication and is not inferred from local evidence.

## Phase 3 Scope

Phase 3 establishes the project baseline-modeling scope only. The baseline families are limited to:

- `nnU-Net v2`
- `MONAI SegResNet`

Phase 3 includes the shared baseline infrastructure needed to exercise those two families
deterministically and audibly:

- deterministic configuration and provenance;
- external artifact-path contracts;
- preprocessing orchestration;
- training orchestration;
- checkpointing and resume behavior;
- deterministic seeds;
- device selection;
- AMP only where the selected device safely supports it;
- sliding-window inference;
- MLflow metadata logging without medical data;
- machine-readable metric JSON;
- CPU shape smoke tests;
- synthetic or tiny-subset overfit tests; and
- failure handling for NaN/Inf, incompatible shapes, missing files, and duplicate predictions.

The required metric interface for both baseline families is:

- tumor Dice;
- IoU;
- HD95;
- normalized surface Dice;
- lesion-wise recall;
- lesion-wise precision;
- lesion F1;
- false-positive lesions per scan;
- volume error; and
- explicit empty-mask behavior.

Phase 3 excludes:

- few-shot support protocols;
- frozen-encoder/head-only, decoder-only, or few-shot full fine-tuning experiments;
- foundation-model adapters;
- retrieval baselines;
- prototype memory;
- ProtoEM-CT E-step or M-step;
- transductive adaptation;
- robustness corruptions;
- uncertainty and calibration experiments;
- external 3D-IRCADb validation;
- LLM/VLM work; and
- claims about segmentation efficacy, generalization, clinical validity, or external validation.

Real development-cohort execution is not part of the initial Phase 3 implementation step. Any
real-data baseline run requires a separately approved later step after smoke, overfit, and contract
verification are complete.

## Phase 3 Branch and Base Commit

- Status: `in progress`
- Base commit: `907ea9d3f559ce959b92bc78c05c75c8187a5f32`
- Branch: `phase/3-baselines`

## Gate 3 Acceptance Criteria

Gate 3 is accepted when both `nnU-Net v2` and `MONAI SegResNet` complete an end-to-end synthetic or
tiny-data path; each path covers preparation, training or tiny overfit, inference, result import,
and metric JSON; CPU smoke tests pass; tiny-subset overfit demonstrates that the training path can
reduce its configured loss; outputs are deterministic where the contract claims determinism;
checkpoint and prediction files remain outside Git; metric JSON is produced from saved predictions
and labels by project code; all repository quality checks and GitHub-hosted CI pass; and Gate 3 is
closed without requiring full training on all 131 development cases.

Phase 4 cannot begin before Gate 3 is merged.

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
