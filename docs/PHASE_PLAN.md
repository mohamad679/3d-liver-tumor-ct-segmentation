# Phase Plan

## Active Phase

Phase 0 is completed.

Phase 1 is completed locally.

Phase 2 Gate 2 close-out is completed locally from the approved real-data development-cohort QA and
leakage artifacts.

Phase 3 Gate 3 baseline scope is treated as completed and merged for later-phase planning purposes
on the user-confirmed repository state. Phase 4 is treated as completed with Gate 4 closed for
planning purposes on the user-confirmed repository state. Phase 5 is treated as completed and
merged for later-phase planning purposes on the user-confirmed repository state. This Phase 6
planning update does not audit, rerun, verify, or modify any completed Phase 5 implementation.

Phase 6 implementation and Gate 6 close-out are completed locally on 2026-08-02.

Phase 7 is user-confirmed complete and merged with Gate 7 passed.

Active phase: Phase 8 Wave 0 planning for external validation only. Wave 0 is documentation-only,
does not access 3D-IRCADb-01 or any external drive, and does not implement source code, tests,
configs, manifests, preregistration artifacts, predictions, metrics, or generated outputs.

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

## Phase 4 Scope

Phase 4 establishes the deterministic few-shot adaptation protocol on the immutable internal
development test cohort only. The objective is protocol definition, support-manifest generation,
adaptation-mode control, and auditable experiment orchestration. Phase 4 must not introduce
ProtoEM-CT, retrieval, prototype memory, robustness corruptions, uncertainty analysis, external
validation, or any Phase 5 or later method work.

Phase 4 includes:

- deterministic patient-level support-manifest generation derived from the existing Phase 2
  development manifest, development split, and lesion-summary artifacts;
- fixed support-set sizes `K = 1, 2, 5, 10, 20`;
- at least three fixed support manifests per `K`;
- strict support/test patient disjointness with the immutable internal-test cohort used only as the
  Phase 4 query/test cohort;
- lesion-burden-aware support selection where feasible from the saved Phase 2 lesion summaries,
  while failing explicitly or falling back to documented deterministic nonstratified ranking when a
  stratum is too small for the requested `K`;
- adaptation-mode support for head-only, decoder-only, and full fine-tuning;
- machine-readable logging of trainable-parameter count and adaptation duration, plus memory usage
  when the runtime can measure it without introducing nondeterministic side effects;
- a deterministic protocol-table artifact enumerating each support manifest, adaptation mode,
  source hashes, and cohort-role assignments;
- CLI and orchestration entry points for generating support manifests and running bounded Phase 4
  few-shot protocol executions; and
- unit, integration, leakage, determinism, and synthetic smoke coverage for the new contracts.

Phase 4 excludes:

- any change to the Phase 2 manifest or split policy artifacts;
- any mutation of the immutable internal-test cohort definition;
- any use of validation or test labels to tune support selection, model choice, or protocol
  selection outside the predeclared deterministic rules;
- any use of the external 3D-IRCADb-01 cohort;
- any robustness, calibration, uncertainty, ensembling, or ablation-report scope beyond the Phase 4
  protocol table and run artifacts; and
- any ProtoEM-CT E-step, M-step, pseudo-labeling, memory-bank, or transductive prototype method.

## Phase 4 Scientific and Data Boundaries

Phase 4 consumes only persisted Phase 2 development artifacts and Phase 3 baseline checkpoints or
model initialization contracts. The immutable internal test cohort remains the only query/test
cohort for Phase 4. Support patients must be drawn from the non-test development partitions only,
with no patient or case overlap against internal test.

Support manifests are patient-level artifacts. Every support manifest must record:

- the exact source development manifest hash;
- the exact source development split hash;
- the exact lesion-summary artifact hash when stratification is used;
- the support-selection policy version;
- the support-manifest seed or ranking seed;
- the requested `K`;
- the manifest replicate identifier; and
- the ordered anonymous patient and case assignments.

Phase 4 may use lesion burden only from saved development-cohort lesion-summary artifacts produced in
Phase 2. No support policy may inspect internal-test labels, internal-test lesion summaries, or any
external labels. If lesion-burden stratification is infeasible for a requested `K` because a
stratum lacks enough training/validation patients, the artifact must record the reason and switch to
the predefined deterministic fallback policy instead of silently changing `K` or sampling rules.

## Phase 4 Planned Implementation Sequence

Each numbered substage is a reviewable implementation unit. Work remains on one named substage at a
time:

1. Define Phase 4 artifact schemas and hashing contracts for support manifests, adaptation configs,
   protocol tables, and few-shot run summaries.
2. Implement deterministic support-candidate extraction from the Phase 2 development manifest,
   development split, and lesion-summary artifacts.
3. Implement lesion-burden stratification buckets and the deterministic fallback path for
   infeasible strata.
4. Generate three or more fixed support manifests for each `K in {1, 2, 5, 10, 20}` with immutable
   replicate identifiers.
5. Add leakage guards that reject any support manifest overlapping the immutable internal-test
   cohort at patient or case level.
6. Add adaptation-mode contracts for head-only, decoder-only, and full fine-tuning, including
   explicit trainable-parameter-group validation and parameter-count logging.
7. Extend the Phase 3 baseline execution path with bounded Phase 4 adaptation-run orchestration,
   duration logging, and memory logging where available.
8. Emit the deterministic protocol-table artifact linking support manifests, adaptation modes,
   checkpoints or initialization sources, and output artifacts.
9. Add Phase 4 CLI commands for support-manifest generation, protocol-table generation, and bounded
   synthetic few-shot smoke execution.
10. Add unit, integration, leakage, determinism, and synthetic smoke tests.
11. Close Gate 4 without expanding into Phase 5 or ProtoEM-CT.

## Phase 4 Planned Artifacts

- `fewshot_support_manifest_v1` artifact for one fixed support cohort replicate
- `fewshot_adaptation_config_v1` artifact for one adaptation mode and runtime contract
- `fewshot_protocol_table_v1` artifact enumerating all planned `K x replicate x adaptation-mode`
  runs
- `fewshot_run_summary_v1` artifact capturing support-manifest hash, adaptation config hash,
  checkpoint or initialization provenance, trainable-parameter count, duration, optional memory
  statistics, metric artifact links, and failure codes when present
- optional deterministic synthetic few-shot fixture artifact if the Phase 3 synthetic dataset is
  extended to exercise support/query adaptation paths without real data

All generated artifacts remain outside Git beneath explicit external output roots.

## Phase 4 Planned Files and Directories

The exact implementation may adjust filenames modestly, but the Phase 4 plan expects work in these
surfaces:

- `configs/phase4_fewshot_protocol.yaml`
- `src/protoem_ct/fewshot/__init__.py`
- `src/protoem_ct/fewshot/artifacts.py`
- `src/protoem_ct/fewshot/support.py`
- `src/protoem_ct/fewshot/stratification.py`
- `src/protoem_ct/fewshot/adaptation.py`
- `src/protoem_ct/fewshot/protocol.py`
- `src/protoem_ct/fewshot/logging.py`
- `src/protoem_ct/cli/main.py`
- `src/protoem_ct/baselines/monai_segresnet.py`
- `src/protoem_ct/baselines/provenance.py`
- `src/protoem_ct/baselines/paths.py`
- `tests/unit/test_phase4_support.py`
- `tests/unit/test_phase4_stratification.py`
- `tests/unit/test_phase4_adaptation.py`
- `tests/unit/test_phase4_protocol_artifacts.py`
- `tests/unit/test_phase4_leakage_guards.py`
- `tests/integration/test_phase4_support_cli.py`
- `tests/integration/test_phase4_protocol_cli.py`
- `tests/integration/test_phase4_synthetic_smoke.py`

If implementation reuses existing Phase 2 or Phase 3 modules instead of introducing a new
`fewshot/` package, the same contracts still apply and the replacement file list must stay narrow
and explicit.

## Phase 4 Adaptation-Mode Contract

Phase 4 defines three allowed adaptation modes:

- `head_only`: only the terminal segmentation head parameters are trainable;
- `decoder_only`: decoder plus segmentation head parameters are trainable while encoder parameters
  remain frozen; and
- `full_finetune`: all model parameters are trainable.

Each mode must:

- expose a deterministic parameter-group selection rule;
- validate that at least one parameter and only the intended parameter groups are trainable;
- emit the exact trainable-parameter count into the run summary artifact; and
- fail explicitly when the selected baseline family cannot represent the requested grouping without
  ambiguous module boundaries.

The initial Phase 4 implementation should target the baseline family whose module structure makes
these groupings explicit and testable. If a second baseline family cannot support the same grouping
contract cleanly, the Phase 4 implementation must record that limitation rather than silently
approximating the mode.

## Phase 4 Protocol-Table Contract

The protocol table is a deterministic machine-readable artifact and, if convenient, a derived
Markdown summary. It must enumerate at minimum:

- support size `K`;
- fixed replicate identifier, with at least three replicates per `K`;
- support-manifest artifact hash;
- support-selection policy version and stratification status;
- adaptation mode;
- source development manifest hash;
- source development split hash;
- immutable internal-test cohort identifier or hash reference;
- initialization or checkpoint provenance reference;
- planned output location identifiers; and
- run status fields once executed.

The protocol table is the canonical Phase 4 inventory of planned few-shot runs. Downstream
reporting must read from this persisted artifact rather than reconstructing the plan ad hoc.

## Phase 4 Planned Verification

- Unit tests for support-manifest schema validation, deterministic serialization, hash stability,
  replicate generation, and ordered assignment behavior
- Unit tests for lesion-burden stratification, explicit infeasibility handling, and deterministic
  fallback selection
- Unit tests for adaptation-mode parameter freezing, trainable-parameter counting, and explicit
  invalid-mode errors
- Unit tests for duration and optional memory logging fields, including the `unavailable` or null
  path when runtime measurement is unsupported
- Leakage tests proving zero support/test patient overlap and zero support/test case overlap for
  every generated support manifest
- Determinism tests proving byte-identical support manifests and protocol tables when rerun with the
  same source artifacts, seeds, and metadata
- Integration tests for CLI generation of support manifests and protocol tables
- Integration tests for bounded synthetic few-shot smoke execution using saved synthetic fixtures and
  no real-data dependency
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src tests`
- targeted pytest selection for the new Phase 4 unit, integration, leakage, determinism, and smoke
  tests

Full-repository test-suite execution is not required by this planning document. The implementation
phase will define the minimal Phase 4-targeted command set needed for Gate 4.

## Phase 4 Risks and Safeguards

- Risk: deterministic support selection can drift if upstream manifest ordering changes.
  Safeguard: derive all rankings from saved artifact hashes, explicit seeds, and canonical sorted
  anonymous identifiers only.
- Risk: lesion-burden stratification can become infeasible for small `K` or skewed lesion
  distributions.
  Safeguard: require explicit feasibility checks and a recorded deterministic fallback policy.
- Risk: support/query leakage can occur through patient duplication, case duplication, or reuse of
  the internal-test cohort.
  Safeguard: enforce patient- and case-level zero-overlap validation before publication and before
  any run starts.
- Risk: adaptation-mode semantics can differ across baseline families.
  Safeguard: require explicit parameter-group validation and reject unsupported groupings.
- Risk: duration and memory metrics can become platform-specific or unavailable.
  Safeguard: log duration deterministically from bounded wall-clock measurement conventions and store
  memory as an explicit optional field with a documented unavailable state.
- Risk: protocol-table regeneration can diverge from executed runs.
  Safeguard: make the protocol table the persisted source of truth and link every run summary back
  to support-manifest and adaptation-config hashes.
- Risk: Phase 4 scope can drift into ProtoEM-CT or later transductive methods.
  Safeguard: keep the implementation limited to deterministic support selection, bounded adaptation
  modes, and auditable orchestration only.

## Gate 4 Acceptance Criteria

Gate 4 is accepted when:

- deterministic patient-level support manifests exist for `K = 1, 2, 5, 10, 20`;
- each `K` has at least three fixed support manifests with stable replicate identifiers;
- every support manifest is linked to the exact source development manifest hash and development
  split hash;
- support patients and cases are disjoint from the immutable internal-test cohort in every
  generated artifact;
- lesion-burden stratification is used where feasible and deterministic fallback behavior is
  explicitly recorded where infeasible;
- head-only, decoder-only, and full-fine-tuning adaptation modes are implemented with explicit
  parameter-group validation;
- trainable-parameter count is logged for every adaptation mode, and duration plus memory are logged
  when available with a documented unavailable state otherwise;
- the deterministic protocol-table artifact enumerates all planned `K x replicate x adaptation-mode`
  runs and their source hashes;
- unit, integration, leakage, determinism, and synthetic smoke tests for the Phase 4 contracts pass;
- no tracked medical data, support manifests containing PHI, checkpoints, predictions, or large
  generated artifacts are Git-visible; and
- Gate 4 close-out remains limited to few-shot protocol definition and bounded adaptation execution,
  without implementing ProtoEM-CT or Phase 5 methods.

## Phase 4 Gate 4 Close-Out Evidence

Verified on 2026-07-30 from branch `phase/4-fewshot-protocol` at user-confirmed `HEAD`
`4f0a356`.

- Deterministic support-manifest generation exists for `K = 1, 2, 5, 10, 20`, with exactly three
  fixed replicates per `K`, for 15 total support manifests.
- Support selection remains patient-level, support/internal-test patient overlap is zero, and
  support/internal-test case overlap is zero in the verified synthetic publication result.
- Lesion-burden stratification and its deterministic fallback are implemented in the Phase 4
  support-generation path and exercised by the passing Phase 4 unit test surface.
- Adaptation modes `head_only`, `decoder_only`, and `full_finetune` are implemented for the
  SegResNet boundary contract. Verified unit-test parameter-count evidence on the deterministic test
  model is:
  - `head_only`: total `98`, trainable `15`, frozen `83`
  - `decoder_only`: total `98`, trainable `38`, frozen `60`
  - `full_finetune`: total `98`, trainable `98`, frozen `0`
- Deterministic adaptation-config construction and deterministic protocol-table generation are
  implemented and verified. The protocol table contains exactly `45` rows for
  `5 K values x 3 replicates x 3 adaptation modes`.
- Deterministic filesystem publication and CLI generation are implemented. One synthetic Phase 4
  publication using existing test-fixture constructors produced:
  - `15` support manifests
  - `45` adaptation configs
  - `45` protocol-table JSON rows
  - `45` protocol-table Markdown data rows
  - zero support/internal-test patient overlap
  - zero support/internal-test case overlap
  - byte-identical artifacts across two separate external output roots

Exact verification commands executed:

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src`
- `uv run pytest -q`
- `uv run protoem-ct generate-phase4-fewshot-protocol --help`
- `mktemp -d /private/tmp/protoem-ct-phase4-gate4.XXXXXX`
- `uv run python -c 'import importlib.util; from pathlib import Path; from protoem_ct.artifacts.phase2_schemas import phase2_artifact_to_json; root = Path("/private/tmp/protoem-ct-phase4-gate4.htyLrT/inputs"); root.mkdir(parents=True, exist_ok=True); helper_path = Path("/Users/mohsenshamsijazeb/Projects/protoem-ct/tests/unit/test_fewshot_protocol.py"); spec = importlib.util.spec_from_file_location("_fewshot_protocol_helpers", helper_path); module = importlib.util.module_from_spec(spec); assert spec is not None and spec.loader is not None; spec.loader.exec_module(module); manifest = module._manifest(); split = module._split(manifest); lesion = module._lesion_artifact(manifest, split); (root / "manifest.json").write_text(phase2_artifact_to_json(manifest), encoding="utf-8"); (root / "split.json").write_text(phase2_artifact_to_json(split), encoding="utf-8"); (root / "lesion.json").write_text(phase2_artifact_to_json(lesion), encoding="utf-8"); print(root / "manifest.json"); print(root / "split.json"); print(root / "lesion.json")'`
- `uv run protoem-ct generate-phase4-fewshot-protocol --manifest /private/tmp/protoem-ct-phase4-gate4.htyLrT/inputs/manifest.json --split /private/tmp/protoem-ct-phase4-gate4.htyLrT/inputs/split.json --lesion-artifact /private/tmp/protoem-ct-phase4-gate4.htyLrT/inputs/lesion.json --output-root /private/tmp/protoem-ct-phase4-gate4.htyLrT/out-a --initialization-reference-type baseline_provenance --initialization-reference-identifier baseline_init_001 --initialization-artifact-sha256 1111111111111111111111111111111111111111111111111111111111111111 --base-seed 1729 --config /Users/mohsenshamsijazeb/Projects/protoem-ct/configs/phase4_fewshot_protocol.yaml`
- `uv run protoem-ct generate-phase4-fewshot-protocol --manifest /private/tmp/protoem-ct-phase4-gate4.htyLrT/inputs/manifest.json --split /private/tmp/protoem-ct-phase4-gate4.htyLrT/inputs/split.json --lesion-artifact /private/tmp/protoem-ct-phase4-gate4.htyLrT/inputs/lesion.json --output-root /private/tmp/protoem-ct-phase4-gate4.htyLrT/out-b --initialization-reference-type baseline_provenance --initialization-reference-identifier baseline_init_001 --initialization-artifact-sha256 1111111111111111111111111111111111111111111111111111111111111111 --base-seed 1729 --config /Users/mohsenshamsijazeb/Projects/protoem-ct/configs/phase4_fewshot_protocol.yaml`
- `uv run python -c 'import json; from pathlib import Path; from protoem_ct.fewshot import fewshot_protocol_table_from_json; root_a = Path("/private/tmp/protoem-ct-phase4-gate4.htyLrT/out-a"); root_b = Path("/private/tmp/protoem-ct-phase4-gate4.htyLrT/out-b"); support_a = sorted(root_a.glob("support_manifests/k*/**/*.json")); support_b = sorted(root_b.glob("support_manifests/k*/**/*.json")); configs_a = sorted((root_a / "adaptation_configs").glob("*.json")); configs_b = sorted((root_b / "adaptation_configs").glob("*.json")); protocol_a = fewshot_protocol_table_from_json((root_a / "protocol" / "fewshot_protocol_table.json").read_bytes()); protocol_b = fewshot_protocol_table_from_json((root_b / "protocol" / "fewshot_protocol_table.json").read_bytes()); md_rows_a = sum(1 for line in (root_a / "protocol" / "fewshot_protocol_table.md").read_text(encoding="utf-8").splitlines() if line.startswith("| ")) - 2; md_rows_b = sum(1 for line in (root_b / "protocol" / "fewshot_protocol_table.md").read_text(encoding="utf-8").splitlines() if line.startswith("| ")) - 2; summary_a = json.loads((root_a / "generation_summary.json").read_text(encoding="utf-8")); summary_b = json.loads((root_b / "generation_summary.json").read_text(encoding="utf-8")); bytes_a = {path.relative_to(root_a).as_posix(): path.read_bytes() for path in root_a.rglob("*") if path.is_file()}; bytes_b = {path.relative_to(root_b).as_posix(): path.read_bytes() for path in root_b.rglob("*") if path.is_file()}; print(json.dumps({"support_manifest_count": len(support_a), "adaptation_config_count": len(configs_a), "protocol_row_count": len(protocol_a.rows), "markdown_data_row_count": md_rows_a, "patient_overlap_count": summary_a["internal_test_patient_overlap_count"], "case_overlap_count": summary_a["internal_test_case_overlap_count"], "leakage_check_passed": summary_a["leakage_check_passed"], "byte_identical_across_output_roots": bytes_a == bytes_b, "generation_summary_equal": summary_a == summary_b, "protocol_rows_equal": len(protocol_a.rows) == len(protocol_b.rows), "support_manifest_count_b": len(support_b), "adaptation_config_count_b": len(configs_b), "protocol_row_count_b": len(protocol_b.rows), "markdown_data_row_count_b": md_rows_b}, sort_keys=True))'`

Exact verified results:

- `uv run ruff check .`: PASS, `All checks passed!`
- `uv run ruff format --check .`: PASS, `132 files already formatted`
- `uv run mypy src`: PASS, `Success: no issues found in 53 source files`
- `uv run pytest -q`: PASS, `769 passed, 2 skipped in 388.27s (0:06:28)`
- `uv run protoem-ct generate-phase4-fewshot-protocol --help`: PASS
- Synthetic Phase 4 publication: PASS
- Synthetic determinism comparison across two external output roots: PASS

Duration and memory remain unavailable for Phase 4 close-out because no adaptation training or
inference execution was performed in Phase 4. This is intentional. The existing
`fewshot_run_summary_v1` schema already supports unavailable execution measurements through nullable
`duration_seconds` and explicit `memory_availability_status` with null memory fields.

## Phase 5 Scope

Phase 5 establishes the retrieval-and-prototype foundation layer that sits between the completed
Phase 4 few-shot protocol and any later transductive refinement. The Phase 5 objective is to build
deterministic feature extraction, deterministic embedding caching, exact nearest-support retrieval,
foreground/background prototype memory, prototype-only inference, and a reproducible comparator
between no-retrieval and nearest-support retrieval. Phase 5 must remain non-transductive.

Phase 5 includes:

- modular `FeatureEncoder3D` and `PromptableSegmenter3D` interfaces with narrow typed contracts;
- one explicit SegResNet feature adapter built on the existing MONAI SegResNet baseline family;
- a deterministic embedding cache keyed by preprocessing hash, checkpoint hash, encoder identity,
  and input identity;
- machine-readable embedding and cache metadata artifacts using canonical JSON and stable SHA-256
  hashing;
- exact cosine-similarity retrieval over persisted support embeddings;
- deterministic foreground/background support prototype construction;
- prototype-only inference with no learnable updates;
- a deterministic no-retrieval versus nearest-support comparison artifact;
- CLI and publication entry points for cache generation, retrieval/prototype publication, and
  bounded synthetic smoke execution; and
- unit, integration, leakage, determinism, and CPU synthetic smoke coverage for the Phase 5
  contracts.

Phase 5 excludes:

- any learnable transductive update on query features or logits;
- any EM-like refinement, E-step, M-step, entropy objective, balance objective, consistency term,
  proximal term, or other iterative self-training procedure;
- any implementation of Phase 6;
- any external 3D-IRCADb-01 validation;
- any mutation of the immutable internal-test definition established earlier;
- any use of external labels for design, retrieval tuning, support selection, or comparator
  selection; and
- any optional foundation-model adapter unless code availability, licensing, practical weights, and
  practical compute feasibility are explicitly verified first.

## Phase 5 Scientific and Data Boundaries

Phase 5 consumes only persisted Phase 2 development artifacts, the immutable internal-test cohort
definition, completed Phase 4 support-manifest artifacts, completed Phase 4 protocol artifacts where
relevant, and existing Phase 3/Phase 4 baseline checkpoints or initialization references. Support
patients remain drawn only from non-test development partitions. Query evaluation remains confined to
the immutable internal-test cohort. Support/query patient disjointness and support/query case
disjointness remain mandatory.

Phase 5 does not update model weights, support embeddings, query embeddings, or prototypes using
query labels or pseudo-labels. Query features may be encoded and compared, but they must not drive
learnable parameter updates in this phase. Any transductive refinement trace belongs to Phase 6 and
must not be claimed, fabricated, or partially implemented in Phase 5.

The optional foundation-model adapter remains out of scope unless all of the following are verified
practical before implementation: usable local code path, acceptable license, compatible weights,
tractable CPU/GPU memory and runtime, and a repository-safe way to reference the adapter without
committing weights. Until then, Phase 5 planning is anchored on the SegResNet feature adapter only.

## Phase 5 Planned Implementation Sequence

Each numbered substage is a reviewable implementation unit. Work remains on one named substage at a
time:

1. Define `FeatureEncoder3D` and `PromptableSegmenter3D` interfaces plus typed retrieval/prototype
   contracts.
2. Implement the SegResNet feature adapter using explicit existing SegResNet feature boundaries.
3. Define embedding-artifact and cache-schema contracts plus their canonical hash payloads.
4. Implement deterministic embedding-cache validation, path safety, and external publication.
5. Implement exact cosine-similarity retrieval with deterministic ordering and tie-breaking.
6. Implement support prototype memory construction for foreground and background classes.
7. Implement prototype-only inference with explicit empty-foreground handling.
8. Implement the no-retrieval versus nearest-support comparison artifact and summary surface.
9. Add CLI/publication paths and bounded CPU synthetic smoke execution for the Phase 5 contracts.
10. Run final Phase 5 verification and evaluate Gate 5 without implementing any Phase 6 method.

## Phase 5 Planned Artifacts

- `retrieval_embedding_manifest_v1` or equivalent artifact enumerating one deterministic embedding
  cache publication
- `retrieval_embedding_record_v1` or equivalent per-input embedding metadata contract keyed by
  preprocessing hash, checkpoint hash, encoder identity, and input identity
- `prototype_memory_v1` artifact recording deterministic foreground/background prototype summaries
  for one support set
- `retrieval_result_v1` artifact recording exact cosine scores, ordered nearest supports, and
  deterministic tie resolution
- `prototype_inference_summary_v1` artifact recording one prototype-only query inference contract
  without inventing metrics
- `retrieval_comparison_table_v1` artifact comparing no-retrieval and nearest-support retrieval
  configurations without implying transductive updates
- optional derived Markdown summaries generated only from saved JSON artifacts

All generated artifacts remain outside Git beneath explicit external output roots.

## Phase 5 Planned Files and Directories

The exact implementation may adjust filenames modestly, but the Phase 5 plan expects work in these
surfaces while avoiding broad refactors:

- `docs/PHASE_PLAN.md`
- `configs/phase5_foundation_retrieval.yaml`
- `src/protoem_ct/models/__init__.py`
- `src/protoem_ct/models/interfaces.py`
- `src/protoem_ct/models/segresnet_adapter.py`
- `src/protoem_ct/retrieval/__init__.py`
- `src/protoem_ct/retrieval/artifacts.py`
- `src/protoem_ct/retrieval/cache.py`
- `src/protoem_ct/retrieval/cosine.py`
- `src/protoem_ct/retrieval/prototypes.py`
- `src/protoem_ct/retrieval/inference.py`
- `src/protoem_ct/retrieval/publication.py`
- `src/protoem_ct/cli/main.py`
- `src/protoem_ct/baselines/monai_segresnet.py`
- `src/protoem_ct/baselines/provenance.py`
- `src/protoem_ct/artifacts/hashing.py`
- `tests/unit/test_phase5_interfaces.py`
- `tests/unit/test_phase5_segresnet_adapter.py`
- `tests/unit/test_phase5_embedding_artifacts.py`
- `tests/unit/test_phase5_cache.py`
- `tests/unit/test_phase5_cosine_retrieval.py`
- `tests/unit/test_phase5_prototypes.py`
- `tests/unit/test_phase5_inference.py`
- `tests/unit/test_phase5_comparison.py`
- `tests/unit/test_phase5_leakage.py`
- `tests/integration/test_phase5_cli.py`
- `tests/integration/test_phase5_publication.py`
- `tests/smoke/test_phase5_retrieval_smoke.py`

The preferred implementation location is the existing `src/protoem_ct/models` and
`src/protoem_ct/retrieval` packages, which are currently narrow placeholders and therefore allow
Phase 5 to be added without disturbing completed Phase 3 and Phase 4 modules.

## Phase 5 Interface and Scientific Contracts

### FeatureEncoder3D

`FeatureEncoder3D` should expose a pure inference-time contract that accepts one preprocessed 3D CT
input tensor plus explicit encoder metadata and returns one deterministic dense feature tensor plus a
machine-readable encoder descriptor. The returned feature tensor must have explicit channel-first
shape `(C, D_f, H_f, W_f)` with a documented spatial downsampling relation to the preprocessed input
shape `(1, D, H, W)`.

For the initial Phase 5 implementation, the SegResNet adapter should define one explicit feature
boundary and keep it fixed for the phase. The contract should document whether the chosen features
come from the final decoder stage before the segmentation head or from another explicit internal
stage. The adapter must fail explicitly if the expected SegResNet boundaries are absent or
ambiguous.

### PromptableSegmenter3D

`PromptableSegmenter3D` should expose a deterministic inference-only contract that accepts one query
feature tensor and one support-derived prompt object, then returns one finite query logit tensor or
probability tensor with explicit spatial shape. In Phase 5 the prompt object is limited to support
prototypes; it must not include learnable query-conditioned updates.

### Feature normalization

Feature normalization should be fixed and explicit. The recommended Phase 5 default is per-voxel
L2-normalization across the feature-channel dimension after feature extraction and before cosine
comparison or prototype construction. The implementation plan should treat this normalization rule
as part of the artifact identity because changing it changes cache validity and retrieval outcomes.

### Cosine similarity

Cosine similarity should be defined exactly as the dot product between two L2-normalized embedding
vectors. If embeddings are compared after spatial pooling, the pooling rule must be fixed first and
recorded in the artifact schema. If embeddings remain dense, the reduction rule over spatial
positions must be fixed and recorded explicitly rather than inferred.

Tie-breaking must be deterministic. The planned rule is descending cosine score, then ascending
anonymous support patient ID, then ascending anonymous support case ID, then ascending support
manifest assignment index. No Python set/dict iteration order may affect retrieval rank.

### Foreground/background prototype construction

Support prototype memory should be built deterministically from support features and support masks
only. The Phase 5 foreground prototype is the mean of all normalized support feature vectors whose
support mask voxel is foreground tumor after any documented feature-resolution alignment step. The
background prototype is the mean of all normalized support feature vectors whose aligned support mask
voxel is background. If feature-resolution alignment requires downsampling masks, the method must be
fixed, explicit, and deterministic.

### Empty-foreground support handling

If a selected support case has no foreground tumor voxels after aligned mask projection, the case
must not silently fabricate a foreground prototype contribution. The artifact must record empty-mask
status explicitly. The planned fallback is to exclude that support case from the foreground prototype
accumulator while retaining its background contribution, then fail explicitly if all support cases
for that support set are foreground-empty.

### Query inference rule

Prototype-only inference should compute query feature vectors, normalize them with the same fixed
rule, and produce class scores from similarity to the background and foreground prototypes without
any gradient-based or iterative update. The initial recommended rule is per-voxel cosine similarity
to the two class prototypes followed by a deterministic two-class argmax or equivalent deterministic
probability mapping. Phase 5 should record whichever rule is chosen and keep it fixed for the phase.

### No-retrieval and nearest-support comparators

The no-retrieval comparator should use the full fixed support set attached to the chosen Phase 4
support manifest for prototype construction, without ranking or filtering by query-support
similarity. The nearest-support comparator should rank support cases by the exact cosine retrieval
rule and build prototypes from the top-ranked support subset under a fixed deterministic policy that
is declared up front in configuration and artifacts. If the initial implementation uses
nearest-support retrieval with one top-ranked support case, that fact must be stated explicitly and
must not be generalized in documentation beyond what is implemented.

### Patient-disjoint and immutable internal-test requirements

Every published Phase 5 artifact must retain links back to the exact support manifest hash, source
development manifest hash, source development split hash, and immutable internal-test cohort hash.
Support and query identities must remain anonymous and patient-disjoint. Any cache or retrieval plan
that mixes support and internal-test identities must fail explicitly.

## Phase 5 Reproducibility Contracts

- Use existing canonical JSON and SHA-256 hashing utilities rather than introducing a second hashing
  stack.
- Derive artifact hashes only from canonical payloads with deterministic key ordering.
- Keep deterministic ordering explicit for embedding records, retrieval result rows, prototype
  members, comparator rows, and any Markdown derived from JSON artifacts.
- Do not place timestamps, hostnames, usernames, environment-specific values, or absolute local
  paths inside deterministic Phase 5 artifacts.
- The embedding cache key must include at least preprocessing hash, checkpoint hash, encoder
  identity, encoder configuration identity, normalization identity, and input identity.
- Changing preprocessing hash or checkpoint hash must invalidate cache reuse deterministically.
- Repeated publication into separate empty output roots with identical explicit inputs must produce
  byte-identical deterministic JSON artifacts.
- Write generated artifacts only beneath explicit external output roots with path-containment and
  symlink-escape guards.
- Do not commit medical data, embeddings, predictions, checkpoints, model weights, or other large
  generated artifacts to Git.

## Phase 5 Planned Verification

- Interface-conformance tests for `FeatureEncoder3D` and `PromptableSegmenter3D`
- Adapter tests proving explicit SegResNet feature output shape and boundary selection
- Deterministic embedding tests for identical explicit inputs
- Cache hit/miss and cache invalidation tests when preprocessing hash or checkpoint hash changes
- Cosine retrieval correctness tests with fixed synthetic embeddings
- Deterministic tie-handling tests
- Prototype construction tests for foreground/background prototypes
- Empty-mask behavior tests proving explicit exclusion/failure semantics
- Prototype-only inference tests for output shape and finite-valued outputs
- No-retrieval versus nearest-support comparison artifact tests
- Leakage tests proving zero support/internal-test patient overlap and zero case overlap in all
  published Phase 5 artifacts
- Integration tests for deterministic CLI/publication paths
- CPU synthetic smoke tests for the end-to-end Phase 5 retrieval/prototype publication surface
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src tests`
- targeted pytest selection for the new Phase 5 unit, integration, leakage, determinism, and smoke
  tests

Full-repository training or real-data inference is not part of the Phase 5 planning scope. Any
Phase 5 smoke path must stay synthetic and CPU-bounded unless the user later authorizes a distinct
execution step.

## Phase 5 Risks and Safeguards

- Risk: SegResNet internal feature boundaries may be ambiguous or unstable across versions.
  Safeguard: use one explicit adapter contract tied to known attributes and fail if the boundary is
  missing or ambiguous.
- Risk: feature-map resolution may not align cleanly with support masks.
  Safeguard: document one deterministic mask-to-feature alignment rule and test it on synthetic
  fixtures.
- Risk: dense 3D embeddings can be large and expensive to cache.
  Safeguard: keep artifact schemas explicit about shape, datatype, and cache scope, and prefer a
  narrow first implementation before adding alternate storage strategies.
- Risk: empty tumor masks can invalidate foreground prototype construction.
  Safeguard: record empty-mask status explicitly and fail when no valid foreground prototype can be
  formed.
- Risk: cache contamination can silently mix embeddings from incompatible preprocessing or
  checkpoints.
  Safeguard: encode preprocessing hash, checkpoint hash, encoder identity, and normalization rule
  directly into cache identity and validation.
- Risk: Phase 5 can drift into Phase 6 transductive refinement.
  Safeguard: prohibit learnable updates, EM-like refinement, iterative pseudo-labeling, and any
  objective trace that belongs to Phase 6.
- Risk: optional foundation-model weights may be impractical or unavailable.
  Safeguard: keep the foundation-model adapter excluded unless code, license, weights, and compute
  are explicitly verified practical first.

## Gate 5 Scope Interpretation

The blueprint wording for Gate 5 is broader than the intended Phase 5 implementation scope in this
repository. For this project, Gate 5 should be interpreted as the non-transductive
retrieval-and-prototype foundation milestone, not as evidence of Phase 6 transductive refinement.

Phase 5 acceptance should therefore require:

- reproducible feature-encoder and promptable-segmenter interfaces;
- a valid explicit SegResNet feature adapter;
- reproducible embedding-cache artifacts with deterministic invalidation;
- exact cosine retrieval with deterministic tie handling;
- reproducible foreground/background prototype artifacts;
- valid prototype-only inference artifacts with finite outputs;
- a deterministic no-retrieval versus nearest-support comparison artifact;
- leakage-safe support/query separation tied to the immutable internal-test cohort; and
- passing Phase 5 unit, integration, determinism, leakage, and synthetic smoke verification.

Phase 5 acceptance must not claim:

- learnable transductive adaptation;
- EM/E-step/M-step behavior;
- entropy or balance optimization;
- iterative refinement traces;
- objective improvements attributable to Phase 6 methods; or
- external-validation findings.

Those broader transductive refinement traces belong to Phase 6 and must remain explicitly out of
scope for Phase 5 planning, implementation, testing, and gate documentation.

## Phase 6 Scope

Phase 6 is limited to ProtoEM-CT objective-driven transductive adaptation on the immutable internal
test cohort using the completed Phase 5 retrieval-and-prototype foundation. Phase 6 does not
include any Phase 7 work. In particular, Phase 6 must not include robustness corruptions, stress
testing, uncertainty analysis, calibration, external validation, reporting claims beyond the
internal development setting, or any new method family that is not part of the ProtoEM-CT
objective-driven transductive adaptation path.

Phase 6 includes:

- deterministic objective contracts for ProtoEM-CT transductive adaptation built on frozen Phase 5
  retrieval/prototype components;
- explicit transductive state schemas for query-conditioned adaptation traces, objective terms, and
  iteration summaries;
- bounded query-time optimization over the declared ProtoEM-CT objective using support-derived
  supervision and unlabeled query volumes only;
- deterministic initialization from Phase 5 prototype-only inference and retrieval artifacts;
- explicit stopping, failure, and non-convergence behavior;
- deterministic publication of adaptation summaries and internal-test evaluation artifacts; and
- unit, integration, determinism, leakage, and bounded synthetic smoke coverage for the Phase 6
  contracts.

Phase 6 excludes:

- any change to completed Phase 5 retrieval, cache, prototype, or baseline implementation beyond
  the narrow integration hooks required to call them;
- any new CLI surface, config expansion, artifact publication, or execution path in this planning
  substage;
- any external 3D-IRCADb-01 access or validation;
- any Phase 7 robustness, corruption, uncertainty, calibration, or ablation work;
- any use of external labels for tuning, selection, or thresholding; and
- any claim that Phase 6 planning alone establishes scientific efficacy.

## Phase 6 Scientific and Data Boundaries

Phase 6 consumes only persisted Phase 2 development artifacts, the immutable internal-test cohort
definition, completed Phase 4 support-manifest/protocol artifacts, completed Phase 5 retrieval and
prototype artifacts, and existing Phase 3/Phase 4 baseline checkpoints or initialization
references. Query adaptation is transductive with respect to unlabeled internal-test query volumes
only. Support labels remain the only label source allowed inside the adaptation objective.

Phase 6 must keep support/query patient disjointness and support/query case disjointness intact. No
query ground-truth labels may enter objective construction, stopping decisions, hyperparameter
selection, checkpoint selection, retrieval tuning, or adaptation-mode selection. Any adaptation
trace must remain auditable from persisted artifacts rather than transient logs or in-memory state.

The planned Phase 6 objective is ProtoEM-CT-specific and must be recorded as explicit weighted
terms with stable naming, deterministic default coefficients, and documented finite-value guards.
Changing term definitions or coefficients changes artifact identity and invalidates direct
comparisons across runs.

## Phase 6 Planned Implementation Sequence

Each numbered substage is a reviewable implementation unit. Work remains on one named substage at a
time in this order:

0. Planning only: define the complete Phase 6 implementation plan in `docs/PHASE_PLAN.md` without
   modifying code, tests, configs, CLI, artifacts, objectives, EM steps, or inference behavior.
1. Define Phase 6 artifact schemas, canonical JSON payloads, and hashing contracts for
   transductive objective configuration, iteration summaries, run summaries, and failure records.
2. Define the narrow Phase 6 integration boundary that reads completed Phase 5 retrieval/prototype
   outputs and produces ProtoEM-CT adaptation inputs without changing Phase 5 semantics.
3. Define the ProtoEM-CT objective contract, including named objective terms, coefficient schema,
   finite-value checks, explicit reduction rules, and deterministic empty-case behavior.
4. Implement deterministic transductive state containers and initialization from Phase 5
   prototype-only query predictions, support prototypes, and retrieval results.
5. Implement bounded iterative adaptation orchestration with explicit iteration order, objective
   evaluation order, stopping criteria, max-iteration handling, and failure/non-convergence status.
6. Implement objective bookkeeping and persisted per-iteration summaries, including total objective,
   per-term values, update magnitude summaries, and termination reason.
7. Implement deterministic final prediction/export wiring for the adapted internal-test query path
   while preserving existing Phase 5 behavior for non-adaptive baselines.
8. Implement publication/reporting surfaces for Phase 6 JSON artifacts and any derived Markdown
   summaries generated only from saved JSON artifacts.
9. Add targeted unit, integration, determinism, leakage, and bounded synthetic smoke tests for the
   Phase 6 contracts.
10. Run final Phase 6 verification and evaluate Gate 6 without implementing any Phase 7 method.

## Phase 6 Planned Artifacts

- `protoem_ct_objective_config_v1` artifact recording the declared Phase 6 objective terms,
  coefficients, initialization identity, and stopping policy
- `protoem_ct_iteration_summary_v1` artifact recording one deterministic adaptation iteration with
  total objective, per-term values, convergence statistics, and failure flags
- `protoem_ct_run_summary_v1` artifact recording one complete Phase 6 transductive run with source
  hashes, support/query identities, stopping reason, iteration count, and output references
- `protoem_ct_failure_record_v1` artifact recording explicit invalid-input, non-finite,
  non-convergent, or unsupported-state failures
- optional derived Markdown summaries generated only from saved JSON artifacts

All generated artifacts remain outside Git beneath explicit external output roots.

## Phase 6 Planned Files and Directories

The exact implementation may adjust filenames modestly, but the Phase 6 plan expects work in these
surfaces while keeping Phase 5 implementation unchanged:

- `docs/PHASE_PLAN.md`
- `configs/phase6_protoem_ct.yaml`
- `src/protoem_ct/protoem/__init__.py`
- `src/protoem_ct/protoem/artifacts.py`
- `src/protoem_ct/protoem/contracts.py`
- `src/protoem_ct/protoem/objective.py`
- `src/protoem_ct/protoem/state.py`
- `src/protoem_ct/protoem/initialize.py`
- `src/protoem_ct/protoem/optimize.py`
- `src/protoem_ct/protoem/stopping.py`
- `src/protoem_ct/protoem/publication.py`
- `src/protoem_ct/protoem/inference.py`
- `src/protoem_ct/retrieval/publication.py`
- `src/protoem_ct/retrieval/inference.py`
- `src/protoem_ct/cli/main.py`
- `src/protoem_ct/artifacts/hashing.py`
- `tests/unit/test_phase6_artifacts.py`
- `tests/unit/test_phase6_objective.py`
- `tests/unit/test_phase6_state.py`
- `tests/unit/test_phase6_initialize.py`
- `tests/unit/test_phase6_optimize.py`
- `tests/unit/test_phase6_stopping.py`
- `tests/unit/test_phase6_failures.py`
- `tests/unit/test_phase6_leakage.py`
- `tests/integration/test_phase6_cli.py`
- `tests/integration/test_phase6_publication.py`
- `tests/integration/test_phase6_transductive_flow.py`
- `tests/smoke/test_phase6_protoem_smoke.py`

## Phase 6 Objective and Orchestration Contracts

Phase 6 must define the ProtoEM-CT objective as a fixed named collection of terms rather than an
implicit computation embedded inside an optimization loop. Each term must state:

- required inputs and their provenance;
- tensor reduction rule and normalization rule;
- coefficient name and default value;
- valid value range or finite-value expectation; and
- failure behavior when required inputs are absent or degenerate.

The orchestration contract must keep iteration order deterministic. Given identical explicit inputs,
identical source artifacts, identical coefficients, and identical iteration limits, the saved
iteration summaries and final run summary must be byte-identical where the implementation claims
determinism.

Stopping behavior must be explicit. The planned contract requires a bounded maximum iteration count,
an explicit improvement threshold or equivalent stopping criterion, and distinct terminal statuses
for converged, max-iteration-reached, invalid-state, and failed runs. Silent early termination is
not allowed.

Phase 6 must preserve a clean comparison boundary against completed Phase 5 artifacts. The
prototype-only Phase 5 path remains the non-adaptive reference. Phase 6 outputs must record the
exact source Phase 5 artifact identities used for initialization so comparisons are traceable and do
not blur the boundary between non-transductive retrieval and ProtoEM-CT transductive adaptation.

## Phase 6 Planned Verification

- Artifact-schema and canonical-serialization tests for all Phase 6 JSON contracts
- Objective-term tests for deterministic values, finite-value guards, and explicit degenerate-case
  handling
- Initialization tests proving deterministic construction from fixed Phase 5 retrieval/prototype
  artifacts
- Optimization-loop tests for iteration ordering, objective bookkeeping, stopping behavior, and
  failure propagation
- Determinism tests proving byte-identical iteration summaries and run summaries for identical
  explicit inputs
- Leakage tests proving no query-label usage, no support/query identity overlap, and no mutation of
  immutable internal-test assignments
- Integration tests for the end-to-end transductive flow using synthetic fixtures and bounded CPU
  execution only
- Synthetic smoke tests for a minimal ProtoEM-CT run surface
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src tests`
- targeted pytest selection for the new Phase 6 unit, integration, leakage, determinism, and smoke
  tests

Full development-cohort training, external validation, Phase 7 robustness execution, and long GPU
runs are not part of the Phase 6 planning scope.

## Phase 6 Risks and Safeguards

- Risk: Phase 6 objective scope can drift into Phase 7 robustness or uncertainty work.
  Safeguard: keep the plan explicitly limited to ProtoEM-CT objective-driven transductive
  adaptation and exclude all Phase 7 surfaces by name.
- Risk: transductive adaptation can accidentally consume query labels or hidden evaluation signals.
  Safeguard: require explicit leakage tests, auditable source-hash linkage, and no query-label
  inputs anywhere in the adaptation contract.
- Risk: iterative optimization can become nondeterministic.
  Safeguard: make iteration order, stopping rules, and artifact serialization explicit and test
  byte-identical outputs where determinism is claimed.
- Risk: objective terms can become numerically unstable or non-finite on degenerate cases.
  Safeguard: require per-term finite-value guards, explicit empty-case handling, and persisted
  failure records.
- Risk: Phase 6 integration can silently alter Phase 5 behavior.
  Safeguard: keep a narrow initialization boundary, record source Phase 5 artifact hashes, and keep
  the Phase 5 non-adaptive path as a reference.
- Risk: transductive state artifacts can become large or difficult to audit.
  Safeguard: persist compact structured summaries by iteration and keep raw intermediate tensors out
  of Git and out of required deterministic artifacts.

## Gate 6 Acceptance Criteria

Gate 6 is accepted when:

- the Phase 6 implementation remains explicitly limited to ProtoEM-CT objective-driven
  transductive adaptation and does not include any Phase 7 work;
- deterministic Phase 6 artifact schemas exist for objective configuration, iteration summaries, run
  summaries, and failure records;
- the ProtoEM-CT objective contract is explicit, finite-guarded, and recorded by named terms with
  deterministic coefficients;
- deterministic initialization from completed Phase 5 retrieval/prototype artifacts is implemented
  and source hashes are recorded in Phase 6 run artifacts;
- bounded iterative adaptation with explicit stopping and failure statuses is implemented;
- deterministic final adapted internal-test query outputs and machine-readable Phase 6 summaries are
  produced from saved artifacts;
- unit, integration, determinism, leakage, and bounded synthetic smoke tests for the Phase 6
  contracts pass;
- no tracked medical data, PHI-bearing manifests, embeddings, predictions, checkpoints, or large
  generated artifacts are Git-visible; and
- Gate 6 close-out does not claim external validation, robustness findings, uncertainty findings,
  calibration findings, or any Phase 7 result.

### Gate 6 Close-Out Status

Gate 6 passed locally on 2026-08-02.

Verified close-out evidence:

- `uv run ruff check .`: PASS
- `uv run ruff format --check .`: PASS after one formatting-only fix in
  `tests/unit/test_phase6_inference.py`
- `uv run mypy src`: PASS
- `uv run pytest -q`: PASS, `1115 passed, 3 skipped`
- `uv run pre-commit run --all-files`: PASS
- `uv run protoem-ct run-phase6-protoem --help`: PASS
- Two independent synthetic `run-phase6-protoem` executions completed successfully
- JSON and Markdown publication artifacts were byte-identical across the two synthetic runs
- The synthetic objective trace was non-empty, zero-based, contiguous, and self-validating
- The synthetic stopping record was self-validating with stop reason `max_iterations`
- The synthetic ablation comparison contained exactly 12 records in canonical order with explicit
  `executed`, `phase5_baseline_link`, `unsupported`, and `provenance_only` statuses
- Query labels and query reference masks remained outside initialization and optimization APIs

This close-out remains limited to ProtoEM-CT objective-driven transductive adaptation on bounded
synthetic executions. It does not claim theoretical convergence, real-data Phase 6 efficacy, GPU
execution, robustness, uncertainty, calibration, external validation, or any Phase 7 result.

## Phase 7 Scope

Phase 7 is limited to deterministic robustness and uncertainty assessment on the completed internal
development workflow. Phase 7 consumes completed Phase 6 prediction and publication contracts where
needed, but it must not re-audit, rerun, redesign, or modify completed Phase 6 scientific behavior.

Phase 7 includes:

- deterministic robustness transforms for HU-window shift, intensity scale and offset, contrast
  shift, Gaussian noise, Gaussian blur, slice-thickness simulation, anisotropic downsampling and
  resampling, and crop/FOV perturbation;
- explicit geometry-safety validation for image/mask alignment, affine, spacing, orientation,
  shape, interpolation mode, binary-mask preservation, common-grid restoration, and geometry-change
  records;
- predictive-entropy uncertainty maps and a deterministic TTA-variance interface or small
  deterministic ensemble interface;
- deterministic calibration, risk-coverage, uncertainty-error correlation, statistically eligible
  failure-detection AUROC, lesion-size subgroup, absolute-degradation, and relative-degradation
  artifacts;
- versioned corruption manifests with persisted severity parameters, deterministic seeds, canonical
  hashes, and machine-readable artifacts;
- JSON-first publication where plots, Markdown, and tables are derived only from validated persisted
  artifacts; and
- bounded CPU synthetic workflow coverage for the complete Phase 7 contract.

Phase 7 explicitly excludes:

- Phase 8 external validation and any access to real 3D-IRCADb-01;
- real-data claims, GPU claims, checkpoint selection, model retraining, or hyperparameter tuning on
  internal test data;
- LLM/VLM work;
- theoretical robustness, calibration, uncertainty, or clinical-validity claims;
- external labels for tuning, model selection, threshold selection, prompt construction, or protocol
  iteration;
- generated-artifact commits, including generated images, masks, predictions, embeddings, reports,
  weights, checkpoints, medical data, or large files; and
- any modification of completed Phase 5 or Phase 6 scientific behavior.

## Phase 7 Dependency-Aware Implementation Sequence

Each numbered substage is a reviewable unit. Work remains on one named substage at a time, and a
downstream substage may not infer or invent an upstream contract before that dependency is approved.

0. Planning only: define the complete Phase 7 implementation plan in `docs/PHASE_PLAN.md` without
   modifying source, tests, configs, CLI, publication, artifacts, transforms, uncertainty
   computation, or evaluation behavior.
1. Artifact contracts: define versioned immutable schemas, strict mapping reconstruction, canonical
   JSON payloads, and self-hash validation for corruption specifications, corruption manifests,
   transform results, geometry records, uncertainty results, calibration results, risk-coverage
   results, degradation results, lesion subgroup results, and Phase 7 run summaries.
2. Geometry contracts: define pure geometry validators, image/mask spatial-alignment checks,
   common-grid restoration contracts, interpolation policy, binary-mask preservation checks, and
   explicit permitted geometry-change records.
3. Uncertainty contracts: define immutable contracts for binary predictive distributions, entropy
   maps, deterministic TTA samples, uncertainty summaries, and evaluation eligibility records.
4. Intensity transforms: implement HU-window shift, intensity scale, intensity offset, and contrast
   shift using deterministic severity-to-parameter mappings and persisted transform parameters.
5. Noise and blur transforms: implement Gaussian noise with local seeded RNG only and Gaussian blur
   with deterministic sigma parameters, finite-output validation, and no mask filtering.
6. Resampling transforms: implement slice-thickness simulation and anisotropic
   downsampling/resampling with explicit spacing changes, image interpolation, nearest-neighbor mask
   interpolation, and common-grid restoration.
7. Crop/FOV transforms: implement deterministic crop/FOV perturbation with persisted crop
   parameters, explicit padding/restoration, no silent label removal, and empty-lesion handling.
8. Predictive entropy: implement binary predictive entropy from validated probabilities with
   explicit zero-probability handling and deterministic finite outputs.
9. TTA or ensemble uncertainty: implement deterministic TTA variance or a small deterministic
   ensemble interface with explicit sample manifests, common-grid alignment before aggregation, and
   mask-free prediction APIs.
10. Calibration evaluation: implement deterministic calibration artifacts using the exact binning
    and expected-calibration-error formula defined below, with explicit degenerate-bin handling.
11. Risk-coverage evaluation: implement deterministic risk-coverage artifacts using uncertainty
    ordering, stable tie-breaking, coverage definitions, and risk definitions recorded below.
12. Failure analysis: implement uncertainty-error correlation and failure-detection AUROC with
    strict eligibility checks for sample count and class diversity. Ineligible AUROC must be
    unavailable, not fabricated.
13. Subgroup and degradation analysis: implement lesion-size subgroup analysis, absolute
    degradation, relative degradation, valid-case counts, empty-lesion counts, and near-zero
    baseline handling.
14. Publication: implement JSON-first atomic publication, deterministic Markdown/tables/plots from
    validated JSON only, path/symlink safety, and no generated artifacts in Git.
15. CLI and synthetic smoke: add a versioned Phase 7 config and bounded CPU synthetic CLI execution
    that covers every required corruption family, uncertainty artifact, and evaluation artifact
    without real data, downloaded checkpoints, GPU use, or external cohort access.
16. Leakage safety: verify that labels enter only evaluation paths, transforms and uncertainty
    prediction APIs do not use labels for prediction, no Phase 8 or real external-cohort access
    exists, and no generated artifacts are Git-visible.
17. Independent scientific review: read-only review first for formula correctness, corruption
    definitions, geometry safety, uncertainty validity, calibration validity, AUROC eligibility,
    degradation interpretation, scope boundaries, and Gate 7 recommendation.
18. Independent engineering review: read-only review first for determinism, canonical hashes,
    serialization, typing, atomic publication, path safety, test coverage, Git hygiene, and Gate 7
    recommendation.
19. Gate 7 close-out: run final repository verification once, execute two independent synthetic
    Phase 7 runs, compare deterministic artifacts, validate geometry safety and corruption-manifest
    reproducibility, update close-out documentation only after verified evidence, and do not begin
    Phase 8.

### Phase 7 Dependency Graph

- Wave 0: substage 0.
- Wave 1: substages 1, 2, and 3 may proceed independently after planning approval.
- Wave 2: substages 4, 5, 6, and 7 depend on approved artifact and geometry contracts.
- Wave 3: substage 8 depends on approved uncertainty and artifact contracts; substage 9 depends on
  approved uncertainty contracts, approved transform implementations, and approved geometry
  contracts.
- Wave 4: substages 10 and 11 depend on approved entropy/TTA or ensemble outputs and artifact
  contracts; substage 12 depends on approved uncertainty implementations and artifact contracts;
  substage 13 depends on approved transform implementations and artifact contracts.
- Wave 5: substage 14 depends on all approved implementation and evaluation contracts; substage 15
  depends on publication and all core implementation agents; substage 16 depends on integrated
  Phase 7 implementation.
- Wave 6: substages 17 and 18 depend on complete integrated implementation; substage 19 depends on
  approved scientific and engineering reviews with all blockers resolved.

## Phase 7 Planned Files and Directories

The exact implementation may adjust filenames modestly, but Phase 7 planning expects work in these
surfaces while preserving completed Phase 5 and Phase 6 behavior:

- `configs/phase7_robustness_uncertainty.yaml`
- `src/protoem_ct/robustness/__init__.py`
- `src/protoem_ct/robustness/artifacts.py`
- `src/protoem_ct/robustness/geometry.py`
- `src/protoem_ct/robustness/intensity.py`
- `src/protoem_ct/robustness/noise.py`
- `src/protoem_ct/robustness/blur.py`
- `src/protoem_ct/robustness/resampling.py`
- `src/protoem_ct/robustness/crop.py`
- `src/protoem_ct/robustness/publication.py`
- `src/protoem_ct/uncertainty/__init__.py`
- `src/protoem_ct/uncertainty/artifacts.py`
- `src/protoem_ct/uncertainty/contracts.py`
- `src/protoem_ct/uncertainty/entropy.py`
- `src/protoem_ct/uncertainty/tta.py`
- `src/protoem_ct/uncertainty/publication.py`
- `src/protoem_ct/evaluation/calibration.py`
- `src/protoem_ct/evaluation/risk_coverage.py`
- `src/protoem_ct/evaluation/failure_detection.py`
- `src/protoem_ct/evaluation/degradation.py`
- `src/protoem_ct/evaluation/subgroups.py`
- `src/protoem_ct/cli/main.py`
- `tests/unit/test_phase7_artifacts.py`
- `tests/unit/test_phase7_geometry.py`
- `tests/unit/test_phase7_uncertainty_contracts.py`
- `tests/unit/test_phase7_intensity.py`
- `tests/unit/test_phase7_noise_blur.py`
- `tests/unit/test_phase7_resampling.py`
- `tests/unit/test_phase7_crop_fov.py`
- `tests/unit/test_phase7_entropy.py`
- `tests/unit/test_phase7_tta.py`
- `tests/unit/test_phase7_calibration.py`
- `tests/unit/test_phase7_risk_coverage.py`
- `tests/unit/test_phase7_failure_detection.py`
- `tests/unit/test_phase7_degradation.py`
- `tests/unit/test_phase7_subgroups.py`
- `tests/unit/test_phase7_leakage.py`
- `tests/integration/test_phase7_publication.py`
- `tests/integration/test_phase7_cli.py`
- `tests/smoke/test_phase7_robustness_smoke.py`

All generated Phase 7 artifacts must be written only beneath explicit external output roots and must
remain untracked.

## Phase 7 Artifact and Contract Plan

Planned versioned schemas:

- `robustness_corruption_spec_v1`: one named corruption with severity, deterministic parameters,
  seed where applicable, input geometry reference, output geometry reference, and identity hash.
- `robustness_corruption_manifest_v1`: ordered collection of corruption specs with canonical
  severity definitions, source config hash, seed policy, and manifest hash.
- `robustness_transform_result_v1`: transform result metadata with input content hash, output
  content hash, geometry record hash, applied parameter payload, mask-preservation status, and
  result hash.
- `robustness_geometry_record_v1`: original grid, transformed grid, restored common grid, spacing,
  orientation, affine hash, shape, interpolation policy, binary-mask-preservation status, and record
  hash.
- `uncertainty_result_v1`: predictive entropy, TTA variance or ensemble variance, probability
  content hashes, uncertainty map hashes, eligibility status, and result hash.
- `calibration_result_v1`: calibration-bin records, ECE, binning policy, valid-voxel count,
  degenerate-bin handling, and result hash.
- `risk_coverage_result_v1`: ordered coverage/risk points, ordering policy, tie policy, valid-voxel
  count, and result hash.
- `failure_detection_result_v1`: uncertainty-error correlation, AUROC eligibility status, AUROC
  value when eligible, ineligibility reason when not eligible, and result hash.
- `degradation_result_v1`: baseline metric, corrupted metric, absolute degradation, relative
  degradation, validity status, and result hash.
- `lesion_subgroup_result_v1`: subgroup thresholds, per-subgroup counts, per-subgroup metrics,
  empty-lesion counts, and result hash.
- `phase7_run_summary_v1`: config hash, corruption manifest hash, transform result hashes,
  uncertainty result hashes, evaluation result hashes, publication hashes, execution status,
  unavailable runtime/memory fields, and run-summary hash.

All schemas must reject unknown fields during reconstruction, validate embedded self-hashes, reject
NaN and Infinity, use existing canonical JSON and SHA-256 hashing utilities, and exclude timestamps,
absolute paths, hostnames, hardware, runtime duration, display-only text, and local filesystem
details from scientific identity payloads where they do not affect scientific results.

## Phase 7 Severity and Geometry Contracts

Severity levels are deterministic named levels: `none`, `low`, `medium`, and `high`. `none` is the
identity transform and must still produce a validated transform record. The first implementation
must define a fixed parameter table in tracked config or artifacts, not fit severity parameters from
data.

Planned default severity semantics:

- HU-window shift: add a deterministic window-center shift in HU before clipping to the configured
  CT window; persisted field `window_center_shift_hu`.
- Intensity scale: multiply CT intensities by a positive finite scale; persisted field
  `scale_factor`.
- Intensity offset: add a finite HU offset; persisted field `offset_hu`.
- Contrast shift: apply `mean + contrast_factor * (x - mean)` using a fixed mean source from config
  or artifact, not a fitted test-set statistic unless explicitly recorded as a synthetic fixture
  constant; persisted fields `contrast_factor` and `contrast_center_hu`.
- Gaussian noise: add local-RNG Gaussian noise with persisted `sigma_hu` and `seed`; no global RNG
  mutation.
- Gaussian blur: apply deterministic image-only Gaussian blur with persisted physical or voxel sigma
  and boundary mode; masks are not blurred.
- Slice-thickness simulation: resample along the slice axis to a coarser spacing and restore to the
  common grid, with persisted original spacing, simulated spacing, interpolation mode, and
  restoration mode.
- Anisotropic downsampling/resampling: downsample selected axes by persisted finite factors and
  restore to the common grid before metrics.
- Crop/FOV perturbation: crop or pad by persisted voxel margins or fractions, restore to the common
  grid, and record whether any foreground label was removed for evaluation diagnostics.

Geometry safety rules:

- Every image and mask must have matching shape, affine, spacing, orientation, and grid identity
  before paired transform or evaluation.
- Transforms that intentionally change geometry must emit a geometry-change record before
  restoration.
- Metrics are computed only after prediction and reference mask are restored to an explicit common
  grid and alignment is revalidated.
- Image interpolation may use documented linear or B-spline interpolation; mask interpolation must
  use nearest neighbor only.
- Masks must remain binary after every transform and restoration step. Non-binary masks fail
  validation instead of being silently repaired.
- Affines must be finite, non-singular, and within the explicit tolerance defined by the geometry
  contract. Shape, spacing, and orientation mismatches must fail unless covered by a validated
  geometry-change record and restored common-grid record.
- Empty-lesion masks are permitted only when explicitly represented in downstream evaluation
  artifacts. Empty-lesion handling must not alter prediction, uncertainty, or optimization logic.

## Phase 7 Uncertainty and Evaluation Formulas

Predictive entropy uses binary probabilities `p_fg` and `p_bg = 1 - p_fg` on valid voxels:

`H(p) = -p_fg * log(p_fg) - p_bg * log(p_bg)`.

The implementation must define the log base in the artifact identity. Natural log is the planned
default. Terms with probability exactly zero contribute zero by definition. Inputs must be finite
and satisfy voxelwise probability-sum validation within a strict tolerance. No clipping is allowed
except an explicitly documented numerical safeguard that preserves the `0 log 0 = 0` convention.

TTA or ensemble variance:

- For `N` aligned probability samples `p_i(x)`, mean probability is
  `mean_p(x) = (1/N) * sum_i p_i(x)`.
- Variance is the population variance
  `var_p(x) = (1/N) * sum_i (p_i(x) - mean_p(x))^2`.
- `N >= 2` is required for variance. Single-sample inputs are ineligible for variance and must
  record an unavailable status.
- Every sample must be generated from an explicit deterministic TTA manifest or approved ensemble
  member manifest and restored to the common grid before aggregation.

Calibration:

- Planned primary metric is expected calibration error (ECE) over valid voxels.
- Confidence is `max(p_fg, p_bg)` unless a downstream binary foreground-confidence contract is
  explicitly selected and recorded.
- Correctness is `prediction == reference_mask` on the common grid after final prediction.
- Fixed bins are half-open intervals `[bin_lower, bin_upper)` except the final bin, which is closed
  on the upper edge.
- Per-bin accuracy is mean correctness for voxels in the bin; per-bin confidence is mean confidence
  for voxels in the bin.
- Empty bins contribute zero weighted error and retain explicit `voxel_count = 0`.
- `ECE = sum_b (voxel_count_b / valid_voxel_count) * abs(accuracy_b - confidence_b)`.
- If `valid_voxel_count = 0`, ECE is unavailable with an explicit reason.

Risk-coverage:

- Uncertainty scores are ordered ascending for retention, because lower uncertainty is retained
  first.
- Tie-breaking is deterministic by flattened voxel index in row-major canonical array order.
- Coverage at prefix length `k` is `k / valid_voxel_count`.
- Risk is the mean error indicator among retained voxels: `mean(prediction != reference_mask)`.
- The curve must include fixed coverage points or every deterministic prefix as defined by the
  artifact policy. Missing points cannot be smoothed or fabricated.
- If no valid voxels exist, risk-coverage is unavailable with an explicit reason.

Uncertainty-error correlation:

- Error is the binary indicator `prediction != reference_mask` on the common grid.
- The planned default is Spearman rank correlation between uncertainty score and error indicator
  with deterministic average ranks for ties.
- Correlation is unavailable when valid voxel count is insufficient, uncertainty is constant, or
  error is constant.

Failure-detection AUROC:

- AUROC is case-level unless a later artifact explicitly records a voxel-level failure-detection
  scope.
- A case is a failure when the selected post-prediction metric crosses the predeclared failure
  threshold, for example Dice below a fixed threshold stored in the artifact.
- The failure score is the case-level uncertainty summary, such as mean entropy or mean TTA
  variance, selected before evaluation and stored in the artifact.
- AUROC is eligible only when there are at least two cases, at least one failure, at least one
  non-failure, finite scores for all included cases, and no missing labels for the eligibility
  scope.
- Ineligible AUROC must be recorded as unavailable with a reason; no fallback value, null-success
  value, or fabricated AUROC is allowed.

Lesion-size subgroup analysis:

- Subgroups are defined from reference connected-component voxel count or physical volume using
  explicit thresholds stored in config/artifacts.
- Planned subgroup names are `empty`, `small`, `medium`, and `large`.
- Empty-lesion cases are counted separately and are not silently merged into a positive-lesion
  subgroup.
- Per-subgroup artifacts must record eligible-case count, empty-lesion count where relevant,
  skipped-case count, metric availability, and metric values only when computable.

Degradation:

- Absolute degradation is `baseline_metric - corrupted_metric` for metrics where higher is better,
  such as Dice and IoU.
- For lower-is-better metrics, the direction must be encoded explicitly before use; the first Phase
  7 implementation should avoid mixing higher-is-better and lower-is-better metrics in one summary
  unless direction metadata is present.
- Relative degradation is `(baseline_metric - corrupted_metric) / abs(baseline_metric)` for
  higher-is-better metrics when `abs(baseline_metric) >= epsilon`.
- If the baseline metric is unavailable, non-finite, or `abs(baseline_metric) < epsilon`, relative
  degradation is unavailable with an explicit reason. It must not be reported as zero or infinity.

## Phase 7 Planned Verification

- Artifact-schema, canonical-serialization, unknown-field rejection, and self-hash validation tests
  for every Phase 7 artifact family.
- Geometry tests for image/mask misalignment, invalid affine, spacing/orientation mismatch, shape
  mismatch, binary-mask preservation, common-grid restoration, and interpolation policy.
- Transform tests for deterministic severity mapping, finite outputs, exact parameter recording,
  local RNG behavior, no global seed mutation, binary-mask preservation, and repeated execution with
  byte-identical logical outputs.
- Uncertainty tests for entropy formulas, zero-probability handling, probability validation,
  deterministic TTA manifests, aligned aggregation, variance eligibility, and no label input to
  prediction APIs.
- Evaluation tests for calibration binning and ECE, risk-coverage ordering and tie-breaking,
  uncertainty-error correlation eligibility, AUROC eligibility and ineligibility, lesion subgroup
  counts, absolute degradation, relative degradation, and degenerate-case behavior.
- Publication tests proving JSON-first derivation for Markdown, tables, and plots; deterministic
  JSON/Markdown bytes; explicit external output roots; atomic staged writes; idempotent
  regeneration; incompatible overwrite rejection; traversal rejection; symlink-escape rejection;
  and no generated artifacts inside the repository.
- CLI and smoke tests for bounded CPU synthetic execution covering every required corruption family,
  uncertainty artifacts, evaluation artifacts, and publication outputs.
- Leakage tests proving labels and reference masks enter only evaluation paths after predictions are
  complete, Phase 8 external access is absent, and real 3D-IRCADb-01 is not used.
- Final Gate 7 commands:
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run mypy src`
  - `uv run pytest -q`
  - `uv run pre-commit run --all-files`
  - Phase 7 CLI help
  - two independent bounded CPU synthetic Phase 7 executions into external temporary roots
  - deterministic JSON/Markdown artifact comparison across those roots
  - geometry-safety validation and corruption-manifest reproducibility validation

## Phase 7 Risks and Safeguards

- Risk: robustness transforms can corrupt masks or misalign image/mask geometry.
  Safeguard: enforce nearest-neighbor mask interpolation, binary-mask validation, geometry-change
  records, and common-grid restoration before metrics.
- Risk: severity parameters can become implicit, tuned, or dataset-dependent.
  Safeguard: persist every severity parameter in manifests and prohibit fitted severity values in
  Phase 7 unless explicitly represented as synthetic constants for tests.
- Risk: random noise or TTA can become nondeterministic.
  Safeguard: use local seeded RNG objects only, record seeds in manifests, and test that global RNG
  state is unchanged.
- Risk: uncertainty metrics can be computed on invalid probability maps.
  Safeguard: require finite probabilities, probability-sum validation, strict shape checks, and
  explicit unavailable statuses for ineligible inputs.
- Risk: AUROC or relative degradation can be fabricated in degenerate cases.
  Safeguard: require eligibility records and unavailable statuses for one-class, insufficient,
  missing, non-finite, or near-zero-baseline inputs.
- Risk: labels can influence prediction, TTA, transform selection, or uncertainty construction.
  Safeguard: keep labels absent from prediction and transform execution APIs and permit reference
  masks only in post-prediction evaluation functions.
- Risk: publication can include derived values not traceable to artifacts.
  Safeguard: generate plots, tables, and Markdown only from validated persisted JSON artifacts.
- Risk: Phase 7 can drift into Phase 8 external validation or LLM/VLM work.
  Safeguard: explicitly prohibit 3D-IRCADb-01, external-validation claims, real-data claims,
  checkpoint selection, model retraining, GPU claims, and LLM/VLM scope.

## Gate 7 Acceptance Criteria

Gate 7 is accepted when:

- every required corruption family exists, records deterministic severity parameters, and is
  reproducible from a versioned corruption manifest;
- masks remain binary and spatially aligned with images, and every geometry-changing transform has
  an explicit geometry-change record plus validated common-grid restoration before metrics;
- predictive entropy is implemented from validated probabilities;
- deterministic TTA variance or an approved deterministic ensemble interface is implemented with
  aligned samples and explicit sample manifests;
- calibration, risk-coverage, uncertainty-error correlation, failure-detection AUROC eligibility,
  lesion-size subgroup, absolute-degradation, and relative-degradation artifacts are implemented
  with explicit degenerate-case handling;
- AUROC is computed only when statistically eligible and recorded as unavailable otherwise;
- labels and reference masks enter only post-prediction evaluation paths and never drive
  corruption selection, TTA, uncertainty construction, prediction, checkpoint selection, or
  hyperparameter tuning;
- deterministic publication writes JSON-first artifacts, derived Markdown/tables/plots, and no
  generated artifacts inside the repository;
- two independent bounded CPU synthetic Phase 7 executions with identical manifests produce
  byte-identical deterministic JSON and Markdown artifacts, with PNG existence and JSON-derived
  plot data validated without requiring cross-platform PNG byte identity;
- repository-wide lint, format, typing, tests, pre-commit, Phase 7 CLI help, synthetic smoke, and
  leakage checks pass;
- no real-data, GPU, external-validation, 3D-IRCADb-01, LLM/VLM, retraining, checkpoint-selection,
  or Phase 8 claim is made; and
- Gate 7 close-out documents only verified evidence and does not begin Phase 8.

## Phase 7 Acceptance Matrix

| Capability | Required artifact or evidence | Required validation |
| --- | --- | --- |
| Corruption manifests | `robustness_corruption_manifest_v1` | Canonical hash, persisted severities, deterministic repeat |
| Intensity corruptions | Transform result records | Exact parameter tests and finite-output validation |
| Noise and blur | Transform result records with seed/sigma | Local RNG tests, repeat determinism, no mask filtering |
| Resampling and slice thickness | Geometry and transform records | Alignment, nearest-neighbor masks, common-grid restoration |
| Crop/FOV perturbation | Crop geometry records | Persisted crop parameters, restoration, empty-lesion handling |
| Predictive entropy | `uncertainty_result_v1` | Exact entropy tests and probability validation |
| TTA or ensemble variance | TTA or ensemble manifest and uncertainty result | Aligned samples, variance eligibility, deterministic repeat |
| Calibration | `calibration_result_v1` | Fixed bins, ECE formula, empty-bin behavior |
| Risk coverage | `risk_coverage_result_v1` | Ordering, tie-breaking, coverage/risk formula |
| Failure analysis | `failure_detection_result_v1` | Correlation eligibility and AUROC eligibility |
| Subgroup/degradation | Subgroup and degradation artifacts | Explicit thresholds, absolute/relative degradation rules |
| Leakage safety | Leakage tests and audit evidence | Labels only after prediction; no Phase 8 access |
| Publication | JSON, Markdown, tables, plots | JSON-first derivation, atomic writes, path safety |
| Gate close-out | Gate 7 evidence | Full checks and two deterministic synthetic executions |

## Phase 8 Scope

Phase 8 is limited to external validation on `3D-IRCADb-01`. It must evaluate a frozen ProtoEM-CT
workflow without tuning and without beginning Phase 9, Phase 10, or the LLM/VLM track.

Phase 8 includes:

- a frozen-decision inventory before external label access;
- versioned, self-validating external-evaluation preregistration JSON before real evaluation;
- approval-gated read-only external-drive discovery for the explicit 3D-IRCADb-01 root;
- an anonymous external manifest and dataset QA with persisted hashes;
- a frozen label-mapping policy before labels are read;
- documentation of scanner/acquisition/domain shift, inclusion/exclusion accounting, missing labels,
  incompatible cases, and preregistration deviations;
- one-time real external inference without tuning after freeze and preregistration;
- immutable prediction inventory outside Git;
- preregistered segmentation metrics, bootstrap confidence intervals, internal-versus-external
  comparison tables, qualitative montages, failure analysis, and limitations computed only from
  persisted artifacts; and
- independent scientific, engineering, and leakage review before Gate 8 close-out.

Phase 8 excludes:

- any checkpoint, model, threshold, support policy, preprocessing, postprocessing, corruption,
  uncertainty, or reporting selection based on external labels or external metrics;
- external superiority, clinical validity, deployment readiness, broad generalization, or
  state-of-the-art claims unless directly supported by preregistered artifacts;
- treating LiTS and MSD Task03 Liver as independent cohorts;
- committing medical data, local paths, source identifiers, external manifests containing PHI,
  masks, predictions, embeddings, checkpoints, model weights, HMAC keys, reverse identifier maps,
  credentials, logs, MLflow runs, or large generated artifacts; and
- any `/Volumes` search or dataset access before Wave 2 approval.

`codex_protoem_ct_blueprint.md` was not present during Wave 0 planning, so no Phase 8 blueprint
section was available for reconciliation.

## Phase 8 Wave 0 Supervisor Planning

Wave 0 used real delegated subagents for four independent read-only planning tracks:

- `P8-SCIENTIFIC-PLAN`
- `P8-DATA-BOUNDARY-PLAN`
- `P8-ENGINEERING-PLAN`
- `P8-LEAKAGE-PLAN`

All four subagents were instructed not to edit files, not to access `/Volumes`, and not to inspect
datasets. Their reports were reconciled into this plan.

Wave 0 scientific conflicts and unresolved decisions:

- Existing Phase 8 and Gate 8 definitions were absent before this planning update.
- Bootstrap CI policy is not implemented yet and must be frozen in Wave 1: CI type, confidence
  level, replicate count, seed, resampling unit, paired/unpaired rules, and unavailable-value
  handling.
- External few-shot support-label use is unresolved. The conservative Wave 0 plan treats external
  labels as evaluation-only. If external support labels are later proposed, support/query partition,
  `K`, replicates, adaptation modes, and label-access rules must be frozen in preregistration before
  any labels are read.
- The real 3D-IRCADb-01 layout is unknown from repository context. Existing
  `IrcadbStyleAdapter` is explicitly synthetic-tested and Phase 2-only; Phase 8 must explicitly
  authorize or replace it for real external discovery.
- Existing Phase 2 `DatasetManifest` is development-oriented and label-bearing; Phase 8 needs
  external image-manifest and label-gated manifest contracts rather than reusing it unchanged.
- External robustness/uncertainty reporting is optional and may be included only if exactly
  preregistered from Phase 7 contracts before external evaluation.
- Phase 7 lesion-size subgrouping and Phase 3 lesion metric connectivity may differ by contract;
  Phase 8 must state the exact rule used for any subgroup reporting.

## Phase 8 Dependency Graph

Every wave must be resumable from repository files and commit hashes on the integration branch
`phase/8-external-validation`. Use small reviewable commits after approved substages. The persistent
handoff file is `docs/phase8/SUPERVISOR_HANDOFF.md` and must record current wave, approved commits,
completed agents, blocked agents, frozen artifact hashes, unresolved blockers, and exact next
action. No hidden chat state is required.

### Wave 0: Planning and Dependency Graph

- Scope: create this complete implementation plan and the persistent Supervisor handoff.
- Dependencies: completed and merged Phases 0-7, user-confirmed Gate 7 passed.
- Allowed files: `docs/PHASE_PLAN.md`, `docs/phase8/SUPERVISOR_HANDOFF.md`.
- Prohibited files: all source, tests, configs, manifests, preregistration JSON, generated outputs,
  medical data, predictions, checkpoints, and external paths.
- Required inputs: project docs, Phase 2-7 code/artifact conventions, four read-only planning
  subagent reports.
- Expected outputs: Phase 8 dependency graph, subagent plan, boundaries, risks, Gate 8 criteria,
  handoff.
- Tests/commands: `git diff --check`, `git diff --stat`, `git status --short --branch`.
- Completion criteria: only allowed files changed, real subagents used, no dataset access, no code
  implementation, no Wave 1 release.
- External drive: disconnected.
- External labels: not read.
- Concurrency: four planning subagents ran concurrently.

### Wave 1: Frozen Contracts, Schemas, and Leakage Controls

Planned subagents:

- `P8-FREEZE-INVENTORY`
  - Scope: define `phase8_decision_freeze_v1` and frozen-decision inventory.
  - Dependencies: Wave 0.
  - Allowed files: `docs/PHASE_PLAN.md`, `docs/DECISIONS.md`,
    `docs/phase8/SUPERVISOR_HANDOFF.md`, planned source/tests under `src/protoem_ct/external/`,
    `tests/unit/test_phase8_freeze*.py`.
  - Prohibited files: real data paths, manifests, predictions, checkpoints, generated artifacts.
  - Required inputs: Phase 2 manifest/split hashes, Phase 4 support/protocol hashes, Phase 5
    retrieval/prototype identities, Phase 6 config/inference/stopping/publication contracts, Phase 7
    publication contracts if used.
  - Expected outputs: schema and tests for preprocessing hash, support policy, thresholds,
    checkpoint/model, label mapping policy, metric config, publication config, Git commit, explicit
    timestamp, and `frozen_before_external_evaluation` status.
  - Tests: schema validation, self-hash validation, unknown-field rejection, no absolute paths,
    no NaN/Inf.
  - Completion criteria: freeze artifact can be created and validated with synthetic fixture
    hashes.
  - External drive: no.
  - External labels: no.
  - Concurrency: may run with `P8-PREREGISTRATION-SCHEMA` and `P8-LEAKAGE-CONTRACTS`.

- `P8-PREREGISTRATION-SCHEMA`
  - Scope: define versioned self-validating `phase8_external_preregistration_v1`.
  - Dependencies: Wave 0.
  - Allowed files: planned Phase 8 external source/tests/docs only.
  - Prohibited files: real preregistration artifacts, real dataset paths, predictions.
  - Required inputs: Wave 0 scientific plan, existing metric and publication conventions.
  - Expected outputs: schema covering cohort definition, inclusion/exclusion rules, metric plan,
    bootstrap CI plan, internal-versus-external comparison plan, label-access policy, deviation
    accounting, and limitations.
  - Tests: required-field validation, hash stability, missing-policy rejection.
  - Completion criteria: synthetic preregistration validates and records no real path.
  - External drive: no.
  - External labels: no.
  - Concurrency: may run with other Wave 1 agents.

- `P8-LEAKAGE-CONTRACTS`
  - Scope: define `external_label_access_ledger_v1`, post-evaluation lock, and leakage tests.
  - Dependencies: Wave 0.
  - Allowed files: planned Phase 8 external source/tests/docs only.
  - Prohibited files: real labels, metrics, prediction outputs, checkpoints.
  - Required inputs: Phase 6/7 label-isolation tests and Wave 0 leakage report.
  - Expected outputs: tests proving external labels cannot tune preprocessing, select support
    policy, choose thresholds, choose checkpoint/model, alter label mapping after freeze, or mutate
    internal-versus-external reporting.
  - Tests: targeted leakage unit tests using synthetic fixtures.
  - Completion criteria: leakage contracts fail closed when freeze or ledger hashes mismatch.
  - External drive: no.
  - External labels: no.
  - Concurrency: may run with other Wave 1 agents.

### Wave 2: External-Drive Discovery and Anonymous Image Inventory

This is the first wave requiring the user to connect the external drive.

Planned subagents:

- `P8-IRCADB-DRYRUN`
  - Scope: approval-gated, filename-only 3D-IRCADb-01 discovery under one explicit root.
  - Dependencies: Wave 1 freeze/preregistration/leakage contracts and explicit user approval.
  - Allowed files: Phase 8 adapter code/tests/docs; external generated output root only for
    nonidentifying dry-run artifact.
  - Prohibited files: repository generated outputs, `/Volumes` enumeration, source-data writes,
    NIfTI/DICOM opening or hashing, labels, predictions.
  - Required inputs: explicit absolute dataset root from user, explicit absolute output root,
    approved layout convention draft.
  - Expected outputs: nonidentifying structural dry-run counts, layout failures, adapter convention
    candidate hash outside Git.
  - Tests: synthetic adapter tests for expected archive-like and alternate layouts, missing pairs,
    duplicate pairs, symlink escape, traversal, unsafe names.
  - Completion criteria: Supervisor approves exact layout before manifest/QA.
  - External drive: yes.
  - External labels: no.
  - Concurrency: cannot run with label or inference agents; may run with read-only docs review.

- `P8-EXTERNAL-IMAGE-MANIFEST`
  - Scope: build anonymous image manifest after approved dry run.
  - Dependencies: `P8-IRCADB-DRYRUN` approval.
  - Allowed files: Phase 8 manifest code/tests; external output root for anonymous manifest.
  - Prohibited files: label reads, reverse ID maps in Git, HMAC keys in Git, absolute paths in
    artifacts.
  - Required inputs: explicit dataset root, dry-run artifact hash, anonymization policy, output
    root.
  - Expected outputs: `phase8_external_image_manifest_v1` with relative paths, image hashes,
    adapter identity, root fingerprint, manifest hash.
  - Tests: collision checks, deterministic ordering, path safety, no absolute paths.
  - Completion criteria: manifest validates, contains no source identifiers or absolute paths.
  - External drive: yes.
  - External labels: no.
  - Concurrency: after dry-run approval, may run before image QA.

- `P8-EXTERNAL-IMAGE-QA`
  - Scope: image-only QA and domain-shift image summaries.
  - Dependencies: external image manifest.
  - Allowed files: Phase 8 QA code/tests; external output root for image-QA artifacts.
  - Prohibited files: label reads, model selection changes, preprocessing fit changes.
  - Required inputs: image manifest hash, fixed QA tolerance/config, fixed histogram bins.
  - Expected outputs: image QA artifact, image-only domain-shift record, inclusion candidates.
  - Tests: finite values, shape/affine/spacing/orientation, fixed-bin histograms, invalid-case
    retention.
  - Completion criteria: image QA status is persisted with frozen rules and no labels.
  - External drive: yes.
  - External labels: no.
  - Concurrency: may run after image manifest; cannot change Wave 4 freeze decisions.

### Wave 3: Label-Mapping Policy, Domain Shift, and Eligibility Accounting

Planned subagents:

- `P8-LABEL-MAPPING-POLICY`
  - Scope: implement and freeze label-mapping policy from documentation/config, not from labels.
  - Dependencies: Wave 1 schemas and Wave 2 layout information; no label reading.
  - Allowed files: Phase 8 label-mapping source/tests/docs and handoff.
  - Prohibited files: real label files, observed label values, metrics, predictions.
  - Required inputs: approved adapter convention, task definition binary tumor vs background.
  - Expected outputs: `phase8_label_mapping_policy_v1` schema/config tests; policy hash to be
    included in freeze/preregistration.
  - Tests: nonzero-tumor-to-foreground synthetic mapping, liver/context masks rejected as target,
    unknown/tampered mapping rejection.
  - Completion criteria: mapping policy is frozen before external labels are opened.
  - External drive: optional for layout artifact only; no new drive access preferred.
  - External labels: no.
  - Concurrency: may run with eligibility accounting.

- `P8-ELIGIBILITY-ACCOUNTING`
  - Scope: predeclare inclusion/exclusion accounting and domain-shift documentation fields.
  - Dependencies: Wave 2 image QA and Wave 1 preregistration schema.
  - Allowed files: Phase 8 accounting source/tests/docs; external output root for image-only
    eligibility artifact if approved.
  - Prohibited files: labels, predictions, metric computation.
  - Required inputs: image QA artifact, fixed invalid-case rules.
  - Expected outputs: image-only eligibility and exclusion accounting; domain-shift image summary.
  - Tests: invalid cases retained with explicit reasons, no silent repair/exclusion.
  - Completion criteria: eligibility rules frozen before labels and metrics.
  - External drive: maybe, only for image QA artifact verification.
  - External labels: no.
  - Concurrency: may run with label-mapping policy.

### Wave 4: Freeze and Preregistration

Planned subagents:

- `P8-FROZEN-DECISION-PACKAGE`
  - Scope: freeze preprocessing, support policy, thresholds, checkpoint, model-selection decision,
    label mapping, metrics, bootstrap, and publication plan.
  - Dependencies: Waves 1-3 complete.
  - Allowed files: Phase 8 freeze/preregistration source/tests/docs; external output root for
    freeze and preregistration JSON.
  - Prohibited files: external labels, real metric results, prediction artifacts.
  - Required inputs: upstream artifact hashes, selected checkpoint SHA-256, fixed threshold config,
    label-mapping policy hash, metric/bootstrap config hash.
  - Expected outputs: `phase8_decision_freeze_v1` and
    `phase8_external_preregistration_v1`, both self-validating.
  - Tests: decision immutability, timestamp/order checks, no multiple candidate checkpoints,
    no threshold override after freeze.
  - Completion criteria: preregistration validates before real evaluation.
  - External drive: not required.
  - External labels: no.
  - Concurrency: sequential gate before Wave 5.

- `P8-FREEZE-LEAKAGE-AUDIT`
  - Scope: read-only audit of the freeze package and preregistration.
  - Dependencies: `P8-FROZEN-DECISION-PACKAGE`.
  - Allowed files: docs/handoff and leakage tests if needed.
  - Prohibited files: external labels, predictions, metrics.
  - Required inputs: freeze hash, preregistration hash, Wave 1 leakage contracts.
  - Expected outputs: audit recommendation and blocker list.
  - Tests: targeted leakage tests from Wave 1.
  - Completion criteria: zero unresolved critical leakage blockers.
  - External drive: no.
  - External labels: no.
  - Concurrency: sequential after freeze package.

### Wave 5: Synthetic/Dry-Run Pipeline Without Real External Labels

Planned subagents:

- `P8-SYNTHETIC-PIPELINE`
  - Scope: bounded synthetic external-validation pipeline using synthetic fixtures only.
  - Dependencies: Wave 4 freeze/preregistration audit.
  - Allowed files: Phase 8 source/tests/configs; temporary output roots outside Git.
  - Prohibited files: real 3D-IRCADb data, external labels, real predictions.
  - Required inputs: synthetic external fixtures, freeze/preregistration JSON, synthetic checkpoint
    reference or deterministic stub allowed only for contract testing.
  - Expected outputs: synthetic manifest, synthetic predictions, synthetic publication artifacts
    outside Git.
  - Tests: end-to-end integration, deterministic rerun, no labels in inference APIs.
  - Completion criteria: synthetic dry run passes without real labels and produces deterministic
    persisted artifacts.
  - External drive: no.
  - External labels: no.
  - Concurrency: may run with `P8-PUBLICATION-CONTRACTS`.

- `P8-PUBLICATION-CONTRACTS`
  - Scope: deterministic inference/publication contracts.
  - Dependencies: Wave 4.
  - Allowed files: Phase 8 publication source/tests/docs.
  - Prohibited files: real generated reports, real montages, external labels.
  - Required inputs: freeze/preregistration schemas and existing Phase 6/7 publication conventions.
  - Expected outputs: JSON-first publication, atomic writes, no-overwrite/idempotent behavior.
  - Tests: Markdown/tables/plots derived from JSON only, path safety, no generated artifacts in Git.
  - Completion criteria: two synthetic roots produce byte-identical deterministic JSON/Markdown.
  - External drive: no.
  - External labels: no.
  - Concurrency: may run with synthetic pipeline.

### Wave 6: One-Time Real External Inference

Planned subagents:

- `P8-REAL-INFERENCE`
  - Scope: one-time inference on real 3D-IRCADb-01 images without tuning.
  - Dependencies: Wave 5 pass, Wave 4 freeze/preregistration hash, explicit user approval.
  - Allowed files: none in Git except handoff/doc updates; external output root for predictions and
    prediction inventory.
  - Prohibited files: external labels, threshold changes, checkpoint changes, support policy
    changes, reruns to improve metrics.
  - Required inputs: explicit dataset root, image manifest, image QA artifact, freeze hash,
    preregistration hash, checkpoint/model artifact outside Git, output root.
  - Expected outputs: immutable prediction inventory and prediction hashes outside Git.
  - Tests: prediction manifest validation, label-free inference path, output-root safety.
  - Completion criteria: prediction lock exists, validates, and links exactly to freeze and
    preregistration hashes.
  - External drive: yes.
  - External labels: no.
  - Concurrency: must run alone.

### Wave 7: External Metrics, Bootstrap CIs, and Internal-Versus-External Tables

This is the first point where external labels may be read, and only for evaluation/QA behind the
frozen label-access ledger.

Planned subagents:

- `P8-LABEL-LEDGER-QA`
  - Scope: open labels under ledger, build label manifest, validate label QA, and apply frozen
    mapping.
  - Dependencies: Wave 6 prediction lock, Wave 4 freeze/preregistration.
  - Allowed files: external output root for label manifest, label QA, ledger.
  - Prohibited files: Git artifacts, prediction reruns, decision mutations, checkpoint changes.
  - Required inputs: freeze hash, preregistration hash, prediction manifest hash, explicit label
    root/layout from approved adapter convention.
  - Expected outputs: `external_label_access_ledger_v1`, label manifest, label QA, mapping result.
  - Tests: ledger required before label reads, mapping hash match, changed labels affect only
    evaluation artifacts.
  - Completion criteria: label access is auditable and cannot alter frozen decisions.
  - External drive: yes.
  - External labels: yes.
  - Concurrency: must precede metrics; no concurrent decision agents.

- `P8-EXTERNAL-METRICS`
  - Scope: compute preregistered external segmentation metrics only from persisted predictions and
    labels.
  - Dependencies: `P8-LABEL-LEDGER-QA`.
  - Allowed files: external output root for metric reports.
  - Prohibited files: source data writes, prediction changes, threshold/model/support changes.
  - Required inputs: prediction manifest, label QA/mapping artifact, metric config hash.
  - Expected outputs: `phase8_external_metric_report_v1` with per-case and aggregate tumor Dice,
    IoU, HD95, NSD, lesion recall/precision/F1, FP lesions/scan, signed/absolute/relative volume
    error, unavailable reasons, valid/invalid counts.
  - Tests: metric artifact validation, empty-mask behavior, undefined values as JSON `null`.
  - Completion criteria: metrics validate and link to freeze, preregistration, predictions, labels.
  - External drive: yes, if label files are needed for persisted evaluation input.
  - External labels: yes.
  - Concurrency: after label ledger; may run before bootstrap.

- `P8-BOOTSTRAP-CI`
  - Scope: compute preregistered bootstrap confidence intervals.
  - Dependencies: external metric report.
  - Allowed files: external output root for bootstrap artifacts.
  - Prohibited files: metric-plan changes, new model outputs, label remapping.
  - Required inputs: metric report hash, bootstrap config hash.
  - Expected outputs: bootstrap estimates and 95% percentile CIs unless a different approved
    preregistered policy exists; patient/case-level resampling, deterministic local RNG, fixed seed,
    replicate count, paired/unpaired status.
  - Tests: deterministic seed/repeat, unavailable CI handling, no voxel/lesion resampling.
  - Completion criteria: CI artifacts validate and record unavailable reasons.
  - External drive: no if metric artifacts persist all inputs.
  - External labels: no new label reads.
  - Concurrency: after metrics.

- `P8-INTERNAL-EXTERNAL-COMPARISON`
  - Scope: compare frozen internal immutable-test results with external results.
  - Dependencies: external metrics and bootstrap CIs; frozen internal metric artifacts.
  - Allowed files: external output root for comparison tables.
  - Prohibited files: internal split changes, checkpoint/model ranking, decision changes.
  - Required inputs: internal metric report hashes, external metric/CI hashes.
  - Expected outputs: side-by-side estimates/CIs and external-minus-internal differences where
    preregistered.
  - Tests: comparison cannot alter freeze hash; report has no decision parameters.
  - Completion criteria: comparison tables validate and are clearly descriptive.
  - External drive: no.
  - External labels: no new label reads.
  - Concurrency: after bootstrap.

### Wave 8: Qualitative Montages, Failure Analysis, and Limitations

Planned subagents:

- `P8-QUALITATIVE-MONTAGES`
  - Scope: anonymous qualitative montages derived from persisted predictions/labels and
    preregistered selection rules.
  - Dependencies: Wave 7 metrics and preregistered montage policy.
  - Allowed files: external output root for generated montages and metadata.
  - Prohibited files: Git images, source identifiers, cherry-picking based on post-hoc claims.
  - Required inputs: prediction/label artifacts, selection policy hash, anonymized IDs.
  - Expected outputs: montage inventory and anonymized montage assets outside Git.
  - Tests: selection is deterministic, no source IDs or paths, metadata links to artifacts.
  - Completion criteria: montages reproduce from persisted artifacts and policy.
  - External drive: no if persisted artifacts suffice; otherwise explicit read-only approval.
  - External labels: no new label reads beyond persisted artifacts.
  - Concurrency: may run with failure analysis.

- `P8-FAILURE-LIMITATIONS`
  - Scope: failure analysis, deviations, limitations, and allowable claims.
  - Dependencies: Wave 7 artifacts.
  - Allowed files: publication source/tests/docs and external output root.
  - Prohibited files: new metrics, tuning, checkpoint ranking, clinical claims.
  - Required inputs: metrics, CIs, comparison tables, QA failures, deviation ledger.
  - Expected outputs: limitations and failure-analysis artifacts derived from persisted JSON.
  - Tests: report values trace to JSON; unsupported claims rejected or flagged.
  - Completion criteria: limitations state cohort bounds, missing/incompatible cases, and all
    preregistration deviations.
  - External drive: no.
  - External labels: no new label reads.
  - Concurrency: may run with qualitative montages.

### Wave 9: Independent Reviews, Gate 8 Verification, and Close-Out

Planned subagents:

- `P8-SCIENTIFIC-REVIEW`
  - Scope: independent read-only scientific review.
  - Dependencies: Waves 1-8 complete.
  - Allowed files: docs/handoff only unless approved blocker fix is needed in a later session.
  - Prohibited files: new data access, reruns, tuning, metrics.
  - Required inputs: freeze, preregistration, manifest, predictions, metrics, CIs, reports.
  - Expected outputs: findings, unresolved scientific conflicts, Gate 8 recommendation.
  - Tests: review artifact links and formulas.
  - Completion criteria: no unresolved critical scientific blockers.
  - External drive: no.
  - External labels: no.
  - Concurrency: may run with engineering/leakage reviews.

- `P8-ENGINEERING-REVIEW`
  - Scope: independent read-only engineering review.
  - Dependencies: Waves 1-8 complete.
  - Allowed files: docs/handoff only unless approved blocker fix is needed later.
  - Prohibited files: real data access, generated artifact commits.
  - Required inputs: code, tests, synthetic outputs, artifact hashes.
  - Expected outputs: determinism/path-safety/publication/test findings.
  - Tests: final command set below.
  - Completion criteria: no unresolved critical engineering blockers.
  - External drive: no.
  - External labels: no.
  - Concurrency: may run with other reviews.

- `P8-LEAKAGE-REVIEW`
  - Scope: independent read-only leakage review.
  - Dependencies: Waves 1-8 complete.
  - Allowed files: docs/handoff and `docs/LEAKAGE_AUDIT.md` close-out only after evidence exists.
  - Prohibited files: new label reads, tuning, checkpoint selection.
  - Required inputs: freeze, label ledger, post-evaluation lock, result lineage.
  - Expected outputs: leakage recommendation and Gate 8 blockers.
  - Tests: no external labels in preprocessing/support/threshold/checkpoint/model-selection paths;
    report immutability.
  - Completion criteria: zero unresolved critical leakage blockers.
  - External drive: no.
  - External labels: no.
  - Concurrency: may run with other reviews.

- `P8-GATE8-CLOSEOUT`
  - Scope: final verification, documentation close-out, and next-phase prompt.
  - Dependencies: all reviews pass and blockers resolved.
  - Allowed files: docs close-out files and acceptance checklist after implementation evidence
    exists.
  - Prohibited files: new external evaluation, reruns to improve results, generated artifacts in Git.
  - Required inputs: all reviewed artifacts and final verification commands.
  - Expected outputs: Gate 8 evidence, changed files, commands, tests, scientific decisions, risks,
    gate status, and next-phase prompt.
  - Tests: final Gate 8 command set.
  - Completion criteria: Gate 8 accepted only from verified evidence, not plans.
  - External drive: no for synthetic gate checks; real artifact hash verification only if explicitly
    approved and required.
  - External labels: no new label reads.
  - Concurrency: sequential final gate.

## Phase 8 Artifact and Preregistration Plan

Planned schemas:

- `phase8_decision_freeze_v1`: frozen preprocessing, support policy, thresholds, checkpoint/model,
  model-selection decision, label-mapping policy, metric config, bootstrap plan, publication config,
  Git commit, explicit timestamp, and self-hash.
- `phase8_external_preregistration_v1`: cohort, inclusion/exclusion policy, label-access policy,
  one-time inference rule, metric hierarchy, bootstrap plan, internal-versus-external comparison,
  deviations, limitations, and self-hash.
- `phase8_external_image_manifest_v1`: anonymous external case IDs, relative image paths, image
  hashes, adapter identity, root fingerprint, no absolute paths, and manifest hash.
- `phase8_external_label_manifest_v1`: label paths/hashes and label metadata generated only after
  the label ledger permits evaluation access.
- `phase8_label_mapping_policy_v1`: frozen mapping from source labels or tumor mask files to binary
  tumor foreground, with liver/context labels excluded from target foreground.
- `external_label_access_ledger_v1`: first label-opening operation, freeze hash, preregistration
  hash, prediction-lock hash, purpose, and rejection rules for mismatch.
- `phase8_external_prediction_manifest_v1`: immutable prediction records, prediction hashes,
  geometry metadata, relative output paths, and freeze/preregistration links.
- `phase8_external_metric_report_v1`: project binary tumor metrics and aggregate reports from
  persisted predictions and labels.
- `phase8_bootstrap_ci_v1`: deterministic case/patient-level bootstrap intervals linked to metric
  reports.
- `phase8_internal_external_comparison_v1`: descriptive comparison tables linked to frozen internal
  and external reports.
- `phase8_publication_summary_v1`: JSON-first report/montage/failure-analysis inventory and
  limitations.

All schemas must use existing canonical JSON and SHA-256 conventions, reject unknown fields, reject
NaN and Infinity, avoid absolute paths and machine-specific metadata, validate embedded self-hashes,
and write generated artifacts only outside Git.

## Phase 8 Planned Files and Tests

Wave 0 changed only planning files. Later waves may add or modify these planned surfaces after
approval:

- `configs/phase8_external_validation.yaml`
- `src/protoem_ct/external/__init__.py`
- `src/protoem_ct/external/artifacts.py`
- `src/protoem_ct/external/freeze.py`
- `src/protoem_ct/external/preregistration.py`
- `src/protoem_ct/external/manifest.py`
- `src/protoem_ct/external/qa.py`
- `src/protoem_ct/external/label_mapping.py`
- `src/protoem_ct/external/ledger.py`
- `src/protoem_ct/external/inference.py`
- `src/protoem_ct/external/metrics.py`
- `src/protoem_ct/external/bootstrap.py`
- `src/protoem_ct/external/comparison.py`
- `src/protoem_ct/external/publication.py`
- `src/protoem_ct/data/adapters/ircadb.py`
- `src/protoem_ct/data/adapters/base.py`
- `src/protoem_ct/data/phase2_paths.py`
- `src/protoem_ct/baselines/metrics.py`
- `src/protoem_ct/baselines/predictions.py`
- `src/protoem_ct/protoem/inference.py`
- `src/protoem_ct/protoem/publication.py`
- `src/protoem_ct/retrieval/*`
- `src/protoem_ct/robustness/*` and `src/protoem_ct/uncertainty/*` only if external
  robustness/uncertainty is preregistered
- `src/protoem_ct/cli/main.py`
- `tests/unit/test_phase8_*.py`
- `tests/integration/test_phase8_*.py`
- `tests/smoke/test_phase8_external_validation_smoke.py`
- `docs/DECISIONS.md`
- `docs/LEAKAGE_AUDIT.md`
- `ACCEPTANCE_CHECKLIST.md`
- `docs/phase8/SUPERVISOR_HANDOFF.md`

Planned test categories:

- schema, serialization, self-hash, unknown-field, and no-NaN/Inf tests;
- path safety and external output-root tests;
- synthetic 3D-IRCADb adapter/layout tests;
- anonymization and collision tests;
- label-mapping freeze and tamper tests;
- leakage tests for preprocessing, support policy, thresholds, checkpoint/model, label mapping,
  and reporting;
- synthetic end-to-end external-validation tests without real data;
- deterministic two-root publication tests;
- metric, bootstrap, CI, and internal-versus-external comparison tests;
- qualitative montage/failure-analysis JSON-first tests; and
- Gate 8 no-generated-artifact Git hygiene checks.

## Phase 8 Gate 8 Acceptance Criteria

Gate 8 is accepted only when:

- preprocessing, support policy, thresholds, checkpoint, and model-selection decision are frozen
  before external evaluation;
- the versioned external-evaluation preregistration JSON exists, self-validates, and predates real
  label access and real evaluation;
- real 3D-IRCADb-01 is evaluated without tuning under the frozen protocol;
- label mapping, scanner/acquisition/domain shift, inclusion/exclusion accounting, missing labels,
  incompatible cases, and all deviations from preregistration are documented in persisted artifacts;
- external segmentation metrics and bootstrap CIs are computed only from persisted predictions,
  labels, preregistration, manifest, and metric artifacts;
- internal-versus-external tables are descriptive and cannot alter frozen decisions;
- qualitative montages, failure analysis, and limitations are anonymous and generated only from
  persisted artifacts;
- final external report reproduces from frozen preregistration, checkpoint metadata, anonymous
  manifest, prediction artifact inventory, and metric artifacts;
- external labels never tune preprocessing, support policy, thresholds, checkpoint/model selection,
  stopping, postprocessing, report selection, or protocol iteration;
- no checkpoint selection occurs after external results;
- label mapping is frozen before evaluation and ledgered before labels are opened;
- independent scientific, engineering, and leakage reviews have no unresolved critical blockers;
- repository-wide lint, format, typing, tests, pre-commit, CLI help, synthetic Phase 8 smoke,
  deterministic synthetic two-root publication checks, and Git hygiene checks pass;
- no tracked medical data, dataset, local path, masks, predictions, embeddings, checkpoints,
  weights, credentials, HMAC keys, reverse maps, MLflow runs, or large generated artifacts are
  Git-visible; and
- close-out claims do not exceed directly supported evidence.

Planned final Gate 8 commands:

- `git status --short --branch`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src tests`
- `uv run pytest -q`
- `uv run pre-commit run --all-files`
- `uv run protoem-ct run-phase8-external-validation --help`
- two independent bounded synthetic Phase 8 executions into explicit external temporary roots;
- deterministic JSON/Markdown comparison across those roots;
- `git status --porcelain --untracked-files=all` after synthetic publication; and
- real external artifact hash verification only when explicitly approved and actually run.

### Phase 8 Definitive Training Orchestration Implementation Note

Status: implementation and synthetic-test completed locally on 2026-08-08 under
`PHASE8-DEFINITIVE-TRAINING-POLICY-V1` (see `docs/DECISIONS.md`). This substage extended
`src/protoem_ct/external/definitive_pipeline.py` with a locked training-execution policy (500 fixed
optimizer steps; checkpoint candidates only at steps 250 and 500; both requiring complete 20-case
development-validation evaluation; no subset-based checkpoint selection; mean-tumor-Dice checkpoint
selection with an earliest-step tie-break; no early stopping; no resume; a 10-hour hard watchdog; no
internal-test, external-data, or external-label use), one continuous single-trajectory
training-with-snapshot runner, full development-validation orchestration over exactly 20 unique
cases loaded one at a time with no persistent full-volume cache, checkpoint selection logic, and a
machine-readable checkpoint-selection-evidence artifact. All work is implementation-only: no
definitive training was executed, no real checkpoint was selected, no freeze occurred, and no
`/Volumes` or real dataset access occurred. Wave 4 and Wave 5 remain `BLOCKED`. This is not itself a
Wave 4/5 readiness release; it prepares orchestration for a future separately approved real
execution.

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
