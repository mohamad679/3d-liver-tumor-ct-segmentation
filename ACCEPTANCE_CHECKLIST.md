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

- [x] Validated Phase 5 initialization boundary is implemented and verified
- [x] Pure deterministic E-step and M-step are implemented and verified
- [x] Support, entropy, class-balance, consistency, proximal, and total objectives are recorded
- [x] Confidence masking is implemented and verified
- [x] Bounded fixed-iteration and tolerance stopping are implemented
- [x] Foreground, background, and no-confident-voxel collapse handling is implemented
- [x] NaN and other non-finite failures are handled explicitly
- [x] Complete per-iteration objective trace is persisted and self-validating
- [x] `fixed_em_like` baseline remains available and unchanged as the default baseline
- [x] Explicit parameterized positive-step schedule is implemented without training
- [x] Final inference is produced only for completed runs
- [x] Deterministic publication is implemented and verified from persisted JSON artifacts
- [x] Deterministic 12-row ablation comparison is implemented and verified
- [x] Query-label leakage is rejected from initialization and optimization APIs
- [x] No Phase 7 robustness, uncertainty, calibration, or external-validation functionality is claimed

### Gate 6 Evidence

Gate 6 status: PASSED locally on 2026-08-02 after one repository defect fix.

Repository-wide verification after the fix:

- `uv run ruff check .`: PASS, `All checks passed!`
- `uv run ruff format --check .`: PASS, `182 files already formatted`
- `uv run mypy src`: PASS, `Success: no issues found in 76 source files`
- `uv run pytest -q`: PASS, `1115 passed, 3 skipped in 602.21s (0:10:02)`
- `uv run pre-commit run --all-files`: PASS
- `uv run protoem-ct run-phase6-protoem --help`: PASS

Required source/test modification during Gate 6 evaluation:

- `tests/unit/test_phase6_inference.py`: formatting-only fix applied after the first
  `uv run ruff format --check .` reported `File would be reformatted`. No scientific behavior,
  source logic, objective, optimization, publication, or Phase 5 behavior was changed.

Independent synthetic Phase 6 executions:

- `ROOT_A="$(mktemp -d /tmp/protoem-ct-phase6-gate-a.XXXXXX)"`
- `ROOT_B="$(mktemp -d /tmp/protoem-ct-phase6-gate-b.XXXXXX)"`
- `uv run protoem-ct run-phase6-protoem --config configs/phase6_protoem_ct.yaml --output-root "$ROOT_A"`:
  PASS
- `uv run protoem-ct run-phase6-protoem --config configs/phase6_protoem_ct.yaml --output-root "$ROOT_B"`:
  PASS

Synthetic run results from the persisted artifacts:

- Execution status: `completed` for both runs
- Config identity: `f408a3d2adfc746a7ac6b3de908b4e1a97436e2f22d5a8eab18f20e570a5437c`
- Initialization identity: `055266b363f5a9aa3455091acc3a4e4ed5d6e0c04ebb4f30811db6041ac9e8fa`
- Objective-trace hash: `acc3b3db5fd5627cc43559a4bf9dad1b006a39e6c4d9a97420b977e189fcf602`
- Stopping-record hash: `dc285d9b281e92c3274652c8dfbac3fc58f684d96b6ec67af467a9296d199c4a`
- Stopping reason: `max_iterations`
- Completed iteration count: `1`
- Converged: `false`
- Failed: `false`
- Final inference exists: `true`

Objective-trace verification:

- Trace is non-empty: `1` iteration
- Iteration indices are zero-based and contiguous: `[0]`
- Trace self-validation succeeded through the Phase 6 serializer
- The persisted iteration record contains:
  `support_objective`, `query_entropy_objective`, `class_balance_objective`,
  `consistency_objective`, `proximal_objective`, `total_objective`,
  `confident_voxel_count`, `foreground_assignment_count`, `background_assignment_count`,
  `foreground_fraction`, `convergence_delta`, `finite_status_ok`,
  `collapse_status_detected`, `state_identity_hash_before`, `state_identity_hash_after`,
  `prototype_identity_hash_before`, and `prototype_identity_hash_after`

Deterministic publication and reproducibility:

- JSON and Markdown outputs were byte-identical across the two synthetic runs for:
  `ablation_comparison.json`, `ablation_comparison_table.md`, `ablation_plan.json`,
  `ablation_run_inventory.json`, `effective_config.json`, `final_inference.json`,
  `initialization_summary.json`, `objective_trace.json`, `phase6_summary.md`,
  `run_summary.json`, and `stopping_record.json`
- `convergence_plot.png` existence was verified and regeneration from
  `objective_trace.json` matched the saved PNG bytes on this platform
- No generated Gate 6 artifacts were found inside the repository worktree

Verified 12-row ablation comparison statuses in canonical order:

1. `full_protoem`: `executed`
2. `no_retrieval`: `phase5_baseline_link`
3. `no_transduction`: `phase5_baseline_link`
4. `no_class_balance`: `executed`
5. `no_proximal`: `executed`
6. `single_prototype`: `executed`
7. `multiple_prototypes`: `unsupported`
8. `fixed_update_schedule`: `executed`
9. `learned_update_schedule`: `unsupported`
10. `head_only`: `provenance_only`
11. `decoder_only`: `provenance_only`
12. `full_finetune`: `provenance_only`

Leakage and scope evidence:

- Query labels and query reference masks are absent from the public initialization and optimization
  APIs by contract and by the Phase 6 unit/integration tests
- The only reference-mask use in the Phase 6 comparison flow is post-prediction metric evaluation
  for synthetic testing; it does not alter initialization identity, optimization traces, stopping
  records, inference identities, or prediction hashes
- Positive-step parameters, when present, are explicit caller-supplied parameters. They are not
  trained from data in Phase 6
- Gate 6 does not claim theoretical convergence, real-data execution, GPU execution, robustness,
  uncertainty, calibration, external validation, or any Phase 7 result

## Gate 7

- [ ] Pending definition.

## Gate 8

- [x] Phase 9 skipped by user decision / not executed
- [x] P10-A artifact inventory generated from saved machine-readable artifacts
- [x] P10-B final tables generated from artifact-backed inventory values
- [x] P10-B final figures generated from artifact-backed numeric values
- [x] P10-B statistical summaries collated from preregistered saved Phase 8 artifacts
- [x] P10-C technical report generated from P10-A/P10-B machine-readable outputs
- [x] Reproduction README created
- [x] MODEL_CARD.md completed without clinical-use or strong-generalization claim
- [x] DATA_CARD.md completed with LiTS/MSD non-independence and external-label no-tuning policy
- [x] One-page CV project summary created
- [x] Motivation-letter evidence paragraph created
- [x] Release checklist created
- [x] Release-candidate metadata created
- [x] `uv run ruff format --check .` passes
- [x] `uv run ruff check .` passes
- [x] `uv run mypy src tests` passes
- [x] `uv run pytest -q` passes
- [x] `make smoke` passes
- [x] `uv run snakemake --cores 4 --rerun-incomplete reports/final_report.html` passes
- [ ] P10-R independent review passes
- [x] Gate 8 passes: final report generated entirely from machine-readable artifacts

### Gate 8 Evidence

Gate 8 status: PASS for artifact-only final-report reproduction; P10-R independent review pending.

Current artifact identities:

- P10-A inventory: `reports/phase10/artifact_inventory.json`
- P10-B outputs manifest: `reports/phase10/phase10_b_outputs_manifest.json`
- Final report manifest: `reports/phase10/final_report_manifest.json`
- Final HTML report: `reports/final_report.html`
- Release candidate metadata: `reports/phase10/release_candidate.json`

Phase 10 does not rerun training, inference, prediction generation, checkpoint selection, threshold
selection, preprocessing changes, or external-label tuning. External validation remains an honest
negative result, and LiTS/MSD Task03 Liver is treated as one LiTS-derived development source.

## Gate 8

- [x] Preregistration published before any external label access
- [x] Image-only external inference completed and predictions locked before any external label
      access
- [x] External labels opened only after preregistration and prediction lock, using the
      unmodified, previously frozen label-mapping/eligibility policy
- [x] All 9 preregistered external metrics computed with case-level bootstrap confidence
      intervals (10,000 resamples, seed 1729, 95% percentile)
- [x] Descriptive-only internal-vs-external comparison published, with no superiority or
      generalization claim
- [x] One known non-scientific comparison-artifact provenance defect identified, minimally
      fixed under a focused regression test, and a corrected artifact published without
      overwriting the original historical record
- [x] Final report documents the objective, frozen configuration, preregistration/lock/
      evaluation identities, all 9 external metrics, limitations, and an explicit negative-result
      interpretation
- [x] Non-scientific reproduction/provenance chain verified from development provenance through
      final report, with a machine-readable closure record
- [x] Lint, formatting, typing, and full local tests pass for all Phase 8 closure changes
- [ ] GitHub-hosted CI passes (not established this session)

### Gate 8 Evidence

Gate 8 status: PASSED locally on 2026-08-09 with a **negative external-validation result**.

External tumor-segmentation performance on the 15 evaluation-eligible 3D-IRCADb-01 cases was very
poor across all 9 preregistered metric families (macro tumor Dice `0.01412`; see
`docs/phase8/FINAL_REPORT.md` for the complete result set). The selected checkpoint also had very
low development-validation tumor Dice (`0.01579295321113191`) before any external evaluation. No
external tuning of any kind occurred at any point. This is reported as an honest negative result,
not reframed as generalization or partial success. Phase 8 is closed on the basis that the locked,
preregistered protocol was executed exactly as designed and reported completely and transparently
— not on the basis of a favorable scientific outcome.

See `docs/phase8/FINAL_REPORT.md` for the full closure report, including frozen configuration
identities, all 9 external metrics with confidence intervals, domain-shift and qualitative-output
references, the corrected internal/external comparison artifact, limitations, and the
reproduction/provenance chain.
