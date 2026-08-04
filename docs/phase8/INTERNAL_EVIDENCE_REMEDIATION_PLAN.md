# Phase 8 Wave 4 Internal-Evidence Remediation Plan

## Scope

This document records Wave 4 blocker-remediation planning only. It does not implement remediation,
train a model, create a checkpoint, rerun readiness, release Wave 5, or claim Gate 8.

Real delegated planning agents were used for:

- `P8R-DEVELOPMENT-DATA-AUDIT`
- `P8R-TRAINING-READINESS-AUDIT`
- `P8R-SELECTION-FREEZE-PLAN`
- `P8R-COMPUTE-EXECUTION-PLAN`

All four agents were read-only. No training, inference, external-label access, external-metric
access, checkpoint creation, generated-run mutation, or NIfTI pixel-array opening occurred.

## Approved Development Evidence

The approved development artifact set for later remediation is the Phase 2 v2 set under
`/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2`.

| Artifact | Path | Embedded identity |
| --- | --- | --- |
| Manifest | `/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/manifest.json` | `manifest_hash=c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b` |
| Split | `/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/split.json` | `split_hash=936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb` |
| Geometry QA | `/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/geometry_qa.json` | `qa_artifact_hash=b8e65c558d5f574a1b40f9a5b15a033efbfbd6952006324fae35173940f94ee2` |
| Lesion components | `/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/lesion_components.json` | `lesion_artifact_hash=f7ce4874801cc8d5387bf93fd1480d376db4028026a90723dda2c5e2f48fbc60` |
| Development summary | `/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/development_summary.json` | `summary_artifact_hash=9b721e10232a08c9a5dd5361a11d7595abab04c071058d584e5895087b91951c` |
| Final QA report | `/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/development_qa_report.json` | `qa_artifact_hash=3f816db6227a1d2a8b349e248a9654eb4eed1e57760b48d32a6a275e1deca4f1` |
| Leakage audit | `/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/leakage_audit.json` | `audit_hash=a32da3ac9d5689ec055c3875076ddc2075ca585219160c2d4200ca36484bf199` |

The v2 split has 91 train, 20 validation, and 20 immutable internal-test cases and patients. The
v2 QA status is 131 passed and 0 failed. The v2 lesion summaries cover 131 analyzed cases, 0
skipped cases, and 908 lesions. The v2 leakage audit passed with zero patient, case, image-hash,
label-hash, and image/label-pair overlap across partitions.

The older `phase2_real_lits_v1` set is noncurrent for Phase 8 planning. It remains preserved
externally but has different manifest and split hashes and is superseded by the committed v2
acceptance and leakage evidence.

## Current Executability

No implemented pathway can currently produce a freeze-eligible real-development Phase 8 checkpoint
package end to end.

| Pathway | Current status |
| --- | --- |
| MONAI SegResNet baseline | Checkpoint-producing code exists, but it is bounded to the Phase 3 synthetic fixture, CPU, and AMP-disabled execution. |
| nnU-Net v2 baseline | Command builders and subprocess contracts exist; real full training/inference orchestration and freeze package are not implemented. |
| `head_only` | Parameter-selection contract exists only; no real adaptation training, resume, checkpoint, or validation-selection orchestration exists. |
| `decoder_only` | Parameter-selection contract exists only; no real adaptation training, resume, checkpoint, or validation-selection orchestration exists. |
| `full_finetune` | Parameter-selection contract exists only; no real adaptation training, resume, checkpoint, or validation-selection orchestration exists. |
| `prototype_only` | Prototype contracts exist, but the public Phase 5 path is synthetic-only. |
| ProtoEM `fixed_em_like` | Objective and publication contracts exist, but the public Phase 6 path is synthetic-only. |

The first freeze-eligible candidate set must include only methods with implemented real-development
training, validation prediction, validation metric, checkpoint metadata, and failure-record
publication. At planning time, no such candidate set is executable. The conservative proposed first
candidate after remediation is a single `monai_segresnet_baseline` candidate, if and only if a real
development runner and selection package are implemented and validated.

## Five-Blocker Dependency Graph

1. `preprocessing_decision`
   - Depends on an approved real preprocessing policy and provenance artifact.
   - Must state whether preprocessing is train-only fitted or no-fit fixed.
   - Must link to the approved v2 manifest hash, v2 split hash, selected candidate, and selected
     checkpoint.

2. `checkpoint_metadata`
   - Depends on an implemented real-development run and a checkpoint file outside Git.
   - Must record checkpoint SHA-256, byte size, format, model family, candidate ID, seed, device,
     AMP flag, training config hash, preprocessing hash, Git commit, package versions, status,
     validation metric artifact hash, and failure codes where applicable.

3. `model_selection_decision`
   - Depends on a predeclared executable candidate inventory, completed candidate run records, and
     validation-only metric artifacts.
   - Must not use immutable internal-test, external images, external labels, Wave 2/3 external
     properties, or external metrics.

4. `threshold_decision`
   - Depends on the selected candidate and either a fixed no-tuning threshold or a
     validation-only threshold artifact.
   - The preferred conservative default is probability threshold `0.5`. The synthetic Phase 6
     threshold `0.55` is not freezeable.

5. `support_policy`
   - Depends on the selected workflow.
   - Conservative default is explicit no-support for external inference:
     `no_external_support_labels` and `no_support_for_external_inference`.
   - Any support alternative remains unresolved unless an internal-only support execution path is
     implemented and frozen before external evaluation.

Wave 4 readiness can be rerun only after all five artifacts exist and the Wave 4 readiness code can
discover or be extended to validate them without accepting synthetic-only evidence.

## Selection Procedure

Candidate inventory must be frozen before execution. It must include only executable candidates. For
each candidate, train on the development train partition and evaluate selection metrics on the
development validation partition only. The immutable internal test remains locked and unused until
after model, checkpoint, preprocessing, threshold, and support/no-support decisions are frozen.

Primary selection metric:

- validation tumor Dice from the accepted Phase 3 binary-tumor metric contract.

Required secondary reporting:

- tumor IoU
- tumor HD95
- tumor normalized surface Dice
- lesion-wise recall
- lesion-wise precision
- lesion F1
- false-positive lesions per scan
- tumor volume error

Deterministic tie-break order:

1. Higher validation tumor Dice.
2. Higher validation lesion F1.
3. Lower validation HD95 among defined cases.
4. Lower false-positive lesions per scan.
5. Smaller trainable-parameter count.
6. Lexicographic candidate ID.

Failed candidates must publish a failure artifact with candidate ID, config hash, failure code, and
missing outputs. Failed candidates are ineligible unless all candidates fail. If all candidates
fail, freeze remains blocked.

## Compute Plan

Tier 1 is a no-data feasibility check: CLI help and import/device probes only. It requires no user
approval and produces no run artifacts.

Tier 2 is a tiny real-development dry or overfit verification after implementation. This is the
first point raw development NIfTI pixel arrays are required. It requires explicit user approval,
must use a tiny train/validation subset, and must write only outside Git.

Tier 3 is definitive development training and selection. This is the first point long compute is
expected. It requires explicit user approval. In the current environment, CUDA is unavailable and
MPS is not available; CPU-only definitive training is possible in principle but likely impractical
for full 3D training. No runtime estimate is claimed.

Current compute blockers:

- no real-development training CLI;
- existing MONAI and nnU-Net Phase 3 paths are synthetic or CPU-constrained;
- Phase 5 and Phase 6 CLIs are synthetic-only;
- no available CUDA or MPS device in the inspected environment;
- AMP is disabled in current accepted configs and contracts;
- no real-run resume contract is implemented.

## Remediation Substages

Each substage should end with a small Git checkpoint after review. Generated run outputs,
checkpoints, predictions, MLflow runs, logs, and medical data remain outside Git.

1. Real-development evidence schema and readiness-input contracts.
   - Add typed artifacts for preprocessing decision, candidate inventory, checkpoint metadata,
     validation metrics link, model-selection decision, threshold decision, and support/no-support
     policy.
   - First exact implementation action: define these artifact schemas and tests without opening
     NIfTI pixels.

2. Real MONAI SegResNet development runner scaffold.
   - Implement manifest/split loading, train/validation partition filtering, no-overwrite external
     output roots, deterministic preprocessing provenance, checkpoint metadata publication, and
     validation prediction/metric publication.
   - Keep `monai_segresnet_baseline` as the only candidate unless additional real pathways are
     implemented before candidate inventory freeze.

3. Tiny real-development dry/overfit verification.
   - Requires user approval because raw development NIfTI pixels are first opened here.
   - Use a bounded tiny subset and a fresh external output root.
   - Do not claim checkpoint eligibility from the tiny dry run unless explicitly marked as a dry-run
     artifact and excluded from selection.

4. Definitive development training and validation-only selection.
   - Requires user approval and likely long compute.
   - Produces real candidate run outputs, validation metrics, selected checkpoint metadata, and
     failed-candidate records if applicable.

5. Freeze missing decisions and optional immutable internal-test evidence.
   - Freeze model selection, checkpoint metadata, preprocessing decision, threshold decision, and
     support/no-support policy before any immutable internal-test evaluation.
   - If internal-test evidence is produced, it must occur after selection freeze and cannot feed
     tuning or selection.

6. Wave 4 readiness remediation.
   - Extend or rerun Wave 4 readiness against the completed internal evidence package.
   - Generate `phase8_decision_freeze.json` and `phase8_external_preregistration.json` only if the
     readiness state becomes `READY`.

Wave 5 remains blocked until Wave 4 is rerun successfully and the Supervisor explicitly releases it.

## Outputs Outside Git

The following remain outside Git:

- raw development NIfTI images and labels;
- preprocessing fit artifacts and preprocessed tensors;
- checkpoints, model weights, optimizer state, and resume state;
- predictions and probability maps;
- validation and internal-test metric JSON outputs if generated as run artifacts;
- logs, MLflow directories, runtime traces, and temporary nnU-Net directories;
- external Wave 2-4 generated JSON artifacts and future Phase 8 generated artifacts.

## Unresolved Scientific Decisions

- Whether Phase 8 should use a baseline-only model or require a real ProtoEM workflow before
  external validation.
- Whether the first candidate inventory should remain single-candidate MONAI SegResNet or wait for
  real nnU-Net or ProtoEM orchestration.
- Whether thresholding is fixed at `0.5` or selected on development validation.
- Whether external inference uses explicit no-support or a frozen internal-only support policy.
- Whether practical definitive training requires access to a CUDA/MPS-capable host.

Gate 8 is not claimed. Wave 5 remains blocked.
