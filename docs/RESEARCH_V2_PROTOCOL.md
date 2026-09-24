# ProtoEM-CT Research V2 Protocol

Protocol ID: `protoem-ct-r2-2026-09-24`  
Schema: `research_v2_protocol.v1`  
Base commit: `ae05bf41665f71339da533068fd5b5324c634669`

## 1. Scope and historical freeze

Research V2 is a new experimental line. It does not revise or reinterpret the historical Phase 8
result. The Phase 8 report, the published final report, the definitive Phase 8 pipeline, and
`image_only_inference.py` are protected historical surfaces. V2 code and configuration live under
new Research-v2 paths.

The objective is measurable improvement on the development cohort followed by locked evaluation.
No target Dice or SOTA claim is guaranteed. Numeric performance claims require real artifacts,
patient count, per-patient outputs, and uncertainty reporting.

## 2. Development cohort and partitions

LiTS and MSD Task03 Liver are treated as one LiTS-derived development cohort, not independent
sources. The protocol binds to the existing patient-level split:

- train: 91 patients; the only partition allowed to fit weights or dataset-derived training state;
- validation: 20 patients; model, threshold, and policy selection only;
- internal test: 20 patients; unavailable during fitting/tuning and read only after candidate lock.

The existing development manifest and split artifacts are authoritative. R0 does not generate a new
91/20/20 split. Their SHA-256 values must be copied into `protocol_v2.yaml` only after the actual
artifacts are materialized and independently verified. Until then, the R0 gate remains blocked.

## 3. External data policy

The historical 3D-IRCADb-01 evaluation is considered seen for Research V2. It may be used only for
explicitly labeled exploratory analysis. It cannot be used for model fitting, model selection,
threshold selection, post-processing selection, or a new confirmatory external claim.

A new external cohort or an official hidden-label test is required for confirmatory external
assessment. External confirmation is inaccessible until the candidate is frozen.

## 4. Fail-closed access matrix

| Purpose | Allowed data | Candidate lock required |
| --- | --- | --- |
| `fit` | train | no |
| `tune` | train + validation | no |
| `fold_planning` | train | no |
| `locked_evaluation` | internal test | yes |
| `external_confirmation` | new external | yes |
| `exploratory_external` | historical external | no |

Any unlisted combination is denied. An empty or implicit data request is denied. Internal-test and
external-confirmation access is denied during tuning regardless of caller intent.

## 5. Split and leakage invariants

Before any performance gate, the following invariants must hold on the real artifacts:

1. every manifest case appears exactly once in the development split;
2. one patient appears in exactly one development partition;
3. image SHA-256 values do not cross train/validation/internal-test boundaries;
4. label SHA-256 values do not cross train/validation/internal-test boundaries;
5. the split `source_manifest_hash` equals the verified manifest hash;
6. patient counts are exactly 91/20/20;
7. rerunning the same manifest/split verification produces the same audit hash.

A collision or mismatch is a hard failure. Cases are not dropped to make a gate pass.

## 6. nnU-Net fold policy

Five patient-grouped folds are created from the 91 train patients only. Research-v2 uses the stable
`patient_hash_round_robin_v1` policy implemented in `protoem_ct.research_v2.protocol`. Validation and
internal-test cases are never members of these folds. A materialized `splits_final.json` must be
generated from the verified real split artifact and stored as a non-PHI artifact before R3.

Dataset fingerprinting/planning derived from all 91 development-train cases must be documented as a
dataset-level shared feature when OOF estimates are reported. Fold model weights must be fit only on
the training patients of that fold.

## 7. Metrics and uncertainty

Primary development metric: patient-level macro tumor Dice on positive cases. Empty/undefined cases
are recorded separately. Secondary metrics include lesion recall, precision and F1, false-positive
lesions per scan, small-lesion sensitivity, volume error, HD95/NSD where defined, inference time,
failure rate, and patient-bootstrap 95% confidence intervals.

Connectivity, lesion matching, surface tolerance, empty-case policy, and post-processing thresholds
must be locked before the experiment that uses them. Comparisons use paired patient-level
differences on the same cases.

## 8. Reproducibility and provenance

Every experiment record contains: experiment ID, git commit, configuration hash, manifest hash,
split hash, data hash/fingerprint where legally recordable, library versions, seed, GPU model,
VRAM, wall time/GPU hours, artifact path, preregistered metric, per-patient outputs, and failures.

Raw medical data, PHI, checkpoints, weights, and sensitive local paths stay outside Git. Only
anonymous manifests, hashes, configuration, code, and non-sensitive reports are tracked.

## 9. Environment contract

The Research-v2 GPU environment is isolated under `environments/research-v2-linux-cuda/` and pins
the Python ML stack separately from the historical Intel/macOS CPU environment. A lockfile must be
created and checked on a networked Linux host, then the environment must pass dependency import and
GPU smoke tests. Runtime CUDA driver, GPU model, RAM/VRAM and wall time are recorded in the gate.

CPU-only CI proves software contracts, not real GPU performance.

## 10. R0 gate

R0 can be marked `PASS` only after all of the following are true:

- `protocol_v2.yaml` validates against the executable contract;
- real manifest and split artifacts are available and their hashes are locked into the protocol;
- patient, case, image-hash and label-hash isolation checks pass on the real 131-case development
  cohort;
- five train-only nnU-Net folds are materialized from the verified split;
- tuning attempts to access internal-test or either external confirmation surface fail closed;
- repeated manifest/split audit yields the same audit hash;
- the pinned Python 3.11 environment is installed from a checked lock;
- lint, format, type, unit, integration and smoke tests pass;
- a real CUDA-capable GPU dependency/import smoke test passes and runtime resources are recorded.

If the real development artifacts, Python 3.11 environment, dependency lock, or CUDA GPU are not
available, the gate is `BLOCKED`; it is not promoted to `PASS` from synthetic tests.

## 11. Artifact identity versus file identity

Research V2 records two different SHA-256 concepts and never substitutes one for the other:

- `manifest_hash` / `split_hash` are schema-defined artifact hashes recomputed from the canonical
  artifact payload after excluding the artifact's own hash field;
- `manifest_file_sha256` / `split_file_sha256` are SHA-256 digests of the exact JSON file bytes.

R0 requires both pairs to be recomputed from the materialized real files. A value quoted in a report
or decision log is only a discovery reference until the corresponding file is opened and verified.
The accepted historical Phase 2 decision log identifies the expected external locations as
`/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/manifest.json` and
`/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/split.json`; those paths are not treated as
current evidence unless the volume is actually mounted in the execution environment.

When the files are available, run:

```bash
python -m protoem_ct.research_v2.r0_audit \
  --protocol configs/research_v2/protocol_v2.yaml \
  --manifest /absolute/path/to/manifest.json \
  --split /absolute/path/to/split.json \
  --output-dir /absolute/non-medical-output/r0-discovery
```

After independently copying the four recomputed hashes into `protocol_v2.yaml`, rerun the same
command with `--require-protocol-lock`. Only the locked rerun can satisfy the real-data part of R0.
