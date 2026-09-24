# ProtoEM-CT Research V2 Protocol

Protocol ID: `protoem-ct-r2-2026-09-24`  
Schema: `research_v2_protocol.v1`  
Gate policy: `research_v2_gate_policy.v2`  
Base commit: `ae05bf41665f71339da533068fd5b5324c634669`

## 1. Scope and historical freeze

Research V2 is a new experimental line. It does not revise or reinterpret historical Phase 8.
Historical Phase 8 reports and definitive code paths remain protected. No target Dice or SOTA claim
is guaranteed; numeric claims require real artifacts and reviewable evidence.

## 2. Development cohort and partitions

LiTS and MSD Task03 Liver are one LiTS-derived development cohort, not independent sources. The
protocol binds to the existing patient-level split: 91 train, 20 validation, and 20 locked internal
test patients. R0 must not generate a replacement split.

The authoritative Phase 2 files are expected at:

```text
/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/manifest.json
/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/split.json
```

Artifact hashes and SHA-256 hashes of the exact files are distinct identities and must be computed
independently from the original bytes. Values quoted in historical reports are comparison references,
not substitutes for opening and hashing the files.

## 3. External data policy

Historical 3D-IRCADb-01 evaluation is considered seen for Research V2 and may be used only for
explicitly labeled exploratory analysis. A new external cohort or official hidden-label test is
required for confirmatory external assessment.

## 4. Fail-closed access matrix

| Purpose | Allowed data | Candidate lock required |
| --- | --- | --- |
| `fit` | train | no |
| `tune` | train + validation | no |
| `fold_planning` | train | no |
| `locked_evaluation` | internal test | yes |
| `external_confirmation` | new external | yes |
| `exploratory_external` | historical external | no |

Any unlisted combination is denied. Internal-test and external-confirmation access is denied during
tuning regardless of caller intent.

## 5. Split and leakage invariants

On the real artifacts all of the following must pass:

1. every manifest case appears exactly once in the split;
2. one patient appears in exactly one development partition;
3. case IDs are unique and partition coverage is exactly 91/20/20;
4. image SHA-256 values do not cross development partition boundaries;
5. label SHA-256 values do not cross development partition boundaries;
6. `split.source_manifest_hash` equals the independently verified manifest artifact hash;
7. a repeated audit produces the same audit hash.

A collision or mismatch is a hard failure. Cases are not dropped and a new split is not created to
make a gate pass.

## 6. nnU-Net fold policy

Five patient-grouped folds are created only from the 91 train patients using
`patient_hash_round_robin_v1`. Development validation and internal-test cases are never fold members.

`r0_audit` writes two separate files:

- `splits_final.json`: the direct nnU-Net input, whose root is the list of fold dictionaries;
- `splits_final.meta.json`: Research-v2 provenance including fold count, source split artifact hash,
  protocol ID, and SHA-256 of the canonical fold list.

Metadata must not wrap the direct nnU-Net split file.

## 7. Metrics and uncertainty

Primary development metric is patient-level macro tumor Dice on positive cases. Empty/undefined
cases are recorded separately. Secondary metrics include lesion recall/precision/F1, false-positive
lesions per scan, small-lesion sensitivity, volume error, HD95/NSD where defined, inference time,
failure rate, and patient-bootstrap 95% confidence intervals.

## 8. Reproducibility and provenance

Experiment records contain experiment ID, git commit, configuration hash, artifact and file hashes,
library versions, seed, resources, wall time, artifacts, metrics and failures. Raw medical data,
patient identifiers, checkpoints, weights and sensitive local paths are not committed to Git.

## 9. Environment contract

The Linux/CUDA environment is isolated under `environments/research-v2-linux-cuda/`. Its `uv.lock`
is tracked and later runs use `uv lock --check` plus `uv sync --frozen` rather than silently creating
a new lock.

CPU/Linux software reproducibility and CUDA hardware readiness are different checks. CPU-only CI can
prove the software contract but cannot prove GPU training readiness.

## 10. Gate policy amendment v2

`research_v2_gate_policy.v2` separates two gates because GPU hardware availability is irrelevant to
deterministic data auditing and CPU-only R1 inspection:

- **R0 research readiness** is required before R1 and requires the real-data audit plus Python 3.11
  software CI. It does **not** require CUDA hardware.
- **GPU training readiness** remains mandatory before any real `R2_training` or `R3_training` run and
  requires a CUDA-capable NVIDIA host, successful CUDA allocation/import smoke, driver/GPU/VRAM
  inventory, and persisted logs.

This is a versioned policy amendment, not a silent promotion. While the real Lexar data audit is
unverified, R0 remains blocked and R1 remains forbidden. After the data/software R0 gate passes, R1
may run on CPU even if GPU training readiness is still blocked.

## 11. Real-data audit procedure on the Mac

The first discovery run on the Mac must not require a protocol lock:

```bash
uv run --python 3.11 python -m protoem_ct.research_v2.r0_audit \
  --protocol configs/research_v2/protocol_v2.yaml \
  --manifest "/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/manifest.json" \
  --split "/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/split.json" \
  --output-dir "/Volumes/Lexar/ProtoEM-CT/runs/research_v2_r0_audit"
```

Compare the recomputed manifest and split artifact hashes to the accepted historical references.
If either differs, stop and investigate; do not create a fresh split. After all four independently
verified values (`manifest_hash`, `manifest_file_sha256`, `split_hash`, `split_file_sha256`) are
copied into the protocol, rerun the same audit with `--require-protocol-lock`.

Only non-PHI evidence may be committed. Original manifests, split files and patient identifiers stay
outside Git.
