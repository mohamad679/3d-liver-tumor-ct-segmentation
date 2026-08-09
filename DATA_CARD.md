# Data Card — ProtoEM-CT Phase 10

## Data Scope

ProtoEM-CT studies binary liver-tumor segmentation in 3D CT. The final Phase 10 package reports from saved machine-readable artifacts only; raw medical datasets are not committed to Git.

## Development Source

The development source is LiTS/MSD Task03 Liver represented as one LiTS-derived development cohort. LiTS and MSD Task03 Liver are not independent cohorts and must not be counted, tuned, or analyzed as separate sources.

Artifact-backed Phase 10 development counts:

- Development cases: `131`
- Train cases: `91`
- Validation cases: `20`
- Immutable internal-test cases: `20`
- Geometry QA passed cases: `131`
- Geometry QA failed cases: `0`
- Development lesion count: `908`

All development split assignments are patient-level and patient-disjoint.

## External Source

The external cohort is 3D-IRCADb-01. Phase 8 used the external cohort only for the locked external-validation protocol.

Artifact-backed external accounting:

- Discovered cases: `20`
- Inference-eligible cases: `20`
- Evaluation-eligible cases: `15`
- Excluded cases: `5`

External labels were not used for tuning, model selection, threshold selection, preprocessing decisions, support-policy decisions, or protocol iteration.

## Labels And Semantics

The primary task is binary tumor foreground versus background. Liver context may be used for preprocessing, quality control, masking, or analysis, but the reported segmentation target is tumor delineation.

## Leakage Controls

- Patient-level development splits.
- Immutable internal-test partition.
- External cohort reserved for Phase 8.
- Preregistration and prediction lock before external label evaluation.
- No external-label tuning.
- LiTS/MSD treated as one development source.

## Access And Licensing Caveats

Dataset access depends on the original dataset providers and local lawful availability. Raw medical files, DICOM/NIfTI arrays, PHI-bearing manifests, predictions, checkpoints, model weights, credentials, and secrets are not tracked in this repository.

## Limitations

- External evaluation is small after eligibility accounting.
- Phase 10 does not create new data, labels, cohorts, or experiments.
- Several planned Phase 3-7 performance surfaces lack durable machine-readable final-package artifacts and are therefore marked unavailable rather than inferred.
