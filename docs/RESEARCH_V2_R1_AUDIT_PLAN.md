# Research V2 R1 — Pixel-to-metric audit plan

Base R0 commit: `c0ebb42b7390b687d0d942371ea339c9fc7f3a6b`

## Scope and access boundary

R1 is a read-only audit of the already-locked development manifest and split. It must not mutate or regenerate either artifact.

Allowed array access in R1:

- `train`: 91 cases
- `validation`: 20 cases

Forbidden array access in R1:

- `internal_test`: 20 cases
- any external cohort, including 3D-IRCADb-01

Reading hashes and metadata in R0 did not authorize opening internal-test or external image/label arrays in R1. The R1 runner constructs its readable case list from the locked split before resolving or loading NIfTI arrays and fails closed if a requested case is not `train` or `validation`.

## Locked R0 identities

- manifest logical artifact hash: `c24244951e050050cf25c4b321f67d61c2087fc0c93fdcf9d112e0e488e1384b`
- manifest raw file SHA-256: `0e21a7d555e20b6011091bc18e462bc150cc23a8f46e522dbd8c01b014a44ae3`
- split logical artifact hash: `936376cd7b5e6070397c2fef16e5125c60fd6569ff3188d7e9bb5428a46ffadb`
- split raw file SHA-256: `416ca83e85c8598fc4f7065316153193211bf6b57875fbda4cafc01c360445f7`

The existing split is authoritative and is not regenerated in R1.

## Geometry and label contract

For LiTS-derived development data, the raw label contract under audit is `{0, 1, 2}`, with tumor defined as raw label `2`. Every allowed case must have finite image and label arrays, three spatial dimensions, integer-valued labels, matching image/label shapes, matching affines within the R1 tolerance, finite positive spacing, and an orientation code derivable from the affine.

The audit records qform/sform codes and matrices when available, native affine/orientation/spacing, raw label values, tumor voxel count, tumor connected-component count and volume, image intensity summary, and the transformation checks needed to reason about RAS reorientation, resampling, crop geometry, and inverse restoration.

Nearest-neighbour interpolation is required for label/binary-mask resampling. Continuous interpolation is required for image/probability resampling. R1 performs no model training and does not tune a probability threshold.

## Metric contract preregistration

R1 metric fixtures are evaluated before any real prediction analysis.

- Exact-metric comparison tolerance for Dice and IoU: absolute error `<= 1e-6`.
- Lesion connectivity: 26-connected components, matching the existing project baseline metric contract.
- Lesion matching: deterministic one-to-one assignment maximizing voxel overlap; a pair counts as a match only when overlap is non-zero.
- Surface connectivity: 6-connected surface extraction, matching the existing project baseline metric contract.
- Surface distances are measured in millimetres using native voxel spacing.
- HD95: the 95th percentile of concatenated bidirectional surface-to-surface distances. Both empty -> `0.0`; exactly one empty -> undefined (`None`).
- NSD tolerance for R1 fixtures: `1.0 mm`, fixed before execution. NSD is the fraction of concatenated bidirectional surface distances `<= 1.0 mm`. Both empty -> `1.0`; exactly one empty -> `0.0`.
- Numeric cross-check tolerance for HD95/NSD against the independent reference implementation: absolute error `<= 1e-6` on deterministic fixtures.

Inputs containing NaN/non-finite values, unknown label values, non-binary metric masks, geometry mismatch, or ambiguous probability/logit semantics must fail closed.

## Golden fixtures

The test suite includes known-answer fixtures for both masks empty, two separated lesions, a small lesion, axis flip, affine mismatch, and a shifted prediction. Dice, IoU and lesion matching are independently recomputed in the R1 reference code and compared with the project evaluator. HD95/NSD use the preregistered native-spacing contract above.

## Real-data audit and visual review

The Mac/Lexar R1 runner must audit exactly all 111 array-access-authorized development cases: 91 train + 20 validation. It must report the exact number actually opened by partition and must never open the 20 internal-test arrays.

At least 10 train cases are selected deterministically after metadata extraction to span differing tumor burden, lesion count, spacing and volume. For each selected train case, the runner writes a local-only tri-planar CT/tumor overlay (axial, sagittal, coronal) to the external evidence directory. These PNG files are review artifacts and must not be committed to Git.

Any real mismatch is recorded with anonymous case identifier, reason, and resolution state. Difficult cases are retained. R1 can pass only with zero unresolved mismatches and explicit human confirmation that the 10+ local overlays are anatomically aligned.

## Gate

R1 remains `BLOCKED_DATA_REVIEW` until real Mac/Lexar evidence is returned and reviewed. Synthetic/unit tests alone cannot promote R1 to PASS. PASS requires locked R0 identities, exactly 91 train + 20 validation arrays audited, zero internal-test/external array opens, zero unresolved mismatches, 10+ train overlays reviewed, metric/rejection/inverse-geometry tests passing, repository CI passing, and no model training or threshold tuning.
