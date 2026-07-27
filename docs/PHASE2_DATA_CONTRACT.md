# Phase 2 Data Contract

## Status and Boundary

Status: accepted planning contract; implementation and real-data execution have not started.

Phase 2 establishes reproducible, read-only development-cohort inventory, anonymous manifests,
patient-level splits, data QA, leakage evidence, and reporting. It includes:

- a LiTS-style NIfTI development-cohort adapter;
- a 3D-IRCADb-style adapter interface exercised only with synthetic format tests;
- anonymous machine-readable dataset manifests;
- deterministic patient-level development splits;
- orientation, spacing, affine, shape, label, and finite-value QA;
- connected-component lesion summaries;
- fixed-bin CT intensity histograms and lesion-size summaries;
- machine-readable development QA artifacts;
- a Markdown QA report generated only from saved artifacts;
- patient-overlap and leakage audit automation; and
- synthetic edge-case tests.

Phase 2 explicitly excludes model training, baseline implementation, preprocessing fitting,
few-shot support selection, robustness experiments, external evaluation, and LLM/VLM work.

## Cohort Policy

- LiTS labeled training data is the sole development cohort for Phase 2.
- LiTS and MSD Task03 Liver must never be treated as independent cohorts. MSD-style layout may
  describe the same LiTS cases, but it does not create a second cohort.
- Development train, validation, and immutable internal-test assignments must be patient-disjoint.
- Real 3D-IRCADb-01 data remains untouched until Phase 8 external validation.
- Phase 2 may implement and synthetic-test a 3D-IRCADb-style adapter, but it must not scan, QA,
  split, tune on, or generate statistics from the real external cohort.
- External labels must never influence development preprocessing, stratification, thresholds,
  model selection, tuning, protocol design, or any other development decision.

## Data-Location and Output Policy

- Dataset roots are supplied only through explicit CLI arguments or ignored local configuration.
- No absolute local dataset or generated-root path may be committed.
- No medical image, label, manifest containing protected health information (PHI), prediction,
  derived volume, source-identifier mapping, credential, or secret may be committed.
- Outputs are written only beneath an explicit external generated root. The implementation must
  reject missing, relative, repository-contained where prohibited, or unsafe output roots.
- Tracked examples and all automated tests use synthetic data only.
- Adapter code must never search the whole computer, inspect parent directories outside its
  explicit dataset root, or infer a dataset location from the current working directory.
- Source data is read-only. Inventory, manifest, split, QA, and report outputs must not be written
  into the dataset root.

## Anonymous Dataset Manifest

### Dataset-level schema

The versioned manifest is deterministic JSON and contains at least:

- `schema_version`
- `manifest_type`
- `dataset_id`
- `cohort_role`
- `adapter_name`
- `adapter_version`
- `generated_at_utc`, supplied explicitly rather than read from the system clock
- `git_commit`, supplied explicitly rather than inferred from Git state
- `dataset_root_fingerprint`, which identifies the intended root without storing its absolute path
- `manifest_hash`
- `case_count`
- `cases`, an ordered array

`manifest_type` distinguishes the dataset manifest from split and QA artifacts.
`cohort_role` uses an explicit development or external role; Phase 2 publication is permitted only
for the LiTS development role. The exact fingerprint construction must be versioned, deterministic,
one-way, exclude the root's absolute text, and be tested before real manifest generation.

Each case contains at least:

- `anonymous_patient_id`
- `anonymous_case_id`
- `image_path`, relative to the explicit dataset root
- `label_path`, relative to the explicit dataset root where labels are allowed
- `image_sha256`
- `label_sha256` where labels are allowed
- `cohort_role`

### Identity, path, uniqueness, and hash rules

- Anonymous IDs are deterministic and contain no raw patient name, medical-record identifier,
  birth date, accession number, or original DICOM identifier.
- The contract does not prescribe reversible pseudonymization.
- If a stable mapping from source identifiers is necessary, the implementation uses a one-way,
  deterministic project namespace, verifies collisions explicitly, and never commits a mapping back
  to source identifiers.
- Paths use normalized POSIX-style relative representation, are nonempty and non-absolute, and
  reject `..`, home expansion, drive-qualified paths, empty path segments, and ambiguous aliases.
- Resolved paths must remain beneath the explicit dataset root.
- Cases are ordered deterministically by documented non-identifying keys.
- Duplicate anonymous patient IDs, anonymous case IDs, resolved image files, and resolved label
  files are rejected. A duplicate patient ID is an error in the dataset manifest because the Phase 2
  LiTS contract expects one case record per patient; a later multi-case design requires an explicit
  schema revision.
- File SHA-256 values are calculated by streaming reads without modifying source files.
- `case_count` must equal the number of ordered cases.
- Manifest hashing uses canonical deterministic JSON and excludes only the manifest's own
  `manifest_hash` field. Serialization ends with a newline and rejects duplicate JSON keys, NaN, and
  Infinity.

## Adapter Interface

A typed, read-only adapter protocol conceptually provides:

1. discovery of candidate cases beneath an explicit dataset root using explicit layout rules;
2. validation of image/label pairing;
3. extraction of non-identifying case keys only; and
4. deterministic case records suitable for manifest construction.

Adapters must:

- accept configurable layout rules instead of assuming every local copy uses identical folder or
  file naming;
- reject ambiguous matches, duplicate pairs, missing images or labels, unsafe paths, and unsupported
  layout states with explicit errors;
- never mutate, rename, repair, copy, or reorganize source data;
- resolve candidate paths and reject symlinks that leave the dataset root; and
- return the same ordered case records for the same root contents and layout configuration,
  regardless of filesystem enumeration order.

Real LiTS ingestion requires a separate explicit dry-run inventory and human review before manifest
publication. The dry run may report anonymous candidate counts and structural failures to the
external generated root; it must not publish a manifest automatically. During Phase 2, only synthetic
fixtures may exercise the 3D-IRCADb-style adapter.

## Deterministic Development Split Contract

Splits are created at patient level, never at volume or slice level. A split request requires:

- an explicit seed;
- an explicit split-policy name and version;
- deterministic patient ordering before assignment;
- train, validation, and immutable internal-test partitions; and
- the exact source dataset `manifest_hash`.

The split artifact must establish:

- zero patient overlap among every partition pair;
- zero case overlap among every partition pair;
- every source patient and case assigned exactly once;
- partition membership linked to the source dataset manifest hash;
- byte-identical split JSON when rerun with the identical manifest, seed, policy version, and
  approved partition configuration; and
- immunity of assignments to changes in input file enumeration or manifest presentation order.

No data-dependent preprocessing is fitted before or during split creation. Development lesion
statistics may be used for documented and audited stratification only after QA artifacts exist, only
under a versioned policy, and never with external data or external labels.

Proposed configurable defaults are `70%` train, `15%` validation, and `15%` immutable internal test,
with deterministic handling for rounding and small patient counts still to be specified. These are
planning defaults only: final ratios, minimum partition sizes, stratification variables and bins,
and the rounding policy require explicit user approval before any real split is generated.

## QA Contract

### Per-case machine-readable fields

Every discovered case, including invalid cases where reading progresses far enough, receives a
deterministic per-case QA record containing at least:

- anonymous patient and case IDs plus source manifest hash;
- dimensionality;
- image and label shape;
- image dtype;
- label dtype;
- image and label affine;
- image and label orientation;
- image and label voxel spacing;
- finite image-values status;
- finite label-values status;
- observed and allowed label values;
- image/label shape-match status;
- image/label affine-match status and the tracked absolute tolerance;
- image intensity minimum, maximum, mean, and standard deviation;
- fixed CT histogram bin edges and counts;
- tumor voxel count;
- tumor physical volume;
- connected-component lesion count;
- per-lesion voxel and physical volumes in deterministic order;
- empty-tumor status;
- overall QA pass/fail; and
- explicit, stable failure reason codes and human-readable details.

The persisted schema must distinguish unavailable values from valid zero values. Affines, spacing,
and floating-point summaries require deterministic JSON representations and explicit units.

### Scientific and processing conventions

- Histogram bin edges are fixed by tracked configuration shared across development QA; bins are not
  fitted separately per case or cohort. The final CT range and bin width/count require explicit
  approval before real QA.
- Connected components use an explicit tracked 3D connectivity. Proposed default: 26-connectivity;
  the final connectivity requires explicit approval before real QA.
- Physical volume in cubic millimetres is voxel count multiplied by the product of the three
  positive voxel-spacing values. Reported millilitres, if included, are derived as
  `cubic_millimetres / 1000`.
- Lesions are ordered deterministically, proposed by descending voxel count with a stable spatial
  tie-breaker; the final tie-break rule must be accepted before real QA.
- QA performs no resampling, reorientation, normalization, clipping, repair, label remapping,
  interpolation, or preprocessing fitting.
- Invalid cases are represented in QA output and fail with explicit reasons; they are never silently
  corrected or skipped.
- Patient-level and dataset-level summaries, including lesion-size summaries and histogram totals,
  are generated only from saved per-case QA JSON.
- The development Markdown QA report is generated only from validated saved QA and summary JSON. It
  does not open NIfTI/DICOM files or recompute statistics from medical volumes.

## Required Synthetic Edge Cases

Synthetic tests cover at least valid alternate layouts, shuffled discovery order, missing pairs,
ambiguous pairs, duplicate IDs and files, unsafe and absolute paths, `..` traversal, symlink escape,
shape and affine mismatch, invalid dimensionality, invalid spacing, unexpected label values, NaN and
infinite values, empty tumors, multiple connected components, connectivity-sensitive lesions,
histogram boundary values, pseudonym collision handling, manifest hash tampering, and split overlap
or incomplete-assignment rejection.

All hosted-CI tests use synthetic fixtures generated beneath test temporary directories. No test
requires a local dataset path, real cohort availability, network access, or external labels.

## Real-Data Execution Approval Boundary

The real LiTS dry-run inventory and all later real LiTS manifest and QA commands require the user to
confirm lawful local availability and supply an explicit dataset root. That confirmation and root
must not be requested during this planning step. A separate explicit approval is also required for
final split ratios, histogram bins, component connectivity, lesion ordering, affine tolerance, and
any lesion-aware stratification before real-data execution.

No Phase 2 action is authorized to access real 3D-IRCADb-01 data.
