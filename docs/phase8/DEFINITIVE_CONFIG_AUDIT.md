# Phase 8 Definitive-Training Configuration Audit

Read-only configuration audit of `build_monai_segresnet_run_config()` and
related Phase 8 real-training-readiness gaps, performed to determine what
must be added, clarified, or explicitly approved before a real Substage 4B
training plan can be published.

## Starting Git State

- Repository: `/Users/mohsenshamsijazeb/Projects/protoem-ct`
- Branch: `phase/8-external-validation`
- HEAD: `809edeb359e359c7c4a2bb8284b99d17ebf469ac`
- Worktree: clean (confirmed before and after the audit — unchanged)

## Subagents Used

- `P8R-DEFINITIVE-CONFIG-AUDIT` — COMPLETE. Produced the configuration gap
  analysis below from committed repository content only.
- `P8R-DEFINITIVE-CONFIG-AUDIT-REVIEW` — COMPLETE. Independently re-opened
  every cited file and line and verified the claims below. Verdict: **PASS**
  (see "Independent Review Findings").

## Final Verdict: CONFIG_AUDIT_READY

All gaps and required approvals identified below were independently
verified against the actual repository state at HEAD `809edeb`.

---

## A. Current Config Inventory — `build_monai_segresnet_run_config()`

Source: `src/protoem_ct/baselines/monai_segresnet.py:217-271`. Called only
from `baselines/gate3.py:406` (synthetic Gate 3 orchestration) and its own
unit tests (`tests/unit/test_monai_segresnet.py:39,48,61`) — **no Phase 8
module** (`definitive_training.py`, `real_development_runner.py`,
`tiny_real_verification.py`, `internal_evidence.py`) imports or calls this
function or `MonaiSegResNetRunConfig` at all (confirmed by repo-wide grep).
The Substage 4A `training_config_reference` field in
`Phase8DefinitiveTrainingConfig` (`definitive_training.py:192`) is a fully
opaque `ArtifactReference` (schema_name/schema_version/artifact_hash/
artifact_role only); in the Substage 4A test suite it is populated with a
synthetic placeholder hash (`tests/unit/test_phase8_definitive_training.py:155`,
`HASH_A = "a" * 64`), never with an actual `MonaiSegResNetRunConfig` payload.
There is currently no code path — CLI, planner, or executor — that
constructs a `training_config_reference` from
`build_monai_segresnet_run_config()`'s output.

| Field | Value | Source | Status | Scientific decision if changed? |
|---|---|---|---|---|
| `device` | `"cpu"` | :255 (enforced :133-134) | engineering gate; "sufficient" for real training is unvalidated | No, but feasibility is undetermined |
| `amp_enabled` | `False` | :256 (enforced :135-136) | generally reusable | No |
| `seed` | `1729` | :219,257 | generally reusable | No, if kept consistent |
| `in_channels` | `1` | :258 (enforced :139-140) | generally reusable | No |
| `num_classes` | `2` | :259 (enforced :141-142) | generally reusable | No |
| `optimizer_name` | `"adam"` | :260 | synthetic Gate-3 choice, never validated on real data | Yes |
| `learning_rate` | `5e-3` | :221,261 | synthetic-only, tuned for a 16-step tiny overfit | Yes |
| `loss_name` | `"cross_entropy"`, unweighted | :262 (`CrossEntropyLoss()` at :401) | synthetic-only — no class weighting despite known tumor/background imbalance | Yes |
| `max_training_steps` | `16` | :220,263 (DECISIONS.md:482 "16 bounded training steps") | synthetic-only — sized for a tiny-fixture overfit demonstration | Yes |
| `fixed_intensity_min`/`max` | `-1000.0`/`1000.0` | :236,264-265 (applied :730-737) | matches the fixed HU window also used in `tiny_real_verification.py:126-127` — one real-pixel-adjacent value, but never frozen as a decision | No, if reused as-is |
| `sliding_window_roi_size` | `(24, 24, 16)` | :222,266 | synthetic-only — equals the entire synthetic fixture volume (`baselines/synthetic.py:27`), not a real patch size | Yes |
| `sliding_window_overlap` | `0.0` | :223,267 | synthetic-only | Yes |
| `sliding_window_batch_size` | `1` | :224,268 | generally reusable | No (bounded by device memory) |
| `thread_count` | `1` | :225,269 (enforced `==1` at :677) | generally reusable | No |
| `overfit_case_identifier` | `"synthetic_train_002"` | :226,244 | fixture-oriented identifier from `BASELINE_SYNTHETIC_TRAINING_CASE_IDENTIFIERS` (`baselines/synthetic.py:36`); models "one overfit case," not a real train/validation partition | Yes — the whole field/concept must be replaced |
| `checkpoint_name` | `"checkpoint_final.pt"` | :46,233 | generally reusable | No |

No field exists at all in `MonaiSegResNetRunConfig` for: epoch count,
validation cadence, checkpoint-selection metric, early stopping, resume
policy, augmentation, worker count, orientation policy, label
interpolation, patch-sampling policy, or positive/negative sampling ratio
(schema at `_RUN_CONFIG_FIELDS`, monai_segresnet.py:50-71) — these are
absent, not merely misconfigured.

---

## B. Policy Classification Table

Classifications: `COMMITTED_AND_UNAMBIGUOUS`, `COMMITTED_BUT_SYNTHETIC_ONLY`,
`PARTIALLY_DEFINED`, `NOT_DEFINED`, `CONFLICTING_DEFINITIONS`.

| Policy | Classification | Basis |
|---|---|---|
| Orientation handling | NOT_DEFINED | No orientation transform anywhere; Phase 2 QA performs "no resampling, reorientation, normalization, clipping" (`docs/PHASE2_DATA_CONTRACT.md:206`) |
| Voxel-spacing handling | NOT_DEFINED | No resampling step in either training/verification loader |
| Intensity clipping/window | COMMITTED_BUT_SYNTHETIC_ONLY / PARTIALLY_DEFINED | `[-1000, 1000]` HU in `monai_segresnet.py:236` and `tiny_real_verification.py:126-127,261-265`; never frozen as a `docs/DECISIONS.md` entry or `Phase8PreprocessingDecision` artifact |
| Normalization | COMMITTED_BUT_SYNTHETIC_ONLY | Same two sites apply linear `[-1,1]` rescale; never formally frozen |
| Label semantics | **COMMITTED_AND_UNAMBIGUOUS** | `tiny_real_verification.py:131-138,1209`: raw domain `{0,1,2}`, `foreground=(raw==2)`; verified against real LiTS/MSD files, cross-checked `docs/phase8/SUPERVISOR_HANDOFF.md:270-296` |
| Image interpolation | NOT_DEFINED | `Phase8PreprocessingDecision.image_interpolation_policy` (`internal_evidence.py:120,149-152`) allows `allow_unresolved=True`, never populated with a resolved value |
| Label interpolation | NOT_DEFINED | Same schema field, same status |
| Foreground-aware patch sampling | NOT_DEFINED | No `RandCropByPosNegLabeld`-equivalent; both modules use a fixed-corner deterministic crop (`tiny_real_verification.py:1206`) |
| Positive/negative sampling ratio | NOT_DEFINED | No such parameter exists in any schema |
| Patch dimensions | COMMITTED_BUT_SYNTHETIC_ONLY (train) / PARTIALLY_DEFINED (verification) | `(24,24,16)` equals the synthetic fixture volume; `MAX_PATCH_DIMS=(32,64,64)` is a bound only |
| Train patches per case/epoch | NOT_DEFINED | Only exists hard-locked to `1` in tiny-verification config; no per-epoch sampling budget concept exists |
| Validation strategy | NOT_DEFINED | No validation-loop code exists in either module |
| Full-volume vs. patch validation | NOT_DEFINED | `Phase8ValidationEvidenceReference` (`internal_evidence.py:363-408`) is metadata-only, no inference-method field |
| Batch size | reusable default, not a scientific commitment | `1` everywhere it appears |
| Gradient accumulation | NOT_DEFINED | No field, no code |
| Optimizer | COMMITTED_BUT_SYNTHETIC_ONLY | `torch.optim.Adam` — Gate-3/verification-only choice |
| Learning rate | COMMITTED_BUT_SYNTHETIC_ONLY | `5e-3` default, sized for 16-step tiny overfit |
| Weight decay | NOT_DEFINED | No field in any Phase 8 schema |
| Loss | COMMITTED_BUT_SYNTHETIC_ONLY | Unweighted `CrossEntropyLoss`; DECISIONS.md:413-437 fixes the *metric* contract, not the training loss |
| Class weighting | NOT_DEFINED | No weighting parameter anywhere despite known foreground/background imbalance |
| Epoch count | NOT_DEFINED | Hard-locked to `1` only in tiny-verification config; no epoch concept for a real multi-case run |
| Maximum optimizer steps | **CONFLICTING_DEFINITIONS** | Gate 3 synthetic: `16`; tiny verification: `1-2`; Substage 4A `authorized_max_training_steps` is a free CLI integer with no fixed value (`definitive_training.py:507-519`) — three regimes, none a real training budget |
| Validation cadence | NOT_DEFINED | No field, no code |
| Checkpoint cadence | NOT_DEFINED | Both modules save exactly one final checkpoint; no periodic/best-checkpoint cadence |
| Early stopping | NOT_DEFINED | No field, no code |
| Deterministic seeds | mechanism COMMITTED_AND_UNAMBIGUOUS / value PARTIALLY_DEFINED | `torch.manual_seed` + deterministic algorithms enforced in both modules; `Phase8DefinitiveTrainingConfig.fixed_seeds` requires nonempty tuple but no concrete real seed list is frozen |
| Augmentation | NOT_DEFINED | No augmentation transform anywhere in any relevant module |
| Device policy | **COMMITTED_AND_UNAMBIGUOUS** as hard gate | `REQUIRED_DEVICE_TYPE="cpu"` enforced in three independent `__post_init__` validators; practicality flagged unresolved (`docs/phase8/INTERNAL_EVIDENCE_REMEDIATION_PLAN.md:145`: "no available CUDA or MPS device," no runtime estimate) |
| AMP policy | **COMMITTED_AND_UNAMBIGUOUS** | Disabled/forbidden everywhere |
| Worker count | COMMITTED (verification-only) | `num_workers=0` hard-locked in tiny verification; no field at all in the Gate-3 config |
| Resume policy | NOT_DEFINED | No resume/checkpoint-loading code path exists anywhere; `Phase8DefinitiveTrainingExecutor.load_training_data`/`construct_model` are unimplemented `Protocol` stubs (`definitive_training.py:706-744`) |
| Checkpoint-selection metric | **COMMITTED_AND_UNAMBIGUOUS** (declared) | `PRIMARY_METRIC_TUMOR_DICE` (`definitive_training.py:118,1082,1093`), cross-confirmed `INTERNAL_EVIDENCE_REMEDIATION_PLAN.md:108` |
| Fixed inference threshold | **COMMITTED_AND_UNAMBIGUOUS** (declared) | `REQUIRED_THRESHOLD_VALUE=0.5` (`definitive_training.py:111,1131`); `INTERNAL_EVIDENCE_REMEDIATION_PLAN.md:87` documents the earlier synthetic `0.55` as "not freezeable," recommending `0.5` |
| Empty-target handling | **COMMITTED_AND_UNAMBIGUOUS** | `docs/DECISIONS.md:415-436` (both-empty → Dice 1.0, etc.) |
| HD95 undefined-case handling | **COMMITTED_AND_UNAMBIGUOUS** | Same `docs/DECISIONS.md:415-436` entry |

---

## C. Source Traceability (for COMMITTED_AND_UNAMBIGUOUS items)

- **Label semantics**: `src/protoem_ct/external/tiny_real_verification.py:131-138,1209`. Verified against real LiTS/MSD label files (`liver_41.nii.gz`, `liver_51.nii.gz`), independently reviewed per `docs/phase8/SUPERVISOR_HANDOFF.md:401-409`.
- **Device/AMP policy**: enforced in three independent `__post_init__` validators (`monai_segresnet.py:133-136`, `definitive_training.py:233-237`, `tiny_real_verification.py:220-225`) — production code, not test-only. CPU-only is committed as an engineering gate, not validated as practically sufficient (`INTERNAL_EVIDENCE_REMEDIATION_PLAN.md:145`).
- **Checkpoint-selection metric = tumor Dice**: `definitive_training.py:118,1093`, cross-confirmed `INTERNAL_EVIDENCE_REMEDIATION_PLAN.md:108`.
- **Fixed threshold = 0.5**: `definitive_training.py:111`, validated `__post_init__` :225-228; cross-confirmed as a considered rejection of the earlier `0.55` value (`INTERNAL_EVIDENCE_REMEDIATION_PLAN.md:87`).
- **Empty-target / HD95-undefined behavior**: `docs/DECISIONS.md:415-436`, a signed-off decision entry with accompanying implementation, not test-only.

**Flag**: the `[-1000, 1000]` HU window and linear-rescale normalization appear identically in two production files, which is stronger than a test-only claim, but no `docs/DECISIONS.md` entry or `Phase8PreprocessingDecision` object with `evidence_status="freeze_ready"` has ever frozen it as the definitive-training policy. Classified `PARTIALLY_DEFINED`/`COMMITTED_BUT_SYNTHETIC_ONLY`, not `COMMITTED_AND_UNAMBIGUOUS`.

---

## D. Proposed Dedicated Config — `monai_segresnet_real_development_v1`

No parameter values are selected below; fields are only categorized by
provenance, per audit scope.

- **Inherited from committed scientific policy**: label semantics
  (`foreground=(raw==2)`); primary checkpoint-selection metric (tumor
  Dice); fixed inference threshold (0.5, `fixed_constant` policy);
  empty-target/HD95-undefined metric behavior; support policy (`no_support`
  for the single preregistered candidate); device=cpu, AMP=disabled (as
  hard gates, with practicality still unresolved — see E/F/H); shared
  metric-report schema.
- **Engineering-only safety bounds**: output-root path safety and
  no-overwrite publication; manifest/split hash verification before any
  file open; internal-test exclusion checks; no-external-data flags;
  checkpoint SHA-256/byte-size identity verification.
- **New scientific choices requiring explicit user approval** (values must
  NOT be silently picked): intensity clip/window and normalization as a
  frozen Phase 8 preprocessing decision; orientation-handling policy;
  voxel-spacing/resampling policy; image and label interpolation methods;
  foreground-aware patch-sampling method and positive/negative ratio;
  patch dimensions for real training; train patches per case/epoch; epoch
  count / total optimizer-step budget; optimizer identity, learning rate,
  and schedule; weight decay; loss function and class weighting;
  validation strategy and cadence; checkpoint cadence and early-stopping
  rule; augmentation policy; concrete fixed seed value(s) for the
  definitive run.
- **Hardware-dependent values requiring measurement or pilot evidence**:
  whether CPU-only execution is practically completable for full-cohort 3D
  training within acceptable wall-clock budget; num_workers/batch size
  bounded by available CPU memory; sliding-window inference ROI/overlap/
  batch size for validation at real CT volume scale.

---

## E. Foreground-Sampling Conclusion

The only patch-sampling code that has touched real pixels is
`tiny_real_verification.py:_load_bounded_patch` (:1176-1218), a
**deterministic fixed-corner crop** (`image_array[0:patch[0], 0:patch[1],
0:patch[2]]`, :1206), explicitly documented as "Never randomly samples the
crop location" (:1186-1187). It is verification-only by design; on the one
real case tested, this crop landed on an all-background region, producing
a degenerate all-background training target
(`docs/phase8/SUPERVISOR_HANDOFF.md:376-383`).

**Conclusion: this fixed-corner crop cannot be used for definitive
training.** It is not a sampling policy — it is a single, non-representative,
position-0 crop with no guarantee of foreground coverage. No foreground-aware
sampling is committed anywhere in the codebase (no positive/negative ratio
field, no lesion-bounding-box field, no random-crop-with-foreground-bias
field in any config schema).

Minimal scientifically defensible options requiring approval (none
selected here):
1. MONAI-style `RandCropByPosNegLabeld` sampler with an explicit pos:neg
   ratio and samples-per-volume count.
2. Deterministic lesion-centered cropping using existing Phase 2 lesion
   connected-component summaries to guarantee foreground inclusion for a
   fixed fraction of patches.
3. Full-volume (or coarse-resampled full-volume) training without patch
   cropping, if computationally tractable.

Approval needed on: (a) whether sampling is foreground-biased vs. uniform
random vs. full-volume; (b) if biased, exact pos:neg ratio and patch count
per case/epoch; (c) whether existing lesion metadata may be consulted for
sampling (a sampling-strategy question, not a leakage question, since it is
the training label itself, but must still be recorded as a frozen decision).

---

## F. Validation-Strategy Conclusion

No committed code or document specifies whether definitive-run validation
should be full-volume sliding-window inference, deterministic bounded
patches, or another method. What is committed:

- The Gate-3 synthetic path uses `monai.inferers.sliding_window_inference`
  with `roi_size=(24,24,16)`, `overlap=0.0` — but that ROI equals the
  *entire* synthetic volume, so no real windowing occurs; this is not
  evidence of a real-scale policy.
- `Phase8ValidationEvidenceReference` requires
  `development_validation_only=True`, `internal_test_not_used=True`,
  `external_data_not_used=True`, and a `metric_artifact_hash`, but has no
  field describing the inference method itself.
- `INTERNAL_EVIDENCE_REMEDIATION_PLAN.md:106-119` fixes which metrics to
  report (tumor Dice primary; IoU, HD95, NSD, lesion recall/precision/F1,
  FP-lesions/scan, volume error secondary) but not how predictions are
  generated.

Full-volume sliding-window inference is the *mechanism* most directly
derivable from committed code (already imported/exercised, and
DECISIONS.md-endorsed as the Gate-3 inference mechanism), but this is a
mechanism choice only — ROI size, overlap, and batch size for real
(non-fixture-sized) volumes have never been set or validated, and current
defaults are fixture-sized and unusable as-is.

Implications of this gap: tumor Dice sensitivity to ROI/overlap choice,
HD95 boundary effects at ROI seams, and false-positive lesions per scan
from stitching artifacts are all currently unmeasured, and checkpoint
selection cannot be considered reliable until a real-scale validation
method is piloted.

---

## G. Implementation Gap

- **Code missing entirely**: dedicated multi-case real-LiTS train-time
  patch/crop pipeline with principled sampling; real train/validation loop
  iterating over the full train partition across epochs (current code only
  trains on one `overfit_case_identifier` for a fixed step count, or one
  tiny-verification case for ≤2 steps); real full-volume/sliding-window
  validation loop producing per-case and aggregate metric artifacts;
  class-weighted loss (or documented decision not to use one);
  epoch/step budget and validation-cadence/checkpoint-cadence/
  early-stopping mechanism; resume-from-checkpoint code path
  (`Phase8DefinitiveTrainingExecutor.load_training_data`/`construct_model`
  are unimplemented `Protocol` stubs); real orientation/spacing handling.
- **Code reusable unchanged**: the metric engine (DECISIONS.md contract);
  the fixed intensity clip/normalize formula (pending freeze as a
  `Phase8PreprocessingDecision`); the label-domain validation and
  binarization rule (already real-data-verified); checkpoint SHA-256/
  byte-size identity verification; output-root path-safety and
  no-overwrite publication helpers; the `Phase8DefinitiveTrainingConfig`/
  `Phase8DefinitiveExecutionRelease` fail-closed release-gating mechanism.
- **Code requiring a narrow extension**: `Phase8CheckpointMetadata`/
  `Phase8ValidationEvidenceReference` schemas likely need no structural
  change, only real instances; `build_phase8_real_development_input_binding`
  needs no change, only a caller that actually trains after binding.
- **Code requiring a new scientific decision before it can be written**:
  the patch-sampling module (needs approved sampling policy from E first);
  the validation-inference module (needs approved ROI/overlap policy from
  F first); the training-loop hyperparameter set (needs D's "new
  scientific choices" resolved first); the orientation/spacing
  preprocessing module (no approved resampling policy exists even as a
  proposal today).
- **Tests**: `tests/unit/test_monai_segresnet.py` only exercises
  `build_monai_segresnet_run_config()`'s synthetic defaults and
  serialization round-trip — not evidence the config is validated for real
  multi-case use; `tests/unit/test_phase8_definitive_training.py` never
  constructs a `training_config_reference` from real
  `MonaiSegResNetRunConfig` content (only an opaque synthetic hash), so no
  test currently checks real-training-config compatibility with the
  Substage 4A contracts.

---

## H. Approval Package — Questions for the User

1. **Patch sampling policy.** Options: (a) foreground-biased random
   sampling with an explicit pos:neg ratio; (b) lesion-metadata-guided
   deterministic cropping using existing Phase 2 lesion-component
   artifacts; (c) full-volume (or resampled coarse full-volume) training
   without cropping. None of these is currently supported by committed
   code or evidence — the only real-pixel-tested crop is verification-only
   and demonstrably unsafe for training (produced an all-background target
   on the one case tested). Operational consequence: without
   foreground-aware sampling, training on real multi-case data risks
   converging to a trivial all-background predictor.

2. **Voxel-spacing / resampling policy.** Options: (a) resample all
   volumes to a fixed target spacing before training; (b) train in native
   per-case spacing with spacing-aware patch sizing; (c) another explicit
   rule. Nothing currently supported by committed code — Phase 2 QA
   deliberately performs no resampling, and no training code resamples
   either. Operational consequence: without an explicit policy, patch size
   in physical units is undefined and cross-case comparability of learned
   features is compromised.

   *Forward reference:* this open question was subsequently resolved and
   approved under `PHASE8-DEFINITIVE-TRAIN-SPACING-V1` (see
   `docs/DECISIONS.md`); this audit section is preserved as-written and
   describes the state at the time of this historical audit, not the
   current approved policy.

3. **Orientation-handling policy.** Options: (a) reorient all volumes to a
   canonical orientation before training; (b) assume/verify native
   orientation consistency and skip reorientation. Nothing currently
   supported by committed code (no orientation transform exists anywhere).
   Operational consequence: inconsistent orientation without correction
   would make spatial semantics of learned filters/patches inconsistent
   across cases.

4. **Intensity window / normalization freeze.** An option with real-code
   support (not yet frozen as a decision) exists: reuse the existing
   `[-1000, 1000]` HU clip + linear `[-1,1]` rescale that appears
   identically in both `monai_segresnet.py` and the real-pixel-verified
   `tiny_real_verification.py`. It has genuine cross-file, real-data-adjacent
   support, but has never been formally frozen via a `docs/DECISIONS.md`
   entry or a `Phase8PreprocessingDecision` instance with
   `evidence_status="freeze_ready"`, so it must still be explicitly
   approved rather than assumed. Operational consequence: if approved
   as-is, no new preprocessing code is needed; if changed, both training
   and validation loaders must be modified together.

5. **Training budget (epochs / total steps) and validation cadence.**
   Options: (a) fixed epoch count over the full train partition with
   periodic validation-Dice checkpointing; (b) fixed total optimizer-step
   budget (as the current `authorized_max_training_steps` CLI field
   implies but never fixes a value for); (c) a convergence-based stopping
   rule. Nothing currently supported by committed code — the only step-count
   precedents (16 for Gate-3 synthetic overfit, 1-2 for tiny verification)
   are both explicitly non-representative and must not be treated as
   already-approved values. Operational consequence: this directly
   determines wall-clock feasibility on the confirmed CPU-only environment,
   already flagged as of unknown practicality — a pilot timing measurement
   is needed regardless of which budget is chosen.

6. **Validation inference method and its parameters (ROI size, overlap,
   batch size) for real-scale volumes.** Options: (a) full-volume
   sliding-window inference (mechanism already exists in code and is
   DECISIONS.md-endorsed as the Gate-3 inference mechanism, though not at
   real-volume-scale parameters); (b) deterministic bounded-patch
   validation (mechanism exists only in tiny-verification form, explicitly
   non-scientific per its own disclaimer). Option (a)'s mechanism is
   closest to already-committed evidence, but its parameters for real CT
   volume sizes have never been set or piloted. Operational consequence:
   ROI/overlap choice affects tumor Dice, HD95 (boundary/seam artifacts),
   and false-positive lesions per scan (stitching artifacts) — none
   currently measurable without a pilot run at real scale.

7. **Optimizer/loss/class-weighting/weight-decay for definitive training.**
   Options: (a) reuse Gate-3's unweighted Adam + cross-entropy at
   `lr=5e-3`; (b) introduce class weighting (e.g., inverse frequency) or a
   Dice-based/combo loss appropriate to the known severe tumor/background
   imbalance; (c) another approach. Nothing beyond the tiny/synthetic
   precedent is currently supported by committed evidence, and that
   precedent was never validated against real multi-case imbalance.
   Operational consequence: an unweighted cross-entropy loss on severely
   imbalanced real tumor/background voxel counts is a known failure-mode
   risk, directly relevant given the Substage 3 all-background-target
   incident already observed on real data.

---

## Independent Review Findings

`P8R-DEFINITIVE-CONFIG-AUDIT-REVIEW` independently re-opened every cited
file and line (monai_segresnet.py, tiny_real_verification.py,
definitive_training.py, docs/DECISIONS.md,
docs/phase8/INTERNAL_EVIDENCE_REMEDIATION_PLAN.md,
docs/phase8/SUPERVISOR_HANDOFF.md, both cited unit test files) and
performed repo-wide greps for `build_monai_segresnet_run_config`,
`MonaiSegResNetRunConfig`, and relevant MONAI transform names
(`Orientationd`, `Spacingd`, `RandCropByPosNegLabeld`, `RandFlipd`,
`RandRotate`, `RandZoom`, `RandGaussianNoise`, `RandAffine`).

Result: all field values, line-level enforcement claims, classifications,
and gap-analysis claims were confirmed accurate. Only trivial (1-3 line)
citation drift was found (e.g., a tuple defined at synthetic.py:36 cited as
:38; DECISIONS.md content at lines 415-436 cited as 413-437), with no
wrong-function or wrong-value errors anywhere. Zero hits confirmed for any
augmentation/orientation/spacing/foreground-sampling transform anywhere in
`src/`.

**Independent review verdict: PASS.**

---

## No-Change / No-Data / No-Compute Confirmation

- No repository file was modified during the audit (read-only inspection
  only, via Read/grep operations by both subagents).
- No `/Volumes` path was accessed.
- No dataset, manifest, split, NIfTI, DICOM, or ZIP file was opened.
- No Python code was executed; torch/MONAI were not imported or run.
- No training, inference, checkpoint, or artifact-publication code was
  executed.
- `git status --short --branch` before and after the audit:
  `## phase/8-external-validation` (clean, no changes).
- HEAD unchanged throughout: `809edeb359e359c7c4a2bb8284b99d17ebf469ac`.
