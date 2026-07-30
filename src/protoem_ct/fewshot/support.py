"""Deterministic support-candidate extraction and support-manifest generation for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from protoem_ct.artifacts import (
    DATASET_MANIFEST_TYPE,
    DEVELOPMENT_SPLIT_MANIFEST_TYPE,
    PHASE2_DEVELOPMENT_COHORT_ROLE,
    DatasetManifest,
    DevelopmentSplitManifest,
    LesionComponentsArtifact,
)
from protoem_ct.artifacts.hashing import sha256_json
from protoem_ct.fewshot.artifacts import (
    SUPPORTED_FEWSHOT_K_VALUES,
    FewshotSupportAssignment,
    FewshotSupportManifest,
)
from protoem_ct.fewshot.stratification import (
    EligibleCandidateBurden,
    FewshotStratificationSelectionResult,
    select_with_optional_stratification,
)

FEWSHOT_SUPPORT_SELECTION_POLICY_VERSION: Final[str] = "fewshot_support_selection_v1"
DEFAULT_REPLICATE_COUNT: Final[int] = 3
DEFAULT_REPLICATE_ID_PREFIX: Final[str] = "replicate"
_NON_TEST_PARTITIONS: Final[frozenset[str]] = frozenset({"train", "validation"})
_INTERNAL_TEST_PARTITION = "internal_test"


class FewshotSupportSelectionError(ValueError):
    """Base error for Phase 4 support-candidate extraction and selection."""


class FewshotSupportInputError(FewshotSupportSelectionError):
    """Raised when Phase 2 artifacts cannot be consumed safely for support selection."""


class FewshotSupportMismatchError(FewshotSupportSelectionError):
    """Raised when manifest, split, and lesion-summary identities do not match."""


class FewshotSupportLeakageError(FewshotSupportSelectionError):
    """Raised when support selection would leak internal-test patients or cases."""


class FewshotSupportInsufficientCandidatesError(FewshotSupportSelectionError):
    """Raised when too few eligible non-test candidates exist for the requested support size."""


@dataclass(frozen=True, slots=True)
class FewshotSupportReplicatePlan:
    """One explicit replicate identifier and selection seed."""

    replicate_id: str
    selection_seed: int

    def __post_init__(self) -> None:
        if not self.replicate_id.startswith(f"{DEFAULT_REPLICATE_ID_PREFIX}_"):
            raise FewshotSupportInputError(
                "replicate_id must use the fixed phase4 replicate_<nn> naming scheme."
            )
        if self.selection_seed < 0:
            raise FewshotSupportInputError("selection_seed must be a nonnegative integer.")


@dataclass(frozen=True, slots=True)
class SupportCandidate:
    """One validated eligible support candidate from a non-test development partition."""

    anonymous_patient_id: str
    anonymous_case_id: str
    partition: str

    def __post_init__(self) -> None:
        if self.partition not in _NON_TEST_PARTITIONS:
            raise FewshotSupportInputError(
                f"SupportCandidate partition must be one of {_NON_TEST_PARTITIONS!r}."
            )


@dataclass(frozen=True, slots=True)
class ValidatedSupportSelectionInputs:
    """Validated and normalized Phase 2 inputs for deterministic support selection."""

    manifest: DatasetManifest
    split: DevelopmentSplitManifest
    eligible_candidates: tuple[SupportCandidate, ...]
    internal_test_patient_ids: frozenset[str]
    internal_test_case_ids: frozenset[str]
    immutable_test_cohort_hash: str
    source_lesion_summary_hash: str | None = None
    eligible_candidate_burdens: tuple[EligibleCandidateBurden, ...] | None = None

    def __post_init__(self) -> None:
        if self.manifest.manifest_type != DATASET_MANIFEST_TYPE:
            raise FewshotSupportInputError("manifest must be a Phase 2 dataset manifest.")
        if self.manifest.cohort_role != PHASE2_DEVELOPMENT_COHORT_ROLE:
            raise FewshotSupportInputError("manifest must use the development cohort role.")
        if self.split.manifest_type != DEVELOPMENT_SPLIT_MANIFEST_TYPE:
            raise FewshotSupportInputError("split must be a Phase 2 development split manifest.")
        if self.split.source_manifest_hash != self.manifest.manifest_hash:
            raise FewshotSupportMismatchError(
                "split.source_manifest_hash must match manifest.manifest_hash."
            )
        candidate_patients: set[str] = set()
        candidate_cases: set[str] = set()
        for candidate in self.eligible_candidates:
            if candidate.anonymous_patient_id in self.internal_test_patient_ids:
                raise FewshotSupportLeakageError(
                    "Eligible candidates must exclude internal-test patients."
                )
            if candidate.anonymous_case_id in self.internal_test_case_ids:
                raise FewshotSupportLeakageError(
                    "Eligible candidates must exclude internal-test cases."
                )
            if candidate.anonymous_patient_id in candidate_patients:
                raise FewshotSupportInputError(
                    "Eligible support candidates must use unique patient IDs."
                )
            if candidate.anonymous_case_id in candidate_cases:
                raise FewshotSupportInputError(
                    "Eligible support candidates must use unique case IDs."
                )
            candidate_patients.add(candidate.anonymous_patient_id)
            candidate_cases.add(candidate.anonymous_case_id)
        if len(self.eligible_candidates) == 0:
            raise FewshotSupportInsufficientCandidatesError(
                "No eligible non-test candidates are available for support selection."
            )
        if self.eligible_candidate_burdens is not None:
            burden_pairs = {
                (item.anonymous_patient_id, item.anonymous_case_id)
                for item in self.eligible_candidate_burdens
            }
            candidate_pairs = {
                (item.anonymous_patient_id, item.anonymous_case_id)
                for item in self.eligible_candidates
            }
            if burden_pairs != candidate_pairs:
                raise FewshotSupportMismatchError(
                    "Eligible lesion-burden records must match eligible support candidates exactly."
                )


def build_default_support_replicate_plans(
    requested_k: int,
    *,
    replicate_count: int = DEFAULT_REPLICATE_COUNT,
) -> tuple[FewshotSupportReplicatePlan, ...]:
    """Return the fixed replicate plans for one support size."""

    if requested_k not in SUPPORTED_FEWSHOT_K_VALUES:
        raise FewshotSupportInputError(
            f"requested_k must be one of {sorted(SUPPORTED_FEWSHOT_K_VALUES)!r}."
        )
    if replicate_count < DEFAULT_REPLICATE_COUNT:
        raise FewshotSupportInputError("replicate_count must be at least three.")
    return tuple(
        FewshotSupportReplicatePlan(
            replicate_id=f"{DEFAULT_REPLICATE_ID_PREFIX}_{index:02d}",
            selection_seed=(requested_k * 1000) + (index * 101),
        )
        for index in range(1, replicate_count + 1)
    )


def hash_immutable_test_cohort(
    split: DevelopmentSplitManifest,
) -> str:
    """Return a deterministic hash of the immutable internal-test cohort assignments."""

    assignments = tuple(
        sorted(
            (
                {
                    "anonymous_case_id": assignment.anonymous_case_id,
                    "anonymous_patient_id": assignment.anonymous_patient_id,
                }
                for assignment in split.assignments
                if assignment.partition == _INTERNAL_TEST_PARTITION
            ),
            key=lambda item: (item["anonymous_patient_id"], item["anonymous_case_id"]),
        )
    )
    return sha256_json(
        {
            "payload_type": "phase4-immutable-test-cohort-v1",
            "source_development_split_hash": split.split_hash,
            "assignments": list(assignments),
        }
    )


def validate_support_selection_inputs(
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
    lesion_artifact: LesionComponentsArtifact | None = None,
) -> ValidatedSupportSelectionInputs:
    """Validate and normalize persisted Phase 2 artifacts for support selection."""

    manifest_pairs = {
        (case.anonymous_patient_id, case.anonymous_case_id): case for case in manifest.cases
    }
    split_pairs = {
        (assignment.anonymous_patient_id, assignment.anonymous_case_id): assignment
        for assignment in split.assignments
    }
    if set(manifest_pairs) != set(split_pairs):
        raise FewshotSupportMismatchError(
            "manifest and split anonymous patient/case assignments must match exactly."
        )
    if any(case.cohort_role != PHASE2_DEVELOPMENT_COHORT_ROLE for case in manifest.cases):
        raise FewshotSupportInputError(
            "External-cohort records are not allowed in support selection."
        )
    internal_test_patient_ids = frozenset(
        assignment.anonymous_patient_id
        for assignment in split.assignments
        if assignment.partition == _INTERNAL_TEST_PARTITION
    )
    internal_test_case_ids = frozenset(
        assignment.anonymous_case_id
        for assignment in split.assignments
        if assignment.partition == _INTERNAL_TEST_PARTITION
    )
    eligible_candidates = tuple(
        sorted(
            (
                SupportCandidate(
                    anonymous_patient_id=assignment.anonymous_patient_id,
                    anonymous_case_id=assignment.anonymous_case_id,
                    partition=assignment.partition,
                )
                for assignment in split.assignments
                if assignment.partition in _NON_TEST_PARTITIONS
            ),
            key=lambda item: (item.anonymous_patient_id, item.anonymous_case_id),
        )
    )
    eligible_burdens: tuple[EligibleCandidateBurden, ...] | None = None
    source_lesion_summary_hash: str | None = None
    if lesion_artifact is not None:
        if lesion_artifact.manifest_hash != manifest.manifest_hash:
            raise FewshotSupportMismatchError(
                "Lesion summary manifest hash must match manifest.manifest_hash."
            )
        if lesion_artifact.split_hash != split.split_hash:
            raise FewshotSupportMismatchError(
                "Lesion summary split hash must match split.split_hash."
            )
        lesion_pairs = {
            (record.anonymous_patient_id, record.anonymous_case_id): record
            for record in lesion_artifact.case_records
        }
        if set(lesion_pairs) != set(manifest_pairs):
            raise FewshotSupportMismatchError(
                "Lesion summary anonymous patient/case assignments must match "
                "manifest and split exactly."
            )
        for pair, record in lesion_pairs.items():
            expected_partition = split_pairs[pair].partition
            if record.partition != expected_partition:
                raise FewshotSupportMismatchError(
                    "Lesion summary partitions must match the persisted split assignments."
                )
        eligible_burdens_list: list[EligibleCandidateBurden] = []
        for candidate in eligible_candidates:
            record = lesion_pairs[(candidate.anonymous_patient_id, candidate.anonymous_case_id)]
            if not record.analysis_performed:
                eligible_burdens = None
                source_lesion_summary_hash = lesion_artifact.lesion_artifact_hash
                break
            assert record.tumor_physical_volume_mm3 is not None
            assert record.lesion_count is not None
            assert record.tumor_voxel_count is not None
            eligible_burdens_list.append(
                EligibleCandidateBurden(
                    anonymous_patient_id=candidate.anonymous_patient_id,
                    anonymous_case_id=candidate.anonymous_case_id,
                    tumor_physical_volume_mm3=record.tumor_physical_volume_mm3,
                    lesion_count=record.lesion_count,
                    tumor_voxel_count=record.tumor_voxel_count,
                )
            )
        else:
            eligible_burdens = tuple(
                sorted(
                    eligible_burdens_list,
                    key=lambda item: (item.anonymous_patient_id, item.anonymous_case_id),
                )
            )
            source_lesion_summary_hash = lesion_artifact.lesion_artifact_hash
    return ValidatedSupportSelectionInputs(
        manifest=manifest,
        split=split,
        eligible_candidates=eligible_candidates,
        internal_test_patient_ids=internal_test_patient_ids,
        internal_test_case_ids=internal_test_case_ids,
        immutable_test_cohort_hash=hash_immutable_test_cohort(split),
        source_lesion_summary_hash=source_lesion_summary_hash,
        eligible_candidate_burdens=eligible_burdens,
    )


def generate_support_manifests_for_k(
    inputs: ValidatedSupportSelectionInputs,
    *,
    requested_k: int,
    replicate_plans: tuple[FewshotSupportReplicatePlan, ...] | None = None,
) -> tuple[FewshotSupportManifest, ...]:
    """Generate the fixed support manifests for one support size."""

    if requested_k not in SUPPORTED_FEWSHOT_K_VALUES:
        raise FewshotSupportInputError(
            f"requested_k must be one of {sorted(SUPPORTED_FEWSHOT_K_VALUES)!r}."
        )
    if requested_k > len(inputs.eligible_candidates):
        raise FewshotSupportInsufficientCandidatesError(
            "requested_k exceeds the eligible non-test candidate count."
        )
    plans = replicate_plans or build_default_support_replicate_plans(requested_k)
    manifests: list[FewshotSupportManifest] = []
    eligible_pairs = tuple(
        (candidate.anonymous_patient_id, candidate.anonymous_case_id)
        for candidate in inputs.eligible_candidates
    )
    for plan in plans:
        selection_result = select_with_optional_stratification(
            eligible_pairs=eligible_pairs,
            burdens=inputs.eligible_candidate_burdens,
            requested_k=requested_k,
            selection_seed=plan.selection_seed,
            selection_policy_version=FEWSHOT_SUPPORT_SELECTION_POLICY_VERSION,
            source_development_manifest_hash=inputs.manifest.manifest_hash,
            source_development_split_hash=inputs.split.split_hash,
            source_lesion_summary_hash=inputs.source_lesion_summary_hash,
        )
        manifests.append(
            _build_support_manifest(
                inputs=inputs,
                requested_k=requested_k,
                plan=plan,
                selection_result=selection_result,
            )
        )
    return tuple(manifests)


def generate_complete_fixed_support_manifest_set(
    manifest: DatasetManifest,
    split: DevelopmentSplitManifest,
    lesion_artifact: LesionComponentsArtifact | None = None,
    *,
    requested_k_values: tuple[int, ...] = (1, 2, 5, 10, 20),
) -> dict[int, tuple[FewshotSupportManifest, ...]]:
    """Generate the complete fixed in-memory support-manifest set for all required support sizes."""

    inputs = validate_support_selection_inputs(manifest, split, lesion_artifact)
    return {
        requested_k: generate_support_manifests_for_k(inputs, requested_k=requested_k)
        for requested_k in requested_k_values
    }


def _build_support_manifest(
    *,
    inputs: ValidatedSupportSelectionInputs,
    requested_k: int,
    plan: FewshotSupportReplicatePlan,
    selection_result: FewshotStratificationSelectionResult,
) -> FewshotSupportManifest:
    selected_pairs = tuple(
        sorted(
            selection_result.selected_patient_case_pairs,
            key=lambda item: (item[0], item[1]),
        )
    )
    _validate_selected_pairs(inputs, selected_pairs, requested_k=requested_k)
    assignments = tuple(
        FewshotSupportAssignment(
            anonymous_patient_id=patient_id,
            anonymous_case_id=case_id,
        )
        for patient_id, case_id in selected_pairs
    )
    manifest_id = f"k{requested_k:02d}_{plan.replicate_id}"
    payload_without_hash = {
        "schema_version": "fewshot_support_manifest_v1",
        "manifest_id": manifest_id,
        "requested_k": requested_k,
        "replicate_id": plan.replicate_id,
        "selection_seed": plan.selection_seed,
        "selection_policy_version": FEWSHOT_SUPPORT_SELECTION_POLICY_VERSION,
        "stratification_status": selection_result.stratification_status,
        "stratification_reason": selection_result.stratification_reason,
        "source_development_manifest_hash": inputs.manifest.manifest_hash,
        "source_development_split_hash": inputs.split.split_hash,
        "source_lesion_summary_hash": inputs.source_lesion_summary_hash,
        "immutable_test_cohort_hash": inputs.immutable_test_cohort_hash,
        "assignments": [
            {
                "anonymous_patient_id": assignment.anonymous_patient_id,
                "anonymous_case_id": assignment.anonymous_case_id,
            }
            for assignment in assignments
        ],
    }
    artifact_hash = sha256_json(payload_without_hash)
    return FewshotSupportManifest(
        artifact_hash=artifact_hash,
        schema_version="fewshot_support_manifest_v1",
        manifest_id=manifest_id,
        requested_k=requested_k,
        replicate_id=plan.replicate_id,
        selection_seed=plan.selection_seed,
        selection_policy_version=FEWSHOT_SUPPORT_SELECTION_POLICY_VERSION,
        stratification_status=selection_result.stratification_status,
        stratification_reason=selection_result.stratification_reason,
        source_development_manifest_hash=inputs.manifest.manifest_hash,
        source_development_split_hash=inputs.split.split_hash,
        source_lesion_summary_hash=inputs.source_lesion_summary_hash,
        immutable_test_cohort_hash=inputs.immutable_test_cohort_hash,
        assignments=assignments,
    )


def _validate_selected_pairs(
    inputs: ValidatedSupportSelectionInputs,
    selected_pairs: tuple[tuple[str, str], ...],
    *,
    requested_k: int,
) -> None:
    if len(selected_pairs) != requested_k:
        raise FewshotSupportInsufficientCandidatesError(
            "Support selection must produce exactly requested_k patient/case assignments."
        )
    patient_ids = {pair[0] for pair in selected_pairs}
    case_ids = {pair[1] for pair in selected_pairs}
    if len(patient_ids) != requested_k or len(case_ids) != requested_k:
        raise FewshotSupportLeakageError(
            "Selected support assignments must use unique patients and cases."
        )
    if patient_ids & inputs.internal_test_patient_ids:
        raise FewshotSupportLeakageError(
            "Selected support patients must not overlap internal test."
        )
    if case_ids & inputs.internal_test_case_ids:
        raise FewshotSupportLeakageError("Selected support cases must not overlap internal test.")
    eligible_pairs = {
        (candidate.anonymous_patient_id, candidate.anonymous_case_id)
        for candidate in inputs.eligible_candidates
    }
    if set(selected_pairs) - eligible_pairs:
        raise FewshotSupportLeakageError(
            "Selected support assignments must come from eligible non-test candidates only."
        )


__all__ = [
    "DEFAULT_REPLICATE_COUNT",
    "DEFAULT_REPLICATE_ID_PREFIX",
    "FEWSHOT_SUPPORT_SELECTION_POLICY_VERSION",
    "FewshotSupportInputError",
    "FewshotSupportInsufficientCandidatesError",
    "FewshotSupportLeakageError",
    "FewshotSupportMismatchError",
    "FewshotSupportReplicatePlan",
    "FewshotSupportSelectionError",
    "SupportCandidate",
    "ValidatedSupportSelectionInputs",
    "build_default_support_replicate_plans",
    "generate_complete_fixed_support_manifest_set",
    "generate_support_manifests_for_k",
    "hash_immutable_test_cohort",
    "validate_support_selection_inputs",
]
