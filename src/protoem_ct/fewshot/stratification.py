"""Deterministic lesion-burden stratification for Phase 4 support selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from protoem_ct.artifacts.hashing import sha256_json

BUCKET_ORDER: Final[tuple[str, str, str]] = ("high", "medium", "low")
_MIN_STRATIFIED_K = len(BUCKET_ORDER)


@dataclass(frozen=True, slots=True)
class EligibleCandidateBurden:
    """One eligible support candidate paired with deterministic burden measurements."""

    anonymous_patient_id: str
    anonymous_case_id: str
    tumor_physical_volume_mm3: float
    lesion_count: int
    tumor_voxel_count: int


@dataclass(frozen=True, slots=True)
class StratifiedCandidate:
    """One candidate assigned to a deterministic lesion-burden bucket."""

    anonymous_patient_id: str
    anonymous_case_id: str
    bucket: str
    tumor_physical_volume_mm3: float
    lesion_count: int
    tumor_voxel_count: int


@dataclass(frozen=True, slots=True)
class FewshotStratificationSelectionResult:
    """Selection outcome with explicit stratification status and reason."""

    selected_patient_case_pairs: tuple[tuple[str, str], ...]
    stratification_status: str
    stratification_reason: str | None


def assign_burden_buckets(
    burdens: tuple[EligibleCandidateBurden, ...],
) -> tuple[StratifiedCandidate, ...]:
    """Assign deterministic high/medium/low buckets from descending burden rank."""

    ranked = sorted(
        burdens,
        key=lambda item: (
            -item.tumor_physical_volume_mm3,
            -item.lesion_count,
            -item.tumor_voxel_count,
            item.anonymous_patient_id,
            item.anonymous_case_id,
        ),
    )
    return tuple(
        StratifiedCandidate(
            anonymous_patient_id=item.anonymous_patient_id,
            anonymous_case_id=item.anonymous_case_id,
            bucket=BUCKET_ORDER[index % len(BUCKET_ORDER)],
            tumor_physical_volume_mm3=item.tumor_physical_volume_mm3,
            lesion_count=item.lesion_count,
            tumor_voxel_count=item.tumor_voxel_count,
        )
        for index, item in enumerate(ranked)
    )


def stratified_bucket_targets(requested_k: int) -> dict[str, int] | None:
    """Return deterministic per-bucket target counts or ``None`` when infeasible."""

    if requested_k < _MIN_STRATIFIED_K:
        return None
    base = requested_k // len(BUCKET_ORDER)
    remainder = requested_k % len(BUCKET_ORDER)
    return {
        bucket: base + (1 if index < remainder else 0) for index, bucket in enumerate(BUCKET_ORDER)
    }


def rank_pairs_for_selection(
    patient_case_pairs: tuple[tuple[str, str], ...],
    *,
    requested_k: int,
    selection_seed: int,
    selection_policy_version: str,
    source_development_manifest_hash: str,
    source_development_split_hash: str,
    source_lesion_summary_hash: str | None,
    bucket: str,
) -> tuple[tuple[str, str], ...]:
    """Return deterministic ranked patient/case pairs for one selection context."""

    return tuple(
        sorted(
            patient_case_pairs,
            key=lambda item: (
                sha256_json(
                    {
                        "anonymous_case_id": item[1],
                        "anonymous_patient_id": item[0],
                        "bucket": bucket,
                        "payload_type": "phase4-support-selection-rank-v1",
                        "requested_k": str(requested_k),
                        "selection_policy_version": selection_policy_version,
                        "selection_seed": str(selection_seed),
                        "source_development_manifest_hash": source_development_manifest_hash,
                        "source_development_split_hash": source_development_split_hash,
                        "source_lesion_summary_hash": source_lesion_summary_hash,
                    }
                ),
                item[0],
                item[1],
            ),
        )
    )


def select_with_optional_stratification(
    *,
    eligible_pairs: tuple[tuple[str, str], ...],
    burdens: tuple[EligibleCandidateBurden, ...] | None,
    requested_k: int,
    selection_seed: int,
    selection_policy_version: str,
    source_development_manifest_hash: str,
    source_development_split_hash: str,
    source_lesion_summary_hash: str | None,
) -> FewshotStratificationSelectionResult:
    """Select deterministic support pairs with stratification or deterministic fallback."""

    if burdens is None:
        return FewshotStratificationSelectionResult(
            selected_patient_case_pairs=rank_pairs_for_selection(
                eligible_pairs,
                requested_k=requested_k,
                selection_seed=selection_seed,
                selection_policy_version=selection_policy_version,
                source_development_manifest_hash=source_development_manifest_hash,
                source_development_split_hash=source_development_split_hash,
                source_lesion_summary_hash=source_lesion_summary_hash,
                bucket="unstratified",
            )[:requested_k],
            stratification_status="fallback_unstratified",
            stratification_reason="lesion_summary_not_provided",
        )
    bucket_targets = stratified_bucket_targets(requested_k)
    if bucket_targets is None:
        return FewshotStratificationSelectionResult(
            selected_patient_case_pairs=rank_pairs_for_selection(
                eligible_pairs,
                requested_k=requested_k,
                selection_seed=selection_seed,
                selection_policy_version=selection_policy_version,
                source_development_manifest_hash=source_development_manifest_hash,
                source_development_split_hash=source_development_split_hash,
                source_lesion_summary_hash=source_lesion_summary_hash,
                bucket="unstratified",
            )[:requested_k],
            stratification_status="fallback_unstratified",
            stratification_reason="requested_k_below_three",
        )

    bucketed = assign_burden_buckets(burdens)
    pairs_by_bucket = {
        bucket: tuple(
            (item.anonymous_patient_id, item.anonymous_case_id)
            for item in bucketed
            if item.bucket == bucket
        )
        for bucket in BUCKET_ORDER
    }
    if any(len(pairs_by_bucket[bucket]) < bucket_targets[bucket] for bucket in BUCKET_ORDER):
        return FewshotStratificationSelectionResult(
            selected_patient_case_pairs=rank_pairs_for_selection(
                eligible_pairs,
                requested_k=requested_k,
                selection_seed=selection_seed,
                selection_policy_version=selection_policy_version,
                source_development_manifest_hash=source_development_manifest_hash,
                source_development_split_hash=source_development_split_hash,
                source_lesion_summary_hash=source_lesion_summary_hash,
                bucket="unstratified",
            )[:requested_k],
            stratification_status="fallback_unstratified",
            stratification_reason="insufficient_bucket_membership",
        )

    selected_pairs: list[tuple[str, str]] = []
    for bucket in BUCKET_ORDER:
        ranked = rank_pairs_for_selection(
            pairs_by_bucket[bucket],
            requested_k=requested_k,
            selection_seed=selection_seed,
            selection_policy_version=selection_policy_version,
            source_development_manifest_hash=source_development_manifest_hash,
            source_development_split_hash=source_development_split_hash,
            source_lesion_summary_hash=source_lesion_summary_hash,
            bucket=bucket,
        )
        selected_pairs.extend(ranked[: bucket_targets[bucket]])
    return FewshotStratificationSelectionResult(
        selected_patient_case_pairs=tuple(selected_pairs),
        stratification_status="stratified",
        stratification_reason=None,
    )


__all__ = [
    "BUCKET_ORDER",
    "EligibleCandidateBurden",
    "FewshotStratificationSelectionResult",
    "StratifiedCandidate",
    "assign_burden_buckets",
    "rank_pairs_for_selection",
    "select_with_optional_stratification",
    "stratified_bucket_targets",
]
