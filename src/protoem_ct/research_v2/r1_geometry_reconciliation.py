"""Reconcile R1 source-geometry observations against the approved Phase 2 contract.

This module operates only on sanitized R1 JSON metadata. It never opens medical arrays, source
NIfTI files, internal-test data, or external data. The approved source image/label affine tolerance
comes from the pre-existing 2026-07-27 Phase 2 real-run decision. Prediction/ground-truth metric
grid validation remains independently strict in :mod:`protoem_ct.research_v2.r1_audit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

R1_SOURCE_AFFINE_ABS_TOLERANCE_MM: Final[float] = 1e-4
R1_SOURCE_SPACING_ABS_TOLERANCE_MM: Final[float] = 1e-4
R1_RECONCILIATION_NUMERICAL_EPS: Final[float] = 1e-12

# Historical Phase 2 envelope recorded before Research V2/R1.
PHASE2_MAX_SELECTED_AFFINE_DELTA_MM: Final[float] = 3.0517578125e-05
PHASE2_MAX_CORNER_DISPLACEMENT_MM: Final[float] = 5.71685607815e-05
PHASE2_MAX_VOXEL_TRANSLATION_DELTA: Final[float] = 4.73484848555e-05
PHASE2_MAX_LINEAR_TRANSFORM_DEVIATION: Final[float] = 1e-7

Matrix = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EffectiveGeometryDiagnostic:
    """Sanitized effective image/label geometry diagnostic for one anonymous case."""

    anonymous_case_id: str
    partition: str
    orientation_match: bool
    shape_match: bool
    effective_affine_match_at_approved_tolerance: bool
    spacing_match_at_approved_tolerance: bool
    max_affine_abs_delta_mm: float
    max_spacing_abs_delta_mm: float
    max_corner_displacement_mm: float
    max_voxel_translation_delta: float
    max_linear_transform_deviation: float
    within_historical_phase2_envelope: bool


def _matrix(record: dict[str, Any], key: str) -> Matrix:
    matrix = np.asarray(record[key], dtype=np.float64)
    if matrix.shape != (4, 4) or not bool(np.isfinite(matrix).all()):
        raise ValueError(f"{key} must be a finite 4x4 matrix")
    return matrix


def _spacing(record: dict[str, Any], key: str) -> npt.NDArray[np.float64]:
    spacing = np.asarray(record[key], dtype=np.float64)
    if spacing.shape != (3,) or not bool(np.isfinite(spacing).all()):
        raise ValueError(f"{key} must be a finite length-3 vector")
    return spacing


def diagnose_effective_geometry(record: dict[str, Any]) -> EffectiveGeometryDiagnostic:
    """Diagnose effective image/label geometry without opening source arrays."""

    image_affine = _matrix(record, "image_affine")
    label_affine = _matrix(record, "label_affine")
    image_spacing = _spacing(record, "image_spacing_mm")
    label_spacing = _spacing(record, "label_spacing_mm")

    shape_raw = record["image_shape"]
    label_shape_raw = record["label_shape"]
    image_shape = tuple(int(value) for value in shape_raw)
    label_shape = tuple(int(value) for value in label_shape_raw)
    if len(image_shape) != 3 or len(label_shape) != 3 or any(value <= 0 for value in image_shape):
        raise ValueError("image_shape and label_shape must describe positive 3D grids")

    max_affine_delta = float(np.max(np.abs(image_affine - label_affine)))
    max_spacing_delta = float(np.max(np.abs(image_spacing - label_spacing)))

    relative = np.linalg.inv(image_affine) @ label_affine
    max_linear = float(np.max(np.abs(relative[:3, :3] - np.eye(3, dtype=np.float64))))
    max_voxel_translation = float(np.max(np.abs(relative[:3, 3])))

    corners = np.asarray(
        [
            [i, j, k, 1.0]
            for i in (0, image_shape[0] - 1)
            for j in (0, image_shape[1] - 1)
            for k in (0, image_shape[2] - 1)
        ],
        dtype=np.float64,
    ).T
    image_world = image_affine @ corners
    label_world = label_affine @ corners
    max_corner = float(np.max(np.linalg.norm(image_world[:3] - label_world[:3], axis=0)))

    orientation_match = list(record["image_orientation"]) == list(record["label_orientation"])
    shape_match = image_shape == label_shape
    affine_match = bool(
        np.allclose(
            image_affine,
            label_affine,
            rtol=0.0,
            atol=R1_SOURCE_AFFINE_ABS_TOLERANCE_MM,
        )
    )
    spacing_match = bool(
        np.allclose(
            image_spacing,
            label_spacing,
            rtol=0.0,
            atol=R1_SOURCE_SPACING_ABS_TOLERANCE_MM,
        )
    )
    eps = R1_RECONCILIATION_NUMERICAL_EPS
    within_historical_envelope = (
        max_affine_delta <= PHASE2_MAX_SELECTED_AFFINE_DELTA_MM + eps
        and max_corner <= PHASE2_MAX_CORNER_DISPLACEMENT_MM + eps
        and max_voxel_translation <= PHASE2_MAX_VOXEL_TRANSLATION_DELTA + eps
        and max_linear <= PHASE2_MAX_LINEAR_TRANSFORM_DEVIATION + eps
    )

    return EffectiveGeometryDiagnostic(
        anonymous_case_id=str(record["anonymous_case_id"]),
        partition=str(record["partition"]),
        orientation_match=orientation_match,
        shape_match=shape_match,
        effective_affine_match_at_approved_tolerance=affine_match,
        spacing_match_at_approved_tolerance=spacing_match,
        max_affine_abs_delta_mm=max_affine_delta,
        max_spacing_abs_delta_mm=max_spacing_delta,
        max_corner_displacement_mm=max_corner,
        max_voxel_translation_delta=max_voxel_translation,
        max_linear_transform_deviation=max_linear,
        within_historical_phase2_envelope=within_historical_envelope,
    )


def reconcile_geometry_mismatches(
    *,
    records: list[dict[str, Any]],
    mismatches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve only the two known R1-v1 geometry over-classifications.

    Unknown reasons are retained as unresolved. qform/sform conflicts remain retained
    observations; they are non-blocking only when effective image/label shape, orientation,
    affine, and spacing are consistent under the pre-existing Phase 2 source-geometry contract.
    An effective-affine mismatch from the erroneous R1 1e-6 source tolerance is resolved only
    when it also lies inside the historical Phase 2 real-run envelope.
    """

    records_by_id = {str(record["anonymous_case_id"]): record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("case records contain duplicate anonymous_case_id values")

    reconciled: list[dict[str, Any]] = []
    diagnostic_cache: dict[str, EffectiveGeometryDiagnostic] = {}
    qform_observation_cases: set[str] = set()
    affine_resolution_cases: set[str] = set()

    for mismatch in mismatches:
        item = dict(mismatch)
        case_id = str(item.get("anonymous_case_id"))
        reason = str(item.get("reason"))
        record = records_by_id.get(case_id)
        if record is None:
            item["resolution_status"] = "unresolved"
            item["resolution"] = "missing sanitized case record"
            reconciled.append(item)
            continue

        diagnostic = diagnostic_cache.setdefault(case_id, diagnose_effective_geometry(record))
        effective_geometry_safe = (
            diagnostic.shape_match
            and diagnostic.orientation_match
            and diagnostic.effective_affine_match_at_approved_tolerance
            and diagnostic.spacing_match_at_approved_tolerance
        )

        if reason in {"label_qform_sform_conflict", "image_qform_sform_conflict"}:
            if effective_geometry_safe:
                item["resolution_status"] = "resolved_nonblocking_observation"
                item["resolution"] = (
                    "retained qform/sform header observation; R1-v1 incorrectly treated this as a "
                    "gate failure although Phase 2 QA and the R1 source-geometry contract govern "
                    "effective image/label geometry"
                )
                qform_observation_cases.add(case_id)
            else:
                item["resolution_status"] = "unresolved"
                item["resolution"] = (
                    "effective image/label geometry is not safe under Phase 2 contract"
                )
        elif reason == "image_label_affine_mismatch":
            if effective_geometry_safe and diagnostic.within_historical_phase2_envelope:
                item["resolution_status"] = "resolved_preexisting_phase2_precision_envelope"
                item["resolution"] = (
                    "R1-v1 used an erroneous 1e-6 source affine tolerance; effective geometry is "
                    "within the pre-existing Phase 2 approved 1e-4 mm tolerance and historical "
                    "real-run floating-point precision envelope"
                )
                affine_resolution_cases.add(case_id)
            else:
                item["resolution_status"] = "unresolved"
                item["resolution"] = "effective affine requires further review"
        else:
            item["resolution_status"] = "unresolved"
            item["resolution"] = item.get("resolution")
        reconciled.append(item)

    unresolved = [
        item
        for item in reconciled
        if str(item["resolution_status"]).startswith("unresolved")
    ]
    diagnostics = [
        {
            "anonymous_case_id": diagnostic.anonymous_case_id,
            "partition": diagnostic.partition,
            "orientation_match": diagnostic.orientation_match,
            "shape_match": diagnostic.shape_match,
            "effective_affine_match_at_approved_tolerance": (
                diagnostic.effective_affine_match_at_approved_tolerance
            ),
            "spacing_match_at_approved_tolerance": diagnostic.spacing_match_at_approved_tolerance,
            "max_affine_abs_delta_mm": diagnostic.max_affine_abs_delta_mm,
            "max_spacing_abs_delta_mm": diagnostic.max_spacing_abs_delta_mm,
            "max_corner_displacement_mm": diagnostic.max_corner_displacement_mm,
            "max_voxel_translation_delta": diagnostic.max_voxel_translation_delta,
            "max_linear_transform_deviation": diagnostic.max_linear_transform_deviation,
            "within_historical_phase2_envelope": diagnostic.within_historical_phase2_envelope,
        }
        for diagnostic in sorted(
            diagnostic_cache.values(), key=lambda value: value.anonymous_case_id
        )
    ]
    summary: dict[str, Any] = {
        "schema_version": "research_v2_r1_geometry_reconciliation.v1",
        "source_arrays_opened_by_reconciliation": 0,
        "source_image_label_affine_tolerance_mm": R1_SOURCE_AFFINE_ABS_TOLERANCE_MM,
        "prediction_ground_truth_metric_affine_tolerance_unchanged": True,
        "original_mismatch_record_count": len(mismatches),
        "reconciled_mismatch_record_count": len(reconciled),
        "resolved_qform_sform_observation_case_count": len(qform_observation_cases),
        "resolved_effective_affine_case_count": len(affine_resolution_cases),
        "unresolved_mismatch_record_count": len(unresolved),
        "unresolved_case_ids": sorted({str(item["anonymous_case_id"]) for item in unresolved}),
        "diagnostics": diagnostics,
        "historical_phase2_basis": {
            "decision_date": "2026-07-27",
            "approved_source_affine_tolerance_mm": R1_SOURCE_AFFINE_ABS_TOLERANCE_MM,
            "max_selected_affine_delta_mm": PHASE2_MAX_SELECTED_AFFINE_DELTA_MM,
            "max_corner_displacement_mm": PHASE2_MAX_CORNER_DISPLACEMENT_MM,
            "max_voxel_translation_delta": PHASE2_MAX_VOXEL_TRANSLATION_DELTA,
            "max_linear_transform_deviation": PHASE2_MAX_LINEAR_TRANSFORM_DEVIATION,
        },
    }
    return reconciled, summary
