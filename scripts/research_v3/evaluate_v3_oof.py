from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SMALL_LESION_LT_ML = 5.0
MEDIUM_LESION_LT_ML = 20.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = int(np.count_nonzero(a))
    bb = int(np.count_nonzero(b))
    inter = int(np.count_nonzero(np.logical_and(a, b)))
    return 1.0 if aa + bb == 0 else (2.0 * inter) / (aa + bb)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(np.logical_or(a, b)))
    return 1.0 if union == 0 else int(np.count_nonzero(np.logical_and(a, b))) / union


def size_bin(volume_ml: float) -> str:
    if volume_ml < SMALL_LESION_LT_ML:
        return "small"
    if volume_ml < MEDIUM_LESION_LT_ML:
        return "medium"
    return "large"


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy import ndimage

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = ndimage.label(mask, structure=structure)
    return labels, int(count)


def _one_to_one_component_matches(
    gt_labels: np.ndarray,
    gt_count: int,
    pred_labels: np.ndarray,
    pred_count: int,
) -> list[tuple[int, int, int]]:
    """Maximum-overlap one-to-one component matching.

    Returns tuples (gt_component_id, pred_component_id, overlap_voxels).
    Zero-overlap assignments are discarded.
    """
    if gt_count == 0 or pred_count == 0:
        return []

    from scipy.optimize import linear_sum_assignment

    overlap = np.zeros((gt_count, pred_count), dtype=np.int64)
    both = np.logical_and(gt_labels > 0, pred_labels > 0)
    if np.any(both):
        g = gt_labels[both].astype(np.int64) - 1
        p = pred_labels[both].astype(np.int64) - 1
        np.add.at(overlap, (g, p), 1)

    rows, cols = linear_sum_assignment(-overlap)
    matches: list[tuple[int, int, int]] = []
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        ov = int(overlap[row, col])
        if ov > 0:
            matches.append((row + 1, col + 1, ov))
    return matches


def lesion_stats(
    gt: np.ndarray,
    pred: np.ndarray,
) -> tuple[int, int, int, float | None, float | None, float, list[tuple[int, int, int]]]:
    gt_labels, gt_count = _label_components(gt)
    pred_labels, pred_count = _label_components(pred)
    matches = _one_to_one_component_matches(gt_labels, gt_count, pred_labels, pred_count)
    tp = len(matches)
    fn = gt_count - tp
    fp = pred_count - tp
    recall = None if gt_count == 0 else tp / gt_count
    precision = None if pred_count == 0 else tp / pred_count
    if recall is None or precision is None or (recall + precision) == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * recall * precision / (recall + precision)
    return tp, fp, fn, recall, precision, f1, matches


def _lesion_size_records(
    gt_tumor: np.ndarray,
    pred_tumor: np.ndarray,
    voxel_ml: float,
) -> list[dict[str, Any]]:
    gt_labels, gt_count = _label_components(gt_tumor)
    pred_labels, pred_count = _label_components(pred_tumor)
    matches = _one_to_one_component_matches(gt_labels, gt_count, pred_labels, pred_count)
    pred_by_gt = {gt_id: pred_id for gt_id, pred_id, _ in matches}

    rows: list[dict[str, Any]] = []
    for gt_id in range(1, gt_count + 1):
        gt_component = gt_labels == gt_id
        gt_volume_ml = float(np.count_nonzero(gt_component) * voxel_ml)
        pred_id = pred_by_gt.get(gt_id)
        pred_component = np.zeros_like(gt_component, dtype=bool)
        if pred_id is not None:
            pred_component = pred_labels == pred_id
        rows.append(
            {
                "size_bin": size_bin(gt_volume_ml),
                "ground_truth_volume_ml": gt_volume_ml,
                "matched": pred_id is not None,
                "dice": dice(gt_component, pred_component),
            }
        )
    return rows


def _fold_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for fold in range(5):
        subset = [row for row in rows if row["fold"] == fold]
        if not subset:
            raise RuntimeError(f"fold {fold} has no OOF cases")
        out[str(fold)] = {
            "case_count": len(subset),
            "mean_tumor_dice": float(np.mean([row["tumor_dice"] for row in subset])),
            "mean_liver_region_dice": float(
                np.mean([row["liver_region_dice"] for row in subset])
            ),
        }
    return out


def evaluate(labels_dir: Path, predictions_dir: Path, folds: dict[str, Any]) -> dict[str, Any]:
    import nibabel as nib

    assignments = folds.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("invalid fold artifact")
    expected = {row["anonymous_case_id"] for row in assignments}
    if len(expected) != len(assignments):
        raise RuntimeError("fold artifact contains duplicate case ids")

    prediction_files = sorted(predictions_dir.glob("*.nii")) + sorted(predictions_dir.glob("*.nii.gz"))
    def _case_stem(path: Path) -> str:
        return path.name.removesuffix(".nii.gz").removesuffix(".nii")
    found = {_case_stem(path) for path in prediction_files}
    if found != expected:
        raise RuntimeError(
            f"OOF prediction set mismatch missing={sorted(expected - found)} "
            f"extra={sorted(found - expected)}"
        )

    fold_by_case = {row["anonymous_case_id"]: int(row["validation_fold"]) for row in assignments}
    rows: list[dict[str, Any]] = []
    lesion_rows: list[dict[str, Any]] = []

    for cid in sorted(expected):
        gt_candidates = [labels_dir / f"{cid}.nii", labels_dir / f"{cid}.nii.gz"]
        pred_candidates = [predictions_dir / f"{cid}.nii", predictions_dir / f"{cid}.nii.gz"]
        gt_path = next((p for p in gt_candidates if p.is_file()), None)
        pred_path = next((p for p in pred_candidates if p.is_file()), None)
        if gt_path is None or pred_path is None:
            raise FileNotFoundError(cid)

        gt_img = nib.load(str(gt_path))
        pred_img = nib.load(str(pred_path))
        if gt_img.shape != pred_img.shape:
            raise RuntimeError(f"geometry shape mismatch: {cid}")
        if not np.allclose(gt_img.affine, pred_img.affine, rtol=0.0, atol=1e-5):
            raise RuntimeError(f"geometry affine mismatch: {cid}")

        gt = np.asarray(gt_img.dataobj)
        pred = np.asarray(pred_img.dataobj)
        if not np.all(np.isfinite(gt)) or not np.all(np.isfinite(pred)):
            raise RuntimeError(f"non-finite segmentation values: {cid}")
        if not set(np.unique(gt).tolist()).issubset({0, 1, 2}):
            raise RuntimeError(f"invalid GT labels: {cid}")
        if not set(np.unique(pred).tolist()).issubset({0, 1, 2}):
            raise RuntimeError(f"invalid prediction labels: {cid}")

        gt_tumor = gt == 2
        pred_tumor = pred == 2
        gt_liver_region = gt > 0
        pred_liver_region = pred > 0

        tp_les, fp_les, fn_les, recall, precision, lesion_f1, _ = lesion_stats(
            gt_tumor, pred_tumor
        )

        spacing = np.sqrt((gt_img.affine[:3, :3] ** 2).sum(axis=0))
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise RuntimeError(f"invalid voxel spacing: {cid}")
        voxel_ml = float(np.prod(spacing) / 1000.0)
        gt_voxels = int(np.count_nonzero(gt_tumor))
        pred_voxels = int(np.count_nonzero(pred_tumor))
        gt_ml = float(gt_voxels * voxel_ml)
        pred_ml = float(pred_voxels * voxel_ml)

        case_row: dict[str, Any] = {
            "case_id": cid,
            "fold": fold_by_case[cid],
            "tumor_dice": dice(gt_tumor, pred_tumor),
            "tumor_iou": iou(gt_tumor, pred_tumor),
            "liver_region_dice": dice(gt_liver_region, pred_liver_region),
            "ground_truth_tumor_voxels": gt_voxels,
            "predicted_tumor_voxels": pred_voxels,
            "ground_truth_lesion_count": tp_les + fn_les,
            "predicted_lesion_count": tp_les + fp_les,
            "true_positive_lesions": tp_les,
            "false_positive_lesions": fp_les,
            "false_negative_lesions": fn_les,
            "lesion_recall": recall,
            "lesion_precision": precision,
            "lesion_f1": lesion_f1,
            "false_positive_lesions_per_scan": fp_les,
            "tumor_volume_error_ml": pred_ml - gt_ml,
            "absolute_tumor_volume_error_ml": abs(pred_ml - gt_ml),
            "prediction_sha256": sha256(pred_path),
        }
        rows.append(case_row)

        for lesion in _lesion_size_records(gt_tumor, pred_tumor, voxel_ml):
            lesion_rows.append({"case_id": cid, "fold": fold_by_case[cid], **lesion})

    mean_tumor = float(np.mean([row["tumor_dice"] for row in rows]))
    mean_liver = float(np.mean([row["liver_region_dice"] for row in rows]))

    size_summary: dict[str, dict[str, float | int | None]] = {}
    for bin_name in ("small", "medium", "large"):
        subset = [row for row in lesion_rows if row["size_bin"] == bin_name]
        size_summary[bin_name] = {
            "lesion_count": len(subset),
            "matched_lesion_count": sum(1 for row in subset if row["matched"]),
            "mean_lesion_dice": None
            if not subset
            else float(np.mean([row["dice"] for row in subset])),
        }

    tumor_pass = mean_tumor >= 0.70
    liver_pass = mean_liver >= 0.90
    report: dict[str, Any] = {
        "schema_version": "protoem_ct.research_v3.oof_report.v2",
        "case_count": len(rows),
        "fold_count": 5,
        "oof_mean_tumor_dice": mean_tumor,
        "oof_mean_liver_region_dice": mean_liver,
        "oof_mean_tumor_iou": float(np.mean([row["tumor_iou"] for row in rows])),
        "oof_mean_false_positive_lesions_per_scan": float(
            np.mean([row["false_positive_lesions_per_scan"] for row in rows])
        ),
        "tumor_pass": tumor_pass,
        "liver_pass": liver_pass,
        "all_131_oof_predictions_present": len(rows) == 131,
        "final_pass": tumor_pass and liver_pass and len(rows) == 131,
        "fold_summary": _fold_summary(rows),
        "lesion_size_summary": size_summary,
        "cases": rows,
        "lesions": lesion_rows,
    }
    return report


def write_case_csv(report: dict[str, Any], path: Path) -> None:
    rows = report["cases"]
    if not rows:
        raise ValueError("report has no cases")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-csv", type=Path)
    args = parser.parse_args()

    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    report = evaluate(args.labels, args.predictions, folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.case_csv:
        write_case_csv(report, args.case_csv)

    print(f"CASE_COUNT={report['case_count']}")
    print(f"OOF_MEAN_TUMOR_DICE={report['oof_mean_tumor_dice']:.6f}")
    print(f"OOF_MEAN_LIVER_REGION_DICE={report['oof_mean_liver_region_dice']:.6f}")
    print(f"FINAL_V3A_PASS={report['final_pass']}")


if __name__ == "__main__":
    main()