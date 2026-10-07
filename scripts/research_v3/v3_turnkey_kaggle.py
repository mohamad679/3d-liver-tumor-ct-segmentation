from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

DATASET_ID = 903
DATASET_NAME = "ProtoEMV3LiverTumor"
DATASET_FOLDER = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
NNUNET_VERSION = "2.8.1"
TRAINER = "nnUNetTrainer_100epochs"
CONFIGURATION = "3d_fullres"
PLANS = "nnUNetPlans"
CHECKPOINT = "checkpoint_final.pth"
LATEST_CHECKPOINT = "checkpoint_latest.pth"
EXPECTED_CASES = 131
REPAIR_IDS = {48, 49, 50, 51, 52}
SEED = 1729
N_SHARDS = 6
SOURCE_LOCK = "4baf8ff86c43f2f2efe6c15f8c1dab4ef24f06eafa6ffd49697a52bc31c19ff5"
SOURCE_LOCK_SHORT = SOURCE_LOCK[:8]
DEFAULT_OWNER = "mohamadasgari"
PART1 = Path("/kaggle/input/datasets/andrewmvd/liver-tumor-segmentation")
PART2 = Path("/kaggle/input/datasets/andrewmvd/liver-tumor-segmentation-part-2")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("COMMAND:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def ensure_nnunet() -> None:
    try:
        version = importlib.metadata.version("nnunetv2")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != NNUNET_VERSION:
        run([sys.executable, "-m", "pip", "install", "--no-cache-dir", f"nnunetv2=={NNUNET_VERSION}"])
    version = importlib.metadata.version("nnunetv2")
    if version != NNUNET_VERSION:
        raise RuntimeError(f"expected nnunetv2 {NNUNET_VERSION}, got {version}")
    print(f"NNUNET_VERSION={version}")


def discover_sources(part1: Path = PART1, part2: Path = PART2) -> tuple[dict[int, Path], dict[int, Path]]:
    vol_re = re.compile(r"^volume-(\d+)\.nii(?:\.gz)?$")
    seg_re = re.compile(r"^segmentation-(\d+)\.nii(?:\.gz)?$")
    volumes: dict[int, Path] = {}
    labels: dict[int, Path] = {}
    for root in (part1, part2):
        if not root.is_dir():
            raise FileNotFoundError(root)
        for p in root.rglob("*"):
            if p.is_file() and (m := vol_re.match(p.name)):
                cid = int(m.group(1))
                if cid in volumes:
                    raise RuntimeError(f"duplicate volume {cid}")
                volumes[cid] = p
    for p in part1.rglob("*"):
        if p.is_file() and (m := seg_re.match(p.name)):
            cid = int(m.group(1))
            if cid in labels:
                raise RuntimeError(f"duplicate label {cid}")
            labels[cid] = p
    expected = set(range(EXPECTED_CASES))
    if set(volumes) != expected or set(labels) != expected:
        raise RuntimeError("source discovery must yield volume/label ids exactly 0..130")
    return volumes, labels


def spacing_from_affine(affine: Any) -> tuple[float, float, float]:
    import numpy as np
    return tuple(float(x) for x in np.linalg.norm(np.asarray(affine, dtype=float)[:3, :3], axis=0))


def _label_stats_streaming(label: Any) -> tuple[set[int], int]:
    """Read a 3D label volume one z-slice at a time to bound peak RAM."""
    import numpy as np

    unique: set[int] = set()
    tumor_voxels = 0
    if len(label.shape) != 3:
        raise RuntimeError(f"expected 3D label, got shape={label.shape}")
    for z in range(int(label.shape[2])):
        slab = np.asanyarray(label.dataobj[:, :, z])
        values, counts = np.unique(slab, return_counts=True)
        unique.update(int(v) for v in values)
        for value, count in zip(values, counts, strict=True):
            if int(value) == 2:
                tumor_voxels += int(count)
        del slab, values, counts
    return unique, tumor_voxels


def semantic_source_lock(volumes: dict[int, Path], labels: dict[int, Path]) -> tuple[str, list[dict[str, Any]]]:
    import nibabel as nib

    records: list[dict[str, Any]] = []
    print("SOURCE_LOCK_SCAN=START cases=131", flush=True)
    for cid in range(EXPECTED_CASES):
        image = nib.load(str(volumes[cid]), mmap=True)
        label = nib.load(str(labels[cid]), mmap=True)
        if image.shape != label.shape:
            raise RuntimeError(f"shape mismatch {cid}")
        unique, tumor_voxels = _label_stats_streaming(label)
        if not unique.issubset({0, 1, 2}):
            raise RuntimeError(f"invalid labels {cid}: {sorted(unique)}")
        records.append({
            "case_id": cid,
            "image_sha256": sha256_file(volumes[cid]),
            "original_label_sha256": sha256_file(labels[cid]),
            "shape": list(image.shape),
            "spacing_mm": [round(float(x), 8) for x in spacing_from_affine(image.affine)],
            "tumor_voxels": tumor_voxels,
            "label_policy": "header_repaired_no_resampling" if cid in REPAIR_IDS else "original",
        })
        if cid % 10 == 0 or cid == EXPECTED_CASES - 1:
            print(f"SOURCE_LOCK_SCAN_PROGRESS={cid + 1}/{EXPECTED_CASES}", flush=True)
        del image, label
    print("SOURCE_LOCK_SCAN=COMPLETE", flush=True)
    return canonical_sha256(records), records


def _voxel_hash(arr: Any) -> str:
    import numpy as np
    data = np.asarray(arr, dtype=np.int16)
    h = hashlib.sha256()
    h.update(json.dumps(list(data.shape), separators=(",", ":")).encode("utf-8"))
    h.update(data.tobytes(order="C"))
    return h.hexdigest()


def _rank_patient(patient_id: str, seed: int = SEED) -> str:
    payload = f"protoem-ct.v3.fold-rank.v1|{seed}|{patient_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_splits(case_ids: list[str], seed: int = SEED) -> tuple[dict[str, Any], list[dict[str, list[str]]]]:
    patients = [f"patient_{int(cid.split('_')[-1]):03d}" for cid in case_ids]
    patient_to_case = dict(zip(patients, case_ids, strict=True))
    ranked = sorted(patients, key=lambda p: (_rank_patient(p, seed), p))
    fold_patients: dict[int, list[str]] = {i: [] for i in range(5)}
    for idx, pid in enumerate(ranked):
        fold_patients[idx % 5].append(pid)
    assignments: list[dict[str, Any]] = []
    for fold in range(5):
        for pid in fold_patients[fold]:
            assignments.append({
                "anonymous_patient_id": pid,
                "anonymous_case_id": patient_to_case[pid],
                "validation_fold": fold,
            })
    assignments.sort(key=lambda x: x["anonymous_case_id"])
    all_cases = sorted(case_ids)
    splits: list[dict[str, list[str]]] = []
    val_union: set[str] = set()
    counts: dict[str, int] = {}
    for fold in range(5):
        val = sorted(x["anonymous_case_id"] for x in assignments if x["validation_fold"] == fold)
        train = sorted(set(all_cases) - set(val))
        if set(train) & set(val):
            raise RuntimeError("fold leakage")
        val_union.update(val)
        counts[str(fold)] = len(val)
        splits.append({"train": train, "val": val})
    if val_union != set(all_cases):
        raise RuntimeError("OOF folds do not cover each case exactly once")
    artifact = {
        "schema_version": "protoem_ct.research_v3.folds.v3",
        "seed": seed,
        "fold_count": 5,
        "case_count": len(case_ids),
        "fold_case_counts": counts,
        "assignments": assignments,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact, splits


def build_balanced_shards(case_work: list[dict[str, Any]], n_shards: int = N_SHARDS) -> list[dict[str, Any]]:
    shards = [{"index": i, "cases": [], "approx_target_voxels": 0.0} for i in range(n_shards)]
    for rec in sorted(case_work, key=lambda x: (-float(x["approx_target_voxels"]), str(x["case"]))):
        target = min(shards, key=lambda s: (float(s["approx_target_voxels"]), int(s["index"])))
        target["cases"].append(str(rec["case"]))
        target["approx_target_voxels"] += float(rec["approx_target_voxels"])
    for shard in shards:
        shard["cases"].sort()
    return shards


def base_paths(root: Path) -> dict[str, Path]:
    base = root / "v3_nnunet"
    return {
        "base": base,
        "raw": base / "nnUNet_raw",
        "preprocessed": base / "nnUNet_preprocessed",
        "results": base / "nnUNet_results",
        "work": root / "v3_work",
        "dataset": base / "nnUNet_raw" / DATASET_FOLDER,
        "pp_dataset": base / "nnUNet_preprocessed" / DATASET_FOLDER,
    }


def build_raw_dataset(root: Path) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np

    paths = base_paths(root)
    dataset = paths["dataset"]
    if dataset.exists():
        print(f"RAW_DATASET=REUSE {dataset}")
        return {"dataset": str(dataset), "reused": True}
    volumes, labels = discover_sources()
    observed_lock, source_records = semantic_source_lock(volumes, labels)
    if observed_lock != SOURCE_LOCK:
        raise RuntimeError(f"source semantic lock mismatch expected={SOURCE_LOCK} observed={observed_lock}")
    print(f"SOURCE_SEMANTIC_LOCK=PASS {observed_lock}", flush=True)
    source_record_by_id = {int(rec["case_id"]): rec for rec in source_records}
    tmp = dataset.with_name(dataset.name + "__tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    images_tr = tmp / "imagesTr"
    labels_tr = tmp / "labelsTr"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)
    case_records = []
    for cid in range(EXPECTED_CASES):
        case = f"case_{cid:03d}"
        image_path = volumes[cid]
        label_path = labels[cid]
        image = nib.load(str(image_path), mmap=True)
        label = nib.load(str(label_path), mmap=True)
        if image.shape != label.shape:
            raise RuntimeError(f"shape mismatch {cid}")
        source_record = source_record_by_id[cid]
        dst_image = images_tr / f"{case}_0000.nii"
        os.symlink(image_path.resolve(), dst_image)
        dst_label = labels_tr / f"{case}.nii"
        if cid in REPAIR_IDS:
            # Only the five preregistered header-repair cases are materialized
            # in full. All other labels remain streaming/symlinked.
            arr = np.asanyarray(label.dataobj)
            unique = set(int(x) for x in np.unique(arr))
            if not unique.issubset({0, 1, 2}):
                raise RuntimeError(f"invalid labels {cid}: {sorted(unique)}")
            before_voxel_hash = _voxel_hash(arr)
            repaired = nib.Nifti1Image(arr, np.asarray(image.affine, dtype=float))
            repaired.set_data_dtype(label.get_data_dtype())
            q, qc = image.get_qform(coded=True)
            s, sc = image.get_sform(coded=True)
            repaired.set_qform(image.affine, code=int(qc) if int(qc) > 0 else 1)
            repaired.set_sform(image.affine, code=int(sc) if int(sc) > 0 else 1)
            nib.save(repaired, str(dst_label))
            repaired_check = nib.load(str(dst_label), mmap=True)
            after_voxel_hash = _voxel_hash(np.asanyarray(repaired_check.dataobj))
            if before_voxel_hash != after_voxel_hash:
                raise RuntimeError(f"repair changed voxel values: {cid}")
            del repaired_check, repaired, arr
        else:
            if not np.allclose(image.affine, label.affine, rtol=0.0, atol=1e-4):
                raise RuntimeError(f"unexpected affine mismatch outside repair set: {cid}")
            os.symlink(label_path.resolve(), dst_label)
        check = nib.load(str(dst_label), mmap=True)
        if check.shape != image.shape or not np.allclose(check.affine, image.affine, rtol=0.0, atol=1e-4):
            raise RuntimeError(f"effective geometry mismatch {cid}")
        case_records.append({
            "anonymous_patient_id": f"patient_{cid:03d}",
            "anonymous_case_id": case,
            "source_case_id": cid,
            "image_sha256": sha256_file(image_path),
            "original_label_sha256": sha256_file(label_path),
            "effective_label_sha256": sha256_file(dst_label),
            "label_source": "header_repaired_no_resampling" if cid in REPAIR_IDS else "original",
            "tumor_voxels": int(source_record["tumor_voxels"]),
        })
        if cid % 10 == 0 or cid == EXPECTED_CASES - 1:
            print(f"RAW_DATASET_BUILD_PROGRESS={cid + 1}/{EXPECTED_CASES}", flush=True)
        del image, label, check
    (tmp / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": 1, "tumor": 2},
        "numTraining": EXPECTED_CASES,
        "file_ending": ".nii",
    }, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "protoem_ct.research_v3.dataset_manifest.v3",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "case_count": EXPECTED_CASES,
        "source_semantic_lock_sha256": SOURCE_LOCK,
        "header_repair_case_ids": sorted(REPAIR_IDS),
        "header_repair_policy": "copy_image_geometry_only_no_resampling",
        "cases": case_records,
    }
    manifest["manifest_content_hash"] = canonical_sha256(manifest)
    work = paths["work"]
    work.mkdir(parents=True, exist_ok=True)
    (work / "v3_dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    folds, splits = build_splits([f"case_{i:03d}" for i in range(EXPECTED_CASES)])
    (work / "v3_folds.json").write_text(json.dumps(folds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (work / "splits_final.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    tmp.rename(dataset)
    print("RAW_DATASET_BUILD=PASS")
    return {"dataset": str(dataset), "reused": False}


def plan_dataset(root: Path) -> dict[str, Any]:
    paths = base_paths(root)
    for p in (paths["raw"], paths["preprocessed"], paths["results"], paths["work"]):
        p.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["nnUNet_raw"] = str(paths["raw"])
    env["nnUNet_preprocessed"] = str(paths["preprocessed"])
    env["nnUNet_results"] = str(paths["results"])
    pp = paths["pp_dataset"]
    plans = pp / "nnUNetPlans.json"
    if not plans.exists():
        run(["nnUNetv2_extract_fingerprint", "-d", str(DATASET_ID), "-np", "4", "--verify_dataset_integrity"], env=env)
        run(["nnUNetv2_plan_experiment", "-d", str(DATASET_ID)], env=env)
    if not plans.exists():
        raise RuntimeError("nnUNetPlans.json missing")
    source_splits = paths["work"] / "splits_final.json"
    target_splits = pp / "splits_final.json"
    if not target_splits.exists():
        shutil.copy2(source_splits, target_splits)
    elif sha256_file(source_splits) != sha256_file(target_splits):
        raise RuntimeError("splits_final mismatch")
    plan = json.loads(plans.read_text(encoding="utf-8"))
    cfg = plan["configurations"][CONFIGURATION]
    print("PLAN_3D_FULLRES=" + json.dumps({k: cfg.get(k) for k in ("spacing", "patch_size", "batch_size", "data_identifier", "preprocessor_name")}, sort_keys=True))
    return cfg


def _dataset_exists(handle: str) -> bool:
    return run(["kaggle", "datasets", "files", handle, "--page-size", "200", "-v"], check=False).returncode == 0


def _remote_complete(handle: str, expected_names: list[str]) -> bool:
    result = run(["kaggle", "datasets", "files", handle, "--page-size", "200", "-v"], check=False)
    if result.returncode != 0:
        return False
    text = result.stdout + result.stderr
    return all(name in text for name in expected_names)


def shard_handles(owner: str) -> list[str]:
    return [f"{owner}/protoem-v3-pp903-{SOURCE_LOCK_SHORT}-s{i:02d}" for i in range(N_SHARDS)]


def preprocess_and_persist_shards(root: Path, owner: str) -> list[str]:
    import nibabel as nib
    import numpy as np
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    paths = base_paths(root)
    cfg = plan_dataset(root)
    pp = paths["pp_dataset"]
    plans_path = pp / "nnUNetPlans.json"
    dataset_json = paths["dataset"] / "dataset.json"
    plans_manager = PlansManager(str(plans_path))
    config_manager = plans_manager.get_configuration(CONFIGURATION)
    preprocessor = DefaultPreprocessor(verbose=False)
    preprocessor.show_progress_bar = False
    target_spacing = np.asarray(config_manager.spacing, dtype=float)
    case_work = []
    for cid in range(EXPECTED_CASES):
        case = f"case_{cid:03d}"
        img = nib.load(str(paths["dataset"] / "imagesTr" / f"{case}_0000.nii"), mmap=True)
        shape = np.asarray(img.shape, dtype=float)
        spacing = np.linalg.norm(np.asarray(img.affine, dtype=float)[:3, :3], axis=0)
        approx = float(np.prod(shape) * np.prod(spacing) / np.prod(target_spacing))
        case_work.append({"case": case, "approx_target_voxels": approx})
        del img
    shards = build_balanced_shards(case_work)
    handles = shard_handles(owner)
    for shard, handle in zip(shards, handles, strict=True):
        expected = [
            name
            for case in shard["cases"]
            for name in (f"{case}.b2nd", f"{case}_seg.b2nd", f"{case}.pkl")
        ]
        expected += ["control_nnUNetPlans.json", "control_dataset.json", "control_splits_final.json", "stage7e_shard_manifest.json"]
        if _remote_complete(handle, expected):
            print(f"SHARD_REMOTE_COMPLETE=SKIP {handle}")
            continue
        shard_dir = paths["work"] / "pp_shards" / f"s{shard['index']:02d}"
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True)
        records = []
        for pos, case in enumerate(shard["cases"], 1):
            free_gib = shutil.disk_usage(root).free / 1024**3
            if free_gib < 7.0:
                raise RuntimeError("disk headroom below 7 GiB")
            print(f"PREPROCESS s{shard['index']:02d} {pos}/{len(shard['cases'])} {case}", flush=True)
            t0 = time.time()
            preprocessor.run_case_save(
                str(shard_dir / case),
                [str(paths["dataset"] / "imagesTr" / f"{case}_0000.nii")],
                str(paths["dataset"] / "labelsTr" / f"{case}.nii"),
                plans_manager,
                config_manager,
                str(dataset_json),
            )
            records.append({"case": case, "seconds": time.time() - t0})
        shutil.copy2(plans_path, shard_dir / "control_nnUNetPlans.json")
        shutil.copy2(dataset_json, shard_dir / "control_dataset.json")
        shutil.copy2(pp / "splits_final.json", shard_dir / "control_splits_final.json")
        manifest = {
            "schema_version": "protoem_ct.research_v3.preprocess_shard.v2",
            "source_semantic_lock_sha256": SOURCE_LOCK,
            "dataset_id": DATASET_ID,
            "configuration": CONFIGURATION,
            "shard_index": shard["index"],
            "case_count": len(shard["cases"]),
            "cases": shard["cases"],
            "records": records,
        }
        manifest["artifact_sha256"] = canonical_sha256(manifest)
        (shard_dir / "stage7e_shard_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (shard_dir / "dataset-metadata.json").write_text(json.dumps({
            "title": handle.split("/", 1)[1],
            "id": handle,
            "licenses": [{"name": "other"}],
        }, indent=2) + "\n", encoding="utf-8")
        if _dataset_exists(handle):
            cmd = ["kaggle", "datasets", "version", "-p", str(shard_dir), "-m", "V3 preprocessed shard", "-q", "-t", "-r", "skip"]
        else:
            cmd = ["kaggle", "datasets", "create", "-p", str(shard_dir), "-q", "-t", "-r", "skip"]
        run(cmd)
        deadline = time.time() + 900
        while time.time() < deadline and not _remote_complete(handle, expected):
            time.sleep(10)
        if not _remote_complete(handle, expected):
            raise RuntimeError(f"remote verification failed: {handle}")
        shutil.rmtree(shard_dir)
        print(f"SHARD_PERSIST=PASS {handle}")
    return handles


def find_attached_shards(owner: str, input_root: Path = Path("/kaggle/input")) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for handle in shard_handles(owner):
        slug = handle.split("/", 1)[1]
        candidates = [p for p in input_root.rglob(slug) if p.is_dir()]
        if len(candidates) == 1:
            found[handle] = candidates[0]
    return found


def assemble_preprocessed_from_mounts(root: Path, owner: str) -> bool:
    paths = base_paths(root)
    attached = find_attached_shards(owner)
    if len(attached) != N_SHARDS:
        print("ACTION_REQUIRED_ATTACH_INPUTS")
        for handle in shard_handles(owner):
            print(handle)
        print(f"ATTACHED_SHARDS={len(attached)}/{N_SHARDS}")
        return False
    cfg = json.loads((paths["pp_dataset"] / "nnUNetPlans.json").read_text(encoding="utf-8"))["configurations"][CONFIGURATION]
    data_identifier = cfg["data_identifier"]
    target = paths["pp_dataset"] / data_identifier
    target.mkdir(parents=True, exist_ok=True)
    seen = set()
    artifact_re = re.compile(r"^case_\d{3}(?:\.b2nd|_seg\.b2nd|\.pkl)$")
    for mount in attached.values():
        for p in mount.iterdir():
            if p.is_file() and artifact_re.match(p.name):
                if p.name in seen:
                    raise RuntimeError(f"duplicate preprocessed artifact: {p.name}")
                seen.add(p.name)
                dst = target / p.name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink(p.resolve(), dst)
    expected = {
        name
        for i in range(EXPECTED_CASES)
        for name in (
            f"case_{i:03d}.b2nd",
            f"case_{i:03d}_seg.b2nd",
            f"case_{i:03d}.pkl",
        )
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise RuntimeError(
            "preprocessed mounted file set mismatch "
            f"missing_count={len(missing)} extra_count={len(extra)} "
            f"missing_sample={missing[:10]} extra_sample={extra[:10]}"
        )
    print(f"ASSEMBLED_PREPROCESSED=PASS files={len(seen)}")
    return True


def model_root(paths: dict[str, Path]) -> Path:
    return paths["results"] / DATASET_FOLDER / f"{TRAINER}__{PLANS}__{CONFIGURATION}"


def fold_remote_handle(owner: str, fold: int) -> str:
    return f"{owner}/protoem-v3a-nnunet100-{SOURCE_LOCK_SHORT}-fold{fold}"


def fold_archive_name(fold: int, final: bool) -> str:
    state = "final" if final else "latest"
    return f"fold{fold}_{state}.zip"


def remote_fold_complete(owner: str, fold: int) -> bool:
    return _remote_complete(fold_remote_handle(owner, fold), [fold_archive_name(fold, True)])


def remote_fold_resume_available(owner: str, fold: int) -> bool:
    handle = fold_remote_handle(owner, fold)
    return _remote_complete(handle, [fold_archive_name(fold, True)]) or _remote_complete(
        handle, [fold_archive_name(fold, False)]
    )


def persist_fold_checkpoint(root: Path, owner: str, fold: int, *, final: bool) -> None:
    import tempfile
    import zipfile

    paths = base_paths(root)
    mroot = model_root(paths)
    fold_dir = mroot / f"fold_{fold}"
    checkpoint_name = CHECKPOINT if final else LATEST_CHECKPOINT
    checkpoint = fold_dir / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    handle = fold_remote_handle(owner, fold)
    upload_root = paths["work"] / "fold_upload" / f"fold_{fold}"
    if upload_root.exists():
        shutil.rmtree(upload_root)
    upload_root.mkdir(parents=True)
    archive_path = upload_root / fold_archive_name(fold, final)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / mroot.name
        staged_fold = stage / f"fold_{fold}"
        staged_fold.mkdir(parents=True)
        shutil.copy2(checkpoint, staged_fold / checkpoint_name)
        for name in ("checkpoint_best.pth",):
            src = fold_dir / name
            if src.is_file():
                shutil.copy2(src, staged_fold / name)
        for name in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
            src = mroot / name
            if src.is_file():
                shutil.copy2(src, stage / name)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as z:
            for p in Path(tmp).rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(tmp))

    state = {
        "schema_version": "protoem_ct.research_v3.fold_persistence.v2",
        "fold": fold,
        "state": "final" if final else "latest",
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "archive_name": archive_path.name,
        "source_semantic_lock_sha256": SOURCE_LOCK,
        "trainer": TRAINER,
        "configuration": CONFIGURATION,
    }
    (upload_root / "fold_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (upload_root / "dataset-metadata.json").write_text(
        json.dumps({"title": handle.split("/", 1)[1], "id": handle, "licenses": [{"name": "other"}]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if _dataset_exists(handle):
        cmd = ["kaggle", "datasets", "version", "-p", str(upload_root), "-m", f"fold {fold} {'final' if final else 'latest'} checkpoint", "-q"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(upload_root), "-q"]
    run(cmd)
    expected = [archive_path.name, "fold_state.json"]
    deadline = time.time() + 900
    while time.time() < deadline and not _remote_complete(handle, expected):
        time.sleep(10)
    if not _remote_complete(handle, expected):
        raise RuntimeError(f"fold persistence verification failed: {fold}")
    shutil.rmtree(upload_root)
    print(f"FOLD_PERSIST=PASS fold={fold} state={'final' if final else 'latest'}")


def restore_fold(root: Path, owner: str, fold: int, *, allow_latest: bool = True) -> str | None:
    import zipfile

    paths = base_paths(root)
    mroot = model_root(paths)
    final_checkpoint = mroot / f"fold_{fold}" / CHECKPOINT
    latest_checkpoint = mroot / f"fold_{fold}" / LATEST_CHECKPOINT
    if final_checkpoint.is_file():
        return "final"
    if allow_latest and latest_checkpoint.is_file():
        return "latest"

    handle = fold_remote_handle(owner, fold)
    if not remote_fold_resume_available(owner, fold):
        return None

    dl = paths["work"] / "restore" / f"fold_{fold}"
    if dl.exists():
        shutil.rmtree(dl)
    dl.mkdir(parents=True)
    run(["kaggle", "datasets", "download", "-d", handle, "-p", str(dl), "--unzip"])

    final_archives = list(dl.rglob(fold_archive_name(fold, True)))
    latest_archives = list(dl.rglob(fold_archive_name(fold, False)))
    archives = final_archives or (latest_archives if allow_latest else [])
    if len(archives) != 1:
        raise RuntimeError(f"expected one restorable archive for fold {fold}, got {len(archives)}")
    with zipfile.ZipFile(archives[0], "r") as z:
        z.extractall(dl / "unzipped")

    restored_checkpoint_name = CHECKPOINT if final_archives else LATEST_CHECKPOINT
    candidates = [
        c for c in (dl / "unzipped").rglob(restored_checkpoint_name) if c.parent.name == f"fold_{fold}"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one {restored_checkpoint_name} for fold {fold}, got {len(candidates)}")
    restored_root = candidates[0].parents[1]
    mroot.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        src = restored_root / name
        if src.is_file():
            shutil.copy2(src, mroot / name)
    target_fold = mroot / f"fold_{fold}"
    target_fold.mkdir(parents=True, exist_ok=True)
    for name in (restored_checkpoint_name, "checkpoint_best.pth"):
        src = restored_root / f"fold_{fold}" / name
        if src.is_file():
            shutil.copy2(src, target_fold / name)
    state = "final" if restored_checkpoint_name == CHECKPOINT else "latest"
    print(f"RESTORE_FOLD=PASS fold={fold} state={state}")
    return state


def _run_training_with_checkpoint_persistence(root: Path, owner: str, fold: int, gpu: str) -> int:
    paths = base_paths(root)
    env = os.environ.copy()
    env["nnUNet_raw"] = str(paths["raw"])
    env["nnUNet_preprocessed"] = str(paths["preprocessed"])
    env["nnUNet_results"] = str(paths["results"])
    env["CUDA_VISIBLE_DEVICES"] = gpu

    restored_state = restore_fold(root, owner, fold, allow_latest=True)
    cmd = [
        "nnUNetv2_train", str(DATASET_ID), CONFIGURATION, str(fold),
        "-tr", TRAINER, "-p", PLANS, "-device", "cuda",
    ]
    if restored_state == "latest":
        cmd.append("--c")

    print("COMMAND:", " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, env=env, text=True)
    latest = model_root(paths) / f"fold_{fold}" / LATEST_CHECKPOINT
    last_persisted_hash: str | None = None
    while process.poll() is None:
        if latest.is_file():
            current_hash = sha256_file(latest)
            if current_hash != last_persisted_hash:
                persist_fold_checkpoint(root, owner, fold, final=False)
                last_persisted_hash = current_hash
        time.sleep(30)
    return_code = process.wait()
    if return_code != 0:
        if latest.is_file():
            current_hash = sha256_file(latest)
            if current_hash != last_persisted_hash:
                persist_fold_checkpoint(root, owner, fold, final=False)
        raise RuntimeError(f"training failed ({return_code}) for fold {fold}")

    final_checkpoint = model_root(paths) / f"fold_{fold}" / CHECKPOINT
    if not final_checkpoint.is_file():
        raise RuntimeError(f"missing final checkpoint after fold {fold}")
    persist_fold_checkpoint(root, owner, fold, final=True)
    return fold


def train_next_folds(root: Path, owner: str, max_parallel: int = 2) -> list[int]:
    incomplete = [f for f in range(5) if not remote_fold_complete(owner, f)]
    if not incomplete:
        print("ALL_FOLDS_REMOTE_COMPLETE=True")
        return []
    selected = incomplete[:max_parallel]
    gpus = [str(i) for i in range(len(selected))]
    done: list[int] = []
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = {
            pool.submit(_run_training_with_checkpoint_persistence, root, owner, fold, gpu): fold
            for fold, gpu in zip(selected, gpus, strict=True)
        }
        for fut in as_completed(futures):
            done.append(fut.result())
    done.sort()
    print("SESSION_TRAINED_FOLDS=" + json.dumps(done))
    return done


def finalize_oof(root: Path, owner: str) -> dict[str, Any]:
    paths = base_paths(root)
    if not assemble_preprocessed_from_mounts(root, owner):
        raise RuntimeError("preprocessed shard inputs must be attached before finalization")
    for fold in range(5):
        state = restore_fold(root, owner, fold, allow_latest=False)
        if state != "final":
            raise RuntimeError(f"final checkpoint unavailable for fold {fold}")
    folds_path = paths["work"] / "v3_folds.json"
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    env_base = os.environ.copy()
    env_base["nnUNet_raw"] = str(paths["raw"])
    env_base["nnUNet_preprocessed"] = str(paths["preprocessed"])
    env_base["nnUNet_results"] = str(paths["results"])
    merged = paths["work"] / "oof_predictions"
    if merged.exists():
        shutil.rmtree(merged)
    merged.mkdir(parents=True)
    fold_by_case = {row["anonymous_case_id"]: int(row["validation_fold"]) for row in folds["assignments"]}
    for fold in range(5):
        input_dir = paths["work"] / "oof_inputs" / f"fold_{fold}"
        output_dir = paths["work"] / "oof_fold_predictions" / f"fold_{fold}"
        if input_dir.exists():
            shutil.rmtree(input_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        for case, assigned_fold in sorted(fold_by_case.items()):
            if assigned_fold != fold:
                continue
            src = paths["dataset"] / "imagesTr" / f"{case}_0000.nii"
            os.symlink(src.resolve(), input_dir / src.name)
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        run([
            "nnUNetv2_predict", "-i", str(input_dir), "-o", str(output_dir),
            "-d", str(DATASET_ID), "-p", PLANS, "-tr", TRAINER,
            "-c", CONFIGURATION, "-f", str(fold), "-chk", CHECKPOINT,
            "-device", "cuda",
        ], env=env)
        for pred in list(output_dir.glob("*.nii")) + list(output_dir.glob("*.nii.gz")):
            dst = merged / pred.name
            if dst.exists():
                raise RuntimeError(f"duplicate OOF prediction {pred.name}")
            shutil.copy2(pred, dst)
    stems = {p.name.removesuffix(".nii.gz").removesuffix(".nii") for p in merged.iterdir() if p.is_file()}
    expected = {f"case_{i:03d}" for i in range(EXPECTED_CASES)}
    if stems != expected:
        raise RuntimeError(f"OOF prediction mismatch missing={sorted(expected-stems)} extra={sorted(stems-expected)}")
    evaluator = Path(__file__).resolve().parent / "evaluate_v3_oof.py"
    report = paths["work"] / "v3_oof_report.json"
    case_csv = paths["work"] / "v3_oof_cases.csv"
    run([
        sys.executable, str(evaluator), "--labels", str(paths["dataset"] / "labelsTr"),
        "--predictions", str(merged), "--folds", str(folds_path),
        "--output", str(report), "--case-csv", str(case_csv),
    ])
    result = json.loads(report.read_text(encoding="utf-8"))
    print(f"FINAL_V3A_PASS={result['final_pass']}")
    print(f"OOF_MEAN_TUMOR_DICE={result['oof_mean_tumor_dice']:.6f}")
    print(f"OOF_MEAN_LIVER_REGION_DICE={result['oof_mean_liver_region_dice']:.6f}")
    print(f"FINAL_REPORT={report}")
    return result

def auto(root: Path, owner: str) -> int:
    os.environ["nnUNet_compile"] = "false"
    os.environ["nnUNet_n_proc_DA"] = "0"
    print("NNUNET_COMPILE=false", flush=True)
    print("NNUNET_N_PROC_DA=0", flush=True)
    ensure_nnunet()
    paths = base_paths(root)
    for p in (paths["raw"], paths["preprocessed"], paths["results"], paths["work"]):
        p.mkdir(parents=True, exist_ok=True)
    build_raw_dataset(root)
    plan_dataset(root)
    handles = preprocess_and_persist_shards(root, owner)
    if not assemble_preprocessed_from_mounts(root, owner):
        print("TURNKEY_STATUS=WAITING_FOR_SHARD_INPUT_ATTACH")
        print("After attaching the six private datasets above, rerun the SAME command.")
        return 0
    done = train_next_folds(root, owner, max_parallel=1)
    print("SESSION_TRAINING_MODE=single_fold_singlethreaded_da", flush=True)\n    remaining = [f for f in range(5) if not remote_fold_complete(owner, f)]
    if remaining:
        print("TURNKEY_STATUS=SESSION_COMPLETE_MORE_FOLDS_REMAIN")
        print("REMAINING_FOLDS=" + json.dumps(remaining))
        print("Start the next Kaggle GPU session and rerun the SAME command.")
        return 0
    print("TURNKEY_STATUS=ALL_FOLDS_COMPLETE")
    finalize_oof(root, owner)
    print("TURNKEY_STATUS=COMPLETE")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("auto", "prepare", "train", "finalize"), default="auto")
    p.add_argument("--root", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--owner", default=DEFAULT_OWNER)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["nnUNet_compile"] = "false"
    os.environ["nnUNet_n_proc_DA"] = "1"
    if args.mode == "auto":
        return auto(args.root, args.owner)
    ensure_nnunet()
    if args.mode == "prepare":
        build_raw_dataset(args.root)
        plan_dataset(args.root)
        preprocess_and_persist_shards(args.root, args.owner)
        assemble_preprocessed_from_mounts(args.root, args.owner)
        return 0
    if args.mode == "finalize":
        finalize_oof(args.root, args.owner)
        return 0
    if not assemble_preprocessed_from_mounts(args.root, args.owner):
        return 0
    train_next_folds(args.root, args.owner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())