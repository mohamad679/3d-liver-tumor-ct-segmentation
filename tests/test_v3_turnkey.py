import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research_v3/v3_turnkey_kaggle.py"
spec = importlib.util.spec_from_file_location("turnkey", SCRIPT)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_dataset_constants_locked():
    assert m.DATASET_ID == 903
    assert m.EXPECTED_CASES == 131
    assert m.TRAINER == "nnUNetTrainer_100epochs"
    assert m.CONFIGURATION == "3d_fullres"
    assert m.REPAIR_IDS == {48, 49, 50, 51, 52}


def test_fold_partition_exact_and_balanced():
    case_ids = [f"case_{i:03d}" for i in range(131)]
    artifact, splits = m.build_splits(case_ids)
    assert artifact["fold_case_counts"] == {"0": 27, "1": 26, "2": 26, "3": 26, "4": 26}
    seen = set()
    for split in splits:
        train, val = set(split["train"]), set(split["val"])
        assert not train & val
        assert train | val == set(case_ids)
        assert not seen & val
        seen |= val
    assert seen == set(case_ids)


def test_fold_generation_deterministic():
    cases = [f"case_{i:03d}" for i in range(131)]
    assert m.build_splits(cases) == m.build_splits(cases)


def test_balanced_shards_cover_once():
    case_work = [
        {"case": f"case_{i:03d}", "approx_target_voxels": float((i % 17) + 1)}
        for i in range(131)
    ]
    shards = m.build_balanced_shards(case_work, 6)
    all_cases = [c for s in shards for c in s["cases"]]
    assert len(all_cases) == len(set(all_cases)) == 131
    assert set(all_cases) == {f"case_{i:03d}" for i in range(131)}
    assert len(shards) == 6


def test_remote_names_are_stable():
    handles = m.shard_handles("alice")
    assert len(handles) == 6
    assert handles[0].startswith("alice/protoem-v3-pp903-")
    assert handles[-1].endswith("-s05")
    assert m.fold_remote_handle("alice", 4) == "alice/protoem-v3a-nnunet100-4baf8ff8-fold4"