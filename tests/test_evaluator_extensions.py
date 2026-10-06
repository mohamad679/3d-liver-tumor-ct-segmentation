import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research_v3/evaluate_v3_oof.py"
spec = importlib.util.spec_from_file_location("evalv3", SCRIPT)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_metric_primitives_still_work():
    import numpy as np
    a = np.array([1, 1, 0, 0], dtype=bool)
    b = np.array([1, 0, 1, 0], dtype=bool)
    assert m.dice(a, b) == 0.5
    assert m.iou(a, b) == 1 / 3