#!/usr/bin/env python3
"""Launch the R2 Kaggle preflight with a deterministic headless Matplotlib backend."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> None:
    # Kaggle notebooks export MPLBACKEND=module://matplotlib_inline.backend_inline.
    # The frozen R2 CUDA environment intentionally does not depend on matplotlib-inline,
    # so force a non-interactive backend before MONAI imports Matplotlib.
    os.environ["MPLBACKEND"] = "Agg"
    target = Path(__file__).with_name("run_r2_kaggle_preflight.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
