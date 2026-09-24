# Research V2 Linux/CUDA environment

This environment is separate from the historical Intel/macOS CPU baseline environment.

The package versions in `pyproject.toml` are exact pins. The Research-v2 Linux/CUDA `uv.lock` is
tracked in this directory and is the reproducibility source for later runs. Normal validation must
use the committed lock; CI must not silently resolve a fresh dependency graph on every run.

CPU/Linux validation commands:

```bash
uv lock --check
uv sync --frozen --python 3.11 --all-groups
uv run --frozen --python 3.11 python -c \
  "import torch, monai, nnunetv2, nibabel; print(torch.__version__, monai.__version__)"
```

GPU training-readiness requires a Linux x86_64 host with a CUDA-capable NVIDIA GPU and compatible
driver. It is a separate gate from R0 data/software readiness. It does not block CPU-only R1 after
the real-data R0 audit passes, but it is mandatory before real R2/R3 training.

GPU validation commands include:

```bash
uv run --frozen --python 3.11 python -c \
  "import torch; assert torch.cuda.is_available(); x=torch.ones(1, device='cuda'); print(torch.cuda.get_device_name(0), x)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

Persist Python, Torch, MONAI, nnU-Net, CUDA runtime, driver, GPU model, VRAM and command logs in the
corresponding evidence artifact and gate record.
