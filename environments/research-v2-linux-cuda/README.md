# Research V2 Linux/CUDA environment

This environment is separate from the historical Intel/macOS CPU baseline environment.

Required validation host:

- Linux x86_64;
- Python 3.11;
- NVIDIA CUDA-capable GPU with a driver compatible with the selected PyTorch wheel;
- sufficient RAM/VRAM for the planned 3D workload.

The package versions in `pyproject.toml` are exact pins. `uv.lock` must be generated and checked on a
networked Linux host before this environment can satisfy the R0 reproducibility gate. Do not copy the
Intel/macOS lockfile into this directory.

Validation commands after the lock exists:

```bash
uv sync --frozen --python 3.11 --all-groups
uv run python -c "import torch, monai, nnunetv2, nibabel; print(torch.cuda.is_available())"
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Record Python, Torch, MONAI, nnU-Net, CUDA runtime, driver, GPU name, RAM, VRAM and wall time in
`reports/research_v2/gates/gate_R0.json`.
