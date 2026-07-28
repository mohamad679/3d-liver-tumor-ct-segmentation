# Phase 3 Intel macOS CPU Environment

This directory defines the isolated Phase 3 baseline dependency environment for Intel macOS CPU
work on ProtoEM-CT. It is a separate dependency project so the core repository environment can stay
on its existing dependency contract while Phase 3 baseline work uses the pinned stack required for
`nnU-Net v2` and `MONAI`.

## Supported Host

- Python 3.11
- Darwin `x86_64`

This host restriction is documented here. No unsupported `uv` project-level platform key is added
in this environment definition.

## Exact Direct Pins

- local editable root project: `protoem-ct`
- `numpy==1.26.4`
- `torch==2.2.2`
- `torchvision==0.17.2`
- `monai==1.4.0`
- `nnunetv2[intel_macos]==2.8.1`
- `mlflow==3.14.0`
- `simpleitk==2.5.5`
- `scikit-image==0.26.0`
- `scipy==1.17.1`
- `nibabel==5.4.2`

## Why This Environment Is Isolated

The root project environment remains the Phase 0-2 core environment and retains its current NumPy
2.x contract. This Phase 3 environment is isolated because the Intel macOS `nnunetv2[intel_macos]`
constraints require `numpy<2` and `torch<2.3`, while the core environment already locks NumPy 2.x.

This isolation keeps baseline dependency churn out of the core environment, its lockfile, and its
existing test surface.

## Installation Status

No installation, sync, import verification, or runtime validation has occurred yet in this step.

`acvl-utils` is locked transitively by this environment, and a later locked sync may require a
local source build for that package on this host.

## Data, Artifacts, and Git Hygiene

- Environment-local `.venv` directories must not be committed.
- Package caches must not be committed.
- MLflow runs must not be committed.
- Checkpoints and predictions must not be committed.
- Data paths must be supplied later through environment variables or untracked local configuration.

## Future Environments

Future Linux and CUDA variants will use sibling environment directories while reusing the same
scientific code.

## Future Commands

Future locked sync command, do not execute in this step:

```bash
uv sync --project environments/phase3-baselines/intel-macos-cpu --locked
```

Future command pattern for running Python through this environment, do not execute imports in this
step:

```bash
uv run --project environments/phase3-baselines/intel-macos-cpu --locked python -c "<command>"
```
