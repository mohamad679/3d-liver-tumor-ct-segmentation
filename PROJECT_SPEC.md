# Project Specification

## Project

ProtoEM-CT: Robust Transductive Few-Shot Adaptation for 3D CT Tumor Segmentation.

## Scientific Scope

The primary task is binary liver-tumor segmentation in 3D CT. The target label space is foreground tumor versus background. Liver context may be used for preprocessing, masking, quality control, or analysis, but the primary segmentation objective is tumor delineation.

The project studies robust transductive few-shot adaptation for CT tumor segmentation. The core scientific question is whether support/query adaptation can improve tumor segmentation under constrained labels, domain shift, robustness stressors, and uncertainty-aware evaluation without leaking information from held-out external labels.

## Cohorts

- Development cohort: LiTS labeled training cohort, optionally represented in MSD-style layout.
- External cohort: 3D-IRCADb-01.

LiTS and MSD Task03 Liver must not be treated as independent cohorts. MSD-style organization can be used as a layout convention only when it represents the same underlying LiTS cases.

All train, validation, test, support, and query selections must be patient-disjoint. Splits must be explicit, reproducible, and auditable.

External validation remains untouched until Phase 8. External labels must never be used for tuning, model selection, threshold selection, early stopping, prompt construction, hyperparameter decisions, or protocol iteration before the external validation phase.

## Engineering Stack

- Python 3.11
- uv
- PyTorch
- MONAI
- nnU-Net v2 wrapper
- Hydra/OmegaConf
- Snakemake
- MLflow
- Typer
- SimpleITK
- nibabel
- pytest
- hypothesis
- ruff
- mypy
- pre-commit
- GitHub Actions
- Docker/Dev Container

## LLM/VLM Track Constraint

The LLM/VLM track cannot begin before baseline modeling, few-shot protocol, robustness evaluation, uncertainty analysis, and external validation are complete. It must not influence Phase 0 through Phase 8 model selection, metric reporting, or external validation procedures.
