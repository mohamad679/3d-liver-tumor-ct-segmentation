# ProtoEM-CT

Reproducible 3D CT liver-tumor segmentation with leakage-controlled development and locked external validation.

## Overview

Liver-tumor segmentation from 3D CT is a challenging medical-imaging problem, especially when labels
are limited and evaluation must account for cross-dataset generalization. Small cohorts, heterogeneous
acquisition protocols, and flexible experimental choices can make reported performance difficult to
interpret unless data separation and external validation are handled rigorously.

ProtoEM-CT implements an end-to-end reproducible framework for developing and evaluating 3D
segmentation models under strict data-separation and external-validation controls. The repository
emphasizes reliable experimental evidence, recorded provenance, and locked evaluation decisions
rather than post-hoc optimization on the external cohort.

## Key Contributions

- End-to-end 3D CT preprocessing, training, inference, and evaluation pipeline.
- Patient-level train, validation, internal-test, and external-validation separation.
- Leakage checks using case, patient, image, and label provenance.
- PyTorch and MONAI implementation for volumetric segmentation.
- Deterministic configuration and recorded preprocessing provenance.
- Controlled checkpoint and model-selection procedure.
- Frozen preprocessing, checkpoint, support policy, and decision threshold before external
  evaluation.
- Preregistered external-validation protocol.
- External prediction locking before external-label evaluation.
- Case-level bootstrap confidence intervals.
- Few-shot protocol and infrastructure support without claiming demonstrated few-shot performance
  gains.
- Machine-readable artifact-driven final reporting.
- Automated testing, static typing, linting, CI, and Snakemake-based reproducibility.

## Research Design

```text
LiTS / MSD Task03 Liver development source
  -> patient-level split
  -> preprocessing, training, and model selection
  -> frozen decision package
  -> external image-only inference on 3D-IRCADb-01
  -> prediction lock
  -> external label evaluation
  -> artifact-driven final reporting
```

## Data

The development source is LiTS / MSD Task03 Liver, treated as one LiTS-derived development source
rather than two independent cohorts. External validation uses 3D-IRCADb-01.

No medical images, masks, predictions, model weights, checkpoints, credentials, or private dataset
paths are committed to Git.

## Technical Stack

- Python
- PyTorch
- MONAI
- Snakemake
- pytest
- mypy
- Ruff
- Git / CI

## Results

| Evaluation | Tumor Dice | Cases |
| --- | ---: | ---: |
| Development validation | 0.01579295321113191 | 20 |
| External validation | 0.01412, 95% CI [0.00434, 0.02501] | 15 eligible |

The frozen configuration did not achieve the level of tumor-segmentation accuracy required for a
strong generalization claim. Because preprocessing, checkpoint selection, threshold, and the
external-evaluation protocol were locked before external labels were evaluated, the external result
represents an untuned assessment of the selected configuration.

This repository is a research artifact and is not intended for clinical deployment.

## Evaluation

The external evaluation includes Dice, IoU, HD95, normalized surface Dice, lesion-level recall,
lesion-level precision, lesion-level F1, false-positive lesions per scan, tumor-volume error, and
bootstrap 95% confidence intervals.

See `reports/final_report.html` or `reports/final_report.md` for the complete metric table and
artifact provenance.

## Reproducibility

Final reports, tables, figures, and statistics are generated from persisted machine-readable
artifacts.

```bash
uv run snakemake --cores 4 --rerun-incomplete reports/final_report.html
```

Final validation currently passes Ruff, mypy, pytest, and reproducible report generation.

## Repository Structure

```text
src/       implementation for data handling, modeling, evaluation, and reporting
tests/     unit, integration, smoke, leakage, and reproducibility tests
workflow/  Snakemake workflow and report-generation scripts
reports/   final report outputs and machine-readable reporting artifacts
docs/      scientific decisions, data documentation, and release materials
```

## Limitations

The frozen model did not achieve strong segmentation accuracy. The external sample size is limited.
No clinical-use claim is made. Some planned analyses lack durable result artifacts and therefore are
not claimed.

## Status

Research pipeline complete.
External validation complete.
Final reproducibility package complete.
