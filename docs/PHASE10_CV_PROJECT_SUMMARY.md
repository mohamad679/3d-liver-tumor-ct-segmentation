# ProtoEM-CT — One-Page CV Project Summary

## Project Title

ProtoEM-CT: Artifact-Driven Final Scientific Package for Leakage-Controlled 3D CT Tumor Segmentation Research

## Research Question

Can a leakage-controlled 3D CT tumor-segmentation workflow be packaged so that final scientific claims, including negative external validation, are reproducible from machine-readable artifacts rather than edited prose or rerun experiments?

## Methods

- Built a phased CT tumor-segmentation research workflow with patient-level development splits and external validation against 3D-IRCADb-01.
- Treated LiTS/MSD Task03 Liver as one LiTS-derived development source, not independent cohorts.
- Preserved preregistration, prediction lock, and no-external-label-tuning boundaries.
- Generated final Phase 10 tables, figures, statistics, and report from saved artifact inventories and reporting manifests.

## Engineering Stack

Python 3.11, uv, Snakemake, pytest, ruff, mypy, JSON-first provenance artifacts, deterministic report rendering, and Git-based phase tracking.

## Reproducibility Contribution

Created a final package in which the HTML technical report, tables, figures, statistical summaries, model card, data card, release checklist, and release-candidate metadata are traceable to machine-readable artifacts.

## Robustness And External Validation Work

The final package preserves the Phase 8 external-validation result as an honest negative result. External macro tumor Dice was `0.0141157` with 95% CI `[0.00434155, 0.025013]`, and the report explicitly avoids claims of clinical readiness or strong external generalization.

## Skills Demonstrated

Scientific governance, leakage-aware evaluation design, artifact provenance, reproducible reporting, test-driven Python tooling, Snakemake workflow integration, statistical-result packaging, and transparent negative-result communication.
