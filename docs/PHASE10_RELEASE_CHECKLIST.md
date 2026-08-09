# Phase 10 Release Checklist

Status: release candidate prepared pending final validation and independent review.

## Scope

- [x] Phase 10 only.
- [x] Phase 9 skipped by user decision / not executed.
- [x] No LLM, VLM, or RAG work implemented.
- [x] Phase 10 is the final core-project phase.

## Artifact-Driven Reporting

- [x] P10-A inventory generated from saved machine-readable artifacts.
- [x] P10-B tables generated from P10-A inventory and saved Phase 8 comparison artifact.
- [x] P10-B figures generated from machine-readable table/inventory values.
- [x] P10-B statistics generated from preregistered saved Phase 8 policy/CI artifacts.
- [x] P10-C final report generated from P10-A/P10-B machine-readable outputs.
- [x] Report manifest records numeric statement provenance.
- [x] Unsupported Phase 3-7 numeric results are marked unavailable, not fabricated.

## Scientific Boundary

- [x] No model training rerun in Phase 10.
- [x] No inference rerun in Phase 10.
- [x] No prediction regeneration in Phase 10.
- [x] No checkpoint reselection.
- [x] No threshold, preprocessing, or support-policy change.
- [x] No external-label tuning.
- [x] LiTS/MSD Task03 Liver treated as one LiTS-derived development source.
- [x] External validation remains an honest negative result.

## Release Hygiene

- [x] No raw datasets committed.
- [x] No DICOM/NIfTI medical arrays committed.
- [x] No checkpoints or model weights committed.
- [x] No predictions committed.
- [x] No credentials or secrets committed.
- [x] Release candidate metadata identifies report, inventory, cards, application materials, and checklist.

## Required Final Commands

These must pass before Phase 10 closure:

- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run mypy src tests`
- [ ] `uv run pytest -q`
- [ ] `make smoke`
- [ ] `uv run snakemake --cores 4 --rerun-incomplete reports/final_report.html`

## Gate 8

- [ ] Gate 8 PASS: final report generated entirely from machine-readable artifacts.

## Final Review

- [ ] P10-R independent review PASS.
- [ ] Working tree clean after final commits.
