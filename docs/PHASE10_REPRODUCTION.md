# Phase 10 Reproduction README

This document describes artifact-only reproduction of the final scientific package.

## Environment

- Python 3.11
- `uv`
- Snakemake
- Repository branch: `phase/10-final-scientific-package`

## Data and Artifact Expectations

Raw medical datasets, DICOM/NIfTI arrays, predictions, checkpoints, model weights, credentials, and secrets are not committed to Git.

The default local Phase 10 inventory command reads saved non-medical JSON/provenance artifacts from the configured run-artifact root. On this workstation the recovered inventory used:

- Inventory hash: `67da0cadfd3d8127f3bfac55f8c3984bd46e6888aa3d6720ad23e8dcf252dea9`
- P10-B manifest hash: `bce7894adbb3d059e6ee6367e7f90f49a30f32652e220d3aeb5b6298f2408035`

Use `--drive-root` on `workflow/scripts/build_phase10_artifact_inventory.py` if your saved artifact root differs.

## Rebuild Commands

```bash
uv run python workflow/scripts/build_phase10_artifact_inventory.py --repo-root "$PWD" --drive-root /path/to/ProtoEM-CT/runs --output-json reports/phase10/artifact_inventory.json --output-markdown reports/phase10/ARTIFACT_INVENTORY.md
uv run python workflow/scripts/build_phase10_scientific_outputs.py --inventory reports/phase10/artifact_inventory.json --reports-root reports
uv run python workflow/scripts/build_phase10_final_report.py --inventory reports/phase10/artifact_inventory.json --b-manifest reports/phase10/phase10_b_outputs_manifest.json --report-md reports/final_report.md --report-html reports/final_report.html --report-manifest reports/phase10/final_report_manifest.json --reproduction-readme docs/PHASE10_REPRODUCTION.md
```

Gate 8 uses the repository DAG:

```bash
uv run snakemake --cores 4 --rerun-incomplete reports/final_report.html
```

The Phase 10 DAG path must not trigger training, inference, external-data experiments, downloads, or prediction regeneration.

## Outputs

- Final Markdown report: `reports/final_report.md`
- Final HTML report: `reports/final_report.html`
- Report manifest: `reports/phase10/final_report_manifest.json`
- Tables: `reports/tables/phase10_*.json` and `reports/tables/phase10_*.csv`
- Figures: `reports/figures/phase10_*.json` and `reports/figures/phase10_*.svg`
- Statistics: `reports/statistics/phase10_*.json`

## Scope

Phase 9 was skipped by user decision and is not executed. Phase 10 is the final core-project reporting and packaging phase.
