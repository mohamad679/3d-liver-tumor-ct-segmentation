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

- Inventory hash: `f06337728c6a39144c7e90bf4c59044c46d75eaae6a570249d4303cb14124da0`
- P10-B manifest hash: `f881180c8d6c760adcc2360860b297e329f682baacfcd564da92fe95224d1951`

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
