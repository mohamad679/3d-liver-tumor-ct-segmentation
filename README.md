# ProtoEM-CT

ProtoEM-CT studies robust transductive few-shot adaptation for 3D CT liver-tumor segmentation, with emphasis on leakage-safe protocol design, robustness evaluation, uncertainty-aware analysis, and external validation discipline.

Phase 8 is closed with an honest negative external-validation result on 3D-IRCADb-01. Phase 9 was skipped by user decision and was not executed. Phase 10 prepares the final scientific package from saved machine-readable artifacts only.

Key Phase 10 outputs:

- Final report: `reports/final_report.html` and `reports/final_report.md`
- Reproduction README: `docs/PHASE10_REPRODUCTION.md`
- Model card: `MODEL_CARD.md`
- Data card: `DATA_CARD.md`
- Release checklist: `docs/PHASE10_RELEASE_CHECKLIST.md`
- Release candidate metadata: `reports/phase10/release_candidate.json`

Gate 8 reproduction command:

```bash
uv run snakemake --cores 4 --rerun-incomplete reports/final_report.html
```

No raw medical data, DICOM/NIfTI arrays, predictions, checkpoints, model weights, credentials, or secrets are committed to this repository.
