# V3 turnkey Kaggle execution

The intended user interaction is deliberately minimal. The V3-A protocol remains fixed: nnU-Net v2.8.1, `3d_fullres`, `nnUNetTrainer_100epochs`, five deterministic folds, no architecture or threshold sweep.

## First run

Attach these two public source datasets to the Kaggle notebook:

- `andrewmvd/liver-tumor-segmentation`
- `andrewmvd/liver-tumor-segmentation-part-2`

Then run the same command for every Kaggle session:

```bash
python scripts/research_v3/v3_turnkey_kaggle.py --mode auto --owner mohamadasgari
```

The runner performs source discovery, recomputes and verifies the locked semantic source hash, applies the preregistered header-only repairs for cases 48-52 with a voxel-value hash check, builds Dataset903, installs the deterministic five-fold split, fingerprints/plans nnU-Net, preprocesses `3d_fullres` into six balanced private shards, persists and verifies those shards, trains incomplete folds, persists checkpoints, restores prior progress, performs OOF inference, and writes the final report.

## Only manual Kaggle handoff

After preprocessing, the runner prints six private preprocessing dataset handles and exits with `TURNKEY_STATUS=WAITING_FOR_SHARD_INPUT_ATTACH`. Attach those six datasets once as notebook inputs, then rerun the exact same command.

## Session recovery

The pinned nnU-Net 2.8.1 trainer writes `checkpoint_latest.pth` every 50 epochs. The runner watches for new latest checkpoints while training and versions them into a fold-specific private Kaggle dataset. On a later session it restores the latest checkpoint and invokes nnU-Net with `--c`; completed folds are represented by a separately named final archive and are skipped. Fold dataset handles include the source-lock prefix to prevent reuse of artifacts from a different source.

If a session ends before the first periodic checkpoint is written, that fold must restart; no training hyperparameter is changed to alter nnU-Net's checkpoint cadence.

## Locked data repair

Only cases 48, 49, 50, 51 and 52 receive a header-only repair. Their voxel arrays are not resampled or interpolated. The effective label geometry is copied from the corresponding CT, and before/after voxel hashes must match.

## Final outputs

After all five folds are remote-complete, the runner restores their final checkpoints, predicts each case only with its assigned fold, verifies exact 131-case OOF coverage, and writes `v3_oof_report.json` plus the case-level CSV. The scientific acceptance criteria remain OOF mean tumor Dice >= 0.70 and OOF mean liver-region Dice >= 0.90.