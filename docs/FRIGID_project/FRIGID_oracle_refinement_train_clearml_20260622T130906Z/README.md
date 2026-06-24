# FRIGID Oracle Refinement ClearML Run

This package contains compact artifacts from the ClearML-connected oracle-refinement training run.

## Remote run

- Host: `spectrum`
- Repository: `/home/nikolenko/work/Projects/FRIGID`
- Run directory: `/home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_refinement_train_clearml_20260622T130906Z`
- Base oracle-trace run: `/home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_refinement_e2e_20260622T124305Z`
- ClearML task id: `64a2505e2c1e4309aa73fbff06206221`
- ClearML URL: `https://app.clearai.innopolis.university/projects/ff2caaabaca64025aca4de5cabd2e798/experiments/64a2505e2c1e4309aa73fbff06206221/output/log`

## Training

- Steps: 2000
- Batch size: 4
- Precision: bf16-mixed
- GPU: NVIDIA A100 80GB PCIe
- Elapsed time: 548.13 seconds
- Final checkpoint on remote: `checkpoints/oracle-refinement-step=2000.ckpt`
- Last checkpoint on remote: `checkpoints/last.ckpt`

Checkpoints are not copied into this repository artifact package because each checkpoint is about 2 GB.

## Validation refinement eval

Evaluation directory: `refinement_eval_val_step2000`

- Rows: 32
- Predictions: 94
- Exact match rate: 0.0
- Mean candidate Tanimoto: 0.04144775717426871
- Mean refined Tanimoto: 0.05681043690564163

The refinement checkpoint improved average Tanimoto on this small validation slice, but the absolute quality is still poor and exact matches are zero. This is not a publication-scale result; it is a real ClearML-connected fine-tune/eval pass on a small oracle trace set.

## Files

- `train.log`: full training log
- `smoke_training_metrics.json`: training metadata and ClearML task id
- `refinement_eval_val_step2000/refinement_summary.json`: aggregate eval metrics
- `refinement_eval_val_step2000/refinement_predictions.csv`: per-sample true, candidate, refined SMILES and Tanimoto values
- `refinement_eval_val_step2000/eval.log`: eval log
