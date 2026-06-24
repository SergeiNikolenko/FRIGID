# FRIGID Oracle Refinement Big ClearML Run

This package contains compact artifacts from a larger oracle-refinement training/evaluation run.

## Remote run

- Host: `spectrum`
- Repository: `/home/nikolenko/work/Projects/FRIGID`
- Run directory: `/home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_refinement_big_clearml_20260622T160222Z`
- ClearML task id: `1ec16fea256d4582ae8234d9d7e6c3ba`
- ClearML URL: `https://app.clearai.innopolis.university/projects/ff2caaabaca64025aca4de5cabd2e798/experiments/1ec16fea256d4582ae8234d9d7e6c3ba/output/log`

## Dataset

- Train candidate spectra: 5000
- Train candidates per spectrum: up to 10
- Train oracle traces: 47875
- Val candidate spectra: 1000
- Val candidates per spectrum: up to 10
- Val oracle traces: 8928

Train trace baseline:

- Mean candidate Tanimoto to true: 0.22024044232113243
- Mean target coverage: 0.46619505289506474
- Candidate formula match rate: 0.5974934725848564

Val trace baseline:

- Mean candidate Tanimoto to true: 0.1187786911563125
- Mean target coverage: 0.25875944707075954
- Candidate formula match rate: 0.1431451612903226

## Training

- Steps: 10000
- Batch size: 8
- Precision: bf16-mixed
- GPU: NVIDIA A100 80GB PCIe
- Elapsed training time: 1080.89 seconds
- Final checkpoint on remote: `training/checkpoints/oracle-refinement-step=10000.ckpt`

Checkpoints are not copied into this repository artifact package. Each checkpoint is about 2 GB.

## Evaluation

Validation evaluation, 256 val traces and 4 samples per trace:

- Predictions: 505
- Exact match rate: 0.0
- Mean candidate Tanimoto: 0.048135767916784776
- Mean refined Tanimoto: 0.03161456272739797

Train-slice diagnostic evaluation, 128 train traces and 2 samples per trace:

- Predictions: 256
- Exact match rate: 0.0234375
- Mean candidate Tanimoto: 0.38357837968528774
- Mean refined Tanimoto: 0.33773230350378786

## Interpretation

The training run is real and completed successfully, but the current refinement approach is not successful yet. It occasionally produces exact train matches, but on average the sampled refined molecules are worse than the input candidates on both train and validation slices.

The most likely issue is the gap between teacher-forced masked-SMILES training and free-running molecule sampling. The model learns the supervised token objective, but the sampler does not reliably preserve useful candidate structure or improve molecular similarity. The next fix should target constrained decoding and candidate-preservation rather than simply increasing the number of training steps.

## Files

- `train_oracle_traces.jsonl`: train repair traces
- `val_oracle_traces.jsonl`: validation repair traces
- `train_oracle_trace_stats.json`: aggregate train trace stats
- `val_oracle_trace_stats.json`: aggregate validation trace stats
- `training/smoke_training_metrics.json`: training metadata and ClearML task id
- `refinement_eval_val_step10000_limit256/refinement_predictions.csv`: validation refined outputs
- `refinement_eval_train_step10000_limit128/refinement_predictions.csv`: train-slice diagnostic refined outputs
- `train.log`: training log
- `eval.log`: validation eval log
