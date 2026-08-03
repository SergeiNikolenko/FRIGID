# FRIGID Oracle Refinement E2E Smoke

Remote run directory:

`/home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_refinement_e2e_20260622T124305Z`

This repository artifact package excludes `.ckpt` files to avoid copying a multi-GB checkpoint bundle locally. The full checkpoints remain on Spectrum:

- `/home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_refinement_e2e_20260622T124305Z/smoke_training/checkpoints/last.ckpt`
- `/home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_refinement_e2e_20260622T124305Z/smoke_training/checkpoints/oracle-refinement-smoke-step=2.ckpt`

Included artifacts:

- `train_candidate_predictions.csv`: split-safe train candidate table built from MSG labels.
- `train_oracle_traces.jsonl`: train oracle refinement traces, 536 rows.
- `train_oracle_trace_stats.json`: train trace statistics.
- `val_candidate_predictions.csv`: split-safe validation candidate table.
- `val_oracle_traces.jsonl`: validation oracle refinement traces, 160 rows.
- `val_oracle_trace_stats.json`: validation trace statistics.
- `smoke_training/smoke_training_metrics.json`: DLM fine-tune smoke metadata.
- `smoke_training/train.log`: Lightning training log.
- `refinement_eval_val_smoke/refinement_predictions.csv`: generated refinement predictions for 4 validation traces x 2 samples.
- `refinement_eval_val_smoke/refinement_summary.json`: smoke refinement metrics.
- `oracle_refinement_e2e_summary.json`: combined machine-readable summary.

Important scope note:

This is an end-to-end smoke slice, not a full oracle-refinement training run. It verifies split-safe trace construction, DLM forward/backward/optimizer steps, checkpoint writing, sampler loading, and refinement generation. It intentionally does not use test split traces for training.
