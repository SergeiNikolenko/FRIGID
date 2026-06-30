# FRIGID DreaMS Adapter Replacement Run Report

Date: 2026-06-30

## Summary

This experiment is prepared as the next serious DreaMS replacement attempt. It
trains a DreaMS residual adapter and fingerprint head directly from spectra,
keeps the encoder frozen during warmup, then unfreezes only upper DreaMS blocks
with a small learning rate. The run is intended for Slurm on `spectrum` and is
queued behind the active DLM adaptation job to avoid A100 contention.

## Run Identity

- Host: `spectrum`.
- Scheduler: Slurm.
- Partition: `gpu`.
- GRES: `gpu:a100:1`.
- Local branch at preparation: `dreams-fingerprint-head`.
- Remote checkout:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head`.
- DreaMS checkout:
  `/home/nikolenko/work/Projects/mist_molforge_autoresearch`.
- Command file:
  `docs/FRIGID_dreams_adapter_replacement_20260630/run_dreams_adapter_replacement.sbatch`.
- Expected run directory:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_adapter_replacement_<UTC>`.
- ClearML project: `FRIGID/DreaMS Replacement`.

## Inputs

- Train rows: 191,216 spectra.
- Validation rows: 19,043 spectra.
- Fingerprint target: 4096-bit Morgan fingerprint, key `ground_truth`.
- MIST train probabilities:
  `runs/mist_dreams_residual_20260629T135651Z/mist_train/fingerprints.npz`.
- MIST validation probabilities:
  `runs/mist_dreams_blend_20260629T131733Z/mist_val/fingerprints.npz`.
- DreaMS HDF5 spectra:
  `runs/dreams_fingerprint_head_msg_full_20260629T110040Z`.

The training script aligns MIST arrays by `spec_name` against the DreaMS
metadata before training and validation.

## What Happened

- Implemented `scripts/train_dreams_adapter_replacement.py`.
- Added a Slurm launcher that uses the DreaMS/MolForge Python environment
  rather than the FRIGID `.venv`.
- Added full-validation threshold/top-k sweeps every epoch.
- Added MIST baseline recomputation inside the same run.
- Added checkpoint, prediction, and summary artifact writing.
- Added optional ClearML logging and model upload.
- Added optional decoder-sensitive case weighting hook for a later downstream
  robustness-derived CSV.

## Metrics

Metrics are pending until the Slurm job starts and reaches validation.

| Metric | MIST Baseline | This Run | Delta |
| --- | ---: | ---: | ---: |
| best validation mean fingerprint Tanimoto | pending recompute | pending | pending |
| median validation fingerprint Tanimoto | pending recompute | pending | pending |
| bit F1 | pending recompute | pending | pending |
| mean predicted active bits | pending recompute | pending | pending |

## Artifacts

Expected remote artifacts:

- `train.log`;
- `run_identity.json`;
- `model/run_config.json`;
- `model/mist_baseline_metrics.json`;
- `model/mist_baseline_metrics.csv`;
- `model/training_metrics.json`;
- `model/summary.json`;
- `model/val_sweep_metrics.json`;
- `model/val_sweep_metrics.csv`;
- `model/val_predictions.npz`;
- `model/val_prediction_metadata.csv`;
- `model/best_model.pt`;
- `model/last_model.pt`.

Expected Slurm logs:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/slurm_logs/frigid-dreams-adapter-<job_id>.out
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/slurm_logs/frigid-dreams-adapter-<job_id>.err
```

## Interpretation

No result yet. This run is designed to answer whether DreaMS can be adapted
without repeating the failure mode of the previous full fine-tune: train-set
overfit while validation remains far below MIST.

## Decision

Pending. Promotion requires beating the recomputed MIST validation fingerprint
baseline by at least `+0.005` mean Tanimoto and then improving downstream DLM
robustness versus `mist_binary`.

## Next Actions

1. Push the code and docs to the fork branch.
2. Pull the branch on `spectrum`.
3. Submit the Slurm job with dependency on DLM job `32`.
4. Record Slurm job id and first validation metrics here.
