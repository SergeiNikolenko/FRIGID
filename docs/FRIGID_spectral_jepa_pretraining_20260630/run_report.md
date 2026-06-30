# FRIGID Spectral JEPA Pretraining Run Report

Date: 2026-06-30

## Summary

This experiment package implements a JEPA-like self-supervised MS/MS spectrum
encoder and a frozen-encoder fingerprint-head evaluation gate.

The run is implemented, pushed, synced to `spectrum`, and queued in Slurm behind
the active DLM and DreaMS GPU jobs.

## Run Identity

- Host: `spectrum`.
- Scheduler: Slurm.
- Partition: `gpu`.
- GRES: `gpu:a100:1`.
- Local branch at preparation: `dreams-fingerprint-head`.
- Status: `queued`.
- Slurm job id: `36`.
- Slurm state after submission: `PENDING`.
- Slurm reason after submission: `Dependency`.
- Slurm dependency: `afterany:34`.
- Submit commit: `8fbfe30`.
- Latest verified status: 2026-06-30 18:20 UTC, still `PENDING` on the same
  dependency.
- Remote checkout:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head`.
- Command file:
  `docs/FRIGID_spectral_jepa_pretraining_20260630/run_spectral_jepa.sbatch`.
- Expected run directory:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/spectral_jepa_pretraining_<UTC>`.
- Slurm logs:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/slurm_logs/frigid-spectral-jepa-36.out`
  and
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/slurm_logs/frigid-spectral-jepa-36.err`.

## What Was Implemented

- Added `scripts/train_spectral_jepa.py`.
- Added JEPA-style context/target masking over peak tokens.
- Added context encoder plus EMA target encoder.
- Added predictor head for latent target prediction.
- Added variance and covariance regularization to reduce representation
  collapse risk.
- Added frozen-encoder fingerprint-head training.
- Added validation threshold/top-k sweep and prediction export.
- Added Slurm launcher, plan, report, and artifact manifest.
- Verified `scripts/train_spectral_jepa.py --help` in the remote DreaMS/MolForge
  Python environment.
- Submitted Slurm job `36` with dependency `afterany:34`, so it does not
  compete with DLM job `32` or DreaMS adapter job `34` for the A100.

## Metrics

Metrics are pending until Slurm job `36` starts and runs.

| Metric | Value |
| --- | ---: |
| best validation mean fingerprint Tanimoto | pending |
| best validation median fingerprint Tanimoto | pending |
| bit F1 | pending |
| mean predicted active bits | pending |
| pretraining final latent loss | pending |

## Interpretation

Pending. The expected first decision is whether JEPA pretraining produces a
frozen spectrum representation that beats earlier DreaMS frozen-head and
full-fine-tune attempts on the same validation fingerprint gate.

## Next Actions

1. Wait for active DLM job `32` and queued DreaMS adapter job `34` to clear.
2. Confirm Slurm job `36` starts and creates `run_identity.json`.
3. Record first metrics and failure modes here.
4. Compare the frozen JEPA encoder against:
   - frozen DreaMS head (`0.1238`);
   - DreaMS distilled (`0.2404`);
   - full DreaMS fine-tune (`0.2580`);
   - MIST baseline (`~0.5420`).
