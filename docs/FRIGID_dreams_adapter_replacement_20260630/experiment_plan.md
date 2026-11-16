# FRIGID DreaMS Adapter Replacement Experiment

Date: 2026-06-30

## Question

Can a staged DreaMS adapter, trained directly from spectra with MIST anchoring
and MIST-error weighting, become a credible replacement candidate for the MIST
spectrum-to-fingerprint encoder?

## Motivation

Previous DreaMS attempts did not clear the MIST replacement gate:

- frozen DreaMS embedding heads were far below MIST;
- calibration and loss variants improved the frozen-head result but still
  failed the gate;
- DreaMS-only distillation remained far below MIST;
- plain full encoder fine-tuning overfit and peaked at about 0.258 validation
  mean fingerprint Tanimoto;
- linear MIST+DreaMS blending selected pure MIST;
- anchored MIST+DreaMS residual learning found only a tiny positive delta.

This experiment is the next serious DreaMS path because it changes both the
training graph and the objective. It does not train a shallow frozen head, and
it does not unfreeze the entire encoder from step zero. It warms up a trainable
adapter and fingerprint head, then unfreezes only upper DreaMS encoder blocks
with a small learning rate.

## Inputs

- Remote host: `spectrum`.
- Scheduler: Slurm, partition `gpu`, GRES `gpu:a100:1`.
- FRIGID checkout:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head`.
- DreaMS/MolForge checkout:
  `/home/nikolenko/work/Projects/mist_molforge_autoresearch`.
- Python:
  `/home/nikolenko/work/Projects/mist_molforge_autoresearch/.venv/bin/python`.
- Base DreaMS dataset:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_full_20260629T110040Z`.
- Train spectra:
  `train_dataset/spectra.hdf5`.
- Validation spectra:
  `val_dataset/spectra.hdf5`.
- Train/validation targets:
  `fingerprints.npz`, key `ground_truth`.
- Train MIST probabilities:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_residual_20260629T135651Z/mist_train/fingerprints.npz`,
  key `probs`.
- Validation MIST probabilities:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_blend_20260629T131733Z/mist_val/fingerprints.npz`,
  key `probs`.
- Baseline: best MIST validation fingerprint sweep from the same validation
  split, recomputed inside the run.

## Method

Train `scripts/train_dreams_adapter_replacement.py`:

```text
MS/MS spectrum
  -> pretrained DreaMS encoder
  -> residual adapter
  -> 4096-bit Morgan fingerprint head
```

Training stages:

1. Warm up only the adapter and fingerprint head for three epochs.
2. Keep the frozen DreaMS encoder in eval mode during warmup.
3. After warmup, unfreeze only the top DreaMS transformer blocks selected by
   regex:

   ```text
   (^head\.|backbone\.transformer_encoder\.(atts|ffs)\.[56]\.|backbone\.transformer_encoder\.scales\.(9|1[0-4])\.)
   ```

4. Use separate AdamW learning rates:
   - encoder: `2e-5`;
   - adapter: `3e-4`;
   - fingerprint head: `1e-3`.
5. Validate on the full validation split every epoch.
6. Select the checkpoint by best validation mean fingerprint Tanimoto, not by
   training loss.

Loss terms:

```text
target_bce
+ mist_anchor_bce
+ soft_tanimoto
+ active_count_penalty
+ MIST-error bit weighting
+ MIST-uncertainty bit weighting
+ false-negative/false-positive weighting
```

## Launch Command

Submit from the remote FRIGID checkout:

```bash
cd /home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head
mkdir -p runs/slurm_logs
sbatch --dependency=afterany:32 docs/FRIGID_dreams_adapter_replacement_20260630/run_dreams_adapter_replacement.sbatch
```

The dependency keeps this job from competing with the active DLM adaptation job
`32` for the same A100.

## Metrics

Primary metric:

- full-validation best mean fingerprint Tanimoto across threshold and top-k
  sweeps.

Secondary metrics:

- median fingerprint Tanimoto;
- bit accuracy;
- bit precision;
- bit recall;
- bit F1;
- mean predicted active bits;
- delta versus the recomputed MIST validation baseline;
- validation loss trend and overfit behavior.

Required baseline:

- MIST probabilities on the same validation split, same metadata alignment, and
  same threshold/top-k sweep.

## Success Criteria

The encoder replacement gate requires:

```text
best validation mean fingerprint Tanimoto >= MIST baseline + 0.005
```

If this gate passes, export `val_predictions.npz` and run the downstream paired
DLM robustness benchmark with a new `dreams_adapter_binary` source.

## Stop Rules

Stop or redirect this DreaMS replacement line if:

- full validation mean fingerprint Tanimoto remains below 0.45 after staged
  adaptation;
- validation repeats the previous full-fine-tune overfit pattern;
- apparent gains come only from harming MIST-easy cases;
- downstream DLM robustness does not improve versus `mist_binary`.

## Expected Artifacts

Remote run directory:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_adapter_replacement_<UTC>
```

Expected files:

- `train.log`;
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
- `model/last_model.pt`;
- Slurm output and error logs under `runs/slurm_logs/`.
