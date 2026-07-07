# Mixed Fingerprint DLM Adaptation

## Goal

Train DLM on a mixture of clean target Morgan fingerprints and MIST-predicted
binary fingerprints so the decoder becomes more robust to realistic MIST errors
without losing clean-fingerprint decoding quality.

This follows the failed 2026-07-06 plain `mist_binary` adaptation. That run
reduced the `ground_truth` vs `mist_binary` gap, but degraded absolute quality:

| Checkpoint | Fingerprint | Tanimoto top-1 |
| --- | --- | ---: |
| Original DLM | `ground_truth` | 0.3897 |
| Original DLM | `mist_binary` | 0.3209 |
| 2,500-step `mist_binary` adaptation | `ground_truth` | 0.3109 |
| 2,500-step `mist_binary` adaptation | `mist_binary` | 0.2796 |

## Hypothesis

Plain `mist_binary` training overfits the decoder toward noisy predicted
fingerprints and damages the clean fingerprint manifold. Mixed training should
keep the decoder anchored to clean fingerprints while exposing it to MIST
fingerprint errors.

## Implementation

Use the existing exported fingerprint NPZ. It already contains:

- `ground_truth`
- `mist_binary`
- `mist_probs`

The new dataset path supports:

```yaml
data:
  predicted_fingerprint_keys:
    - ground_truth
    - mist_binary
  predicted_fingerprint_probs:
    - 0.5
    - 0.5
```

Primary config:

```text
configs/fp2mol_finetune_mixed_fingerprints.yaml
```

Training defaults:

- base checkpoint: `repro_cache/DLM.ckpt`
- hidden size: `896`
- intermediate size: `3584`
- attention heads: `14`
- learning rate: `1e-5`
- global batch size: `32`
- max steps: `10000`
- checkpoint interval: `500`

## Smoke Test

Before a long run, verify:

1. dataset can load both fingerprint arrays;
2. mixed sampling returns valid 4096-bit fingerprints;
3. 2-step DLM smoke train writes checkpoints;
4. no CPU/CUDA mismatch or checkpoint architecture mismatch occurs.

2026-07-07 smoke result:

- remote run root: `/home/nikolenko/work/Projects/FRIGID_dlm_mist_adapt_cbc854`
- dataset: `runs/mist_fingerprint_exports/train_subset_4096`
- dataset check: 4096 rows, 4096-bit `float32` fingerprints
- train smoke: `runs/dlm_mixed_smoke_2`
- checkpoints: `checkpoints/1.ckpt`, `checkpoints/2.ckpt`
- log: `train.log`

The first full-run attempt with `global_batch_size: 512` exhausted A100 80GB
memory. The default was reduced to `32` and re-checked with:

- train smoke: `runs/dlm_mixed_batch32_smoke_2`
- checkpoints: `checkpoints/1.ckpt`, `checkpoints/2.ckpt`

## Evaluation

Run the paired robustness benchmark with the same diagnostic settings:

```text
max_spectra: 64 first, then 200/1024 if promising
formula_matches: 2
max_attempts: 20
batch_size: 4
fingerprint_sources: ground_truth mist_binary
```

Required comparison:

- original DLM + `ground_truth`
- original DLM + `mist_binary`
- mixed-adapted DLM + `ground_truth`
- mixed-adapted DLM + `mist_binary`

## Success Gate

The mixed-adapted checkpoint must improve `mist_binary` decoding without
collapsing `ground_truth` decoding.

Minimum 64-spectrum gate:

- `mist_binary` top-1 Tanimoto must beat original DLM `mist_binary`;
- `ground_truth` top-1 Tanimoto must stay within 5 percent relative of original
  DLM `ground_truth`;
- exact-match metrics must not regress if they become non-zero;
- benchmark runtime should stay in the same order of magnitude.

If the gate fails, do not extend the run. Move to soft `mist_probs`, partial
freezing, or conditioning-only training.
