# DreaMS Fingerprint-Head Round 2 Experiment Plan

Date: 2026-06-29

## Objective

Run a full second-round DreaMS-head experiment on `spectrum` before moving to
DreaMS backbone fine-tuning. The first full run showed that frozen DreaMS
embeddings plus an auto-positive-weight BCE head overpredicted active bits:

- mean validation Tanimoto: `0.1238048323`
- bit precision: `0.1238794961`
- bit recall: `0.5741983662`
- mean predicted active bits: `295.9601440430`
- mean target active bits: `63.8514404297`

Round 2 tests whether this is mainly a loss/calibration problem. It uses the
existing full DreaMS embeddings and target fingerprints, so no embedding
extraction is repeated.

## Inputs

- Base full run:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_full_20260629T110040Z`
- Train embeddings:
  `train_embeddings/dreams_embeddings.npz`
- Train targets:
  `train_dataset/fingerprints.npz`
- Validation embeddings:
  `val_embeddings/dreams_embeddings.npz`
- Validation targets:
  `val_dataset/fingerprints.npz`
- Validation metadata:
  `val_dataset/metadata.csv`

## Experiment Matrix

Each variant trains on the full `191,216` train split and evaluates on the full
`19,043` validation split. The fixed architecture remains:

```text
1024 DreaMS embedding -> LayerNorm -> Linear(1024) -> SiLU -> Dropout -> 4096 logits
```

Planned variants:

| Variant | Positive weight | Soft Tanimoto loss | Active-count loss | Purpose |
| --- | ---: | ---: | ---: | --- |
| `pos1_bce` | 1 | 0 | 0 | unweighted BCE baseline |
| `pos5_bce` | 5 | 0 | 0 | low positive weight |
| `pos10_bce` | 10 | 0 | 0 | moderate positive weight |
| `pos25_bce` | 25 | 0 | 0 | capped positive weight |
| `pos10_soft025` | 10 | 0.25 | 0 | optimize overlap directly |
| `pos10_count005` | 10 | 0 | 0.05 | penalize active-bit overprediction |
| `pos5_soft025_count005` | 5 | 0.25 | 0.05 | combined overlap and sparsity pressure |

All variants use 25 epochs, batch size 512, learning rate `1e-3`, weight decay
`1e-4`, and threshold `0.5` during training-time validation.

## Calibration Sweep

After each variant, validation probabilities are swept with:

- threshold values from `0.02` to `0.8`
- fixed top-k active bit counts: `32, 48, 64, 80, 96, 128, 160, 192, 256, 320`
- an oracle diagnostic that uses the target active-bit count per row

The selected metric is full-validation mean Tanimoto after calibration. The
secondary checks are median Tanimoto, bit precision/recall/F1, and predicted
active-bit count.

## Decision Gates

- If the best calibrated mean Tanimoto remains below `0.25`, the frozen-DreaMS
  head path is considered non-competitive and the next experiment should be
  partial DreaMS fine-tuning or a different spectral representation.
- If it reaches `0.25-0.45`, inspect per-spectrum failures and consider a
  larger head or adapter fine-tune, but do not run a full DLM benchmark yet.
- If it reaches `0.45-0.52`, run a 200-spectrum DLM diagnostic.
- If it exceeds the MIST reference gate of about `0.52`, run a larger DLM
  benchmark and compare against MIST and oracle fingerprints.

## Expected Output

Remote run directory:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_round2_20260629T*
```

For each variant:

- `fingerprint_head/dreams_fingerprint_head.pt`
- `fingerprint_head/training_metrics.json`
- `val_predictions/fingerprints.npz`
- `val_predictions/summary.json`
- `calibration_sweep/sweep_metrics.json`
- `calibration_sweep/sweep_metrics.csv`

The final docs update should copy compact metrics and summaries into this
directory under `docs/`.

