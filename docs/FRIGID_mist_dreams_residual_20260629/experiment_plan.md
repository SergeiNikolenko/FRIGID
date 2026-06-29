# MIST + DreaMS Residual Adapter Experiment

Date: 2026-06-29

## Objective

Test a stronger DreaMS integration than probability blending. MIST remains the
base fingerprint encoder, while a DreaMS-conditioned adapter learns a residual
correction to MIST logits:

```text
final_logits = mist_logits + residual_scale * adapter(dreams_embedding, mist_logits)
```

This tests whether frozen DreaMS embeddings contain signal about MIST errors.

## Baselines

Full MSG validation baseline from the MIST+DreaMS blend experiment:

```text
pure MIST mean Tanimoto = 0.5420425056
best threshold = 0.25
```

Best standalone DreaMS Round 2 head:

```text
mean Tanimoto = 0.2342031586
variant = pos10_count005
threshold = 0.4
```

Linear probability blending did not improve MIST:

```text
best blend alpha_primary = 1.0
mean Tanimoto = 0.5420425056
```

## Full Experiment

1. Export MIST probabilities for the full MSG train split.
2. Reuse existing MIST validation probabilities from the blend run.
3. Align train and validation rows by `spec_name` across:
   - DreaMS embeddings
   - MIST probabilities
   - ground-truth Morgan fingerprints
4. Train residual adapters on the full train split.
5. Sweep validation thresholds and top-k values after every epoch.
6. Compare every model against the same pure-MIST validation baseline.

## Model

The residual adapter has two branches:

- DreaMS embedding branch
- MIST-logit branch

The branches are concatenated and mapped to a 4096-bit residual. The final
layer is initialized to zero, so the model starts exactly at pure MIST and can
only improve by learning useful residual corrections.

## Run Matrix

The initial serious matrix uses full MSG train/validation data:

| variant | residual scale | loss |
| --- | ---: | --- |
| conservative_bce | 0.25 | unweighted BCE |
| balanced_count | 0.5 | unweighted BCE + count loss |
| pos5_count | 0.5 | BCE pos_weight 5 + count loss |
| pos10_soft_count | 0.5 | BCE pos_weight 10 + soft Tanimoto + count loss |

## Decision Gate

Run DLM only if the residual adapter improves full-validation mean Tanimoto over
pure MIST by at least:

```text
+0.005
```

If the best residual result is below this gate, the frozen-DreaMS residual path
should be considered negative for near-term FRIGID quality work.

## Expected Outputs

Remote run directory:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_residual_20260629T*
```

Per variant:

- `summary.json`
- `training_metrics.json`
- `mist_baseline_metrics.json`
- `mist_baseline_metrics.csv`
- `val_sweep_metrics.json` if the variant improves over MIST
- `best_model.pt` if the variant improves over MIST
- `last_model.pt`

Local docs package should store compact JSON/CSV summaries and the run script.
Large checkpoints and prediction arrays stay on the remote host.
