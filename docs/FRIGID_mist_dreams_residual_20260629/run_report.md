# MIST + DreaMS Residual Adapter Run Report

Date: 2026-06-29

## Goal

Run a full MSG train-to-validation experiment testing whether frozen DreaMS
embeddings can improve the existing MIST fingerprint encoder through a learned
residual correction:

```text
final_logits = mist_logits + residual_scale * adapter(dreams_embedding, mist_logits)
```

This is stronger than the earlier probability blend because the adapter can
learn bit-specific and spectrum-specific corrections instead of using one
global alpha.

## Data And Baseline

Training rows: `191216`

Validation rows: `19043`

Fingerprint bits: `4096`

DreaMS embedding dimension: `1024`

Pure MIST validation baseline:

| mode | value | mean Tanimoto | median Tanimoto | bit F1 | mean active bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| threshold | 0.25 | 0.5420425046 | 0.5483870968 | 0.6909004105 | 57.5802132017 |

Gate for a useful encoder-level improvement:

```text
mean Tanimoto improvement >= +0.005
```

## Run 1: Unanchored Residual Matrix

Remote run directory:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_residual_20260629T135651Z
```

Commit:

```text
bd22600d1ee1a0dd63c6d3a89330c621ec142ca5
```

Variants:

| variant | best epoch | best mean Tanimoto | delta vs MIST | result |
| --- | ---: | ---: | ---: | --- |
| balanced_count | 0 | 0.5420425046 | 0.0000000000 | no improvement |
| conservative_bce | 0 | 0.5420425046 | 0.0000000000 | no improvement |
| pos10_soft_count | 0 | 0.5420425046 | 0.0000000000 | no improvement |
| pos5_count | 0 | 0.5420425046 | 0.0000000000 | no improvement |

Every trained epoch was worse than the epoch-0 MIST initialization. Best epoch
1 results were already below MIST:

| variant | epoch 1 mean Tanimoto | epoch 1 delta |
| --- | ---: | ---: |
| balanced_count | 0.5017572700 | -0.0402852350 |
| conservative_bce | 0.5091020810 | -0.0329404230 |
| pos10_soft_count | 0.5081113120 | -0.0339311920 |
| pos5_count | 0.5117228590 | -0.0303196450 |

Interpretation: ordinary target BCE trains away from the strong MIST solution.
It lowers train loss but damages validation Tanimoto.

## Run 2: Anchored Residual Matrix

Remote run directory:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_residual_anchored_20260629T160033Z
```

Commit:

```text
8e0201632ef9576f1f6c1f2cc2a3b3bc11d9f768
```

This run added:

- distillation anchor to MIST probabilities;
- residual L2 penalty;
- lower learning rates;
- smaller residual scales.

Leaderboard:

| variant | best epoch | mode | value | mean Tanimoto | delta vs MIST | gate |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| anchor10_scale005 | 3 | threshold | 0.25 | 0.5427262996 | +0.0006837950 | fail |
| anchor10_scale01_count | 1 | threshold | 0.25 | 0.5426932803 | +0.0006507756 | fail |
| anchor30_pos5_scale01 | 3 | threshold | 0.30 | 0.5426415289 | +0.0005990242 | fail |
| anchor30_scale005 | 3 | threshold | 0.25 | 0.5423324762 | +0.0002899716 | fail |

Best anchored result:

```text
anchor10_scale005
epoch = 3
mean Tanimoto = 0.5427262996
delta vs MIST = +0.0006837950
```

## Interpretation

The anchored residual experiment found a real direction above pure MIST, but
the effect is very small: about `+0.00068` mean Tanimoto. This is far below the
predefined `+0.005` gate for spending DLM compute.

The result is stronger than the earlier linear blend because the learned
residual can beat MIST slightly, while linear blending could not. However, the
gain is not large enough to justify downstream molecule decoding or claiming
that frozen DreaMS materially improves FRIGID quality.

## Decision

Do not run DLM for this residual checkpoint yet.

The current evidence says:

1. Standalone DreaMS fingerprint head is much weaker than MIST.
2. Linear MIST+DreaMS probability blending does not beat MIST.
3. Unanchored residual training overfits or drifts away from MIST.
4. Anchored residual training gives a small positive signal but does not clear
   the encoder gate.

## Next Recommended Experiment

The next useful experiment is not another frozen-DreaMS head. The next serious
direction should be either:

1. Train a residual objective that targets only high-confidence MIST error
   regions, using MIST correctness/uncertainty masks instead of all bits.
2. Partially fine-tune a DreaMS adapter or upper encoder layers on the
   spectrum-to-fingerprint objective, because frozen embeddings appear too weak
   for a large gain.
3. If the project goal is near-term FRIGID quality, return to MIST-centered
   DLM integration and calibration rather than DreaMS replacement.

## Local Artifacts

Compact artifacts are stored under:

```text
docs/FRIGID_mist_dreams_residual_20260629/remote_runs/
```

Included:

- leaderboards;
- run info;
- per-variant summaries;
- per-epoch training metrics;
- validation sweep metrics for anchored variants;
- MIST baseline sweeps.

Excluded intentionally:

- `.pt` checkpoints;
- `.npz` prediction arrays;
- large metadata CSVs;
- multi-megabyte raw logs.
