# DreaMS Distilled Replacement Candidate Experiment

Date: 2026-06-29

## Objective

Train a DreaMS-only spectrum-to-fingerprint model that can eventually replace
MIST, while using MIST as a teacher during training.

The model input is only the frozen DreaMS embedding. MIST probabilities are used
only as training-time supervision and error-focus metadata.

## Motivation

Previous results:

- frozen DreaMS shallow head: far below MIST;
- DreaMS Round 2 calibration: improved but still far below MIST;
- linear MIST+DreaMS blend: did not beat pure MIST;
- MIST+DreaMS residual: small positive delta, but still MIST-dependent.

To move toward a DreaMS replacement, the next model should learn:

1. the strong global fingerprint distribution that MIST already captures;
2. corrections on bits where MIST is wrong or uncertain.

## Method

Train:

```text
DreaMS embedding -> larger MLP head -> 4096 Morgan fingerprint logits
```

Loss terms:

- target BCE against ground-truth Morgan fingerprints;
- MIST distillation BCE against MIST probabilities;
- extra weight on MIST false-positive and false-negative bits;
- extra weight on MIST-uncertain bits;
- optional soft Tanimoto loss;
- optional active-bit count loss.

This is not a MIST+DreaMS residual. At inference time, the candidate model does
not need MIST.

## Data

Full MSG train and validation splits:

```text
train rows = 191216
validation rows = 19043
DreaMS embedding dimension = 1024
fingerprint bits = 4096
```

The experiment reuses:

- DreaMS embeddings from `dreams_fingerprint_head_msg_full_20260629T110040Z`;
- MIST train probabilities exported in `mist_dreams_residual_20260629T135651Z`;
- MIST validation probabilities exported in `mist_dreams_blend_20260629T131733Z`;
- ground-truth Morgan fingerprints from the prepared DreaMS datasets.

## Run Matrix

| variant | hidden dim | depth | target | distill | error focus | extra losses |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| distill_balanced | 2048 | 2 | 1.0 | 1.0 | 4.0 | count 0.02 |
| distill_error_heavy | 2048 | 2 | 1.0 | 0.5 | 8.0 | soft Tanimoto 0.1, count 0.02 |
| distill_teacher_heavy | 2048 | 2 | 0.75 | 2.0 | 4.0 | count 0.01 |
| distill_deep | 3072 | 3 | 1.0 | 1.0 | 6.0 | soft Tanimoto 0.05, count 0.02 |

## Gate

This is a replacement-candidate gate, so it must beat the pure MIST validation
baseline by at least:

```text
+0.005 mean Tanimoto
```

If it does not beat MIST, do not promote it to DLM yet. If it does beat MIST,
run a DLM diagnostic using the DreaMS-only predictions.

## Expected Remote Run

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_distilled_replacement_20260629T*
```

Local docs should store compact summaries and the leaderboard; large model
checkpoints and prediction arrays remain on `spectrum`.
