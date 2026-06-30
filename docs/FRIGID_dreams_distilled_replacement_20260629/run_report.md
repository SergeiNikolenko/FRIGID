# FRIGID DreaMS Distilled Replacement Run Report

Date: 2026-06-30

## Objective

Test whether a DreaMS-only fingerprint predictor can replace MIST when trained
as a student on frozen DreaMS embeddings with a mixture of ground-truth Morgan
fingerprint supervision and MIST-probability distillation.

This experiment was intended to be stronger than the plain frozen-head DreaMS
run, but still cheaper than full DreaMS encoder fine-tuning.

## Remote Run

- Host: `spectrum`
- Run directory:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_distilled_replacement_20260629T171603Z`
- Train rows: 191,216
- Validation rows: 19,043
- Fingerprint bits: 4096
- DreaMS embedding dimension: 1024
- MIST replacement gate: DreaMS must beat the MIST validation baseline by at
  least +0.005 mean Tanimoto.

MIST validation baseline:

```text
mean_tanimoto = 0.5420425046
mode = threshold
value = 0.25
```

## Variants

| Variant | Best epoch | Mean Tanimoto | Delta vs MIST | Gate |
| --- | ---: | ---: | ---: | --- |
| `distill_teacher_heavy` | 7 | 0.2403539014 | -0.3016886032 | fail |
| `distill_balanced` | 5 | 0.2320045740 | -0.3100379306 | fail |
| `distill_deep` | 4 | 0.2291250453 | -0.3129174593 | fail |
| `distill_error_heavy` | 3 | 0.2217641807 | -0.3202783239 | fail |

## Interpretation

The distilled DreaMS-only replacement failed decisively. The best variant,
`distill_teacher_heavy`, reached only 0.2404 mean validation Tanimoto, roughly
0.302 below the MIST baseline. This is similar to the earlier frozen-DreaMS
results and far below the level needed for downstream FRIGID/DLM evaluation.

This closes another plausible DreaMS replacement path:

- frozen DreaMS embeddings plus a plain head are not competitive;
- frozen DreaMS embeddings plus MIST distillation are still not competitive;
- full DreaMS encoder fine-tuning also failed to approach MIST.

## Decision

Do not continue DreaMS-only replacement runs with the same frozen-embedding
student setup. The evidence does not justify DLM decoding or more variants of
the same objective.
