# MIST + DreaMS Fingerprint Blend Run Report

Date: 2026-06-29

## Goal

Test whether the best calibrated DreaMS fingerprint head from Round 2 adds a
useful complementary signal to the existing MIST fingerprint encoder on the
full MSG validation split.

This is an encoder-level experiment. It does not run the DLM decoder. The
decision criterion is validation Morgan-fingerprint Tanimoto quality before
downstream molecule decoding.

## Inputs

Remote project checkout:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head
```

Remote run directory:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_blend_20260629T131733Z
```

Code commit:

```text
c7e4c574295cf25a18274f25ace3ff0c5d790317
```

MIST checkpoint:

```text
/home/nikolenko/work/Projects/FRIGID/repro_cache/mist_msg.pt
```

MSG data directory:

```text
/home/nikolenko/work/Projects/FRIGID/repro_cache/msg
```

DreaMS source predictions:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_round2_20260629T123254Z/variants/pos10_count005/val_predictions/fingerprints.npz
```

Ground-truth fingerprints:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_full_20260629T110040Z/val_dataset/fingerprints.npz
```

## Method

1. Export MIST fingerprint probabilities on the MSG validation split using the
   FRIGID environment.
2. Align MIST predictions, DreaMS predictions, and ground-truth fingerprints by
   `spec_name`.
3. Evaluate probability blends:

   ```text
   p_blend = alpha * p_mist + (1 - alpha) * p_dreams
   ```

4. Sweep these alpha values:

   ```text
   0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0
   ```

5. For every alpha, sweep thresholds and fixed top-k active-bit counts.
6. Rank by full-validation mean Tanimoto.

## Results

Validation rows: `19043`

Fingerprint bits: `4096`

Best overall result:

| mode | alpha_primary | threshold/top-k | mean Tanimoto | median Tanimoto | bit F1 | mean active bits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold | 1.0 | 0.25 | 0.5420425056 | 0.5483871102 | 0.6909004105 | 57.5802116394 |

Best pure DreaMS point:

| mode | alpha_primary | threshold/top-k | mean Tanimoto | median Tanimoto | bit F1 | mean active bits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold | 0.0 | 0.4 | 0.2342031586 | 0.2156862766 | 0.3836226357 | 48.8542785645 |

Best near-blend points:

| mode | alpha_primary | threshold/top-k | mean Tanimoto | median Tanimoto | bit F1 | mean active bits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold | 0.95 | 0.25 | 0.5420244869 | 0.5454545617 | 0.6911619710 | 57.4212036133 |
| threshold | 0.9 | 0.25 | 0.5415965924 | 0.5423728824 | 0.6911011090 | 57.4037704468 |
| threshold | 0.8 | 0.3 | 0.5363206287 | 0.5217391253 | 0.6899351021 | 52.9993171692 |

Best result by alpha:

| alpha_primary | best mode | value | mean Tanimoto |
| ---: | --- | ---: | ---: |
| 0.0 | threshold | 0.4 | 0.2342031586 |
| 0.05 | threshold | 0.4 | 0.2406511007 |
| 0.1 | threshold | 0.35 | 0.2482396116 |
| 0.2 | threshold | 0.15 | 0.2938532500 |
| 0.3 | threshold | 0.2 | 0.3560918182 |
| 0.4 | threshold | 0.3 | 0.4090431786 |
| 0.5 | threshold | 0.35 | 0.4522323968 |
| 0.6 | threshold | 0.4 | 0.4927730337 |
| 0.7 | threshold | 0.35 | 0.5227742849 |
| 0.8 | threshold | 0.3 | 0.5363206287 |
| 0.9 | threshold | 0.25 | 0.5415965924 |
| 0.95 | threshold | 0.25 | 0.5420244869 |
| 1.0 | threshold | 0.25 | 0.5420425056 |

## Interpretation

The best alpha is `1.0`, which is pure MIST. The best 95% MIST / 5% DreaMS
blend is extremely close but still lower by mean Tanimoto:

```text
0.5420425056 - 0.5420244869 = 0.0000180187
```

The current DreaMS fingerprint head is therefore not a useful linear
probability-blend correction for MIST on MSG validation. It improves with more
MIST weight because MIST is much stronger, not because DreaMS adds a measurable
orthogonal signal.

This does not rule out DreaMS as a research direction. It rules out this
specific use: fixed DreaMS embeddings plus a shallow fingerprint head blended
linearly with MIST probabilities.

## Follow-Up Decision

Do not run DLM on the current DreaMS blend. The encoder result does not beat
pure MIST.

The next serious experiment should be one of:

1. Fine-tune the spectral encoder or a larger adapter objective, not only a
   shallow fingerprint head on frozen DreaMS embeddings.
2. Train a residual correction model on MIST errors using DreaMS embeddings and
   MIST logits as inputs, with a held-out validation split to avoid simply
   learning MIST calibration.
3. If the project priority is near-term quality, continue with MIST-centered
   diagnostics and DLM integration rather than DreaMS replacement.

## Local Artifacts

Stored in this docs package:

- `blend_summary.json`
- `blend_metrics.json`
- `blend_metrics.csv`
- `mist_export_summary.json`
- `run_blend.sh`

Large prediction arrays are intentionally not copied into the repository.
