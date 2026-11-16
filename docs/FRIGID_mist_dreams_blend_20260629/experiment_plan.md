# MIST + DreaMS Fingerprint Blend Experiment

Date: 2026-06-29

## Objective

Test whether DreaMS-head probabilities add useful signal to the current MIST
fingerprint encoder. The standalone DreaMS-head Round 2 result improved over
the first DreaMS run but remained far below the MIST quality gate:

- best Round 2 DreaMS-head mean validation Tanimoto: `0.2342031586`
- best mode: `pos10_count005`, threshold `0.4`
- original first DreaMS-head run: `0.1238048323`

The next question is not whether DreaMS should replace MIST. The question is
whether DreaMS can act as a complementary correction signal.

## Method

1. Export MIST fingerprint probabilities on the same validation split and row
   identity as the DreaMS-head predictions.
2. Evaluate pure MIST and pure DreaMS against the same ground-truth Morgan
   fingerprints.
3. Sweep probability blends:

   ```text
   p_blend = alpha * p_mist + (1 - alpha) * p_dreams
   ```

4. For every alpha, sweep thresholds and fixed top-k active-bit counts.
5. Use full-validation mean Tanimoto as the primary encoder gate.

## Decision Rules

- If the best alpha is `1.0`, DreaMS adds no useful signal to MIST in this
  representation.
- If `0 < alpha < 1` beats pure MIST, train a residual/gated blend on the train
  split.
- If the blend beats MIST by a meaningful margin and clears the encoder gate,
  then run a DLM diagnostic.

## Expected Artifacts

Remote run directory:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_blend_20260629T*
```

Local docs package should store compact summaries and blend metrics under this
directory.

