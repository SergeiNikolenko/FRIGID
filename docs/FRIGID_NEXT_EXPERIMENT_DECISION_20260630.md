# FRIGID Next Experiment Decision

Date: 2026-06-30

## Decision

The next serious FRIGID experiment should be DLM adaptation to MIST-predicted
fingerprints, not another direct DreaMS replacement attempt.

Concretely:

1. Export full train-split MIST fingerprints.
2. Fine-tune DLM from the existing DLM checkpoint using those predicted
   fingerprints through `configs/fp2mol_finetune_mist_fingerprints.yaml`.
3. Evaluate the adapted DLM with the paired robustness benchmark against the
   original DLM on both clean `ground_truth` fingerprints and `mist_binary`
   fingerprints.
4. Promote the adapted decoder only if it narrows the `ground_truth` vs
   `mist_binary` generation gap without destroying clean-fingerprint decoding.

## Why This Is The Right Next Move

The current evidence is now consistent across independent experiments:

- MIST is still the strongest validated spectrum-to-fingerprint encoder.
- Direct DreaMS replacement is not close to MIST.
- Frozen DreaMS heads, MIST/DreaMS blends, residual adapters, distilled
  DreaMS-only students, and full DreaMS encoder fine-tuning did not clear the
  replacement gate.
- The largest measured system-level bottleneck is not "DreaMS is missing"; it
  is the mismatch between MIST-predicted fingerprints and the DLM decoder.

The 1,400-spectrum DLM robustness partial run showed a large paired gap:

| Metric | Ground-truth FP | MIST binary FP | Delta |
| --- | ---: | ---: | ---: |
| Exact match top-1 | 0.4879 | 0.1386 | -0.3493 |
| Exact match top-10 | 0.4921 | 0.1400 | -0.3521 |
| Tanimoto top-1 mean | 0.8130 | 0.5677 | -0.2453 |
| Formula-match success | 0.7643 | 0.6936 | -0.0707 |

This is much larger than any DreaMS-side gain found so far. The most direct way
to attack it is to make DLM robust to the fingerprint distribution that MIST
actually emits.

## What Not To Do Next

Do not spend the next long run on another plain DreaMS fingerprint objective:

- The best pure MIST validation fingerprint Tanimoto is approximately 0.5420.
- The best frozen DreaMS/distilled result is approximately 0.2404.
- The full DreaMS encoder fine-tune reached only 0.2580 and overfit strongly.
- The best anchored MIST+DreaMS residual improved MIST by only +0.00068, below
  the +0.005 encoder gate.

Those results are enough to stop treating direct DreaMS replacement as the main
near-term path.

## Proposed Experiment Design

### Training

Use the existing DLM adaptation path:

```bash
python scripts/export_mist_fingerprints.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir repro_cache/msg \
  --mist-checkpoint repro_cache/mist_msg.pt \
  --split train \
  --output-dir runs/mist_fingerprints_train_full
```

Then fine-tune DLM:

```bash
python scripts/train.py --config-name fp2mol_finetune_mist_fingerprints \
  load_weights_only=repro_cache/DLM.ckpt \
  data.predicted_fingerprint_metadata=runs/mist_fingerprints_train_full/metadata.csv \
  data.predicted_fingerprint_npz=runs/mist_fingerprints_train_full/fingerprints.npz \
  data.predicted_fingerprint_key=mist_binary
```

The existing config starts from:

- global batch size: 512;
- learning rate: 5e-5;
- max steps: 50,000;
- fingerprint dropout: 0.25;
- shared cross-attention enabled;
- checkpoint every 2,500 train steps.

### Evaluation

Run the paired robustness benchmark with both the original and adapted DLM:

```bash
python scripts/benchmark_dlm_fingerprint_robustness.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir repro_cache/msg \
  --mist-checkpoint repro_cache/mist_msg.pt \
  --dlm-checkpoint <adapted_dlm_checkpoint> \
  --use-shared-cross-attention \
  --formula-matches 10 \
  --max-attempts 100 \
  --batch-size 16 \
  --partial-save-every 100 \
  --fingerprint-sources ground_truth mist_binary \
  --output-dir runs/dlm_mist_adapted_robustness_full
```

The first evaluation can use a bounded 1,400- or 2,048-spectrum subset to match
the previous robustness evidence. If it improves, run the full 17,082-spectrum
test split.

## Success Criteria

Primary success criterion:

- Improve `mist_binary` DLM generation quality over the original DLM by at
  least +0.05 top-1 molecular Tanimoto on a paired validation/test subset.

Secondary criteria:

- Reduce exact-match top-1/top-10 gap between `ground_truth` and `mist_binary`.
- Maintain or only modestly reduce `ground_truth`-conditioned decoding quality.
- Improve formula-match success or at least avoid a regression.
- Save partial outputs during long robustness runs so interrupted jobs remain
  interpretable.

## Expected Outcomes

If DLM adaptation helps, the main FRIGID path should be:

```text
MIST encoder stays -> DLM is adapted to MIST fingerprint distribution -> rerun
FRIGID benchmark with adapted DLM.
```

If DLM adaptation does not help, then MIST is losing structure-critical
information before DLM sees the fingerprint. At that point the next direction
should be MIST-side training or a decoder-compatible MIST objective, not DreaMS
replacement.
