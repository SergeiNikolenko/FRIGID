---
marp: true
title: MS/MS Encoder Benchmark for FRIGID
description: Evidence, protocol, shortlist, and execution decision
---

# MS/MS Encoder Benchmark for FRIGID

Evidence, protocol, shortlist, and execution decision
2026-07-15

---

## The actual question

- FRIGID consumes an ordered Morgan-4096 fingerprint.
- A 4096-dimensional vector is not sufficient; bit semantics must match.
- The replacement must beat MIST on identical spectra without leakage.
- DLM matters only after an encoder-level win.

---

## Data are available

- MSG: 231,104 total spectra.
- Eligible train / validation / test: 191,216 / 19,043 / 17,082.
- MIST and DLM checkpoints are present and hashed.
- Historical metrics and candidate checkpoints are present.
- The missing full per-spectrum validation NPZ has been regenerated and hashed.

No additional data handoff is required to start.

---

## Current baseline

| Model | Mean fingerprint Tanimoto |
|---|---:|
| MIST | **0.542043** |
| DreaMS full fine-tune | 0.2580 |
| DreaMS distillation | 0.240354 |
| DreaMS frozen head | 0.123805 |
| MIST + DreaMS residual | 0.542726 |

Residual gain: **+0.000684**, below the **+0.005** gate.

Independent replay: MIST `0.5420425046`, zero train/evaluation structure
overlap, 19,043 per-spectrum rows retained.

---

## Decision: close the DreaMS replacement line

- Frozen, distilled, adapter, JEPA, and full fine-tune variants all fail.
- Full fine-tuning overfits strongly.
- Residual fusion adds less than the required effect.
- MSAlign remains a retrieval watch item, not another direct replacement run.

---

## Locked comparison contract

- Morgan radius 2, 4096 bits, `useChirality=false`.
- Molecule-disjoint train / calibration / evaluation partitions.
- Explicit spectrum-ID alignment; no positional assumptions.
- Threshold frozen on calibration before evaluation.
- Train/evaluation InChIKey overlap audit.
- Same probe, optimizer budget, and downstream generation budget.

---

## Promotion gate

A candidate proceeds only if:

1. paired mean Tanimoto gain is at least 0.005;
2. molecule-cluster bootstrap lower 95% bound is above zero;
3. training overlap is zero;
4. data, checkpoint, code, predictions, and manifests are hashed.

---

## First wave

| Candidate | Why now | Treatment |
|---|---|---|
| MSBERT | Released dense checkpoint | Frozen 512->4096 probe |
| MS2DeepScore | Mature weights and tooling | Frozen 500->4096 probe + retrieval |
| JESTR | Closest to structure ranking | Frozen 512->4096 probe + reranking |
| SpecEmbedding | Strong contrastive control | Frozen 512->4096 probe |

Every model receives the same `LayerNorm -> Linear` head.

---

## Second wave

- **IDSL_MINT:** direct active-bit sequence model, trained as Morgan-4096.
- **DiffMS:** inspect raw 4096 encoder output and full generator separately.
- **MSFlow:** compare as an end-to-end generator, not as a direct fingerprint.
- **CMSSP / ChemEmbed:** follow-up dense probes if Wave 1 justifies expansion.

---

## Why two tracks are necessary

### Fingerprint track

Can the encoder improve the exact FRIGID/DLM interface?

### Retrieval track

Can a dense embedding improve candidate ranking even if its Morgan probe is
weak?

Failure in one track does not imply failure in the other.

---

## Reproducible evidence package

- strict prediction-bundle validation;
- per-spectrum FP/FN and Tanimoto;
- paired wins, losses, ties, and cluster bootstrap;
- molecule-balanced and stratified metrics;
- immutable output directory;
- code, data, checkpoint, and ordered-ID hashes.

---

## Final decision

- Keep MIST in production.
- Stop investing in direct DreaMS replacement.
- Run four identical frozen probes first.
- Train IDSL_MINT only after the cheap probe wave is locked.
- Connect only gate-passing encoders to DLM.

This converts an open-ended survey into a finite, falsifiable benchmark.
