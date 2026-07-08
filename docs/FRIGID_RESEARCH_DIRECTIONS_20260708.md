# FRIGID research directions after the strict-threshold gate

Date: 2026-07-08

## Current experimental state

The fixed MIST threshold sweep produced a weak but useful signal:

- 64-spectrum gate: default `0.187` tan@1 `0.3209`; strict `0.50` tan@1 `0.3326`.
- 200-spectrum gate: default `0.187` tan@1 `0.2781`; strict `0.50` tan@1 `0.2861`.
- Paired 200 delta: tan@1 `+0.0080`, tan@10 `+0.0066`, wins/losses `104/96`.
- Bootstrap CI at 200 spectra crosses zero: tan@1 delta 95% CI `[-0.0042, +0.0206]`.

Decision: keep `0.50` as the current inference-side baseline, but do not promote
the fixed threshold directly to 1024/full. The next step should search for a
stronger sparsification or generation strategy.

The first fixed top-k follow-up did not survive scale-up:

- 32-spectrum grid: top-k `32` tan@1 `0.4439` vs fixed `0.50` tan@1 `0.3927`.
- 64-spectrum gate: top-k `32` tan@1 `0.3401` vs fixed `0.50` tan@1 `0.3326`.
- 200-spectrum gate: top-k `32` tan@1 `0.2668` vs fixed `0.50` tan@1 `0.2861`.
- Paired 200 delta vs fixed `0.50`: tan@1 `-0.0193`, 95% CI `[-0.0350, -0.0030]`.

Decision: reject fixed top-k `32`. It was a useful small-subset signal, but the
200-spectrum gate shows that fixed sparsity is not robust. The next
sparsification step must be confidence-gated or calibrated per spectrum.

## Literature signals

### 1. Diffusion decoder with formula constraints

DiffMS uses a transformer spectrum encoder and a discrete graph diffusion decoder
restricted by the known formula. The important idea for FRIGID is not only
diffusion itself, but decoder pretraining on large fingerprint-structure pairs
before bridging spectrum embeddings to structures.

Sources:

- PMLR paper: https://proceedings.mlr.press/v267/bohde25a.html
- Code: https://github.com/coleygroup/DiffMS

FRIGID experiment:

- Treat the current MIST/DLM path as a baseline and evaluate whether a DiffMS-like
  decoder or pretrained graph diffusion decoder can be used as a second candidate
  generator.
- First gate: run a small identical MassSpecGym subset if code/data alignment is
  feasible; compare tan@1/top-k and exact@k against the current strict-threshold
  DLM path.

### 2. Fingerprint decoder choice matters

Recent work on MIST + MolForge argues that the fingerprint-to-structure decoder is
critical and reports large gains from using a decoder trained on larger and broader
fingerprint-structure data. This directly matches the FRIGID failure mode: DLM is
fragile to MIST fingerprint errors, and raw `mist_probs` failed.

Source:

- "One Small Step with Fingerprints, One Giant Leap for De Novo Molecule
  Generation from Mass Spectra": https://arxiv.org/html/2508.04180v4

FRIGID experiment:

- Do not spend more time on full-DLM adaptation first.
- Compare DLM against an external fingerprint decoder path, ideally MolForge or a
  MolForge-like decoder, on the same MIST fingerprints.
- If integration is too slow, start with an offline decoder smoke on 32 spectra.

### 3. Retrieval-augmented spectral prediction and reranking

MARASON shows that retrieval augmentation plus neural graph matching can improve
mass spectrum simulation and downstream retrieval. ICEBERG-style spectral
prediction ranks candidate structures by comparing predicted spectra to the
experimental spectrum.

Sources:

- MARASON: https://arxiv.org/abs/2502.17874
- PMLR MARASON page: https://proceedings.mlr.press/v267/wang25dg.html
- ICEBERG web/repo entry point: https://github.com/coleygroup/ms-pred

FRIGID experiment:

- Use DLM/strict-threshold output as a proposal generator, not the final answer.
- Rerank generated candidates with a spectral predictor, initially ICEBERG if the
  local workflow can run it.
- Small gate: 16 or 32 spectra, compare top-1/top-10 before and after reranking.

### 4. Iterative optimization around seeds

FOAM formulates structure elucidation as formula-constrained iterative
optimization guided by predicted spectral similarity. It reports that seed
quality and spectral-prediction oracle quality are key determinants.

Source:

- FOAM: https://arxiv.org/html/2602.07709v1

FRIGID experiment:

- Use FRIGID/DLM candidates as seeds.
- Mutate/crossover around the best generated candidates under the known formula.
- Score with ICEBERG or another spectral predictor.
- Gate on hard cases where strict-threshold DLM improves MIST fingerprint quality
  but still misses exact structures.

### 5. Edge-aware graph generation

Many-body enhanced diffusion reports that explicit bond-bond or edge-edge
interactions improve chemical plausibility and similarity in MS-conditioned
generation. This suggests that pure fingerprint conditioning is too lossy for
some failures.

Source:

- Many-body enhanced diffusion: https://ojs.aaai.org/index.php/AAAI/article/view/37074/41036

FRIGID experiment:

- Treat this as a longer architecture track, not a quick benchmark knob.
- Start by analyzing whether current failures are mostly bond topology errors
  despite reasonable substructure/fingerprint similarity.
- If yes, create a graph-decoder track rather than more SMILES/DLM tuning.

### 6. Foundation spectrum encoders

DreaMS shows that large self-supervised MS/MS pretraining can produce structural
representations from unannotated spectra. Earlier direct DreaMS replacement did
not solve FRIGID, but the paper supports using embeddings for confidence,
retrieval, calibration, or reranking rather than as a direct MIST replacement.

Source:

- Nature Biotechnology DreaMS paper: https://www.nature.com/articles/s41587-025-02663-3

FRIGID experiment:

- Use DreaMS embedding distance or confidence as a gate for adaptive MIST
  sparsification.
- Use DreaMS nearest neighbors to seed retrieval/reranking, not as the only
  fingerprint predictor.

### 7. Flow matching decoders

FlowMS and MSFlow propose spectrum-conditioned flow-matching graph decoders as
alternatives to autoregressive or diffusion decoders. This is relevant because
the current fixed-threshold and top-k experiments show that the FRIGID bottleneck
is not only the MIST fingerprint, but also how brittle the downstream decoder is
when the fingerprint has plausible but imperfect substructure bits.

Sources:

- FlowMS: https://arxiv.org/abs/2603.18397
- MSFlow: https://arxiv.org/html/2602.19912v1
- MSFlow code: https://github.com/ghaith-mq/MSFlow

FRIGID experiment:

- Treat flow matching as a decoder-replacement track, not a threshold knob.
- First gate: find whether a public checkpoint or runnable inference path can
  score/generate for a 16/32 MassSpecGym subset.
- Compare against current fixed `0.50` DLM on the same spectra and formulas.

### 8. Scaffold and anchor-conditioned generation

MADGEN and MSAnchor use scaffold or anchor-extended representations to reduce
the search space before full molecule generation. This targets the exact-match
failure mode more directly than fingerprint sparsification: if the scaffold is
wrong, DLM can produce high-Tanimoto but wrong molecules.

Sources:

- MADGEN: https://arxiv.org/abs/2501.01950
- MSAnchor: https://ojs.aaai.org/index.php/AAAI/article/view/37064

FRIGID experiment:

- Start as retrieval/conditioning side information, not a full rewrite.
- First gate: on 32 hard spectra, test whether generated candidates already
  contain the correct or near-correct scaffold; if not, a scaffold-first track is
  justified.

## Immediate experiment backlog

### A. Confidence-gated MIST sparsification

Goal: improve over fixed threshold `0.50` without using ground truth. Fixed
top-k `32` is rejected after the 200 gate, so do not repeat fixed top-k as the
main hypothesis.

Candidates:

- top-k MIST bits: `k in {32, 64, 96, 128, 160, 256}`;
- quantile threshold: `q in {0.75, 0.85, 0.90, 0.95}`;
- confidence-gated top-k based on entropy or max probability;
- optional probability calibration before sparsification.
- prior-adjusted threshold matching the training-set active-bit prior.

Gate:

- first write MIST-only diagnostics: entropy, top-k probability mass,
  high-confidence ratio, probability quantiles, active-bit counts;
- retrospective analysis on completed 200 gates, using no target labels for the
  gate features;
- 32-spectrum prospective gate, paired against fixed `0.50`;
- promote only if tan@1 improves by at least `+0.005` on 32 and does not collapse
  formula success;
- 64 gate for the best candidate;
- 200 gate only if 64 is positive.

### B. Spectral-predictor reranking

Goal: keep DLM generation but rerank candidates with spectrum consistency.

Candidates:

- ICEBERG rerank of DLM candidate lists;
- formula/mass/fingerprint weighted rerank as a lightweight baseline;
- diversity-aware reranking to improve exact@10.

Gate:

- 16/32 spectra first because this is expensive;
- primary metrics: exact@1, exact@10, tan@1, tan@10;
- report generation time and reranking time separately.

### C. External decoder comparison

Goal: determine whether DLM is the bottleneck after MIST.

Candidates:

- MolForge or MolForge-like fingerprint decoder;
- DiffMS decoder path if integration is feasible;
- current DLM as control.

Gate:

- same MIST fingerprints, same spectrum subset, same formula constraints;
- compare exact@k and tan@k;
- do not tune thresholds on the evaluation subset.

### D. Evaluation hardening

Goal: avoid promoting artifacts.

Required improvements:

- fail fast on missing MIST/DLM checkpoints;
- write preflight/run manifest with checkpoint hashes;
- store paired subset manifests;
- bootstrap CI for paired deltas;
- validate artifact schemas after each run.

## Current decision

Do not scale fixed `0.50` or fixed top-k directly to full test yet. The next
active experimental track is confidence-gated sparsification diagnostics plus a
parallel decoder/reranking track. The highest-upside external comparisons are
MolForge/MSFlow/FlowMS/DiffMS decoder replacement and ICEBERG/MARASON-style
spectral reranking of existing DLM candidate lists.
