# Decoder programme — state of evidence and ranked work

Written 2026-08-12. Everything here is measured unless marked otherwise; every number
carries the file it came from. This document exists so that no experiment is repeated
and no conclusion outlives the measurement that supported it.

## 1. Objective

Beat MARLIN's and FRIGID's reported numbers **by training the decoder better**, and be
able to show that is what happened.

Consequences of that framing, which order everything below:

- **The decoder is the object of study.** Encoders are taken as given. We may change how
  their output is *processed* (thresholding, calibration, cross-fitting the probe that
  turns embeddings into fingerprint bits) but we do not train or redesign an encoder.
- **Generation speed is a first-class requirement**, not an optimisation. Hypotheses per
  day is the limiting resource, and today one evaluation of the 321-spectrum panel costs
  ~10 CPU-hours.
- **Hopeless attempts must die early.** Compute spent on an attempt that cannot produce a
  candidate is compute stolen from a hypothesis.
- **Metrics must be FRIGID-convention and honest**, so the comparison to both baselines is
  real rather than a denominator trick. See §7.

## 2. What we may measure on

| Panel | Spectra | Status |
|---|---:|---|
| Locked test split | 803 | **Usable.** 0/803 share a connectivity block with training. |
| Clean validation panel `nplib1_val_clean322_v1` | 321 | **Usable.** Built 2026-08-10 by removing 75 contaminated rows from val396. |
| val396, micro128, micro32, micro4 | — | **Burned.** 75/396 val spectra share a connectivity block with the 6,748-spectrum adaptation training set. |

Every exact hit ever recorded on a micro panel is the same memorised molecule,
`CCMSLIB00005465125` / block `SMEROWZSTRWXGI` (lithocholic acid), whose exact SMILES sits
twice in the adaptation training set. On micro128: 2/2 hits contaminated, 0 of 107 clean
spectra solved. Any conclusion drawn from those panels — the first non-zero Exact@1, the
step-20,000 peak, the four-arm mask comparison, the learning-rate isolation, the 8-vs-64
budget claim — is void.

**Rule:** a panel without a training-overlap gate is not a panel. Add the check to
`scripts/build_marlin_nplib1_benchmarks.py` quality_checks before building another one.

## 3. Honest numbers today

| Measurement | Value | Source |
|---|---:|---|
| Exact@1, locked test, 8 candidates | **2.74%** (22/803) | `evaluations/full803-c8-100k` |
| Exact@1, clean panel, 8 candidates | **0.93%** (3/321) | `evaluations/clean-before` |
| Exact@1, test, 64 candidates (190 paired) | 5.26% vs 4.21% at 8 | paired subset of `full803-c64-100k` |
| Paper target | 16.94% | MARLIN paper |
| Teacher-forced per-token top-1, true fingerprint | 0.750 | 24-molecule probe |
| Teacher-forced per-token top-1, DreaMS fingerprint | 0.547 | same probe |
| Median Tanimoto, DreaMS vs true fingerprint @0.95 | 0.293 val / 0.304 test / 0.274 clean panel | full-split recomputation |

### Loss decomposition on the locked test split (8 candidates)

| Outcome | Spectra | Share |
|---|---:|---:|
| Returned nothing | 499 | **62.1%** |
| Returned only wrong candidates | 278 | 34.6% |
| Right answer present but not first | 4 | **0.5%** |
| Right answer first | 22 | 2.74% |

This decomposition is the single most important table in the project. It says the yield
problem is generation, not ranking.

## 4. Established mechanisms

### 4.1 Wall clock is the grammar mask, not the model

- Full-support mask call: 270 ms at prefix length 30, **5–23 s at length 65–88**, one call
  measured at **343.2 s**.
- Model forward pass: ~10 ms.
- Single-token `SafeGrammarMask.admits` probe: **0.47 ms**; lru_cache hit 77 µs.
- 70.3% of positions have exactly one token above p=0.01.
- One 943 s spectrum spent 202.4 s in 500 of its 551 mask calls.

### 4.2 "max_length" endings are deadline truncations

`max_length` is 256 and the position embedding is (256, 896), yet attempts stop at
56/88/96/104 tokens — all block-width multiples. The same spectrum and seed reaches 96
tokens under a 900 s budget and 104 under a longer one.

### 4.3 EOS is blocked by open ring labels, not by mass

In 21 of 32 traced attempts EOS was grammatically illegal because 2–16 SAFE ring labels
were still open, while the mass shell would have accepted termination in 17 of those 21.
The decoder builds a molecule whose mass is right and cannot say so.

### 4.4 Commitment is irreversible

There is no remasking, no backtracking, no revision. Traces show every attempt leaving the
gold path at token 1–5 of 60, with the gold token still inside the admitted support and
ranked 2 by the model. Attempts are 8 independent draws that never learn from each other.

### 4.5 The conditioning ladder is broken at three points

| Stage | Median Tanimoto to true Morgan | Active bits |
|---|---:|---:|
| Pretraining (true Morgan + symmetric noise) | 0.811 | ~50 |
| Adaptation (DreaMS **in-sample**, threshold 0.90) | 0.398 | ~80 |
| Inference (out-of-sample, threshold 0.95, +30% dropout) | 0.223–0.293 | 32 |

Three separable defects: the train-split DreaMS predictions are in-sample for the probe
that produced them (0.475 train vs 0.282 val); the adaptation threshold is 0.90 while every
evaluation ran at 0.95; inference diversity dropout only *removes* bits while training
taught symmetric corruption.

The checkpoint under test is already 100k steps of adaptation on predicted fingerprints —
"fine-tune on predicted fingerprints" is **done**. What remains is parity.

### 4.6 What the fingerprint actually carries

Morgan-radius recall: 0.902 (r0) / 0.519 (r1) / **0.287 (r2)**. Bare heteroatom bits 0.856
(bare O 0.971) against carbonyl context 0.345, aryl-carbonyl 0.147, nitro and nitrile 0.000.
53.4% of the ON bits handed to the decoder are simply the 48 most frequent training bits.

Within the mass shell the decoder already enforces (±0.5 Da, 7,144-molecule pool, median 15
competitors), the true fingerprint ranks the gold structure first **99.2%** of the time; the
DreaMS fingerprint **48.3%**. Use that pair as the encoder-side metric — Tanimoto is
unreadable by comparison.

### 4.7 Ceiling

Calibrated against the teacher-forced anchors, a *perfect* fingerprint puts this decoder at
**10.75–17.99% Exact@1**. Reaching 16.94% needs per-token 0.795, above what the decoder
achieves with the true fingerprint (0.750). **The fingerprint alone cannot get us there —
the decoder has to get better.** That is the programme.

## 5. Killed — do not revisit without new evidence

| Idea | The number that kills it |
|---|---|
| Any new candidate reranker (ICEBERG, MSAlign, learned LTR) | Exact@10 == oracle over all stored candidates; total headroom +0.50 pp on 803, and **zero** on the clean panel (Exact@1 == Exact@10 == 3/321) |
| Sweeping the DreaMS binarisation threshold | Fine sweep 0.02–0.9999 puts the argmax exactly at 0.95; per-row oracle threshold reaches only 0.321 |
| Mass-first / mass-gated ranking | 100% of returned candidates already sit inside 10 ppm; mass-first sort degrades 5.26% → 2.63% |
| Consensus / self-consistency ranking | Gains 2 spectra on 190, loses 2 on 803; McNemar p = 0.500 |
| Candidate budget as the primary lever | 8 → 64 costs 5.6× compute for +1.05 pp |
| Retrieval instead of generation | 0 of 321 clean-panel structures are in the train library; measured top-1 0.00% |
| Lower temperature / greedy / top-p | Gold was argmax at **0 of 32** divergence points; mean p(argmax)/p(gold) = 36.6× |
| Dedup by InChIKey first block | Zero duplicates exist in the returned lists |
| "Return the largest fragment" | Median Tanimoto gain +0.0035–0.0048; Exact@1 moves by exactly zero on all three datasets |
| Reject multi-fragment candidates as a filter | Exact@1 unchanged; mean top-1 Tanimoto falls 0.1575 → 0.1146 |
| Hydrogen padding as a mass story | Hydrogen-only fragments carry a **median 0.0000** of target mass; mass is bought with C, O and whole sugar rings |
| Exotic charge tokens as an accuracy story | 0/1278 returned candidates carry one |
| Early abort as a standalone fix | 0/803, 0/321 spectra hit the time cap; every spectrum ran its full attempt budget, so freed time has nowhere to go |
| Formula conditioning, *now* | FRIGID's 0 → 53% jump is real but comes from a decoder trained with `formula_dropout_prob = 0.0`; adopting it needs a full retrain |
| `safe_to_smiles(fix=True)` rescue | Rescues 60 of 91 unparsable strings; 0 of them land inside the 10 ppm shell |

## 6. The programme, ranked

Ordered by (expected gain) / (cost), with the objective's priorities applied: speed and
early rejection first because they buy hypotheses, then decoder training, then encoder
*processing*.

### R1 — Lazy top-k mask (speed, unblocks everything)
Replace the full-support call with a top-k admissibility probe, and move the deadline check
from the block loop into the per-token loop. `src/marlin/sampler.py:485-489`, calling the
existing `SafeGrammarMask.admits` (`src/marlin/grammar.py:1269`).
**Gain:** 10–60× throughput at identical outputs; recovers the attempts truncated by the
clock rather than by chemistry. **Cost:** 30–50 lines, CPU-only validation by replaying
stored prefixes and asserting the chosen token matches the full-support choice, then the
gold-mask-walk gate. **Risk:** low — outputs must be bit-identical, which is testable.

### R2 — EOS reachability as a mask invariant (yield)
Forbid any token that leaves more open ring labels than the remaining mass and valence
budget can close; forbid `.` while the deficit cannot host another fragment.
**Gain:** bounded by the empty-return bucket. At the observed conditional Exact@1 of 7.24%,
converting a quarter of the 499 empty test spectra is +0.9 pp, half is +1.8 pp.
**Cost:** 1–2 days in `grammar.py` / `token_properties.py`, plus the gold-mask-walk gate.

### R3 — Honest checkpoint selection
Evaluate step=20000, 30000 and 100000 on the clean panel. The current checkpoint was chosen
without a clean-panel comparison.
**Cost:** ~2 h wall per checkpoint on 6 shards after R1; far more before it.

### R4 — Conditioning parity re-adaptation (the decoder-training experiment)
One run with: out-of-fold train fingerprints (cross-fit the probe k-fold so
`train/dreams_predictions.npz` stops being in-sample), the training gate identical to the
evaluation gate (soft, 0.95), and a paired arm with symmetric noise switched off.
**Gain:** targets the teacher-forced gap directly (0.750 vs 0.547 per-token). External
anchor: MS-BART reports 1.71% → 7.45% top-1 on NPLIB1 from exactly this class of fix.
**Cost:** ~26 GPU-hours for 20,000 steps. **Blocked on:** pushing `marlin-reproduction`
(35 commits ahead of origin) — ClearML delivers training code by git clone, and
`scripts/train_marlin_spectrum_adaptation.py:311-313` refuses a dirty tree.

### R5 — Self-correction training (the decoder-training experiment that matters most)
With probability p, replace a fraction of the visible prefix with *wrong* tokens rather than
masks, keeping the loss on those positions, so the model learns to repair its own errors
instead of only filling masks. Today it never sees a wrong prefix, so every step after its
first mistake is off-distribution — which is exactly what the traces show.
Patch: `/tmp/selfcorrect/patch.diff`, design `/tmp/selfcorrect/design.md`. At p=0 it
reproduces the current objective bit-exactly, so the control arm is free.
**Blocked on:** the same push.

### R6 — Prefix-only beam (attacks the measured divergence point)
Enumerate the top-2 tokens at positions 1–3 (8 prefixes) and continue each with the existing
sampler: 8 fingerprint draws × 8 prefixes rather than 8 independent draws. A plain beam
collapses because the second-best probability is ~1e-4 under one conditioning vector, which
is why it must be prefix-only and combined with conditioning diversity.
Patch: `/tmp/beam_search.patch` (613 lines, applies cleanly). **Cost:** one paired evaluation.

### R7 — Encoder *processing* only
Cross-fit the DreaMS→Morgan head k-fold and regenerate `train/dreams_predictions.npz`
out-of-fold; replace LayerNorm+Linear with a small MLP; select on ON-bit recall. The head
trains in 5.3 s, so k-fold is ~30 s. **This is processing, not encoder training.**
Also worth one paired CPU diagnostic: MIST fingerprints vs DreaMS on the same rows, scored
by mass-shell top-1 (§4.6), before anyone spends GPU on a lane switch.

### R8 — Hygiene, ship with the next grammar change
- Monovalent atoms may not carry a ring-closure label. 0 of 7,947 gold answers do; gate
  byte-identical (32 rejections, same rows). Patch `/tmp/hydrogen/patch.diff`.
- Bound formal charge per element on the parsed atom. 473 of 1880 tokens carry |q| ≥ 2 and
  45 survive the current filters; the mask admits one at 97.8% of positions and they are
  modelled *lighter* than real atoms, so they are the cheapest mass sinks available.
  Patch `/tmp/exotic_charge.patch`; support shrinks from median 291 to 259 tokens.
Both are **zero-accuracy** changes. Ship them as hygiene and claim nothing.

## 7. Metrics contract

`src/marlin/frigid_convention.py` already states the problem: both projects average Exact@k
over every spectrum and count a no-return as a miss, but **Tanimoto@k differs by denominator**
— we average over spectra that returned something, FRIGID averages over all with a missing
candidate scoring 0.0. FRIGID pads its candidate list until it reaches the requested count,
so it almost never returns nothing and the two denominators nearly coincide *for it*. For a
mass-shell constrained decoder they differ by the candidate return rate, which is 0.38–0.44.

Rules from here on:

1. **Every reported number carries its denominator.** `frigid_convention_metrics` returns
   them; use it, and score both projects' prediction files with the same scorer.
2. **Headline metrics** are Exact@1 and Exact@10 over all spectra, plus FRIGID-convention
   Tanimoto@1 over all spectra, plus candidate return rate. Never quote a conditional
   Tanimoto without the return rate beside it.
3. **Never quote Exact@10 as headroom.** It equals the oracle over stored candidates because
   the sampler rarely returns more than a handful of unique ones.
4. **Panels:** locked test 803 for headline claims, clean 321 for iteration. Nothing else.
5. **Every panel gets a training-overlap gate** before it is used for a decision.
6. **A truncated run is reported as truncated.** `MarlinGenerationStats.truncated` exists;
   surface it in the metrics file so a partially searched spectrum can never be mistaken for
   a fully searched one.

## 8. Infrastructure

- **ClearML queue `sience`** (id `e0841e72c8a544efa9c54b5e768b1683`), worker `aiagent03`,
  two GPU slots. Both are currently held by our own runs.
- **The worker's `/mnt/netstorage` is NOT the shared NFS** — it is a per-worker local ext4
  (983 G, 106 G free) carrying the same path name. Plans that "just point at the NFS path"
  are wrong by construction. The control-r2 checkpoint happens to be present there with a
  matching sha256, which is incidental state on one worker's 89%-full disk.
- **Code delivery** to a ClearML worker is by git clone of a pushed commit; the training
  script additionally refuses a dirty tree. Any queued *training* therefore needs the branch
  pushed.
- The local A100 is shared with a colleague's FRIGID encoder fine-tuning run
  (`peak_resampler_finetuned`, ClearML task `c2f1bfe44bef465192458ace0b7a0c40`). Do not
  disturb it; treat the local GPU as unavailable.

### Runs in flight (2026-08-12 evening)

| Task | What | State |
|---|---|---|
| `1adf897039114c38a1c9a06e8384211e` | Clean panel, **oracle** fingerprint, 8 candidates | running, ETA ~02:10Z |
| `99e971d1ec1644bb80f5e0ed7106e93d` | Clean panel, DreaMS fingerprint, 8 candidates | running, ETA ~04:30Z |
| `c25c3ac3c6b44e66bbeffe04187ba445` | Clean panel, 64 candidates | queued behind our own jobs; **resubmit with ~24 shards** — as submitted it extrapolates to 3–4 days |

The oracle/DreaMS pair is matched flag-for-flag and even in sparsity (48.64 vs 48.67 mean
active bits), so the difference is attributable to the conditioning alone.

## 9. Open decisions for the project lead

1. **Push `marlin-reproduction` to origin?** R4 and R5 — the two decoder-training
   experiments, i.e. the actual objective — cannot be queued until the branch is pushed.
2. Cancel and resubmit run C with 24 shards, or drop the 64-candidate arm entirely given
   that budget was already killed as a primary lever (§5).
