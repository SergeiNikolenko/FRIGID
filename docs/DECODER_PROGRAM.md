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

### 4.7 The mask now agrees with RDKit on every finished string it admits

The mask tracked lexis, mass and connectivity and never asked two chemical
questions, so it admitted finished strings RDKit refuses in two named classes:
an aromatic atom outside any ring (`c12ccccc1.c13.[H]1.[o+]23`) and an aromatic
ring with no Kekule structure (`c1cccccc1-1.[H+]2.O2-1`). Both are closed now.

- **An aromatic atom must be able to reach a ring.** Decided on the parsed atom:
  take the bonds written so far, add one node standing for everything unwritten,
  join it to each atom by one edge per bond that atom can still receive — one per
  ring label it holds open, plus its spare valence when it is still the current
  atom or on the branch stack — and require every aromatic atom to lie on a cycle
  of that graph. `src/marlin/grammar.py:_aromatic_atoms_can_reach_a_ring`.
- **A closing aromatic ring must kekulize.** Asked once, of the finished string,
  in front of EOS. `src/marlin/grammar.py:_aromatic_system_kekulizes`.

Measured over the three stored runs, terminal strings the mask admits and RDKit
refuses go **116 → 0** (`full803-c8-100k`), **13 → 0** (`clean-before`) and
**79 → 0** (`full803-c64-100k`). Nothing that parses is lost: strings RDKit reads
and the mask admits stay at 360, 3 and 190, and returned candidates stay at
397/534, 132/247 and 316/497. The two classes are 241 of the 1,413 sampled
terminals of the 803-spectrum run (17.06%), and **all 241 are now refused** — 162
while the string is still being written, 79 at EOS. Per spectrum, 116 of the 499
empty-return test spectra, 25 of 188 on the clean panel and 30 of 54 on c64 had
at least one attempt killed by one of them.

Read the size of the prize honestly: the median refusal lands at **96.7% of the
string**. The decoder does not get the attempt back, it gets its last few tokens
back — the doomed token leaves the support and the model picks another instead of
ending on a string nothing can read. The whole-attempt lever is still R2.

**The acceptance gate constant moved: 32 gold rejections → 28**, same 7,551
targets and 431,904 positions, still 0 on test. The 28 are a strict subset of the
32 and no gold answer is newly refused; the four rescued rows (train 1781, 3962,
4063, 4070, e.g. `c13cccc2nnsc12.CSC3=O`) come from widening the mass-viability
probe, which now also tries the ring label an aromatic atom must open before any
atom may follow it.

### 4.8 Ceiling

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
| An admissible-ring-size rule for kekulisation | Ring perception over the 7,947 gold answers says aromatic rings are 5 and 6 (1,703 and 11,337 of 13,056, plus 10 sevens, 5 sixteens, one three), but a mask sees *closures*, and the smallest all-aromatic cycle a gold closure completes is 3, 5, 6, 7, 8, 9 (456), 10 (573), 11, 12, 14, 16, 17 and 22, with 1,045 completing none at all — fused polycycles close their perimeter first. A size cut costs thousands of gold answers. `scripts/audit_marlin_aromatic_rings.py` |

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
7. **A repaired candidate is not a candidate.** `safe_to_smiles` defaults to `fix=True`
   (`src/dlm/utils/utils_chem.py:26`), which deletes every fragment that will not decode and
   returns the stump — `"c1cccccc1-1.[H+]2.O2-1"` becomes `"O"`. Every candidate now carries
   `repaired`, every row carries `generated_candidate_count` / `repaired_candidate_count`,
   and a repaired candidate is dropped before scoring unless `--keep-repaired-candidates`
   says otherwise. Measured cost on every panel we have: **zero** — 0/247 clean321,
   0/534 test803-c8, 0/497 c64 returned candidates needed the repair, and all four headline
   runs already ran with `safe_decode_fix = False`. Prediction files written before this
   report `repaired_candidate_rate: null`, which means *unknown*, not clean.

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

## 10. Back on full panels — measurement wave of 2026-08-12, 19:00–19:30Z

Appended, not merged into anything above. Sections 1–9 are unedited.

### 10.1 The throughput regression is still in the code that is running

`git log --oneline -8` at the time of writing carries no lazy or top-k mask commit. The
lazy top-k probe exists only as an **uncommitted working-tree change** to
`src/marlin/sampler.py` (`_probe_token`, `lazy_probe_width`), `src/marlin/grammar.py`
(`_decoded_prefix` / `_prefix_scan` caches) and the untracked
`scripts/audit_lazy_probe_identity.py`. Those files were still being edited at 19:05Z while
this measurement ran, so R1 is in flight, not landed.

Code does **not** reach the ClearML worker by git clone for evaluation: the entry point is
`marlin_clean_panel_eval.sh`, and the tree arrives as artifact `code` of payload task
`3993339892da42c9be34c7e76022376e`
(`code.tar.gz` sha256 `30cd5d937b23acb5d206af85f212c2d9933d141113b89f8af1b6b694d32c1bba`).
That tarball's `src/marlin/sampler.py` and `src/marlin/grammar.py` hash to
`d03f5b2031df…` and `de9f6a0f469c…`, byte-identical to commit `9d0a919`, and contain no
`lazy_probe`. **Both running evaluations therefore execute the pre-lazy-mask code.**

Measured from the two tasks' own 10-minute heartbeats rather than from the commit titles,
over the steady-state window 1,212 s → 5,412/6,012 s:

| Run | rows/h | s/spectrum wall (6 shards) | s/spectrum of shard time |
|---|---:|---:|---:|
| `1adf897039114c38a1c9a06e8384211e` oracle | 52.3 | 68.9 | **413** |
| `99e971d1ec1644bb80f5e0ed7106e93d` DreaMS | 49.7 | 72.7 | **436** |

Baseline for the same 321 spectra at commit `2049e10`, from
`/mnt/netstorage/nikolenko/marlin/evaluations/clean-before/shard0*.log`: median **54.5 s**,
mean **108.7 s**, **34,882 shard-seconds** in total.

So the current cost is **3.80x / 4.01x the baseline mean** and **7.6x / 8.0x the baseline
median** — and that is measured under a per-spectrum cap of **300 s** against the baseline's
1,800 s, so the like-for-like factor is larger than these ratios. The reported ~10.8x
regression is real and unfixed in shipped code.

The cap is also overshot: mean shard time 413–436 s under a 300 s budget.

### 10.2 The pair now running is a truncated evaluation

At baseline speed 23 of 321 spectra (7.2%) ran past 300 s; `clean-before` truncated **0**
rows at its 1,800 s cap. At the measured ~4x, the rows that will hit the new 300 s cap are
those past ~75 s at baseline: **124 of 321 (38.6%)**.

Both arms are truncated identically, so the oracle-vs-predicted *contrast* survives; the
*absolute* Exact@1 of these two runs will **not** be comparable with `clean-before`'s 0.93%
and must be published with its `truncated_spectra` count (§7 rule 6).

A local paired probe on one clean-panel spectrum (`CCMSLIB00000077068`, GPU baseline 28.7 s)
run on CPU on the login host — which is simultaneously carrying three other workflows, so
only the ratio is claimed — gives 409.9 s at `9d0a919` against 314.4 s with the
working-tree lazy probe applied. Both saturate the 300 s cap, so this measures **deadline
overshoot, not throughput**: 1.37x over budget without the per-token deadline check, 1.05x
with it. `/tmp/speed_out_head/head.log`, `/tmp/speed_out_lazy/lazy.log`.

### 10.3 Runs in flight: true state and what was done

| Task | What | True state at 19:25Z | Action |
|---|---|---|---|
| `1adf897039114c38a1c9a06e8384211e` | clean panel 321, **oracle** fingerprint, c8, 6 shards, cap 300 | `in_progress`, 73/321 rows at 5,412 s | **left running**, ETA ≈ 00:05Z |
| `99e971d1ec1644bb80f5e0ed7106e93d` | clean panel 321, DreaMS fingerprint, c8, 6 shards, cap 300 | `in_progress`, 77/321 rows at 6,012 s | **left running**, ETA ≈ 00:20Z |
| `c25c3ac3c6b44e66bbeffe04187ba445` | clean panel 321, DreaMS, **c64**, 6 shards, cap 900 | `queued`, never started | **cancelled** (`stopped`) and removed from queue `sience`, which is now empty with both GPU slots on A and B |

ETAs are computed from the measured 68.9 / 72.7 s per row above, not assumed; they exclude
the shard tail, since the six interleaved shards do not finish together.

**Why C was cancelled.** The 64-candidate budget is already a killed lever (§5: 5.6x compute
for +1.05 pp). At the throughput measured in §10.1 the 803-split c64:c8 compute ratio puts it
near **33 h on 6 shards**, and it was the only entry in queue `sience`, so it would have
blocked the slot that the post-R1 reruns need. Cancelling it costs nothing that §5 has not
already priced.

**Why A and B were not cancelled and relaunched.** Relaunching on *current HEAD* buys nothing
— the payload already is current HEAD, verified by hash above — so the only gain would be
more shards. That gain is bounded: the worker's GPU is a 40 GB A100
(`nvidia-smi` line in the task log, `host=… nproc=64`), each shard holds its own copy of a
2.75 GB checkpoint, and six shards are the known-good configuration; ten would sit at the
edge of the card. Against ~1 h of saving, a relaunch discards the 150 rows already decoded
and risks an OOM that costs the night. The decisive fact is that R1 lands within hours: once
the lazy mask is committed and gated, the same full-panel pair costs a fraction of this, so
the right rerun is a *post-R1* one and not a resharded repeat of the slow code.

### 10.4 The decisive pair is already full-size and matched

Verified rather than assumed:

- Panel: `configs/benchmarks/nplib1_v1/nplib1_val_clean322_v1.tsv`, sha256
  `23426dabf916822ab4961d83e3942c1f59c317d6500480e7c812f2fea58a4a14`, identical in the local
  tree and in the payload tarball; its 321 spec names are exactly the union of the six
  `clean-before` shards. The launcher shards the whole file — no `MARLIN_MAX_SPECTRA`.
- Flags are identical apart from the fingerprint source: `--candidates 8`,
  `--diversity-dropout 0.3 --temperature 1.0 --threshold 0.95 --sample-tokens
  --soft-fingerprint --mass-reachability-prune --forbid-isotope-tokens
  --restrict-organic-elements --isotope-token omit --no-ema --generation-mode block
  --ppm-tolerance 10.0 --eos-boost 1.0 --seed 42 --per-spectrum-seconds 300`.
- The oracle arm's fingerprints are the ClearML artifact `oracle_fingerprints`, which is
  `np.array_equal`-identical to `val/fingerprints.npz['ground_truth']` (396 × 4,096, mean
  46.93 active bits) with a `spectrum_ids` key added for keying. The DreaMS arm reads
  `val/dreams_predictions.npz` key `probs`.

So the requirement "both arms, entire 321-spectrum clean panel, 8 candidates, identical flags
apart from the fingerprint source" is already satisfied by the pair in flight; what is not
satisfied is the shard count and the cap, and both of those are worth fixing only after R1.

### 10.5 One scorer over everything finished

`src/marlin/frigid_convention.py`, FRIGID convention, every denominator "all spectra" unless
named otherwise. Nothing here is new decoding; it is the existing evidence re-scored so the
table is comparable end to end.

| Run | Panel | n | Exact@1 | Exact@10 | Tanimoto@1 | Tanimoto (all candidates) | Formula@1 | Return rate | Truncated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full803-c8-100k` | locked test | 803 | **2.74%** (22) | 3.24% (26) | 0.1697 | 0.1602 | 0.3238 | 0.3786 | 0 |
| `full803-c64-100k` | locked test, partial | 190 | 5.26% (10) | 7.89% (15) | 0.3166 | 0.2660 | 0.5368 | 0.7158 | 1 |
| `clean-before` | clean 321 | 321 | **0.93%** (3) | 0.93% (3) | 0.1575 | 0.1473 | 0.2399 | 0.4143 | 0 |
| `hgate-prefix` | **val396, burned** | 314 | 1.59% (5) | 1.91% (6) | 0.1372 | 0.1302 | 0.2516 | 0.3185 | 0 |
| `hgate-postfix` | **val396, burned** | 316 | 1.58% (5) | 1.90% (6) | 0.1656 | 0.1552 | 0.2563 | 0.4019 | 0 |
| MARLIN paper (DreaMS) | NPLIB1 test | — | 16.94% | 23.54% | 0.55 | — | 0.767 | — | — |

Readings that matter:

- The `full803-c64-100k` row is **190 of 803 spectra**, not a panel result; it is the paired
  subset already reported in §3 and is listed only so its denominator travels with it.
- `hgate-prefix` and `hgate-postfix` ran on `nplib1_val_full396_v1`, which §2 declares burned.
  They are printed here so that nobody re-scores them later believing they are panels. **Do
  not quote them.**
- Two evaluation directories hold no usable result: `clean-after` (3 rows, aborted) and
  `paper-proto-100k` / `paper-protocol-step100000` (0 rows).
- Exact@10 on the clean panel equals Exact@1 exactly, which is §5's reranker verdict
  restated: 3 = 3, zero ranking headroom on the clean panel.

### 10.6 What this wave changes

1. Every evaluation from here runs on the clean 321 panel or the locked 803 split. The 8-,
   32- and 48-spectrum arms are retired; nothing in §10.5 is drawn from one.
2. The 64-candidate arm is gone from the queue and should stay gone (§5).
3. The next full-panel launch waits on R1, and when it goes it should carry `cap = 1800` so
   it is comparable with `clean-before`, and more shards than six — bounded by the 40 GB card
   and one 2.75 GB checkpoint per shard, so roughly 8–10 per GPU, not 24.

## 11. FRIGID prior art and the comparison axis

Appended 2026-08-12. Sections 1–10 are unedited. Everything below is either verified in this
section or explicitly marked unverified. Numbers taken from a README are labelled as such and
are never used as a target.

### 11.1 The comparison target on our axis

Our panels are **NPLIB1/CANOPUS**, not MassSpecGym. They are built through FRIGID's own
CANOPUS loader — `scripts/prepare_marlin_nplib1.py:58` pins
`splits/canopus_hplus_100_0.tsv` and calls `load_spec_data` from
`scripts/benchmark_spec2mol.py:320`, which hands the same split file to
`PresetSpectraSplitter` (`scripts/benchmark_spec2mol.py:354-357`). The raw split is
6,810 train / 401 val / 819 test; our usable panels after fingerprint computation are
6,748 / 396 / 803 (§2). MassSpecGym shares no rows with any of it.

So the comparable claims on **our** axis are:

| Claim | Value | Axis | Status |
|---|---:|---|---|
| MARLIN paper target | 16.94% Exact@1 | NPLIB1 | The number to beat |
| FRIGID README, base | 19.80% Top-1 | NPLIB1 | `README.md:77`, **README only, never reproduced here** |
| FRIGID README, scaled | 25.03% Top-1 | NPLIB1 | `README.md:78`, **README only, never reproduced here** |
| Ours, `full803-c8-100k` | **2.74%** (22/803) | NPLIB1 | Measured (§3, §10.5) |

Numbers that are **not** on our axis and must never be set against 2.74%:
FRIGID README MassSpecGym 16.09% / 18.29% (`README.md:77-78`), FRIGID's measured MassSpecGym
13.86% (`docs/MSMS_ENCODER_BENCHMARK_REPORT.md:370-373`, MIST-binary row of the 1,400-spectrum
paired diagnostic), and the full-MSG 10.97% below. Mixing MassSpecGym and NPLIB1 numbers is
the single easiest way to misreport this project.

### 11.2 Measured versus README, on both sides

FRIGID does not reproduce its own headline either. FRIGID's own full MassSpecGym run over
17,082 eligible spectra at threshold 0.187 measured **exact top-1 0.109706** and
**generated-structure Tanimoto top-1 0.459838**
(`docs/MSMS_ENCODER_BENCHMARK_REPORT.md:362-366`). The README claims 16.09% base / 18.29%
full on the same benchmark (`README.md:77-78`). That is a gap of roughly five points on
FRIGID's own numbers — the same *class* of gap this project suspects behind MARLIN's
reported 16.94%.

The consequence is a rule, not a datum: **compare against measured numbers only.** There is
no measured FRIGID NPLIB1 number available to us at all — the NPLIB1 19.80%/25.03% figures
exist only in the README, and no NPLIB1 reproduction of FRIGID has been run on this machine.
Until one is, the 16.94% paper target is the only NPLIB1 anchor we have, and it is itself a
paper number.

The strongest *honest* FRIGID evidence in the record is not a full-split number at all. It is
the locked 1,024 molecule-diverse gate (FRIGID experiments 19 and 23), which measured
four-source union over DLM control at Tanimoto top-1 `+0.0189` `[+0.0148, +0.0232]` and
Exact top-1 `+0.0127` `[+0.0049, +0.0215]` (`docs/FRIGID_EXPERIMENT_REPORT_RU.md:751-769`
on `origin/research/msg-quality-gates`). The full four-source 17,082 confirmation that report
describes as running **never completed**: Slurm shows `frigid-ctl-0/1000/2000/3000`
(jobs 133–136) COMPLETED and `frigid-ctl-4000` (job 137) plus every `frigid-ctl-5000`…`17000`
and every `frigid-t08-*` (jobs 138–168) **CANCELLED at 2026-07-14T12:15:35**. Coverage
reached roughly 4,000 of 17,082 rows on the control source and **zero** on the
temperature-0.8 source. Experiments 24, 25, 34, 36 and 37 in that report are described as
in flight and are not. Every `FRIGID_*_runs` artifact directory has since been deleted from a
94%-full root filesystem, so the partial shards are gone too. Treat
`docs/FRIGID_EXPERIMENT_REPORT_RU.md` as frozen at 2026-07-13.

### 11.3 Settled negatives that bear on training

These are FRIGID's, measured, and they constrain what a MARLIN training arm may claim to be
new. They do **not** all transfer — the transfer conditions are stated with each.

| Killed | Evidence | What it forbids |
|---|---|---|
| Fine-tuning the decoder on the encoder's *own* predicted fingerprints (FRIGID experiment 8) | Ground-truth Tanimoto top-1 0.3897 → 0.3109, MIST-fingerprint 0.3209 → 0.2796 (`FRIGID_EXPERIMENT_REPORT_RU.md:378-399`; restated `MSMS_ENCODER_BENCHMARK_REPORT.md:381-385`) | Adapting on noisy conditioning alone closes the clean/noisy gap **by getting worse in both regimes**. Any arm that trains only on corrupted conditioning must report the clean arm too, or it is repeating this. |
| A 50/50 clean+noisy mixture objective | 10,000-step mixed adaptation: 0.3486 clean / 0.2870 noisy, still below the 0.3897/0.3209 baseline on both arms (`MSMS_ENCODER_BENCHMARK_REPORT.md:385`) | The obvious repair for the row above — mix clean and noisy — **was already tried and failed both arms.** A mixture is not, by itself, a new idea. |
| Raw probability conditioning | Reported to us as Tanimoto@1 0.1258 with formula success 0.0000. **Unverified:** no surviving artifact for these two numbers was found in either repo. The directionally identical, verified statement is that `mist_binary` conditioning degrades every metric against ground truth on the 64-spectrum paired diagnostic (`DLM_FINGERPRINT_ROBUSTNESS_RESULTS.md:68-78`) | Feeding unthresholded probabilities to a decoder trained on binary fingerprints is a distribution break, not a conditioning improvement. |
| Threshold-only fingerprint processing (FRIGID experiment 9) | threshold 0.50 gave `+0.0118` Tanimoto on 64 spectra, shrinking to `+0.0080` with a CI containing zero on 200; top-32 bits worse than baseline; confidence gate lost on a fresh holdout (`FRIGID_EXPERIMENT_REPORT_RU.md:402-422`) | "Just move the threshold" is closed. Our 0.95 soft gate is admissible only because it is a *parity* change — it makes training match evaluation — not because a higher threshold is expected to help. |
| Encoder swaps, as a class (undocumented FRIGID "experiment 38", 2026-07-15/16, locked 15,325-row molecule-disjoint partition) | MIST 0.5415; DiffMS MIST-512 0.437133; JESTR 0.2778; MS2DeepScore 2.0 0.2277; SpecEmbedding 0.1942; MSBERT 0.1844. **Nothing met the +0.005 gate.** Results live in this repo: `docs/MSMS_ENCODER_BENCHMARK_REPORT.md:7-10,162-205` | No off-the-shelf encoder beats MIST on fingerprint Tanimoto. R7 stands as *processing* only; a lane switch has no supporting evidence. |

The decoder upper bound from the same source is worth keeping in view: on a 1,400-spectrum
paired diagnostic, ground-truth conditioning gave Exact top-1 0.4879 and MIST-binary gave
0.1386 (`MSMS_ENCODER_BENCHMARK_REPORT.md:370-373`). The conditioning gap is the dominant
term over there too, which is the same diagnosis as §4.5.

### 11.4 Two operational traps

**Trap 1 — the FRIGID checkout is on the wrong branch.**
`/home/nikolenko/work/Projects/FRIGID` is on `main` at `65c0855`. Verified: that tree's
`scripts/` contains ten files and **no** `--spec-manifest` / `--start-index` support, no
`submit_*.sbatch`, no fusion, no retrieval, no MolForge tooling. `grep -rn "spec-manifest"`
over the working tree returns nothing. The runnable protocol is
`origin/research/msg-quality-gates` at `e14c6f8`, where `--spec-manifest` appears in
`scripts/benchmark_dlm_fingerprint_robustness.py`, `scripts/finalize_full_four_source.py`,
`scripts/submit_msg_full_retrieval.sbatch`, `scripts/submit_msg_full_molforge_resume.sbatch`
and `src/dlm/utils/benchmark_selection.py`. Anyone running FRIGID from that checkout as it
stands is running a different, weaker protocol than the one the report describes.

**Trap 2 — the config still ships the bug that invalidated two long runs.**
`configs/spec2mol_benchmark_msg.yaml:56` is `randomness: 10.0` on **both** `main` and
`origin/research/msg-quality-gates`. It is only ever corrected on the command line:
`scripts/benchmark_spec2mol.py` overrides it exclusively under
`if args.randomness is not None:` (in `merge_config_with_args`). A run launched "from the
config", with the flag omitted, therefore silently reproduces the discarded setting and looks
like a valid run. Fixing this is out of scope here (the FRIGID tree is read-only for us), but
no FRIGID launch may omit `--randomness`.

### 11.5 Cost, measured against the Slurm record

The report states "примерно `10-14 s/spectrum`" for a DLM full run
(`FRIGID_EXPERIMENT_REPORT_RU.md:785`). The Slurm record disagrees. The four completed
control chunks were 1,000 spectra each on a **whole** A100 (`gres/gpu=1`, not `gres/shard`):

| Job | Name | Elapsed | s/spectrum |
|---|---|---:|---:|
| 133 | `frigid-ctl-0` | 07:06:47 | 25.6 |
| 134 | `frigid-ctl-1000` | 06:45:34 | 24.3 |
| 135 | `frigid-ctl-2000` | 06:40:31 | 24.0 |
| 136 | `frigid-ctl-3000` | 06:56:43 | 25.0 |

Mean **24.7 s/spectrum**, i.e. **1.8×–2.5× the quoted range** — the "optimistic by about 2×"
claim is confirmed, and job 133's 7.11 A100-hours per 1,000 rows is consistent with the
reported ~7.0 A100-hours for a 1,024-spectrum control gate.

**Not confirmed:** the ~9–12.5 A100-hours attributed to the temperature-0.8 source. Every
`frigid-t08-*` job (151–168) shows `Start=None` and `Elapsed=00:00:00` — that source never
ran a single spectrum on this cluster, and `sacct` over 2026-07-01…2026-07-20 lists no other
GPU job longer than two hours besides 133–136 and an unrelated `frigid-spectral-jepa`. The
t08 figure must come from a non-Slurm run or from an estimate; it is not in the record we can
reach. The safe planning number is therefore **~7 A100-hours per 1,000 spectra per DLM
source**, and a two-source 1,024 gate should be budgeted at ≥14 A100-hours with the second
source unmeasured.

### 11.6 Reconciliation with the three queued training arms

All three sit on ClearML queue `sience` (`e0841e72c8a544efa9c54b5e768b1683`), all
`status: queued`, all pinned to entry point `scripts/run_marlin_faro_spectrum_adaptation.sh`
at commit `4fc867a`, all sharing the out-of-fold runtime bundle
`runtime-inputs-spectrum-oof-v1` (sha256 `4305cf00…`), warm-started from the same
`step=100000.ckpt` (sha256 `aed408c7…`), soft fingerprint on, train and validation threshold
both 0.95, 20,000 steps, evaluated on `nplib1_val_clean322_v1` (321 spectra, 8 candidates).

**One fact applies to all three and must be recorded before the individual verdicts.**
None of the three sets `MARLIN_NOISE_PROBABILITY`, so all three inherit the default `0.5`
(`scripts/run_marlin_faro_spectrum_adaptation.sh:96`), which
`src/marlin/training.py:671-681` applies per row every step. **Every arm is already a 50/50
clean+noisy conditioning mixture.** That is the same *shape* as the FRIGID mixture killed in
§11.3. It is not the same *content*: FRIGID's noisy half was MIST's real predicted
fingerprints, while ours is synthetic bit noise over 10–30% of the ON bits
(`src/marlin/noise.py:8-38`), and — decisively — this setting was already in the run that
produced our 2.74% baseline, so it is the incumbent, not a proposal. The honest reading is
that the mixture question is **not open** in any of these arms; none of them tests it, and
none of them may claim it as the novelty.

**T1a — `781de15ee38c408aa2cc068fade83110`, conditioning parity (R4).** Distinct, and it is
the arm to keep. Its only levers against the 2.74% baseline are (a) the out-of-fold
fingerprint bundle, which removes the in-sample probe leak so `train/dreams_predictions.npz`
stops being a rehearsal of its own answers, and (b) `MARLIN_TRAIN_FINGERPRINT_THRESHOLD` and
`MARLIN_VALIDATION_FINGERPRINT_THRESHOLD` both at 0.95 with `MARLIN_SOFT_FINGERPRINT=1`, so
the training gate is the evaluation gate. Neither is what FRIGID rejected. Experiment 8
killed *fine-tuning on the encoder's noisy output*; T1a changes *whose* fingerprints the
model sees at the same noise level, and the direction is toward less leakage, not more noise.
Experiment 9 killed *threshold-tuning as a quality lever*; T1a does not tune the threshold to
find a better one, it copies the evaluation threshold into training to remove a train/test
mismatch — the opposite operation. This arm attacks the measured 0.750-vs-0.547 teacher-forced
gap of §4.5 directly and has an external anchor in MS-BART's 1.71% → 7.45% on NPLIB1. Run it.

**T2 — `3f163b90b6494d0f91762fa8cd92171a`, self-correction on top of parity (R5).** Distinct,
and the most distinct of the three. It is T1a plus
`MARLIN_CONTEXT_CORRUPTION_PROBABILITY=0.5`, warmup 1,000 steps, corrupted fraction 0.05–0.25,
`MARLIN_RESTORATION_LOSS_WEIGHT=0.5`. The corruption here is on the **decoded SAFE prefix**,
not on the conditioning fingerprint: a fraction of clean-stream *content positions* is
replaced by the model's own runner-up tokens while every cross-entropy target stays gold
(`src/marlin/model.py:384-394`). FRIGID never tried this — every FRIGID negative in §11.3 is
about the fingerprint input, and no FRIGID experiment touched the decoder's context stream at
all. It also addresses a defect we measured ourselves and FRIGID never looked for: §4.4's
irreversible commitment, where every step after the first mistake is off-distribution. The one
caveat is that it is confounded with T1a — it changes two things at once — so its result is
only interpretable against T1a's, which means T1a must run and must not be cancelled to make
room for it. Run it, second.

**T1b — `50ef8dcd85e348b6861fdbadea2cecdb`, one-sided training corruption.** Distinct from
FRIGID's negatives, but weakly motivated and the one to cut if compute is short. Its sole
delta from T1a is `MARLIN_FINGERPRINT_NOISE_MODE=dropout`, switching
`symmetric_fingerprint_noise` (drop *n* ON bits, add *n* OFF bits) for
`one_sided_fingerprint_dropout` (drop ON bits only) at
`src/marlin/training.py:671-674`. This is not FRIGID's 50/50 mixture and not experiment 8:
the noise *rate* is unchanged at 0.5, only its *shape* changes, and it is motivated by a real
observation — that the encoder's dominant error is false negatives, with false-negative bit
count correlating −0.5604 with quality delta (`DLM_FINGERPRINT_ROBUSTNESS_RESULTS.md:90`).
So it is a legitimate, non-duplicate hypothesis. But note what it is **not**: §6's R4 asked
for "a paired arm with symmetric noise switched off", and T1b does not switch noise off — it
reshapes it. The genuine noise-ablation control that R4 specified is therefore still missing
from the queue, and T1b is a third variant of a nuisance parameter rather than the control it
is standing in for. **Recommendation: not a cancel-on-duplication call — it duplicates
nothing — but it is the lowest-value of the three.** If the two GPU slots are contended,
deprioritise T1b behind T1a and T2, and if it is requeued, requeue it as
`MARLIN_NOISE_PROBABILITY=0.0` (the true control R4 asked for) rather than as a second noise
shape. Nothing here is cancelled by this document; the decision is the project lead's (§9).

## 12. The oracle-vs-predicted fingerprint verdict — both arms finished, 2026-08-13

Appended 2026-08-13. Sections 1–11 are unedited. This section closes the pair that §10.3
left running and supersedes §10.2's expectation that the pair would be unusably truncated.

### 12.1 True state: both tasks completed, both panels are full

| Task | Name | Status | `MARLIN_RUN_NAME` | Rows landed |
|---|---|---|---|---:|
| `1adf897039114c38a1c9a06e8384211e` | `marlin-B-full-clean-panel-ORACLE-c8` | **completed** 2026-08-13 01:10:23Z, worker `aiagent03:gpu0` | `clean-new-oracle-c8` | **321 / 321** |
| `99e971d1ec1644bb80f5e0ed7106e93d` | `marlin-A-full-clean-panel-dreams-c8` | **completed** 2026-08-13 02:28:30Z, worker `aiagent03:gpu1` | `clean-new-c8` | **321 / 321** |

Neither run is partial. Each of the six interleaved shards wrote its own
`predictions.jsonl` (54/54/54/53/53/53 = 321) plus `metrics.json` and `run_signature.json`.

**Where the outputs actually are.** `/mnt/netstorage/nikolenko/marlin/evaluations/clean-new-*`
does **not** exist on the login host: `aiagent03`'s `/mnt/netstorage` is a different
filesystem. The worker paths the launcher verified —
`/mnt/netstorage/nikolenko/marlin/runs/spectrum-fingerprint-adaptation-3f6a2461d77b4261a916a4d8259be0a5/checkpoints/step=100000.ckpt`
and `/mnt/netstorage/nikolenko/marlin/runtime-inputs-spectrum-v1` — are both absent here,
while the login host's mount is `10.100.10.100:/pool0/ai/datastorage`. The results survived
only because the launcher tars the run directory into the ClearML artifact `results`. Both
tarballs were fetched and unpacked onto the login host at the same run names:

- `/mnt/netstorage/nikolenko/marlin/evaluations/clean-new-oracle-c8/predictions.jsonl`
  sha256 `363779bc5d71995fa319c751a6c4c5af6e6b47f376f472e19006fe4bf6825661`
- `/mnt/netstorage/nikolenko/marlin/evaluations/clean-new-c8/predictions.jsonl`
  sha256 `fabaaea363d3288358923b201554f04568c191745814d2290b9f07fe3f08b80f`

The directories `oracle-vs-dreams-A` (8 rows), `oracle-vs-dreams-B` (9 rows) and
`oracle-vs-dreams-A-aborted-massorder0` (0 rows) on the login host are **not** these runs:
their `run_signature.json` names `/tmp/panel48.tsv` and a 240 s cap. They are the retired
48-spectrum pilots of §10.6 rule 1. Do not quote them.

**Matched flag-for-flag, verified.** Diffing the two `shard00/run_signature.json` settings
blocks leaves exactly three keys: `fingerprint_key` (`ground_truth` vs `probs`), the shard
manifest path, and `git_commit`. The commit divergence
(`cc9f064a…` vs `d35d615f…`) is an artefact, not a code difference: both arms unpack the
**same** payload artifact from the **same** payload task `3993339892da42c9be34c7e76022376e`
(`code.tar.gz` sha256 `30cd5d937b23…`, §10.1), and the launcher then runs `git init && git
commit` inside the container because `evaluate_marlin_nplib1.py:224` records provenance with
`git rev-parse HEAD`. A fresh commit hash depends on its timestamp, so two containers
committing byte-identical trees one minute apart necessarily disagree. The six per-shard
`ordered_spec_names_sha256` values are **pairwise identical between the arms**
(`d88435d0…`, `24aba668…`, `485eed32…`, `51718198…`, `f0427d8d…`, `b39d1229…`), which is the
statement that both arms decoded the same 321 spectra in the same order. Both ran cap 300 s,
8 candidates, checkpoint `aed408c7…` (`control-r2 step=100000`).

### 12.2 Both arms, one scorer

`src/marlin/frigid_convention.py`, every denominator "all spectra" unless named. The
worker-side `frigid_convention` artifact of each task reproduces these values exactly.

| | Oracle fingerprint | DreaMS fingerprint | `clean-before` (DreaMS, cap 1800 s) |
|---|---:|---:|---:|
| Spectra | 321 | 321 | 321 |
| **Exact@1** | **19.00% (61/321)** | **1.25% (4/321)** | 0.93% (3/321) |
| Exact@10 | 19.00% (61/321) | 1.25% (4/321) | 0.93% (3/321) |
| Tanimoto@1 (FRIGID convention) | 0.3973 | 0.1527 | 0.1575 |
| Tanimoto, all returned candidates | 0.3364 | 0.1418 | 0.1473 |
| Formula@1 | 0.4984 | 0.2617 | 0.2399 |
| Candidate return rate | 59.50% (191/321) | 40.81% (131/321) | 41.43% |
| Never-matched rate | 40.50% | 59.19% | 58.57% |
| **Truncated spectra** | **261/321** | **298/321** | 0 |
| Attempts (mean / to match) | 8.0 / 8.0 | 8.0 / 8.0 | — |
| Repair provenance | absent (pre-dates the check) | absent | absent |
| Total shard time | 38.3 h | 43.5 h | 9.7 h |

95% Wilson intervals on Exact@1: oracle **[15.09%, 23.65%]**, DreaMS **[0.49%, 3.16%]**.
They do not overlap and are not close to overlapping.

Exact@10 equals Exact@1 in both arms, which is §5's reranker verdict restated at a much
higher accuracy: when a correct structure is in the returned set it is already ranked first,
so there is still zero ranking headroom — 61 = 61 and 4 = 4.

### 12.3 The paired contrast

The two arms cover the identical 321 spec names, so the pairing is complete — no spectrum is
dropped and no unpaired comparison is involved.

|  | DreaMS correct | DreaMS wrong |
|---|---:|---:|
| **Oracle correct** | 4 | **57** |
| **Oracle wrong** | **0** | 260 |

Every single spectrum the DreaMS arm solved, the oracle arm also solved. The discordance is
57–0. Exact McNemar, two-sided: **p = 1.39e-17** on 57 discordant pairs. Paired difference
**+17.76 pp**, 95% CI **[13.58, 21.94] pp**. Paired Tanimoto@1 difference **+0.2446**
(sd 0.3373, se 0.0188, t = 12.99, n = 321).

**This is not inside the noise.** It is roughly nine paired standard errors.

Three progressively harsher subsets, each removing a possible confound:

| Subset | n | Oracle Exact@1 | DreaMS Exact@1 | Discordance | Exact McNemar p |
|---|---:|---:|---:|---|---:|
| Whole panel | 321 | 19.00% (61) | 1.25% (4) | 57–0 | 1.39e-17 |
| Both arms returned a candidate | 105 | **38.10% (40)** | 3.81% (4) | 36–0 | 2.91e-11 |
| Neither arm truncated | 12 | 75.00% (9) | 33.33% (4) | 5–0 | 6.25e-02 |

The middle row is the one that matters: restricted to the 105 spectra where **both** arms
returned a molecule at all, the oracle arm is right 10.0x as often, and mean Tanimoto@1 is
**0.7070 vs 0.4243**. So the gap is not merely a yield gap — conditioned on the decoder
having produced something, the answer under the true fingerprint is right ten times more
often and much closer even when wrong.

### 12.4 What the truncation does and does not cost

§10.2 predicted the cap would make these runs unusable. It over-predicted, and the reason is
measurable rather than assumed.

- Every spectrum in both arms recorded `attempts = 8`. All eight candidates were launched in
  a single batch, so no spectrum lost a *batch* of its budget; `truncated` here fires at
  `src/marlin/sampler.py:463-464` and `:585-591`, meaning the deadline elapsed while
  candidates inside that one batch were still decoding, and those unfinished rows are
  abandoned. So truncation costs candidates, not attempts, and it costs them silently.
- The size of that cost is measurable against `clean-before`, which is the **same panel, same
  checkpoint, same DreaMS fingerprint, 1,800 s cap, 0 truncations**: 0.93% (3/321) there
  against **1.25% (4/321)** here. Six times the time budget bought the DreaMS lane nothing
  distinguishable from zero. The 300 s cap is therefore not what is holding the predicted-
  fingerprint arm at 1%.
- Truncation is itself an *effect* of the arm, not only a handicap on it: 261 oracle vs 298
  DreaMS spectra truncated (paired: 250 both, 48 DreaMS-only, 11 oracle-only). Worse
  conditioning produces longer, deader decodes. The 29 exact matches the oracle arm scored on
  *truncated* spectra against the DreaMS arm's **0 of 298** says the oracle arm wins even
  where the clock is against it.

Consequently the honest statement is narrower than "not comparable": the **DreaMS** arm's
1.25% is comparable to `clean-before`'s 0.93% (the cap costs it nothing), while the
**oracle** arm's 19.00% is a **floor**, not a ceiling — 261 of its 321 spectra were still
decoding when time ran out, and on the 60 it finished it scored 32 (53.3%). The direction of
the remaining bias is known and it is against the headline.

### 12.5 Verdict: the conditioning is the ceiling, the decoder is sound

The question §10.4 posed was which of the two is the binding constraint. The answer is not
ambiguous.

Under the true fingerprint, this decoder — the *same* `control-r2 step=100000` weights, the
same grammar mask, the same mass shell, the same 8 candidates, and a time cap that costs it
261 truncations — reaches **19.00% Exact@1 on the clean 321 panel**, above MARLIN's paper
16.94% and level with CoRe-Gen's measured 19.54% on our axis. Under the DreaMS fingerprint it
reaches 1.25%. **The decoder is not the binding constraint. The fingerprint it is conditioned
on is, and it accounts for essentially the entire distance to the state of the art on this
axis.**

The three qualifications that keep this honest:

1. **19.00% is not a benchmark number.** It is a ceiling probe: an oracle fingerprint is not
   available at inference. MARLIN's 16.94% and CoRe-Gen's 19.54% are *predicted*-fingerprint
   numbers. The comparable cell of our table is **1.25%**, and that is the number that has to
   move.
2. This bounds what better *decoder training* alone can buy on the present conditioning: the
   oracle arm shows the weights already contain a 19% solution, so the remaining decoder-side
   headroom at fixed conditioning is the part of the 1.25 → 19.00 gap that a decoder can close
   by becoming robust to a wrong fingerprint — which is exactly the objective §5 of
   `TRAINING_RECIPE_FINDINGS.md` ranks first (frequency-aware fitted corruption at
   pretraining scale) and second (distillation against the oracle-conditioned distribution of
   this very decoder). Those two now have a measured target rather than a hope: the oracle
   arm **is** the teacher, and its output on these 321 spectra is on disk.
3. It does **not** license "fix the encoder instead". The project's objective is to move the
   number by training the decoder; the finding redirects *which* decoder objective, from
   "decode better" to "decode correctly under a corrupted fingerprint", and it prices that
   objective at up to +17.8 pp on this panel.

### 12.6 Sample size

The current sample is more than sufficient — this is the rare case where no more data is
needed. The paired design on n = 321 with 57–0 discordance gives p = 1.39e-17; even the
harshest confound-free subset (105 spectra where both arms returned) gives 36–0 and
p = 2.91e-11. The 95% CI on the paired difference, [13.58, 21.94] pp, excludes zero by more
than six standard errors.

For calibration of *future* arms on this panel, at α = 0.05 and 80% power with one-directional
discordance, a paired McNemar on the clean 321 panel resolves a **3.0 pp** difference
(n ≈ 262 needed) and, at the limit, a **2.45 pp** one; it cannot resolve **2.0 pp** (n ≈ 393),
**1.0 pp** (n ≈ 785) or **0.5 pp** (n ≈ 1,570). So the clean 321 panel is the right instrument
for effects of ~3 pp and larger — which covers every lever in §5 of
`TRAINING_RECIPE_FINDINGS.md` including CoRe-Gen's −4.77 pp corruption ablation — and the
locked 803 test split is required for anything smaller. Note that the only sub-3 pp row
already measured (`clean-before` 0.93% vs this DreaMS arm 1.25%, +0.32 pp) is correctly
reported above as indistinguishable from zero.

### 12.7 What this changes

1. **§5's ranking is confirmed by measurement, not argument.** Fingerprint-corruption
   robustness moves from "CoRe-Gen says −4.77 pp" to "worth up to +17.8 pp here". It is the
   first experiment.
2. **The oracle arm is now a teacher, not just a control.** Distillation (`TRAINING_RECIPE_
   FINDINGS.md` §5.2) has its teacher outputs already computed on the panel it will be scored
   on: `clean-new-oracle-c8/predictions.jsonl`.
3. **Any future arm must publish `truncated_spectra`** (§7 rule 6) and, given §12.4, must also
   publish `attempts` — a run can be 81% truncated and still have spent every attempt, and the
   two words mean different things.
4. **Results must be harvested from the ClearML `results` artifact, not from
   `/mnt/netstorage`.** The worker's netstorage is not the login host's. Any launcher that
   writes only to `$ROOT` and does not tar it into an artifact loses its run.

## 13. Campaign state, 2026-08-13 evening — what is built, what is launched, what is refused

Written after four specification agents, three build agents and three adversarial
reviewers. The reviewers changed the plan more than the specs did; §13.2 records
which arms died and why, because an arm killed before it burns a fleet-day is the
cheapest result this project can buy.

### 13.1 The fleet is not what the plan assumed

| Resource | Assumed | Measured, 2026-08-13 17:00–17:35 |
|---|---|---|
| Local A100 80 GB | free, 0 MiB | **held by another user**: pid 1993492, `m.isangulov`, 7,587 MiB, 98–99% util |
| Slurm `gpu`/`gpu-shared` | a separate pool | **the same card.** `hostname` = `spectrum`; `scontrol show node spectrum` reports the same 24 cores and `gpu:a100:1` |
| ClearML `sience`, 2 slots | free | **both busy**: `aiagent03:gpu0` = T1a, `aiagent03:gpu1` = T1b |

So the whole fleet is one contended A100 plus two occupied queue slots. Slurm
reports `spectrum` IDLE while a bare non-Slurm process holds the card at 99%,
which is why "the scheduler says idle" is not the test to use.

**Measured queue rate.** T1a advanced 280 → 311 iterations in 120 s = **0.258
steps/s** at global batch 256 (66 mol/s). Its 20,000 steps therefore finish in
**~21.2 h**, and both slots are unavailable until then. That also settles half of
the unexplained 10.6× throughput question in `TRAINING_RECIPE_FINDINGS.md`: the
real queue rate is 0.258 steps/s, not the 2.30 measured on a free 80 GB card, so
any A100-hour budget written at 2.30 is optimistic by ~9× **on the queue**.

### 13.2 Arms a reviewer called fatal, and what happened to them

| Arm | Verdict | Action |
|---|---|---|
| Corpus stage-1 A (fitted) vs B (rate-matched control) | best design in the set, but unlaunchable | **not launched.** The integration (`fitted` branch in `training.py`, `Fp2MolStream` wiring) does not exist: grep for `Fp2MolStream\|EncoderErrorModel` in `src/marlin/training.py` returns nothing. And the ClearML worker's `/mnt/netstorage` is a per-worker local ext4, so the 67 GB corpus is not visible from the only compute that could take it. |
| MIST vs DreaMS paired decode | six variables at once; 87 sh (clean 321) to 285 sh (locked 803) for a central estimate at or below panel resolution | **replaced** by the forward-pass lane probe, §13.4. |
| Formula Stage A (A0/A1/A2) | ceiling 1.81 pp against a 2.45 pp panel resolution — arithmetically unreadable | **not launched.** |
| CoRe-Gen formula-distance reranker | effect is exactly zero by measurement (Exact@10 == Exact@1 in both §12.2 arms) | **not launched**; it would claim nothing. |
| Any decode at cap 300 on the pre-R1 tree | `clean-new-c8` spent 43.46 shard-hours to truncate 298/321 | **fixed at the root**, §13.3. |

### 13.3 R1 landed: a token is now committed without building the support

`a766389`. The block lane masked all 1,880 tokens at every position, and that call
is the decode's wall clock. Measured on this host over real val prefixes at the
run's own mask settings:

| prefix length | full support | one `admits` probe | ratio |
|---:|---:|---:|---:|
| 30 | 223.2 ms | 0.233 ms | 958× |
| 50 | 76.3 ms | 0.307 ms | 249× |
| 65 | 123.6 ms | 0.565 ms | 219× |
| 80 | 1101.4 ms | 4.505 ms | 244× |

against a ~10 ms model forward. Both selection rules read the masked distribution
only through an argmax and restricting the support rescales the survivors by one
positive constant, so walking the ranked tokens and stopping at the first the
grammar admits commits the same token; `scripts/audit_lazy_probe_identity.py`
asserts that on replayed decode prefixes under both argmax and multinomial
selection, and asserts the generator ends in the same state.

**Acceptance gate on the merged tree**: test 803 targets / 45,351 positions / 0
gold rejections; train 6,748 / 386,553 / 28. Totals **7,551 / 431,904 / 28, 0 on
test** — the constants of §4.7. **581 tests pass.**

### 13.4 Ranking the conditioning lanes without buying a decode

Four specs each wanted a paired decode to choose a conditioning lane. The ordering
question is answerable with forward passes: `scripts/probe_conditioning_lanes.py`
runs the existing teacher-forced conditioning probe once per lane over one fixed
256-row set drawn from the locked 803, so lanes differ in nothing but which
fingerprint file is read. Two checkpoints, because a lane comparison on a
checkpoint adapted 100,000 steps to one lane's bit vocabulary is biased toward
that lane; two probe seeds, because a gate with no measured spread is not a gate.

Read this as an **ordering**, not a price: the map from per-token top-1 to Exact@1
is steep and non-linear (0.547 → 1.25%, 0.750 → 19.00%, §12), and these absolute
values are not comparable to §3's 24-molecule anchors because the probe masks a
different fraction of positions.

Lanes probed, all on the locked 803 (the only panel where every lane is honest —
the clean 321 is drawn from val396, which is MIST's early-stopping fold):

| Lane | Source | Threshold |
|---|---|---|
| `true` | `runs/true_fingerprints/test/fingerprints.npz` (gold, 46.34 mean on-bits) | 0.5 |
| `dreams095` | `runs/dreams/probe/test_predictions.npz` | 0.95 |
| `mist_formula_blind_010` | `runs/mist_cf_peakformula_fingerprints_job693/fingerprints.npz` | 0.10 |
| `mist_oracle_formula_020` | `runs/mist/test/fingerprints_with_ids.npz` | 0.20 |

The fourth lane is labelled **oracle** deliberately. `runs/mist/*` was produced
with ground-truth-formula subformulae; the deployable MIST lane is the
formula-blind job693 export. Their medians differ by 0.16 Tanimoto (0.549 against
0.393), so any "MIST buys +23.6 pp of conditioning" claim taken from `runs/mist`
is an oracle-formula number and must not be used to justify a decode.

Results land in `runs/conditioning-lanes/test803_lanes_v1.json`; the run was still
executing when this section was written. Read it with the two guards the script
enforces: `probe_top1_true` must be identical across lanes on a given checkpoint
(it is the same gold fingerprint regardless of which predicted file is loaded, so
a difference there means the row sets drifted), and the per-seed spread must be
smaller than any lane gap being claimed.

### 13.5 The corruption model no longer contains the panel it will be scored on

`58aaaa0`. `report.json` recorded `fit_split: "nplib1 locked test 803"` and
`validation_split: "nplib1 val 396"` — the headline panel and the superset of the
iteration panel were both inside the fit. The fit split is now an argument and the
model is refitted on the 6,748 **out-of-fold** train DreaMS predictions
(`runs/dreams/probe/train_predictions_oof.npz`), validated on the locked 803.

Three things that refit establishes:

1. **The out-of-fold train predictions are not leaky.** Median Tanimoto 0.3000
   against the test split's 0.3043, KS 0.0293 (p 0.56). The 0.444 that made the
   train split unusable belongs to the in-sample probe, not to the OOF one. So
   train is a legitimate fit surface and the panels can stay outside the fit.
2. **The frequency dependence replicates out of fold.** Sensitivity rises
   **0.239 → 0.992** across corpus-frequency deciles and the false-positive rate
   **0.00074 → 0.889**, against the in-fold 0.252 → 0.991 and 0.00079 → 0.872.
   "The incumbent noise is structurally inverted" survives the leak removal.
3. **The level fit degrades, honestly.** KS against the held-out 803 is **0.0738
   (p 6.65e-4)** where the in-fold fit scored 0.0503 (p 0.315), against a real
   train-vs-test noise floor of KS 0.0293. The model is measurably imperfect out
   of fold — and still far closer than the rate-matched control (0.2214) or the
   incumbent symmetric (0.8727) and dropout (0.9308) noise.

Artefacts, `/mnt/netstorage/nikolenko/marlin/artifacts/encoder-error-model-oof-v1/`:
`encoder_error_model_oof.npz` `815da77b47ef1f07e3e4948e229bfcbad3f8181e82d5720377ee25eafb089748`,
`encoder_error_model_oof_control.npz` `f99b512febc35f90aeddeb6a47344ca6127395a8ba50788e19d698c8e8f578f3`,
`report_oof.json` `cdf51ba6fafa4e04259969ca63a97d21986b49b527b2e9b00b776d490f315dac`.

### 13.6 The corpus stream can no longer train on the panel

`bfedcba`. Measured here: `data/nplib1_test_inchikeys.csv` covers 701/701 test
connectivity blocks, **0/394 val blocks and 0/320 clean-panel blocks**, and
`Fp2MolStream` defaulted `exclude_inchikeys` to None and skipped the check on an
empty set. In the first 10,000,000 fp2mol rows — one paired stage-1 experiment —
the corpus carries 74 of the 701 locked-test blocks and **23 of the 320
clean-panel blocks**, against a 0.93% (3/321) baseline and a sought effect of
about 15 spectra. The stream now refuses to start without a list, refuses a list
that excludes nothing, and `configs/marlin_nplib1.yaml` points at
`data/nplib1_holdout_inchikeys_v2.csv` (1,095 blocks, sha256
`7d1f45937f284dbc9dc93be0ff6ae6eedf02b1cc293496acff7ffcd8c5dab44a`, rebuilt by
`scripts/build_nplib1_holdout_inchikeys.py`). When comparing to CoRe-Gen's 19.54%,
say that this excludes more than they do: the paper removes test overlap only.

### 13.7 Run table

| Run | Where | Task / job | Control | Answers | State |
|---|---|---|---|---|---|
| T1a `marlin-T1a-…-20k-fixedrecipe-r1` | ClearML `sience`, aiagent03:gpu0 | `356c0d926d004d28a78f6f97e07afe5c` | T1b | recipe fixes + symmetric noise | in flight, 0.258 steps/s, ETA ~21.2 h |
| T1b `…-dropoutnoise-20k-fixedrecipe-r1` | ClearML `sience`, aiagent03:gpu1 | `a6e928d83acf40ccab31bb665f4eae62` | T1a | same, dropout noise | in flight |
| conditioning-lane probe | local CPU, 12 threads | `runs/conditioning-lanes/test803_lanes_v1.json` | `true` lane and `frigid-warmstart-step0` checkpoint | which fingerprint lane to condition on | finished |
| encoder error model, OOF refit | local CPU | `artifacts/encoder-error-model-oof-v1` | rate-matched uniform control | does the inversion survive leak removal | finished |
| FRIGID reference, clean 321 | Slurm `gpu`, node `spectrum` | job **786**, array 0-1 | arm 0 fingerprint-only vs arm 1 formula+fingerprint | the number our re-implementation must beat | running, ~11 min/arm |

**The FRIGID reference costs minutes, not hours.** The brief's cost model —
24.7 s/spectrum, ~7 A100-hours per 1,000 spectra — does not hold for this
configuration. Measured on job 786 at 8 candidates: **2.1 s/spectrum**, so the
whole clean 321 panel is ~11 minutes per arm and the pair is ~22 minutes on one
card. Every plan that priced a FRIGID-side comparison in A100-hours was pricing
it ~700× too high. Two operational notes paid for in a wasted submission:
`evaluate_frigid_parity.py --max-spectra` **defaults to 4**, so an sbatch that
omits it silently evaluates four spectra and reports a metrics block that looks
complete (job 782 did exactly this); and the script refuses to reuse an output
directory, which is what caught it.

The gate that launched it is `scripts/submit_when_gpu_idle.sh`: it polls for
compute processes owned by another user and submits only when there are none.
The card cleared at 17:47 and the job went in unattended, without ever competing
with the run that held it.

Both T1a and T1b warm-start from `control-r2 step=100000`
(`aed408c7d2c01c86a4b257e5119c28b11404971c3df3fd09a76c054ef0e7b14f`), not from
the released DLM. They therefore price the recipe fixes *on top of* a checkpoint
that already absorbed 3,846 replays at ~190× its terminal lr. That is a real
confound in the "training is what did it" claim and the released-DLM arm the
reviewers asked for (arm D) is still owed.

### 13.8 What is owed before the headline pair can run

1. Integrate `Fp2MolStream` and `EncoderErrorModel` into `src/marlin/training.py`
   behind `--fingerprint-noise-mode fitted|fitted_control`, and set **stage 2 to
   the same corruption mode as stage 1**. As specified, stage 2 runs the
   incumbent symmetric noise at p=0.5 for 20,000 steps against stage 1's 5,000 —
   50–80% of each arm's optimizer steps would run the measured-inverted law and
   the arms would converge by construction.
2. Stage the corpus where the compute is, or run stage 1 on the local card. A
   5,120,000-row subset is ~395 MB and `Fp2MolStream` takes an explicit
   `row_groups` list.
3. Add **arm D**: released DLM → recipe-fixed NPLIB1 stage 2 only, no corpus. D→B
   prices the corpus, A−B prices the placement law, D alone prices the recipe
   fixes. Without D a result of A=4.5%, B=1.5% cannot separate "the frequency-aware
   law worked" from "5.12M distinct molecules worked".
4. Emit a per-row `seen_in_corpus` flag in the prediction file and report Exact@1
   on the held-out complement as well as overall. With the §13.6 exclusion in
   place it should be identically zero; if it is not, the exclusion is not wired
   and the run is void.
5. Re-price one clean-panel shard end to end on the post-R1 tree before buying any
   paired decode, and evaluate at cap 1800, never 300.

## 14. Run order, decided 2026-08-13 14:53Z

Appended after §13. Sections 1–13 are unedited. §13.7's run table is superseded on
one row (T1b) and one row is added (T2); everything else in §13 stands. Every state
below was read from the ClearML API today, not from a plan.

### 14.1 What is now true, in the four terms the wave was set

**1. The recipe is fixed and the arms are running on it — verified in the running
tasks, not in the code.** Both surviving arms carry, in their own
`resolved_config`: `lr_schedule = warmup_cosine`, `learning_rate =
3.164707848388532e-07`, `lr_min = 5.2697058404552555e-08`, `lr_warmup_steps =
1000`, `loss_reduction = token_mean`, `time_sampling = per_sequence_antithetic`,
`fp32_forward = True`, `probe_early_stopping_metric = probe_top1_predicted`
(patience 4, probe interval 1,000), global batch 256, `max_steps = 20000`, warm
start `aed408c7…`. T1a's own log at 14:00:49.959Z prints
`MARLIN adaptation learning rate: schedule=warmup_cosine peak=3.16471e-07 (6.0x
the released terminal 5.2697058404552555e-08), displacement bound 0.0036653`.

The `--fp32-forward` memory risk did not materialise: T1a's task monitor reports
**8.65 GB used of 40.0 GB** (31.35 GB free) on `aiagent03:gpu0`.

**Measured cost of the fixed recipe.** T1a advanced step 675 → 752 between
14:44:59Z and 14:50:46Z = **4.51 s/step**; T1b advanced 288 → 364 over the same
349 s = **4.59 s/step** (both windows read from `Training metrics/learning_rate`
iterations). §13.1's 0.258 steps/s (3.88 s/step) was a 120 s window; over ~350 s
the rate is 4.5 s/step. Against the old bf16 recipe's **3.292 s/step** (100,000
steps in 91.45 h, ClearML `3f6a2461d77b4261a916a4d8259be0a5`) the fixed recipe
costs **1.37×**, so **20,000 steps = 21.2–25.1 h of training per arm** before any
in-training evaluation. Both windows were measured with two arms sharing the node;
T1a now runs alone and should be re-measured.

**2. The speed regression: the overshoot is gone, the throughput is still
unmeasured — and the running arms do not have R1 at all.** Same spectrum, same
host, same flags, same 300 s cap, one process on CPU:

| Tree | `runtime_seconds` | overshoot | outcome |
|---|---:|---:|---|
| `9d0a919` (pre-R1) | 409.860 | +36.6% | truncated, 0 candidates |
| working-tree lazy probe, 2026-08-12 | 314.430 | +4.8% | truncated, 0 candidates |
| **HEAD `58aaaa0`** (R1 committed at `a766389`) | **300.314** | **+0.10%** | truncated, 0 candidates |

`/tmp/speed_out_head`, `/tmp/speed_out_lazy`, `/tmp/speed_out_r1`, spectrum
`CCMSLIB00000077068`. So the deadline is now honoured to 0.3 s where it used to be
missed by 110 s, which is what §10.1 called the cap overshoot. It is **not** a
throughput number: the spectrum still spends its whole budget and returns nothing,
so §13.8 item 5 — re-price one clean-panel shard end to end on the post-R1 tree —
is still owed and every decode cost below is quoted as unpriced until it lands.

**The arms in flight are pinned to `94290c032a051bad49d7727fe744d2b59cfe3c5d`,
which predates `a766389`.** Their in-training panel evaluations therefore run the
pre-R1 decoder, at 16 shards, on the 321-spectrum clean panel, **with no
per-spectrum cap at all**: nothing sets `evaluation.per_spectrum_seconds`, and
`src/marlin/periodic_evaluation.py:153-155` passes `--per-spectrum-seconds` only
when it is set. That is four blocking evaluations per arm (`evaluation_interval =
checkpoint_interval = 5000`) whose cost is unknown and whose result nothing reads:
`evaluation/selection/enabled = False`, and early stopping runs off the
conditioning probe. See §14.4 risk 1.

**3. The oracle-vs-predicted verdict is decisive and closed.** §12: 19.00% (61/321)
oracle against 1.25% (4/321) DreaMS on the same 321 spectra, same weights, same
flags; discordance **57–0**, exact McNemar **p = 1.39e-17**, paired difference
**+17.76 pp** [13.58, 21.94]; on the 105 spectra where both arms returned anything,
38.10% against 3.81%, 36–0, p = 2.91e-11. **n = 321 paired.** No further sample is
needed (§12.6). The decoder is sound; the fingerprint it is conditioned on is the
binding constraint, and the decoder-side objective is therefore *decode correctly
under a wrong fingerprint*, not *decode better*.

**4. The corpus pipeline is built and measured but not connected to a trainer.**
Corpus, error model, control, leakage guard and their audits all exist and are
committed (§13.5, §13.6). What does not exist is the join:
`src/marlin/training.py:602` still accepts only `{"symmetric", "dropout"}`, and
neither `Fp2MolStream` nor `EncoderErrorModel` is imported by `training.py`,
`scripts/train_marlin_spectrum_adaptation.py` or the launcher. Two further gaps
that only bite on the worker: the corpus is on the login host's NFS and the
worker's `/mnt/netstorage` is a different filesystem (§12.1, §13.1), and the
exclusion list `configs/marlin_nplib1.yaml:32` points at
`/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/data/nplib1_holdout_inchikeys_v2.csv`
— **outside the git repo**, so a worker that gets its code by `git clone` does not
have it. Verified present on the login host: 1,095 blocks, sha256
`7d1f45937f284dbc9dc93be0ff6ae6eedf02b1cc293496acff7ffcd8c5dab44a`.

### 14.2 Two changes made to the fleet today

**T2 was never rejected by the recipe — it lost a `git clone`.** Task
`d7771805d55242fcbf4b23a412b11aaf` died at 13:55:26Z, five minutes in, on
`error: RPC failed; curl 92 HTTP/2 stream 0 was not closed cleanly` /
`fetch-pack: unexpected disconnect` / `fatal: early EOF` while cloning
`https://github.com/SergeiNikolenko/FRIGID.git`. Nothing in the arm is wrong. It
was **reset and re-enqueued to `sience` at 14:52Z** and is `queued`, still pinned
to `94290c0` so that its only training delta from T1a stays
`context_corruption_probability 0.5` against `0.0`.

**T1b was stopped at step 400 and its slot given to T2.** Task
`a6e928d83acf40ccab31bb665f4eae62`, `stopped` at 14:53:07Z. §11.6 pre-registered it
as the arm to cut if compute were short, and today's out-of-fold refit prices the
cut: T1b's whole delta from T1a is `fingerprint_noise_mode = dropout`, and against
the real held-out DreaMS Tanimoto distribution dropout noise is the **worst** of
the four laws measured — KS **0.9308** (median 0.8070) against symmetric 0.8727
(0.6667), rate-matched control 0.2214 (0.3333) and the fitted model 0.0738
(0.2933), with the real test distribution at median **0.3043**
(`artifacts/encoder-error-model-oof-v1/report_oof.json`). Spending 25 GPU-hours to
reshape a corruption law that is 2.65× too mild, while the law that fits is
already fitted and waiting for a trainer, is the wrong use of the second slot.
T1b is recoverable at any time: clone the task and enqueue it.

### 14.3 The queue

Two GPU slots on ClearML `sience` (`aiagent03:gpu0`, `gpu1`). Training has both.
GPU-hours are quoted at the measured 4.51 s/step; decode costs are marked unpriced
until §13.8 item 5 lands. Every arm names the control it is read against, because
an arm without one cannot support the claim this project exists to make.

| # | Run | Control it is read against | GPU-hours | State |
|---:|---|---|---:|---|
| 1 | **T1a** conditioning parity, fixed recipe, 20k `356c0d926d004d28a78f6f97e07afe5c` | the incumbent it warm-starts from (`control-r2 step=100000`, 2.74% test / 1.25% clean panel), and arm D for the warm-start confound | ~24.1 remaining + 4 uncapped panel evaluations | **running**, gpu0, step 752 at 14:50Z |
| 2 | **T2** self-correction on top of parity, 20k `d7771805d55242fcbf4b23a412b11aaf` | T1a — same commit, same recipe, same corpus; the only training delta is `context_corruption_probability` 0.5 vs 0.0 | 25.1 + 4 evaluations | **queued**, takes gpu1 when T1b's container exits |
| 3 | **D** — released DLM → recipe-fixed adaptation, 20k, no corpus | T1a. D vs T1a prices the corrupted warm start; D alone is the only arm that can say *the recipe fix* moved the number (§13.7, §13.8 item 3) | 25.1 | **not submitted.** Needs only `DLM.ckpt` staged on the worker — no new code |
| 4 | **C-A1 / C-A0** — fitted frequency-aware corruption vs rate-matched uniform control, fp2mol stage 1, 10k steps each | each other: `EncoderErrorModel.rate_matched_uniform_control()` holds pooled sensitivity and pooled FPR fixed and removes only the frequency dependence and the row latent, which is exactly CoRe-Gen's −4.77 pp ablation | 12.5 each, **25.1 for the pair** | **blocked on integration** (§14.1 point 4, §13.8 items 1–2, 4) |
| 5 | Locked-803 scoring of whichever checkpoints survive 1–4, cap 1800, post-R1 tree | the stored `full803-c8-100k` (2.74%) | unpriced | **blocked** on §13.8 item 5 |
| 6 | Distillation KL anchor against the oracle-conditioned teacher | T1a; teacher outputs already on disk at `clean-new-oracle-c8/predictions.jsonl` | unpriced, unwired | not designed |
| — | ~~T1b one-sided noise~~ | — | — | **stopped at step 400**, §14.2 |

Why this order and not another:

- **1 and 2 are already paid for.** T1a is 3.8% of the way through a 25-hour run
  and is the control every later arm is read against; stopping it to start
  something else throws away the only fixed-recipe baseline we will have.
- **3 before 4 because it is launchable and 4 is not.** The §13.8 integration is a
  day of code with no GPU in it, so the honest schedule runs D in the first free
  slot and C-A in the second the moment the join lands. If the integration is
  finished before a slot frees, swap them: C-A carries the larger expected effect
  (§12.7: up to +17.8 pp is on the table; CoRe-Gen prices its own corruption at
  −4.77 pp when ablated), D carries the attribution.
- **5 is not optional.** The clean 321 panel resolves **2.45 pp** at best (§12.6)
  and the number to move on it is **1.25%**. An arm that buys +1.5 pp is real and
  unreadable there. The locked 803 resolves +1.25 pp at 2% discordance. Every
  headline claim from arms 1–4 has to end on the 803.
- **6 last** because it is the only entry with neither code nor a price.

### 14.4 What could waste the next day of compute

1. **The uncapped, pre-R1, 16-shard periodic evaluation.** T1a reaches step 5,000
   around 19:30Z today and will fork 16 evaluators over the full 321-spectrum
   panel at commit `94290c0`, with no per-spectrum deadline, blocking training
   until they finish (`_run_sharded` waits on every child). Nothing consumes the
   result. It is fail-soft (`periodic_evaluation.py:421-427`) so it cannot kill
   the run, but it can eat the night. **Watch T1a's first one; if it costs more
   than ~2 h, requeue both arms with `MARLIN_EVALUATION_INTERVAL` =
   `MARLIN_CHECKPOINT_INTERVAL` = 20000** (they must stay equal —
   `periodic_evaluation.py:52` is what killed the first submission) and rely on
   `last.ckpt`.
2. **GPU memory at that same moment.** The callback calls
   `torch.cuda.empty_cache()` and then forks 16 evaluators at roughly 1.8 GB each
   (its own docstring, measured on twelve holding 22 GB) onto a card with 31.35 GB
   free — about 29 GB of shards against 31.35 GB — and the trainer then has to
   allocate its own working set back.
3. **Worker disk.** `aiagent03` reported `disk_free_percent = 8.04` while T1a ran.
   Each checkpoint upload is 1,722.69 MB and each arm writes step 5,000 / 10,000 /
   15,000 / 20,000 plus `last`. A full disk takes both arms.
4. **Panel resolution.** Reporting an arm only on the clean 321 panel risks
   calling a real +1.5 pp "no effect". Budget the 803 decode into the arm, not
   after it.
5. **The corpus arm cannot see its own inputs from the worker.** The 67 GB corpus
   and the 1,095-block exclusion list both live where the compute is not (§14.1
   point 4). Stage a row-group subset and the list as run inputs, or run stage 1
   on the local card — and never take the `Fp2MolStream` opt-out that `bfedcba`
   left in place, because 10M fp2mol rows carry 23 of the 320 clean-panel blocks
   against a 3-spectrum baseline.
6. **A transient `git clone` kills an arm five minutes in.** That is exactly how T2
   died. Check every newly started arm within 15 minutes of its start.
7. **Probe early stopping can end an arm long before step 20,000** (patience 4 at a
   1,000-step cadence), leaving `last.ckpt` at an odd step. That is the intended
   bound on 3,846-fold replay — but such a checkpoint must never be described as a
   20,000-step run.
8. **The warm-start confound is still unpaid.** T1a and T2 both resume
   `control-r2 step=100000`, itself 100,000 steps of the broken recipe (272.8× the
   displacement budget, §6 of `TRAINING_RECIPE_FINDINGS.md`). Until arm D runs, a
   gain from these arms cannot be attributed to the recipe fix, which is the claim
   the project exists to make.

## 15. The corpus and the fitted corruption are joined to the trainer, 2026-08-13 night

Appended after §14. Sections 1–14 are unedited. This section closes §13.8 items 1–4
and §14.1 point 4. Every number below was measured on this host today; the two that
were not measured here are labelled with the run that measured them.

### 15.1 The switch now has four laws, and two of them are fitted

`src/marlin/training.py` accepts `fingerprint_noise_mode ∈ {symmetric, dropout,
fitted, fitted_control}` and a `fingerprint_error_model` path.
`marlin.training.fingerprint_corruption` loads
`artifacts/encoder-error-model-oof-v1/encoder_error_model_oof.npz` and derives the
control from **the same file** with `rate_matched_uniform_control()`, so the pair
cannot drift apart: the control is the fitted model with its frequency dependence and
row latent removed and its pooled rates held, which is exactly CoRe-Gen's −4.77 pp
ablation.

The incumbent branch is untouched argument for argument
(`MarlinLightningModule.corrupt_conditioning`), and
`test_symmetric_still_draws_exactly_what_it_drew_before` asserts that a `symmetric`
run draws the identical tensor from the same seed. The frozen-loss identity in
`tests/test_marlin_training_recipe.py` (`45.71464157104492`) still passes.

Measured on **64 real fp2mol molecules** (mean 49.4 true on-bits), corruption
probability 1.0:

| law | recall of true on-bits | invented bits/row | median Tanimoto to truth |
|---|---:|---:|---:|
| `fitted` | 0.555 | 17.75 | **0.3538** |
| `fitted_control` | 0.487 | 23.56 | 0.3373 |
| `symmetric` (incumbent) | 0.798 | 10.00 | 0.6508 |

The real held-out DreaMS median is **0.3043** (`report_oof.json`). The fitted law
lands at 0.354 on corpus molecules; the incumbent at 0.651 is 2.1× too clean, and it
is flat where the real encoder is not. That is §4 of `TRAINING_RECIPE_FINDINGS.md`
reproduced on the corpus the arm will actually train on.

### 15.2 Stage 2 can no longer erase stage 1

The reviewer's finding is now a refusal, not a note. `resolve_inherited_corruption`
reads the corruption law out of the warm-start checkpoint's own hyper-parameters
(`checkpoint_fingerprint_corruption`) and:

- an omitted `--fingerprint-noise-mode` **inherits** it;
- a different mode, or a different fitted file, **raises** unless
  `--allow-corruption-mode-change` says the change is the experiment;
- a checkpoint that records nothing — the released DLM does not — falls back to
  `symmetric`, which is what every run on record used.

`scripts/run_marlin_faro_spectrum_adaptation.sh` no longer passes
`--fingerprint-noise-mode symmetric` unconditionally; an unset
`MARLIN_FINGERPRINT_NOISE_MODE` now means "inherit". Tests:
`test_stage_two_may_not_silently_revert_to_the_incumbent_noise`,
`test_the_two_fitted_arms_do_not_inherit_each_others_law`,
`test_a_different_fitted_file_is_also_a_change`.

Why it matters, in the units of the experiment: stage 1 is 10,000 steps and a stage 2
is 20,000, so a reverting stage 2 would run **two thirds** of the pair's optimizer
steps under a law whose KS distance to the real held-out DreaMS error is 0.8727
against the fitted law's 0.0738, and arms A and B would converge by construction.

### 15.3 Data locality: the corpus streams, it does not move

Measured from **inside a Slurm allocation on node `spectrum`** (job 790, `gpu-shared`,
8 CPUs, COMPLETED in 25 s). The login host **is** the Slurm node, and the Slurm job
sees the real NFS:

| quantity | measured |
|---|---:|
| corpus visible from the job | 94 shards, **934 row groups**, 67 GB |
| one 1,048,576-row group off NFS | **0.189 s = 5.54 M rows/s** |
| `Fp2MolStream`, single worker | **407 mol/s/core** (5 rejections in 4,000, all `too_long`) |
| `Fp2MolStream` + real collator, 8 workers | **599 mol/s** |
| training demand, global batch 256 at 4.51 s/step | **56.8 mol/s** |
| headroom | **10.6×** |
| fitted corruption, batch of 32 on CPU | 6.399 ms = 5,001 mol/s |

So the 67 GB corpus needs no staging, no subset copy and no cache: at 10.6× headroom
the stream is never the bottleneck, and the corruption runs on the training device.
The alternative — staging to the ClearML worker — is refused by measurement rather
than by preference: that worker's `/mnt/netstorage` is a per-worker ext4 with
**8.04% free** (§14.4 item 3) and both its slots are held until T1a and T2 finish.

**The exclusion list now travels with the code.** `data/nplib1_holdout_inchikeys_v2.csv`
(1,095 blocks, sha256 `7d1f4593…`) is committed **inside the repository**;
`configs/marlin_nplib1.yaml` names it repo-relative and
`marlin.corpus_stream.resolve_repository_path` resolves it against the repository
rather than the working directory, so a `git clone` on any worker carries it.
`test_the_holdout_list_is_inside_the_repository` pins the digest and the count.

### 15.4 The contamination guard is in the prediction file

`scripts/evaluate_marlin_nplib1.py --corpus-exclude-inchikeys` writes a per-row
`seen_in_corpus` flag — true when the corpus stream was **allowed** to emit that
structure — and `metrics.json` gains `seen_in_corpus_spectra`,
`corpus_unseen_spectra`, `exact_top1_corpus_unseen` and `exact_top1_corpus_seen`.
A row written without the flag reports `null`, which means **unknown, not clean**,
the same convention the repair provenance uses. The flag is threaded through
`src/marlin/periodic_evaluation.py` and `scripts/evaluate_marlin_sharded.sh`
(`CORPUS_EXCLUDE=…`), so an in-training panel evaluation of a corpus arm carries it
too.

`seen_in_corpus_spectra` **must be 0**. Its precondition is measured rather than
assumed: `test_the_panel_is_entirely_inside_the_packaged_exclusion` checks that every
connectivity block of `nplib1_val_clean322_v1` is in the packaged list. If a corpus
arm ever reports a non-zero count, the exclusion was not in the path and the run is
void.

### 15.5 The pair, and what it costs

Stage 1 is `scripts/slurm_marlin_corpus_stage1.sbatch`. It warm-starts from the
**released DLM in MARLIN layout**
(`checkpoints/frigid-warmstart-marlin/step=0.ckpt`, sha256
`71f463709a68071396b1b030923c17a98a12f9dfa51fc5fe89a620faaab35244`) and not from
control-r2, for two reasons: control-r2 exists only on the ClearML worker's local
disk, and it is itself 100,000 steps of the broken recipe, which is §14.4 item 8's
unpaid confound. Its own `hyper_parameters` carry no corruption law, verified by
reading them, so stage 1 must name its mode and the inheritance rule has nothing to
inherit — which is correct.

```bash
# arm C-A1, the fitted law
MARLIN_ERROR_MODEL_VARIANT=fitted \
  scripts/submit_when_gpu_idle.sh scripts/slurm_marlin_corpus_stage1.sbatch

# arm C-A0, the rate-matched uniform control
MARLIN_ERROR_MODEL_VARIANT=fitted_control \
  scripts/submit_when_gpu_idle.sh scripts/slurm_marlin_corpus_stage1.sbatch
```

`submit_when_gpu_idle.sh` polls for compute processes owned by another user and
submits only when there are none, so neither arm can evict the colleague's run.

Cost. The data side is measured above and is not the constraint. The optimizer step
is quoted from the fleet's own measurement, not from this host: **4.51 s/step** at
global batch 256 with `--fp32-forward`, read from T1a's iteration timestamps
(§14.1). 10,000 steps is therefore **≈12.5 GPU-hours per arm, ≈25.1 for the pair**,
and 10,000 × 256 = 2,560,000 molecules against the 8 row groups (≈8.4 M eligible
molecules) the script streams — **no replay at all**, against the 770 replays a
20,000-step NPLIB1 arm performs. The local A100 is 80 GB against the worker's 40 GB
and is otherwise unmeasured for this recipe; the first arm must be re-timed in its
first ten minutes and the figure corrected.

Verified end to end on CPU before any GPU was asked for: `Fp2MolStream` → real
`MarlinCollator` → `fitted` corruption → four Lightning optimizer steps, weights
moved, `fingerprint_error_model_sha256 = 815da77b…` recorded in the module's
hyper-parameters.

### 15.6 What this does not yet do

1. **Stage 2 is not written as a script.** The inheritance rule is in the trainer and
   tested, but the NPLIB1 stage 2 that reads a stage-1 checkpoint still has to be
   submitted by hand, omitting `--fingerprint-noise-mode` so that it inherits.
2. **No decode has been priced on the post-R1 tree** (§13.8 item 5), so every decode
   cost in this section is still marked unpriced.
3. **Arm D is still owed** (§13.8 item 3). Without it, A−B prices the placement law
   but nothing separates "the corpus worked" from "the recipe fix worked".
