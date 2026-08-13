# Training findings, 2026-08-13

Three research agents finished before the account's weekly token limit stopped the rest.
Everything here is measured or quoted from a paper; sources are inline. This supersedes two
claims in `DECODER_PROGRAM.md` — see §4.

## 1. Our adaptation recipe is broken in three places

Extracted from the released DLM checkpoint's own `hyper_parameters`, `lr_schedulers`,
`optimizer_states` and `ema`, against `scripts/train_marlin_spectrum_adaptation.py`,
`src/marlin/training.py` and the control-r2 checkpoint's stored hparams.

| Knob | Released DLM | FRIGID exp-8 config | Ours (control-r2) | Risk |
|---|---|---|---|---|
| Peak lr | 1.3e-4 | 5e-5 | 1e-5 | — |
| **Terminal lr** | **5.2697e-8** (cosine fully annealed) | 1e-8 | **1e-5, never anneals** | **HIGH: we restart at ~190x the lr the weights were left at** |
| **Schedule** | LinearLR 1e-6→1 over 6,000 steps, then cosine T_max 520,000, eta_min 1e-8 | warmup 2,000 + cosine T_max 50,000 | **none at all** (`lr_schedulers: []`) | **HIGH** |
| Global batch | 1,920 (6×64×5 accum) | 512 | 256 (8×32) | MEDIUM |
| **Steps / data** | 520,000 steps, ≥998M samples, **epoch 0** | 50,000 | 100,000 steps over 6,748 rows / 6,032 unique molecules, **epoch 3,846** | **HIGHEST** |
| Loss reduction | `global_mean_loss` = sum / total valid tokens | same | per-example sum / ceil(n_valid/8) then batch mean ≈ **8× a token mean** | MEDIUM |
| Time sampling | one t per **sequence**, antithetic/stratified, t ∈ [1e-3, 1] | same | i.i.d. t per (example, **block**), no antithetic, floor 1e-4 | MEDIUM: 10× heavier 1/t tail at 1/7.5 the batch |
| Precision | trainer bf16 but forward forced to fp32 (`src/dlm/model.py:949`) | same | `bf16-mixed`, no fp32 override | MEDIUM |
| Grad-spike detection | on (×100, ema 0.97, min 800) — absent from public FRIGID code | absent | absent | LOW-MED |
| Noise schedule | loglinear, weight 1/t | same | mask prob = t, weight = 1/t — **identical** | none |

Read the top two rows together: we resume 249.5M weights that were annealed to 5.3e-8 at a
constant 1e-5 with no schedule, and then replay 6,032 molecules 3,846 times. That is a recipe
for catastrophic forgetting regardless of the objective.

## 2. The data situation is worse than the recipe

- Adaptation corpus: 6,649 rows, **6,032 unique molecules**, 2,764 Murcko scaffolds — against a
  249.5M-parameter decoder.
- A 20k-step run at batch 256 replays that corpus **770 times** while seeing **0.51%** of the
  pretraining budget. A 100k run: 3,850 replays, 2.56%.
- The **fp2mol corpus (datamol-io/safe-gpt, 933,382,869 molecules, 67 GB) is readable today and
  unused** for adaptation.
- The encoder error is **simulatable**: per-bit-index conditional rates plus one per-row
  difficulty latent reproduce the real Tanimoto distribution; the bit-index space is 100% shared
  with fp2mol; generation runs at 3,353 mol/s/core against a 55.6 mol/s requirement.
  **The same 25.6 GPU-hours that today buys 6,032 distinct molecules could buy 5,120,000.**
- FRIGID's pretraining config has the hook `fingerprint_flip_prob` set to 0.000, and the released
  checkpoint's config lacks the key entirely: **the decoder has never seen a corrupted
  fingerprint.**

## 3. FRIGID experiment 8 is probably a checkpoint-selection artefact

MS-BART early-stops on validation Top-1 Tanimoto every 200 steps with patience 3. FRIGID's
finetune config sets `validation.enabled: False` and checkpoints every 2,500 steps, so the
"adapted DLM" reported in experiment 8 is the **first** checkpoint of a 50,000-step cosine,
caught at peak lr 5e-5 straight after a 2,000-step warmup with no annealing.

So the negative that closed "fine-tune on predicted fingerprints" is at least partly an artefact
of when the checkpoint was taken, not proof the objective fails. It should be re-run with
validation-based selection before being treated as settled.

## 4. Two corrections to DECODER_PROGRAM.md

1. **The MS-BART anchor was misread.** `DECODER_PROGRAM.md:221,625` cites 1.71% → 7.45% as the
   gain from fine-tuning on predicted fingerprints. That is MS-BART's **pretraining** ablation
   (their Table 2), where every row already includes the predicted-fingerprint finetune. The
   correct anchor for the finetune itself is MassSpecGym **0.00% → 1.07%** (their §4.6).
2. **A new SOTA reference exists on our axis.** CoRe-Gen (arXiv 2605.12980, May 2026) reports
   **19.54% Top-1 on NPLIB1**. Its ablation prices frequency-aware fingerprint corruption at
   **−4.77 pp** when removed (19.54 → 14.77) — the second-largest component of the system — and
   applies that corruption during **decoder pretraining** on a 2.8M corpus, not as an adaptation
   on a few thousand molecules.

Our training noise is not merely too weak: it is **structurally inverted** relative to the real
encoder error.

## 5. Ranked next experiments

1. **Frequency-aware fitted corruption during pretraining-scale training** on fp2mol, using the
   measured per-bit error model. This is CoRe-Gen's largest single component and the measurement
   says we can generate the data for free.
2. **Distillation as a KL anchor**: train the predicted-fingerprint model against the output
   distribution of the same decoder under the true fingerprint. Targets the 0.750 vs 0.547
   teacher-forced gap directly.
3. **Self-correction** — already queued and paid for; only its diagnostic is missing (teacher-
   forced top-1 under a corrupted prefix).
4. **Mixture/curriculum** — last. The mixture half is both the incumbent (every queued arm
   already runs `MARLIN_NOISE_PROBABILITY=0.5`) and already failed at FRIGID.

Recipe fixes to apply to **every** arm regardless of objective: restore a cosine schedule with
warmup, start from a terminal-lr-compatible peak, fix the loss reduction to a true token mean,
adopt per-sequence antithetic time sampling, and cap replay of the tiny corpus.

## 6. The recipe is fixed, behind flags, 2026-08-13

Each of the four corrections in §1 is now implemented with the **current behaviour as the
default**, so a run that asks for none of them is bit-identical to every run already on
record. `tests/test_marlin_training_recipe.py` freezes that identity on a fixed seed and
batch (`block_mean` / `per_block_iid` loss `45.71464157104492`), and each correction has its
own test.

| Correction | Flag | Default | Corrected value |
|---|---|---|---|
| (a) lr schedule | `--lr-schedule` / `--derive-learning-rate` | `constant` | `warmup_cosine`, peak **3.1647e-7** |
| (b) loss reduction | `--loss-reduction` | `block_mean` | `token_mean` |
| (c) time sampling | `--time-sampling` | `per_block_iid` | `per_sequence_antithetic` |
| (d) precision | `--fp32-forward` | off | on |
| replay bound | `--probe-early-stopping-metric` | off | `probe_top1_predicted`, patience 3 |

### The peak, derived rather than guessed

Read verbatim from `DLM.ckpt["lr_schedulers"][0]` (sha256
`b6177c2d43448380aba80ff41c01461ea34ca2ca93b213986954c5afb7f0f457`): `LinearLR` from
`start_factor 1e-6` over 6,000 steps into `CosineAnnealingLR(T_max=520000, eta_min=1e-8)`
on `base_lr 1.3e-4`, stopped at cosine step 514,000 with
`_last_lr = 5.2697058404552555e-08`. `src/marlin/lr_schedule.py:released_learning_rate`
reproduces that terminal value to 1e-9 relative, which is what makes the rest arithmetic
rather than assertion.

AdamW's per-step update has magnitude of order `lr`, so `sum_t lr_t` bounds how far a run
can move any one weight. The weight scale to compare it against is measured: the RMS of the
173,801,752 float parameters under `decoder.` in the control-r2 checkpoint the arms
warm-start from is **0.082472829079876**.

Three anchors, all computed in `src/marlin/lr_schedule.py`:

1. **Pure continuation.** Our 20,000 steps at global batch 256 are 5.12M examples, which the
   released run covered in 2,667 of its steps at batch 1,920; its schedule there reads
   **2.32e-8**, *below* the rate the weights were left at. No continuation criterion can
   justify anything larger, which is the honest statement of the problem.
2. **A full retrain at our batch.** `1.3e-4 * 256 / 1920 = 1.7333e-5`. **Today's constant
   `1e-5` is 58% of that** — the current "fine-tune" runs at a pretraining peak, flat,
   forever. That is the defect, stated in the units that matter.
3. **The released run's final decade** — the 14,890 steps over which its own rate fell from
   `10x` terminal to terminal. `sum_t lr_t` over that segment is **3.6653e-3**, or 4.44% of
   the weight RMS. It is the last well-defined stretch in which the released optimiser was
   still making updates of the order we are about to make, and it is the budget adopted.

Solving anchor 3 for a 20,000-step warmup-cosine with a 1,000-step warmup and a floor at the
checkpoint's terminal rate gives **peak = 3.1647e-7**: `6.0x` the terminal rate rather than
`190x`, and `54x` below the batch-scaled pretraining peak.

For scale, the same budget prices what has already been spent: the 20,000-step arms at a
constant `1e-5` were about to spend **54.6x** that budget, and control-r2's 100,000 steps
spent **272.8x** it — a displacement bound of **12.1x the weight RMS**.

The warmup is 1,000 steps because the optimizer state is not restored: AdamW's second-moment
estimate has an averaging window of `1 / (1 - beta2) = 1000` at the default `beta2 = 0.999`,
and the peak is not the peak it was chosen to be before that window has filled.

### Bounding the replay

`--probe-early-stopping-metric probe_top1_predicted` stops a run when teacher-forced
per-token top-1 **under the predicted fingerprint**, on structures the optimizer never sees,
stops improving for `--probe-early-stopping-patience` probes. That is MS-BART's rule (§3) at
a cadence the conditioning probe can afford — four short forward passes rather than the hours
a molecular panel costs — so a run stops when it starts forgetting instead of at an arbitrary
step count.

### Why the three arms had to be resubmitted anyway

`781de15ee38c408aa2cc068fade83110`, `3f163b90b6494d0f91762fa8cd92171a` and
`50ef8dcd85e348b6861fdbadea2cecdb` are **`failed`**, not queued: all three died 5 minutes in
at commit `4fc867a` with
`ValueError: evaluation.interval_steps must equal output.checkpoint_interval`
(`src/marlin/periodic_evaluation.py:52`), because they were submitted with
`MARLIN_CHECKPOINT_INTERVAL=5000` against `MARLIN_EVALUATION_INTERVAL=20000`.
