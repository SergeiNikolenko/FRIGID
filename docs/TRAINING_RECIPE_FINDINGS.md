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
