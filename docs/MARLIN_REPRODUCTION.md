# MARLIN clean-room reproduction and conformance ledger

This branch is a clean-room reproduction of MARLIN (arXiv:2607.04774). The
authors' implementation and trained weights were not public when this work
started. Results from this branch must not be described as bitwise-identical,
author-verified, or produced by the authors' code.

The arXiv v1 source archive contains the main manuscript, bibliography, class
file, and figures. It does not contain a separate supplementary manuscript or
appendix. Consequently, the method specification available for reproduction is
the main paper itself. Details absent from that source are marked **inferred**
below and in generated manifests.

## Paper-to-code conformance

| Paper requirement | Reproduction location | Status |
| --- | --- | --- |
| SAFE sequences and deterministic SAFE-to-molecule decoding | `marlin.tokenizer`, official `safe-mol==0.1.13`, strict `safe_to_smiles(..., fix=False)` | Implemented |
| SAFE vocabulary size 1,880 | `configs/marlin_nplib1.yaml` and the published `datamol-io/safe-gpt` tokenizer artifact | Implemented; artifact hash recorded at run time. The corpus and its pinned revision are inferred because the paper does not identify its training corpus. |
| Decoder width/layers/heads 896/12/14 | `MarlinDecoderConfig` | Implemented |
| Block width 8 | `MarlinDecoderConfig.block_width` | Implemented |
| Block-causal clean/noisy two-stream attention | `marlin.model.two_stream_attention_mask` and `two_stream_logits` | Implemented and tested |
| Per-block continuous time, absorbing masking, and `1/t` NELBO | `MarlinDecoder.diffusion_loss` | Implemented; all valid blocks are averaged in one forward pass, an unbiased realization of uniform block sampling |
| Score every unresolved position and reveal the globally highest-confidence position | `MarlinSampler` | Global reveal is implemented and tested. Exact position-aware grammar support through partial-block holes remains underdetermined by the paper. |
| Precursor mass Fourier token | `FourierMassEncoder` | Implemented |
| Optional M+1/M and M+2/M isotope token, enabled in training and disabled by default in inference | `marlin.isotopes`, `MarlinCollator`, `MarlinConditioner` | Implemented; inference omits the isotope token rather than inserting an untrained placeholder. Isotope-envelope calculation is inferred. |
| One conditioning token per active Morgan bit, 4,096 bits, radius 2 | `SparseFingerprintEncoder`, `MarlinCollator` | Implemented |
| Symmetric fingerprint corruption with `p=0.5`, `rho ~ U(0.1, 0.3)`, equal drop/add counts | `marlin.noise.symmetric_fingerprint_noise` | Implemented |
| Warm start from FRIGID | `marlin.warm_start.load_frigid_decoder` | Implemented; the official checkpoint SHA-256, tensor shapes, tokenizer hash, and pinned dataset revision are checked or recorded. Exact vocabulary-order identity cannot be proven from the weight-only checkpoint. |
| AdamW, learning rate `5e-5`, batch 256, fixed EMA decay 0.9999 | config and `MarlinLightningModule` | Implemented. The inherited DLM EMA update-count warm-up is explicitly disabled. |
| Preserve complete SAFE targets within the decoder context | streaming length filter and `MarlinCollator` | Silent truncation is forbidden. Targets longer than the inferred 256-token context are excluded before batching; the collator still fails closed if one reaches it. This inferred policy and its pinned million-row audit are hashed in the run manifest. |
| Heavy-atom token masses excluding hydrogen; zero mass for special tokens | `marlin.token_properties`, `MassShellConstraint` | Implemented |
| Prune when `h + mu_v > M + delta` | `MassShellConstraint.apply` | Implemented |
| Hydrogen-aware reachable interval with valence slack 4 | grammar state and `MassShellConstraint` | Implemented conservatively |
| Forbid early EOS and boost EOS when no positive-mass token fits | `MassShellConstraint.apply`, `MarlinSampler._mass_state` | Implemented. Tokens to the right of a committed EOS are excluded from the semantic prefix and mass state; this suffix handling and the boost magnitude are inferred because the paper does not specify within-block EOS padding. |
| Grammar and forbidden-token masks at every reveal | `SafeGrammarMask` and sampler | Partial/inferred: full syntactic support is retained for contiguous prefixes and EOS is forbidden across an unresolved hole. Other tokens are conservatively not grammar-pruned across holes because the paper does not define the required existential grammar state. Completed blocks still pass strict SAFE decoding. Chemical mass reachability is not folded into the syntax mask. |
| Decode after each committed block and accept only valid structures within 10 ppm | `MarlinSampler` | Implemented |
| 384 candidates with independent 0.3 on-bit conditioning dropout | `MarlinSampler.generate_ranked_with_stats` | Implemented |
| Rank by Tanimoto to the unperturbed predicted fingerprint | `MarlinSampler` | Implemented |
| DreaMS raw-spectrum and MIST predicted-formula feature lanes | evaluation inputs and lane-specific fingerprints | DreaMS is implemented. The original MIST export used ground-truth-formula subformula annotations and is retained only as a leaked oracle; canonical evaluation rejects that path and requires a hashed MIST-CF predicted-formula provenance manifest. |
| Exact Top-1/Top-10, Morgan Tanimoto Top-1/Top-10, and MCES Top-1/Top-10 | evaluation scripts | Implemented; MCES runtime is isolated and smoke-tested with `myopic-mces==1.2.0` and PuLP 3.3.2 |
| Formula recovery and mass bins `<300`, `300-500`, `>=500` Da | evaluation post-processing | Implemented |
| Saved per-spectrum predictions and runtime | `evaluate_marlin_nplib1.py` | Implemented. The sampler does not yet use the paper's committed-prefix KV cache, so its runtime is not comparable to the paper. |
| ClearML curves and local TensorBoard events | `train_marlin.py`, `MarlinLightningModule` | Verified by Slurm job 293: ClearML task `3854ab071bdd4602a5678f720fe2d629` contains finite `train_loss`, `learning_rate`, `fingerprint_noise_fraction`, and `grad_norm` series; the run also saved a TensorBoard event and EMA checkpoint. Evidence: `manifests/clearml_smoke_293.json`. |
| Immutable training provenance | `train_marlin.py` | A canonical run refuses a dirty Git checkout and writes `run_manifest.json` with the commit, resolved config, input hashes, package versions, CUDA/GPU identity, inferred settings, and Slurm job ID before loading the training stream. |

## Explicitly inferred choices

The paper does not disclose the following items. They are not presented as
author settings:

- the decoder training corpus and exact split; this reproduction uses the
  public `datamol-io/safe-gpt` train stream with all NPLIB1 test connectivity
  keys excluded;
- the Hugging Face dataset revision and streaming shuffle buffer;
- 100,000 adaptation steps, checkpoint cadence, and random seed;
- maximum sequence length 256, FFN width 3,584, dropout 0.1, gradient clipping,
  weight decay, and the absence of a learning-rate schedule/warmup;
- exclusion, before batching, of complete SAFE targets longer than the fixed
  256-position DLM context;
- 64 geometric Fourier frequencies spanning `1e-3` to `1.0`;
- theoretical training isotope ratios computed from RDKit natural abundances;
- exact molecular mass as the clean training proxy for measured neutral
  precursor mass, without instrument/adduct noise augmentation;
- the exact SAFE partial-prefix grammar, partial-block hole semantics, and
  treatment of tokens to the right of EOS;
- the additive EOS logit boost magnitude;
- encoder fingerprint binarization thresholds;
- the exact myopic-MCES package version, solver, threshold, and timeout;
- neutral-mass conversion from the NPLIB1 `[M+H]+` precursor convention;
- all engineering parameters related to batching, data-loader workers,
  caches, and GPU execution.

## Reproduction gates

A full training run is eligible for final evaluation only when all of the
following hold:

1. strict official SAFE round trips pass without fragment repair;
2. random reveal-order oracle tests show that valid ground-truth tokens are not
   removed by the partial-block grammar mask;
3. the sampler reveals the globally highest-confidence supported position,
   not an implicitly leftmost position;
4. a Slurm smoke run produces a finite loss, EMA checkpoint, TensorBoard event
   file, and a non-empty ClearML scalar series for loss, learning rate,
   fingerprint-noise fraction, and gradient norm;
5. checkpoint, tokenizer, dataset revision, code commit, environment, and all
   inferred settings are captured in a provenance manifest;
6. evaluation resume refuses incompatible settings or duplicate spectrum IDs;
7. the final two 803-spectrum lanes report every paper metric plus validity,
   uniqueness, mass-validity, formula recovery, mass bins, and runtime.
8. training data are never silently truncated; any overlength exclusion or
   context-length change is recorded as an inferred policy and audited before
   the canonical submission.

The ClearML/TensorBoard/checkpoint gate was satisfied by Slurm job 293. The
supported random-reveal oracle is produced by
`scripts/audit_marlin_safe_oracle.py` and must be retained with the run
manifests before the full training submission. The pinned stream length audit
is produced by `scripts/audit_marlin_training_lengths.py`; it records the
sample size, length distribution, overlength count, input revision, tokenizer
hash, and code commit.

## Rejected non-canonical checkpoints

`runs/decoder-bos-nelbo-v2/checkpoints/step=40000.ckpt` is mechanically
load-compatible with the current model, but it is not eligible for the canonical
run. Slurm job 282 trained it at commit `dd98fbe` before real isotope-ratio
conditioning and strict `safe_to_smiles(..., fix=False)` were introduced. The
four isotope-MLP parameters have no AdamW moments, while the missing-isotope
embedding does. The job also predates pinned input hashes and ClearML. Resuming
it would change the training task after 40,000 steps, so it is retained only as
a diagnostic hybrid/ablation artifact. Its SHA-256 is
`9ccdc21b1690594f193c0c1f222f330bd3cb6757866b235a61c23d66233ff454`.

## Encoder evidence status

- DreaMS job 214 produced a formula-free test fingerprint Tanimoto of
  `0.337694`. The frozen-head fitting protocol remains a clean-room choice
  because the paper does not publish its encoder-training details.
- MIST job 215 produced `0.533575`, but its peak-to-subformula features used the
  NPLIB1 ground-truth formula. It is therefore a leakage-positive oracle, not a
  MARLIN(MIST) result. Final MIST evaluation requires MIST-CF top-1 predicted
  formulas, regenerated subformula annotations, and a new fingerprint export.
  The required official MIST-CF and CANOPUS checkpoints and SIRIUS 5.5.7 were
  not present in the inspected Spectrum paths. Their training splits must also
  be checked for overlap with the 803 evaluation spectra before use.

## Paper reference values

The published NPLIB1 results are comparison targets, not expected assertions:

| Lane | Exact Top-1 | MCES Top-1 | Tanimoto Top-1 | Exact Top-10 | MCES Top-10 | Tanimoto Top-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MARLIN (DreaMS) | 16.94% | 6.79 | 0.55 | 23.54% | 5.83 | 0.60 |
| MARLIN (MIST) | 19.18% | 8.59 | 0.51 | 26.65% | 7.40 | 0.57 |

The paper also reports Top-1 formula recovery of 76.7% over all DreaMS spectra
(96.9% among spectra with a returned candidate) and 72.2% for the MIST lane.
