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
| Score every unresolved position and reveal the globally highest-confidence position | `MarlinSampler` | Implemented and tested; exact position-aware partial-block grammar support remains underdetermined by the paper |
| Precursor mass Fourier token | `FourierMassEncoder` | Implemented |
| Optional M+1/M and M+2/M isotope token, enabled in training and disabled by default in inference | `marlin.isotopes`, `MarlinCollator`, `MarlinConditioner` | Implemented; isotope-envelope calculation is inferred |
| One conditioning token per active Morgan bit, 4,096 bits, radius 2 | `SparseFingerprintEncoder`, `MarlinCollator` | Implemented |
| Symmetric fingerprint corruption with `p=0.5`, `rho ~ U(0.1, 0.3)`, equal drop/add counts | `marlin.noise.symmetric_fingerprint_noise` | Implemented |
| Warm start from FRIGID | `marlin.warm_start.load_frigid_decoder` | Implemented; the official checkpoint SHA-256, tensor shapes, tokenizer hash, and pinned dataset revision are checked or recorded. Exact vocabulary-order identity cannot be proven from the weight-only checkpoint. |
| AdamW, learning rate `5e-5`, batch 256, EMA 0.9999 | config and `MarlinLightningModule` | Implemented |
| Heavy-atom token masses excluding hydrogen; zero mass for special tokens | `marlin.token_properties`, `MassShellConstraint` | Implemented |
| Prune when `h + mu_v > M + delta` | `MassShellConstraint.apply` | Implemented |
| Hydrogen-aware reachable interval with valence slack 4 | grammar state and `MassShellConstraint` | Implemented conservatively |
| Forbid early EOS and boost EOS when no positive-mass token fits | `MassShellConstraint.apply` | Implemented; boost magnitude is inferred |
| Grammar and forbidden-token masks at every reveal | `SafeGrammarMask` and sampler | Full syntactic support is retained for contiguous prefixes. A conservative no-prune rule is used across unresolved holes; this is inferred because the paper does not define hole semantics. Chemical mass reachability is not folded into the syntax mask. |
| Decode after each committed block and accept only valid structures within 10 ppm | `MarlinSampler` | Implemented |
| 384 candidates with independent 0.3 on-bit conditioning dropout | `MarlinSampler.generate_ranked_with_stats` | Implemented |
| Rank by Tanimoto to the unperturbed predicted fingerprint | `MarlinSampler` | Implemented |
| DreaMS raw-spectrum and MIST predicted-formula feature lanes | evaluation inputs and lane-specific fingerprints | Implemented |
| Exact Top-1/Top-10, Morgan Tanimoto Top-1/Top-10, and MCES Top-1/Top-10 | evaluation scripts | Implemented; MCES runtime is isolated and smoke-tested with `myopic-mces==1.2.0` and PuLP 3.3.2 |
| Formula recovery and mass bins `<300`, `300-500`, `>=500` Da | evaluation post-processing | Implemented |
| Saved per-spectrum predictions and runtime | `evaluate_marlin_nplib1.py` | Implemented |
| ClearML curves and local TensorBoard events | `train_marlin.py`, `MarlinLightningModule` | Instrumented. Task creation is verified, but scalar curves remain a smoke-test gate because job 283 failed before its first training step when the external warm-start file disappeared. |

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
- 64 geometric Fourier frequencies spanning `1e-3` to `1.0`;
- theoretical training isotope ratios computed from RDKit natural abundances;
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

## Paper reference values

The published NPLIB1 results are comparison targets, not expected assertions:

| Lane | Exact Top-1 | MCES Top-1 | Tanimoto Top-1 | Exact Top-10 | MCES Top-10 | Tanimoto Top-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MARLIN (DreaMS) | 16.94% | 6.79 | 0.55 | 23.54% | 5.83 | 0.60 |
| MARLIN (MIST) | 19.18% | 8.59 | 0.51 | 26.65% | 7.40 | 0.57 |

The paper also reports Top-1 formula recovery of 76.7% over all DreaMS spectra
(96.9% among spectra with a returned candidate) and 72.2% for the MIST lane.
