# MARLIN clean-room reproduction and conformance ledger

This branch is a clean-room reproduction of MARLIN (arXiv:2607.04774). The
authors' implementation and trained weights were not public when this work
started. Results from this branch must not be described as bitwise-identical,
author-verified, or produced by the authors' code.

The bounded held-out search protocol, immutable evaluator contract, and current
architecture/inference results are recorded in
[`MARLIN_AUTORESEARCH.md`](MARLIN_AUTORESEARCH.md).

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
| Warm start from FRIGID | `marlin.warm_start.load_frigid_decoder`, `scripts/audit_marlin_frigid_parity.py` | Implemented and functionally audited against the released model. For checkpoint SHA-256 `b6177c2d43448380aba80ff41c01461ea34ca2ca93b213986954c5afb7f0f457`, fingerprint token sets agree within `3.73e-8` and final logits within `1.15e-5`. The machine-readable result is `/mnt/netstorage/nikolenko/marlin/runs/autoresearch/parity/frigid-parity-4701a8a.json`. Exact vocabulary-order identity still cannot be proven from the weight-only checkpoint. |
| AdamW, learning rate `5e-5`, batch 256, fixed EMA decay 0.9999 | config and `MarlinLightningModule` | Implemented. The inherited DLM EMA update-count warm-up is explicitly disabled. |
| Preserve complete SAFE targets within the decoder context | streaming length filter and `MarlinCollator` | Silent truncation is forbidden. Targets longer than the inferred 256-token context are excluded before batching; the collator still fails closed if one reaches it. This inferred policy and its pinned million-row audit are hashed in the run manifest. |
| Heavy-atom token masses excluding hydrogen; zero mass for special tokens | `marlin.token_properties`, `MassShellConstraint` | Implemented; the running state sums every committed token across unresolved holes and ignores only MASK plus the suffix after EOS. |
| Prune when `h + mu_v > M + delta` | `MassShellConstraint.apply` | Implemented |
| Hydrogen-aware reachable interval with valence slack 4 | grammar state and `MassShellConstraint` | Implemented conservatively |
| Forbid early EOS and boost EOS when no positive-mass token fits | `MassShellConstraint.apply`, `MarlinSampler._mass_state` | Implemented. Tokens to the right of a committed EOS are excluded from the semantic prefix and mass state; this suffix handling and the boost magnitude are inferred because the paper does not specify within-block EOS padding. |
| Grammar and forbidden-token masks at every reveal | `SafeGrammarMask` and sampler | Partial/inferred: full syntactic support is retained for contiguous prefixes and EOS is forbidden across an unresolved hole. Other tokens are conservatively not grammar-pruned across holes because the paper does not define the required existential grammar state. Completed blocks still pass strict SAFE decoding. Chemical mass reachability is not folded into the syntax mask on the canonical path; `--mass-reachability-prune` enables it as an opt-in diagnostic deviation and is recorded as `mass_reachability_prune` in the run manifest. |
| Decode after each committed block and accept only valid structures within 10 ppm | `MarlinSampler` | Implemented |
| 384 candidates with independent 0.3 on-bit conditioning dropout | `MarlinSampler.generate_ranked_with_stats` | Implemented |
| Rank by Tanimoto to the unperturbed predicted fingerprint | `MarlinSampler` | Implemented |
| DreaMS raw-spectrum and MIST predicted-formula feature lanes | evaluation inputs and lane-specific fingerprints | DreaMS is implemented. The original MIST export used ground-truth-formula subformula annotations and is retained only as a leaked oracle; canonical evaluation rejects that path and requires a hashed MIST-CF predicted-formula provenance manifest. |
| Exact Top-1/Top-10, Morgan Tanimoto Top-1/Top-10, and MCES Top-1/Top-10 | evaluation scripts | Implemented; MCES runtime is isolated and smoke-tested with `myopic-mces==1.2.0` and PuLP 3.3.2 |
| Formula recovery and mass bins `<300`, `300-500`, `>=500` Da | evaluation post-processing | Implemented |
| Saved per-spectrum predictions and runtime | `evaluate_marlin_nplib1.py` | Implemented. The sampler does not yet use the paper's committed-prefix KV cache, so its runtime is not comparable to the paper. |
| ClearML curves and local TensorBoard events | `train_marlin.py`, `MarlinLightningModule` | The final strict-SAFE gate, Slurm job 317 at commit `60d42e8`, completed three optimizer steps and produced a ClearML task with loss, learning-rate, fingerprint-noise, gradient-norm, actual micro-batch-size, GPU, and machine-monitor series. It also saved a TensorBoard event and a checkpoint with 232 EMA shadow tensors. Evidence: `manifests/clearml_smoke_317.json`. |
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

The final ClearML/TensorBoard/checkpoint gate was satisfied by Slurm job 317 at
commit `60d42e81d512e16d64dce6bed36424109923460a`. Its three reported losses were
`56.201488`, `39.083832`, and `33.567577`; ClearML also retained 15 samples for
each configured GPU and machine-monitor series. The checkpoint SHA-256 is
`8c1f7b06ec722be2d0c629c5367af55eb198fe02fc2165fe18dbbee09d27e4a1`,
and the TensorBoard event SHA-256 is
`3a6e06297d9bb9c5af4a046439a4704fcc78b9d97ad45ac0230b45ab2244c1be`.
The supported random-reveal oracle is produced by
`scripts/audit_marlin_safe_oracle.py` and must be retained with the run
manifests before the full training submission. The pinned stream length audit
is produced by `scripts/audit_marlin_training_lengths.py`; it records the
sample size, overlength count, strict decode failures, NPLIB1 exclusions,
eligible count, input revision, tokenizer hash, and code commit. The pinned
million-row audit at commit `a7a49f6` examined 1,000,000 records and retained
998,678: 1,305 were overlength, seven failed strict SAFE decoding without
repair, and ten matched excluded NPLIB1 test connectivity keys. Its SHA-256 is
`c6267bfcffe51b42fa82d3514cb148ba15fa9e7c0141119682aa12c64413b0fd`.
Full snapshot SHA-256 verification is cached beside the immutable snapshot
manifest after the first successful pass. The cache is bound to the manifest
digest, canonical file-list digest, filesystem identity and timestamps, and a
sampled content digest for every shard. Unchanged runs therefore read only
small samples instead of rehashing the 71.3 GB snapshot; any identity or sample
change falls back to full hashing before training.

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

Slurm job 298 is also rejected and was cancelled after approximately 370
optimizer steps. Its `safe_to_smiles(..., fix=False)` call reached a wrapper
that did not forward `fix=False` to the official SAFE decoder, so malformed
sequences could be repaired. Its collator also skipped invalid, overlength, or
excluded examples after batching and could silently produce fewer than eight
examples. Commit `a7a49f6` fixed both defects by forwarding strict decoding and
filtering eligibility before batching; the collator now fails closed. No job
298 checkpoint may be resumed for canonical training.

## Encoder evidence status

- DreaMS job 214 produced a formula-free test fingerprint Tanimoto of
  `0.337694`. The frozen-head fitting protocol remains a clean-room choice
  because the paper does not publish its encoder-training details.
- MIST job 215 produced `0.533575`, but its peak-to-subformula features used the
  NPLIB1 ground-truth formula. It is therefore a leakage-positive oracle, not a
  MARLIN(MIST) result. Final MIST evaluation requires formula-blind MIST-CF
  predictions, regenerated subformula annotations, and a new fingerprint export.
  The official MIST-CF and MIST checkpoints are pinned in the isolated
  reproduction root. The released `split_1.tsv` has
  10,709 rows (7,727 train, 777 validation, 2,205 test). All 803 NPLIB1 IDs are
  present, but 586 are in its training fold, 60 in validation, and only 157 in
  test. A stricter archive scan found 889 spectra sharing the 701 unique NPLIB1
  connectivity blocks: 633 train, 73 validation, and 183 test, for 706 fitting
  overlaps. Therefore the released MIST-CF checkpoint is contamination-positive
  for this benchmark. Its formula-blind predictions may be reported only as a
  paper-like released-checkpoint lane with that disclosure, not as an unbiased
  canonical MIST lane. The clean retraining split moves all 706 train/validation
  connectivity overlaps to test before fitting either the fast filter or the
  MIST-CF scorer. Slurm job 315 trains the scorer on this connectivity-clean
  split with the official public MIST-CF architecture and optimization
  parameters. The released scorer remains available only as the explicitly
  contamination-positive paper-like lane. After that fit completes,
  `slurm_mist_cf_clean_predict.sbatch` runs the same official formula-blind
  prospective inference path used for the released-checkpoint lane, changing
  only the scorer checkpoint. Its fast formula filter is the released generic
  biomolecular-formula model rather than an NPLIB1 spectrum-fit model.
  Section III-A of MARLIN states a narrower contract than the first clean-room
  attempt assumed: the top-1 MIST-CF formula is used to compute
  peak-to-subformula features, and the ground-truth formula is never used. The
  canonical all-803 implementation therefore applies the same official MIST-CF
  subformula assignment to every spectrum and passes those positive fragment
  formulae to the released MIST PeakFormula encoder. It does not select a lower
  ranked candidate based on SIRIUS compatibility and does not invent signed or
  neutral-loss formulae. Four spectra have no peak assignment within the
  official MIST-CF threshold; their trees contain only the predicted root/CLS
  formula, exactly as the released MIST featurizer handles an empty fragment
  list. This is disclosed as an out-of-distribution limitation and will receive
  a separate sensitivity analysis rather than being hidden or imputed.
  `slurm_mist_cf_peakformula_fingerprints.sbatch` binds the top-1 formula
  manifest, all selected MIST-CF subformula JSONs, generated PeakFormula trees,
  MIST labels, final fingerprint NPZ, reference metadata, both source commits,
  both checkpoints, and the four root-only IDs. This is reported as “MIST with
  MIST-CF subformula adapter”, not as an official MIST-SIRIUS reproduction. The
  earlier SIRIUS jobs are retained only as rejected diagnostic evidence: one
  spectrum produced the signed node `C8H10N5-O`, which the official MIST formula
  parser cannot represent, and all seven available ranked candidates failed the
  SIRIUS bridge. Silently removing the sign or converting it to oxygen would
  change the chemistry.
  The official MIST code runs in a separate, frozen Python 3.8 environment;
  it is not allowed to mutate the completed MIST-CF scorer environment.

## Paper reference values

The published NPLIB1 results are comparison targets, not expected assertions:

| Lane | Exact Top-1 | MCES Top-1 | Tanimoto Top-1 | Exact Top-10 | MCES Top-10 | Tanimoto Top-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MARLIN (DreaMS) | 16.94% | 6.79 | 0.55 | 23.54% | 5.83 | 0.60 |
| MARLIN (MIST) | 19.18% | 8.59 | 0.51 | 26.65% | 7.40 | 0.57 |

The paper also reports Top-1 formula recovery of 76.7% over all DreaMS spectra
(96.9% among spectra with a returned candidate) and 72.2% for the MIST lane.
