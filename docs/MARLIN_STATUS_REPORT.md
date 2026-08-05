# MARLIN reproduction status, 15 July to 5 August 2026

## Summary

The goal is to reproduce MARLIN (arXiv:2607.04774), de novo molecular structure
generation from an MS/MS spectrum. The paper reports `Exact@1 = 16.94%` on NPLIB1
in the formula-unknown setting with the DreaMS encoder.

This reproduction reports `Exact@1 = 0%`.

The cause was identified on 4 and 5 August and it is structural: **every adaptation
run so far started from the wrong decoder.** The paper's only stated initialization
is a warm start from the released FRIGID model. This lineage instead adapted a
from-scratch decoder trained for 30,000 steps, and that decoder's architecture made
a FRIGID warm start impossible to apply.

## Work completed

| | |
| --- | --- |
| period | 15 July to 5 August, three weeks |
| commits | ~360 |
| training and evaluation runs | 30 |
| cluster jobs | 100 |
| artifacts on shared storage | 90 GB |
| automated tests | 274, all passing |

The infrastructure works: decoder pretraining and spectrum adaptation, mass-shell
constrained decoding with its safety argument, the SAFE grammar mask, an evaluation
harness with hashed manifests and per-spectrum predictions, ClearML and SLURM
integration, and a 34-row paper-conformance ledger.

## Root cause

The paper's hyperparameter table, `Decoder training` block, states one line about
initialization:

> Initialization — warm-start from FRIGID

The paper names no other pretraining: no corpus, no step count, no dataset size.
Verified across all 663 lines of the LaTeX source.

| | paper | this reproduction |
| --- | --- | --- |
| initial weights | released FRIGID DLM | from-scratch, 30,000 steps |
| molecule presentations | 998,000,000 | 2,720,000 |
| shortfall | | **367x** |

The reproduction's own pretraining log records `validity = 0.0` and
`candidate_return_rate = 0.0` at every checkpoint from step 20,000 to 30,000, so the
base model never generated valid molecules.

The FRIGID checkpoint was on disk the whole time and its loader was written and
audited to `1.15e-5` logit parity, but the loader was committed on 25 July at 22:35,
after both pretraining runs had finished, and their output became the source for
every later adaptation. No `warm_start.json`, which `train_marlin.py` writes when the
warm start is applied, exists anywhere on disk.

## Why the warm start could not simply be switched on

Loading FRIGID weights failed three times in sequence. The pretrained architecture
diverges from the repository's own canonical config in three places:

| parameter | `configs/marlin_nplib1.yaml` | pretrained checkpoint |
| --- | --- | --- |
| `frigid_compatible_layer_order` | true | false |
| `fingerprint_self_attention_layers` | 3 | 0 |
| `fingerprint_layer_norm` | true | absent |

All three carry the comment "Inferred from the released FRIGID warm-start
checkpoint". The pretraining therefore built a different network, without the
fingerprint set-encoder self-attention stack and with a different normalization
order, which cannot be initialized from FRIGID under any setting.

## Defects found and fixed

Nine independent defects were found while diagnosing the zero result. All are fixed
and each is backed by a measurement.

Three changed what the model learns:

| defect | measured effect of the fix |
| --- | --- |
| targets were built as plain SMILES instead of the BRICS-sliced SAFE of the pretraining corpus, differing on 90.7% of molecules | teacher-forced top-1 `0.407 -> 0.556` |
| the threshold was never applied to soft fingerprints, so the effective value was 0.5 rather than 0.90 | conditioning Tanimoto `0.265 -> 0.495` |
| symmetric noise wrote injected false bits at amplitude 1.0 against genuine bits averaging 0.965 | noise no longer outranks real predictions |

Three changed generation: the grammar mask admitted absorbing states it could not
leave; the isotope conditioning token was present in every training batch and absent
from every sampling pass; Lightning deleted each periodic checkpoint when the next
one landed.

Three concern measurement and throughput: `validity` conflated "the decoder emitted
nonsense" with "the mask refused every continuation", so `completed_validity` is now
reported alongside; decoding is 5.81x faster with bitwise-identical predictions; and
one of our own optimizations was reverted after it measured 0.98x with a 300 MB
memory regression.

## Measurements that close whole directions

**The generation budget is not the constraint.** The paper prescribes 384 candidates
per spectrum and this lineage evaluated 4 to 8. Scaling 4 to 32 to 128 candidates
returns zero molecules throughout, with the dead-end share moving only `87% -> 81% ->
79.5%`. The failure is deterministic and more attempts cannot reach it.

**Fingerprint quality is not the primary blocker.** On NPLIB1, DreaMS reaches 0.338
and formula-blind MIST 0.421 against true Morgan r=2/4096. The paper reaches 16.94%
with DreaMS, so the choice of encoder is not what separates us from it.

**The training budget is far larger than the adaptation set warrants.** 6,649 molecules
at global batch 256 is 26 steps per epoch, so 100,000 steps are about 3,850 epochs, and
whole-sequence accuracy on training data reaches 0.956 with the conditioning gain flat.
Held-out validity does not clearly turn, however: the nine points to step 90,000 swing by
up to 0.12 between neighbours and end near their maximum, so a 32-spectrum panel cannot
resolve whether overfitting has begun. The recipe has no validation loss and no early
stopping, so it could not detect a turn either way.

**Mass-reachability pruning works and was switched off.** Enabling it raises candidate
return 5x and mass validity 3.8x, and walking the gold SAFE string of all 32 panel
targets shows it never rejects the correct molecule. It is now wired into the periodic
evaluation behind `--mass-reachability-prune`.

## State on 5 August

| run | step | initial weights | state |
| --- | --- | --- | --- |
| control | 90,000 | from scratch, no fixes | training saturated, held-out flat within panel noise |
| with twelve fixes | 25,000 | from scratch | flat |
| **FRIGID warm start** | starting | **correct weights and architecture** | launched 5 August |

The third run is the first in this project to start from what the paper prescribes.
Its checkpoint was built from the released FRIGID DLM (EMA, 520,000 updates,
sha256 `b6177c2d...`) against the canonical MARLIN config, and reloading it reproduces
all 257 tensors exactly.

## Next steps

1. Read the first evaluation of the FRIGID run at step 10,000. The control reported
   `Validity 0.246` and 5.375 constraint dead ends there, with everything else zero.
   A working warm start should show materially higher validity from the start.
2. Give the recipe a held-out signal it can act on. The molecular evaluation on 32
   spectra swings by up to 0.12 between neighbouring points, which is larger than any
   effect it would need to detect, so a validation loss or a wider panel is required
   before the 3,850-epoch budget can be tuned on evidence rather than guessed.
3. Keep mass-reachability pruning on in evaluation.

One caveat must not be lost. A FRIGID warm start is a necessary condition, not a
sufficient one. FRIGID's decoder is full-sequence while MARLIN makes attention
block-causal at width 8, so the weights arrive from a different attention pattern and
need adaptation; a single zero-shot probe of the warm start also returns zero.

## Lesson

The divergence surfaced only from a line-by-line comparison of the paper against the
code, not from debugging metrics. The real source of weights is not visible in the
ClearML task parameters: it is an environment variable inside the container command
line, and the field that looks like an input, `Args/checkpoint`, is an output written
by the runner after it downloads the artifact. That is why the divergence survived
three weeks unnoticed.

Of 42 divergences found between the paper and this code, exactly one survived the
check "could this account for a zero result". The other 41 are real but small. The
natural response to "nothing works" is to look for many causes and fix everything;
the correct move was to compare the paper with the code sooner, because one
structural error masked the effect of all nine fixes.
