# MARLIN reproduction status, 15 July to 6 August 2026

## Summary

The goal is to reproduce MARLIN (arXiv:2607.04774), de novo molecular structure
generation from an MS/MS spectrum. The paper reports `Exact@1 = 16.94%` on NPLIB1
in the formula-unknown setting with the DreaMS encoder.

Through 5 August this reproduction reported `Exact@1 = 0%` everywhere, including a
completed 100,000-step run. On 6 August the first non-zero value appeared: `0.0312`
at step 30,000 of a run started from the released FRIGID checkpoint, reproduced
exactly by an independent offline evaluation. That is one molecule of 32 and far from
the paper, but it is the first departure from zero in three weeks.

The cause of the zero was identified on 4 and 5 August and it is structural: **every
adaptation run before that started from the wrong decoder.** The paper's only stated
initialization is a warm start from the released FRIGID model. This lineage instead
adapted a from-scratch decoder trained for 30,000 steps, and that decoder's
architecture made a FRIGID warm start impossible to apply.

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

**The generation budget was not the constraint, and now it is.** Measured before the
warm start, scaling 4 to 32 to 128 candidates returned zero molecules throughout, with
the dead-end share moving only `87% -> 81% -> 79.5%`: the failure was deterministic and
more attempts could not reach it. That has changed. On the step=30000 warm-start
checkpoint with the prune and both vocabulary restrictions, over the identical 23
spectra and with only the budget differing:

| | returned | mass valid | Exact@1 | dead ends per attempt |
| --- | ---: | ---: | ---: | ---: |
| 8 candidates | 9/23 (0.391) | 9 | 1 | 0.620 |
| 64 candidates | 12/23 (0.522) | 12 | 1 | 0.677 |

Candidate return rises by a third when the budget rises eightfold, so the failure mode
is now probabilistic rather than deterministic, and the paper's 384 candidates finally
have a mechanism through which to help. The exact hit reproduces at 64 candidates on
the same spectrum, there with 6 candidates returned instead of 1.

Both arms then completed all 32 spectra:

| | 8 candidates | 64 candidates |
| --- | ---: | ---: |
| Exact@1 | 0.0312 | 0.0312 |
| Exact@10 | 0.0312 | 0.0312 |
| candidate return | 0.2812 | **0.5312** |
| uniqueness | 0.2634 | **0.4981** |
| mass validity | 0.1594 | 0.1879 |
| Tanimoto@10 | 0.2731 | 0.3201 |
| validity | 0.2539 | 0.2104 |
| completed validity | 0.6679 | 0.6442 |
| dead ends per attempt | 0.680 | 0.712 |
| panel wall clock | 1.50 h | 15.23 h |

The honest reading has two halves. An eightfold budget nearly doubles candidate return
and uniqueness, and moves Tanimoto@10 from 0.2731 to 0.3201, so the extra candidates
are both more numerous and somewhat closer to the target. It produces **no additional
exact hit**: both Exact@1 and Exact@10 stay at the same single molecule. So the regime
change is established for returning candidates, while whether exact accuracy scales
with budget is not, and cannot be settled on a panel whose floor is 1/32.

That result also bounds the paper protocol in wall clock. At 15.23 h for 32 spectra and
64 candidates, one spectrum-candidate costs about 27 s. The paper's 384 candidates on
this same 32-spectrum panel is therefore about 91 h, and on the full 803-spectrum
validation split about 2,290 GPU-hours, roughly 95 days on one device. That is after the
4.87x already gained from the support restrictions. The paper protocol needs either
parallelism across many workers or another order of magnitude in decode speed; it is not
reachable by waiting.

**Fingerprint quality is not the primary blocker.** On NPLIB1, DreaMS reaches 0.338
at threshold 0.95 and formula-blind MIST 0.421 at threshold 0.15 against true Morgan
r=2/4096. An earlier revision reported 0.376 for MIST, measured on a threshold grid
that started at 0.30 and missed its optimum; MIST is calibrated far lower than DreaMS
and each predictor needs its own threshold. The paper reaches 16.94% with DreaMS, so
the choice of encoder is not what separates us from it.

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
evaluation behind `--mass-reachability-prune`. Re-measured end to end on the
step=30000 warm-start checkpoint at 8 candidates, the prune moves candidate return
`0.0312 -> 0.2500`, mass validity `0.0208 -> 0.1125` and uniqueness `0.0078 -> 0.2321`
while `Exact@1` holds at 0.0312, so it does not cost the hit.

**Two thirds of the vocabulary can never be right, and withholding it is free.** The
tokenizer is inherited from the pretraining corpus and can write 123 distinct
elements; 479 entries are isotope-labelled bracket atoms and 959 introduce an element
outside CHNOPS and the halogens. Tokenizing all 7,144 adaptation targets uses none of
either group. Withholding both in the mask rather than only in the sampler logits cuts
mass-reachable support per position from 1,208 to 234 tokens and runs the mask 4.59x
faster; on the full panel the isotope ban alone cuts wall clock from 7.30 h to 4.19 h
with `Exact@1`, candidate return and uniqueness unchanged. Only 53 of the 1,880 entries appear in any target; narrowing to
those would be fitting the answer set, so it was not done.

All four configurations were then run end to end on the same 32-spectrum panel,
same checkpoint, same seed, 8 candidates:

| | no prune | prune | prune + isotope | prune + isotope + elements |
| --- | ---: | ---: | ---: | ---: |
| Exact@1 | 0.0312 | 0.0312 | 0.0312 | 0.0312 |
| completed validity | 0.5156 | 0.5707 | 0.5368 | **0.6679** |
| mass validity | 0.0208 | 0.1125 | 0.1146 | **0.1594** |
| candidate return | 0.0312 | 0.2500 | 0.2500 | **0.2812** |
| uniqueness | 0.0078 | 0.2321 | 0.2321 | **0.2634** |
| dead ends of 8 | 6.63 | 5.28 | 5.06 | 5.44 |
| Tanimoto@1 | 1.0000 | 0.2889 | 0.2938 | 0.2731 |
| panel wall clock | 0.53 h | 7.30 h | 4.19 h | **1.50 h** |

The element restriction is therefore not only a throughput result: against the prune
alone it raises completed validity by 17%, mass validity by 42% and candidate return
by 12.5% while running 4.87x faster. Two rows move the other way and are recorded as
such: Tanimoto@1 on returned candidates falls from 0.2889 to 0.2731, and dead ends per
eight attempts rise from 5.06 to 5.44 against the isotope-only arm, which is
consistent with a narrower support wasting fewer attempts on chemistry that cannot
finish while walling off more individual branches. Exact@1 is unchanged everywhere
because one molecule of 32 is this panel's resolution floor, so no configuration can
be separated from another on it. The Tanimoto@1 of 1.0000 in the no-prune column is an
artifact of a single returned candidate that happened to be the exact answer.

## State on 5 August

| run | step | initial weights | Exact@1 | state |
| --- | --- | --- | --- | --- |
| control | 100,000 | from scratch, no fixes | 0.0000 at all ten points | completed; train token and sequence accuracy both 1.000, loss 0.002 |
| with twelve fixes | 40,000 | from scratch | 0.0000 | stopped after four consecutive declining points, validity `0.0898 -> 0.0586` |
| **FRIGID warm start** | 32,000 | **released FRIGID DLM** | **0.0312 at step 30,000** | running |

The FRIGID run's trajectory is the first that improves:

| step | validity | dead ends | mass validity | candidate return | Exact@1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.0039 | 7.844 | 0.0000 | 0.0000 | 0.0000 |
| 20,000 | 0.0547 | 7.438 | 0.0000 | 0.0000 | 0.0000 |
| 30,000 | 0.1328 | 6.625 | 0.0208 | 0.0312 | 0.0312 |
| 40,000 | 0.2031 | 6.125 | 0.0156 | 0.0312 | 0.0312 |
| 50,000 | 0.1992 | 6.156 | 0.0156 | 0.0312 | 0.0312 |
| 60,000 | 0.1602 | 6.375 | — | 0.0312 | 0.0312 |

The hit has now held at three consecutive evaluation points rather than vanishing,
so it is not a single lucky draw. It does not yet show a rising rate, and cannot:
one molecule of 32 is the resolution floor of this panel, so `Exact@1 = 0.0312`
three times is consistent with any true rate between roughly 1.6% and 4.7%.
Resolving movement needs a wider panel or more candidates, not more steps.

Nothing should be read into the 40,000 to 50,000 interval in either direction. Every
change across it — validity `0.2031 -> 0.1992`, completed validity `0.6849 -> 0.6599`,
dead ends `6.125 -> 6.156` — is smaller than the up-to-0.12 swing this 32-spectrum
panel shows between neighbouring points, which is what forced the earlier withdrawal
of a claimed peak-and-decline. What is outside that band is the rise in completed
validity over the run as a whole, `0.0312 -> 0.6599`.

Step 60,000 needs a different reading and is deliberately not yet a claim. Exact@1 held
for a fourth consecutive point, but completed validity fell `0.6599 -> 0.5005` and
validity `0.1992 -> 0.1602`. The completed-validity drop of 0.159 is larger than the
0.12 swing band, so unlike the previous interval it is not obviously noise. Two points
are still not a trend, and a claimed peak-and-decline on this same panel had to be
withdrawn once already, so this is recorded as the thing to watch at step 70,000 rather
than as a turn.

Note that it is *below* the control on validity at the same steps, 0.133 against 0.293,
while being the only run with a non-zero Exact@1. The control learned to write
syntactically valid strings that are never the target; validity therefore cannot serve
as the progress indicator, because on it the best run looks like the worst.

The third run is the first in this project to start from what the paper prescribes.
Its checkpoint was built from the released FRIGID DLM (EMA, 520,000 updates,
sha256 `b6177c2d...`) against the canonical MARLIN config, and reloading it reproduces
all 257 tensors exactly.

## Withdrawn claims

Four readings were published and then withdrawn against later measurement. They are
listed because the pattern matters more than any of them.

| claim | why it fell |
| --- | --- |
| held-out validity peaked at step 60,000 and then declined | step 90,000 returned 0.3164, the second highest value; the 32-spectrum panel swings by up to 0.12 between neighbouring points |
| formula-blind MIST reaches only 0.376 | the threshold grid started at 0.30; MIST's optimum is 0.15, where it reaches 0.421 |
| the FRIGID run is not learning | it left a plateau after about 10,000 steps; loss fell from 16.0 to 0.78 and token accuracy rose from 0.318 to 0.972 |
| the block-causal mask destroys the FRIGID weights | the comparison was unseeded, and the mass bridge is randomly initialized |

The last one produced a finding of its own. `load_frigid_decoder` transfers every
architecture-compatible weight and leaves the Fourier mass encoder new, so its
projection is randomly initialized. Across five seeds with identical weights and data,
token accuracy on the pretraining corpus ranges `0.0938` to `0.1387`, a 1.48x spread
from one untrained tensor. Zero-initializing that projection removes the variance, as
all seeds then return exactly `0.0156`, but makes the absolute result worse, so the
obvious repair is not the right one. What this establishes is narrower than it first
appeared: any single measurement of the warm start is a draw from a distribution, not
a value, even though 99.9% of the weights are loaded deterministically from a file.

## Infrastructure

The ClearML worker `aiagent01:gpu0` is registered and accepts tasks but fails within
seconds with `RuntimeError: No CUDA GPUs are available`, thrown from
`trainer.fit` during `strategy.setup_environment()` before any project code runs. It
serves `high_q_80`, `sience_80` and `default`, and because it is usually idle it takes
the next task before the busy workers do. It killed two runs on 5 August, each costing
about an hour to notice. Both were relaunched on the `sience` queue, which only
`aiagent03` serves, and both then ran normally.

## Next steps

1. Read the FRIGID run's step 40,000 and 50,000 evaluations. One molecule of 32 is a
   single hit, not a measured rate; the question is whether Exact@1 rises. Offline
   evaluations of the step 30,000 checkpoint at 8 and 64 candidates, with and without
   the mass-reachability prune, are queued to test whether the failure has become
   probabilistic, since raising the budget previously changed nothing at all.
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
