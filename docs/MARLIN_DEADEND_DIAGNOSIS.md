# Constraint dead ends in the 803-spectrum NPLIB1 run

## Summary

* The variable that separates the 161 all-dead-end spectra from the 304 that returned is the
  **input fingerprint**: thresholded-DreaMS recall against the true Morgan bits, AUC 0.226,
  p = 2e-22, and the per-attempt dead-end rate falls from 0.78 to 0.34 across its quintiles.
  It is not specific to dead ends, separating the mass-miss failures from the returns just as
  strongly (0.293). What decides *which* failure you get is **target size**: neutral mass is
  worth nothing between mass-miss and returned (AUC 0.500, p = 0.99) but separates
  all-dead-end from mass-miss (0.300, p = 3e-12). Light targets dead-end, heavy ones miss the
  mass window. The gold SAFE fragment count and the target's hydrogen mass fraction separate
  nothing.
* Of the 3,093 recorded dead ends, **2,518 (81.4%) are mass-shell deadlocks**, 376 (12.2%) are
  grammar deadlocks with an empty syntax support, 199 (6.4%) are killed by the token-table
  mass shell while the grammar still offered something, and 0 are unexplained. The 12.2%
  **contradicts the established "0% grammar deadlocks with the prune on"**.
* They die **late**: median 87.4% of the target mass committed as heavy atoms, 92.4% counting
  forced hydrogens, 1.06 times the gold heavy-atom count already placed, and only 0.7% have a
  minimum mass above the target. The decoder paints itself into a corner in the last tenth of
  the budget, it does not overshoot early.
* All 91 unparsable terminal strings from the 61 no-parse spectra are terminal under our own
  grammar. RDKit refuses them for kekulization (43), a ring closure duplicating an existing
  bond (33), an aromatic atom outside a ring (11), valence (2) and bracket syntax (2). The
  33 are a defect in our mask, not model error.
* Three defects, section 5: the element and isotope restrictions are enforced per token and the
  vocabulary spells around them while `grammar._ATOM_MASSES` weighs the results at zero
  (1,130 of 3,093 dead-end prefixes carry a foreign element, 0 of 534 accepted candidates do);
  the target mass of a permanently charged molecule is short by a proton, putting 15 of 803
  gold answers outside the run's own acceptance window; and the mask admits a ring closure that
  duplicates an existing bond.
* The mask itself is not the problem: it admits every one of the 803 gold answers whose target
  mass is right, both at the run's revision and at current HEAD.

## What was measured

The completed evaluation over the locked 803-spectrum NPLIB1 test split, formula-unknown,
DreaMS lane, `step=100000` checkpoint, 8 candidates, mass-reachability prune with the
isotope and organic-element restrictions:

```
/mnt/netstorage/nikolenko/marlin/evaluations/full803-c8-100k/predictions.jsonl
```

Run settings that matter here, from `shard00/run_signature.json`: `ppm_tolerance 10.0`,
`valence_slack 4.0`, `mass_reachability_prune true`, `forbid_isotope_tokens true`,
`restrict_organic_elements true`, `chemistry_forbidden_token_count 1287`,
`generation_mode block`, `token_selection multinomial`, `seed 42`, `weights raw`,
`git_commit 7b157ab78de835afbd08ef137e0821e09426a505`.

Every mask probe below was run against commit `1f562ab`, whose `grammar.py`, `mass_shell.py`
and `token_properties.py` are byte-identical to the run's `7b157ab` (`git diff 7b157ab 1f562ab
-- src/marlin/grammar.py src/marlin/mass_shell.py src/marlin/token_properties.py` is empty).
This matters because `1224bb0`, committed while this diagnosis was being measured, narrows the
terminal EOS gate from a hydrogen interval to an exact count; the one place where the two
revisions disagree is called out with both numbers.

Outcome per spectrum, with the 8 attempts of each spectrum broken out. The three attempt
columns account for all 6,424 attempts exactly.

| outcome | spectra | attempts | dead ended | EOS, rejected | mass-valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| returned a candidate | 304 | 2,432 | 824 | 582 | 1,026 |
| valid molecules, none on the mass shell | 277 | 2,216 | 1,421 | 795 | 0 |
| every attempt hit a constraint dead end | 161 | 1,288 | 1,288 | 0 | 0 |
| completed, nothing parsed | 61 | 488 | 397 | 91 | 0 |
| total | 803 | 6,424 | 3,930 | 1,468 | 1,026 |

The dead end is not a property of 161 spectra; it is the fate of **3,930 of 6,424 attempts
(61.2%)**, spread over 727 of 803 spectra, including 824 of the 2,432 attempts inside
spectra that did return an answer. The 161 are the spectra where all 8 attempts died.
0 of 6,424 attempts hit `max_length`, and the longest gold tokenisation is 183 ids including
BOS and EOS against `max_length 256`, which confirms the established finding that the canvas
is not the constraint.

## 1. What separates the 161 from the 304

Sample sizes: 161 all-dead-end, 304 returned, 277 mass-miss, 61 no-parse. AUC is the
Mann-Whitney U scaled to [0, 1]: 0.5 means the two groups are interleaved, and values
below 0.5 mean the first group is the lower one. `(ns)` marks p > 0.05.

| variable | median 161 all-dead-end | median 304 returned | median 277 mass-miss | AUC dead vs returned | AUC dead vs mass-miss | AUC mass-miss vs returned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| neutral mass | 260.1 | 373.2 | 366.2 | 0.287 | 0.300 | 0.500 (ns) |
| heavy atoms | 18.0 | 27.0 | 26.0 | 0.271 | 0.294 | 0.487 (ns) |
| rings | 2.000 | 4.000 | 3.000 | 0.230 | 0.326 | 0.404 |
| aromatic rings | 1.000 | 2.000 | 2.000 | 0.375 | 0.450 (ns) | 0.442 |
| heteroatom fraction | 0.279 | 0.227 | 0.250 | 0.688 | 0.621 | 0.572 |
| nitrogen fraction | 0.074 | 0.000 | 0.051 | 0.715 | 0.559 | 0.668 |
| oxygen fraction | 0.188 | 0.192 | 0.167 | 0.476 (ns) | 0.561 | 0.409 |
| gold SAFE fragments | 4.000 | 4.000 | 5.000 | 0.452 (ns) | 0.420 | 0.534 (ns) |
| gold tokens | 39.0 | 56.0 | 54.0 | 0.298 | 0.316 | 0.487 (ns) |
| hydrogen mass fraction | 0.062 | 0.063 | 0.065 | 0.480 (ns) | 0.466 (ns) | 0.515 (ns) |
| fingerprint Tanimoto | 0.254 | 0.395 | 0.275 | 0.235 | 0.425 | 0.299 |
| fingerprint recall | 0.413 | 0.600 | 0.433 | 0.226 | 0.422 | 0.293 |
| predicted bits at 0.95 | 33.0 | 51.0 | 45.0 | 0.219 | 0.300 | 0.400 |
| true Morgan bits | 36.0 | 50.0 | 46.0 | 0.294 | 0.313 | 0.469 (ns) |

Fingerprint Tanimoto is between the DreaMS prediction thresholded at 0.95 and the true
Morgan r=2, 4096-bit fingerprint; recall is the same intersection over the true bit count,
which removes the size dependence of the raw count.

**Two variables separate, and they separate different things.**

The *input fingerprint* separates success from failure. It is the strongest single
variable against the returned group (predicted bits AUC 0.219, recall 0.226, Tanimoto
0.235, all p < 1e-20), and the per-attempt dead-end rate over all 803 spectra falls
monotonically across its quintiles:

| fingerprint recall quintile (160 or 161 spectra each) | mean recall | mean dead-end rate per attempt |
| --- | ---: | ---: |
| 1 | 0.282 | 0.777 |
| 2 | 0.380 | 0.723 |
| 3 | 0.470 | 0.675 |
| 4 | 0.601 | 0.541 |
| 5 | 0.788 | 0.342 |

But the fingerprint is *not specific to dead ends*: it separates the 277 mass-miss
spectra from the 304 returns just as strongly (recall AUC 0.293, p = 5.6e-18) and barely
separates dead-end from mass-miss (0.422). A bad fingerprint makes the run fail; it does
not decide how.

*Target size* decides which failure you get. Neutral mass is worth nothing between
mass-miss and returned (AUC 0.500, p = 0.99; heavy atoms 0.487, p = 0.58; gold tokens
0.487, p = 0.59), yet it separates all-dead-end from mass-miss (0.300, p = 3.0e-12) and
from returned (0.287). Light targets dead-end; heavy targets produce molecules that miss
the mass window:

| neutral-mass quintile (160 or 161 spectra each) | mean mass | mean dead-end rate per attempt |
| --- | ---: | ---: |
| 1 | 205.6 | 0.800 |
| 2 | 293.0 | 0.558 |
| 3 | 355.0 | 0.588 |
| 4 | 431.1 | 0.525 |
| 5 | 661.2 | 0.588 |

The two effects are not each other in disguise. Split the 465 all-dead-end-plus-returned
spectra into tertiles of both and the dead share moves along both axes:

Share of each cell that is all-dead-end rather than returned, cell size in parentheses:

| fingerprint recall tertile | neutral mass <= 286 Da | <= 408 Da | above 408 Da |
| --- | ---: | ---: | ---: |
| recall <= 0.43 | 0.736 (n=72) | 0.426 (n=47) | 0.444 (n=36) |
| recall <= 0.62 | 0.556 (n=54) | 0.215 (n=65) | 0.250 (n=36) |
| recall above 0.62 | 0.276 (n=29) | 0.093 (n=43) | 0.084 (n=83) |

Both gradients survive conditioning on the other: 0.736 down to 0.276 along the fingerprint
axis at low mass, 0.736 down to 0.444 along the mass axis at low recall.

This refines, and partly contradicts, a sentence in `docs/MARLIN_STATUS_REPORT.md`: "The
returned subset is not simply the light end of the benchmark either, its median neutral mass
being 373 against 338 for the spectra that returned nothing, so coverage is not a size
effect." Both numbers reproduce here (373.2 over 304, 338.1 over 499), but the 338 pools two
populations with opposite behaviour: 260.1 Da median for the 161 all-dead-end spectra against
366.2 for the 277 mass-miss and 360.1 for the 61 no-parse. Coverage is not a size effect
overall; the dead-end half of it is.

Measurements that do **not** separate, and so rule out three plausible stories: the number
of SAFE fragments in the gold answer (AUC 0.452, p = 0.087), the hydrogen mass fraction of
the target (0.480, p = 0.49), and the oxygen fraction (0.476, p = 0.39). The nitrogen and
heteroatom fractions are raised in the dead-end group (0.715, 0.688), but most of that is
the returned group being nitrogen-free natural products: against mass-miss, nitrogen
fraction only reaches 0.559.

## 2. Grammar deadlock or mass-shell deadlock

Every recorded dead-end prefix was replayed through a `SafeGrammarMask` built on the run's
own tokenizer with `forbidden_token_ids` from `isotope_token_ids` and
`foreign_element_token_ids` (1,287 ids), `ppm_tolerance 10.0`, `valence_slack 4.0`, and the
size of `_valid_token_ids(prefix)` compared with `_mass_reachable_token_ids(prefix,
neutral_mass)`.

A dead end fires in the sampler when the mass shell, the forbidden-token list and the
grammar mask leave no finite logit, so a third class exists beside the two asked for: the
grammar still offers something and the token-table mass shell excludes all of it. Every
recorded dead end falls in one of the three, none is unexplained.

| class | definition | all 3,093 dead ends | the 805 of the 161 all-dead-end spectra |
| --- | --- | ---: | ---: |
| mass-shell deadlock | syntax support non-empty, mass-reachable support empty | 2,518 (81.4%) | 651 (80.9%) |
| grammar deadlock | syntax support empty | 376 (12.2%) | 89 (11.1%) |
| token-prune deadlock | mass-reachable non-empty, disjoint from the shell's own budget | 199 (6.4%) | 65 (8.1%) |
| unexplained | both supports non-empty and overlapping | 0 | 0 |

The split barely moves between outcome groups: 80.9 / 11.1 / 8.1 percent for the
all-dead-end spectra against 79.0 / 14.5 / 6.5 for the dead ends inside spectra that did
return an answer. The mechanism is the same everywhere; the 161 are the spectra where all 8
attempts met it.

**This contradicts the established finding that "with the prune on, 0% were grammar
deadlocks".** On the 32-spectrum panel that was true; over the full split 376 of 3,093 dead
ends (12.2%) have an empty *syntax* support, and 89 of the 805 belonging to the 161 (11.1%).
The cause is one specific trap, and all 376 share it:

| property of the 376 grammar deadlocks | count |
| --- | ---: |
| prefix ends inside an unclosed `[` | 376 of 376 |
| prefix ends in the bare `[` followed by digits, 50 distinct mass numbers | 376 of 376 |
| support becomes non-empty once the 1,287-id chemistry block list is removed | 376 of 376 |
| median support without the block list (min 2, max 11) | 9 tokens |

The tails are `[62`, `[56`, `[55`, `[66`, `[201`, ...: the model emitted the bare `[` token
followed by digit tokens, which reads as an isotope mass number. `_partial_bracket_symbol`
then only admits an element that actually has an isotope at that mass, no organic element
has one at 52 to 66, and every element that does is on the block list. The state is legal,
scannable, and has no successor. Section 5.1 is the same defect seen from the other side.

For the other two classes the support left before the dead end is not large: the median
mass-shell deadlock still had 191 syntactically legal tokens of 590 allowed, but the median
token-prune deadlock had 12, and its shell residual was 5.2 Da against 31.2 Da for the
mass-shell class, with a median 27.0 Da disagreement between the shell's mass for the prefix
and the grammar's.

### Is the mask refusing correct chemistry?

Before blaming the mask, it was asked whether it can accept the right answer at all. Every
gold SAFE string was scanned and tested for a terminal state on the mass shell
(`_has_hydrogen_only_exact_mass`):

| group | gold answers | admitted as terminal, run's mask | admitted, HEAD after `1224bb0` | gold mass outside the 10 ppm window |
| --- | ---: | ---: | ---: | ---: |
| all-dead-end | 161 | 158 | 154 | 7 |
| mass-miss | 277 | 277 | 271 | 6 |
| no-parse | 61 | 61 | 61 | 0 |
| returned | 304 | 304 | 302 | 2 |
| total | 803 | 800 | 788 | 15 |

**Every gold answer whose target mass is right is admitted, under both revisions.** The run's
mask admitted 800 of 803; the 3 it refused are all outside the window, at -12.69, 15.02 and
16.38 ppm, and it let the other 12 through because its terminal gate accepted any hydrogen
count in a roughly 6 Da interval, which absorbs the whole-proton errors of section 5.2. The
narrowed gate on current HEAD admits 788 and refuses exactly the 15 whose own gold mass is
outside the window. Either way the mask is not what stops the right answer.

Replaying the gold token sequence through `_mass_reachable_token_ids` position by position says
the same at prefix level. 15 targets were walked, the 10 lightest all-dead-end ones plus all 5
charged ones: under the run's mask **0 of the 9 inside-window targets was refused at any
position**, and the 2 refusals are both outside the window, `mona_1770` at -12.69 ppm and
`CCMSLIB00004679999` at 15.02 ppm. Note which ones survive: a target mass wrong by a whole
proton walks cleanly, because "one hydrogen fewer" is a reachable molecule, while a target
wrong by a fraction of a hydrogen (12.69 ppm is 0.0019 Da) has no integer-hydrogen solution and
is refused mid-string. The prune is faithful to the mass it is given; it is the mass and the
prefix that are wrong.

## 3. Where along the mass budget the dead ends happen

Fractions are of the run's target neutral mass. "Heavy mass placed" is the grammar's own
`sum(atom_masses)` for the prefix; "minimum mass" is `state.minimum_mass(4.0)`, the heavy
mass plus the hydrogens the sealed atoms cannot avoid; "mass-shell heavy mass" is the
`MassShellState` the sampler recorded from the token property table.

### The 805 recorded dead ends of the 161 all-dead-end spectra

| quantity | p10 | p25 | median | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| heavy mass placed / target mass | 0.794 | 0.833 | 0.874 | 0.895 | 0.914 |
| minimum mass (heavy + forced H) / target | 0.841 | 0.888 | 0.924 | 0.953 | 0.973 |
| mass-shell heavy mass / target | 0.818 | 0.860 | 0.899 | 0.942 | 0.991 |
| heavy atoms placed / gold heavy atoms | 0.941 | 1.000 | 1.059 | 1.154 | 1.308 |

### All 3,093 recorded dead ends, 727 spectra

| quantity | p10 | p25 | median | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| heavy mass placed / target mass | 0.821 | 0.859 | 0.888 | 0.907 | 0.922 |
| minimum mass (heavy + forced H) / target | 0.876 | 0.917 | 0.947 | 0.967 | 0.978 |
| mass-shell heavy mass / target | 0.842 | 0.885 | 0.917 | 0.969 | 0.996 |
| heavy atoms placed / gold heavy atoms | 0.955 | 1.000 | 1.059 | 1.154 | 1.261 |

**The decoder does not overshoot the budget early; it paints itself into a corner at the
end.** At the median dead end 87.4% of the target mass is already committed as heavy atoms
and 92.4% once the forced hydrogens are counted, with 1.06 times as many heavy atoms as
the answer needs. Only 6 of 805 dead ends (0.7%), and 25 of all 3,093 (0.8%), have a minimum
mass already above the target; 0 of 3,093 have a heavy mass above it, the maximum being 0.9996
of the target. A hard overshoot is therefore not the mechanism: the prefix is still under the
target and there is simply no token left that lands on it exactly. The last 8 to 10% of the
budget is where the decode dies.

### What the decoder writes in that last stretch

The prefixes are full of bracket hydrogen: `.[H].[H].` tails, and `[HH]`, `[H-]`, `[H+]`
fragments. Because the grammar treats `[H]` as a one-atom molecule with one explicit
hydrogen (`_scan("C[H]")` is refused for exceeding hydrogen's valence, `_scan("C.[H]")` is
not), these can only appear as their own fragments, which is exactly how they appear.
456 of the 805 all-dead-end prefixes (56.6%) hold at least one, 317 (39.4%) end in one, and
they account for 1,758 fragments; over all 3,093 dead ends it is 1,958 (63.3%), 1,299 (42.0%)
and 8,516.

Whether the constraints *forced* that tail was tested by cutting 25 such prefixes back to the
point where their hydrogen tail begins, over 15 distinct spectra, and asking for the support
there, intersected with the mass shell reconstructed at that point:

| support at that point, of 590 allowed tokens | probes | hydrogen spellings | heavier | mass-free |
| --- | ---: | ---: | ---: | ---: |
| 12 tokens | 13 | 5 | 6 | 1 |
| 10 tokens | 4 | 5 | 4 | 1 |
| 6 tokens | 8 | 5 | 0 | 1 |

The mass-shell residual at those points was 2.2 to 68.4 Da, median 40.2, so the budget was not
the binding thing. The answer is therefore "partly": in 8 of 25 probes nothing heavier than
hydrogen was legal at all, and in the other 17 the support had already collapsed to 10 or 12
tokens of which exactly 5 were bracket-hydrogen spellings that no gold answer ever uses. The
decode is not forced into the hydrogen tail every time, but by then it chooses from a handful
of tokens, 5 of which are junk that keeps the string alive without advancing it. Across all
3,093 dead ends, 241 (7.8%) had no positive-mass token fitting the shell residual at all and a
further 594 (19.2%) had only hydrogen-mass tokens fitting.

## 4. The 61 spectra that completed without a parsed molecule

These 61 spectra spent 397 of their 488 attempts on dead ends and reached EOS 91 times; all
91 recorded terminal SAFE strings are in the table below. Not one was truncated by the
512-character recording limit, and none terminated at `max_length`.

Each string was handed to `dlm.utils.utils_chem.safe_to_smiles(s, fix=False)`, the call the
run used, with the RDKit log of `Chem.MolFromSmiles(s)` captured to name the cause.

| RDKit rejection | strings | share of 91 | our mask's verdict on the same string |
| --- | ---: | ---: | --- |
| cannot kekulize an aromatic system | 43 | 47.3% | terminal, accepted |
| ring closure duplicates an existing bond | 33 | 36.3% | terminal, accepted |
| aromatic atom that is not in a ring | 11 | 12.1% | terminal, accepted |
| explicit valence above permitted | 2 | 2.2% | terminal, accepted |
| syntax error inside a bracket atom | 2 | 2.2% | terminal, accepted |
| parses (would be a false bucket) | 0 | 0.0% | — |

**All 91 of 91 are terminal under our own grammar**: `_scan` returns a state with no open
ring label, no open branch and no incomplete token for every one of them, so the mask
believed each was a finished molecule and let the model emit EOS. This bucket is therefore
not "genuine model error" versus "our defect" — it is our mask being lexical where RDKit is
chemical. 32 of 91 also carry an atom the grammar's mass table treats as weightless and 11
carry a non-organic element, which is section 5.1 below.

Two follow-ups worth recording:

* the duplicate-ring-bond bucket (33 of 91) is a defect in our emission, not model error:
  the grammar refuses only a ring label closing onto its own opening atom and never checks
  whether the pair is already bonded. `_scan("C12CC12")`, `_scan("c1ccccc1.C12.C12")` and
  `_scan("C%99C%99")` all return terminal states, and RDKit rejects all three;
* `safe_to_smiles(s, fix=True)`, which the run disabled (`safe_decode_fix false`), rescues
  60 of 91 into some molecule, but **0 of those 60 land inside the 10 ppm mass shell**, so
  enabling it would convert 0 of the 61 spectra into returns. It repairs the string by
  dropping fragments, which moves the mass.

## 5. Defects found

### 5.1 The element and isotope restrictions are per token, and the vocabulary spells around them

`isotope_token_ids` and `foreign_element_token_ids` (`src/marlin/token_properties.py:28`
and `:84`) inspect each vocabulary entry on its own. The vocabulary also holds 216
bracket-opening partials and 222 bracket-closing partials; after the 1,287-id block list,
36 openers (`[`, `[C`, `[N`, `[O`, `[Z`, `[A`, `[H`, ...) and 183 closers (`g]`, `a-]`,
`r]`, `H]`, ...) plus the bare digits remain allowed. Any forbidden atom can therefore be
spelled across several allowed tokens:

| spelling | tokenises as | every piece allowed? |
| --- | --- | ---: |
| `[Og]` | `['[O', 'g]']` | yes |
| `[Po-90]` | `['[P', 'o', '-', '9', '0', ']']` | yes |
| `[H-21]` | `['[H', '-', '2', '1', ']']` | yes |
| `[Na-]` | `['[Na-]']` as one token | no, blocked; `['[N', 'a-]']` is not |

What the run did with that, counted over the recorded strings:

| emitted strings | with a non-organic bracket element | with an isotope-labelled bracket |
| --- | ---: | ---: |
| 3,093 dead-end prefixes | 1,130 (36.5%) | 110 (3.6%) |
| 1,413 unaccepted terminal strings | 193 (13.7%) | 8 (0.6%) |
| 534 accepted candidates | **0** | **0** |

41 distinct foreign elements appear, including Og, Cn, Fl, Hs, Sg, Fm, Np, Pu, Fr and Nd.
The leak never reaches an accepted answer; it only consumes attempts.

The compounding half of the defect is that `marlin.grammar._ATOM_MASSES` has 18 entries and
no hydrogen, so `_advance` files every leaked element *and every bracket hydrogen* at
**mass 0.0** with the default valence 4.0, while `token_properties` gives the same
characters their real mass. The two mass models the decoder runs simultaneously therefore
disagree on the same string:

| string | grammar heavy mass | mass-shell token mass |
| --- | ---: | ---: |
| `[H]`, `[HH]`, `[H-]` | 0.0 (+1 explicit H) | 1.0078 |
| `[Og]` | 0.0 | 294.2139 |
| `[Cr-]` | 0.0 | 51.9405 |
| `[Fe]` | 0.0 | 55.9349 |
| `C` | 12.0 | 12.0 |

Measured consequence: **2,222 of 3,093 dead-end prefixes (71.8%) have the two mass models
disagreeing by more than 0.01 Da**, and 649 (21.0%) disagree even on the heavy-atom count. The
median gap between the mass shell's `heavy_mass` and the grammar's own is 2.016 Da, exactly two
hydrogens, for the 805 all-dead-end prefixes, and 27.0 Da for the 199 token-prune deadlocks.
539 of 3,093 prefixes (17.4%) cannot even be re-tokenised into the state the run recorded,
because the greedy re-encoding merges a composed spelling back into the single forbidden token.
A zero-mass atom is present in 2,240 of 3,093 dead-end prefixes (72.4%), 66.0% of the 805
all-dead-end ones, 77.3% of those inside returned spectra, and 58 of 534 accepted candidates
(10.9%, all bracket hydrogen, no foreign element). The weightless symbols are hydrogen 1,967
times and 41 foreign elements after that, led by Og 174, Cn 135, No 113 and Os 96.

Cost attributable to this defect at the dead-end level: **575 of 3,093 dead ends (18.6%)**,
namely the 376 grammar deadlocks, every one of which is the bare `[` plus digits trap, and the
199 token-prune deadlocks, where the two mass models disagree by a median 27.0 Da.

**A fix is provably admissible on this data.** Over the 7,947 gold answers of train, val and
test, the gold SAFE tokenisation uses **53 distinct vocabulary ids**, of which 9 are
complete bracket atoms (`[C-] [N+] [N-] [O-] [S+] [SH] [n+] [nH] [o+]`), and **0 of the 438
partial bracket tokens, with 0 uses of the bare `[`**. Withholding the partial bracket
spellings therefore admits 7,947 of 7,947 gold answers. Adding hydrogen and the remaining
elements to `_ATOM_MASSES` is a separate, orthogonal repair.

### 5.2 The target mass of a permanently charged molecule is short by a proton

14 of 803 test targets carry a non-zero formal charge. For 13 of them the exact mass of the
gold SMILES minus the run's `neutral_mass` is 1.0073 Da, 1,269 to 5,354 ppm, because
`neutral_mass = precursor_mz - proton` is applied to species that are already cations.

| gold answers outside the run's own 10 ppm window | spectra | of group |
| --- | ---: | ---: |
| all-dead-end | 7 | 4.3% of 161 |
| mass-miss | 6 | 2.2% of 277 |
| no-parse | 0 | 0.0% of 61 |
| returned | 2 | 0.7% of 304 |
| total | 15 | 1.9% of 803 |

13 of the 15 are the charged targets, 2 are precursor error on neutral molecules. Those 15
spectra cannot produce the gold answer as a returned candidate: acceptance is
`MassShellConstraint.accepts_smiles`, an equality between RDKit's exact mass and the run's
target within 10 ppm, and the gold molecule misses it by 1,269 to 5,354 ppm. Under the run's
mask the gate at EOS was wide enough to *emit* 12 of the 15 gold strings anyway, so the failure
was silent; the narrowed gate on current HEAD refuses them at EOS instead, which makes it
visible. The 2 in the returned group returned something that cannot be the right answer.
This is small, 4.3% of the 161, but it is unanswerable by construction, and it belongs with the
hydrogen-accounting work already in flight.

### 5.3 The mask admits a ring closure that duplicates an existing bond

`_advance` refuses a ring label only when it closes onto its own opening atom
(`src/marlin/grammar.py:605`) and never asks whether the two atoms are already bonded.
`_scan("C12CC12")`, `_scan("c1ccccc1.C12.C12")` and `_scan("C%99C%99")` all return terminal
states; RDKit rejects each with "ring closure duplicates bond". This is 33 of the 91
unparsable terminal strings (36.3%). A fix is admissible by construction, since RDKit
refuses such a string and every gold answer is an RDKit-valid molecule, but the empirical
re-scan of the 7,947 gold answers under the fix was **not** run here.

### 5.4 Smaller gaps in the same class

Kekulizability (43 of 91) and "aromatic atom not in a ring" (11 of 91) are chemistry the
lexical mask does not model, and `[Cn--11]` shows `_BRACKET_ATOM`
(`src/marlin/grammar.py:93`) accepting a repeated charge sign RDKit refuses. No verified fix
is proposed for these.

## Commands

Everything above comes from `scripts/diagnose_marlin_dead_ends.py`, run from the repository
root, pinned to four cores and single-threaded in practice, while a 16-shard evaluation held
the rest of the machine. The mask probe is the only expensive step, about one second per
prefix over 3,093 prefixes, so the second command reuses its CSV instead of asking the mask
again.

```
CACHE=/mnt/netstorage/nikolenko/marlin/cache/runtime-inputs-spectrum-v1/16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3
EVAL=/mnt/netstorage/nikolenko/marlin/evaluations/full803-c8-100k

# 1. features, parse failures and the mask probe over every recorded dead end (~55 min)
env -u LD_PRELOAD PYTHONPATH=src taskset -c 20,21,22,23 ./.venv/bin/python \
  scripts/diagnose_marlin_dead_ends.py \
  --predictions $EVAL/predictions.jsonl --cache-dir $CACHE \
  --dead-end-scope all --output /tmp/marlin_deadend/report_all.json

# 2. the same report plus the hydrogen-funnel and gold-walk probes, reusing step 1
env -u LD_PRELOAD PYTHONPATH=src taskset -c 20,21,22,23 ./.venv/bin/python \
  scripts/diagnose_marlin_dead_ends.py \
  --predictions $EVAL/predictions.jsonl --cache-dir $CACHE \
  --dead-end-scope all --reuse-dead-ends /tmp/marlin_deadend/report_all.dead_ends.csv \
  --gold-walk-sample 10 --funnel-sample 25 \
  --output /tmp/marlin_deadend/report_final.json

# 3. step 2 again against the mask the run actually used, since 1224bb0 landed
#    mid-measurement; only the gold-answer probes differ between the two
git worktree add --detach /tmp/marlin_runmask 1f562ab
cp scripts/diagnose_marlin_dead_ends.py /tmp/marlin_runmask/scripts/
(cd /tmp/marlin_runmask && env -u LD_PRELOAD PYTHONPATH=/tmp/marlin_runmask/src \
  <repo>/.venv/bin/python scripts/diagnose_marlin_dead_ends.py \
  --predictions $EVAL/predictions.jsonl --cache-dir $CACHE \
  --dead-end-scope all --reuse-dead-ends /tmp/marlin_deadend/report_all.dead_ends.csv \
  --gold-walk-sample 10 --funnel-sample 25 \
  --output /tmp/marlin_deadend/report_runmask.json)
git worktree remove /tmp/marlin_runmask
```

Step 1 started at 01:16 and `1224bb0` landed at 01:24, so step 1 held the run's mask in memory
throughout; steps 2 and 3 differ only in the gold-answer probes, and the classification, mass
budget, funnel and parse tables are identical in both.

The report and its two CSVs (`report_final.features.csv`, one row per spectrum, and
`report_final.dead_ends.csv`, one row per recorded dead-end prefix) are the artifacts behind
every number here. The run signature quoted at the top is
`$EVAL/shard00/run_signature.json`.

Nothing in `src/` was changed: this is a measurement, and the defects in section 5 are
reported rather than repaired. `tests/test_diagnose_marlin_dead_ends.py` locks the outcome
grouping, the zero-mass atom count and the duplicate-ring-bond repro:

```
env -u LD_PRELOAD PYTHONPATH=src ./.venv/bin/python -m pytest \
  tests/test_diagnose_marlin_dead_ends.py -q -p no:warnings   # 3 passed
```
