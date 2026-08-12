# What the recorded decode shows, and what to fix

Measured on the checkpoint the clean panel used (`control-r2 … step=100000`), the
321-spectrum clean-panel run, the 396-spectrum validation fingerprints, and a
token-by-token trace of two spectra recorded with
`evaluate_marlin_nplib1.py --trace-output`.

Everything below is a count, not an impression. Reproduce with
`scripts/build_attempt_fan.py`, `scripts/enrich_bit_panel.py` and the audit
snippet at the end.

---

## 1. The conditioning is the largest error source

DreaMS predicts 4,096 Morgan bit probabilities from the spectrum; the run keeps
those at p ≥ 0.95 and hands them to the decoder with the precursor mass.

| Measure (396 validation spectra) | Value |
|---|---|
| Median recall of the real Morgan bits | **0.449** |
| Median precision | **0.467** |
| Mean real bits on | 46.9 |
| Mean predicted bits on | 47.9 |
| Spectra whose recall is below 0.5 | **229 of 396 (58%)** |

So the decoder is typically told about half of what the molecule contains, and
about half of what it is told is not there. The count is roughly right, the
identity is not.

The trace shows what that costs. For benzaldehyde (`O=Cc1ccccc1`, 106 Da) the
prediction keeps the bare-oxygen bit (`O`, p = 0.998) but loses every bit that
places that oxygen:

| Bit | Substructure | Predicted p |
|---|---|---|
| 3052 | `C=O` | 0.074 |
| 1063 | `cC=O` | 0.004 |
| 963 | `cc(c)C=O` | 0.0002 |
| 2980 | `ccc(C=O)cc` | 0.0001 |

The decoder therefore opens with an aromatic carbon at p = 0.995 (`C` 0.005,
`O` 1e-5) and never writes the aldehyde. It is answering the question it was
asked.

**Fixes, in order of expected effect**

1. Train the decoder on predicted fingerprints, not on ground-truth Morgan bits.
   The decoder currently learns from a clean fingerprint and is evaluated on one
   with ~45% recall — a train/test shift that no decoding rule can repair.
2. Feed probabilities instead of a 0.95 threshold. The threshold throws away
   calibration the encoder produced and turns a soft claim into a hard fact.
3. Report conditioning recall per spectrum next to every decode metric, so a
   decoding defect is never confused with an encoding defect.
4. Consider a bit subset that is worth conditioning on: bits whose predicted
   probability is informative (high AUC per bit), rather than all 4,096.

## 2. The vocabulary admits chemistry that cannot exist

| Measure (1,880 tokens) | Value |
|---|---|
| Withheld by the chemistry filter | 1,286 |
| Tokens carrying charge ≥ 2 | 411 |
| …of those, **still admitted** | **45** |
| Hydrogen-family tokens | 46 |

Admitted examples: `[C+4]`, `[CH4+2]`, `[CH+2]`, `[Cl+3]`, `[ClH3+2]`, `[Br+3]`,
`[I+4]`, `[I-3]`, `[N+2]`. Hydrogen family includes `[8H]`, `[2HH]`, `[3HH]`,
`[1H]`, `[HH]`, `[H-]` and isotopes of mercury and holmium.

`[H+7]` appears in a recorded attempt, so these are not theoretical: the mask
lets the sampler spend mass on them.

**Fixes**

1. Extend `forbidden_token_ids` to everything with |formal charge| ≥ 2, and to
   the isotope-hydrogen family (`isotope_token_ids` catches `[2H]` but not
   `[2HH]`, `[3HH]`, `[8H]`).
2. Add a unit test that walks the whole vocabulary and asserts every admitted
   token parses as a chemically possible fragment with a sane valence, so a new
   tokenizer cannot reintroduce them.
3. Reconsider whether standalone hydrogen fragments (`[H]`, `[H+]`, `[HH]`)
   belong in the vocabulary at all: they are the padding material of §3.

## 3. Half the returned candidates are not one molecule

321-spectrum clean-panel run, 8 candidates per spectrum:

| Measure | Value |
|---|---|
| Spectra returning at least one candidate | **133 of 321 (41%)** |
| Spectra where all 8 attempts died | **188 of 321 (59%)** |
| Candidates returned | 247 |
| Candidates made of more than one fragment | **115 of 247 (47%)** |
| …of those, padded with one-atom fragments | **90** |
| Fragment histogram | 1: 132 · 2: 69 · 3: 21 · 4: 13 · 5: 4 · 6: 4 · 8: 1 · 10: 1 |
| Mean dead ends per spectrum | 6.2 of 8 attempts |

The mechanism is visible in the trace: after closing a ring the decoder writes
the fragment separator `.` at p = 1.0 and starts a new fragment, then spends the
remaining mass budget on hydrogen fragments — `.` appears 35 times, `[H+]` 15,
`[H]` 10 across the eight benzaldehyde attempts. One attempt reached the length
cap as toluene plus fifteen loose hydrogens.

**Fixes**

1. Keep the connectivity rule (added after this run) and re-measure the same
   panel: this table is the before-picture and should be repeated as after.
2. Refuse a separator when the remaining mass is below the lightest fragment
   worth writing, instead of letting the decoder open a fragment it can only
   fill with hydrogen.
3. Reject one-heavy-atom fragments outright at the candidate gate.
4. Score fragment count in the metric set — a two-piece candidate that hits the
   mass window is not a hit.

## 4. Six of eight attempts die, and the budget is spent on doomed ones

The decoder commits eight positions per forward pass and never backtracks: when
the mask empties, the attempt is abandoned and a fresh one starts from scratch.
With 6.2 of 8 attempts dying, most of the compute buys nothing. The run is CPU
bound (grammar mask and mass-reachability search), so the waste is real work,
not idle GPU.

**Fixes, cheapest first**

1. **Abort an attempt early on its own signal.** A row whose admitted-token
   count has collapsed, or whose top probability has gone flat, or which has
   already opened more fragments than the target mass can justify, is almost
   always heading for a dead end. Killing it frees the per-row mask work, which
   is what dominates the runtime.
2. **Refill the freed slot.** Aborting only helps if a new attempt starts in the
   same batch; otherwise the batch just runs shorter with fewer live rows.
3. **Backtrack to the last split instead of restarting from nothing.** The tree
   shows attempts sharing long prefixes; re-deciding the last block is far
   cheaper than rewriting 100 positions.
4. **Make the mask look one block ahead.** Most dead ends are mass deadlocks:
   the prefix was already unsatisfiable several positions before the mask
   noticed. Checking reachability over the next block, not the next token,
   converts a late death into an early refusal.

Any of these changes recall as well as speed, so each needs the same panel
before and after: candidate-return rate, exact top-1, fragment count.

## 5. The metrics flatter the decoder

On the benzaldehyde spectrum the best attempt covers 88% of the gold heavy atoms
— it built the aromatic ring — while the molecule is wrong and the Tanimoto is
0.14. Atom-level coverage, `validity` and mass-window pass rate all look
respectable on candidates that are chemically nonsense.

**Fixes**

1. Report `candidate_return_rate`, single-fragment rate and exact connectivity
   together; none of them is meaningful alone.
2. Divide validity by completed attempts *and* by all attempts, as the run
   already does — but surface both in the summary, not only in the JSON.
3. Add "conditioning recall" as a column, so results can be read as decoder
   quality at a given conditioning quality.

---

## Reproducing the numbers

```python
# vocabulary and conditioning
PYTHONPATH=src ./.venv/bin/python scripts/audit_decoder.py
# returned candidates
cat evaluations/clean-before/shard*/predictions.jsonl > /tmp/clean_before.jsonl
```

The token-by-token evidence is on the decoding page
(`docs/decoding-page/index.html`, tab **Real runs**), built from
`evaluate_marlin_nplib1.py --trace-output` by `scripts/build_attempt_fan.py`.
