# MARLIN NPLIB1 validation protocol

This protocol mirrors the validated FRIGID benchmark workflow while scaling the
panels to the smaller NPLIB1 validation split (396 spectra, 394 connectivity
clusters).

## Fixed panels

- `val_micro32`: early technical and futility check.
- `val_micro64` plus `val_macro64`: required paired promotion check. The macro
  panel contains one spectrum per molecule and has no molecule overlap with the
  micro sequence.
- `val_micro128`: optional stability check for close decisions.
- `val_full396`: final validation confirmation.
- `test_locked_full803`: one final held-out result after the recipe is frozen.
  It must not be used for hyperparameter or checkpoint selection.

The micro panels are nested and deterministic. Their selection balances neutral
mass, predicted-fingerprint active-bit count, heavy-atom count, replicate count,
and element-presence indicators. Target structures are used only to construct
balanced fixed panels and compute evaluation metrics; they are never model
inputs or ranking signals. The checked-in selection report records source
hashes, ordered panel hashes, overlap checks, and distribution discrepancies.

## Paired decision contract

Compare a reference and candidate on the same panel with the same spectrum
order, seed, fingerprint source and threshold, candidate budget, temperature,
dropout, grammar and mass-shell settings. Only the intended intervention may
differ, such as the checkpoint after spectrum-fingerprint adaptation.

There are two explicitly different decoding budgets:

- **screening:** 16 decodes per spectrum on `val_micro32`, used only to reject
  broken or clearly futile candidates cheaply;
- **paper-comparable confirmation:** 384 decodes per spectrum, matching Table I
  of MARLIN. This budget is mandatory for promotion on `val_micro64` plus
  `val_macro64`, `val_full396`, and the single locked-test evaluation.

Results produced with the screening budget must be labelled `screening` and
must never be compared numerically with the paper's reported Top-1/Top-10
accuracy. The paper-comparable inference contract is block width 8, 4096-bit
radius-2 Morgan conditioning, 10 ppm acceptance, and conditioning-diversity
dropout 0.3. Exact model identity, checkpoint, seed, threshold, temperature,
and all decoding settings remain part of every run signature.

Report per-spectrum and aggregate:

- Exact@1 and Exact@10;
- Tanimoto@1 and Tanimoto@10, with zero for no returned candidate in paired
  comparisons;
- candidate return rate;
- formula@1 and formula@10;
- validity, mass validity and uniqueness.

Use 10,000 paired bootstrap resamples over molecule-connectivity clusters.
Promote only when micro64 and the independent macro64 agree directionally and
the primary metric confidence interval does not support a material regression.
Exact@1 and Exact@10 are the final primary metrics; loss alone cannot promote a
run. Confirmation uses at least three fixed seeds and reports each seed plus
mean and standard deviation; the final locked-test run uses the frozen recipe
without any further selection.

## Autoresearch prerequisite gates

The fixed-panel research loop uses a lexicographic `research_score` only while
Exact retrieval is still zero:

1. syntactically valid decoding;
2. mass-valid decoding;
3. strict mass-shell candidate return;
4. Exact retrieval.

Each completed gate occupies a disjoint score band, so a small improvement in a
later prerequisite always outranks a perfect earlier prerequisite. Once
Exact@1 or Exact@10 becomes non-zero, the score is `3 + 0.6 * Exact@1 + 0.4 *
Exact@10`. The staged score is a validation search signal, not a paper metric;
all final comparisons continue to report Exact@1 and Exact@10 directly.

Autoresearch scoring must use the checked-in validation manifest, spectrum-
derived DreaMS `probs`, and a fixed candidate/seed budget. Ground-truth
fingerprints are diagnostic upper bounds only and cannot produce a promotable
candidate.

## Commands

Run an evaluation with a fixed panel:

```bash
.venv/bin/python scripts/evaluate_marlin_nplib1.py \
  ... \
  --spec-manifest configs/benchmarks/nplib1_v1/nplib1_val_micro64_v1.tsv \
  --max-spectra 64
```

Compare two completed runs:

```bash
.venv/bin/python scripts/compare_marlin_benchmark_runs.py \
  --reference /path/to/reference \
  --candidate /path/to/candidate \
  --output-dir /path/to/paired-comparison
```

The comparison refuses mismatched stochastic and decoding settings and writes
`comparison_summary.json`, `paired_deltas.csv`, and `bootstrap_ci.json`.
