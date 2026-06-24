# DLM Fingerprint Robustness Results

This note summarizes the initial MIST-to-DLM fingerprint mismatch diagnostic
run. The goal was to isolate whether DLM generation quality drops when the
decoder is conditioned on MIST-predicted fingerprints instead of clean
ground-truth Morgan fingerprints.

## Question

FRIGID uses MIST to predict a molecular fingerprint from an MS/MS spectrum, then
uses DLM to decode a molecule conditioned on that fingerprint. If DLM was mostly
trained or evaluated with clean Morgan fingerprints, the decoder may be brittle
to realistic MIST errors. The robustness benchmark keeps the spectrum, target
formula, formula filter, DLM checkpoint, generation seed, and generation
settings fixed while changing only the fingerprint source.

Compared fingerprint sources:

- `ground_truth`: Morgan fingerprint computed directly from the target SMILES.
- `mist_binary`: MIST probabilities thresholded with the benchmark fingerprint
  threshold.

## Implementation

The workflow added in this change includes:

- `scripts/benchmark_dlm_fingerprint_robustness.py`: paired robustness
  benchmark for clean vs MIST-predicted fingerprints.
- `scripts/export_mist_fingerprints.py`: export MIST fingerprints for DLM
  adaptation without changing the MIST checkpoint.
- `configs/fp2mol_finetune_mist_fingerprints.yaml`: DLM fine-tuning config that
  consumes exported predicted fingerprints.
- `src/dlm/utils/utils_data.py`: dataset/collator support for externally
  provided fingerprints.
- `docs/FRIGID_OPERATIONAL_USAGE.md`: operational commands for the benchmark and
  adaptation workflow.

## Runs

The benchmark was run remotely on `kolmogorov` using an isolated worktree from
commit `910c39f`. Local execution was limited to code inspection and lightweight
syntax/import checks.

Diagnostic settings:

- Dataset/config: MSG test split, `configs/spec2mol_benchmark_msg.yaml`.
- Checkpoints: `repro_cache/mist_msg.pt`, `repro_cache/DLM.ckpt`.
- Fingerprint sources: `ground_truth`, `mist_binary`.
- Formula matches: `2`.
- Max attempts: `20`.
- Batch size: `4`.
- Shared cross-attention enabled.

These settings are intentionally cheaper than a paper-style FRIGID run. They are
intended to measure paired sensitivity, not absolute production accuracy.

## Completed 64-Spectrum Diagnostic

Output directory:

```text
/home/nikolenko/work/Projects/FRIGID_dlm_fp_robustness_910c39f/runs/dlm_fp_robustness_64_20260623T163605Z
```

Aggregate results:

| Metric | `ground_truth` | `mist_binary` | Delta |
| --- | ---: | ---: | ---: |
| Spectra | 64 | 64 | 0 |
| Exact match top-1 | 0.0000 | 0.0000 | 0.0000 |
| Exact match top-10 | 0.0000 | 0.0000 | 0.0000 |
| Tanimoto top-1 mean | 0.5051 | 0.4474 | -0.0576 |
| Tanimoto top-10 mean | 0.5051 | 0.4501 | -0.0550 |
| Formula-match success rate | 0.7813 | 0.7031 | -0.0781 |
| Never-matched rate | 0.2188 | 0.2969 | +0.0781 |
| Average formula matches | 1.4844 | 1.2344 | -0.2500 |
| Average generated molecules | 13.6094 | 15.5469 | +1.9375 |

Sensitivity summary:

| Statistic | Value |
| --- | ---: |
| Mean MIST-vs-ground-truth fingerprint Tanimoto | 0.7545 |
| Median MIST-vs-ground-truth fingerprint Tanimoto | 0.8504 |
| Mean paired top-1 Tanimoto delta | -0.0576 |
| Median paired top-1 Tanimoto delta | -0.0323 |
| Worse cases for `mist_binary` | 40 / 64 |
| Better-or-equal cases for `mist_binary` | 24 / 64 |
| Correlation: MIST fingerprint Tanimoto vs quality delta | +0.5743 |
| Correlation: false-negative bits vs quality delta | -0.5604 |
| Correlation: false-positive bits vs quality delta | -0.5874 |

## Full-Split Attempt

A full test-split run over 17,082 spectra was started on `kolmogorov` and then
stopped on request before completion.

Stopped-run directory:

```text
/home/nikolenko/work/Projects/FRIGID_dlm_fp_robustness_910c39f/runs/dlm_fp_robustness_full_20260623T170455Z
```

The run processed approximately 1,377 / 17,082 spectra, about 8% of the split.
No final aggregate JSON or CSV was produced because the current benchmark writes
its output at normal completion. The benchmark now writes partial outputs every
`--partial-save-every` spectra so future interrupted full runs retain usable
partial metrics.

## Interpretation

The 64-spectrum paired diagnostic supports the fingerprint mismatch hypothesis:
DLM performs worse when conditioned on MIST-predicted binary fingerprints than
when conditioned on clean target-derived fingerprints, while all other
generation inputs are held fixed.

The exact-match metrics are not informative in this cheap diagnostic run because
the generation budget is intentionally small. The meaningful signal is the
paired drop in molecular similarity and the increased formula-filter failure
rate under `mist_binary`.

The correlations suggest that both false-positive and false-negative MIST
fingerprint errors matter for DLM decoding. This makes DLM adaptation on
MIST-predicted fingerprints a plausible next experiment. If adaptation reduces
the `ground_truth` vs `mist_binary` gap, the bottleneck is at least partly DLM
distribution brittleness rather than only MIST recall or downstream ranking.

## 5-Hour Spectrum Partial Run

A larger partial run was executed on `spectrum` with a 5-hour timeout and
partial saves every 25 spectra.

Output directory:

```text
/home/nikolenko/work/Projects/FRIGID_dlm_fp_robustness_2a3c00e/runs/dlm_fp_robustness_5h_spectrum_20260623T230432Z
```

The run stopped by timeout after saving 1,400 / 17,082 spectra. This run is
partial, but it is large enough to make the MIST-to-DLM mismatch signal much
clearer than the initial 64-spectrum diagnostic.

Aggregate results:

| Metric | `ground_truth` | `mist_binary` | Delta |
| --- | ---: | ---: | ---: |
| Spectra | 1,400 | 1,400 | 0 |
| Exact match top-1 | 0.4879 | 0.1386 | -0.3493 |
| Exact match top-10 | 0.4921 | 0.1400 | -0.3521 |
| Tanimoto top-1 mean | 0.8130 | 0.5677 | -0.2453 |
| Tanimoto top-10 mean | 0.8130 | 0.5723 | -0.2407 |
| Formula-match success rate | 0.7643 | 0.6936 | -0.0707 |

Paired sensitivity:

| Statistic | Value |
| --- | ---: |
| Worse cases for `mist_binary` | 1,024 / 1,400 |
| Equal cases | 174 / 1,400 |
| Better cases for `mist_binary` | 202 / 1,400 |
| Mean paired top-1 Tanimoto delta | -0.2453 |
| Median paired top-1 Tanimoto delta | -0.1694 |
| Mean MIST-vs-ground-truth fingerprint Tanimoto | 0.7162 |
| Median MIST-vs-ground-truth fingerprint Tanimoto | 0.7882 |
| Correlation: MIST fingerprint Tanimoto vs quality delta | +0.7948 |
| Correlation: false-negative bits vs quality delta | -0.7381 |
| Correlation: false-positive bits vs quality delta | -0.5945 |

## Main Problem Statement

The current evidence points to the MIST-to-DLM fingerprint interface as a major
FRIGID bottleneck.

DLM can decode substantially better when it receives a clean target-derived
Morgan fingerprint. When the same decoder receives the binary fingerprint
predicted by MIST for the same spectrum, generation quality drops sharply. The
drop is paired, reproducible across a larger partial run, and strongly related
to MIST fingerprint errors.

This means that changing DLM sampling parameters, formula-filter thresholds, or
ranking alone is unlikely to solve the main failure mode. The system needs one
of the following:

- better MIST fingerprint prediction;
- DLM adaptation/fine-tuning on the fingerprint distribution that MIST actually
  emits;
- an interface objective that makes MIST fingerprints more decoder-compatible,
  not just independently fingerprint-like.

The fastest next experiment is DLM adaptation on exported MIST fingerprints
without changing the MIST checkpoint. If this reduces the `ground_truth` vs
`mist_binary` gap, the bottleneck is confirmed as decoder brittleness to MIST's
fingerprint distribution. If it does not, MIST is likely losing too much
structure-relevant information and needs to be improved directly.

## Recommended Next Steps

1. Add periodic partial writes or resume support to the robustness benchmark
   before rerunning the full 17k-spectrum split. Partial writes are now enabled
   by default through `--partial-save-every 100`; resume support is still a
   possible follow-up.
2. Run an intermediate 1,024- or 2,048-spectrum subset with the same paired
   setup to confirm that the 64-spectrum effect is stable.
3. Export train-split MIST fingerprints and fine-tune DLM with
   `configs/fp2mol_finetune_mist_fingerprints.yaml`.
4. Re-evaluate the fine-tuned DLM with the same robustness benchmark and compare
   original vs adapted DLM on both `ground_truth` and `mist_binary`.
