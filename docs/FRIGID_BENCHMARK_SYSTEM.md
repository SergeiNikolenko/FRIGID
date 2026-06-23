# FRIGID Benchmark System

Date: 2026-06-23

This document defines the reproducible benchmark procedure for FRIGID quality comparisons. It complements `docs/FRIGID_OPERATIONAL_USAGE.md`, which documents the stable CLI.

## Goals

Every reported result should answer:

- Which split and dataset were used?
- Which MIST, DLM, NGBoost, and optional ICEBERG checkpoints were used?
- Which generation parameters were used?
- How many expected spectra were evaluated?
- Were any expected rows missing?
- Which exact, Tanimoto, formula, MIST, and runtime metrics were produced?

## Benchmark Tiers

| Tier | Purpose | Size | Expected runtime | Required output |
|---|---|---:|---|---|
| Smoke | Verify wiring after code changes. | 1-5 spectra | Minutes | Manifest plus aggregate statistics. |
| Balanced diagnostic | Compare model/config quality quickly. | 200 spectra | Remote GPU job | Aggregate statistics plus per-spectrum CSV. |
| Full MSG reproduction | Publication-grade MassSpecGym number. | Full test split | Remote multi-GPU job | Full coverage manifest, all shards, aggregate, QA summary. |
| Oracle fingerprint ceiling | Separate MIST from DLM limitations. | 20, 200, or hard subsets | Remote GPU job | Oracle aggregate plus comparison against baseline. |
| ICEBERG bounded scaling | Test inference-time scaling mechanics. | 20-50 spectra first | Remote CPU/GPU job | Round traces, ICEBERG logs, aggregate, timing. |

## Canonical Command Path

Use the CLI for new benchmark runs:

```bash
frigid benchmark \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir data/msg \
  --mist-checkpoint checkpoints/mist_msg.pt \
  --dlm-checkpoint checkpoints/DLM.ckpt \
  --scaler ngboost \
  --token-model token_models/models/best_ngboost_MSG.joblib \
  --use-shared-cross-attention \
  --split test \
  --max-spectra 200 \
  --output-dir runs/msg_balanced_200_ngboost
```

For ICEBERG:

```bash
frigid benchmark \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir data/msg \
  --mist-checkpoint checkpoints/mist_msg.pt \
  --dlm-checkpoint checkpoints/DLM.ckpt \
  --scaler iceberg \
  --token-model token_models/models/best_ngboost_MSG.joblib \
  --iceberg-gen-ckpt checkpoints/iceberg/gen.ckpt \
  --iceberg-inten-ckpt checkpoints/iceberg/inten.ckpt \
  --num-rounds 2 \
  --num-unique-to-refine 16 \
  --masks-per-molecule 4 \
  --max-spectra 50 \
  --output-dir runs/msg_iceberg_50_r2
```

The CLI writes `run_manifest.json` after the underlying benchmark script finishes. The manifest records command, git commit, dirty status, runtime, inputs, checkpoints, parameters, and expected output paths.

## Required Run Artifacts

| File | Required | Meaning |
|---|---|---|
| `run_manifest.json` | Yes for new CLI runs | Reproducibility record. |
| `aggregate_statistics.json` or `oracle_aggregate_statistics.json` | Yes | Primary metric summary. |
| `detailed_results.csv` | Yes | Per-spectrum metrics and diagnostics. |
| `predictions.csv` or `oracle_predictions.csv` | Yes | Candidate predictions. |
| `config.yaml` | Recommended | Resolved benchmark config. |
| Logs | Recommended | Runtime failures, warnings, shard progress. |
| QA summary | Required for full runs | Coverage, denominator, symlink, and schema validation. |

## Metrics

| Metric | Meaning |
|---|---|
| `total_spectra` | Number of spectra included in the aggregate. |
| `exact_match_top1` | Fraction where top candidate has the target InChIKey. |
| `exact_match_top10` | Fraction where any top-10 candidate has the target InChIKey. |
| `tanimoto_top1_mean` | Mean Morgan fingerprint Tanimoto for the top candidate. |
| `tanimoto_top10_mean` | Mean best Morgan fingerprint Tanimoto in top 10. |
| `mist_tanimoto_mean` | Mean similarity between MIST-predicted and target fingerprint. |
| `avg_formula_matches` | Mean number of generated molecules matching target formula. |
| `formula_match_success_rate` | Share of spectra with at least one formula-matched candidate. |
| `spectra_per_second` | End-to-end throughput from aggregate timing. |
| `never_matched_rate` | Share of spectra with no formula-matched candidate. |

For full reproduction runs, also report `manifest_test_samples`, `missing_test_objects`, `exact_top1_missing_as_failures`, and `exact_top10_missing_as_failures`.

## Baseline Summary Generation

Use `scripts/summarize_benchmark_runs.py` to turn completed runs into comparable JSON and CSV:

```bash
python3 scripts/summarize_benchmark_runs.py \
  --run msg_full_ngboost=docs/FRIGID_project/FRIGID_experiments_20260618/E28_msg_base_full_ngboost_aggregate \
  --qa-summary msg_full_ngboost=docs/FRIGID_project/FRIGID_spectrum_base_final_20260621/reproduction_qa_sources/qa_summary.json \
  --run iceberg_msg_50_r2_cefix=docs/FRIGID_project/FRIGID_experiments_20260618/E37_iceberg_scaling_msg_50_r2_cpuiceberg_cefix/results \
  --run oracle_fingerprint_balanced_200=docs/FRIGID_project/FRIGID_experiments_20260618/E26_oracle_fingerprint_balanced_200 \
  --output-json docs/FRIGID_project/FRIGID_project_report_20260622/benchmark_baselines.json \
  --output-csv docs/FRIGID_project/FRIGID_project_report_20260622/benchmark_baselines.csv
```

This script reads existing artifacts only. It does not run models and is safe for local use.

## Full-Run Quality Gate

A full benchmark should not be promoted to a final reproduction unless:

1. The aggregate row count equals the manifest split count.
2. Every shard listed in the run plan is present.
3. No copied shard-data symlink is broken in the published artifact package.
4. `predictions.csv` and `detailed_results.csv` have matching spectrum IDs.
5. RDKit validity and formula checks are rerun in a controlled environment.
6. The final report includes both observed-denominator and missing-as-failure metrics.

The existing full MSG aggregate does not pass this gate because the QA package reports 474 missing manifest test objects and broken copied symlinks.

## Comparison Protocol

When comparing two experiments:

1. Use the same split and spectrum subset.
2. Use the same target formula source.
3. Use the same top-k metric definitions.
4. Use the same missing-row policy.
5. Report checkpoint paths and hashes where available.
6. Compare against at least one MIST diagnostic and one oracle fingerprint ceiling when the change touches conditioning or decoding.

## Recommended Benchmark Matrix

| Change type | Required checks |
|---|---|
| MIST encoder or thresholding | MIST balanced-200, NGBoost balanced-200, oracle fingerprint balanced-200 as ceiling. |
| DLM architecture or sampling | NGBoost smoke, NGBoost balanced-200, oracle fingerprint balanced-200, hard oracle-miss subset. |
| Formula or length model | NGBoost balanced-200 with formula success and never-matched rate. |
| ICEBERG code | ICEBERG 20-spectrum smoke, ICEBERG 50-spectrum bounded run, round trace analysis. |
| Packaging or CLI changes | Smoke benchmark plus manifest inspection. |

## Current Baselines

The current baseline summary is stored in:

- `docs/FRIGID_project/FRIGID_project_report_20260622/benchmark_baselines.json`
- `docs/FRIGID_project/FRIGID_project_report_20260622/benchmark_baselines.csv`

The full MSG aggregate should be used as diagnostic evidence only until a clean full rerun passes the quality gate.
