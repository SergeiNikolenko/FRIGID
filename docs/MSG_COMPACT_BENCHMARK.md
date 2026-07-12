# MSG Compact Benchmark

## Purpose

The compact benchmark reduces iteration time without replacing the locked
1,024-spectrum and full MassSpecGym evaluations. It combines two complementary
views of the production test distribution:

- Micro panels reproduce the 17,082-row distribution, including repeated
  spectra for common molecules.
- The macro panel samples unique connectivity blocks and measures transfer to
  molecules that do not occur in the micro or prior diverse panels.

The selector never uses target quality metrics or model predictions. Target
labels are joined only to verify identity and molecule-disjointness.

## Generate and validate

```bash
uv run python scripts/build_msg_compact_benchmark.py \
  --metadata-csv runs/msg_full/mist_export/metadata.csv \
  --labels-tsv data/msg/labels.tsv \
  --fingerprints-npz runs/msg_full/mist_export/fingerprints.npz \
  --spectrum-dir data/msg/spec_files \
  --exclude-manifest configs/benchmarks/msg_diverse_dev64_v1.tsv \
  --exclude-manifest configs/benchmarks/msg_diverse_confirm200_v1.tsv \
  --exclude-manifest configs/benchmarks/msg_diverse_gate1024_v1.tsv \
  --output-dir runs/benchmarks/msg_compact_v1
```

The command validates the 17,082-row grain, unique spectrum identifiers, full
label and file coverage, fingerprint order, formula parsing, nested micro
panels, and zero forbidden molecule overlap. It fails closed when an official
panel exceeds its distribution threshold.

Observed maximum standardized mean differences were `0.0552`, `0.0888`, and
`0.0968` for micro128/256/512, and `0.1156` for macro64. Maximum categorical
share differences were at most `0.0023` for micro and `0.0083` for macro. All
locked panels passed their predefined limits.

## Run a model

```bash
uv run python scripts/benchmark_dlm_fingerprint_robustness.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir data/msg \
  --mist-checkpoint checkpoints/mist_msg.pt \
  --dlm-checkpoint checkpoints/DLM.ckpt \
  --spec-manifest configs/benchmarks/msg_compact_micro256_v1.tsv \
  --batch-size 16 \
  --softmax-temp 1.0 \
  --randomness 0.1 \
  --formula-matches 10 \
  --max-attempts 100 \
  --fingerprint-sources mist_binary \
  --use-shared-cross-attention \
  --seed 42 \
  --fp-sparsify-mode threshold \
  --fp-threshold 0.187 \
  --output-dir runs/benchmarks/compact_candidate
```

## Compare paired runs

Micro panels contain repeated molecules, so confidence intervals must resample
complete molecule clusters:

```bash
uv run python scripts/compare_paired_benchmark_runs.py \
  --reference runs/benchmarks/compact_reference/detailed_results.csv \
  --candidate runs/benchmarks/compact_candidate/detailed_results.csv \
  --bootstrap-unit molecule \
  --cluster-column target_inchi_key \
  --output-dir runs/benchmarks/compact_candidate/paired_comparison
```

Use row bootstrap only for a panel that contains exactly one row per molecule.

## Decision protocol

1. Use 16-32 rows only for runtime smoke tests.
2. Freeze a hypothesis after the existing development gate; do not tune on the
   compact panels.
3. Use micro128 only for futility rejection. It cannot establish an improvement.
4. Require agreement on micro256 and molecule-disjoint macro64 before promotion.
5. Run micro512 only for a predeclared borderline result.
6. Promote only candidates with a useful effect and a positive paired cluster-
   bootstrap interval, then verify on 1,024 and full data.

Compact results are screening evidence. The locked 1,024 and full evaluations
remain the final quality evidence.
