# FRIGID Operational Usage

This document describes the stable project-facing FRIGID entrypoints. Research
orchestration systems should call these commands instead of reaching into
benchmark internals directly.

## Entry Points

FRIGID exposes three intended surfaces:

- `frigid predict`: run FRIGID on one or more spectra from a configured dataset
  split and write prediction artifacts.
- `frigid benchmark`: run a comparable benchmark over a split or split subset.
- `python scripts/train.py`: train the DLM model from Hydra configs.

The CLI currently reuses the repository benchmark scripts for model loading,
generation, and metrics so output remains comparable with existing paper
reproduction runs.

## NGBoost Prediction

```bash
frigid predict \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir data/msg \
  --mist-checkpoint checkpoints/mist_msg.pt \
  --dlm-checkpoint checkpoints/DLM.ckpt \
  --scaler ngboost \
  --token-model token_models/models/best_ngboost_MSG.joblib \
  --use-shared-cross-attention \
  --max-spectra 1 \
  --output-dir runs/example_msg_ngboost
```

Inputs:

- `data/msg/labels.tsv`
- `data/msg/split.tsv`
- `data/msg/spec_files/`
- `data/msg/subformulae/default_subformulae/`
- `checkpoints/mist_msg.pt`
- `checkpoints/DLM.ckpt`
- `token_models/models/best_ngboost_MSG.joblib`

Outputs:

- `aggregate_statistics.json`: aggregate quality and timing metrics.
- `detailed_results.csv`: one row per spectrum with target, proposal, timing,
  formula matching, duplicate, and diagnostic fields.
- `predictions.csv`: target SMILES, spectrum name, and ranked predicted SMILES.
- `config.yaml`: resolved benchmark configuration.
- `run_manifest.json`: command, repo revision, runtime, inputs, checkpoints, and
  output paths.

## NGBoost Benchmark

```bash
frigid benchmark \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir data/msg \
  --mist-checkpoint checkpoints/mist_msg.pt \
  --dlm-checkpoint checkpoints/DLM.ckpt \
  --scaler ngboost \
  --token-model token_models/models/best_ngboost_MSG.joblib \
  --use-shared-cross-attention \
  --formula-matches 10 \
  --max-attempts 100 \
  --batch-size 16 \
  --output-dir runs/msg_ngboost_benchmark
```

Use `--max-spectra` only for smoke tests. Paper-like runs should use the full
configured split and must record the exact command and checkpoint hashes from
`run_manifest.json`.

## DLM Fingerprint Robustness

Use the paired robustness benchmark to isolate decoder sensitivity to the MIST
fingerprint interface. It evaluates the same spectra, formula filter, DLM
checkpoint, seed, and generation settings with different fingerprint sources.

```bash
python scripts/benchmark_dlm_fingerprint_robustness.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir data/msg \
  --mist-checkpoint checkpoints/mist_msg.pt \
  --dlm-checkpoint checkpoints/DLM.ckpt \
  --use-shared-cross-attention \
  --formula-matches 10 \
  --max-attempts 100 \
  --batch-size 16 \
  --partial-save-every 100 \
  --fingerprint-sources ground_truth mist_binary \
  --max-spectra 64 \
  --output-dir runs/msg_dlm_fp_robustness_smoke
```

Outputs:

- `aggregate_statistics.json`: metrics by fingerprint source and
  `mist_binary - ground_truth` deltas.
- `detailed_results.csv`: one row per spectrum and fingerprint source.
- `paired_comparison.csv`: per-spectrum metric deltas for sensitivity analysis.
- `fingerprint_drift.csv`: MIST-vs-ground-truth FP error summary.
- `predictions_ground_truth.csv` and `predictions_mist_binary.csv`: compatible
  with the existing `multi_compute.py`/`compute_metrics.py` analysis format.
- `run_state.json`: completion state and progress. During long runs,
  `aggregate_statistics.json`, CSV outputs, and `run_state.json` are refreshed
  every `--partial-save-every` spectra so interrupted runs still leave usable
  partial metrics.

Use a small subset for smoke tests, then rerun on the full split remotely. A
large gap between `ground_truth` and `mist_binary` indicates decoder brittleness
to the MIST interface; a small gap shifts attention toward ranking, formula
filtering, ICEBERG refinement, or MIST recall itself.

## MIST-Fingerprint DLM Adaptation

Export train-split MIST fingerprints without changing the MIST checkpoint:

```bash
python scripts/export_mist_fingerprints.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir data/msg \
  --mist-checkpoint checkpoints/mist_msg.pt \
  --split train \
  --output-dir results/mist_fingerprints_train
```

Fine-tune DLM on the exported predicted fingerprints:

```bash
python scripts/train.py --config-name fp2mol_finetune_mist_fingerprints \
  load_weights_only=checkpoints/DLM.ckpt \
  data.predicted_fingerprint_metadata=results/mist_fingerprints_train/metadata.csv \
  data.predicted_fingerprint_npz=results/mist_fingerprints_train/fingerprints.npz \
  data.predicted_fingerprint_key=mist_binary
```

Evaluate the fine-tuned checkpoint with the robustness benchmark and compare it
against the original DLM on both `ground_truth` and `mist_binary` sources. This
tests whether decoder adaptation helps without modifying MIST.

## ICEBERG Scaling

```bash
frigid benchmark \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir data/msg \
  --mist-checkpoint checkpoints/mist_msg.pt \
  --dlm-checkpoint checkpoints/DLM.ckpt \
  --scaler iceberg \
  --token-model token_models/models/best_ngboost_MSG.joblib \
  --iceberg-gen-ckpt checkpoints/iceberg/iceberg_msg_model1.ckpt \
  --iceberg-inten-ckpt checkpoints/iceberg/iceberg_msg_model2.ckpt \
  --iceberg-python-path python \
  --num-rounds 3 \
  --batch-size 128 \
  --output-dir runs/msg_iceberg_benchmark
```

ICEBERG mode requires a working `ms-pred` installation and its forward model
dependencies. Keep ICEBERG outputs separate from NGBoost outputs when comparing
throughput because ICEBERG uses a different scoring/refinement path.

## Training

```bash
python scripts/train.py --config-name fp2mol_pretraining
```

The training harness keeps string fields such as `formula: list[str]` on the
host while moving tensor fields to the training device. This avoids Lightning
batch-transfer failures when formula and fingerprint conditioning are enabled.

## Artifact Layout

Use one directory per run:

```text
runs/
  <timestamp>_<mode>_<scaler>/
    run_manifest.json
    aggregate_statistics.json
    detailed_results.csv
    predictions.csv
    config.yaml
```

Do not overwrite previous run directories. Treat `run_manifest.json` as the
minimum provenance needed to compare results across hosts or commits.

## Environment Notes

Use `uv sync` for a fresh clone:

```bash
git clone --recurse-submodules <repo-url>
cd frigid
uv sync
source .venv/bin/activate
frigid --help
```

ICEBERG is part of the default project environment. The clone must include the
`ms-pred` submodule because `pyproject.toml` installs it as an editable path
dependency.

```bash
git submodule update --init --recursive
uv sync
```

The base Python environment includes:

- `transformers==4.32.0`
- `safe-mol==0.1.13`
- `ngboost>=0.5.1`
- `bionemo-moco==0.0.2.1`

ICEBERG mode additionally needs the `ms-pred` submodule and its runtime stack,
including DGL/Ray dependencies matching the target CUDA installation.
