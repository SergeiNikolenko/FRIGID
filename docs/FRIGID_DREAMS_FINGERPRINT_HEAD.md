# FRIGID DreaMS Fingerprint-Head Workflow

Date: 2026-06-29

This document defines the first serious DreaMS integration path for FRIGID. The
goal is not to replace the full FRIGID stack at once. The goal is to replace the
weakest proven interface:

```text
MS/MS spectrum -> 4096-bit Morgan fingerprint probabilities
```

The current production path uses MIST for that step. The new path tests:

```text
MS/MS spectrum -> DreaMS embedding -> supervised fingerprint head -> DLM
```

## Why This Path

Existing FRIGID diagnostics show that MIST-to-DLM conditioning is the main
quality bottleneck:

- Oracle fingerprints lift balanced-200 Exact Top-1/Top-10 to 53%.
- MIST threshold, top-N, per-bit calibration, and ridge adapter experiments did
  not solve the fingerprint-quality problem.
- Paired DLM robustness runs show a large quality drop when DLM receives
  MIST-predicted binary fingerprints instead of ground-truth fingerprints.

The safest DreaMS experiment is therefore a controlled spectrum-to-fingerprint
replacement, measured before and after DLM generation.

## Added Components

- `src/frigid/fingerprint_backends.py`
  - Adds a small backend interface for spectrum-to-fingerprint encoders.
  - Keeps `mist` as the default backend.
  - Adds `precomputed` / `dreams_precomputed` backend for probabilities saved
    by a DreaMS fingerprint head.
- `src/frigid/dreams_fingerprint_head.py`
  - Shared MLP fingerprint head and fingerprint-quality metrics.
- `scripts/prepare_dreams_fingerprint_dataset.py`
  - Creates `spectra.mgf`, `metadata.csv`, and target `fingerprints.npz`.
- `scripts/extract_dreams_embeddings.py`
  - Thin wrapper around `dreams.api.dreams_embeddings`.
  - Run this in a DreaMS-capable environment.
- `scripts/train_dreams_fingerprint_head.py`
  - Trains `DreaMS embedding -> 4096-bit Morgan logits`.
- `scripts/predict_dreams_fingerprint_head.py`
  - Writes DreaMS-head probabilities in the precomputed backend format.
- `scripts/evaluate_fingerprint_encoder.py`
  - Computes fingerprint Tanimoto, bit precision/recall/F1, and per-case CSV.
- `configs/dreams_fingerprint_head_msg.yaml`
  - Path and hyperparameter recipe for the MassSpecGym run.

## Remote Run Plan on `spectrum`

Use the remote GPU host for real runs. The local machine should only prepare,
review, and package outputs.

### 1. Prepare Train/Validation Data

```bash
python scripts/prepare_dreams_fingerprint_dataset.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir /home/nikolenko/work/Projects/FRIGID/data/msg \
  --split train \
  --output-dir runs/dreams_fingerprint_head_msg/train_dataset

python scripts/prepare_dreams_fingerprint_dataset.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir /home/nikolenko/work/Projects/FRIGID/data/msg \
  --split val \
  --output-dir runs/dreams_fingerprint_head_msg/val_dataset
```

### 2. Extract DreaMS Embeddings

Run in an environment where DreaMS is installed:

```bash
python scripts/extract_dreams_embeddings.py \
  --mgf runs/dreams_fingerprint_head_msg/train_dataset/spectra.mgf \
  --output-dir runs/dreams_fingerprint_head_msg/train_embeddings \
  --batch-size 128

python scripts/extract_dreams_embeddings.py \
  --mgf runs/dreams_fingerprint_head_msg/val_dataset/spectra.mgf \
  --output-dir runs/dreams_fingerprint_head_msg/val_embeddings \
  --batch-size 128
```

The expected output array key is `embeddings`.

### 3. Train Fingerprint Head

```bash
python scripts/train_dreams_fingerprint_head.py \
  --embeddings-npz runs/dreams_fingerprint_head_msg/train_embeddings/dreams_embeddings.npz \
  --fingerprints-npz runs/dreams_fingerprint_head_msg/train_dataset/fingerprints.npz \
  --val-embeddings-npz runs/dreams_fingerprint_head_msg/val_embeddings/dreams_embeddings.npz \
  --val-fingerprints-npz runs/dreams_fingerprint_head_msg/val_dataset/fingerprints.npz \
  --output-dir runs/dreams_fingerprint_head_msg/fingerprint_head \
  --hidden-dim 1024 \
  --epochs 20 \
  --batch-size 256 \
  --threshold 0.5
```

Primary output:

```text
runs/dreams_fingerprint_head_msg/fingerprint_head/dreams_fingerprint_head.pt
```

### 4. Predict and Evaluate Validation Fingerprints

```bash
python scripts/predict_dreams_fingerprint_head.py \
  --checkpoint runs/dreams_fingerprint_head_msg/fingerprint_head/dreams_fingerprint_head.pt \
  --embeddings-npz runs/dreams_fingerprint_head_msg/val_embeddings/dreams_embeddings.npz \
  --metadata-csv runs/dreams_fingerprint_head_msg/val_dataset/metadata.csv \
  --output-dir runs/dreams_fingerprint_head_msg/val_predictions

python scripts/evaluate_fingerprint_encoder.py \
  --predictions-npz runs/dreams_fingerprint_head_msg/val_predictions/fingerprints.npz \
  --targets-npz runs/dreams_fingerprint_head_msg/val_dataset/fingerprints.npz \
  --metadata-csv runs/dreams_fingerprint_head_msg/val_dataset/metadata.csv \
  --output-dir runs/dreams_fingerprint_head_msg/val_evaluation \
  --threshold 0.5
```

First success criterion:

- The validation mean fingerprint Tanimoto should beat the current MIST
  balanced-200 reference of about 0.52 before any DLM benchmark is run.

### 5. Run DLM with DreaMS-Head Fingerprints

```bash
python scripts/benchmark_spec2mol.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir /home/nikolenko/work/Projects/FRIGID/data/msg \
  --dlm-checkpoint /home/nikolenko/work/Projects/FRIGID/repro_cache/DLM.ckpt \
  --encoder-backend dreams_precomputed \
  --fingerprint-npz runs/dreams_fingerprint_head_msg/val_predictions/fingerprints.npz \
  --fingerprint-metadata runs/dreams_fingerprint_head_msg/val_predictions/metadata.csv \
  --fingerprint-probability-key probs \
  --split val \
  --max-spectra 200 \
  --formula-matches 10 \
  --max-attempts 100 \
  --batch-size 16 \
  --use-shared-cross-attention \
  --output-dir runs/dreams_fingerprint_head_msg/val_dlm_benchmark
```

Compare against the existing MIST baseline, the MIST-to-DLM robustness results,
and the oracle-fingerprint ceiling.

## Completion Criteria

The experiment is complete only when these artifacts exist from a real remote
run:

- Train and validation dataset packages.
- DreaMS embeddings for train and validation.
- Trained fingerprint-head checkpoint.
- Validation fingerprint metrics JSON and per-case CSV.
- DLM benchmark aggregate using `dreams_precomputed`.
- A short result note under `docs/` summarizing exact metrics and conclusions.

## Interpretation Rules

- If DreaMS-head fingerprint Tanimoto does not beat MIST, do not run a full DLM
  benchmark yet; improve the head or fine-tune DreaMS.
- If fingerprint metrics improve but DLM metrics do not, the next step is DLM
  adaptation on DreaMS-predicted fingerprints.
- If both improve, promote DreaMS-head to a first-class encoder backend and run
  full split validation with the benchmark quality gates.
