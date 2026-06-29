#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
FRIGID=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head
DATA=/home/nikolenko/work/Projects/FRIGID/repro_cache/msg
DLM=/home/nikolenko/work/Projects/FRIGID/repro_cache/DLM.ckpt
FRIGID_PY=/home/nikolenko/work/Projects/FRIGID/.venv/bin/python
DREAMS_PY=/home/nikolenko/work/Projects/mist_molforge_autoresearch/.venv/bin/python
DREAMS_SRC=/home/nikolenko/work/Projects/mist_molforge_autoresearch/src
mkdir -p "$RUN_DIR/logs"
exec > >(tee -a "$RUN_DIR/full_run.log") 2>&1

echo "RUN_DIR=$RUN_DIR"
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
cd "$FRIGID"
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

PYTHONPATH="$FRIGID/src" "$FRIGID_PY" scripts/prepare_dreams_fingerprint_dataset.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir "$DATA" \
  --split train \
  --output-dir "$RUN_DIR/train_dataset"

PYTHONPATH="$FRIGID/src" "$FRIGID_PY" scripts/prepare_dreams_fingerprint_dataset.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir "$DATA" \
  --split val \
  --output-dir "$RUN_DIR/val_dataset"

PYTHONPATH="$FRIGID/src:$DREAMS_SRC" "$DREAMS_PY" scripts/extract_dreams_embeddings.py \
  --mgf "$RUN_DIR/train_dataset/spectra.mgf" \
  --output-dir "$RUN_DIR/train_embeddings" \
  --batch-size 128

PYTHONPATH="$FRIGID/src:$DREAMS_SRC" "$DREAMS_PY" scripts/extract_dreams_embeddings.py \
  --mgf "$RUN_DIR/val_dataset/spectra.mgf" \
  --output-dir "$RUN_DIR/val_embeddings" \
  --batch-size 128

PYTHONPATH="$FRIGID/src:$DREAMS_SRC" "$DREAMS_PY" scripts/train_dreams_fingerprint_head.py \
  --embeddings-npz "$RUN_DIR/train_embeddings/dreams_embeddings.npz" \
  --fingerprints-npz "$RUN_DIR/train_dataset/fingerprints.npz" \
  --val-embeddings-npz "$RUN_DIR/val_embeddings/dreams_embeddings.npz" \
  --val-fingerprints-npz "$RUN_DIR/val_dataset/fingerprints.npz" \
  --output-dir "$RUN_DIR/fingerprint_head" \
  --hidden-dim 1024 \
  --epochs 20 \
  --batch-size 512 \
  --threshold 0.5

PYTHONPATH="$FRIGID/src:$DREAMS_SRC" "$DREAMS_PY" scripts/predict_dreams_fingerprint_head.py \
  --checkpoint "$RUN_DIR/fingerprint_head/dreams_fingerprint_head.pt" \
  --embeddings-npz "$RUN_DIR/val_embeddings/dreams_embeddings.npz" \
  --metadata-csv "$RUN_DIR/val_dataset/metadata.csv" \
  --output-dir "$RUN_DIR/val_predictions" \
  --batch-size 512

PYTHONPATH="$FRIGID/src:$DREAMS_SRC" "$DREAMS_PY" scripts/evaluate_fingerprint_encoder.py \
  --predictions-npz "$RUN_DIR/val_predictions/fingerprints.npz" \
  --targets-npz "$RUN_DIR/val_dataset/fingerprints.npz" \
  --metadata-csv "$RUN_DIR/val_dataset/metadata.csv" \
  --output-dir "$RUN_DIR/val_evaluation" \
  --threshold 0.5

PYTHONPATH="$FRIGID/src" "$FRIGID_PY" scripts/benchmark_spec2mol.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir "$DATA" \
  --dlm-checkpoint "$DLM" \
  --encoder-backend dreams_precomputed \
  --fingerprint-npz "$RUN_DIR/val_predictions/fingerprints.npz" \
  --fingerprint-metadata "$RUN_DIR/val_predictions/metadata.csv" \
  --fingerprint-probability-key probs \
  --split val \
  --max-spectra 200 \
  --formula-matches 10 \
  --max-attempts 100 \
  --batch-size 16 \
  --use-shared-cross-attention \
  --output-dir "$RUN_DIR/val_dlm_benchmark_200"

date -u
echo "DONE"
