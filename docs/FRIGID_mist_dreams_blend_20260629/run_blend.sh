#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a "/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_blend_20260629T131733Z/blend.log") 2>&1
echo RUN=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_blend_20260629T131733Z
date -u
cd "/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head"
echo branch=$(git branch --show-current)
echo commit=$(git rev-parse HEAD)
"/home/nikolenko/work/Projects/FRIGID/.venv/bin/python" - <<'PY'
import sys, torch
print('python', sys.executable)
print('torch', torch.__version__)
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
PYTHONPATH="/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/src" "/home/nikolenko/work/Projects/FRIGID/.venv/bin/python" scripts/export_mist_fingerprints.py   --config configs/spec2mol_benchmark_msg.yaml   --data-dir "/home/nikolenko/work/Projects/FRIGID/repro_cache/msg"   --mist-checkpoint "/home/nikolenko/work/Projects/FRIGID/repro_cache/mist_msg.pt"   --split val   --output-dir "/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_blend_20260629T131733Z/mist_val"   --batch-size 1   --array-key probs
PYTHONPATH="/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/src" "/home/nikolenko/work/Projects/FRIGID/.venv/bin/python" scripts/blend_fingerprint_predictions.py   --primary-npz "/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_blend_20260629T131733Z/mist_val/fingerprints.npz"   --primary-metadata "/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_blend_20260629T131733Z/mist_val/metadata.csv"   --primary-key probs   --secondary-npz "/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_round2_20260629T123254Z/variants/pos10_count005/val_predictions/fingerprints.npz"   --secondary-metadata "/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_round2_20260629T123254Z/variants/pos10_count005/val_predictions/metadata.csv"   --secondary-key probs   --targets-npz "/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_full_20260629T110040Z/val_dataset/fingerprints.npz"   --targets-metadata "/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_full_20260629T110040Z/val_dataset/metadata.csv"   --targets-key ground_truth   --output-dir "/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_blend_20260629T131733Z/blend_val"
date -u
echo DONE
