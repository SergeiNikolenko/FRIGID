#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head
PY=/home/nikolenko/work/Projects/FRIGID/.venv/bin/python
DATA=/home/nikolenko/work/Projects/FRIGID/repro_cache/msg
MIST_CKPT=/home/nikolenko/work/Projects/FRIGID/repro_cache/mist_msg.pt
FULL_RUN=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_full_20260629T110040Z
BLEND_RUN=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_blend_20260629T131733Z
RUN=${RUN:-/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_residual_$(date -u +%Y%m%dT%H%M%SZ)}

mkdir -p "$RUN"
cd "$PROJECT"

{
  echo "RUN=$RUN"
  date -u
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "commit=$(git rev-parse HEAD)"
  echo "python=$PY"
  "$PY" - <<'PY'
import torch
print("torch", torch.__version__)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(i, props.name, props.total_memory)
PY
} | tee "$RUN/run_info.txt"

if [ ! -f "$RUN/mist_train/fingerprints.npz" ]; then
  "$PY" scripts/export_mist_fingerprints.py \
    --config configs/spec2mol_benchmark_msg.yaml \
    --data-dir "$DATA" \
    --mist-checkpoint "$MIST_CKPT" \
    --split train \
    --output-dir "$RUN/mist_train" \
    --batch-size 1 \
    --array-key probs 2>&1 | tee "$RUN/mist_train_export.log"
fi

run_variant() {
  local name=$1
  shift
  mkdir -p "$RUN/variants/$name"
  "$PY" scripts/train_mist_dreams_residual.py \
    --train-embeddings-npz "$FULL_RUN/train_embeddings/dreams_embeddings.npz" \
    --train-embeddings-metadata "$FULL_RUN/train_dataset/metadata.csv" \
    --train-mist-npz "$RUN/mist_train/fingerprints.npz" \
    --train-mist-metadata "$RUN/mist_train/metadata.csv" \
    --train-targets-npz "$FULL_RUN/train_dataset/fingerprints.npz" \
    --train-targets-metadata "$FULL_RUN/train_dataset/metadata.csv" \
    --val-embeddings-npz "$FULL_RUN/val_embeddings/dreams_embeddings.npz" \
    --val-embeddings-metadata "$FULL_RUN/val_dataset/metadata.csv" \
    --val-mist-npz "$BLEND_RUN/mist_val/fingerprints.npz" \
    --val-mist-metadata "$BLEND_RUN/mist_val/metadata.csv" \
    --val-targets-npz "$FULL_RUN/val_dataset/fingerprints.npz" \
    --val-targets-metadata "$FULL_RUN/val_dataset/metadata.csv" \
    --output-dir "$RUN/variants/$name" \
    --batch-size 256 \
    --epochs 12 \
    --hidden-dim 1024 \
    --dropout 0.15 \
    "$@" 2>&1 | tee "$RUN/variants/$name/train.log"
}

run_variant conservative_bce \
  --disable-pos-weight \
  --residual-scale 0.25 \
  --lr 3e-4 \
  --weight-decay 1e-4

run_variant balanced_count \
  --disable-pos-weight \
  --residual-scale 0.5 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --count-loss-weight 0.02

run_variant pos5_count \
  --pos-weight 5 \
  --residual-scale 0.5 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --count-loss-weight 0.02

run_variant pos10_soft_count \
  --pos-weight 10 \
  --residual-scale 0.5 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --soft-tanimoto-weight 0.1 \
  --count-loss-weight 0.02

"$PY" - <<'PY' "$RUN"
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
rows = []
for summary_path in sorted((run / "variants").glob("*/summary.json")):
    payload = json.loads(summary_path.read_text())
    rows.append({
        "variant": summary_path.parent.name,
        "best_epoch": payload["best"]["epoch"],
        "mean_tanimoto": payload["best"]["best_metrics"]["mean_tanimoto"],
        "delta_vs_mist": payload["best"]["best_delta_vs_mist"],
        "mode": payload["best"]["best_metrics"]["mode"],
        "value": payload["best"]["best_metrics"]["value"],
        "passed_encoder_gate": payload["passed_encoder_gate"],
    })
rows.sort(key=lambda row: row["mean_tanimoto"], reverse=True)
(run / "leaderboard.json").write_text(json.dumps({"rows": rows}, indent=2))
print(json.dumps(rows, indent=2))
PY
