#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head
PY=/home/nikolenko/work/Projects/FRIGID/.venv/bin/python
FULL_RUN=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_full_20260629T110040Z
MIST_TRAIN_RUN=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_residual_20260629T135651Z
MIST_VAL_RUN=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_blend_20260629T131733Z
RUN=${RUN:-/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_distilled_replacement_$(date -u +%Y%m%dT%H%M%SZ)}

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

for path in \
  "$FULL_RUN/train_embeddings/dreams_embeddings.npz" \
  "$FULL_RUN/val_embeddings/dreams_embeddings.npz" \
  "$MIST_TRAIN_RUN/mist_train/fingerprints.npz" \
  "$MIST_VAL_RUN/mist_val/fingerprints.npz"; do
  if [ ! -f "$path" ]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

run_variant() {
  local name=$1
  shift
  mkdir -p "$RUN/variants/$name"
  "$PY" scripts/train_dreams_mist_distilled_head.py \
    --train-embeddings-npz "$FULL_RUN/train_embeddings/dreams_embeddings.npz" \
    --train-embeddings-metadata "$FULL_RUN/train_dataset/metadata.csv" \
    --train-mist-npz "$MIST_TRAIN_RUN/mist_train/fingerprints.npz" \
    --train-mist-metadata "$MIST_TRAIN_RUN/mist_train/metadata.csv" \
    --train-targets-npz "$FULL_RUN/train_dataset/fingerprints.npz" \
    --train-targets-metadata "$FULL_RUN/train_dataset/metadata.csv" \
    --val-embeddings-npz "$FULL_RUN/val_embeddings/dreams_embeddings.npz" \
    --val-embeddings-metadata "$FULL_RUN/val_dataset/metadata.csv" \
    --val-mist-npz "$MIST_VAL_RUN/mist_val/fingerprints.npz" \
    --val-mist-metadata "$MIST_VAL_RUN/mist_val/metadata.csv" \
    --val-targets-npz "$FULL_RUN/val_dataset/fingerprints.npz" \
    --val-targets-metadata "$FULL_RUN/val_dataset/metadata.csv" \
    --output-dir "$RUN/variants/$name" \
    --batch-size 256 \
    --epochs 30 \
    --dropout 0.2 \
    --mist-threshold 0.25 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    "$@" 2>&1 | tee "$RUN/variants/$name/train.log"
}

run_variant distill_balanced \
  --hidden-dim 2048 \
  --depth 2 \
  --target-weight 1.0 \
  --distill-weight 1.0 \
  --mist-error-weight 4.0 \
  --mist-uncertainty-weight 1.0 \
  --count-loss-weight 0.02

run_variant distill_error_heavy \
  --hidden-dim 2048 \
  --depth 2 \
  --target-weight 1.0 \
  --distill-weight 0.5 \
  --mist-error-weight 8.0 \
  --mist-uncertainty-weight 1.0 \
  --false-negative-weight 2.0 \
  --false-positive-weight 1.0 \
  --soft-tanimoto-weight 0.1 \
  --count-loss-weight 0.02

run_variant distill_teacher_heavy \
  --hidden-dim 2048 \
  --depth 2 \
  --target-weight 0.75 \
  --distill-weight 2.0 \
  --mist-error-weight 4.0 \
  --mist-uncertainty-weight 0.5 \
  --count-loss-weight 0.01

run_variant distill_deep \
  --hidden-dim 3072 \
  --depth 3 \
  --target-weight 1.0 \
  --distill-weight 1.0 \
  --mist-error-weight 6.0 \
  --mist-uncertainty-weight 1.0 \
  --soft-tanimoto-weight 0.05 \
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
        "passed_mist_replacement_gate": payload["passed_mist_replacement_gate"],
    })
rows.sort(key=lambda row: row["mean_tanimoto"], reverse=True)
(run / "leaderboard.json").write_text(json.dumps({"rows": rows}, indent=2))
print(json.dumps(rows, indent=2))
PY
