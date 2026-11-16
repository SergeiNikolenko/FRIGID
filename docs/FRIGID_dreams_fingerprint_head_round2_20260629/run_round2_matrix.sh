#!/usr/bin/env bash
set -euo pipefail

FRIGID=${FRIGID:-/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head}
BASE_RUN=${BASE_RUN:-/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_full_20260629T110040Z}
RUN_DIR=${RUN_DIR:-/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_round2_$(date -u +%Y%m%dT%H%M%SZ)}
DREAMS_PY=${DREAMS_PY:-/home/nikolenko/work/Projects/mist_molforge_autoresearch/.venv/bin/python}
DREAMS_SRC=${DREAMS_SRC:-/home/nikolenko/work/Projects/mist_molforge_autoresearch/src}
export RUN_DIR

mkdir -p "$RUN_DIR/logs"
exec > >(tee -a "$RUN_DIR/round2_matrix.log") 2>&1

echo "RUN_DIR=$RUN_DIR"
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

cd "$FRIGID"
echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse HEAD)"

TRAIN_EMB="$BASE_RUN/train_embeddings/dreams_embeddings.npz"
TRAIN_FP="$BASE_RUN/train_dataset/fingerprints.npz"
VAL_EMB="$BASE_RUN/val_embeddings/dreams_embeddings.npz"
VAL_FP="$BASE_RUN/val_dataset/fingerprints.npz"
VAL_META="$BASE_RUN/val_dataset/metadata.csv"

for required in "$TRAIN_EMB" "$TRAIN_FP" "$VAL_EMB" "$VAL_FP" "$VAL_META"; do
  test -f "$required" || { echo "Missing required artifact: $required" >&2; exit 1; }
done

run_variant() {
  local name="$1"
  shift
  local variant_dir="$RUN_DIR/variants/$name"
  mkdir -p "$variant_dir"

  echo
  echo "===== VARIANT $name ====="
  date -u

  PYTHONPATH="$FRIGID/src:$DREAMS_SRC" "$DREAMS_PY" scripts/train_dreams_fingerprint_head.py \
    --embeddings-npz "$TRAIN_EMB" \
    --fingerprints-npz "$TRAIN_FP" \
    --val-embeddings-npz "$VAL_EMB" \
    --val-fingerprints-npz "$VAL_FP" \
    --output-dir "$variant_dir/fingerprint_head" \
    --hidden-dim 1024 \
    --epochs 25 \
    --batch-size 512 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --threshold 0.5 \
    "$@"

  PYTHONPATH="$FRIGID/src:$DREAMS_SRC" "$DREAMS_PY" scripts/predict_dreams_fingerprint_head.py \
    --checkpoint "$variant_dir/fingerprint_head/dreams_fingerprint_head.pt" \
    --embeddings-npz "$VAL_EMB" \
    --metadata-csv "$VAL_META" \
    --output-dir "$variant_dir/val_predictions" \
    --batch-size 512

  PYTHONPATH="$FRIGID/src:$DREAMS_SRC" "$DREAMS_PY" scripts/sweep_fingerprint_predictions.py \
    --predictions-npz "$variant_dir/val_predictions/fingerprints.npz" \
    --targets-npz "$VAL_FP" \
    --output-dir "$variant_dir/calibration_sweep" \
    --include-target-count-topk
}

run_variant pos1_bce --disable-pos-weight
run_variant pos5_bce --pos-weight 5
run_variant pos10_bce --pos-weight 10
run_variant pos25_bce --pos-weight 25
run_variant pos10_soft025 --pos-weight 10 --soft-tanimoto-weight 0.25
run_variant pos10_count005 --pos-weight 10 --count-loss-weight 0.05
run_variant pos5_soft025_count005 --pos-weight 5 --soft-tanimoto-weight 0.25 --count-loss-weight 0.05

PYTHONPATH="$FRIGID/src:$DREAMS_SRC" "$DREAMS_PY" - <<'PY'
import csv
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
rows = []
for sweep_path in sorted(run_dir.glob("variants/*/calibration_sweep/sweep_metrics.json")):
    variant = sweep_path.parts[-3]
    sweep = json.loads(sweep_path.read_text())["rows"]
    best = sweep[0]
    train_metrics = json.loads(
        (run_dir / "variants" / variant / "fingerprint_head" / "training_metrics.json").read_text()
    )
    history = train_metrics["history"]
    best_train = max(history, key=lambda row: row.get("mean_tanimoto", -1.0))
    rows.append(
        {
            "variant": variant,
            "best_calibration_mode": best["mode"],
            "best_calibration_value": best["value"],
            "best_calibrated_mean_tanimoto": best["mean_tanimoto"],
            "best_calibrated_median_tanimoto": best["median_tanimoto"],
            "best_calibrated_bit_precision": best["bit_precision"],
            "best_calibrated_bit_recall": best["bit_recall"],
            "best_calibrated_bit_f1": best["bit_f1"],
            "best_calibrated_mean_active_bits": best["mean_active_bits"],
            "mean_target_active_bits": best["mean_target_active_bits"],
            "best_training_epoch": best_train["epoch"],
            "training_threshold_mean_tanimoto": best_train["mean_tanimoto"],
            "training_threshold_bit_f1": best_train["bit_f1"],
            "pos_weight": train_metrics["pos_weight"],
            "disable_pos_weight": train_metrics["disable_pos_weight"],
            "soft_tanimoto_weight": train_metrics["soft_tanimoto_weight"],
            "count_loss_weight": train_metrics["count_loss_weight"],
        }
    )

rows.sort(key=lambda row: row["best_calibrated_mean_tanimoto"], reverse=True)
(run_dir / "leaderboard.json").write_text(json.dumps({"rows": rows}, indent=2))
with (run_dir / "leaderboard.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY

date -u
echo "DONE"
