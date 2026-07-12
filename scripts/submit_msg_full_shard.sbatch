#!/usr/bin/env bash
#SBATCH --job-name=frigid-msg
#SBATCH --partition=gpu-shared
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=72:00:00

set -euo pipefail

: "${FRIGID_REPO:?Set FRIGID_REPO to the Spectrum FRIGID checkout}"
: "${OUT_DIR:?Set OUT_DIR to the shard run directory}"
: "${START_INDEX:?Set START_INDEX within the ordered test split}"
: "${MAX_SPECTRA:?Set MAX_SPECTRA for this shard}"
: "${SOFTMAX_TEMP:?Set SOFTMAX_TEMP}"
: "${MAX_ATTEMPTS:?Set MAX_ATTEMPTS}"

PYTHON_BIN="${PYTHON_BIN:-$FRIGID_REPO/.venv/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INFERENCE_DTYPE="${INFERENCE_DTYPE:-float32}"
mkdir -p "$OUT_DIR"
cat > "$OUT_DIR/RUN_MANIFEST.json" <<JSON
{
  "host": "spectrum",
  "slurm_job_id": "${SLURM_JOB_ID:-unknown}",
  "slurm_partition": "${SLURM_JOB_PARTITION:-unknown}",
  "slurm_gres": "${SLURM_JOB_GRES:-unknown}",
  "code_commit": "$(git -C "$FRIGID_REPO" rev-parse HEAD)",
  "data_dir": "$FRIGID_REPO/data/msg",
  "mist_checkpoint": "$FRIGID_REPO/repro_cache/mist_msg.pt",
  "dlm_checkpoint": "$FRIGID_REPO/repro_cache/DLM.ckpt",
  "split": "test",
  "start_index": $START_INDEX,
  "max_spectra": $MAX_SPECTRA,
  "softmax_temp": $SOFTMAX_TEMP,
  "randomness": 0.1,
  "formula_matches": 10,
  "max_attempts": $MAX_ATTEMPTS,
  "batch_size": $BATCH_SIZE,
  "inference_dtype": "$INFERENCE_DTYPE",
  "fingerprint_source": "mist_binary",
  "seed": 42
}
JSON

cd "$FRIGID_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" \
  scripts/benchmark_dlm_fingerprint_robustness.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir data/msg \
  --mist-checkpoint repro_cache/mist_msg.pt \
  --dlm-checkpoint repro_cache/DLM.ckpt \
  --output-dir "$OUT_DIR" \
  --split test \
  --batch-size "$BATCH_SIZE" \
  --inference-dtype "$INFERENCE_DTYPE" \
  --start-index "$START_INDEX" \
  --max-spectra "$MAX_SPECTRA" \
  --softmax-temp "$SOFTMAX_TEMP" \
  --randomness 0.1 \
  --formula-matches 10 \
  --max-attempts "$MAX_ATTEMPTS" \
  --fingerprint-sources mist_binary \
  --partial-save-every 250 \
  --use-shared-cross-attention \
  --seed 42 \
  --fp-sparsify-mode threshold \
  --fp-threshold 0.187
