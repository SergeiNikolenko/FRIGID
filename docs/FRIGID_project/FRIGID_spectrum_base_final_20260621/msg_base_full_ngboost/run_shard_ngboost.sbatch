#!/bin/bash
#SBATCH --job-name=frigid-base-ngb
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-35%7
#SBATCH --output=/home/nikolenko/work/Projects/FRIGID/repro_runs/20260616T203000Z_msg_paper_repro/msg_base_full_ngboost/logs/shard-%A_%a.out
#SBATCH --error=/home/nikolenko/work/Projects/FRIGID/repro_runs/20260616T203000Z_msg_paper_repro/msg_base_full_ngboost/logs/shard-%A_%a.err
set -euo pipefail
cd /home/nikolenko/work/Projects/FRIGID
BASE=/home/nikolenko/work/Projects/FRIGID/repro_runs/20260616T203000Z_msg_paper_repro/msg_base_full_ngboost
IDX=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
DATA_DIR="$BASE/shard_data/shard_${IDX}"
OUT_DIR="$BASE/shard_outputs/shard_${IDX}"
TOKEN_MODEL=/home/nikolenko/work/Projects/FRIGID/token_models/models/best_ngboost_MSG.joblib
mkdir -p "$OUT_DIR"
if [ ! -s "$TOKEN_MODEL" ]; then
  echo "Missing token model: $TOKEN_MODEL" >&2
  exit 44
fi
echo "START $(date '+%F %T') job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID} CVD=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
.venv/bin/python scripts/benchmark_spec2mol.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --mist-checkpoint repro_cache/mist_msg.pt \
  --dlm-checkpoint repro_cache/DLM.ckpt \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT_DIR" \
  --formula-matches 10 \
  --max-attempts 100 \
  --batch-size 16 \
  --seed 42 \
  --use-shared-cross-attention \
  --token-model "$TOKEN_MODEL" \
  --sigma-lambda 3.0
echo "DONE $(date '+%F %T') task=${SLURM_ARRAY_TASK_ID}"
