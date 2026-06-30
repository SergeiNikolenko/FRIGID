#!/usr/bin/env bash
#SBATCH --job-name=frigid-spectral-jepa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/slurm_logs/%x-%j.out
#SBATCH --error=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/slurm_logs/%x-%j.err

set -euo pipefail

FRIGID=${FRIGID:-/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head}
DREAMS_ROOT=${DREAMS_ROOT:-/home/nikolenko/work/Projects/mist_molforge_autoresearch}
DREAMS_PY=${DREAMS_PY:-$DREAMS_ROOT/.venv/bin/python}
DREAMS_SRC=${DREAMS_SRC:-$DREAMS_ROOT/src}
BASE_RUN=${BASE_RUN:-$FRIGID/runs/dreams_fingerprint_head_msg_full_20260629T110040Z}
RUN_DIR=${RUN_DIR:-$FRIGID/runs/spectral_jepa_pretraining_$(date -u +%Y%m%dT%H%M%SZ)}

mkdir -p "$RUN_DIR" "$FRIGID/runs/slurm_logs"
exec > >(tee -a "$RUN_DIR/train.log") 2>&1

cd "$FRIGID"

echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse HEAD)"
echo "run_dir=$RUN_DIR"
echo "dreams_python=$DREAMS_PY"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

for required_path in \
  "$DREAMS_PY" \
  "$DREAMS_SRC" \
  "$BASE_RUN/train_dataset/spectra.hdf5" \
  "$BASE_RUN/train_dataset/metadata.csv" \
  "$BASE_RUN/train_dataset/fingerprints.npz" \
  "$BASE_RUN/val_dataset/spectra.hdf5" \
  "$BASE_RUN/val_dataset/metadata.csv" \
  "$BASE_RUN/val_dataset/fingerprints.npz"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required path: $required_path" >&2
    exit 2
  fi
done

cat > "$RUN_DIR/run_identity.json" <<JSON
{
  "utc_start": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id": "${SLURM_JOB_ID:-manual}",
  "host": "$(hostname)",
  "branch": "$(git branch --show-current)",
  "commit": "$(git rev-parse HEAD)",
  "frigid": "$FRIGID",
  "dreams_root": "$DREAMS_ROOT",
  "run_dir": "$RUN_DIR",
  "base_run": "$BASE_RUN"
}
JSON

export PYTHONPATH="$FRIGID/src:$DREAMS_SRC:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$DREAMS_PY" scripts/train_spectral_jepa.py \
  --train-spectra-hdf5 "$BASE_RUN/train_dataset/spectra.hdf5" \
  --train-metadata "$BASE_RUN/train_dataset/metadata.csv" \
  --train-targets-npz "$BASE_RUN/train_dataset/fingerprints.npz" \
  --val-spectra-hdf5 "$BASE_RUN/val_dataset/spectra.hdf5" \
  --val-metadata "$BASE_RUN/val_dataset/metadata.csv" \
  --val-targets-npz "$BASE_RUN/val_dataset/fingerprints.npz" \
  --output-dir "$RUN_DIR/model" \
  --pretrain-epochs 50 \
  --head-epochs 20 \
  --batch-size 512 \
  --num-workers 4 \
  --embed-dim 384 \
  --transformer-layers 6 \
  --attention-heads 8 \
  --ff-dim 1024 \
  --pretrain-lr 3e-4 \
  --head-lr 1e-3 \
  --ema-decay 0.995 \
  --target-fraction 0.25 \
  --target-window-fraction 0.18 \
  --mask-modes mz_window,high_intensity,random \
  --variance-weight 0.05 \
  --covariance-weight 0.005

echo "utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

