#!/usr/bin/env bash
#SBATCH --job-name=frigid-dreams-adapter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/slurm_logs/%x-%j.out
#SBATCH --error=/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/slurm_logs/%x-%j.err

set -euo pipefail

FRIGID=${FRIGID:-/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head}
DREAMS_ROOT=${DREAMS_ROOT:-/home/nikolenko/work/Projects/mist_molforge_autoresearch}
DREAMS_PY=${DREAMS_PY:-$DREAMS_ROOT/.venv/bin/python}
DREAMS_SRC=${DREAMS_SRC:-$DREAMS_ROOT/src}
BASE_RUN=${BASE_RUN:-$FRIGID/runs/dreams_fingerprint_head_msg_full_20260629T110040Z}
MIST_TRAIN_RUN=${MIST_TRAIN_RUN:-$FRIGID/runs/mist_dreams_residual_20260629T135651Z/mist_train}
MIST_VAL_RUN=${MIST_VAL_RUN:-$FRIGID/runs/mist_dreams_blend_20260629T131733Z/mist_val}
RUN_DIR=${RUN_DIR:-$FRIGID/runs/dreams_adapter_replacement_$(date -u +%Y%m%dT%H%M%SZ)}
ENABLE_CLEARML=${ENABLE_CLEARML:-1}

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
  "$BASE_RUN/val_dataset/fingerprints.npz" \
  "$MIST_TRAIN_RUN/fingerprints.npz" \
  "$MIST_TRAIN_RUN/metadata.csv" \
  "$MIST_VAL_RUN/fingerprints.npz" \
  "$MIST_VAL_RUN/metadata.csv"; do
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
  "run_dir": "$RUN_DIR"
}
JSON

clearml_args=()
if [[ "$ENABLE_CLEARML" == "1" ]]; then
  clearml_args+=(--clearml)
fi

decoder_args=()
if [[ -n "${DECODER_SENSITIVE_CSV:-}" ]]; then
  if [[ ! -f "$DECODER_SENSITIVE_CSV" ]]; then
    echo "DECODER_SENSITIVE_CSV was set but does not exist: $DECODER_SENSITIVE_CSV" >&2
    exit 2
  fi
  decoder_args+=(--decoder-sensitive-csv "$DECODER_SENSITIVE_CSV")
fi

export PYTHONPATH="$FRIGID/src:$DREAMS_SRC:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$DREAMS_PY" scripts/train_dreams_adapter_replacement.py \
  --train-spectra-hdf5 "$BASE_RUN/train_dataset/spectra.hdf5" \
  --train-metadata "$BASE_RUN/train_dataset/metadata.csv" \
  --train-targets-npz "$BASE_RUN/train_dataset/fingerprints.npz" \
  --train-mist-npz "$MIST_TRAIN_RUN/fingerprints.npz" \
  --train-mist-metadata "$MIST_TRAIN_RUN/metadata.csv" \
  --val-spectra-hdf5 "$BASE_RUN/val_dataset/spectra.hdf5" \
  --val-metadata "$BASE_RUN/val_dataset/metadata.csv" \
  --val-targets-npz "$BASE_RUN/val_dataset/fingerprints.npz" \
  --val-mist-npz "$MIST_VAL_RUN/fingerprints.npz" \
  --val-mist-metadata "$MIST_VAL_RUN/metadata.csv" \
  --output-dir "$RUN_DIR/model" \
  --epochs 60 \
  --warmup-epochs 3 \
  --batch-size 512 \
  --num-workers 4 \
  --head-lr 1e-3 \
  --adapter-lr 3e-4 \
  --encoder-lr 2e-5 \
  --mist-anchor-weight 0.5 \
  --mist-error-weight 4.0 \
  --mist-uncertainty-weight 0.5 \
  --false-negative-weight 2.0 \
  --false-positive-weight 1.0 \
  --soft-tanimoto-weight 0.5 \
  --count-loss-weight 0.02 \
  --clearml-task-name "FRIGID DreaMS adapter replacement MSG ${SLURM_JOB_ID:-manual}" \
  "${clearml_args[@]}" \
  "${decoder_args[@]}"

echo "utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
