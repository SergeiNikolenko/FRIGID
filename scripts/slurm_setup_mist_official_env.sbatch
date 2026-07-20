#!/bin/bash
#SBATCH --job-name=mist-env
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/mist-env-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
SOURCE_ENV="$ROOT/envs/mist_cf_py38"
TARGET_ENV="$ROOT/envs/mist_official_py38"
SOURCE_LOCK="$SOURCE_ENV/requirements.lock.txt"
TARGET_LOCK="$TARGET_ENV/requirements.lock.txt"

if test -e "$TARGET_ENV"; then
  echo "Refusing to overwrite existing $TARGET_ENV" >&2
  exit 2
fi

uv venv --python "$SOURCE_ENV/bin/python" "$TARGET_ENV"
uv pip install \
  --python "$TARGET_ENV/bin/python" \
  --find-links https://download.pytorch.org/whl/torch_stable.html \
  --requirement "$SOURCE_LOCK" \
  h5py==3.11.0
uv pip freeze --python "$TARGET_ENV/bin/python" > "$TARGET_LOCK"

PYTHONPATH="$ROOT/vendor/mist-official/src" \
  "$TARGET_ENV/bin/python" -c 'import h5py, mist.pred_fp, pytorch_lightning, torch'
sha256sum "$TARGET_LOCK"
