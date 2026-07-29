#!/bin/bash
#SBATCH --job-name=marlin-score
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/netstorage/nikolenko/marlin/runs/autoresearch/slurm/score-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="${MARLIN_CODE_ROOT:-$ROOT/code}"
PYTHON="$ROOT/.venv/bin/python"
export MARLIN_REPRO_ROOT="${MARLIN_REPRO_ROOT:-$ROOT}"
export PYTHONPATH="$CODE/src:$ROOT/vendor"
export HF_HOME=/mnt/netstorage/nikolenko/marlin/cache/huggingface
export TOKENIZERS_PARALLELISM=false

source "$CODE/scripts/marlin_runtime_env.sh"
cd "$CODE"
"$PYTHON" scripts/autoresearch_marlin_score.py "$@"
