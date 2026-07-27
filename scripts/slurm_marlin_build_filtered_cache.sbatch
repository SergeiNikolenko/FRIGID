#!/bin/bash
#SBATCH --job-name=marlin-cache
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/cache-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
source "$CODE/scripts/marlin_runtime_env.sh"
export PYTHONPATH="$CODE/src:$ROOT/vendor"

"$PYTHON" "$CODE/scripts/build_marlin_filtered_prefix_cache.py" \
  --snapshot-manifest /mnt/netstorage/nikolenko/marlin/safe-gpt-16d0be9ad6177ae683a32a86204530e8ee624a0f/manifest.json \
  --dataset datamol-io/safe-gpt \
  --revision 16d0be9ad6177ae683a32a86204530e8ee624a0f \
  --file-list-sha256 ca5356076d4e6e4920a019d55af0a769b0984d736f3feb9799045f3c49244e6f \
  --tokenizer "$ROOT/data/safe-gpt/tokenizer.json" \
  --exclude-inchikeys "$ROOT/data/nplib1_test_inchikeys.csv" \
  --rows 100000 \
  --output-dir /mnt/netstorage/nikolenko/marlin/cache/filtered-safe-prefix-v1
