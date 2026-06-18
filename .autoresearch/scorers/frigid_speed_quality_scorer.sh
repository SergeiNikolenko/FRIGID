#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="$REPO_ROOT/.autoresearch/artifacts/current"
mkdir -p "$ARTIFACT_DIR"

REMOTE_HOST="${FRIGID_SCORER_HOST:-kolmogorov}"
REMOTE_REPO="${FRIGID_REMOTE_REPO:-/home/nikolenko/work/Projects/FRIGID}"
REMOTE_RUN_DIR="${FRIGID_REMOTE_SCORER_DIR:-/home/nikolenko/work/Projects/FRIGID/.autoresearch_scorer_runs/current}"
MAX_SPECTRA="${FRIGID_SCORER_MAX_SPECTRA:-16}"
FORMULA_MATCHES="${FRIGID_SCORER_FORMULA_MATCHES:-10}"
MAX_ATTEMPTS="${FRIGID_SCORER_MAX_ATTEMPTS:-100}"
BATCH_SIZE="${FRIGID_SCORER_BATCH_SIZE:-16}"
CUDA_VISIBLE_DEVICES_VALUE="${FRIGID_SCORER_CUDA_VISIBLE_DEVICES:-1}"
TOKEN_MODEL="${FRIGID_SCORER_TOKEN_MODEL:-token_models/models/best_ngboost_MSG.joblib}"
SIGMA_LAMBDA="${FRIGID_SCORER_SIGMA_LAMBDA:-3.0}"
SYNC_LOCAL_SOURCE="${FRIGID_SCORER_SYNC_LOCAL_SOURCE:-1}"
PROFILE_GENERATION="${FRIGID_SCORER_PROFILE_GENERATION:-0}"
NUM_TOKENS_UNMASK="${FRIGID_SCORER_NUM_TOKENS_UNMASK:-1}"
FORMULA_PRUNING_CHUNK_SIZE="${FRIGID_SCORER_FORMULA_PRUNING_CHUNK_SIZE:-}"

if [[ "${ALLOW_CONCURRENT_FRIGID_SCORER:-0}" != "1" ]]; then
  if ssh -o StrictHostKeyChecking=accept-new "$REMOTE_HOST" -- "tmux has-session -t frigid_spectrum_base 2>/dev/null"; then
    echo "Refusing to run scorer while frigid_spectrum_base is active on $REMOTE_HOST." >&2
    echo "Set ALLOW_CONCURRENT_FRIGID_SCORER=1 only if the active full run may be interrupted by scorer load." >&2
    exit 42
  fi
fi

if [[ "$SYNC_LOCAL_SOURCE" == "1" ]]; then
  rsync -az \
    "$REPO_ROOT/scripts/benchmark_spec2mol.py" \
    "$REPO_ROOT/src/dlm/model.py" \
    "$REPO_ROOT/src/dlm/sampler.py" \
    "$REPO_ROOT/src/dlm/utils/benchmark_utils.py" \
    "$REPO_ROOT/src/dlm/utils/spec2mol.py" \
    "$REMOTE_HOST:$REMOTE_REPO/"
  ssh -o StrictHostKeyChecking=accept-new "$REMOTE_HOST" -- "bash -seuo pipefail" <<REMOTE_SYNC
cd "$REMOTE_REPO"
mkdir -p scripts src/dlm src/dlm/utils
if [[ -f benchmark_spec2mol.py ]]; then mv benchmark_spec2mol.py scripts/benchmark_spec2mol.py; fi
if [[ -f model.py ]]; then mv model.py src/dlm/model.py; fi
if [[ -f sampler.py ]]; then mv sampler.py src/dlm/sampler.py; fi
if [[ -f benchmark_utils.py ]]; then mv benchmark_utils.py src/dlm/utils/benchmark_utils.py; fi
if [[ -f spec2mol.py ]]; then mv spec2mol.py src/dlm/utils/spec2mol.py; fi
REMOTE_SYNC
fi

ssh -o StrictHostKeyChecking=accept-new "$REMOTE_HOST" -- "bash -seuo pipefail" <<REMOTE
cd "$REMOTE_REPO"
rm -rf "$REMOTE_RUN_DIR"
mkdir -p "$REMOTE_RUN_DIR"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
cmd=(
  python scripts/benchmark_spec2mol.py
  --config configs/spec2mol_benchmark_msg.yaml
  --mist-checkpoint repro_cache/mist_msg.pt
  --dlm-checkpoint repro_cache/DLM.ckpt
  --data-dir repro_cache/msg
  --output-dir "$REMOTE_RUN_DIR/output"
  --split test
  --max-spectra "$MAX_SPECTRA"
  --batch-size "$BATCH_SIZE"
  --formula-matches "$FORMULA_MATCHES"
  --max-attempts "$MAX_ATTEMPTS"
  --seed 42
  --use-shared-cross-attention
  --token-model "$TOKEN_MODEL"
  --sigma-lambda "$SIGMA_LAMBDA"
  --num-tokens-unmask "$NUM_TOKENS_UNMASK"
)
if [[ "$PROFILE_GENERATION" == "1" ]]; then
  cmd+=(--profile-generation)
fi
if [[ -n "$FORMULA_PRUNING_CHUNK_SIZE" ]]; then
  cmd+=(--formula-pruning-chunk-size "$FORMULA_PRUNING_CHUNK_SIZE")
fi
"\${cmd[@]}"
python - <<'PY'
import json, pathlib, csv, os
run = pathlib.Path(os.environ.get("REMOTE_RUN_DIR", "$REMOTE_RUN_DIR"))
stats_path = run / "output" / "aggregate_statistics.json"
stats = json.load(open(stats_path))
total = max(float(stats.get("total_spectra", 0) or 0), 1.0)
elapsed = float(stats.get("elapsed_time_seconds", stats.get("total_elapsed_time_seconds", 0.0)) or 0.0)
metrics = {
    "remote_host": "$REMOTE_HOST",
    "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES_VALUE",
    "batch_size": int("$BATCH_SIZE"),
    "max_spectra": int("$MAX_SPECTRA"),
    "formula_matches": int("$FORMULA_MATCHES"),
    "max_attempts": int("$MAX_ATTEMPTS"),
    "sigma_lambda": float("$SIGMA_LAMBDA"),
    "profile_generation": "$PROFILE_GENERATION" == "1",
    "num_tokens_unmask": int("$NUM_TOKENS_UNMASK"),
    "token_model": "$TOKEN_MODEL",
    "seconds_per_case": elapsed / total,
    "elapsed_time_seconds": elapsed,
    "total_spectra": int(stats.get("total_spectra", 0) or 0),
    "exact_top1": float(stats.get("exact_match_top1", 0.0)),
    "exact_top10": float(stats.get("exact_match_top10", 0.0)),
    "tanimoto_top1": float(stats.get("tanimoto_top1_mean", 0.0)),
    "tanimoto_top10": float(stats.get("tanimoto_top10_mean", 0.0)),
    "formula_success": float(stats.get("formula_match_success_rate", 0.0)),
    "avg_total_generated": float(stats.get("avg_total_generated", 0.0)),
    "avg_unique_valid_smiles": float(stats.get("avg_unique_valid_smiles", 0.0)),
    "avg_duplicate_valid_smiles": float(stats.get("avg_duplicate_valid_smiles", 0.0)),
    "avg_valid_duplicate_rate": float(stats.get("avg_valid_duplicate_rate", 0.0)),
    "avg_formula_duplicate_matches": float(stats.get("avg_formula_duplicate_matches", 0.0)),
    "generation_time_percentage": float(stats.get("generation_time_percentage", 0.0)),
}
for key, value in stats.items():
    if key.startswith(("avg_generation_", "total_generation_profile_")):
        metrics[key] = value
json.dump(metrics, open(run / "metrics.json", "w"), indent=2)
PY
REMOTE

rsync -az "$REMOTE_HOST:$REMOTE_RUN_DIR/metrics.json" "$ARTIFACT_DIR/metrics.json"
rsync -az "$REMOTE_HOST:$REMOTE_RUN_DIR/output/aggregate_statistics.json" "$ARTIFACT_DIR/aggregate_statistics.json"
rsync -az "$REMOTE_HOST:$REMOTE_RUN_DIR/output/detailed_results.csv" "$ARTIFACT_DIR/detailed_results.csv"
rsync -az "$REMOTE_HOST:$REMOTE_RUN_DIR/output/predictions.csv" "$ARTIFACT_DIR/predictions.csv"
