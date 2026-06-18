#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="$REPO_ROOT/.autoresearch/artifacts/current"
mkdir -p "$ARTIFACT_DIR"

REMOTE_HOST="${FRIGID_SCORER_HOST:-spectrum}"
REMOTE_REPO="${FRIGID_REMOTE_REPO:-/home/nikolenko/work/Projects/FRIGID}"
REMOTE_RUN_DIR="${FRIGID_REMOTE_SCORER_DIR:-/home/nikolenko/work/Projects/FRIGID/.autoresearch_scorer_runs/current}"
MAX_SPECTRA="${FRIGID_SCORER_MAX_SPECTRA:-16}"
FORMULA_MATCHES="${FRIGID_SCORER_FORMULA_MATCHES:-10}"
MAX_ATTEMPTS="${FRIGID_SCORER_MAX_ATTEMPTS:-100}"
BATCH_SIZE="${FRIGID_SCORER_BATCH_SIZE:-16}"
CUDA_VISIBLE_DEVICES_VALUE="${FRIGID_SCORER_CUDA_VISIBLE_DEVICES:-0}"

if [[ "${ALLOW_CONCURRENT_FRIGID_SCORER:-0}" != "1" ]]; then
  if ssh "$REMOTE_HOST" -- "tmux has-session -t frigid_spectrum_base 2>/dev/null"; then
    echo "Refusing to run scorer while frigid_spectrum_base is active on $REMOTE_HOST." >&2
    echo "Set ALLOW_CONCURRENT_FRIGID_SCORER=1 only if the active full run may be interrupted by scorer load." >&2
    exit 42
  fi
fi

ssh "$REMOTE_HOST" -- "bash -seuo pipefail" <<REMOTE
cd "$REMOTE_REPO"
rm -rf "$REMOTE_RUN_DIR"
mkdir -p "$REMOTE_RUN_DIR"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
START_TS=\\\$(date +%s)
python scripts/benchmark_spec2mol.py \\
  --config configs/spec2mol_benchmark_msg.yaml \\
  --mist-checkpoint repro_cache/mist_msg.pt \\
  --dlm-checkpoint repro_cache/DLM.ckpt \\
  --data-dir repro_cache/msg \\
  --output-dir "$REMOTE_RUN_DIR/output" \\
  --split test \\
  --max-spectra "$MAX_SPECTRA" \\
  --batch-size "$BATCH_SIZE" \\
  --formula-matches "$FORMULA_MATCHES" \\
  --max-attempts "$MAX_ATTEMPTS" \\
  --seed 42
END_TS=\\\$(date +%s)
python - <<'PY'
import json, pathlib, csv, os
run = pathlib.Path(os.environ.get("REMOTE_RUN_DIR", "$REMOTE_RUN_DIR"))
stats_path = run / "output" / "aggregate_statistics.json"
stats = json.load(open(stats_path))
total = max(float(stats.get("total_spectra", 0) or 0), 1.0)
metrics = {
    "seconds_per_case": float(stats.get("elapsed_time_seconds", 0.0)) / total,
    "elapsed_time_seconds": float(stats.get("elapsed_time_seconds", 0.0)),
    "total_spectra": int(stats.get("total_spectra", 0) or 0),
    "exact_top1": float(stats.get("exact_match_top1", 0.0)),
    "exact_top10": float(stats.get("exact_match_top10", 0.0)),
    "tanimoto_top1": float(stats.get("tanimoto_top1_mean", 0.0)),
    "tanimoto_top10": float(stats.get("tanimoto_top10_mean", 0.0)),
    "formula_success": float(stats.get("formula_match_success_rate", 0.0)),
    "avg_total_generated": float(stats.get("avg_total_generated", 0.0)),
    "generation_time_percentage": float(stats.get("generation_time_percentage", 0.0)),
}
json.dump(metrics, open(run / "metrics.json", "w"), indent=2)
PY
REMOTE

rsync -az "$REMOTE_HOST:$REMOTE_RUN_DIR/metrics.json" "$ARTIFACT_DIR/metrics.json"
rsync -az "$REMOTE_HOST:$REMOTE_RUN_DIR/output/aggregate_statistics.json" "$ARTIFACT_DIR/aggregate_statistics.json"
rsync -az "$REMOTE_HOST:$REMOTE_RUN_DIR/output/detailed_results.csv" "$ARTIFACT_DIR/detailed_results.csv"
