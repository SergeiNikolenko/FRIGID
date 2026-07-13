#!/usr/bin/env bash
#SBATCH --job-name=frigid-msg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=36:00:00

set -euo pipefail

: "${FRIGID_REPO:?Set FRIGID_REPO to a clean Spectrum FRIGID worktree}"
: "${FRIGID_ASSET_ROOT:?Set FRIGID_ASSET_ROOT to the immutable model/data assets}"
: "${OUT_DIR:?Set OUT_DIR to a new shard run directory}"
: "${SOURCE_NAME:?Set SOURCE_NAME to the frozen candidate source name}"
: "${EXPECTED_MANIFEST:?Set EXPECTED_MANIFEST to the ordered full test manifest}"
: "${EXPECTED_CODE_COMMIT:?Set EXPECTED_CODE_COMMIT}"
: "${EXPECTED_CONFIG_SHA256:?Set EXPECTED_CONFIG_SHA256}"
: "${EXPECTED_DATA_SPLIT_SHA256:?Set EXPECTED_DATA_SPLIT_SHA256}"
: "${EXPECTED_LABELS_SHA256:?Set EXPECTED_LABELS_SHA256}"
: "${EXPECTED_MIST_CHECKPOINT_SHA256:?Set EXPECTED_MIST_CHECKPOINT_SHA256}"
: "${EXPECTED_DLM_CHECKPOINT_SHA256:?Set EXPECTED_DLM_CHECKPOINT_SHA256}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256}"
: "${START_INDEX:?Set START_INDEX within the ordered test split}"
: "${MAX_SPECTRA:?Set MAX_SPECTRA for this shard}"
: "${SOFTMAX_TEMP:?Set SOFTMAX_TEMP}"
: "${MAX_ATTEMPTS:?Set MAX_ATTEMPTS}"

PYTHON_BIN="${PYTHON_BIN:-$FRIGID_ASSET_ROOT/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$FRIGID_REPO/configs/spec2mol_benchmark_msg.yaml}"
DATA_DIR="${DATA_DIR:-$FRIGID_ASSET_ROOT/data/msg}"
DATA_SPLIT_PATH="${DATA_SPLIT_PATH:-$DATA_DIR/split.tsv}"
LABELS_PATH="${LABELS_PATH:-$DATA_DIR/labels.tsv}"
MIST_CHECKPOINT="${MIST_CHECKPOINT:-$FRIGID_ASSET_ROOT/repro_cache/mist_msg.pt}"
DLM_CHECKPOINT="${DLM_CHECKPOINT:-$FRIGID_ASSET_ROOT/repro_cache/DLM.ckpt}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INFERENCE_DTYPE="${INFERENCE_DTYPE:-float32}"
CACHE_CONDITIONING="${CACHE_CONDITIONING:-1}"
RUN_MANIFEST="$OUT_DIR/RUN_MANIFEST.json"

for path in \
  "$PYTHON_BIN" \
  "$CONFIG_PATH" \
  "$DATA_SPLIT_PATH" \
  "$LABELS_PATH" \
  "$MIST_CHECKPOINT" \
  "$DLM_CHECKPOINT" \
  "$EXPECTED_MANIFEST"; do
  if [[ ! -e "$path" ]]; then
    echo "Required input does not exist: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUT_DIR" ]]; then
  echo "Shard output directory already exists: $OUT_DIR" >&2
  exit 2
fi

actual_commit="$(git -C "$FRIGID_REPO" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_CODE_COMMIT" ]]; then
  echo "Code commit mismatch: expected $EXPECTED_CODE_COMMIT, got $actual_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$FRIGID_REPO" status --porcelain --untracked-files=all)" ]]; then
  echo "FRIGID_REPO must be a clean worktree: $FRIGID_REPO" >&2
  exit 2
fi

sha256_file() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

verify_sha256() {
  local label="$1"
  local path="$2"
  local expected="$3"
  local actual
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA-256 mismatch: expected $expected, got $actual" >&2
    exit 2
  fi
}

verify_sha256 "Config" "$CONFIG_PATH" "$EXPECTED_CONFIG_SHA256"
verify_sha256 "Data split" "$DATA_SPLIT_PATH" "$EXPECTED_DATA_SPLIT_SHA256"
verify_sha256 "Labels" "$LABELS_PATH" "$EXPECTED_LABELS_SHA256"
verify_sha256 "MIST checkpoint" "$MIST_CHECKPOINT" "$EXPECTED_MIST_CHECKPOINT_SHA256"
verify_sha256 "DLM checkpoint" "$DLM_CHECKPOINT" "$EXPECTED_DLM_CHECKPOINT_SHA256"
verify_sha256 "Expected manifest" "$EXPECTED_MANIFEST" "$EXPECTED_MANIFEST_SHA256"

export FRIGID_REPO OUT_DIR SOURCE_NAME EXPECTED_MANIFEST EXPECTED_CODE_COMMIT
export CONFIG_PATH DATA_DIR DATA_SPLIT_PATH LABELS_PATH MIST_CHECKPOINT DLM_CHECKPOINT
export EXPECTED_CONFIG_SHA256 EXPECTED_DATA_SPLIT_SHA256 EXPECTED_LABELS_SHA256
export EXPECTED_MIST_CHECKPOINT_SHA256 EXPECTED_DLM_CHECKPOINT_SHA256
export EXPECTED_MANIFEST_SHA256 START_INDEX MAX_SPECTRA SOFTMAX_TEMP MAX_ATTEMPTS
export BATCH_SIZE INFERENCE_DTYPE CACHE_CONDITIONING RUN_MANIFEST

mkdir -p "$OUT_DIR"
"$PYTHON_BIN" - <<'PY'
import csv
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path


def read_names(path: Path, column: str, split: str | None = None) -> list[str]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter=delimiter)
        if rows.fieldnames is None or column not in rows.fieldnames:
            raise ValueError(f"Missing {column} column in {path}")
        return [
            str(row[column]).strip()
            for row in rows
            if split is None or str(row.get("split", "")).strip() == split
        ]


expected_path = Path(os.environ["EXPECTED_MANIFEST"])
expected_names = read_names(expected_path, "spec_name")
test_names = read_names(Path(os.environ["DATA_SPLIT_PATH"]), "name", "test")
if expected_names != test_names:
    raise ValueError("Expected manifest does not exactly match the ordered test split")

start = int(os.environ["START_INDEX"])
size = int(os.environ["MAX_SPECTRA"])
if start < 0 or size <= 0 or start + size > len(expected_names):
    raise ValueError(f"Invalid shard range [{start}, {start + size})")


def input_record(path_name: str, hash_name: str) -> dict:
    return {
        "path": str(Path(os.environ[path_name]).resolve()),
        "sha256": os.environ[hash_name],
    }


manifest = {
    "schema_version": 2,
    "purpose": "frigid_msg_full_dlm_shard",
    "status": "running",
    "source_name": os.environ["SOURCE_NAME"],
    "start_timestamp": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS", "unknown"),
    },
    "code": {
        "repo": str(Path(os.environ["FRIGID_REPO"]).resolve()),
        "commit": os.environ["EXPECTED_CODE_COMMIT"],
        "dirty": False,
    },
    "inputs": {
        "config": input_record("CONFIG_PATH", "EXPECTED_CONFIG_SHA256"),
        "data_split": input_record("DATA_SPLIT_PATH", "EXPECTED_DATA_SPLIT_SHA256"),
        "labels": input_record("LABELS_PATH", "EXPECTED_LABELS_SHA256"),
        "mist_checkpoint": input_record(
            "MIST_CHECKPOINT", "EXPECTED_MIST_CHECKPOINT_SHA256"
        ),
        "dlm_checkpoint": input_record(
            "DLM_CHECKPOINT", "EXPECTED_DLM_CHECKPOINT_SHA256"
        ),
        "expected_manifest": input_record(
            "EXPECTED_MANIFEST", "EXPECTED_MANIFEST_SHA256"
        ),
    },
    "selection": {
        "split": "test",
        "start_index": start,
        "max_spectra": size,
    },
    "settings": {
        "softmax_temp": float(os.environ["SOFTMAX_TEMP"]),
        "randomness": 0.1,
        "formula_matches": 10,
        "max_attempts": int(os.environ["MAX_ATTEMPTS"]),
        "batch_size": int(os.environ["BATCH_SIZE"]),
        "inference_dtype": os.environ["INFERENCE_DTYPE"],
        "cache_conditioning": os.environ["CACHE_CONDITIONING"] == "1",
        "fingerprint_source": "mist_binary",
        "fingerprint_sparsify_mode": "threshold",
        "fingerprint_threshold": 0.187,
        "use_shared_cross_attention": True,
        "token_model": None,
        "seed": 42,
    },
    "outputs": {},
}
Path(os.environ["RUN_MANIFEST"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

mark_failed() {
  local status="$?"
  trap - EXIT
  if [[ -f "$RUN_MANIFEST" ]]; then
    EXIT_CODE="$status" "$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["RUN_MANIFEST"])
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["status"] = "failed"
manifest["exit_code"] = int(os.environ["EXIT_CODE"])
manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  exit "$status"
}
trap mark_failed EXIT

CACHE_ARGS=()
if [[ "$CACHE_CONDITIONING" == "1" ]]; then
  CACHE_ARGS+=(--cache-conditioning)
fi

cd "$FRIGID_REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" \
  scripts/benchmark_dlm_fingerprint_robustness.py \
  --config "$CONFIG_PATH" \
  --data-dir "$DATA_DIR" \
  --mist-checkpoint "$MIST_CHECKPOINT" \
  --dlm-checkpoint "$DLM_CHECKPOINT" \
  --output-dir "$OUT_DIR" \
  --split test \
  --batch-size "$BATCH_SIZE" \
  --inference-dtype "$INFERENCE_DTYPE" \
  "${CACHE_ARGS[@]}" \
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

"$PYTHON_BIN" - <<'PY'
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


out_dir = Path(os.environ["OUT_DIR"])
manifest_path = Path(os.environ["RUN_MANIFEST"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_rows = read_rows(Path(os.environ["EXPECTED_MANIFEST"]))
start = int(os.environ["START_INDEX"])
size = int(os.environ["MAX_SPECTRA"])
expected_names = [row["spec_name"] for row in expected_rows[start : start + size]]

complete_files = {
    "predictions_mist_binary.csv": "name",
    "detailed_results.csv": "spec_name",
    "fingerprint_drift.csv": "spec_name",
    "paired_comparison.csv": "spec_name",
}
for filename, column in complete_files.items():
    path = out_dir / filename
    rows = read_rows(path)
    observed = [row[column] for row in rows]
    if observed != expected_names:
        raise ValueError(f"{filename} does not match the declared ordered shard range")

score_rows = read_rows(out_dir / "prediction_scores_mist_binary.csv")
if not {row["spec_name"] for row in score_rows}.issubset(set(expected_names)):
    raise ValueError("prediction_scores_mist_binary.csv contains an unknown query")

run_state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
if not run_state.get("completed") or run_state.get("processed_spectra") != size:
    raise ValueError("Benchmark run_state is not complete for the declared shard size")

outputs = {}
for path in sorted(out_dir.iterdir()):
    if not path.is_file() or path.name == "RUN_MANIFEST.json":
        continue
    record = {"sha256": sha256_file(path)}
    if path.suffix == ".csv":
        record["row_count"] = len(read_rows(path))
    outputs[path.name] = record

manifest["status"] = "completed"
manifest["exit_code"] = 0
manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
manifest["outputs"] = outputs
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

trap - EXIT
