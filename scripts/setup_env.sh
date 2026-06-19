#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_env.sh [--dev]

Compatibility wrapper around `uv sync`. The default project environment installs
FRIGID plus the dependencies needed for NGBoost and ICEBERG inference.

Examples:
  bash scripts/setup_env.sh
  bash scripts/setup_env.sh --dev
EOF
}

INSTALL_DEV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      INSTALL_DEV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun this script." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  uv venv --python 3.10 .venv
fi

git submodule update --init --recursive

if [[ "$INSTALL_DEV" == "1" ]]; then
  uv sync --group dev
else
  uv sync
fi

cat <<EOF
FRIGID environment is ready.

Activate:
  source .venv/bin/activate

Check:
  frigid --help
EOF
