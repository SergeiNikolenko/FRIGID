#!/bin/bash

# This file is sourced by the Slurm launchers after ROOT and CODE are set.
# Keep the training interpreter and TLS trust store inside the isolated MARLIN
# workspace so deleting an unrelated checkout cannot break a running job.

PYTHON="${MARLIN_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "MARLIN Python is not executable: $PYTHON" >&2
    exit 2
fi

CERTIFI_CA="$("$PYTHON" -c 'import certifi; print(certifi.where())')"
if [[ ! -r "$CERTIFI_CA" ]]; then
    echo "certifi CA bundle is not readable: $CERTIFI_CA" >&2
    exit 2
fi

CA_DIR="$ROOT/cache/tls"
CA_BUNDLE="$CA_DIR/cacert.pem"
mkdir -p "$CA_DIR"
install -m 0644 "$CERTIFI_CA" "$CA_BUNDLE"

export REQUESTS_CA_BUNDLE="$CA_BUNDLE"
export SSL_CERT_FILE="$CA_BUNDLE"
export CURL_CA_BUNDLE="$CA_BUNDLE"

"$PYTHON" -c '
import os
from pathlib import Path

for name in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
    path = Path(os.environ[name])
    if not path.is_file():
        raise SystemExit(f"{name} does not point to a readable file: {path}")
'
