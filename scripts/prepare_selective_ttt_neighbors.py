#!/usr/bin/env python
"""Prepare audit-safe train-only neighbor bundles for selective DLM TTT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FINGERPRINT_ARRAY = "mist_binary"
FINGERPRINT_BITS = 4096
TRAIN_FIELDS = ("spec_name", "split", "smiles", "inchi_key_first_block")
QUERY_FIELDS = ("spec_name",)
EXCLUSION_FIELDS = ("spec_name", "connectivity_exclusion_token")
CONNECTIVITY_TOKEN_DOMAIN = "frigid-spa139-connectivity-v1"
QUERY_STATE_TOKEN_DOMAIN = "frigid-spa139-query-state-v1"
INCHIKEY_FIRST_BLOCK_RE = re.compile(r"[A-Z]{14}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
POPCOUNT_TABLE = np.asarray(
    [value.bit_count() for value in range(256)],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class TrainRecord:
    row_index: int
    spec_name: str
    smiles: str
    inchi_key_first_block: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_inchi_key_first_block(value: str) -> str:
    normalized = str(value).strip().upper()
    if INCHIKEY_FIRST_BLOCK_RE.fullmatch(normalized) is None:
        raise ValueError(
            "InChIKey connectivity blocks must contain exactly 14 uppercase letters."
        )
    return normalized


def connectivity_exclusion_token(inchi_key_first_block: str) -> str:
    normalized = normalize_inchi_key_first_block(inchi_key_first_block)
    payload = f"{CONNECTIVITY_TOKEN_DOMAIN}\0{normalized}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load_csv_rows(
    path: Path,
    expected_fields: tuple[str, ...],
    purpose: str,
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != expected_fields:
            raise ValueError(
                f"{purpose} columns must be exactly {list(expected_fields)}; "
                f"got {list(actual_fields)}."
            )
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"{purpose} has undeclared values on line {line_number}."
                )
            if any(row.get(field) is None for field in expected_fields):
                raise ValueError(f"{purpose} has missing values on line {line_number}.")
            rows.append(
                {field: str(row[field]).strip() for field in expected_fields}
            )
    if not rows:
        raise ValueError(f"{purpose} is empty: {path}")
    return rows


def load_train_library(path: Path) -> list[TrainRecord]:
    rows = _load_csv_rows(path, TRAIN_FIELDS, "Train library")
    records: list[TrainRecord] = []
    seen_specs: set[str] = set()
    for row_index, row in enumerate(rows):
        spec_name = row["spec_name"]
        if not spec_name:
            raise ValueError(f"Train library has an empty spec_name at row {row_index}.")
        if spec_name in seen_specs:
            raise ValueError(f"Train library has duplicate spec_name: {spec_name}")
        seen_specs.add(spec_name)
        if row["split"].lower() != "train":
            raise ValueError(
                "Train library must contain only split=train rows; "
                f"got {row['split']!r} for {spec_name}."
            )
        if not row["smiles"]:
            raise ValueError(f"Train library has empty smiles for {spec_name}.")
        records.append(
            TrainRecord(
                row_index=row_index,
                spec_name=spec_name,
                smiles=row["smiles"],
                inchi_key_first_block=normalize_inchi_key_first_block(
                    row["inchi_key_first_block"]
                ),
            )
        )
    return records


def load_query_specs(path: Path) -> list[str]:
    rows = _load_csv_rows(path, QUERY_FIELDS, "Query index")
    spec_names = [row["spec_name"] for row in rows]
    if any(not spec_name for spec_name in spec_names):
        raise ValueError("Query index contains an empty spec_name.")
    if len(spec_names) != len(set(spec_names)):
        raise ValueError("Query index contains duplicate spec_name values.")
    return spec_names


def load_exclusion_tokens(path: Path, query_specs: list[str]) -> dict[str, str]:
    rows = _load_csv_rows(path, EXCLUSION_FIELDS, "Query exclusion token file")
    tokens: dict[str, str] = {}
    for row in rows:
        spec_name = row["spec_name"]
        token = row["connectivity_exclusion_token"].lower()
        if not spec_name:
            raise ValueError("Query exclusion token file contains an empty spec_name.")
        if spec_name in tokens:
            raise ValueError(f"Duplicate exclusion token row for {spec_name}.")
        if SHA256_RE.fullmatch(token) is None:
            raise ValueError(f"Invalid connectivity exclusion token for {spec_name}.")
        tokens[spec_name] = token

    expected = set(query_specs)
    actual = set(tokens)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "Query exclusion tokens must match the query index exactly: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    return tokens


def load_mist_binary(
    path: Path,
    expected_rows: int,
    purpose: str,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        arrays = set(archive.files)
        if arrays != {FINGERPRINT_ARRAY}:
            raise ValueError(
                f"{purpose} NPZ arrays must be exactly ['{FINGERPRINT_ARRAY}']; "
                f"got {sorted(arrays)}."
            )
        fingerprints = np.asarray(archive[FINGERPRINT_ARRAY])

    expected_shape = (expected_rows, FINGERPRINT_BITS)
    if fingerprints.shape != expected_shape:
        raise ValueError(
            f"{purpose} {FINGERPRINT_ARRAY} shape must be {expected_shape}; "
            f"got {fingerprints.shape}."
        )
    if not (
        np.issubdtype(fingerprints.dtype, np.number)
        or np.issubdtype(fingerprints.dtype, np.bool_)
    ):
        raise ValueError(f"{purpose} fingerprints must be numeric or boolean.")
    if not np.isfinite(fingerprints).all():
        raise ValueError(f"{purpose} fingerprints contain non-finite values.")
    if not np.logical_or(fingerprints == 0, fingerprints == 1).all():
        raise ValueError(f"{purpose} {FINGERPRINT_ARRAY} must be binary.")

    fingerprints = np.ascontiguousarray(fingerprints, dtype=np.uint8)
    zero_rows = np.flatnonzero(fingerprints.sum(axis=1) == 0)
    if zero_rows.size:
        raise ValueError(
            f"{purpose} contains zero-active-bit fingerprints at rows "
            f"{zero_rows[:5].tolist()}."
        )
    return fingerprints


def _binary_tanimoto_scores(
    train_packed: np.ndarray,
    train_active_bits: np.ndarray,
    query_packed: np.ndarray,
    query_active_bits: int,
    block_size: int = 8192,
) -> np.ndarray:
    scores = np.empty(train_packed.shape[0], dtype=np.float64)
    for start in range(0, train_packed.shape[0], block_size):
        stop = min(start + block_size, train_packed.shape[0])
        intersection = POPCOUNT_TABLE[
            np.bitwise_and(train_packed[start:stop], query_packed)
        ].sum(axis=1, dtype=np.uint16)
        union = (
            train_active_bits[start:stop].astype(np.int32)
            + query_active_bits
            - intersection.astype(np.int32)
        )
        scores[start:stop] = intersection / union
    return scores


def rank_train_neighbors(
    train_records: list[TrainRecord],
    train_fingerprints: np.ndarray,
    query_specs: list[str],
    query_fingerprints: np.ndarray,
    exclusion_tokens: dict[str, str],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    train_names = np.asarray([record.spec_name for record in train_records], dtype=str)
    train_tokens = np.asarray(
        [
            connectivity_exclusion_token(record.inchi_key_first_block)
            for record in train_records
        ],
        dtype=str,
    )
    train_active_bits = train_fingerprints.sum(axis=1, dtype=np.uint16)
    train_packed = np.packbits(train_fingerprints, axis=1, bitorder="little")
    query_packed = np.packbits(query_fingerprints, axis=1, bitorder="little")
    query_active_bits = query_fingerprints.sum(axis=1, dtype=np.uint16)

    neighbor_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for query_index, query_spec_name in enumerate(query_specs):
        exclusion_token = exclusion_tokens[query_spec_name]
        eligible = np.logical_and(
            train_tokens != exclusion_token,
            train_names != query_spec_name,
        )
        eligible_count = int(eligible.sum())
        if eligible_count < top_k:
            raise ValueError(
                f"Only {eligible_count} eligible train rows remain for "
                f"{query_spec_name}; top_k={top_k}."
            )

        scores = _binary_tanimoto_scores(
            train_packed=train_packed,
            train_active_bits=train_active_bits,
            query_packed=query_packed[query_index],
            query_active_bits=int(query_active_bits[query_index]),
        )
        scores[~eligible] = -1.0
        order = np.lexsort((train_names, -scores))[:top_k]

        selected_names: list[str] = []
        first_neighbor_row = len(neighbor_rows)
        for rank, train_index in enumerate(order, start=1):
            record = train_records[int(train_index)]
            selected_names.append(record.spec_name)
            neighbor_rows.append(
                {
                    "query_spec_name": query_spec_name,
                    "rank": rank,
                    "train_spec_name": record.spec_name,
                    "train_fingerprint_row": record.row_index,
                    "train_smiles": record.smiles,
                    "train_inchi_key_first_block": record.inchi_key_first_block,
                    "tanimoto_to_query_mist": float(scores[train_index]),
                    "query_mist_active_bits": int(query_active_bits[query_index]),
                    "train_mist_active_bits": int(train_active_bits[train_index]),
                }
            )
        query_rows.append(
            {
                "schema_version": 1,
                "query_spec_name": query_spec_name,
                "query_fingerprint_row": query_index,
                "neighbor_csv_first_data_row": first_neighbor_row,
                "neighbor_count": top_k,
                "neighbor_train_spec_names": selected_names,
                "target_fields_available": False,
            }
        )
    return neighbor_rows, query_rows


def _input_artifact(path: Path, **metadata: Any) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        **metadata,
    }


def prepare_selective_ttt_neighbors(
    train_library_csv: Path,
    train_fingerprints_npz: Path,
    query_index_csv: Path,
    query_fingerprints_npz: Path,
    query_exclusion_tokens_csv: Path,
    mist_checkpoint: Path,
    base_dlm_checkpoint: Path,
    output_dir: Path,
    top_k: int = 64,
) -> dict[str, Any]:
    input_paths = (
        train_library_csv,
        train_fingerprints_npz,
        query_index_csv,
        query_fingerprints_npz,
        query_exclusion_tokens_csv,
        mist_checkpoint,
        base_dlm_checkpoint,
    )
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise ValueError(f"Input files do not exist: {missing}")
    if train_fingerprints_npz.resolve() == query_fingerprints_npz.resolve():
        raise ValueError("Train and query fingerprints must be separate artifacts.")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ValueError(f"Output directory must be empty: {output_dir}")

    train_records = load_train_library(train_library_csv)
    query_specs = load_query_specs(query_index_csv)
    train_spec_names = {record.spec_name for record in train_records}
    query_train_overlap = sorted(train_spec_names.intersection(query_specs))
    if query_train_overlap:
        raise ValueError(
            "Query spec_name values overlap the train library: "
            f"{query_train_overlap[:5]}."
        )
    exclusion_tokens = load_exclusion_tokens(
        query_exclusion_tokens_csv,
        query_specs,
    )
    train_fingerprints = load_mist_binary(
        train_fingerprints_npz,
        expected_rows=len(train_records),
        purpose="Train",
    )
    query_fingerprints = load_mist_binary(
        query_fingerprints_npz,
        expected_rows=len(query_specs),
        purpose="Query",
    )
    neighbor_rows, query_rows = rank_train_neighbors(
        train_records=train_records,
        train_fingerprints=train_fingerprints,
        query_specs=query_specs,
        query_fingerprints=query_fingerprints,
        exclusion_tokens=exclusion_tokens,
        top_k=top_k,
    )

    mist_checkpoint_sha256 = sha256_file(mist_checkpoint)
    base_dlm_checkpoint_sha256 = sha256_file(base_dlm_checkpoint)
    state_isolation_contract = {
        "scope": "query_local",
        "initialize_from_base_checkpoint_for_every_query": True,
        "cross_query_learned_state_reuse_allowed": False,
        "base_dlm_checkpoint_sha256": base_dlm_checkpoint_sha256,
        "adapter_execution_implemented": False,
    }
    for query_row in query_rows:
        state_payload = (
            f"{QUERY_STATE_TOKEN_DOMAIN}\0{base_dlm_checkpoint_sha256}\0"
            f"{query_row['query_spec_name']}"
        ).encode("utf-8")
        query_row["state_isolation_contract"] = {
            **state_isolation_contract,
            "planned_query_state_id": hashlib.sha256(state_payload).hexdigest(),
            "prior_query_state_inputs": [],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    neighbors_csv = output_dir / "neighbors.csv"
    query_bundles_jsonl = output_dir / "query_bundles.jsonl"
    run_manifest_json = output_dir / "run_manifest.json"

    neighbor_fields = (
        "query_spec_name",
        "rank",
        "train_spec_name",
        "train_fingerprint_row",
        "train_smiles",
        "train_inchi_key_first_block",
        "tanimoto_to_query_mist",
        "query_mist_active_bits",
        "train_mist_active_bits",
    )
    with neighbors_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=neighbor_fields)
        writer.writeheader()
        writer.writerows(neighbor_rows)

    with query_bundles_jsonl.open("w", encoding="utf-8") as handle:
        for query_row in query_rows:
            handle.write(json.dumps(query_row, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 1,
        "purpose": "selective_ttt_train_neighbor_preparation",
        "script": _input_artifact(Path(__file__).resolve()),
        "inputs": {
            "train_library_csv": _input_artifact(
                train_library_csv,
                row_count=len(train_records),
                required_split="train",
                allowed_columns=list(TRAIN_FIELDS),
            ),
            "train_fingerprints_npz": _input_artifact(
                train_fingerprints_npz,
                row_count=len(train_records),
                allowed_arrays=[FINGERPRINT_ARRAY],
            ),
            "query_index_csv": _input_artifact(
                query_index_csv,
                row_count=len(query_specs),
                allowed_columns=list(QUERY_FIELDS),
            ),
            "query_fingerprints_npz": _input_artifact(
                query_fingerprints_npz,
                row_count=len(query_specs),
                allowed_arrays=[FINGERPRINT_ARRAY],
            ),
            "query_exclusion_tokens_csv": _input_artifact(
                query_exclusion_tokens_csv,
                row_count=len(query_specs),
                allowed_columns=list(EXCLUSION_FIELDS),
                token_domain=CONNECTIVITY_TOKEN_DOMAIN,
            ),
            "mist_checkpoint": _input_artifact(
                mist_checkpoint,
                sha256=mist_checkpoint_sha256,
            ),
            "base_dlm_checkpoint": _input_artifact(
                base_dlm_checkpoint,
                sha256=base_dlm_checkpoint_sha256,
            ),
        },
        "parameters": {
            "top_k": top_k,
            "fingerprint_array": FINGERPRINT_ARRAY,
            "fingerprint_bits": FINGERPRINT_BITS,
            "similarity": "binary_tanimoto",
        },
        "safeguards": {
            "target_fields_used_for_ranking": [],
            "target_derived_exclusion_token_used_only_for_filtering": True,
            "query_metadata_allowlist": list(QUERY_FIELDS),
            "query_npz_allowlist": [FINGERPRINT_ARRAY],
            "train_split_required": "train",
            "raw_query_inchi_key_allowed": False,
            "raw_query_smiles_allowed": False,
            "raw_query_formula_allowed": False,
            "connectivity_exclusion_mode": "domain_separated_sha256_token",
            "fingerprint_checkpoint_binding": "declared_not_verified",
        },
        "state_isolation_contract": state_isolation_contract,
        "counts": {
            "train_rows": len(train_records),
            "query_rows": len(query_specs),
            "neighbor_rows": len(neighbor_rows),
        },
        "outputs": {
            "neighbors_csv": _input_artifact(neighbors_csv),
            "query_bundles_jsonl": _input_artifact(query_bundles_jsonl),
            "run_manifest_json": str(run_manifest_json),
        },
    }
    run_manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-library-csv", required=True)
    parser.add_argument("--train-fingerprints-npz", required=True)
    parser.add_argument("--query-index-csv", required=True)
    parser.add_argument("--query-fingerprints-npz", required=True)
    parser.add_argument(
        "--query-exclusion-tokens-csv",
        required=True,
        help=(
            "Sanitized spec_name/token mapping produced by a trusted split audit; "
            "raw query InChIKeys are not accepted."
        ),
    )
    parser.add_argument("--mist-checkpoint", required=True)
    parser.add_argument("--base-dlm-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_selective_ttt_neighbors(
        train_library_csv=Path(args.train_library_csv).expanduser().resolve(),
        train_fingerprints_npz=Path(args.train_fingerprints_npz).expanduser().resolve(),
        query_index_csv=Path(args.query_index_csv).expanduser().resolve(),
        query_fingerprints_npz=Path(args.query_fingerprints_npz).expanduser().resolve(),
        query_exclusion_tokens_csv=Path(
            args.query_exclusion_tokens_csv
        ).expanduser().resolve(),
        mist_checkpoint=Path(args.mist_checkpoint).expanduser().resolve(),
        base_dlm_checkpoint=Path(args.base_dlm_checkpoint).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        top_k=args.top_k,
    )
    print(
        "Prepared "
        f"{manifest['counts']['neighbor_rows']} train-only neighbors for "
        f"{manifest['counts']['query_rows']} queries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
