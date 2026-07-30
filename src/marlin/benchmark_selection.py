"""Helpers for reproducible, ordered MARLIN benchmark panels."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd


def load_spec_manifest(path: str | Path) -> list[str]:
    """Load a non-empty, ordered and unique ``spec_name`` column."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Spectrum manifest not found: {manifest_path}")
    delimiter = "\t" if manifest_path.suffix.lower() in {".tsv", ".tab"} else ","
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None or "spec_name" not in reader.fieldnames:
            raise ValueError(
                f"Spectrum manifest must contain a 'spec_name' column: {manifest_path}"
            )
        names = [str(row["spec_name"]).strip() for row in reader]
    if not names:
        raise ValueError(f"Spectrum manifest is empty: {manifest_path}")
    if any(not name for name in names):
        raise ValueError(f"Spectrum manifest contains an empty spec_name: {manifest_path}")
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(
            "Spectrum manifest contains duplicate spec_name values: "
            + ", ".join(duplicates[:5])
        )
    return names


def hash_spec_names(names: Sequence[str]) -> str:
    payload = "".join(f"{name}\n" for name in names).encode()
    return hashlib.sha256(payload).hexdigest()


def select_metadata(
    metadata: pd.DataFrame,
    manifest_names: Sequence[str] | None,
    max_spectra: int | None,
) -> pd.DataFrame:
    """Select an ordered manifest or a leading technical-smoke subset."""
    if "spec_name" not in metadata:
        raise ValueError("Metadata must contain a spec_name column")
    names = metadata["spec_name"].astype(str)
    if names.duplicated().any():
        duplicate = names[names.duplicated()].iloc[0]
        raise ValueError(f"Metadata contains duplicate spec_name: {duplicate}")
    if max_spectra is not None and max_spectra <= 0:
        raise ValueError("--max-spectra must be positive")
    if manifest_names is None:
        return metadata.iloc[:max_spectra].copy() if max_spectra else metadata.copy()
    if max_spectra is not None and max_spectra != len(manifest_names):
        raise ValueError(
            "--max-spectra cannot truncate --spec-manifest; omit it or set it to "
            f"{len(manifest_names)}"
        )
    indexed = metadata.assign(spec_name=names).set_index("spec_name", drop=False)
    missing = [name for name in manifest_names if name not in indexed.index]
    if missing:
        raise ValueError(
            "Spectrum manifest contains names absent from metadata: "
            + ", ".join(missing[:5])
        )
    return indexed.loc[list(manifest_names)].reset_index(drop=True)
