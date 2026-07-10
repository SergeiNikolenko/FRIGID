"""Selection helpers for reproducible benchmark subsets."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Sequence


def load_spec_manifest(path: str | Path) -> list[str]:
    """Load an ordered, unique ``spec_name`` column from CSV or TSV."""
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
        spec_names = [str(row["spec_name"]).strip() for row in reader]

    if not spec_names:
        raise ValueError(f"Spectrum manifest is empty: {manifest_path}")
    if any(not name for name in spec_names):
        raise ValueError(
            f"Spectrum manifest contains an empty spec_name: {manifest_path}"
        )

    duplicates = sorted(
        name for name, count in Counter(spec_names).items() if count > 1
    )
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(
            f"Spectrum manifest contains duplicate spec_name values: {preview}"
        )
    return spec_names


def hash_spec_names(spec_names: Sequence[str]) -> str:
    payload = "".join(f"{name}\n" for name in spec_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_selected_indices(
    split_data: Sequence,
    spec_names: Sequence[str] | None,
    start_index: int,
    max_spectra: int | None,
) -> list[int]:
    """Resolve either an ordered manifest or a contiguous benchmark slice."""
    if not split_data:
        raise ValueError("Selected split is empty.")
    if start_index < 0 or start_index >= len(split_data):
        raise ValueError(f"start_index must be in [0, {len(split_data) - 1}]")
    if max_spectra is not None and max_spectra <= 0:
        raise ValueError("max_spectra must be positive when provided.")

    if spec_names is None:
        end_index = len(split_data)
        if max_spectra is not None:
            end_index = min(end_index, start_index + max_spectra)
        return list(range(start_index, end_index))

    if start_index != 0:
        raise ValueError("start_index cannot be combined with spec_manifest.")
    if max_spectra is not None and max_spectra != len(spec_names):
        raise ValueError(
            "max_spectra cannot truncate spec_manifest; omit it or set it to the "
            f"manifest size ({len(spec_names)})."
        )

    name_to_index: dict[str, int] = {}
    for index, (spectrum, _molecule) in enumerate(split_data):
        name = str(spectrum.get_spec_name())
        if name in name_to_index:
            raise ValueError(f"Selected split contains duplicate spec_name: {name}")
        name_to_index[name] = index

    missing = [name for name in spec_names if name not in name_to_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Spectrum manifest contains names absent from the split: {preview}"
        )
    return [name_to_index[name] for name in spec_names]
