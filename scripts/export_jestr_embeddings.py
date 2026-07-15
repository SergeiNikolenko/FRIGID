#!/usr/bin/env python
"""Export frozen JESTR spectrum embeddings for row-locked FRIGID spectra.

The exporter verifies a pinned checkout of the official JESTR repository, then
loads only the released spectral MLP weights.  Reimplementing this small MLP in
isolation avoids importing JESTR's molecule-only DGL dependencies while keeping
the checkpoint architecture and preprocessing contract explicit and testable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OFFICIAL_REPOSITORY_URL = "https://github.com/HassounLab/JESTR1"
OFFICIAL_PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11601792/"
OFFICIAL_SOURCE_REVISION = "a5619c18a85a49171d60ead079684ae667cc0dd0"
OFFICIAL_CHECKPOINT_RELATIVE_PATH = (
    "data/MassSpecGym/pretrained_spec_enc_model_1741546103623_best.pt"
)
OFFICIAL_CHECKPOINT_SHA256 = (
    "b9f2ccc25ae7710d17d30bfa7cc5ca6e3065962fd0bf24d57af9527d804972fa"
)
OFFICIAL_MODEL_SOURCE_SHA256 = (
    "17d99b1599ee6383c76e30c0d9064e2e159ea7a8796d6ada1620ffff640d5af2"
)
OFFICIAL_UTILS_SOURCE_SHA256 = (
    "9aaceea8142b82f313e02a97f8ba1c53fe17072927a82b5b6ed2efd677eb3d2b"
)
OFFICIAL_PARAMS_SOURCE_SHA256 = (
    "9e767fb7612eb44f6d3a3e0a6b883e555d68e4358c38d14f8c45672e287684b8"
)

BIN_COUNT = 1000
BIN_RESOLUTION = 1.0
MAX_MZ = 1000.0
MAX_NORMALIZED_INTENSITY = 999.0
EMBEDDING_DIMENSION = 512
HIDDEN_DIMENSION = 1024
FINGERPRINT_BITS = 4096
FINGERPRINT_RADIUS = 2


@dataclass(frozen=True)
class PreparedSpectrum:
    """One spectrum encoded according to the released JESTR input contract."""

    binned: np.ndarray
    ionization: str
    source_peak_count: int
    retained_peak_count: int
    removed_above_max_mz_count: int
    source_max_intensity: float


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA-256 without loading the whole file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ms_file(path: str | Path) -> tuple[str, list[tuple[float, float]]]:
    """Read ionization and ``(m/z, intensity)`` peaks from one FRIGID file."""

    path = Path(path)
    ionization: str | None = None
    peaks: list[tuple[float, float]] = []
    reading_peaks = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith(">ionization "):
            ionization = line.split(maxsplit=1)[1].strip()
        elif lower == ">ms2peaks":
            reading_peaks = True
        elif reading_peaks and line.startswith(">"):
            break
        elif reading_peaks and line and not line.startswith("#"):
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed peak row in {path}: {line!r}")
            mz, intensity = float(fields[0]), float(fields[1])
            if not np.isfinite(mz) or not np.isfinite(intensity):
                raise ValueError(f"Non-finite peak row in {path}: {line!r}")
            if mz <= 0.0:
                raise ValueError(f"Non-positive m/z in {path}: {line!r}")
            if intensity < 0.0:
                raise ValueError(f"Negative peak intensity in {path}: {line!r}")
            peaks.append((mz, intensity))

    if ionization is None:
        raise ValueError(f"Missing >ionization value in {path}")
    if not peaks:
        raise ValueError(f"No MS2 peaks found in {path}")
    return ionization, peaks


def prepare_spectrum(path: str | Path) -> PreparedSpectrum:
    """Convert one FRIGID spectrum to JESTR's released 1000-bin input.

    JESTR stores each peak internally as ``[intensity, m/z]``.  Its published
    preprocessing scales each spectrum's maximum peak intensity to 999; the
    released ``get_ms_array`` then keeps ``m/z < 1001``, assigns bins with
    ``int((m/z - 1) / 1)``, sums collisions, and applies
    ``log10(binned_intensity + 1) / 3``.  This function deliberately preserves
    that boundary and indexing behavior.
    """

    ionization, peaks = _parse_ms_file(path)
    source_max_intensity = max(intensity for _, intensity in peaks)
    if source_max_intensity <= 0.0:
        raise ValueError(f"Spectrum has no positive peak intensity: {path}")
    intensity_scale = MAX_NORMALIZED_INTENSITY / source_max_intensity

    binned = np.zeros(BIN_COUNT, dtype=np.float32)
    retained_peak_count = 0
    for mz, intensity in peaks:
        if mz >= MAX_MZ + BIN_RESOLUTION:
            continue
        bin_index = int((mz - BIN_RESOLUTION) / BIN_RESOLUTION)
        if not 0 <= bin_index < BIN_COUNT:
            raise ValueError(
                f"JESTR bin index {bin_index} is outside [0, {BIN_COUNT}) "
                f"for m/z {mz} in {path}"
            )
        binned[bin_index] += intensity * intensity_scale
        retained_peak_count += 1

    transformed = np.log10(binned + np.float32(1.0)) / np.float32(3.0)
    return PreparedSpectrum(
        binned=transformed.astype(np.float32, copy=False),
        ionization=ionization,
        source_peak_count=len(peaks),
        retained_peak_count=retained_peak_count,
        removed_above_max_mz_count=len(peaks) - retained_peak_count,
        source_max_intensity=source_max_intensity,
    )


def _resolve_inchikey_column(metadata: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in metadata:
            raise ValueError(f"Metadata is missing InChIKey column {requested!r}")
        return requested
    for candidate in ("inchikey", "inchi_key", "inchi_key_first_block"):
        if candidate in metadata:
            return candidate
    raise ValueError(
        "Metadata must contain an InChIKey column; tried 'inchikey', "
        "'inchi_key', and 'inchi_key_first_block'"
    )


def load_ordered_metadata(
    path: str | Path,
    *,
    id_column: str,
    index_column: str,
    inchikey_column: str | None,
) -> tuple[pd.DataFrame, str]:
    """Load metadata and enforce one deterministic spectrum row order."""

    path = Path(path)
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    metadata = pd.read_csv(path, sep=delimiter)
    if metadata.empty:
        raise ValueError(f"Metadata is empty: {path}")
    if id_column not in metadata:
        raise ValueError(f"Metadata {path} is missing ID column {id_column!r}")

    spectrum_ids = metadata[id_column].astype(str).str.strip()
    if (
        spectrum_ids.eq("").any()
        or spectrum_ids.str.lower().eq("nan").any()
        or spectrum_ids.duplicated().any()
    ):
        raise ValueError(f"Metadata column {id_column!r} must contain unique non-empty IDs")
    metadata = metadata.assign(**{id_column: spectrum_ids})

    resolved_inchikey_column = _resolve_inchikey_column(metadata, inchikey_column)
    inchikeys = metadata[resolved_inchikey_column].astype(str).str.strip()
    if inchikeys.eq("").any() or inchikeys.str.lower().eq("nan").any():
        raise ValueError(
            f"Metadata column {resolved_inchikey_column!r} must contain non-empty InChIKeys"
        )
    metadata = metadata.assign(**{resolved_inchikey_column: inchikeys})

    if index_column in metadata:
        numeric_index = pd.to_numeric(metadata[index_column], errors="raise").to_numpy()
        if not np.equal(numeric_index, numeric_index.astype(np.int64)).all():
            raise ValueError(f"Metadata column {index_column!r} must contain integer indexes")
        metadata = metadata.assign(
            **{index_column: numeric_index.astype(np.int64)}
        ).sort_values(index_column, kind="stable")
        expected_index = np.arange(len(metadata), dtype=np.int64)
        if not np.array_equal(metadata[index_column].to_numpy(), expected_index):
            raise ValueError(
                f"Metadata column {index_column!r} must contain every index from "
                f"0 to {len(metadata) - 1} exactly once"
            )
    return metadata.reset_index(drop=True), resolved_inchikey_column


def compute_morgan_targets(
    metadata: pd.DataFrame,
    *,
    smiles_column: str,
    id_column: str,
) -> np.ndarray:
    """Compute non-chiral Morgan radius-2/4096 targets in metadata row order."""

    if smiles_column not in metadata:
        raise ValueError(
            f"Metadata is missing SMILES column {smiles_column!r} required for targets"
        )

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.warning")
    targets = np.empty((len(metadata), FINGERPRINT_BITS), dtype=np.uint8)
    target_by_smiles: dict[str, np.ndarray] = {}
    for index, row in metadata.iterrows():
        smiles = str(row[smiles_column]).strip()
        if smiles in target_by_smiles:
            targets[index] = target_by_smiles[smiles]
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(
                f"Could not parse SMILES for {row[id_column]!r}: {smiles!r}"
            )
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            FINGERPRINT_RADIUS,
            nBits=FINGERPRINT_BITS,
            useChirality=False,
        )
        target = np.zeros(FINGERPRINT_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, target)
        targets[index] = target
        target_by_smiles[smiles] = target
    return targets


def _load_targets(path: Path, key: str, expected_rows: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        if key not in arrays:
            raise ValueError(
                f"Target key {key!r} is missing from {path}; available keys: {arrays.files}"
            )
        targets = np.asarray(arrays[key])
    if targets.shape != (expected_rows, FINGERPRINT_BITS):
        raise ValueError(
            f"Targets have shape {targets.shape}; expected "
            f"({expected_rows}, {FINGERPRINT_BITS})"
        )
    if not np.logical_or(np.isclose(targets, 0), np.isclose(targets, 1)).all():
        raise ValueError("Targets must be binary")
    return targets.astype(np.uint8, copy=False)


def _git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _try_git_revision(repository: Path) -> str | None:
    try:
        return _git_revision(repository)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _build_spectrum_encoder(torch: Any) -> Any:
    """Build the exact released ``SpecEncMLP_BIN`` spectral architecture."""

    class JestrSpectrumEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = torch.nn.Dropout(0.4)
            self.mz_fc1 = torch.nn.Linear(BIN_COUNT, HIDDEN_DIMENSION)
            self.mz_fc2 = torch.nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION)
            self.mz_fc3 = torch.nn.Linear(HIDDEN_DIMENSION, EMBEDDING_DIMENSION)
            self.relu = torch.nn.ReLU()

        def forward(self, binned_spectra: Any) -> Any:
            hidden = self.dropout(self.relu(self.mz_fc1(binned_spectra)))
            hidden = self.dropout(self.relu(self.mz_fc2(hidden)))
            return self.dropout(self.mz_fc3(hidden))

    return JestrSpectrumEncoder()


def _load_official_model(
    repository: Path,
    checkpoint: Path,
    *,
    expected_revision: str,
    expected_checkpoint_sha256: str,
    expected_model_source_sha256: str,
    expected_utils_source_sha256: str,
    expected_params_source_sha256: str,
    device: str,
) -> tuple[Any, Any, str, str, str]:
    import torch

    revision = _git_revision(repository)
    if revision != expected_revision:
        raise ValueError(
            f"JESTR source revision is {revision}; expected {expected_revision}"
        )

    source_expectations = {
        repository / "models.py": expected_model_source_sha256,
        repository / "utils.py": expected_utils_source_sha256,
        repository / "params.yaml": expected_params_source_sha256,
    }
    for source_path, expected_sha256 in source_expectations.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"Official JESTR source is missing: {source_path}")
        source_sha256 = sha256_file(source_path)
        if source_sha256 != expected_sha256:
            raise ValueError(
                f"JESTR source SHA-256 for {source_path.name} is {source_sha256}; "
                f"expected {expected_sha256}"
            )

    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"JESTR checkpoint SHA-256 is {checkpoint_sha256}; "
            f"expected {expected_checkpoint_sha256}"
        )

    resolved_device = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {resolved_device}")

    model = _build_spectrum_encoder(torch)
    try:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(checkpoint, map_location="cpu")
    state_dict = loaded.get("model_state_dict", loaded)
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(resolved_device)
    return model, torch, resolved_device, revision, checkpoint_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official frozen JESTR embeddings for row-locked FRIGID spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--spectra-dir", required=True)
    parser.add_argument("--jestr-repo", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-metadata")
    parser.add_argument("--manifest")
    parser.add_argument("--id-column", default="spec_name")
    parser.add_argument("--index-column", default="fingerprint_index")
    parser.add_argument("--inchikey-column")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--exporter-revision",
        help="FRIGID commit containing this exporter; auto-detected when possible",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--targets-npz")
    target_group.add_argument("--compute-morgan-targets", action="store_true")
    parser.add_argument("--targets-key", default="ground_truth")
    parser.add_argument("--expected-source-revision", default=OFFICIAL_SOURCE_REVISION)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=OFFICIAL_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--expected-model-source-sha256", default=OFFICIAL_MODEL_SOURCE_SHA256
    )
    parser.add_argument(
        "--expected-utils-source-sha256", default=OFFICIAL_UTILS_SOURCE_SHA256
    )
    parser.add_argument(
        "--expected-params-source-sha256", default=OFFICIAL_PARAMS_SOURCE_SHA256
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    metadata_path = Path(args.metadata).resolve()
    spectra_dir = Path(args.spectra_dir).resolve()
    repository = Path(args.jestr_repo).resolve()
    checkpoint = Path(
        args.checkpoint or repository / OFFICIAL_CHECKPOINT_RELATIVE_PATH
    ).resolve()
    output_path = Path(args.output).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    output_metadata_path = Path(
        args.output_metadata or output_path.with_suffix(".metadata.csv")
    ).resolve()
    manifest_path = Path(args.manifest or output_path.with_suffix(".manifest.json")).resolve()
    for path in (output_path, output_metadata_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    metadata, inchikey_column = load_ordered_metadata(
        metadata_path,
        id_column=args.id_column,
        index_column=args.index_column,
        inchikey_column=args.inchikey_column,
    )
    spectrum_ids = metadata[args.id_column].to_numpy(dtype=str)
    inchikeys = metadata[inchikey_column].to_numpy(dtype=str)

    model, torch, device, revision, checkpoint_sha256 = _load_official_model(
        repository,
        checkpoint,
        expected_revision=args.expected_source_revision,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_model_source_sha256=args.expected_model_source_sha256,
        expected_utils_source_sha256=args.expected_utils_source_sha256,
        expected_params_source_sha256=args.expected_params_source_sha256,
        device=args.device,
    )

    target_start = time.perf_counter()
    targets_path: Path | None = None
    targets_sha256: str | None = None
    if args.targets_npz:
        targets_path = Path(args.targets_npz).resolve()
        targets_sha256 = sha256_file(targets_path)
        targets = _load_targets(targets_path, args.targets_key, len(metadata))
        targets_source = f"{targets_path}:{args.targets_key}"
    else:
        targets = compute_morgan_targets(
            metadata,
            smiles_column=args.smiles_column,
            id_column=args.id_column,
        )
        targets_source = (
            f"{metadata_path}:{args.smiles_column}; Morgan radius={FINGERPRINT_RADIUS}, "
            f"bits={FINGERPRINT_BITS}, useChirality=false"
        )
    target_seconds = time.perf_counter() - target_start

    embeddings = np.empty((len(metadata), EMBEDDING_DIMENSION), dtype=np.float32)
    inference_seconds = np.empty(len(metadata), dtype=np.float64)
    source_peak_counts = np.empty(len(metadata), dtype=np.int32)
    retained_peak_counts = np.empty(len(metadata), dtype=np.int32)
    removed_peak_counts = np.empty(len(metadata), dtype=np.int32)
    source_max_intensities = np.empty(len(metadata), dtype=np.float32)
    ionizations: list[str] = []
    aggregate_inference_seconds = 0.0
    preprocessing_seconds = 0.0
    export_start = time.perf_counter()

    for start in range(0, len(metadata), args.batch_size):
        stop = min(start + args.batch_size, len(metadata))
        preprocessing_start = time.perf_counter()
        prepared = [
            prepare_spectrum(spectra_dir / f"{spectrum_id}.ms")
            for spectrum_id in spectrum_ids[start:stop]
        ]
        batch = torch.as_tensor(
            np.stack([item.binned for item in prepared]),
            dtype=torch.float32,
            device=device,
        )
        preprocessing_seconds += time.perf_counter() - preprocessing_start

        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        with torch.inference_mode():
            batch_embeddings = model(batch)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        batch_inference_seconds = time.perf_counter() - inference_start
        aggregate_inference_seconds += batch_inference_seconds
        inference_seconds[start:stop] = batch_inference_seconds / (stop - start)

        batch_embeddings = batch_embeddings.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        if batch_embeddings.shape != (stop - start, EMBEDDING_DIMENSION):
            raise ValueError(
                f"JESTR produced shape {batch_embeddings.shape}; expected "
                f"({stop - start}, {EMBEDDING_DIMENSION})"
            )
        if not np.isfinite(batch_embeddings).all():
            raise ValueError(f"JESTR produced non-finite embeddings for rows {start}:{stop}")
        embeddings[start:stop] = batch_embeddings
        source_peak_counts[start:stop] = [item.source_peak_count for item in prepared]
        retained_peak_counts[start:stop] = [item.retained_peak_count for item in prepared]
        removed_peak_counts[start:stop] = [
            item.removed_above_max_mz_count for item in prepared
        ]
        source_max_intensities[start:stop] = [
            item.source_max_intensity for item in prepared
        ]
        ionizations.extend(item.ionization for item in prepared)

    compute_wall_seconds = time.perf_counter() - export_start
    metadata = metadata.copy()
    metadata["jestr_embedding_index"] = np.arange(len(metadata), dtype=np.int64)
    metadata["jestr_ionization"] = ionizations
    metadata["jestr_source_peak_count"] = source_peak_counts
    metadata["jestr_retained_peak_count"] = retained_peak_counts
    metadata["jestr_removed_above_max_mz_count"] = removed_peak_counts
    metadata["jestr_source_max_intensity"] = source_max_intensities
    metadata["jestr_inference_seconds"] = inference_seconds
    metadata.to_csv(output_metadata_path, index=False)

    arrays: dict[str, np.ndarray] = {
        "embeddings": embeddings,
        "ground_truth": targets,
        "spectrum_ids": spectrum_ids,
        "inchikeys": inchikeys,
        "inference_seconds": inference_seconds,
        "aggregate_inference_seconds": np.asarray(
            aggregate_inference_seconds, dtype=np.float64
        ),
    }
    if args.index_column in metadata:
        arrays[args.index_column] = metadata[args.index_column].to_numpy(dtype=np.int64)
    serialization_start = time.perf_counter()
    np.savez(output_path, **arrays)
    serialization_seconds = time.perf_counter() - serialization_start

    ordered_ids_sha256 = hashlib.sha256(
        "\n".join(spectrum_ids).encode("utf-8")
    ).hexdigest()
    ionization_counts = dict(sorted(Counter(ionizations).items()))
    exporter_path = Path(__file__).resolve()
    exporter_revision = args.exporter_revision or _try_git_revision(
        exporter_path.parents[1]
    )
    manifest = {
        "schema_version": 1,
        "exporter": {
            "path": str(exporter_path),
            "sha256": sha256_file(exporter_path),
            "frigid_revision": exporter_revision,
        },
        "model": {
            "name": "JESTR",
            "repository": OFFICIAL_REPOSITORY_URL,
            "paper": OFFICIAL_PAPER_URL,
            "source_revision": revision,
            "model_source_sha256": args.expected_model_source_sha256,
            "utils_source_sha256": args.expected_utils_source_sha256,
            "params_source_sha256": args.expected_params_source_sha256,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "l2_normalized": False,
            "implementation": (
                "architecture-only port of official SpecEncMLP_BIN; strict checkpoint load"
            ),
        },
        "inputs": {
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "spectra_dir": str(spectra_dir),
            "id_column": args.id_column,
            "inchikey_column": inchikey_column,
            "targets": str(targets_path) if targets_path else None,
            "targets_sha256": targets_sha256,
            "targets_source": targets_source,
            "ordered_spectrum_ids_sha256": ordered_ids_sha256,
            "rows": len(metadata),
        },
        "preprocessing": {
            "source_peak_layout": "FRIGID rows are [m/z, intensity]",
            "model_peak_layout": "JESTR rows are [intensity, m/z]",
            "intensity_normalization": "per-spectrum maximum scaled to 999 before filtering",
            "bin_count": BIN_COUNT,
            "bin_resolution": BIN_RESOLUTION,
            "mz_filter": "m/z < 1001",
            "bin_index": "int((m/z - 1) / 1)",
            "collision_reduction": "sum",
            "transformation": "log10(sum + 1) / 3",
            "precursor_peak_added": False,
            "adduct_input": False,
            "adduct_policy": "no row filtering; export every supplied spectrum",
        },
        "coverage": {
            "ionization_counts": ionization_counts,
            "spectra_with_above_max_mz_peaks": int(np.count_nonzero(removed_peak_counts)),
            "above_max_mz_peaks_removed": int(removed_peak_counts.sum()),
            "upstream_scope_warning": (
                "Official JESTR dataset loaders hard-filter [M+H]+, while the released "
                "MassSpecGym benchmark also contains [M+Na]+. Sodium-adduct rows are "
                "exported here but must be reported as an out-of-training-loader stratum."
            ),
        },
        "fingerprint_targets": {
            "array_key": "ground_truth",
            "type": "Morgan",
            "radius": FINGERPRINT_RADIUS,
            "bits": FINGERPRINT_BITS,
            "use_chirality": False,
            "dtype": str(targets.dtype),
        },
        "runtime": {
            "device": device,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "batch_size": args.batch_size,
            "aggregate_inference_seconds": aggregate_inference_seconds,
            "per_row_inference_method": "batch elapsed time divided equally across batch rows",
            "preprocessing_seconds": preprocessing_seconds,
            "target_seconds": target_seconds,
            "compute_wall_seconds": compute_wall_seconds,
            "serialization_seconds": serialization_seconds,
            "inference_spectra_per_second": (
                len(metadata) / aggregate_inference_seconds
                if aggregate_inference_seconds > 0.0
                else None
            ),
        },
        "outputs": {
            "bundle": str(output_path),
            "bundle_sha256": sha256_file(output_path),
            "metadata": str(output_metadata_path),
            "metadata_sha256": sha256_file(output_metadata_path),
            "npz_keys": sorted(arrays),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
