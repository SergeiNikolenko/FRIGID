#!/usr/bin/env python
"""Export released MS2DeepScore 2.0 embeddings for row-locked FRIGID spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OFFICIAL_REPOSITORY_URL = "https://github.com/matchms/ms2deepscore"
OFFICIAL_RELEASE_TAG = "2.5.3"
OFFICIAL_RELEASE_REVISION = "5de7c58c12b4209bbff5c9dbd4ca47bb70507971"
OFFICIAL_MODEL_RECORD = "https://doi.org/10.5281/zenodo.14290920"
OFFICIAL_CHECKPOINT_SHA256 = (
    "e7e0c57a5d25bfd328e27d00af9e5559ebc5b85eda1640c61c8436b3497f9821"
)
OFFICIAL_SOURCE_SHA256 = {
    "ms2deepscore/models/load_model.py": (
        "f65560bbbca218908db7c2bf0a29fccffdd50d527a2e6a32525df85072172279"
    ),
    "ms2deepscore/models/SiameseSpectralModel.py": (
        "08d281da78f82699296531077163572f9ce606805c724eb128e999d9c2d6cbf8"
    ),
    "ms2deepscore/tensorize_spectra.py": (
        "fe9aa58b1b9687d1f123a872c2478e475cbce6c0cd564b82d8aed0cd68117faf"
    ),
    "ms2deepscore/MetadataFeatureGenerator.py": (
        "fac8669299982c00f3ab479bb245efb76399a1ab73cd3f983de8b5cd68faac7c"
    ),
}

EMBEDDING_DIMENSION = 500
MIN_MZ = 10.0
MAX_MZ = 1000.0
MZ_BIN_WIDTH = 0.1
FINGERPRINT_BITS = 4096
FINGERPRINT_RADIUS = 2


@dataclass(frozen=True)
class PreparedSpectrum:
    """One validated spectrum represented in the public model's input domain."""

    precursor_mz: float
    ion_mode: str
    mz: np.ndarray
    intensities: np.ndarray
    source_peak_count: int
    removed_domain_peak_count: int


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA-256 without loading the full file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ms_file(path: str | Path) -> PreparedSpectrum:
    """Read one FRIGID ``.ms`` file and apply the released m/z domain."""

    path = Path(path)
    precursor_mz: float | None = None
    ionization: str | None = None
    peaks: list[tuple[float, float]] = []
    reading_peaks = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith(">parentmass "):
            precursor_mz = float(line.split(maxsplit=1)[1])
        elif lower.startswith(">ionization "):
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
            if mz <= 0.0 or intensity < 0.0:
                raise ValueError(f"Invalid peak row in {path}: {line!r}")
            peaks.append((mz, intensity))

    if precursor_mz is None or not np.isfinite(precursor_mz) or precursor_mz <= 0.0:
        raise ValueError(f"Missing positive finite >parentmass value in {path}")
    if ionization is None:
        raise ValueError(f"Missing >ionization value in {path}")
    if ionization.endswith("+"):
        ion_mode = "positive"
    elif ionization.endswith("-"):
        ion_mode = "negative"
    else:
        raise ValueError(f"Unsupported ionization value in {path}: {ionization!r}")
    if not peaks:
        raise ValueError(f"No MS2 peaks found in {path}")

    retained = [(mz, intensity) for mz, intensity in peaks if MIN_MZ <= mz < MAX_MZ]
    if not retained:
        raise ValueError(
            f"No MS2 peaks remain in the released [{MIN_MZ:g}, {MAX_MZ:g}) domain: {path}"
        )
    maximum_intensity = max(intensity for _, intensity in retained)
    if maximum_intensity <= 0.0:
        raise ValueError(f"Spectrum has no positive retained intensity: {path}")

    return PreparedSpectrum(
        precursor_mz=precursor_mz,
        ion_mode=ion_mode,
        mz=np.asarray([mz for mz, _ in retained], dtype=np.float64),
        intensities=np.asarray(
            [intensity / maximum_intensity for _, intensity in retained],
            dtype=np.float64,
        ),
        source_peak_count=len(peaks),
        removed_domain_peak_count=len(peaks) - len(retained),
    )


def load_ordered_metadata(
    path: str | Path,
    *,
    id_column: str,
    index_column: str,
    inchikey_column: str,
) -> pd.DataFrame:
    """Load metadata and enforce one unique deterministic spectrum order."""

    path = Path(path)
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    metadata = pd.read_csv(path, sep=delimiter)
    required = {id_column, inchikey_column}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata {path} is missing columns: {missing}")
    if metadata.empty:
        raise ValueError(f"Metadata is empty: {path}")

    for column in (id_column, inchikey_column):
        values = metadata[column].astype(str).str.strip()
        if values.eq("").any() or values.str.lower().eq("nan").any():
            raise ValueError(f"Metadata column {column!r} contains an empty value")
        if column == id_column and values.duplicated().any():
            raise ValueError(f"Metadata column {column!r} contains duplicate IDs")
        metadata = metadata.assign(**{column: values})

    if index_column in metadata:
        indexes = pd.to_numeric(metadata[index_column], errors="raise").to_numpy()
        if not np.equal(indexes, indexes.astype(np.int64)).all():
            raise ValueError(f"Metadata column {index_column!r} must contain integers")
        metadata = metadata.assign(
            **{index_column: indexes.astype(np.int64)}
        ).sort_values(index_column, kind="stable")
        expected = np.arange(len(metadata), dtype=np.int64)
        if not np.array_equal(metadata[index_column].to_numpy(), expected):
            raise ValueError(
                f"Metadata column {index_column!r} must contain every index from "
                f"0 to {len(metadata) - 1} exactly once"
            )
    return metadata.reset_index(drop=True)


def compute_morgan_targets(
    metadata: pd.DataFrame,
    *,
    smiles_column: str,
    id_column: str,
) -> np.ndarray:
    """Compute the exact non-chiral Morgan radius-2/4096 target contract."""

    if smiles_column not in metadata:
        raise ValueError(f"Metadata is missing SMILES column {smiles_column!r}")

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


def _git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_official_model(
    repository: Path,
    checkpoint: Path,
    *,
    expected_revision: str,
    expected_checkpoint_sha256: str,
    device: str,
) -> tuple[Any, Any, Any, str, str]:
    """Verify the pinned release and load its unmodified public checkpoint."""

    revision = _git_revision(repository)
    if revision != expected_revision:
        raise ValueError(
            f"MS2DeepScore source revision is {revision}; expected {expected_revision}"
        )
    for relative_path, expected_sha256 in OFFICIAL_SOURCE_SHA256.items():
        source_path = repository / relative_path
        actual_sha256 = sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Source SHA-256 mismatch for {source_path}: "
                f"{actual_sha256} != {expected_sha256}"
            )

    checkpoint_sha256 = sha256_file(checkpoint)
    if not expected_checkpoint_sha256:
        raise ValueError("An expected checkpoint SHA-256 is required")
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 is {checkpoint_sha256}; "
            f"expected {expected_checkpoint_sha256}"
        )

    sys.path.insert(0, str(repository))
    import torch
    from matchms import Spectrum
    from ms2deepscore.models import load_model
    from ms2deepscore.tensorize_spectra import tensorize_spectra

    resolved_device = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {resolved_device}")

    model = load_model(checkpoint)
    settings = model.model_settings
    expected_settings = {
        "embedding_dim": EMBEDDING_DIMENSION,
        "min_mz": int(MIN_MZ),
        "max_mz": int(MAX_MZ),
        "mz_bin_width": MZ_BIN_WIDTH,
        "ionisation_mode": "both",
    }
    for name, expected in expected_settings.items():
        actual = getattr(settings, name)
        if actual != expected:
            raise ValueError(f"Unexpected checkpoint setting {name}: {actual!r}")
    metadata_fields = {
        str(item[1].get("metadata_field")) for item in settings.additional_metadata
    }
    if metadata_fields != {"precursor_mz", "ionmode"}:
        raise ValueError(
            f"Unexpected checkpoint metadata fields: {sorted(metadata_fields)}"
        )
    model.eval().to(resolved_device)
    return model, torch, (Spectrum, tensorize_spectra), resolved_device, checkpoint_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official MS2DeepScore 2.0 embeddings for FRIGID spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--spectra-dir", required=True)
    parser.add_argument("--ms2deepscore-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-metadata")
    parser.add_argument("--manifest")
    parser.add_argument("--id-column", default="spec_name")
    parser.add_argument("--index-column", default="fingerprint_index")
    parser.add_argument("--inchikey-column", default="inchi_key")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--targets-npz")
    parser.add_argument("--targets-key", default="ground_truth")
    parser.add_argument("--compute-morgan-targets", action="store_true")
    parser.add_argument("--expected-source-revision", default=OFFICIAL_RELEASE_REVISION)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=OFFICIAL_CHECKPOINT_SHA256,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.targets_npz and args.compute_morgan_targets:
        raise ValueError("Use either --targets-npz or --compute-morgan-targets, not both")

    metadata_path = Path(args.metadata).resolve()
    spectra_dir = Path(args.spectra_dir).resolve()
    repository = Path(args.ms2deepscore_repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    output_metadata_path = Path(
        args.output_metadata or output_path.with_suffix(".metadata.csv")
    ).resolve()
    manifest_path = Path(args.manifest or output_path.with_suffix(".manifest.json")).resolve()
    if len({output_path, output_metadata_path, manifest_path}) != 3:
        raise ValueError("Output NPZ, metadata, and manifest paths must be distinct")
    for path in (output_path, output_metadata_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_ordered_metadata(
        metadata_path,
        id_column=args.id_column,
        index_column=args.index_column,
        inchikey_column=args.inchikey_column,
    )
    spectrum_ids = metadata[args.id_column].to_numpy(dtype=str)
    inchikeys = metadata[args.inchikey_column].to_numpy(dtype=str)
    model, torch, upstream, device, checkpoint_sha256 = _load_official_model(
        repository,
        checkpoint,
        expected_revision=args.expected_source_revision,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        device=args.device,
    )
    Spectrum, tensorize_spectra = upstream

    embeddings = np.empty((len(metadata), EMBEDDING_DIMENSION), dtype=np.float32)
    source_peak_counts = np.empty(len(metadata), dtype=np.int32)
    retained_peak_counts = np.empty(len(metadata), dtype=np.int32)
    removed_domain_peak_counts = np.empty(len(metadata), dtype=np.int32)
    ion_modes = np.empty(len(metadata), dtype="U8")
    inference_seconds = 0.0
    wall_started = time.perf_counter()

    for start in range(0, len(metadata), args.batch_size):
        stop = min(start + args.batch_size, len(metadata))
        prepared = [
            parse_ms_file(spectra_dir / f"{spectrum_id}.ms")
            for spectrum_id in spectrum_ids[start:stop]
        ]
        spectra = [
            Spectrum(
                mz=item.mz,
                intensities=item.intensities,
                metadata={
                    "precursor_mz": item.precursor_mz,
                    "ionmode": item.ion_mode,
                },
            )
            for item in prepared
        ]
        peak_tensors, metadata_tensors = tensorize_spectra(
            spectra, model.model_settings
        )
        peak_tensors = peak_tensors.to(device)
        metadata_tensors = metadata_tensors.to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            batch_embeddings = model.encoder(peak_tensors, metadata_tensors)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - inference_started
        batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float32, copy=False)
        if batch_embeddings.shape != (stop - start, EMBEDDING_DIMENSION):
            raise ValueError(
                f"Unexpected embedding shape {batch_embeddings.shape} for rows {start}:{stop}"
            )
        if not np.isfinite(batch_embeddings).all():
            raise ValueError(f"Non-finite embeddings for rows {start}:{stop}")

        embeddings[start:stop] = batch_embeddings
        source_peak_counts[start:stop] = [item.source_peak_count for item in prepared]
        retained_peak_counts[start:stop] = [len(item.mz) for item in prepared]
        removed_domain_peak_counts[start:stop] = [
            item.removed_domain_peak_count for item in prepared
        ]
        ion_modes[start:stop] = [item.ion_mode for item in prepared]

    wall_seconds = time.perf_counter() - wall_started
    metadata = metadata.copy()
    metadata["ms2deepscore_embedding_index"] = np.arange(len(metadata), dtype=np.int64)
    metadata["ms2deepscore_source_peak_count"] = source_peak_counts
    metadata["ms2deepscore_retained_peak_count"] = retained_peak_counts
    metadata["ms2deepscore_removed_domain_peak_count"] = removed_domain_peak_counts
    metadata["ms2deepscore_ion_mode"] = ion_modes
    metadata.to_csv(output_metadata_path, index=False)

    arrays: dict[str, np.ndarray] = {
        "embeddings": embeddings,
        "spectrum_ids": spectrum_ids,
        "inchikeys": inchikeys,
        "inference_seconds": np.full(
            len(metadata), inference_seconds / len(metadata), dtype=np.float64
        ),
    }
    if args.index_column in metadata:
        arrays[args.index_column] = metadata[args.index_column].to_numpy(dtype=np.int64)

    targets_path = None
    targets_sha256 = None
    targets_source = None
    if args.targets_npz:
        targets_path = Path(args.targets_npz).resolve()
        targets_sha256 = sha256_file(targets_path)
        targets_source = f"{targets_path}:{args.targets_key}"
        with np.load(targets_path, allow_pickle=False) as target_arrays:
            if args.targets_key not in target_arrays:
                raise ValueError(f"Target key {args.targets_key!r} is missing")
            targets = np.asarray(target_arrays[args.targets_key])
        if targets.shape != (len(metadata), FINGERPRINT_BITS):
            raise ValueError(
                f"Targets have shape {targets.shape}; "
                f"expected {(len(metadata), FINGERPRINT_BITS)}"
            )
        arrays[args.targets_key] = targets.astype(np.uint8, copy=False)
    elif args.compute_morgan_targets:
        arrays[args.targets_key] = compute_morgan_targets(
            metadata,
            smiles_column=args.smiles_column,
            id_column=args.id_column,
        )
        targets_source = (
            f"{metadata_path}:{args.smiles_column}; Morgan radius=2, bits=4096, "
            "useChirality=false"
        )
    np.savez(output_path, **arrays)

    ordered_ids_sha256 = hashlib.sha256(
        "\n".join(spectrum_ids).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "model": {
            "name": "MS2DeepScore 2.0",
            "repository": OFFICIAL_REPOSITORY_URL,
            "release_tag": OFFICIAL_RELEASE_TAG,
            "source_revision": args.expected_source_revision,
            "source_sha256": OFFICIAL_SOURCE_SHA256,
            "model_record": OFFICIAL_MODEL_RECORD,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "l2_normalized": False,
        },
        "inputs": {
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "spectra_dir": str(spectra_dir),
            "targets": str(targets_path) if targets_path else None,
            "targets_sha256": targets_sha256,
            "targets_source": targets_source,
            "ordered_spectrum_ids_sha256": ordered_ids_sha256,
            "rows": len(metadata),
        },
        "preprocessing": {
            "ion_modes": sorted(set(ion_modes.tolist())),
            "fragment_mz_interval": [MIN_MZ, MAX_MZ],
            "fragment_mz_upper_bound_exclusive": True,
            "mz_bin_width": MZ_BIN_WIDTH,
            "intensity_normalization": "divide retained peaks by spectrum maximum",
            "intensity_scaling": "square root inside official tensorize_spectra",
            "additional_metadata": ["precursor_mz", "ionmode"],
        },
        "coverage": {
            "spectra_with_fewer_than_five_retained_peaks": int(
                np.count_nonzero(retained_peak_counts < 5)
            ),
            "spectra_with_domain_peaks_removed": int(
                np.count_nonzero(removed_domain_peak_counts)
            ),
            "domain_peaks_removed": int(removed_domain_peak_counts.sum()),
        },
        "fingerprint_targets": {
            "included": args.targets_key in arrays,
            "array_key": args.targets_key if args.targets_key in arrays else None,
            "type": "Morgan",
            "radius": FINGERPRINT_RADIUS,
            "bits": FINGERPRINT_BITS,
            "use_chirality": False,
        },
        "runtime": {
            "device": device,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "batch_size": args.batch_size,
            "inference_seconds": inference_seconds,
            "inference_timing_scope": "synchronized encoder forward only",
            "wall_seconds": wall_seconds,
            "inference_spectra_per_second": len(metadata) / inference_seconds,
        },
        "outputs": {
            "embeddings": str(output_path),
            "embeddings_sha256": sha256_file(output_path),
            "metadata": str(output_metadata_path),
            "metadata_sha256": sha256_file(output_metadata_path),
            "npz_keys": sorted(arrays),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
