#!/usr/bin/env python
"""Export row-locked MSBERT embeddings for FRIGID ``.ms`` spectra.

The adapter intentionally imports the model definition from an explicit
checkout of the official MSBERT repository. It does not vendor or silently
modify the upstream architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OFFICIAL_REPOSITORY_URL = "https://github.com/zhanghailiangcsu/MSBERT"
OFFICIAL_RELEASE_TAG = "1.0"
OFFICIAL_RELEASE_REVISION = "8e4372abcd93aa3c9d9345527de1d250b3cd0aa8"
OFFICIAL_CHECKPOINT_SHA256 = (
    "f6a50e1a5650504370e563a9daf9117b4f4ecd60f5ba508dce2ee9e81f064605"
)
OFFICIAL_MODEL_SOURCE_SHA256 = (
    "33839b9ab63d11fbad6a92cdc33a6e37678ccd52a94053d1baca45b1a7ff42e3"
)

VOCABULARY_SIZE = 100_002
EMBEDDING_DIMENSION = 512
MAX_SEQUENCE_LENGTH = 100
MAX_FRAGMENT_PEAKS = MAX_SEQUENCE_LENGTH - 1
MIN_TRAINING_MZ = 10.0
MAX_TRAINING_MZ = 1000.0


@dataclass(frozen=True)
class PreparedSpectrum:
    """One spectrum encoded according to the released MSBERT input contract."""

    input_ids: np.ndarray
    intensity: np.ndarray
    source_peak_count: int
    retained_peak_count: int
    removed_domain_peak_count: int
    truncated_peak_count: int


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_released_vocabulary() -> dict[str, int]:
    """Build the exact 0.01-Da vocabulary used by ``MakeTrainData``."""

    words = np.round(
        np.linspace(0, 1000, 100 * 1000, endpoint=False),
        2,
    )
    vocabulary = {"[PAD]": 0, "[MASK]": 1}
    vocabulary.update({f"{word:.2f}": index + 2 for index, word in enumerate(words)})
    return vocabulary


def _parse_ms_file(path: str | Path) -> tuple[float, str, list[tuple[float, float]]]:
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
            if intensity < 0:
                raise ValueError(f"Negative peak intensity in {path}: {line!r}")
            peaks.append((mz, intensity))

    if precursor_mz is None or not np.isfinite(precursor_mz):
        raise ValueError(f"Missing finite >parentmass value in {path}")
    if ionization is None:
        raise ValueError(f"Missing >ionization value in {path}")
    if not ionization.endswith("+"):
        raise ValueError(
            f"MSBERT release 1.0 was trained for positive ion mode; "
            f"found {ionization!r} in {path}"
        )
    if not MIN_TRAINING_MZ <= precursor_mz < MAX_TRAINING_MZ:
        raise ValueError(
            f"Precursor m/z {precursor_mz} in {path} is outside the released "
            f"[{MIN_TRAINING_MZ:g}, {MAX_TRAINING_MZ:g}) domain"
        )
    return precursor_mz, ionization, peaks


def prepare_spectrum(
    path: str | Path,
    vocabulary: dict[str, int],
) -> PreparedSpectrum:
    """Convert one FRIGID spectrum into released MSBERT model tensors.

    The publication-defined 10 <= m/z < 1000 domain filter is applied before
    the released code's top-99 fragment selection. The precursor occupies the
    first token, with intensity 2; the sequence is then normalized and padded
    to 100 tokens exactly as ``ProDataset(..., n_max=99)`` followed by
    ``MakeTrainData(..., maxlen=100)``.
    """

    precursor_mz, _, source_peaks = _parse_ms_file(path)
    domain_peaks = [
        (mz, intensity)
        for mz, intensity in source_peaks
        if MIN_TRAINING_MZ <= mz < MAX_TRAINING_MZ
    ]
    removed_domain_peak_count = len(source_peaks) - len(domain_peaks)

    truncated_peak_count = max(0, len(domain_peaks) - MAX_FRAGMENT_PEAKS)
    if truncated_peak_count:
        intensities = np.asarray([intensity for _, intensity in domain_peaks])
        retained_indices = np.argsort(intensities)[::-1][:MAX_FRAGMENT_PEAKS]
        retained_indices = sorted(int(index) for index in retained_indices)
        domain_peaks = [domain_peaks[index] for index in retained_indices]

    mz_words = [f"{precursor_mz:.2f}"] + [f"{mz:.2f}" for mz, _ in domain_peaks]
    try:
        input_ids = [vocabulary[word] for word in mz_words]
    except KeyError as error:
        raise ValueError(f"m/z token {error.args[0]!r} from {path} is outside the vocabulary")
    input_ids.extend([vocabulary["[PAD]"]] * (MAX_SEQUENCE_LENGTH - len(input_ids)))

    intensity = np.asarray(
        [2.0] + [peak_intensity for _, peak_intensity in domain_peaks],
        dtype=np.float64,
    )
    intensity /= intensity.max()
    intensity = np.pad(intensity, (0, MAX_SEQUENCE_LENGTH - len(intensity)))

    return PreparedSpectrum(
        input_ids=np.asarray(input_ids, dtype=np.int64),
        intensity=intensity.astype(np.float32, copy=False),
        source_peak_count=len(source_peaks),
        retained_peak_count=len(domain_peaks),
        removed_domain_peak_count=removed_domain_peak_count,
        truncated_peak_count=truncated_peak_count,
    )


def load_ordered_metadata(
    path: str | Path,
    *,
    id_column: str,
    index_column: str,
) -> pd.DataFrame:
    """Load metadata and enforce a unique, deterministic spectrum order."""

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
    return metadata.reset_index(drop=True)


def compute_morgan_targets(
    metadata: pd.DataFrame,
    *,
    smiles_column: str,
    id_column: str,
    radius: int,
    bits: int,
) -> np.ndarray:
    """Compute the FRIGID target fingerprint contract in metadata row order."""

    if smiles_column not in metadata:
        raise ValueError(
            f"Metadata is missing SMILES column {smiles_column!r} required for targets"
        )
    if radius <= 0 or bits <= 0:
        raise ValueError("Morgan fingerprint radius and bit count must be positive")

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.warning")
    targets = np.empty((len(metadata), bits), dtype=np.uint8)
    for index, row in metadata.iterrows():
        smiles = str(row[smiles_column]).strip()
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(
                f"Could not parse SMILES for {row[id_column]!r}: {smiles!r}"
            )
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            radius,
            nBits=bits,
            useChirality=False,
        )
        target = np.zeros(bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, target)
        targets[index] = target
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
    expected_model_source_sha256: str,
    device: str,
) -> tuple[Any, Any, str, str, str]:
    import torch

    revision = _git_revision(repository)
    if revision != expected_revision:
        raise ValueError(
            f"MSBERT source revision is {revision}; expected {expected_revision}. "
            f"Check out release tag {OFFICIAL_RELEASE_TAG!r}."
        )

    model_source = repository / "model" / "MSBERTModel.py"
    if not model_source.is_file():
        raise FileNotFoundError(f"Official model source is missing: {model_source}")
    model_source_sha256 = sha256_file(model_source)
    if model_source_sha256 != expected_model_source_sha256:
        raise ValueError(
            f"MSBERT model source SHA-256 is {model_source_sha256}; "
            f"expected {expected_model_source_sha256}"
        )

    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"MSBERT checkpoint SHA-256 is {checkpoint_sha256}; "
            f"expected {expected_checkpoint_sha256}"
        )

    module_spec = importlib.util.spec_from_file_location(
        "official_msbert_release_model",
        model_source,
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not import official model source: {model_source}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {resolved_device}")

    model = module.MSBERT(
        VOCABULARY_SIZE,
        EMBEDDING_DIMENSION,
        6,
        16,
        0,
        MAX_SEQUENCE_LENGTH,
        3,
    )
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(resolved_device)
    return model, torch, resolved_device, revision, checkpoint_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official MSBERT 1.0 embeddings for row-locked FRIGID spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--spectra-dir", required=True)
    parser.add_argument("--msbert-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-metadata")
    parser.add_argument("--manifest")
    parser.add_argument("--id-column", default="spec_name")
    parser.add_argument("--index-column", default="fingerprint_index")
    parser.add_argument("--inchikey-column", default="inchi_key")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--targets-npz")
    parser.add_argument("--targets-key", default="ground_truth")
    parser.add_argument("--compute-morgan-targets", action="store_true")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=4096)
    parser.add_argument("--expected-source-revision", default=OFFICIAL_RELEASE_REVISION)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=OFFICIAL_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected-model-source-sha256",
        default=OFFICIAL_MODEL_SOURCE_SHA256,
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
    repository = Path(args.msbert_repo).resolve()
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_ordered_metadata(
        metadata_path,
        id_column=args.id_column,
        index_column=args.index_column,
    )
    spectrum_ids = metadata[args.id_column].to_numpy(dtype=str)
    if args.inchikey_column not in metadata:
        raise ValueError(
            f"Metadata is missing InChIKey column {args.inchikey_column!r}"
        )
    inchikeys = metadata[args.inchikey_column].astype(str).str.strip().to_numpy(dtype=str)
    if any(not value or value.lower() == "nan" for value in inchikeys):
        raise ValueError(
            f"Metadata column {args.inchikey_column!r} contains empty InChIKeys"
        )
    vocabulary = build_released_vocabulary()
    model, torch, device, revision, checkpoint_sha256 = _load_official_model(
        repository,
        checkpoint,
        expected_revision=args.expected_source_revision,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_model_source_sha256=args.expected_model_source_sha256,
        device=args.device,
    )

    embeddings = np.empty((len(metadata), EMBEDDING_DIMENSION), dtype=np.float32)
    source_peak_counts = np.empty(len(metadata), dtype=np.int32)
    retained_peak_counts = np.empty(len(metadata), dtype=np.int32)
    removed_domain_peak_counts = np.empty(len(metadata), dtype=np.int32)
    truncated_peak_counts = np.empty(len(metadata), dtype=np.int32)
    inference_seconds = 0.0
    wall_start = time.perf_counter()

    for start in range(0, len(metadata), args.batch_size):
        stop = min(start + args.batch_size, len(metadata))
        prepared = [
            prepare_spectrum(spectra_dir / f"{spectrum_id}.ms", vocabulary)
            for spectrum_id in spectrum_ids[start:stop]
        ]
        input_ids = torch.as_tensor(
            np.stack([item.input_ids for item in prepared]),
            dtype=torch.long,
            device=device,
        )
        intensity = torch.as_tensor(
            np.stack([item.intensity for item in prepared])[:, None, :],
            dtype=torch.float32,
            device=device,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        with torch.inference_mode():
            batch_embeddings = model.predict(input_ids, intensity).reshape(
                stop - start,
                EMBEDDING_DIMENSION,
            )
        batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float32, copy=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - inference_start

        if not np.isfinite(batch_embeddings).all():
            raise ValueError(f"MSBERT produced non-finite embeddings for rows {start}:{stop}")
        embeddings[start:stop] = batch_embeddings
        source_peak_counts[start:stop] = [item.source_peak_count for item in prepared]
        retained_peak_counts[start:stop] = [item.retained_peak_count for item in prepared]
        removed_domain_peak_counts[start:stop] = [
            item.removed_domain_peak_count for item in prepared
        ]
        truncated_peak_counts[start:stop] = [item.truncated_peak_count for item in prepared]

    wall_seconds = time.perf_counter() - wall_start
    metadata = metadata.copy()
    metadata["msbert_embedding_index"] = np.arange(len(metadata), dtype=np.int64)
    metadata["msbert_source_peak_count"] = source_peak_counts
    metadata["msbert_retained_peak_count"] = retained_peak_counts
    metadata["msbert_removed_domain_peak_count"] = removed_domain_peak_counts
    metadata["msbert_truncated_peak_count"] = truncated_peak_counts
    metadata.to_csv(output_metadata_path, index=False)

    arrays: dict[str, np.ndarray] = {
        "embeddings": embeddings,
        "spectrum_ids": spectrum_ids,
        "inchikeys": inchikeys,
        "inference_seconds": np.full(
            len(metadata),
            inference_seconds / len(metadata),
            dtype=np.float64,
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
                raise ValueError(
                    f"Target key {args.targets_key!r} is missing from {targets_path}; "
                    f"available keys: {target_arrays.files}"
                )
            targets = np.asarray(target_arrays[args.targets_key])
        expected_target_shape = (len(metadata), args.fingerprint_bits)
        if targets.shape != expected_target_shape:
            raise ValueError(
                f"Targets have shape {targets.shape}; expected {expected_target_shape}"
            )
        if not np.logical_or(np.isclose(targets, 0), np.isclose(targets, 1)).all():
            raise ValueError("Targets must be binary")
        arrays[args.targets_key] = targets.astype(np.uint8, copy=False)
    elif args.compute_morgan_targets:
        arrays[args.targets_key] = compute_morgan_targets(
            metadata,
            smiles_column=args.smiles_column,
            id_column=args.id_column,
            radius=args.fingerprint_radius,
            bits=args.fingerprint_bits,
        )
        targets_source = (
            f"{metadata_path}:{args.smiles_column}; Morgan radius={args.fingerprint_radius}, "
            f"bits={args.fingerprint_bits}, useChirality=false"
        )
    np.savez(output_path, **arrays)

    ordered_ids_sha256 = hashlib.sha256(
        "\n".join(spectrum_ids).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "model": {
            "name": "MSBERT",
            "repository": OFFICIAL_REPOSITORY_URL,
            "release_tag": OFFICIAL_RELEASE_TAG,
            "source_revision": revision,
            "model_source_sha256": args.expected_model_source_sha256,
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
            "ion_mode": "positive",
            "fragment_mz_interval": [MIN_TRAINING_MZ, MAX_TRAINING_MZ],
            "fragment_mz_upper_bound_exclusive": True,
            "mz_decimals": 2,
            "max_fragment_peaks": MAX_FRAGMENT_PEAKS,
            "fragment_selection": "highest intensity, original order restored",
            "precursor_first": True,
            "precursor_intensity": 2.0,
            "intensity_normalization": "divide by maximum after precursor insertion",
            "sequence_length": MAX_SEQUENCE_LENGTH,
            "position_embedding": False,
        },
        "coverage": {
            "spectra_with_fewer_than_five_retained_peaks": int(
                np.count_nonzero(retained_peak_counts < 5)
            ),
            "spectra_with_domain_peaks_removed": int(
                np.count_nonzero(removed_domain_peak_counts)
            ),
            "domain_peaks_removed": int(removed_domain_peak_counts.sum()),
            "spectra_truncated": int(np.count_nonzero(truncated_peak_counts)),
            "peaks_truncated": int(truncated_peak_counts.sum()),
        },
        "fingerprint_targets": {
            "included": args.targets_key in arrays,
            "array_key": args.targets_key if args.targets_key in arrays else None,
            "type": "Morgan",
            "radius": args.fingerprint_radius,
            "bits": int(arrays[args.targets_key].shape[1])
            if args.targets_key in arrays
            else args.fingerprint_bits,
            "use_chirality": False,
        },
        "runtime": {
            "device": device,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "batch_size": args.batch_size,
            "inference_seconds": inference_seconds,
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
