#!/usr/bin/env python
"""Export embeddings from the public SpecEmbedding Hugging Face Space release."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_msbert_embeddings import (  # noqa: E402
    compute_morgan_targets,
    load_ordered_metadata,
    sha256_file,
)


OFFICIAL_GITHUB_URL = "https://github.com/sword-nan/SpecEmbedding"
OFFICIAL_SPACE_URL = "https://huggingface.co/spaces/xp113280/SpecEmbedding"
OFFICIAL_SPACE_REVISION = "26665447238100728da3640675927f2c8bfb10cd"
OFFICIAL_CHECKPOINT_SHA256 = (
    "0ca0aa002a0d061a95410f7a4055e82c7fcb428d0ba04b5714ac3a4e7f0f5cca"
)
OFFICIAL_MODEL_SOURCE_SHA256 = (
    "9f4e5613eed5291e1c8f01e1b486d29e990dd30e6a722d45c92aabd108bd2c95"
)

EMBEDDING_DIMENSION = 512
MAX_SEQUENCE_LENGTH = 100
MAX_FRAGMENT_PEAKS = MAX_SEQUENCE_LENGTH - 1


@dataclass(frozen=True)
class PreparedSpectrum:
    """One spectrum tokenized according to the public Space implementation."""

    mz: np.ndarray
    intensity: np.ndarray
    mask: np.ndarray
    source_peak_count: int
    retained_peak_count: int
    truncated_peak_count: int


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
            if mz <= 0.0 or intensity < 0.0:
                raise ValueError(f"Invalid peak row in {path}: {line!r}")
            peaks.append((mz, intensity))

    if precursor_mz is None or not np.isfinite(precursor_mz) or precursor_mz <= 0.0:
        raise ValueError(f"Missing positive finite >parentmass value in {path}")
    if ionization is None:
        raise ValueError(f"Missing >ionization value in {path}")
    if not ionization.endswith("+"):
        raise ValueError(
            f"The public SpecEmbedding Space accepts positive-ion spectra; "
            f"found {ionization!r} in {path}"
        )
    if not peaks:
        raise ValueError(f"No MS2 peaks found in {path}")
    return precursor_mz, ionization, peaks


def prepare_spectrum(path: str | Path) -> PreparedSpectrum:
    """Reproduce the Space tokenizer's top-99 selection and precursor token."""

    precursor_mz, _, peaks = _parse_ms_file(path)
    maximum_intensity = max(intensity for _, intensity in peaks)
    if maximum_intensity <= 0.0:
        raise ValueError(f"Spectrum has no positive peak intensity: {path}")

    truncated_peak_count = max(0, len(peaks) - MAX_FRAGMENT_PEAKS)
    if truncated_peak_count:
        intensities = np.asarray([intensity for _, intensity in peaks])
        retained_indexes = np.sort(
            np.argsort(intensities)[::-1][:MAX_FRAGMENT_PEAKS]
        )
        peaks = [peaks[int(index)] for index in retained_indexes]
    retained_maximum = max(intensity for _, intensity in peaks)

    mz = np.asarray(
        [precursor_mz] + [peak_mz for peak_mz, _ in peaks], dtype=np.float32
    )
    intensity = np.asarray(
        [2.0] + [peak_intensity / retained_maximum for _, peak_intensity in peaks],
        dtype=np.float32,
    )
    mask = np.zeros(MAX_SEQUENCE_LENGTH, dtype=bool)
    if len(mz) < MAX_SEQUENCE_LENGTH:
        mask[len(mz) :] = True
        mz = np.pad(mz, (0, MAX_SEQUENCE_LENGTH - len(mz)))
        intensity = np.pad(intensity, (0, MAX_SEQUENCE_LENGTH - len(intensity)))
    return PreparedSpectrum(
        mz=mz,
        intensity=intensity,
        mask=mask,
        source_peak_count=len(peaks) + truncated_peak_count,
        retained_peak_count=len(peaks),
        truncated_peak_count=truncated_peak_count,
    )


def _load_official_model(
    source_path: Path,
    checkpoint: Path,
    *,
    expected_model_source_sha256: str,
    expected_checkpoint_sha256: str,
    device: str,
) -> tuple[Any, Any, str, str]:
    """Load the exact model architecture and checkpoint used by the public Space."""

    source_sha256 = sha256_file(source_path)
    if source_sha256 != expected_model_source_sha256:
        raise ValueError(
            f"SpecEmbedding model source SHA-256 is {source_sha256}; "
            f"expected {expected_model_source_sha256}"
        )
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"SpecEmbedding checkpoint SHA-256 is {checkpoint_sha256}; "
            f"expected {expected_checkpoint_sha256}"
        )

    import torch

    module_spec = importlib.util.spec_from_file_location(
        "official_specembedding_space_model", source_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not import SpecEmbedding model source: {source_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    resolved_device = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {resolved_device}")

    model = module.SiameseModel(
        embedding_dim=512,
        n_head=16,
        n_layer=4,
        dim_feedward=512,
        dim_target=512,
        feedward_activation="selu",
    )
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(resolved_device)
    return model, torch, resolved_device, checkpoint_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export embeddings from the released SpecEmbedding Space model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--spectra-dir", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-metadata")
    parser.add_argument("--manifest")
    parser.add_argument("--id-column", default="spec_name")
    parser.add_argument("--index-column", default="fingerprint_index")
    parser.add_argument("--inchikey-column", default="inchi_key")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--targets-npz")
    parser.add_argument("--targets-key", default="ground_truth")
    parser.add_argument("--compute-morgan-targets", action="store_true")
    parser.add_argument(
        "--expected-model-source-sha256", default=OFFICIAL_MODEL_SOURCE_SHA256
    )
    parser.add_argument(
        "--expected-checkpoint-sha256", default=OFFICIAL_CHECKPOINT_SHA256
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
    source_path = Path(args.model_source).resolve()
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
    )
    if args.inchikey_column not in metadata:
        raise ValueError(
            f"Metadata is missing InChIKey column {args.inchikey_column!r}"
        )
    spectrum_ids = metadata[args.id_column].to_numpy(dtype=str)
    inchikeys = metadata[args.inchikey_column].astype(str).str.strip().to_numpy(dtype=str)
    if any(not value or value.lower() == "nan" for value in inchikeys):
        raise ValueError(f"Metadata column {args.inchikey_column!r} contains empty values")

    model, torch, device, checkpoint_sha256 = _load_official_model(
        source_path,
        checkpoint,
        expected_model_source_sha256=args.expected_model_source_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        device=args.device,
    )
    embeddings = np.empty((len(metadata), EMBEDDING_DIMENSION), dtype=np.float32)
    source_peak_counts = np.empty(len(metadata), dtype=np.int32)
    retained_peak_counts = np.empty(len(metadata), dtype=np.int32)
    truncated_peak_counts = np.empty(len(metadata), dtype=np.int32)
    inference_seconds = 0.0
    wall_started = time.perf_counter()

    for start in range(0, len(metadata), args.batch_size):
        stop = min(start + args.batch_size, len(metadata))
        prepared = [
            prepare_spectrum(spectra_dir / f"{spectrum_id}.ms")
            for spectrum_id in spectrum_ids[start:stop]
        ]
        mz = torch.as_tensor(
            np.stack([item.mz for item in prepared]),
            dtype=torch.float32,
            device=device,
        )
        intensity = torch.as_tensor(
            np.stack([item.intensity for item in prepared]),
            dtype=torch.float32,
            device=device,
        )
        mask = torch.as_tensor(
            np.stack([item.mask for item in prepared]),
            dtype=torch.bool,
            device=device,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            batch_embeddings = model(mz, intensity, mask)
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
        retained_peak_counts[start:stop] = [item.retained_peak_count for item in prepared]
        truncated_peak_counts[start:stop] = [item.truncated_peak_count for item in prepared]

    wall_seconds = time.perf_counter() - wall_started
    metadata = metadata.copy()
    metadata["specembedding_embedding_index"] = np.arange(len(metadata), dtype=np.int64)
    metadata["specembedding_source_peak_count"] = source_peak_counts
    metadata["specembedding_retained_peak_count"] = retained_peak_counts
    metadata["specembedding_truncated_peak_count"] = truncated_peak_counts
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
        expected_shape = (len(metadata), 4096)
        if targets.shape != expected_shape:
            raise ValueError(f"Targets have shape {targets.shape}; expected {expected_shape}")
        arrays[args.targets_key] = targets.astype(np.uint8, copy=False)
    elif args.compute_morgan_targets:
        arrays[args.targets_key] = compute_morgan_targets(
            metadata,
            smiles_column=args.smiles_column,
            id_column=args.id_column,
            radius=2,
            bits=4096,
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
            "name": "SpecEmbedding",
            "github": OFFICIAL_GITHUB_URL,
            "space": OFFICIAL_SPACE_URL,
            "space_revision": OFFICIAL_SPACE_REVISION,
            "model_source": str(source_path),
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
            "max_fragment_peaks": MAX_FRAGMENT_PEAKS,
            "fragment_selection": "highest intensity, original order restored",
            "fragment_intensity_normalization": "divide by retained maximum",
            "precursor_first": True,
            "precursor_intensity": 2.0,
            "sequence_length": MAX_SEQUENCE_LENGTH,
            "pooling": "unmasked mean over all 100 positions in public Space source",
        },
        "coverage": {
            "spectra_with_fewer_than_five_retained_peaks": int(
                np.count_nonzero(retained_peak_counts < 5)
            ),
            "spectra_truncated": int(np.count_nonzero(truncated_peak_counts)),
            "peaks_truncated": int(truncated_peak_counts.sum()),
        },
        "fingerprint_targets": {
            "included": args.targets_key in arrays,
            "array_key": args.targets_key if args.targets_key in arrays else None,
            "type": "Morgan",
            "radius": 2,
            "bits": 4096,
            "use_chirality": False,
        },
        "runtime": {
            "device": device,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "batch_size": args.batch_size,
            "inference_seconds": inference_seconds,
            "inference_timing_scope": "synchronized model forward only",
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
