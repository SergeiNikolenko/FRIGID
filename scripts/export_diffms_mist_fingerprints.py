#!/usr/bin/env python
"""Export the released DiffMS MIST encoder probabilities on a locked manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


FINGERPRINT_BITS = 4096
ENCODER_PREFIX = "encoder."


@dataclass(frozen=True)
class LockedSpectrumRow:
    """One spectrum row resolved against the immutable MSG inputs."""

    spectrum_id: str
    formula: str
    instrument: str
    spectrum_path: Path
    subformula_path: Path


class SpectrumOnlyDataset(Dataset):
    """Featurize only the spectrum fields consumed by the MIST encoder."""

    def __init__(
        self,
        spectra: Sequence[Any],
        featurizer: Any,
    ) -> None:
        self.spectra = list(spectra)
        self.featurizer = featurizer

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.featurizer.featurize(self.spectra[index], train_mode=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw Morgan-4096 probabilities from the released DiffMS "
            "pretrained MIST spectrum encoder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--diffms-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--released-checkpoint-archive", default=None)
    parser.add_argument(
        "--released-encoder-member", default="checkpoints/encoder_msg.pt"
    )
    parser.add_argument("--reference-metadata", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--spectra-dir", required=True)
    parser.add_argument("--subformula-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-column", default="spec_name")
    parser.add_argument("--index-column", default="fingerprint_index")
    parser.add_argument("--expected-split", default="val")
    parser.add_argument("--expected-rows", type=int, default=19043)
    parser.add_argument("--max-rows", type=int, default=None, help="Smoke-test prefix only")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--code-revision", default=None)
    return parser.parse_args(argv)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in stable key order."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _decode_nonempty(values: pd.Series, *, source: str) -> list[str]:
    decoded = values.astype(str).str.strip().tolist()
    if any(not value or value.lower() == "nan" for value in decoded):
        raise ValueError(f"{source} contains an empty value")
    return decoded


def load_locked_rows(
    reference_metadata: str | Path,
    labels_path: str | Path,
    split_path: str | Path,
    spectra_dir: str | Path,
    subformula_dir: str | Path,
    *,
    id_column: str,
    index_column: str,
    expected_split: str,
    expected_rows: int | None,
) -> tuple[list[LockedSpectrumRow], str]:
    """Resolve and strictly validate every row in the locked benchmark order."""

    metadata = pd.read_csv(reference_metadata)
    required_metadata = {id_column, index_column}
    missing_metadata = sorted(required_metadata - set(metadata.columns))
    if missing_metadata:
        raise ValueError(
            f"Reference metadata is missing columns {missing_metadata}: {reference_metadata}"
        )
    indexes = pd.to_numeric(metadata[index_column], errors="raise").to_numpy()
    if not np.equal(indexes, indexes.astype(np.int64)).all():
        raise ValueError(f"{index_column} must contain integer indexes")
    metadata = metadata.assign(**{index_column: indexes.astype(np.int64)}).sort_values(
        index_column, kind="stable"
    )
    expected_indexes = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(metadata[index_column].to_numpy(), expected_indexes):
        raise ValueError(
            f"{index_column} must contain every index from 0 to {len(metadata) - 1} exactly once"
        )
    spectrum_ids = _decode_nonempty(metadata[id_column], source=str(reference_metadata))
    if len(set(spectrum_ids)) != len(spectrum_ids):
        raise ValueError(f"{reference_metadata}:{id_column} contains duplicate IDs")
    if expected_rows is not None and len(spectrum_ids) != expected_rows:
        raise ValueError(
            f"Locked manifest has {len(spectrum_ids)} rows; expected {expected_rows}"
        )

    labels = pd.read_csv(labels_path, sep="\t", dtype=str)
    required_labels = {"spec", "formula"}
    missing_labels = sorted(required_labels - set(labels.columns))
    if missing_labels:
        raise ValueError(f"Labels are missing columns {missing_labels}: {labels_path}")
    if labels["spec"].duplicated().any():
        duplicates = labels.loc[labels["spec"].duplicated(), "spec"].head(5).tolist()
        raise ValueError(f"Labels contain duplicate spectrum IDs: {duplicates}")
    labels = labels.set_index("spec", drop=False)

    split = pd.read_csv(split_path, sep="\t", dtype=str)
    if not {"name", "split"}.issubset(split.columns):
        raise ValueError(f"Split file must contain 'name' and 'split': {split_path}")
    if split["name"].duplicated().any():
        duplicates = split.loc[split["name"].duplicated(), "name"].head(5).tolist()
        raise ValueError(f"Split file contains duplicate spectrum IDs: {duplicates}")
    split_by_name = split.set_index("name")["split"]

    missing_labels = [value for value in spectrum_ids if value not in labels.index]
    missing_split = [value for value in spectrum_ids if value not in split_by_name.index]
    if missing_labels:
        raise ValueError(f"Locked IDs missing from labels: {missing_labels[:5]}")
    if missing_split:
        raise ValueError(f"Locked IDs missing from split file: {missing_split[:5]}")
    wrong_split = [
        value for value in spectrum_ids if split_by_name.loc[value] != expected_split
    ]
    if wrong_split:
        raise ValueError(
            f"Locked IDs outside expected split {expected_split!r}: {wrong_split[:5]}"
        )

    spectra_dir = Path(spectra_dir).resolve()
    subformula_dir = Path(subformula_dir).resolve()
    rows: list[LockedSpectrumRow] = []
    subformula_digest = hashlib.sha256()
    missing_spectra: list[str] = []
    missing_subformulae: list[str] = []
    for spectrum_id in spectrum_ids:
        label = labels.loc[spectrum_id]
        formula = str(label["formula"]).strip()
        if not formula or formula.lower() == "nan":
            raise ValueError(f"Empty molecular formula for {spectrum_id}")
        instrument = ""
        if "instrument" in labels.columns and pd.notna(label["instrument"]):
            instrument = str(label["instrument"]).strip()
        spectrum_path = spectra_dir / f"{spectrum_id}.ms"
        subformula_path = subformula_dir / f"{spectrum_id}.json"
        if not spectrum_path.is_file():
            missing_spectra.append(spectrum_id)
        if not subformula_path.is_file():
            missing_subformulae.append(spectrum_id)
        else:
            subformula_digest.update(spectrum_id.encode("utf-8"))
            subformula_digest.update(b"\0")
            subformula_digest.update(sha256_file(subformula_path).encode("ascii"))
            subformula_digest.update(b"\n")
        rows.append(
            LockedSpectrumRow(
                spectrum_id=spectrum_id,
                formula=formula,
                instrument=instrument,
                spectrum_path=spectrum_path,
                subformula_path=subformula_path,
            )
        )
    if missing_spectra:
        raise ValueError(f"Locked spectra files are missing: {missing_spectra[:5]}")
    if missing_subformulae:
        raise ValueError(f"Locked subformula files are missing: {missing_subformulae[:5]}")
    return rows, subformula_digest.hexdigest()


def extract_encoder_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extract only the released spectrum encoder from a DiffMS checkpoint."""

    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise ValueError("Checkpoint must be a state dictionary or contain 'state_dict'")
    encoder_state = {
        str(key)[len(ENCODER_PREFIX) :]: value
        for key, value in raw_state.items()
        if str(key).startswith(ENCODER_PREFIX)
    }
    if not encoder_state:
        raise ValueError(f"Checkpoint contains no keys with prefix {ENCODER_PREFIX!r}")
    if not all(isinstance(value, torch.Tensor) for value in encoder_state.values()):
        raise ValueError("Encoder state contains a non-tensor value")
    return encoder_state


def load_released_encoder_member(
    archive_path: str | Path,
    member_name: str,
) -> tuple[dict[str, torch.Tensor], str, int]:
    """Read and hash a standalone encoder directly from the official archive."""

    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember(member_name)
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"Unable to read {member_name!r} from {archive_path}")
        payload = handle.read()
    raw_state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(raw_state, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in raw_state.values()
    ):
        raise ValueError(f"Released encoder member is not a tensor state dict: {member_name}")
    return dict(raw_state), hashlib.sha256(payload).hexdigest(), len(payload)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(value)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    expected_ids: Sequence[str],
    *,
    device: torch.device,
    fingerprint_bits: int = FINGERPRINT_BITS,
    warmup_batches: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run row-locked inference and return probabilities and forward-only latency."""

    model.eval()
    if warmup_batches > 0:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                device_batch = move_batch(batch, device)
                model(device_batch)
                if batch_index + 1 >= warmup_batches:
                    break
        synchronize(device)

    probabilities = np.empty((len(expected_ids), fingerprint_bits), dtype=np.float32)
    inference_seconds = np.empty(len(expected_ids), dtype=np.float64)
    row_offset = 0
    aggregate_seconds = 0.0
    with torch.inference_mode():
        for batch in loader:
            names = [str(value) for value in batch.get("names", [])]
            stop = row_offset + len(names)
            if not names:
                raise ValueError("Encoder batch contains no spectrum names")
            if names != list(expected_ids[row_offset:stop]):
                raise ValueError(
                    f"Batch IDs diverge from the locked manifest at row {row_offset}: "
                    f"got {names[:3]}, expected {list(expected_ids[row_offset:stop])[:3]}"
                )
            device_batch = move_batch(batch, device)
            synchronize(device)
            started = time.perf_counter()
            output = model(device_batch)
            synchronize(device)
            elapsed = time.perf_counter() - started
            batch_probabilities = output[0] if isinstance(output, tuple) else output
            if not isinstance(batch_probabilities, torch.Tensor):
                raise ValueError("Encoder did not return a tensor of probabilities")
            batch_array = batch_probabilities.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
            expected_shape = (len(names), fingerprint_bits)
            if batch_array.shape != expected_shape:
                raise ValueError(
                    f"Encoder output has shape {batch_array.shape}; expected {expected_shape}"
                )
            if not np.isfinite(batch_array).all():
                raise ValueError(f"Encoder returned non-finite probabilities at rows {row_offset}:{stop}")
            if np.any(batch_array < 0.0) or np.any(batch_array > 1.0):
                raise ValueError(f"Encoder probabilities fall outside [0, 1] at rows {row_offset}:{stop}")
            probabilities[row_offset:stop] = batch_array
            inference_seconds[row_offset:stop] = elapsed / len(names)
            aggregate_seconds += elapsed
            row_offset = stop
            if progress is not None:
                progress(row_offset, len(expected_ids))
    if row_offset != len(expected_ids):
        raise ValueError(f"Encoder produced {row_offset} rows; expected {len(expected_ids)}")
    return probabilities, inference_seconds, aggregate_seconds


def import_diffms_components(diffms_repo: Path) -> tuple[Any, Any, Any]:
    """Import MIST classes from the selected immutable DiffMS checkout."""

    source_root = (diffms_repo / "src").resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"DiffMS source directory does not exist: {source_root}")
    sys.path.insert(0, str(source_root))
    import mist  # type: ignore[import-not-found]
    from mist.data.data import Spectra  # type: ignore[import-not-found]
    from mist.data.featurizers import PeakFormula  # type: ignore[import-not-found]
    from mist.models.spectra_encoder import (  # type: ignore[import-not-found]
        SpectraEncoderGrowing,
    )

    module_roots = [Path(value).resolve() for value in mist.__path__]
    expected_module_root = source_root / "mist"
    if expected_module_root not in module_roots:
        raise RuntimeError(
            f"Imported mist from {module_roots}, outside selected DiffMS source {source_root}"
        )
    return Spectra, PeakFormula, SpectraEncoderGrowing


def build_encoder(model_class: Any) -> nn.Module:
    """Instantiate the exact MSG encoder architecture released with DiffMS."""

    return model_class(
        inten_transform="float",
        inten_prob=0.1,
        remove_prob=0.5,
        peak_attn_layers=2,
        num_heads=8,
        pairwise_featurization=True,
        embed_instrument=False,
        cls_type="ms1",
        set_pooling="cls",
        spec_features="peakformula",
        mol_features="fingerprint",
        form_embedder="pos-cos",
        output_size=FINGERPRINT_BITS,
        hidden_size=512,
        spectra_dropout=0.1,
        top_layers=1,
        refine_layers=4,
        magma_modulo=2048,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_rows <= 0:
        raise ValueError("expected-rows must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("max-rows must be positive when supplied")
    if args.batch_size <= 0 or args.num_workers < 0 or args.warmup_batches < 0:
        raise ValueError("batch-size must be positive; worker and warmup counts must be non-negative")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    diffms_repo = Path(args.diffms_repo).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    reference_metadata = Path(args.reference_metadata).resolve()
    labels_path = Path(args.labels).resolve()
    split_path = Path(args.split_file).resolve()
    spectra_dir = Path(args.spectra_dir).resolve()
    subformula_dir = Path(args.subformula_dir).resolve()

    wall_started = time.perf_counter()
    rows, subformula_manifest_sha256 = load_locked_rows(
        reference_metadata,
        labels_path,
        split_path,
        spectra_dir,
        subformula_dir,
        id_column=args.id_column,
        index_column=args.index_column,
        expected_split=args.expected_split,
        expected_rows=args.expected_rows,
    )
    source_rows = len(rows)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    expected_ids = [row.spectrum_id for row in rows]

    Spectra, PeakFormula, SpectraEncoderGrowing = import_diffms_components(diffms_repo)
    featurizer = PeakFormula(
        subform_folder=str(subformula_dir),
        augment_data=False,
        remove_prob=0.1,
        remove_weights="exp",
        inten_prob=0.1,
        cls_type="ms1",
        magma_aux_loss=False,
        inten_transform="float",
        magma_modulo=2048,
        cache_featurizers=False,
    )
    spectra = [
        Spectra(
            spectra_name=row.spectrum_id,
            spectra_file=str(row.spectrum_path),
            spectra_formula=row.formula,
            instrument=row.instrument,
        )
        for row in rows
    ]
    dataset = SpectrumOnlyDataset(spectra, featurizer)
    device = resolve_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=PeakFormula.collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder_state = extract_encoder_state(checkpoint)
    released_encoder_identity = None
    if args.released_checkpoint_archive is not None:
        archive_path = Path(args.released_checkpoint_archive).resolve()
        standalone_state, standalone_sha256, standalone_size = (
            load_released_encoder_member(archive_path, args.released_encoder_member)
        )
        if set(standalone_state) != set(encoder_state) or any(
            not torch.equal(standalone_state[key], encoder_state[key])
            for key in encoder_state
        ):
            raise ValueError(
                "Full DiffMS checkpoint encoder does not exactly match the released "
                f"standalone member {args.released_encoder_member!r}"
            )
        released_encoder_identity = {
            "archive": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "member": args.released_encoder_member,
            "member_size_bytes": standalone_size,
            "member_sha256": standalone_sha256,
            "all_tensors_exact_match": True,
        }
    encoder = build_encoder(SpectraEncoderGrowing)
    load_result = encoder.load_state_dict(encoder_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Strict encoder load failed: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    encoder = encoder.to(device)

    last_reported = -1

    def report_progress(done: int, total: int) -> None:
        nonlocal last_reported
        percent = int(100 * done / total)
        if percent >= last_reported + 5 or done == total:
            print(f"Processed {done}/{total} spectra ({percent}%)", flush=True)
            last_reported = percent

    probabilities, inference_seconds, aggregate_inference_seconds = run_inference(
        encoder,
        loader,
        expected_ids,
        device=device,
        warmup_batches=args.warmup_batches,
        progress=report_progress,
    )

    predictions_path = output_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        probs=probabilities,
        spectrum_ids=np.asarray(expected_ids, dtype=str),
        inference_seconds=inference_seconds,
    )
    wall_seconds = time.perf_counter() - wall_started
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "candidate": {
            "name": "diffms_released_mist_msg_512",
            "scope": "released pretrained MIST spectrum encoder from DiffMS",
            "is_end_to_end_diffms_gain": False,
            "native_output": "Morgan-4096 probabilities",
        },
        "rows": len(expected_ids),
        "source_rows": source_rows,
        "is_smoke": args.max_rows is not None,
        "fingerprint_bits": FINGERPRINT_BITS,
        "ordered_spectrum_ids_sha256": ordered_ids_sha256(expected_ids),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "encoder_prefix": ENCODER_PREFIX,
            "encoder_state_sha256": state_dict_sha256(encoder_state),
            "encoder_tensor_count": len(encoder_state),
            "encoder_parameter_count": int(sum(value.numel() for value in encoder_state.values())),
            "strict_load_missing_keys": list(load_result.missing_keys),
            "strict_load_unexpected_keys": list(load_result.unexpected_keys),
            "released_standalone_encoder": released_encoder_identity,
        },
        "inputs": {
            "reference_metadata": str(reference_metadata),
            "reference_metadata_sha256": sha256_file(reference_metadata),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "split_file": str(split_path),
            "split_file_sha256": sha256_file(split_path),
            "expected_split": args.expected_split,
            "spectra_dir": str(spectra_dir),
            "subformula_dir": str(subformula_dir),
            "locked_subformula_manifest_sha256": subformula_manifest_sha256,
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "warmup_batches": args.warmup_batches,
            "timing_method": (
                "synchronized encoder forward only; excludes featurization, host-to-device "
                "transfer, device-to-host transfer, and NPZ serialization"
            ),
            "aggregate_inference_seconds": aggregate_inference_seconds,
            "mean_inference_seconds": float(inference_seconds.mean()),
            "p95_inference_seconds": float(np.quantile(inference_seconds, 0.95)),
            "wall_seconds": wall_seconds,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "diffms_repo": str(diffms_repo),
            "diffms_commit": git_commit(diffms_repo),
            "code_revision": args.code_revision,
            "exporter": str(script_path),
            "exporter_sha256": sha256_file(script_path),
        },
        "outputs": {
            "predictions": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
