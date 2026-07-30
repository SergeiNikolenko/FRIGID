"""Fingerprint encoder backends for spectrum-to-molecule benchmarks.

The benchmark code historically called MIST directly. This module keeps that
path intact while adding a small contract for alternative spectrum encoders that
produce 4096-bit Morgan fingerprint probabilities, such as a DreaMS embedding
model followed by a supervised fingerprint head.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from mist.models import SpectraEncoderGrowing


class FingerprintBackend:
    """Base interface for spectrum-to-fingerprint backends."""

    name = "base"

    def predict_probs(
        self,
        batch: Dict[str, Any],
        *,
        spec_name: Optional[str] = None,
    ) -> np.ndarray:
        """Return one fingerprint probability vector for a benchmark sample."""
        raise NotImplementedError


class MistFingerprintBackend(FingerprintBackend):
    """MIST encoder backend returning raw fingerprint probabilities."""

    name = "mist"

    def __init__(self, encoder: torch.nn.Module, device: torch.device):
        self.encoder = encoder
        self.device = device

    def predict_probs(
        self,
        batch: Dict[str, Any],
        *,
        spec_name: Optional[str] = None,
    ) -> np.ndarray:
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            pred_probs, _ = self.encoder(batch)
        return pred_probs.detach().cpu().numpy()[0].astype(np.float32)


@dataclass
class PrecomputedFingerprintBackend(FingerprintBackend):
    """Backend for probabilities produced offline by DreaMS or another encoder.

    The metadata CSV must contain a spectrum identifier column, usually
    ``spec_name``. The NPZ array must have the same row order as the metadata.
    """

    npz_path: str
    metadata_path: str
    probability_key: str = "probs"
    spec_name_column: str = "spec_name"
    name: str = "precomputed"

    def __post_init__(self):
        metadata = pd.read_csv(self.metadata_path)
        if self.spec_name_column not in metadata.columns:
            raise ValueError(
                f"Missing {self.spec_name_column!r} column in {self.metadata_path}"
            )

        arrays = np.load(self.npz_path)
        if self.probability_key not in arrays:
            keys = ", ".join(arrays.files)
            raise ValueError(
                f"Missing array {self.probability_key!r} in {self.npz_path}; "
                f"available arrays: {keys}"
            )

        probs = np.asarray(arrays[self.probability_key], dtype=np.float32)
        if probs.ndim != 2:
            raise ValueError(
                f"Expected a 2D probability array, got shape {probs.shape}"
            )
        if len(metadata) != len(probs):
            raise ValueError(
                f"Metadata rows ({len(metadata)}) do not match probability rows "
                f"({len(probs)})"
            )

        self._probs = probs
        self._index_by_spec_name = {
            str(spec_name): i
            for i, spec_name in enumerate(metadata[self.spec_name_column])
        }

    def predict_probs(
        self,
        batch: Dict[str, Any],
        *,
        spec_name: Optional[str] = None,
    ) -> np.ndarray:
        if spec_name is None:
            raise ValueError("Precomputed fingerprint backend requires spec_name")
        try:
            index = self._index_by_spec_name[str(spec_name)]
        except KeyError as exc:
            raise KeyError(
                f"Spectrum {spec_name!r} was not found in {self.metadata_path}"
            ) from exc
        return self._probs[index]


def load_mist_encoder(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Load a MIST encoder from the existing FRIGID config shape."""
    checkpoint_path = config["checkpoint"]
    print(f"\nLoading MIST encoder from: {checkpoint_path}")

    encoder = SpectraEncoderGrowing(
        form_embedder=config.get("form_embedder", "pos-cos"),
        output_size=config.get("output_size", 4096),
        hidden_size=config.get("hidden_size", 512),
        spectra_dropout=config.get("spectra_dropout", 0.1),
        peak_attn_layers=config.get("peak_attn_layers", 2),
        num_heads=config.get("num_heads", 8),
        set_pooling=config.get("set_pooling", "cls"),
        refine_layers=config.get("refine_layers", 4),
        pairwise_featurization=config.get("pairwise_featurization", True),
        embed_instrument=config.get("embed_instrument", False),
        inten_transform=config.get("inten_transform", "float"),
        magma_modulo=config.get("magma_modulo", 2048),
        inten_prob=config.get("inten_prob", 0.1),
        remove_prob=config.get("remove_prob", 0.5),
        cls_type=config.get("cls_type", "ms1"),
        spec_features=config.get("spec_features", "peakformula"),
        mol_features=config.get("mol_features", "fingerprint"),
        top_layers=config.get("top_layers", 1),
    )

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            state_dict = {
                key.replace("encoder.", ""): value
                for key, value in state_dict.items()
                if key.startswith("encoder.")
            }
            if not state_dict:
                state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        encoder.load_state_dict(state_dict, strict=False)
        print("Loaded MIST encoder weights")
    else:
        print(f"WARNING: MIST checkpoint not found: {checkpoint_path}")

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def build_fingerprint_backend(
    config: Dict[str, Any],
    device: torch.device,
) -> FingerprintBackend:
    """Build the configured fingerprint backend.

    Config shape:

    ``encoder_backend.name: mist`` keeps the historical behavior.
    ``encoder_backend.name: precomputed`` reads probabilities from an NPZ/CSV
    pair, which is the handoff format emitted by the DreaMS fingerprint-head
    scripts.
    """
    backend_cfg = config.get("encoder_backend", {}) or {}
    backend_name = backend_cfg.get("name", "mist")

    if backend_name == "mist":
        return MistFingerprintBackend(
            load_mist_encoder(config["mist_encoder"], device),
            device,
        )

    if backend_name in {"precomputed", "dreams_precomputed"}:
        return PrecomputedFingerprintBackend(
            npz_path=backend_cfg["fingerprint_npz"],
            metadata_path=backend_cfg["metadata_csv"],
            probability_key=backend_cfg.get("probability_key", "probs"),
            spec_name_column=backend_cfg.get("spec_name_column", "spec_name"),
            name=backend_name,
        )

    raise ValueError(f"Unknown encoder backend: {backend_name}")
