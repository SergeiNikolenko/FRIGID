"""Training utilities for the clean-room MARLIN decoder."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from dlm.utils.utils_chem import safe_to_smiles
from dlm.utils.ema import ExponentialMovingAverage
from marlin.model import MarlinDecoder, MarlinDecoderConfig
from marlin.noise import symmetric_fingerprint_noise


class MarlinCollator:
    def __init__(
        self,
        tokenizer,
        *,
        max_length: int = 256,
        fingerprint_bits: int = 4096,
        exclude_inchikeys: str | Path | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fingerprint_bits = fingerprint_bits
        self.exclude = set()
        if exclude_inchikeys:
            table = pd.read_csv(exclude_inchikeys)
            column = "inchi" if "inchi" in table else "inchikey"
            self.exclude = {str(value).split("-")[0] for value in table[column].dropna()}

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        safes: list[str] = []
        fingerprints: list[torch.Tensor] = []
        masses: list[float] = []
        for example in examples:
            safe = example.get("safe", example.get("input"))
            if not safe:
                continue
            smiles = safe_to_smiles(safe, fix=True)
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                continue
            key = Chem.MolToInchiKey(molecule).split("-")[0]
            if key in self.exclude:
                continue
            fingerprint = AllChem.GetMorganGenerator(
                radius=2, fpSize=self.fingerprint_bits
            ).GetFingerprint(molecule)
            array = np.zeros(self.fingerprint_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fingerprint, array)
            safes.append(safe)
            fingerprints.append(torch.from_numpy(array))
            masses.append(Descriptors.ExactMolWt(molecule))
        if not safes:
            raise ValueError("batch has no valid non-excluded molecules")
        tokens = self.tokenizer(
            safes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": tokens["input_ids"],
            "fingerprint": torch.stack(fingerprints),
            "precursor_mass": torch.tensor(masses, dtype=torch.float32),
        }


class MarlinLightningModule(L.LightningModule):
    def __init__(
        self,
        config: MarlinDecoderConfig,
        *,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        noise_probability: float = 0.5,
        noise_min_fraction: float = 0.1,
        noise_max_fraction: float = 0.3,
        ema_decay: float = 0.9999,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "config": asdict(config),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "noise_probability": noise_probability,
                "noise_min_fraction": noise_min_fraction,
                "noise_max_fraction": noise_max_fraction,
                "ema_decay": ema_decay,
            }
        )
        self.decoder = MarlinDecoder(config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.noise_probability = noise_probability
        self.noise_min_fraction = noise_min_fraction
        self.noise_max_fraction = noise_max_fraction
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.decoder.parameters(), decay=ema_decay)

    def reset_ema(self) -> None:
        """Reset EMA after loading warm-start weights."""
        self.ema = ExponentialMovingAverage(self.decoder.parameters(), decay=self.ema_decay)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        fingerprint = symmetric_fingerprint_noise(
            batch["fingerprint"],
            corruption_probability=self.noise_probability,
            min_fraction=self.noise_min_fraction,
            max_fraction=self.noise_max_fraction,
        )
        loss = self.decoder.diffusion_loss(
            batch["input_ids"], batch["precursor_mass"], fingerprint
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def on_train_start(self) -> None:
        self.ema.move_shadow_params_to_device(self.device)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.decoder.parameters())

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
