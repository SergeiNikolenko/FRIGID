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
from marlin.isotopes import theoretical_isotope_ratios


def load_excluded_connectivity_keys(path: str | Path | None) -> set[str]:
    """Load first-block InChIKeys used to keep evaluation structures out."""

    if path is None:
        return set()
    table = pd.read_csv(path)
    column = "inchi" if "inchi" in table else "inchikey"
    return {str(value).split("-")[0] for value in table[column].dropna()}


class MarlinTrainingFilter:
    """Pickleable pre-batch filter for the canonical MARLIN stream."""

    def __init__(
        self,
        tokenizer,
        max_length: int,
        exclude_inchikeys: str | Path | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.exclude = load_excluded_connectivity_keys(exclude_inchikeys)

    def __call__(self, example: dict) -> bool:
        safe = example.get("safe", example.get("input"))
        if not safe:
            return False
        if len(self.tokenizer.encode(safe, add_special_tokens=True)) > self.max_length:
            return False
        smiles = safe_to_smiles(safe, fix=False)
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            return False
        key = Chem.MolToInchiKey(molecule).split("-")[0]
        return key not in self.exclude


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
        self.exclude = load_excluded_connectivity_keys(exclude_inchikeys)

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        safes: list[str] = []
        fingerprints: list[torch.Tensor] = []
        masses: list[float] = []
        isotope_ratios: list[torch.Tensor] = []
        for example in examples:
            safe = example.get("safe", example.get("input"))
            if not safe:
                raise ValueError("pre-batch filter admitted an empty SAFE sequence")
            smiles = safe_to_smiles(safe, fix=False)
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                raise ValueError("pre-batch filter admitted an invalid SAFE sequence")
            key = Chem.MolToInchiKey(molecule).split("-")[0]
            if key in self.exclude:
                raise ValueError("pre-batch filter admitted an excluded test structure")
            fingerprint = AllChem.GetMorganGenerator(
                radius=2, fpSize=self.fingerprint_bits
            ).GetFingerprint(molecule)
            array = np.zeros(self.fingerprint_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fingerprint, array)
            safes.append(safe)
            fingerprints.append(torch.from_numpy(array))
            masses.append(Descriptors.ExactMolWt(molecule))
            isotope_ratios.append(theoretical_isotope_ratios(molecule))
        if len(safes) != len(examples):
            raise AssertionError("MARLIN collator changed the pre-filtered batch size")
        tokens = self.tokenizer(
            safes,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        sequence_length = int(tokens["input_ids"].shape[1])
        if sequence_length > self.max_length:
            raise ValueError(
                "SAFE token length exceeds the decoder maximum: "
                f"batch_max={sequence_length}, model_max={self.max_length}; "
                "refusing silent truncation"
            )
        return {
            "input_ids": tokens["input_ids"],
            "fingerprint": torch.stack(fingerprints),
            "precursor_mass": torch.tensor(masses, dtype=torch.float32),
            "isotope_ratios": torch.stack(isotope_ratios),
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
        metric_interval: int = 50,
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
                "metric_interval": metric_interval,
            }
        )
        self.decoder = MarlinDecoder(config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.noise_probability = noise_probability
        self.noise_min_fraction = noise_min_fraction
        self.noise_max_fraction = noise_max_fraction
        self.ema_decay = ema_decay
        self.metric_interval = metric_interval
        self._last_metric_step = -1
        self.ema = ExponentialMovingAverage(
            self.decoder.parameters(), decay=ema_decay, use_num_updates=False
        )

    def reset_ema(self) -> None:
        """Reset EMA after loading warm-start weights."""
        self.ema = ExponentialMovingAverage(
            self.decoder.parameters(), decay=self.ema_decay, use_num_updates=False
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        fingerprint = symmetric_fingerprint_noise(
            batch["fingerprint"],
            corruption_probability=self.noise_probability,
            min_fraction=self.noise_min_fraction,
            max_fraction=self.noise_max_fraction,
        )
        collect_metrics = (
            self.global_step % self.metric_interval == 0
            and self.global_step != self._last_metric_step
        )
        loss, reconstruction_metrics = self.decoder.diffusion_objective(
            batch["input_ids"],
            batch["precursor_mass"],
            fingerprint,
            isotope_ratios=batch["isotope_ratios"],
            collect_metrics=collect_metrics,
        )
        if collect_metrics:
            self._last_metric_step = self.global_step
            for name, value in reconstruction_metrics.items():
                self.log(
                    f"train_{name}",
                    value,
                    on_step=True,
                    sync_dist=True,
                )
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log(
            "fingerprint_noise_fraction",
            (fingerprint != batch["fingerprint"]).float().mean(),
            on_step=True,
            sync_dist=True,
        )
        optimizer = self.optimizers()
        self.log(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "micro_batch_size",
            float(batch["input_ids"].shape[0]),
            on_step=True,
            sync_dist=True,
        )
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.global_step % 50 != 0:
            return
        parameter_norms = [
            parameter.grad.detach().norm(2)
            for parameter in self.parameters()
            if parameter.grad is not None
        ]
        if parameter_norms:
            grad_norm = torch.stack(parameter_norms).norm(2)
            self.log("grad_norm", grad_norm, on_step=True, sync_dist=True)

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
