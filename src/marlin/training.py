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

from dlm.utils.utils_chem import safe_to_smiles, smiles_to_safe
from marlin.ema import AllParameterExponentialMovingAverage
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
            provided_fingerprint = example.get("fingerprint")
            if provided_fingerprint is None:
                fingerprint = AllChem.GetMorganGenerator(
                    radius=2, fpSize=self.fingerprint_bits
                ).GetFingerprint(molecule)
                array = np.zeros(self.fingerprint_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fingerprint, array)
            else:
                array = np.asarray(provided_fingerprint, dtype=np.float32)
                if array.shape != (self.fingerprint_bits,):
                    raise ValueError(
                        "provided fingerprint must have shape "
                        f"({self.fingerprint_bits},), got {array.shape}"
                    )
                if not np.array_equal(array, array.astype(bool)):
                    raise ValueError("provided fingerprint must be binary")
            safes.append(safe)
            fingerprints.append(torch.from_numpy(array))
            masses.append(
                float(example.get("precursor_mass", Descriptors.ExactMolWt(molecule)))
            )
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


class MarlinMetadataDataset(torch.utils.data.Dataset):
    """Finite, filtered molecular table used by held-out audits and training."""

    def __init__(
        self,
        metadata_csv: str | Path,
        tokenizer,
        *,
        max_length: int,
        exclude_inchikeys: str | Path | None = None,
    ) -> None:
        table = pd.read_csv(metadata_csv)
        excluded = load_excluded_connectivity_keys(exclude_inchikeys)
        rows = []
        for record in table.to_dict("records"):
            smiles = str(record.get("smiles", ""))
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                continue
            key = Chem.MolToInchiKey(molecule).split("-")[0]
            if key in excluded:
                continue
            safe = smiles_to_safe(smiles)
            if len(tokenizer.encode(safe, add_special_tokens=True)) <= max_length:
                rows.append({"safe": safe})
        if not rows:
            raise ValueError(f"no MARLIN-compatible rows in {metadata_csv}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


class MarlinSpectrumFingerprintDataset(torch.utils.data.Dataset):
    """Finite spectrum-to-molecule adaptation set with predicted fingerprints."""

    def __init__(
        self,
        metadata_csv: str | Path,
        fingerprint_npz: str | Path,
        tokenizer,
        *,
        fingerprint_key: str,
        threshold: float,
        max_length: int,
        exclude_inchikeys: str | Path | None = None,
        exclude_metadata_csvs: tuple[str | Path, ...] = (),
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("fingerprint threshold must be in [0, 1]")
        table = pd.read_csv(metadata_csv)
        required_columns = {
            "spec_name",
            "smiles",
            "inchikey_first_block",
            "neutral_mass",
        }
        missing_columns = sorted(required_columns - set(table.columns))
        if missing_columns:
            raise ValueError(
                f"spectrum metadata is missing columns: {missing_columns}"
            )

        excluded = load_excluded_connectivity_keys(exclude_inchikeys)
        for path in exclude_metadata_csvs:
            excluded.update(
                pd.read_csv(path)["inchikey_first_block"].dropna().astype(str)
            )

        with np.load(fingerprint_npz, allow_pickle=False) as arrays:
            if fingerprint_key not in arrays:
                raise KeyError(
                    f"{fingerprint_key!r} not found in {fingerprint_npz}; "
                    f"keys={arrays.files}"
                )
            values = np.asarray(arrays[fingerprint_key])
            if "spectrum_ids" not in arrays:
                raise ValueError(
                    "predicted fingerprint bundle must contain spectrum_ids"
                )
            spectrum_ids = [str(value) for value in arrays["spectrum_ids"]]
        if values.ndim != 2 or values.shape[1] != 4096:
            raise ValueError(
                "predicted fingerprints must have shape [rows, 4096], "
                f"got {values.shape}"
            )
        if len(spectrum_ids) != len(values) or len(set(spectrum_ids)) != len(
            spectrum_ids
        ):
            raise ValueError(
                "predicted fingerprint spectrum_ids are missing or duplicated"
            )
        positions = {value: index for index, value in enumerate(spectrum_ids)}

        rows = []
        for record in table.to_dict("records"):
            key = str(record["inchikey_first_block"])
            if key in excluded:
                continue
            spec_name = str(record["spec_name"])
            if spec_name not in positions:
                raise ValueError(
                    f"predicted fingerprint bundle is missing {spec_name}"
                )
            smiles = str(record["smiles"])
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            safe = smiles_to_safe(smiles)
            if len(tokenizer.encode(safe, add_special_tokens=True)) > max_length:
                continue
            rows.append(
                {
                    "safe": safe,
                    "fingerprint": (
                        values[positions[spec_name]] >= threshold
                    ).astype(np.float32),
                    "precursor_mass": float(record["neutral_mass"]),
                    "spec_name": spec_name,
                    "inchikey_first_block": key,
                }
            )
        if not rows:
            raise ValueError(
                f"no MARLIN-compatible spectrum rows in {metadata_csv}"
            )
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]


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
        eos_loss_weight: float = 1.0,
        eos_mask_probability: float = 0.0,
        balanced_token_loss_alpha: float = 0.0,
        token_loss_weight_max: float = 20.0,
        full_sequence_mask_probability: float = 0.0,
        conditioning_only_steps: int = 0,
        cross_attention_only_steps: int = 0,
        adapt_fingerprint: bool = False,
    ) -> None:
        super().__init__()
        if conditioning_only_steps < 0 or cross_attention_only_steps < 0:
            raise ValueError("adaptation stage durations must be non-negative")
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
                "eos_loss_weight": eos_loss_weight,
                "eos_mask_probability": eos_mask_probability,
                "balanced_token_loss_alpha": balanced_token_loss_alpha,
                "token_loss_weight_max": token_loss_weight_max,
                "full_sequence_mask_probability": full_sequence_mask_probability,
                "conditioning_only_steps": conditioning_only_steps,
                "cross_attention_only_steps": cross_attention_only_steps,
                "adapt_fingerprint": adapt_fingerprint,
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
        self.eos_loss_weight = eos_loss_weight
        self.eos_mask_probability = eos_mask_probability
        self.balanced_token_loss_alpha = balanced_token_loss_alpha
        self.token_loss_weight_max = token_loss_weight_max
        self.full_sequence_mask_probability = full_sequence_mask_probability
        self.conditioning_only_steps = conditioning_only_steps
        self.cross_attention_only_steps = cross_attention_only_steps
        self.adapt_fingerprint = adapt_fingerprint
        self._last_metric_step = -1
        self._active_adaptation_stage: str | None = None
        self._trainable_parameter_fraction = 1.0
        self.ema = AllParameterExponentialMovingAverage(
            self.decoder.parameters(), decay=ema_decay, use_num_updates=False
        )

    def reset_ema(self) -> None:
        """Reset EMA after loading warm-start weights."""
        self.ema = AllParameterExponentialMovingAverage(
            self.decoder.parameters(), decay=self.ema_decay, use_num_updates=False
        )

    def adaptation_stage(self, step: int) -> str:
        """Return the staged FRIGID adaptation phase for an optimizer step."""
        if step < self.conditioning_only_steps:
            return "conditioning"
        if step < (
            self.conditioning_only_steps + self.cross_attention_only_steps
        ):
            return "cross_attention"
        return "full"

    def _stage_parameter_is_trainable(self, name: str, stage: str) -> bool:
        conditioning = (
            name.startswith("conditioner.mass.")
            or name.startswith("conditioner.isotope.")
            or (
                self.adapt_fingerprint
                and name.startswith("conditioner.fingerprint.")
            )
        )
        if stage == "conditioning":
            return conditioning
        if stage == "cross_attention":
            return (
                conditioning
                or ".cross_attention." in name
                or ".norm2." in name
            )
        if stage == "full":
            return True
        raise ValueError(f"unknown adaptation stage: {stage}")

    def apply_adaptation_stage(self, step: int) -> str:
        """Freeze or unfreeze decoder parameters without changing EMA order."""
        stage = self.adaptation_stage(step)
        if stage == self._active_adaptation_stage:
            return stage
        trainable = 0
        total = 0
        for name, parameter in self.decoder.named_parameters():
            parameter.requires_grad_(
                self._stage_parameter_is_trainable(name, stage)
            )
            count = parameter.numel()
            total += count
            if parameter.requires_grad:
                trainable += count
        self._active_adaptation_stage = stage
        self._trainable_parameter_fraction = trainable / total
        print(
            "MARLIN adaptation stage "
            f"{stage!r} at step {step}: "
            f"{trainable}/{total} trainable parameters "
            f"({self._trainable_parameter_fraction:.6f})",
            flush=True,
        )
        return stage

    def on_train_batch_start(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        self.apply_adaptation_stage(int(self.global_step))

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
            eos_loss_weight=self.eos_loss_weight,
            eos_mask_probability=self.eos_mask_probability,
            balanced_token_loss_alpha=self.balanced_token_loss_alpha,
            token_loss_weight_max=self.token_loss_weight_max,
            full_sequence_mask_probability=self.full_sequence_mask_probability,
            collect_metrics=collect_metrics,
        )
        if collect_metrics:
            self._last_metric_step = self.global_step
            stage_codes = {
                "conditioning": 0.0,
                "cross_attention": 1.0,
                "full": 2.0,
            }
            stage = self._active_adaptation_stage or self.adaptation_stage(
                int(self.global_step)
            )
            self.log(
                "adaptation_stage",
                stage_codes[stage],
                on_step=True,
                sync_dist=True,
            )
            self.log(
                "trainable_parameter_fraction",
                self._trainable_parameter_fraction,
                on_step=True,
                sync_dist=True,
            )
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
        self.apply_adaptation_stage(int(self.global_step))
        self.ema.move_shadow_params_to_device(self.device)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.decoder.parameters())

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
