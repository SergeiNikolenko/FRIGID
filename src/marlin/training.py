"""Training utilities for the clean-room MARLIN decoder."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from dlm.utils.utils_chem import safe_to_smiles, smiles_to_safe
from dlm.utils.ema import ExponentialMovingAverage
from marlin.grammar import SafeGrammarMask
from marlin.mass_shell import MassShellConstraint
from marlin.model import MarlinDecoder, MarlinDecoderConfig
from marlin.noise import symmetric_fingerprint_noise
from marlin.isotopes import theoretical_isotope_ratios
from marlin.sampler import MarlinSampler
from marlin.token_properties import build_token_property_table


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
            if not safe and example.get("smiles"):
                safe = smiles_to_safe(str(example["smiles"]))
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


class MarlinMetadataDataset(torch.utils.data.Dataset):
    """Finite local metadata table for short NPLIB domain fine-tuning."""

    def __init__(
        self,
        metadata_csv: str | Path,
        tokenizer,
        *,
        max_length: int,
        exclude_inchikeys: str | Path | None = None,
    ) -> None:
        table = pd.read_csv(metadata_csv)
        exclude = load_excluded_connectivity_keys(exclude_inchikeys)
        rows = []
        for record in table.to_dict("records"):
            smiles = str(record.get("smiles", ""))
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                continue
            key = Chem.MolToInchiKey(molecule).split("-")[0]
            if key in exclude:
                continue
            safe = smiles_to_safe(smiles)
            if len(tokenizer.encode(safe, add_special_tokens=True)) > max_length:
                continue
            rows.append({"safe": safe})
        if not rows:
            raise ValueError(f"no MARLIN-compatible rows in {metadata_csv}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
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
                "eos_loss_weight": eos_loss_weight,
                "eos_mask_probability": eos_mask_probability,
                "balanced_token_loss_alpha": balanced_token_loss_alpha,
                "token_loss_weight_max": token_loss_weight_max,
                "full_sequence_mask_probability": full_sequence_mask_probability,
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
            eos_loss_weight=self.eos_loss_weight,
            eos_mask_probability=self.eos_mask_probability,
            balanced_token_loss_alpha=self.balanced_token_loss_alpha,
            token_loss_weight_max=self.token_loss_weight_max,
            full_sequence_mask_probability=self.full_sequence_mask_probability,
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


class MarlinMolecularValidationCallback(L.Callback):
    """Log bounded oracle-conditioned molecular generation metrics during training."""

    def __init__(
        self,
        tokenizer,
        metadata_csv: str | Path,
        *,
        output_dir: str | Path,
        fingerprint_bits: int = 4096,
        every_n_steps: int = 500,
        samples: int = 2,
        candidates: int = 4,
        temperature: float = 1.0,
    ) -> None:
        if every_n_steps <= 0:
            raise ValueError("molecular validation interval must be positive")
        if samples <= 0 or candidates <= 0:
            raise ValueError(
                "molecular validation samples and candidates must be positive"
            )
        if temperature <= 0:
            raise ValueError("molecular validation temperature must be positive")

        table = pd.read_csv(metadata_csv)
        if "smiles" not in table:
            raise ValueError("molecular validation CSV must contain a smiles column")
        records = []
        fingerprint_generator = AllChem.GetMorganGenerator(
            radius=2, fpSize=fingerprint_bits
        )
        for smiles in table["smiles"].dropna().astype(str):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            fingerprint = fingerprint_generator.GetFingerprint(molecule)
            array = np.zeros(fingerprint_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fingerprint, array)
            records.append(
                {
                    "smiles": Chem.MolToSmiles(molecule, canonical=True),
                    "connectivity": Chem.MolToInchiKey(molecule).split("-")[0],
                    "fingerprint": torch.from_numpy(array),
                    "mass": float(Descriptors.ExactMolWt(molecule)),
                }
            )
            if len(records) >= samples:
                break
        if not records:
            raise ValueError("molecular validation CSV contains no valid molecules")

        self.tokenizer = tokenizer
        self.records = records
        self.output_path = Path(output_dir) / "molecular_validation.jsonl"
        self.every_n_steps = every_n_steps
        self.candidates = candidates
        self.temperature = temperature
        self._last_step = -1

        special_ids = {
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
        }
        token_masses, token_atoms, token_valences = build_token_property_table(
            len(tokenizer), tokenizer.convert_ids_to_tokens, special_ids
        )
        self.constraint = MassShellConstraint(
            token_masses,
            token_atoms,
            token_valences,
            eos_token_id=tokenizer.eos_token_id,
            ppm_tolerance=10.0,
            valence_slack=4.0,
            eos_boost=1.0,
        )
        self.grammar = SafeGrammarMask(
            [
                tokenizer.convert_ids_to_tokens(index)
                for index in range(len(tokenizer))
            ],
            lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            eos_token_id=tokenizer.eos_token_id,
            mask_token_id=tokenizer.mask_token_id,
            special_token_ids=tuple(special_ids) + (tokenizer.unk_token_id,),
            ppm_tolerance=10.0,
            valence_slack=4.0,
        )

    def _sampler(self, model: MarlinDecoder) -> MarlinSampler:
        return MarlinSampler(
            model,
            self.constraint,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
            decode_tokens=lambda ids: self.tokenizer.decode(
                ids, skip_special_tokens=True
            ),
            safe_to_smiles=lambda safe: safe_to_smiles(safe, fix=True),
            grammar_mask=self.grammar,
            forbidden_token_ids=(
                self.tokenizer.unk_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.mask_token_id,
                self.tokenizer.pad_token_id,
            ),
            mass_shell_enabled=True,
            generation_mode="block",
        )

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: MarlinLightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        step = int(trainer.global_step)
        if (
            not trainer.is_global_zero
            or step <= 0
            or step % self.every_n_steps
            or step == self._last_step
        ):
            return
        self._last_step = step

        parameters = [
            parameter
            for parameter in pl_module.decoder.parameters()
            if parameter.requires_grad
        ]
        was_training = pl_module.decoder.training
        pl_module.ema.store(parameters)
        pl_module.ema.copy_to(parameters)
        pl_module.decoder.eval()
        try:
            sampler = self._sampler(pl_module.decoder)
            attempts = valid = mass_valid = unique_mass_valid = returned = exact = 0
            top1_tanimoto = []
            sample_rows = []
            device = pl_module.device
            rng = torch.Generator(device=device).manual_seed(10_000 + step)
            for record in self.records:
                ranked, stats = sampler.generate_ranked_with_stats(
                    record["fingerprint"].to(device),
                    record["mass"],
                    candidates=self.candidates,
                    diversity_dropout=0.0,
                    temperature=self.temperature,
                    generator=rng,
                )
                attempts += stats.attempts
                valid += stats.valid
                mass_valid += stats.mass_valid
                unique_mass_valid += stats.unique_mass_valid
                returned += int(bool(ranked))
                if ranked:
                    top = ranked[0]
                    molecule = Chem.MolFromSmiles(top.smiles)
                    predicted_key = (
                        Chem.MolToInchiKey(molecule).split("-")[0]
                        if molecule is not None
                        else None
                    )
                    exact += int(predicted_key == record["connectivity"])
                    top1_tanimoto.append(top.tanimoto)
                else:
                    top1_tanimoto.append(0.0)
                sample_rows.append(
                    {
                        "target_smiles": record["smiles"],
                        "top1_smiles": ranked[0].smiles if ranked else None,
                        "valid": stats.valid,
                        "mass_valid": stats.mass_valid,
                    }
                )

            count = len(self.records)
            metrics = {
                "validity": valid / attempts,
                "mass_validity": mass_valid / attempts,
                "unique_mass_validity": unique_mass_valid / attempts,
                "candidate_return_rate": returned / count,
                "exact_top1": exact / count,
                "tanimoto_top1": float(np.mean(top1_tanimoto)),
            }
            for name, value in metrics.items():
                pl_module.log(
                    f"molecular_{name}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "metrics": metrics,
                            "samples": sample_rows,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        finally:
            pl_module.ema.restore(parameters)
            pl_module.decoder.train(was_training)
