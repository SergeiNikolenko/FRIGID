"""Training utilities for the clean-room MARLIN decoder."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors

from dlm.utils.utils_chem import safe_to_smiles, smiles_to_safe
from marlin.distillation import (
    ALL_DECODER_TRAINABLE_SCOPE,
    CYCLIC_ROLLOUT_PREFIX_SCHEDULE,
    FairBlockInputs,
    FrigidDistillationSettings,
    build_fair_block_inputs,
    expected_distillation_trainable_parameters,
    frigid_distillation_loss,
)
from marlin.ema import AllParameterExponentialMovingAverage
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
        include_formula: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fingerprint_bits = fingerprint_bits
        self.exclude = load_excluded_connectivity_keys(exclude_inchikeys)
        self.include_formula = include_formula

    def __call__(self, examples: list[dict]) -> dict[str, object]:
        safes: list[str] = []
        fingerprints: list[torch.Tensor] = []
        masses: list[float] = []
        isotope_ratios: list[torch.Tensor] = []
        formulas: list[str] = []
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
            if self.include_formula:
                formulas.append(rdMolDescriptors.CalcMolFormula(molecule))
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
        batch: dict[str, object] = {
            "input_ids": tokens["input_ids"],
            "fingerprint": torch.stack(fingerprints),
            "precursor_mass": torch.tensor(masses, dtype=torch.float32),
            "isotope_ratios": torch.stack(isotope_ratios),
        }
        if self.include_formula:
            batch["formula"] = formulas
        return batch


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
        distillation: FrigidDistillationSettings | None = None,
        frigid_teacher=None,
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
                "distillation": asdict(distillation) if distillation else None,
            }
        )
        self.decoder = MarlinDecoder(config)
        if (distillation is None) != (frigid_teacher is None):
            raise ValueError(
                "distillation settings and FRIGID teacher must be provided together"
            )
        self.distillation = distillation
        object.__setattr__(self, "_frigid_teacher", frigid_teacher)
        if distillation is not None:
            if (
                distillation.attention_mode == "block"
                and config.block_width != distillation.block_width_override
            ):
                raise ValueError(
                    "block distillation requires model.block_width to match "
                    "adaptation.block_width_override"
                )
            self._configure_distillation_trainability()
            self._set_distillation_training_modes()
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
        self._last_distillation_context_step = -1
        self.ema = AllParameterExponentialMovingAverage(
            self.decoder.parameters(), decay=ema_decay, use_num_updates=False
        )

    def reset_ema(self) -> None:
        """Reset EMA after loading warm-start weights."""
        self.ema = AllParameterExponentialMovingAverage(
            self.decoder.parameters(), decay=self.ema_decay, use_num_updates=False
        )

    def _configure_distillation_trainability(self) -> None:
        """Apply the explicit decoder parameter scope for this adaptation stage."""

        if self.distillation is None:
            return
        expected = expected_distillation_trainable_parameters(
            self.distillation.trainable_scope,
            num_layers=self.decoder.config.num_layers,
            fingerprint_self_attention_layers=(
                self.decoder.config.fingerprint_self_attention_layers
            ),
            frigid_compatible_layer_order=(
                self.decoder.config.frigid_compatible_layer_order
            ),
        )
        named_parameters = dict(self.decoder.named_parameters())
        missing = expected.difference(named_parameters)
        if missing:
            raise RuntimeError(
                "decoder does not expose the requested trainable parameters: "
                + ", ".join(sorted(missing))
            )
        if self.distillation.trainable_scope == ALL_DECODER_TRAINABLE_SCOPE:
            uncovered = set(named_parameters).difference(expected)
            if uncovered:
                raise RuntimeError(
                    "all_decoder scope does not cover decoder parameters: "
                    + ", ".join(sorted(uncovered))
                )
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(name in expected)
        actual = {
            name
            for name, parameter in named_parameters.items()
            if parameter.requires_grad
        }
        if actual != set(expected):
            raise RuntimeError("distillation trainable scope was not applied exactly")

    def _set_distillation_training_modes(self) -> None:
        """Keep frozen modules deterministic while training the selected scope."""

        if self.distillation is None:
            return
        # Evaluation mode disables dropout without disabling autograd. Exact
        # parameter trainability remains controlled solely by requires_grad.
        self.decoder.eval()

    def train(self, mode: bool = True):
        result = super().train(mode)
        if mode and getattr(self, "distillation", None) is not None:
            self._set_distillation_training_modes()
        return result

    @torch.no_grad()
    def _distillation_context_response_metrics(
        self,
        *,
        input_ids: torch.Tensor,
        fair: FairBlockInputs,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        formulas: list[str],
    ) -> dict[str, torch.Tensor]:
        """Measure whether revealed current-block tokens affect predictions."""

        settings = self.distillation
        teacher = object.__getattribute__(self, "_frigid_teacher")
        if settings is None or settings.attention_mode != "block" or teacher is None:
            return {}
        revealed = fair.current_content_mask & fair.input_ids.ne(
            self.decoder.config.mask_token_id
        )
        comparable_rows = revealed.any(dim=1) & fair.loss_mask.any(dim=1)
        comparison_mask = fair.loss_mask & comparable_rows.reshape(-1, 1)
        if not comparison_mask.any():
            zero = precursor_mass.new_zeros((), dtype=torch.float32)
            return {
                "distillation_student_context_logit_l1": zero,
                "distillation_teacher_context_logit_l1": zero,
                "distillation_context_response_ratio": zero,
                "distillation_student_context_argmax_change_fraction": zero,
                "distillation_teacher_context_argmax_change_fraction": zero,
                "distillation_context_tokens": zero,
            }

        fully_masked = fair.input_ids.masked_fill(
            fair.current_content_mask,
            self.decoder.config.mask_token_id,
        )
        was_training = self.decoder.training
        self.decoder.eval()
        try:
            partial_student = self.decoder.two_stream_logits(
                input_ids,
                fair.input_ids,
                precursor_mass,
                fingerprint,
                isotope_ratios=None,
            )
            fully_masked_student = self.decoder.two_stream_logits(
                input_ids,
                fully_masked,
                precursor_mass,
                fingerprint,
                isotope_ratios=None,
            )
            partial_teacher = teacher.logits(
                fair.input_ids,
                formulas,
                fingerprint,
            )
            fully_masked_teacher = teacher.logits(
                fully_masked,
                formulas,
                fingerprint,
            )
        finally:
            self.decoder.train(was_training)
            if was_training:
                self._set_distillation_training_modes()

        student_delta = (
            partial_student.float() - fully_masked_student.float()
        ).abs()[comparison_mask].mean()
        teacher_delta = (
            partial_teacher.float() - fully_masked_teacher.float()
        ).abs()[comparison_mask].mean()
        student_argmax_change = (
            partial_student.argmax(dim=-1)[comparison_mask]
            != fully_masked_student.argmax(dim=-1)[comparison_mask]
        ).float().mean()
        teacher_argmax_change = (
            partial_teacher.argmax(dim=-1)[comparison_mask]
            != fully_masked_teacher.argmax(dim=-1)[comparison_mask]
        ).float().mean()
        return {
            "distillation_student_context_logit_l1": student_delta,
            "distillation_teacher_context_logit_l1": teacher_delta,
            "distillation_context_response_ratio": (
                student_delta / teacher_delta.clamp_min(1e-8)
            ),
            "distillation_student_context_argmax_change_fraction": (
                student_argmax_change
            ),
            "distillation_teacher_context_argmax_change_fraction": (
                teacher_argmax_change
            ),
            "distillation_context_tokens": comparison_mask.sum().float(),
        }

    def frigid_distillation_objective(
        self,
        batch: dict[str, object],
        *,
        generator: torch.Generator | None = None,
        collect_context_metrics: bool = False,
        rollout_prefix_step: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the distilled student objective on one fair masked block."""

        settings = self.distillation
        if settings is None:
            raise RuntimeError("FRIGID distillation is not enabled")
        teacher = object.__getattribute__(self, "_frigid_teacher")
        if teacher is None or not callable(getattr(teacher, "logits", None)):
            raise RuntimeError(
                "FRIGID teacher is not loaded; training must run on_train_start first"
            )

        input_ids = batch["input_ids"]
        precursor_mass = batch["precursor_mass"]
        fingerprint = batch["fingerprint"]
        formulas = batch.get("formula")
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a tensor")
        if not isinstance(precursor_mass, torch.Tensor):
            raise TypeError("precursor_mass must be a tensor")
        if not isinstance(fingerprint, torch.Tensor):
            raise TypeError("fingerprint must be a tensor")
        if (
            not isinstance(formulas, (list, tuple))
            or len(formulas) != input_ids.shape[0]
            or not all(isinstance(formula, str) for formula in formulas)
        ):
            raise ValueError(
                "FRIGID distillation requires one true molecular formula per row"
            )

        fair = build_fair_block_inputs(
            input_ids,
            block_width=settings.block_width_override,
            bos_token_id=getattr(self.decoder.config, "bos_token_id", 1),
            pad_token_id=self.decoder.config.pad_token_id,
            mask_token_id=self.decoder.config.mask_token_id,
            generator=generator,
            current_block_masking=settings.current_block_masking,
            full_block_mask_probability=settings.full_block_mask_probability,
            rollout_prefix_probability=settings.rollout_prefix_probability,
            rollout_prefix_schedule=settings.rollout_prefix_schedule,
            rollout_prefix_step=rollout_prefix_step,
        )
        if settings.attention_mode == "frigid_full":
            student_logits = self.decoder(
                fair.input_ids,
                precursor_mass,
                fingerprint,
                isotope_ratios=None,
                attention_mode="frigid_full",
                block_width_override=settings.block_width_override,
            )
        else:
            student_logits = self.decoder.two_stream_logits(
                input_ids,
                fair.input_ids,
                precursor_mass,
                fingerprint,
                isotope_ratios=None,
            )
        teacher_logits = teacher.logits(
            fair.input_ids,
            list(formulas),
            fingerprint,
        )
        target_balance_mask = input_ids.ne(self.decoder.config.pad_token_id)
        target_balance_mask[:, 0] = False
        result = frigid_distillation_loss(
            student_logits,
            teacher_logits,
            input_ids,
            fair.loss_mask,
            temperature=settings.temperature,
            kl_weight=settings.kl_weight,
            loss_weights=fair.loss_weights,
            normalization_mask=(
                fair.current_content_mask
                if settings.current_block_masking == "continuous_time"
                else None
            ),
            balanced_token_loss_alpha=self.balanced_token_loss_alpha,
            token_loss_weight_max=self.token_loss_weight_max,
            eos_token_id=self.decoder.config.eos_token_id,
            target_balance_mask=target_balance_mask,
        )
        predictions = student_logits.detach().argmax(dim=-1)
        current_accuracy = (
            predictions[fair.loss_mask]
            .eq(input_ids[fair.loss_mask])
            .float()
            .mean()
        )
        content_targets = target_balance_mask & input_ids.ne(
            self.decoder.config.eos_token_id
        )
        token_counts = torch.bincount(
            input_ids[content_targets],
            minlength=self.decoder.config.vocab_size,
        )
        majority_token_id = token_counts.argmax()
        non_majority_mask = fair.loss_mask & input_ids.ne(majority_token_id)
        non_majority_tokens = non_majority_mask.sum()
        non_majority_correct = (
            predictions.eq(input_ids) & non_majority_mask
        ).sum()
        non_majority_accuracy = (
            non_majority_correct.float()
            / non_majority_tokens.clamp_min(1).float()
        )
        metrics = {
            "distillation_cross_entropy": result.cross_entropy.detach(),
            "distillation_kl": result.kl.detach(),
            "distillation_student_teacher_top1_agreement": (
                result.student_teacher_top1_agreement.detach()
            ),
            "distillation_current_token_accuracy": current_accuracy,
            "distillation_non_majority_token_accuracy": (
                non_majority_accuracy
            ),
            "distillation_non_majority_tokens": non_majority_tokens.float(),
            "distillation_selected_target_weight": (
                result.selected_target_weight.detach()
            ),
            "current_tokens": result.current_tokens.detach(),
            "distillation_mask_probability": (
                fair.mask_probabilities.mean().detach()
            ),
            "distillation_full_block_mask_fraction": (
                fair.full_block_masked.float().mean().detach()
            ),
            "distillation_rollout_prefix_fraction": (
                fair.rollout_prefix_masked.float().mean().detach()
            ),
        }
        if collect_context_metrics:
            metrics.update(
                self._distillation_context_response_metrics(
                    input_ids=input_ids,
                    fair=fair,
                    precursor_mass=precursor_mass,
                    fingerprint=fingerprint,
                    formulas=list(formulas),
                )
            )
        return result.loss, metrics

    def training_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        if self.distillation is not None:
            next_step = int(self.global_step) + 1
            collect_context_metrics = (
                self.distillation.attention_mode == "block"
                and (next_step == 1 or next_step % self.metric_interval == 0)
                and int(self.global_step) != self._last_distillation_context_step
            )
            loss, distillation_metrics = self.frigid_distillation_objective(
                batch,
                collect_context_metrics=collect_context_metrics,
                rollout_prefix_step=(
                    int(self.global_step)
                    if self.distillation.rollout_prefix_schedule
                    == CYCLIC_ROLLOUT_PREFIX_SCHEDULE
                    else None
                ),
            )
            if collect_context_metrics:
                self._last_distillation_context_step = int(self.global_step)
            for name, value in distillation_metrics.items():
                self.log(
                    f"train_{name}",
                    value,
                    on_step=True,
                    sync_dist=True,
                )
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                sync_dist=True,
            )
            self.log(
                "fingerprint_noise_fraction",
                0.0,
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
            input_ids = batch["input_ids"]
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError("input_ids must be a tensor")
            self.log(
                "micro_batch_size",
                float(input_ids.shape[0]),
                on_step=True,
                sync_dist=True,
            )
            return loss
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
        parameters = [
            parameter
            for parameter in self.decoder.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("MARLIN decoder has no trainable parameters")
        return torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def on_train_start(self) -> None:
        self.ema.move_shadow_params_to_device(self.device)
        if self.distillation is None:
            return

        teacher = object.__getattribute__(self, "_frigid_teacher")
        load = getattr(teacher, "load", None)
        if callable(load):
            teacher = load()
        if teacher is None:
            raise RuntimeError("FRIGID teacher loader returned no teacher")
        move = getattr(teacher, "to", None)
        evaluate = getattr(teacher, "eval", None)
        if not callable(move) or not callable(evaluate):
            raise TypeError("FRIGID teacher must provide to() and eval()")
        moved_teacher = move(self.device)
        if moved_teacher is not None:
            teacher = moved_teacher
        teacher.eval()
        object.__setattr__(self, "_frigid_teacher", teacher)
        self._set_distillation_training_modes()

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.decoder.parameters())

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])


class ClearMLScalarCallback(L.Callback):
    """Publish selected training scalars directly to the active ClearML task."""

    metric_names = (
        "train_loss",
        "train_distillation_cross_entropy",
        "train_distillation_kl",
        "train_distillation_student_teacher_top1_agreement",
        "train_distillation_current_token_accuracy",
        "train_distillation_non_majority_token_accuracy",
        "train_distillation_non_majority_tokens",
        "train_distillation_selected_target_weight",
        "train_distillation_mask_probability",
        "train_distillation_full_block_mask_fraction",
        "train_distillation_rollout_prefix_fraction",
        "train_distillation_student_context_logit_l1",
        "train_distillation_teacher_context_logit_l1",
        "train_distillation_context_response_ratio",
        "train_distillation_student_context_argmax_change_fraction",
        "train_distillation_teacher_context_argmax_change_fraction",
        "train_distillation_context_tokens",
        "train_current_tokens",
        "learning_rate",
        "grad_norm",
        "micro_batch_size",
        "fingerprint_noise_fraction",
    )

    def __init__(self, clearml_task, *, every_n_steps: int = 10) -> None:
        if every_n_steps <= 0:
            raise ValueError("ClearML scalar interval must be positive")
        self.clearml_task = clearml_task
        self.every_n_steps = every_n_steps
        self._last_step = -1

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: MarlinLightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        step = int(trainer.global_step)
        if (
            not trainer.is_global_zero
            or step <= 0
            or (step != 1 and step % self.every_n_steps)
            or step == self._last_step
        ):
            return

        logger = self.clearml_task.get_logger()
        reported = False
        for name in self.metric_names:
            value = trainer.logged_metrics.get(name)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                scalar = float(value.detach().float().cpu().item())
            else:
                scalar = float(value)
            if not math.isfinite(scalar):
                continue
            logger.report_scalar(
                title="MARLIN training",
                series=name,
                value=scalar,
                iteration=step,
            )
            reported = True
        if reported:
            self._last_step = step


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
        use_ema: bool = True,
        clearml_task=None,
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
        self.latest_output_path = (
            Path(output_dir) / "molecular_validation_latest.json"
        )
        self.every_n_steps = every_n_steps
        self.candidates = candidates
        self.temperature = temperature
        self.use_ema = use_ema
        self.weights = "ema" if use_ema else "raw"
        self.clearml_task = clearml_task
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
            strict_safe_to_smiles=lambda safe: safe_to_smiles(safe, fix=False),
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

    def _report_clearml_samples(
        self,
        *,
        step: int,
        epoch: float,
        sample_rows: list[dict],
    ) -> None:
        if self.clearml_task is None:
            return
        logger = self.clearml_task.get_logger()
        table = pd.DataFrame(sample_rows)
        for column in ("terminal_safe_examples", "dead_end_examples"):
            if column in table:
                table[column] = table[column].map(
                    lambda value: json.dumps(value, sort_keys=True)
                )
        table.insert(0, "epoch", epoch)
        table.insert(0, "step", step)
        logger.report_table(
            title="MARLIN molecular validation",
            series="target vs generated",
            iteration=step,
            table_plot=table,
        )

        molecules = []
        legends = []
        for index, row in enumerate(sample_rows):
            target = Chem.MolFromSmiles(row["target_smiles"])
            if target is not None:
                molecules.append(target)
                legends.append(f"{index + 1} target\n{row['target_smiles']}")
            generated_smiles = row["generated_smiles"]
            generated = (
                Chem.MolFromSmiles(generated_smiles)
                if generated_smiles
                else None
            )
            if generated is not None:
                molecules.append(generated)
                legends.append(
                    f"{index + 1} generated\n"
                    f"Tanimoto={row['tanimoto']:.3f} "
                    f"mass={row['mass_error_ppm']:.1f} ppm\n"
                    f"{generated_smiles}"
                )
        if molecules:
            image = Draw.MolsToGridImage(
                molecules,
                legends=legends,
                molsPerRow=2,
                subImgSize=(420, 300),
                useSVG=False,
            )
            logger.report_image(
                title="MARLIN generated molecules",
                series="target vs generated",
                iteration=step,
                image=image,
                max_image_history=-1,
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

        parameters = list(pl_module.decoder.parameters())
        was_training = pl_module.decoder.training
        if self.use_ema:
            pl_module.ema.store(parameters)
            pl_module.ema.copy_to(parameters)
        pl_module.decoder.eval()
        try:
            sampler = self._sampler(pl_module.decoder)
            attempts = valid = strict_valid = 0
            mass_valid = unique_mass_valid = returned = exact = 0
            dead_ends = eos_terminated = max_length_terminated = 0
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
                strict_valid += stats.strict_valid
                mass_valid += stats.mass_valid
                unique_mass_valid += stats.unique_mass_valid
                dead_ends += stats.constraint_dead_ends
                eos_terminated += stats.eos_terminated
                max_length_terminated += stats.max_length_terminated
                returned += int(bool(ranked))
                generated_smiles = ranked[0].smiles if ranked else None
                if generated_smiles is None:
                    for terminal_safe in stats.sample_terminal_safes:
                        decoded = safe_to_smiles(terminal_safe, fix=True)
                        if decoded and Chem.MolFromSmiles(decoded) is not None:
                            generated_smiles = decoded
                            break
                generated_molecule = (
                    Chem.MolFromSmiles(generated_smiles)
                    if generated_smiles
                    else None
                )
                generated_tanimoto = 0.0
                generated_mass_error_ppm = None
                if generated_molecule is not None:
                    generated_fingerprint = AllChem.GetMorganGenerator(
                        radius=2,
                        fpSize=record["fingerprint"].numel(),
                    ).GetFingerprint(generated_molecule)
                    generated_tanimoto = DataStructs.TanimotoSimilarity(
                        _fingerprint_from_tensor(record["fingerprint"]),
                        generated_fingerprint,
                    )
                    generated_mass_error_ppm = (
                        abs(
                            Descriptors.ExactMolWt(generated_molecule)
                            - record["mass"]
                        )
                        / record["mass"]
                        * 1e6
                    )
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
                        "generated_smiles": generated_smiles,
                        "tanimoto": generated_tanimoto,
                        "mass_error_ppm": generated_mass_error_ppm,
                        "valid": stats.valid,
                        "strict_valid": stats.strict_valid,
                        "mass_valid": stats.mass_valid,
                        "constraint_dead_ends": stats.constraint_dead_ends,
                        "eos_terminated": stats.eos_terminated,
                        "max_length_terminated": stats.max_length_terminated,
                        "terminal_safe_examples": list(
                            stats.sample_terminal_safes
                        ),
                        "dead_end_examples": list(stats.sample_dead_ends),
                    }
                )

            count = len(self.records)
            metrics = {
                "validity": valid / attempts,
                "strict_validity": strict_valid / attempts,
                "mass_validity": mass_valid / attempts,
                "unique_mass_validity": unique_mass_valid / attempts,
                "candidate_return_rate": returned / count,
                "exact_top1": exact / count,
                "tanimoto_top1": float(np.mean(top1_tanimoto)),
                "constraint_dead_end_rate": dead_ends / attempts,
                "eos_termination_rate": eos_terminated / attempts,
                "max_length_termination_rate": max_length_terminated / attempts,
            }
            for name, value in metrics.items():
                pl_module.log(
                    f"molecular_{self.weights}_{name}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )
            if self.clearml_task is not None:
                logger = self.clearml_task.get_logger()
                for name, value in metrics.items():
                    logger.report_scalar(
                        title=f"MARLIN molecular metrics ({self.weights})",
                        series=name,
                        value=float(value),
                        iteration=step,
                    )
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "step": step,
                "weights": self.weights,
                "metrics": metrics,
                "samples": sample_rows,
            }
            serialized = json.dumps(result, sort_keys=True)
            with self.output_path.open("a") as handle:
                handle.write(serialized + "\n")
            latest_temporary_path = self.latest_output_path.with_suffix(".tmp")
            latest_temporary_path.write_text(serialized + "\n")
            latest_temporary_path.replace(self.latest_output_path)
            self._report_clearml_samples(
                step=step,
                epoch=float(trainer.current_epoch),
                sample_rows=sample_rows,
            )
        finally:
            if self.use_ema:
                pl_module.ema.restore(parameters)
            pl_module.decoder.train(was_training)
            if pl_module.distillation is not None:
                pl_module._set_distillation_training_modes()


def _fingerprint_from_tensor(array: torch.Tensor):
    bit_vector = DataStructs.ExplicitBitVect(array.numel())
    for index in torch.nonzero(array.reshape(-1), as_tuple=False).flatten().tolist():
        bit_vector.SetBit(index)
    return bit_vector
