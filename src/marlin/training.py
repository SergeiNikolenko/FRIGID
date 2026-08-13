"""Training utilities for the clean-room MARLIN decoder."""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterable, Mapping, Sequence
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
from marlin.lr_schedule import (
    ADAMW_SECOND_MOMENT_WINDOW,
    RELEASED_TERMINAL_LEARNING_RATE,
    build_warmup_cosine_scheduler,
)
from marlin.model import (
    LOSS_REDUCTIONS,
    TIME_SAMPLING_MODES,
    MarlinDecoder,
    MarlinDecoderConfig,
)
from marlin.noise import one_sided_fingerprint_dropout, symmetric_fingerprint_noise
from marlin.isotopes import theoretical_isotope_ratios


LR_SCHEDULES = ("constant", "warmup_cosine")


def fp32_forward_context(enabled: bool, device_type: str):
    """The released model's fp32 forward override, as a context manager.

    ``src/dlm/model.py:949`` wraps its whole training forward in
    ``torch.amp.autocast('cuda', dtype=torch.float32)``. Torch does not support
    fp32 as an autocast target, so that call warns and disables autocasting for
    the region -- i.e. the released model computes its forward in fp32 while the
    trainer is nominally in bf16. ``torch.autocast(..., enabled=False)`` is that
    same region without the warning, and it is what this returns. Our runs use
    ``precision="bf16-mixed"`` with no such override, so the objective they
    optimise is not the one the checkpoint was trained under.
    """
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, enabled=False)


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
        allow_soft_fingerprints: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fingerprint_bits = fingerprint_bits
        self.exclude = load_excluded_connectivity_keys(exclude_inchikeys)
        self.allow_soft_fingerprints = allow_soft_fingerprints

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
                if self.allow_soft_fingerprints:
                    if np.any(array < 0.0) or np.any(array > 1.0):
                        raise ValueError(
                            "provided soft fingerprint must be in [0, 1]"
                        )
                elif not np.array_equal(array, array.astype(bool)):
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
        preserve_probabilities: bool = False,
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
            fingerprint = values[positions[spec_name]]
            if preserve_probabilities:
                if np.any(fingerprint < 0.0) or np.any(fingerprint > 1.0):
                    raise ValueError("soft fingerprints must be probabilities in [0, 1]")
                # The encoder activates every bit above 0.5, so without this gate
                # a soft bundle conditions on far more bits than the threshold
                # the run records.
                fingerprint = np.where(fingerprint >= threshold, fingerprint, 0.0)
            else:
                fingerprint = fingerprint >= threshold
            rows.append(
                {
                    "safe": safe,
                    "fingerprint": fingerprint.astype(np.float32),
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


def _unit_hash(payload: str) -> float:
    """Map a string to [0, 1) deterministically across processes and versions."""
    digest = hashlib.sha256(payload.encode()).digest()[:8]
    return int.from_bytes(digest, "big") / 2**64


def structure_disjoint_holdout(
    rows: Sequence[Mapping[str, object]],
    *,
    fraction: float,
    seed: int = 0,
    key: str = "inchikey_first_block",
) -> tuple[list[int], list[int]]:
    """Split dataset rows into training and held-out indices by structure.

    The split is taken on the InChIKey connectivity block rather than on the row,
    so every spectrum of one molecule lands on the same side and the held-out
    slice shares no structure with what the optimizer sees. It is a hash of the
    key rather than a shuffle, so it depends on neither row order, row count, nor
    global RNG state: adding spectra moves no existing structure across the
    boundary, and two runs with the same seed hold out the same molecules.

    ``fraction = 0`` returns every row as training data and an empty held-out
    slice, which is the historical behaviour.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("held-out fraction must be in [0, 1)")
    train_indices: list[int] = []
    holdout_indices: list[int] = []
    for index, row in enumerate(rows):
        value = row.get(key)
        if value is None or not str(value):
            raise ValueError(f"row {index} has no {key} to split on")
        if fraction > 0.0 and _unit_hash(f"{seed}:{value}") < fraction:
            holdout_indices.append(index)
        else:
            train_indices.append(index)
    if fraction > 0.0 and not holdout_indices:
        raise ValueError(
            f"held-out fraction {fraction} selected no structures from "
            f"{len(rows)} rows"
        )
    if not train_indices:
        raise ValueError(
            f"held-out fraction {fraction} selected every structure; nothing "
            "would be left to train on"
        )
    return train_indices, holdout_indices


def holdout_split_report(
    rows: Sequence[Mapping[str, object]],
    train_indices: Iterable[int],
    holdout_indices: Iterable[int],
    *,
    key: str = "inchikey_first_block",
) -> dict[str, int]:
    """Verify and describe the disjointness of a held-out split.

    Raises rather than reporting a violation: a validation loss measured on a
    structure the optimizer also sees is not held out, and a run must not be able
    to record one as if it were.
    """
    training = list(train_indices)
    holdout = list(holdout_indices)
    train_keys = {str(rows[index][key]) for index in training}
    holdout_keys = {str(rows[index][key]) for index in holdout}
    shared = sorted(train_keys & holdout_keys)
    if shared:
        raise ValueError(
            "held-out slice shares structures with training: "
            + ", ".join(shared[:5])
        )
    return {
        "training_rows": len(training),
        "holdout_rows": len(holdout),
        "training_structures": len(train_keys),
        "holdout_structures": len(holdout_keys),
        "shared_structures": 0,
    }


class PeriodicHeldOutLoss(L.Callback):
    """Log the masked-diffusion objective on a structure-disjoint slice.

    The recipe had no validation loss at all, so a run could only be judged by a
    molecular panel that cost hours. This is the cheap half of the held-out
    signal: the same objective as ``training_step``, with the same weighting, on
    molecules the optimizer never sees.

    Two deliberate differences from training make consecutive checkpoints
    comparable:

    - the fingerprint is not corrupted. Symmetric corruption is a training
      augmentation, and inference conditions on the uncorrupted predicted
      fingerprint, so corrupting here would inject variance into the signal;
    - the masking generator is re-seeded at every pass, so every checkpoint is
      scored on the same mask draws and only the weights differ.
    """

    def __init__(
        self,
        loader,
        *,
        interval_steps: int,
        seed: int = 0,
        max_batches: int | None = None,
        clearml_task=None,
        title: str = "Held-out loss",
    ) -> None:
        super().__init__()
        if interval_steps <= 0:
            raise ValueError("held-out loss interval must be positive")
        if max_batches is not None and max_batches <= 0:
            raise ValueError("held-out loss max_batches must be positive")
        self.loader = loader
        self.interval_steps = interval_steps
        self.seed = seed
        self.max_batches = max_batches
        self.clearml_task = clearml_task
        self.title = title
        self._last_step = -1

    def evaluate(self, pl_module) -> dict[str, float]:
        was_training = pl_module.training
        pl_module.eval()
        device = pl_module.device
        generator = torch.Generator(device=device).manual_seed(self.seed)
        totals: dict[str, float] = {}
        batches = 0
        try:
            with torch.no_grad():
                for index, batch in enumerate(self.loader):
                    if self.max_batches is not None and index >= self.max_batches:
                        break
                    batch = {
                        name: value.to(device) for name, value in batch.items()
                    }
                    loss, metrics = pl_module.decoder.diffusion_objective(
                        batch["input_ids"],
                        batch["precursor_mass"],
                        batch["fingerprint"],
                        isotope_ratios=batch["isotope_ratios"],
                        generator=generator,
                        eos_loss_weight=pl_module.eos_loss_weight,
                        eos_mask_probability=pl_module.eos_mask_probability,
                        balanced_token_loss_alpha=(
                            pl_module.balanced_token_loss_alpha
                        ),
                        token_loss_weight_max=pl_module.token_loss_weight_max,
                        full_sequence_mask_probability=(
                            pl_module.full_sequence_mask_probability
                        ),
                        # The held-out loss has to be the training objective or
                        # it cannot be read against it, so it follows whatever
                        # reduction and time sampling the run selected.
                        time_sampling=getattr(
                            pl_module, "time_sampling", "per_block_iid"
                        ),
                        time_sampling_eps=getattr(
                            pl_module, "time_sampling_eps", 1e-3
                        ),
                        loss_reduction=getattr(
                            pl_module, "loss_reduction", "block_mean"
                        ),
                        collect_metrics=True,
                    )
                    batches += 1
                    totals["loss"] = totals.get("loss", 0.0) + float(loss)
                    for name, value in metrics.items():
                        totals[name] = totals.get(name, 0.0) + float(value)
        finally:
            pl_module.train(was_training)
        if not batches:
            raise ValueError("held-out loader produced no batches")
        return {name: value / batches for name, value in totals.items()}

    def _report(self, step: int, values: dict[str, float]) -> None:
        if self.clearml_task is None:
            return
        logger = self.clearml_task.get_logger()
        for name, value in values.items():
            logger.report_scalar(
                title=self.title,
                series=f"val_{name}",
                value=value,
                iteration=step,
            )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        del outputs, batch, batch_idx
        step = int(trainer.global_step)
        if step <= 0 or step % self.interval_steps or step == self._last_step:
            return
        self._last_step = step
        values = self.evaluate(pl_module)
        for name, value in values.items():
            pl_module.log(f"val_{name}", value, on_step=True, sync_dist=False)
        print(
            f"Held-out masked-diffusion loss at step {step}: "
            f"{values['loss']:.6f}",
            flush=True,
        )
        if trainer.is_global_zero:
            self._report(step, values)


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
        fingerprint_noise_mode: str = "symmetric",
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
        context_corruption_probability: float = 0.0,
        context_corruption_warmup_steps: int = 0,
        context_corruption_min_fraction: float = 0.05,
        context_corruption_max_fraction: float = 0.25,
        restoration_loss_weight: float = 0.0,
        lr_schedule: str = "constant",
        lr_warmup_steps: int = ADAMW_SECOND_MOMENT_WINDOW,
        lr_total_steps: int | None = None,
        lr_min: float = RELEASED_TERMINAL_LEARNING_RATE,
        time_sampling: str = "per_block_iid",
        time_sampling_eps: float = 1e-3,
        loss_reduction: str = "block_mean",
        fp32_forward: bool = False,
    ) -> None:
        super().__init__()
        if conditioning_only_steps < 0 or cross_attention_only_steps < 0:
            raise ValueError("adaptation stage durations must be non-negative")
        if lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"unknown lr_schedule {lr_schedule!r}; expected one of "
                f"{', '.join(LR_SCHEDULES)}"
            )
        if lr_schedule == "warmup_cosine":
            if not lr_total_steps or lr_total_steps <= 0:
                raise ValueError(
                    "a warmup-cosine schedule needs lr_total_steps: a cosine "
                    "with no end is not a schedule"
                )
            if not 0 <= lr_warmup_steps < lr_total_steps:
                raise ValueError(
                    "lr_warmup_steps must satisfy 0 <= warmup < lr_total_steps"
                )
            if not 0.0 <= lr_min <= learning_rate:
                raise ValueError(
                    f"lr_min {lr_min:g} must be in [0, peak {learning_rate:g}]"
                )
        if time_sampling not in TIME_SAMPLING_MODES:
            raise ValueError(
                f"unknown time_sampling {time_sampling!r}; expected one of "
                f"{', '.join(TIME_SAMPLING_MODES)}"
            )
        if loss_reduction not in LOSS_REDUCTIONS:
            raise ValueError(
                f"unknown loss_reduction {loss_reduction!r}; expected one of "
                f"{', '.join(LOSS_REDUCTIONS)}"
            )
        if fingerprint_noise_mode not in {"symmetric", "dropout"}:
            raise ValueError(
                "fingerprint_noise_mode must be 'symmetric' or 'dropout', "
                f"got {fingerprint_noise_mode!r}"
            )
        if context_corruption_warmup_steps < 0:
            raise ValueError("context_corruption_warmup_steps must be non-negative")
        self.save_hyperparameters(
            {
                "config": asdict(config),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "noise_probability": noise_probability,
                "noise_min_fraction": noise_min_fraction,
                "noise_max_fraction": noise_max_fraction,
                "fingerprint_noise_mode": fingerprint_noise_mode,
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
                "context_corruption_probability": context_corruption_probability,
                "context_corruption_warmup_steps": context_corruption_warmup_steps,
                "context_corruption_min_fraction": context_corruption_min_fraction,
                "context_corruption_max_fraction": context_corruption_max_fraction,
                "restoration_loss_weight": restoration_loss_weight,
                "lr_schedule": lr_schedule,
                "lr_warmup_steps": lr_warmup_steps,
                "lr_total_steps": lr_total_steps,
                "lr_min": lr_min,
                "time_sampling": time_sampling,
                "time_sampling_eps": time_sampling_eps,
                "loss_reduction": loss_reduction,
                "fp32_forward": fp32_forward,
            }
        )
        self.decoder = MarlinDecoder(config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.noise_probability = noise_probability
        self.fingerprint_noise_mode = fingerprint_noise_mode
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
        self.context_corruption_probability = context_corruption_probability
        self.context_corruption_warmup_steps = context_corruption_warmup_steps
        self.context_corruption_min_fraction = context_corruption_min_fraction
        self.context_corruption_max_fraction = context_corruption_max_fraction
        self.restoration_loss_weight = restoration_loss_weight
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_total_steps = lr_total_steps
        self.lr_min = lr_min
        self.time_sampling = time_sampling
        self.time_sampling_eps = time_sampling_eps
        self.loss_reduction = loss_reduction
        self.fp32_forward = fp32_forward
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

    def context_corruption_schedule(self, step: int) -> float:
        """Ramp the corrupted-context probability in from zero.

        Step 0 must reproduce the source checkpoint's objective exactly, so any
        divergence in the first logged steps is attributable to the ramp and not
        to a silent change of objective. The ramp is linear and short.
        """
        if not self.context_corruption_probability:
            return 0.0
        if step >= self.context_corruption_warmup_steps:
            return self.context_corruption_probability
        return self.context_corruption_probability * (
            step / max(self.context_corruption_warmup_steps, 1)
        )

    def on_train_batch_start(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        self.apply_adaptation_stage(int(self.global_step))

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        corrupt_fingerprint = (
            symmetric_fingerprint_noise
            if self.fingerprint_noise_mode == "symmetric"
            else one_sided_fingerprint_dropout
        )
        fingerprint = corrupt_fingerprint(
            batch["fingerprint"],
            corruption_probability=self.noise_probability,
            min_fraction=self.noise_min_fraction,
            max_fraction=self.noise_max_fraction,
        )
        collect_metrics = (
            self.global_step % self.metric_interval == 0
            and self.global_step != self._last_metric_step
        )
        corruption_probability = self.context_corruption_schedule(
            int(self.global_step)
        )
        confusion_ids = None
        if corruption_probability:
            was_training = self.decoder.training
            # Dropout would make the confusion a different model's mistake.
            self.decoder.eval()
            with fp32_forward_context(self.fp32_forward, self.device.type):
                confusion_ids = self.decoder.sample_confusions(
                    batch["input_ids"],
                    batch["precursor_mass"],
                    fingerprint,
                    batch["isotope_ratios"],
                )
            self.decoder.train(was_training)
        with fp32_forward_context(self.fp32_forward, self.device.type):
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
                context_corruption_probability=corruption_probability,
                context_corruption_min_fraction=self.context_corruption_min_fraction,
                context_corruption_max_fraction=self.context_corruption_max_fraction,
                restoration_loss_weight=self.restoration_loss_weight,
                confusion_ids=confusion_ids,
                time_sampling=self.time_sampling,
                time_sampling_eps=self.time_sampling_eps,
                loss_reduction=self.loss_reduction,
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
        self.log(
            "context_corruption_probability",
            corruption_probability,
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
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.lr_schedule == "constant":
            return optimizer
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=int(self.lr_total_steps),
            warmup_steps=int(self.lr_warmup_steps),
            floor=float(self.lr_min),
        )
        return {
            "optimizer": optimizer,
            # Per optimizer step, not per epoch: the adaptation set is replayed
            # thousands of times, so an epoch-stepped cosine would finish in the
            # first few minutes.
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

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
