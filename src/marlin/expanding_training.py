"""Lightning training module for conditional MARLIN EFlow and EFM."""

from __future__ import annotations

from dataclasses import asdict

import lightning as L
import torch

from dlm.utils.ema import ExponentialMovingAverage
from marlin.expanding import (
    ExpandingFlowConfig,
    ExpandingMarlinModel,
    eflow_objective,
    efm_objective,
)
from marlin.model import MarlinDecoderConfig
from marlin.noise import symmetric_fingerprint_noise


class ExpandingMarlinLightningModule(L.LightningModule):
    """Train an EFlow teacher or distill an EFM student."""

    def __init__(
        self,
        decoder_config: MarlinDecoderConfig,
        flow_config: ExpandingFlowConfig,
        *,
        stage: str = "eflow",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 2500,
        noise_probability: float = 0.5,
        noise_min_fraction: float = 0.1,
        noise_max_fraction: float = 0.3,
        ema_decay: float = 0.9999,
        metric_interval: int = 50,
    ) -> None:
        super().__init__()
        if stage not in {"eflow", "efm"}:
            raise ValueError("stage must be 'eflow' or 'efm'")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.save_hyperparameters(
            {
                "decoder_config": asdict(decoder_config),
                "flow_config": asdict(flow_config),
                "stage": stage,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "noise_probability": noise_probability,
                "noise_min_fraction": noise_min_fraction,
                "noise_max_fraction": noise_max_fraction,
                "ema_decay": ema_decay,
                "metric_interval": metric_interval,
            }
        )
        self.model = ExpandingMarlinModel(decoder_config, flow_config)
        self.stage = stage
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.noise_probability = noise_probability
        self.noise_min_fraction = noise_min_fraction
        self.noise_max_fraction = noise_max_fraction
        self.ema_decay = ema_decay
        self.metric_interval = metric_interval
        self._last_metric_step = -1
        self.__dict__["_teacher"] = None
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), decay=ema_decay, use_num_updates=False
        )

    @property
    def teacher(self) -> ExpandingMarlinModel | None:
        return self.__dict__.get("_teacher")

    def set_teacher(self, teacher: ExpandingMarlinModel) -> None:
        if self.stage != "efm":
            raise RuntimeError("an EFlow module does not use a teacher")
        teacher.requires_grad_(False).eval()
        self.__dict__["_teacher"] = teacher

    def reset_ema(self) -> None:
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), decay=self.ema_decay, use_num_updates=False
        )

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        fingerprint = symmetric_fingerprint_noise(
            batch["fingerprint"],
            corruption_probability=self.noise_probability,
            min_fraction=self.noise_min_fraction,
            max_fraction=self.noise_max_fraction,
        )
        detach_insertion = (
            self.global_step < self.model.flow_config.insertion_detach_steps
        )
        if self.stage == "eflow":
            loss, metrics = eflow_objective(
                self.model,
                batch["input_ids"],
                batch["precursor_mass"],
                fingerprint,
                isotope_ratios=batch.get("isotope_ratios"),
                detach_insertion_backbone=detach_insertion,
            )
        else:
            teacher = self.teacher
            if teacher is None:
                raise RuntimeError("EFM training requires a frozen EFlow teacher")
            loss, metrics = efm_objective(
                self.model,
                teacher,
                batch["input_ids"],
                batch["precursor_mass"],
                fingerprint,
                isotope_ratios=batch.get("isotope_ratios"),
                detach_insertion_backbone=detach_insertion,
            )
        collect_metrics = (
            self.global_step % self.metric_interval == 0
            and self.global_step != self._last_metric_step
        )
        if collect_metrics:
            self._last_metric_step = self.global_step
            for name, value in metrics.items():
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
            "insertion_backbone_detached",
            float(detach_insertion),
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
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        if not self.warmup_steps:
            return optimizer

        def schedule(step: int) -> float:
            return min(1.0, (step + 1) / self.warmup_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_start(self) -> None:
        self.ema.move_shadow_params_to_device(self.device)
        if self.teacher is not None:
            self.teacher.to(self.device).eval()

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model.parameters())

    def on_before_optimizer_step(self, optimizer) -> None:
        del optimizer
        if self.global_step % 50:
            return
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if norms:
            self.log(
                "grad_norm",
                torch.stack(norms).norm(2),
                on_step=True,
                sync_dist=True,
            )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
