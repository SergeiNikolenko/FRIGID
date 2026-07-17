#!/usr/bin/env python3
# ruff: noqa: E402
"""Train the clean-room MARLIN block-diffusion decoder."""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets
import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from marlin.model import MarlinDecoderConfig
from marlin.tokenizer import load_safe_tokenizer
from marlin.training import MarlinCollator, MarlinLightningModule
from marlin.warm_start import load_frigid_decoder


@hydra.main(version_base=None, config_path="../configs", config_name="marlin_nplib1")
def main(config: DictConfig) -> None:
    L.seed_everything(config.seed, workers=True)
    tokenizer = load_safe_tokenizer(config.data.tokenizer_file)
    decoder_config = MarlinDecoderConfig(
        **OmegaConf.to_container(config.model, resolve=True)
    )
    module = MarlinLightningModule(
        decoder_config,
        learning_rate=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
        noise_probability=config.training.noise_probability,
        noise_min_fraction=config.training.noise_min_fraction,
        noise_max_fraction=config.training.noise_max_fraction,
        ema_decay=config.training.ema_decay,
    )
    if config.get("warm_start_checkpoint"):
        report = load_frigid_decoder(module.decoder, config.warm_start_checkpoint)
        module.reset_ema()
        report_path = Path(config.output.root) / "warm_start.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    dataset = datasets.load_dataset(
        config.data.dataset,
        split="train",
        streaming=True,
        cache_dir=config.data.hf_cache_dir,
    ).shuffle(seed=config.seed, buffer_size=config.data.shuffle_buffer)
    collator = MarlinCollator(
        tokenizer,
        max_length=decoder_config.max_length,
        fingerprint_bits=decoder_config.fingerprint_bits,
        exclude_inchikeys=config.data.exclude_inchikeys,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=config.output.checkpoints,
        filename="{step}",
        every_n_train_steps=config.output.checkpoint_interval,
        save_top_k=-1,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=config.trainer.devices,
        strategy="ddp" if config.trainer.devices > 1 else "auto",
        precision=config.trainer.precision,
        max_steps=config.trainer.max_steps,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        callbacks=[checkpoint],
        default_root_dir=config.output.root,
    )
    trainer.fit(module, loader, ckpt_path=config.get("resume_checkpoint"))


if __name__ == "__main__":
    main()
