from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

from marlin.model import MarlinDecoder, MarlinDecoderConfig
from marlin.warm_start import load_marlin_decoder_weights


def tiny_config(**overrides) -> MarlinDecoderConfig:
    values = {
        "vocab_size": 16,
        "hidden_size": 8,
        "num_layers": 1,
        "num_heads": 2,
        "intermediate_size": 16,
        "max_length": 8,
        "block_width": 2,
        "fingerprint_bits": 8,
        "dropout": 0.0,
        "pad_token_id": 3,
    }
    values.update(overrides)
    return MarlinDecoderConfig(**values)


def write_checkpoint(path, model: MarlinDecoder) -> None:
    torch.save(
        {
            "global_step": 123,
            "hyper_parameters": {"config": asdict(model.config)},
            "state_dict": {
                f"decoder.{name}": value
                for name, value in model.state_dict().items()
            },
        },
        path,
    )


def test_architecture_upgrade_whitelists_fingerprint_layer_norm(tmp_path) -> None:
    source = MarlinDecoder(tiny_config(fingerprint_layer_norm=False))
    checkpoint = tmp_path / "source.ckpt"
    write_checkpoint(checkpoint, source)
    target = MarlinDecoder(tiny_config(fingerprint_layer_norm=True))

    report = load_marlin_decoder_weights(
        target,
        checkpoint,
        architecture_upgrade=True,
    )

    assert report["source_global_step"] == 123
    assert report["initialized_target_keys"] == [
        "conditioner.fingerprint.layer_norm.bias",
        "conditioner.fingerprint.layer_norm.weight",
    ]
    assert torch.equal(target.token_embedding.weight, source.token_embedding.weight)
    assert torch.all(target.conditioner.fingerprint.layer_norm.weight == 1)
    assert torch.all(target.conditioner.fingerprint.layer_norm.bias == 0)


def test_strict_checkpoint_load_rejects_architecture_change(tmp_path) -> None:
    source = MarlinDecoder(tiny_config(fingerprint_layer_norm=False))
    checkpoint = tmp_path / "source.ckpt"
    write_checkpoint(checkpoint, source)
    target = MarlinDecoder(tiny_config(fingerprint_layer_norm=True))

    with pytest.raises(ValueError, match="architecture mismatch"):
        load_marlin_decoder_weights(target, checkpoint)


def test_upgrade_zero_initializes_new_token_type_embedding(tmp_path) -> None:
    source_config = tiny_config(
        fingerprint_layer_norm=False,
        frigid_compatible_layer_order=False,
    )
    source = MarlinDecoder(source_config)
    checkpoint = tmp_path / "source.ckpt"
    write_checkpoint(checkpoint, source)
    target = MarlinDecoder(
        replace(source_config, frigid_compatible_layer_order=True)
    )

    report = load_marlin_decoder_weights(
        target,
        checkpoint,
        architecture_upgrade=True,
    )

    assert "token_type_embedding.weight" in report["initialized_target_keys"]
    assert torch.count_nonzero(target.token_type_embedding.weight) == 0
