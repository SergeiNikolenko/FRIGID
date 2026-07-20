"""Warm-start the clean-room decoder from the released FRIGID checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from marlin.model import MarlinDecoder


def _copy(parameter: torch.Tensor, source: torch.Tensor, name: str) -> None:
    if parameter.shape != source.shape:
        raise ValueError(f"shape mismatch for {name}: {tuple(parameter.shape)} != {tuple(source.shape)}")
    parameter.data.copy_(source.to(dtype=parameter.dtype))


def _copy_attention(layer, state: dict[str, torch.Tensor], prefix: str, name: str) -> None:
    _copy(
        layer.in_proj_weight,
        torch.cat([state[f"{prefix}.{part}.weight"] for part in ("query", "key", "value")]),
        f"{name}.in_proj_weight",
    )
    _copy(
        layer.in_proj_bias,
        torch.cat([state[f"{prefix}.{part}.bias"] for part in ("query", "key", "value")]),
        f"{name}.in_proj_bias",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frigid_decoder(
    model: MarlinDecoder,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, int | str]:
    """Load all architecture-compatible FRIGID weights and leave mass tokens new."""
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
        raise ValueError(
            f"FRIGID checkpoint SHA-256 is {checkpoint_sha256}; "
            f"expected {expected_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    _copy(model.token_embedding.weight, state["backbone.bert.embeddings.word_embeddings.weight"], "tokens")
    _copy(model.position_embedding.weight, state["backbone.bert.embeddings.position_embeddings.weight"], "positions")
    _copy(
        model.conditioner.fingerprint.embedding.weight,
        state["fingerprint_conditioner.fingerprint_encoder.bit_embeddings.weight"],
        "fingerprint bits",
    )

    for index, layer in enumerate(model.layers):
        source = f"backbone.bert.encoder.layer.{index}"
        self_prefix = f"{source}.bert_layer.attention.self"
        _copy_attention(layer.self_attention, state, self_prefix, f"layer {index} self attention")
        _copy(
            layer.self_attention.out_proj.weight,
            state[f"{source}.bert_layer.attention.output.dense.weight"],
            f"layer {index} self output weight",
        )
        _copy(
            layer.self_attention.out_proj.bias,
            state[f"{source}.bert_layer.attention.output.dense.bias"],
            f"layer {index} self output bias",
        )
        _copy_attention(layer.cross_attention, state, f"{source}.cross_attention", f"layer {index} cross attention")
        _copy(
            layer.cross_attention.out_proj.weight,
            state[f"{source}.cross_attention.output_dense.weight"],
            f"layer {index} cross output weight",
        )
        _copy(
            layer.cross_attention.out_proj.bias,
            state[f"{source}.cross_attention.output_dense.bias"],
            f"layer {index} cross output bias",
        )
        mappings = (
            (layer.norm1, f"{source}.bert_layer.attention.output.LayerNorm"),
            (layer.norm2, f"{source}.cross_attention.output_layer_norm"),
            (layer.norm3, f"{source}.bert_layer.output.LayerNorm"),
        )
        for target, key in mappings:
            _copy(target.weight, state[f"{key}.weight"], f"{key}.weight")
            _copy(target.bias, state[f"{key}.bias"], f"{key}.bias")
        _copy(layer.linear1.weight, state[f"{source}.bert_layer.intermediate.dense.weight"], "ffn1 weight")
        _copy(layer.linear1.bias, state[f"{source}.bert_layer.intermediate.dense.bias"], "ffn1 bias")
        _copy(layer.linear2.weight, state[f"{source}.bert_layer.output.dense.weight"], "ffn2 weight")
        _copy(layer.linear2.bias, state[f"{source}.bert_layer.output.dense.bias"], "ffn2 bias")

    head = "backbone.cls.predictions.transform"
    _copy(model.prediction_dense.weight, state[f"{head}.dense.weight"], "prediction dense weight")
    _copy(model.prediction_dense.bias, state[f"{head}.dense.bias"], "prediction dense bias")
    _copy(model.prediction_norm.weight, state[f"{head}.LayerNorm.weight"], "prediction norm weight")
    _copy(model.prediction_norm.bias, state[f"{head}.LayerNorm.bias"], "prediction norm bias")
    _copy(model.output_bias, state["backbone.cls.predictions.bias"], "prediction bias")
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "layers_loaded": len(model.layers),
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
    }
