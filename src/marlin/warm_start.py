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


def _copy_embeddings(
    model: MarlinDecoder,
    state: dict[str, torch.Tensor],
    prefix: str,
) -> None:
    if model.token_type_embedding is None or not isinstance(
        model.embedding_norm, torch.nn.LayerNorm
    ):
        raise ValueError(
            "FRIGID embedding transfer requires the compatible embedding stack"
        )
    _copy(
        model.token_embedding.weight,
        state[f"{prefix}.word_embeddings.weight"],
        "token embeddings",
    )
    _copy(
        model.position_embedding.weight,
        state[f"{prefix}.position_embeddings.weight"],
        "position embeddings",
    )
    _copy(
        model.token_type_embedding.weight,
        state[f"{prefix}.token_type_embeddings.weight"],
        "token type embeddings",
    )
    _copy(
        model.embedding_norm.weight,
        state[f"{prefix}.LayerNorm.weight"],
        "embedding layer norm weight",
    )
    _copy(
        model.embedding_norm.bias,
        state[f"{prefix}.LayerNorm.bias"],
        "embedding layer norm bias",
    )


def _copy_fingerprint_encoder(
    encoder,
    state: dict[str, torch.Tensor],
    prefix: str,
) -> None:
    _copy(
        encoder.embedding.weight,
        state[f"{prefix}.bit_embeddings.weight"],
        "fingerprint bits",
    )
    _copy(
        encoder.layer_norm.weight,
        state[f"{prefix}.layer_norm.weight"],
        "fingerprint layer norm weight",
    )
    _copy(
        encoder.layer_norm.bias,
        state[f"{prefix}.layer_norm.bias"],
        "fingerprint layer norm bias",
    )
    source_layers = {
        int(key.split(".self_attention_layers.", 1)[1].split(".", 1)[0])
        for key in state
        if key.startswith(f"{prefix}.self_attention_layers.")
    }
    if len(encoder.self_attention_layers) != len(source_layers):
        raise ValueError(
            "fingerprint set-encoder layer mismatch: "
            f"{len(encoder.self_attention_layers)} != {len(source_layers)}"
        )
    for index, layer in enumerate(encoder.self_attention_layers):
        source = f"{prefix}.self_attention_layers.{index}"
        _copy_attention(
            layer.attention,
            state,
            source,
            f"fingerprint set attention {index}",
        )
        _copy(
            layer.attention.out_proj.weight,
            state[f"{source}.output_dense.weight"],
            f"fingerprint set attention {index} output weight",
        )
        _copy(
            layer.attention.out_proj.bias,
            state[f"{source}.output_dense.bias"],
            f"fingerprint set attention {index} output bias",
        )
        _copy(
            layer.norm.weight,
            state[f"{source}.output_layer_norm.weight"],
            f"fingerprint set attention {index} norm weight",
        )
        _copy(
            layer.norm.bias,
            state[f"{source}.output_layer_norm.bias"],
            f"fingerprint set attention {index} norm bias",
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
) -> dict[str, bool | int | str]:
    """Load all architecture-compatible FRIGID weights and leave mass tokens new."""
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
        raise ValueError(
            f"FRIGID checkpoint SHA-256 is {checkpoint_sha256}; "
            f"expected {expected_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not model.config.frigid_compatible_layer_order:
        raise ValueError(
            "FRIGID warm-start requires frigid_compatible_layer_order=true"
        )
    _copy_embeddings(model, state, "backbone.bert.embeddings")
    _copy_fingerprint_encoder(
        model.conditioner.fingerprint,
        state,
        "fingerprint_conditioner.fingerprint_encoder",
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
        "fingerprint_set_layers": len(
            model.conditioner.fingerprint.self_attention_layers
        ),
        "frigid_compatible_layer_order": bool(
            model.config.frigid_compatible_layer_order
        ),
    }
