"""Warm-start the clean-room decoder from the released FRIGID checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from marlin.model import MarlinDecoder, MarlinDecoderConfig


_ARCHITECTURE_UPGRADE_PREFIXES = (
    "conditioner.fingerprint.layer_norm.",
    "conditioner.fingerprint.self_attention_layers.",
    "token_type_embedding.",
    "embedding_norm.",
)


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


def _tensor_storage_key(tensor: torch.Tensor) -> tuple:
    storage = tensor.untyped_storage()
    return (
        storage.data_ptr(),
        storage.nbytes(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _ema_backbone_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Replace raw backbone parameters with the EMA used by FRIGID inference."""
    state = checkpoint.get("state_dict", checkpoint)
    ema = checkpoint.get("ema")
    if not isinstance(ema, dict) or "shadow_params" not in ema:
        return state

    canonical_names = []
    alias_to_canonical = {}
    storage_to_name = {}
    for name, tensor in state.items():
        if not name.startswith("backbone."):
            continue
        storage_key = _tensor_storage_key(tensor)
        canonical_name = storage_to_name.get(storage_key)
        if canonical_name is None:
            storage_to_name[storage_key] = name
            canonical_names.append(name)
        else:
            alias_to_canonical[name] = canonical_name

    shadows = ema["shadow_params"]
    if len(canonical_names) != len(shadows):
        raise ValueError(
            "FRIGID EMA/backbone parameter count mismatch: "
            f"{len(shadows)} != {len(canonical_names)}"
        )

    effective = dict(state)
    for name, shadow in zip(canonical_names, shadows):
        if state[name].shape != shadow.shape:
            raise ValueError(
                f"FRIGID EMA shape mismatch for {name}: "
                f"{tuple(shadow.shape)} != {tuple(state[name].shape)}"
            )
        effective[name] = shadow
    for alias, canonical in alias_to_canonical.items():
        effective[alias] = effective[canonical]
    return effective


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marlin_decoder_from_checkpoint(
    checkpoint: dict,
    *,
    use_ema: bool,
) -> MarlinDecoder:
    config_values = dict(checkpoint["hyper_parameters"]["config"])
    checkpoint_state = checkpoint["state_dict"]
    if (
        "decoder.conditioner.fingerprint.layer_norm.weight"
        not in checkpoint_state
    ):
        config_values["fingerprint_layer_norm"] = False
    source = MarlinDecoder(MarlinDecoderConfig(**config_values))
    decoder_state = {
        key.removeprefix("decoder."): value
        for key, value in checkpoint_state.items()
        if key.startswith("decoder.")
    }
    source.load_state_dict(decoder_state, strict=True)
    if use_ema and checkpoint.get("ema"):
        parameters = [
            parameter for parameter in source.parameters()
            if parameter.requires_grad
        ]
        shadows = checkpoint["ema"]["shadow_params"]
        if len(parameters) != len(shadows):
            raise ValueError(
                "MARLIN EMA parameter count mismatch: "
                f"{len(parameters)} != {len(shadows)}"
            )
        with torch.no_grad():
            for index, (parameter, shadow) in enumerate(
                zip(parameters, shadows)
            ):
                if parameter.shape != shadow.shape:
                    raise ValueError(
                        "MARLIN EMA shape mismatch at position "
                        f"{index}: {tuple(parameter.shape)} != "
                        f"{tuple(shadow.shape)}"
                    )
                parameter.copy_(shadow)
    return source


def load_marlin_decoder_weights(
    model: MarlinDecoder,
    checkpoint_path: str | Path,
    *,
    architecture_upgrade: bool = False,
    use_ema: bool = True,
) -> dict:
    """Load a MARLIN checkpoint, optionally adapting only known paper modules."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    source = _marlin_decoder_from_checkpoint(checkpoint, use_ema=use_ema)
    source_state = source.state_dict()
    target_state = model.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    shape_mismatches = sorted(
        key
        for key in set(source_state) & set(target_state)
        if source_state[key].shape != target_state[key].shape
    )
    changed = missing + unexpected + shape_mismatches
    if changed and not architecture_upgrade:
        raise ValueError(
            "MARLIN checkpoint architecture mismatch; set the explicit "
            f"architecture-upgrade mode only for known paper modules: {changed}"
        )
    forbidden = [
        key
        for key in changed
        if not key.startswith(_ARCHITECTURE_UPGRADE_PREFIXES)
    ]
    if forbidden:
        raise ValueError(
            "MARLIN architecture upgrade contains non-whitelisted keys: "
            f"{forbidden}"
        )
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and value.shape == target_state[key].shape
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    if sorted(incompatible.missing_keys) != missing:
        raise AssertionError("MARLIN upgrade missing-key accounting changed")
    if incompatible.unexpected_keys:
        raise AssertionError("filtered MARLIN upgrade has unexpected keys")
    if model.token_type_embedding is not None and any(
        key.startswith("token_type_embedding.") for key in missing
    ):
        torch.nn.init.zeros_(model.token_type_embedding.weight)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_global_step": int(checkpoint.get("global_step", -1)),
        "source_weights": "ema" if use_ema and checkpoint.get("ema") else "raw",
        "architecture_upgrade": architecture_upgrade,
        "loaded_keys": len(compatible),
        "initialized_target_keys": missing,
        "ignored_source_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "optimizer_state_restored": False,
        "trainer_loop_state_restored": False,
    }


def load_frigid_decoder(
    model: MarlinDecoder,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, bool | float | int | str]:
    """Load all architecture-compatible FRIGID weights and leave mass tokens new."""
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
        raise ValueError(
            f"FRIGID checkpoint SHA-256 is {checkpoint_sha256}; "
            f"expected {expected_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _ema_backbone_state(checkpoint)
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
        "backbone_weights": "ema" if "ema" in checkpoint else "raw",
        "frigid_ema_decay": checkpoint.get("ema", {}).get("decay", 0.0),
        "frigid_ema_updates": checkpoint.get("ema", {}).get("num_updates", 0),
    }
