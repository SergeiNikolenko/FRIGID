"""Checkpoint I/O for MARLIN Expanding Flows and Flow Maps."""

from __future__ import annotations

from pathlib import Path

import torch

from marlin.expanding import ExpandingFlowConfig, ExpandingMarlinModel
from marlin.model import MarlinDecoderConfig


def expanding_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    use_ema: bool = True,
) -> tuple[ExpandingMarlinModel, str]:
    """Load an EFlow/EFM model and its declared training stage."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hyperparameters = checkpoint.get("hyper_parameters", {})
    if "decoder_config" not in hyperparameters or "flow_config" not in hyperparameters:
        raise ValueError(f"not an expanding MARLIN checkpoint: {path}")
    decoder_config = MarlinDecoderConfig(**hyperparameters["decoder_config"])
    flow_config = ExpandingFlowConfig(**hyperparameters["flow_config"])
    stage = str(hyperparameters.get("stage", "eflow"))
    if stage not in {"eflow", "efm"}:
        raise ValueError(f"invalid expanding checkpoint stage {stage!r}")
    model = ExpandingMarlinModel(decoder_config, flow_config)
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    if use_ema:
        ema = checkpoint.get("ema")
        if not ema:
            raise ValueError(f"checkpoint has no EMA state: {path}")
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        shadows = ema["shadow_params"]
        if len(parameters) != len(shadows):
            raise ValueError(
                "EMA parameter count mismatch: "
                f"model={len(parameters)} checkpoint={len(shadows)}"
            )
        with torch.no_grad():
            for parameter, shadow in zip(parameters, shadows):
                if parameter.shape != shadow.shape:
                    raise ValueError(
                        "EMA shape mismatch: "
                        f"model={tuple(parameter.shape)} "
                        f"checkpoint={tuple(shadow.shape)}"
                    )
                parameter.copy_(shadow)
    return model.eval().to(device), stage


def expanding_raw_state(checkpoint_path: str | Path) -> tuple[dict, str]:
    """Return model state and stage without constructing the architecture."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hyperparameters = checkpoint.get("hyper_parameters", {})
    stage = str(hyperparameters.get("stage", ""))
    if stage not in {"eflow", "efm"}:
        raise ValueError(f"not an expanding MARLIN checkpoint: {path}")
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    return state, stage
