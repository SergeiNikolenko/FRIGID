#!/usr/bin/env python3
"""Compare transferred MARLIN logits with the released FRIGID backbone."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from dlm.model import DLM
from marlin.model import MarlinDecoder, MarlinDecoderConfig
from marlin.warm_start import load_frigid_decoder, sha256_file


DEFAULT_CHECKPOINT = Path(
    "/mnt/netstorage/nikolenko/marlin/checkpoints/frigid/DLM.ckpt"
)
DEFAULT_CACHE = Path("/mnt/netstorage/nikolenko/marlin/cache/huggingface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def copy_frigid_ema_to_backbone(model: DLM, checkpoint: dict) -> None:
    shadows = checkpoint["ema"]["shadow_params"]
    parameters = list(model.backbone.parameters())
    if len(parameters) != len(shadows):
        raise ValueError(
            "official FRIGID EMA parameter count mismatch: "
            f"{len(parameters)} != {len(shadows)}"
        )
    with torch.no_grad():
        for index, (parameter, shadow) in enumerate(zip(parameters, shadows)):
            if parameter.shape != shadow.shape:
                raise ValueError(
                    "official FRIGID EMA shape mismatch at position "
                    f"{index}: {tuple(parameter.shape)} != {tuple(shadow.shape)}"
                )
            parameter.copy_(shadow)


def marlin_config_from_frigid(checkpoint: dict) -> MarlinDecoderConfig:
    config = checkpoint["hyper_parameters"]["config"].model
    return MarlinDecoderConfig(
        vocab_size=int(config.vocab_size),
        hidden_size=int(config.hidden_size),
        num_layers=int(config.num_hidden_layers),
        num_heads=int(config.num_attention_heads),
        intermediate_size=int(config.intermediate_size),
        max_length=int(config.max_position_embeddings),
        block_width=8,
        fingerprint_bits=int(config.fingerprint_bits),
        dropout=float(config.hidden_dropout_prob),
        layer_norm_eps=float(config.layer_norm_eps),
        cross_attention_layer_norm_eps=1e-5,
        fingerprint_layer_norm_eps=1e-5,
        fingerprint_layer_norm=True,
        fingerprint_self_attention_layers=int(
            config.fingerprint_num_self_attention_layers
        ),
        frigid_compatible_layer_order=True,
        eos_token_id=2,
        mask_token_id=4,
        pad_token_id=int(config.pad_token_id),
    )


def main() -> None:
    args = parse_args()
    if args.atol <= 0 or args.rtol <= 0:
        raise ValueError("--atol and --rtol must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["hyper_parameters"]["config"]
    config.data.hf_cache_dir = str(args.hf_cache)

    official = DLM(config)
    official.load_state_dict(checkpoint["state_dict"], strict=True)
    copy_frigid_ema_to_backbone(official, checkpoint)
    official.eval().to(device)

    marlin = MarlinDecoder(marlin_config_from_frigid(checkpoint))
    transfer = load_frigid_decoder(
        marlin,
        args.checkpoint,
        expected_sha256=sha256_file(args.checkpoint),
    )
    marlin.eval().to(device)

    input_ids = torch.tensor(
        [
            [1, 4, 17, 205, 4, 91, 732, 4, 311, 2],
            [1, 19, 4, 4, 511, 87, 4, 131, 29, 2],
        ],
        dtype=torch.long,
        device=device,
    )
    fingerprint = torch.zeros((2, 4096), device=device)
    fingerprint[0, [1, 7, 31, 255, 1023, 2047, 4095]] = 1.0
    fingerprint[1, [0, 11, 47, 128, 511, 777, 1555]] = 1.0
    attention = torch.ones_like(input_ids)

    with torch.inference_mode():
        official_condition, official_condition_mask = (
            official.fingerprint_conditioner.encode_fingerprint(fingerprint)
        )
        marlin_condition, marlin_condition_mask = (
            marlin.conditioner.fingerprint(fingerprint)
        )
        official_logits = official.backbone(
            input_ids=input_ids,
            attention_mask=attention,
            condition_embeddings=official_condition,
            condition_mask=official_condition_mask,
        ).logits
        marlin_logits = marlin._forward_with_mask(
            input_ids,
            torch.zeros((1,), device=device),
            fingerprint,
            None,
            positions=torch.arange(input_ids.shape[1], device=device),
            include_mass_conditioning=False,
            attention_mask=torch.zeros(
                (input_ids.shape[1], input_ids.shape[1]),
                dtype=torch.bool,
                device=device,
            ),
        )

    # FRIGID uses unsorted ``topk`` while MARLIN uses ascending ``nonzero``.
    # Set attention and decoder cross-attention are permutation invariant, so
    # compare the conditioning token sets rather than their storage order.
    condition_set_errors = []
    for row in range(fingerprint.shape[0]):
        official_valid = official_condition[
            row, official_condition_mask[row].bool()
        ]
        marlin_valid = marlin_condition[row, marlin_condition_mask[row].bool()]
        distances = torch.cdist(
            official_valid.float(),
            marlin_valid.float(),
            p=float("inf"),
        )
        condition_set_errors.append(
            torch.maximum(
                distances.min(dim=0).values.max(),
                distances.min(dim=1).values.max(),
            )
        )
    condition_set_error = torch.stack(condition_set_errors).max()
    condition_aligned_delta = (official_condition - marlin_condition).abs()
    logits_delta = (official_logits - marlin_logits).abs()
    logits_close = torch.allclose(
        official_logits,
        marlin_logits,
        atol=args.atol,
        rtol=args.rtol,
    )
    condition_set_close = bool(condition_set_error <= args.atol)
    mask_counts_equal = torch.equal(
        official_condition_mask.bool().sum(dim=1),
        marlin_condition_mask.bool().sum(dim=1),
    )
    cosine = torch.nn.functional.cosine_similarity(
        official_logits.flatten(),
        marlin_logits.flatten(),
        dim=0,
    )
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": str(device),
        "atol": args.atol,
        "rtol": args.rtol,
        "condition_token_order": {
            "official": "unsorted_topk",
            "marlin": "ascending_nonzero",
        },
        "condition_mask_counts_equal": mask_counts_equal,
        "condition_set_allclose": condition_set_close,
        "condition_set_max_abs_error": float(condition_set_error),
        "condition_aligned_max_abs_error": float(
            condition_aligned_delta.max()
        ),
        "condition_aligned_mean_abs_error": float(
            condition_aligned_delta.mean()
        ),
        "logits_allclose": logits_close,
        "logits_max_abs_error": float(logits_delta.max()),
        "logits_mean_abs_error": float(logits_delta.mean()),
        "logits_cosine_similarity": float(cosine),
        "passed": bool(
            mask_counts_equal and condition_set_close and logits_close
        ),
        "transfer": transfer,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
