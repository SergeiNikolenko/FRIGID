"""Rank every conditioning lane by teacher-forced top-1, before buying a decode.

Four specs each proposed a paired decode to answer "which fingerprint should we
condition on". Priced against this tree's measured decode cost, that is 87 to 285
shard-hours per pair on a fleet of one contended A100 plus two queue slots, to
resolve an effect the panels cannot resolve: the locked 803 resolves 0.98 pp and
the clean 321 resolves 2.45 pp (``docs/DECODER_PROGRAM.md`` 12.6).

The same ordering question is answerable with forward passes only. Teacher-forced
per-token top-1 is the project's own conditioning anchor --- 0.750 under the true
fingerprint against 0.547 under DreaMS (``docs/DECODER_PROGRAM.md``:53-54) --- and
``marlin.conditioning_probe`` already computes it. This script runs that probe
once per lane on a fixed row set, so the lanes differ in nothing but which
fingerprint file is read.

What it does NOT do: convert a probe number into Exact@1. The map from per-token
top-1 to Exact@1 is steep and non-linear (0.547 -> 1.25%, 0.750 -> 19.00%), so
this ranks hypotheses, it does not price them.

Run it against more than one checkpoint. A lane comparison on a checkpoint that
was adapted for 100,000 steps to one lane's bit vocabulary is biased toward that
lane; the released warm start has been adapted to none of them.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from marlin.conditioning_probe import (  # noqa: E402
    conditioning_probe_metrics,
    probe_batches_from_paths,
)
from marlin.model import MarlinDecoder, MarlinDecoderConfig  # noqa: E402
from marlin.tokenizer import load_safe_tokenizer  # noqa: E402


def load_decoder(checkpoint_path: Path, device: torch.device) -> MarlinDecoder:
    """Load the EMA weights the evaluator would load, and nothing else."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = MarlinDecoderConfig(**checkpoint["hyper_parameters"]["config"])
    state_dict = checkpoint["state_dict"]
    if "decoder.conditioner.fingerprint.layer_norm.weight" not in state_dict:
        config = replace(config, fingerprint_layer_norm=False)
    model = MarlinDecoder(config)
    model.load_state_dict(
        {
            key.removeprefix("decoder."): value
            for key, value in state_dict.items()
            if key.startswith("decoder.")
        },
        strict=True,
    )
    ema = checkpoint.get("ema")
    if ema:
        parameters = [p for p in model.parameters() if p.requires_grad]
        shadows = ema["shadow_params"]
        if len(parameters) != len(shadows):
            raise ValueError(
                f"{checkpoint_path}: {len(parameters)} parameters against "
                f"{len(shadows)} EMA shadows"
            )
        with torch.no_grad():
            for parameter, shadow in zip(parameters, shadows):
                parameter.copy_(shadow.to(parameter.dtype))
    return model.eval().to(device)


def parse_lane(spec: str) -> dict:
    """``name=path::key::threshold`` --- one lane, spelled where it is read."""
    name, _, rest = spec.partition("=")
    path, key, threshold = rest.split("::")
    return {
        "name": name,
        "fingerprints": path,
        "fingerprint_key": key,
        "threshold": float(threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="name=path, repeatable")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--lane", action="append", required=True,
                        help="name=fingerprints.npz::key::threshold, repeatable")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fingerprint-bits", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-seed", action="append", type=int, default=None,
                        help="repeat the probe at each seed to expose its spread")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    probe_seeds = arguments.probe_seed or [0, 1]
    device = torch.device(arguments.device)
    tokenizer = load_safe_tokenizer(arguments.tokenizer)
    lanes = [parse_lane(spec) for spec in arguments.lane]

    # Every lane is built from the same metadata with the same selection seed, so
    # the rows and the batch boundaries are shared and only the fingerprint file
    # differs.
    batches = {}
    for lane in lanes:
        batches[lane["name"]] = probe_batches_from_paths(
            arguments.metadata,
            lane["fingerprints"],
            tokenizer,
            fingerprint_key=lane["fingerprint_key"],
            threshold=lane["threshold"],
            max_length=arguments.max_length,
            fingerprint_bits=arguments.fingerprint_bits,
            size=arguments.size,
            batch_size=arguments.batch_size,
            seed=arguments.seed,
        )
        print(f"lane {lane['name']}: {len(batches[lane['name']])} batches")

    results = []
    for spec in arguments.checkpoint:
        name, _, path = spec.partition("=")
        started = time.time()
        decoder = load_decoder(Path(path), device)
        print(f"checkpoint {name} loaded in {time.time() - started:.1f}s")
        for lane in lanes:
            for probe_seed in probe_seeds:
                start = time.time()
                metrics = conditioning_probe_metrics(
                    decoder, batches[lane["name"]], seed=probe_seed
                )
                row = {
                    "checkpoint": name,
                    "checkpoint_path": path,
                    "lane": lane["name"],
                    "fingerprints": lane["fingerprints"],
                    "fingerprint_key": lane["fingerprint_key"],
                    "threshold": lane["threshold"],
                    "probe_seed": probe_seed,
                    "seconds": round(time.time() - start, 1),
                    **{k: round(float(v), 6) for k, v in metrics.items()},
                }
                results.append(row)
                print(
                    f"  {name:<22} {lane['name']:<22} seed={probe_seed} "
                    f"top1_predicted={row.get('probe_top1_predicted')} "
                    f"top1_true={row.get('probe_top1_true')} "
                    f"({row['seconds']}s)"
                )
        del decoder

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "metadata": arguments.metadata,
                "size": arguments.size,
                "selection_seed": arguments.seed,
                "probe_seeds": probe_seeds,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
