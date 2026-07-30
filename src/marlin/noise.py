"""Fingerprint corruption used for MARLIN training and candidate diversity."""

from __future__ import annotations

import torch


def symmetric_fingerprint_noise(
    fingerprints: torch.Tensor,
    *,
    corruption_probability: float = 0.5,
    min_fraction: float = 0.1,
    max_fraction: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop on-bits and add the same number of off-bits per selected example."""
    if fingerprints.ndim != 2:
        raise ValueError("fingerprints must have shape [batch, bits]")
    if not 0 <= corruption_probability <= 1:
        raise ValueError("corruption_probability must be in [0, 1]")
    if not 0 <= min_fraction <= max_fraction <= 1:
        raise ValueError("noise fractions must satisfy 0 <= min <= max <= 1")

    binary = fingerprints > 0.5
    output = binary.clone()
    device = fingerprints.device
    for row in range(binary.shape[0]):
        if torch.rand((), device=device, generator=generator) >= corruption_probability:
            continue
        on = torch.nonzero(binary[row], as_tuple=False).flatten()
        off = torch.nonzero(~binary[row], as_tuple=False).flatten()
        if on.numel() == 0 or off.numel() == 0:
            continue
        fraction = torch.empty((), device=device).uniform_(
            min_fraction, max_fraction, generator=generator
        )
        count = min(int(round(on.numel() * float(fraction))), off.numel())
        if count == 0:
            continue
        drop = on[torch.randperm(on.numel(), device=device, generator=generator)[:count]]
        add = off[torch.randperm(off.numel(), device=device, generator=generator)[:count]]
        output[row, drop] = False
        output[row, add] = True
    return output.to(dtype=fingerprints.dtype)


def perturb_fingerprint(
    fingerprint: torch.Tensor,
    *,
    dropout: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Swap active and inactive bits for one inference candidate.

    Candidate conditioning must follow the same cardinality-preserving
    corruption family used during training.  The legacy implementation only
    dropped active bits, shrinking the sparse conditioning sequence by roughly
    ``dropout`` for every candidate.
    """
    if not 0 <= dropout <= 1:
        raise ValueError("dropout must be in [0, 1]")
    if fingerprint.ndim not in {1, 2}:
        raise ValueError("fingerprint must have shape [bits] or [batch, bits]")
    batched = fingerprint.unsqueeze(0) if fingerprint.ndim == 1 else fingerprint
    perturbed = symmetric_fingerprint_noise(
        batched,
        corruption_probability=1.0,
        min_fraction=dropout,
        max_fraction=dropout,
        generator=generator,
    )
    return perturbed[0] if fingerprint.ndim == 1 else perturbed
