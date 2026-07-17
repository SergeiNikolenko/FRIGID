"""Precursor-mass, isotope, and sparse-fingerprint conditioning."""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierMassEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_frequencies: int = 64,
        min_frequency: float = 1e-3,
        max_frequency: float = 1.0,
    ) -> None:
        super().__init__()
        frequencies = torch.logspace(
            math.log10(min_frequency), math.log10(max_frequency), num_frequencies
        )
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Sequential(
            nn.Linear(2 * num_frequencies, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, mass: torch.Tensor) -> torch.Tensor:
        angles = mass.float().reshape(-1, 1) * self.frequencies.reshape(1, -1)
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.projection(features)


class SparseFingerprintEncoder(nn.Module):
    """Encode each active Morgan bit as a separate conditioning token."""

    def __init__(self, bits: int, hidden_size: int) -> None:
        super().__init__()
        self.bits = bits
        self.embedding = nn.Embedding(bits, hidden_size)

    def forward(self, fingerprint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if fingerprint.ndim != 2 or fingerprint.shape[1] != self.bits:
            raise ValueError(f"fingerprint must have shape [batch, {self.bits}]")
        active = fingerprint > 0.5
        lengths = active.sum(dim=1)
        width = max(int(lengths.max().item()), 1)
        indices = torch.zeros((fingerprint.shape[0], width), dtype=torch.long, device=fingerprint.device)
        mask = torch.zeros((fingerprint.shape[0], width), dtype=torch.bool, device=fingerprint.device)
        for row in range(fingerprint.shape[0]):
            row_indices = torch.nonzero(active[row], as_tuple=False).flatten()
            indices[row, : row_indices.numel()] = row_indices
            mask[row, : row_indices.numel()] = True
        return self.embedding(indices), mask


class MarlinConditioner(nn.Module):
    """Build ``[mass; isotope; active fingerprint bits]`` conditioning tokens."""

    def __init__(
        self,
        hidden_size: int,
        fingerprint_bits: int = 4096,
        num_mass_frequencies: int = 64,
    ) -> None:
        super().__init__()
        self.mass = FourierMassEncoder(hidden_size, num_mass_frequencies)
        self.isotope = nn.Sequential(nn.Linear(2, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.fingerprint = SparseFingerprintEncoder(fingerprint_bits, hidden_size)
        self.missing_isotope = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        isotope_ratios: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mass_token = self.mass(precursor_mass).unsqueeze(1)
        if isotope_ratios is None:
            isotope_token = self.missing_isotope.expand(fingerprint.shape[0], -1).unsqueeze(1)
        else:
            if isotope_ratios.shape != (fingerprint.shape[0], 2):
                raise ValueError("isotope_ratios must have shape [batch, 2]")
            isotope_token = self.isotope(isotope_ratios.float()).unsqueeze(1)
        fingerprint_tokens, fingerprint_mask = self.fingerprint(fingerprint)
        tokens = torch.cat((mass_token, isotope_token, fingerprint_tokens), dim=1)
        prefix_mask = torch.ones((fingerprint.shape[0], 2), dtype=torch.bool, device=fingerprint.device)
        return tokens, torch.cat((prefix_mask, fingerprint_mask), dim=1)
