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

    def __init__(
        self,
        bits: int,
        hidden_size: int,
        *,
        num_heads: int = 1,
        num_self_attention_layers: int = 0,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-12,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.embedding = nn.Embedding(bits, hidden_size)
        self.layer_norm = (
            nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            if use_layer_norm
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.self_attention_layers = nn.ModuleList(
            FingerprintSetAttentionLayer(
                hidden_size,
                num_heads,
                dropout,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_self_attention_layers)
        )

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
        tokens = self.dropout(self.layer_norm(self.embedding(indices)))
        # Binary Morgan fingerprints (the warm-start contract) have unit
        # weights, while soft DreaMS probabilities retain confidence on the
        # same active-bit token set without changing parameter shapes.
        weights = torch.gather(fingerprint, 1, indices)
        tokens = tokens * weights.unsqueeze(-1)
        attention_mask = mask.clone()
        empty = ~attention_mask.any(dim=1)
        if empty.any():
            attention_mask[empty, 0] = True
        for layer in self.self_attention_layers:
            tokens = layer(tokens, attention_mask)
        return tokens * mask.unsqueeze(-1), mask


class FingerprintSetAttentionLayer(nn.Module):
    """Permutation-equivariant residual self-attention over active fingerprint bits."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        *,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        update, _ = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        return self.norm(hidden + self.dropout(update))


class MarlinConditioner(nn.Module):
    """Build ``[mass; isotope; active fingerprint bits]`` conditioning tokens."""

    def __init__(
        self,
        hidden_size: int,
        fingerprint_bits: int = 4096,
        num_mass_frequencies: int = 64,
        num_heads: int = 1,
        fingerprint_self_attention_layers: int = 0,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-12,
        fingerprint_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.mass = FourierMassEncoder(hidden_size, num_mass_frequencies)
        self.isotope = nn.Sequential(nn.Linear(2, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.fingerprint = SparseFingerprintEncoder(
            fingerprint_bits,
            hidden_size,
            num_heads=num_heads,
            num_self_attention_layers=fingerprint_self_attention_layers,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            use_layer_norm=fingerprint_layer_norm,
        )

    def forward(
        self,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        isotope_ratios: torch.Tensor | None = None,
        *,
        include_mass: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_tokens = []
        if include_mass:
            prefix_tokens.append(self.mass(precursor_mass).unsqueeze(1))
        if isotope_ratios is not None:
            if isotope_ratios.shape != (fingerprint.shape[0], 2):
                raise ValueError("isotope_ratios must have shape [batch, 2]")
            prefix_tokens.append(self.isotope(isotope_ratios.float()).unsqueeze(1))
        fingerprint_tokens, fingerprint_mask = self.fingerprint(fingerprint)
        if not prefix_tokens:
            empty = ~fingerprint_mask.any(dim=1)
            if empty.any():
                # MultiheadAttention cannot consume an all-masked condition.
                # This constant token carries no mass or fingerprint information.
                fingerprint_tokens = fingerprint_tokens.clone()
                fingerprint_mask = fingerprint_mask.clone()
                fingerprint_tokens[empty, 0] = 0
                fingerprint_mask[empty, 0] = True
        tokens = torch.cat((*prefix_tokens, fingerprint_tokens), dim=1)
        prefix_mask = torch.ones(
            (fingerprint.shape[0], len(prefix_tokens)),
            dtype=torch.bool,
            device=fingerprint.device,
        )
        return tokens, torch.cat((prefix_mask, fingerprint_mask), dim=1)
