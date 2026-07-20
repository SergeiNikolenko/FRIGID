"""Block-causal masked-diffusion decoder for the MARLIN reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from marlin.conditioning import MarlinConditioner


@dataclass(frozen=True)
class MarlinDecoderConfig:
    vocab_size: int = 1880
    hidden_size: int = 896
    num_layers: int = 12
    num_heads: int = 14
    intermediate_size: int = 3584
    max_length: int = 256
    block_width: int = 8
    fingerprint_bits: int = 4096
    dropout: float = 0.1
    mask_token_id: int = 4
    pad_token_id: int = 0


def _bos_aware_block_ids(
    length: int,
    block_width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    positions = torch.arange(length, device=device)
    content_blocks = (positions - 1).clamp_min(0).div(
        block_width, rounding_mode="floor"
    )
    return content_blocks.masked_fill(positions.eq(0), -1)


def block_causal_attention_mask(length: int, block_width: int, device: torch.device | None = None) -> torch.Tensor:
    """Return a mask where a token sees its block and all earlier blocks."""
    blocks = _bos_aware_block_ids(length, block_width, device)
    query_blocks = blocks.reshape(-1, 1)
    key_blocks = blocks.reshape(1, -1)
    return key_blocks > query_blocks


def two_stream_attention_mask(length: int, block_width: int, device: torch.device | None = None) -> torch.Tensor:
    """Mask for clean-prefix/noisy-current training in one forward pass."""
    blocks = _bos_aware_block_ids(length, block_width, device)
    clean_query_blocks = blocks.reshape(-1, 1)
    clean_key_blocks = blocks.reshape(1, -1)
    clean_to_clean = clean_key_blocks > clean_query_blocks
    clean_to_noisy = torch.ones((length, length), dtype=torch.bool, device=device)
    noisy_to_clean = clean_key_blocks >= clean_query_blocks
    noisy_to_noisy = clean_key_blocks != clean_query_blocks
    return torch.cat(
        (
            torch.cat((clean_to_clean, clean_to_noisy), dim=1),
            torch.cat((noisy_to_clean, noisy_to_noisy), dim=1),
        ),
        dim=0,
    )


class MarlinDecoderLayer(nn.Module):
    def __init__(self, config: MarlinDecoderConfig) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            config.hidden_size, config.num_heads, config.dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_size, config.num_heads, config.dropout, batch_first=True
        )
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        padding_mask: torch.Tensor | None,
        condition_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        update, _ = self.self_attention(
            hidden,
            hidden,
            hidden,
            attn_mask=attention_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = self.norm1(hidden + self.dropout(update))
        update, _ = self.cross_attention(
            hidden,
            condition,
            condition,
            key_padding_mask=condition_padding_mask,
            need_weights=False,
        )
        hidden = self.norm2(hidden + self.dropout(update))
        update = self.linear2(self.dropout(F.gelu(self.linear1(hidden))))
        return self.norm3(hidden + self.dropout(update))


class MarlinDecoder(nn.Module):
    def __init__(self, config: MarlinDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)
        self.conditioner = MarlinConditioner(config.hidden_size, config.fingerprint_bits)
        self.layers = nn.ModuleList(MarlinDecoderLayer(config) for _ in range(config.num_layers))
        self.prediction_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.prediction_norm = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.output.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        isotope_ratios: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError("sequence exceeds max_length")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        attention_mask = block_causal_attention_mask(
            input_ids.shape[1], self.config.block_width, input_ids.device
        )
        return self._forward_with_mask(
            input_ids,
            precursor_mass,
            fingerprint,
            isotope_ratios,
            positions=positions,
            attention_mask=attention_mask,
        )

    def _forward_with_mask(
        self,
        input_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        isotope_ratios: torch.Tensor | None,
        *,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)
        condition, condition_mask = self.conditioner(precursor_mass, fingerprint, isotope_ratios)
        padding_mask = input_ids.eq(self.config.pad_token_id)
        for layer in self.layers:
            hidden = layer(
                hidden,
                condition,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                condition_padding_mask=~condition_mask,
            )
        hidden = self.prediction_norm(F.gelu(self.prediction_dense(hidden)))
        return self.output(hidden) + self.output_bias

    def two_stream_logits(
        self,
        clean_ids: torch.Tensor,
        noised_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        isotope_ratios: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict every noisy block against clean earlier blocks in one pass."""
        if clean_ids.shape != noised_ids.shape:
            raise ValueError("clean and noised streams must have equal shapes")
        length = clean_ids.shape[1]
        streams = torch.cat((clean_ids, noised_ids), dim=1)
        positions = torch.arange(length, device=clean_ids.device).repeat(2)
        mask = two_stream_attention_mask(length, self.config.block_width, clean_ids.device)
        logits = self._forward_with_mask(
            streams,
            precursor_mass,
            fingerprint,
            isotope_ratios,
            positions=positions,
            attention_mask=mask,
        )
        return logits[:, length:]

    def sampling_logits(
        self,
        input_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        isotope_ratios: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the noisy stream using committed earlier blocks as clean context."""
        return self.two_stream_logits(
            input_ids,
            input_ids,
            precursor_mass,
            fingerprint,
            isotope_ratios,
        )

    def diffusion_loss(
        self,
        clean_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        *,
        isotope_ratios: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Continuous-time absorbing NELBO, sampled independently per block."""
        valid = clean_ids.ne(self.config.pad_token_id)
        valid[:, 0] = False
        batch, length = clean_ids.shape
        block_ids = (torch.arange(length, device=clean_ids.device) - 1).clamp_min(0).div(
            self.config.block_width,
            rounding_mode="floor",
        )
        block_count = int(block_ids.max().item()) + 1
        times = torch.rand((batch, block_count), device=clean_ids.device, generator=generator).clamp_min(1e-4)
        probabilities = times[:, block_ids]
        masked = (torch.rand(clean_ids.shape, device=clean_ids.device, generator=generator) < probabilities) & valid
        noised = clean_ids.masked_fill(masked, self.config.mask_token_id)
        logits = self.two_stream_logits(
            clean_ids,
            noised,
            precursor_mass,
            fingerprint,
            isotope_ratios,
        )
        losses = F.cross_entropy(logits.transpose(1, 2), clean_ids, reduction="none")
        weights = probabilities.reciprocal()
        valid_block_counts = valid.sum(dim=1).add(self.config.block_width - 1).div(
            self.config.block_width,
            rounding_mode="floor",
        ).clamp_min(1)
        per_example = (losses * weights * masked).sum(dim=1) / valid_block_counts
        return per_example.mean()
