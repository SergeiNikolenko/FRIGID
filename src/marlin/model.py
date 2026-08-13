"""Block-causal masked-diffusion decoder for the MARLIN reproduction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from marlin.conditioning import MarlinConditioner


# How the diffusion time is drawn. ``per_block_iid`` is what every MARLIN run
# recorded so far used; ``per_sequence_antithetic`` is what the released
# checkpoint was trained with.
TIME_SAMPLING_MODES = ("per_block_iid", "per_sequence_antithetic")

# How the weighted token losses are reduced to a scalar. ``block_mean`` is the
# historical per-example sum over ceil(n_valid / block_width); ``token_mean`` is
# the checkpoint's ``global_mean_loss``.
LOSS_REDUCTIONS = ("block_mean", "token_mean")


def antithetic_uniform_times(
    count: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    sampling_eps: float = 1e-3,
) -> torch.Tensor:
    """One stratified, antithetic diffusion time per sequence.

    A literal port of the time distribution the released checkpoint used
    (``src/dlm/utils/utils_moco.py:82-86``, reached from
    ``src/dlm/model.py:222-223``): draw ``count`` uniforms, compress each into
    its own 1/count-wide stratum, offset it by that stratum's start, and map the
    result onto ``[sampling_eps, 1]``. The strata make a batch cover the time
    axis instead of clustering, which is what removes most of the variance the
    ``1/t`` importance weight injects.
    """
    if count <= 0:
        raise ValueError("antithetic time sampling needs at least one sequence")
    if not 0.0 < sampling_eps < 1.0:
        raise ValueError("sampling_eps must be in (0, 1)")
    draws = torch.rand(count, device=device, generator=generator)
    offset = torch.arange(count, device=device, dtype=draws.dtype) / count
    times = (draws / count + offset) % 1
    return (1 - sampling_eps) * times + sampling_eps


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
    layer_norm_eps: float = 1e-12
    cross_attention_layer_norm_eps: float = 1e-5
    fingerprint_layer_norm_eps: float = 1e-5
    fingerprint_layer_norm: bool = True
    fingerprint_self_attention_layers: int = 0
    frigid_compatible_layer_order: bool = False
    layer0_long_residual_scale: float = 0.0
    eos_token_id: int = 2
    mask_token_id: int = 4
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.layer0_long_residual_scale)
            or self.layer0_long_residual_scale < 0.0
        ):
            raise ValueError(
                "layer0_long_residual_scale must be finite and non-negative"
            )
        if self.layer0_long_residual_scale and self.num_layers < 1:
            raise ValueError(
                "layer0_long_residual_scale requires at least one decoder layer"
            )


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
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(
            config.hidden_size, eps=config.cross_attention_layer_norm_eps
        )
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.frigid_compatible_layer_order = config.frigid_compatible_layer_order

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
        if self.frigid_compatible_layer_order:
            update = self.linear2(self.dropout(F.gelu(self.linear1(hidden))))
            hidden = self.norm3(hidden + self.dropout(update))
            update, _ = self.cross_attention(
                hidden,
                condition,
                condition,
                key_padding_mask=condition_padding_mask,
                need_weights=False,
            )
            return self.norm2(hidden + self.dropout(update))
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
        if config.frigid_compatible_layer_order:
            self.token_type_embedding = nn.Embedding(2, config.hidden_size)
            self.embedding_norm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
        else:
            self.token_type_embedding = None
            self.embedding_norm = nn.Identity()
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.conditioner = MarlinConditioner(
            config.hidden_size,
            config.fingerprint_bits,
            num_heads=config.num_heads,
            fingerprint_self_attention_layers=config.fingerprint_self_attention_layers,
            dropout=config.dropout,
            layer_norm_eps=config.fingerprint_layer_norm_eps,
            fingerprint_layer_norm=config.fingerprint_layer_norm,
        )
        self.layers = nn.ModuleList(MarlinDecoderLayer(config) for _ in range(config.num_layers))
        self.prediction_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.prediction_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.output.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        isotope_ratios: torch.Tensor | None = None,
        *,
        include_mass_conditioning: bool = True,
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
            include_mass_conditioning=include_mass_conditioning,
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
        include_mass_conditioning: bool,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.token_embedding(input_ids) + self.position_embedding(
            positions
        ).unsqueeze(0)
        if self.token_type_embedding is not None:
            hidden = hidden + self.token_type_embedding(torch.zeros_like(input_ids))
            hidden = self.embedding_dropout(self.embedding_norm(hidden))
        condition, condition_mask = self.conditioner(
            precursor_mass,
            fingerprint,
            isotope_ratios,
            include_mass=include_mass_conditioning,
        )
        padding_mask = input_ids.eq(self.config.pad_token_id)
        layer0_hidden = None
        for layer_index, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                condition,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                condition_padding_mask=~condition_mask,
            )
            if layer_index == 0:
                layer0_hidden = hidden
        if self.config.layer0_long_residual_scale:
            if layer0_hidden is None:
                raise RuntimeError("layer0 long residual has no source layer")
            hidden = (
                hidden
                + self.config.layer0_long_residual_scale * layer0_hidden
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
        *,
        include_mass_conditioning: bool = True,
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
            include_mass_conditioning=include_mass_conditioning,
            attention_mask=mask,
        )
        return logits[:, length:]

    def sampling_logits(
        self,
        input_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        isotope_ratios: torch.Tensor | None = None,
        *,
        include_mass_conditioning: bool = True,
    ) -> torch.Tensor:
        """Predict the noisy stream using committed earlier blocks as clean context."""
        return self.two_stream_logits(
            input_ids,
            input_ids,
            precursor_mass,
            fingerprint,
            isotope_ratios,
            include_mass_conditioning=include_mass_conditioning,
        )

    @torch.no_grad()
    def sample_confusions(
        self,
        clean_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        isotope_ratios: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return, per position, the token this model would most plausibly
        write instead of the gold one.

        One forward pass with the whole content masked puts every block in the
        state the sampler is in when it commits that block's first token; the
        runner-up there is the mistake the sampler actually makes at temperature
        1, which is a far more informative corruption than a uniform draw. No
        gradient flows through it.
        """
        valid = clean_ids.ne(self.config.pad_token_id)
        valid[:, 0] = False
        masked_ids = clean_ids.masked_fill(valid, self.config.mask_token_id)
        logits = self.two_stream_logits(
            clean_ids,
            masked_ids,
            precursor_mass,
            fingerprint,
            isotope_ratios,
            include_mass_conditioning=True,
        )
        top2 = logits.topk(2, dim=-1).indices
        return torch.where(top2[..., 0].eq(clean_ids), top2[..., 1], top2[..., 0])

    def diffusion_loss(
        self,
        clean_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        *,
        isotope_ratios: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        eos_loss_weight: float = 1.0,
        eos_mask_probability: float = 0.0,
        balanced_token_loss_alpha: float = 0.0,
        token_loss_weight_max: float = 20.0,
        full_sequence_mask_probability: float = 0.0,
    ) -> torch.Tensor:
        """Continuous-time absorbing NELBO, sampled independently per block."""
        loss, _ = self.diffusion_objective(
            clean_ids,
            precursor_mass,
            fingerprint,
            isotope_ratios=isotope_ratios,
            generator=generator,
            eos_loss_weight=eos_loss_weight,
            eos_mask_probability=eos_mask_probability,
            balanced_token_loss_alpha=balanced_token_loss_alpha,
            token_loss_weight_max=token_loss_weight_max,
            full_sequence_mask_probability=full_sequence_mask_probability,
            collect_metrics=False,
        )
        return loss

    def diffusion_objective(
        self,
        clean_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        *,
        isotope_ratios: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        eos_loss_weight: float = 1.0,
        eos_mask_probability: float = 0.0,
        balanced_token_loss_alpha: float = 0.0,
        token_loss_weight_max: float = 20.0,
        full_sequence_mask_probability: float = 0.0,
        context_corruption_probability: float = 0.0,
        context_corruption_min_fraction: float = 0.05,
        context_corruption_max_fraction: float = 0.25,
        restoration_loss_weight: float = 0.0,
        confusion_ids: torch.Tensor | None = None,
        time_sampling: str = "per_block_iid",
        time_sampling_eps: float = 1e-3,
        loss_reduction: str = "block_mean",
        collect_metrics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the NELBO and optional reconstruction diagnostics.

        ``context_corruption_probability`` teaches the decoder to survive its own
        mistakes. At sampling time ``sampling_logits`` feeds the *committed*
        prefix into the clean stream, so a wrong committed token becomes the
        clean context of every later block; during training the clean stream is
        always gold, and nothing in the objective ever shows the model a wrong
        visible token. With this option a selected example has a fraction of its
        clean-stream content positions replaced by ``confusion_ids`` (the
        model's own runner-up tokens), while every cross-entropy target stays
        gold. ``restoration_loss_weight`` additionally scores the corrupted
        positions that were not masked, which is the explicit
        "this token is wrong, here is the right one" signal.

        ``time_sampling`` and ``loss_reduction`` exist because the released
        checkpoint was not trained the way this function samples or reduces.

        - ``per_block_iid`` (the historical default) draws one diffusion time
          per (example, block), so a 256-token sequence carries 32 independent
          draws and the heavy 1/t tail is sampled 32x per row. ``src/dlm`` draws
          **one t per sequence** with antithetic stratification over
          ``[time_sampling_eps, 1]`` (``src/dlm/model.py:222-223``,
          ``src/dlm/utils/utils_moco.py:82-86``), which is what
          ``per_sequence_antithetic`` reproduces.
        - ``block_mean`` (the historical default) divides each example's summed
          token loss by ``ceil(n_valid / block_width)``, which is about ``8x`` a
          token mean at ``block_width = 8``. ``token_mean`` is the checkpoint's
          ``global_mean_loss``: the whole batch's weighted token losses summed
          and divided by the number of valid tokens in the batch
          (``src/dlm/model.py:953``, ``bionemo/moco`` MDLM ``loss(...,
          global_mean=True)``).

        Both defaults are the historical behaviour, so a run that passes
        neither is bit-identical to every run recorded before they existed.
        """
        if eos_loss_weight <= 0:
            raise ValueError("eos_loss_weight must be positive")
        if not 0.0 <= eos_mask_probability <= 1.0:
            raise ValueError("eos_mask_probability must be in [0, 1]")
        if not 0.0 <= full_sequence_mask_probability <= 1.0:
            raise ValueError("full_sequence_mask_probability must be in [0, 1]")
        if balanced_token_loss_alpha < 0:
            raise ValueError("balanced_token_loss_alpha must be non-negative")
        if token_loss_weight_max < 1:
            raise ValueError("token_loss_weight_max must be at least 1")
        if not 0.0 <= context_corruption_probability <= 1.0:
            raise ValueError("context_corruption_probability must be in [0, 1]")
        if not (
            0.0
            <= context_corruption_min_fraction
            <= context_corruption_max_fraction
            <= 1.0
        ):
            raise ValueError(
                "context corruption fractions must satisfy 0 <= min <= max <= 1"
            )
        if restoration_loss_weight < 0:
            raise ValueError("restoration_loss_weight must be non-negative")
        if context_corruption_probability and confusion_ids is None:
            raise ValueError(
                "context corruption needs confusion_ids; call sample_confusions"
            )
        if time_sampling not in TIME_SAMPLING_MODES:
            raise ValueError(
                f"unknown time_sampling {time_sampling!r}; expected one of "
                f"{', '.join(TIME_SAMPLING_MODES)}"
            )
        if not 0.0 < time_sampling_eps < 1.0:
            raise ValueError("time_sampling_eps must be in (0, 1)")
        if loss_reduction not in LOSS_REDUCTIONS:
            raise ValueError(
                f"unknown loss_reduction {loss_reduction!r}; expected one of "
                f"{', '.join(LOSS_REDUCTIONS)}"
            )
        valid = clean_ids.ne(self.config.pad_token_id)
        valid[:, 0] = False
        batch, length = clean_ids.shape
        block_ids = (torch.arange(length, device=clean_ids.device) - 1).clamp_min(0).div(
            self.config.block_width,
            rounding_mode="floor",
        )
        block_count = int(block_ids.max().item()) + 1
        if time_sampling == "per_block_iid":
            times = torch.rand((batch, block_count), device=clean_ids.device, generator=generator).clamp_min(1e-4)
            probabilities = times[:, block_ids]
        else:
            times = antithetic_uniform_times(
                batch,
                device=clean_ids.device,
                generator=generator,
                sampling_eps=time_sampling_eps,
            )
            probabilities = times.unsqueeze(1).expand(batch, length)
        masked = (torch.rand(clean_ids.shape, device=clean_ids.device, generator=generator) < probabilities) & valid
        full_sequence_masked = torch.zeros((batch,), dtype=torch.bool, device=clean_ids.device)
        if full_sequence_mask_probability:
            full_sequence_masked = (
                torch.rand((batch,), device=clean_ids.device, generator=generator)
                < full_sequence_mask_probability
            )
            masked = masked | (full_sequence_masked.unsqueeze(1) & valid)
        loss_probabilities = probabilities.masked_fill(
            full_sequence_masked.unsqueeze(1) & valid,
            1.0,
        )
        eos_targets = valid & clean_ids.eq(self.config.eos_token_id)
        if eos_mask_probability:
            eos_masked = (
                torch.rand(clean_ids.shape, device=clean_ids.device, generator=generator)
                < eos_mask_probability
            ) & eos_targets
            masked = masked | eos_masked
        corrupted = torch.zeros_like(valid)
        context_ids = clean_ids
        if context_corruption_probability:
            if confusion_ids.shape != clean_ids.shape:
                raise ValueError("confusion_ids must match clean_ids in shape")
            selected = (
                torch.rand((batch, 1), device=clean_ids.device, generator=generator)
                < context_corruption_probability
            )
            span = context_corruption_max_fraction - context_corruption_min_fraction
            fractions = context_corruption_min_fraction + span * torch.rand(
                (batch, 1), device=clean_ids.device, generator=generator
            )
            corrupted = (
                torch.rand(clean_ids.shape, device=clean_ids.device, generator=generator)
                < fractions
            ) & valid & selected
            # A confusion that happens to equal the gold token corrupts nothing,
            # so it must not be scored as a restoration target either.
            corrupted &= confusion_ids.ne(clean_ids)
            context_ids = torch.where(corrupted, confusion_ids.detach(), clean_ids)
        noised = context_ids.masked_fill(masked, self.config.mask_token_id)
        logits = self.two_stream_logits(
            context_ids,
            noised,
            precursor_mass,
            fingerprint,
            isotope_ratios,
            include_mass_conditioning=True,
        )
        losses = F.cross_entropy(logits.transpose(1, 2), clean_ids, reduction="none")
        target_weights = torch.ones_like(losses)
        if balanced_token_loss_alpha:
            content_targets = valid & clean_ids.ne(self.config.eos_token_id)
            valid_targets = clean_ids[content_targets]
            counts = torch.bincount(
                valid_targets,
                minlength=self.config.vocab_size,
            ).to(losses.dtype)
            positive_counts = counts[counts > 0]
            mean_count = positive_counts.mean() if positive_counts.numel() else counts.new_tensor(1.0)
            token_weights = torch.ones_like(counts)
            token_weights[counts > 0] = (mean_count / counts[counts > 0]).pow(
                balanced_token_loss_alpha
            )
            token_weights = token_weights.clamp(max=token_loss_weight_max)
            target_weights = target_weights.masked_scatter(
                content_targets,
                token_weights[clean_ids[content_targets]],
            )
        if eos_loss_weight != 1.0:
            target_weights = target_weights.masked_fill(eos_targets, eos_loss_weight)
        weights = loss_probabilities.reciprocal()
        weighted = losses * target_weights * weights * masked
        if loss_reduction == "block_mean":
            valid_block_counts = valid.sum(dim=1).add(self.config.block_width - 1).div(
                self.config.block_width,
                rounding_mode="floor",
            ).clamp_min(1)
            per_example = weighted.sum(dim=1) / valid_block_counts
            loss = per_example.mean()
        else:
            # The checkpoint's global_mean_loss: one sum over the whole batch
            # divided by the batch's valid-token count, so a long sequence
            # contributes more tokens rather than being renormalised away.
            loss = weighted.sum() / valid.sum().clamp_min(1)
        # Visible-but-wrong positions carry no 1/t importance weight - they are
        # not part of the absorbing NELBO - so they enter as a separate mean
        # with its own coefficient and leave the NELBO estimator untouched.
        restoration = corrupted & ~masked
        restoration_count = restoration.sum()
        restoration_loss = torch.where(
            restoration_count > 0,
            (losses * restoration).sum() / restoration_count.clamp_min(1),
            losses.new_tensor(0.0),
        )
        if restoration_loss_weight:
            loss = loss + restoration_loss_weight * restoration_loss
        if not collect_metrics:
            return loss, {}

        masked_count = masked.sum().clamp_min(1)
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(clean_ids) & masked
        top_k = min(10, self.config.vocab_size)
        top10 = logits.topk(top_k, dim=-1).indices.eq(clean_ids.unsqueeze(-1)).any(dim=-1)
        log_probabilities = logits.float().log_softmax(dim=-1)
        probabilities = log_probabilities.exp()
        target_probabilities = probabilities.gather(
            -1, clean_ids.unsqueeze(-1)
        ).squeeze(-1)
        masked_probabilities = probabilities[masked]
        masked_entropy = -(masked_probabilities * log_probabilities[masked]).sum(dim=-1).mean()
        masked_top1_confidence = masked_probabilities.max(dim=-1).values.mean()
        masked_eos_targets = masked & eos_targets
        eos_target_count = masked_eos_targets.sum()
        eos_target_probability = torch.where(
            eos_target_count > 0,
            target_probabilities[masked_eos_targets].mean(),
            target_probabilities.new_tensor(0.0),
        )
        eos_logits = logits[..., self.config.eos_token_id]
        eos_target_rank = torch.where(
            eos_target_count > 0,
            (logits[masked_eos_targets] > eos_logits[masked_eos_targets].unsqueeze(-1))
            .sum(dim=-1)
            .add(1)
            .float()
            .mean(),
            eos_logits.new_tensor(0.0),
        )
        masked_nll = (losses * masked).sum() / masked_count
        first_block_masked = masked & block_ids.eq(0).unsqueeze(0)
        first_block_masked_count = first_block_masked.sum()
        first_block_denominator = first_block_masked_count.clamp_min(1)
        conditioning_metrics = {
            "first_block_conditioning_nll_gain": losses.new_tensor(0.0),
            "first_block_conditioning_target_probability_gain": losses.new_tensor(
                0.0
            ),
            "first_block_conditioning_token_accuracy_top1_gain": losses.new_tensor(
                0.0
            ),
            "first_block_conditioning_token_accuracy_top10_gain": losses.new_tensor(
                0.0
            ),
        }
        if batch > 1 and first_block_masked_count > 0:
            # Compare matched and cyclically shuffled molecular conditions on
            # the exact same noised tokens. Eval mode removes dropout as a
            # source of variance; no gradients or RNG state affect training.
            was_training = self.training
            self.eval()
            try:
                with torch.no_grad():
                    matched_logits = self.two_stream_logits(
                        clean_ids,
                        noised,
                        precursor_mass,
                        fingerprint,
                        isotope_ratios,
                        include_mass_conditioning=True,
                    )
                    shuffled_logits = self.two_stream_logits(
                        clean_ids,
                        noised,
                        precursor_mass.roll(1, dims=0),
                        fingerprint.roll(1, dims=0),
                        (
                            isotope_ratios.roll(1, dims=0)
                            if isotope_ratios is not None
                            else None
                        ),
                        include_mass_conditioning=True,
                    )
            finally:
                self.train(was_training)

            matched_log_probabilities = matched_logits.float().log_softmax(dim=-1)
            shuffled_log_probabilities = shuffled_logits.float().log_softmax(dim=-1)
            matched_target_probabilities = matched_log_probabilities.gather(
                -1, clean_ids.unsqueeze(-1)
            ).squeeze(-1).exp()
            shuffled_target_probabilities = shuffled_log_probabilities.gather(
                -1, clean_ids.unsqueeze(-1)
            ).squeeze(-1).exp()
            matched_losses = F.cross_entropy(
                matched_logits.transpose(1, 2),
                clean_ids,
                reduction="none",
            )
            shuffled_losses = F.cross_entropy(
                shuffled_logits.transpose(1, 2),
                clean_ids,
                reduction="none",
            )
            matched_top1 = matched_logits.argmax(dim=-1).eq(clean_ids)
            shuffled_top1 = shuffled_logits.argmax(dim=-1).eq(clean_ids)
            matched_top10 = (
                matched_logits.topk(top_k, dim=-1)
                .indices.eq(clean_ids.unsqueeze(-1))
                .any(dim=-1)
            )
            shuffled_top10 = (
                shuffled_logits.topk(top_k, dim=-1)
                .indices.eq(clean_ids.unsqueeze(-1))
                .any(dim=-1)
            )
            conditioning_metrics = {
                "first_block_conditioning_nll_gain": (
                    (shuffled_losses - matched_losses) * first_block_masked
                ).sum()
                / first_block_denominator,
                "first_block_conditioning_target_probability_gain": (
                    (matched_target_probabilities - shuffled_target_probabilities)
                    * first_block_masked
                ).sum()
                / first_block_denominator,
                "first_block_conditioning_token_accuracy_top1_gain": (
                    (matched_top1.float() - shuffled_top1.float())
                    * first_block_masked
                ).sum()
                / first_block_denominator,
                "first_block_conditioning_token_accuracy_top10_gain": (
                    (matched_top10.float() - shuffled_top10.float())
                    * first_block_masked
                ).sum()
                / first_block_denominator,
            }
        metrics = {
            "masked_token_accuracy_top1": correct.sum() / masked_count,
            "masked_token_accuracy_top10": (top10 & masked).sum() / masked_count,
            "masked_target_probability": (target_probabilities * masked).sum()
            / masked_count,
            "masked_token_nll": masked_nll,
            "masked_token_perplexity": masked_nll.clamp_max(20).exp(),
            "masked_prediction_entropy": masked_entropy,
            "masked_top1_confidence": masked_top1_confidence,
            "masked_argmax_eos_fraction": (
                predictions.eq(self.config.eos_token_id) & masked
            ).sum() / masked_count,
            "masked_eos_target_count": eos_target_count.float(),
            "masked_eos_target_probability": eos_target_probability,
            "masked_eos_target_rank": eos_target_rank,
            "mask_fraction": masked.sum() / valid.sum().clamp_min(1),
            "context_corruption_fraction": corrupted.sum() / valid.sum().clamp_min(1),
            "restoration_loss": restoration_loss,
            "restoration_token_count": restoration_count.float(),
            # Did the model put the gold token back where it saw a wrong one?
            "restoration_token_accuracy_top1": (
                predictions.eq(clean_ids) & restoration
            ).sum() / restoration_count.clamp_min(1),
            # ... or did it simply copy the wrong token it was shown?
            "restoration_copy_rate": (
                predictions.eq(context_ids) & restoration
            ).sum() / restoration_count.clamp_min(1),
            "masked_token_accuracy_top1_uncorrupted_rows": (
                correct & masked & ~corrupted.any(dim=1, keepdim=True)
            ).sum()
            / (masked & ~corrupted.any(dim=1, keepdim=True)).sum().clamp_min(1),
            "full_sequence_mask_fraction": full_sequence_masked.float().mean(),
            "masked_sequence_accuracy": (
                (correct | ~masked).all(dim=1) & masked.any(dim=1)
            ).float().sum()
            / masked.any(dim=1).sum().clamp_min(1),
            # The first content block sees BOS plus mass/fingerprint/isotope
            # conditioning, but no clean molecular prefix. These metrics
            # therefore separate conditioning progress from easy continuation.
            "first_block_masked_count": first_block_masked_count.float(),
            "first_block_masked_token_accuracy_top1": (
                correct & first_block_masked
            ).sum()
            / first_block_denominator,
            "first_block_masked_token_accuracy_top10": (
                top10 & first_block_masked
            ).sum()
            / first_block_denominator,
            "first_block_masked_target_probability": (
                target_probabilities * first_block_masked
            ).sum()
            / first_block_denominator,
            "first_block_masked_token_nll": (
                losses * first_block_masked
            ).sum()
            / first_block_denominator,
            **conditioning_metrics,
        }
        return loss, metrics
