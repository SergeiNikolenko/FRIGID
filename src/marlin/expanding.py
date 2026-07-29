"""Conditional discrete Expanding Flows and Flow Maps for MARLIN.

This module adapts the variable-length sequence construction from
``Expanding Flow Maps`` (arXiv:2607.21585) to SAFE molecular strings.  BOS and
EOS are treated as permanent anchors; molecular tokens are inserted between
them, transported from Gaussian vocabulary-space noise, and conditioned on the
MARLIN fingerprint, precursor mass, and isotope envelope.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from torch import nn
from torch.nn import functional as F

from marlin.mass_shell import MassShellConstraint
from marlin.model import MarlinDecoder, MarlinDecoderConfig
from marlin.noise import perturb_fingerprint
from marlin.sampler import MarlinCandidate, MarlinGenerationStats


@dataclass(frozen=True)
class ExpandingFlowConfig:
    """Hyperparameters specific to the expanding interpolant."""

    prior_scale: float = 1.25
    insertion_cutoff: float = 0.5
    insertion_loss_weight: float = 1.0
    time_embedding_size: int = 128
    time_fourier_dim: int = 64
    time_warp_points: int = 2048
    time_warp_quadrature: int = 48
    diagonal_probability: float = 0.75
    boundary_probability: float = 1.0 / 32.0
    adaptive_loss_c: float = 1e-6
    adaptive_loss_r: float = 0.5
    insertion_detach_steps: int = 2000
    sampling_time_floor: float = 1e-3
    eflow_early_time_probability: float = 0.5
    prior_type: str = "gaussian"
    time_warp_type: str = "vocabulary_error"

    def __post_init__(self) -> None:
        if self.prior_scale <= 0:
            raise ValueError("prior_scale must be positive")
        if not 0 < self.insertion_cutoff <= 1:
            raise ValueError("insertion_cutoff must be in (0, 1]")
        if self.insertion_loss_weight < 0:
            raise ValueError("insertion_loss_weight must be non-negative")
        if self.time_embedding_size <= 0 or self.time_fourier_dim < 2:
            raise ValueError("time embedding dimensions must be positive")
        if self.time_fourier_dim % 2:
            raise ValueError("time_fourier_dim must be even")
        if self.time_warp_points < 32 or self.time_warp_quadrature < 8:
            raise ValueError("time-warp resolution is too small")
        if not 0 <= self.diagonal_probability <= 1:
            raise ValueError("diagonal_probability must be in [0, 1]")
        if not 0 <= self.boundary_probability <= 1:
            raise ValueError("boundary_probability must be in [0, 1]")
        if self.adaptive_loss_c <= 0 or self.adaptive_loss_r < 0:
            raise ValueError("adaptive loss parameters are invalid")
        if self.insertion_detach_steps < 0:
            raise ValueError("insertion_detach_steps must be non-negative")
        if not 0 < self.sampling_time_floor < 1:
            raise ValueError("sampling_time_floor must be in (0, 1)")
        if not 0 < self.eflow_early_time_probability < 1:
            raise ValueError("eflow_early_time_probability must be in (0, 1)")
        if self.prior_type not in {"gaussian", "mask"}:
            raise ValueError("prior_type must be 'gaussian' or 'mask'")
        if self.time_warp_type not in {"vocabulary_error", "identity"}:
            raise ValueError(
                "time_warp_type must be 'vocabulary_error' or 'identity'"
            )


class CosineInsertionSchedule:
    """Cosine CDF confined to the paper's insertion window."""

    def __init__(self, cutoff: float) -> None:
        if not 0 < cutoff <= 1:
            raise ValueError("cutoff must be in (0, 1]")
        self.cutoff = float(cutoff)

    def alpha(self, time: torch.Tensor) -> torch.Tensor:
        scaled = (time / self.cutoff).clamp(0.0, 1.0)
        return 1.0 - torch.cos(0.5 * math.pi * scaled)

    def derivative(self, time: torch.Tensor) -> torch.Tensor:
        scaled = time / self.cutoff
        value = (
            0.5
            * math.pi
            / self.cutoff
            * torch.sin(0.5 * math.pi * scaled.clamp(0.0, 1.0))
        )
        return value.masked_fill((scaled < 0) | (scaled >= 1), 0.0)

    def inverse(self, probability: torch.Tensor) -> torch.Tensor:
        probability = probability.clamp(0.0, 1.0)
        return (
            self.cutoff
            * 2.0
            / math.pi
            * torch.acos((1.0 - probability).clamp(-1.0, 1.0))
        )

    def hazard(self, time: torch.Tensor) -> torch.Tensor:
        """Instantaneous insertion hazard alpha'(t) / (1 - alpha(t))."""
        denominator = 1.0 - self.alpha(time)
        active = time < self.cutoff
        return torch.where(
            active,
            self.derivative(time) / denominator.clamp_min(1e-6),
            torch.zeros_like(time),
        )

    def interval_fraction(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        alpha_source = self.alpha(source)
        return (
            (self.alpha(target) - alpha_source)
            / (1.0 - alpha_source).clamp_min(1e-6)
        ).clamp(0.0, 1.0)


class VocabularyTimeWarp(nn.Module):
    """Numerical inverse of the large-vocabulary decoding-error time warp.

    For the Gaussian-to-one-hot interpolant, the probability that the target
    category is the largest coordinate is evaluated by Gauss-Hermite
    quadrature.  The resulting monotone table implements Eq. 86 of the paper.
    """

    def __init__(
        self,
        vocabulary_size: int,
        *,
        points: int = 2048,
        quadrature: int = 48,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least two")
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature)
        z = torch.from_numpy(nodes * math.sqrt(2.0)).double()
        gh_weights = torch.from_numpy(weights / math.sqrt(math.pi)).double()
        time = torch.linspace(0.0, 1.0 - epsilon, points, dtype=torch.float64)
        signal = time / (1.0 - time)
        cdf = 0.5 * (
            1.0
            + torch.erf(
                (z.reshape(1, -1) + signal.reshape(-1, 1)) / math.sqrt(2.0)
            )
        )
        correct = (
            cdf.clamp_min(torch.finfo(torch.float64).tiny)
            .log()
            .mul(vocabulary_size - 1)
            .exp()
            .mul(gh_weights.reshape(1, -1))
            .sum(dim=1)
        )
        error = (1.0 - correct).clamp_min(0.0)
        error0 = 1.0 - 1.0 / vocabulary_size
        tau = ((error0 - error) / error0).clamp(0.0, 1.0)
        tau = torch.cummax(tau, dim=0).values
        # Explicit unique endpoints make interpolation stable at boundaries.
        before_endpoint = tau < 1.0 - 1e-7
        time = torch.cat(
            (time[before_endpoint], torch.ones(1, dtype=time.dtype))
        )
        tau = torch.cat(
            (tau[before_endpoint], torch.ones(1, dtype=tau.dtype))
        )
        keep = torch.ones_like(tau, dtype=torch.bool)
        keep[1:] = tau[1:] > tau[:-1] + 1e-12
        keep[-1] = True
        self.register_buffer("time_table", time[keep].float(), persistent=True)
        self.register_buffer("tau_table", tau[keep].float(), persistent=True)

    @staticmethod
    def _interpolate(
        values: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        flat = values.reshape(-1)
        indices = torch.searchsorted(x, flat).clamp(1, x.numel() - 1)
        x0 = x[indices - 1]
        x1 = x[indices]
        y0 = y[indices - 1]
        y1 = y[indices]
        fraction = (flat - x0) / (x1 - x0).clamp_min(1e-12)
        return (y0 + fraction * (y1 - y0)).reshape(values.shape)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return self._interpolate(
            time.clamp(0.0, 1.0), self.time_table, self.tau_table
        )

    def inverse(self, tau: torch.Tensor) -> torch.Tensor:
        return self._interpolate(
            tau.clamp(0.0, 1.0), self.tau_table, self.time_table
        )


class IdentityTimeWarp(nn.Module):
    """Identity schedule for categorical masked-token priors."""

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return time.clamp(0.0, 1.0)

    def inverse(self, tau: torch.Tensor) -> torch.Tensor:
        return tau.clamp(0.0, 1.0)


class FourierTimeEmbedding(nn.Module):
    def __init__(self, fourier_dim: int, hidden_size: int) -> None:
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), fourier_dim // 2)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * time.unsqueeze(-1) * self.frequencies
        return self.projection(torch.cat((angles.sin(), angles.cos()), dim=-1))


@dataclass
class ExpandingModelOutput:
    logits: torch.Tensor
    insertion_means: torch.Tensor
    insertion_mask: torch.Tensor
    hidden: torch.Tensor


class ExpandingMarlinModel(nn.Module):
    """FRIGID-initializable conditional EFlow/EFM sequence network."""

    def __init__(
        self,
        decoder_config: MarlinDecoderConfig,
        flow_config: ExpandingFlowConfig,
    ) -> None:
        super().__init__()
        self.decoder_config = decoder_config
        self.flow_config = flow_config
        self.backbone = MarlinDecoder(decoder_config)
        self.time_warp = (
            VocabularyTimeWarp(
                decoder_config.vocab_size,
                points=flow_config.time_warp_points,
                quadrature=flow_config.time_warp_quadrature,
            )
            if flow_config.time_warp_type == "vocabulary_error"
            else IdentityTimeWarp()
        )
        hidden = decoder_config.hidden_size
        self.source_time = FourierTimeEmbedding(
            flow_config.time_fourier_dim, hidden
        )
        self.target_time = FourierTimeEmbedding(
            flow_config.time_fourier_dim, hidden
        )
        # A zero target projection starts as a single-time EFlow denoiser.
        self.target_projection = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.target_projection.weight)
        nn.init.zeros_(self.target_projection.bias)
        self.layer_modulation = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
            for _ in range(decoder_config.num_layers)
        )
        self.output_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, 2 * hidden)
        )
        for modulation in (*self.layer_modulation, self.output_modulation):
            nn.init.zeros_(modulation[-1].weight)
            nn.init.zeros_(modulation[-1].bias)
        self.left_boundary = nn.Parameter(torch.zeros(hidden))
        self.right_boundary = nn.Parameter(torch.zeros(hidden))
        self.gap_left = nn.Linear(hidden, flow_config.time_embedding_size)
        self.gap_right = nn.Linear(hidden, flow_config.time_embedding_size)
        self.gap_time = nn.Linear(hidden, flow_config.time_embedding_size)
        self.insertion_output = nn.Sequential(
            nn.SiLU(),
            nn.Linear(flow_config.time_embedding_size, 1),
        )

    @property
    def config(self) -> MarlinDecoderConfig:
        """Sampler compatibility with the original MARLIN model."""
        return self.decoder_config

    def _gap_means(
        self,
        hidden: torch.Tensor,
        lengths: torch.Tensor,
        time_condition: torch.Tensor,
        *,
        detach_backbone: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if detach_backbone:
            hidden = hidden.detach()
        batch, width, feature_size = hidden.shape
        positions = torch.arange(width + 1, device=hidden.device).unsqueeze(0)
        left_indices = (positions - 1).clamp(0, width - 1)
        right_indices = positions.clamp(0, width - 1)
        left = hidden.gather(
            1, left_indices.unsqueeze(-1).expand(batch, -1, feature_size)
        )
        right = hidden.gather(
            1, right_indices.unsqueeze(-1).expand(batch, -1, feature_size)
        )
        left = torch.where(
            positions.unsqueeze(-1).eq(0),
            self.left_boundary.reshape(1, 1, -1),
            left,
        )
        right = torch.where(
            positions.unsqueeze(-1).ge(lengths.reshape(-1, 1, 1)),
            self.right_boundary.reshape(1, 1, -1),
            right,
        )
        gap_hidden = (
            self.gap_left(left)
            + self.gap_right(right)
            + self.gap_time(time_condition).unsqueeze(1)
        )
        means = F.softplus(self.insertion_output(gap_hidden).squeeze(-1)) + 1e-6
        mask = positions <= lengths.unsqueeze(1)
        means = means.masked_fill(~mask, 0.0)
        return means, mask

    def forward(
        self,
        latent_tokens: torch.Tensor,
        local_times: torch.Tensor,
        padding_mask: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        *,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        isotope_ratios: torch.Tensor | None = None,
        detach_insertion_backbone: bool = False,
    ) -> ExpandingModelOutput:
        if latent_tokens.ndim != 3:
            raise ValueError("latent_tokens must have shape [batch, length, vocab]")
        batch, length, vocabulary = latent_tokens.shape
        if vocabulary != self.decoder_config.vocab_size:
            raise ValueError("latent vocabulary does not match decoder")
        expected = (batch, length)
        if local_times.shape != expected or padding_mask.shape != expected:
            raise ValueError("local_times and padding_mask must match latent tokens")
        if source_time.shape != (batch,) or target_time.shape != (batch,):
            raise ValueError("source_time and target_time must have shape [batch]")
        if length > self.decoder_config.max_length:
            raise ValueError("expanding sequence exceeds max_length")

        positions = torch.arange(length, device=latent_tokens.device)
        hidden = latent_tokens @ self.backbone.token_embedding.weight
        hidden = hidden + self.backbone.position_embedding(positions).unsqueeze(0)
        if self.backbone.token_type_embedding is not None:
            token_types = torch.zeros(expected, dtype=torch.long, device=hidden.device)
            hidden = hidden + self.backbone.token_type_embedding(token_types)
            hidden = self.backbone.embedding_norm(hidden)
        hidden = self.backbone.embedding_dropout(hidden)
        condition, condition_mask = self.backbone.conditioner(
            precursor_mass,
            fingerprint,
            isotope_ratios,
            include_mass=True,
        )
        local_condition = self.source_time(local_times)
        target_condition = self.target_projection(self.target_time(target_time))
        time_condition = local_condition + target_condition.unsqueeze(1)
        attention_mask = torch.zeros(
            (length, length), dtype=torch.bool, device=hidden.device
        )
        layer0_hidden = None
        for layer_index, (layer, modulation) in enumerate(
            zip(self.backbone.layers, self.layer_modulation)
        ):
            shift, scale = modulation(time_condition).chunk(2, dim=-1)
            hidden = hidden * (1.0 + scale) + shift
            hidden = layer(
                hidden,
                condition,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                condition_padding_mask=~condition_mask,
            )
            if layer_index == 0:
                layer0_hidden = hidden
        if self.decoder_config.layer0_long_residual_scale:
            if layer0_hidden is None:
                raise RuntimeError("layer0 long residual has no source")
            hidden = (
                hidden
                + self.decoder_config.layer0_long_residual_scale * layer0_hidden
            )
        output_shift, output_scale = self.output_modulation(
            time_condition
        ).chunk(2, dim=-1)
        hidden = hidden * (1.0 + output_scale) + output_shift
        prediction_hidden = self.backbone.prediction_norm(
            F.gelu(self.backbone.prediction_dense(hidden))
        )
        logits = (
            self.backbone.output(prediction_hidden) + self.backbone.output_bias
        )
        lengths = (~padding_mask).sum(dim=1)
        insertion_means, insertion_mask = self._gap_means(
            prediction_hidden,
            lengths,
            self.source_time(source_time)
            + self.target_projection(self.target_time(target_time)),
            detach_backbone=detach_insertion_backbone,
        )
        return ExpandingModelOutput(
            logits=logits,
            insertion_means=insertion_means,
            insertion_mask=insertion_mask,
            hidden=prediction_hidden,
        )


@dataclass
class ExpandingBatch:
    latent_tokens: torch.Tensor
    target_ids: torch.Tensor
    local_times: torch.Tensor
    padding_mask: torch.Tensor
    token_loss_mask: torch.Tensor
    gap_targets: torch.Tensor
    gap_mask: torch.Tensor
    source_time: torch.Tensor
    target_time: torch.Tensor
    sample_weights: torch.Tensor


def _rand(
    shape: Sequence[int],
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=generator)


def _randn(
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _sample_prior(
    count: int,
    config: MarlinDecoderConfig,
    flow_config: ExpandingFlowConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if flow_config.prior_type == "mask":
        ids = torch.full(
            (count,), config.mask_token_id, device=device, dtype=torch.long
        )
        return F.one_hot(ids, num_classes=config.vocab_size).to(dtype)
    return _randn(
        (count, config.vocab_size),
        device=device,
        dtype=dtype,
        generator=generator,
    ).mul(flow_config.prior_scale)


def _special_masks(
    clean_ids: torch.Tensor, config: MarlinDecoderConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = clean_ids.ne(config.pad_token_id)
    bos = torch.zeros_like(valid)
    bos[:, 0] = valid[:, 0]
    eos = valid & clean_ids.eq(config.eos_token_id)
    content = valid & ~bos & ~eos
    return valid, eos, content


def _local_times(
    global_time: torch.Tensor, insertion_times: torch.Tensor
) -> torch.Tensor:
    expanded_time = global_time.unsqueeze(1)
    active = insertion_times <= expanded_time
    return torch.where(
        active,
        (expanded_time - insertion_times)
        / (1.0 - insertion_times).clamp_min(1e-6),
        torch.zeros_like(insertion_times),
    ).clamp(0.0, 1.0)


def _compact_batch(
    clean_ids: torch.Tensor,
    insertion_times: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    active_at_target: torch.Tensor,
    config: MarlinDecoderConfig,
    flow_config: ExpandingFlowConfig,
    *,
    generator: torch.Generator | None,
    sample_weights: torch.Tensor | None = None,
) -> ExpandingBatch:
    device = clean_ids.device
    batch = clean_ids.shape[0]
    vocabulary = config.vocab_size
    valid, eos, content = _special_masks(clean_ids, config)
    source_local = _local_times(source_time, insertion_times)
    anchors = valid & ~content
    source_local = source_local.masked_fill(anchors, 1.0)
    active_at_target = active_at_target & valid
    active_at_target = active_at_target | anchors
    active_lengths = active_at_target.sum(dim=1)
    width = int(active_lengths.max().item())
    latent = torch.zeros(
        (batch, width, vocabulary), device=device, dtype=torch.float32
    )
    target_ids = torch.full(
        (batch, width), config.pad_token_id, device=device, dtype=torch.long
    )
    compact_local = torch.zeros((batch, width), device=device)
    padding = torch.ones((batch, width), device=device, dtype=torch.bool)
    token_loss_mask = torch.zeros_like(padding)
    gap_targets = torch.zeros((batch, width + 1), device=device)
    gap_mask = torch.zeros((batch, width + 1), device=device, dtype=torch.bool)

    for row in range(batch):
        original = torch.nonzero(active_at_target[row], as_tuple=False).flatten()
        count = original.numel()
        ids = clean_ids[row, original]
        local = source_local[row, original]
        noise = _sample_prior(
            count,
            config,
            flow_config,
            device=device,
            dtype=latent.dtype,
            generator=generator,
        )
        one_hot = F.one_hot(ids, num_classes=vocabulary).to(latent.dtype)
        mixed = (1.0 - local.unsqueeze(1)) * noise + local.unsqueeze(1) * one_hot
        latent[row, :count] = mixed
        target_ids[row, :count] = ids
        compact_local[row, :count] = local
        padding[row, :count] = False
        token_loss_mask[row, :count] = content[row, original]

        gap_mask[row, : count + 1] = True
        # Positions before BOS and after EOS are structurally forbidden.  Each
        # internal target is the number of clean tokens not active at source.
        if count >= 2:
            gaps = original[1:] - original[:-1] - 1
            gap_targets[row, 1:count] = gaps.to(gap_targets.dtype)

    return ExpandingBatch(
        latent_tokens=latent,
        target_ids=target_ids,
        local_times=compact_local,
        padding_mask=padding,
        token_loss_mask=token_loss_mask,
        gap_targets=gap_targets,
        gap_mask=gap_mask,
        source_time=source_time,
        target_time=target_time,
        sample_weights=(
            torch.ones_like(source_time)
            if sample_weights is None
            else sample_weights
        ),
    )


def sample_eflow_batch(
    clean_ids: torch.Tensor,
    config: MarlinDecoderConfig,
    flow_config: ExpandingFlowConfig,
    schedule: CosineInsertionSchedule,
    warp: VocabularyTimeWarp,
    *,
    generator: torch.Generator | None = None,
) -> ExpandingBatch:
    """Sample Algorithm 1's expanding interpolant for SAFE sequences."""
    device = clean_ids.device
    batch = clean_ids.shape[0]
    # The vocabulary time warp places only a tiny fraction of uniform tau
    # samples before the insertion cutoff (about 1% for the SAFE vocabulary).
    # Stratify both regions and attach exact importance weights so small-batch
    # training remains unbiased while seeing useful insertion examples.
    early_probability = flow_config.eflow_early_time_probability
    cutoff_tau = warp(
        torch.tensor(flow_config.insertion_cutoff, device=device)
    ).clamp(1e-6, 1.0 - 1e-6)
    region_draw = _rand((batch,), device=device, generator=generator)
    within_region = _rand((batch,), device=device, generator=generator)
    early = region_draw < early_probability
    tau = torch.where(
        early,
        within_region * cutoff_tau,
        cutoff_tau + within_region * (1.0 - cutoff_tau),
    )
    sample_weights = torch.where(
        early,
        cutoff_tau / early_probability,
        (1.0 - cutoff_tau) / (1.0 - early_probability),
    )
    global_time = warp.inverse(tau).clamp_min(flow_config.sampling_time_floor)
    insertion_times = schedule.inverse(
        _rand(clean_ids.shape, device=device, generator=generator)
    )
    valid, _, content = _special_masks(clean_ids, config)
    insertion_times = insertion_times.masked_fill(~content, 0.0)
    insertion_times = insertion_times.masked_fill(~valid, 2.0)
    active = insertion_times <= global_time.unsqueeze(1)
    return _compact_batch(
        clean_ids,
        insertion_times,
        global_time,
        global_time,
        active,
        config,
        flow_config,
        generator=generator,
        sample_weights=sample_weights,
    )


def _masked_soft_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    losses = -(targets * logits.float().log_softmax(dim=-1)).sum(dim=-1)
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def _masked_hard_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.transpose(1, 2), target_ids, reduction="none"
    )
    weights = (
        torch.ones(
            (logits.shape[0], 1),
            device=logits.device,
            dtype=losses.dtype,
        )
        if sample_weights is None
        else sample_weights.unsqueeze(1)
    )
    weighted_mask = mask * weights
    return (losses * weighted_mask).sum() / weighted_mask.sum().clamp_min(1)


def _poisson_loss(
    means: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    losses = means - targets * means.clamp_min(1e-8).log() + torch.lgamma(
        targets + 1.0
    )
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def _hazard_weighted_poisson_loss(
    means: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    hazard: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Paper Eq. 26 diagonal insertion loss with importance weighting."""
    losses = means - targets * means.clamp_min(1e-8).log() + torch.lgamma(
        targets + 1.0
    )
    row_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    weights = hazard * sample_weights
    return (row_losses * weights).sum() / weights.sum().clamp_min(1e-8)


def eflow_objective(
    model: ExpandingMarlinModel,
    clean_ids: torch.Tensor,
    precursor_mass: torch.Tensor,
    fingerprint: torch.Tensor,
    *,
    isotope_ratios: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    detach_insertion_backbone: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    schedule = CosineInsertionSchedule(model.flow_config.insertion_cutoff)
    batch = sample_eflow_batch(
        clean_ids,
        model.decoder_config,
        model.flow_config,
        schedule,
        model.time_warp,
        generator=generator,
    )
    output = model(
        batch.latent_tokens,
        batch.local_times,
        batch.padding_mask,
        precursor_mass,
        fingerprint,
        source_time=batch.source_time,
        target_time=batch.target_time,
        isotope_ratios=isotope_ratios,
        detach_insertion_backbone=detach_insertion_backbone,
    )
    token_loss = _masked_hard_cross_entropy(
        output.logits,
        batch.target_ids,
        batch.token_loss_mask,
        batch.sample_weights,
    )
    insertion_mask = batch.gap_mask & output.insertion_mask
    insertion_hazard = schedule.hazard(batch.source_time)
    insertion_loss = _hazard_weighted_poisson_loss(
        output.insertion_means,
        batch.gap_targets,
        insertion_mask,
        insertion_hazard,
        batch.sample_weights,
    )
    loss = token_loss + model.flow_config.insertion_loss_weight * insertion_loss
    with torch.no_grad():
        predictions = output.logits.argmax(dim=-1)
        token_count = batch.token_loss_mask.sum().clamp_min(1)
        accuracy = (
            predictions.eq(batch.target_ids) & batch.token_loss_mask
        ).sum() / token_count
        predicted_remaining = (
            output.insertion_means * insertion_mask
        ).sum(dim=1)
        actual_remaining = (batch.gap_targets * insertion_mask).sum(dim=1)
        length_mae = (predicted_remaining - actual_remaining).abs().mean()
    return loss, {
        "token_loss": token_loss.detach(),
        "insertion_loss": insertion_loss.detach(),
        "token_accuracy": accuracy.detach(),
        "active_fraction": batch.token_loss_mask.sum()
        / clean_ids.ne(model.decoder_config.pad_token_id).sum().clamp_min(1),
        "remaining_length_mae": length_mae.detach(),
        "source_time": batch.source_time.mean().detach(),
        "insertion_hazard": insertion_hazard.mean().detach(),
        "early_time_fraction": (
            batch.source_time < model.flow_config.insertion_cutoff
        )
        .float()
        .mean()
        .detach(),
        "importance_weight": batch.sample_weights.mean().detach(),
    }


@dataclass
class EFMOffDiagonalBatch:
    expanded: ExpandingBatch
    partial: ExpandingBatch
    insertion_times: torch.Tensor
    active_by_u: torch.Tensor
    source_time: torch.Tensor
    middle_time: torch.Tensor
    target_time: torch.Tensor


def sample_efm_off_diagonal_batch(
    clean_ids: torch.Tensor,
    config: MarlinDecoderConfig,
    flow_config: ExpandingFlowConfig,
    schedule: CosineInsertionSchedule,
    warp: VocabularyTimeWarp,
    *,
    generator: torch.Generator | None = None,
) -> EFMOffDiagonalBatch:
    device = clean_ids.device
    batch = clean_ids.shape[0]
    tau_source = _rand((batch,), device=device, generator=generator)
    difference = _rand((batch,), device=device, generator=generator)
    tau_target = tau_source + (1.0 - tau_source) * difference
    boundary = _rand((batch,), device=device, generator=generator)
    force_boundary = boundary < flow_config.boundary_probability
    tau_source = torch.where(force_boundary, torch.zeros_like(tau_source), tau_source)
    tau_target = torch.where(force_boundary, torch.ones_like(tau_target), tau_target)
    source = warp.inverse(tau_source).clamp_max(1.0 - 2e-4)
    target = warp.inverse(tau_target).clamp_min(source + 1e-4).clamp_max(1.0)
    middle = 0.5 * (source + target)
    insertion_times = schedule.inverse(
        _rand(clean_ids.shape, device=device, generator=generator)
    )
    valid, _, content = _special_masks(clean_ids, config)
    insertion_times = insertion_times.masked_fill(~content, 0.0)
    insertion_times = insertion_times.masked_fill(~valid, 2.0)
    active_target = insertion_times <= target.unsqueeze(1)
    active_source = insertion_times <= source.unsqueeze(1)
    active_middle = insertion_times <= middle.unsqueeze(1)
    expanded = _compact_batch(
        clean_ids,
        insertion_times,
        source,
        target,
        active_target,
        config,
        flow_config,
        generator=generator,
    )
    partial = _compact_batch(
        clean_ids,
        insertion_times,
        source,
        target,
        active_source,
        config,
        flow_config,
        generator=generator,
    )
    # Compact the middle-activation mask in target-active order.
    compact_middle = torch.zeros_like(expanded.padding_mask)
    for row in range(batch):
        target_indices = torch.nonzero(
            active_target[row] & valid[row], as_tuple=False
        ).flatten()
        compact_middle[row, : target_indices.numel()] = active_middle[
            row, target_indices
        ]
    return EFMOffDiagonalBatch(
        expanded=expanded,
        partial=partial,
        insertion_times=insertion_times,
        active_by_u=compact_middle,
        source_time=source,
        middle_time=middle,
        target_time=target,
    )


def _flow_map(
    state: torch.Tensor,
    prediction: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    update_mask: torch.Tensor,
) -> torch.Tensor:
    denominator = (1.0 - source).clamp_min(1e-6)
    keep = ((1.0 - target) / denominator).reshape(-1, 1, 1)
    move = ((target - source) / denominator).reshape(-1, 1, 1)
    mapped = keep * state + move * prediction
    return torch.where(update_mask.unsqueeze(-1), mapped, state)


def efm_objective(
    student: ExpandingMarlinModel,
    teacher: ExpandingMarlinModel,
    clean_ids: torch.Tensor,
    precursor_mass: torch.Tensor,
    fingerprint: torch.Tensor,
    *,
    isotope_ratios: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    detach_insertion_backbone: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Teacher-distilled diagonal + semigroup EFM objective."""
    if _rand(
        (1,), device=clean_ids.device, generator=generator
    ).item() < student.flow_config.diagonal_probability:
        schedule = CosineInsertionSchedule(student.flow_config.insertion_cutoff)
        batch = sample_eflow_batch(
            clean_ids,
            student.decoder_config,
            student.flow_config,
            schedule,
            student.time_warp,
            generator=generator,
        )
        with torch.no_grad():
            teacher_output = teacher(
                batch.latent_tokens,
                batch.local_times,
                batch.padding_mask,
                precursor_mass,
                fingerprint,
                source_time=batch.source_time,
                target_time=batch.target_time,
                isotope_ratios=isotope_ratios,
            )
            target_probabilities = teacher_output.logits.float().softmax(dim=-1)
        student_output = student(
            batch.latent_tokens,
            batch.local_times,
            batch.padding_mask,
            precursor_mass,
            fingerprint,
            source_time=batch.source_time,
            target_time=batch.target_time,
            isotope_ratios=isotope_ratios,
            detach_insertion_backbone=detach_insertion_backbone,
        )
        token_loss = _masked_soft_cross_entropy(
            student_output.logits, target_probabilities, batch.token_loss_mask
        )
        insertion_mask = batch.gap_mask & student_output.insertion_mask
        insertion_loss = _poisson_loss(
            student_output.insertion_means, batch.gap_targets, insertion_mask
        )
        loss = (
            token_loss
            + student.flow_config.insertion_loss_weight * insertion_loss
        )
        return loss, {
            "token_loss": token_loss.detach(),
            "insertion_loss": insertion_loss.detach(),
            "diagonal_fraction": loss.new_tensor(1.0),
            "source_time": batch.source_time.mean().detach(),
            "target_time": batch.target_time.mean().detach(),
        }

    schedule = CosineInsertionSchedule(student.flow_config.insertion_cutoff)
    sampled = sample_efm_off_diagonal_batch(
        clean_ids,
        student.decoder_config,
        student.flow_config,
        schedule,
        student.time_warp,
        generator=generator,
    )
    batch = sampled.expanded
    first = student(
        batch.latent_tokens,
        batch.local_times,
        batch.padding_mask,
        precursor_mass,
        fingerprint,
        source_time=sampled.source_time,
        target_time=sampled.middle_time,
        isotope_ratios=isotope_ratios,
    )
    first_probabilities = first.logits.float().softmax(dim=-1)
    middle_state = _flow_map(
        batch.latent_tokens,
        first_probabilities,
        sampled.source_time,
        sampled.middle_time,
        sampled.active_by_u & ~batch.padding_mask,
    )
    # Existing coordinates advance to their local time at u; coordinates whose
    # insertion lies after u remain pure noise at local time zero.
    middle_local = torch.zeros_like(batch.local_times)
    valid, _, _ = _special_masks(clean_ids, student.decoder_config)
    for row in range(clean_ids.shape[0]):
        target_indices = torch.nonzero(
            (sampled.insertion_times[row] <= sampled.target_time[row])
            & valid[row],
            as_tuple=False,
        ).flatten()
        local = _local_times(
            sampled.middle_time[row : row + 1],
            sampled.insertion_times[row : row + 1, target_indices],
        )[0]
        anchor = ~(
            (target_indices != 0)
            & clean_ids[row, target_indices].ne(student.decoder_config.eos_token_id)
        )
        local = local.masked_fill(anchor, 1.0)
        middle_local[row, : target_indices.numel()] = local
    second = student(
        middle_state,
        middle_local,
        batch.padding_mask,
        precursor_mass,
        fingerprint,
        source_time=sampled.middle_time,
        target_time=sampled.target_time,
        isotope_ratios=isotope_ratios,
    )
    second_probabilities = second.logits.float().softmax(dim=-1)
    source = sampled.source_time
    middle = sampled.middle_time
    target = sampled.target_time
    omega = (
        (middle - source)
        * (1.0 - target)
        / ((target - source) * (1.0 - middle)).clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    consistency_target = (
        omega.reshape(-1, 1, 1) * first_probabilities
        + (1.0 - omega).reshape(-1, 1, 1) * second_probabilities
    ).detach()
    direct = student(
        batch.latent_tokens,
        batch.local_times,
        batch.padding_mask,
        precursor_mass,
        fingerprint,
        source_time=source,
        target_time=target,
        isotope_ratios=isotope_ratios,
    )
    log_probabilities = direct.logits.float().log_softmax(dim=-1)
    per_position = -(consistency_target * log_probabilities).sum(dim=-1)
    mismatch = (
        direct.logits.float().softmax(dim=-1) - consistency_target
    ).square().sum(dim=-1)
    adaptive_weight = (
        mismatch + student.flow_config.adaptive_loss_c
    ).pow(-student.flow_config.adaptive_loss_r).detach()
    token_mask = batch.token_loss_mask
    token_loss = (
        per_position * adaptive_weight * token_mask
    ).sum() / (adaptive_weight * token_mask).sum().clamp_min(1)

    partial_output = student(
        sampled.partial.latent_tokens,
        sampled.partial.local_times,
        sampled.partial.padding_mask,
        precursor_mass,
        fingerprint,
        source_time=source,
        target_time=target,
        isotope_ratios=isotope_ratios,
        detach_insertion_backbone=detach_insertion_backbone,
    )
    # The compact partial targets currently store all tokens absent at source.
    # Restrict them to the realized source->target interval.
    interval_targets = sampled.partial.gap_targets.clone()
    for row in range(clean_ids.shape[0]):
        source_indices = torch.nonzero(
            (sampled.insertion_times[row] <= source[row]) & valid[row],
            as_tuple=False,
        ).flatten()
        target_active = sampled.insertion_times[row] <= target[row]
        for gap in range(1, source_indices.numel()):
            left = int(source_indices[gap - 1])
            right = int(source_indices[gap])
            interval_targets[row, gap] = target_active[
                left + 1 : right
            ].sum()
    insertion_mask = sampled.partial.gap_mask & partial_output.insertion_mask
    insertion_loss = _poisson_loss(
        partial_output.insertion_means, interval_targets, insertion_mask
    )
    loss = token_loss + student.flow_config.insertion_loss_weight * insertion_loss
    return loss, {
        "token_loss": token_loss.detach(),
        "insertion_loss": insertion_loss.detach(),
        "diagonal_fraction": loss.new_tensor(0.0),
        "source_time": source.mean().detach(),
        "target_time": target.mean().detach(),
        "consistency_mismatch": mismatch[token_mask].mean().detach(),
    }


@dataclass
class _SequenceState:
    latent: torch.Tensor
    insertion_times: torch.Tensor


class ExpandingMarlinSampler:
    """Variable-length EFlow/EFM sampler with molecular post-filtering."""

    def __init__(
        self,
        model: ExpandingMarlinModel,
        constraint: MassShellConstraint,
        *,
        stage: str,
        bos_token_id: int,
        eos_token_id: int,
        decode_tokens: Callable[[Sequence[int]], str],
        safe_to_smiles: Callable[[str], str | None],
        forbidden_token_ids: Sequence[int] = (),
        mass_shell_enabled: bool = True,
        steps: int = 32,
    ) -> None:
        if stage not in {"eflow", "efm"}:
            raise ValueError("stage must be 'eflow' or 'efm'")
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.model = model
        self.constraint = constraint
        self.stage = stage
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decode_tokens = decode_tokens
        self.safe_to_smiles = safe_to_smiles
        self.forbidden_token_ids = tuple(forbidden_token_ids)
        self.mass_shell_enabled = mass_shell_enabled
        self.steps = steps
        self.schedule = CosineInsertionSchedule(
            model.flow_config.insertion_cutoff
        )

    def _initial_state(
        self, device: torch.device, dtype: torch.dtype
    ) -> _SequenceState:
        ids = torch.tensor(
            [self.bos_token_id, self.eos_token_id], device=device
        )
        latent = F.one_hot(
            ids, num_classes=self.model.decoder_config.vocab_size
        ).to(dtype)
        insertion_times = torch.full((2,), -1.0, device=device)
        return _SequenceState(latent=latent, insertion_times=insertion_times)

    def _pack(
        self, states: list[_SequenceState], global_time: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = states[0].latent.device
        batch = len(states)
        width = max(state.latent.shape[0] for state in states)
        vocabulary = self.model.decoder_config.vocab_size
        latent = torch.zeros(
            (batch, width, vocabulary),
            device=device,
            dtype=states[0].latent.dtype,
        )
        local = torch.zeros((batch, width), device=device)
        padding = torch.ones((batch, width), device=device, dtype=torch.bool)
        for row, state in enumerate(states):
            count = state.latent.shape[0]
            latent[row, :count] = state.latent
            padding[row, :count] = False
            times = state.insertion_times
            values = torch.where(
                times < 0,
                torch.ones_like(times),
                (global_time - times)
                / (1.0 - times).clamp_min(1e-6),
            ).clamp(0.0, 1.0)
            local[row, :count] = values
        return latent, local, padding

    def _draw_insertions(
        self,
        means: torch.Tensor,
        current_length: int,
        *,
        generator: torch.Generator | None,
    ) -> list[int]:
        budget = self.model.decoder_config.max_length - current_length
        counts = [0] * means.numel()
        if budget <= 0:
            return counts
        remaining = budget
        for gap in range(1, means.numel() - 1):
            if remaining <= 0:
                break
            probability = float((means[gap] / budget).clamp(0.0, 1.0))
            draw = int(
                torch.binomial(
                    torch.tensor(
                        float(budget), device=means.device, dtype=means.dtype
                    ),
                    torch.tensor(
                        probability, device=means.device, dtype=means.dtype
                    ),
                    generator=generator,
                ).item()
            )
            counts[gap] = min(draw, remaining)
            remaining -= counts[gap]
        return counts

    def _insert(
        self,
        state: _SequenceState,
        counts: list[int],
        source_time: float,
        *,
        generator: torch.Generator | None,
    ) -> _SequenceState:
        pieces = []
        times = []
        for index in range(state.latent.shape[0]):
            if index < len(counts) and counts[index]:
                noise = _sample_prior(
                    counts[index],
                    self.model.decoder_config,
                    self.model.flow_config,
                    device=state.latent.device,
                    dtype=state.latent.dtype,
                    generator=generator,
                )
                pieces.append(noise)
                times.append(
                    torch.full(
                        (counts[index],),
                        source_time,
                        device=state.latent.device,
                    )
                )
            pieces.append(state.latent[index : index + 1])
            times.append(state.insertion_times[index : index + 1])
        return _SequenceState(
            latent=torch.cat(pieces, dim=0),
            insertion_times=torch.cat(times, dim=0),
        )

    def _project_tokens(self, probabilities: torch.Tensor) -> list[int]:
        if self.forbidden_token_ids:
            probabilities = probabilities.clone()
            probabilities[:, list(self.forbidden_token_ids)] = 0.0
        return probabilities.argmax(dim=-1).tolist()

    @torch.no_grad()
    def _generate_many(
        self,
        fingerprint: torch.Tensor,
        target_mass: float,
        *,
        candidates: int,
        diversity_dropout: float,
        temperature: float,
        generator: torch.Generator | None,
    ) -> tuple[list[tuple[str, str, bool] | None], int, dict[str, object]]:
        del temperature  # the flow state carries stochasticity continuously
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        original = fingerprint.to(device=device, dtype=torch.float32)
        conditioned = torch.stack(
            [
                perturb_fingerprint(
                    original, dropout=diversity_dropout, generator=generator
                )
                for _ in range(candidates)
            ]
        )
        masses = torch.full((candidates,), target_mass, device=device)
        states = [self._initial_state(device, dtype) for _ in range(candidates)]
        tau_grid = torch.linspace(
            0.0, 1.0, self.steps + 1, device=device
        )
        time_grid = self.model.time_warp.inverse(tau_grid)
        time_grid[0] = 0.0
        time_grid[-1] = 1.0

        for step in range(self.steps):
            source = float(time_grid[step])
            target = float(time_grid[step + 1])
            packed, local, padding = self._pack(states, source)
            source_tensor = torch.full((candidates,), source, device=device)
            target_tensor = torch.full((candidates,), target, device=device)
            insertion_output = self.model(
                packed,
                local,
                padding,
                masses,
                conditioned,
                source_time=source_tensor,
                target_time=(
                    target_tensor if self.stage == "efm" else source_tensor
                ),
            )
            expanded_states = []
            for row, state in enumerate(states):
                count = state.latent.shape[0]
                means = insertion_output.insertion_means[row, : count + 1]
                if self.stage == "eflow":
                    fraction = self.schedule.interval_fraction(
                        source_tensor[row], target_tensor[row]
                    )
                    means = means * fraction
                counts = self._draw_insertions(
                    means,
                    count,
                    generator=generator,
                )
                expanded_states.append(
                    self._insert(state, counts, source, generator=generator)
                )
            states = expanded_states
            packed, local_source, padding = self._pack(states, source)
            denoiser_source_tensor = source_tensor
            if self.stage == "efm" and self.steps == 1 and step == 0:
                # App. C.5: condition the one-step full-noise sequence at a
                # small positive denoising time while retaining the true
                # insertion interval for the expansion head.
                denoising_floor = self.model.flow_config.sampling_time_floor
                denoiser_source_tensor = torch.full(
                    (candidates,), denoising_floor, device=device
                )
                non_anchor = ~padding
                non_anchor[:, 0] = False
                for row, state in enumerate(states):
                    non_anchor[row, state.latent.shape[0] - 1] = False
                local_source = local_source.masked_fill(
                    non_anchor, denoising_floor
                )
            denoised = self.model(
                packed,
                local_source,
                padding,
                masses,
                conditioned,
                source_time=denoiser_source_tensor,
                target_time=(
                    target_tensor if self.stage == "efm" else source_tensor
                ),
            )
            probabilities = denoised.logits.float().softmax(dim=-1)
            for row, state in enumerate(states):
                count = state.latent.shape[0]
                insertion = state.insertion_times
                local_s = torch.where(
                    insertion < 0,
                    torch.ones_like(insertion),
                    (source - insertion)
                    / (1.0 - insertion).clamp_min(1e-6),
                ).clamp(0.0, 1.0)
                local_t = torch.where(
                    insertion < 0,
                    torch.ones_like(insertion),
                    (target - insertion)
                    / (1.0 - insertion).clamp_min(1e-6),
                ).clamp(0.0, 1.0)
                keep = (1.0 - local_t) / (1.0 - local_s).clamp_min(1e-6)
                move = (local_t - local_s) / (1.0 - local_s).clamp_min(1e-6)
                updated = (
                    keep.unsqueeze(1) * state.latent
                    + move.unsqueeze(1) * probabilities[row, :count]
                )
                updated[0] = state.latent[0]
                updated[-1] = state.latent[-1]
                states[row] = _SequenceState(updated, insertion)

        results: list[tuple[str, str, bool] | None] = [None] * candidates
        valid = 0
        terminal_safes = []
        for row, state in enumerate(states):
            token_ids = self._project_tokens(state.latent[1:-1].float())
            safe = self.decode_tokens(token_ids)
            if len(terminal_safes) < 5:
                terminal_safes.append(safe[:512])
            smiles = self.safe_to_smiles(safe)
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                continue
            valid += 1
            is_mass_valid = self.constraint.accepts_smiles(smiles, target_mass)
            if is_mass_valid or not self.mass_shell_enabled:
                results[row] = (safe, smiles, is_mass_valid)
        diagnostics: dict[str, object] = {
            "constraint_dead_ends": 0,
            "eos_terminated": candidates,
            "max_length_terminated": sum(
                state.latent.shape[0] >= self.model.decoder_config.max_length
                for state in states
            ),
            "sample_terminal_safes": terminal_safes,
            "sample_dead_ends": [],
        }
        return results, valid, diagnostics

    @torch.no_grad()
    def generate_ranked_with_stats(
        self,
        fingerprint: torch.Tensor,
        target_mass: float,
        *,
        candidates: int = 384,
        diversity_dropout: float = 0.3,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[list[MarlinCandidate], MarlinGenerationStats]:
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        generated, valid, diagnostics = self._generate_many(
            fingerprint,
            target_mass,
            candidates=candidates,
            diversity_dropout=diversity_dropout,
            temperature=temperature,
            generator=generator,
        )
        unique: dict[str, tuple[str, str, bool]] = {}
        mass_valid = 0
        for result in generated:
            if result is None:
                continue
            safe, smiles, is_mass_valid = result
            mass_valid += int(is_mass_valid)
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            canonical = Chem.MolToSmiles(molecule, canonical=True)
            unique.setdefault(canonical, (safe, canonical, is_mass_valid))
        reference = _fingerprint_from_tensor(fingerprint)
        ranked = []
        unique_mass_valid = 0
        for safe, smiles, is_mass_valid in unique.values():
            unique_mass_valid += int(is_mass_valid)
            molecule = Chem.MolFromSmiles(smiles)
            exact_mass = Descriptors.ExactMolWt(molecule)
            ranked.append(
                MarlinCandidate(
                    smiles=smiles,
                    safe=safe,
                    tanimoto=DataStructs.TanimotoSimilarity(
                        reference,
                        AllChem.GetMorganGenerator(
                            radius=2, fpSize=fingerprint.numel()
                        ).GetFingerprint(molecule),
                    ),
                    mass_error_ppm=1e6
                    * (exact_mass - target_mass)
                    / target_mass,
                )
            )
        ranked.sort(
            key=lambda candidate: (
                -candidate.tanimoto,
                abs(candidate.mass_error_ppm),
            )
        )
        return ranked, MarlinGenerationStats(
            attempts=candidates,
            valid=valid,
            mass_valid=mass_valid,
            unique_mass_valid=unique_mass_valid,
            constraint_dead_ends=int(diagnostics["constraint_dead_ends"]),
            eos_terminated=int(diagnostics["eos_terminated"]),
            max_length_terminated=int(diagnostics["max_length_terminated"]),
            sample_terminal_safes=tuple(diagnostics["sample_terminal_safes"]),
            sample_dead_ends=tuple(diagnostics["sample_dead_ends"]),
        )

    def generate_ranked(self, *args, **kwargs) -> list[MarlinCandidate]:
        ranked, _ = self.generate_ranked_with_stats(*args, **kwargs)
        return ranked


def _fingerprint_from_tensor(fingerprint: torch.Tensor):
    bits = (fingerprint.detach().cpu().numpy() > 0.5).astype(np.uint8)
    return DataStructs.CreateFromBitString("".join(str(int(bit)) for bit in bits))


def expanding_config_dict(
    decoder: MarlinDecoderConfig, flow: ExpandingFlowConfig
) -> dict[str, dict[str, object]]:
    return {"decoder_config": asdict(decoder), "flow_config": asdict(flow)}
