"""Explicit FRIGID-distilled MARLIN adaptation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from marlin.losses import balanced_token_target_weights
from marlin.warm_start import sha256_file


FRIGID_DISTILLED_MARLIN_MODE = "frigid_distilled_marlin"
MASS_ONLY_TRAINABLE_SCOPE = "mass_only"
ATTENTION_BRIDGE_TRAINABLE_SCOPE = "attention_bridge"
ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE = "attention_plus_top4_ffn"
ATTENTION_PLUS_TOP4_FFN_PREDICTION_HEAD_TRAINABLE_SCOPE = (
    "attention_plus_top4_ffn_prediction_head"
)
ATTENTION_PLUS_ALL_FFN_TRAINABLE_SCOPE = "attention_plus_all_ffn"
RANDOM_ROLLOUT_PREFIX_SCHEDULE = "random"
CYCLIC_ROLLOUT_PREFIX_SCHEDULE = "cyclic"
ROLLOUT_PREFIX_SCHEDULES = frozenset(
    {RANDOM_ROLLOUT_PREFIX_SCHEDULE, CYCLIC_ROLLOUT_PREFIX_SCHEDULE}
)
TRAINABLE_SCOPES = frozenset(
    {
        MASS_ONLY_TRAINABLE_SCOPE,
        ATTENTION_BRIDGE_TRAINABLE_SCOPE,
        ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE,
        ATTENTION_PLUS_TOP4_FFN_PREDICTION_HEAD_TRAINABLE_SCOPE,
        ATTENTION_PLUS_ALL_FFN_TRAINABLE_SCOPE,
    }
)


def expected_distillation_trainable_parameters(
    scope: str,
    *,
    num_layers: int,
) -> frozenset[str]:
    """Return the exact, fail-closed set of trainable decoder parameters."""

    if scope not in TRAINABLE_SCOPES:
        raise ValueError(
            f"unsupported FRIGID distillation trainable_scope: {scope!r}"
        )
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    names = {
        "conditioner.mass.projection.0.weight",
        "conditioner.mass.projection.0.bias",
        "conditioner.mass.projection.2.weight",
        "conditioner.mass.projection.2.bias",
    }
    if scope == MASS_ONLY_TRAINABLE_SCOPE:
        return frozenset(names)

    attention_modules = (
        "self_attention.in_proj_weight",
        "self_attention.in_proj_bias",
        "self_attention.out_proj.weight",
        "self_attention.out_proj.bias",
        "cross_attention.in_proj_weight",
        "cross_attention.in_proj_bias",
        "cross_attention.out_proj.weight",
        "cross_attention.out_proj.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
    )
    for layer_index in range(num_layers):
        names.update(
            f"layers.{layer_index}.{suffix}" for suffix in attention_modules
        )

    if scope in {
        ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE,
        ATTENTION_PLUS_TOP4_FFN_PREDICTION_HEAD_TRAINABLE_SCOPE,
        ATTENTION_PLUS_ALL_FFN_TRAINABLE_SCOPE,
    }:
        ffn_modules = (
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
            "norm3.weight",
            "norm3.bias",
        )
        first_ffn_layer = (
            0
            if scope == ATTENTION_PLUS_ALL_FFN_TRAINABLE_SCOPE
            else max(0, num_layers - 4)
        )
        for layer_index in range(first_ffn_layer, num_layers):
            names.update(
                f"layers.{layer_index}.{suffix}" for suffix in ffn_modules
            )
    if scope == ATTENTION_PLUS_TOP4_FFN_PREDICTION_HEAD_TRAINABLE_SCOPE:
        names.update(
            {
                "prediction_dense.weight",
                "prediction_dense.bias",
                "prediction_norm.weight",
                "prediction_norm.bias",
                "output_bias",
            }
        )
    return frozenset(names)


@dataclass(frozen=True)
class FrigidDistillationSettings:
    """Configuration for the opt-in, non-reproduction adaptation mode."""

    mode: str
    trainable_scope: str = MASS_ONLY_TRAINABLE_SCOPE
    block_width_override: int = 256
    attention_mode: str = "frigid_full"
    current_block_masking: str = "full"
    full_block_mask_probability: float = 0.0
    rollout_prefix_probability: float = 0.0
    rollout_prefix_schedule: str = RANDOM_ROLLOUT_PREFIX_SCHEDULE
    temperature: float = 2.0
    kl_weight: float = 1.0
    use_isotope: bool = False

    def __post_init__(self) -> None:
        if self.mode != FRIGID_DISTILLED_MARLIN_MODE:
            raise ValueError(
                "FRIGID distillation requires adaptation mode "
                f"{FRIGID_DISTILLED_MARLIN_MODE!r}"
            )
        if self.block_width_override <= 0:
            raise ValueError("block_width_override must be positive")
        if self.attention_mode not in {"frigid_full", "block"}:
            raise ValueError(
                "FRIGID distillation attention_mode must be 'frigid_full' or 'block'"
            )
        if self.current_block_masking not in {"full", "continuous_time"}:
            raise ValueError(
                "FRIGID distillation current_block_masking must be "
                "'full' or 'continuous_time'"
            )
        if not 0.0 <= self.full_block_mask_probability <= 1.0:
            raise ValueError("full_block_mask_probability must be in [0, 1]")
        if not 0.0 <= self.rollout_prefix_probability <= 1.0:
            raise ValueError("rollout_prefix_probability must be in [0, 1]")
        if self.rollout_prefix_schedule not in ROLLOUT_PREFIX_SCHEDULES:
            raise ValueError(
                "rollout_prefix_schedule must be 'random' or 'cyclic'"
            )
        if (
            self.rollout_prefix_probability > 0.0
            and self.current_block_masking != "continuous_time"
        ):
            raise ValueError(
                "rollout-prefix states require continuous-time current-block masking"
            )
        if self.rollout_prefix_schedule == CYCLIC_ROLLOUT_PREFIX_SCHEDULE:
            if self.rollout_prefix_probability != 1.0:
                raise ValueError(
                    "cyclic rollout-prefix schedule requires probability 1"
                )
            if self.full_block_mask_probability != 0.0:
                raise ValueError(
                    "cyclic rollout-prefix schedule requires full-block probability 0"
                )
        if (
            self.current_block_masking == "continuous_time"
            and self.attention_mode != "block"
        ):
            raise ValueError(
                "continuous-time current-block masking requires block attention"
            )
        if self.trainable_scope not in TRAINABLE_SCOPES:
            raise ValueError(
                "unsupported FRIGID distillation trainable_scope: "
                f"{self.trainable_scope!r}"
            )
        if (
            self.trainable_scope == MASS_ONLY_TRAINABLE_SCOPE
            and self.attention_mode != "frigid_full"
        ):
            raise ValueError("mass_only adaptation requires frigid_full attention")
        if (
            self.trainable_scope != MASS_ONLY_TRAINABLE_SCOPE
            and self.attention_mode != "block"
        ):
            raise ValueError(
                "decoder adaptation scopes require exact block attention"
            )
        if self.temperature <= 0:
            raise ValueError("distillation temperature must be positive")
        if self.kl_weight < 0:
            raise ValueError("distillation KL weight must be non-negative")
        if self.use_isotope:
            raise ValueError("FRIGID distillation must omit isotope conditioning")


@dataclass(frozen=True)
class FairBlockInputs:
    input_ids: torch.Tensor
    current_mask: torch.Tensor
    loss_mask: torch.Tensor
    current_content_mask: torch.Tensor
    selected_blocks: torch.Tensor
    mask_probabilities: torch.Tensor
    loss_weights: torch.Tensor
    full_block_masked: torch.Tensor
    rollout_prefix_masked: torch.Tensor


@dataclass(frozen=True)
class FrigidDistillationLoss:
    loss: torch.Tensor
    cross_entropy: torch.Tensor
    kl: torch.Tensor
    student_teacher_top1_agreement: torch.Tensor
    current_tokens: torch.Tensor
    selected_target_weight: torch.Tensor


def build_fair_block_inputs(
    clean_ids: torch.Tensor,
    *,
    block_width: int,
    bos_token_id: int,
    pad_token_id: int,
    mask_token_id: int,
    generator: torch.Generator | None = None,
    selected_blocks: torch.Tensor | None = None,
    current_block_masking: str = "full",
    full_block_mask_probability: float = 0.0,
    rollout_prefix_probability: float = 0.0,
    rollout_prefix_schedule: str = RANDOM_ROLLOUT_PREFIX_SCHEDULE,
    rollout_prefix_step: int | None = None,
    current_mask_probabilities: torch.Tensor | None = None,
) -> FairBlockInputs:
    """Build a fair current-block state for teacher and two-stream student.

    Full masking preserves the stage-zero bridge. Continuous-time masking
    samples ``t ~ U(0, 1]`` and independently masks current-block tokens with
    probability ``t``. A configurable auxiliary path exposes a clean prefix
    and masks the remainder of the current block, matching production grammar
    rollouts. Future blocks become PAD so the full-attention teacher cannot
    exploit a target-length-bearing MASK suffix that the two-stream student
    cannot attend to. Diffusion rows score all masked current-block tokens;
    rollout rows score only the leftmost unresolved token because that is the
    only action admitted by the production SAFE grammar before logits are
    recomputed.
    """

    if clean_ids.ndim != 2:
        raise ValueError("clean_ids must have shape [batch, length]")
    if block_width <= 0:
        raise ValueError("block_width must be positive")
    if current_block_masking not in {"full", "continuous_time"}:
        raise ValueError(
            "current_block_masking must be 'full' or 'continuous_time'"
        )
    if not 0.0 <= full_block_mask_probability <= 1.0:
        raise ValueError("full_block_mask_probability must be in [0, 1]")
    if not 0.0 <= rollout_prefix_probability <= 1.0:
        raise ValueError("rollout_prefix_probability must be in [0, 1]")
    if rollout_prefix_schedule not in ROLLOUT_PREFIX_SCHEDULES:
        raise ValueError("rollout_prefix_schedule must be 'random' or 'cyclic'")
    if rollout_prefix_schedule == CYCLIC_ROLLOUT_PREFIX_SCHEDULE:
        if rollout_prefix_probability != 1.0:
            raise ValueError("cyclic rollout-prefix schedule requires probability 1")
        if full_block_mask_probability != 0.0:
            raise ValueError(
                "cyclic rollout-prefix schedule requires full-block probability 0"
            )
        if (
            isinstance(rollout_prefix_step, bool)
            or not isinstance(rollout_prefix_step, int)
            or rollout_prefix_step < 0
        ):
            raise ValueError(
                "cyclic rollout-prefix schedule requires a non-negative integer step"
            )
        if current_mask_probabilities is not None:
            raise ValueError(
                "cyclic rollout-prefix schedule forbids random mask probabilities"
            )
    elif rollout_prefix_step is not None:
        raise ValueError("rollout_prefix_step requires cyclic rollout-prefix schedule")
    if current_block_masking == "full" and rollout_prefix_probability > 0.0:
        raise ValueError(
            "rollout_prefix_probability requires continuous-time masking"
        )
    if (
        current_block_masking == "full"
        and current_mask_probabilities is not None
    ):
        raise ValueError(
            "current_mask_probabilities require continuous-time masking"
        )

    batch_size, length = clean_ids.shape
    positions = torch.arange(length, device=clean_ids.device).reshape(1, -1)
    block_ids = (positions - 1).clamp_min(0).div(
        block_width, rounding_mode="floor"
    )
    content = clean_ids.ne(pad_token_id) & positions.ne(0)
    if (clean_ids[:, 0] != bos_token_id).any():
        raise ValueError("every distillation sequence must start with BOS")
    if not content.any(dim=1).all():
        raise ValueError("every distillation sequence must contain a target token")

    block_counts = (
        block_ids.expand(batch_size, -1)
        .masked_fill(~content, -1)
        .amax(dim=1)
        .add(1)
    )
    if selected_blocks is None:
        random_values = torch.rand(
            batch_size,
            device=clean_ids.device,
            generator=generator,
        )
        selected_blocks = torch.floor(random_values * block_counts).long()
    else:
        selected_blocks = selected_blocks.to(
            device=clean_ids.device, dtype=torch.long
        )
        if selected_blocks.shape != (batch_size,):
            raise ValueError("selected_blocks must have shape [batch]")
        if ((selected_blocks < 0) | (selected_blocks >= block_counts)).any():
            raise ValueError("selected block is outside a sequence")

    selected = selected_blocks.reshape(-1, 1)
    current_content = content & block_ids.eq(selected)
    suffix = content & block_ids.gt(selected)
    fallback_rows = torch.zeros(
        batch_size,
        device=clean_ids.device,
        dtype=torch.bool,
    )
    if current_block_masking == "full":
        mask_probabilities = torch.ones(
            batch_size,
            device=clean_ids.device,
            dtype=torch.float32,
        )
        full_block_masked = torch.ones(
            batch_size,
            device=clean_ids.device,
            dtype=torch.bool,
        )
        rollout_prefix_masked = torch.zeros(
            batch_size,
            device=clean_ids.device,
            dtype=torch.bool,
        )
        current_mask = current_content
        input_ids = clean_ids.masked_fill(
            current_content | suffix,
            mask_token_id,
        )
    else:
        current_lengths = current_content.sum(dim=1)
        within_block = (positions - 1).clamp_min(0).remainder(block_width)
        if rollout_prefix_schedule == CYCLIC_ROLLOUT_PREFIX_SCHEDULE:
            step = torch.full_like(current_lengths, rollout_prefix_step)
            revealed_counts = torch.remainder(step, current_lengths)
            mask_probabilities = (
                (current_lengths - revealed_counts).float()
                / current_lengths.float()
            )
            rollout_prefix_masked = torch.ones(
                batch_size,
                device=clean_ids.device,
                dtype=torch.bool,
            )
            current_mask = current_content & within_block.ge(
                revealed_counts.reshape(-1, 1)
            )
        elif current_mask_probabilities is None:
            mask_probabilities = torch.rand(
                batch_size,
                device=clean_ids.device,
                generator=generator,
            ).clamp_min(1e-4)
            full_block_masked = (
                torch.rand(
                    batch_size,
                    device=clean_ids.device,
                    generator=generator,
                )
                < full_block_mask_probability
            )
            mask_probabilities = mask_probabilities.masked_fill(
                full_block_masked,
                1.0,
            )
        else:
            mask_probabilities = current_mask_probabilities.to(
                device=clean_ids.device,
                dtype=torch.float32,
            )
            if mask_probabilities.shape != (batch_size,):
                raise ValueError(
                    "current_mask_probabilities must have shape [batch]"
                )
            if (
                ~torch.isfinite(mask_probabilities)
                | (mask_probabilities <= 0)
                | (mask_probabilities > 1)
            ).any():
                raise ValueError(
                    "current_mask_probabilities must be finite and in (0, 1]"
                )
            full_block_masked = mask_probabilities.eq(1.0)
        if rollout_prefix_schedule == RANDOM_ROLLOUT_PREFIX_SCHEDULE:
            mask_draws = torch.rand(
                clean_ids.shape,
                device=clean_ids.device,
                generator=generator,
            )
            current_mask = (
                current_content
                & mask_draws.lt(mask_probabilities.reshape(-1, 1))
            )
            rollout_prefix_masked = (
                torch.rand(
                    batch_size,
                    device=clean_ids.device,
                    generator=generator,
                )
                < rollout_prefix_probability
            ) & ~full_block_masked
            # With t ~ Uniform(0, 1], floor((1 - t) * length) uniformly
            # selects every production prefix length from zero through
            # length - 1, including the BOS-only first-token action.
            revealed_counts = torch.floor(
                (1.0 - mask_probabilities) * current_lengths.float()
            ).long()
            revealed_counts = torch.minimum(
                revealed_counts,
                (current_lengths - 1).clamp_min(0),
            )
            prefix_suffix_mask = current_content & within_block.ge(
                revealed_counts.reshape(-1, 1)
            )
            current_mask = torch.where(
                rollout_prefix_masked.reshape(-1, 1),
                prefix_suffix_mask,
                current_mask,
            )
        if not current_mask.any():
            fallback_row = int(mask_probabilities.argmax().item())
            first_current_positions = current_content.float().argmax(dim=1)
            current_mask[fallback_row, first_current_positions[fallback_row]] = True
            fallback_rows[fallback_row] = True
        full_block_masked = (current_mask | ~current_content).all(dim=1)
        input_ids = clean_ids.masked_fill(current_mask, mask_token_id)
        input_ids = input_ids.masked_fill(suffix, pad_token_id)
    loss_mask = current_mask.clone()
    if rollout_prefix_masked.any():
        first_unresolved = current_mask.float().argmax(dim=1)
        rollout_loss_mask = torch.zeros_like(current_mask)
        rollout_rows = torch.nonzero(
            rollout_prefix_masked,
            as_tuple=False,
        ).flatten()
        rollout_loss_mask[
            rollout_rows,
            first_unresolved[rollout_rows],
        ] = True
        loss_mask = torch.where(
            rollout_prefix_masked.reshape(-1, 1),
            rollout_loss_mask,
            loss_mask,
        )
    loss_weights = torch.zeros(
        clean_ids.shape,
        device=clean_ids.device,
        dtype=torch.float32,
    )
    inverse_probabilities = mask_probabilities.reciprocal().masked_fill(
        fallback_rows,
        1.0,
    )
    inverse_probabilities = torch.where(
        rollout_prefix_masked,
        current_content.sum(dim=1).to(dtype=torch.float32),
        inverse_probabilities,
    )
    loss_weights = loss_weights.masked_scatter(
        loss_mask,
        inverse_probabilities
        .reshape(-1, 1)
        .expand_as(clean_ids)[loss_mask],
    )
    return FairBlockInputs(
        input_ids=input_ids,
        current_mask=current_mask,
        loss_mask=loss_mask,
        current_content_mask=current_content,
        selected_blocks=selected_blocks,
        mask_probabilities=mask_probabilities,
        loss_weights=loss_weights,
        full_block_masked=full_block_masked,
        rollout_prefix_masked=rollout_prefix_masked,
    )


def frigid_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    current_mask: torch.Tensor,
    *,
    temperature: float,
    kl_weight: float,
    loss_weights: torch.Tensor | None = None,
    normalization_mask: torch.Tensor | None = None,
    balanced_token_loss_alpha: float = 0.0,
    token_loss_weight_max: float = 20.0,
    eos_token_id: int | None = None,
    target_balance_mask: torch.Tensor | None = None,
) -> FrigidDistillationLoss:
    """Compute weighted target CE plus temperature-scaled teacher KL."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have equal shapes")
    if student_logits.shape[:-1] != targets.shape:
        raise ValueError("logits and target token shapes do not match")
    if current_mask.shape != targets.shape or current_mask.dtype != torch.bool:
        raise ValueError("current_mask must be boolean with the target shape")
    if loss_weights is None:
        loss_weights = torch.ones_like(targets, dtype=torch.float32)
    if loss_weights.shape != targets.shape:
        raise ValueError("loss_weights must have the target shape")
    if (~torch.isfinite(loss_weights) | (loss_weights < 0)).any():
        raise ValueError("loss_weights must be finite and non-negative")
    if normalization_mask is not None:
        if (
            normalization_mask.shape != targets.shape
            or normalization_mask.dtype != torch.bool
        ):
            raise ValueError(
                "normalization_mask must be boolean with the target shape"
            )
        if (current_mask & ~normalization_mask).any():
            raise ValueError(
                "current_mask must be contained in normalization_mask"
            )
        if not normalization_mask.any(dim=1).all():
            raise ValueError(
                "normalization_mask must contain a target in every row"
            )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if kl_weight < 0:
        raise ValueError("kl_weight must be non-negative")
    current_tokens = current_mask.sum()
    if current_tokens.item() == 0:
        raise ValueError("the selected block contains no target tokens")

    if target_balance_mask is None or eos_token_id is None:
        if balanced_token_loss_alpha != 0:
            raise ValueError(
                "balanced distillation requires target_balance_mask and eos_token_id"
            )
        balance_mask = torch.zeros_like(current_mask)
        effective_eos_token_id = 0
    else:
        balance_mask = target_balance_mask
        effective_eos_token_id = eos_token_id
    target_weights = balanced_token_target_weights(
        targets,
        balance_mask,
        vocab_size=student_logits.shape[-1],
        eos_token_id=effective_eos_token_id,
        alpha=balanced_token_loss_alpha,
        weight_max=token_loss_weight_max,
    ).to(device=student_logits.device)

    selected_student = student_logits[current_mask].float()
    selected_teacher = teacher_logits.detach()[current_mask].float()
    selected_targets = targets[current_mask]
    selected_sampling_weights = loss_weights[current_mask].float()
    selected_target_weights = target_weights[current_mask].float()
    if selected_sampling_weights.sum().item() <= 0:
        raise ValueError("masked current tokens must have positive loss weights")
    token_cross_entropy = F.cross_entropy(
        selected_student,
        selected_targets,
        reduction="none",
    )
    token_kl = F.kl_div(
        F.log_softmax(selected_student / temperature, dim=-1),
        F.softmax(selected_teacher / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1) * temperature**2
    if normalization_mask is None:
        weight_sum = selected_sampling_weights.sum()
        cross_entropy = (
            token_cross_entropy
            * selected_sampling_weights
            * selected_target_weights
        ).sum() / weight_sum
        kl = (token_kl * selected_sampling_weights).sum() / weight_sum
    else:
        batch_size = targets.shape[0]
        row_indices = (
            torch.arange(batch_size, device=targets.device)
            .reshape(-1, 1)
            .expand_as(targets)[current_mask]
        )
        cross_entropy_by_row = torch.zeros(
            batch_size,
            device=selected_student.device,
            dtype=selected_student.dtype,
        )
        kl_by_row = torch.zeros_like(cross_entropy_by_row)
        cross_entropy_by_row.scatter_add_(
            0,
            row_indices,
            token_cross_entropy
            * selected_sampling_weights
            * selected_target_weights,
        )
        kl_by_row.scatter_add_(
            0,
            row_indices,
            token_kl * selected_sampling_weights,
        )
        fixed_denominator = normalization_mask.sum(dim=1).to(
            dtype=selected_student.dtype
        )
        cross_entropy = (cross_entropy_by_row / fixed_denominator).mean()
        kl = (kl_by_row / fixed_denominator).mean()
    agreement = (
        selected_student.argmax(dim=-1)
        .eq(selected_teacher.argmax(dim=-1))
        .float()
        .mean()
    )
    return FrigidDistillationLoss(
        loss=cross_entropy + kl_weight * kl,
        cross_entropy=cross_entropy,
        kl=kl,
        student_teacher_top1_agreement=agreement,
        current_tokens=current_tokens,
        selected_target_weight=selected_target_weights.mean(),
    )


class FrozenFrigidTeacher:
    """Official FRIGID model with native formula/fingerprint conditioning.

    This wrapper deliberately is not an ``nn.Module``. The teacher therefore
    cannot enter the MARLIN checkpoint or optimizer by accidental registration.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        expected_sha256: str,
        expected_tokenizer_vocab: dict[str, int] | None = None,
        expected_special_token_ids: dict[str, int] | None = None,
    ) -> "FrozenFrigidTeacher":
        checkpoint_path = Path(checkpoint_path)
        actual_sha256 = sha256_file(checkpoint_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"FRIGID teacher SHA-256 is {actual_sha256}; "
                f"expected {expected_sha256}"
            )

        from dlm.model import DLM

        model = DLM.load_from_checkpoint(
            str(checkpoint_path),
            map_location="cpu",
            strict=True,
        )
        if model.ema is None:
            raise ValueError("official FRIGID teacher checkpoint must contain EMA")
        model.ema.copy_to(model.backbone.parameters())
        model.ema = None
        teacher = cls(model)
        teacher.validate_student_tokenizer(
            expected_tokenizer_vocab,
            expected_special_token_ids,
        )
        return teacher

    def validate_student_tokenizer(
        self,
        expected_vocab: dict[str, int] | None,
        expected_special_token_ids: dict[str, int] | None,
    ) -> None:
        if expected_vocab is not None:
            tokenizer = self.model.tokenizer
            teacher_vocab = dict(tokenizer.get_vocab())
            tokenizer_vocab_size = int(getattr(tokenizer, "vocab_size"))
            backbone = getattr(self.model, "backbone", None)
            backbone_config = getattr(backbone, "config", None)
            model_vocab_size = int(
                getattr(backbone_config, "vocab_size", tokenizer_vocab_size)
            )
            if model_vocab_size != tokenizer_vocab_size:
                raise ValueError(
                    "FRIGID tokenizer/model vocabulary sizes differ: "
                    f"tokenizer={tokenizer_vocab_size}, model={model_vocab_size}"
                )

            expected_vocab = dict(expected_vocab)
            expected_ids = set(expected_vocab.values())
            if expected_ids != set(range(model_vocab_size)):
                raise ValueError(
                    "student tokenizer IDs do not exactly cover the FRIGID model "
                    f"vocabulary [0, {model_vocab_size})"
                )
            teacher_model_vocab = {
                token: token_id
                for token, token_id in teacher_vocab.items()
                if token_id < model_vocab_size
            }
            if teacher_model_vocab != expected_vocab:
                keys = set(teacher_model_vocab) | set(expected_vocab)
                mismatches = sorted(
                    key
                    for key in keys
                    if teacher_model_vocab.get(key) != expected_vocab.get(key)
                )
                raise ValueError(
                    "FRIGID teacher/student model vocabularies differ at: "
                    + ", ".join(mismatches[:5])
                )
        if expected_special_token_ids is not None:
            for name in ("unk", "bos", "eos", "pad", "mask"):
                expected = expected_special_token_ids.get(name)
                actual = getattr(self.model.tokenizer, f"{name}_token_id")
                if expected is not None and actual != expected:
                    raise ValueError(
                        f"FRIGID teacher/student {name} token IDs differ: "
                        f"teacher={actual}, student={expected}"
                    )

    def to(self, device: torch.device | str) -> "FrozenFrigidTeacher":
        self.model.to(device)
        return self

    def eval(self) -> "FrozenFrigidTeacher":
        self.model.eval()
        return self

    @torch.no_grad()
    def logits(
        self,
        input_ids: torch.Tensor,
        formulas: Sequence[str],
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        if len(formulas) != input_ids.shape[0]:
            raise ValueError("one molecular formula is required per teacher input")
        attention_mask = input_ids.ne(self.model.tokenizer.pad_token_id)
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            formula=list(formulas),
            fingerprint=fingerprint.float(),
        )


@dataclass(frozen=True)
class FrozenFrigidTeacherCheckpoint:
    """Small serializable spec that lazy-loads the 2.5 GB teacher per rank."""

    checkpoint_path: str
    expected_sha256: str
    expected_tokenizer_vocab: dict[str, int]
    expected_special_token_ids: dict[str, int]

    def load(self) -> FrozenFrigidTeacher:
        return FrozenFrigidTeacher.from_checkpoint(
            self.checkpoint_path,
            expected_sha256=self.expected_sha256,
            expected_tokenizer_vocab=self.expected_tokenizer_vocab,
            expected_special_token_ids=self.expected_special_token_ids,
        )
