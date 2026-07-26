"""Explicit FRIGID-distilled MARLIN adaptation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from marlin.warm_start import sha256_file


FRIGID_DISTILLED_MARLIN_MODE = "frigid_distilled_marlin"
MASS_ONLY_TRAINABLE_SCOPE = "mass_only"
ATTENTION_BRIDGE_TRAINABLE_SCOPE = "attention_bridge"
ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE = "attention_plus_top4_ffn"
TRAINABLE_SCOPES = frozenset(
    {
        MASS_ONLY_TRAINABLE_SCOPE,
        ATTENTION_BRIDGE_TRAINABLE_SCOPE,
        ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE,
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

    if scope == ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE:
        ffn_modules = (
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
            "norm3.weight",
            "norm3.bias",
        )
        for layer_index in range(max(0, num_layers - 4), num_layers):
            names.update(
                f"layers.{layer_index}.{suffix}" for suffix in ffn_modules
            )
    return frozenset(names)


@dataclass(frozen=True)
class FrigidDistillationSettings:
    """Configuration for the opt-in, non-reproduction adaptation mode."""

    mode: str
    trainable_scope: str = MASS_ONLY_TRAINABLE_SCOPE
    block_width_override: int = 256
    attention_mode: str = "frigid_full"
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
    selected_blocks: torch.Tensor


@dataclass(frozen=True)
class FrigidDistillationLoss:
    loss: torch.Tensor
    cross_entropy: torch.Tensor
    kl: torch.Tensor
    student_teacher_top1_agreement: torch.Tensor
    current_tokens: torch.Tensor


def build_fair_block_inputs(
    clean_ids: torch.Tensor,
    *,
    block_width: int,
    bos_token_id: int,
    pad_token_id: int,
    mask_token_id: int,
    generator: torch.Generator | None = None,
    selected_blocks: torch.Tensor | None = None,
) -> FairBlockInputs:
    """Keep the prefix clean and MASK the selected block plus its suffix.

    One block is selected independently for each row. Only tokens in that
    current block contribute to either CE or KL, so teacher and student see the
    same evidence and are scored on the same positions.
    """

    if clean_ids.ndim != 2:
        raise ValueError("clean_ids must have shape [batch, length]")
    if block_width <= 0:
        raise ValueError("block_width must be positive")

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
    current_mask = content & block_ids.eq(selected)
    current_or_suffix = content & block_ids.ge(selected)
    input_ids = clean_ids.masked_fill(current_or_suffix, mask_token_id)
    return FairBlockInputs(
        input_ids=input_ids,
        current_mask=current_mask,
        selected_blocks=selected_blocks,
    )


def frigid_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    current_mask: torch.Tensor,
    *,
    temperature: float,
    kl_weight: float,
) -> FrigidDistillationLoss:
    """Compute target CE plus temperature-scaled teacher KL on one block."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have equal shapes")
    if student_logits.shape[:-1] != targets.shape:
        raise ValueError("logits and target token shapes do not match")
    if current_mask.shape != targets.shape or current_mask.dtype != torch.bool:
        raise ValueError("current_mask must be boolean with the target shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if kl_weight < 0:
        raise ValueError("kl_weight must be non-negative")
    current_tokens = current_mask.sum()
    if current_tokens.item() == 0:
        raise ValueError("the selected block contains no target tokens")

    selected_student = student_logits[current_mask].float()
    selected_teacher = teacher_logits.detach()[current_mask].float()
    selected_targets = targets[current_mask]
    cross_entropy = F.cross_entropy(selected_student, selected_targets)
    kl = F.kl_div(
        F.log_softmax(selected_student / temperature, dim=-1),
        F.softmax(selected_teacher / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
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
