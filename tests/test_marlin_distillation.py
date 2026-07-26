import importlib.util
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from marlin.distillation import (
    ATTENTION_BRIDGE_TRAINABLE_SCOPE,
    ATTENTION_PLUS_ALL_FFN_TRAINABLE_SCOPE,
    ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE,
    FRIGID_DISTILLED_MARLIN_MODE,
    MASS_ONLY_TRAINABLE_SCOPE,
    FrozenFrigidTeacher,
    FrigidDistillationSettings,
    build_fair_block_inputs,
    expected_distillation_trainable_parameters,
    frigid_distillation_loss,
)
from marlin.ema import AllParameterExponentialMovingAverage
from marlin.model import MarlinDecoderConfig
from marlin.training import ClearMLScalarCallback, MarlinLightningModule


def _tiny_config() -> MarlinDecoderConfig:
    return MarlinDecoderConfig(
        vocab_size=7,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=8,
        block_width=2,
        fingerprint_bits=8,
        dropout=0.0,
        mask_token_id=4,
        pad_token_id=0,
    )


def test_fair_block_inputs_keep_prefix_and_mask_current_suffix():
    clean = torch.tensor([[1, 5, 6, 5, 6, 2, 0]])

    fair = build_fair_block_inputs(
        clean,
        block_width=2,
        bos_token_id=1,
        pad_token_id=0,
        mask_token_id=4,
        selected_blocks=torch.tensor([1]),
    )

    assert fair.input_ids.tolist() == [[1, 5, 6, 4, 4, 4, 0]]
    assert fair.current_mask.tolist() == [
        [False, False, False, True, True, False, False]
    ]
    assert fair.current_content_mask.tolist() == fair.current_mask.tolist()
    assert fair.loss_mask.tolist() == fair.current_mask.tolist()
    assert fair.selected_blocks.tolist() == [1]
    assert fair.mask_probabilities.tolist() == [1.0]
    assert fair.loss_weights.tolist() == [
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    ]
    assert fair.full_block_masked.tolist() == [True]
    assert fair.rollout_prefix_masked.tolist() == [False]


def test_continuous_time_block_inputs_cover_partial_rollout_states():
    clean = torch.tensor([[1, 5, 6, 5, 6, 5, 6, 2, 0]])
    fair = build_fair_block_inputs(
        clean,
        block_width=4,
        bos_token_id=1,
        pad_token_id=0,
        mask_token_id=4,
        selected_blocks=torch.tensor([0]),
        current_block_masking="continuous_time",
        current_mask_probabilities=torch.tensor([0.5]),
        generator=torch.Generator().manual_seed(0),
    )

    assert fair.current_mask[0, 1:5].any()
    assert not fair.current_mask[0, 1:5].all()
    assert torch.equal(fair.input_ids[0, 5:8], torch.tensor([0, 0, 0]))
    assert fair.input_ids[0, 1:5].eq(4).any()
    assert fair.input_ids[0, 1:5].ne(4).any()
    assert torch.equal(fair.loss_mask, fair.current_mask)
    assert fair.loss_weights[fair.current_mask].eq(2.0).all()
    assert fair.full_block_masked.tolist() == [False]
    assert fair.rollout_prefix_masked.tolist() == [False]


def test_rollout_prefix_inputs_match_production_left_to_right_states():
    clean = torch.tensor([[1, 5, 6, 5, 6, 2, 0]])
    fair = build_fair_block_inputs(
        clean,
        block_width=4,
        bos_token_id=1,
        pad_token_id=0,
        mask_token_id=4,
        selected_blocks=torch.tensor([0]),
        current_block_masking="continuous_time",
        current_mask_probabilities=torch.tensor([0.5]),
        rollout_prefix_probability=1.0,
        generator=torch.Generator().manual_seed(0),
    )

    assert fair.input_ids.tolist() == [[1, 5, 6, 4, 4, 0, 0]]
    assert fair.current_mask.tolist() == [
        [False, False, False, True, True, False, False]
    ]
    assert fair.loss_mask.tolist() == [
        [False, False, False, True, False, False, False]
    ]
    assert fair.loss_weights[fair.loss_mask].tolist() == [4.0]
    assert not fair.loss_weights[fair.current_mask & ~fair.loss_mask].any()
    assert fair.full_block_masked.tolist() == [False]
    assert fair.rollout_prefix_masked.tolist() == [True]


def test_rollout_prefix_reveal_counts_cover_every_production_action():
    clean = torch.tensor(
        [
            [1, 5, 6, 5, 2],
            [1, 5, 6, 5, 2],
            [1, 5, 6, 5, 2],
            [1, 5, 6, 5, 2],
        ]
    )

    fair = build_fair_block_inputs(
        clean,
        block_width=4,
        bos_token_id=1,
        pad_token_id=0,
        mask_token_id=4,
        selected_blocks=torch.zeros(4, dtype=torch.long),
        current_block_masking="continuous_time",
        current_mask_probabilities=torch.tensor([0.99, 0.70, 0.49, 0.20]),
        rollout_prefix_probability=1.0,
        generator=torch.Generator().manual_seed(0),
    )

    assert fair.input_ids.tolist() == [
        [1, 4, 4, 4, 4],
        [1, 5, 4, 4, 4],
        [1, 5, 6, 4, 4],
        [1, 5, 6, 5, 4],
    ]
    assert fair.loss_mask.tolist() == [
        [False, True, False, False, False],
        [False, False, True, False, False],
        [False, False, False, True, False],
        [False, False, False, False, True],
    ]
    assert fair.loss_weights[fair.loss_mask].tolist() == [4.0] * 4
    assert fair.rollout_prefix_masked.tolist() == [True] * 4
    assert fair.full_block_masked.tolist() == [True, False, False, False]


def test_empty_continuous_time_microbatch_uses_bounded_fallback_weight():
    clean = torch.tensor([[1, 5, 6, 5, 6, 0]])
    fair = build_fair_block_inputs(
        clean,
        block_width=4,
        bos_token_id=1,
        pad_token_id=0,
        mask_token_id=4,
        selected_blocks=torch.tensor([0]),
        current_block_masking="continuous_time",
        current_mask_probabilities=torch.tensor([1e-4]),
        generator=torch.Generator().manual_seed(0),
    )

    assert fair.current_mask.sum().item() == 1
    assert torch.equal(fair.loss_mask, fair.current_mask)
    assert fair.loss_weights[fair.current_mask].tolist() == [1.0]
    assert fair.rollout_prefix_masked.tolist() == [False]


def test_frigid_distillation_loss_ignores_logits_outside_current_block():
    targets = torch.tensor([[1, 2, 3]])
    current = torch.tensor([[False, True, False]])
    student = torch.tensor(
        [[[30.0, -30.0, 0.0, 0.0], [0.0, 2.0, 1.0, 0.0], [-20.0, 20.0, 0.0, 0.0]]]
    )
    teacher = torch.tensor(
        [[[-30.0, 30.0, 0.0, 0.0], [0.0, 0.5, 2.5, 0.0], [20.0, -20.0, 0.0, 0.0]]]
    )

    first = frigid_distillation_loss(
        student,
        teacher,
        targets,
        current,
        temperature=2.0,
        kl_weight=0.75,
    )
    student[:, (0, 2)] *= -100
    teacher[:, (0, 2)] *= -100
    second = frigid_distillation_loss(
        student,
        teacher,
        targets,
        current,
        temperature=2.0,
        kl_weight=0.75,
    )

    assert torch.allclose(first.loss, first.cross_entropy + 0.75 * first.kl)
    assert torch.allclose(first.loss, second.loss)
    assert torch.allclose(first.cross_entropy, second.cross_entropy)
    assert torch.allclose(first.kl, second.kl)


def test_frigid_distillation_loss_applies_inverse_time_weights():
    targets = torch.tensor([[0, 0]])
    current = torch.tensor([[True, True]])
    student = torch.tensor([[[0.0, 0.0], [0.0, 2.0]]])
    teacher = torch.zeros_like(student)
    weights = torch.tensor([[1.0, 3.0]])

    result = frigid_distillation_loss(
        student,
        teacher,
        targets,
        current,
        temperature=1.0,
        kl_weight=0.0,
        loss_weights=weights,
        normalization_mask=current,
    )
    per_token = torch.nn.functional.cross_entropy(
        student.reshape(-1, 2),
        targets.reshape(-1),
        reduction="none",
    )

    assert torch.allclose(
        result.cross_entropy,
        (per_token * weights.reshape(-1)).sum() / current.sum(),
    )


def test_all_parameter_ema_preserves_frozen_parameter_order():
    first = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    frozen = torch.nn.Parameter(torch.tensor([10.0]), requires_grad=False)
    ema = AllParameterExponentialMovingAverage([first, frozen], decay=0.5)

    first.data.fill_(3.0)
    frozen.data.fill_(30.0)
    ema.update([first, frozen])
    first.data.fill_(-1.0)
    frozen.data.fill_(-1.0)
    ema.store([first, frozen])
    ema.copy_to([first, frozen])

    assert first.item() == 2.0
    assert frozen.item() == 20.0
    assert len(ema.shadow_params) == 2

    ema.restore([first, frozen])
    assert first.item() == -1.0
    assert frozen.item() == -1.0


def test_stage0_freezes_pretrained_student_and_omits_isotope():
    class FrozenTeacher:
        def __init__(self):
            self.calls = []

        def to(self, _device):
            return self

        def eval(self):
            return self

        @torch.no_grad()
        def logits(self, input_ids, formulas, fingerprint):
            self.calls.append((input_ids.clone(), list(formulas), fingerprint.clone()))
            logits = torch.zeros((*input_ids.shape, 7), device=input_ids.device)
            logits[..., 5] = 2.0
            return logits

    class FailIfCalled(torch.nn.Module):
        def forward(self, _ratios):
            raise AssertionError("stage-0 student must not consume isotope ratios")

    teacher = FrozenTeacher()
    module = MarlinLightningModule(
        _tiny_config(),
        distillation=FrigidDistillationSettings(
            mode=FRIGID_DISTILLED_MARLIN_MODE,
            block_width_override=8,
            temperature=2.0,
            kl_weight=1.0,
        ),
        frigid_teacher=teacher,
    )
    module.decoder.conditioner.isotope = FailIfCalled()
    batch = {
        "input_ids": torch.tensor([[1, 5, 6, 2, 0]]),
        "fingerprint": torch.tensor(
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "precursor_mass": torch.tensor([44.0]),
        "isotope_ratios": torch.tensor([[0.1, 0.01]]),
        "formula": ["C2H4O"],
    }

    with patch.object(
        module.decoder, "forward", wraps=module.decoder.forward
    ) as student_forward:
        loss, metrics = module.frigid_distillation_objective(batch)

    assert student_forward.call_args.kwargs["attention_mode"] == "frigid_full"

    trainable = {
        name for name, parameter in module.decoder.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("conditioner.mass.") for name in trainable)
    assert not any("frigid_teacher" in name for name in module.state_dict())
    assert teacher.calls[0][0].tolist() == [[1, 4, 4, 4, 0]]
    assert teacher.calls[0][1] == ["C2H4O"]
    assert torch.isfinite(loss)
    assert metrics["current_tokens"].item() == 3


def test_block_distillation_uses_two_stream_and_exact_attention_scope():
    class FrozenTeacher:
        def __init__(self):
            self.input_ids = None

        @torch.no_grad()
        def logits(self, input_ids, formulas, fingerprint):
            del formulas, fingerprint
            self.input_ids = input_ids.clone()
            return torch.zeros((*input_ids.shape, 7), device=input_ids.device)

    teacher = FrozenTeacher()
    module = MarlinLightningModule(
        _tiny_config(),
        distillation=FrigidDistillationSettings(
            mode=FRIGID_DISTILLED_MARLIN_MODE,
            trainable_scope=ATTENTION_BRIDGE_TRAINABLE_SCOPE,
            attention_mode="block",
            block_width_override=2,
            current_block_masking="continuous_time",
            full_block_mask_probability=1.0,
            temperature=2.0,
            kl_weight=0.5,
        ),
        frigid_teacher=teacher,
    )
    batch = {
        "input_ids": torch.tensor([[1, 5, 6, 2, 0]]),
        "fingerprint": torch.tensor(
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "precursor_mass": torch.tensor([44.0]),
        "formula": ["C2H4O"],
    }

    with patch.object(
        module.decoder,
        "forward",
        side_effect=AssertionError("block adaptation must not call forward"),
    ), patch.object(
        module.decoder,
        "two_stream_logits",
        wraps=module.decoder.two_stream_logits,
    ) as two_stream:
        loss, metrics = module.frigid_distillation_objective(
            batch,
            generator=torch.Generator().manual_seed(5),
        )

    clean_ids, noised_ids = two_stream.call_args.args[:2]
    assert torch.equal(clean_ids, batch["input_ids"])
    assert torch.equal(noised_ids, teacher.input_ids)
    expected = expected_distillation_trainable_parameters(
        ATTENTION_BRIDGE_TRAINABLE_SCOPE,
        num_layers=module.decoder.config.num_layers,
    )
    actual = {
        name
        for name, parameter in module.decoder.named_parameters()
        if parameter.requires_grad
    }
    assert actual == set(expected)
    assert torch.isfinite(loss)
    assert metrics["distillation_mask_probability"].item() == 1.0
    assert metrics["distillation_full_block_mask_fraction"].item() == 1.0


def test_block_distillation_reports_deterministic_revealed_context_response():
    class InputSensitiveTeacher:
        def logits(self, input_ids, formulas, fingerprint):
            del formulas, fingerprint
            previous = torch.cat(
                (torch.zeros_like(input_ids[:, :1]), input_ids[:, :-1]),
                dim=1,
            )
            return torch.nn.functional.one_hot(previous, num_classes=7).float()

    module = MarlinLightningModule(
        _tiny_config(),
        distillation=FrigidDistillationSettings(
            mode=FRIGID_DISTILLED_MARLIN_MODE,
            trainable_scope=ATTENTION_BRIDGE_TRAINABLE_SCOPE,
            attention_mode="block",
            block_width_override=2,
            current_block_masking="continuous_time",
            full_block_mask_probability=0.0,
            rollout_prefix_probability=1.0,
            temperature=2.0,
            kl_weight=0.5,
        ),
        frigid_teacher=InputSensitiveTeacher(),
    )
    batch = {
        "input_ids": torch.tensor([[1, 5, 6, 5, 6, 5, 6]]),
        "fingerprint": torch.tensor(
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "precursor_mass": torch.tensor([44.0]),
        "formula": ["C2H4O"],
    }

    loss, metrics = module.frigid_distillation_objective(
        batch,
        generator=torch.Generator().manual_seed(5),
        collect_context_metrics=True,
    )

    assert torch.isfinite(loss)
    assert metrics["distillation_context_tokens"].item() > 0
    assert metrics["distillation_student_context_logit_l1"].item() > 0
    assert metrics["distillation_teacher_context_logit_l1"].item() > 0
    assert torch.isfinite(metrics["distillation_context_response_ratio"])
    assert not module.decoder.training


@pytest.mark.parametrize(
    ("scope", "attention_mode"),
    (
        (MASS_ONLY_TRAINABLE_SCOPE, "frigid_full"),
        (ATTENTION_BRIDGE_TRAINABLE_SCOPE, "block"),
        (ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE, "block"),
        (ATTENTION_PLUS_ALL_FFN_TRAINABLE_SCOPE, "block"),
    ),
)
def test_distillation_scopes_disable_dropout_but_keep_selected_gradients(
    scope,
    attention_mode,
):
    class ConstantTeacher:
        def logits(self, input_ids, formulas, fingerprint):
            del formulas, fingerprint
            return torch.zeros((*input_ids.shape, 7), device=input_ids.device)

    config = replace(_tiny_config(), block_width=8, dropout=0.5)
    module = MarlinLightningModule(
        config,
        distillation=FrigidDistillationSettings(
            mode=FRIGID_DISTILLED_MARLIN_MODE,
            trainable_scope=scope,
            attention_mode=attention_mode,
            block_width_override=8,
            current_block_masking="full",
            temperature=2.0,
            kl_weight=0.5,
        ),
        frigid_teacher=ConstantTeacher(),
    )
    module.train()

    assert module.training
    assert not module.decoder.training
    assert all(
        not submodule.training
        for submodule in module.decoder.modules()
        if isinstance(submodule, torch.nn.Dropout)
    )

    expected = expected_distillation_trainable_parameters(
        scope,
        num_layers=module.decoder.config.num_layers,
    )
    named_parameters = dict(module.decoder.named_parameters())
    actual = {
        name for name, parameter in named_parameters.items() if parameter.requires_grad
    }
    assert actual == set(expected)
    optimizer = module.configure_optimizers()
    optimized = {
        id(parameter)
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    }
    assert optimized == {id(named_parameters[name]) for name in expected}

    batch = {
        "input_ids": torch.tensor([[1, 5, 6, 2, 0]]),
        "fingerprint": torch.tensor(
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "precursor_mass": torch.tensor([44.0]),
        "formula": ["C2H4O"],
    }
    loss, _ = module.frigid_distillation_objective(
        batch,
        generator=torch.Generator().manual_seed(5),
    )
    loss.backward()

    assert {
        name for name in expected if named_parameters[name].grad is not None
    } == set(expected)


def test_block_context_metrics_keep_keys_when_batch_has_no_revealed_tokens():
    class ConstantTeacher:
        def logits(self, input_ids, formulas, fingerprint):
            del formulas, fingerprint
            return torch.zeros((*input_ids.shape, 7), device=input_ids.device)

    module = MarlinLightningModule(
        _tiny_config(),
        distillation=FrigidDistillationSettings(
            mode=FRIGID_DISTILLED_MARLIN_MODE,
            trainable_scope=ATTENTION_BRIDGE_TRAINABLE_SCOPE,
            attention_mode="block",
            block_width_override=2,
            current_block_masking="continuous_time",
            full_block_mask_probability=1.0,
            rollout_prefix_probability=0.0,
        ),
        frigid_teacher=ConstantTeacher(),
    )
    batch = {
        "input_ids": torch.tensor([[1, 5, 6, 5, 6]]),
        "fingerprint": torch.zeros((1, 8)),
        "precursor_mass": torch.tensor([44.0]),
        "formula": ["C2H4O"],
    }

    _, metrics = module.frigid_distillation_objective(
        batch,
        generator=torch.Generator().manual_seed(5),
        collect_context_metrics=True,
    )

    assert metrics["distillation_context_tokens"].item() == 0.0
    assert metrics["distillation_context_response_ratio"].item() == 0.0
    assert metrics["distillation_student_context_logit_l1"].item() == 0.0


def test_curriculum_scopes_have_fail_closed_parameter_counts():
    stage1 = expected_distillation_trainable_parameters(
        ATTENTION_BRIDGE_TRAINABLE_SCOPE,
        num_layers=12,
    )
    stage2 = expected_distillation_trainable_parameters(
        ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE,
        num_layers=12,
    )
    all_ffn = expected_distillation_trainable_parameters(
        ATTENTION_PLUS_ALL_FFN_TRAINABLE_SCOPE,
        num_layers=12,
    )

    assert len(stage1) == 148
    assert len(stage2) == 172
    assert len(all_ffn) == 220
    assert "token_embedding.weight" not in stage1
    assert "prediction_dense.weight" not in stage2
    assert "conditioner.fingerprint.embedding.weight" not in all_ffn
    assert "layers.8.linear1.weight" in stage2
    assert "layers.7.linear1.weight" not in stage2
    assert "layers.0.linear1.weight" in all_ffn
    assert "layers.11.norm3.bias" in all_ffn


def test_all_ffn_scope_has_exact_decoder_parameter_set():
    num_layers = 12
    mass_parameters = {
        "conditioner.mass.projection.0.weight",
        "conditioner.mass.projection.0.bias",
        "conditioner.mass.projection.2.weight",
        "conditioner.mass.projection.2.bias",
    }
    attention_and_norm_parameters = {
        f"layers.{layer_index}.{suffix}"
        for layer_index in range(num_layers)
        for suffix in (
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
    }
    all_ffn_parameters = {
        f"layers.{layer_index}.{suffix}"
        for layer_index in range(num_layers)
        for suffix in (
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
            "norm3.weight",
            "norm3.bias",
        )
    }
    exact = mass_parameters | attention_and_norm_parameters | all_ffn_parameters

    expected = expected_distillation_trainable_parameters(
        ATTENTION_PLUS_ALL_FFN_TRAINABLE_SCOPE,
        num_layers=num_layers,
    )

    assert expected == exact
    assert len(expected) == 220
    assert not any(name.startswith("conditioner.fingerprint.") for name in expected)
    assert not any("embedding" in name for name in expected)
    assert not any(name.startswith("prediction_") for name in expected)
    assert not any(name.startswith("output") for name in expected)


def test_block_curriculum_configs_are_conservative_and_raw_evaluated():
    root = Path(__file__).resolve().parents[1]
    stage1 = OmegaConf.load(root / "configs/marlin_frigid_distilled_stage1.yaml")
    stage2 = OmegaConf.load(root / "configs/marlin_frigid_distilled_stage2.yaml")
    stage1b = OmegaConf.load(root / "configs/marlin_frigid_distilled_stage1b.yaml")
    stage1c = OmegaConf.load(root / "configs/marlin_frigid_distilled_stage1c.yaml")
    tiny = OmegaConf.load(
        root / "configs/marlin_frigid_distilled_tiny_overfit_c16h12o3.yaml"
    )

    assert stage1.adaptation.trainable_scope == ATTENTION_BRIDGE_TRAINABLE_SCOPE
    assert stage1.adaptation.block_width_override == 32
    assert stage1.adaptation.current_block_masking == "continuous_time"
    assert stage1.adaptation.full_block_mask_probability == pytest.approx(0.25)
    assert stage1.adaptation.rollout_prefix_probability == pytest.approx(0.5)
    assert stage1.adaptation.kl_weight == 0.5
    assert stage1.optim.learning_rate == pytest.approx(1e-5)
    assert stage1.training.ema_decay == pytest.approx(0.99)
    assert stage1.training.molecular_validation_csv.endswith(
        "/datasets/nplib1/val-metadata.csv"
    )
    assert stage1.training.molecular_validation_use_ema is False
    assert (
        stage2.adaptation.trainable_scope
        == ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE
    )
    assert stage2.adaptation.block_width_override == 8
    assert stage2.adaptation.current_block_masking == "continuous_time"
    assert stage2.adaptation.full_block_mask_probability == pytest.approx(0.25)
    assert stage2.adaptation.rollout_prefix_probability == pytest.approx(0.5)
    assert stage2.adaptation.kl_weight == 0.25
    assert stage2.optim.learning_rate == pytest.approx(5e-6)
    assert stage1b.adaptation.trainable_scope == ATTENTION_BRIDGE_TRAINABLE_SCOPE
    assert stage1b.adaptation.block_width_override == 32
    assert stage1b.adaptation.rollout_prefix_probability == pytest.approx(0.5)
    assert stage1b.optim.learning_rate == pytest.approx(1e-5)
    assert (
        stage1c.adaptation.trainable_scope
        == ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE
    )
    assert tiny.adaptation.stage.startswith("diagnostic-tiny-overfit-")
    assert tiny.data.metadata_csv.endswith(
        "/datasets/marlin-tiny-overfit-block32-v2/train-metadata.csv"
    )
    assert tiny.loader.batch_size == 4
    assert tiny.loader.num_workers == 0
    assert tiny.optim.learning_rate == pytest.approx(2e-5)
    assert tiny.trainer.max_steps == 400
    assert tiny.training.molecular_validation_samples == 4
    assert tiny.training.molecular_validation_candidates == 1
    assert tiny.training.molecular_validation_use_ema is False
    assert stage1c.optim.learning_rate == pytest.approx(5e-6)


def test_all_ffn_tiny_config_only_changes_capacity_and_run_identity():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        top4 = compose(config_name="marlin_frigid_distilled_tiny_overfit_c16h12o3")
        all_ffn = compose(
            config_name="marlin_frigid_distilled_tiny_overfit_c16h12o3_all_ffn"
        )

    top4_container = OmegaConf.to_container(top4, resolve=True)
    all_ffn_container = OmegaConf.to_container(all_ffn, resolve=True)
    assert isinstance(top4_container, dict)
    assert isinstance(all_ffn_container, dict)

    expected_differences = {
        "adaptation.stage",
        "adaptation.trainable_scope",
        "optim.learning_rate",
        "output.root",
        "output.checkpoints",
        "tracking.clearml.task_name",
        "tracking.clearml.tags",
    }

    def differing_paths(left, right, prefix=""):
        if isinstance(left, dict) and isinstance(right, dict):
            assert left.keys() == right.keys()
            return {
                path
                for key in left
                for path in differing_paths(
                    left[key],
                    right[key],
                    f"{prefix}.{key}" if prefix else key,
                )
            }
        return {prefix} if left != right else set()

    assert differing_paths(top4_container, all_ffn_container) == expected_differences
    assert all_ffn.adaptation.trainable_scope == ATTENTION_PLUS_ALL_FFN_TRAINABLE_SCOPE
    assert all_ffn.optim.learning_rate == pytest.approx(1e-5)
    assert all_ffn.data.metadata_csv == top4.data.metadata_csv
    assert all_ffn.trainer == top4.trainer
    assert "diagnostic-only" in all_ffn.tracking.clearml.tags
    assert "attention-plus-all-ffn" in all_ffn.tracking.clearml.tags


def test_action_ce_tiny_config_only_changes_objective_and_run_identity():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        top4 = compose(config_name="marlin_frigid_distilled_tiny_overfit_c16h12o3")
        action_ce = compose(
            config_name=(
                "marlin_frigid_distilled_tiny_overfit_c16h12o3_action_ce"
            )
        )

    top4_container = OmegaConf.to_container(top4, resolve=True)
    action_ce_container = OmegaConf.to_container(action_ce, resolve=True)
    assert isinstance(top4_container, dict)
    assert isinstance(action_ce_container, dict)

    expected_differences = {
        "adaptation.stage",
        "adaptation.full_block_mask_probability",
        "adaptation.rollout_prefix_probability",
        "adaptation.kl_weight",
        "output.root",
        "output.checkpoints",
        "tracking.clearml.task_name",
        "tracking.clearml.tags",
    }

    def differing_paths(left, right, prefix=""):
        if isinstance(left, dict) and isinstance(right, dict):
            assert left.keys() == right.keys()
            return {
                path
                for key in left
                for path in differing_paths(
                    left[key],
                    right[key],
                    f"{prefix}.{key}" if prefix else key,
                )
            }
        return {prefix} if left != right else set()

    assert differing_paths(top4_container, action_ce_container) == expected_differences
    assert (
        action_ce.adaptation.trainable_scope
        == ATTENTION_PLUS_TOP4_FFN_TRAINABLE_SCOPE
    )
    assert action_ce.adaptation.full_block_mask_probability == 0.0
    assert action_ce.adaptation.rollout_prefix_probability == 1.0
    assert action_ce.adaptation.kl_weight == 0.0
    assert action_ce.data == top4.data
    assert action_ce.loader == top4.loader
    assert action_ce.model == top4.model
    assert action_ce.optim == top4.optim
    assert action_ce.training == top4.training
    assert action_ce.trainer == top4.trainer
    assert "production-action-ce" in action_ce.tracking.clearml.tags
    assert "rollout-prefix-only" in action_ce.tracking.clearml.tags
    assert "kl-zero" in action_ce.tracking.clearml.tags


def test_frigid_teacher_rejects_tokenizer_mismatch():
    class Tokenizer:
        unk_token_id = 0
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 3
        mask_token_id = 4
        vocab_size = 5

        @staticmethod
        def model_vocab():
            return {
                "[UNK]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "[PAD]": 3,
                "[MASK]": 4,
            }

        @classmethod
        def get_vocab(cls):
            return {**cls.model_vocab(), "<": 5, ">": 6}

    class Model:
        tokenizer = Tokenizer()

        def requires_grad_(self, _requires_grad):
            return self

        def eval(self):
            return self

    teacher = FrozenFrigidTeacher(Model())
    teacher.validate_student_tokenizer(
        Tokenizer.model_vocab(),
        {"unk": 0, "bos": 1, "eos": 2, "pad": 3, "mask": 4},
    )

    with pytest.raises(ValueError, match="vocabularies differ"):
        teacher.validate_student_tokenizer(
            {**Tokenizer.model_vocab(), "[EOS]": 3, "[PAD]": 2},
            None,
        )
    with pytest.raises(ValueError, match="mask token IDs differ"):
        teacher.validate_student_tokenizer(
            Tokenizer.model_vocab(),
            {"mask": 9},
        )


def test_frigid_distilled_config_is_separate_and_explicit():
    root = Path(__file__).resolve().parents[1]
    strict = OmegaConf.load(root / "configs/marlin_nplib1.yaml")
    distilled = OmegaConf.load(root / "configs/marlin_frigid_distilled_stage0.yaml")

    assert strict.adaptation.mode == "strict_marlin"
    assert distilled.adaptation.mode == FRIGID_DISTILLED_MARLIN_MODE
    assert distilled.tracking.clearml.task_name.startswith("frigid-distilled-marlin")
    assert "not-strict-reproduction" in distilled.tracking.clearml.tags


def test_strict_lane_allows_hashed_warmstart_but_rejects_teacher_fields():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "train_marlin_adaptation_test",
        root / "scripts/train_marlin.py",
    )
    assert spec is not None and spec.loader is not None
    train_marlin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_marlin)

    strict = OmegaConf.load(root / "configs/marlin_nplib1.yaml")
    strict.frigid_warm_start_checkpoint = "/checkpoints/DLM.ckpt"
    strict.frigid_warm_start_sha256 = "b6177c2d"

    assert train_marlin.adaptation_mode(strict) == "strict_marlin"

    strict.adaptation.teacher_checkpoint = "/checkpoints/DLM.ckpt"
    strict.adaptation.teacher_sha256 = "b6177c2d"
    with pytest.raises(
        ValueError,
        match="strict_marlin forbids FRIGID teacher fields",
    ):
        train_marlin.adaptation_mode(strict)


def test_distilled_resume_supersedes_provenance_warmstart():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "train_marlin_resume_test",
        root / "scripts/train_marlin.py",
    )
    assert spec is not None and spec.loader is not None
    train_marlin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_marlin)

    distilled = OmegaConf.load(root / "configs/marlin_frigid_distilled_stage0.yaml")
    distilled.resume_checkpoint = "/runs/stage0/checkpoints/step=100.ckpt"

    assert train_marlin.initialization_source(distilled) == "resume_checkpoint"

    distilled.resume_checkpoint = None
    distilled.resume_weights_only_checkpoint = "/runs/other.ckpt"

    assert (
        train_marlin.initialization_source(distilled)
        == "resume_weights_only_checkpoint"
    )

    distilled.resume_checkpoint = "/runs/stage0/checkpoints/step=100.ckpt"
    with pytest.raises(ValueError, match="choose exactly one initialization source"):
        train_marlin.initialization_source(distilled)


def test_clearml_scalar_callback_reports_first_step_and_interval():
    class Logger:
        def __init__(self):
            self.calls = []

        def report_scalar(self, **kwargs):
            self.calls.append(kwargs)

    class Task:
        def __init__(self):
            self.logger = Logger()

        def get_logger(self):
            return self.logger

    class Trainer:
        is_global_zero = True
        global_step = 1
        logged_metrics = {
            "train_loss": torch.tensor(14.5),
            "train_distillation_kl": torch.tensor(2.25),
            "ignored": torch.tensor(99.0),
        }

    task = Task()
    trainer = Trainer()
    callback = ClearMLScalarCallback(task, every_n_steps=10)

    callback.on_train_batch_end(trainer, None, None, None, 0)
    callback.on_train_batch_end(trainer, None, None, None, 0)
    trainer.global_step = 2
    callback.on_train_batch_end(trainer, None, None, None, 1)
    trainer.global_step = 10
    callback.on_train_batch_end(trainer, None, None, None, 9)

    assert [(call["series"], call["iteration"]) for call in task.logger.calls] == [
        ("train_loss", 1),
        ("train_distillation_kl", 1),
        ("train_loss", 10),
        ("train_distillation_kl", 10),
    ]
