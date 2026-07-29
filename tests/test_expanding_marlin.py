from pathlib import Path

import torch
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import Descriptors

from marlin.expanding import (
    CosineInsertionSchedule,
    ExpandingFlowConfig,
    ExpandingMarlinModel,
    ExpandingMarlinSampler,
    ExpandingModelOutput,
    VocabularyTimeWarp,
    eflow_objective,
    efm_objective,
    sample_eflow_batch,
)
from marlin.expanding_checkpoint import expanding_model_from_checkpoint
from marlin.expanding_training import ExpandingMarlinLightningModule
from marlin.mass_shell import MassShellConstraint
from marlin.model import MarlinDecoderConfig


def tiny_configs(**flow_overrides):
    decoder = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=8,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        eos_token_id=2,
        mask_token_id=3,
        pad_token_id=0,
    )
    defaults = {
        "prior_scale": 1.0,
        "insertion_cutoff": 0.5,
        "time_embedding_size": 8,
        "time_fourier_dim": 8,
        "time_warp_points": 64,
        "time_warp_quadrature": 12,
        "insertion_detach_steps": 0,
    }
    defaults.update(flow_overrides)
    return decoder, ExpandingFlowConfig(**defaults)


def test_cosine_insertion_schedule_is_invertible_and_bounded():
    schedule = CosineInsertionSchedule(0.5)
    probability = torch.linspace(0, 1, 11)
    time = schedule.inverse(probability)

    assert torch.allclose(schedule.alpha(time), probability, atol=1e-6)
    assert time.min() == 0
    assert time.max() <= 0.5
    assert schedule.alpha(torch.tensor([0.75])).item() == 1.0
    assert schedule.derivative(torch.tensor([0.75])).item() == 0.0


def test_vocabulary_time_warp_is_monotone_and_round_trips():
    warp = VocabularyTimeWarp(32, points=128, quadrature=16)
    tau = torch.linspace(0, 1, 21)
    time = warp.inverse(tau)

    assert (time[1:] >= time[:-1]).all()
    assert torch.allclose(warp(time), tau, atol=0.03)
    assert time[0] == 0
    assert time[-1] == 1


def test_eflow_batch_compacts_active_tokens_and_preserves_anchors():
    decoder, flow = tiny_configs()
    model = ExpandingMarlinModel(decoder, flow)
    clean = torch.tensor(
        [
            [1, 4, 5, 6, 2, 0],
            [1, 7, 2, 0, 0, 0],
        ]
    )
    sampled = sample_eflow_batch(
        clean,
        decoder,
        flow,
        CosineInsertionSchedule(flow.insertion_cutoff),
        model.time_warp,
        generator=torch.Generator().manual_seed(2),
    )

    assert sampled.latent_tokens.shape[:2] == sampled.target_ids.shape
    assert sampled.latent_tokens.shape[-1] == decoder.vocab_size
    assert (sampled.target_ids[:, 0] == 1).all()
    lengths = (~sampled.padding_mask).sum(dim=1)
    assert (sampled.target_ids[torch.arange(2), lengths - 1] == 2).all()
    assert not sampled.token_loss_mask[:, 0].any()
    assert not sampled.token_loss_mask[
        torch.arange(2), lengths - 1
    ].any()
    assert sampled.gap_targets.sum() >= 0


def test_expanding_model_outputs_token_and_gap_predictions():
    decoder, flow = tiny_configs()
    model = ExpandingMarlinModel(decoder, flow)
    latent = torch.randn((2, 4, decoder.vocab_size))
    local = torch.rand((2, 4))
    padding = torch.tensor(
        [[False, False, False, False], [False, False, True, True]]
    )
    output = model(
        latent,
        local,
        padding,
        torch.tensor([100.0, 120.0]),
        torch.zeros((2, decoder.fingerprint_bits)),
        source_time=torch.tensor([0.2, 0.4]),
        target_time=torch.tensor([0.5, 0.8]),
    )

    assert output.logits.shape == (2, 4, decoder.vocab_size)
    assert output.insertion_means.shape == (2, 5)
    assert output.insertion_mask[0].sum() == 5
    assert output.insertion_mask[1].sum() == 3
    assert torch.isfinite(output.logits).all()
    assert (output.insertion_means[output.insertion_mask] > 0).all()


def test_eflow_objective_is_finite_and_updates_new_heads():
    decoder, flow = tiny_configs()
    model = ExpandingMarlinModel(decoder, flow)
    clean = torch.tensor([[1, 4, 5, 6, 2, 0]])
    loss, metrics = eflow_objective(
        model,
        clean,
        torch.tensor([100.0]),
        torch.zeros((1, decoder.fingerprint_bits)),
        generator=torch.Generator().manual_seed(3),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(metrics) >= {
        "token_loss",
        "insertion_loss",
        "token_accuracy",
        "remaining_length_mae",
    }
    assert model.insertion_output[-1].weight.grad is not None
    assert model.source_time.projection[0].weight.grad is not None


def test_eflow_staged_adaptation_preserves_ema_parameter_order():
    decoder, flow = tiny_configs()
    module = ExpandingMarlinLightningModule(
        decoder,
        flow,
        flow_modules_only_steps=2,
    )

    assert module.apply_adaptation_stage(0) == "flow_modules"
    assert not module.model.backbone.token_embedding.weight.requires_grad
    assert module.model.source_time.projection[0].weight.requires_grad
    assert len(module.ema.shadow_params) == len(list(module.model.parameters()))
    module.ema.update(module.model.parameters())

    assert module.apply_adaptation_stage(2) == "full"
    assert all(parameter.requires_grad for parameter in module.model.parameters())
    module.ema.update(module.model.parameters())


def test_efm_diagonal_and_semigroup_objectives_are_finite():
    decoder, diagonal_flow = tiny_configs(diagonal_probability=1.0)
    teacher = ExpandingMarlinModel(decoder, diagonal_flow).eval()
    student = ExpandingMarlinModel(decoder, diagonal_flow)
    clean = torch.tensor([[1, 4, 5, 6, 2, 0]])
    diagonal_loss, diagonal_metrics = efm_objective(
        student,
        teacher,
        clean,
        torch.tensor([100.0]),
        torch.zeros((1, decoder.fingerprint_bits)),
        generator=torch.Generator().manual_seed(4),
    )

    _, off_diagonal_flow = tiny_configs(
        diagonal_probability=0.0,
        boundary_probability=0.0,
    )
    teacher_off = ExpandingMarlinModel(decoder, off_diagonal_flow).eval()
    student_off = ExpandingMarlinModel(decoder, off_diagonal_flow)
    consistency_loss, consistency_metrics = efm_objective(
        student_off,
        teacher_off,
        clean,
        torch.tensor([100.0]),
        torch.zeros((1, decoder.fingerprint_bits)),
        generator=torch.Generator().manual_seed(5),
    )

    assert torch.isfinite(diagonal_loss)
    assert diagonal_metrics["diagonal_fraction"] == 1
    assert torch.isfinite(consistency_loss)
    assert consistency_metrics["diagonal_fraction"] == 0
    assert torch.isfinite(consistency_metrics["consistency_mismatch"])


def test_expanding_sampler_grows_and_returns_a_mass_valid_molecule():
    decoder = MarlinDecoderConfig(
        vocab_size=6,
        hidden_size=4,
        num_layers=1,
        num_heads=1,
        intermediate_size=8,
        max_length=3,
        block_width=1,
        fingerprint_bits=8,
        dropout=0.0,
        eos_token_id=2,
        mask_token_id=3,
        pad_token_id=0,
    )
    flow = ExpandingFlowConfig(
        prior_scale=1.0,
        insertion_cutoff=0.5,
        time_embedding_size=4,
        time_fourier_dim=4,
        time_warp_points=32,
        time_warp_quadrature=8,
    )

    class IdentityWarp:
        def inverse(self, value):
            return value

    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.decoder_config = decoder
            self.flow_config = flow
            self.time_warp = IdentityWarp()

        def forward(
            self,
            latent_tokens,
            local_times,
            padding_mask,
            precursor_mass,
            fingerprint,
            *,
            source_time,
            target_time,
            **kwargs,
        ):
            del (
                local_times,
                precursor_mass,
                fingerprint,
                source_time,
                target_time,
                kwargs,
            )
            batch, length, vocabulary = latent_tokens.shape
            logits = torch.full(
                (batch, length, vocabulary),
                -10.0,
                device=latent_tokens.device,
            )
            logits[..., 4] = 10.0
            means = torch.zeros(
                (batch, length + 1), device=latent_tokens.device
            )
            means[:, 1] = padding_mask.logical_not().sum(dim=1).eq(2).float()
            mask = (
                torch.arange(length + 1, device=latent_tokens.device)
                .unsqueeze(0)
                .le(padding_mask.logical_not().sum(dim=1).unsqueeze(1))
            )
            return ExpandingModelOutput(logits, means, mask, logits[..., :4])

    molecule = Chem.MolFromSmiles("C")
    target_mass = Descriptors.ExactMolWt(molecule)
    sampler = ExpandingMarlinSampler(
        FixedModel(),
        MassShellConstraint(
            [0.0] * decoder.vocab_size,
            eos_token_id=decoder.eos_token_id,
            ppm_tolerance=10,
        ),
        stage="efm",
        bos_token_id=1,
        eos_token_id=2,
        decode_tokens=lambda ids: "C" if ids == [4] else "",
        safe_to_smiles=lambda safe: safe or None,
        forbidden_token_ids=(0, 1, 2, 3),
        steps=2,
    )

    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(decoder.fingerprint_bits),
        target_mass,
        candidates=2,
        generator=torch.Generator().manual_seed(7),
    )

    assert stats.valid == 2
    assert stats.mass_valid == 2
    assert [candidate.smiles for candidate in ranked] == ["C"]


def test_expanding_checkpoint_round_trip_with_ema(tmp_path):
    decoder, flow = tiny_configs()
    module = ExpandingMarlinLightningModule(
        decoder,
        flow,
        stage="eflow",
        warmup_steps=0,
    )
    checkpoint = {
        "hyper_parameters": dict(module.hparams),
        "state_dict": module.state_dict(),
        "ema": module.ema.state_dict(),
    }
    path = tmp_path / "eflow.ckpt"
    torch.save(checkpoint, path)

    restored, stage = expanding_model_from_checkpoint(path, use_ema=True)

    assert stage == "eflow"
    assert restored.decoder_config == decoder
    assert restored.flow_config == flow
    assert all(
        torch.equal(left, right)
        for left, right in zip(module.model.parameters(), restored.parameters())
    )


def test_expanding_config_is_separate_from_strict_marlin_recipe():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs/expanding_marlin_nplib1.yaml")

    assert config.architecture == "expanding"
    assert config.training.flow_modules_only_steps == 500
    assert config.stage == "eflow"
    assert config.flow.prior_scale == 1.25
    assert config.flow.insertion_cutoff == 0.5
    assert config.flow.diagonal_probability == 0.75
    assert config.optim.learning_rate == 3e-4
    assert config.optim.warmup_steps == 2500
    assert config.trainer.max_steps == 200000
    assert "non-paper-architecture" in config.tracking.clearml.tags
