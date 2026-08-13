from pathlib import Path

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from marlin.conditioning import MarlinConditioner
from marlin.mass_shell import MassShellConstraint, MassShellState
from marlin.model import (
    MarlinDecoder,
    MarlinDecoderConfig,
    block_causal_attention_mask,
    two_stream_attention_mask,
)
from marlin.noise import symmetric_fingerprint_noise
from marlin.isotopes import theoretical_isotope_ratios
from marlin.sampler import MarlinSampler
from marlin.token_properties import build_token_property_table, token_properties
from marlin.tokenizer import load_safe_tokenizer
from marlin.training import (
    MarlinCollator,
    MarlinLightningModule,
    MarlinTrainingFilter,
)
from marlin.warm_start import _copy_attention, sha256_file


def test_block_causal_mask_allows_current_and_previous_blocks():
    mask = block_causal_attention_mask(6, 2)
    assert mask[0, 1]
    assert mask[0, 2]
    assert not mask[2, 0]
    assert mask[2, 3]
    assert mask[2, 4]


def test_two_stream_mask_exposes_only_clean_prefix_and_noisy_current_block():
    mask = two_stream_attention_mask(5, 2)
    noisy_query = 5 + 3
    assert not mask[noisy_query, 0]
    assert mask[noisy_query, 3]
    assert not mask[noisy_query, 5 + 4]
    assert mask[noisy_query, 5 + 1]


def test_symmetric_noise_preserves_number_of_on_bits():
    fingerprint = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    generator = torch.Generator().manual_seed(7)
    noisy = symmetric_fingerprint_noise(
        fingerprint,
        corruption_probability=1.0,
        min_fraction=0.5,
        max_fraction=0.5,
        generator=generator,
    )
    assert noisy.sum() == fingerprint.sum()
    assert not torch.equal(noisy, fingerprint)


def test_conditioner_emits_mass_isotope_and_active_bit_tokens():
    conditioner = MarlinConditioner(
        hidden_size=28, fingerprint_bits=8, num_mass_frequencies=4
    )
    fingerprint = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    tokens, mask = conditioner(
        torch.tensor([250.0]), fingerprint, torch.tensor([[0.1, 0.02]])
    )
    assert tokens.shape == (1, 4, 28)
    assert mask.tolist() == [[True, True, True, True]]


def test_conditioner_preserves_soft_active_bit_confidence():
    conditioner = MarlinConditioner(
        hidden_size=8, fingerprint_bits=4, num_mass_frequencies=2
    )
    with torch.no_grad():
        conditioner.fingerprint.embedding.weight.zero_()
        conditioner.fingerprint.embedding.weight[0].copy_(
            torch.arange(8, dtype=torch.float32)
        )
    binary_tokens, _ = conditioner(
        torch.tensor([250.0]), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    soft_tokens, _ = conditioner(
        torch.tensor([250.0]), torch.tensor([[0.75, 0.0, 0.0, 0.0]])
    )
    assert torch.allclose(soft_tokens[:, 1], binary_tokens[:, 1] * 0.75)


def test_conditioner_omits_disabled_isotope_token():
    conditioner = MarlinConditioner(
        hidden_size=28, fingerprint_bits=8, num_mass_frequencies=4
    )
    fingerprint = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)

    tokens, mask = conditioner(torch.tensor([250.0]), fingerprint)

    assert tokens.shape == (1, 3, 28)
    assert mask.tolist() == [[True, True, True]]


def test_marlin_uses_fixed_decay_ema():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    module = MarlinLightningModule(config, ema_decay=0.9999)

    assert module.ema.decay == 0.9999
    assert module.ema.num_updates is None


def test_staged_adaptation_preserves_ema_parameter_order():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    module = MarlinLightningModule(
        config,
        conditioning_only_steps=2,
        cross_attention_only_steps=3,
    )
    parameter_count = len(list(module.decoder.parameters()))

    assert module.apply_adaptation_stage(0) == "conditioning"
    assert len(module.ema.shadow_params) == parameter_count
    assert {
        name
        for name, parameter in module.decoder.named_parameters()
        if parameter.requires_grad
    } == {
        name
        for name, _ in module.decoder.named_parameters()
        if name.startswith(("conditioner.mass.", "conditioner.isotope."))
    }

    assert module.apply_adaptation_stage(2) == "cross_attention"
    assert any(
        parameter.requires_grad
        for name, parameter in module.decoder.named_parameters()
        if ".cross_attention." in name
    )
    assert not module.decoder.layers[0].self_attention.in_proj_weight.requires_grad

    assert module.apply_adaptation_stage(5) == "full"
    assert all(parameter.requires_grad for parameter in module.decoder.parameters())
    module.ema.update(module.decoder.parameters())
    assert len(module.ema.shadow_params) == parameter_count


def test_staged_adaptation_can_train_fingerprint_conditioner():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    module = MarlinLightningModule(
        config,
        conditioning_only_steps=2,
        adapt_fingerprint=True,
    )

    assert module.apply_adaptation_stage(0) == "conditioning"
    assert all(
        parameter.requires_grad
        for name, parameter in module.decoder.named_parameters()
        if name.startswith("conditioner.fingerprint.")
    )
    assert not module.decoder.layers[0].self_attention.in_proj_weight.requires_grad
    assert {
        name
        for name, parameter in module.decoder.named_parameters()
        if parameter.requires_grad
    } == {
        name
        for name, _ in module.decoder.named_parameters()
        if name.startswith(
            (
                "conditioner.mass.",
                "conditioner.isotope.",
                "conditioner.fingerprint.",
            )
        )
    }


@pytest.mark.parametrize(
    ("conditioning_steps", "cross_attention_steps"),
    ((-1, 0), (0, -1)),
)
def test_staged_adaptation_rejects_negative_durations(
    conditioning_steps,
    cross_attention_steps,
):
    with pytest.raises(ValueError, match="stage durations"):
        MarlinLightningModule(
            MarlinDecoderConfig(
                vocab_size=8,
                hidden_size=8,
                num_layers=1,
                num_heads=1,
                intermediate_size=16,
                max_length=5,
                block_width=2,
                fingerprint_bits=4,
            ),
            conditioning_only_steps=conditioning_steps,
            cross_attention_only_steps=cross_attention_steps,
        )


def test_layer0_residual_defaults_off_and_rejects_invalid_scales():
    assert MarlinDecoderConfig().layer0_long_residual_scale == 0.0
    for scale in (-1.0, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(
            ValueError,
            match="layer0_long_residual_scale must be finite and non-negative",
        ):
            MarlinDecoderConfig(layer0_long_residual_scale=scale)


def test_layer0_residual_preserves_checkpoint_compatibility():
    common = dict(
        vocab_size=9,
        hidden_size=8,
        num_layers=2,
        num_heads=1,
        intermediate_size=16,
        max_length=6,
        block_width=2,
        fingerprint_bits=8,
        dropout=0.0,
        mask_token_id=4,
        pad_token_id=0,
    )
    baseline = MarlinDecoder(
        MarlinDecoderConfig(**common, layer0_long_residual_scale=0.0)
    ).eval()
    residual = MarlinDecoder(
        MarlinDecoderConfig(**common, layer0_long_residual_scale=1.0)
    ).eval()
    residual.load_state_dict(baseline.state_dict(), strict=True)

    assert residual.state_dict().keys() == baseline.state_dict().keys()
    input_ids = torch.tensor([[1, 5, 4, 4, 4, 4]])
    mass = torch.tensor([100.0])
    fingerprint = torch.zeros((1, 8))
    with torch.inference_mode():
        baseline_logits = baseline(input_ids, mass, fingerprint)
        residual_logits = residual(input_ids, mass, fingerprint)
    assert not torch.equal(residual_logits, baseline_logits)


def test_training_accepts_optional_eos_recovery_controls():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    module = MarlinLightningModule(
        config,
        eos_loss_weight=5.0,
        eos_mask_probability=1.0,
        balanced_token_loss_alpha=0.5,
        token_loss_weight_max=10.0,
        full_sequence_mask_probability=0.25,
    )

    assert module.eos_loss_weight == 5.0
    assert module.eos_mask_probability == 1.0
    assert module.hparams["eos_loss_weight"] == 5.0
    assert module.hparams["eos_mask_probability"] == 1.0
    assert module.balanced_token_loss_alpha == 0.5
    assert module.token_loss_weight_max == 10.0
    assert module.full_sequence_mask_probability == 0.25
    assert module.hparams["balanced_token_loss_alpha"] == 0.5
    assert module.hparams["token_loss_weight_max"] == 10.0
    assert module.hparams["full_sequence_mask_probability"] == 0.25


def test_fingerprint_layer_norm_can_be_disabled_for_legacy_checkpoints():
    config = MarlinDecoderConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        intermediate_size=32,
        max_length=16,
        block_width=4,
        fingerprint_bits=32,
        fingerprint_layer_norm=False,
    )

    model = MarlinDecoder(config)

    assert isinstance(model.conditioner.fingerprint.layer_norm, torch.nn.Identity)


def test_theoretical_isotope_ratios_include_m_plus_one_and_two():
    ratios = theoretical_isotope_ratios(Chem.MolFromSmiles("CCl"))
    assert ratios.shape == (2,)
    assert ratios[0] > 0
    assert ratios[1] > 0.3


def test_mass_shell_prunes_overshoot_and_forbids_early_eos():
    constraint = MassShellConstraint(
        [0.0, 12.0, 16.0],
        [0, 1, 1],
        [0.0, 4.0, 2.0],
        eos_token_id=0,
        ppm_tolerance=10,
    )
    logits = constraint.apply(torch.zeros(3), MassShellState(heavy_mass=90.0), 100.0)
    assert torch.isneginf(logits[0])
    assert torch.isneginf(logits[1])
    assert torch.isneginf(logits[2])


def test_mass_shell_boosts_eos_when_no_nonzero_token_fits():
    constraint = MassShellConstraint(
        [0.0, 12.0, 16.0],
        [0, 1, 1],
        [0.0, 4.0, 2.0],
        eos_token_id=0,
        ppm_tolerance=10,
        eos_boost=1.5,
    )
    logits = constraint.apply(
        torch.zeros(3),
        MassShellState(heavy_mass=95.0, heavy_atoms=1, valence_sum=4.0),
        100.0,
    )
    assert logits[0] == 1.5
    assert torch.isneginf(logits[1:]).all()


def test_token_properties_ignore_safe_grammar_characters():
    properties = token_properties("C1=CC(Cl)=CC=C1")
    assert properties.heavy_atoms == 7
    assert properties.heavy_mass > 100


def test_token_properties_cover_rare_official_safe_elements():
    properties = token_properties("[208Tl+]")
    assert properties.heavy_atoms == 1
    assert properties.heavy_mass > 200


def test_token_properties_use_explicit_isotope_mass():
    boron_10 = token_properties("[10B]")
    boron_default = token_properties("[B]")
    carbon_13 = token_properties("[13C]")
    carbon_default = token_properties("[C]")

    assert boron_10.heavy_atoms == 1
    assert boron_10.heavy_mass < boron_default.heavy_mass
    assert carbon_13.heavy_atoms == 1
    assert carbon_13.heavy_mass > carbon_default.heavy_mass


def test_token_properties_ignore_unknown_safe_bracket_tokens():
    properties = token_properties("[Z]")
    assert properties.heavy_atoms == 0
    assert properties.heavy_mass == 0.0
    assert properties.valence_sum == 0.0


def test_official_safe_vocabulary_builds_token_property_table():
    tokenizer = load_safe_tokenizer(
        "/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/data/safe-gpt/tokenizer.json"
    )
    special_ids = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
        tokenizer.pad_token_id,
    }
    masses, atom_counts, valences = build_token_property_table(
        len(tokenizer), tokenizer.convert_ids_to_tokens, special_ids
    )

    assert len(masses) == len(tokenizer)
    assert len(atom_counts) == len(tokenizer)
    assert len(valences) == len(tokenizer)


def test_training_collator_refuses_to_truncate_safe(monkeypatch):
    class OverlengthTokenizer:
        def __call__(self, values, **kwargs):
            assert values == ["C"]
            assert kwargs["truncation"] is False
            return {"input_ids": torch.tensor([[1, 4, 5, 2]])}

    monkeypatch.setattr("marlin.training.safe_to_smiles", lambda safe, fix: safe)
    collator = MarlinCollator(
        OverlengthTokenizer(), max_length=3, fingerprint_bits=16
    )

    with pytest.raises(ValueError, match="refusing silent truncation"):
        collator([{"safe": "C"}])


def test_stream_filter_excludes_invalid_overlength_and_test_safe(
    monkeypatch, tmp_path
):
    class LengthTokenizer:
        def encode(self, safe, add_special_tokens):
            assert add_special_tokens is True
            return list(range(len(safe) + 2))

    exclusions = tmp_path / "exclude.csv"
    exclusions.write_text("inchikey\nOKKJLVBELUTLKV-UHFFFAOYSA-N\n")
    monkeypatch.setattr(
        "marlin.training.safe_to_smiles",
        lambda safe, fix: {"CC": "CC", "CO": "CO"}.get(safe),
    )
    stream_filter = MarlinTrainingFilter(LengthTokenizer(), 4, exclusions)

    assert stream_filter({"safe": "CC"})
    assert not stream_filter({"safe": "CO"})
    assert not stream_filter({"safe": "CCC"})
    assert not stream_filter({"safe": "invalid"})
    assert not stream_filter({})


def test_small_decoder_forward_and_loss():
    config = MarlinDecoderConfig(
        vocab_size=32,
        hidden_size=28,
        num_layers=2,
        num_heads=4,
        intermediate_size=56,
        max_length=8,
        block_width=2,
        fingerprint_bits=16,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    model = MarlinDecoder(config)
    tokens = torch.tensor([[1, 5, 6, 2, 0, 0]])
    mass = torch.tensor([100.0])
    fingerprint = torch.zeros((1, 16))
    fingerprint[0, [2, 7]] = 1
    logits = model(tokens, mass, fingerprint)
    sampling_logits = model.sampling_logits(tokens, mass, fingerprint)
    expected_sampling_logits = model.two_stream_logits(
        tokens, tokens, mass, fingerprint
    )
    loss = model.diffusion_loss(
        tokens, mass, fingerprint, generator=torch.Generator().manual_seed(3)
    )
    assert logits.shape == (1, 6, 32)
    assert torch.equal(sampling_logits, expected_sampling_logits)
    assert torch.isfinite(loss)


def test_diffusion_objective_reports_reconstruction_metrics():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    model = MarlinDecoder(config)
    loss, metrics = model.diffusion_objective(
        torch.tensor([[1, 4, 5, 6, 2]]),
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        generator=torch.Generator().manual_seed(7),
        collect_metrics=True,
    )

    assert torch.isfinite(loss)
    assert set(metrics) == {
        "masked_token_accuracy_top1",
        "masked_token_accuracy_top10",
        "masked_target_probability",
        "masked_token_nll",
        "masked_token_perplexity",
        "masked_prediction_entropy",
        "masked_top1_confidence",
        "masked_argmax_eos_fraction",
        "masked_eos_target_count",
        "masked_eos_target_probability",
        "masked_eos_target_rank",
        "mask_fraction",
        # Reported at every step so a corrupted-context run is legible; all
        # zero, and the loss unchanged, while the corruption probability is 0.
        "context_corruption_fraction",
        "restoration_loss",
        "restoration_token_count",
        "restoration_token_accuracy_top1",
        "restoration_copy_rate",
        "masked_token_accuracy_top1_uncorrupted_rows",
        "full_sequence_mask_fraction",
        "masked_sequence_accuracy",
        "first_block_masked_count",
        "first_block_masked_token_accuracy_top1",
        "first_block_masked_token_accuracy_top10",
        "first_block_masked_target_probability",
        "first_block_masked_token_nll",
        "first_block_conditioning_nll_gain",
        "first_block_conditioning_target_probability_gain",
        "first_block_conditioning_token_accuracy_top1_gain",
        "first_block_conditioning_token_accuracy_top10_gain",
    }
    assert all(torch.isfinite(value) for value in metrics.values())
    assert 0 <= metrics["masked_token_accuracy_top1"] <= 1
    assert 0 <= metrics["masked_token_accuracy_top10"] <= 1
    assert metrics["masked_token_accuracy_top10"] >= metrics["masked_token_accuracy_top1"]
    assert 0 <= metrics["masked_top1_confidence"] <= 1
    assert 0 <= metrics["masked_argmax_eos_fraction"] <= 1
    assert metrics["masked_eos_target_count"] >= 0
    assert metrics["first_block_masked_count"] >= 0
    assert (
        metrics["first_block_masked_token_accuracy_top10"]
        >= metrics["first_block_masked_token_accuracy_top1"]
    )


def test_diffusion_keeps_bos_clean():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    model = MarlinDecoder(config)
    tokens = torch.tensor([[1, 4, 5, 6, 2]])
    captured = {}
    original = model.two_stream_logits

    def capture(
        clean_ids,
        noised_ids,
        precursor_mass,
        fingerprint,
        isotope_ratios=None,
        **kwargs,
    ):
        captured["noised_ids"] = noised_ids.clone()
        captured["isotope_ratios"] = isotope_ratios
        return original(
            clean_ids,
            noised_ids,
            precursor_mass,
            fingerprint,
            isotope_ratios,
            **kwargs,
        )

    model.two_stream_logits = capture
    model.diffusion_loss(
        tokens,
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        generator=torch.Generator().manual_seed(1),
    )

    assert captured["noised_ids"][0, 0] == tokens[0, 0]


def test_diffusion_passes_isotope_ratios_to_conditioner():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    model = MarlinDecoder(config)
    captured = {}
    original = model.conditioner.forward

    def capture(precursor_mass, fingerprint, isotope_ratios=None, **kwargs):
        captured["isotope_ratios"] = isotope_ratios
        return original(precursor_mass, fingerprint, isotope_ratios, **kwargs)

    model.conditioner.forward = capture
    isotope_ratios = torch.tensor([[0.12, 0.03]])
    model.diffusion_loss(
        torch.tensor([[1, 4, 5, 6, 2]]),
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        isotope_ratios=isotope_ratios,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.equal(captured["isotope_ratios"], isotope_ratios)


def test_diffusion_loss_averages_the_weighted_token_sum_over_blocks():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    model = MarlinDecoder(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    tokens = torch.tensor([[1, 4, 5, 6, 2]])
    seed = 5
    expected_generator = torch.Generator().manual_seed(seed)
    times = torch.rand((1, 2), generator=expected_generator).clamp_min(1e-4)
    probabilities = times[:, torch.tensor([0, 0, 0, 1, 1])]
    valid = torch.tensor([[False, True, True, True, True]])
    masked = (
        torch.rand(tokens.shape, generator=expected_generator) < probabilities
    ) & valid
    expected = (
        masked.float().mul(probabilities.reciprocal()).sum()
        * torch.log(torch.tensor(float(config.vocab_size)))
        / 2
    )

    loss = model.diffusion_loss(
        tokens,
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        generator=torch.Generator().manual_seed(seed),
    )

    assert torch.allclose(loss, expected)


def test_attention_warm_start_concatenates_qkv():
    attention = torch.nn.MultiheadAttention(4, 1, batch_first=True)
    state = {}
    for offset, part in enumerate(("query", "key", "value")):
        state[f"x.{part}.weight"] = torch.full((4, 4), float(offset + 1))
        state[f"x.{part}.bias"] = torch.full((4,), float(offset + 1))
    _copy_attention(attention, state, "x", "test")
    assert torch.equal(attention.in_proj_weight[:4], state["x.query.weight"])
    assert torch.equal(attention.in_proj_weight[-4:], state["x.value.weight"])


def test_warm_start_sha256_file(tmp_path):
    checkpoint = tmp_path / "DLM.ckpt"
    checkpoint.write_bytes(b"official checkpoint bytes")

    assert (
        sha256_file(checkpoint)
        == "39508f80120dcafa9564e61482326a83ebe2b06d09d2bee7b3ee0e1fbc932d97"
    )


def test_batched_sampler_reports_attempt_validity_and_uniqueness():
    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
            logits[..., 1] = 10.0
            return logits

    molecule = Chem.MolFromSmiles("C")
    target_mass = Descriptors.ExactMolWt(molecule)
    constraint = MassShellConstraint(
        [0.0] * 4,
        eos_token_id=2,
        ppm_tolerance=10,
    )
    sampler = MarlinSampler(
        FixedModel(),
        constraint,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda _: "C",
        safe_to_smiles=lambda _: "C",
        forbidden_token_ids=(0, 3),
    )
    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=3
    )
    assert stats.attempts == 3
    assert stats.valid == 3
    assert stats.mass_valid == 3
    assert stats.unique_mass_valid == 1
    assert len(ranked) == 1

    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8),
        target_mass,
        candidates=3,
        candidate_batch_size=2,
    )
    assert stats.attempts == 3
    assert stats.valid == 3
    assert stats.mass_valid == 3
    assert stats.unique_mass_valid == 1
    assert len(ranked) == 1


def test_constrained_sampler_uses_confidence_order_within_block():
    class PositionConfidenceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.seen.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 5), -torch.inf)
            logits[:, 1, 1] = 2.0
            logits[:, 1, 4] = 1.0
            logits[:, 2, 1] = 5.0
            logits[:, 2, 4] = 0.0
            return logits

    model = PositionConfidenceModel()
    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("CC"))
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 16.0],
            [0, 1, 0, 0, 1],
            [0.0, 4.0, 0.0, 0.0, 2.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "".join({1: "C", 4: "O"}.get(i, "") for i in ids),
        safe_to_smiles=lambda safe: safe,
        grammar_mask=lambda _prefix, logits, _mass: logits,
        forbidden_token_ids=(0, 3),
    )

    result = sampler.generate_one(torch.zeros(8), target_mass)

    assert result == ("CC", "CC")
    assert model.seen[1].tolist() == [[0, 3, 1]]


def test_constrained_sampler_rejects_confident_suffix_that_breaks_mass_shell():
    class SuffixBiasedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.seen.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 5), -torch.inf, device=input_ids.device)
            logits[:, 1, 1] = 1.0
            logits[:, 1, 4] = 0.0
            if input_ids[0, 1].item() == 1:
                logits[:, 2, 2] = 10.0
            else:
                logits[:, 2, 4] = 10.0
            return logits

    model = SuffixBiasedModel()
    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 16.0],
            [0, 1, 0, 0, 1],
            [0.0, 4.0, 0.0, 0.0, 2.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" if 1 in ids else "",
        safe_to_smiles=lambda safe: safe or None,
        grammar_mask=lambda _prefix, logits, _mass: logits,
        forbidden_token_ids=(0, 3),
    )

    assert sampler.generate_one(torch.zeros(8), target_mass) is None
    assert model.seen[1].tolist() == [[0, 3, 4]]


def test_sampler_can_disable_mass_shell_for_unconstrained_diagnostics():
    class SuffixConfidentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.seen.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 4), -torch.inf, device=input_ids.device)
            logits[:, 1, 2] = 1.0
            logits[:, 1, 1] = 0.0
            logits[:, 2, 1] = 10.0
            logits[:, 2, 2] = 0.0
            return logits

    model = SuffixConfidentModel()
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0],
            [0, 1, 0, 0],
            [0.0, 4.0, 0.0, 0.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" if 1 in ids else "",
        safe_to_smiles=lambda safe: safe or None,
        forbidden_token_ids=(0, 3),
        mass_shell_enabled=False,
    )

    sampler.generate_ranked_with_stats(torch.zeros(8), target_mass=12.0, candidates=1)

    assert model.seen[1].tolist() == [[0, 3, 1]]


def test_batched_sampler_discards_tokens_after_eos_before_mass_and_decoding():
    class EosThenJunkModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=4,
                block_width=3,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            logits = torch.zeros((*input_ids.shape, 5), device=input_ids.device)
            logits[:, 1, 1] = 20.0
            if input_ids.shape[1] > 2:
                logits[:, 2, 2] = 15.0
                logits[:, 3, 4] = 10.0
            return logits

    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    constraint = MassShellConstraint(
        [0.0, 12.0, 0.0, 0.0, 16.0],
        [0, 1, 0, 0, 1],
        [0.0, 4.0, 0.0, 0.0, 2.0],
        eos_token_id=2,
        ppm_tolerance=10,
    )
    sampler = MarlinSampler(
        EosThenJunkModel(),
        constraint,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: (
            "".join({1: "C", 4: "X"}.get(i, "") for i in ids) if 2 in ids else ""
        ),
        safe_to_smiles=lambda safe: safe if safe == "C" else None,
        forbidden_token_ids=(0, 3),
    )

    assert sampler.generate_one(torch.zeros(8), target_mass) == ("C", "C")
    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=1
    )

    assert stats.valid == 1
    assert stats.mass_valid == 1
    assert [candidate.smiles for candidate in ranked] == ["C"]


def test_batched_sampler_rejects_suffix_revealed_before_eos_when_over_mass():
    class SuffixFirstModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []
            self.config = MarlinDecoderConfig(
                vocab_size=7,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=4,
                block_width=3,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.seen.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 7), -torch.inf)
            logits[:, 1, 1] = 10.0
            logits[:, 1, 5] = 0.0
            logits[:, 2, 2] = 5.0
            logits[:, 2, 5] = 0.0
            logits[:, 2, 6] = 0.0
            logits[:, 3, 4] = 20.0
            logits[:, 3, 5] = 0.0
            return logits

    model = SuffixFirstModel()
    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            [0, 1, 0, 0, 1, 0, 0],
            [0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" if 1 in ids and 2 in ids else "",
        safe_to_smiles=lambda safe: safe or None,
        forbidden_token_ids=(0, 3),
    )

    assert sampler.generate_one(torch.zeros(8), target_mass) is None
    assert model.seen[1].tolist() == [[0, 3, 3, 4]]
    model.seen.clear()
    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=1
    )

    assert model.seen[1].tolist() == [[0, 3, 3, 4]]
    assert stats.mass_valid == 0
    assert ranked == []


def test_mass_state_counts_committed_tokens_across_holes_and_stops_at_eos():
    class MinimalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(vocab_size=6, mask_token_id=3)

    constraint = MassShellConstraint(
        [0.0, 12.0, 0.0, 0.0, 16.0, 14.0],
        [0, 1, 0, 0, 1, 1],
        [0.0, 4.0, 0.0, 0.0, 2.0, 3.0],
        eos_token_id=2,
    )
    sampler = MarlinSampler(
        MinimalModel(),
        constraint,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda _: "",
        safe_to_smiles=lambda _: None,
    )

    state = sampler._mass_state([0, 3, 1, 3, 4, 2, 5])

    assert state == MassShellState(
        heavy_mass=28.0,
        heavy_atoms=2,
        valence_sum=6.0,
    )


def test_batched_sampler_keeps_mass_valid_candidate_at_max_length():
    class CarbonModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=1,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            logits = torch.full(
                (*input_ids.shape, 4), -torch.inf, device=input_ids.device
            )
            logits[..., 1] = 0.0
            return logits

    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("CC"))
    constraint = MassShellConstraint(
        [0.0, 12.0, 0.0, 0.0],
        [0, 1, 0, 0],
        [0.0, 4.0, 0.0, 0.0],
        eos_token_id=2,
        ppm_tolerance=10,
    )
    sampler = MarlinSampler(
        CarbonModel(),
        constraint,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" * ids.count(1),
        safe_to_smiles=lambda safe: safe,
        forbidden_token_ids=(0, 3),
    )

    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=1
    )

    assert stats.valid == 1
    assert stats.mass_valid == 1
    assert [candidate.smiles for candidate in ranked] == ["CC"]


def test_batched_sampler_aligns_first_generated_block_after_bos():
    class BlockAlignedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=4,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            assert input_ids.shape[1] == 3
            logits = torch.full(
                (*input_ids.shape, 4), -torch.inf, device=input_ids.device
            )
            logits[:, 1, 1] = 1.0
            logits[:, 2, 2] = 0.0
            return logits

    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    sampler = MarlinSampler(
        BlockAlignedModel(),
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0],
            [0, 1, 0, 0],
            [0.0, 4.0, 0.0, 0.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" if 1 in ids else "",
        safe_to_smiles=lambda safe: safe,
        forbidden_token_ids=(0, 3),
    )

    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=1
    )

    assert stats.mass_valid == 1
    assert [candidate.smiles for candidate in ranked] == ["C"]


def test_sampler_supplies_the_isotope_token_that_training_always_emits():
    captured: dict[str, object] = {}

    class CapturingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                fingerprint_bits=8,
                max_length=3,
                block_width=2,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def sampling_logits(
            self, input_ids, precursor_mass, fingerprint, isotope_ratios=None
        ):
            captured["isotope_ratios"] = isotope_ratios
            logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
            logits[..., 1] = 10.0
            return logits

    molecule = Chem.MolFromSmiles("C")
    sampler = MarlinSampler(
        CapturingModel(),
        MassShellConstraint([0.0] * 4, eos_token_id=2, ppm_tolerance=10),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda _: "C",
        safe_to_smiles=lambda _: "C",
        forbidden_token_ids=(0, 3),
    )
    target_mass = Descriptors.ExactMolWt(molecule)

    sampler.generate_ranked_with_stats(torch.zeros(8), target_mass, candidates=3)
    assert captured["isotope_ratios"] is None

    sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=3, isotope_ratios=(0.033, 0.002)
    )
    supplied = captured["isotope_ratios"]
    assert supplied is not None
    assert supplied.shape == (3, 2)
    assert torch.allclose(supplied[0], torch.tensor([0.033, 0.002]))


def test_soft_fingerprints_use_the_threshold_for_sparsity():
    import tempfile
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from marlin.evaluation import load_fingerprints

    probabilities = np.zeros((1, 4096), dtype=np.float32)
    probabilities[0, 0] = 0.97
    probabilities[0, 1] = 0.62
    probabilities[0, 2] = 0.10
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "soft.npz"
        np.savez(path, probs=probabilities, spectrum_ids=np.array(["s0"], dtype="U8"))
        metadata = pd.DataFrame({"spec_name": ["s0"]})

        values = load_fingerprints(
            path, "probs", 0.95, metadata, preserve_probabilities=True
        )

    # The encoder activates any bit above 0.5, so a sub-threshold probability
    # would silently join the conditioning set unless it is zeroed here.
    assert values[0, 0] == pytest.approx(0.97)
    assert values[0, 1] == 0.0
    assert values[0, 2] == 0.0


def test_symmetric_noise_moves_soft_amplitudes_instead_of_saturating_them():
    fingerprint = torch.tensor([[0.92, 0.96, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    generator = torch.Generator().manual_seed(3)

    noisy = symmetric_fingerprint_noise(
        fingerprint,
        corruption_probability=1.0,
        min_fraction=0.5,
        max_fraction=0.5,
        generator=generator,
    )

    # A saturated injected bit would outrank every genuine DreaMS probability and
    # make the noise the most confident conditioning token in the batch.
    assert noisy.max() <= fingerprint.max()
    assert sorted(noisy[noisy > 0].tolist()) == sorted(
        fingerprint[fingerprint > 0].tolist()
    )


def test_time_budget_truncates_between_batches_and_says_so():
    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
            logits[..., 1] = 10.0
            return logits

    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    sampler = MarlinSampler(
        FixedModel(),
        MassShellConstraint([0.0] * 4, eos_token_id=2, ppm_tolerance=10),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda _: "C",
        safe_to_smiles=lambda _: "C",
        forbidden_token_ids=(0, 3),
    )

    _, full = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=8, candidate_batch_size=2
    )
    assert full.attempts == 8
    assert not full.truncated

    # A budget already spent must stop after the first batch, never mid candidate.
    _, capped = sampler.generate_ranked_with_stats(
        torch.zeros(8),
        target_mass,
        candidates=8,
        candidate_batch_size=2,
        time_budget_seconds=1e-9,
    )
    assert capped.truncated
    assert capped.attempts == 2
    assert capped.valid <= capped.attempts

    with pytest.raises(ValueError):
        sampler.generate_ranked_with_stats(
            torch.zeros(8), target_mass, candidates=2, time_budget_seconds=0
        )

    # scripts/evaluate_marlin_nplib1.py asks for 8 candidates in one batch of 8,
    # so a deadline read only between batches is never read at all: a spectrum of
    # the clean 321-spectrum panel ran 19,311 s against --per-spectrum-seconds
    # 1800, and the panel's own three-spectrum arm never finished.
    _, single_batch = sampler.generate_ranked_with_stats(
        torch.zeros(8),
        target_mass,
        candidates=8,
        candidate_batch_size=8,
        time_budget_seconds=1e-9,
    )
    assert single_batch.truncated


LAZY_PROBE_TOKENIZER_PATHS = (
    Path(
        "/mnt/netstorage/nikolenko/marlin/cache/runtime-inputs-spectrum-v1/"
        "16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3/"
        "tokenizer.json"
    ),
    Path(
        "/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/"
        "data/safe-gpt/tokenizer.json"
    ),
)
# Prefixes taken from the recorded decodes under
# /mnt/netstorage/nikolenko/marlin/evaluations/attempt-fan{,-big}/trace.jsonl:
# the opening of an attempt, a ring under construction, a branch, a closed ring
# and a long chain, which is where the full-support call costs 5-23 s.
LAZY_PROBE_PREFIXES = (
    "",
    "C",
    "CC",
    "c1ccc",
    "COc1cc(",
    "CC1(C)CCc2c(O1)",
    "CC(=O)N1CCC(O)CC1",
    "Cc1c(C)c2c(OCC(=O)N3CCC(O)CC3)cc3c(c2oc1=O)",
    "C" * 24,
)


def _lazy_probe_fixture():
    """Build the sampler pieces one clean-panel evaluation actually runs with."""
    from marlin.grammar import SafeGrammarMask
    from marlin.token_properties import (
        foreign_element_token_ids,
        isotope_token_ids,
    )

    for path in LAZY_PROBE_TOKENIZER_PATHS:
        if path.exists():
            tokenizer = load_safe_tokenizer(path)
            break
    else:
        pytest.skip(f"no real SAFE tokenizer under {LAZY_PROBE_TOKENIZER_PATHS}")

    special_ids = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
        tokenizer.pad_token_id,
    }
    token_strings = [
        tokenizer.convert_ids_to_tokens(index) for index in range(len(tokenizer))
    ]
    masses, atoms, valences = build_token_property_table(
        len(tokenizer), tokenizer.convert_ids_to_tokens, special_ids
    )
    constraint = MassShellConstraint(
        masses,
        atoms,
        valences,
        ppm_tolerance=10.0,
        valence_slack=4.0,
        eos_boost=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    chemistry_forbidden_ids = tuple(
        sorted(
            set(isotope_token_ids(token_strings))
            | set(foreign_element_token_ids(token_strings))
        )
    )
    mask = SafeGrammarMask(
        token_strings,
        lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id,
        special_token_ids=tuple(special_ids) + (tokenizer.unk_token_id,),
        forbidden_token_ids=chemistry_forbidden_ids,
        ppm_tolerance=10.0,
        valence_slack=4.0,
        mass_reachability_prune=True,
        restrict_organic_elements=True,
        forbid_isotopes=True,
    )
    forbidden = tuple(
        token_id
        for token_id in (
            tokenizer.unk_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
        )
        + chemistry_forbidden_ids
        if token_id != tokenizer.eos_token_id
    )
    return tokenizer, constraint, mask, forbidden


class _ProbeOnlyModel(torch.nn.Module):
    """A stand-in for the decoder: ``_probe_token`` never reads the model."""

    def __init__(self, vocab_size: int, mask_token_id: int) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = MarlinDecoderConfig(
            vocab_size=vocab_size,
            hidden_size=4,
            num_layers=1,
            num_heads=1,
            intermediate_size=4,
            max_length=8,
            block_width=4,
            fingerprint_bits=8,
            dropout=0.0,
            mask_token_id=mask_token_id,
            pad_token_id=0,
        )

    def forward(self, input_ids, precursor_mass, fingerprint):
        return torch.zeros((*input_ids.shape, self.config.vocab_size))


def test_lazy_probe_commits_the_token_the_full_support_would_commit():
    tokenizer, constraint, mask, forbidden = _lazy_probe_fixture()
    sampler = MarlinSampler(
        _ProbeOnlyModel(len(tokenizer), tokenizer.mask_token_id),
        constraint,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id,
        decode_tokens=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        safe_to_smiles=lambda safe: None,
        grammar_mask=mask,
        forbidden_token_ids=forbidden,
    )
    assert sampler._lazy_probe_available

    target_mass = 415.197710533379
    checked = 0
    for prefix_text in LAZY_PROBE_PREFIXES:
        prefix = [tokenizer.bos_token_id] + (
            tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            if prefix_text
            else []
        )
        state = MassShellState()
        for token_id in prefix[1:]:
            state = constraint.advance(state, token_id)
        for logit_seed in range(3):
            draw = torch.Generator().manual_seed(logit_seed)
            # Peaked like the decoder's own output: 70.3% of the positions of a
            # real decode have exactly one token above p=0.01.
            logits = 6.0 * torch.randn(len(tokenizer), generator=draw)
            shell = constraint.apply(logits, state, target_mass)
            shell[list(forbidden)] = -torch.inf
            masked = mask(prefix, shell.clone(), target_mass)
            probabilities = masked.softmax(dim=-1)
            alive = bool(torch.isfinite(probabilities.max(dim=-1).values))

            for sample_tokens in (False, True):
                sampler.sample_tokens = sample_tokens
                for seed in range(4):
                    full = torch.Generator().manual_seed(seed)
                    if not alive:
                        expected = None
                    elif sample_tokens:
                        expected = int(
                            torch.multinomial(
                                probabilities, num_samples=1, generator=full
                            ).item()
                        )
                    else:
                        expected = int(probabilities.argmax().item())
                    lazy = torch.Generator().manual_seed(seed)
                    assert (
                        sampler._probe_token(
                            prefix, shell.clone(), target_mass, lazy
                        )
                        == expected
                    )
                    # The stream has to advance identically or every later
                    # position of the decode diverges.
                    assert torch.equal(full.get_state(), lazy.get_state())
                    checked += 1
    assert checked == len(LAZY_PROBE_PREFIXES) * 3 * 2 * 4
    assert sampler.lazy_probe_positions == checked


def test_lazy_probe_walks_past_its_width_and_builds_the_support_only_when_it_must():
    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "a", "b", "c", "d")
    admitted = {8}

    class NarrowMask:
        eos_token_id = 2

        def __init__(self):
            self.supports_built = 0

        def admits(self, prefix_ids, token_id, target_mass=None):
            return token_id in admitted

        def __call__(self, prefix_ids, logits, target_mass=None):
            self.supports_built += 1
            support = torch.zeros(len(tokens), dtype=torch.bool)
            support[list(admitted)] = True
            return torch.where(support, logits, torch.full_like(logits, -torch.inf))

    grammar = NarrowMask()
    sampler = MarlinSampler(
        _ProbeOnlyModel(len(tokens), 4),
        MassShellConstraint([0.0] * len(tokens), eos_token_id=2, ppm_tolerance=10),
        bos_token_id=1,
        eos_token_id=2,
        mask_token_id=4,
        decode_tokens=lambda ids: "",
        safe_to_smiles=lambda safe: None,
        grammar_mask=grammar,
        forbidden_token_ids=(0, 1, 3, 4),
        lazy_probe_width=2,
        sample_tokens=False,
    )
    logits = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 9.0, 8.0, 7.0, 6.0])

    # The answer is the lowest-ranked token the mass shell left standing, well
    # past the probe width. Building the support instead would cost 13-23 s at
    # the prefix lengths where that happens, so the walk simply continues.
    assert sampler._probe_token([1], logits.clone(), 100.0, None) == 8
    assert sampler.lazy_probe_misses == 1
    assert sampler.lazy_probe_fallbacks == 0
    assert grammar.supports_built == 0
    assert sampler.lazy_probe_admits_calls == 4

    # Nothing admissible anywhere is a dead end the walk proves on its own, and
    # it leaves the generator where the full-support path -- which never reaches
    # its multinomial call -- would have left it.
    admitted.clear()
    sampler.sample_tokens = True
    generator = torch.Generator().manual_seed(3)
    before = generator.get_state()
    assert sampler._probe_token([1], logits.clone(), 100.0, generator) is None
    assert torch.equal(generator.get_state(), before)
    assert grammar.supports_built == 0

    # A winner the float32 path could rank differently is handed to that path.
    admitted.update({5, 6})
    sampler.sample_tokens = False
    tied = logits.clone()
    tied[6] = tied[5]
    assert sampler._probe_token([1], tied, 100.0, None) == 5
    assert sampler.lazy_probe_fallbacks == 1
    assert grammar.supports_built == 1


def test_time_budget_stops_a_decode_inside_a_block():
    import time as _time

    class SlowModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.calls = 0
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=5,
                block_width=4,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.calls += 1
            _time.sleep(0.05)
            logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
            logits[..., 1] = 10.0
            return logits

    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))

    def build():
        model = SlowModel()
        decoded: list[list[int]] = []

        def decode(token_ids):
            decoded.append(list(token_ids))
            return "C"

        return model, decoded, MarlinSampler(
            model,
            MassShellConstraint([0.0] * 4, eos_token_id=2, ppm_tolerance=10),
            bos_token_id=0,
            eos_token_id=2,
            mask_token_id=3,
            decode_tokens=decode,
            safe_to_smiles=lambda _: "C",
            forbidden_token_ids=(0, 3),
        )

    model, _, sampler = build()
    _, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=1
    )
    assert not stats.truncated
    # One block of four positions, so four forward passes when nothing stops it.
    assert model.calls == 4

    # A budget of one position's work has to stop inside that block; read only at
    # the block boundary it would run all four. A 900 s budget produced a 1,308 s
    # spectrum for exactly this reason.
    model, decoded, sampler = build()
    _, capped = sampler.generate_ranked_with_stats(
        torch.zeros(8),
        target_mass,
        candidates=1,
        time_budget_seconds=0.06,
    )
    assert capped.truncated
    assert 0 < model.calls < 4
    # Positions are resolved leftmost first, so an abandoned block leaves masks
    # only at the end: what is decoded is a prefix, never a string with a hole.
    assert decoded
    for token_ids in decoded:
        holes = [index for index, token in enumerate(token_ids) if token == 3]
        assert holes == list(range(len(token_ids) - len(holes), len(token_ids)))
