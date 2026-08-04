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
