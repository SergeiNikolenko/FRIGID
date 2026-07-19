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
from marlin.sampler import MarlinSampler
from marlin.token_properties import token_properties
from marlin.warm_start import _copy_attention


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
    tokens, mask = conditioner(torch.tensor([250.0]), fingerprint)
    assert tokens.shape == (1, 4, 28)
    assert mask.tolist() == [[True, True, True, True]]


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


def test_token_properties_ignore_safe_grammar_characters():
    properties = token_properties("C1=CC(Cl)=CC=C1")
    assert properties.heavy_atoms == 7
    assert properties.heavy_mass > 100


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

    def capture(clean_ids, noised_ids, precursor_mass, fingerprint):
        captured["noised_ids"] = noised_ids.clone()
        return original(clean_ids, noised_ids, precursor_mass, fingerprint)

    model.two_stream_logits = capture
    model.diffusion_loss(
        tokens,
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        generator=torch.Generator().manual_seed(1),
    )

    assert captured["noised_ids"][0, 0] == tokens[0, 0]


def test_attention_warm_start_concatenates_qkv():
    attention = torch.nn.MultiheadAttention(4, 1, batch_first=True)
    state = {}
    for offset, part in enumerate(("query", "key", "value")):
        state[f"x.{part}.weight"] = torch.full((4, 4), float(offset + 1))
        state[f"x.{part}.bias"] = torch.full((4,), float(offset + 1))
    _copy_attention(attention, state, "x", "test")
    assert torch.equal(attention.in_proj_weight[:4], state["x.query.weight"])
    assert torch.equal(attention.in_proj_weight[-4:], state["x.value.weight"])


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


def test_batched_sampler_discards_tokens_after_eos_before_decoding():
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
        [0.0, 12.0, 0.0, 0.0, 0.0],
        [0, 1, 0, 0, 0],
        [0.0, 4.0, 0.0, 0.0, 0.0],
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

    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=1
    )

    assert stats.valid == 1
    assert stats.mass_valid == 1
    assert [candidate.smiles for candidate in ranked] == ["C"]


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
