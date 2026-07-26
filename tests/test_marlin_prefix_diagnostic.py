import torch

from marlin.mass_shell import MassShellConstraint
from marlin.model import MarlinDecoderConfig
from marlin.prefix_diagnostic import diagnose_production_prefix
from marlin.sampler import MarlinSampler


TOKENS = ("[PAD]", "[BOS]", "[EOS]", "[MASK]", "C", "O", "N")


class PrefixActionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen = []
        self.config = MarlinDecoderConfig(
            vocab_size=len(TOKENS),
            hidden_size=4,
            num_layers=1,
            num_heads=1,
            intermediate_size=4,
            max_length=5,
            block_width=2,
            fingerprint_bits=8,
            dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            mask_token_id=3,
            pad_token_id=0,
        )

    def sampling_logits(self, input_ids, precursor_mass, fingerprint):
        self.seen.append(input_ids.detach().clone())
        logits = torch.zeros(
            (*input_ids.shape, len(TOKENS)),
            device=input_ids.device,
        )
        logits[..., 3] = 10.0
        logits[:, 1, 4] = 8.0
        if input_ids.shape[1] > 2:
            logits[:, 2, 5] = 8.0
        if input_ids.shape[1] > 3:
            logits[:, 3, 2] = 8.0
        return logits


def make_sampler(*, grammar_mask=None):
    model = PrefixActionModel()
    constraint = MassShellConstraint(
        [0.0, 0.0, 0.0, 0.0, 12.0, 15.99491462, 14.003074004],
        [0, 0, 0, 0, 1, 1, 1],
        [0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 3.0],
        eos_token_id=2,
        ppm_tolerance=10.0,
    )
    return model, MarlinSampler(
        model,
        constraint,
        bos_token_id=1,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "".join(
            TOKENS[token_id] for token_id in ids if token_id >= 4
        ),
        safe_to_smiles=lambda safe: safe or None,
        grammar_mask=grammar_mask,
        forbidden_token_ids=(0, 1, 3),
    )


def test_prefix_diagnostic_reveals_true_actions_across_production_blocks():
    model, sampler = make_sampler()

    result = diagnose_production_prefix(
        sampler,
        [1, 4, 5, 2],
        torch.zeros(8),
        32.026214748,
        lambda token_id: TOKENS[token_id],
    )

    assert [canvas.tolist() for canvas in model.seen] == [
        [[1, 3, 3]],
        [[1, 4, 3]],
        [[1, 4, 5, 3, 3]],
    ]
    assert [action["target_token"] for action in result["actions"]] == [
        "C",
        "O",
        "[EOS]",
    ]
    assert [action["raw"]["top1_id"] for action in result["actions"]] == [3, 3, 3]
    assert [
        action["constrained"]["top1_id"] for action in result["actions"]
    ] == [4, 5, 2]
    assert result["summary"]["target_allowed_rate"] == 1.0
    assert result["summary"]["constrained_top1_accuracy"] == 1.0
    assert result["summary"]["raw_top1_accuracy"] == 0.0


def test_prefix_diagnostic_records_disallowed_target_and_continues_to_eos():
    def reject_oxygen(prefix_ids, logits, _target_mass):
        constrained = logits.clone()
        if prefix_ids == [1, 4]:
            constrained[5] = -torch.inf
        return constrained

    model, sampler = make_sampler(grammar_mask=reject_oxygen)

    result = diagnose_production_prefix(
        sampler,
        [1, 4, 5, 2],
        torch.zeros(8),
        32.026214748,
        lambda token_id: TOKENS[token_id],
    )

    assert len(model.seen) == 3
    assert [action["target_allowed"] for action in result["actions"]] == [
        True,
        False,
        True,
    ]
    assert result["actions"][1]["constrained"]["target_rank"] is None
    assert result["actions"][1]["constrained"]["target_probability"] == 0.0
    assert result["summary"]["target_allowed"] == 2
    assert result["summary"]["first_disallowed_position"] == 2
