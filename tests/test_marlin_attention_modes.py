import pytest
import torch

from marlin.model import (
    MarlinDecoder,
    MarlinDecoderConfig,
    block_causal_attention_mask,
)


def _tiny_decoder() -> MarlinDecoder:
    return MarlinDecoder(
        MarlinDecoderConfig(
            vocab_size=8,
            hidden_size=8,
            num_layers=1,
            num_heads=1,
            intermediate_size=16,
            max_length=8,
            block_width=2,
            fingerprint_bits=4,
            dropout=0.0,
        )
    )


def _capture_forward_mask(
    monkeypatch: pytest.MonkeyPatch,
    model: MarlinDecoder,
) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def fake_forward_with_mask(
        input_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        isotope_ratios: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        del precursor_mass, fingerprint, isotope_ratios
        captured["attention_mask"] = kwargs["attention_mask"]  # type: ignore[assignment]
        return torch.zeros(
            (*input_ids.shape, model.config.vocab_size),
            device=input_ids.device,
        )

    monkeypatch.setattr(model, "_forward_with_mask", fake_forward_with_mask)
    return captured


def test_frigid_full_attention_is_unmasked_including_bos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_decoder()
    captured = _capture_forward_mask(monkeypatch, model)

    model(
        torch.tensor([[1, 4, 4, 2, 0]]),
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        attention_mode="frigid_full",
    )

    mask = captured["attention_mask"]
    assert mask.dtype == torch.bool
    assert mask.shape == (5, 5)
    assert not mask.any()
    assert not mask[0].any()
    assert not mask[:, 0].any()


def test_block_attention_accepts_a_positive_width_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_decoder()
    captured = _capture_forward_mask(monkeypatch, model)
    tokens = torch.tensor([[1, 4, 4, 2, 0]])

    model(
        tokens,
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        attention_mode="block",
        block_width_override=4,
    )

    assert torch.equal(
        captured["attention_mask"],
        block_causal_attention_mask(5, 4, tokens.device),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"attention_mode": "unknown"}, "unknown attention mode"),
        ({"block_width_override": 0}, "block_width_override must be positive"),
    ],
)
def test_decoder_rejects_invalid_attention_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    model = _tiny_decoder()

    with pytest.raises(ValueError, match=message):
        model(
            torch.tensor([[1, 4, 2]]),
            torch.tensor([50.0]),
            torch.zeros((1, 4)),
            **kwargs,
        )
