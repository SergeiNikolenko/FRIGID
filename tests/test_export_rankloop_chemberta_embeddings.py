from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "export_rankloop_chemberta_embeddings.py"
)
SPEC = importlib.util.spec_from_file_location(
    "export_rankloop_chemberta_embeddings",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cls_and_masked_mean_pooling():
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
            [[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    cls = MODULE.pool_hidden_states(hidden, mask, "cls")
    mean = MODULE.pool_hidden_states(hidden, mask, "mean")

    assert torch.equal(cls, torch.tensor([[1.0, 2.0], [2.0, 4.0]]))
    assert torch.equal(mean, torch.tensor([[2.0, 3.0], [4.0, 8.0]]))


class _MaskedLanguageModelStub:
    def __init__(self, *, tied: bool) -> None:
        self.input_embeddings = nn.Embedding(4, 3)
        self.output_embeddings = nn.Linear(3, 4, bias=False)
        if tied:
            self.output_embeddings.weight = self.input_embeddings.weight

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings


def test_chemberta_loader_requires_tied_mlm_embeddings():
    MODULE.require_tied_input_output_embeddings(_MaskedLanguageModelStub(tied=True))

    with pytest.raises(RuntimeError, match="not tied"):
        MODULE.require_tied_input_output_embeddings(
            _MaskedLanguageModelStub(tied=False)
        )
