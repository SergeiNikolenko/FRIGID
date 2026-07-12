from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


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
