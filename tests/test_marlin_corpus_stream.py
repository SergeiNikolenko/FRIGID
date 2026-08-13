"""Tests for the fp2mol corpus stream."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from marlin.corpus_stream import Fp2MolStream, corpus_row_groups

# Two of these carry stereo centres; one is a repeat structure written the
# other way round, so a destereo stream must collapse them.
SMILES = [
    "CCOc1cc(ccc1NC(=O)N[C@@H]2CCCC[C@@H]2O)F",
    "CCOc1cc(ccc1NC(=O)N[C@H]2CCCC[C@H]2O)F",
    "Cc1ccccc1O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2[nH]ccc2c1",
]


class _Tokenizer:
    """Length-only stand-in: the stream only asks for a token count."""

    def __init__(self, per_char: float = 1.0) -> None:
        self.per_char = per_char

    def encode(self, text, add_special_tokens=True):
        return [0] * (int(len(text) * self.per_char) + (2 if add_special_tokens else 0))


@pytest.fixture()
def snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "data" / "train").mkdir(parents=True)
    path = root / "data" / "train" / "train-00000.parquet"
    table = pa.table({"smiles": SMILES, "safe": ["x"] * len(SMILES)})
    pq.write_table(table, path, row_group_size=3)
    (root / "manifest.json").write_text(
        json.dumps({"files": [{"path": "data/train/train-00000.parquet"}]})
    )
    return root


def _drain(stream, count):
    out = []
    for example in stream:
        out.append(example)
        if len(out) >= count:
            break
    return out


def test_row_groups_are_enumerated_without_reading_data(snapshot):
    refs = corpus_row_groups(snapshot)
    assert len(refs) == 2  # 5 rows at row_group_size=3
    assert all(ref.path.endswith("train-00000.parquet") for ref in refs)


def test_examples_carry_the_shape_the_collator_expects(snapshot):
    stream = Fp2MolStream(
        snapshot=snapshot, tokenizer=_Tokenizer(), limit=5, allow_evaluation_structures=True
    )
    examples = list(stream)
    assert len(examples) == 5
    for example in examples:
        assert set(example) == {"safe", "fingerprint", "precursor_mass"}
        assert example["fingerprint"].shape == (4096,)
        assert example["fingerprint"].dtype == np.float32
        assert set(np.unique(example["fingerprint"])) <= {0.0, 1.0}
        assert example["precursor_mass"] > 0.0
        assert isinstance(example["safe"], str) and example["safe"]


def test_the_safe_string_decodes_to_the_molecule_the_fingerprint_describes(snapshot):
    from dlm.utils.utils_chem import safe_to_smiles

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    for example in Fp2MolStream(
        snapshot=snapshot, tokenizer=_Tokenizer(), limit=5, allow_evaluation_structures=True
    ):
        molecule = Chem.MolFromSmiles(safe_to_smiles(example["safe"], fix=False))
        assert molecule is not None
        expected = generator.GetFingerprintAsNumPy(molecule).astype(np.float32)
        assert np.array_equal(expected, example["fingerprint"])


def test_stereo_is_stripped_because_the_adaptation_set_has_none(snapshot):
    examples = list(Fp2MolStream(
        snapshot=snapshot, tokenizer=_Tokenizer(), limit=5, allow_evaluation_structures=True
    ))
    assert all("@" not in example["safe"] for example in examples)
    # the two enantiomers collapse onto one SAFE string, so five inputs give four
    assert len({example["safe"] for example in examples}) == 4


def test_stereo_survives_when_the_stream_is_told_to_keep_it(snapshot):
    examples = list(
        Fp2MolStream(
            snapshot=snapshot,
            tokenizer=_Tokenizer(),
            limit=5,
            remove_stereo=False,
            allow_evaluation_structures=True,
        )
    )
    assert any("@" in example["safe"] for example in examples)
    assert len({example["safe"] for example in examples}) == 5


def test_a_molecule_too_long_for_the_decoder_is_dropped_not_truncated(snapshot):
    # "CC(=O)Oc1ccccc1C(=O)O" is the longest of the five, so a cut just under it
    # drops that one and keeps the rest.
    stream = Fp2MolStream(
        snapshot=snapshot,
        tokenizer=_Tokenizer(),
        max_length=22,
        limit=4,
        allow_evaluation_structures=True,
    )
    examples = list(stream)
    assert len(examples) == 4
    assert stream.rejections["too_long"] > 0
    assert all(len(example["safe"]) + 2 <= 22 for example in examples)


def test_a_corpus_that_yields_nothing_raises_instead_of_hanging(snapshot):
    stream = Fp2MolStream(
        snapshot=snapshot,
        tokenizer=_Tokenizer(per_char=10.0),
        max_length=8,
        allow_evaluation_structures=True,
    )
    with pytest.raises(ValueError, match="emitted nothing"):
        list(stream)


def test_evaluation_structures_are_refused(snapshot, tmp_path):
    key = Chem.MolToInchiKey(Chem.MolFromSmiles("Cc1ccccc1O")).split("-")[0]
    exclusion = tmp_path / "exclude.csv"
    exclusion.write_text(f"inchikey\n{key}-UHFFFAOYSA-N\n")
    # Two full passes over the five rows, so the excluded molecule is offered
    # regardless of where the shuffle puts it.
    stream = Fp2MolStream(
        snapshot=snapshot,
        tokenizer=_Tokenizer(),
        exclude_inchikeys=exclusion,
        limit=8,
    )
    examples = list(stream)
    assert len(examples) == 8
    assert stream.rejections.get("evaluation_structure", 0) >= 1
    from dlm.utils.utils_chem import safe_to_smiles

    seen = {
        Chem.MolToInchiKey(
            Chem.MolFromSmiles(safe_to_smiles(example["safe"], fix=False))
        ).split("-")[0]
        for example in examples
    }
    assert key not in seen


def test_the_stream_wraps_rather_than_stopping(snapshot):
    examples = _drain(
        Fp2MolStream(
            snapshot=snapshot, tokenizer=_Tokenizer(), allow_evaluation_structures=True
        ),
        12,
    )
    assert len(examples) == 12  # 5 distinct molecules, so it must have wrapped


def test_the_order_is_a_function_of_the_seed(snapshot):
    def order(seed):
        return tuple(
            example["safe"]
            for example in Fp2MolStream(
                snapshot=snapshot,
                tokenizer=_Tokenizer(),
                seed=seed,
                limit=5,
                allow_evaluation_structures=True,
            )
        )

    # Reproducibility is the hard requirement, so it is asserted exactly.
    assert order(0) == order(0)
    assert order(3) == order(3)
    # Five molecules over two row groups admit few distinct orders, so a single
    # seed pair can collide by chance; the claim is that the seed is *used*, and
    # eight seeds all colliding is not chance.
    assert len({order(seed) for seed in range(8)}) > 1


def test_an_empty_snapshot_is_refused(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"files": []}))
    with pytest.raises(ValueError, match="no parquet row groups"):
        Fp2MolStream(
            snapshot=root, tokenizer=_Tokenizer(), allow_evaluation_structures=True
        )


def test_an_unfiltered_stream_is_refused_unless_it_is_asked_for(snapshot):
    # The corpus carries 23 of the 320 clean-panel connectivity blocks in its
    # first ten million rows, so defaulting to no exclusion trains on the panel.
    with pytest.raises(ValueError, match="needs exclude_inchikeys"):
        Fp2MolStream(snapshot=snapshot, tokenizer=_Tokenizer())


def test_an_exclusion_file_that_excludes_nothing_is_refused(snapshot, tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("inchikey\n")
    with pytest.raises(ValueError, match="lists no connectivity blocks"):
        Fp2MolStream(
            snapshot=snapshot, tokenizer=_Tokenizer(), exclude_inchikeys=empty
        )
