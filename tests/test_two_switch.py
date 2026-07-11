from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from frigid.two_switch import generate_two_switch_neighbors


SEED_SMILES = "CCOC(=O)NCC1CCCCC1"


def _connectivity_key(mol: Chem.Mol) -> str:
    return Chem.MolToInchiKey(mol).split("-", maxsplit=1)[0]


def _atom_signature(mol: Chem.Mol):
    return [
        (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope())
        for atom in mol.GetAtoms()
    ]


def _degrees(mol: Chem.Mol):
    return [atom.GetDegree() for atom in mol.GetAtoms()]


def _valences(mol: Chem.Mol):
    return [atom.GetTotalValence() for atom in mol.GetAtoms()]


def test_neighbors_preserve_atoms_formula_degrees_valence_and_connectivity():
    seed_mol = Chem.MolFromSmiles(SEED_SMILES)
    assert seed_mol is not None
    seed_formula = rdMolDescriptors.CalcMolFormula(seed_mol)
    seed_key = _connectivity_key(seed_mol)

    neighbors, stats = generate_two_switch_neighbors(
        SEED_SMILES,
        seed=134,
        max_proposals=256,
        max_neighbors=32,
    )

    assert neighbors
    assert stats.accepted_unique == len(neighbors)
    assert len({neighbor.inchi_key_connectivity for neighbor in neighbors}) == len(
        neighbors
    )
    for neighbor in neighbors:
        mol = Chem.MolFromSmiles(neighbor.smiles)
        assert mol is not None
        assert mol.GetNumAtoms() == seed_mol.GetNumAtoms()
        assert sorted(_atom_signature(mol)) == sorted(_atom_signature(seed_mol))
        assert rdMolDescriptors.CalcMolFormula(mol) == seed_formula
        assert sorted(_degrees(mol)) == sorted(_degrees(seed_mol))
        assert sorted(_valences(mol)) == sorted(_valences(seed_mol))
        assert len(Chem.GetMolFrags(mol)) == 1
        assert _connectivity_key(mol) == neighbor.inchi_key_connectivity
        assert neighbor.inchi_key_connectivity != seed_key


def test_generation_is_deterministic_for_seed_and_caps():
    kwargs = {
        "seed": 13420260711,
        "max_proposals": 24,
        "max_neighbors": 6,
    }
    first, first_stats = generate_two_switch_neighbors(SEED_SMILES, **kwargs)
    second, second_stats = generate_two_switch_neighbors(SEED_SMILES, **kwargs)

    assert first == second
    assert first_stats.as_dict() == second_stats.as_dict()
    assert len(first) <= kwargs["max_neighbors"]
    assert first_stats.proposals_considered <= kwargs["max_proposals"]


def test_excluded_connectivity_keys_are_not_returned():
    initial, _ = generate_two_switch_neighbors(
        SEED_SMILES,
        seed=7,
        max_proposals=256,
        max_neighbors=8,
    )
    assert initial

    excluded = {initial[0].inchi_key_connectivity}
    filtered, stats = generate_two_switch_neighbors(
        SEED_SMILES,
        seed=7,
        max_proposals=256,
        max_neighbors=8,
        exclude_connectivity_keys=excluded,
    )

    assert excluded.isdisjoint(neighbor.inchi_key_connectivity for neighbor in filtered)
    assert stats.invalid_counts["duplicate_connectivity"] >= 1


def test_invalid_seed_and_nonpositive_caps_are_rejected():
    for kwargs in (
        {"smiles": "not-smiles", "seed": 1, "max_proposals": 1, "max_neighbors": 1},
        {"smiles": "CC", "seed": 1, "max_proposals": 0, "max_neighbors": 1},
        {"smiles": "CC", "seed": 1, "max_proposals": 1, "max_neighbors": 0},
    ):
        try:
            generate_two_switch_neighbors(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")
