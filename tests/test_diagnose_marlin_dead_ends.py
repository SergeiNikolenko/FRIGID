"""Lock the classifications the dead-end diagnosis reports."""

from rdkit import Chem

from marlin.grammar import _scan
from scripts.diagnose_marlin_dead_ends import (
    classify_parse_failure,
    group_of,
    phantom_atoms,
)


def _row(**overrides):
    row = {
        "attempts": 8,
        "candidate_returned": False,
        "constraint_dead_ends": 0,
        "valid": 0,
    }
    row.update(overrides)
    return row


def test_group_of_separates_the_four_outcomes():
    assert group_of(_row(candidate_returned=True, valid=3)) == "returned"
    assert group_of(_row(constraint_dead_ends=8)) == "all_dead_end"
    assert group_of(_row(constraint_dead_ends=4, valid=2)) == "mass_miss"
    assert group_of(_row(constraint_dead_ends=4, valid=0)) == "no_parse"


def test_phantom_atoms_counts_what_the_grammar_weighs_at_zero():
    # marlin.grammar._ATOM_MASSES has no hydrogen entry and only 18 elements, so
    # a bracket hydrogen and any element outside that table enter the state at
    # mass 0.0 while the mass-shell token table charges their real mass.
    counts = phantom_atoms(_scan("C.[H].[Og]"))

    assert counts["atoms"] == 3
    assert counts["zero_mass_atoms"] == 2
    assert counts["bracket_hydrogen_atoms"] == 1
    assert counts["foreign_element_atoms"] == 1
    assert phantom_atoms(_scan("CCO"))["zero_mass_atoms"] == 0


def test_classify_parse_failure_names_the_rdkit_cause():
    assert classify_parse_failure("CCO")["bucket"] == "parses"
    assert classify_parse_failure("c1cccc1")["bucket"] == "kekulization"
    # The diagnosis recorded 33 of 91 unparsable terminal strings as a ring closure
    # duplicating an existing bond, which the grammar accepted and RDKit refused. The
    # grammar now refuses it too, so the classifier is exercised on the string alone.
    assert _scan("C12CC12") is None
    assert Chem.MolFromSmiles("C12CC12") is None
    assert classify_parse_failure("C12CC12")["bucket"] == "duplicate_ring_bond"
