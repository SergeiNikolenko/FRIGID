from importlib.metadata import version

import rdkit
from packaging.version import Version


def test_rdkit_import_matches_installed_distribution() -> None:
    assert Version(rdkit.__version__) == Version(version("rdkit")), (
        "The imported rdkit package does not match the installed rdkit distribution"
    )
