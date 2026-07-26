import hashlib
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts import train_marlin


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_metadata_override_can_add_or_replace_hydra_keys():
    config_dir = str(PROJECT_ROOT / "configs")
    overrides = [
        "++data.metadata_csv=/tmp/override.csv",
        "++data.metadata_csv_sha256=" + "a" * 64,
    ]

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        strict = compose(config_name="marlin_nplib1", overrides=overrides)
        tiny = compose(
            config_name="marlin_frigid_distilled_tiny_overfit_c16h12o3",
            overrides=overrides,
        )

    for config in (strict, tiny):
        assert config.data.metadata_csv == "/tmp/override.csv"
        assert config.data.metadata_csv_sha256 == "a" * 64


def test_metadata_csv_is_verified_fail_closed(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("smiles\nCCO\n")
    digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
    config = OmegaConf.create(
        {
            "data": {
                "metadata_csv": str(metadata),
                "metadata_csv_sha256": digest,
            }
        }
    )

    assert train_marlin.validate_metadata_csv(config) == (str(metadata), digest)

    del config.data.metadata_csv_sha256
    with pytest.raises(ValueError, match="metadata_csv_sha256 is required"):
        train_marlin.validate_metadata_csv(config)

    config.data.metadata_csv_sha256 = digest
    metadata.write_text("smiles\nCCC\n")
    with pytest.raises(ValueError, match="metadata CSV SHA-256"):
        train_marlin.validate_metadata_csv(config)


def test_metadata_digest_is_written_to_run_manifest(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("smiles\nCCO\n")
    digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
    supporting_input = tmp_path / "supporting-input"
    supporting_input.write_text("pinned\n")
    config = OmegaConf.create(
        {
            "adaptation": {"mode": "strict_marlin"},
            "data": {
                "dataset": "diagnostic/metadata",
                "revision": "local",
                "metadata_csv": str(metadata),
                "metadata_csv_sha256": digest,
                "exclude_inchikeys": str(supporting_input),
                "length_audit_manifest": str(supporting_input),
                "snapshot_manifest": str(supporting_input),
            },
            "output": {"root": str(tmp_path / "run")},
        }
    )
    monkeypatch.setattr(train_marlin, "git_state", lambda: ("abc123", []))

    verified_path, verified_digest = train_marlin.validate_metadata_csv(config)
    assert verified_path == str(metadata)
    manifest = train_marlin.write_run_manifest(
        config,
        "tokenizer-digest",
        metadata_csv_sha256=verified_digest,
    )

    assert manifest["inputs"]["metadata_csv"] == str(metadata)
    assert manifest["inputs"]["metadata_csv_sha256"] == digest


def test_tiny_overfit_config_pins_metadata_digest():
    tiny = OmegaConf.load(
        PROJECT_ROOT / "configs/marlin_frigid_distilled_tiny_overfit_c16h12o3.yaml"
    )

    assert (
        tiny.data.metadata_csv_sha256
        == "866a53ba9f390475e6dd086aa358313a50e1d56438002f90dde22effe868193b"
    )
