#!/usr/bin/env python
"""Fit the encoder error model, then validate the fit by sampling.

The fit is on the locked 803-spectrum test split and the validation is on the
396-row val split, both of which are *out of sample* for the DreaMS->Morgan
probe. The 6,748 train rows are deliberately excluded from the fit: the probe
saw them, and their Tanimoto is 0.444 against 0.293/0.304 out of sample, so
fitting on them would fit the leak rather than the encoder.

Validation is a two-sample KS test of the FULL Tanimoto distribution, sampled
from the model against the real held-out one, with the real fit-vs-held-out KS
printed beside it as the noise floor a fit cannot beat.

    env -u LD_PRELOAD PYTHONPATH=src .venv/bin/python \
        scripts/fit_encoder_error_model.py --output artefacts/encoder_error_model.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import ks_2samp

from marlin.encoder_error_model import (
    FINGERPRINT_BITS,
    EncoderErrorModel,
    fit_encoder_error_model,
)
from marlin.noise import one_sided_fingerprint_dropout, symmetric_fingerprint_noise

RDLogger.DisableLog("rdApp.*")

DEFAULT_BUNDLE = Path(
    "/mnt/netstorage/nikolenko/marlin/cache/runtime-inputs-spectrum-v1/"
    "16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3"
)


def tanimoto(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersection = (left & right).sum(1)
    union = (left | right).sum(1)
    return np.where(union > 0, intersection / np.maximum(union, 1), 0.0)


def true_fingerprints(smiles: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=FINGERPRINT_BITS
    )
    out = np.zeros((len(smiles), FINGERPRINT_BITS), dtype=bool)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"unparsable benchmark smiles at row {index}: {value}")
        out[index] = generator.GetFingerprintAsNumPy(molecule).astype(bool)
    return out


def load_split(bundle: Path, split: str, threshold: float):
    metadata = pd.read_csv(bundle / split / "metadata.csv")
    with np.load(bundle / split / "dreams_predictions.npz", allow_pickle=False) as z:
        probabilities = np.asarray(z["probs"])
        ids = [str(value) for value in z["spectrum_ids"]]
    if ids != [str(value) for value in metadata["spec_name"]]:
        raise ValueError(f"{split}: prediction rows are not aligned with metadata")
    return true_fingerprints(metadata["smiles"].tolist()), probabilities >= threshold


def describe(name: str, values: np.ndarray, reference: np.ndarray) -> dict:
    statistic = ks_2samp(values, reference)
    row = {
        "name": name,
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "sd": float(values.std()),
        "ks": float(statistic.statistic),
        "ks_p": float(statistic.pvalue),
    }
    print(
        f"{name:<34}{row['median']:>8.4f}{row['mean']:>8.4f}{row['q10']:>8.4f}"
        f"{row['q25']:>8.4f}{row['q75']:>8.4f}{row['q90']:>8.4f}{row['sd']:>8.4f}"
        f"{row['ks']:>9.4f}{row['ks_p']:>11.3g}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--corpus-frequency", type=Path, required=True,
                        help="npz with freq (4096,) and n, from audit_corpus_bit_frequency.py")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    # Which split the rates are fitted on is a correctness question, not a
    # convenience one: a model fitted on the panel a run is later scored on makes
    # that panel part of the fit, and the campaign forbids quoting a training-arm
    # gain against its own fit.
    parser.add_argument("--fit-split", default="test")
    parser.add_argument("--held-out-split", default="val")
    parser.add_argument("--name", default=None)
    arguments = parser.parse_args()

    import torch

    with np.load(arguments.corpus_frequency, allow_pickle=False) as z:
        frequency = np.asarray(z["freq"], dtype=np.float64) / float(z["n"])
        corpus_rows = int(z["n"])

    fit_true, fit_pred = load_split(
        arguments.bundle, arguments.fit_split, arguments.threshold
    )
    held_true, held_pred = load_split(
        arguments.bundle, arguments.held_out_split, arguments.threshold
    )
    print(
        f"fit on {arguments.fit_split} n={len(fit_true)}, "
        f"held out {arguments.held_out_split} n={len(held_true)}, "
        f"corpus frequency from {corpus_rows:,} molecules, threshold {arguments.threshold}"
    )

    model = fit_encoder_error_model(
        fit_true,
        fit_pred,
        frequency,
        bins=arguments.bins,
        name=arguments.name or f"dreams_nplib1_{arguments.fit_split}",
        extra_metadata={
            "threshold": arguments.threshold,
            "corpus_frequency_rows": corpus_rows,
            "fit_split": f"nplib1 {arguments.fit_split} ({len(fit_true)} rows)",
            "validation_split": f"nplib1 {arguments.held_out_split} ({len(held_true)} rows)",
        },
    )

    print("\n## Per-bin conditional rates (bins of corpus frequency, equal true-ON mass) ##")
    print(f"{'bin':>4}{'bits':>7}{'p_corpus_med':>15}{'sens':>9}{'fpr':>11}")
    for index in range(model.bins):
        print(
            f"{index:>4}{model.metadata['bin_bit_counts'][index]:>7}"
            f"{model.metadata['bin_median_corpus_frequency'][index]:>15.3e}"
            f"{model.sensitivity[index]:>9.4f}{model.false_positive_rate[index]:>11.6f}"
        )

    real_held = tanimoto(held_true, held_pred)
    print(
        f"\n{'distribution':<34}{'median':>8}{'mean':>8}{'q10':>8}{'q25':>8}"
        f"{'q75':>8}{'q90':>8}{'sd':>8}{'KS':>9}{'KS p':>11}"
    )
    rows = [describe(f"REAL {arguments.held_out_split} (held out)", real_held, real_held)]
    rows.append(
        describe(
            f"REAL {arguments.fit_split} (fit split)",
            tanimoto(fit_true, fit_pred),
            real_held,
        )
    )

    torch.manual_seed(arguments.seed)
    stacked = np.tile(held_true, (arguments.replicates, 1))
    truth = torch.from_numpy(stacked.astype(np.float32))

    sampled = model.corrupt(truth).numpy() > 0.5
    rows.append(describe("FITTED (frequency + latent)", tanimoto(stacked, sampled), real_held))

    flat = model.rate_matched_uniform_control()
    sampled_flat = flat.corrupt(truth).numpy() > 0.5
    rows.append(
        describe("CONTROL rate-matched uniform", tanimoto(stacked, sampled_flat), real_held)
    )

    incumbent = symmetric_fingerprint_noise(truth, corruption_probability=1.0).numpy() > 0.5
    rows.append(describe("INCUMBENT symmetric 0.1-0.3", tanimoto(stacked, incumbent), real_held))
    dropout = one_sided_fingerprint_dropout(truth, corruption_probability=1.0).numpy() > 0.5
    rows.append(describe("INCUMBENT dropout 0.1-0.3", tanimoto(stacked, dropout), real_held))

    model.save(arguments.output)
    flat.save(arguments.output.with_name(arguments.output.stem + "_control.npz"))
    print(f"\nwrote {arguments.output} and its rate-matched control")

    if arguments.report:
        arguments.report.write_text(
            json.dumps(
                {"rows": rows, "metadata": model.metadata,
                 "sensitivity": model.sensitivity.tolist(),
                 "false_positive_rate": model.false_positive_rate.tolist()},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
