"""Build leakage-safe candidate lists for RankLoop training and validation."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


ALLOWED_ATOMS = frozenset({"C", "O", "P", "N", "S", "Cl", "F", "H"})
FORBIDDEN_SOURCE_COLUMNS = frozenset(
    {
        "ground_truth",
        "is_positive",
        "label",
        "target_formula",
        "target_inchi_key",
        "target_smiles",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _git_revision(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


@dataclass(frozen=True)
class MoleculeRecord:
    smiles: str
    inchi_key_first_block: str
    provided_inchi_key_first_block: str
    formula: str
    scaffold: str
    scaffold_fallback: bool
    exact_mass: float
    fingerprint: np.ndarray

    @property
    def partition_group(self) -> str:
        return self.scaffold or self.inchi_key_first_block


@dataclass(frozen=True)
class SpectrumRecord:
    spec_name: str
    dataset_split: str
    ionization: str
    instrument: str
    molecule: MoleculeRecord


@dataclass(frozen=True)
class SourceCandidate:
    query_spec_name: str
    source_name: str
    source_rank: int
    molecule: MoleculeRecord


@dataclass(frozen=True)
class CandidateChoice:
    molecule: MoleculeRecord
    source_name: str
    source_rank: int
    negative_tier: str
    morgan_similarity: float


def molecule_record_from_smiles(
    smiles: str,
    *,
    formula: str | None = None,
    inchi_key: str | None = None,
    fingerprint_bits: int = 2048,
    fingerprint_radius: int = 2,
) -> MoleculeRecord | None:
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        return None
    if {atom.GetSymbol() for atom in molecule.GetAtoms()}.difference(ALLOWED_ATOMS):
        return None

    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    computed_inchi_key = Chem.MolToInchiKey(molecule)
    computed_first_block = computed_inchi_key.split("-", maxsplit=1)[0]
    provided_first_block = str(inchi_key or "").split("-", maxsplit=1)[0]
    inchi_key_first_block = computed_first_block
    if not inchi_key_first_block:
        return None

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=fingerprint_radius,
        fpSize=fingerprint_bits,
    )
    bit_vector = generator.GetFingerprint(molecule)
    fingerprint = np.zeros((fingerprint_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, fingerprint)
    scaffold_fallback = False
    try:
        scaffold_molecule = MurckoScaffold.GetScaffoldForMol(Chem.Mol(molecule))
        Chem.RemoveStereochemistry(scaffold_molecule)
        scaffold = Chem.MolToSmiles(
            scaffold_molecule,
            canonical=True,
            isomericSmiles=False,
        )
    except RuntimeError:
        scaffold = ""
        scaffold_fallback = True
    return MoleculeRecord(
        smiles=canonical_smiles,
        inchi_key_first_block=inchi_key_first_block,
        provided_inchi_key_first_block=provided_first_block,
        formula=str(formula or rdMolDescriptors.CalcMolFormula(molecule)),
        scaffold=scaffold,
        scaffold_fallback=scaffold_fallback,
        exact_mass=float(Descriptors.ExactMolWt(molecule)),
        fingerprint=fingerprint,
    )


def fingerprint_tanimoto(left: np.ndarray, right: np.ndarray) -> float:
    intersection = float(np.logical_and(left, right).sum())
    union = float(np.logical_or(left, right).sum())
    return intersection / union if union else 0.0


def load_spectrum_records(
    labels_tsv: str | Path,
    split_tsv: str | Path,
    *,
    dataset_split: str = "train",
    fingerprint_bits: int = 2048,
    fingerprint_radius: int = 2,
) -> tuple[list[SpectrumRecord], dict[str, int]]:
    labels = pd.read_csv(labels_tsv, sep="\t", dtype=str).fillna("")
    splits = pd.read_csv(split_tsv, sep="\t", dtype=str).fillna("")
    required_labels = {"spec", "formula", "smiles", "inchikey"}
    required_splits = {"name", "split"}
    if missing := sorted(required_labels.difference(labels.columns)):
        raise ValueError(f"Labels file is missing columns: {missing}")
    if missing := sorted(required_splits.difference(splits.columns)):
        raise ValueError(f"Split file is missing columns: {missing}")
    if labels["spec"].duplicated().any():
        duplicate = labels.loc[labels["spec"].duplicated(), "spec"].iloc[0]
        raise ValueError(f"Duplicate spectrum in labels file: {duplicate}")
    if splits["name"].duplicated().any():
        duplicate = splits.loc[splits["name"].duplicated(), "name"].iloc[0]
        raise ValueError(f"Duplicate spectrum in split file: {duplicate}")

    split_map = dict(splits[["name", "split"]].itertuples(index=False, name=None))
    molecule_cache: dict[tuple[str, str, str], MoleculeRecord | None] = {}
    records: list[SpectrumRecord] = []
    rejected = Counter()
    for row in labels.itertuples(index=False):
        spec_name = str(getattr(row, "spec")).strip()
        row_split = split_map.get(spec_name)
        if row_split != dataset_split:
            continue
        smiles = str(getattr(row, "smiles")).strip()
        formula = str(getattr(row, "formula")).strip()
        inchi_key = str(getattr(row, "inchikey")).strip()
        cache_key = (smiles, formula, inchi_key)
        if cache_key not in molecule_cache:
            molecule_cache[cache_key] = molecule_record_from_smiles(
                smiles,
                formula=formula,
                inchi_key=inchi_key,
                fingerprint_bits=fingerprint_bits,
                fingerprint_radius=fingerprint_radius,
            )
        molecule = molecule_cache[cache_key]
        if molecule is None:
            rejected["invalid_or_unsupported_molecule"] += 1
            continue
        if (
            molecule.provided_inchi_key_first_block
            and molecule.provided_inchi_key_first_block
            != molecule.inchi_key_first_block
        ):
            rejected["inchi_key_mismatch_rows"] += 1
        if molecule.scaffold_fallback:
            rejected["scaffold_fallback_rows"] += 1
        records.append(
            SpectrumRecord(
                spec_name=spec_name,
                dataset_split=row_split,
                ionization=str(getattr(row, "ionization", "")).strip(),
                instrument=str(getattr(row, "instrument", "")).strip(),
                molecule=molecule,
            )
        )
    if not records:
        raise ValueError(f"No valid spectra found for split {dataset_split!r}.")
    records.sort(key=lambda record: record.spec_name)
    return records, dict(rejected)


def split_partition_groups(
    records: Sequence[SpectrumRecord],
    *,
    development_fraction: float,
    seed: int,
) -> dict[str, str]:
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between zero and one.")
    groups = sorted({record.molecule.partition_group for record in records})
    if len(groups) < 2:
        raise ValueError("At least two scaffold/connectivity groups are required.")
    ordered = sorted(
        groups,
        key=lambda value: (_stable_digest(seed, f"partition:{value}"), value),
    )
    development_count = max(
        1, min(len(groups) - 1, round(len(groups) * development_fraction))
    )
    development = set(ordered[:development_count])
    return {
        group: "development" if group in development else "train" for group in groups
    }


def _resolve_source_columns(frame: pd.DataFrame) -> tuple[str, str, str | None]:
    lowered = {column.lower() for column in frame.columns}
    forbidden = sorted(
        column
        for column in lowered
        if column in FORBIDDEN_SOURCE_COLUMNS
        or column.startswith("target_")
        or "ground_truth" in column
    )
    if forbidden:
        raise ValueError(
            f"Candidate source contains forbidden supervision columns: {forbidden}"
        )
    query_column = next(
        (
            column
            for column in ("query_spec_name", "spec_name")
            if column in frame.columns
        ),
        None,
    )
    smiles_column = next(
        (
            column
            for column in ("candidate_smiles", "smiles")
            if column in frame.columns
        ),
        None,
    )
    if query_column is None or smiles_column is None:
        raise ValueError(
            "Candidate source requires query_spec_name/spec_name and candidate_smiles/smiles."
        )
    rank_column = "rank" if "rank" in frame.columns else None
    return query_column, smiles_column, rank_column


def load_candidate_source(
    source_name: str,
    source_path: str | Path,
    *,
    allowed_queries: set[str],
    fingerprint_bits: int,
    fingerprint_radius: int,
) -> tuple[dict[str, list[SourceCandidate]], dict[str, int]]:
    frame = pd.read_csv(source_path, dtype=str).fillna("")
    query_column, smiles_column, rank_column = _resolve_source_columns(frame)
    rows_by_query: dict[str, list[SourceCandidate]] = defaultdict(list)
    stats = Counter()
    molecule_cache: dict[str, MoleculeRecord | None] = {}
    for row_index, row in frame.iterrows():
        query = str(row[query_column]).strip()
        if query not in allowed_queries:
            stats["rows_outside_query_split"] += 1
            continue
        smiles = str(row[smiles_column]).strip()
        if smiles not in molecule_cache:
            molecule_cache[smiles] = molecule_record_from_smiles(
                smiles,
                fingerprint_bits=fingerprint_bits,
                fingerprint_radius=fingerprint_radius,
            )
        molecule = molecule_cache[smiles]
        if molecule is None:
            stats["invalid_or_unsupported_candidate"] += 1
            continue
        if molecule.scaffold_fallback:
            stats["scaffold_fallback_rows"] += 1
        if rank_column:
            try:
                rank = int(float(str(row[rank_column]).strip()))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid rank at row {row_index} in source {source_name!r}."
                ) from exc
        else:
            rank = row_index + 1
        if rank < 0:
            raise ValueError(
                f"Negative rank at row {row_index} in source {source_name!r}."
            )
        rows_by_query[query].append(
            SourceCandidate(
                query_spec_name=query,
                source_name=source_name,
                source_rank=rank,
                molecule=molecule,
            )
        )
        stats["accepted_rows"] += 1
    for query_rows in rows_by_query.values():
        query_rows.sort(
            key=lambda candidate: (
                candidate.source_rank,
                candidate.molecule.inchi_key_first_block,
                candidate.molecule.smiles,
            )
        )
    return dict(rows_by_query), dict(stats)


def _dedupe_choices(choices: Iterable[CandidateChoice]) -> list[CandidateChoice]:
    by_connectivity: dict[str, CandidateChoice] = {}
    for choice in choices:
        key = choice.molecule.inchi_key_first_block
        current = by_connectivity.get(key)
        ordering = (
            choice.source_rank,
            choice.source_name,
            choice.negative_tier,
            choice.molecule.smiles,
        )
        if current is None or ordering < (
            current.source_rank,
            current.source_name,
            current.negative_tier,
            current.molecule.smiles,
        ):
            by_connectivity[key] = choice
    return list(by_connectivity.values())


def _rank_library_candidates(
    query: MoleculeRecord,
    candidates: Iterable[MoleculeRecord],
    tier: str,
) -> list[CandidateChoice]:
    choices = []
    for rank, molecule in enumerate(candidates, start=1):
        if molecule.inchi_key_first_block == query.inchi_key_first_block:
            continue
        choices.append(
            CandidateChoice(
                molecule=molecule,
                source_name="train_library",
                source_rank=rank,
                negative_tier=tier,
                morgan_similarity=fingerprint_tanimoto(
                    query.fingerprint,
                    molecule.fingerprint,
                ),
            )
        )
    choices.sort(
        key=lambda choice: (
            -choice.morgan_similarity,
            abs(choice.molecule.exact_mass - query.exact_mass),
            choice.molecule.inchi_key_first_block,
        )
    )
    return choices


def choose_candidates(
    query: MoleculeRecord,
    *,
    source_candidates: Sequence[CandidateChoice],
    train_library: Sequence[MoleculeRecord],
    formula_index: dict[str, list[MoleculeRecord]],
    scaffold_index: dict[str, list[MoleculeRecord]],
    nominal_mass_index: dict[int, list[MoleculeRecord]],
    negatives_per_query: int,
    seed: int,
) -> list[CandidateChoice]:
    if negatives_per_query <= 0:
        raise ValueError("negatives_per_query must be positive.")
    source_choices = [
        candidate
        for candidate in source_candidates
        if candidate.molecule.inchi_key_first_block != query.inchi_key_first_block
    ]
    tiers: list[list[CandidateChoice]] = [
        sorted(
            source_choices,
            key=lambda choice: (
                choice.source_rank,
                -choice.morgan_similarity,
                choice.source_name,
                choice.molecule.inchi_key_first_block,
            ),
        ),
        _rank_library_candidates(
            query, formula_index.get(query.formula, ()), "formula_morgan"
        ),
    ]
    if query.scaffold:
        tiers.append(
            _rank_library_candidates(
                query,
                scaffold_index.get(query.scaffold, ()),
                "scaffold_morgan",
            )
        )
    tiers.append(
        _rank_library_candidates(
            query,
            nominal_mass_index.get(round(query.exact_mass), ()),
            "nominal_mass_morgan",
        )
    )

    selected: list[CandidateChoice] = []
    seen = {query.inchi_key_first_block}
    for tier_choices in tiers:
        for choice in tier_choices:
            connectivity = choice.molecule.inchi_key_first_block
            if connectivity in seen:
                continue
            seen.add(connectivity)
            selected.append(choice)
            if len(selected) == negatives_per_query:
                return selected

    rng = random.Random(
        int(
            _stable_digest(
                seed,
                f"random-negatives:{query.inchi_key_first_block}",
            )[:16],
            16,
        )
    )
    remaining = [
        molecule
        for molecule in train_library
        if molecule.inchi_key_first_block not in seen
    ]
    rng.shuffle(remaining)
    for rank, molecule in enumerate(remaining, start=1):
        selected.append(
            CandidateChoice(
                molecule=molecule,
                source_name="train_library",
                source_rank=rank,
                negative_tier="random",
                morgan_similarity=fingerprint_tanimoto(
                    query.fingerprint,
                    molecule.fingerprint,
                ),
            )
        )
        if len(selected) == negatives_per_query:
            break
    if len(selected) != negatives_per_query:
        raise ValueError(
            f"Only {len(selected)} negatives available for {query.inchi_key_first_block}; "
            f"requested {negatives_per_query}."
        )
    return selected


def build_rankloop_corpus(
    records: Sequence[SpectrumRecord],
    *,
    source_candidates: dict[str, list[SourceCandidate]] | None = None,
    development_fraction: float = 0.1,
    negatives_per_query: int = 32,
    max_spectra_per_molecule: int | None = 4,
    max_query_spectra: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    partition_by_group = split_partition_groups(
        records,
        development_fraction=development_fraction,
        seed=seed,
    )
    by_connectivity: dict[str, list[SpectrumRecord]] = defaultdict(list)
    molecules: dict[str, MoleculeRecord] = {}
    for record in records:
        by_connectivity[record.molecule.inchi_key_first_block].append(record)
        molecules.setdefault(record.molecule.inchi_key_first_block, record.molecule)

    selected_records: list[SpectrumRecord] = []
    for connectivity, molecule_records in sorted(by_connectivity.items()):
        molecule_records.sort(key=lambda record: record.spec_name)
        limit = max_spectra_per_molecule
        selected_records.extend(
            molecule_records if not limit else molecule_records[:limit]
        )
    selected_records.sort(
        key=lambda record: (
            _stable_digest(
                seed,
                f"query-sampling:{record.molecule.inchi_key_first_block}",
            ),
            record.spec_name,
        )
    )
    if max_query_spectra is not None:
        if max_query_spectra <= 0:
            raise ValueError("max_query_spectra must be positive.")
        selected_records = selected_records[:max_query_spectra]
    selected_records.sort(key=lambda record: record.spec_name)

    train_library = sorted(
        (
            molecule
            for molecule in molecules.values()
            if partition_by_group[molecule.partition_group] == "train"
        ),
        key=lambda molecule: molecule.inchi_key_first_block,
    )
    if len(train_library) <= negatives_per_query:
        raise ValueError(
            "Training molecule library is too small for the requested negatives."
        )
    formula_index: dict[str, list[MoleculeRecord]] = defaultdict(list)
    scaffold_index: dict[str, list[MoleculeRecord]] = defaultdict(list)
    nominal_mass_index: dict[int, list[MoleculeRecord]] = defaultdict(list)
    for molecule in train_library:
        formula_index[molecule.formula].append(molecule)
        if molecule.scaffold:
            scaffold_index[molecule.scaffold].append(molecule)
        nominal_mass_index[round(molecule.exact_mass)].append(molecule)

    source_candidates = source_candidates or {}
    candidate_cache: dict[str, list[CandidateChoice]] = {}
    rows: list[dict[str, object]] = []
    for record in selected_records:
        query = record.molecule
        query_sources = source_candidates.get(record.spec_name, ())
        source_choices = _dedupe_choices(
            CandidateChoice(
                molecule=candidate.molecule,
                source_name=candidate.source_name,
                source_rank=candidate.source_rank,
                negative_tier="source",
                morgan_similarity=fingerprint_tanimoto(
                    query.fingerprint,
                    candidate.molecule.fingerprint,
                ),
            )
            for candidate in query_sources
        )
        if source_choices:
            negative_choices = choose_candidates(
                query,
                source_candidates=source_choices,
                train_library=train_library,
                formula_index=formula_index,
                scaffold_index=scaffold_index,
                nominal_mass_index=nominal_mass_index,
                negatives_per_query=negatives_per_query,
                seed=seed,
            )
        else:
            if query.inchi_key_first_block not in candidate_cache:
                candidate_cache[query.inchi_key_first_block] = choose_candidates(
                    query,
                    source_candidates=(),
                    train_library=train_library,
                    formula_index=formula_index,
                    scaffold_index=scaffold_index,
                    nominal_mass_index=nominal_mass_index,
                    negatives_per_query=negatives_per_query,
                    seed=seed,
                )
            negative_choices = candidate_cache[query.inchi_key_first_block]

        positive_source = min(
            (
                candidate
                for candidate in query_sources
                if candidate.molecule.inchi_key_first_block
                == query.inchi_key_first_block
            ),
            key=lambda candidate: (
                candidate.source_rank,
                candidate.source_name,
            ),
            default=None,
        )
        partition = partition_by_group[query.partition_group]
        choices = [
            CandidateChoice(
                molecule=query,
                source_name=(
                    positive_source.source_name if positive_source else "supervision"
                ),
                source_rank=(positive_source.source_rank if positive_source else 0),
                negative_tier=("positive_source" if positive_source else "positive"),
                morgan_similarity=1.0,
            ),
            *negative_choices,
        ]
        random.Random(
            int(
                _stable_digest(seed, f"candidate-order:{record.spec_name}")[:16],
                16,
            )
        ).shuffle(choices)
        for candidate_rank, choice in enumerate(choices, start=1):
            candidate = choice.molecule
            rows.append(
                {
                    "query_spec_name": record.spec_name,
                    "rankloop_split": partition,
                    "query_formula": query.formula,
                    "query_inchi_key_first_block": query.inchi_key_first_block,
                    "query_label_inchi_key_first_block": (
                        query.provided_inchi_key_first_block
                    ),
                    "query_scaffold": query.scaffold,
                    "query_scaffold_fallback": int(query.scaffold_fallback),
                    "ionization": record.ionization,
                    "instrument": record.instrument,
                    "candidate_rank": candidate_rank,
                    "candidate_smiles": candidate.smiles,
                    "candidate_inchi_key_first_block": candidate.inchi_key_first_block,
                    "candidate_formula": candidate.formula,
                    "candidate_scaffold": candidate.scaffold,
                    "candidate_scaffold_fallback": int(candidate.scaffold_fallback),
                    "source_name": choice.source_name,
                    "source_rank": choice.source_rank,
                    "negative_tier": choice.negative_tier,
                    "morgan_similarity": choice.morgan_similarity,
                    "formula_match": int(candidate.formula == query.formula),
                    "label": int(
                        candidate.inchi_key_first_block == query.inchi_key_first_block
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    if set(frame["rankloop_split"]) != {"train", "development"}:
        raise ValueError(
            "Selected query spectra must include both train and development partitions."
        )
    frame.sort_values(
        ["rankloop_split", "query_spec_name", "candidate_rank"],
        inplace=True,
        ignore_index=True,
    )
    positive_counts = frame.groupby("query_spec_name", sort=False)["label"].sum()
    if not positive_counts.eq(1).all():
        raise AssertionError("Every query must have exactly one positive candidate.")

    split_scaffolds: dict[str, set[str]] = {}
    split_connectivity: dict[str, set[str]] = {}
    for partition, partition_frame in frame.groupby("rankloop_split"):
        split_scaffolds[str(partition)] = set(partition_frame["query_scaffold"]) - {""}
        split_connectivity[str(partition)] = set(
            partition_frame["query_inchi_key_first_block"]
        )
    scaffold_overlap = split_scaffolds.get("train", set()).intersection(
        split_scaffolds.get("development", set())
    )
    connectivity_overlap = split_connectivity.get("train", set()).intersection(
        split_connectivity.get("development", set())
    )
    if scaffold_overlap or connectivity_overlap:
        raise AssertionError("RankLoop train/development molecule groups overlap.")

    report: dict[str, object] = {
        "schema_version": 1,
        "query_count": int(frame["query_spec_name"].nunique()),
        "query_molecule_count": int(frame["query_inchi_key_first_block"].nunique()),
        "candidate_row_count": int(len(frame)),
        "negatives_per_query": negatives_per_query,
        "partition_query_counts": {
            str(key): int(value)
            for key, value in frame.groupby("rankloop_split")["query_spec_name"]
            .nunique()
            .items()
        },
        "negative_tier_counts": {
            str(key): int(value)
            for key, value in frame["negative_tier"].value_counts().items()
        },
        "source_counts": {
            str(key): int(value)
            for key, value in frame["source_name"].value_counts().items()
        },
        "formula_match_rate": float(
            frame.loc[frame["label"] == 0, "formula_match"].mean()
        ),
        "train_library_molecule_count": len(train_library),
        "scaffold_overlap_count": len(scaffold_overlap),
        "connectivity_overlap_count": len(connectivity_overlap),
    }
    return frame, report


def write_rankloop_corpus(
    frame: pd.DataFrame,
    report: dict[str, object],
    *,
    output_dir: str | Path,
    labels_tsv: str | Path,
    split_tsv: str | Path,
    source_paths: Sequence[tuple[str, str | Path]],
    parameters: dict[str, object],
    repo_root: str | Path,
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    corpus_path = output_path / "candidate_corpus.csv"
    query_manifest_path = output_path / "query_manifest.tsv"
    quality_path = output_path / "quality_report.json"
    manifest_path = output_path / "run_manifest.json"
    frame.to_csv(corpus_path, index=False, float_format="%.8g")
    query_manifest = (
        frame[["query_spec_name", "rankloop_split"]]
        .drop_duplicates("query_spec_name", keep="first")
        .rename(columns={"query_spec_name": "spec_name"})
    )
    query_manifest.to_csv(query_manifest_path, sep="\t", index=False)

    final_report = dict(report)
    final_report["candidate_corpus_sha256"] = sha256_file(corpus_path)
    quality_path.write_text(
        json.dumps(final_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    revision, dirty = _git_revision(Path(repo_root))
    manifest = {
        "schema_version": 1,
        "repo": {"commit": revision, "dirty": dirty},
        "inputs": {
            "labels_tsv": {
                "path": str(Path(labels_tsv).resolve()),
                "sha256": sha256_file(labels_tsv),
            },
            "split_tsv": {
                "path": str(Path(split_tsv).resolve()),
                "sha256": sha256_file(split_tsv),
            },
            "candidate_sources": [
                {
                    "name": name,
                    "path": str(Path(path).resolve()),
                    "sha256": sha256_file(path),
                }
                for name, path in source_paths
            ],
        },
        "parameters": parameters,
        "outputs": {
            "candidate_corpus_csv": str(corpus_path.resolve()),
            "candidate_corpus_sha256": final_report["candidate_corpus_sha256"],
            "query_manifest_tsv": str(query_manifest_path.resolve()),
            "query_manifest_sha256": sha256_file(query_manifest_path),
            "quality_report_json": str(quality_path.resolve()),
            "quality_report_sha256": sha256_file(quality_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
