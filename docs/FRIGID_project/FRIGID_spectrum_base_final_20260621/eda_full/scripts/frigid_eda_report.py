#!/usr/bin/env python3
"""Generate a full EDA report for the FRIGID MSG base benchmark output."""

from __future__ import annotations

import html
import json
import math
import re
import sys
from pathlib import Path

PYDEPS = Path("/tmp/frigid_eda_pydeps")
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RUN_ROOT = Path("/Users/nikolenko/Desktop/FRIGID_spectrum_base_final_20260621/msg_base_full_ngboost")
OUT_ROOT = Path("/Users/nikolenko/Desktop/FRIGID_spectrum_base_final_20260621/eda_full")
TABLE_DIR = OUT_ROOT / "tables"
CHART_DIR = OUT_ROOT / "charts"

PAPER_TARGETS = {
    "exact_match_top1": 0.1609,
    "exact_match_top10": 0.1819,
}

NUMERIC_METRICS = [
    "exact_match_top1",
    "exact_match_top10",
    "tanimoto_top1",
    "tanimoto_top10",
    "tanimoto_mean",
    "mist_tanimoto",
    "num_predictions",
    "formula_matches_collected",
    "total_generated",
    "total_valid",
    "total_formula_matched",
    "generation_time",
    "valid_rate",
    "formula_match_rate_generated",
    "target_smiles_len",
    "target_atom_token_count",
    "target_chiral_markers",
    "target_ring_digit_count",
]


def mkdirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict], pd.DataFrame]:
    aggregate_dir = RUN_ROOT / "aggregate"
    detailed = pd.read_csv(aggregate_dir / "detailed_results.csv")
    predictions = pd.read_csv(aggregate_dir / "predictions.csv")
    stats = json.loads((aggregate_dir / "aggregate_statistics.json").read_text())
    completed = json.loads((aggregate_dir / "completed_shards.json").read_text())

    shard_frames: list[pd.DataFrame] = []
    for path in sorted((RUN_ROOT / "shard_outputs").glob("shard_*/detailed_results.csv")):
        shard = path.parent.name
        frame = pd.read_csv(path)
        frame.insert(0, "shard", shard)
        frame.insert(1, "shard_index", int(shard.split("_")[1]))
        shard_frames.append(frame)
    sharded = pd.concat(shard_frames, ignore_index=True) if shard_frames else detailed.copy()
    return detailed, predictions, stats, completed, sharded


def extract_spec_id(spec_name: str) -> float:
    match = re.search(r"(\d+)$", str(spec_name))
    return float(match.group(1)) if match else np.nan


ATOM_PATTERN = re.compile(r"Cl|Br|Si|Se|Na|Li|Mg|Ca|Al|[BCNOFPSIHK]|[cnops]")


def atom_token_count(smiles: str) -> int:
    if not isinstance(smiles, str):
        return 0
    return len(ATOM_PATTERN.findall(smiles))


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spec_id"] = df["spec_name"].map(extract_spec_id)
    df["top1_hit"] = df["exact_match_top1"] > 0
    df["top10_hit"] = df["exact_match_top10"] > 0
    df["top10_only_hit"] = (~df["top1_hit"]) & df["top10_hit"]
    df["formula_success"] = df["total_formula_matched"] > 0
    df["valid_rate"] = np.where(df["total_generated"] > 0, df["total_valid"] / df["total_generated"], np.nan)
    df["formula_match_rate_generated"] = np.where(
        df["total_generated"] > 0,
        df["total_formula_matched"] / df["total_generated"],
        np.nan,
    )
    df["seconds_per_generated"] = np.where(
        df["total_generated"] > 0,
        df["generation_time"] / df["total_generated"],
        np.nan,
    )
    df["target_smiles_len"] = df["target_smiles"].astype(str).str.len()
    df["proposal_smiles_len"] = df["proposal_smiles"].astype(str).str.len()
    df["target_atom_token_count"] = df["target_smiles"].map(atom_token_count)
    df["proposal_atom_token_count"] = df["proposal_smiles"].map(atom_token_count)
    df["target_chiral_markers"] = df["target_smiles"].astype(str).str.count("@")
    df["target_ring_digit_count"] = df["target_smiles"].astype(str).str.count(r"\d")

    df["mist_bin"] = pd.cut(
        df["mist_tanimoto"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.0001],
        labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    )
    df["tanimoto_top10_bin"] = pd.cut(
        df["tanimoto_top10"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.0001],
        labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    )
    df["formula_actual_bin"] = pd.cut(
        df["total_formula_matched"],
        bins=[-0.1, 0, 2, 5, 9, np.inf],
        labels=["0", "1-2", "3-5", "6-9", "10+"],
    )
    df["target_length_bin"] = pd.qcut(
        df["target_smiles_len"],
        q=5,
        duplicates="drop",
    ).astype(str)

    conditions = [
        df["top1_hit"],
        df["top10_only_hit"],
        (~df["top10_hit"]) & (df["tanimoto_top10"] >= 0.8),
        (~df["top10_hit"]) & (df["tanimoto_top10"] >= 0.5),
        (~df["top10_hit"]) & (df["tanimoto_top10"] < 0.5),
    ]
    choices = [
        "Exact Top-1",
        "Exact Top-10 only",
        "No exact, high Tanimoto >=0.8",
        "No exact, medium Tanimoto 0.5-0.8",
        "No exact, low Tanimoto <0.5",
    ]
    df["outcome_bucket"] = np.select(conditions, choices, default="Other")
    return df


def percent(x: float) -> str:
    return f"{100 * x:.2f}%"


def f4(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.4f}"


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=index)
    return path


def distribution_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for col in metrics:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append(
            {
                "metric": col,
                "count": int(s.count()),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p05": s.quantile(0.05),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "p95": s.quantile(0.95),
                "max": s.max(),
            }
        )
    return pd.DataFrame(rows)


def grouped_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=False, dropna=False)
    out = g.agg(
        spectra=("spec_name", "count"),
        exact_top1=("exact_match_top1", "mean"),
        exact_top10=("exact_match_top10", "mean"),
        tanimoto_top1=("tanimoto_top1", "mean"),
        tanimoto_top10=("tanimoto_top10", "mean"),
        mist_tanimoto=("mist_tanimoto", "mean"),
        formula_success=("formula_success", "mean"),
        avg_actual_formula_matches=("total_formula_matched", "mean"),
        avg_generated=("total_generated", "mean"),
        avg_generation_time_s=("generation_time", "mean"),
        median_target_smiles_len=("target_smiles_len", "median"),
    ).reset_index()
    return out


def make_tables(
    detailed: pd.DataFrame,
    predictions: pd.DataFrame,
    stats: dict,
    completed: list[dict],
    df: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    recomputed = {
        "total_spectra": len(df),
        "exact_match_top1": df["exact_match_top1"].mean(),
        "exact_match_top10": df["exact_match_top10"].mean(),
        "tanimoto_top1_mean": df["tanimoto_top1"].mean(),
        "tanimoto_top10_mean": df["tanimoto_top10"].mean(),
        "mist_tanimoto_mean": df["mist_tanimoto"].mean(),
        "avg_formula_matches": df["total_formula_matched"].mean(),
        "avg_predictions_collected": df["formula_matches_collected"].mean(),
        "avg_total_generated": df["total_generated"].mean(),
        "formula_match_success_rate": df["formula_success"].mean(),
    }

    headline_rows = []
    for metric, value in recomputed.items():
        row = {
            "metric": metric,
            "recomputed_from_shard_csv": value,
            "aggregate_json": stats.get(metric),
            "abs_diff_recomputed_vs_json": (
                abs(value - stats.get(metric)) if isinstance(value, (int, float)) and isinstance(stats.get(metric), (int, float)) else None
            ),
            "paper_target": PAPER_TARGETS.get(metric.replace("_mean", "")),
            "delta_vs_paper": None,
        }
        if metric in PAPER_TARGETS:
            row["paper_target"] = PAPER_TARGETS[metric]
            row["delta_vs_paper"] = value - PAPER_TARGETS[metric]
        headline_rows.append(row)
    headline = pd.DataFrame(headline_rows)
    paths["headline_metrics"] = save_table(headline, "headline_metrics.csv")

    data_quality = pd.DataFrame(
        [
            {"check": "aggregate_detailed_rows", "value": len(detailed), "status": "ok" if len(detailed) == 17082 else "review"},
            {"check": "sharded_detailed_rows", "value": len(df), "status": "ok" if len(df) == 17082 else "review"},
            {"check": "prediction_rows", "value": len(predictions), "status": "ok" if len(predictions) == len(detailed) else "review"},
            {"check": "unique_spec_names", "value": df["spec_name"].nunique(), "status": "ok" if df["spec_name"].is_unique else "review"},
            {
                "check": "prediction_name_set_matches_detailed_spec_set",
                "value": str(set(predictions["name"]) == set(detailed["spec_name"])),
                "status": "ok" if set(predictions["name"]) == set(detailed["spec_name"]) else "review",
            },
            {
                "check": "completed_shard_count",
                "value": len(completed),
                "status": "ok" if len(completed) == 36 else "review",
            },
            {
                "check": "all_rows_have_10_predictions_recorded",
                "value": int((df["num_predictions"] == 10).sum()),
                "status": "ok" if int((df["num_predictions"] == 10).sum()) == len(df) else "review",
            },
            {
                "check": "all_rows_have_formula_proposal_source",
                "value": int((df["proposal_source"] == "formula").sum()),
                "status": "ok" if int((df["proposal_source"] == "formula").sum()) == len(df) else "review",
            },
        ]
    )
    paths["data_quality_checks"] = save_table(data_quality, "data_quality_checks.csv")

    missing = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )
    missing["missing_rate"] = missing["missing_count"] / len(df)
    paths["missing_values"] = save_table(missing, "missing_values.csv")

    dist = distribution_table(df, NUMERIC_METRICS)
    paths["metric_distributions"] = save_table(dist, "metric_distributions.csv")

    paths["by_mist_bin"] = save_table(grouped_summary(df, "mist_bin"), "by_mist_bin.csv")
    paths["by_formula_actual_bin"] = save_table(grouped_summary(df, "formula_actual_bin"), "by_formula_actual_bin.csv")
    paths["by_target_length_bin"] = save_table(grouped_summary(df, "target_length_bin"), "by_target_length_bin.csv")
    paths["by_tanimoto_top10_bin"] = save_table(grouped_summary(df, "tanimoto_top10_bin"), "by_tanimoto_top10_bin.csv")

    by_shard = grouped_summary(df, "shard")
    by_shard["shard_index"] = by_shard["shard"].str.extract(r"(\d+)").astype(int)
    by_shard = by_shard.sort_values("shard_index")
    paths["by_shard"] = save_table(by_shard, "by_shard.csv")
    paths["worst_shards"] = save_table(by_shard.sort_values(["exact_top1", "exact_top10", "tanimoto_top10"]).head(12), "worst_shards.csv")
    paths["best_shards"] = save_table(by_shard.sort_values(["exact_top1", "exact_top10", "tanimoto_top10"], ascending=False).head(12), "best_shards.csv")

    outcome = (
        df.groupby("outcome_bucket", observed=False)
        .agg(
            spectra=("spec_name", "count"),
            share=("spec_name", lambda s: len(s) / len(df)),
            avg_tanimoto_top10=("tanimoto_top10", "mean"),
            avg_mist_tanimoto=("mist_tanimoto", "mean"),
            formula_success=("formula_success", "mean"),
            median_target_smiles_len=("target_smiles_len", "median"),
        )
        .reset_index()
        .sort_values("spectra", ascending=False)
    )
    paths["outcome_breakdown"] = save_table(outcome, "outcome_breakdown.csv")

    corr_cols = [
        "exact_match_top1",
        "exact_match_top10",
        "tanimoto_top1",
        "tanimoto_top10",
        "mist_tanimoto",
        "formula_success",
        "total_formula_matched",
        "total_generated",
        "generation_time",
        "target_smiles_len",
        "target_atom_token_count",
        "target_chiral_markers",
        "target_ring_digit_count",
    ]
    corr_input = df[corr_cols].copy()
    corr_input["formula_success"] = corr_input["formula_success"].astype(float)
    rows = []
    for col in corr_cols:
        if col in {"tanimoto_top10", "exact_match_top1", "exact_match_top10"}:
            continue
        rows.append(
            {
                "feature": col,
                "pearson_with_tanimoto_top10": corr_input[col].corr(corr_input["tanimoto_top10"], method="pearson"),
                "spearman_with_tanimoto_top10": corr_input[col].corr(corr_input["tanimoto_top10"], method="spearman"),
                "pearson_with_exact_top1": corr_input[col].corr(corr_input["exact_match_top1"], method="pearson"),
                "spearman_with_exact_top1": corr_input[col].corr(corr_input["exact_match_top1"], method="spearman"),
                "pearson_with_exact_top10": corr_input[col].corr(corr_input["exact_match_top10"], method="pearson"),
                "spearman_with_exact_top10": corr_input[col].corr(corr_input["exact_match_top10"], method="spearman"),
            }
        )
    paths["correlations"] = save_table(pd.DataFrame(rows), "correlations.csv")

    completed_rows = []
    for item in completed:
        row = {"shard": item["shard"], "rows": item.get("rows")}
        row.update(item.get("aggregate", {}))
        completed_rows.append(row)
    runtime = pd.DataFrame(completed_rows)
    runtime["shard_index"] = runtime["shard"].str.extract(r"(\d+)").astype(int)
    runtime = runtime.sort_values("shard_index")
    paths["runtime_by_shard"] = save_table(runtime, "runtime_by_shard.csv")

    unique_targets = (
        df.groupby("target_inchi_key")
        .agg(
            spectra=("spec_name", "count"),
            exact_top1=("exact_match_top1", "mean"),
            exact_top10=("exact_match_top10", "mean"),
            tanimoto_top10=("tanimoto_top10", "mean"),
            mist_tanimoto=("mist_tanimoto", "mean"),
            median_target_smiles_len=("target_smiles_len", "median"),
            example_smiles=("target_smiles", "first"),
        )
        .reset_index()
        .sort_values("spectra", ascending=False)
    )
    paths["by_target_molecule"] = save_table(unique_targets, "by_target_molecule.csv")

    examples = []
    example_specs = [
        ("top1_exact_examples", df[df["top1_hit"]].sort_values("tanimoto_top1", ascending=False).head(10)),
        ("top10_only_examples", df[df["top10_only_hit"]].sort_values("tanimoto_top10", ascending=False).head(10)),
        (
            "near_miss_high_tanimoto_no_exact",
            df[(~df["top10_hit"]) & (df["tanimoto_top10"] >= 0.8)].sort_values("tanimoto_top10", ascending=False).head(10),
        ),
        (
            "bad_low_mist_low_tanimoto",
            df[(~df["top10_hit"]) & (df["mist_tanimoto"] < 0.4) & (df["tanimoto_top10"] < 0.4)].sort_values("mist_tanimoto").head(10),
        ),
        (
            "bad_high_mist_but_no_exact",
            df[(~df["top10_hit"]) & (df["mist_tanimoto"] >= 0.8)].sort_values("tanimoto_top10").head(10),
        ),
    ]
    keep = [
        "example_type",
        "shard",
        "spec_name",
        "target_inchi_key",
        "exact_match_top1",
        "exact_match_top10",
        "tanimoto_top1",
        "tanimoto_top10",
        "mist_tanimoto",
        "total_formula_matched",
        "total_generated",
        "generation_time",
        "target_smiles",
        "proposal_smiles",
        "top_prediction_smiles",
    ]
    for name, part in example_specs:
        tmp = part.copy()
        tmp.insert(0, "example_type", name)
        examples.append(tmp[keep])
    paths["example_cases"] = save_table(pd.concat(examples, ignore_index=True), "example_cases.csv")

    return paths


def savefig(name: str) -> Path:
    path = CHART_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path


def make_charts(df: pd.DataFrame, tables: dict[str, Path]) -> dict[str, Path]:
    sns.set_theme(style="whitegrid", context="notebook")
    charts: dict[str, Path] = {}

    headline = pd.read_csv(tables["headline_metrics"])
    comp = headline[headline["metric"].isin(["exact_match_top1", "exact_match_top10"])].copy()
    comp["Metric"] = comp["metric"].map(
        {
            "exact_match_top1": "Exact Top-1",
            "exact_match_top10": "Exact Top-10",
        }
    )
    comp = comp.rename(
        columns={
            "recomputed_from_shard_csv": "Reproduced run",
            "paper_target": "Paper target",
        }
    )
    plot_df = comp.melt(
        id_vars=["Metric"],
        value_vars=["Reproduced run", "Paper target"],
        var_name="Series",
        value_name="value",
    )
    plt.figure(figsize=(8.5, 5.2))
    ax = sns.barplot(data=plot_df, x="Metric", y="value", hue="Series", palette=["#3b82f6", "#ef4444"])
    ax.set_title("Exact match is below the paper target")
    ax.set_ylabel("Rate")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(title="", loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=9)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v.get_height():.1%}" for v in container], fontsize=10)
    charts["headline_vs_paper"] = savefig("headline_vs_paper.png")

    by_shard = pd.read_csv(tables["by_shard"])
    plt.figure(figsize=(13, 6))
    for col, label in [
        ("exact_top1", "Exact Top-1"),
        ("exact_top10", "Exact Top-10"),
        ("tanimoto_top10", "Tanimoto Top-10"),
        ("mist_tanimoto", "MIST Tanimoto"),
        ("formula_success", "Formula success"),
    ]:
        sns.lineplot(data=by_shard, x="shard_index", y=col, marker="o", label=label)
    plt.title("Shard-level quality reveals a weak tail")
    plt.xlabel("Shard index")
    plt.ylabel("Mean / rate")
    plt.ylim(0, 1)
    charts["shard_metrics"] = savefig("shard_metrics.png")

    heat_cols = ["exact_top1", "exact_top10", "tanimoto_top10", "mist_tanimoto", "formula_success"]
    heat = by_shard.set_index("shard")[heat_cols].T
    plt.figure(figsize=(15, 4.5))
    ax = sns.heatmap(heat, cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "rate / mean"})
    ax.set_title("Shard metric heatmap")
    ax.set_xlabel("Shard")
    ax.set_ylabel("")
    charts["shard_heatmap"] = savefig("shard_heatmap.png")

    sample = df.sample(min(len(df), 7000), random_state=17)
    plt.figure(figsize=(8, 7))
    ax = sns.scatterplot(
        data=sample,
        x="mist_tanimoto",
        y="tanimoto_top10",
        hue="top10_hit",
        alpha=0.35,
        s=18,
        palette={False: "#64748b", True: "#16a34a"},
    )
    ax.set_title("MIST fingerprint quality tracks final candidate quality")
    ax.set_xlabel("MIST Tanimoto")
    ax.set_ylabel("Best generated Tanimoto in Top-10")
    charts["mist_vs_tanimoto_scatter"] = savefig("mist_vs_tanimoto_scatter.png")

    plt.figure(figsize=(10, 5))
    sns.histplot(df["mist_tanimoto"], bins=40, stat="density", color="#2563eb", alpha=0.45, label="MIST Tanimoto")
    sns.histplot(df["tanimoto_top10"], bins=40, stat="density", color="#f97316", alpha=0.45, label="Top-10 Tanimoto")
    plt.title("Distribution of encoder and final candidate similarity")
    plt.xlabel("Tanimoto")
    plt.ylabel("Density")
    plt.legend()
    charts["similarity_distributions"] = savefig("similarity_distributions.png")

    by_mist = pd.read_csv(tables["by_mist_bin"])
    plot_cols = ["exact_top1", "exact_top10", "tanimoto_top10", "formula_success"]
    mist_plot = by_mist.melt(id_vars=["mist_bin", "spectra"], value_vars=plot_cols, var_name="metric", value_name="value")
    mist_plot["Metric"] = mist_plot["metric"].map(
        {
            "exact_top1": "Exact Top-1",
            "exact_top10": "Exact Top-10",
            "tanimoto_top10": "Tanimoto Top-10",
            "formula_success": "Formula success",
        }
    )
    plt.figure(figsize=(11.5, 6))
    ax = sns.barplot(data=mist_plot, x="mist_bin", y="value", hue="Metric")
    ax.set_title("Exact recovery rises sharply with MIST quality")
    ax.set_xlabel("MIST Tanimoto bin")
    ax.set_ylabel("Rate / mean")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=9)
    charts["exact_by_mist_bin"] = savefig("exact_by_mist_bin.png")

    plt.figure(figsize=(10, 5))
    sns.histplot(df["total_formula_matched"], bins=range(0, int(df["total_formula_matched"].max()) + 2), color="#0891b2")
    plt.title("Actual formula matches per spectrum")
    plt.xlabel("Formula-matched generations out of up to 100 attempts")
    plt.ylabel("Spectra")
    charts["formula_matches_distribution"] = savefig("formula_matches_distribution.png")

    outcome = pd.read_csv(tables["outcome_breakdown"])
    plt.figure(figsize=(11, 5))
    ax = sns.barplot(data=outcome, x="share", y="outcome_bucket", color="#7c3aed")
    ax.set_title("Most spectra are misses, but many remain chemically close")
    ax.set_xlabel("Share of spectra")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v.get_width():.1%}" for v in container], fontsize=9)
    charts["outcome_breakdown"] = savefig("outcome_breakdown.png")

    runtime = pd.read_csv(tables["runtime_by_shard"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["generation_time"], bins=50, ax=axes[0], color="#475569")
    axes[0].set_title("Per-spectrum generation time")
    axes[0].set_xlabel("Seconds")
    axes[0].set_ylabel("Spectra")
    sns.lineplot(data=runtime, x="shard_index", y="spectra_per_second", marker="o", ax=axes[1], color="#dc2626")
    axes[1].set_title("Throughput by shard")
    axes[1].set_xlabel("Shard index")
    axes[1].set_ylabel("Spectra per second")
    charts["runtime"] = savefig("runtime.png")

    by_len = pd.read_csv(tables["by_target_length_bin"])
    len_plot = by_len.melt(
        id_vars=["target_length_bin", "spectra"],
        value_vars=["exact_top1", "exact_top10", "tanimoto_top10", "mist_tanimoto"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(data=len_plot, x="target_length_bin", y="value", hue="metric", marker="o")
    ax.set_title("Longer SMILES targets are harder by exact match")
    ax.set_xlabel("Target SMILES length quintile")
    ax.set_ylabel("Rate / mean")
    plt.xticks(rotation=30, ha="right")
    charts["quality_by_target_length"] = savefig("quality_by_target_length.png")

    return charts


def table_html(path: Path, rows: int = 10, floatfmt: str = "{:.4f}") -> str:
    df = pd.read_csv(path).head(rows)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))
    return df.to_html(index=False, escape=True, classes="data-table")


def img_html(path: Path, alt: str) -> str:
    rel = path.relative_to(OUT_ROOT)
    return f'<figure><img src="{html.escape(str(rel))}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def build_report(
    df: pd.DataFrame,
    stats: dict,
    tables: dict[str, Path],
    charts: dict[str, Path],
) -> Path:
    headline = pd.read_csv(tables["headline_metrics"])
    by_mist = pd.read_csv(tables["by_mist_bin"])
    by_shard = pd.read_csv(tables["by_shard"])
    runtime = pd.read_csv(tables["runtime_by_shard"])
    corr = pd.read_csv(tables["correlations"])
    outcome = pd.read_csv(tables["outcome_breakdown"])
    dist = pd.read_csv(tables["metric_distributions"])

    top1 = df["exact_match_top1"].mean()
    top10 = df["exact_match_top10"].mean()
    tan10 = df["tanimoto_top10"].mean()
    mist = df["mist_tanimoto"].mean()
    formula_success = df["formula_success"].mean()
    delta_top1 = top1 - PAPER_TARGETS["exact_match_top1"]
    delta_top10 = top10 - PAPER_TARGETS["exact_match_top10"]
    top10_lift = top10 - top1
    unique_mols = df["target_inchi_key"].nunique()
    total_elapsed_h = runtime["elapsed_time_seconds"].sum() / 3600
    total_generation_h = runtime["total_generation_time"].sum() / 3600
    gen_share = runtime["generation_time_percentage"].mean() / 100

    corr_mist_tan = corr.loc[corr["feature"] == "mist_tanimoto", "spearman_with_tanimoto_top10"].iloc[0]
    corr_mist_exact = corr.loc[corr["feature"] == "mist_tanimoto", "spearman_with_exact_top10"].iloc[0]
    worst_shards = pd.read_csv(tables["worst_shards"]).head(5)
    best_shards = pd.read_csv(tables["best_shards"]).head(5)

    low_mist = by_mist[by_mist["mist_bin"].isin(["0.0-0.2", "0.2-0.4"])]
    high_mist = by_mist[by_mist["mist_bin"].isin(["0.8-1.0"])]
    low_mist_spectra = int(low_mist["spectra"].sum())
    high_mist_top10 = float(high_mist["exact_top10"].iloc[0]) if len(high_mist) else np.nan
    low_mist_top10 = (
        float(np.average(low_mist["exact_top10"], weights=low_mist["spectra"])) if low_mist_spectra else np.nan
    )

    zero_formula = int((df["total_formula_matched"] == 0).sum())
    target_len_corr = corr.loc[corr["feature"] == "target_smiles_len", "spearman_with_exact_top10"].iloc[0]

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f8fafc; color: #111827; }
    main { max-width: 1180px; margin: 0 auto; padding: 36px 28px 80px; }
    h1 { font-size: 34px; margin: 0 0 8px; }
    h2 { margin-top: 42px; font-size: 24px; border-top: 1px solid #d1d5db; padding-top: 28px; }
    h3 { margin-top: 28px; font-size: 18px; }
    p, li { line-height: 1.55; font-size: 15px; }
    .subtitle { color: #4b5563; margin-bottom: 26px; }
    .kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 24px 0; }
    .kpi { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 16px; }
    .kpi .label { color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .kpi .value { font-size: 24px; font-weight: 700; margin-top: 6px; }
    .kpi .note { color: #6b7280; font-size: 12px; margin-top: 4px; }
    figure { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; margin: 18px 0; }
    figure img { max-width: 100%; display: block; margin: 0 auto; }
    figcaption { color: #6b7280; font-size: 12px; text-align: center; margin-top: 8px; }
    .data-table { border-collapse: collapse; width: 100%; background: #ffffff; font-size: 12px; margin: 14px 0 24px; }
    .data-table th, .data-table td { border: 1px solid #e5e7eb; padding: 7px 8px; vertical-align: top; }
    .data-table th { background: #f1f5f9; text-align: left; }
    .callout { background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 16px; margin: 18px 0; }
    code { background: #eef2ff; padding: 1px 4px; border-radius: 4px; }
    @media (max-width: 850px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    """

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FRIGID MSG Base Reproduction EDA</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>FRIGID MSG Base Reproduction EDA</h1>
  <p class="subtitle">Technical EDA for the full MSG FRIGID-base NGBoost run copied to Desktop. Source run: <code>{html.escape(str(RUN_ROOT))}</code>.</p>

  <h2>Technical summary</h2>
  <p><strong>The full run is internally consistent but does not reproduce the paper-level exact-match rates.</strong>
  Across {len(df):,} spectra and {unique_mols:,} unique target connectivity keys, Exact Top-1 is {percent(top1)} and Exact Top-10 is {percent(top10)}.
  The manifest paper targets are {percent(PAPER_TARGETS["exact_match_top1"])} and {percent(PAPER_TARGETS["exact_match_top10"])} respectively, leaving gaps of {delta_top1 * 100:.2f} and {delta_top10 * 100:.2f} percentage points.</p>
  <p><strong>The dominant quality driver is upstream fingerprint quality, not candidate-list length.</strong>
  The run almost always writes ten candidates, but Top-10 improves over Top-1 by only {top10_lift * 100:.2f} percentage points.
  Spearman correlation between MIST Tanimoto and final Top-10 Tanimoto is {corr_mist_tan:.3f}; correlation with Exact Top-10 is weaker but still directional at {corr_mist_exact:.3f}.</p>
  <p><strong>The formula filter is useful but not enough to guarantee exact recovery.</strong>
  Formula success is {percent(formula_success)}, yet Exact Top-10 remains {percent(top10)}. This means many candidates pass formula filtering while having the wrong connectivity.</p>

  <div class="kpis">
    <div class="kpi"><div class="label">Spectra</div><div class="value">{len(df):,}</div><div class="note">Full MSG test split in this run</div></div>
    <div class="kpi"><div class="label">Exact Top-1</div><div class="value">{percent(top1)}</div><div class="note">Paper target {percent(PAPER_TARGETS["exact_match_top1"])}</div></div>
    <div class="kpi"><div class="label">Exact Top-10</div><div class="value">{percent(top10)}</div><div class="note">Paper target {percent(PAPER_TARGETS["exact_match_top10"])}</div></div>
    <div class="kpi"><div class="label">Formula success</div><div class="value">{percent(formula_success)}</div><div class="note">{zero_formula:,} spectra had zero formula matches</div></div>
  </div>

  <h2>Exact-match gap versus the target</h2>
  <p>The aggregate JSON and a recomputation from per-shard CSVs agree to floating-point precision. The benchmark result is therefore not a reporting artifact in the copied output. The problem is analytical: the generated and reranked candidate lists miss the correct connectivity too often.</p>
  {img_html(charts["headline_vs_paper"], "Exact Top-1 and Top-10 compared with paper targets")}
  {table_html(tables["headline_metrics"], rows=12)}

  <h2>Data integrity and metric definitions</h2>
  <p>The EDA uses <code>detailed_results.csv</code>, <code>predictions.csv</code>, per-shard <code>detailed_results.csv</code>, and <code>completed_shards.json</code>. Exact match is computed by benchmark code using the first block of InChIKey, so it ignores stereochemistry. Tanimoto is Morgan fingerprint similarity to the ground truth molecule for evaluation. MIST Tanimoto measures predicted fingerprint quality before molecule generation.</p>
  {table_html(tables["data_quality_checks"], rows=20)}
  <p>One row records eight predictions instead of ten (<code>MassSpecGymID0188658</code> in <code>shard_010</code>), and that row is an Exact Top-1/Top-10 success. It is not a material source of the aggregate quality gap. Formula statistics need careful reading: benchmark aggregation uses <code>total_formula_matched</code> for actual formula matches, not the padded <code>formula_matches_collected</code> candidate count.</p>

  <h2>Outcome distribution</h2>
  <p>Exact hits are a small part of the output. A meaningful slice of misses still has high structural similarity, but the low-similarity miss bucket is too large to be explained only by tie-breaking or ranking noise.</p>
  {img_html(charts["outcome_breakdown"], "Outcome buckets based on exact-match and Top-10 Tanimoto")}
  {table_html(tables["outcome_breakdown"], rows=10)}

  <h2>MIST fingerprint quality is the main bottleneck</h2>
  <p>Cases with low MIST similarity almost never recover the exact molecule. In the two lowest MIST bins there are {low_mist_spectra:,} spectra and weighted Exact Top-10 is {percent(low_mist_top10)}. In the highest MIST bin, Exact Top-10 rises to {percent(high_mist_top10)}. The final generator still makes mistakes at high MIST, but the sharp separation by MIST bin makes the encoder a primary failure surface.</p>
  {img_html(charts["mist_vs_tanimoto_scatter"], "Per-spectrum relationship between MIST Tanimoto and best generated Top-10 Tanimoto")}
  {img_html(charts["exact_by_mist_bin"], "Quality and exact-match metrics by MIST Tanimoto bin")}
  {img_html(charts["similarity_distributions"], "Similarity distributions for MIST and generated Top-10 candidates")}
  {table_html(tables["by_mist_bin"], rows=10)}

  <h2>Formula filtering mostly works but does not solve connectivity</h2>
  <p>The run generates valid molecules and usually finds at least one formula-matched candidate. However, formula success and exact recovery are far apart: {percent(formula_success)} versus {percent(top10)}. This is consistent with formula being a coarse constraint; many structures share the formula and the current ranking by predicted-fingerprint similarity does not reliably put the correct connectivity first.</p>
  {img_html(charts["formula_matches_distribution"], "Distribution of actual formula matches per spectrum")}
  {table_html(tables["by_formula_actual_bin"], rows=10)}

  <h2>Shard-level failures are concentrated in the tail</h2>
  <p>Quality is not uniform across the 36 shards. Several tail shards have near-zero Exact Top-1 and Top-10 even when some retain moderate Tanimoto, which pulls down the aggregate. This is either a distribution shift in later MassSpecGym IDs, a data ordering issue, or a subset where MIST/formula-conditioned generation is systematically weaker.</p>
  {img_html(charts["shard_metrics"], "Shard-level metric trends")}
  {img_html(charts["shard_heatmap"], "Heatmap of shard-level quality metrics")}
  <h3>Worst shards by exact match</h3>
  {table_html(tables["worst_shards"], rows=12)}
  <h3>Best shards by exact match</h3>
  {table_html(tables["best_shards"], rows=12)}

  <h2>Runtime is generation-bound</h2>
  <p>Summed shard elapsed time is {total_elapsed_h:.1f} hours and summed generation time is {total_generation_h:.1f} hours. The mean shard-level generation-time percentage is {percent(gen_share)}, so the benchmark is dominated by molecule generation loops rather than CSV aggregation or scoring. Per-spectrum generation time is therefore a direct function of attempts, batch size, model latency, and early stopping after enough unique formula matches.</p>
  {img_html(charts["runtime"], "Runtime distribution and throughput by shard")}
  {table_html(tables["runtime_by_shard"], rows=12)}

  <h2>Target complexity is a secondary stressor</h2>
  <p>Using SMILES length and token counts as rough local complexity proxies, longer targets generally show lower exact recovery. This is not a chemistry-perfect descriptor, but it is enough to flag that large or stereochemically dense molecules are harder in this run. Spearman correlation between target SMILES length and Exact Top-10 is {target_len_corr:.3f}.</p>
  {img_html(charts["quality_by_target_length"], "Quality by target SMILES length quintile")}
  {table_html(tables["by_target_length_bin"], rows=10)}

  <h2>Case-level examples</h2>
  <p>The examples CSV contains full target/proposal/prediction SMILES for exact hits, Top-10-only hits, high-Tanimoto near misses, low-MIST failures, and high-MIST no-exact failures. This is the right starting point for manual chemistry inspection.</p>
  {table_html(tables["example_cases"], rows=25)}

  <h2>Limitations and robustness checks</h2>
  <ul>
    <li>The EDA does not recompute RDKit fingerprints locally; it trusts the benchmark's recorded exact and Tanimoto values. This avoids changing chemistry libraries between run and analysis.</li>
    <li>Local complexity metrics are lightweight SMILES proxies, not curated molecular descriptors.</li>
    <li>The report covers FRIGID-base with NGBoost. It does not include a completed full ICEBERG scaling run.</li>
    <li>Shard labels come from per-shard output folders, not from inferred row order.</li>
  </ul>

  <h2>Recommended next steps</h2>
  <ol>
    <li><strong>Audit the weak tail shards first.</strong> Compare shard input distributions, spectra quality, target formulas, and molecule-size distributions for worst versus best shards.</li>
    <li><strong>Run an encoder-vs-decoder ablation.</strong> For a stratified sample, generate with ground-truth fingerprints or oracle formula/fingerprint inputs to separate MIST error from decoder/ranking error.</li>
    <li><strong>Inspect high-MIST no-exact failures.</strong> These are the clearest decoder/reranker failures because the fingerprint signal looks good but exact recovery fails.</li>
    <li><strong>Inspect low-MIST failures separately.</strong> These should focus on MIST spectra preprocessing, checkpoint compatibility, and whether the same MassSpecGym subset has known spectrum/metadata issues.</li>
    <li><strong>Do not treat more Top-K alone as the main fix.</strong> Top-10 only adds {top10_lift * 100:.2f} percentage points over Top-1, so the candidate set itself often lacks the correct connectivity.</li>
  </ol>

  <h2>Supporting artifacts</h2>
  <p>All source tables, chart PNGs, and generated CSV summaries are saved under <code>{html.escape(str(OUT_ROOT))}</code>.</p>
</main>
</body>
</html>
"""

    path = OUT_ROOT / "frigid_msg_base_eda_report.html"
    path.write_text(html_text)
    return path


def main() -> None:
    mkdirs()
    detailed, predictions, stats, completed, sharded = load_data()
    df = enrich(sharded)
    tables = make_tables(detailed, predictions, stats, completed, df)
    charts = make_charts(df, tables)
    report = build_report(df, stats, tables, charts)

    summary = {
        "report": str(report),
        "run_root": str(RUN_ROOT),
        "output_root": str(OUT_ROOT),
        "rows": int(len(df)),
        "unique_target_inchi_keys": int(df["target_inchi_key"].nunique()),
        "exact_match_top1": float(df["exact_match_top1"].mean()),
        "exact_match_top10": float(df["exact_match_top10"].mean()),
        "tanimoto_top1_mean": float(df["tanimoto_top1"].mean()),
        "tanimoto_top10_mean": float(df["tanimoto_top10"].mean()),
        "mist_tanimoto_mean": float(df["mist_tanimoto"].mean()),
        "formula_match_success_rate": float(df["formula_success"].mean()),
        "paper_target_exact_top1": PAPER_TARGETS["exact_match_top1"],
        "paper_target_exact_top10": PAPER_TARGETS["exact_match_top10"],
        "delta_exact_top1_vs_paper": float(df["exact_match_top1"].mean() - PAPER_TARGETS["exact_match_top1"]),
        "delta_exact_top10_vs_paper": float(df["exact_match_top10"].mean() - PAPER_TARGETS["exact_match_top10"]),
        "tables": {k: str(v) for k, v in tables.items()},
        "charts": {k: str(v) for k, v in charts.items()},
    }
    (OUT_ROOT / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
