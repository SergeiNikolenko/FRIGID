#!/usr/bin/env python3
"""Build an interactive Plotly FRIGID chemistry and train-coverage HTML report."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

PYDEPS = Path("/tmp/frigid_eda_pydeps")
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

import numpy as np
import pandas as pd


ROOT = Path("/Users/nikolenko/Desktop/FRIGID_spectrum_base_final_20260621")
CHEM = ROOT / "eda_chemical_full"
TRAIN = ROOT / "eda_train_overlap"
QA = ROOT / "reproduction_qa_sources"
OUT = CHEM
REPORT = OUT / "frigid_chemical_eda_report.html"
PLOTLY_JS = OUT / "vendor" / "plotly-2.35.2.min.js"

BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#16a34a"
SLATE = "#64748b"
AMBER = "#d97706"
INK = "#101827"
MUTED = "#5c6678"
GRID = "#dbe1ea"


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def js(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def base_layout(title: str, height: int) -> dict:
    return {
        "title": {"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 20, "color": INK}},
        "height": height,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", "color": INK, "size": 13},
        "margin": {"l": 72, "r": 48, "t": 72, "b": 70},
        "hovermode": "closest",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        "xaxis": {"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID, "automargin": True},
        "yaxis": {"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID, "automargin": True},
        "hoverlabel": {"bgcolor": "white", "bordercolor": GRID, "font": {"color": INK}},
    }


def plotly_div(chart_id: str, data: list[dict], layout: dict, caption: str) -> str:
    config = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    }
    return f"""
    <section class="plot-card">
      <div id="{html.escape(chart_id)}" class="plot" style="height: {int(layout.get("height", 640))}px;"></div>
      <p class="plot-caption">{html.escape(caption)}</p>
      <script>
        Plotly.newPlot({js(chart_id)}, {js(data)}, {js(layout)}, {js(config)});
      </script>
    </section>
    """


def chart_train_coverage_bins(cov_bin: pd.DataFrame) -> str:
    order = ["ood_lt_0.50", "weak_train_0.50_0.70", "near_train_0.70_0.85", "near_train_ge_0.85"]
    labels = {
        "ood_lt_0.50": "No close train<br>NN < 0.50",
        "weak_train_0.50_0.70": "Weak train<br>0.50-0.70",
        "near_train_0.70_0.85": "Near train<br>0.70-0.85",
        "near_train_ge_0.85": "Very near train<br>>= 0.85",
    }
    df = cov_bin.set_index("train_coverage_bin").loc[order].reset_index().copy()
    df["share_targets"] = df["targets"] / df["targets"].sum()
    custom = [
        [int(r.targets), int(r.spectra), float(r.exact_top10), float(r.bad_low_tanimoto_share), float(r.median_nn_train_tanimoto)]
        for r in df.itertuples()
    ]
    data = [
        {
            "type": "bar",
            "x": [labels[x] for x in df["train_coverage_bin"]],
            "y": [float(x) for x in df["share_targets"]],
            "text": [f"{x:.1%}" for x in df["share_targets"]],
            "textposition": "outside",
            "marker": {"color": [RED, AMBER, "#60a5fa", BLUE], "line": {"color": "#111827", "width": 0.4}},
            "customdata": custom,
            "hovertemplate": (
                "<b>%{x}</b><br>"
                "Targets: %{customdata[0]:,}<br>"
                "Spectra: %{customdata[1]:,}<br>"
                "Share of targets: %{y:.1%}<br>"
                "Exact Top-10: %{customdata[2]:.2%}<br>"
                "Low-Tanimoto miss: %{customdata[3]:.2%}<br>"
                "Median train NN: %{customdata[4]:.3f}<extra></extra>"
            ),
        }
    ]
    layout = base_layout("Most evaluated targets are far from train molecules", 620)
    layout["yaxis"].update({"title": "Share of unique test targets", "tickformat": ".0%", "range": [0, 0.86]})
    layout["xaxis"].update({"title": ""})
    return plotly_div(
        "chart_train_coverage_bins",
        data,
        layout,
        "Hover bars to inspect target counts, spectra counts, Exact Top-10, low-Tanimoto miss share, and nearest-train similarity.",
    )


def chart_reproduction_qa_shard_coverage(shard_coverage: pd.DataFrame) -> str:
    df = shard_coverage.copy()
    df["shard_num"] = df["shard"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("shard_num")
    colors = [RED if x > 0 else BLUE for x in df["missing"]]
    custom = [
        [int(r.expected), int(r.observed), int(r.missing), float(r.coverage)]
        for r in df.itertuples()
    ]
    data = [
        {
            "type": "bar",
            "name": "Missing test rows",
            "x": df["shard"].tolist(),
            "y": [int(x) for x in df["missing"]],
            "customdata": custom,
            "marker": {"color": colors, "line": {"color": "#111827", "width": 0.4}},
            "hovertemplate": (
                "<b>%{x}</b><br>"
                "Expected: %{customdata[0]:,}<br>"
                "Observed: %{customdata[1]:,}<br>"
                "Missing: %{customdata[2]:,}<br>"
                "Coverage: %{customdata[3]:.1%}<extra></extra>"
            ),
        }
    ]
    layout = base_layout("Run is marked complete while 21 shards are missing rows", 620)
    layout["xaxis"].update({"title": "Shard", "tickangle": -45})
    layout["yaxis"].update({"title": "Missing test rows"})
    return plotly_div(
        "chart_reproduction_qa_shard_coverage",
        data,
        layout,
        "This verifies the P0 coverage blocker: manifest expects 17,556 test rows, but aggregate contains 17,082 rows.",
    )


def chart_reproduction_qa_denominator(metric_denominator: pd.DataFrame) -> str:
    df = metric_denominator.copy()
    labels = ["Exact Top-1", "Exact Top-10"]
    traces = [
        {
            "type": "bar",
            "name": "Reported observed denominator",
            "x": labels,
            "y": [float(x) for x in df["reported_observed_denominator"]],
            "marker": {"color": BLUE},
            "hovertemplate": "<b>%{x}</b><br>Reported: %{y:.2%}<extra></extra>",
        },
        {
            "type": "bar",
            "name": "Missing rows counted as failures",
            "x": labels,
            "y": [float(x) for x in df["missing_as_failures_denominator"]],
            "marker": {"color": RED},
            "customdata": [[int(r.successes), int(r.observed_rows), int(r.manifest_test_samples)] for r in df.itertuples()],
            "hovertemplate": (
                "<b>%{x}</b><br>"
                "Adjusted: %{y:.2%}<br>"
                "Successes: %{customdata[0]:,}<br>"
                "Observed rows: %{customdata[1]:,}<br>"
                "Manifest denominator: %{customdata[2]:,}<extra></extra>"
            ),
        },
    ]
    layout = base_layout("Missing rows lower benchmark metrics when counted as failures", 560)
    layout.update({"barmode": "group"})
    layout["yaxis"].update({"title": "Exact match rate", "tickformat": ".0%", "range": [0, 0.16]})
    layout["xaxis"].update({"title": ""})
    return plotly_div(
        "chart_reproduction_qa_denominator",
        traces,
        layout,
        "The published aggregate uses the observed 17,082-row denominator; a strict manifest denominator gives 10.67% Top-1 and 12.06% Top-10.",
    )


def chart_reproduction_qa_proposal_position(proposal_position: pd.DataFrame) -> str:
    df = proposal_position.copy()
    order = [str(i) for i in range(1, 11)] + ["not_found"]
    df["position"] = pd.Categorical(df["position"].astype(str), categories=order, ordered=True)
    df = df.sort_values("position")
    colors = [GREEN if str(x) == "1" else RED if str(x) == "not_found" else AMBER for x in df["position"].astype(str)]
    data = [
        {
            "type": "bar",
            "x": [str(x) for x in df["position"]],
            "y": [int(x) for x in df["rows"]],
            "marker": {"color": colors, "line": {"color": "#111827", "width": 0.4}},
            "hovertemplate": "<b>Position %{x}</b><br>Rows: %{y:,}<extra></extra>",
        }
    ]
    layout = base_layout("Scored top-1 is often not pred_smiles_1", 560)
    layout["xaxis"].update({"title": "Position of detailed_results.proposal_smiles inside predictions.csv pred_smiles_1..10"})
    layout["yaxis"].update({"title": "Rows"})
    return plotly_div(
        "chart_reproduction_qa_proposal_position",
        data,
        layout,
        "Only 9,805 / 17,082 rows have the scored proposal in pred_smiles_1; downstream consumers should not treat pred_smiles_1 as the reported Top-1.",
    )


def chart_reproduction_qa_micro_macro(micro_macro: pd.DataFrame) -> str:
    df = micro_macro.copy()
    label_map = {
        "exact_match_top1": "Exact Top-1",
        "exact_match_top10": "Exact Top-10",
        "tanimoto_top1": "Tanimoto Top-1",
        "mist_tanimoto": "MIST Tanimoto",
    }
    labels = [label_map.get(x, x) for x in df["metric"]]
    traces = [
        {
            "type": "bar",
            "name": "Micro per spectrum",
            "x": labels,
            "y": [float(x) for x in df["micro_per_spectrum"]],
            "marker": {"color": BLUE},
            "hovertemplate": "<b>%{x}</b><br>Micro: %{y:.3f}<extra></extra>",
        },
        {
            "type": "bar",
            "name": "Macro per target SMILES",
            "x": labels,
            "y": [float(x) for x in df["macro_per_target_smiles"]],
            "marker": {"color": AMBER},
            "hovertemplate": "<b>%{x}</b><br>Macro: %{y:.3f}<extra></extra>",
        },
    ]
    layout = base_layout("Repeated spectra make micro and macro metrics differ", 560)
    layout.update({"barmode": "group"})
    layout["yaxis"].update({"title": "Metric value", "range": [0, 0.62]})
    layout["xaxis"].update({"title": ""})
    return plotly_div(
        "chart_reproduction_qa_micro_macro",
        traces,
        layout,
        "The run has 17,082 spectra but only 3,076 unique target SMILES, so both per-spectrum and per-molecule metrics should be reported.",
    )


def chart_class_quality_bars(class_summary: pd.DataFrame) -> str:
    df = class_summary.sort_values("exact_top10", ascending=True).copy()
    labels = [x.replace("_", " ") for x in df["chem_class"]]
    custom = [
        [
            int(r.spectra),
            int(r.unique_targets),
            float(r.formula_success),
            float(r.tanimoto_top10),
            float(r.mol_wt_median),
            float(r.heavy_atoms_median),
            float(r.chiral_centers_median),
        ]
        for r in df.itertuples()
    ]
    traces = []
    for col, name, color in [
        ("exact_top10", "Exact Top-10", GREEN),
        ("bad_low_tanimoto_share", "Low-Tanimoto miss", RED),
        ("mist_tanimoto", "MIST Tanimoto", SLATE),
    ]:
        traces.append(
            {
                "type": "bar",
                "orientation": "h",
                "name": name,
                "y": labels,
                "x": [float(x) for x in df[col]],
                "customdata": custom,
                "marker": {"color": color},
                "hovertemplate": (
                    f"<b>{name}</b><br>%{{y}}<br>"
                    "Value: %{x:.2%}<br>"
                    "Spectra: %{customdata[0]:,}<br>"
                    "Targets: %{customdata[1]:,}<br>"
                    "Formula success: %{customdata[2]:.2%}<br>"
                    "Tanimoto Top-10: %{customdata[3]:.3f}<br>"
                    "Median MW: %{customdata[4]:.1f}<br>"
                    "Heavy atoms: %{customdata[5]:.0f}<br>"
                    "Chiral centers: %{customdata[6]:.0f}<extra></extra>"
                ),
            }
        )
    layout = base_layout("Hard classes have low exact recovery even when MIST is not zero", 820)
    layout.update({"barmode": "group"})
    layout["xaxis"].update({"title": "Rate / mean", "tickformat": ".0%", "range": [0, 0.86]})
    layout["yaxis"].update({"title": "", "automargin": True})
    return plotly_div(
        "chart_class_quality_bars",
        traces,
        layout,
        "Toggle metrics in the legend to isolate exact recovery, MIST quality, or low-Tanimoto misses by chemical class.",
    )


def chart_class_descriptor_heatmap(class_summary: pd.DataFrame) -> str:
    metrics = [
        ("mol_wt_median", "MW"),
        ("heavy_atoms_median", "Heavy atoms"),
        ("ring_count_median", "Rings"),
        ("max_ring_size_median", "Max ring"),
        ("amide_count_median", "Amides"),
        ("ester_count_median", "Esters"),
        ("chiral_centers_median", "Chiral centers"),
        ("rotatable_bonds_median", "Rotatable bonds"),
        ("formula_success", "Formula success"),
        ("exact_top10", "Exact Top-10"),
    ]
    df = class_summary.sort_values("exact_top10", ascending=True).copy()
    labels = [x.replace("_", " ") for x in df["chem_class"]]
    z = []
    raw = []
    for _, row in df.iterrows():
        zrow = []
        rawrow = []
        for col, _ in metrics:
            series = class_summary[col].astype(float)
            lo = float(series.min())
            hi = float(series.max())
            value = float(row[col])
            zrow.append((value - lo) / (hi - lo) if hi > lo else 0.0)
            rawrow.append(value)
        z.append(zrow)
        raw.append(rawrow)
    data = [
        {
            "type": "heatmap",
            "x": [label for _, label in metrics],
            "y": labels,
            "z": z,
            "customdata": raw,
            "colorscale": [
                [0.0, "#eff6ff"],
                [0.5, "#60a5fa"],
                [1.0, "#1e3a8a"],
            ],
            "colorbar": {"title": "Within-metric<br>normalized"},
            "hovertemplate": "<b>%{y}</b><br>%{x}<br>Raw value: %{customdata:.3f}<br>Normalized: %{z:.3f}<extra></extra>",
        }
    ]
    layout = base_layout("Hard classes are chemically larger and more stereochemical", 780)
    layout["margin"].update({"l": 190, "r": 70, "b": 95})
    layout["xaxis"].update({"title": "", "side": "top"})
    layout["yaxis"].update({"title": ""})
    return plotly_div(
        "chart_class_descriptor_heatmap",
        data,
        layout,
        "Each column is normalized within that descriptor, so the heatmap shows which classes are chemically high or low for each property; hover gives raw values.",
    )


def chart_worst_best_descriptor_profile(q: pd.DataFrame) -> str:
    selected = [
        ("mol_wt_median", "Mol weight", "raw"),
        ("heavy_atoms_median", "Heavy atoms", "raw"),
        ("chiral_centers_median", "Chiral centers", "raw"),
        ("rotatable_bonds_median", "Rotatable bonds", "raw"),
        ("formula_success", "Formula success", "rate"),
        ("bad_low_tanimoto_share", "Low-Tanimoto miss", "rate"),
        ("exact_top10", "Exact Top-10", "rate"),
    ]
    rows = []
    groups = [("worst_8_shards", "worst 8 shards", RED), ("best_8_shards", "best 8 shards", BLUE)]
    scale_metrics = {label for _, label, kind in selected if kind == "raw"}
    raw_max = {}
    for col, label, kind in selected:
        if kind == "raw":
            raw_max[label] = float(q[col].max())
    for source_group, label_group, _ in groups:
        r = q[q["shard_quality_group"] == source_group].iloc[0]
        for col, metric, kind in selected:
            raw_value = float(r[col])
            plot_value = raw_value / raw_max[metric] if metric in scale_metrics and raw_max[metric] else raw_value
            rows.append({"group": label_group, "metric": metric, "kind": kind, "raw_value": raw_value, "plot_value": plot_value})
    traces = []
    for group, _, color in groups:
        label_group = group.replace("_", " ")
        dfg = [r for r in rows if r["group"] == label_group]
        traces.append(
            {
                "type": "bar",
                "orientation": "h",
                "name": label_group,
                "y": [r["metric"] for r in dfg],
                "x": [r["plot_value"] for r in dfg],
                "customdata": [[r["raw_value"], r["kind"]] for r in dfg],
                "marker": {"color": color},
                "hovertemplate": (
                    "<b>%{fullData.name}</b><br>%{y}<br>"
                    "Plotted value: %{x:.3f}<br>"
                    "Raw value: %{customdata[0]:.3f}<br>"
                    "Scale: %{customdata[1]}<extra></extra>"
                ),
            }
        )
    layout = base_layout("Worst shards are larger, more stereochemical, and much less recoverable", 640)
    layout.update({"barmode": "group"})
    layout["xaxis"].update({"title": "Normalized descriptor value for size descriptors; raw rate for rate metrics", "range": [0, 1.08]})
    layout["yaxis"].update({"title": ""})
    return plotly_div(
        "chart_worst_best_descriptor_profile",
        traces,
        layout,
        "The size-like descriptors are normalized only for side-by-side readability; hover shows the raw values.",
    )


def chart_class_mix_by_shard_quality(class_mix: pd.DataFrame) -> str:
    df = class_mix.sort_values("worst_minus_best_share", ascending=True).copy()
    labels = [x.replace("_", " ") for x in df["chem_class"]]
    custom = [
        [
            float(r.best_8_shards),
            float(r.middle_20_shards),
            float(r.worst_8_shards),
            float(r.worst_minus_best_share),
        ]
        for r in df.itertuples()
    ]
    traces = []
    for col, name, color in [
        ("best_8_shards", "Best 8 shards", BLUE),
        ("middle_20_shards", "Middle 20 shards", SLATE),
        ("worst_8_shards", "Worst 8 shards", RED),
    ]:
        traces.append(
            {
                "type": "bar",
                "orientation": "h",
                "name": name,
                "y": labels,
                "x": [float(x) for x in df[col]],
                "customdata": custom,
                "marker": {"color": color},
                "hovertemplate": (
                    f"<b>{name}</b><br>%{{y}}<br>"
                    "Class share: %{x:.2%}<br>"
                    "Best share: %{customdata[0]:.2%}<br>"
                    "Middle share: %{customdata[1]:.2%}<br>"
                    "Worst share: %{customdata[2]:.2%}<br>"
                    "Worst minus best: %{customdata[3]:+.2%}<extra></extra>"
                ),
            }
        )
    layout = base_layout("Worst shards are enriched for glycoside and polycyclic aliphatic chemistry", 780)
    layout.update({"barmode": "group"})
    layout["xaxis"].update({"title": "Share of spectra inside shard-quality group", "tickformat": ".0%"})
    layout["yaxis"].update({"title": ""})
    return plotly_div(
        "chart_class_mix_by_shard_quality",
        traces,
        layout,
        "This replaces the class-mix table: compare how much each chemical class contributes to best, middle, and worst shard groups.",
    )


def scaled_sizes(values: pd.Series, min_size: int = 16, max_size: int = 72) -> list[float]:
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return [float((min_size + max_size) / 2)] * len(values)
    return [float(min_size + (x - lo) / (hi - lo) * (max_size - min_size)) for x in values]


def chart_class_failure_matrix(class_cov: pd.DataFrame) -> str:
    df = class_cov[class_cov["targets"] >= 8].copy()
    df["label"] = df["chem_class"].str.replace("_", " ")
    text = [label if targets >= 35 or exact <= 0.035 else "" for label, targets, exact in zip(df["label"], df["targets"], df["exact_top10"])]
    custom = [
        [
            r.chem_class,
            int(r.targets),
            int(r.spectra),
            float(r.p25_nn_train_tanimoto),
            float(r.p75_nn_train_tanimoto),
            float(r.share_nn_ge_070),
            float(r.share_nn_ge_085),
            float(r.bad_low_tanimoto_share),
            float(r.mist_tanimoto),
            float(r.mol_wt_median),
            float(r.heavy_atoms_median),
            float(r.chiral_centers_median),
        ]
        for r in df.itertuples()
    ]
    data = [
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": [float(x) for x in df["median_nn_train_tanimoto"]],
            "y": [float(x) for x in df["exact_top10"]],
            "text": text,
            "textposition": "top center",
            "textfont": {"size": 12, "color": INK},
            "customdata": custom,
            "marker": {
                "size": scaled_sizes(df["targets"]),
                "color": [float(x) for x in df["bad_low_tanimoto_share"]],
                "colorscale": "Magma",
                "reversescale": True,
                "showscale": True,
                "colorbar": {"title": "Low-Tanimoto<br>miss", "tickformat": ".0%"},
                "line": {"color": "#111827", "width": 0.8},
                "opacity": 0.88,
            },
            "hovertemplate": (
                "<b>%{customdata[0]}</b><br>"
                "Targets: %{customdata[1]:,}<br>"
                "Spectra: %{customdata[2]:,}<br>"
                "Median train NN: %{x:.3f}<br>"
                "Train NN IQR: %{customdata[3]:.3f}-%{customdata[4]:.3f}<br>"
                "Train NN >= 0.70: %{customdata[5]:.2%}<br>"
                "Train NN >= 0.85: %{customdata[6]:.2%}<br>"
                "Exact Top-10: %{y:.2%}<br>"
                "Low-Tanimoto miss: %{customdata[7]:.2%}<br>"
                "MIST Tanimoto: %{customdata[8]:.3f}<br>"
                "Median MW: %{customdata[9]:.1f}<br>"
                "Heavy atoms: %{customdata[10]:.0f}<br>"
                "Chiral centers: %{customdata[11]:.0f}<extra></extra>"
            ),
        }
    ]
    layout = base_layout("Class difficulty is not explained by train similarity alone", 760)
    layout["xaxis"].update({"title": "Median nearest non-exact train Tanimoto", "range": [0.34, 0.71]})
    layout["yaxis"].update({"title": "Exact Top-10", "tickformat": ".0%", "range": [-0.015, 0.205]})
    return plotly_div(
        "chart_class_failure_matrix",
        data,
        layout,
        "Bubble area reflects target count; color reflects low-Tanimoto miss share. Hover gives per-class train coverage and chemistry descriptors.",
    )


def chart_train_coverage_by_class(class_cov: pd.DataFrame) -> str:
    df = class_cov.sort_values("median_nn_train_tanimoto", ascending=True).copy()
    labels = [x.replace("_", " ") for x in df["chem_class"]]
    custom = [
        [
            int(r.targets),
            int(r.spectra),
            float(r.median_nn_train_tanimoto),
            float(r.p25_nn_train_tanimoto),
            float(r.p75_nn_train_tanimoto),
            float(r.exact_top10),
            float(r.bad_low_tanimoto_share),
        ]
        for r in df.itertuples()
    ]
    traces = []
    for col, name, color in [
        ("share_nn_ge_070", "Train NN >= 0.70", BLUE),
        ("share_nn_ge_085", "Train NN >= 0.85", AMBER),
        ("exact_top10", "Exact Top-10", GREEN),
    ]:
        traces.append(
            {
                "type": "bar",
                "orientation": "h",
                "name": name,
                "y": labels,
                "x": [float(x) for x in df[col]],
                "customdata": custom,
                "marker": {"color": color},
                "hovertemplate": (
                    f"<b>{name}</b><br>%{{y}}<br>"
                    "Value: %{x:.2%}<br>"
                    "Targets: %{customdata[0]:,}<br>"
                    "Spectra: %{customdata[1]:,}<br>"
                    "Median train NN: %{customdata[2]:.3f}<br>"
                    "Train NN IQR: %{customdata[3]:.3f}-%{customdata[4]:.3f}<br>"
                    "Exact Top-10: %{customdata[5]:.2%}<br>"
                    "Low-Tanimoto miss: %{customdata[6]:.2%}<extra></extra>"
                ),
            }
        )
    layout = base_layout("Nearest-train coverage by chemical class", 820)
    layout.update({"barmode": "group"})
    layout["xaxis"].update({"title": "Share / Exact Top-10", "tickformat": ".0%", "range": [0, 1.0]})
    layout["yaxis"].update({"title": ""})
    return plotly_div(
        "chart_train_coverage_by_class",
        traces,
        layout,
        "This replaces the train-coverage table: the chart shows which classes have close train neighbors and whether that translates into exact recovery.",
    )


def chart_cluster_failure_matrix(cluster_cov: pd.DataFrame) -> str:
    df = cluster_cov.copy()
    labels = [str(int(x)) for x in df["cluster"]]
    custom = [
        [
            int(r.cluster),
            int(r.targets),
            int(r.spectra),
            float(r.p25_nn_train_tanimoto),
            float(r.p75_nn_train_tanimoto),
            float(r.share_nn_ge_070),
            float(r.share_nn_ge_085),
            float(r.bad_low_tanimoto_share),
            float(r.mist_tanimoto),
            float(r.mol_wt_median),
            float(r.heavy_atoms_median),
            float(r.chiral_centers_median),
        ]
        for r in df.itertuples()
    ]
    data = [
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": [float(x) for x in df["median_nn_train_tanimoto"]],
            "y": [float(x) for x in df["exact_top10"]],
            "text": labels,
            "textposition": "middle center",
            "textfont": {"size": 14, "color": "white"},
            "customdata": custom,
            "marker": {
                "size": scaled_sizes(df["targets"], 22, 78),
                "color": [float(x) for x in df["bad_low_tanimoto_share"]],
                "colorscale": "Viridis",
                "reversescale": True,
                "showscale": True,
                "colorbar": {"title": "Low-Tanimoto<br>miss", "tickformat": ".0%"},
                "line": {"color": "#111827", "width": 0.9},
                "opacity": 0.9,
            },
            "hovertemplate": (
                "<b>Cluster %{customdata[0]}</b><br>"
                "Targets: %{customdata[1]:,}<br>"
                "Spectra: %{customdata[2]:,}<br>"
                "Median train NN: %{x:.3f}<br>"
                "Train NN IQR: %{customdata[3]:.3f}-%{customdata[4]:.3f}<br>"
                "Train NN >= 0.70: %{customdata[5]:.2%}<br>"
                "Train NN >= 0.85: %{customdata[6]:.2%}<br>"
                "Exact Top-10: %{y:.2%}<br>"
                "Low-Tanimoto miss: %{customdata[7]:.2%}<br>"
                "MIST Tanimoto: %{customdata[8]:.3f}<br>"
                "Median MW: %{customdata[9]:.1f}<br>"
                "Heavy atoms: %{customdata[10]:.0f}<br>"
                "Chiral centers: %{customdata[11]:.0f}<extra></extra>"
            ),
        }
    ]
    layout = base_layout("Fingerprint clusters reveal distinct failure regimes", 740)
    layout["xaxis"].update({"title": "Median nearest non-exact train Tanimoto", "range": [0.32, 0.74]})
    layout["yaxis"].update({"title": "Exact Top-10", "tickformat": ".0%", "range": [-0.015, 0.205]})
    return plotly_div(
        "chart_cluster_failure_matrix",
        data,
        layout,
        "Cluster labels are drawn inside the points; hover shows train-neighbor coverage, quality, and median chemical descriptors.",
    )


def chart_cluster_quality_bars(cluster_cov: pd.DataFrame) -> str:
    df = cluster_cov.sort_values("exact_top10", ascending=True).copy()
    labels = [f"cluster {int(x)}" for x in df["cluster"]]
    custom = [
        [
            int(r.targets),
            int(r.spectra),
            float(r.median_nn_train_tanimoto),
            float(r.share_nn_ge_070),
            float(r.share_nn_ge_085),
            float(r.mol_wt_median),
            float(r.chiral_centers_median),
        ]
        for r in df.itertuples()
    ]
    traces = []
    for col, name, color in [
        ("exact_top10", "Exact Top-10", GREEN),
        ("bad_low_tanimoto_share", "Low-Tanimoto miss", RED),
        ("mist_tanimoto", "MIST Tanimoto", SLATE),
    ]:
        traces.append(
            {
                "type": "bar",
                "orientation": "h",
                "name": name,
                "y": labels,
                "x": [float(x) for x in df[col]],
                "customdata": custom,
                "marker": {"color": color},
                "hovertemplate": (
                    f"<b>{name}</b><br>%{{y}}<br>"
                    "Value: %{x:.2%}<br>"
                    "Targets: %{customdata[0]:,}<br>"
                    "Spectra: %{customdata[1]:,}<br>"
                    "Median train NN: %{customdata[2]:.3f}<br>"
                    "Train NN >= 0.70: %{customdata[3]:.2%}<br>"
                    "Train NN >= 0.85: %{customdata[4]:.2%}<br>"
                    "Median MW: %{customdata[5]:.1f}<br>"
                    "Chiral centers: %{customdata[6]:.0f}<extra></extra>"
                ),
            }
        )
    layout = base_layout("Cluster quality leaderboard", 820)
    layout.update({"barmode": "group"})
    layout["xaxis"].update({"title": "Rate / mean", "tickformat": ".0%", "range": [0, 0.92]})
    layout["yaxis"].update({"title": ""})
    return plotly_div(
        "chart_cluster_quality_bars",
        traces,
        layout,
        "This replaces the cluster table: sort order exposes zero-recovery clusters first while retaining MIST and low-Tanimoto miss context.",
    )


def chart_descriptor_associations(desc: pd.DataFrame) -> str:
    df = desc.copy()
    df["score"] = df[["bad_minus_top10_hit_cohen_d", "worst8_minus_best8_cohen_d"]].abs().max(axis=1)
    df = df.sort_values("score", ascending=False).head(18).sort_values("bad_minus_top10_hit_cohen_d")
    labels = [x.replace("_", " ") for x in df["descriptor"]]
    custom = [
        [
            float(r.top10_hit_median),
            float(r.bad_low_tanimoto_median),
            float(r.best8_median),
            float(r.worst8_median),
            float(r.spearman_with_exact_top10),
            float(r.spearman_with_tanimoto_top10),
            float(r.spearman_with_mist_tanimoto),
        ]
        for r in df.itertuples()
    ]
    traces = [
        {
            "type": "bar",
            "orientation": "h",
            "name": "Bad miss vs Top-10 hit",
            "y": labels,
            "x": [float(x) for x in df["bad_minus_top10_hit_cohen_d"]],
            "customdata": custom,
            "marker": {"color": RED},
            "hovertemplate": (
                "<b>%{y}</b><br>"
                "Cohen d: %{x:.3f}<br>"
                "Top-10 hit median: %{customdata[0]:.3f}<br>"
                "Bad miss median: %{customdata[1]:.3f}<br>"
                "Spearman exact Top-10: %{customdata[4]:.3f}<br>"
                "Spearman Tanimoto Top-10: %{customdata[5]:.3f}<br>"
                "Spearman MIST: %{customdata[6]:.3f}<extra></extra>"
            ),
        },
        {
            "type": "bar",
            "orientation": "h",
            "name": "Worst 8 vs Best 8 shards",
            "y": labels,
            "x": [float(x) for x in df["worst8_minus_best8_cohen_d"]],
            "customdata": custom,
            "marker": {"color": BLUE},
            "hovertemplate": (
                "<b>%{y}</b><br>"
                "Cohen d: %{x:.3f}<br>"
                "Best 8 median: %{customdata[2]:.3f}<br>"
                "Worst 8 median: %{customdata[3]:.3f}<br>"
                "Spearman exact Top-10: %{customdata[4]:.3f}<br>"
                "Spearman Tanimoto Top-10: %{customdata[5]:.3f}<br>"
                "Spearman MIST: %{customdata[6]:.3f}<extra></extra>"
            ),
        },
    ]
    layout = base_layout("Descriptors separating failures from recoverable cases", 860)
    layout.update({"barmode": "group", "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": -0.5, "y1": len(labels) - 0.5, "line": {"color": "#111827", "width": 1}}]})
    layout["xaxis"].update({"title": "Cohen d; positive means larger in bad/worst group", "zeroline": True, "zerolinecolor": "#111827"})
    layout["yaxis"].update({"title": ""})
    return plotly_div(
        "chart_descriptor_associations",
        traces,
        layout,
        "Positive effects mean the descriptor is larger in bad misses or worst shards; negative effects mean the descriptor is larger in easier/recovered cases.",
    )


def build() -> None:
    if not PLOTLY_JS.exists():
        raise FileNotFoundError(f"Plotly bundle is missing: {PLOTLY_JS}")

    aggregate = json.loads((ROOT / "msg_base_full_ngboost" / "aggregate" / "aggregate_statistics.json").read_text())
    train_summary = json.loads((TRAIN / "train_coverage_summary.json").read_text())
    qa_summary = json.loads((QA / "qa_summary.json").read_text())
    shard_coverage = pd.read_csv(QA / "shard_coverage.csv")
    metric_denominator = pd.read_csv(QA / "metric_denominator_comparison.csv")
    proposal_position = pd.read_csv(QA / "proposal_position_in_predictions.csv")
    micro_macro = pd.read_csv(QA / "micro_vs_macro_metrics.csv")
    q = pd.read_csv(CHEM / "tables" / "quality_group_summary.csv")
    class_summary = pd.read_csv(CHEM / "tables" / "class_summary.csv")
    class_mix = pd.read_csv(CHEM / "tables" / "class_mix_worst_vs_best.csv")
    desc = pd.read_csv(CHEM / "tables" / "descriptor_associations.csv")
    cluster_cov = pd.read_csv(TRAIN / "tables" / "cluster_train_coverage.csv")
    class_cov = pd.read_csv(TRAIN / "tables" / "class_train_coverage.csv")
    cov_bin = pd.read_csv(TRAIN / "tables" / "coverage_bin_summary.csv")

    charts = {
        "reproduction_qa_shard_coverage": chart_reproduction_qa_shard_coverage(shard_coverage),
        "reproduction_qa_denominator": chart_reproduction_qa_denominator(metric_denominator),
        "reproduction_qa_proposal_position": chart_reproduction_qa_proposal_position(proposal_position),
        "reproduction_qa_micro_macro": chart_reproduction_qa_micro_macro(micro_macro),
        "train_coverage_bins": chart_train_coverage_bins(cov_bin),
        "class_quality_bars": chart_class_quality_bars(class_summary),
        "class_descriptor_heatmap": chart_class_descriptor_heatmap(class_summary),
        "worst_best_descriptor_profile": chart_worst_best_descriptor_profile(q),
        "class_mix_by_shard_quality": chart_class_mix_by_shard_quality(class_mix),
        "class_failure_matrix": chart_class_failure_matrix(class_cov),
        "train_coverage_by_class": chart_train_coverage_by_class(class_cov),
        "cluster_failure_matrix": chart_cluster_failure_matrix(cluster_cov),
        "cluster_quality_bars": chart_cluster_quality_bars(cluster_cov),
        "descriptor_associations": chart_descriptor_associations(desc),
    }

    worst = q[q["shard_quality_group"] == "worst_8_shards"].iloc[0]
    best = q[q["shard_quality_group"] == "best_8_shards"].iloc[0]
    hardest_classes = class_summary.sort_values("exact_top10").head(6)
    enriched = class_mix.sort_values("worst_minus_best_share", ascending=False).head(4)
    cluster_cov_worst = cluster_cov.sort_values(["exact_top10", "bad_low_tanimoto_share"], ascending=[True, False]).head(5)

    css = """
    :root {
      --bg: #f6f7f9; --panel: #ffffff; --ink: #101827; --muted: #5c6678;
      --line: #dbe1ea; --blue: #2563eb; --red: #dc2626; --green: #16a34a;
      --soft-blue: #eaf1ff; --soft-red: #fff1f2;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { max-width: 1560px; margin: 0 auto; padding: 30px 28px 90px; }
    header { padding: 8px 0 22px; border-bottom: 1px solid var(--line); }
    h1 { margin: 0; font-size: 36px; letter-spacing: 0; line-height: 1.1; }
    h2 { margin: 44px 0 16px; font-size: 26px; letter-spacing: 0; }
    h3 { margin: 0 0 10px; font-size: 17px; }
    p, li { font-size: 15px; line-height: 1.58; }
    .subtitle { color: var(--muted); max-width: 1040px; margin: 10px 0 0; }
    .kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 20px 0 6px; }
    .kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 15px; min-height: 96px; }
    .kpi .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
    .kpi .value { font-size: 25px; font-weight: 740; margin-top: 7px; }
    .kpi .note { color: var(--muted); font-size: 12px; margin-top: 5px; }
    .panel, .plot-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .panel { padding: 16px 18px; margin: 12px 0 16px; max-width: 1120px; }
    .plot-card { padding: 14px 16px 10px; margin: 12px 0 14px; }
    .plot { width: 100%; }
    .plot-caption { color: var(--muted); font-size: 12px; margin: 8px 2px 0; }
    .takeaway { border-left: 4px solid var(--blue); background: var(--soft-blue); padding: 13px 15px; border-radius: 6px; margin: 12px 0; }
    .warning { border-left-color: var(--red); background: var(--soft-red); }
    code { background: #eef2ff; border-radius: 4px; padding: 1px 4px; }
    .pillrow { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .pill { border: 1px solid var(--line); border-radius: 999px; padding: 6px 9px; font-size: 12px; background: #fff; color: var(--muted); }
    @media (max-width: 980px) {
      main { padding: 20px 14px 70px; }
      .kpis { grid-template-columns: 1fr; }
      h1 { font-size: 30px; }
    }
    """

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FRIGID MSG Base: Chemical Failure and Train Coverage</title>
  <script src="vendor/plotly-2.35.2.min.js"></script>
  <style>{css}</style>
</head>
<body>
<main>
  <header>
    <h1>FRIGID MSG Base: Chemical Failure and Train Coverage</h1>
    <p class="subtitle">Interactive report over the full MSG FRIGID-base NGBoost run, the chemical EDA, and the MassSpecGym train-overlap analysis. Plotly charts support hover, zoom, pan, scroll zoom, and legend toggling.</p>
    <div class="kpis">
      <div class="kpi"><div class="label">Spectra</div><div class="value">{aggregate["total_spectra"]:,}</div><div class="note">full benchmark output</div></div>
      <div class="kpi"><div class="label">Exact Top-10</div><div class="value">{pct(aggregate["exact_match_top10"])}</div><div class="note">paper target {pct(aggregate["paper_target_msg_frigid_base_top10"])}</div></div>
      <div class="kpi"><div class="label">Worst 8 shards</div><div class="value">{pct(worst["exact_top10"])}</div><div class="note">Exact Top-10</div></div>
      <div class="kpi"><div class="label">Exact train overlap</div><div class="value">{pct(train_summary["exact_in_train_share"])}</div><div class="note">same InChIKey in train</div></div>
      <div class="kpi"><div class="label">Near train >= 0.85</div><div class="value">{pct(train_summary["nn_ge_085_share"])}</div><div class="note">non-exact nearest neighbor</div></div>
    </div>
  </header>

  <h2>Main Conclusions</h2>
  <div class="panel">
    <p><strong>1. The weak shard tail is real and chemically structured.</strong> Worst 8 shards are {pct(worst["exact_top10"])} Exact Top-10 versus {pct(best["exact_top10"])} for best 8 shards. They are much heavier and more stereochemically dense: median MW {worst["mol_wt_median"]:.0f} vs {best["mol_wt_median"]:.0f}, heavy atoms {worst["heavy_atoms_median"]:.0f} vs {best["heavy_atoms_median"]:.0f}, chiral centers {worst["chiral_centers_median"]:.0f} vs {best["chiral_centers_median"]:.0f}.</p>
    <p><strong>2. The hard classes are not generic small molecules.</strong> Cyclic peptide-like, cyclic depsipeptide-like, large peptide-like, glycoside/sugar-rich, macrocycle, and polycyclic aliphatic/steroid-like regions have the weakest exact recovery.</p>
    <p><strong>3. Train exact overlap is zero for evaluated test targets.</strong> None of the 2,905 benchmark target InChIKeys appear in train. That is good split hygiene, but it means the model must generalize by chemistry rather than memorize exact molecules.</p>
    <p><strong>4. Most test targets are not close to train by Morgan fingerprint.</strong> Only {pct(train_summary["nn_ge_085_share"])} have a non-exact train neighbor with Tanimoto >= 0.85, and {pct(train_summary["nn_ge_070_share"])} have one >= 0.70. The difficult glycoside/peptide/polycyclic families are therefore often OOD or near-OOD in structure space.</p>
  </div>

  <h2>Reproduction QA Blockers</h2>
  <div class="panel">
    <p><strong>This run should not be treated as a final paper reproduction yet.</strong> The locally verified QA audit found P0/P1 packaging and aggregation blockers before any chemistry interpretation: {qa_summary["missing_test_objects"]:,} manifest test objects are absent from the aggregate, {qa_summary["broken_symlinks"]:,}/{qa_summary["symlinks_total"]:,} shard-data symlinks are broken in the Desktop package, and <code>pred_smiles_1</code> matches the scored <code>proposal_smiles</code> in only {pct(qa_summary["proposal_equals_pred_smiles_1_share"])} of rows.</p>
    <p><strong>What remains useful:</strong> the chemical EDA below is still useful for explaining the observed 17,082-row run. <strong>What is not acceptable yet:</strong> calling this a clean final benchmark reproduction against the 17,556-row manifest denominator.</p>
    <div class="pillrow">
      <span class="pill">Manifest rows: {qa_summary["manifest_test_samples"]:,}</span>
      <span class="pill">Observed rows: {qa_summary["aggregate_detailed_rows"]:,}</span>
      <span class="pill">Missing: {qa_summary["missing_test_objects"]:,} ({pct(qa_summary["missing_test_object_share"])})</span>
      <span class="pill">Broken symlinks: {qa_summary["broken_symlinks"]:,}</span>
      <span class="pill">len.pk warnings: {qa_summary["len_pk_missing_warnings"]}</span>
    </div>
  </div>
  {charts["reproduction_qa_shard_coverage"]}
  {charts["reproduction_qa_denominator"]}
  {charts["reproduction_qa_proposal_position"]}
  {charts["reproduction_qa_micro_macro"]}
  <div class="panel">
    <h3>Source Files Added</h3>
    <p>The reproducibility QA sources are saved in <code>/Users/nikolenko/Desktop/FRIGID_spectrum_base_final_20260621/reproduction_qa_sources</code>. They include the copied pasted audit, recomputed CSV evidence, a machine-readable summary, and a README. RDKit-dependent formula/invalid-SMILES checks from the pasted audit were not rerun locally because RDKit is not available in the current local Python environment.</p>
  </div>

  <h2>Train Coverage Answer</h2>
  {charts["train_coverage_bins"]}
  <div class="panel">
    <h3>Interpretation</h3>
    <p>The answer to “were such structures in train?” is mostly no at exact-target level and mostly weak at nearest-neighbor level.</p>
    <div class="takeaway warning"><strong>Exact same molecules:</strong> 0% train overlap by InChIKey.</div>
    <div class="takeaway"><strong>Similar molecules:</strong> only {pct(train_summary["nn_ge_085_share"])} of test targets have a very close non-exact train neighbor.</div>
    <p>This supports an OOD/generalization explanation, but not uniformly: a few bad clusters have moderate train-neighbor similarity and still fail, so decoder/ranking remains a second failure mode.</p>
  </div>

  <h2>Chemical Classes</h2>
  {charts["class_quality_bars"]}
  {charts["class_descriptor_heatmap"]}
  <div class="panel">
    <h3>What fails</h3>
    <p>Hardest classes by Exact Top-10:</p>
    <div class="pillrow">
      {''.join(f'<span class="pill">{html.escape(row.chem_class)}: {pct(row.exact_top10)}</span>' for row in hardest_classes.itertuples())}
    </div>
    <p>These are chemistry-heavy regions: peptides/depsipeptides, glycosides, macrocycles, and polycyclic aliphatic-like structures. They are large, oxygen-rich, stereochemically rich, and often have many rings or flexible substituents.</p>
  </div>

  <h2>Worst Shards Versus Best Shards</h2>
  {charts["worst_best_descriptor_profile"]}
  {charts["class_mix_by_shard_quality"]}
  <div class="panel">
    <h3>What is enriched in the tail</h3>
    <p>The worst shard group is enriched for:</p>
    <div class="pillrow">
      {''.join(f'<span class="pill">{html.escape(row.chem_class)} +{100*row.worst_minus_best_share:.1f} pp</span>' for row in enriched.itertuples())}
    </div>
    <p>The biggest shift is glycoside/sugar-rich chemistry and polycyclic aliphatic-like chemistry. Small heteroaromatics and aromatic-rich polycycles are much more common in the best shards.</p>
  </div>

  <h2>Train Similarity Does Not Fully Explain Difficulty</h2>
  {charts["class_failure_matrix"]}
  {charts["train_coverage_by_class"]}
  <div class="panel">
    <h3>Reading the chart</h3>
    <p>X-axis is median nearest non-exact train Tanimoto. Y-axis is Exact Top-10. Bubble size is number of unique test targets. Color is the low-Tanimoto miss share.</p>
    <p>Some classes are far from train and bad. Others have moderate train-neighbor similarity but remain bad, which points to generation/ranking limitations in addition to OOD.</p>
  </div>

  <h2>Cluster-Level Failure Regimes</h2>
  {charts["cluster_failure_matrix"]}
  {charts["cluster_quality_bars"]}
  <div class="panel">
    <h3>Worst clusters</h3>
    <p>Clusters with zero or near-zero Exact Top-10:</p>
    <div class="pillrow">
      {''.join(f'<span class="pill">cluster {int(row.cluster)}: {pct(row.exact_top10)}, NN {row.median_nn_train_tanimoto:.2f}</span>' for row in cluster_cov_worst.itertuples())}
    </div>
    <p>Cluster 7 is peptide/depsipeptide-like; cluster 10 is polycyclic aliphatic/glycoside-like; cluster 9 has moderate nearest-train similarity but still zero exact, making it a prime decoder/ranking audit target.</p>
  </div>

  <h2>Descriptor Evidence</h2>
  {charts["descriptor_associations"]}
  <div class="panel">
    <h3>What the descriptors say</h3>
    <p>Bad misses are associated with more aliphatic rings, more oxygen atoms, more heavy atoms, more ring/scaffold complexity, higher molecular weight, more chiral centers, higher HBA/TPSA, and more rotatable bonds. Halogenated and small heteroaromatic regions are easier.</p>
    <p>This is consistent with the chemical class result: the failures concentrate in large, stereochemically rich natural-product-like chemistry.</p>
  </div>

  <h2>Actionable Next Tests</h2>
  <div class="panel">
    <ol>
      <li><strong>Oracle fingerprint ablation by class and cluster.</strong> Run cyclic peptide/depsipeptide, glycoside, polycyclic aliphatic, and cluster 7/9/10 with ground-truth fingerprints. If exact jumps, MIST/preprocessing is the primary blocker; if not, DLM/ranking is.</li>
      <li><strong>Nearest-train stratified sampling.</strong> Compare OOD targets below 0.50 train NN, weak 0.50-0.70, near 0.70-0.85, and very-near >=0.85. The very-near failures are the cleanest decoder/ranking cases.</li>
      <li><strong>Separate reporting by chemistry class.</strong> A single MSG aggregate hides that small heteroaromatics behave very differently from glycoside/peptide/polycyclic natural-product-like chemistry.</li>
      <li><strong>Do not spend first on Top-100 globally.</strong> Top-100 should be tested only with raw candidate pool saving and stratified by class. Otherwise it will mix OOD failures and generation-depth effects.</li>
    </ol>
  </div>
</main>
</body>
</html>
"""
    REPORT.write_text(html_text)
    print(json.dumps({"report": str(REPORT), "plotly": str(PLOTLY_JS), "charts": sorted(charts)}, indent=2))


if __name__ == "__main__":
    build()
