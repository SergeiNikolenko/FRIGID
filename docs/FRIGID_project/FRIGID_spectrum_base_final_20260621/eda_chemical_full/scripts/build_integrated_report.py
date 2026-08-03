#!/usr/bin/env python3
"""Build an integrated FRIGID chemistry and train-coverage HTML report."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path("/Users/nikolenko/Desktop/FRIGID_spectrum_base_final_20260621")
CHEM = ROOT / "eda_chemical_full"
TRAIN = ROOT / "eda_train_overlap"
OUT = CHEM
CHARTS = OUT / "enhanced_charts"
REPORT = OUT / "frigid_chemical_eda_report.html"


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def table(path: Path, rows: int = 10) -> str:
    df = pd.read_csv(path).head(rows)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    return df.to_html(index=False, escape=True, classes="data-table")


def img(path: Path, alt: str) -> str:
    rel = path.relative_to(OUT)
    return f'<figure><img src="{html.escape(str(rel))}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def savefig(name: str) -> Path:
    path = CHARTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def chart_train_coverage_bins(cov_bin: pd.DataFrame) -> Path:
    order = ["ood_lt_0.50", "weak_train_0.50_0.70", "near_train_0.70_0.85", "near_train_ge_0.85"]
    labels = {
        "ood_lt_0.50": "No close train\nNN < 0.50",
        "weak_train_0.50_0.70": "Weak train\n0.50-0.70",
        "near_train_0.70_0.85": "Near train\n0.70-0.85",
        "near_train_ge_0.85": "Very near train\n>= 0.85",
    }
    df = cov_bin.copy()
    df["share_targets"] = df["targets"] / df["targets"].sum()
    df["label"] = df["train_coverage_bin"].map(labels)
    df["label"] = pd.Categorical(df["label"], [labels[x] for x in order], ordered=True)
    plt.figure(figsize=(9.5, 5.2))
    ax = sns.barplot(data=df.sort_values("label"), x="label", y="share_targets", color="#2563eb")
    ax.set_title("Most evaluated targets are far from train molecules")
    ax.set_xlabel("")
    ax.set_ylabel("Share of unique test targets")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    for c in ax.containers:
        ax.bar_label(c, labels=[f"{bar.get_height():.1%}" for bar in c], fontsize=10)
    return savefig("train_coverage_bins.png")


def chart_class_failure_matrix(class_cov: pd.DataFrame) -> Path:
    df = class_cov[class_cov["targets"] >= 8].copy()
    df["label"] = df["chem_class"].str.replace("_", " ")
    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(
        data=df,
        x="median_nn_train_tanimoto",
        y="exact_top10",
        size="targets",
        hue="bad_low_tanimoto_share",
        palette="magma_r",
        sizes=(80, 900),
        edgecolor="#111827",
        linewidth=0.35,
    )
    offsets = {
        "cyclic_depsipeptide_like": (8, 8),
        "cyclic_peptide_like": (8, 22),
        "large_peptide_like": (8, 8),
        "lipid_like_flexible": (8, 8),
    }
    for _, r in df.iterrows():
        if r["targets"] >= 35 or r["exact_top10"] <= 0.035:
            dx, dy = offsets.get(r["chem_class"], (8, 8))
            ax.annotate(
                r["label"],
                (r["median_nn_train_tanimoto"], r["exact_top10"]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8,
            )
    ax.set_title("Class difficulty is not explained by train similarity alone")
    ax.set_xlabel("Median nearest non-exact train Tanimoto")
    ax.set_ylabel("Exact Top-10")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="")
    return savefig("class_failure_matrix.png")


def chart_cluster_failure_matrix(cluster_cov: pd.DataFrame) -> Path:
    df = cluster_cov.copy()
    df["cluster"] = df["cluster"].astype(str)
    plt.figure(figsize=(9.5, 6.3))
    ax = sns.scatterplot(
        data=df,
        x="median_nn_train_tanimoto",
        y="exact_top10",
        size="targets",
        hue="bad_low_tanimoto_share",
        palette="viridis_r",
        sizes=(120, 950),
        edgecolor="#111827",
        linewidth=0.45,
    )
    for _, r in df.iterrows():
        ax.text(r["median_nn_train_tanimoto"] + 0.004, r["exact_top10"] + 0.003, str(r["cluster"]), fontsize=10, weight="bold")
    ax.set_title("Fingerprint clusters reveal distinct failure regimes")
    ax.set_xlabel("Median nearest non-exact train Tanimoto")
    ax.set_ylabel("Exact Top-10")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="")
    return savefig("cluster_failure_matrix.png")


def chart_worst_best_descriptor_profile(q: pd.DataFrame) -> Path:
    rows = []
    selected = [
        ("mol_wt_median", "Mol weight"),
        ("heavy_atoms_median", "Heavy atoms"),
        ("chiral_centers_median", "Chiral centers"),
        ("rotatable_bonds_median", "Rotatable bonds"),
        ("formula_success", "Formula success"),
        ("bad_low_tanimoto_share", "Low-Tanimoto miss"),
        ("exact_top10", "Exact Top-10"),
    ]
    for _, r in q[q["shard_quality_group"].isin(["worst_8_shards", "best_8_shards"])].iterrows():
        for col, label in selected:
            rows.append({"group": r["shard_quality_group"].replace("_", " "), "metric": label, "value": r[col]})
    df = pd.DataFrame(rows)
    scale_metrics = {"Mol weight", "Heavy atoms", "Chiral centers", "Rotatable bonds"}
    df["plot_value"] = df["value"]
    for metric in scale_metrics:
        mask = df["metric"] == metric
        maxv = df.loc[mask, "value"].max()
        df.loc[mask, "plot_value"] = df.loc[mask, "value"] / maxv if maxv else 0
    plt.figure(figsize=(10.5, 6))
    palette = {"worst 8 shards": "#dc2626", "best 8 shards": "#2563eb"}
    ax = sns.barplot(
        data=df,
        y="metric",
        x="plot_value",
        hue="group",
        hue_order=["worst 8 shards", "best 8 shards"],
        palette=palette,
    )
    ax.set_title("Worst shards are larger, more stereochemical, and much less recoverable")
    ax.set_xlabel("Normalized descriptor value for size descriptors; raw rate for rate metrics")
    ax.set_ylabel("")
    ax.legend(title="")
    return savefig("worst_best_descriptor_profile.png")


def chart_class_quality_bars(class_summary: pd.DataFrame) -> Path:
    df = class_summary.sort_values("exact_top10").copy()
    df["label"] = df["chem_class"].str.replace("_", " ")
    plot = df.melt(
        id_vars=["label", "spectra"],
        value_vars=["exact_top10", "bad_low_tanimoto_share", "mist_tanimoto"],
        var_name="metric",
        value_name="value",
    )
    metric_labels = {
        "exact_top10": "Exact Top-10",
        "bad_low_tanimoto_share": "Low-Tanimoto miss",
        "mist_tanimoto": "MIST Tanimoto",
    }
    plot["metric"] = plot["metric"].map(metric_labels)
    plt.figure(figsize=(12, 7.2))
    ax = sns.barplot(data=plot, y="label", x="value", hue="metric", palette=["#16a34a", "#dc2626", "#64748b"])
    ax.set_title("Hard classes have low exact recovery even when MIST is not zero")
    ax.set_xlabel("Rate / mean")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(title="", loc="center left", bbox_to_anchor=(1.01, 0.5))
    return savefig("class_quality_bars.png")


def build() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    aggregate = json.loads((ROOT / "msg_base_full_ngboost" / "aggregate" / "aggregate_statistics.json").read_text())
    train_summary = json.loads((TRAIN / "train_coverage_summary.json").read_text())
    q = pd.read_csv(CHEM / "tables" / "quality_group_summary.csv")
    class_summary = pd.read_csv(CHEM / "tables" / "class_summary.csv")
    class_mix = pd.read_csv(CHEM / "tables" / "class_mix_worst_vs_best.csv")
    desc = pd.read_csv(CHEM / "tables" / "descriptor_associations.csv")
    cluster = pd.read_csv(CHEM / "tables" / "cluster_summary.csv")
    class_cov = pd.read_csv(TRAIN / "tables" / "class_train_coverage.csv")
    cluster_cov = pd.read_csv(TRAIN / "tables" / "cluster_train_coverage.csv")
    cov_bin = pd.read_csv(TRAIN / "tables" / "coverage_bin_summary.csv")

    charts = {
        "train_coverage_bins": chart_train_coverage_bins(cov_bin),
        "class_failure_matrix": chart_class_failure_matrix(class_cov),
        "cluster_failure_matrix": chart_cluster_failure_matrix(cluster_cov),
        "worst_best_descriptor_profile": chart_worst_best_descriptor_profile(q),
        "class_quality_bars": chart_class_quality_bars(class_summary),
    }

    worst = q[q["shard_quality_group"] == "worst_8_shards"].iloc[0]
    best = q[q["shard_quality_group"] == "best_8_shards"].iloc[0]
    hardest_classes = class_summary.sort_values("exact_top10").head(6)
    enriched = class_mix.sort_values("worst_minus_best_share", ascending=False).head(4)
    worst_clusters = cluster.sort_values(["exact_top10", "bad_low_tanimoto_share"], ascending=[True, False]).head(5)
    cluster_cov_worst = cluster_cov.sort_values(["exact_top10", "bad_low_tanimoto_share"], ascending=[True, False]).head(5)

    css = """
    :root {
      --bg: #f6f7f9; --panel: #ffffff; --ink: #101827; --muted: #5c6678;
      --line: #dbe1ea; --blue: #2563eb; --red: #dc2626; --green: #16a34a;
      --amber: #d97706; --soft-blue: #eaf1ff; --soft-red: #fff1f2;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { max-width: 1240px; margin: 0 auto; padding: 30px 26px 80px; }
    header { padding: 8px 0 22px; border-bottom: 1px solid var(--line); }
    h1 { margin: 0; font-size: 34px; letter-spacing: 0; line-height: 1.1; }
    h2 { margin: 38px 0 14px; font-size: 24px; letter-spacing: 0; }
    h3 { margin: 24px 0 10px; font-size: 17px; }
    p, li { font-size: 15px; line-height: 1.58; }
    .subtitle { color: var(--muted); max-width: 950px; margin: 10px 0 0; }
    .kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin: 18px 0 6px; }
    .kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 13px 14px; min-height: 92px; }
    .kpi .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
    .kpi .value { font-size: 24px; font-weight: 740; margin-top: 7px; }
    .kpi .note { color: var(--muted); font-size: 12px; margin-top: 5px; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(320px, .85fr); gap: 18px; align-items: start; }
    .panel, figure { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .panel { padding: 16px 18px; }
    figure { padding: 10px; margin: 0; }
    figure img { width: 100%; height: auto; display: block; }
    figcaption { color: var(--muted); font-size: 12px; text-align: center; margin-top: 8px; }
    .takeaway { border-left: 4px solid var(--blue); background: var(--soft-blue); padding: 13px 15px; border-radius: 6px; margin: 12px 0; }
    .warning { border-left-color: var(--red); background: var(--soft-red); }
    .data-table { border-collapse: collapse; width: 100%; background: var(--panel); font-size: 12px; margin: 12px 0 22px; }
    .data-table th, .data-table td { border: 1px solid var(--line); padding: 7px 8px; vertical-align: top; }
    .data-table th { background: #eef2f7; text-align: left; }
    code { background: #eef2ff; border-radius: 4px; padding: 1px 4px; }
    .pillrow { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .pill { border: 1px solid var(--line); border-radius: 999px; padding: 6px 9px; font-size: 12px; background: #fff; color: var(--muted); }
    @media (max-width: 980px) { .grid, .kpis { grid-template-columns: 1fr; } }
    """

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FRIGID MSG Base: Chemical Failure and Train Coverage</title>
  <style>{css}</style>
</head>
<body>
<main>
  <header>
    <h1>FRIGID MSG Base: Chemical Failure and Train Coverage</h1>
    <p class="subtitle">Integrated technical report over the full MSG FRIGID-base NGBoost run, the chemical EDA, and the MassSpecGym train-overlap analysis. The goal is to answer which molecules fail and whether that chemistry exists in train.</p>
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

  <h2>Train Coverage Answer</h2>
  <div class="grid">
    {img(charts["train_coverage_bins"], "Train coverage bins for unique test targets")}
    <div class="panel">
      <h3>Interpretation</h3>
      <p>The answer to “were such structures in train?” is mostly no at exact-target level and mostly weak at nearest-neighbor level.</p>
      <div class="takeaway warning"><strong>Exact same molecules:</strong> 0% train overlap by InChIKey.</div>
      <div class="takeaway"><strong>Similar molecules:</strong> only {pct(train_summary["nn_ge_085_share"])} of test targets have a very close non-exact train neighbor.</div>
      <p>This supports an OOD/generalization explanation, but not uniformly: a few bad clusters have moderate train-neighbor similarity and still fail, so decoder/ranking remains a second failure mode.</p>
    </div>
  </div>
  {table(TRAIN / "tables" / "coverage_bin_summary.csv", rows=8)}

  <h2>Chemical Classes</h2>
  <div class="grid">
    {img(charts["class_quality_bars"], "Exact recovery and low-Tanimoto miss rate by chemical class")}
    <div class="panel">
      <h3>What fails</h3>
      <p>Hardest classes by Exact Top-10:</p>
      <div class="pillrow">
        {''.join(f'<span class="pill">{html.escape(row.chem_class)}: {pct(row.exact_top10)}</span>' for row in hardest_classes.itertuples())}
      </div>
      <p>These are chemistry-heavy regions: peptides/depsipeptides, glycosides, macrocycles, and polycyclic aliphatic-like structures. They are large, oxygen-rich, stereochemically rich, and often have many rings or flexible substituents.</p>
    </div>
  </div>
  {table(CHEM / "tables" / "class_summary.csv", rows=12)}

  <h2>Worst Shards Versus Best Shards</h2>
  <div class="grid">
    {img(charts["worst_best_descriptor_profile"], "Descriptor profile for worst versus best shards")}
    <div class="panel">
      <h3>What is enriched in the tail</h3>
      <p>The worst shard group is enriched for:</p>
      <div class="pillrow">
        {''.join(f'<span class="pill">{html.escape(row.chem_class)} +{100*row.worst_minus_best_share:.1f} pp</span>' for row in enriched.itertuples())}
      </div>
      <p>The biggest shift is glycoside/sugar-rich chemistry and polycyclic aliphatic-like chemistry. Small heteroaromatics and aromatic-rich polycycles are much more common in the best shards.</p>
    </div>
  </div>
  {table(CHEM / "tables" / "class_mix_worst_vs_best.csv", rows=12)}

  <h2>Train Similarity Does Not Fully Explain Difficulty</h2>
  <div class="grid">
    {img(charts["class_failure_matrix"], "Class difficulty versus nearest train similarity")}
    <div class="panel">
      <h3>Reading the chart</h3>
      <p>X-axis is median nearest non-exact train Tanimoto. Y-axis is Exact Top-10. Bubble size is number of unique test targets. Color is the low-Tanimoto miss share.</p>
      <p>Some classes are far from train and bad. Others have moderate train-neighbor similarity but remain bad, which points to generation/ranking limitations in addition to OOD.</p>
    </div>
  </div>
  {table(TRAIN / "tables" / "class_train_coverage.csv", rows=12)}

  <h2>Cluster-Level Failure Regimes</h2>
  <div class="grid">
    {img(charts["cluster_failure_matrix"], "Cluster quality versus train-neighbor coverage")}
    <div class="panel">
      <h3>Worst clusters</h3>
      <p>Clusters with zero or near-zero Exact Top-10:</p>
      <div class="pillrow">
        {''.join(f'<span class="pill">cluster {int(row.cluster)}: {pct(row.exact_top10)}, NN {row.median_nn_train_tanimoto:.2f}</span>' for row in cluster_cov_worst.itertuples())}
      </div>
      <p>Cluster 7 is peptide/depsipeptide-like; cluster 10 is polycyclic aliphatic/glycoside-like; cluster 9 has moderate nearest-train similarity but still zero exact, making it a prime decoder/ranking audit target.</p>
    </div>
  </div>
  {table(TRAIN / "tables" / "cluster_train_coverage.csv", rows=14)}

  <h2>Descriptor Evidence</h2>
  <div class="grid">
    <figure><img src="charts/descriptor_shift_bad_vs_hit.png" alt="Descriptor shift bad versus hit"><figcaption>Descriptor shift: bad low-Tanimoto misses versus Exact Top-10 hits</figcaption></figure>
    <div class="panel">
      <h3>What the descriptors say</h3>
      <p>Bad misses are associated with more aliphatic rings, more oxygen atoms, more heavy atoms, more ring/scaffold complexity, higher molecular weight, more chiral centers, higher HBA/TPSA, and more rotatable bonds. Halogenated and small heteroaromatic regions are easier.</p>
      <p>This is consistent with the chemical class result: the failures concentrate in large, stereochemically rich natural-product-like chemistry.</p>
    </div>
  </div>
  {table(CHEM / "tables" / "descriptor_associations.csv", rows=16)}

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
    print(json.dumps({"report": str(REPORT), "charts": {k: str(v) for k, v in charts.items()}}, indent=2))


if __name__ == "__main__":
    build()
