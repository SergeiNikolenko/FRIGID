#!/usr/bin/env python
"""Render static report figures for the MS/MS encoder benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "encoder_benchmark_figures"

INK = "#202735"
MUTED = "#667085"
GRID = "#D9DEE7"
BLUE = "#2F5D8A"
BLUE_DARK = "#1D3E60"
GOLD = "#C58A16"
NEUTRAL = "#AAB2BF"
BACKGROUND = "#FBFCFE"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
        }
    )


def historical_encoder_chart() -> None:
    data = pd.read_csv(FIGURE_DIR / "historical_encoder_metrics.csv").sort_values(
        "mean_fingerprint_tanimoto"
    )
    colors = {
        "MIST": BLUE,
        "Hybrid": BLUE_DARK,
        "Alternative": GOLD,
        "DreaMS": NEUTRAL,
    }
    fig, ax = plt.subplots(figsize=(11.2, 6.8))
    bars = ax.barh(
        data["model"],
        data["mean_fingerprint_tanimoto"],
        color=[colors[value] for value in data["series"]],
        edgecolor=INK,
        linewidth=0.55,
        height=0.67,
    )
    for bar, value in zip(bars, data["mean_fingerprint_tanimoto"], strict=True):
        ax.text(
            value + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            color=INK,
            fontsize=9.5,
        )

    baseline = float(
        data.loc[data["model"].eq("MIST baseline"), "mean_fingerprint_tanimoto"].iloc[0]
    )
    gate = baseline + 0.005
    ax.axvline(gate, color=INK, linestyle="--", linewidth=1.2)
    ax.text(gate + 0.004, 0.1, f"promotion gate {gate:.4f}", rotation=90, fontsize=9)
    ax.set_xlim(0.0, 0.59)
    ax.set_xlabel("Mean fingerprint Tanimoto")
    ax.set_title("Historical fingerprint-encoder results", loc="left", pad=34)
    ax.text(
        0.0,
        1.025,
        "Context only: historical validation surfaces; future candidates use the locked 15,325-row evaluation manifest",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.5,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "historical_encoder_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def threshold_calibration_chart() -> None:
    data = pd.read_csv(FIGURE_DIR / "mist_threshold_calibration.csv")
    selected = data.sort_values(
        ["mean_fingerprint_tanimoto", "threshold"], ascending=[False, True]
    ).iloc[0]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(
        data["threshold"],
        data["mean_fingerprint_tanimoto"],
        color=BLUE,
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        markerfacecolor=BACKGROUND,
        markeredgecolor=BLUE,
    )
    ax.scatter(
        [selected["threshold"]],
        [selected["mean_fingerprint_tanimoto"]],
        s=90,
        color=GOLD,
        edgecolor=INK,
        linewidth=0.7,
        zorder=3,
    )
    ax.annotate(
        f"selected {selected['threshold']:.2f}\nTanimoto {selected['mean_fingerprint_tanimoto']:.4f}",
        xy=(selected["threshold"], selected["mean_fingerprint_tanimoto"]),
        xytext=(0.39, 0.485),
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1.0},
        fontsize=10,
        color=INK,
    )
    ax.set_xlim(0.0, 0.82)
    ax.set_ylim(0.20, 0.57)
    ax.set_xlabel("Binary fingerprint threshold")
    ax.set_ylabel("Mean fingerprint Tanimoto")
    ax.set_title("MIST threshold calibration curve", loc="left", pad=34)
    ax.text(
        0.0,
        1.025,
        "Molecule-disjoint calibration partition, 3,718 spectra and 614 structure clusters",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.5,
    )
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "mist_threshold_calibration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def dlm_upper_bound_chart() -> None:
    data = pd.read_csv(FIGURE_DIR / "dlm_fingerprint_upper_bound.csv")
    positions = np.arange(len(data))
    height = 0.32
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ground_truth = ax.barh(
        positions + height / 2,
        data["ground_truth_fingerprint"],
        height,
        label="Ground-truth fingerprint",
        color=BLUE,
        edgecolor=INK,
        linewidth=0.55,
    )
    mist = ax.barh(
        positions - height / 2,
        data["mist_binary_fingerprint"],
        height,
        label="MIST binary fingerprint",
        color=GOLD,
        edgecolor=INK,
        linewidth=0.55,
    )
    ax.bar_label(ground_truth, fmt="%.4f", padding=4, fontsize=9)
    ax.bar_label(mist, fmt="%.4f", padding=4, fontsize=9)
    ax.set_yticks(positions, data["metric"])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.9)
    ax.set_xlabel("Metric value")
    ax.set_title("DLM outcomes by fingerprint source", loc="left", pad=34)
    ax.text(
        0.0,
        1.025,
        "Same DLM and settings; paired diagnostic on 1,400 spectra (partial run)",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.5,
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dlm_fingerprint_upper_bound.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    historical_encoder_chart()
    threshold_calibration_chart()
    dlm_upper_bound_chart()


if __name__ == "__main__":
    main()
