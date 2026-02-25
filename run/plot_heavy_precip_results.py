#!/usr/bin/env python3
"""
Plot heavy precipitation evaluation results from MRMS verification.

Creates:
1. RMSE/MAE/Bias vs lead time (continuous metrics)
2. Frequency Bias vs lead time (by threshold)
3. ETS/CSI vs lead time (by threshold)

Usage:
    python plot_heavy_precip_results.py [--csv case_342_heavy_precip_results.csv]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colorblind-friendly palette
COLORS = {
    "WeatherMesh-4": "#0072B2",
    "IFS": "#D55E00",
    "AIFS": "#009E73",
    "GFS": "#CC79A7",
    "GFS-Ens-Mean": "#E69F00",
}

MODEL_ORDER = ["WeatherMesh-4", "IFS", "AIFS", "GFS", "GFS-Ens-Mean"]

OUTPUT_DIR = Path("/huge/proc/eva/ewb_case_342")


def parse_lead_hours(lead_str: str) -> float:
    """Convert pandas Timedelta string to hours."""
    td = pd.Timedelta(lead_str)
    return td.total_seconds() / 3600


def plot_continuous_metrics(df: pd.DataFrame, output_dir: Path) -> list[str]:
    """Plot RMSE, MAE, and Bias vs lead time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cont = df[df["target_variable"] == "tp_6hr"].copy()
    cont["lead_h"] = cont["lead_time"].apply(parse_lead_hours)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    metrics = ["RootMeanSquaredError", "MeanAbsoluteError", "MeanError"]
    titles = ["RMSE (mm)", "MAE (mm)", "Bias (mm)"]

    for ax, metric, title in zip(axes, metrics, titles):
        for model in MODEL_ORDER:
            m = cont[(cont["forecast_source"] == model) & (cont["metric"] == metric)]
            if len(m) == 0:
                continue
            agg = m.groupby("lead_h")["value"].mean().sort_index()
            ax.plot(agg.index, agg.values,
                    color=COLORS[model], label=model,
                    linewidth=1.8)

        ax.set_xlabel("Forecast Hour")
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 366)

        if metric == "MeanError":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Case 342: CA Christmas AR — 6-hr Precip vs MRMS\n"
        "Dec 20–27, 2025 | All available init times",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()

    fname = output_dir / "heavy_precip_continuous_metrics.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")
    return [str(fname)]


def plot_frequency_bias(df: pd.DataFrame, output_dir: Path) -> list[str]:
    """Plot Frequency Bias vs lead time for each threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fbias = df[df["metric"] == "FrequencyBias"].copy()
    fbias["lead_h"] = fbias["lead_time"].apply(parse_lead_hours)

    thresholds = ["tp_6hr_1mm", "tp_6hr_2.5mm", "tp_6hr_5mm", "tp_6hr_10mm", "tp_6hr_25mm"]
    labels = ["1 mm", "2.5 mm", "5 mm", "10 mm", "25 mm"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=300, sharey=True)

    for ax, thresh, label in zip(axes, thresholds, labels):
        sub = fbias[fbias["target_variable"] == thresh]
        for model in MODEL_ORDER:
            m = sub[sub["forecast_source"] == model]
            if len(m) == 0:
                continue
            agg = m.groupby("lead_h")["value"].mean().sort_index()
            ax.plot(agg.index, agg.values,
                    color=COLORS[model], label=model,
                    linewidth=1.5)

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xlabel("Forecast Hour", fontsize=9)
        ax.set_title(f"Threshold: {label}", fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 366)
        ax.set_ylim(0, max(5, sub["value"].quantile(0.95) * 1.1) if len(sub) > 0 else 5)

    axes[0].set_ylabel("Frequency Bias\n(1.0 = perfect)")
    axes[0].legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Case 342: CA Christmas AR — 6-hr Precip Frequency Bias vs MRMS\n"
        "Dec 20–27, 2025 | Bias > 1 = over-forecast, < 1 = under-forecast",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()

    fname = output_dir / "heavy_precip_frequency_bias.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")
    return [str(fname)]


def plot_skill_scores(df: pd.DataFrame, output_dir: Path) -> list[str]:
    """Plot ETS and CSI vs lead time for each threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = ["tp_6hr_1mm", "tp_6hr_2.5mm", "tp_6hr_5mm", "tp_6hr_10mm", "tp_6hr_25mm"]
    labels = ["1 mm", "2.5 mm", "5 mm", "10 mm", "25 mm"]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9), dpi=300, sharey="row")

    for row, metric_name, ylabel in [
        (0, "EquitableThreatScore", "ETS"),
        (1, "CriticalSuccessIndex", "CSI"),
    ]:
        sub_metric = df[df["metric"] == metric_name].copy()
        sub_metric["lead_h"] = sub_metric["lead_time"].apply(parse_lead_hours)

        for col, (thresh, label) in enumerate(zip(thresholds, labels)):
            ax = axes[row, col]
            sub = sub_metric[sub_metric["target_variable"] == thresh]

            for model in MODEL_ORDER:
                m = sub[sub["forecast_source"] == model]
                if len(m) == 0:
                    continue
                agg = m.groupby("lead_h")["value"].mean().sort_index()
                ax.plot(agg.index, agg.values,
                        color=COLORS[model], label=model,
                        linewidth=1.5)

            ax.set_xlim(0, 366)
            ax.set_ylim(-0.05, 0.7)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(f"{label}", fontweight="bold", fontsize=10)
            if row == 1:
                ax.set_xlabel("Forecast Hour", fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontweight="bold")

    axes[0, 0].legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Case 342: CA Christmas AR — 6-hr Precip Skill vs MRMS\n"
        "Dec 20–27, 2025 | ETS (top) and CSI (bottom) by threshold",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()

    fname = output_dir / "heavy_precip_skill_scores.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")
    return [str(fname)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="case_342_heavy_precip_results.csv")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    output_dir = Path(args.output_dir)

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Models: {df['forecast_source'].unique()}")
    print(f"Metrics: {df['metric'].unique()}")

    all_plots = []
    all_plots.extend(plot_continuous_metrics(df, output_dir))
    all_plots.extend(plot_frequency_bias(df, output_dir))
    all_plots.extend(plot_skill_scores(df, output_dir))

    print(f"\nGenerated {len(all_plots)} plots")
    return all_plots


if __name__ == "__main__":
    main()
