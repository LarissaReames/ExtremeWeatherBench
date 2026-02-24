#!/usr/bin/env python3
"""
Forecast-hour vs. init-time heatmaps for non-event-based metrics.

Produces two figures per (metric, variable) combination:
  A) Raw heatmaps: one subplot per model, shared sequential colorbar.
  B) Difference heatmaps: IFS-Ens-Mean raw as first panel, then
     (model − IFS-Ens-Mean) for every other model, shared diverging colorbar.

IFS-Ens-Mean is always the first (leftmost) panel so ordering is consistent.

Orientation: bottom-left is closest to the actual event — forecast hour
increases upward (short lead at bottom), init time increases leftward
(latest init at left).

Usage:
    python plot_heatmaps.py results.csv
    python plot_heatmaps.py results.csv --output-dir plots_heatmap
    python plot_heatmaps.py results.csv --reference-model "IFS-Ens-Mean"
    python plot_heatmaps.py results.csv --metrics RootMeanSquaredError MeanAbsoluteError
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


REFERENCE_MODEL = "IFS-Ens-Mean"


def parse_lead_time_to_hours(lt_series: pd.Series) -> pd.Series:
    """Convert lead_time strings like '2 days 12:00:00' to float hours."""
    td = pd.to_timedelta(lt_series)
    return td.dt.total_seconds() / 3600.0


def pivot_to_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a single-model/metric slice into (forecast_hour x init_time) matrix.

    Returns a DataFrame with forecast hours as index (ascending, short lead
    at bottom of plot) and init_time timestamps as columns sorted so that
    the latest init time is on the LEFT (descending).
    """
    sub = df.copy()
    sub["forecast_hour"] = parse_lead_time_to_hours(sub["lead_time"])
    sub["init_time"] = pd.to_datetime(sub["init_time"])

    pivoted = sub.pivot_table(
        index="forecast_hour",
        columns="init_time",
        values="value",
        aggfunc="mean",
    )
    # Ascending forecast hour (short lead at row 0 → bottom of plot)
    # Descending init time (latest date at col 0 → left of plot)
    return pivoted.sort_index(ascending=True).sort_index(axis=1, ascending=False)


def _model_sort_key(model: str, reference: str) -> tuple[int, str]:
    """Sort key that puts the reference model first, rest alphabetical."""
    if model == reference:
        return (0, model)
    return (1, model)


def _pretty_name(name: str) -> str:
    """CamelCase or snake_case → readable label."""
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    name = name.replace("_", " ")
    return name.strip()


def _setup_axes(ax, piv, col_index, is_left_col=False):
    """Configure x/y ticks and labels for a heatmap panel.

    Uses date-aware x-axis ticking and clean forecast-hour y labels.
    """
    # --- Y-axis: forecast hours ---
    hours = piv.index.values
    n_y = len(hours)
    ytick_step = max(1, n_y // 20)
    ytick_positions = list(range(0, n_y, ytick_step))
    ax.set_yticks([p + 0.5 for p in ytick_positions])
    ax.set_yticklabels([f"{int(hours[p])}h" for p in ytick_positions], fontsize=8)
    ax.set_ylabel("Forecast hour", fontsize=9)

    # --- X-axis: init times (descending — latest on left) ---
    timestamps = piv.columns  # already sorted descending
    n_x = len(timestamps)

    date_range_days = (timestamps.max() - timestamps.min()).days
    if date_range_days > 14:
        target_labels = 6
    elif date_range_days > 7:
        target_labels = 8
    else:
        target_labels = min(n_x, 10)

    step = max(1, n_x // target_labels)
    xtick_positions = list(range(0, n_x, step))

    # Use "Jun 15" style for multi-day ranges, add "12Z" only if sub-daily inits
    has_sub_daily = len(set(timestamps.hour)) > 1
    if has_sub_daily:
        fmt = "%b %d\n%HZ"
    else:
        fmt = "%b %d"

    ax.set_xticks([p + 0.5 for p in xtick_positions])
    ax.set_xticklabels(
        [timestamps[p].strftime(fmt) for p in xtick_positions],
        fontsize=7.5,
        ha="center",
    )
    ax.set_xlabel("Init time →  earlier", fontsize=9)


def plot_raw_heatmaps(
    df: pd.DataFrame,
    metric: str,
    variable: str,
    reference_model: str,
    output_dir: Path,
    case_label: str = "",
):
    """Figure A: raw metric heatmaps, one subplot per model, shared colorbar."""
    models = sorted(
        df["forecast_source"].unique(),
        key=lambda m: _model_sort_key(m, reference_model),
    )
    n_models = len(models)

    pivots = {}
    for model in models:
        model_df = df[df["forecast_source"] == model]
        if model_df.empty:
            continue
        pivots[model] = pivot_to_heatmap(model_df)

    if not pivots:
        return

    vmin = min(p.min().min() for p in pivots.values())
    vmax = max(p.max().max() for p in pivots.values())

    ncols = min(n_models, 4)
    nrows = (n_models + ncols - 1) // ncols
    fig_width = 5.5 * ncols + 1.5
    fig_height = 5 * nrows + 1.8

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.06, right=0.88, top=0.88, bottom=0.12,
        wspace=0.30, hspace=0.40,
    )

    cmap = plt.cm.YlOrRd

    for idx, model in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        piv = pivots[model]

        im = ax.pcolormesh(
            np.arange(piv.shape[1] + 1),
            np.arange(piv.shape[0] + 1),
            piv.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )

        _setup_axes(ax, piv, col, is_left_col=(col == 0))
        ax.set_title(model, fontsize=11, fontweight="bold")

    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02, aspect=30,
    )
    cbar.set_label(f"{_pretty_name(metric)} ({_pretty_name(variable)})", fontsize=10)

    title = f"{_pretty_name(metric)} – {_pretty_name(variable)}"
    if case_label:
        title = f"{case_label}\n{title}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    safe_metric = metric.lower().replace(" ", "_")
    safe_var = variable.replace(" ", "_")
    out_path = output_dir / f"heatmap_raw_{safe_metric}_{safe_var}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_diff_heatmaps(
    df: pd.DataFrame,
    metric: str,
    variable: str,
    reference_model: str,
    output_dir: Path,
    case_label: str = "",
    diff_percentile: float = 90,
):
    """Figure B: IFS-Ens-Mean raw as panel 1, then (model − ref) for the rest."""
    models = sorted(
        df["forecast_source"].unique(),
        key=lambda m: _model_sort_key(m, reference_model),
    )

    if reference_model not in models:
        print(f"  WARNING: reference model '{reference_model}' not in data, skipping diff plot")
        return

    pivots = {}
    for model in models:
        model_df = df[df["forecast_source"] == model]
        if model_df.empty:
            continue
        pivots[model] = pivot_to_heatmap(model_df)

    if reference_model not in pivots:
        return

    ref_pivot = pivots[reference_model]
    other_models = [m for m in models if m != reference_model]

    diffs = {}
    for model in other_models:
        aligned_ref, aligned_model = ref_pivot.align(pivots[model], join="inner")
        diffs[model] = aligned_model - aligned_ref

    if not diffs:
        return

    n_panels = 1 + len(other_models)
    ncols = min(n_panels, 4)
    nrows = (n_panels + ncols - 1) // ncols
    fig_width = 5.5 * ncols + 1.5
    fig_height = 5 * nrows + 1.8

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.06, right=0.88, top=0.88, bottom=0.12,
        wspace=0.30, hspace=0.40,
    )

    # --- Panel 0: reference model raw values ---
    ax0 = axes[0, 0]
    ref_cmap = plt.cm.YlOrRd
    ref_vmin = ref_pivot.min().min()
    ref_vmax = ref_pivot.max().max()

    im_ref = ax0.pcolormesh(
        np.arange(ref_pivot.shape[1] + 1),
        np.arange(ref_pivot.shape[0] + 1),
        ref_pivot.values,
        cmap=ref_cmap,
        vmin=ref_vmin,
        vmax=ref_vmax,
        shading="flat",
    )
    _setup_axes(ax0, ref_pivot, 0, is_left_col=True)
    ax0.set_title(f"{reference_model}\n(reference)", fontsize=11, fontweight="bold")

    cb_ref = fig.colorbar(im_ref, ax=ax0, shrink=0.85, pad=0.02)
    cb_ref.set_label(_pretty_name(metric), fontsize=9)

    # --- Remaining panels: difference heatmaps ---
    # Use a percentile-based clipping so a few extreme values don't wash out
    # the structure in other models.  Values beyond the clip show as arrows.
    all_diff_vals = np.concatenate([d.values.ravel() for d in diffs.values()])
    all_diff_vals = all_diff_vals[np.isfinite(all_diff_vals)]
    diff_clip = np.percentile(np.abs(all_diff_vals), diff_percentile)
    diff_clip = max(diff_clip, 1e-6)  # avoid degenerate zero range
    diff_norm = mcolors.TwoSlopeNorm(vmin=-diff_clip, vcenter=0, vmax=diff_clip)
    diff_cmap = plt.cm.RdBu_r

    diff_axes = []
    for panel_idx, model in enumerate(other_models, start=1):
        row, col = divmod(panel_idx, ncols)
        ax = axes[row, col]
        diff_axes.append(ax)
        d = diffs[model]

        im_diff = ax.pcolormesh(
            np.arange(d.shape[1] + 1),
            np.arange(d.shape[0] + 1),
            np.clip(d.values, -diff_clip, diff_clip),
            cmap=diff_cmap,
            norm=diff_norm,
            shading="flat",
        )

        _setup_axes(ax, d, col, is_left_col=(col == 0))
        ax.set_title(model, fontsize=11, fontweight="bold")

    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    if diff_axes:
        sm = plt.cm.ScalarMappable(cmap=diff_cmap, norm=diff_norm)
        sm.set_array([])
        cb_diff = fig.colorbar(
            sm, ax=diff_axes, shrink=0.8, pad=0.02, aspect=30,
            extend="both",
        )
        cb_diff.set_label(
            f"Δ {_pretty_name(metric)} (positive = model worse)", fontsize=10,
        )

    title = (
        f"Δ {_pretty_name(metric)} from {reference_model} – "
        f"{_pretty_name(variable)}"
    )
    if case_label:
        title = f"{case_label}\n{title}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    safe_metric = metric.lower().replace(" ", "_")
    safe_var = variable.replace(" ", "_")
    out_path = output_dir / f"heatmap_diff_{safe_metric}_{safe_var}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Forecast-hour vs. init-time heatmaps for non-event-based metrics"
    )
    parser.add_argument("csv", type=Path, help="Results CSV file")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: plots_heatmap/ next to CSV)",
    )
    parser.add_argument(
        "--reference-model", type=str, default=REFERENCE_MODEL,
        help=f"Reference model for difference plots (default: {REFERENCE_MODEL})",
    )
    parser.add_argument(
        "--metrics", nargs="*", default=None,
        help="Metrics to plot (default: auto-detect non-event metrics with both lead_time and init_time)",
    )
    parser.add_argument(
        "--diff-percentile", type=float, default=90,
        help="Percentile of |diff| values to set colorbar endpoints (default: 90). "
             "Values beyond this are clipped and shown with extend arrows.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    output_dir = args.output_dir or args.csv.parent / "plots_heatmap"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics:
        target_metrics = args.metrics
    else:
        target_metrics = []
        for m in df["metric"].unique():
            sub = df[df["metric"] == m]
            if sub["lead_time"].notna().any() and sub["init_time"].notna().any():
                target_metrics.append(m)
        print(f"Auto-detected metrics with (lead_time, init_time): {target_metrics}")

    if not target_metrics:
        print("No plottable metrics found.")
        return

    case_label = args.csv.stem.replace("_results", "").replace("_", " ").title()

    for metric in target_metrics:
        for variable in df.loc[df["metric"] == metric, "target_variable"].unique():
            sub = df[(df["metric"] == metric) & (df["target_variable"] == variable)]
            sub = sub.dropna(subset=["lead_time", "init_time", "value"])

            if sub.empty:
                print(f"  Skipping {metric}/{variable}: no valid data")
                continue

            print(f"\n{'='*60}")
            print(f"  {metric} / {variable}: {len(sub)} data points")
            print(f"  Models: {sorted(sub['forecast_source'].unique())}")
            print(f"{'='*60}")

            plot_raw_heatmaps(
                sub, metric, variable, args.reference_model, output_dir,
                case_label=case_label,
            )
            plot_diff_heatmaps(
                sub, metric, variable, args.reference_model, output_dir,
                case_label=case_label,
                diff_percentile=args.diff_percentile,
            )


if __name__ == "__main__":
    main()
