#!/usr/bin/env python3
"""
Hovmöller diagrams for AR Case 342 — IVT along the California coast over time.

For a single init time, shows how each model's IVT forecast evolves along the coast
as forecast hour progresses. ERA5 panel shows truth, model panels show difference
(model - ERA5) to highlight timing/intensity errors.

Uses cached derived variables from evaluate_case_2025.py --cache-derived.

Usage:
    python plot_ar_hovmoller.py --init-time 2025-12-20T00
    python plot_ar_hovmoller.py --init-time 2025-12-21T12 --cache-dir /huge/proc/eva/ewb_cache
"""
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# Coastal band for California: ~235-240°E (125-120°W)
COASTAL_LON_MIN = 235.0
COASTAL_LON_MAX = 240.0

OUTPUT_DIR = Path("/huge/proc/eva/ewb_case_342")

MODEL_DISPLAY = {
    "ERA5": {"label": "ERA5 (Truth)"},
    "WeatherMesh-4": {"label": "WeatherMesh-4"},
    "IFS": {"label": "IFS"},
    "AIFS": {"label": "AIFS"},
    "GFS": {"label": "GFS"},
    "GFS-Ens-Mean": {"label": "GFS-Ens-Mean"},
}

MODEL_ORDER = ["WeatherMesh-4", "IFS", "AIFS", "GFS", "GFS-Ens-Mean"]


def load_era5_hovmoller(cache_dir: Path) -> xr.DataArray:
    """Load ERA5 truth IVT and compute coastal band max."""
    era5_path = cache_dir / "case_342" / "ERA5" / "derived.nc"
    assert era5_path.exists(), f"ERA5 cache not found: {era5_path}"

    ds = xr.open_dataset(era5_path)
    ivt = ds["integrated_vapor_transport"]

    # Subset to coastal band and take max along longitude
    coastal = ivt.sel(longitude=slice(COASTAL_LON_MIN, COASTAL_LON_MAX))
    hovmoller = coastal.max(dim="longitude")
    return hovmoller


def load_model_cache(cache_dir: Path, model_name: str) -> xr.Dataset | None:
    """Load the full cached derived dataset for a model."""
    model_path = cache_dir / "case_342" / model_name / "derived.nc"
    if not model_path.exists():
        print(f"  Cache not found for {model_name}: {model_path}")
        return None
    return xr.open_dataset(model_path)


def extract_init_hovmoller(ds: xr.Dataset, init_time: np.datetime64) -> xr.DataArray | None:
    """Extract IVT coastal-band max Hovmöller for a single init time.

    The cached model data has dims (lead_time, valid_time, latitude, longitude).
    For a given init time, valid_time = init_time + lead_time.
    We select valid_times that match this init's lead times.

    Returns DataArray with dims (valid_time, latitude).
    """
    ivt = ds["integrated_vapor_transport"]

    # The cached data has valid_time and lead_time dims
    # valid_time for a given init is: init_time + lead_time
    if "lead_time" not in ivt.dims:
        # ERA5 — no lead_time dimension, just return the full valid_time series
        coastal = ivt.sel(longitude=slice(COASTAL_LON_MIN, COASTAL_LON_MAX))
        return coastal.max(dim="longitude")

    lead_times = ivt.lead_time.values
    expected_valid_times = init_time + lead_times

    # Find which valid_times in the dataset match this init
    available_valid = ivt.valid_time.values
    results = []
    result_valid_times = []

    for lt, expected_vt in zip(lead_times, expected_valid_times):
        # Check if this valid_time exists in the dataset
        if expected_vt in available_valid:
            slice_data = ivt.sel(lead_time=lt, valid_time=expected_vt)
            coastal = slice_data.sel(longitude=slice(COASTAL_LON_MIN, COASTAL_LON_MAX))
            results.append(coastal.max(dim="longitude").values)
            result_valid_times.append(expected_vt)

    if len(results) == 0:
        return None

    lat = ivt.latitude.values
    data = np.stack(results, axis=0)  # (time, lat)
    return xr.DataArray(
        data,
        coords={"valid_time": result_valid_times, "latitude": lat},
        dims=("valid_time", "latitude"),
    )


def plot_init_hovmoller(cache_dir: Path, init_time_str: str, output_dir: Path = OUTPUT_DIR):
    """Plot Hovmöller for a single init time: ERA5 truth + model differences."""
    output_dir.mkdir(parents=True, exist_ok=True)

    init_time = np.datetime64(init_time_str)
    init_dt = datetime.fromisoformat(init_time_str.replace("T", " ").replace("Z", ""))
    init_label = init_dt.strftime("%Y-%m-%d %HZ")

    # Load ERA5 truth
    print("Loading ERA5...")
    era5_hov = load_era5_hovmoller(cache_dir)

    # Load available models
    model_hovs = {}
    for name in MODEL_ORDER:
        print(f"Loading {name}...")
        ds = load_model_cache(cache_dir, name)
        if ds is None:
            continue
        hov = extract_init_hovmoller(ds, init_time)
        if hov is not None and len(hov.valid_time) > 2:
            model_hovs[name] = hov
        else:
            print(f"  {name}: no data for init {init_label} (or too few valid times)")

    if not model_hovs:
        print("No model data available for this init time!")
        return None

    n_models = len(model_hovs)
    # Layout: ERA5 on top row (full width), model diffs below (3 per row)
    ncols = 3
    nrows_models = (n_models + ncols - 1) // ncols

    fig = plt.figure(figsize=(5.5 * ncols, 3.5 + 3.5 * nrows_models), dpi=300)

    # ERA5 truth panel on top (spans full width)
    ax_era5 = fig.add_axes([0.08, 0.55 + 0.02 * nrows_models,
                             0.82, 0.30 / (1 + 0.3 * nrows_models)])

    # IVT levels for truth panel
    ivt_levels = np.arange(0, 850, 50)
    cmap_ivt = "YlOrRd"

    # Subset ERA5 to the event window relevant to this init
    # Show from init time to init + 360h
    end_time = init_time + np.timedelta64(360, "h")
    event_end = np.datetime64("2025-12-27T00")
    display_end = min(end_time, event_end)
    era5_sub = era5_hov.sel(valid_time=slice(init_time, display_end))

    times_era5 = era5_sub.valid_time.values
    lats = era5_sub.latitude.values
    T_e, L_e = np.meshgrid(times_era5, lats)
    cf_era5 = ax_era5.contourf(T_e, L_e, era5_sub.values.T, levels=ivt_levels,
                                 cmap=cmap_ivt, extend="max")
    cb_era5 = plt.colorbar(cf_era5, ax=ax_era5, shrink=0.8, pad=0.02)
    cb_era5.set_label("IVT (kg/m/s)", fontsize=9)
    ax_era5.set_title("ERA5 (Truth)", fontweight="bold", fontsize=11)
    ax_era5.set_ylabel("Latitude (°N)")
    ax_era5.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %HZ"))
    ax_era5.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    plt.setp(ax_era5.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    for spine in ax_era5.spines.values():
        spine.set_edgecolor("#DAA520")
        spine.set_linewidth(2.5)

    # Difference panels (model - ERA5)
    diff_levels = np.arange(-400, 450, 50)
    diff_levels = diff_levels[diff_levels != 0]
    cmap_diff = "RdBu_r"
    norm_diff = mcolors.TwoSlopeNorm(vmin=-400, vcenter=0, vmax=400)

    # Create model subplot axes
    left_margin = 0.08
    right_margin = 0.92
    bottom = 0.08
    top = 0.50
    hspace = 0.12
    wspace = 0.08

    panel_w = (right_margin - left_margin - (ncols - 1) * wspace) / ncols
    panel_h = (top - bottom - (nrows_models - 1) * hspace) / nrows_models

    last_cf_diff = None
    for i, (name, hov) in enumerate(model_hovs.items()):
        row = i // ncols
        col = i % ncols
        x0 = left_margin + col * (panel_w + wspace)
        y0 = top - (row + 1) * panel_h - row * hspace
        ax = fig.add_axes([x0, y0, panel_w, panel_h])

        # Compute difference: model - ERA5 at matched valid times
        model_times = hov.valid_time.values
        era5_at_model = era5_hov.sel(valid_time=model_times, method="nearest")

        diff = hov.values - era5_at_model.values

        T_m, L_m = np.meshgrid(model_times, lats)
        cf_diff = ax.contourf(T_m, L_m, diff.T, levels=diff_levels,
                                cmap=cmap_diff, norm=norm_diff, extend="both")
        last_cf_diff = cf_diff

        display = MODEL_DISPLAY.get(name, {"label": name})
        # Compute forecast hour range for this model
        fhr_min = int((model_times[0] - init_time) / np.timedelta64(1, "h"))
        fhr_max = int((model_times[-1] - init_time) / np.timedelta64(1, "h"))
        ax.set_title(f"{display['label']} − ERA5\nF{fhr_min:03d}–F{fhr_max:03d}",
                      fontweight="bold", fontsize=9)

        if col == 0:
            ax.set_ylabel("Lat (°N)", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # Colorbar for differences
    if last_cf_diff is not None:
        cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
        cb = plt.colorbar(last_cf_diff, cax=cbar_ax, orientation="horizontal")
        cb.set_label("IVT Difference (kg/m/s)", fontsize=9)

    fig.suptitle(
        f"Hovmöller: IVT Coastal-Band Max — Init {init_label}\n"
        f"Lon band: {COASTAL_LON_MIN-360:.0f}°W to {COASTAL_LON_MAX-360:.0f}°W | "
        f"Case 342: CA Christmas AR",
        fontsize=12, fontweight="bold", y=0.98,
    )

    fname = output_dir / f"ar_hovmoller_init_{init_time_str.replace(':', '').replace('T', '_')}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
    return str(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AR Hovmöller diagrams — single init time")
    parser.add_argument("--init-time", type=str, required=True,
                        help="Init time in ISO format, e.g. 2025-12-20T00")
    parser.add_argument("--cache-dir", type=str, default="/huge/proc/eva/ewb_cache",
                        help="Path to cached derived variables")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for plots")
    args = parser.parse_args()

    plot_init_hovmoller(
        cache_dir=Path(args.cache_dir),
        init_time_str=args.init_time,
        output_dir=Path(args.output_dir),
    )
