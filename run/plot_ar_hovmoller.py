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

# Coast-following band: approximate CA/OR/WA coastline as a set of (lat, lon) pairs
# and take IVT within COAST_BUFFER degrees of this line at each latitude.
# Longitudes in 0-360 convention (°E).
COAST_POINTS = np.array([
    # (lat, lon_E)  — approximate Pacific coast
    (48.5, 235.5),   # WA coast
    (47.0, 235.7),
    (46.0, 236.0),   # OR/WA border
    (44.0, 236.0),
    (43.0, 235.7),   # Southern OR
    (42.0, 235.6),   # OR/CA border
    (41.0, 235.8),
    (40.0, 235.7),   # Cape Mendocino
    (38.0, 237.0),   # Point Reyes
    (37.0, 237.5),   # SF
    (36.0, 238.3),   # Monterey
    (35.0, 239.3),   # Point Conception bend
    (34.5, 240.0),
    (34.0, 241.3),   # LA
    (33.5, 242.0),   # San Diego area
    (33.0, 242.5),
    (32.5, 243.0),   # US/Mexico border
    (30.0, 244.0),   # Northern Baja
    (28.0, 245.0),   # Mid Baja
    (26.0, 247.0),   # Southern Baja (tip)
    (25.0, 248.0),
])
COAST_BUFFER = 1.5  # degrees from coast to include

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


def _coast_lon_at_lat(lat: float) -> float:
    """Interpolate coastline longitude (°E) at a given latitude."""
    coast_lats = COAST_POINTS[:, 0]
    coast_lons = COAST_POINTS[:, 1]
    return float(np.interp(lat, coast_lats[::-1], coast_lons[::-1]))


def _extract_coast_band(ivt: xr.DataArray) -> xr.DataArray:
    """Extract max IVT within COAST_BUFFER of the coastline at each latitude.

    Returns DataArray with the longitude dimension removed.
    """
    lats = ivt.latitude.values
    lons = ivt.longitude.values
    vals = ivt.values  # (..., lat, lon)

    # Build mask: for each lat, only keep lons within COAST_BUFFER of coastline
    lat_idx = ivt.dims.index("latitude")
    lon_idx = ivt.dims.index("longitude")

    # Pre-compute coast longitude for each latitude
    coast_lons = np.array([_coast_lon_at_lat(lat) for lat in lats])

    # Create 2D mask (lat, lon)
    lon_grid = lons[np.newaxis, :]  # (1, nlon)
    coast_grid = coast_lons[:, np.newaxis]  # (nlat, 1)
    mask = np.abs(lon_grid - coast_grid) <= COAST_BUFFER  # (nlat, nlon)

    # Apply mask and take max along longitude
    # Handle arbitrary leading dims by broadcasting
    masked = np.where(mask, vals, np.nan)
    result = np.nanmax(masked, axis=lon_idx)

    # Build new DataArray without the longitude dim
    new_dims = [d for d in ivt.dims if d != "longitude"]
    new_coords = {k: v for k, v in ivt.coords.items() if "longitude" not in v.dims and k != "longitude"}
    return xr.DataArray(result, dims=new_dims, coords=new_coords)


def load_era5_hovmoller(cache_dir: Path) -> xr.DataArray:
    """Load ERA5 truth IVT and compute coast-following band max."""
    era5_path = cache_dir / "case_342" / "ERA5" / "derived.nc"
    assert era5_path.exists(), f"ERA5 cache not found: {era5_path}"

    ds = xr.open_dataset(era5_path)
    ivt = ds["integrated_vapor_transport"]
    return _extract_coast_band(ivt)


def load_model_cache(cache_dir: Path, model_name: str) -> xr.Dataset | None:
    """Load the full cached derived dataset for a model."""
    model_path = cache_dir / "case_342" / model_name / "derived.nc"
    if not model_path.exists():
        print(f"  Cache not found for {model_name}: {model_path}")
        return None
    return xr.open_dataset(model_path)


def extract_init_hovmoller(ds: xr.Dataset, init_time: np.datetime64) -> xr.DataArray | None:
    """Extract IVT coast-following band max Hovmöller for a single init time.

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
        return _extract_coast_band(ivt)

    lead_times = ivt.lead_time.values
    expected_valid_times = init_time + lead_times

    # Find which valid_times in the dataset match this init
    available_valid = ivt.valid_time.values
    results = []
    result_valid_times = []

    for lt, expected_vt in zip(lead_times, expected_valid_times):
        if expected_vt in available_valid:
            slice_data = ivt.sel(lead_time=lt, valid_time=expected_vt)
            # slice_data has dims (latitude, longitude)
            coast_band = _extract_coast_band(slice_data)
            results.append(coast_band.values)
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
    n_panels = 1 + n_models  # ERA5 + model diffs
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.4 * nrows),
                              dpi=300, squeeze=False)

    # IVT levels for truth panel
    ivt_levels = np.arange(0, 850, 50)
    cmap_ivt = "YlOrRd"

    # Subset ERA5 to the event window relevant to this init
    end_time = init_time + np.timedelta64(360, "h")
    event_end = np.datetime64("2025-12-27T00")
    display_end = min(end_time, event_end)
    era5_sub = era5_hov.sel(valid_time=slice(init_time, display_end))

    times_era5 = era5_sub.valid_time.values
    lats = era5_sub.latitude.values

    # Panel 0: ERA5 truth
    ax_era5 = axes[0][0]
    T_e, L_e = np.meshgrid(times_era5, lats)
    cf_era5 = ax_era5.contourf(T_e, L_e, era5_sub.values.T, levels=ivt_levels,
                                 cmap=cmap_ivt, extend="max")
    # No per-panel colorbar — ERA5 gets a shared one at the bottom too
    ax_era5.set_title("ERA5 (Truth)", fontweight="bold", fontsize=10)
    ax_era5.set_ylabel("Latitude (°N)")
    for spine in ax_era5.spines.values():
        spine.set_edgecolor("#DAA520")
        spine.set_linewidth(2.5)

    # Difference panels (model - ERA5)
    diff_levels = np.arange(-400, 450, 50)
    diff_levels = diff_levels[diff_levels != 0]
    cmap_diff = "RdBu_r"
    norm_diff = mcolors.TwoSlopeNorm(vmin=-400, vcenter=0, vmax=400)

    last_cf_diff = None
    for i, (name, hov) in enumerate(model_hovs.items()):
        panel_idx = i + 1  # ERA5 is panel 0
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row][col]

        # Compute difference: model - ERA5 at matched valid times
        model_times = hov.valid_time.values
        era5_at_model = era5_hov.sel(valid_time=model_times, method="nearest")
        diff = hov.values - era5_at_model.values

        T_m, L_m = np.meshgrid(model_times, lats)
        cf_diff = ax.contourf(T_m, L_m, diff.T, levels=diff_levels,
                                cmap=cmap_diff, norm=norm_diff, extend="both")
        last_cf_diff = cf_diff

        display = MODEL_DISPLAY.get(name, {"label": name})
        fhr_min = int((model_times[0] - init_time) / np.timedelta64(1, "h"))
        fhr_max = int((model_times[-1] - init_time) / np.timedelta64(1, "h"))
        ax.set_title(f"{display['label']} − ERA5\nF{fhr_min:03d}–F{fhr_max:03d}",
                      fontweight="bold", fontsize=9)

        if col == 0:
            ax.set_ylabel("Lat (°N)", fontsize=9)

    # Format all axes: hide x-axis labels on non-bottom rows to prevent overlap
    for idx in range(n_panels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        is_bottom = (row == nrows - 1) or not axes[row + 1][col].get_visible() if row < nrows - 1 else True
        if is_bottom:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        else:
            ax.tick_params(labelbottom=False)

    # Add city labels on the right y-axis of rightmost column panels
    CITIES = {
        32.7: "San Diego",
        34.0: "Los Angeles",
        37.8: "San Francisco",
        45.5: "Portland",
        47.6: "Seattle",
    }
    for row_idx in range(nrows):
        rightmost_col = ncols - 1
        # Find the rightmost visible panel in this row
        while rightmost_col >= 0:
            panel_num = row_idx * ncols + rightmost_col
            if panel_num < n_panels:
                break
            rightmost_col -= 1
        if rightmost_col < 0:
            continue
        ax = axes[row_idx][rightmost_col]
        ax2 = ax.twinx()
        city_lats = sorted(CITIES.keys())
        city_names = [CITIES[lat] for lat in city_lats]
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(city_lats)
        ax2.set_yticklabels(city_names, fontsize=7, color="gray")
        ax2.tick_params(axis="y", length=3, width=0.5, colors="gray")

    # Hide unused subplots
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Colorbars: IVT truth on left side (vertical), diff at bottom (horizontal)
    fig.subplots_adjust(left=0.14, bottom=0.12, hspace=0.30)

    # ERA5 IVT colorbar — vertical, tight to left of ERA5 panel
    # Labels and title on the LEFT side of the colorbar
    era5_pos = axes[0][0].get_position()
    cbar_ax_ivt = fig.add_axes([era5_pos.x0 - 0.06, era5_pos.y0, 0.012, era5_pos.height])
    cb_ivt = plt.colorbar(cf_era5, cax=cbar_ax_ivt, orientation="vertical")
    cb_ivt.ax.yaxis.set_ticks_position("left")
    cb_ivt.ax.yaxis.set_label_position("left")
    cb_ivt.set_label("IVT (kg/m/s)", fontsize=8)
    cb_ivt.ax.tick_params(labelsize=7)

    # Difference colorbar — horizontal, centered at bottom (nudged down for clearance)
    if last_cf_diff is not None:
        cbar_ax_diff = fig.add_axes([0.25, 0.01, 0.5, 0.015])
        cb_diff = plt.colorbar(last_cf_diff, cax=cbar_ax_diff, orientation="horizontal")
        cb_diff.set_label("IVT Difference (kg/m/s)", fontsize=9)

    fig.suptitle(
        f"Hovmöller: IVT Coast-Following Band Max — Init {init_label}\n"
        f"±{COAST_BUFFER:.1f}° from Pacific coastline | "
        f"Case 342: CA Christmas AR",
        fontsize=12, fontweight="bold", y=0.99,
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
