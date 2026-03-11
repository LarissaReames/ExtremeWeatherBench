#!/usr/bin/env python3
"""
Plot AR IVT snapshot maps with AR mask overlays for case 342.

Creates side-by-side panels showing IVT fields and AR land intersection masks
for multiple models vs ERA5 at a specific valid time and forecast lead time.

Usage:
    python plot_ar_maps.py --valid-time 2025-12-23T00 --lead-hours 48
    python plot_ar_maps.py --valid-time 2025-12-22T12 --lead-hours 72
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

# EWB's AR detection functions
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from extremeweatherbench.events.atmospheric_river import (
    integrated_vapor_transport,
    integrated_vapor_transport_laplacian,
    atmospheric_river_mask,
)
from extremeweatherbench import calc

# --- Config ---
ZARR_BASE = Path("/huge/proc/met-data/zarr")
OUTPUT_DIR = Path("/huge/proc/eva/ewb_case_342")

MODELS = ["WeatherMesh-4", "IFS", "AIFS", "GFS"]
MODEL_DISPLAY = {
    "WeatherMesh-4": "WeatherMesh-4",
    "IFS": "IFS (det)",
    "AIFS": "AIFS (det)",
    "GFS": "GFS (det)",
}

# Bounding box (case 342): lat 25-50N, lon 215-250E (145W-110W)
LAT_MIN, LAT_MAX = 25.0, 50.0
LON_MIN, LON_MAX = 215.0, 250.0

# Variable mappings per model in zarr
VAR_MAP = {
    "u": {"WeatherMesh-4": "u", "IFS": "u", "AIFS": "u", "GFS": "u", "ERA5": "u"},
    "v": {"WeatherMesh-4": "v", "IFS": "v", "AIFS": "v", "GFS": "v", "ERA5": "v"},
    "q": {"WeatherMesh-4": "q", "IFS": "q", "AIFS": "q", "GFS": "q", "ERA5": "q"},
}


def _load_global_uvq(zarr_path: Path, step_idx: int):
    """Load u, v, q on the full global grid at a specific step."""
    ds = xr.open_zarr(str(zarr_path), chunks=None, decode_times=False)
    ds_step = ds.isel(step=step_idx)

    u = ds_step["u"].rename({"lat": "latitude", "lon": "longitude"})
    v = ds_step["v"].rename({"lat": "latitude", "lon": "longitude"})
    q = ds_step["q"].rename({"lat": "latitude", "lon": "longitude"})
    ds_step = ds_step.rename({"lat": "latitude", "lon": "longitude"})

    return u, v, q, ds_step


def load_zarr_data(model: str, init_time_str: str, step_idx: int):
    """Load u, v, q from zarr archive on the full global grid."""
    zarr_path = ZARR_BASE / init_time_str / model / f"{model}_{init_time_str}.zarr"
    assert zarr_path.exists(), f"Zarr not found: {zarr_path}"
    return _load_global_uvq(zarr_path, step_idx)


def load_era5_data(valid_time_str: str):
    """Load ERA5 analysis (step=0) on the full global grid."""
    zarr_path = ZARR_BASE / valid_time_str / "ERA5" / f"ERA5_{valid_time_str}.zarr"
    assert zarr_path.exists(), f"ERA5 zarr not found: {zarr_path}"
    return _load_global_uvq(zarr_path, step_idx=0)


def compute_ivt_and_mask(u, v, q, levels):
    """Compute IVT and AR mask on the FULL global grid, then subset for plotting.

    The AR detection (Laplacian + dilation + min_size) must run on the full grid
    to match the evaluation pipeline. Subsetting first gives wrong results.
    """
    # Rename level dim if needed
    if "level" not in u.dims:
        for possible in ["isobaricInhPa", "pressure_level"]:
            if possible in u.dims:
                u = u.rename({possible: "level"})
                v = v.rename({possible: "level"})
                q = q.rename({possible: "level"})
                break

    if levels is None:
        levels = u.coords["level"]

    # Compute IVT on full global grid
    ivt = integrated_vapor_transport(
        specific_humidity=q,
        eastward_wind=u,
        northward_wind=v,
        levels=levels,
    )

    # Compute Laplacian on full global grid
    ivt_laplacian = integrated_vapor_transport_laplacian(ivt, sigma=3)

    # Compute AR mask on full global grid
    ar_mask_result = atmospheric_river_mask(
        ivt=ivt, ivt_laplacian=ivt_laplacian,
    )

    # Land intersection on full global grid
    land_intersection = calc.find_land_intersection(ar_mask_result)

    # NOW subset to bounding box for plotting
    lat = ivt.coords["latitude"].values
    lon = ivt.coords["longitude"].values
    lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)

    ivt_sub = ivt.sel(latitude=lat[lat_mask], longitude=lon[lon_mask])
    mask_sub = ar_mask_result.sel(latitude=lat[lat_mask], longitude=lon[lon_mask])
    land_sub = land_intersection.sel(latitude=lat[lat_mask], longitude=lon[lon_mask])

    return ivt_sub, mask_sub, land_sub


def plot_ivt_panels(
    data_dict: dict,
    valid_time: datetime,
    lead_hours: int,
    output_path: Path,
):
    """Create multi-panel IVT + AR mask map (3 per row).

    data_dict: {model_name: (ivt, ar_mask, land_intersect, lat, lon)}
    ERA5 is always first and labeled as "(Truth)".
    """
    n_panels = len(data_dict)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    # Aspect ratio ~35°lon × 25°lat → wider than tall
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes_flat = np.array(axes).flatten()

    # Hide unused axes
    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # IVT colormap: 0-800 kg/m/s
    ivt_levels = np.arange(0, 850, 50)
    ivt_cmap = plt.cm.YlOrRd
    ivt_norm = mcolors.BoundaryNorm(ivt_levels, ivt_cmap.N, extend="max")

    cf = None
    for ax, (model_name, (ivt, ar_mask, land_int, lat, lon)) in zip(
        axes_flat, data_dict.items()
    ):
        lon_plot = np.where(lon > 180, lon - 360, lon)

        cf = ax.contourf(
            lon_plot, lat, ivt.values,
            levels=ivt_levels, cmap=ivt_cmap, norm=ivt_norm,
            transform=ccrs.PlateCarree(), extend="max",
        )

        if ar_mask.values.max() > 0:
            ax.contour(
                lon_plot, lat, ar_mask.values,
                levels=[0.5], colors=["blue"], linewidths=2,
                transform=ccrs.PlateCarree(),
            )

        if land_int.values.max() > 0:
            ax.contour(
                lon_plot, lat, land_int.values,
                levels=[0.5], colors=["magenta"], linewidths=2.5,
                linestyles="dashed",
                transform=ccrs.PlateCarree(),
            )

        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
        ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":")
        ax.set_extent(
            [LON_MIN - 360, LON_MAX - 360, LAT_MIN, LAT_MAX],
            crs=ccrs.PlateCarree(),
        )

        # Mark ERA5 as truth
        display_name = MODEL_DISPLAY.get(model_name, model_name)
        if model_name == "ERA5":
            display_name = "ERA5 (Truth)"
            # Gold border to distinguish truth panel
            for spine in ax.spines.values():
                spine.set_edgecolor("#D4AF37")
                spine.set_linewidth(3)
        ax.set_title(display_name, fontsize=13, fontweight="bold")

    # Colorbar at bottom
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("IVT (kg/m/s)", fontsize=12)

    init_time = valid_time - timedelta(hours=lead_hours)
    fig.suptitle(
        f"Case 342: California Christmas AR — IVT + AR Mask\n"
        f"Valid: {valid_time.strftime('%Y-%m-%d %HZ')} | "
        f"Init: {init_time.strftime('%Y-%m-%d %HZ')} | "
        f"Lead: {lead_hours}h\n"
        f"Blue contour = AR mask | Magenta dashed = land intersection",
        fontsize=14, fontweight="bold",
    )

    fig.subplots_adjust(top=0.85, bottom=0.10, hspace=0.15, wspace=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot AR IVT maps for case 342")
    parser.add_argument("--valid-time", required=True, help="Valid time (YYYY-MM-DDTHH)")
    parser.add_argument("--lead-hours", type=int, default=48, help="Forecast lead time in hours")
    args = parser.parse_args()

    valid_time = datetime.strptime(args.valid_time, "%Y-%m-%dT%H")
    lead_hours = args.lead_hours
    init_time = valid_time - timedelta(hours=lead_hours)
    init_str = init_time.strftime("%Y%m%d%H")
    valid_str = valid_time.strftime("%Y%m%d%H")

    print(f"Valid time: {valid_time}")
    print(f"Init time:  {init_time}")
    print(f"Lead time:  {lead_hours}h")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_dict = {}

    # Load ERA5 first (ground truth)
    print("Loading ERA5...")
    try:
        u, v, q, ds = load_era5_data(valid_str)
        levels = ds["level"] if "level" in ds.coords else ds.coords.get("isobaricInhPa")
        ivt, ar_mask_result, land_int = compute_ivt_and_mask(u, v, q, levels)
        lat = ivt.coords["latitude"].values
        lon = ivt.coords["longitude"].values
        data_dict["ERA5"] = (ivt, ar_mask_result, land_int, lat, lon)
        print(f"  ERA5: IVT range [{float(ivt.min()):.0f}, {float(ivt.max()):.0f}] kg/m/s")
    except Exception as e:
        print(f"  ERA5 failed: {e}")

    # Load each model
    for model in MODELS:
        print(f"Loading {model}...")
        try:
            # Find the right step index for this lead time
            zarr_path = ZARR_BASE / init_str / model / f"{model}_{init_str}.zarr"
            if not zarr_path.exists():
                print(f"  {model}: zarr not found at {zarr_path}")
                continue

            ds_check = xr.open_zarr(str(zarr_path), chunks=None, decode_times=False)
            steps = ds_check.step.values  # float hours (e.g. 0, 6, 12, ...)
            step_matches = np.where(np.isclose(steps, lead_hours))[0]
            assert len(step_matches) > 0, f"No step matching {lead_hours}h in {model} (available: {steps[:5]}...)"
            step_idx = int(step_matches[0])

            u, v, q, ds_sub = load_zarr_data(model, init_str, step_idx)
            levels = ds_sub["level"] if "level" in ds_sub.coords else None
            ivt, ar_mask_result, land_int = compute_ivt_and_mask(u, v, q, levels)
            lat = ivt.coords["latitude"].values
            lon = ivt.coords["longitude"].values
            data_dict[model] = (ivt, ar_mask_result, land_int, lat, lon)
            print(f"  {model}: IVT range [{float(ivt.min()):.0f}, {float(ivt.max()):.0f}] kg/m/s")
        except Exception as e:
            print(f"  {model} failed: {e}")
            import traceback
            traceback.print_exc()

    if not data_dict:
        print("No data loaded!")
        return

    # Plot
    output_name = f"ar_ivt_map_{valid_time.strftime('%Y%m%d%H')}_{lead_hours}h.png"
    output_path = OUTPUT_DIR / output_name
    plot_ivt_panels(data_dict, valid_time, lead_hours, output_path)


if __name__ == "__main__":
    main()
