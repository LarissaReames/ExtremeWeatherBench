#!/usr/bin/env python
"""
Plot TC forecast tracks and SLP fields for any tropical cyclone case.

Creates:
  1. Per-model track overview PNGs (all init times on one map)
  2. Per-init multi-model track PNGs (all models on one map, no MSLP)
  3. (Optional, --slp-pdfs) Per-model per-init multi-page PDFs with SLP
     contour fills, black contour lines, minimum MSLP labels, full
     forecast track, and observed track on every frame.

Usage:
    python plot_hurricane_tracks.py <case_id>
    python plot_hurricane_tracks.py 236  # Hurricane Laura
    python plot_hurricane_tracks.py 338  # Hurricane Melissa
    python plot_hurricane_tracks.py 338 --slp-pdfs  # also generate SLP PDFs
"""
import warnings

warnings.filterwarnings("ignore")

import argparse
import importlib.resources
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Use a non-interactive backend for batch plot generation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import yaml

from extremeweatherbench import calc

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

TRACKS_DIR = Path("/huge/users/larissa/ExtremeWeatherBench/run")
OUTPUT_BASE = Path("/huge/users/larissa/ExtremeWeatherBench/run/tc_track_plots")

# Model configurations by year/data source
# For 2020 cases (like Laura, case 236)
MODEL_SOURCES_2020 = {
    "WeatherMesh-4": {
        "source": "/huge/proc/weathermesh4_2020.zarr",
        "storage_options": {},
    },
    "HRES": {
        "source": "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        "storage_options": {"remote_options": {"anon": True}},
    },
    "GraphCast": {
        "source": (
            "gs://weatherbench2/datasets/graphcast/2020/"
            "date_range_2019-11-16_2021-02-01_12_hours.zarr"
        ),
        "storage_options": {"remote_options": {"anon": True}},
    },
    "Pangu-Weather": {
        "source": "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
        "storage_options": {"remote_options": {"anon": True}},
    },
    "GenCast": {
        "source": "gs://weatherbench2/datasets/gencast/2020-1440x721_mean.zarr",
        "storage_options": {"remote_options": {"anon": True}},
    },
}

# For 2025 cases (like Melissa, case 338)
# These use per-init zarr files: /huge/proc/met-data/zarr/{YYYYMMDDHH}/{MODEL}/{MODEL}_{YYYYMMDDHH}.zarr
MODEL_SOURCES_2025 = {
    "WeatherMesh-4": {
        "source": "/huge/proc/met-data/zarr",
        "per_init": True,
    },
    "WeatherMesh-4p5-Ens-Mean": {
        "source": "/huge/proc/met-data/zarr",
        "per_init": True,
    },
    "IFS-Ens-Mean": {
        "source": "/huge/proc/met-data/zarr",
        "per_init": True,
    },
    "AIFS-Ens-Mean": {
        "source": "/huge/proc/met-data/zarr",
        "per_init": True,
    },
    "GFS-Ens-Mean": {
        "source": "/huge/proc/met-data/zarr",
        "per_init": True,
    },
}


# ── Model visual styling (consistent with analyze_results.py) ─────────────────
# Physical ensemble models: blue tones, solid lines
# WeatherMesh variants: black/gray, solid thick lines
# AI models: varied warm colours, dashed lines
MODEL_STYLES = {
    # Physical ensembles
    "IFS-Ens-Mean":              {"color": "#0173B2", "marker": "o", "linestyle": "-",  "linewidth": 2.5, "zorder": 4},
    "GFS-Ens-Mean":              {"color": "#56B4E9", "marker": "s", "linestyle": "-",  "linewidth": 2.5, "zorder": 4},
    "HRES":                      {"color": "#0173B2", "marker": "o", "linestyle": "-",  "linewidth": 2.5, "zorder": 4},
    # WeatherMesh – bold red/orange, thick, high zorder to pop (observed track is black)
    "WeatherMesh-4":             {"color": "#D62728", "marker": "D", "linestyle": "-",  "linewidth": 4.0, "zorder": 10},
    "WeatherMesh-4p5-Ens-Mean":  {"color": "#FF7F0E", "marker": "P", "linestyle": "-",  "linewidth": 3.5, "zorder": 9},
    # AI models – warm/varied colours, dashed
    "AIFS-Ens-Mean":             {"color": "#DE8F05", "marker": "^", "linestyle": "--", "linewidth": 2.2, "zorder": 3},
    "GraphCast":                 {"color": "#029E73", "marker": "*", "linestyle": "--", "linewidth": 2.2, "zorder": 3},
    "GraphCast (IFS)":           {"color": "#029E73", "marker": "*", "linestyle": "--", "linewidth": 2.2, "zorder": 3},
    "Pangu-Weather":             {"color": "#CC78BC", "marker": "X", "linestyle": "--", "linewidth": 2.2, "zorder": 3},
    "Pangu-Weather (IFS)":       {"color": "#CC78BC", "marker": "X", "linestyle": "--", "linewidth": 2.2, "zorder": 3},
    "GenCast":                   {"color": "#CA9161", "marker": "h", "linestyle": "--", "linewidth": 2.2, "zorder": 3},
    "FourCastNet-v2 (IFS)":      {"color": "#D55E00", "marker": "v", "linestyle": "--", "linewidth": 2.2, "zorder": 3},
    "Aurora (IFS)":              {"color": "#949494", "marker": "h", "linestyle": "--", "linewidth": 2.2, "zorder": 3},
    "WeatherNext2":              {"color": "#ECE133", "marker": "8", "linestyle": "--", "linewidth": 2.2, "zorder": 3},
}
_DEFAULT_STYLE = {"color": "gray", "marker": "o", "linestyle": ":", "linewidth": 2.0, "zorder": 2}

# Preferred legend order: physical → WeatherMesh → AI
MODEL_PLOT_ORDER = [
    "IFS-Ens-Mean", "GFS-Ens-Mean", "HRES",
    "WeatherMesh-4", "WeatherMesh-4p5-Ens-Mean",
    "AIFS-Ens-Mean", "FourCastNet-v2 (IFS)", "GraphCast", "GraphCast (IFS)",
    "Pangu-Weather", "Pangu-Weather (IFS)", "GenCast", "Aurora (IFS)", "WeatherNext2",
]


def get_track_model_style(model_name: str) -> dict:
    """Resolve a CSV model name to its visual style, handling underscore variants."""
    if model_name in MODEL_STYLES:
        return MODEL_STYLES[model_name]
    # Try converting underscores back to spaces/parens for CSV-safe names
    canonical = model_name.replace("_", " ").replace(" (", "_(")
    # Also try: GraphCast_(IFS) -> GraphCast (IFS)
    canonical2 = model_name.replace("_", " ")
    for variant in (canonical, canonical2, model_name):
        if variant in MODEL_STYLES:
            return MODEL_STYLES[variant]
    return _DEFAULT_STYLE


def order_models_for_plot(model_names: list[str]) -> list[str]:
    """Sort models by the preferred legend order; unknowns at the end."""
    # Build a lookup allowing underscore variants
    order_map = {}
    for i, name in enumerate(MODEL_PLOT_ORDER):
        order_map[name] = i
        order_map[name.replace(" ", "_").replace("/", "_")] = i
    ordered = sorted(model_names, key=lambda m: order_map.get(m, 999))
    return ordered


# SLP contour levels: 900 to 1030 hPa, every 4 hPa (descending for cool→warm)
SLP_LEVELS = np.arange(900, 1031, 4)

# IBTrACS source for observed track
IBTRACS_URI = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
    "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)
# Cache the full IBTrACS database (shared across all cases)
IBTRACS_FULL_CACHE = TRACKS_DIR / "ibtracs_full_database.csv"


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def load_observed_track(case: dict, cache_dir: Path) -> pd.DataFrame:
    """Load IBTrACS data and filter to this storm.

    Returns a DataFrame with columns: valid_time, latitude, longitude,
    pressure_hpa, wind_kts.
    """
    # Create case-specific cache file for the filtered track
    case_id = case["case_id_number"]
    case_cache = cache_dir / f"ibtracs_case_{case_id}_obs_track.csv"
    
    if case_cache.exists():
        print(f"  Using cached observed track from {case_cache.name}")
        obs = pd.read_csv(case_cache, parse_dates=["valid_time"])
        return obs

    # Load full IBTrACS database (download once, cache forever)
    if IBTRACS_FULL_CACHE.exists():
        print(f"  Loading full IBTrACS database from cache ({IBTRACS_FULL_CACHE.name})")
        raw = pd.read_csv(
            IBTRACS_FULL_CACHE,
            dtype=str,
            low_memory=False,
        )
    else:
        print("  Downloading IBTrACS CSV (this may take a moment) …")
        # Read only the columns we need; skip the units row (row 1 after header)
        cols_to_use = [
            "SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON",
            "USA_WIND", "USA_PRES", "WMO_WIND", "WMO_PRES",
        ]
        raw = pd.read_csv(
            IBTRACS_URI,
            usecols=cols_to_use,
            skiprows=[1],  # units row
            dtype=str,
            low_memory=False,
        )
        # Cache the full database for future use
        raw.to_csv(IBTRACS_FULL_CACHE, index=False)
        print(f"  Cached full IBTrACS database → {IBTRACS_FULL_CACHE}")

    # Filter to matching storm
    storm_name = case["title"].upper().strip()
    start = pd.Timestamp(case["start_date"])
    season = start.year if start.month <= 11 else start.year + 1

    print(f"  Filtering for: storm='{storm_name}', season={season}")

    mask = (
        (raw["SEASON"].str.strip() == str(season))
        & (raw["NAME"].str.strip().str.upper() == storm_name)
    )
    subset = raw.loc[mask].copy()

    if subset.empty:
        raise ValueError(
            f"No IBTrACS rows found for storm '{storm_name}' season {season}"
        )

    # Parse columns
    subset["valid_time"] = pd.to_datetime(subset["ISO_TIME"].str.strip())
    subset["latitude"] = pd.to_numeric(subset["LAT"].str.strip(), errors="coerce")
    subset["longitude"] = pd.to_numeric(subset["LON"].str.strip(), errors="coerce")

    # Pressure: prefer USA_PRES, fall back to WMO_PRES (hPa in IBTrACS)
    subset["pressure_hpa"] = pd.to_numeric(
        subset["USA_PRES"].str.strip(), errors="coerce"
    ).fillna(
        pd.to_numeric(subset["WMO_PRES"].str.strip(), errors="coerce")
    )
    # Wind: prefer USA_WIND, fall back to WMO_WIND (knots in IBTrACS)
    subset["wind_kts"] = pd.to_numeric(
        subset["USA_WIND"].str.strip(), errors="coerce"
    ).fillna(
        pd.to_numeric(subset["WMO_WIND"].str.strip(), errors="coerce")
    )

    obs = subset[
        ["valid_time", "latitude", "longitude", "pressure_hpa", "wind_kts"]
    ].dropna(subset=["latitude", "longitude"]).sort_values("valid_time").reset_index(drop=True)

    # Convert longitude to -180/180
    obs["longitude"] = obs["longitude"].apply(lambda x: x - 360 if x > 180 else x)

    obs.to_csv(case_cache, index=False)
    print(f"  Cached {len(obs)} track points for this case → {case_cache.name}")
    return obs


def load_case_metadata(case_id: int) -> dict:
    """Load case metadata from events.yaml (lightweight, no ewb import)."""
    import extremeweatherbench.data  # only used to locate the bundled YAML file

    events_file = importlib.resources.files(extremeweatherbench.data).joinpath(
        "events.yaml"
    )
    with importlib.resources.as_file(events_file) as fpath:
        with open(fpath) as f:
            all_cases = yaml.safe_load(f)

    matching = [c for c in all_cases if c["case_id_number"] == case_id]
    if not matching:
        raise ValueError(f"Case {case_id} not found in events.yaml")
    return matching[0]


def get_domain(case: dict) -> list[float]:
    """Return [lon_min, lon_max, lat_min, lat_max] in -180/180 for cartopy."""
    params = case["location"]["parameters"]
    lon_min = params["longitude_min"]
    lon_max = params["longitude_max"]
    lat_min = params["latitude_min"]
    lat_max = params["latitude_max"]

    # Convert 0-360 → -180/180
    if lon_min > 180:
        lon_min -= 360
    if lon_max > 180:
        lon_max -= 360

    return [lon_min, lon_max, lat_min, lat_max]


def find_model_source(model_csv_name: str, model_sources: dict):
    """Match a CSV model name back to its MODEL_SOURCES entry."""
    if model_csv_name in model_sources:
        return model_csv_name, model_sources[model_csv_name]
    # Try safe-name matching
    for name, cfg in model_sources.items():
        safe = name.replace(" ", "_").replace("/", "_")
        if safe == model_csv_name:
            return name, cfg
    return None, None


def get_model_sources_for_case(case: dict) -> dict:
    """Determine which MODEL_SOURCES dict to use based on case year."""
    start_date = pd.Timestamp(case["start_date"])
    year = start_date.year
    
    if year <= 2022:
        print(f"  Using 2020-era data sources (case year: {year})")
        return MODEL_SOURCES_2020
    else:
        print(f"  Using 2025-era data sources (case year: {year})")
        return MODEL_SOURCES_2025


def open_slp_dataset(
    model_name: str, config: dict, case: dict, init_times: list = None
) -> xr.Dataset | None:
    """Open forecast zarr, extract SLP, subset to case domain, normalise coords.
    
    For per_init models, init_times should be a list of pandas Timestamps to load.
    """
    source = config["source"]
    storage_options = config.get("storage_options") or {}
    per_init = config.get("per_init", False)

    if per_init:
        # Load multiple per-init zarr files
        if not init_times:
            print(f"    ⚠ per_init=True but no init_times provided")
            return None
        
        print(f"    Loading {len(init_times)} per-init zarr files...")
        all_ds = []
        
        for init_ts in init_times:
            init_str = init_ts.strftime("%Y%m%d%H")
            # Path pattern: /huge/proc/met-data/zarr/{YYYYMMDDHH}/{MODEL}/{MODEL}_{YYYYMMDDHH}.zarr
            zarr_path = Path(source) / init_str / model_name / f"{model_name}_{init_str}.zarr"
            
            if not zarr_path.exists():
                print(f"      ⚠ Missing: {zarr_path}")
                continue
            
            try:
                ds = xr.open_zarr(zarr_path, chunks="auto", decode_timedelta=True)
                # Add init_time as a coordinate
                ds = ds.expand_dims({"init_time": [init_ts]})
                all_ds.append(ds)
            except Exception as e:
                print(f"      ⚠ Failed to open {zarr_path}: {e}")
                continue
        
        if not all_ds:
            print(f"    ⚠ No valid zarr files found")
            return None
        
        # Concatenate along init_time
        ds = xr.concat(all_ds, dim="init_time")
        print(f"    ✓ Loaded {len(all_ds)} init times")
    else:
        # Load single consolidated zarr
        print(f"    Opening {source} ...")
        try:
            ds = xr.open_zarr(
                source,
                storage_options=storage_options,
                chunks="auto",
                decode_timedelta=True,
            )
        except Exception as e:
            print(f"    ⚠ Failed to open zarr: {e}")
            return None

    # ── Rename coordinates / variables to standard names ──
    renames = {}
    if "time" in ds.dims and "init_time" not in ds.dims:
        renames["time"] = "init_time"
    if "prediction_timedelta" in ds.dims and "lead_time" not in ds.dims:
        renames["prediction_timedelta"] = "lead_time"
    if "step" in ds.dims and "lead_time" not in ds.dims:
        renames["step"] = "lead_time"
    if "lat" in ds.dims and "latitude" not in ds.dims:
        renames["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        renames["lon"] = "longitude"

    # SLP variable
    for raw_name in ("mean_sea_level_pressure", "msl"):
        if raw_name in ds and "air_pressure_at_mean_sea_level" not in ds:
            renames[raw_name] = "air_pressure_at_mean_sea_level"
            break

    if renames:
        ds = ds.rename(renames)

    slp_var = "air_pressure_at_mean_sea_level"
    if slp_var not in ds:
        print(f"    ⚠ SLP variable not found. Available: {list(ds.data_vars)}")
        return None

    ds = ds[[slp_var]]

    # ── Spatial subset ──
    params = case["location"]["parameters"]
    lon_min = params["longitude_min"]
    lon_max = params["longitude_max"]
    lat_min = params["latitude_min"]
    lat_max = params["latitude_max"]

    # Adjust bounds to dataset longitude convention
    ds_lons = ds.longitude.values
    if ds_lons.min() < 0:
        if lon_min > 180:
            lon_min -= 360
        if lon_max > 180:
            lon_max -= 360

    # Handle ascending / descending latitude
    lat_ascending = ds.latitude.values[0] < ds.latitude.values[-1]
    if lat_ascending:
        lat_slice = slice(lat_min, lat_max)
    else:
        lat_slice = slice(lat_max, lat_min)
    lon_slice = slice(min(lon_min, lon_max), max(lon_min, lon_max))

    ds = ds.sel(latitude=lat_slice, longitude=lon_slice)

    # ── Temporal subset ──
    if not per_init:  # Only filter if not already filtered by per-init loading
        start_date = pd.Timestamp(case["start_date"])
        end_date = pd.Timestamp(case["end_date"])
        if "init_time" in ds.dims:
            ds = ds.sel(init_time=slice(start_date, end_date))

    # Filter lead times to 12-hourly, 12 h – 240 h
    if "lead_time" in ds.dims:
        # Check if lead_time is timedelta or numeric hours
        if np.issubdtype(ds.lead_time.dtype, np.timedelta64):
            # Convert timedelta to hours
            lead_hours = ds.lead_time.values / np.timedelta64(1, "h")
        else:
            # Already numeric hours
            lead_hours = ds.lead_time.values
        
        valid_lt = (lead_hours > 0) & (lead_hours % 12 == 0) & (lead_hours <= 240)
        ds = ds.isel(lead_time=valid_lt)

    # ── Convert longitude to -180/180 for plotting ──
    if ds.longitude.values.max() > 180:
        ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
        ds = ds.sortby("longitude")

    return ds


def add_map_features(ax, domain):
    """Set extent and add standard map features to an axis."""
    ax.set_extent(domain, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3, zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3, zorder=0)


def add_gridlines(ax, fontsize=8):
    """Add gridlines with labels to an axis."""
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--", zorder=3
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": fontsize}
    gl.ylabel_style = {"size": fontsize}
    return gl


def _in_map_bounds(ax, lon: float, lat: float) -> bool:
    """Return True if (lon, lat) falls within the current map extent."""
    extent = ax.get_extent(crs=ccrs.PlateCarree())  # [x0, x1, y0, y1]
    return extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]


def plot_observed_track(ax, obs_df: pd.DataFrame, label="Observed (IBTrACS)", zorder=4):
    """Plot the observed track."""
    obs_lons = obs_df["longitude"].values
    obs_lats = obs_df["latitude"].values
    ax.plot(
        obs_lons, obs_lats, "k-",
        linewidth=3, alpha=0.8,
        transform=ccrs.PlateCarree(), label=label, zorder=zorder,
    )
    ax.plot(
        obs_lons, obs_lats, "ko",
        markersize=4,
        transform=ccrs.PlateCarree(), zorder=zorder,
    )


def annotate_observed_track_time_bounds(
    ax,
    obs_df: pd.DataFrame,
    zorder: int = 11,
):
    """Annotate the first/last observed IBTrACS timestamps on the map."""
    if len(obs_df) == 0:
        return

    obs_sorted = obs_df.sort_values("valid_time")
    first = obs_sorted.iloc[0]
    last = obs_sorted.iloc[-1]

    if _in_map_bounds(ax, float(first["longitude"]), float(first["latitude"])):
        first_time = pd.Timestamp(first["valid_time"]).strftime("%Y-%m-%d %H:%M UTC")
        ax.text(
            float(first["longitude"]),
            float(first["latitude"]) - 0.6,
            f"OBS START\n{first_time}",
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="top",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.85),
            transform=ccrs.PlateCarree(),
            zorder=zorder,
        )

    if _in_map_bounds(ax, float(last["longitude"]), float(last["latitude"])):
        last_time = pd.Timestamp(last["valid_time"]).strftime("%Y-%m-%d %H:%M UTC")
        ax.text(
            float(last["longitude"]),
            float(last["latitude"]) + 0.6,
            f"OBS END\n{last_time}",
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.85),
            transform=ccrs.PlateCarree(),
            zorder=zorder,
        )


def annotate_observed_track_periodic(
    ax,
    obs_df: pd.DataFrame,
    step_days: int = 3,
    zorder: int = 10,
):
    """Annotate observed track every N days with arrow callouts."""
    if len(obs_df) == 0:
        return

    obs_sorted = obs_df.sort_values("valid_time").copy()
    obs_sorted["valid_time"] = pd.to_datetime(obs_sorted["valid_time"], errors="coerce")
    obs_sorted = obs_sorted[obs_sorted["valid_time"].notna()].copy()
    if len(obs_sorted) == 0:
        return

    first_time = pd.Timestamp(obs_sorted["valid_time"].iloc[0]).floor("D")
    last_time = pd.Timestamp(obs_sorted["valid_time"].iloc[-1])

    target_times = []
    t = first_time
    while t <= last_time:
        target_times.append(t)
        t += pd.Timedelta(days=step_days)

    # Exclude endpoints because those are labeled separately.
    interior_times = target_times[1:-1]
    if not interior_times:
        return

    used_indices = set()
    for i, t_target in enumerate(interior_times):
        diffs = (obs_sorted["valid_time"] - t_target).abs()
        idx = int(diffs.idxmin())
        if idx in used_indices:
            continue
        used_indices.add(idx)

        row = obs_sorted.loc[idx]
        lon = float(row["longitude"])
        lat = float(row["latitude"])

        if not _in_map_bounds(ax, lon, lat):
            continue

        label = pd.Timestamp(row["valid_time"]).strftime("%m-%d")

        # Alternate offsets to keep labels from stacking on the track.
        x_offset = 1.2 if i % 2 == 0 else -1.6
        y_offset = 1.0 if i % 3 != 0 else -1.1

        ax.annotate(
            label,
            xy=(lon, lat),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(lon + x_offset, lat + y_offset),
            textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9, shrinkA=2, shrinkB=2),
            zorder=zorder,
        )


def extract_observed_landfalls(obs_df: pd.DataFrame) -> list[tuple[int, pd.Timestamp, float, float]]:
    """Return observed landfall markers as (landfall_id, valid_time, lat, lon)."""
    if len(obs_df) == 0:
        return []

    track_da = xr.DataArray(
        obs_df["pressure_hpa"].fillna(np.nan).to_numpy(),
        dims=["valid_time"],
        coords={
            "valid_time": obs_df["valid_time"].to_numpy(),
            "latitude": ("valid_time", obs_df["latitude"].to_numpy()),
            "longitude": ("valid_time", obs_df["longitude"].to_numpy()),
        },
        name="air_pressure_at_mean_sea_level",
    )

    landfalls = calc.find_landfalls(track_da, return_next_landfall=True)
    if landfalls is None:
        return []

    out: list[tuple[int, pd.Timestamp, float, float]] = []
    if "landfall" in landfalls.dims:
        for lf_idx in range(len(landfalls.landfall)):
            lf = landfalls.isel(landfall=lf_idx)
            lf_time = pd.Timestamp(lf.coords["valid_time"].values)
            lf_lat = float(lf.coords["latitude"].values)
            lf_lon = float(lf.coords["longitude"].values)
            out.append((int(lf_idx), lf_time, lf_lat, lf_lon))
        return out

    lf_time = pd.Timestamp(landfalls.coords["valid_time"].values)
    lf_lat = float(landfalls.coords["latitude"].values)
    lf_lon = float(landfalls.coords["longitude"].values)
    return [(0, lf_time, lf_lat, lf_lon)]


def plot_landfall_annotations(ax, landfalls: list[tuple[int, pd.Timestamp, float, float]], zorder=9):
    """Plot LF# markers and labels on a map axis."""
    if not landfalls:
        return

    for lf_idx, _lf_time, lat, lon in landfalls:
        if not _in_map_bounds(ax, lon, lat):
            continue
        ax.plot(
            lon,
            lat,
            marker="^",
            color="gold",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.8,
            transform=ccrs.PlateCarree(),
            zorder=zorder,
        )
        ax.text(
            lon,
            lat + 0.4,
            f"LF{lf_idx}",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.85),
            transform=ccrs.PlateCarree(),
            zorder=zorder + 1,
        )


def plot_forecast_track(ax, track_lats, track_lons, label="Forecast Track", zorder=5):
    """Plot the full forecast track."""
    if len(track_lats) < 2:
        return
    ax.plot(
        track_lons, track_lats, "-",
        color="cyan", linewidth=2.5,
        transform=ccrs.PlateCarree(), label=label, zorder=zorder,
    )
    ax.plot(
        track_lons, track_lats, "o",
        color="cyan", markersize=4, markeredgecolor="black", markeredgewidth=0.5,
        transform=ccrs.PlateCarree(), zorder=zorder,
    )


def get_track_arrays(df: pd.DataFrame):
    """Extract valid (non-NaN) lat/lon arrays from a track DataFrame."""
    lats = df["latitude"].values
    lons = df["longitude"].values
    valid = ~(np.isnan(lats) | np.isnan(lons))
    return lats[valid], lons[valid]


def compute_track_domain(
    all_lats: list[np.ndarray],
    all_lons: list[np.ndarray],
    pad_deg: float = 5.0,
    min_span_deg: float = 10.0,
) -> list[float]:
    """Compute a tight [lon_min, lon_max, lat_min, lat_max] domain from track arrays.

    Adds *pad_deg* around the bounding box of all points and ensures a minimum
    span of *min_span_deg* in each direction so the plot isn't too cramped.
    """
    lats = np.concatenate(all_lats)
    lons = np.concatenate(all_lons)

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    # Ensure minimum span
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    lat_span = max(lat_max - lat_min, min_span_deg)
    lon_span = max(lon_max - lon_min, min_span_deg)

    lat_min = lat_center - lat_span / 2 - pad_deg
    lat_max = lat_center + lat_span / 2 + pad_deg
    lon_min = lon_center - lon_span / 2 - pad_deg
    lon_max = lon_center + lon_span / 2 + pad_deg

    # Clamp to valid ranges
    lat_min = max(lat_min, -90)
    lat_max = min(lat_max, 90)

    return [lon_min, lon_max, lat_min, lat_max]


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Parse arguments ──
    parser = argparse.ArgumentParser(
        description="Plot TC forecast tracks (and optionally SLP fields) for any tropical cyclone case"
    )
    parser.add_argument(
        "case_id",
        type=int,
        help="Case ID number from events.yaml (e.g., 236 for Hurricane Laura, 338 for Hurricane Melissa)",
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=TRACKS_DIR,
        help=f"Directory containing forecast_tracks_*.csv files (default: {TRACKS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: tc_track_plots/case_<id>)",
    )
    parser.add_argument(
        "--slp-pdfs",
        action="store_true",
        default=False,
        help="Generate per-model per-init SLP contour PDFs (slow; disabled by default)",
    )
    args = parser.parse_args()
    
    case_id = args.case_id
    tracks_dir = args.tracks_dir

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = OUTPUT_BASE / f"case_{case_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"TC Track & SLP Visualization – Case {case_id}")
    print("=" * 80)

    # ── Load case metadata ──
    case = load_case_metadata(case_id)
    domain = get_domain(case)
    case_title = case.get("title", f"Case {case_id}")
    print(f"Case {case_id}: {case_title}")
    print(
        f"Domain: lat [{domain[2]:.1f}, {domain[3]:.1f}], "
        f"lon [{domain[0]:.1f}, {domain[1]:.1f}]"
    )
    print(f"Period: {case['start_date']} → {case['end_date']}")

    # ── Determine model sources based on case year ──
    model_sources = get_model_sources_for_case(case)

    # ── Load observed track from IBTrACS ──
    obs_track = load_observed_track(case, tracks_dir)
    observed_landfalls = extract_observed_landfalls(obs_track)
    print(
        f"  Observed track: {len(obs_track)} points, "
        f"{obs_track['valid_time'].min()} → {obs_track['valid_time'].max()}"
    )
    if observed_landfalls:
        print(f"  Observed landfalls: {len(observed_landfalls)}")
        for lf_idx, lf_time, _lf_lat, _lf_lon in observed_landfalls:
            print(f"    LF{lf_idx}: {lf_time}")

    # ── Load track CSVs ──
    track_csv_files = sorted(tracks_dir.glob("forecast_tracks_*.csv"))
    if not track_csv_files:
        print(f"❌ No forecast_tracks_*.csv files found in {tracks_dir}")
        print("   Run evaluate_case_2025.py or evaluate_case_2020.py first to generate tracks")
        return 1

    all_dfs = []
    for csv_file in track_csv_files:
        model_name = csv_file.stem.replace("forecast_tracks_", "")
        df = pd.read_csv(csv_file)
        df["model"] = model_name
        all_dfs.append(df)
        print(f"  ✓ {csv_file.name}: {len(df)} pts  (model: {model_name})")

    tracks_df = pd.concat(all_dfs, ignore_index=True)
    tracks_df["valid_time"] = pd.to_datetime(tracks_df["valid_time"])

    if "init_time" not in tracks_df.columns and "lead_time_hours" in tracks_df.columns:
        tracks_df["init_time"] = tracks_df["valid_time"] - pd.to_timedelta(
            tracks_df["lead_time_hours"], unit="h"
        )
    if "init_time" in tracks_df.columns:
        tracks_df["init_time"] = pd.to_datetime(tracks_df["init_time"])

    models = sorted(tracks_df["model"].unique())
    print(f"\n{len(models)} model(s): {models}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Part 1 – Per-model track overview PNGs
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Part 1: Per-model track overview PNGs")
    print("=" * 80)

    for model in models:
        model_df = tracks_df[tracks_df["model"] == model]
        if "init_time" in model_df.columns:
            init_times = sorted(model_df["init_time"].unique())
        else:
            init_times = [None]
        n_inits = len(init_times)

        fig = plt.figure(figsize=(16, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())
        add_map_features(ax, domain)
        colors = cm.viridis(np.linspace(0, 1, max(n_inits, 1)))

        for i, init_time in enumerate(init_times):
            if init_time is not None:
                track = model_df[model_df["init_time"] == init_time].sort_values("valid_time")
                label = f"Init {pd.Timestamp(init_time).strftime('%m/%d %H:%M')}"
            else:
                track = model_df.sort_values("valid_time")
                label = "Forecast"

            lats, lons = get_track_arrays(track)
            if len(lats) < 2:
                continue

            ax.plot(
                lons,
                lats,
                color=colors[i],
                linewidth=2.5,
                alpha=0.8,
                transform=ccrs.PlateCarree(),
                label=label,
                zorder=5,
            )
            ax.plot(
                lons[0],
                lats[0],
                "o",
                color=colors[i],
                markersize=8,
                transform=ccrs.PlateCarree(),
                zorder=6,
            )
            ax.plot(
                lons[-1],
                lats[-1],
                "s",
                color=colors[i],
                markersize=8,
                transform=ccrs.PlateCarree(),
                zorder=6,
            )

        plot_observed_track(ax, obs_track, zorder=7)
        annotate_observed_track_periodic(ax, obs_track, step_days=3, zorder=11)
        annotate_observed_track_time_bounds(ax, obs_track, zorder=12)
        plot_landfall_annotations(ax, observed_landfalls, zorder=8)
        add_gridlines(ax, fontsize=10)
        ax.set_title(
            f"{model} – TC Track Forecasts – {case_title}\n{n_inits} Initialization Time(s)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        legend = ax.legend(loc="upper left", fontsize=9, framealpha=0.95, ncol=2)
        legend.set_zorder(100)

        safe_model = model.replace(" ", "_").replace("/", "_")
        out_png = output_dir / f"{safe_model}_all_tracks.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved: {out_png}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Part 2 – Per-init multi-model track PNGs
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Part 2: Per-init multi-model track PNGs")
    print("=" * 80)

    if "init_time" in tracks_df.columns:
        all_init_times = sorted(tracks_df["init_time"].unique())
    else:
        all_init_times = []
        print("  ⚠ No init_time column found, skipping multi-model track plots")

    for init_time in all_init_times:
        init_ts = pd.Timestamp(init_time)
        init_str = init_ts.strftime("%Y%m%d_%H%M")

        # First pass: collect all track arrays to compute a tight domain
        init_track_lats = []
        init_track_lons = []
        init_model_tracks = {}
        for model in models:
            model_df = tracks_df[
                (tracks_df["model"] == model)
                & (tracks_df["init_time"] == init_time)
            ].sort_values("valid_time")
            if model_df.empty:
                continue
            lats, lons = get_track_arrays(model_df)
            if len(lats) < 2:
                continue
            init_model_tracks[model] = (lats, lons)
            init_track_lats.append(lats)
            init_track_lons.append(lons)

        if not init_model_tracks:
            continue

        # Domain is computed from forecast tracks only; the full observed
        # track is still plotted but doesn't influence the zoom.
        init_domain = compute_track_domain(init_track_lats, init_track_lons)

        fig = plt.figure(figsize=(16, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())
        add_map_features(ax, init_domain)

        for model in order_models_for_plot(list(init_model_tracks.keys())):
            lats, lons = init_model_tracks[model]
            style = get_track_model_style(model)
            ax.plot(
                lons, lats,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                label=model,
                zorder=style["zorder"],
            )
            ax.plot(
                lons, lats,
                marker=style["marker"],
                linestyle="None",
                color=style["color"],
                markersize=5 if "WeatherMesh" in model else 4,
                markeredgecolor="black",
                markeredgewidth=0.5,
                transform=ccrs.PlateCarree(),
                zorder=style["zorder"],
            )

        plot_observed_track(ax, obs_track, zorder=7)
        annotate_observed_track_periodic(ax, obs_track, step_days=3, zorder=11)
        annotate_observed_track_time_bounds(ax, obs_track, zorder=12)
        plot_landfall_annotations(ax, observed_landfalls, zorder=8)
        add_gridlines(ax, fontsize=10)
        ax.set_title(
            f"All Models – TC Track Forecasts – {case_title}\n"
            f"Init: {init_ts.strftime('%Y-%m-%d %H:%M UTC')}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        legend = ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
        legend.set_zorder(100)

        out_png = output_dir / f"all_models_init_{init_str}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved: {out_png}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Part 3 – Per-model per-init SLP PDFs  (only with --slp-pdfs)
    # ═══════════════════════════════════════════════════════════════════════════
    if not args.slp_pdfs:
        print("\n" + "=" * 80)
        print("Part 3: Per-model per-init SLP PDFs  [SKIPPED – pass --slp-pdfs to enable]")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Part 3: Per-model per-init SLP PDFs")
        print("=" * 80)

        for model in models:
            model_df = tracks_df[tracks_df["model"] == model]
            if "init_time" not in model_df.columns:
                print(f"  ⚠ No init_time column for {model}, skipping SLP PDFs")
                continue

            init_times = sorted(model_df["init_time"].unique())
            if not len(init_times):
                print(f"  ⚠ No init times for {model}, skipping")
                continue

            real_name, config = find_model_source(model, model_sources)
            if config is None:
                print(f"  ⚠ No data source configured for '{model}', skipping SLP PDFs")
                continue

            print(f"\n  {model} ({real_name}): {len(init_times)} init time(s)")
            init_timestamps = [pd.Timestamp(it) for it in init_times]
            ds = open_slp_dataset(real_name, config, case, init_times=init_timestamps)
            if ds is None:
                continue

            safe_model = model.replace(" ", "_").replace("/", "_")

            for init_time in init_times:
                init_ts = pd.Timestamp(init_time)
                print(f"    Init {init_ts.strftime('%Y-%m-%d %H:%M')} …")

                init_track = model_df[model_df["init_time"] == init_time].sort_values("valid_time")
                track_lats, track_lons = get_track_arrays(init_track)

                try:
                    ds_init = ds.sel(init_time=init_ts, method="nearest")
                except Exception as e:
                    print(f"      ⚠ Could not select init_time {init_ts}: {e}")
                    continue

                if "lead_time" not in ds_init.dims:
                    print("      ⚠ No lead_time dimension, skipping")
                    continue

                lead_times = ds_init.lead_time.values
                n_leads = len(lead_times)
                if n_leads == 0:
                    print("      ⚠ No lead times available, skipping")
                    continue

                init_str = init_ts.strftime("%Y%m%d_%H%M")
                pdf_path = output_dir / f"{safe_model}_init_{init_str}_slp.pdf"

                with PdfPages(pdf_path) as pdf:
                    for lt in lead_times:
                        try:
                            lt_hours = float(lt / np.timedelta64(1, "h"))
                        except Exception:
                            lt_hours = float(lt)
                        valid_time = init_ts + pd.Timedelta(hours=lt_hours)

                        try:
                            slp_pa = (
                                ds_init["air_pressure_at_mean_sea_level"]
                                .sel(lead_time=lt)
                                .compute()
                            )
                        except Exception as e:
                            print(f"      ⚠ Lead {lt_hours:.0f}h failed: {e}")
                            continue

                        slp_hpa = slp_pa.values / 100.0  # Pa → hPa
                        lats = slp_pa.latitude.values
                        lons = slp_pa.longitude.values

                        if np.all(np.isnan(slp_hpa)):
                            print(f"      ⚠ Lead {lt_hours:.0f}h is all-NaN, skipping page")
                            continue
                        min_idx = np.nanargmin(slp_hpa)
                        min_lat_idx, min_lon_idx = np.unravel_index(min_idx, slp_hpa.shape)
                        min_slp = round(float(slp_hpa[min_lat_idx, min_lon_idx]))
                        min_lat = float(lats[min_lat_idx])
                        min_lon = float(lons[min_lon_idx])

                        fig = plt.figure(figsize=(14, 10))
                        ax = plt.axes(projection=ccrs.PlateCarree())
                        add_map_features(ax, domain)

                        lon2d, lat2d = np.meshgrid(lons, lats)
                        cf = ax.contourf(
                            lon2d,
                            lat2d,
                            slp_hpa,
                            levels=SLP_LEVELS,
                            cmap="RdYlBu_r",
                            extend="both",
                            transform=ccrs.PlateCarree(),
                            zorder=2,
                        )
                        plt.colorbar(
                            cf,
                            ax=ax,
                            orientation="vertical",
                            label="MSLP (hPa)",
                            shrink=0.8,
                            pad=0.02,
                        )

                        cs = ax.contour(
                            lon2d,
                            lat2d,
                            slp_hpa,
                            levels=SLP_LEVELS,
                            colors="black",
                            linewidths=0.5,
                            transform=ccrs.PlateCarree(),
                            zorder=2,
                        )
                        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")

                        plot_forecast_track(ax, track_lats, track_lons)
                        plot_observed_track(ax, obs_track)
                        annotate_observed_track_periodic(ax, obs_track, step_days=3, zorder=11)
                        annotate_observed_track_time_bounds(ax, obs_track, zorder=12)
                        plot_landfall_annotations(ax, observed_landfalls, zorder=8)

                        ax.plot(
                            min_lon,
                            min_lat,
                            "w*",
                            markersize=15,
                            markeredgecolor="black",
                            markeredgewidth=1,
                            transform=ccrs.PlateCarree(),
                            zorder=6,
                        )
                        ax.text(
                            min_lon + 1.5,
                            min_lat + 1.5,
                            f"{min_slp:.0f} hPa",
                            fontsize=10,
                            fontweight="bold",
                            color="darkblue",
                            bbox=dict(
                                boxstyle="round",
                                facecolor="white",
                                alpha=0.9,
                                edgecolor="darkblue",
                            ),
                            transform=ccrs.PlateCarree(),
                            zorder=6,
                        )

                        add_gridlines(ax)
                        ax.set_title(
                            f"{model} – MSLP\n"
                            f"Init: {init_ts.strftime('%Y-%m-%d %H:%M UTC')}    "
                            f"Valid: {valid_time.strftime('%Y-%m-%d %H:%M UTC')}    "
                            f"F{int(lt_hours):03d}",
                            fontsize=12,
                            fontweight="bold",
                        )
                        ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

                        plt.tight_layout()
                        pdf.savefig(fig, dpi=150)
                        plt.close(fig)

                print(f"      ✓ {pdf_path.name}  ({n_leads} pages)")

            ds.close()

    print("\n" + "=" * 80)
    print("Done!")
    print(f"All outputs in: {output_dir}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())
