#!/usr/bin/env python3
"""
Analyze and plot evaluation results from ExtremeWeatherBench.

This script loads CSV results and creates comparison plots with:
- Colorblind-friendly palette
- Thicker lines for better visibility
- Varied markers for each model
- WeatherMesh always plotted in black for emphasis

Usage:
    python analyze_results.py <results_csv>

Example:
    python analyze_results.py case_29_heat_wave_results.csv
"""

import argparse
import importlib.resources
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import yaml

from extremeweatherbench import calc
import extremeweatherbench.data

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11

# Palettes for the two model families
_PASTEL2 = sns.color_palette("Pastel2")   # physical models
_DEEP = sns.color_palette("deep")         # AI models

# Physical models: IFS, IFS-Ens-Mean, GFS, GFS-Ens-Mean — solid lines, Pastel2
PHYSICAL_MODELS = {"IFS", "IFS-Ens-Mean", "GFS", "GFS-Ens-Mean"}

# WeatherMesh variants — hardcoded colors, solid lines
_WM_STYLES = {
    "WeatherMesh-4":             {"color": "#555555", "marker": "D", "linestyle": "-"},
    "WeatherMesh-4p5-Ens-Mean":  {"color": "#888888", "marker": "P", "linestyle": "-"},
    "WeatherMesh-5c-Ens-Mean":   {"color": "#000000", "marker": "d", "linestyle": "-"},
}

# Markers to cycle through for physical and AI models
_PHYS_MARKERS = ["o", "s", "^", "v"]
_AI_MARKERS = ["^", "v", "*", "X", "h", "8", "p", "H"]

# Counters for dynamic palette assignment (populated at runtime)
_PHYS_IDX = 0
_AI_IDX = 0

# Runtime cache: model name → style dict
MODEL_GROUPS: dict[str, dict] = {}
MODEL_GROUPS.update(_WM_STYLES)


def _assign_style(model: str) -> dict:
    """Lazily assign a consistent style for a model the first time it's seen."""
    global _PHYS_IDX, _AI_IDX
    if model in MODEL_GROUPS:
        return MODEL_GROUPS[model]
    if model in PHYSICAL_MODELS:
        color = _PASTEL2[_PHYS_IDX % len(_PASTEL2)]
        marker = _PHYS_MARKERS[_PHYS_IDX % len(_PHYS_MARKERS)]
        style = {"color": color, "marker": marker, "linestyle": "-"}
        _PHYS_IDX += 1
    else:
        color = _DEEP[_AI_IDX % len(_DEEP)]
        marker = _AI_MARKERS[_AI_IDX % len(_AI_MARKERS)]
        style = {"color": color, "marker": marker, "linestyle": "--"}
        _AI_IDX += 1
    MODEL_GROUPS[model] = style
    return style


# Preferred ordering for legends/plot traces: physical -> WeatherMesh -> AI.
MODEL_GROUP_ORDER = [
    "IFS-Ens-Mean",
    "GFS-Ens-Mean",
    "WeatherMesh-4",
    "WeatherMesh-4p5-Ens-Mean",
    "WeatherMesh-5c-Ens-Mean",
    "AIFS-Ens-Mean",
    "FourCastNet-v2 (IFS)",
    "GraphCast (IFS)",
    "Pangu-Weather (IFS)",
    "Aurora (IFS)",
    "WeatherNext2",
]

# Fallback for unknown models
DEFAULT_STYLE = {"color": "gray", "marker": "o", "linestyle": ":"}


def _station_color(model: str) -> str:
    """Return the model's line color for station map scatter plots."""
    style = _assign_style(model)
    return style["color"]

# Variable name mappings (internal name → display name)
VARIABLE_DISPLAY_NAMES = {
    "air_pressure_at_mean_sea_level": "MSLP",
    "surface_air_temperature": "T2m",
    "air_temperature": "Air Temperature",
    "specific_humidity": "Specific Humidity",
    "geopotential": "Geopotential Height",
    "surface_eastward_wind": "U10m",
    "surface_northward_wind": "V10m",
    "eastward_wind": "U-wind",
    "northward_wind": "V-wind",
    "total_precipitation": "Precipitation",
    "total_precipitation_6hr": "6hr Precipitation",
    "total_precipitation_12hr": "12hr Precipitation",
}

# Metric name mappings (internal name → display abbreviation)
# Note: CSV uses lowercase names, so include both CamelCase and lowercase versions
METRIC_DISPLAY_NAMES = {
    "RootMeanSquaredError": "RMSE",
    "MeanAbsoluteError": "MAE",
    "MaximumMeanAbsoluteError": "Max MAE",
    "MinimumMeanAbsoluteError": "Min MAE",
    "DurationMeanError": "Duration Error",
    "LandfallDisplacement": "Landfall Distance Error",
    "LandfallTimeMeanError": "Landfall Time Error",
    "LandfallIntensityMeanAbsoluteError": "Landfall Intensity MAE",
    "LandfallIntensityRootMeanSquaredError": "Landfall Intensity RMSE",
    "TrackIntensityMeanAbsoluteError": "Track Intensity MAE",
    "AlongTrackError": "Along-Track Error",
    "CrossTrackError": "Cross-Track Error",
    "TotalTrackError": "Total Track Error",
    # Lowercase versions (as they appear in CSV)
    "landfall_displacement": "Landfall Distance Error",
    "landfall_time_me": "Landfall Time Error",
    "landfall_intensity_mae": "Landfall Intensity MAE",
    "landfall_intensity_rmse": "Landfall Intensity RMSE",
    "track_intensity_mae": "Track Intensity MAE",
    "duration_error": "Duration Error",
    "along_track_error": "Along-Track Error",
    "cross_track_error": "Cross-Track Error",
    "total_track_error": "Total Track Error",
}

# Units for each variable
VARIABLE_UNITS = {
    "air_pressure_at_mean_sea_level": "hPa",
    "surface_air_temperature": "K",
    "air_temperature": "K",
    "specific_humidity": "kg/kg",
    "geopotential": "m²/s²",
    "surface_eastward_wind": "m/s",
    "surface_northward_wind": "m/s",
    "eastward_wind": "m/s",
    "northward_wind": "m/s",
    "total_precipitation": "mm",
    "total_precipitation_6hr": "mm",
    "total_precipitation_12hr": "mm",
}

# Special metrics that don't need variable units (they have their own)
METRIC_UNITS = {
    "DurationMeanError": "hours",
    "LandfallDisplacement": "km",
    "LandfallTimeMeanError": "hours",
    "AlongTrackError": "km",
    "CrossTrackError": "km",
    "TotalTrackError": "km",
    # Lowercase versions (as they appear in CSV)
    "duration_error": "hours",
    "landfall_displacement": "km",
    "landfall_time_me": "hours",
    "landfall_intensity_mae": "hPa",  # Pressure difference at landfall
    "landfall_intensity_rmse": "hPa",  # Pressure difference at landfall
    "track_intensity_mae": "hPa",  # Pressure difference along matched TC track
    "along_track_error": "km",
    "cross_track_error": "km",
    "total_track_error": "km",
    # CamelCase variants (if emitted by some pipelines)
    "TrackIntensityMeanAbsoluteError": "hPa",
    "LandfallIntensityRootMeanSquaredError": "hPa",
    # Heat/freeze event timing metrics (all measured in hours)
    "heat_wave_onset_error": "hours",
    "heat_wave_end_error": "hours",
    "heat_wave_duration_error": "hours",
    "heat_wave_peak_timing_bias": "hours",
    "heat_wave_peak_timing_rmse": "hours",
    "freeze_onset_error": "hours",
    "freeze_end_error": "hours",
    "freeze_duration_error": "hours",
    "freeze_peak_timing_bias": "hours",
    "freeze_peak_timing_rmse": "hours",
}

# Event timing metrics — init-time plots should be trimmed to the range
# where at least one model has non-NaN values.
EVENT_TIMING_METRICS = {
    "heat_wave_onset_error",
    "heat_wave_end_error",
    "heat_wave_duration_error",
    "heat_wave_peak_timing_bias",
    "heat_wave_peak_timing_rmse",
    "freeze_onset_error",
    "freeze_end_error",
    "freeze_duration_error",
    "freeze_peak_timing_bias",
    "freeze_peak_timing_rmse",
}

TRACK_ERROR_METRICS = {
    "along_track_error",
    "cross_track_error",
    "total_track_error",
    "AlongTrackError",
    "CrossTrackError",
    "TotalTrackError",
}

# Metrics whose values can be positive or negative — always draw a bold 0-line.
SIGNED_METRICS = TRACK_ERROR_METRICS | {
    "heat_wave_onset_error",
    "heat_wave_end_error",
    "heat_wave_duration_error",
    "heat_wave_peak_timing_bias",
    "freeze_onset_error",
    "freeze_end_error",
    "freeze_duration_error",
    "freeze_peak_timing_bias",
}


IBTRACS_URI = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
    "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)
IBTRACS_FULL_CACHE = Path(__file__).resolve().parent / "ibtracs_full_database.csv"


def _load_case_metadata(case_id: int) -> dict:
    events_file = importlib.resources.files(extremeweatherbench.data).joinpath("events.yaml")
    with importlib.resources.as_file(events_file) as fpath:
        with open(fpath) as f:
            all_cases = yaml.safe_load(f)
    for case in all_cases:
        if int(case["case_id_number"]) == int(case_id):
            return case
    raise ValueError(f"Case {case_id} not found in events.yaml")


def _load_observed_track_for_case(case_id: int) -> pd.DataFrame:
    """Load observed IBTrACS track for a case (cached full database)."""
    case = _load_case_metadata(case_id)
    storm_name = str(case["title"]).upper().strip()
    start = pd.Timestamp(case["start_date"])
    season = start.year if start.month <= 11 else start.year + 1

    if IBTRACS_FULL_CACHE.exists():
        raw = pd.read_csv(IBTRACS_FULL_CACHE, dtype=str, low_memory=False)
    else:
        cols_to_use = [
            "SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON",
            "USA_WIND", "USA_PRES", "WMO_WIND", "WMO_PRES",
        ]
        raw = pd.read_csv(
            IBTRACS_URI,
            usecols=cols_to_use,
            skiprows=[1],
            dtype=str,
            low_memory=False,
        )
        raw.to_csv(IBTRACS_FULL_CACHE, index=False)

    mask = (
        (raw["SEASON"].str.strip() == str(season))
        & (raw["NAME"].str.strip().str.upper() == storm_name)
    )
    subset = raw.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=["valid_time", "latitude", "longitude", "pressure_hpa"])

    subset["valid_time"] = pd.to_datetime(subset["ISO_TIME"].str.strip())
    subset["latitude"] = pd.to_numeric(subset["LAT"].str.strip(), errors="coerce")
    subset["longitude"] = pd.to_numeric(subset["LON"].str.strip(), errors="coerce")
    subset["pressure_hpa"] = pd.to_numeric(
        subset["USA_PRES"].str.strip(), errors="coerce"
    ).fillna(pd.to_numeric(subset["WMO_PRES"].str.strip(), errors="coerce"))
    subset["longitude"] = subset["longitude"].apply(lambda x: x - 360 if pd.notna(x) and x > 180 else x)

    obs = subset[
        ["valid_time", "latitude", "longitude", "pressure_hpa"]
    ].dropna(subset=["valid_time", "latitude", "longitude"]).sort_values("valid_time")
    return obs.reset_index(drop=True)


def extract_event_times(df: pd.DataFrame) -> list[tuple[str, pd.Timestamp]] | None:
    """Extract event start/end dates from events.yaml for heat_wave/freeze cases.

    Returns a list like [("Event Start", ts), ("Event End", ts)] or None if not
    applicable (e.g. tropical_cyclone cases).
    """
    if "case_id_number" not in df.columns or df["case_id_number"].dropna().empty:
        return None
    if "event_type" not in df.columns or df["event_type"].dropna().empty:
        return None

    event_type = str(df["event_type"].dropna().iloc[0]).strip()
    if event_type not in ("heat_wave", "freeze"):
        return None

    try:
        case_id = int(df["case_id_number"].dropna().iloc[0])
        meta = _load_case_metadata(case_id)
    except Exception:
        return None

    times: list[tuple[str, pd.Timestamp]] = []
    if "start_date" in meta and meta["start_date"]:
        times.append(("Event Start", pd.Timestamp(meta["start_date"])))
    if "end_date" in meta and meta["end_date"]:
        times.append(("Event End", pd.Timestamp(meta["end_date"])))
    return times if times else None


def extract_landfall_times(df: pd.DataFrame) -> list[tuple]:
    """Extract observed landfall times from IBTrACS for this case."""
    if "case_id_number" not in df.columns or df["case_id_number"].dropna().empty:
        return []

    try:
        case_id = int(df["case_id_number"].dropna().iloc[0])
    except Exception:
        return []

    obs = _load_observed_track_for_case(case_id)
    if obs.empty:
        return []

    # Build a track DataArray compatible with calc.find_landfalls
    track_da = xr.DataArray(
        obs["pressure_hpa"].to_numpy(),
        dims=["valid_time"],
        coords={
            "valid_time": obs["valid_time"].to_numpy(),
            "latitude": ("valid_time", obs["latitude"].to_numpy()),
            "longitude": ("valid_time", obs["longitude"].to_numpy()),
        },
        name="air_pressure_at_mean_sea_level",
    )

    # Important: use return_next_landfall=True to get ALL observed landfalls.
    # False returns only the first landfall.
    target_landfalls = calc.find_landfalls(track_da, return_next_landfall=True)
    if target_landfalls is None:
        return []

    if "landfall" in target_landfalls.dims:
        times = target_landfalls.coords["valid_time"].values
        return [(int(i), pd.Timestamp(t)) for i, t in enumerate(times)]

    # Single-landfall fallback
    if "valid_time" in target_landfalls.coords:
        return [(0, pd.Timestamp(target_landfalls.coords["valid_time"].values))]
    return []


def get_display_label(variable: str, metric: str, include_units: bool = True) -> str:
    """
    Get a human-readable label for a variable-metric combination.
    
    Args:
        variable: Variable name (e.g., 'air_pressure_at_mean_sea_level')
        metric: Metric name (e.g., 'MeanAbsoluteError')
        include_units: Whether to include units in parentheses
    
    Returns:
        Display label (e.g., 'MSLP MAE (hPa)' or 'MSLP MAE')
    """
    # Track error and landfall metrics are position/intensity-based and do not need
    # a variable prefix in their display labels.
    track_error_metrics = [
        "along_track_error",
        "cross_track_error",
        "total_track_error",
        "AlongTrackError",
        "CrossTrackError",
        "TotalTrackError",
    ]
    landfall_metrics = [
        "landfall_displacement",
        "landfall_time_me",
        "landfall_intensity_mae",
        "landfall_intensity_rmse",
        "LandfallDisplacement",
        "LandfallTimeMeanError",
        "LandfallIntensityMeanAbsoluteError",
        "LandfallIntensityRootMeanSquaredError",
    ]

    if metric in track_error_metrics or metric in landfall_metrics:
        label = METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())
    else:
        var_display = VARIABLE_DISPLAY_NAMES.get(variable, variable.replace("_", " ").title())
        metric_display = METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())
        label = f"{var_display} {metric_display}"

    if include_units:
        if metric in METRIC_UNITS:
            units = METRIC_UNITS[metric]
        else:
            units = VARIABLE_UNITS.get(variable, "")
        if units:
            label = f"{label} ({units})"

    return label


def get_filename_safe_label(variable: str, metric: str) -> str:
    """
    Get a filename-safe label for a variable-metric combination.
    
    Args:
        variable: Variable name (e.g., 'air_pressure_at_mean_sea_level')
        metric: Metric name (e.g., 'MeanAbsoluteError')
    
    Returns:
        Filename-safe label (e.g., 'mslp_mae')
    """
    var_display = VARIABLE_DISPLAY_NAMES.get(variable, variable).lower().replace(" ", "_")
    metric_display = METRIC_DISPLAY_NAMES.get(metric, metric).lower().replace(" ", "_")
    
    return f"{var_display}_{metric_display}"


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results CSV and do basic filtering."""
    print(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"  Loaded {len(df)} rows")
    
    # Normalize column names - ExtremeWeatherBench uses different names
    column_mapping = {
        'forecast_source': 'forecast_name',
        'forecast': 'forecast_name',
        'metric': 'metric_name',
        'value': 'metric_value',
    }
    
    # Apply mappings for columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Verify required columns exist
    required_cols = ['forecast_name', 'metric_name', 'metric_value']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"  Available columns: {list(df.columns)}")
        raise ValueError(f"Results CSV missing required columns after mapping: {missing}")
    
    print(f"  Forecasts: {df['forecast_name'].unique()}")
    print(f"  Metrics: {df['metric_name'].unique()}")
    
    # DEBUG: Count landfall metrics before filtering
    if "landfall" in df["metric_name"].str.cat():
        landfall_before = len(df[df["metric_name"].str.contains("landfall", case=False, na=False)])
        print(f"  Landfall metrics before filtering: {landfall_before} rows")
    
    # Filter lead_time to 12-hourly intervals for plotting
    # BUT only for non-duration metrics to avoid affecting duration calculations
    if "lead_time" in df.columns:
        df["lead_time_hours"] = pd.to_timedelta(df["lead_time"]).dt.total_seconds() / 3600
        
        # Identify rows that have no lead_time at all (e.g. landfall metrics
        # which are indexed by init_time instead).  These must be preserved.
        no_lead_time = df["lead_time"].isna() | (df["lead_time"] == "")
        
        duration_metrics = df["metric_name"].str.contains("duration", case=False, na=False)
        non_duration_mask = ~duration_metrics
        
        # Enforce 12-hourly intervals (not just range) for non-duration metrics
        valid_lead_times = (
            (df["lead_time_hours"] > 0)
            & (df["lead_time_hours"] % 12 == 0)
        )
        df = df[no_lead_time | duration_metrics | (non_duration_mask & valid_lead_times)]
        
        max_lh = int(df["lead_time_hours"].max())
        print(f"  After lead time filtering (12-hourly, up to {max_lh}h): {len(df)} rows")

    # Filter init_time to 00Z/12Z only (consistent with evaluation scripts)
    if "init_time" in df.columns:
        init_dt = pd.to_datetime(df["init_time"], errors="coerce")
        no_init = df["init_time"].isna() | (df["init_time"] == "") | init_dt.isna()
        valid_init_hours = init_dt.dt.hour.isin([0, 12])
        before = len(df)
        df = df[no_init | valid_init_hours]
        after = len(df)
        if after < before:
            print(f"  After init time filtering (00Z/12Z only): {after} rows (dropped {before - after})")

    # Derive valid_time when possible (for by-valid-time analysis).
    # CSV may only include init_time + lead_time.
    if "init_time" in df.columns and "lead_time" in df.columns:
        init_dt = pd.to_datetime(df["init_time"], errors="coerce")
        lead_td = pd.to_timedelta(df["lead_time"], errors="coerce")
        df["valid_time"] = init_dt + lead_td
        n_valid = int(df["valid_time"].notna().sum())
        if n_valid > 0:
            print(f"  Derived valid_time for {n_valid} rows from init_time + lead_time")
    
    # Convert pressure-based metrics from Pa to hPa
    pa_to_hpa_metrics = [
        "landfall_intensity_mae",
        "landfall_intensity_rmse",
        "track_intensity_mae",
        "TrackIntensityMeanAbsoluteError",
        "LandfallIntensityRootMeanSquaredError",
    ]
    pa_mask = df["metric_name"].isin(pa_to_hpa_metrics)
    if pa_mask.any():
        df.loc[pa_mask, "metric_value"] = df.loc[pa_mask, "metric_value"] / 100.0
        print(f"  Converted {pa_mask.sum()} rows from Pa → hPa ({', '.join(pa_to_hpa_metrics)})")
    
    return df


def get_model_style(forecast_name: str) -> dict:
    """
    Get the visual style (color, marker, linestyle) for a forecast model.
    Returns a dict with keys: 'color', 'marker', 'linestyle'
    """
    return _assign_style(forecast_name)



def order_forecasts(forecast_names) -> list[str]:
    """Order models by style groups instead of alphabetical order.

    Also pre-assigns palette colours so ordering is deterministic
    (physical models get Pastel2 slots first, then AI models get deep slots).
    """
    unique_names = list(dict.fromkeys([str(n) for n in forecast_names if pd.notna(n)]))
    ordered = [name for name in MODEL_GROUP_ORDER if name in unique_names]
    remaining = sorted([name for name in unique_names if name not in ordered])
    result = ordered + remaining
    for name in result:
        _assign_style(name)
    return result


def get_model_plot_params(forecast_name: str, base_markersize: float) -> dict:
    """Return per-model plot emphasis while preserving color/style groups."""
    params = {
        "linewidth": 2.5,
        "markersize": base_markersize,
        "markeredgewidth": 1.5 if base_markersize >= 7 else 1.2,
        "markeredgecolor": "white",
        "alpha": 0.9,
        "zorder": 2,
    }
    if "WeatherMesh" in forecast_name:
        params["linewidth"] = 3.0
        params["zorder"] = 4
    if forecast_name == "WeatherNext2":
        params["linewidth"] = 3.2
        params["markersize"] = base_markersize + 1.5
        params["markeredgewidth"] = max(params["markeredgewidth"], 1.8)
        params["markeredgecolor"] = "black"
        params["alpha"] = 0.98
        params["zorder"] = 5
    return params


def compute_consensus_init_start(
    df: pd.DataFrame,
    min_fraction: float = 0.6,
) -> pd.Timestamp | None:
    """First init_time where a majority of models have valid TC track metrics."""
    if "init_time" not in df.columns:
        return None

    track_metrics = {
        "along_track_error",
        "cross_track_error",
        "total_track_error",
        "AlongTrackError",
        "CrossTrackError",
        "TotalTrackError",
    }
    track_df = df[df["metric_name"].isin(track_metrics)].copy()
    if track_df.empty:
        return None

    track_df = track_df[
        track_df["init_time"].notna()
        & (track_df["init_time"] != "")
        & track_df["metric_value"].notna()
    ].copy()
    if track_df.empty:
        return None

    track_df["init_time"] = pd.to_datetime(track_df["init_time"], errors="coerce")
    track_df = track_df[track_df["init_time"].notna()]
    if track_df.empty:
        return None

    n_models = len(df["forecast_name"].dropna().unique())
    if n_models == 0:
        return None
    required = int(np.ceil(n_models * min_fraction))

    active_models_by_init = (
        track_df.groupby("init_time")["forecast_name"].nunique().sort_index()
    )
    eligible = active_models_by_init[active_models_by_init >= required]
    if eligible.empty:
        return None

    return pd.Timestamp(eligible.index.min())


def plot_metric_by_leadtime(
    df: pd.DataFrame,
    metric_name: str,
    output_dir: Path,
    case_label: str = "",
):
    """Plot a metric as a function of lead time."""
    metric_df = df[df["metric_name"] == metric_name].copy()
    
    if len(metric_df) == 0:
        print(f"  ⚠️  No data for {metric_name}, skipping")
        return
    
    # Skip if no lead_time data (e.g., landfall metrics)
    if "lead_time" not in metric_df.columns or metric_df["lead_time"].isna().all():
        print(f"  ⚠️  No lead_time data for {metric_name}, skipping lead time plot")
        return
    
    # Filter out rows with empty lead_time
    metric_df = metric_df[metric_df["lead_time"].notna() & (metric_df["lead_time"] != "")]
    
    if len(metric_df) == 0:
        print(f"  ⚠️  No valid lead_time data for {metric_name}, skipping lead time plot")
        return
    
    # Get the variable name (should be consistent within one metric)
    if "target_variable" in metric_df.columns:
        variable = metric_df["target_variable"].iloc[0]
    else:
        variable = "unknown"
    
    # Convert lead_time to hours
    if "lead_time_hours" not in metric_df.columns:
        try:
            metric_df["lead_time_hours"] = (
                pd.to_timedelta(metric_df["lead_time"]).dt.total_seconds() / 3600
            )
        except Exception:
            print(f"  ⚠️  Could not parse lead_time for {metric_name}, skipping lead time plot")
            return
    
    # Take absolute value for track error metrics (before averaging across inits)
    track_error_metrics = ["along_track_error", "cross_track_error", "AlongTrackError", "CrossTrackError"]
    if metric_name in track_error_metrics:
        metric_df["metric_value"] = metric_df["metric_value"].abs()

    # Only keep init times where ALL models have data
    if "init_time" in metric_df.columns:
        n_models = metric_df["forecast_name"].nunique()
        has_init = metric_df["init_time"].notna()
        models_per_init = (
            metric_df.loc[has_init]
            .groupby("init_time")["forecast_name"]
            .nunique()
        )
        complete_inits = models_per_init[models_per_init == n_models].index
        n_before = metric_df.loc[has_init, "init_time"].nunique()
        metric_df = metric_df[~has_init | metric_df["init_time"].isin(complete_inits)].copy()
        n_after = metric_df.loc[metric_df["init_time"].notna(), "init_time"].nunique()
        if n_after < n_before:
            print(f"    Dropped {n_before - n_after} init times with incomplete model coverage")

    # Group by forecast and lead time, compute mean
    grouped = (
        metric_df.groupby(["forecast_name", "lead_time_hours"])["metric_value"]
        .mean()
        .reset_index()
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each forecast with its designated style
    for forecast in order_forecasts(grouped["forecast_name"].unique()):
        forecast_data = grouped[grouped["forecast_name"] == forecast]
        style = get_model_style(forecast)
        draw = get_model_plot_params(forecast, base_markersize=8)
        
        ax.plot(
            forecast_data["lead_time_hours"],
            forecast_data["metric_value"],
            label=forecast,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=draw["linewidth"],
            markersize=draw["markersize"],
            markeredgewidth=draw["markeredgewidth"],
            markeredgecolor=draw["markeredgecolor"],
            alpha=draw["alpha"],
            zorder=draw["zorder"],
        )
    
    # Get display labels
    y_label = get_display_label(variable, metric_name, include_units=True)
    title_label = get_display_label(variable, metric_name, include_units=False)
    
    ax.set_xlabel("Lead Time (hours)", fontsize=13, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=13, fontweight="bold")
    ax.set_title(f"{title_label} vs Lead Time", fontsize=15, fontweight="bold")
    ax.legend(frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    if metric_name in SIGNED_METRICS:
        ax.axhline(0.0, color="black", linewidth=2, alpha=0.8, zorder=1)
    
    # Save with case label in filename
    prefix = f"{case_label}_" if case_label else ""
    filename_label = get_filename_safe_label(variable, metric_name)
    output_file = output_dir / f"{prefix}{filename_label}_by_leadtime.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved {output_file.name}")
    
    # Create zoomed version (0-96 hours)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each forecast, filtered to 0-96 hours
    for forecast in order_forecasts(grouped["forecast_name"].unique()):
        forecast_data = grouped[grouped["forecast_name"] == forecast]
        # Filter to 0-96 hours
        forecast_data = forecast_data[
            (forecast_data["lead_time_hours"] >= 0) & 
            (forecast_data["lead_time_hours"] <= 96)
        ]
        
        if len(forecast_data) > 0:
            style = get_model_style(forecast)
            draw = get_model_plot_params(forecast, base_markersize=8)
            
            ax.plot(
                forecast_data["lead_time_hours"],
                forecast_data["metric_value"],
                label=forecast,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=draw["linewidth"],
                markersize=draw["markersize"],
                markeredgewidth=draw["markeredgewidth"],
                markeredgecolor=draw["markeredgecolor"],
                alpha=draw["alpha"],
                zorder=draw["zorder"],
            )
    
    ax.set_xlabel("Lead Time (hours)", fontsize=13, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=13, fontweight="bold")
    ax.set_title(f"{title_label} vs Lead Time (0-96h)", fontsize=15, fontweight="bold")
    ax.legend(frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    if metric_name in SIGNED_METRICS:
        ax.axhline(0.0, color="black", linewidth=2, alpha=0.8, zorder=1)
    ax.set_xlim(0, 96)
    
    # Save zoomed version
    output_file_zoomed = output_dir / f"{prefix}{filename_label}_by_leadtime_0-96h.png"
    plt.tight_layout()
    plt.savefig(output_file_zoomed, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved {output_file_zoomed.name} (zoomed 0-96h)")


def plot_metric_by_validtime(
    df: pd.DataFrame,
    metric_name: str,
    output_dir: Path,
    case_label: str = "",
    landfall_times: list[tuple] | None = None,
    event_times: list[tuple[str, pd.Timestamp]] | None = None,
):
    """Plot a metric as a function of valid time."""
    metric_df = df[df["metric_name"] == metric_name].copy()
    if len(metric_df) == 0:
        return

    if "valid_time" not in metric_df.columns or metric_df["valid_time"].isna().all():
        print(f"  ⚠️  No valid_time data for {metric_name}, skipping valid time plot")
        return

    metric_df = metric_df[metric_df["valid_time"].notna()].copy()
    if len(metric_df) == 0:
        print(f"  ⚠️  No non-NaN valid_time data for {metric_name}, skipping valid time plot")
        return

    metric_df["valid_time"] = pd.to_datetime(metric_df["valid_time"], errors="coerce")
    metric_df = metric_df[metric_df["valid_time"].notna()]
    if len(metric_df) == 0:
        print(f"  ⚠️  Could not parse valid_time for {metric_name}, skipping valid time plot")
        return

    if "target_variable" in metric_df.columns:
        variable = metric_df["target_variable"].iloc[0]
    else:
        variable = "unknown"

    grouped = (
        metric_df.groupby(["forecast_name", "valid_time"])["metric_value"]
        .mean()
        .reset_index()
    )
    if len(grouped) == 0:
        print(f"  ⚠️  No grouped valid_time values for {metric_name}, skipping")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    for forecast in order_forecasts(grouped["forecast_name"].unique()):
        forecast_data = grouped[grouped["forecast_name"] == forecast]
        style = get_model_style(forecast)
        draw = get_model_plot_params(forecast, base_markersize=7)
        ax.plot(
            forecast_data["valid_time"],
            forecast_data["metric_value"],
            label=forecast,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=draw["linewidth"],
            markersize=draw["markersize"],
            markeredgewidth=draw["markeredgewidth"],
            markeredgecolor=draw["markeredgecolor"],
            alpha=draw["alpha"],
            zorder=draw["zorder"],
        )

    y_label = get_display_label(variable, metric_name, include_units=True)
    title_label = get_display_label(variable, metric_name, include_units=False)
    ax.set_xlabel("Valid Time", fontsize=13, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=13, fontweight="bold")
    ax.set_title(f"{title_label} vs Valid Time", fontsize=15, fontweight="bold")
    ax.legend(frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    if metric_name in SIGNED_METRICS:
        ax.axhline(0.0, color="black", linewidth=2, alpha=0.8, zorder=1)

    if landfall_times and len(landfall_times) > 0:
        # Use only timestamps that have at least one plotted (non-NaN) value.
        plotted_valid = grouped[grouped["metric_value"].notna()]
        if len(plotted_valid) == 0:
            plotted_valid = grouped
        valid_time_min = plotted_valid["valid_time"].min()
        valid_time_max = plotted_valid["valid_time"].max()
        y_lim = ax.get_ylim()
        y_range = y_lim[1] - y_lim[0]
        for landfall_id, landfall_time in landfall_times:
            landfall_ts = pd.Timestamp(landfall_time)
            # Only annotate landfalls that are inside the plotted valid-time data range.
            if landfall_ts < valid_time_min or landfall_ts > valid_time_max:
                continue
            ax.axvline(
                landfall_ts,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                zorder=1,
            )
            ax.text(
                landfall_ts,
                y_lim[1] - 0.05 * y_range,
                f"LF{landfall_id}",
                rotation=0,
                ha="center",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                ),
                zorder=10,
            )

    # Annotate heat/freeze event start/end on valid-time plots.
    if event_times:
        y_lim = ax.get_ylim()
        y_range = y_lim[1] - y_lim[0]
        for evt_label, evt_ts in event_times:
            ax.axvline(
                evt_ts,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                zorder=1,
            )
            ax.text(
                evt_ts,
                y_lim[1] - 0.05 * y_range,
                evt_label,
                rotation=0,
                ha="center",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="lightyellow",
                    edgecolor="red",
                    alpha=0.9,
                ),
                zorder=10,
            )

    # Show later timestamps on the left.
    ax.invert_xaxis()
    plt.xticks(rotation=45, ha="right")

    prefix = f"{case_label}_" if case_label else ""
    filename_label = get_filename_safe_label(variable, metric_name)
    output_file = output_dir / f"{prefix}{filename_label}_by_validtime.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {output_file.name}")


def _pick_first_available_metric(
    available: set[str],
    candidates: list[str],
) -> str | None:
    """Pick first available metric from a priority-ordered list."""
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _resolve_landfall_panel_metrics(df: pd.DataFrame) -> dict[str, str | None]:
    """Resolve metric names for the 4-panel landfall-relative plot."""
    available = set(df["metric_name"].dropna().astype(str).unique())

    # Track metrics: handle lowercase/camelcase variants.
    total_track = _pick_first_available_metric(
        available, ["total_track_error", "TotalTrackError"]
    )
    along_track = _pick_first_available_metric(
        available, ["along_track_error", "AlongTrackError"]
    )
    cross_track = _pick_first_available_metric(
        available, ["cross_track_error", "CrossTrackError"]
    )

    # MSLP panel metric:
    # Prefer track-intensity MAE (same variable lineage as landfall_intensity_mae).
    # Fall back to minimum-MSLP-style metrics if present.
    mslp_metric = None
    if "target_variable" in df.columns:
        mslp_df = df[df["target_variable"] == "air_pressure_at_mean_sea_level"].copy()
        if len(mslp_df) > 0:
            mslp_names = mslp_df["metric_name"].dropna().astype(str).unique().tolist()
            # First preference: explicit track intensity MAE metric.
            for preferred in [
                "track_intensity_rmse",
                "TrackIntensityRootMeanSquaredError",
                "track_intensity_mae",
                "TrackIntensityMeanAbsoluteError",
            ]:
                if preferred in mslp_names:
                    mslp_metric = preferred
                    break

            # Second preference: known minimum-intensity metric names.
            for preferred in [
                "MinimumMeanAbsoluteError",
                "minimum_mean_absolute_error",
                "minimum_mae",
                "min_mae",
            ]:
                if mslp_metric is not None:
                    break
                if preferred in mslp_names:
                    mslp_metric = preferred
                    break
            if mslp_metric is None:
                # Fallback pattern matching.
                for name in mslp_names:
                    low = name.lower()
                    if "min" in low and ("absolute" in low or "mae" in low):
                        mslp_metric = name
                        break

    return {
        "mslp": mslp_metric,
        "total": total_track,
        "along": along_track,
        "cross": cross_track,
    }


def _prepare_landfall_metric_grouped(
    df: pd.DataFrame,
    metric_name: str | None,
    all_models: list[str] | None = None,
) -> tuple[pd.DataFrame | None, str, str] | tuple[None, None, None]:
    """Prepare grouped valid-time data for a single panel metric."""
    if metric_name is None:
        return None, None, None

    metric_df = df[df["metric_name"] == metric_name].copy()
    if len(metric_df) == 0:
        return None, None, None
    if "valid_time" not in metric_df.columns:
        return None, None, None

    metric_df = metric_df[metric_df["valid_time"].notna()].copy()
    metric_df["valid_time"] = pd.to_datetime(metric_df["valid_time"], errors="coerce")
    metric_df = metric_df[metric_df["valid_time"].notna()].copy()
    if len(metric_df) == 0:
        return None, None, None

    # Fairness guard: only keep init_times where all models have non-NaN values.
    if "init_time" in metric_df.columns and all_models:
        metric_df["init_time"] = pd.to_datetime(metric_df["init_time"], errors="coerce")
        init_ready = metric_df[
            metric_df["init_time"].notna() & metric_df["metric_value"].notna()
        ].copy()
        if len(init_ready) > 0:
            models_per_init = (
                init_ready.groupby("init_time")["forecast_name"].nunique().sort_index()
            )
            required_models = len(all_models)
            common_inits = models_per_init[models_per_init >= required_models].index
            if len(common_inits) == 0:
                print(
                    f"  ⚠️  No common init_times across all models for {metric_name}; "
                    "excluding from landfall panel"
                )
                return None, None, None
            before_n = metric_df["init_time"].nunique()
            metric_df = metric_df[metric_df["init_time"].isin(common_inits)].copy()
            after_n = metric_df["init_time"].nunique()
            print(
                f"    Fair-init filter for {metric_name}: "
                f"{after_n}/{before_n} init_times kept "
                f"(all {required_models} models present)"
            )

    grouped = (
        metric_df.groupby(["forecast_name", "valid_time"])["metric_value"]
        .mean()
        .reset_index()
    )
    grouped = grouped[grouped["metric_value"].notna()].copy()
    if len(grouped) == 0:
        return None, None, None

    if "target_variable" in metric_df.columns:
        non_nan_var = metric_df["target_variable"].dropna()
        variable = str(non_nan_var.iloc[0]) if len(non_nan_var) > 0 else "unknown"
    else:
        variable = "unknown"

    y_label = get_display_label(variable, metric_name, include_units=True)
    title_label = get_display_label(variable, metric_name, include_units=False)
    return grouped, y_label, title_label


def plot_landfall_relative_panels(
    df: pd.DataFrame,
    output_dir: Path,
    case_label: str = "",
    landfall_times: list[tuple] | None = None,
    all_models: list[str] | None = None,
):
    """Create 4-panel landfall-relative plots (MSLP, total/along/cross track)."""
    if not landfall_times:
        return

    metric_map = _resolve_landfall_panel_metrics(df)
    panel_order = [
        ("mslp", "MSLP Error"),
        ("total", "Total Track Error"),
        ("along", "Along-Track Error"),
        ("cross", "Cross-Track Error"),
    ]

    prepared: dict[str, tuple[pd.DataFrame, str, str] | None] = {}
    for key, _fallback_title in panel_order:
        grouped, y_label, title_label = _prepare_landfall_metric_grouped(
            df, metric_map[key], all_models=all_models
        )
        prepared[key] = (grouped, y_label, title_label) if grouped is not None else None

    if all(prepared[key] is None for key, _ in panel_order):
        print("  ⚠️  No data available for landfall-relative panel metrics; skipping")
        return

    # Only keep landfalls that are within at least one panel's valid-time range.
    in_range_landfalls = []
    for lf_id, lf_time in landfall_times:
        lf_ts = pd.Timestamp(lf_time)
        in_any = False
        for key, _ in panel_order:
            if prepared[key] is None:
                continue
            grouped = prepared[key][0]
            if grouped["valid_time"].min() <= lf_ts <= grouped["valid_time"].max():
                in_any = True
                break
        if in_any:
            in_range_landfalls.append((lf_id, lf_ts))

    if len(in_range_landfalls) == 0:
        print("  ⚠️  No in-range landfalls for panel plots; skipping")
        return

    prefix = f"{case_label}_" if case_label else ""
    for lf_id, lf_ts in in_range_landfalls:
        fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
        all_hours = []

        for ax_idx, (key, fallback_title) in enumerate(panel_order):
            ax = axes[ax_idx]
            panel_data = prepared[key]
            ax.grid(True, alpha=0.3)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
            if key in {"total", "along", "cross"}:
                ax.axhline(0.0, color="black", linewidth=2, alpha=0.8, zorder=1)

            if panel_data is None:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {fallback_title}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                ax.set_ylabel(fallback_title, fontsize=11, fontweight="bold")
                continue

            grouped, y_label, title_label = panel_data
            rel_df = grouped.copy()
            rel_df["hours_from_landfall"] = (
                rel_df["valid_time"] - lf_ts
            ).dt.total_seconds() / 3600.0
            rel_df = rel_df[rel_df["metric_value"].notna()].copy()
            if len(rel_df) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No values near LF{lf_id}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
                ax.set_title(title_label, fontsize=12, fontweight="bold")
                continue

            all_hours.extend(rel_df["hours_from_landfall"].tolist())
            for forecast in order_forecasts(rel_df["forecast_name"].unique()):
                forecast_data = (
                    rel_df[rel_df["forecast_name"] == forecast]
                    .sort_values("hours_from_landfall")
                )
                style = get_model_style(forecast)
                draw = get_model_plot_params(forecast, base_markersize=6)
                ax.plot(
                    forecast_data["hours_from_landfall"],
                    forecast_data["metric_value"],
                    label=forecast,
                    marker=style["marker"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=draw["linewidth"],
                    markersize=draw["markersize"],
                    markeredgewidth=draw["markeredgewidth"],
                    markeredgecolor=draw["markeredgecolor"],
                    alpha=draw["alpha"],
                    zorder=draw["zorder"],
                )

            ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
            ax.set_title(title_label, fontsize=12, fontweight="bold")

            if ax_idx == 0:
                y_lim = ax.get_ylim()
                y_range = y_lim[1] - y_lim[0]
                ax.text(
                    0.0,
                    y_lim[1] - 0.05 * y_range,
                    f"LF{lf_id}",
                    rotation=0,
                    ha="center",
                    va="top",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="black",
                        alpha=0.8,
                    ),
                    zorder=10,
                )
                ax.legend(frameon=True, shadow=True, fontsize=9, loc="best")

            # Match the user's preferred direction: closer/later on the left.
            ax.invert_xaxis()

        if len(all_hours) > 0:
            min_h = float(np.nanmin(all_hours))
            max_h = float(np.nanmax(all_hours))
            axes[-1].set_xlim(min_h, max_h)

        axes[-1].set_xlabel(
            f"Hours Relative to LF{lf_id} (closer to landfall on the left)",
            fontsize=13,
            fontweight="bold",
        )
        fig.suptitle(
            f"Landfall-Relative Metrics Around LF{lf_id} ({lf_ts})",
            fontsize=15,
            fontweight="bold",
        )

        output_file = output_dir / f"{prefix}landfall_relative_panel_lf{lf_id}.png"
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {output_file.name}")


def plot_metric_by_inittime(
    df: pd.DataFrame,
    metric_name: str,
    output_dir: Path,
    case_label: str = "",
    landfall_times: list[tuple] = None,
    min_init_time: pd.Timestamp | None = None,
    event_times: list[tuple[str, pd.Timestamp]] | None = None,
):
    """Plot a metric as a function of initialization time.
    
    Args:
        df: Results dataframe
        metric_name: Name of metric to plot
        output_dir: Directory to save plots
        case_label: Label for filename
        landfall_times: List of (landfall_id, datetime) tuples for observed landfalls
    """
    metric_df = df[df["metric_name"] == metric_name].copy()

    if len(metric_df) == 0:
        print(f"  ⚠️  No data for {metric_name}, skipping")
        return

    if "init_time" not in metric_df.columns or metric_df["init_time"].isna().all():
        print(f"  ⚠️  No init_time data for {metric_name}, skipping init time plot")
        return

    metric_df = metric_df[metric_df["init_time"].notna() & (metric_df["init_time"] != "")].copy()
    if len(metric_df) == 0:
        print(f"  ⚠️  No valid init_time data for {metric_name}, skipping init time plot")
        return

    if "target_variable" in metric_df.columns:
        variable = metric_df["target_variable"].iloc[0]
    else:
        variable = "unknown"

    metric_df["init_time"] = pd.to_datetime(metric_df["init_time"], errors="coerce")
    metric_df = metric_df[metric_df["init_time"].notna()].copy()
    if len(metric_df) == 0:
        print(f"  ⚠️  Could not parse init_time for {metric_name}, skipping init time plot")
        return

    if min_init_time is not None:
        metric_df = metric_df[metric_df["init_time"] >= min_init_time].copy()
        if len(metric_df) == 0:
            print(
                f"  ⚠️  No init_time data for {metric_name} after cutoff "
                f"{pd.Timestamp(min_init_time)}, skipping"
            )
            return

    landfall_specific_metrics = [
        "landfall_displacement",
        "landfall_time_me",
        "landfall_intensity_mae",
        "landfall_intensity_rmse",
    ]
    is_landfall_metric = metric_name in landfall_specific_metrics

    if "landfall" in metric_df.columns and is_landfall_metric:
        landfalls = sorted(metric_df["landfall"].dropna().unique())
        if len(landfalls) == 0:
            landfalls = [None]
    else:
        landfalls = [None]

    group_cols = ["forecast_name", "init_time"]
    y_label = get_display_label(variable, metric_name, include_units=True)
    title_label = get_display_label(variable, metric_name, include_units=False)
    prefix = f"{case_label}_" if case_label else ""
    filename_label = get_filename_safe_label(variable, metric_name)

    for landfall_idx in landfalls:
        if landfall_idx is not None:
            plot_df = metric_df[metric_df["landfall"] == landfall_idx].copy()
            landfall_label = f" (Landfall {landfall_idx})"
            landfall_suffix = f"_lf{landfall_idx}"
        else:
            plot_df = metric_df.copy()
            landfall_label = ""
            landfall_suffix = ""

        if "metric_value" not in plot_df.columns or plot_df["metric_value"].isna().all():
            if landfall_idx is not None:
                print(
                    f"  ⚠️  No valid values for {metric_name} at landfall={landfall_idx}; "
                    "skipping this plot"
                )
            else:
                print(f"  ⚠️  No valid values for {metric_name}; skipping init time plot")
            continue

        # Carry event_inherited and n_event_points through the groupby
        # if the columns exist.
        has_inherited_col = (
            "event_inherited" in plot_df.columns
            and plot_df["event_inherited"].notna().any()
        )
        has_npts_col = (
            "n_event_points" in plot_df.columns
            and plot_df["n_event_points"].notna().any()
        )
        agg_dict: dict = {"metric_value": "mean"}
        if has_inherited_col:
            # Convert to bool-like for aggregation (True if any row is inherited)
            plot_df["event_inherited"] = plot_df["event_inherited"].fillna(False).astype(bool)
            agg_dict["event_inherited"] = "max"
        if has_npts_col:
            agg_dict["n_event_points"] = "mean"
        has_ntgt_col = (
            "n_target_events" in plot_df.columns
            and plot_df["n_target_events"].notna().any()
        )
        if has_ntgt_col:
            agg_dict["n_target_events"] = "mean"
        grouped = plot_df.groupby(group_cols).agg(agg_dict).reset_index()

        if len(grouped) == 0:
            if landfall_idx is not None:
                print(
                    f"  ⚠️  No grouped values for {metric_name} at landfall={landfall_idx}; "
                    "skipping this plot"
                )
            else:
                print(f"  ⚠️  No grouped values for {metric_name}; skipping init time plot")
            continue

        # Drop init times where ANY model is missing data so comparisons
        # are always apples-to-apples across all models.
        n_models = grouped["forecast_name"].nunique()
        models_per_init = (
            grouped[grouped["metric_value"].notna()]
            .groupby("init_time")["forecast_name"]
            .nunique()
        )
        complete_inits = models_per_init[models_per_init == n_models].index
        n_before = grouped["init_time"].nunique()
        grouped = grouped[grouped["init_time"].isin(complete_inits)].copy()
        n_after = grouped["init_time"].nunique()
        if n_after < n_before:
            print(f"    Dropped {n_before - n_after} init times with incomplete model coverage")

        # For event timing metrics, mask out init times where a model has
        # n_event_points == 0 (no forecast event detected at any station).
        # Those values are pure penalty scores and not meaningful skill.
        if metric_name in EVENT_TIMING_METRICS and has_npts_col and "n_event_points" in grouped.columns:
            no_fc_event = grouped["n_event_points"] == 0
            grouped.loc[no_fc_event, "metric_value"] = np.nan

        # For event timing metrics, trim to init times where at least one
        # model has a finite value — later inits don't have enough lead time
        # to capture the full event, so they are all NaN.
        if metric_name in EVENT_TIMING_METRICS:
            finite_mask = grouped["metric_value"].notna()
            if finite_mask.any():
                init_times_with_data = grouped.loc[finite_mask, "init_time"]
                earliest_useful = init_times_with_data.min()
                latest_useful = init_times_with_data.max()
                # Keep only init times within the range that has real data,
                # plus a small buffer so the plot doesn't clip markers.
                buf = pd.Timedelta(hours=6)
                grouped = grouped[
                    (grouped["init_time"] >= earliest_useful - buf)
                    & (grouped["init_time"] <= latest_useful + buf)
                ].copy()

        # Use a two-panel figure for event timing metrics when point
        # counts are available, otherwise a single-panel figure.
        show_npts = (
            has_npts_col
            and metric_name in EVENT_TIMING_METRICS
            and "n_event_points" in grouped.columns
            and grouped["n_event_points"].notna().any()
        )
        # Label depends on whether this is station-based verification
        _is_station_target = (
            "target_source" in plot_df.columns
            and plot_df["target_source"].str.contains("GHCN", case=False, na=False).any()
        )
        _npts_label = "# Stations" if _is_station_target else "# Grid Points"
        if show_npts:
            fig, (ax, ax_n) = plt.subplots(
                2, 1, figsize=(14, 9.5), sharex=True,
                gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
            )
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax_n = None

        for forecast in order_forecasts(grouped["forecast_name"].unique()):
            forecast_data = grouped[grouped["forecast_name"] == forecast]
            style = get_model_style(forecast)
            draw = get_model_plot_params(forecast, base_markersize=7)

            # Plot the full connected line (all points, solid markers)
            ax.plot(
                forecast_data["init_time"],
                forecast_data["metric_value"],
                label=forecast,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=draw["linewidth"],
                markersize=draw["markersize"],
                markeredgewidth=draw["markeredgewidth"],
                markeredgecolor=draw["markeredgecolor"],
                alpha=draw["alpha"],
                zorder=draw["zorder"],
            )

            # Overlay open/unfilled markers on inherited points
            if has_inherited_col and "event_inherited" in forecast_data.columns:
                inherited = forecast_data[
                    forecast_data["event_inherited"].fillna(False).astype(bool)
                ]
                if len(inherited) > 0:
                    ax.scatter(
                        inherited["init_time"],
                        inherited["metric_value"],
                        marker=style["marker"],
                        facecolors="white",
                        edgecolors=style["color"],
                        linewidths=draw["markeredgewidth"] + 0.5,
                        s=(draw["markersize"] + 2) ** 2,
                        alpha=draw["alpha"],
                        zorder=draw["zorder"] + 1,
                    )

            # Bottom panel: station / grid-point count
            if ax_n is not None and "n_event_points" in forecast_data.columns:
                npts = forecast_data["n_event_points"].values
                if np.any(np.isfinite(npts)):
                    ax_n.plot(
                        forecast_data["init_time"],
                        npts,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=draw["linewidth"] * 0.8,
                        marker=style["marker"],
                        markersize=draw["markersize"] * 0.7,
                        alpha=draw["alpha"],
                    )

        # Observed event count line on bottom panel (same for all models,
        # so average across models at each init time to get one line).
        if ax_n is not None and has_ntgt_col and "n_target_events" in grouped.columns:
            obs_counts = (
                grouped.groupby("init_time")["n_target_events"]
                .mean()
                .reset_index()
                .sort_values("init_time")
            )
            obs_counts = obs_counts[obs_counts["n_target_events"].notna()]
            if not obs_counts.empty:
                ax_n.plot(
                    obs_counts["init_time"],
                    obs_counts["n_target_events"],
                    color="red", linewidth=1.5, linestyle="--",
                    label="Observed",
                    zorder=10,
                )
                ax_n.legend(fontsize=8, loc="upper right")

        # For event timing metrics, lock x-limits to the data range so that
        # out-of-range markers don't stretch the axis.
        # Use normal order here; invert_xaxis() below flips it.
        _evt_xlim_set = False
        if metric_name in EVENT_TIMING_METRICS and len(grouped) > 0:
            data_xmin = grouped["init_time"].min()
            data_xmax = grouped["init_time"].max()
            pad = pd.Timedelta(hours=12)
            ax.set_xlim(
                mdates.date2num(data_xmin - pad),
                mdates.date2num(data_xmax + pad),
            )
            _evt_xlim_set = True

        # Annotate landfall and event markers, but only if they fall within
        # the visible x range (prevents stretching for event-timing plots).
        x_lim_num = ax.get_xlim()
        xlo, xhi = min(x_lim_num), max(x_lim_num)
        y_lim = ax.get_ylim()
        y_range = y_lim[1] - y_lim[0]

        if not is_landfall_metric and landfall_times and len(landfall_times) > 0:
            for landfall_id, landfall_time in landfall_times:
                if _evt_xlim_set and not (xlo <= mdates.date2num(landfall_time) <= xhi):
                    continue
                ax.axvline(
                    landfall_time,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.6,
                    zorder=1,
                )
                label_text = f"LF{landfall_id}" if len(landfall_times) > 1 else "Landfall"
                ax.text(
                    landfall_time,
                    y_lim[1] - 0.05 * y_range,
                    label_text,
                    rotation=0,
                    ha="center",
                    va="top",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="black",
                        alpha=0.8,
                    ),
                    zorder=10,
                )

        if event_times:
            for evt_label, evt_ts in event_times:
                evt_num = mdates.date2num(evt_ts)
                # Only draw if the marker falls within the visible x range.
                if _evt_xlim_set and not (xlo <= evt_num <= xhi):
                    continue
                ax.axvline(
                    evt_ts,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=1,
                )
                ax.text(
                    evt_ts,
                    y_lim[1] - 0.05 * y_range,
                    evt_label,
                    rotation=0,
                    ha="center",
                    va="top",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="lightyellow",
                        edgecolor="red",
                        alpha=0.9,
                    ),
                    zorder=10,
                )

        ax.set_ylabel(y_label, fontsize=13, fontweight="bold")
        ax.set_title(
            f"{title_label}{landfall_label} vs Initialization Time",
            fontsize=15,
            fontweight="bold",
        )
        # Add an "inherited" legend entry if any inherited points were plotted
        if has_inherited_col and grouped["event_inherited"].any():
            from matplotlib.lines import Line2D
            handles, labels = ax.get_legend_handles_labels()
            handles.append(
                Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor="none", markeredgecolor="gray",
                    markeredgewidth=1.5, markersize=9,
                    label="Inherited (open)",
                )
            )
            labels.append("Inherited (open)")
            ax.legend(handles=handles, labels=labels, frameon=True, shadow=True, fontsize=10)
        else:
            ax.legend(frameon=True, shadow=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        if metric_name in SIGNED_METRICS:
            ax.axhline(0.0, color="black", linewidth=2, alpha=0.8, zorder=1)

        # Configure bottom panel (point/station count) or main-axis x-label
        if ax_n is not None:
            from matplotlib.ticker import MaxNLocator
            ax_n.set_ylabel(_npts_label, fontsize=11, fontweight="bold")
            ax_n.set_xlabel("Initialization Time", fontsize=13, fontweight="bold")
            ax_n.grid(True, alpha=0.3)
            ax_n.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.sca(ax_n)
            plt.xticks(rotation=45, ha="right")
            # Hide x-tick labels on the main axis since they share x
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Initialization Time", fontsize=13, fontweight="bold")
            plt.xticks(rotation=45, ha="right")

        # Show later timestamps on the left (only call once; sharex propagates).
        ax.invert_xaxis()

        output_file = output_dir / f"{prefix}{filename_label}_by_inittime{landfall_suffix}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved {output_file.name}")

        if show_npts:
            fig, (ax, ax_n) = plt.subplots(
                2, 1, figsize=(14, 9.5), sharex=True,
                gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
            )
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax_n = None

        max_init_time = grouped["init_time"].max()
        cutoff_time = max_init_time - pd.Timedelta(days=4)

        for forecast in order_forecasts(grouped["forecast_name"].unique()):
            forecast_data = grouped[grouped["forecast_name"] == forecast]
            forecast_data = forecast_data[forecast_data["init_time"] >= cutoff_time]
            if len(forecast_data) == 0:
                continue
            style = get_model_style(forecast)
            draw = get_model_plot_params(forecast, base_markersize=7)

            ax.plot(
                forecast_data["init_time"],
                forecast_data["metric_value"],
                label=forecast,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=draw["linewidth"],
                markersize=draw["markersize"],
                markeredgewidth=draw["markeredgewidth"],
                markeredgecolor=draw["markeredgecolor"],
                alpha=draw["alpha"],
                zorder=draw["zorder"],
            )
            if has_inherited_col and "event_inherited" in forecast_data.columns:
                inherited = forecast_data[
                    forecast_data["event_inherited"].fillna(False).astype(bool)
                ]
                if len(inherited) > 0:
                    ax.scatter(
                        inherited["init_time"],
                        inherited["metric_value"],
                        marker=style["marker"],
                        facecolors="white",
                        edgecolors=style["color"],
                        linewidths=draw["markeredgewidth"] + 0.5,
                        s=(draw["markersize"] + 2) ** 2,
                        alpha=draw["alpha"],
                        zorder=draw["zorder"] + 1,
                    )

            # Bottom panel: station / grid-point count
            if ax_n is not None and "n_event_points" in forecast_data.columns:
                npts = forecast_data["n_event_points"].values
                if np.any(np.isfinite(npts)):
                    ax_n.plot(
                        forecast_data["init_time"],
                        npts,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=draw["linewidth"] * 0.8,
                        marker=style["marker"],
                        markersize=draw["markersize"] * 0.7,
                        alpha=draw["alpha"],
                    )

        # Observed event count line on bottom panel (zoomed)
        if ax_n is not None and has_ntgt_col and "n_target_events" in grouped.columns:
            obs_counts_z = (
                grouped[grouped["init_time"] >= cutoff_time]
                .groupby("init_time")["n_target_events"]
                .mean()
                .reset_index()
                .sort_values("init_time")
            )
            obs_counts_z = obs_counts_z[obs_counts_z["n_target_events"].notna()]
            if not obs_counts_z.empty:
                ax_n.plot(
                    obs_counts_z["init_time"],
                    obs_counts_z["n_target_events"],
                    color="red", linewidth=1.5, linestyle="--",
                    label="Observed",
                    zorder=10,
                )
                ax_n.legend(fontsize=8, loc="upper right")

        ax.set_ylabel(y_label, fontsize=13, fontweight="bold")
        ax.set_title(
            f"{title_label}{landfall_label} vs Initialization Time (Last 4 Days)",
            fontsize=15,
            fontweight="bold",
        )
        ax.legend(frameon=True, shadow=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        if metric_name in SIGNED_METRICS:
            ax.axhline(0.0, color="black", linewidth=2, alpha=0.8, zorder=1)

        if ax_n is not None:
            from matplotlib.ticker import MaxNLocator
            ax_n.set_ylabel(_npts_label, fontsize=11, fontweight="bold")
            ax_n.set_xlabel("Initialization Time", fontsize=13, fontweight="bold")
            ax_n.grid(True, alpha=0.3)
            ax_n.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.sca(ax_n)
            plt.xticks(rotation=45, ha="right")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Initialization Time", fontsize=13, fontweight="bold")
            plt.xticks(rotation=45, ha="right")

        # Show later timestamps on the left (only call once; sharex propagates).
        ax.invert_xaxis()

        output_file_zoomed = (
            output_dir / f"{prefix}{filename_label}_by_inittime_last4days{landfall_suffix}.png"
        )
        plt.tight_layout()
        plt.savefig(output_file_zoomed, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved {output_file_zoomed.name} (last 4 days)")


def plot_station_error_maps(
    station_csv: Path,
    output_dir: Path,
    case_label: str = "",
    hourly_maps: bool = False,
) -> None:
    """Create geographic station-error maps from a per-station results CSV.

    Always generates event-timing maps (onset, duration, best-model).

    When *hourly_maps* is True, also generates per-lead-hour maps:
    * One multi-panel figure (one subplot per model) showing RMSE at each station.
    * One multi-panel figure showing bias (mean error) at each station.
    * A "best model" map where each station is coloured by which model has
      the lowest RMSE.

    Per-lead-hour maps are slow to generate, so they are off by default.
    Pass ``--hourly-maps`` on the CLI to enable them.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    print("\n📍 Creating station error maps...")

    sdf = pd.read_csv(station_csv)
    if sdf.empty:
        print("  ⚠️  Station CSV is empty — skipping maps")
        return

    variables = sdf["variable"].unique()
    lead_hours = sorted(int(x) for x in sdf["lead_time_hours"].dropna().unique())
    models = order_forecasts(sdf["forecast_source"].unique())
    prefix = f"{case_label}_" if case_label else ""

    # Determine map extent from station locations (with padding)
    lat_min, lat_max = sdf["latitude"].min(), sdf["latitude"].max()
    lon_min, lon_max = sdf["longitude"].min(), sdf["longitude"].max()
    lat_pad = max((lat_max - lat_min) * 0.15, 1.0)
    lon_pad = max((lon_max - lon_min) * 0.15, 1.0)
    extent = [
        lon_min - lon_pad, lon_max + lon_pad,
        lat_min - lat_pad, lat_max + lat_pad,
    ]
    # Handle 0-360 longitude convention — convert for cartopy PlateCarree
    if extent[0] > 180:
        extent[0] -= 360
    if extent[1] > 180:
        extent[1] -= 360

    # ── Event-timing station maps (before per-lead-hour maps) ──────────────
    EVENT_METRIC_COLS = [
        ("onset_error", "Onset Error", "hours"),
        ("end_error", "End Error", "hours"),
        ("duration_error", "Duration Error", "hours"),
        ("peak_timing_error", "Peak Timing Error", "hours"),
        ("peak_value_error", "Peak Value Error", "K"),
    ]
    has_event_cols = all(c in sdf.columns for c, _, _ in EVENT_METRIC_COLS)
    has_missed_col = "fc_event_missed" in sdf.columns
    if has_event_cols:
        # Include rows with event errors OR missed-event flags
        event_mask = sdf["onset_error"].notna() | sdf["end_error"].notna()
        if has_missed_col:
            event_mask = event_mask | (sdf["fc_event_missed"] == True)  # noqa: E712
        edf = sdf[event_mask].copy()
        if not edf.empty:
            print("\n   📊 Creating event-timing station maps...")
            for var in edf["variable"].unique():
                var_sub = edf[edf["variable"] == var]
                if var_sub.empty:
                    continue
                var_display = VARIABLE_DISPLAY_NAMES.get(var, var)
                ev_models = order_forecasts(var_sub["forecast_source"].unique())
                n_ev = len(ev_models)
                if n_ev == 0:
                    continue

                # Build missed-event lookup: {model: DataFrame of missed rows}
                missed_by_model: dict[str, pd.DataFrame] = {}
                if has_missed_col:
                    missed_all = var_sub[var_sub["fc_event_missed"] == True]  # noqa: E712
                    for mdl in ev_models:
                        mm = missed_all[missed_all["forecast_source"] == mdl]
                        if not mm.empty:
                            missed_by_model[mdl] = mm

                for col, label, unit in EVENT_METRIC_COLS:
                    col_sub = var_sub[var_sub[col].notna()]
                    # Allow figure even if col_sub is empty but missed stations exist
                    if col_sub.empty and not missed_by_model:
                        continue

                    ncols = min(n_ev, 3)
                    nrows = int(np.ceil(n_ev / ncols))
                    fig, axes = plt.subplots(
                        nrows, ncols,
                        figsize=(5.5 * ncols, 4.5 * nrows),
                        subplot_kw={"projection": ccrs.PlateCarree()},
                        layout="constrained",
                    )
                    axes_flat = np.atleast_1d(axes).ravel()

                    # Shared colour scale — symmetric (all event errors can be +/-)
                    all_vals = col_sub[col].dropna().values
                    if len(all_vals) == 0 and not missed_by_model:
                        plt.close(fig)
                        continue
                    if len(all_vals) > 0:
                        abs_max = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals)))
                        if abs_max == 0:
                            abs_max = 1.0
                    else:
                        abs_max = 1.0
                    vmin, vmax = -abs_max, abs_max
                    cmap_name = "RdBu_r"

                    sc = None
                    has_missed_marker = False
                    for mi, model in enumerate(ev_models):
                        ax = axes_flat[mi]
                        ax.set_extent(extent, crs=ccrs.PlateCarree())
                        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
                        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
                        ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)

                        m_sub = col_sub[col_sub["forecast_source"] == model]
                        if not m_sub.empty:
                            lons = m_sub["longitude"].values.copy()
                            lons = np.where(lons > 180, lons - 360, lons)
                            sc = ax.scatter(
                                lons, m_sub["latitude"].values,
                                c=m_sub[col].values,
                                cmap=cmap_name,
                                vmin=vmin, vmax=vmax,
                                s=28, edgecolors="k", linewidths=0.3,
                                transform=ccrs.PlateCarree(),
                                zorder=5,
                            )

                        # Overlay red X only for stations where the forecast
                        # completely missed the event (no error value for this
                        # column).  Stations that have a valid error value are
                        # already shown as colored dots.
                        if model in missed_by_model:
                            mm = missed_by_model[model]
                            mm = mm[mm[col].isna()] if col in mm.columns else mm
                            if not mm.empty:
                                mlons = mm["longitude"].values.copy()
                                mlons = np.where(mlons > 180, mlons - 360, mlons)
                                ax.scatter(
                                    mlons, mm["latitude"].values,
                                    marker="o", s=28, facecolors="none",
                                    edgecolors="k", linewidths=0.3,
                                    transform=ccrs.PlateCarree(),
                                    zorder=6,
                                )
                                ax.scatter(
                                    mlons, mm["latitude"].values,
                                    marker="x", s=20, c="red", linewidths=1.2,
                                    transform=ccrs.PlateCarree(),
                                    zorder=7,
                                )
                                has_missed_marker = True

                        ax.set_title(model, fontsize=10, fontweight="bold")

                    for j in range(mi + 1, len(axes_flat)):
                        axes_flat[j].set_visible(False)

                    if sc is None and not has_missed_marker:
                        plt.close(fig)
                        continue

                    if sc is not None:
                        cbar = fig.colorbar(sc, ax=axes_flat[:n_ev], shrink=0.7, pad=0.02)
                        cbar.set_label(f"{label} ({unit})", fontsize=11)

                    # Legend entry for missed-event marker
                    if has_missed_marker:
                        from matplotlib.lines import Line2D
                        legend_elements = [
                            Line2D(
                                [0], [0], marker="x", color="red",
                                markeredgewidth=1.2, markersize=6,
                                linestyle="None",
                                label="Obs event, fc missed",
                            ),
                        ]
                        fig.legend(
                            handles=legend_elements, loc="lower center",
                            ncol=1, fontsize=9, framealpha=0.9,
                        )

                    fig.suptitle(
                        f"{var_display} {label} at Stations",
                        fontsize=14, fontweight="bold",
                    )

                    safe_col = col.replace("_", "")
                    fname = f"{prefix}station_{safe_col}_{var}.png"
                    fig.savefig(output_dir / fname, dpi=250, bbox_inches="tight")
                    plt.close(fig)
                    print(f"  ✓ Saved {fname}")

                # ── Best-model map per event metric (min |error|) ─────────
                for col, label, unit in EVENT_METRIC_COLS:
                    col_sub = var_sub[var_sub[col].notna()]
                    if col_sub.empty or len(ev_models) < 2:
                        continue

                    pivot = col_sub.pivot_table(
                        index=["latitude", "longitude"],
                        columns="forecast_source",
                        values=col,
                        aggfunc="mean",
                    ).reset_index()

                    model_cols = [c for c in pivot.columns if c in set(ev_models)]
                    if len(model_cols) < 2:
                        continue

                    # Best = smallest absolute error (closest to 0)
                    abs_pivot = pivot[model_cols].abs()
                    pivot["best_model"] = abs_pivot.idxmin(axis=1)

                    model_to_idx = {m: i for i, m in enumerate(ev_models)}
                    best_idx = pivot["best_model"].map(model_to_idx).values

                    fig, ax = plt.subplots(
                        figsize=(8, 6),
                        subplot_kw={"projection": ccrs.PlateCarree()},
                    )
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
                    ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)

                    from matplotlib.colors import ListedColormap
                    cmap_best = ListedColormap(
                        [_station_color(m) for m in ev_models]
                    )

                    lons = pivot["longitude"].values.copy()
                    lons = np.where(lons > 180, lons - 360, lons)
                    ax.scatter(
                        lons, pivot["latitude"].values,
                        c=best_idx,
                        cmap=cmap_best,
                        vmin=-0.5, vmax=len(ev_models) - 0.5,
                        s=32, edgecolors="k", linewidths=0.3,
                        transform=ccrs.PlateCarree(),
                        zorder=5,
                    )

                    from matplotlib.lines import Line2D
                    legend_handles = [
                        Line2D(
                            [0], [0], marker="o", color="w",
                            markerfacecolor=_station_color(m),
                            markeredgecolor="k", markersize=8,
                            label=m,
                        )
                        for m in ev_models
                    ]
                    ax.legend(
                        handles=legend_handles, loc="lower left",
                        fontsize=8, frameon=True,
                    )

                    ax.set_title(
                        f"Best Model per Station — {var_display} {label} (min |error|)",
                        fontsize=12, fontweight="bold",
                    )

                    safe_col = col.replace("_", "")
                    fname = f"{prefix}station_best_{safe_col}_{var}.png"
                    fig.savefig(output_dir / fname, dpi=250, bbox_inches="tight")
                    plt.close(fig)
                    print(f"  ✓ Saved {fname}")

    # ── Per-lead-hour RMSE / bias / best-model maps ─────────────────────────
    if not hourly_maps:
        print("  ⏭️  Skipping per-lead-hour station maps (use --hourly-maps to enable)")
        return

    for var in variables:
        for lh in lead_hours:
            sub = sdf[(sdf["variable"] == var) & (sdf["lead_time_hours"] == lh)]
            if sub.empty:
                continue

            n_models = len(models)
            if n_models == 0:
                continue

            # --- RMSE and bias multi-panel maps ---
            for metric_col, metric_label, cmap_name in [
                ("rmse", "RMSE", "YlOrRd"),
                ("bias", "Bias", "RdBu_r"),
            ]:
                ncols = min(n_models, 3)
                nrows = int(np.ceil(n_models / ncols))
                fig, axes = plt.subplots(
                    nrows, ncols,
                    figsize=(5.5 * ncols, 4.5 * nrows),
                    subplot_kw={"projection": ccrs.PlateCarree()},
                    layout="constrained",
                )
                axes_flat = np.atleast_1d(axes).ravel()

                # Shared colour scale across all panels
                all_vals = []
                for model in models:
                    m_sub = sub[sub["forecast_source"] == model]
                    if not m_sub.empty:
                        all_vals.extend(m_sub[metric_col].dropna().tolist())
                if not all_vals:
                    plt.close(fig)
                    continue

                if metric_col == "bias":
                    abs_max = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals)))
                    vmin, vmax = -abs_max, abs_max
                else:
                    vmin, vmax = 0, np.nanpercentile(all_vals, 95)

                for mi, model in enumerate(models):
                    ax = axes_flat[mi]
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
                    ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)

                    m_sub = sub[sub["forecast_source"] == model]
                    if m_sub.empty:
                        ax.set_title(model, fontsize=10, fontweight="bold")
                        continue

                    lons = m_sub["longitude"].values.copy()
                    lons = np.where(lons > 180, lons - 360, lons)
                    sc = ax.scatter(
                        lons, m_sub["latitude"].values,
                        c=m_sub[metric_col].values,
                        cmap=cmap_name,
                        vmin=vmin, vmax=vmax,
                        s=28, edgecolors="k", linewidths=0.3,
                        transform=ccrs.PlateCarree(),
                        zorder=5,
                    )
                    ax.set_title(model, fontsize=10, fontweight="bold")

                # Hide unused axes
                for j in range(mi + 1, len(axes_flat)):
                    axes_flat[j].set_visible(False)

                # Shared colorbar
                cbar = fig.colorbar(sc, ax=axes_flat[:n_models], shrink=0.7, pad=0.02)
                var_display = VARIABLE_DISPLAY_NAMES.get(var, var)
                unit = VARIABLE_UNITS.get(var, "")
                unit_str = f" ({unit})" if unit else ""
                cbar.set_label(f"{metric_label}{unit_str}", fontsize=11)

                fig.suptitle(
                    f"{var_display} {metric_label} at Stations — Lead {lh}h",
                    fontsize=14, fontweight="bold",
                )

                fname = f"{prefix}station_{metric_col}_{var}_lead{lh:03d}h.png"
                fig.savefig(output_dir / fname, dpi=250, bbox_inches="tight")
                plt.close(fig)
                print(f"  ✓ Saved {fname}")

            # --- Best model map ---
            if n_models >= 2:
                fig, ax = plt.subplots(
                    figsize=(10, 7),
                    subplot_kw={"projection": ccrs.PlateCarree()},
                    layout="constrained",
                )
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
                ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
                ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)

                # For each station find the model with lowest RMSE
                pivot = sub.pivot_table(
                    index=["station_idx", "latitude", "longitude"],
                    columns="forecast_source",
                    values="rmse",
                ).reset_index()

                model_cols = [c for c in pivot.columns if c in set(models)]
                if len(model_cols) >= 2:
                    pivot["best_model"] = pivot[model_cols].idxmin(axis=1)

                    # Assign each model a colour index
                    model_to_idx = {m: i for i, m in enumerate(models)}
                    best_idx = pivot["best_model"].map(model_to_idx).values

                    # Use vivid map-friendly colours
                    from matplotlib.colors import ListedColormap
                    cmap_best = ListedColormap(
                        [_station_color(m) for m in models]
                    )

                    lons = pivot["longitude"].values.copy()
                    lons = np.where(lons > 180, lons - 360, lons)
                    sc = ax.scatter(
                        lons, pivot["latitude"].values,
                        c=best_idx,
                        cmap=cmap_best,
                        vmin=-0.5, vmax=len(models) - 0.5,
                        s=32, edgecolors="k", linewidths=0.3,
                        transform=ccrs.PlateCarree(),
                        zorder=5,
                    )
                    # Legend
                    from matplotlib.lines import Line2D
                    legend_handles = [
                        Line2D(
                            [0], [0], marker="o", color="w",
                            markerfacecolor=_station_color(m),
                            markeredgecolor="k", markersize=8,
                            label=m,
                        )
                        for m in models
                    ]
                    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, frameon=True)

                var_display = VARIABLE_DISPLAY_NAMES.get(var, var)
                ax.set_title(
                    f"Best Model per Station — {var_display} RMSE — Lead {lh}h",
                    fontsize=13, fontweight="bold",
                )

                fname = f"{prefix}station_best_model_{var}_lead{lh:03d}h.png"
                fig.savefig(output_dir / fname, dpi=250, bbox_inches="tight")
                plt.close(fig)
                print(f"  ✓ Saved {fname}")


def create_model_comparison_table(df: pd.DataFrame, output_dir: Path, case_label: str = ""):
    """Create a table comparing all models across all metrics."""
    print("\n📊 Creating model comparison table...")
    
    # Compute mean metric values for each forecast
    comparison = (
        df.groupby(["forecast_name", "metric_name"])["metric_value"]
        .mean()
        .reset_index()
    )
    
    # Pivot to wide format
    comparison_wide = comparison.pivot(
        index="metric_name", columns="forecast_name", values="metric_value"
    )
    
    # Save as CSV with case label in filename
    prefix = f"{case_label}_" if case_label else ""
    output_file = output_dir / f"{prefix}model_comparison_table.csv"
    comparison_wide.to_csv(output_file)
    print(f"  ✓ Saved {output_file.name}")
    
    # Also print to console
    print("\n" + "=" * 80)
    print("Model Comparison (Mean Values)")
    print("=" * 80)
    print(comparison_wide.to_string())
    print("=" * 80)


def create_all_plots(
    df: pd.DataFrame,
    output_dir: Path,
    case_label: str = "",
    by_valid_only: bool = False,
    by_landfall_relative_only: bool = False,
    hourly_maps: bool = False,
):
    """Create all plots for the results."""
    print("\n📈 Creating plots...")

    # Get all metrics
    metrics = df["metric_name"].unique()
    all_models = order_forecasts(df["forecast_name"].dropna().unique())

    # Extract landfall times for TC cases (used for init/valid annotations).
    landfall_times = extract_landfall_times(df)
    if landfall_times:
        print(f"\n🌀 Detected {len(landfall_times)} landfall(s):")
        for lf_id, lf_time in landfall_times:
            print(f"    Landfall {lf_id}: {lf_time}")

    # Extract event start/end for heat_wave / freeze cases.
    event_times = extract_event_times(df)
    if event_times:
        print(f"\n🌡️  Event reference times:")
        for label, ts in event_times:
            print(f"    {label}: {ts}")

    if by_valid_only:
        print("\n⚡ --by-valid-only enabled: generating only valid-time plots.")
        print("\n📊 Plotting metrics vs valid time...")
        for metric in metrics:
            plot_metric_by_validtime(
                df,
                metric,
                output_dir,
                case_label,
                landfall_times=landfall_times,
                event_times=event_times,
            )
        return

    if by_landfall_relative_only:
        print("\n⚡ --by-landfall-relative-only enabled: generating only landfall-relative panels.")
        print("   Panels: MSLP error, total track, along track, cross track")
        plot_landfall_relative_panels(
            df,
            output_dir,
            case_label,
            landfall_times=landfall_times,
            all_models=all_models,
        )
        return

    init_cutoff = compute_consensus_init_start(df, min_fraction=0.6)
    if init_cutoff is not None:
        # Build diagnostics so the chosen cutoff is transparent.
        track_metrics = {
            "along_track_error",
            "cross_track_error",
            "total_track_error",
            "AlongTrackError",
            "CrossTrackError",
            "TotalTrackError",
        }
        track_df = df[df["metric_name"].isin(track_metrics)].copy()
        track_df = track_df[
            track_df["init_time"].notna()
            & (track_df["init_time"] != "")
            & track_df["metric_value"].notna()
        ].copy()
        track_df["init_time"] = pd.to_datetime(track_df["init_time"], errors="coerce")
        track_df = track_df[track_df["init_time"].notna()]

        n_models = len(df["forecast_name"].dropna().unique())
        required = int(np.ceil(n_models * 0.6))
        active_models_by_init = (
            track_df.groupby("init_time")["forecast_name"].nunique().sort_index()
        )
        models_at_cutoff = int(active_models_by_init.get(init_cutoff, 0))

        print(
            f"\n🕒 Using init-time cutoff: {init_cutoff} "
            "(>=60% of models with valid TC track metrics)"
        )
        print(
            f"    Models with valid track metrics at cutoff: "
            f"{models_at_cutoff}/{n_models} (required >= {required})"
        )
        # Show nearby init-time coverage for context.
        nearby = active_models_by_init.loc[
            (active_models_by_init.index >= init_cutoff - pd.Timedelta(days=1))
            & (active_models_by_init.index <= init_cutoff + pd.Timedelta(days=1))
        ]
        if len(nearby) > 0:
            print("    Nearby init-time model coverage:")
            for it, n_active in nearby.items():
                marker = " <- cutoff" if pd.Timestamp(it) == pd.Timestamp(init_cutoff) else ""
                print(f"      {pd.Timestamp(it)}: {int(n_active)}/{n_models}{marker}")
    else:
        print(
            "\n🕒 No consensus init-time cutoff found; using all available init times."
        )
    
    # Print model styling info
    forecast_names = order_forecasts(df["forecast_name"].unique())
    print(f"\n🎨 Model styling:")
    print("  Physical Ensembles (blue tones, solid):")
    for name in forecast_names:
        if "Ens-Mean" in name and ("IFS" in name or "GFS" in name or "AIFS" in name):
            style = get_model_style(name)
            print(f"    {name}: {style['marker']} marker")
    print("  WeatherMesh (black/gray, solid):")
    for name in forecast_names:
        if "WeatherMesh" in name:
            style = get_model_style(name)
            print(f"    {name}: {style['marker']} marker")
    print("  AI Models (varied colors, dashed):")
    for name in forecast_names:
        if name not in MODEL_GROUPS:
            continue
        if "WeatherMesh" not in name and not ("Ens-Mean" in name and ("IFS" in name or "GFS" in name or "AIFS" in name)):
            style = get_model_style(name)
            print(f"    {name}: {style['marker']} marker")
    
    # Create plots by lead time
    print("\n📊 Plotting metrics vs lead time...")
    for metric in metrics:
        if metric != "duration_error":  # Duration error is better by init time
            plot_metric_by_leadtime(
                df,
                metric,
                output_dir,
                case_label,
            )
    
    # Create plots by init time (with landfall markers)
    print("\n📊 Plotting metrics vs init time...")
    for metric in metrics:
        plot_metric_by_inittime(
            df,
            metric,
            output_dir,
            case_label,
            landfall_times=landfall_times,
            min_init_time=init_cutoff,
            event_times=event_times,
        )

    # Create plots by valid time
    print("\n📊 Plotting metrics vs valid time...")
    for metric in metrics:
        plot_metric_by_validtime(
            df,
            metric,
            output_dir,
            case_label,
            landfall_times=landfall_times,
            event_times=event_times,
        )

    # Create landfall-relative multi-panel plots.
    print("\n📊 Plotting landfall-relative metric panels...")
    plot_landfall_relative_panels(
        df,
        output_dir,
        case_label,
        landfall_times=landfall_times,
        all_models=all_models,
    )
    
    # Create comparison table
    create_model_comparison_table(df, output_dir, case_label)

    # Station error maps — only for station-based (GHCN) verification.
    is_station_verification = (
        "target_source" in df.columns
        and df["target_source"].str.contains("GHCN", case=False, na=False).any()
    )
    if is_station_verification:
        results_dir = output_dir.parent
        label_parts = case_label.split("_")
        case_id = label_parts[0].replace("case", "") if label_parts[0].startswith("case") else None
        glob_pat = f"case_{case_id}_*_station_results.csv" if case_id else "*_station_results.csv"
        station_csvs = sorted(results_dir.glob(glob_pat))
        for station_csv in station_csvs:
            plot_station_error_maps(station_csv, output_dir, case_label, hourly_maps=hourly_maps)


def main(
    csv_path: str,
    by_valid_only: bool = False,
    by_landfall_relative_only: bool = False,
    hourly_maps: bool = False,
):
    """Main analysis function."""
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} does not exist")
        sys.exit(1)
    
    # Extract case info from CSV filename (e.g., case_29_heat_wave_results.csv)
    # Expected format: case_{id}_{event_type}_results.csv
    stem = csv_path.stem  # e.g., "case_29_heat_wave_results"
    parts = stem.split("_")
    
    if len(parts) >= 3 and parts[0] == "case":
        case_id = parts[1]
        # Event type is everything between case_id and _results
        event_type = "_".join(parts[2:]).replace("_results", "")
        case_label = f"case{case_id}_{event_type}"
    else:
        # Fallback if filename doesn't match expected pattern
        case_label = stem.replace("_results", "")
    
    # Create descriptive output directory
    output_dir = csv_path.parent / f"plots_{case_label}"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("ExtremeWeatherBench - Results Analysis")
    print("=" * 80)
    print(f"Input: {csv_path}")
    print(f"Output: {output_dir}")
    print(f"Case: {case_label}")
    
    # Load results
    df = load_results(csv_path)
    
    # Create plots
    create_all_plots(
        df,
        output_dir,
        case_label,
        by_valid_only=by_valid_only,
        by_landfall_relative_only=by_landfall_relative_only,
        hourly_maps=hourly_maps,
    )
    
    print("\n✅ Analysis complete!")
    print(f"   Plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and plot ExtremeWeatherBench results"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to results CSV file",
    )
    parser.add_argument(
        "--by-valid-only",
        action="store_true",
        help="Generate only valid-time plots (skip lead/init-time plots and comparison table).",
    )
    parser.add_argument(
        "--by-landfall-relative-only",
        action="store_true",
        help="Generate only landfall-relative 4-panel plots (MSLP, total/along/cross track).",
    )
    parser.add_argument(
        "--hourly-maps",
        action="store_true",
        help=(
            "Generate per-lead-hour station RMSE/bias/best-model maps. "
            "Off by default because these are slow to generate."
        ),
    )

    args = parser.parse_args()
    
    try:
        main(
            args.csv_path,
            by_valid_only=args.by_valid_only,
            by_landfall_relative_only=args.by_landfall_relative_only,
            hourly_maps=args.hourly_maps,
        )
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


