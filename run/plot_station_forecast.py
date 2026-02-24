#!/usr/bin/env python3
"""
Plot a single model init's temperature forecast at a station vs observations
and climatology, to visually debug event detection.

Supports heat_wave and freeze event types, auto-detected from the case metadata.

Heat wave (Perkins & Alexander 2013, condition 1):
  Daily Tmax >= P90(Tmax climatology) for 3+ consecutive days
  P90 = mean + 1.282 * std

Freeze:
  Daily Tmin <= P5(Tmin climatology) for 3+ consecutive days
  P5  = mean - 1.645 * std

Usage:
    python plot_station_forecast.py 340 --station 12 --model IFS-Ens-Mean --init 2025061500
    python plot_station_forecast.py 341 --station 5 --model WeatherMesh-4 --init 2026011500
    python plot_station_forecast.py 340 --station 12 --model IFS-Ens-Mean WeatherMesh-4 --init 2025061500
"""

import argparse
import sys
sys.path.insert(0, "/huge/users/larissa/ExtremeWeatherBench/src")

import numpy as np
import pandas as pd
import xarray as xr
import sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

import extremeweatherbench as ewb
from extremeweatherbench.metrics import (
    _load_t2m_climatology,
    _find_events_vectorized,
    _to_daily_agg,
)
from extremeweatherbench.utils import build_station_bilinear_map, apply_bilinear_weights

# ── Constants ─────────────────────────────────────────────────────────────
LOCAL_ZARR_ROOT = Path("/huge/proc/met-data/zarr")
VARIABLE_MAPPING = {
    "msl": "air_pressure_at_mean_sea_level",
    "t2": "surface_air_temperature",
    "t2m": "surface_air_temperature",
}
EVENT_MIN_DAYS = 3
GHCN_DIR = Path("/huge/users/larissa/ExtremeWeatherBench/data_prep")
STANDARD_HOURS = [0, 6, 12, 18]

# Event-type-specific configuration
EVENT_CONFIG = {
    "heat_wave": {
        "z_score": 1.282,         # P90
        "daily_agg": "max",       # daily Tmax
        "compare": "ge",          # >=
        "threshold_label": "P90",
        "daily_label": "Tmax",
        "event_label": "heat wave",
    },
    "freeze": {
        "z_score": -1.645,        # P5
        "daily_agg": "min",       # daily Tmin
        "compare": "le",          # <=
        "threshold_label": "P5",
        "daily_label": "Tmin",
        "event_label": "freeze",
    },
}


def load_station_locations(case_id: int):
    """Load station locations from the case's station results CSV."""
    pattern = f"case_{case_id}_*_station_results.csv"
    matches = sorted(Path(".").glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No station CSV found matching {pattern} in {Path('.').resolve()}"
        )
    csv_path = matches[0]
    print(f"  Using station CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    locs = df.groupby("station_idx").first()[["latitude", "longitude"]].reset_index()
    return locs


def load_local_zarr_single_init(model_name: str, init_str: str) -> xr.Dataset:
    """Load a single local zarr init for the given model."""
    zarr_path = (
        LOCAL_ZARR_ROOT / init_str / model_name
        / f"{model_name}_{init_str}.zarr"
    )
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr not found: {zarr_path}")

    ds = xr.open_zarr(str(zarr_path), chunks=None)
    needed = {"t2", "t2m"}
    keep = [v for v in needed if v in ds.data_vars]
    if not keep:
        raise ValueError(f"No temperature variable in {zarr_path}. Vars: {list(ds.data_vars)}")
    ds = ds[keep]

    step_hours = ds["step"].values.astype(float)
    lead_time = pd.to_timedelta(step_hours, unit="h")
    ds = ds.assign_coords(lead_time=("step", lead_time))
    ds = ds.swap_dims({"step": "lead_time"})

    if "valid_time" in ds.coords:
        vt_vals = ds["valid_time"].values
        ds = ds.drop_vars("valid_time")
        ds = ds.assign_coords(valid_time=("lead_time", vt_vals))

    ds = ds.drop_vars("step", errors="ignore")

    renames = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        renames["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        renames["lon"] = "longitude"
    for src, dst in VARIABLE_MAPPING.items():
        if src in ds.data_vars and dst not in ds.data_vars:
            renames[src] = dst
    if renames:
        ds = ds.rename(renames)

    if "init_time" not in ds.coords:
        init_dt = pd.Timestamp(datetime.strptime(init_str, "%Y%m%d%H"))
        ds = ds.assign_coords(init_time=init_dt)

    return ds


def _resolve_ghcn_sources(case, explicit_source: str | None) -> list[str]:
    """Return the parquet file path(s) needed for this case's date range."""
    if explicit_source is not None:
        return [explicit_source]
    years = sorted({case.start_date.year, case.end_date.year})
    paths = []
    for y in years:
        p = GHCN_DIR / f"ghcnh_all_{y}.parq"
        if not p.exists():
            raise FileNotFoundError(
                f"GHCN parquet for {y} not found: {p}\n"
                f"Available: {sorted(GHCN_DIR.glob('ghcnh_all_*.parq'))}"
            )
        paths.append(str(p))
    return paths


def load_ghcn_target_ds(case, ghcn_sources: list[str]) -> xr.Dataset:
    """Load GHCN target as xr.Dataset, concatenating multiple years if needed."""
    variables = ["surface_air_temperature"]
    datasets = []
    for src in ghcn_sources:
        target = ewb.inputs.GHCN(variables=variables, source=src)
        raw = target.open_and_maybe_preprocess_data_from_source()
        raw = target.subset_data_to_case(raw, case)
        ds = target.maybe_convert_to_dataset(raw)
        if isinstance(ds, xr.Dataset) and ds.sizes.get("valid_time", 0) > 0:
            datasets.append(ds)

    if not datasets:
        raise ValueError(
            f"Empty GHCN target after pipeline. Sources: {ghcn_sources}"
        )
    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(datasets, dim="valid_time")


def extract_obs_at_station(target_ds, station_lat, station_lon):
    """Extract observations at a single station from the sparse GHCN Dataset."""
    var = "surface_air_temperature"
    da = target_ds[var]
    valid_times = target_ds["valid_time"].values
    lat_vals = np.asarray(da["latitude"].values, dtype=np.float64)
    lon_vals = np.asarray(da["longitude"].values, dtype=np.float64)
    station_lon_360 = station_lon + 360 if station_lon < 0 else station_lon

    if isinstance(da.data, sp.COO):
        da_dense = da.data.todense()
    else:
        da_dense = np.asarray(da.values)

    has_data = np.any(np.isfinite(da_dense) & (da_dense != 0), axis=0)
    occ_li, occ_lj = np.where(has_data)
    if len(occ_li) == 0:
        has_data = np.any(da_dense != 0, axis=0)
        occ_li, occ_lj = np.where(has_data)
    if len(occ_li) == 0:
        raise ValueError("No occupied cells found in GHCN target")

    occ_lats = lat_vals[occ_li]
    occ_lons = lon_vals[occ_lj]
    dist = (occ_lats - station_lat) ** 2 + (occ_lons - station_lon_360) ** 2
    nearest = np.argmin(dist)
    li, lj = occ_li[nearest], occ_lj[nearest]
    print(f"  Matched GHCN cell: ({occ_lats[nearest]:.4f}, "
          f"{occ_lons[nearest] - 360 if occ_lons[nearest] > 180 else occ_lons[nearest]:.4f}), "
          f"dist={np.sqrt(dist[nearest]):.4f}°")

    obs_vals = da_dense[:, li, lj].astype(np.float32)
    return valid_times, obs_vals


# ── Climatology helpers ──────────────────────────────────────────────────

_CLIMO_CACHE = {}

def _get_climo_ds():
    """Load climatology into memory (once)."""
    if "ds" not in _CLIMO_CACHE:
        print("  (loading climatology into memory...)")
        _CLIMO_CACHE["ds"] = _load_t2m_climatology().load()
    return _CLIMO_CACHE["ds"]


def _get_climo_bmap(station_lat, station_lon):
    """Build/cache bilinear map from climo grid to a single station."""
    key = (station_lat, station_lon)
    if key not in _CLIMO_CACHE:
        ds = _get_climo_ds()
        climo_lat = np.asarray(ds["t2m_mean"]["latitude"].values, dtype=np.float32)
        climo_lon = np.asarray(ds["t2m_mean"]["longitude"].values, dtype=np.float32)
        lon360 = np.float32(station_lon + 360 if station_lon < 0 else station_lon)
        _CLIMO_CACHE[key] = build_station_bilinear_map(
            climo_lon, climo_lat,
            np.array([lon360]), np.array([np.float32(station_lat)]),
        )
    return _CLIMO_CACHE[key]


def get_climatology_at_station(valid_times, station_lat, station_lon):
    """Get climatological mean at station for given valid times (Kelvin)."""
    ds = _get_climo_ds()
    climo_mean = ds["t2m_mean"]
    bmap = _get_climo_bmap(station_lat, station_lon)

    climo_vals = np.full(len(valid_times), np.nan, dtype=np.float32)
    cache = {}
    for i, vt in enumerate(valid_times):
        ts = pd.Timestamp(vt)
        doy = ts.month * 100 + ts.day
        hr = ts.hour
        key = (doy, hr)
        if key not in cache:
            m2d = climo_mean.sel(DOY=doy, hour=hr, method="nearest").values
            cache[key] = apply_bilinear_weights(m2d[np.newaxis], bmap)[0, 0]
        climo_vals[i] = cache[key]
    return climo_vals


def get_threshold_at_station(valid_times, station_lat, station_lon, z_score: float):
    """Get threshold (mean + z_score*std) at station for given valid times (Kelvin)."""
    ds = _get_climo_ds()
    climo_mean = ds["t2m_mean"]
    climo_std = ds["t2m_std"]
    bmap = _get_climo_bmap(station_lat, station_lon)

    thr_vals = np.full(len(valid_times), np.nan, dtype=np.float32)
    cache = {}
    for i, vt in enumerate(valid_times):
        ts = pd.Timestamp(vt)
        doy = ts.month * 100 + ts.day
        hr = ts.hour
        key = (doy, hr)
        if key not in cache:
            m2d = climo_mean.sel(DOY=doy, hour=hr, method="nearest").values
            s2d = climo_std.sel(DOY=doy, hour=hr, method="nearest").values
            mean_val = apply_bilinear_weights(m2d[np.newaxis], bmap)[0, 0]
            std_val = apply_bilinear_weights(s2d[np.newaxis], bmap)[0, 0]
            cache[key] = mean_val + z_score * std_val
        thr_vals[i] = cache[key]
    return thr_vals


def get_daily_threshold(unique_days, station_lat, station_lon,
                        z_score: float, daily_agg: str):
    """Compute the daily threshold using all 4 standard hours per day.

    For heat waves (daily_agg="max"): daily max of hourly P90 thresholds.
    For freezes  (daily_agg="min"): daily min of hourly P5 thresholds.
    """
    ds = _get_climo_ds()
    climo_mean = ds["t2m_mean"]
    climo_std = ds["t2m_std"]
    bmap = _get_climo_bmap(station_lat, station_lon)

    agg_fn = max if daily_agg == "max" else min
    n_days = len(unique_days)
    daily_thr = np.full(n_days, np.nan, dtype=np.float32)
    cache = {}

    for d in range(n_days):
        day = pd.Timestamp(unique_days[d])
        doy = day.month * 100 + day.day
        hour_thrs = []
        for hr in STANDARD_HOURS:
            key = (doy, hr)
            if key not in cache:
                m2d = climo_mean.sel(DOY=doy, hour=hr, method="nearest").values
                s2d = climo_std.sel(DOY=doy, hour=hr, method="nearest").values
                mean_val = apply_bilinear_weights(m2d[np.newaxis], bmap)[0, 0]
                std_val = apply_bilinear_weights(s2d[np.newaxis], bmap)[0, 0]
                cache[key] = mean_val + z_score * std_val
            hour_thrs.append(cache[key])
        daily_thr[d] = agg_fn(hour_thrs)

    return daily_thr


def interp_forecast_to_station(ds, station_lat, station_lon):
    """Bilinear interpolation of forecast grid to a single station point."""
    fc_lat = np.asarray(ds["latitude"].values, dtype=np.float32)
    fc_lon = np.asarray(ds["longitude"].values, dtype=np.float32)

    lon360 = np.float32(station_lon + 360 if station_lon < 0 else station_lon)
    lat_f = np.float32(station_lat)

    bmap = build_station_bilinear_map(
        fc_lon, fc_lat, np.array([lon360]), np.array([lat_f]),
    )

    var = "surface_air_temperature"
    da = ds[var]
    if "level" in da.dims:
        da = da.isel(level=0)
    vals = np.asarray(da.values, dtype=np.float32)
    fc_at_station = apply_bilinear_weights(vals, bmap)[:, 0]

    if "valid_time" in ds.coords:
        vt = ds["valid_time"].values
    else:
        init_time = pd.Timestamp(ds["init_time"].values)
        lead_times = ds["lead_time"].values
        vt = np.array([np.datetime64(init_time + pd.Timedelta(lt)) for lt in lead_times])

    return vt, fc_at_station


def _exceed(values, threshold, compare: str):
    """Apply exceedance comparison."""
    if compare == "ge":
        return values >= threshold
    return values <= threshold


def to_celsius(kelvin):
    return kelvin - 273.15


def main():
    parser = argparse.ArgumentParser(
        description="Plot station forecast vs obs vs climo for event debugging",
    )
    parser.add_argument("case_id", type=int, help="Case ID (e.g., 340 for heat wave, 341 for freeze)")
    parser.add_argument("--station", required=True, help="Station index (int) or lat,lon")
    parser.add_argument("--model", required=True, nargs="+",
                        help="Model name(s), space-separated or comma-separated")
    parser.add_argument("--init", required=True, help="Init time as YYYYMMDDHH")
    parser.add_argument("--ghcn-source", default=None,
                        help="Path to GHCN parquet file (auto-detected from case year if omitted)")
    parser.add_argument("--output", default=None, help="Output filename")
    parser.add_argument("--kelvin", action="store_true", help="Plot in Kelvin (default: Celsius)")
    args = parser.parse_args()

    model_names = []
    for m in args.model:
        model_names.extend(m.split(","))
    model_names = [m.strip() for m in model_names if m.strip()]

    MODEL_COLORS = [
        "tab:red", "tab:blue", "tab:green", "tab:purple",
        "tab:brown", "tab:pink", "tab:cyan", "tab:olive",
    ]

    # ── Load case metadata ───────────────────────────────────────────────
    all_cases = ewb.cases.load_ewb_events_yaml_into_case_list()
    matching = [c for c in all_cases if c.case_id_number == args.case_id]
    if not matching:
        print(f"Case {args.case_id} not found"); sys.exit(1)
    case = matching[0]
    event_type = case.event_type
    print(f"Case: {case.title} ({case.start_date} to {case.end_date})")
    print(f"Event type: {event_type}")

    if event_type not in EVENT_CONFIG:
        print(f"Unsupported event type '{event_type}'. Supported: {list(EVENT_CONFIG.keys())}")
        sys.exit(1)

    ecfg = EVENT_CONFIG[event_type]
    z_score = ecfg["z_score"]
    daily_agg = ecfg["daily_agg"]
    compare = ecfg["compare"]
    thr_label = ecfg["threshold_label"]
    daily_label = ecfg["daily_label"]
    event_label = ecfg["event_label"]
    z_str = f"{'+' if z_score > 0 else ''}{z_score:.3f}"

    # ── Resolve station ──────────────────────────────────────────────────
    locs = load_station_locations(args.case_id)
    if "," in args.station:
        lat, lon = map(float, args.station.split(","))
    else:
        si = int(args.station)
        row = locs[locs["station_idx"] == si]
        if row.empty:
            print(f"Station {si} not found. Available: {sorted(locs['station_idx'].tolist())}")
            sys.exit(1)
        lat = row["latitude"].values[0]
        lon_360 = row["longitude"].values[0]
        lon = lon_360 - 360 if lon_360 > 180 else lon_360
    print(f"Station: idx={args.station}, lat={lat:.4f}, lon={lon:.4f}")

    # ── Load all forecasts ───────────────────────────────────────────────
    forecasts = {}
    all_fc_valid_times = None
    for mi, model_name in enumerate(model_names):
        print(f"\nLoading {model_name} init {args.init}...")
        try:
            ds = load_local_zarr_single_init(model_name, args.init)
        except FileNotFoundError as e:
            print(f"  Skipping {model_name}: {e}")
            continue
        print(f"  Dims: {dict(ds.sizes)}")
        fc_vt, fc_v = interp_forecast_to_station(ds, lat, lon)
        print(f"  {len(fc_v)} lead times: "
              f"{pd.Timestamp(fc_vt[0])} -> {pd.Timestamp(fc_vt[-1])}")

        fc_daily_arr, _ = _to_daily_agg(
            fc_v[:, np.newaxis],
            np.zeros_like(fc_v[:, np.newaxis]),
            valid_times=fc_vt,
            agg=daily_agg,
        )
        fc_d = fc_daily_arr[:, 0]
        fc_udays = np.unique(np.array([pd.Timestamp(vt).normalize() for vt in fc_vt]))
        forecasts[model_name] = dict(
            valid_times=fc_vt, vals=fc_v,
            daily=fc_d, unique_days=fc_udays,
        )
        if all_fc_valid_times is None:
            all_fc_valid_times = fc_vt.copy()
        else:
            all_fc_valid_times = np.union1d(all_fc_valid_times, fc_vt)

    if not forecasts:
        print("No models loaded successfully."); sys.exit(1)

    ref_model = list(forecasts.keys())[0]
    ref_fc_vt = forecasts[ref_model]["valid_times"]
    ref_fc_udays = forecasts[ref_model]["unique_days"]

    # ── Load GHCN observations ───────────────────────────────────────────
    ghcn_sources = _resolve_ghcn_sources(case, args.ghcn_source)
    print(f"\nLoading GHCN observations from {ghcn_sources}...")
    target_ds = load_ghcn_target_ds(case, ghcn_sources)
    print(f"  Target DS: {dict(target_ds.sizes)}")
    obs_valid_times, obs_vals = extract_obs_at_station(target_ds, lat, lon)
    n_finite = np.isfinite(obs_vals).sum()
    print(f"  {n_finite}/{len(obs_vals)} finite obs, "
          f"{pd.Timestamp(obs_valid_times[0])} -> {pd.Timestamp(obs_valid_times[-1])}")

    # Obs daily agg — align obs to the reference forecast's valid times
    obs_aligned_mask = np.isin(obs_valid_times, ref_fc_vt)
    obs_daily = obs_daily_days = None
    if obs_aligned_mask.any():
        obs_aligned_vals = obs_vals[obs_aligned_mask]
        obs_aligned_times = obs_valid_times[obs_aligned_mask]
        n_aligned = int(obs_aligned_mask.sum())
        print(f"  Obs aligned to forecast times: {n_aligned}/{len(obs_vals)} "
              f"({n_aligned * 100 / len(obs_vals):.0f}%)")
        obs_daily_arr, _ = _to_daily_agg(
            obs_aligned_vals[:, np.newaxis],
            np.zeros_like(obs_aligned_vals[:, np.newaxis]),
            valid_times=obs_aligned_times,
            agg=daily_agg,
        )
        obs_daily = obs_daily_arr[:, 0]
        obs_daily_days = np.unique(np.array([
            pd.Timestamp(vt).normalize() for vt in obs_aligned_times
        ]))

    # ── Compute daily threshold (all 4 standard hours) ───────────────────
    all_unique_days = ref_fc_udays.copy()
    for finfo in forecasts.values():
        all_unique_days = np.union1d(all_unique_days, finfo["unique_days"])
    print(f"\nComputing {thr_label} daily {daily_label} threshold "
          f"(mean {z_str}*std, all 4 hours)...")
    thr_daily_all = get_daily_threshold(all_unique_days, lat, lon, z_score, daily_agg)
    thr_daily_lookup = dict(zip(all_unique_days, thr_daily_all))

    for model_name, finfo in forecasts.items():
        thr_for_model = np.array([thr_daily_lookup[d] for d in finfo["unique_days"]])
        finfo["thr_daily"] = thr_for_model
        finfo["exceed"] = _exceed(finfo["daily"], thr_for_model, compare)

    # Sub-daily threshold and climo for the top panel
    all_times = np.sort(np.union1d(all_fc_valid_times, obs_valid_times))
    print(f"Computing sub-daily climatology for {len(all_times)} times...")
    climo_vals_all = get_climatology_at_station(all_times, lat, lon)
    thr_vals_all = get_threshold_at_station(all_times, lat, lon, z_score)

    # ── Event detection (text output per model) ──────────────────────────
    cmp_symbol = ">=" if compare == "ge" else "<="
    print(f"\n{'='*70}")
    print(f"DAILY {daily_label.upper()} EVENT DETECTION")
    print(f"  Threshold: {thr_label} of {daily_label} climatology (mean {z_str}*std)")
    print(f"  Exceedance: {daily_label} {cmp_symbol} {thr_label}")
    print(f"  Min consecutive days: {EVENT_MIN_DAYS}")
    print(f"{'='*70}")

    for model_name, finfo in forecasts.items():
        fc_d = finfo["daily"]
        fc_ex = finfo["exceed"]
        fc_udays = finfo["unique_days"]
        thr_d = finfo["thr_daily"]

        print(f"\nFORECAST ({model_name}, init {args.init}):")
        print(f"  {'Day':>3s}  {'Date':>12s}  {daily_label+'_fc':>10s}  "
              f"{thr_label+'_thr':>10s}  {'Diff':>8s}  {'Exceed?':>8s}")
        for d in range(len(fc_d)):
            diff = fc_d[d] - thr_d[d]
            marker = "YES" if fc_ex[d] else ""
            print(f"  {d:3d}  {str(fc_udays[d])[:10]:>12s}  "
                  f"{to_celsius(fc_d[d]):8.1f}C  "
                  f"{to_celsius(thr_d[d]):8.1f}C  "
                  f"{diff:+6.1f}K  {marker:>8s}")

        n_ex = int(fc_ex.sum())
        if n_ex >= EVENT_MIN_DAYS:
            has_ev, ev_start, ev_end = _find_events_vectorized(
                fc_ex[:, np.newaxis].astype(np.int32), EVENT_MIN_DAYS, time_axis=0,
            )
            if has_ev[0]:
                print(f"\n  FC {event_label.upper()}: days {ev_start[0]}-{ev_end[0]-1} "
                      f"({fc_udays[ev_start[0]]} to {fc_udays[ev_end[0]-1]})")
            else:
                print(f"\n  No {EVENT_MIN_DAYS}-consecutive run despite {n_ex} exceed days")
        else:
            print(f"\n  Only {n_ex} exceed days (need {EVENT_MIN_DAYS})")

    if obs_daily is not None:
        thr_obs_daily = get_daily_threshold(obs_daily_days, lat, lon, z_score, daily_agg)
        obs_exceed = _exceed(obs_daily, thr_obs_daily, compare)
        n_obs_exceed = int(obs_exceed.sum())

        print(f"\nOBSERVATIONS:")
        print(f"  {'Day':>3s}  {'Date':>12s}  {daily_label+'_obs':>10s}  "
              f"{thr_label+'_thr':>10s}  {'Diff':>8s}  {'Exceed?':>8s}")
        for d in range(len(obs_daily)):
            diff = obs_daily[d] - thr_obs_daily[d]
            marker = "YES" if obs_exceed[d] else ""
            print(f"  {d:3d}  {str(obs_daily_days[d])[:10]:>12s}  "
                  f"{to_celsius(obs_daily[d]):8.1f}C  "
                  f"{to_celsius(thr_obs_daily[d]):8.1f}C  "
                  f"{diff:+6.1f}K  {marker:>8s}")

        if n_obs_exceed >= EVENT_MIN_DAYS:
            has_ev, ev_start, ev_end = _find_events_vectorized(
                obs_exceed[:, np.newaxis].astype(np.int32), EVENT_MIN_DAYS, time_axis=0,
            )
            if has_ev[0]:
                print(f"\n  OBS {event_label.upper()}: days {ev_start[0]}-{ev_end[0]-1} "
                      f"({obs_daily_days[ev_start[0]]} to {obs_daily_days[ev_end[0]-1]})")
            else:
                print(f"\n  No {EVENT_MIN_DAYS}-consecutive run despite {n_obs_exceed} exceed days")
        else:
            print(f"\n  Only {n_obs_exceed} exceed days (need {EVENT_MIN_DAYS})")

    # ── PLOT ─────────────────────────────────────────────────────────────
    convert = (lambda x: x) if args.kelvin else to_celsius
    unit = "K" if args.kelvin else "C"

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})

    # ── Top panel: sub-daily time series ─────────────────────────────────
    ax = axes[0]

    for mi, (model_name, finfo) in enumerate(forecasts.items()):
        color = MODEL_COLORS[mi % len(MODEL_COLORS)]
        fc_times_plot = pd.to_datetime(finfo["valid_times"])
        ax.plot(fc_times_plot, convert(finfo["vals"]), color=color, linewidth=2,
                label=f"{model_name}", zorder=3 + mi)

    obs_times_plot = pd.to_datetime(obs_valid_times)
    obs_in_fc_range = (obs_valid_times >= all_fc_valid_times[0]) & (obs_valid_times <= all_fc_valid_times[-1])
    if obs_in_fc_range.any():
        ax.plot(obs_times_plot[obs_in_fc_range], convert(obs_vals[obs_in_fc_range]),
                color="black", linewidth=2.5, label="GHCN observations", zorder=2)

    climo_in_range = (all_times >= all_fc_valid_times[0]) & (all_times <= all_fc_valid_times[-1])
    all_times_plot = pd.to_datetime(all_times)
    ax.plot(all_times_plot[climo_in_range], convert(climo_vals_all[climo_in_range]),
            color="gray", linewidth=1.5, linestyle="--", label="Climatology mean", zorder=1)
    ax.plot(all_times_plot[climo_in_range], convert(thr_vals_all[climo_in_range]),
            color="tab:orange", linewidth=1.5, linestyle=":",
            label=f"{thr_label} threshold (mean {z_str}\u03c3)", zorder=1)

    ax.set_ylabel(f"Temperature ({unit})")
    ax.set_title(
        f"Station {args.station} ({lat:.2f}N, {lon:.2f}W) - init {args.init}\n"
        f"Sub-daily temperatures",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %HZ"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.tick_params(axis="x", rotation=45)

    # ── Bottom panel: daily agg ──────────────────────────────────────────
    ax = axes[1]

    for mi, (model_name, finfo) in enumerate(forecasts.items()):
        color = MODEL_COLORS[mi % len(MODEL_COLORS)]
        fc_day_dates = pd.to_datetime(finfo["unique_days"])
        ax.plot(fc_day_dates, convert(finfo["daily"]), color=color, linewidth=2,
                marker="s", markersize=5, label=f"{model_name}", zorder=3 + mi)

    if obs_daily is not None:
        obs_day_dates = pd.to_datetime(obs_daily_days)
        ax.plot(obs_day_dates, convert(obs_daily), color="black", linewidth=2.5,
                marker="o", markersize=6, label=f"Obs daily {daily_label}", zorder=2)

    all_day_dates = pd.to_datetime(all_unique_days)
    ax.plot(all_day_dates, convert(thr_daily_all), color="tab:orange", linewidth=2,
            linestyle=":", marker="^", markersize=5,
            label=f"{thr_label} daily {daily_label} threshold", zorder=1)

    ax.set_ylabel(f"Daily {daily_label} ({unit})")
    ax.set_xlabel("Date")
    ax.set_title(
        f"Daily {daily_label.lower()} temperatures - "
        f"{event_label} = {EVENT_MIN_DAYS}+ days with {daily_label} {cmp_symbol} {thr_label}",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if args.output:
        outfile = args.output
    else:
        safe_models = "_".join(m.replace(" ", "_") for m in forecasts.keys())
        outfile = f"station_debug_case{args.case_id}_stn{args.station}_{safe_models}_{args.init}.png"

    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
