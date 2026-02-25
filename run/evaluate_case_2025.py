#!/usr/bin/env python3
"""
Evaluate forecasts for 2025 cases from ExtremeWeatherBench events.yaml

This script handles 2025 cases using two data sources:
1. Local per-init zarr archive at /huge/proc/met-data/zarr/
2. NOAA OAR MLWP NetCDF archive at s3://noaa-oar-mlwp-data/

Models:
  Local:  WeatherMesh-4,WeatherMesh-4p5-Ens-Mean, IFS-Ens-Mean, AIFS-Ens-Mean, GFS-Ens-Mean
  NOAA:   FourCastNet-v2 (IFS), GraphCast (IFS), Pangu-Weather (IFS), Aurora (IFS)
  GCS:    WeatherNext2

Usage:
    python evaluate_case_2025.py --case-id <case_number> [--force]

Example:
    python evaluate_case_2025.py --case-id 338          # Hurricane Melissa
    python evaluate_case_2025.py --case-id 338 --force  # Regenerate results
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timedelta
from pathlib import Path
import re

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import extremeweatherbench as ewb
import h5py
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

warnings.filterwarnings("ignore")

# Setup logging
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Paths and constants
# ─────────────────────────────────────────────────────────────────────────────
LOCAL_ZARR_ROOT = Path("/huge/proc/met-data/zarr")
NOAA_S3_BUCKET = "noaa-oar-mlwp-data"
NOAA_CACHE_DIR = Path("/huge/proc/larissa/noaa_s3_cache")
WEATHERNEXT_CACHE_DIR = Path("/huge/proc/larissa/weathernext2_cache")
OUTPUT_DIR = Path("./")
TRACK_DEBUG_DIR = Path("/huge/users/larissa/ExtremeWeatherBench/run")
TC_TRACK_CACHE_DIR = Path("/huge/proc/larissa/tc_tracks_cache")
WEATHERNEXT_GCS_ROOT = "gs://weathernext/weathernext_2_0_0/zarr/2025_to_present"

# Maximum lead time (hours) to retain when filtering forecasts.
# Overridable via --max-lead-hours CLI flag.
MAX_LEAD_HOURS: int = 240

# Local models to evaluate (all share the same zarr structure)
LOCAL_MODELS = [
#    "WeatherMesh-4",
    "WeatherMesh-4p5-Ens-Mean",
    "WeatherMesh-5c-Ens-Mean",
    "IFS-Ens-Mean",
    "AIFS-Ens-Mean",
#    "GFS-Ens-Mean",
]

# Models with upper-air data (q, u, v at pressure levels) — needed for AR events
LOCAL_MODELS_UPPER_AIR = [
    "WeatherMesh-4",
    "IFS",
    "AIFS",
    "GFS",
    "GFS-Ens-Mean",
]

# NOAA S3 models (IFS-initialised AI models)
NOAA_MODELS = {
#    "FourCastNet-v2 (IFS)": "FOUR_v200_IFS",
#    "GraphCast (IFS)":      "GRAP_v100_IFS",
#    "Pangu-Weather (IFS)":  "PANG_v100_IFS",
#    "Aurora (IFS)":         "AURO_v100_IFS",
}

GCS_MODELS = {
    "WeatherNext2": {
        "source": WEATHERNEXT_GCS_ROOT,
    },
}

# Variable mapping: local/NOAA names → EWB standard names
VARIABLE_MAPPING = {
    "msl":  "air_pressure_at_mean_sea_level",
    "z":    "geopotential",
    "u10":  "surface_eastward_wind",
    "v10":  "surface_northward_wind",
    "u":    "eastward_wind",
    "v":    "northward_wind",
    "t":    "air_temperature",
    "t2":   "surface_air_temperature",
    "t2m":  "surface_air_temperature",
    "gh":   "geopotential",      # local zarr uses gh (geopotential height in m)
    "q":    "specific_humidity",
    # WeatherNext2 names
    "mean_sea_level_pressure": "air_pressure_at_mean_sea_level",
    "10m_u_component_of_wind": "surface_eastward_wind",
    "10m_v_component_of_wind": "surface_northward_wind",
    "2m_temperature": "surface_air_temperature",
}

# Enforce common temporal sampling across all models.
ALLOWED_INIT_HOURS = (0, 12)


# ─────────────────────────────────────────────────────────────────────────────
#  Local zarr helpers
# ─────────────────────────────────────────────────────────────────────────────

def _discover_local_inits(
    model_name: str,
    start_date: datetime,
    end_date: datetime,
    init_hours: tuple[int, ...] = (0, 12),
    lookback_days: int = 15,
) -> list[Path]:
    """Find all local zarr init-time paths for *model_name* covering the case.

    We need inits that start up to 15 days *before* end_date (360 h lead time)
    and no later than end_date itself.  We restrict to 00Z and 12Z by default.
    """
    # Start searching lookback_days before case start.
    search_start = start_date - timedelta(days=lookback_days)
    search_end = end_date

    paths: list[Path] = []
    current = search_start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current <= search_end:
        for hour in init_hours:
            init_dt = current.replace(hour=hour)
            if init_dt < search_start or init_dt > search_end:
                continue
            init_str = init_dt.strftime("%Y%m%d%H")
            zarr_path = (
                LOCAL_ZARR_ROOT / init_str / model_name
                / f"{model_name}_{init_str}.zarr"
            )
            if zarr_path.exists():
                paths.append(zarr_path)
        current += timedelta(days=1)

    return sorted(paths)


def _local_needed_vars_for_event(event_type: str) -> set[str]:
    """Source variable names needed from local zarr for this event type."""
    if event_type == "tropical_cyclone":
        return {"msl", "u10", "v10"}
    if event_type == "atmospheric_river":
        return {"u", "v", "q"}
    if event_type == "heavy_precip":
        return {"tp_6hr"}
    return {"t2", "t2m"}


def _open_single_local_zarr(zarr_path: Path, needed_vars: set[str]) -> xr.Dataset:
    """Open one local zarr init and standardise to EWB conventions.

    Local zarr structure:
        dims:   step (hours float32), lat, lon, level
        coords: step, valid_time, lat, lon, level, init_time (scalar)
        vars:   msl, gh, u10, v10, ...

    Target EWB structure:
        dims:   lead_time (timedelta), latitude, longitude, level
        coords: init_time (scalar → will become dim after concat)
    """
    ds = xr.open_zarr(str(zarr_path), chunks="auto")
    # Only keep the variables we need to avoid loading the full ~19 GB file
    keep = [v for v in needed_vars if v in ds.data_vars]
    ds = ds[keep]

    # Convert step (hours as float32) → lead_time (timedelta64)
    step_hours = ds["step"].values.astype(float)
    lead_time = pd.to_timedelta(step_hours, unit="h")
    ds = ds.assign_coords(lead_time=("step", lead_time))
    ds = ds.swap_dims({"step": "lead_time"})
    
    # Preserve valid_time but re-index it by lead_time
    if "valid_time" in ds.coords:
        valid_time_values = ds["valid_time"].values
        ds = ds.drop_vars("valid_time")
        ds = ds.assign_coords(valid_time=("lead_time", valid_time_values))
    
    ds = ds.drop_vars("step", errors="ignore")

    # Rename spatial coords
    renames = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        renames["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        renames["lon"] = "longitude"

    # Rename variables
    for src, dst in VARIABLE_MAPPING.items():
        if src in ds.data_vars and dst not in ds.data_vars:
            renames[src] = dst

    if renames:
        ds = ds.rename(renames)

    # Ensure init_time is a proper coordinate
    if "init_time" not in ds.coords:
        # Parse from path: …/2025101000/Model/Model_2025101000.zarr
        init_str = zarr_path.parent.parent.name
        init_dt = pd.Timestamp(datetime.strptime(init_str, "%Y%m%d%H"))
        ds = ds.assign_coords(init_time=init_dt)

    return ds


def assemble_local_forecast(
    model_name: str,
    start_date: datetime,
    end_date: datetime,
    event_type: str,
    lookback_days: int = 15,
) -> xr.Dataset:
    """Assemble a multi-init dataset for a local model."""
    paths = _discover_local_inits(
        model_name,
        start_date,
        end_date,
        lookback_days=lookback_days,
    )
    if not paths:
        raise FileNotFoundError(
            f"No local zarr files found for {model_name} "
            f"in range {start_date} – {end_date}"
        )
    print(f"      Found {len(paths)} init times")
    needed_vars = _local_needed_vars_for_event(event_type)

    datasets = []
    for i, p in enumerate(paths, 1):
        try:
            if i == 1 or i % 10 == 0 or i == len(paths):
                print(f"      [{i}/{len(paths)}] {p.name}", flush=True)
            ds = _open_single_local_zarr(p, needed_vars=needed_vars)
            # Expand init_time to a dimension for concatenation
            ds = ds.expand_dims("init_time")
            datasets.append(ds)
        except Exception as e:
            print(f"      ⚠ Skipping {p.name}: {e}")

    if not datasets:
        raise FileNotFoundError(
            f"Could not open any zarr files for {model_name}"
        )

    combined = xr.concat(datasets, dim="init_time", coords="minimal", compat="override")
    
    # Recompute valid_time as 2D (init_time, lead_time) after concat
    # This ensures all init×lead combinations have correct valid_time
    if "init_time" in combined.dims and "lead_time" in combined.dims:
        init_times = combined.init_time.values
        lead_times = combined.lead_time.values
        # Broadcast: valid_time[i,j] = init_time[i] + lead_time[j]
        valid_time_2d = init_times[:, np.newaxis] + lead_times[np.newaxis, :]
        combined = combined.assign_coords(
            valid_time=(["init_time", "lead_time"], valid_time_2d)
        )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  NOAA S3 helpers
# ─────────────────────────────────────────────────────────────────────────────

_s3fs_instance: s3fs.S3FileSystem | None = None
_gcsfs_instance = None


def _get_s3fs() -> s3fs.S3FileSystem:
    """Lazy singleton for the S3 filesystem."""
    global _s3fs_instance
    if _s3fs_instance is None:
        _s3fs_instance = s3fs.S3FileSystem(anon=True)
    return _s3fs_instance


def _get_boto3_s3():
    """Lazy anonymous boto3 S3 client (faster for targeted reads)."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _get_gcsfs():
    """Lazy singleton for GCS filesystem (prefers authenticated access)."""
    global _gcsfs_instance
    if _gcsfs_instance is None:
        try:
            import gcsfs  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "gcsfs is required to load WeatherNext2 from Google Cloud Storage. "
                "Install with: conda install -n main gcsfs or pip install gcsfs"
            ) from exc
        # Prefer authenticated credentials (same identity used by gsutil/gcloud).
        # Fall back to anonymous for public buckets.
        try:
            _gcsfs_instance = gcsfs.GCSFileSystem()
        except Exception:
            _gcsfs_instance = gcsfs.GCSFileSystem(token="anon")
    return _gcsfs_instance


def _parse_weathernext_init_from_uri(uri: str) -> pd.Timestamp | None:
    """Parse init time from URI path segment like YYYYMMDD_HHhr_01_preds."""
    folder = uri.rstrip("/").split("/")[-2] if uri.endswith("/predictions.zarr") else uri.rstrip("/").split("/")[-1]
    m = re.match(r"(\d{8})_(\d{2})hr_.*_preds$", folder)
    if not m:
        return None
    ymd, hh = m.group(1), m.group(2)
    try:
        return pd.Timestamp(datetime.strptime(f"{ymd}{hh}", "%Y%m%d%H"))
    except ValueError:
        return None


def _discover_weathernext_zarrs(
    start_date: datetime,
    end_date: datetime,
    lookback_days: int = 10,
) -> list[tuple[pd.Timestamp, str]]:
    """Discover WeatherNext2 per-init zarr stores in date range.

    Uses deterministic candidate paths and fast existence checks:
      gs://.../YYYYMMDD_[00,06,12,18]hr_01_preds/predictions.zarr
    """
    fs = _get_gcsfs()

    # 240h max lead time => commonly 10 days before case start.
    search_start = start_date - timedelta(days=lookback_days)
    search_end = end_date
    found: list[tuple[pd.Timestamp, str]] = []
    t0 = time.time()
    n_days = (search_end.date() - search_start.date()).days + 1
    print(
        f"      Discovering WeatherNext2 zarrs: {search_start:%Y-%m-%d} to {search_end:%Y-%m-%d} "
        f"({n_days} day prefixes)",
        flush=True,
    )

    current = search_start.replace(hour=0, minute=0, second=0, microsecond=0)
    day_idx = 0
    init_hours = ALLOWED_INIT_HOURS
    while current <= search_end:
        day_idx += 1
        ymd = current.strftime("%Y%m%d")
        if day_idx == 1 or day_idx % 5 == 0 or day_idx == n_days:
            print(
                f"      [discover {day_idx}/{n_days}] scanning {ymd} "
                f"({time.time()-t0:.0f}s elapsed)",
                flush=True,
            )

        for hh in init_hours:
            init_dt = current.replace(hour=hh)
            if init_dt < search_start or init_dt > search_end:
                continue
            uri = (
                f"{WEATHERNEXT_GCS_ROOT}/"
                f"{ymd}_{hh:02d}hr_01_preds/predictions.zarr"
            )
            # gcsfs is fastest/reliable when checking an object within the zarr store.
            zmeta_obj = uri.replace("gs://", "").rstrip("/") + "/.zmetadata"
            try:
                if fs.exists(zmeta_obj):
                    found.append((pd.Timestamp(init_dt), uri))
            except Exception as exc:
                msg = str(exc)
                if (
                    "storage.objects.list" in msg
                    or "Permission" in msg
                    or "Anonymous caller" in msg
                ):
                    raise PermissionError(
                        "Could not access WeatherNext2 objects in GCS. "
                        "Run gcloud auth application-default login (or ensure service-account creds) "
                        "for Python/gcsfs access."
                    ) from exc
                print(
                    f"      ⚠ WeatherNext2 exists check error for {ymd} {hh:02d}Z: {exc}",
                    flush=True,
                )
        current += timedelta(days=1)

    unique = {}
    for init_dt, uri in found:
        unique[(init_dt, uri)] = None
    results = sorted(unique.keys(), key=lambda x: x[0])
    print(
        f"      WeatherNext2 discovery complete: {len(results)} init stores "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )
    return results


def _weathernext_needed_vars_for_event(event_type: str) -> list[str]:
    """Source variable names needed from WeatherNext2 for this event type."""
    if event_type == "tropical_cyclone":
        return [
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ]
    return ["2m_temperature"]


def _weathernext_cache_path(init_dt: pd.Timestamp) -> Path:
    """Build local cache path for one WeatherNext2 init."""
    init_str = pd.Timestamp(init_dt).strftime("%Y%m%d%H")
    yyyy = init_str[:4]
    mmdd = init_str[4:8]
    return WEATHERNEXT_CACHE_DIR / yyyy / mmdd / f"WN2_{init_str}_processed.zarr"


def _wn2_normalize(ds: xr.Dataset, init_dt: pd.Timestamp, needed_vars: list[str]) -> xr.Dataset:
    """Normalize a raw WN2 dataset: subset vars, reduce sample dim, rename, build lead_time."""
    keep = [v for v in needed_vars if v in ds.data_vars]
    if keep:
        ds = ds[keep]
    if "sample" in ds.dims:
        ds = ds.mean(dim="sample") if ds.sizes["sample"] > 1 else ds.isel(sample=0, drop=True)
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
        ds = ds.assign_coords(init_time=init_dt)
    if "time" in ds.dims:
        time_vals = ds["time"].values
        if np.issubdtype(np.asarray(time_vals).dtype, np.timedelta64):
            lead_time = time_vals
        elif np.issubdtype(np.asarray(time_vals).dtype, np.number):
            lead_time = pd.to_timedelta(np.asarray(time_vals).astype(float), unit="h")
        elif np.issubdtype(np.asarray(time_vals).dtype, np.datetime64):
            lead_time = pd.to_datetime(time_vals) - pd.Timestamp(ds.init_time.values)
        else:
            lead_time = pd.to_timedelta(np.asarray(time_vals))
        ds = ds.assign_coords(lead_time=("time", lead_time))
        ds = ds.swap_dims({"time": "lead_time"})
        if "datetime" in ds.coords:
            ds = ds.assign_coords(valid_time=("lead_time", pd.to_datetime(ds["datetime"].values, errors="coerce")))
        elif "valid_time" not in ds.coords:
            ds = ds.assign_coords(valid_time=("lead_time", pd.Timestamp(ds.init_time.values) + pd.to_timedelta(lead_time)))
        ds = ds.drop_vars(["time", "datetime"], errors="ignore")
    if "lead_time" in ds.dims:
        lead_hours = ds.lead_time / pd.Timedelta(hours=1)
        mask = (lead_hours > 0) & (lead_hours % 12 == 0) & (lead_hours <= MAX_LEAD_HOURS)
        ds = ds.sel(lead_time=ds.lead_time[mask])
    return ds


def _wn2_extend_cache(
    zarr_uri: str,
    init_dt: pd.Timestamp,
    needed_vars: list[str],
    cached_ds: xr.Dataset,
    cache_path: Path,
) -> xr.Dataset:
    """Extend a partial WN2 cache by lazily fetching only the missing lead times from GCS."""
    init_tag = pd.Timestamp(init_dt).strftime("%Y%m%d%H")
    fs = _get_gcsfs()
    gcs_path = zarr_uri.replace("gs://", "")
    mapper = fs.get_mapper(gcs_path)
    remote = xr.open_zarr(mapper, consolidated=True, chunks="auto")
    remote = _wn2_normalize(remote, init_dt, needed_vars)

    cached_lts = set(
        (pd.to_timedelta(cached_ds.lead_time.values).total_seconds() / 3600).astype(int)
    )
    remote_lts = pd.to_timedelta(remote.lead_time.values).total_seconds() / 3600
    new_mask = np.array([int(h) not in cached_lts for h in remote_lts])
    n_new = int(new_mask.sum())
    if n_new == 0:
        print(f"      [WN2 {init_tag}] no new lead times to add", flush=True)
        return cached_ds

    # Only load (download) the new lead times
    t_dl = time.time()
    new_part = remote.isel(lead_time=new_mask).load()
    print(
        f"      [WN2 {init_tag}] downloaded {n_new} new lead times in {time.time()-t_dl:.2f}s",
        flush=True,
    )

    merged = xr.concat([cached_ds.load(), new_part], dim="lead_time").sortby("lead_time")
    if cache_path.exists():
        shutil.rmtree(cache_path, ignore_errors=True)
    merged.to_zarr(cache_path, mode="w", zarr_version=2)
    return xr.open_zarr(cache_path, zarr_version=2)


def _open_single_weathernext_zarr(
    zarr_uri: str,
    init_dt: pd.Timestamp,
    needed_vars: list[str],
    use_mapper: bool = False,
) -> xr.Dataset:
    """Open and normalize one WeatherNext2 per-init zarr dataset."""
    t_total = time.time()
    init_tag = pd.Timestamp(init_dt).strftime("%Y%m%d%H")
    cache_path = _weathernext_cache_path(init_dt)
    if cache_path.exists():
        try:
            t_cache = time.time()
            cached_ds = xr.open_zarr(cache_path, zarr_version=2)
            cached_max_h = int(pd.to_timedelta(cached_ds.lead_time.values[-1]).total_seconds() / 3600)
            if cached_max_h >= MAX_LEAD_HOURS:
                cached_ds.attrs["_loaded_from_cache"] = True
                print(
                    f"      [WN2 {init_tag}] cache hit ({cached_max_h}h) in {time.time()-t_cache:.2f}s",
                    flush=True,
                )
                return cached_ds
            # Cache is short — extend it by fetching only the missing lead times
            # from GCS via lazy gcsfs open (no full download needed).
            print(
                f"      [WN2 {init_tag}] cache has {cached_max_h}h, need {MAX_LEAD_HOURS}h; "
                f"fetching missing lead times only",
                flush=True,
            )
            try:
                ds = _wn2_extend_cache(
                    zarr_uri, init_dt, needed_vars, cached_ds, cache_path,
                )
                print(
                    f"      [WN2 {init_tag}] cache extended in {time.time()-t_cache:.2f}s",
                    flush=True,
                )
                return ds
            except Exception as exc:
                print(
                    f"      [WN2 {init_tag}] ⚠ extend failed ({exc}); full refetch",
                    flush=True,
                )
        except Exception as exc:
            print(f"      [WN2 {init_tag}] ⚠ Invalid cache ({exc}); refetching")
            try:
                if cache_path.is_dir():
                    shutil.rmtree(cache_path, ignore_errors=True)
                else:
                    cache_path.unlink(missing_ok=True)
            except Exception:
                pass

    # --- Full download path (cache miss or extend failed) ---

    stage_dir = Path(tempfile.mkdtemp(prefix=f"wn2_stage_{init_tag}_", dir="/tmp"))
    stage_store = stage_dir / "predictions.zarr"
    stage_store.mkdir(parents=True, exist_ok=True)

    def _run_copy(src: str, dst: str, recursive: bool, required: bool = True):
        if shutil.which("gcloud"):
            cmd = ["gcloud", "storage", "cp"]
            if recursive:
                cmd.append("-r")
            cmd.extend([src, dst])
        elif shutil.which("gsutil"):
            cmd = ["gsutil", "-m", "cp"]
            if recursive:
                cmd.append("-r")
            cmd.extend([src, dst])
        else:
            raise RuntimeError("Neither gcloud nor gsutil is available for WeatherNext2 staging.")
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        except subprocess.CalledProcessError as exc:
            if required:
                err = (exc.stderr or exc.stdout or "").strip()
                raise RuntimeError(f"copy failed: {' '.join(cmd)} :: {err}") from exc

    try:
        t_stage = time.time()
        print(
            f"      [WN2 {init_tag}] stage download start: {zarr_uri} -> {stage_store}",
            flush=True,
        )
        for fname in [".zgroup", ".zattrs", ".zmetadata"]:
            _run_copy(f"{zarr_uri}/{fname}", str(stage_store), recursive=False, required=False)

        required_dirs = list(needed_vars) + ["sample", "time", "lat", "lon"]
        optional_dirs = ["datetime", "init_time"]
        for dname in required_dirs:
            _run_copy(f"{zarr_uri}/{dname}", str(stage_store), recursive=True, required=True)
        for dname in optional_dirs:
            _run_copy(f"{zarr_uri}/{dname}", str(stage_store), recursive=True, required=False)

        download_seconds = time.time() - t_stage
        print(
            f"      [WN2 {init_tag}] stage download done in {download_seconds:.2f}s",
            flush=True,
        )

        t_process_write = time.time()

        try:
            ds = xr.open_zarr(stage_store, consolidated=True, chunks="auto", zarr_version=2)
        except Exception:
            ds = xr.open_zarr(stage_store, consolidated=False, chunks="auto", zarr_version=2)

        ds = _wn2_normalize(ds, init_dt, needed_vars)

        t_write = time.time()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            shutil.rmtree(cache_path, ignore_errors=True)
        ds.to_zarr(cache_path, mode="w", zarr_version=2)
        print(f"      [WN2 {init_tag}] cache write done in {time.time()-t_write:.2f}s", flush=True)

        ds = xr.open_zarr(cache_path, zarr_version=2)
        print(
            f"      [WN2 {init_tag}] total pipeline (cache miss) {time.time()-t_total:.2f}s "
            f"(dl={download_seconds:.2f}s)",
            flush=True,
        )
        ds.attrs["_loaded_from_cache"] = False
        return ds
    finally:
        # Remove raw staged files after processing to avoid duplicate storage growth.
        try:
            if stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)
                print(
                    f"      [WN2 {init_tag}] cleaned staged raw store: {stage_dir}",
                    flush=True,
                )
        except Exception as cleanup_exc:
            print(
                f"      [WN2 {init_tag}] ⚠ Failed to clean staged raw store {stage_dir}: {cleanup_exc}",
                flush=True,
            )


def assemble_weathernext_forecast(
    display_name: str,
    start_date: datetime,
    end_date: datetime,
    event_type: str,
    max_workers: int = 1,
    lookback_days: int = 10,
    use_mapper: bool = False,
) -> xr.Dataset:
    """Assemble WeatherNext2 from GCS per-init zarr stores."""
    entries = _discover_weathernext_zarrs(
        start_date,
        end_date,
        lookback_days=lookback_days,
    )
    needed_vars = _weathernext_needed_vars_for_event(event_type)
    if not entries:
        raise FileNotFoundError(
            f"No WeatherNext2 zarr stores found in range {start_date} – {end_date}"
        )

    print(f"      Found {len(entries)} init zarr stores on GCS")
    datasets: list[xr.Dataset] = []
    errors: list[str] = []
    t0 = time.time()

    def _load_one(entry: tuple[pd.Timestamp, str]) -> tuple[str, xr.Dataset | None, str | None, bool, float]:
        init_dt, uri = entry
        folder_name = uri.split("/")[-2]
        t_init = time.time()
        try:
            ds = _open_single_weathernext_zarr(
                uri,
                init_dt,
                needed_vars=needed_vars,
                use_mapper=use_mapper,
            )
            from_cache = ds.attrs.pop("_loaded_from_cache", False)
            ds = ds.expand_dims("init_time")
            return folder_name, ds, None, from_cache, time.time() - t_init
        except Exception as exc:
            return folder_name, None, str(exc), False, time.time() - t_init

    n_workers = min(max_workers, len(entries))
    done = 0
    print(f"      Loading WeatherNext2 with {n_workers} worker(s)...", flush=True)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pending = {pool.submit(_load_one, e) for e in entries}
        while pending:
            ready, pending = wait(
                pending,
                timeout=20,
                return_when=FIRST_COMPLETED,
            )
            if not ready:
                print(
                    f"      ...still waiting ({done}/{len(entries)} complete, "
                    f"{time.time()-t0:.0f}s elapsed)",
                    flush=True,
                )
                continue

            for future in ready:
                done += 1
                folder_name, ds, err, from_cache, dt = future.result()
                if ds is not None:
                    datasets.append(ds)
                    cache_tag = " (cached)" if from_cache else " (fetched+cached)"
                    print(
                        f"      [{done}/{len(entries)}] {folder_name}{cache_tag} ✓ "
                        f"({dt:.1f}s, {time.time()-t0:.0f}s elapsed)",
                        flush=True,
                    )
                else:
                    errors.append(f"{folder_name}: {err}")
                    print(
                        f"      [{done}/{len(entries)}] {folder_name} ⚠ {err}",
                        flush=True,
                    )

    if not datasets:
        raise FileNotFoundError(f"Could not open any WeatherNext2 zarr stores for {display_name}")

    combined = xr.concat(datasets, dim="init_time", coords="minimal", compat="override")
    combined = combined.sortby("init_time")

    if "init_time" in combined.dims and "lead_time" in combined.dims:
        init_times = combined.init_time.values
        lead_times = combined.lead_time.values
        valid_time_2d = init_times[:, np.newaxis] + lead_times[np.newaxis, :]
        combined = combined.assign_coords(
            valid_time=(["init_time", "lead_time"], valid_time_2d)
        )

    return combined


def _discover_noaa_files(
    s3_model_key: str,
    start_date: datetime,
    end_date: datetime,
    lookback_days: int = 10,
) -> list[str]:
    """List NOAA S3 NetCDF *keys* (no ``s3://`` prefix) for a model.

    Path pattern:
        noaa-oar-mlwp-data/{MODEL}/{YYYY}/{MMDD}/{MODEL}_{YYYYMMDDHH}_f000_f240_06.nc
    """
    fs = _get_s3fs()

    # Search lookback_days before start (240 h max lead time for NOAA data)
    search_start = start_date - timedelta(days=lookback_days)
    search_end = end_date

    keys: list[str] = []
    current = search_start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current <= search_end:
        mmdd = current.strftime("%m%d")
        year = current.strftime("%Y")
        prefix = f"{NOAA_S3_BUCKET}/{s3_model_key}/{year}/{mmdd}/"
        try:
            items = fs.ls(prefix)
            for item in items:
                if item.endswith(".nc"):
                    fname = item.rsplit("/", 1)[-1]
                    try:
                        init_str = fname.split("_")[3]  # YYYYMMDDHH
                        init_hour = int(init_str[-2:])
                    except Exception:
                        continue
                    if init_hour not in ALLOWED_INIT_HOURS:
                        continue
                    # item is bucket-relative already (no s3://)
                    keys.append(item)
        except FileNotFoundError:
            pass  # day not available
        current += timedelta(days=1)

    return sorted(keys)


def _noaa_needed_vars_for_event(event_type: str) -> list[str]:
    """Source variable names needed from NOAA files for this event type."""
    if event_type == "tropical_cyclone":
        return ["msl", "u10", "v10"]
    return ["t2", "t2m"]


def _open_single_noaa_nc(s3_key: str, needed_vars: list[str]) -> xr.Dataset:
    """Open one NOAA S3 NetCDF with zarr caching.

    First checks if processed zarr cache exists in NOAA_CACHE_DIR.
    If cached: loads from zarr (~1-2s)
    If not: downloads from S3, processes, saves to zarr cache, returns (~100s first time)
    
    Cache structure: /huge/proc/larissa/noaa_s3_cache/{MODEL}/{YYYY}/{MMDD}/{file}_processed.zarr/
    
    Cached data: only surface vars (msl→air_pressure_at_mean_sea_level, u10, v10)
                 at 12-hourly intervals, ~260 MB per file vs 2360 MB original.
    """
    import tempfile
    import os
    
    # Parse metadata from S3 key
    fname = s3_key.rsplit("/", 1)[-1]  # e.g., "FOUR_v200_IFS_2025101000_f000_f240_06.nc"
    model = s3_key.split("/")[1]  # e.g., "FOUR_v200_IFS"
    init_str = fname.split("_")[3]  # e.g., "2025101000"
    init_dt = pd.Timestamp(datetime.strptime(init_str, "%Y%m%d%H"))
    
    # Build cache path
    yyyy = init_str[:4]
    mmdd = init_str[4:8]
    cache_fname = fname.replace(".nc", "_processed.zarr")
    cache_path = NOAA_CACHE_DIR / model / yyyy / mmdd / cache_fname
    
    _noaa_cached_ds = None
    if cache_path.exists():
        ds = xr.open_zarr(cache_path)
        cached_max_h = int(pd.to_timedelta(ds.lead_time.values[-1]).total_seconds() / 3600)
        if cached_max_h >= MAX_LEAD_HOURS:
            if "valid_time" not in ds.coords and "init_time" in ds.coords and "lead_time" in ds.coords:
                valid_times = ds.init_time.values + ds.lead_time.values
                ds = ds.assign_coords(valid_time=("lead_time", valid_times))
            ds.attrs['_loaded_from_cache'] = True
            return ds
        print(f"      [{model}] cache has {cached_max_h}h, need {MAX_LEAD_HOURS}h; will extend")
        _noaa_cached_ds = ds.load()
    
    # Not cached: download from S3, process, save to zarr
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket = s3_key.split("/")[0]
    key = "/".join(s3_key.split("/")[1:])
    
    # Download to /tmp
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False, dir="/tmp")
    tmp_path = tmp.name
    tmp.close()
    
    try:
        s3.download_file(bucket, key, tmp_path)
        
        # Open and process: extract surface vars at 12-hourly
        ds = xr.open_dataset(tmp_path, engine="h5netcdf")
        keep = [v for v in needed_vars if v in ds.data_vars]
        ds = ds[keep].isel(time=slice(None, None, 2))  # 12-hourly
        ds.load()
        ds.close()
        
        # Convert time → lead_time, but keep valid_time
        valid_times = pd.DatetimeIndex(ds["time"].values)
        lead_times = valid_times - init_dt
        ds = ds.assign_coords(lead_time=("time", lead_times))
        ds = ds.assign_coords(valid_time=("time", valid_times))
        ds = ds.swap_dims({"time": "lead_time"})
        ds = ds.drop_vars("time", errors="ignore")
        
        # Add init_time
        ds = ds.assign_coords(init_time=init_dt)
        
        # Rename variables to EWB conventions
        renames = {}
        for src, dst in VARIABLE_MAPPING.items():
            if src in ds.data_vars and dst not in ds.data_vars:
                renames[src] = dst
        if renames:
            ds = ds.rename(renames)
        
        # Merge with existing partial cache if extending.
        if _noaa_cached_ds is not None:
            cached_lts = set(
                (pd.to_timedelta(_noaa_cached_ds.lead_time.values).total_seconds() / 3600).astype(int)
            )
            new_lts = pd.to_timedelta(ds.lead_time.values).total_seconds() / 3600
            new_mask = ~np.isin(new_lts.astype(int), list(cached_lts))
            if new_mask.any():
                new_part = ds.isel(lead_time=new_mask)
                ds = xr.concat([_noaa_cached_ds, new_part], dim="lead_time").sortby("lead_time")
                print(f"      [{model}] extended: {len(cached_lts)} existing + {int(new_mask.sum())} new lead times")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            shutil.rmtree(cache_path, ignore_errors=True)
        ds.to_zarr(cache_path, mode="w")
        
        return ds
    
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def assemble_noaa_forecast(
    display_name: str,
    s3_model_key: str,
    start_date: datetime,
    end_date: datetime,
    event_type: str,
    max_workers: int = 6,
    lookback_days: int = 10,
) -> xr.Dataset:
    """Assemble a multi-init dataset from the NOAA S3 bucket.

    Downloads files in parallel using ThreadPoolExecutor (pattern from
    deep/met-analysis/utils/file_tools.py).
    """
    keys = _discover_noaa_files(
        s3_model_key,
        start_date,
        end_date,
        lookback_days=lookback_days,
    )
    needed_vars = _noaa_needed_vars_for_event(event_type)
    if not keys:
        raise FileNotFoundError(
            f"No NOAA S3 files found for {display_name} ({s3_model_key}) "
            f"in range {start_date} – {end_date}"
        )
    print(f"      Found {len(keys)} init files on S3")

    datasets: list[xr.Dataset] = []
    errors: list[str] = []
    t0 = time.time()

    def _load_one(key: str) -> tuple[str, xr.Dataset | None, str | None, bool]:
        fname = key.rsplit("/", 1)[-1]
        try:
            ds = _open_single_noaa_nc(key, needed_vars=needed_vars)
            from_cache = ds.attrs.pop('_loaded_from_cache', False)
            
            # Ensure init_time is a coordinate (may be missing in cached zarr)
            if "init_time" not in ds.coords:
                # Parse from filename as fallback
                init_str = fname.split("_")[3]
                init_dt = pd.Timestamp(datetime.strptime(init_str, "%Y%m%d%H"))
                ds = ds.assign_coords(init_time=init_dt)
            
            # Expand init_time to a dimension for concatenation
            ds = ds.expand_dims("init_time")
            return fname, ds, None, from_cache
        except Exception as e:
            return fname, None, str(e), False

    n_workers = min(max_workers, len(keys))
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_one, k): k for k in keys}
        for future in as_completed(futures):
            done += 1
            fname, ds, err, from_cache = future.result()
            if ds is not None:
                datasets.append(ds)
                cache_tag = " (cached)" if from_cache else ""
                print(f"      [{done}/{len(keys)}] {fname}{cache_tag} ✓ "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
            else:
                errors.append(f"{fname}: {err}")
                print(f"      [{done}/{len(keys)}] {fname} ⚠ {err}", flush=True)

    if not datasets:
        raise FileNotFoundError(
            f"Could not open any files for {display_name}"
        )

    print(f"      Concatenating {len(datasets)} datasets...", flush=True)
    combined = xr.concat(datasets, dim="init_time", coords="minimal", compat="override")
    # Sort by init_time since ThreadPoolExecutor results come in any order
    combined = combined.sortby("init_time")
    
    # Recompute valid_time as 2D (init_time, lead_time) after concat
    # This ensures all init×lead combinations have correct valid_time
    if "init_time" in combined.dims and "lead_time" in combined.dims:
        init_times = combined.init_time.values
        lead_times = combined.lead_time.values
        # Broadcast: valid_time[i,j] = init_time[i] + lead_time[j]
        valid_time_2d = init_times[:, np.newaxis] + lead_times[np.newaxis, :]
        combined = combined.assign_coords(
            valid_time=(["init_time", "lead_time"], valid_time_2d)
        )
    print(f"      Done in {time.time()-t0:.0f}s total")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing (shared by all models)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_12hourly(ds: xr.Dataset, forecast_name: str = "Unknown") -> xr.Dataset:
    """Filter lead_time to 12-hourly (excluding hour 0), up to MAX_LEAD_HOURS."""
    if "lead_time" not in ds.dims:
        return ds

    lead_hours = ds.lead_time / pd.Timedelta(hours=1)
    mask = (lead_hours > 0) & (lead_hours % 12 == 0) & (lead_hours <= MAX_LEAD_HOURS)
    ds = ds.sel(lead_time=ds.lead_time[mask])
    return ds


def preprocess_6hourly(ds: xr.Dataset, forecast_name: str = "Unknown") -> xr.Dataset:
    """Filter lead_time to 6-hourly (excluding hour 0), up to MAX_LEAD_HOURS."""
    if "lead_time" not in ds.dims:
        return ds

    lead_hours = ds.lead_time / pd.Timedelta(hours=1)
    mask = (lead_hours > 0) & (lead_hours % 6 == 0) & (lead_hours <= MAX_LEAD_HOURS)
    ds = ds.sel(lead_time=ds.lead_time[mask])
    return ds


def preprocess_init_hours(ds: xr.Dataset, forecast_name: str = "Unknown") -> xr.Dataset:
    """Keep only common init cycles (00Z/12Z) across models."""
    if "init_time" not in ds.dims:
        return ds

    init_vals = pd.to_datetime(ds["init_time"].values, errors="coerce")
    keep_mask = np.array(
        [(not pd.isna(ts)) and (ts.hour in ALLOWED_INIT_HOURS) for ts in init_vals],
        dtype=bool,
    )
    before = int(len(init_vals))
    after = int(np.sum(keep_mask))
    if after < before:
        print(
            f"      → Filtering init_time to 00/12Z for {forecast_name}: "
            f"{before} -> {after}"
        )
    if after == 0:
        print(f"      ⚠ No 00/12Z init_time values for {forecast_name} after filtering")
        return ds.isel(init_time=slice(0, 0))
    return ds.sel(init_time=ds.init_time[keep_mask])


def preprocess_tc(
    ds: xr.Dataset,
    forecast_name: str = "Unknown",
    is_true_geopotential: bool = False,
) -> xr.Dataset:
    """TC-specific preprocessing: lon conversion + geopotential thickness.

    Args:
        ds: The forecast dataset.
        forecast_name: Display name for log messages.
        is_true_geopotential: If True, the ``geopotential`` variable is in
            m²/s² (e.g. NOAA ``z``).  If False (default), it is geopotential
            *height* in metres (e.g. local ``gh``).
    """
    # Convert longitude 0-360 → -180..180
    if "longitude" in ds.coords:
        lon = ds.longitude
        if float(lon.min()) >= 0 and float(lon.max()) > 180:
            print(f"      → Converting longitude to -180..180 for {forecast_name}")
            ds = ds.assign_coords(longitude=(lon + 180) % 360 - 180)
            ds = ds.sortby("longitude")

    # Compute geopotential thickness if possible
    if "geopotential" in ds.data_vars and "level" in ds.dims:
        try:
            ds["geopotential_thickness"] = ewb.calc.geopotential_thickness(
                ds["geopotential"],
                top_level=300,
                bottom_level=500,
                geopotential=is_true_geopotential,
            )
        except (KeyError, ValueError):
            pass

    # Add dummy thickness if not computed
    if "geopotential_thickness" not in ds.data_vars:
        for var_name in ["air_pressure_at_mean_sea_level"]:
            if var_name in ds.data_vars:
                ds["geopotential_thickness"] = xr.full_like(
                    ds[var_name], float("nan"), dtype=float
                )
                ds["geopotential_thickness"].attrs["note"] = (
                    "Dummy variable – TC tracking will skip warm-core check"
                )
                break

    return ds


# ─────────────────────────────────────────────────────────────────────────────
#  Event-type helpers (reused from evaluate_case_2020.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_variables_for_event_type(event_type: str) -> list:
    """Required EWB variable names for a given event type.
    
    Note: For NOAA S3 models, we don't include 'geopotential' because:
    - It's 2+ GB per file (13 pressure levels)
    - We provide 'geopotential_thickness' as a dummy (NaN-filled)
    - TC tracking will skip the warm-core check when thickness is NaN
    """
    if event_type == "tropical_cyclone":
        return [
            "air_pressure_at_mean_sea_level",
            # "geopotential",  # Skipped for NOAA models - use dummy thickness instead
            "surface_eastward_wind",
            "surface_northward_wind",
        ]
    elif event_type == "atmospheric_river":
        return [
            ewb.derived.AtmosphericRiverVariables(
                output_variables=[
                    "atmospheric_river_land_intersection",
                    "integrated_vapor_transport",
                ]
            )
        ]
    elif event_type == "heavy_precip":
        return ["tp_6hr"]
    elif event_type == "heat_wave":
        return ["surface_air_temperature"]
    else:
        return ["surface_air_temperature"]


def get_metrics_for_event_type(event_type: str) -> list:
    """Metric instances appropriate for a given event type."""
    if event_type == "tropical_cyclone":
        mslp = "air_pressure_at_mean_sea_level"
        return [
            ewb.metrics.RootMeanSquaredError(
                forecast_variable=mslp, target_variable=mslp,
            ),
            ewb.metrics.MeanAbsoluteError(
                forecast_variable=mslp, target_variable=mslp,
            ),
            ewb.metrics.LandfallDisplacement(
                forecast_variable=mslp, target_variable=mslp,
            ),
            ewb.metrics.LandfallTimeMeanError(
                forecast_variable=mslp, target_variable=mslp,
            ),
            ewb.metrics.LandfallIntensityMeanAbsoluteError(
                forecast_variable=mslp, target_variable=mslp,
            ),
            ewb.metrics.LandfallIntensityRootMeanSquaredError(
                forecast_variable=mslp, target_variable=mslp,
            ),
            ewb.metrics.TrackIntensityMeanAbsoluteError(
                forecast_variable=mslp, target_variable=mslp,
            ),
            ewb.metrics.AlongTrackError(
                forecast_variable=mslp, target_variable=mslp,
            ),
            ewb.metrics.CrossTrackError(
                forecast_variable=mslp, target_variable=mslp,
            ),
            ewb.metrics.TotalTrackError(
                forecast_variable=mslp, target_variable=mslp,
            ),
        ]
    elif event_type == "atmospheric_river":
        ar_var = "atmospheric_river_land_intersection"
        ivt_var = "integrated_vapor_transport"
        return [
            # Binary AR detection metrics
            ewb.metrics.CriticalSuccessIndex(
                forecast_variable=ar_var, target_variable=ar_var,
            ),
            ewb.metrics.SpatialDisplacement(
                forecast_variable=ar_var, target_variable=ar_var,
            ),
            ewb.metrics.EarlySignal(
                forecast_variable=ar_var, target_variable=ar_var,
            ),
            # IVT intensity metrics
            ewb.metrics.RootMeanSquaredError(
                forecast_variable=ivt_var, target_variable=ivt_var,
            ),
            ewb.metrics.MeanAbsoluteError(
                forecast_variable=ivt_var, target_variable=ivt_var,
            ),
            ewb.metrics.MeanError(
                forecast_variable=ivt_var, target_variable=ivt_var,
            ),
        ]
    elif event_type == "heavy_precip":
        v = "tp_6hr"
        metrics = [
            ewb.metrics.RootMeanSquaredError(
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=v, target_variable=v,
            ),
            ewb.metrics.MeanAbsoluteError(
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=v, target_variable=v,
            ),
            ewb.metrics.MeanError(
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=v, target_variable=v,
            ),
        ]
        for thresh_m in PRECIP_THRESHOLDS_M:
            metrics.append(ewb.metrics.ThresholdMetric(
                name=f"threshold_{thresh_m}",
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=v, target_variable=v,
                forecast_threshold=thresh_m, target_threshold=thresh_m,
                metrics=[
                    ewb.metrics.FrequencyBias,
                    ewb.metrics.EquitableThreatScore,
                    ewb.metrics.CriticalSuccessIndex,
                ],
            ))
        return metrics
    elif event_type == "heat_wave":
        t = "surface_air_temperature"
        return [
            ewb.metrics.RootMeanSquaredError(
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=t, target_variable=t,
            ),
            ewb.metrics.MeanAbsoluteError(
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=t, target_variable=t,
            ),
            ewb.metrics.MeanError(
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=t, target_variable=t,
            ),
            ewb.metrics.ClimatologyEventTimingComposite(
                event_kind="heat_wave",
                preserve_dims="init_time",
                reduce_spatial_dims=[],
                forecast_variable=t,
                target_variable=t,
            ),
        ]
    elif event_type == "freeze":
        t = "surface_air_temperature"
        return [
            ewb.metrics.RootMeanSquaredError(
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=t, target_variable=t,
            ),
            ewb.metrics.MeanAbsoluteError(
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=t, target_variable=t,
            ),
            ewb.metrics.MeanError(
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=t, target_variable=t,
            ),
            ewb.metrics.ClimatologyEventTimingComposite(
                event_kind="freeze",
                preserve_dims="init_time",
                reduce_spatial_dims=[],
                forecast_variable=t,
                target_variable=t,
            ),
        ]
    else:
        t = "surface_air_temperature"
        return [
            ewb.metrics.RootMeanSquaredError(
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=t, target_variable=t,
            ),
            ewb.metrics.MeanAbsoluteError(
                preserve_dims=["lead_time", "valid_time"],
                forecast_variable=t, target_variable=t,
            ),
        ]


def get_metrics_for_event_type_fast(event_type: str) -> list:
    """Fast baseline metric set for quick iteration runs."""
    if event_type == "tropical_cyclone":
        # Keep TC metric behavior unchanged for now.
        return get_metrics_for_event_type(event_type)
    t = "surface_air_temperature"
    return [
        ewb.metrics.RootMeanSquaredError(
            preserve_dims=["lead_time", "valid_time"],
            forecast_variable=t, target_variable=t,
        ),
        ewb.metrics.MeanAbsoluteError(
            preserve_dims=["lead_time", "valid_time"],
            forecast_variable=t, target_variable=t,
        ),
    ]


def _track_csv_path_for_forecast(forecast_name: str) -> Path:
    """Return per-model debug track CSV path used by metrics.py."""
    safe_name = forecast_name.replace(" ", "_").replace("/", "_")
    return TRACK_DEBUG_DIR / f"forecast_tracks_{safe_name}.csv"


def _tc_track_cache_path_for_forecast(
    case_id: int,
    forecast_name: str,
    min_track_timesteps: int = 5,
) -> Path:
    """Return TC track zarr cache path used by TropicalCycloneTrackVariables."""
    cache_key = f"{forecast_name}_case{case_id}_mintrack{min_track_timesteps}"
    return TC_TRACK_CACHE_DIR / f"{cache_key}.zarr"


def print_tc_track_cache_status(
    case: ewb.IndividualCase,
    forecasts: list[ewb.inputs.XarrayForecast],
    min_track_timesteps: int = 5,
):
    """Print whether per-model TC track zarr cache exists."""
    print("\n🌀 TC Track Compute Cache Status:")
    print(f"   Cache directory: {TC_TRACK_CACHE_DIR}")
    for forecast in forecasts:
        cache_path = _tc_track_cache_path_for_forecast(
            case_id=case.case_id_number,
            forecast_name=forecast.name,
            min_track_timesteps=min_track_timesteps,
        )
        if cache_path.exists():
            print(
                f"   • {forecast.name}: FOUND {cache_path.name} "
                "-> track detection likely loaded from cache"
            )
        else:
            print(
                f"   • {forecast.name}: MISSING {cache_path.name} "
                "-> track detection will be recomputed"
            )


def clear_tc_track_cache(case_id: int) -> int:
    """Delete case-specific TC track zarr caches.

    Returns:
        Number of cache entries removed.
    """
    print("\n🧹 Clearing TC track cache...")
    print(f"   Directory: {TC_TRACK_CACHE_DIR}")
    pattern = f"*_case{case_id}_mintrack*.zarr"
    candidates = sorted(TC_TRACK_CACHE_DIR.glob(pattern))
    if not candidates:
        print(f"   No matching cache entries found for case {case_id}")
        return 0

    removed = 0
    for path in candidates:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            removed += 1
            print(f"   • removed {path.name}")
        except Exception as exc:
            print(f"   • failed to remove {path.name}: {exc}")

    print(f"   Removed {removed}/{len(candidates)} cache entries")
    return removed


def print_track_csv_cache_status(forecasts: list[ewb.inputs.XarrayForecast]):
    """Print whether per-model track debug CSVs already exist."""
    print("\n🧭 Track Debug CSV Status:")
    print("   Storm tracking metrics are recomputed from forecast fields each run.")
    print("   forecast_tracks_*.csv files are debug outputs that may be reused.")
    print(f"   Debug CSV directory: {TRACK_DEBUG_DIR}")
    for forecast in forecasts:
        csv_path = _track_csv_path_for_forecast(forecast.name)
        if csv_path.exists():
            print(
                f"   • {forecast.name}: found {csv_path.name} "
                "-> existing debug CSV will be reused (no rewrite)"
            )
        else:
            print(
                f"   • {forecast.name}: missing {csv_path.name} "
                "-> debug CSV will be created during evaluation"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Forecast loaders (build XarrayForecast objects)
# ─────────────────────────────────────────────────────────────────────────────

def _make_preprocess(
    name: str,
    event_type: str,
    is_true_geopotential: bool = False,
):
    """Create a preprocessing pipeline for a forecast."""
    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        ds = preprocess_init_hours(ds, forecast_name=name)
        if event_type == "heavy_precip":
            ds = preprocess_6hourly(ds, forecast_name=name)
        else:
            ds = preprocess_12hourly(ds, forecast_name=name)
        if event_type == "tropical_cyclone":
            ds = preprocess_tc(
                ds,
                forecast_name=name,
                is_true_geopotential=is_true_geopotential,
            )
        return ds
    return _preprocess


def _required_base_variables(event_type: str) -> list[str]:
    """Return required string variables for the event type."""
    return [v for v in get_variables_for_event_type(event_type) if isinstance(v, str)]


def _missing_required_variables(ds: xr.Dataset, required_vars: list[str]) -> list[str]:
    """List required variables missing from dataset."""
    return [v for v in required_vars if v not in ds.data_vars]


def load_local_forecasts(
    case: ewb.IndividualCase,
    lookback_days: int | None = None,
    only_model: str | None = None,
) -> list[ewb.inputs.XarrayForecast]:
    """Load all local models as XarrayForecast objects."""
    forecasts = []
    variables = get_variables_for_event_type(case.event_type)
    required_vars = _required_base_variables(case.event_type)

    # Add TC derived variables if needed
    if case.event_type == "tropical_cyclone":
        variables = variables + [
            ewb.derived.TropicalCycloneTrackVariables(min_track_timesteps=5)
        ]

    # AR events need models with upper-air data (q, u, v at pressure levels)
    model_list = LOCAL_MODELS_UPPER_AIR if case.event_type == "atmospheric_river" else LOCAL_MODELS

    for model_name in model_list:
        if only_model and model_name != only_model:
            continue
        print(f"   → {model_name} (local)")
        try:
            lb = lookback_days if lookback_days is not None else max(15, -(-MAX_LEAD_HOURS // 24))
            ds = assemble_local_forecast(
                model_name,
                case.start_date,
                case.end_date,
                event_type=case.event_type,
                lookback_days=lb,
            )

            # Apply preprocessing
            preprocess_fn = _make_preprocess(model_name, case.event_type)
            ds = preprocess_fn(ds)
            missing_vars = _missing_required_variables(ds, required_vars)
            if missing_vars:
                print(
                    f"      ⏭ Skipping {model_name}: missing required variables "
                    f"{missing_vars} for event_type={case.event_type}"
                )
                continue
            
            # Debug: print dataset info
            print(f"      Dataset info before XarrayForecast:")
            print(f"        Dims: {dict(ds.dims)}")
            print(f"        Coords: {list(ds.coords)}")
            if "valid_time" in ds.coords:
                vt = ds.valid_time.values
                print(f"        valid_time range: {vt.min()} to {vt.max()}")
            else:
                print(f"        ⚠ valid_time MISSING!")
            if "init_time" in ds.coords:
                it = ds.init_time.values
                if hasattr(it, 'min'):
                    print(f"        init_time range: {it.min()} to {it.max()}")
                else:
                    print(f"        init_time: {it.min() if it.size > 1 else it}")
            print(f"        Variables: {list(ds.data_vars)[:5]}...")

            forecast = ewb.inputs.XarrayForecast(
                ds=ds,
                name=model_name,
                variables=variables,
            )
            forecasts.append(forecast)
            print(f"      ✓ {len(ds.init_time)} inits, "
                  f"{len(ds.lead_time)} lead times")
        except Exception as e:
            print(f"      ✗ Failed: {e}")

    return forecasts


def load_noaa_forecasts(
    case: ewb.IndividualCase,
    lookback_days: int | None = None,
    only_model: str | None = None,
) -> list[ewb.inputs.XarrayForecast]:
    """Load all NOAA S3 models as XarrayForecast objects."""
    forecasts = []
    variables = get_variables_for_event_type(case.event_type)
    required_vars = _required_base_variables(case.event_type)

    if case.event_type == "tropical_cyclone":
        variables = variables + [
            ewb.derived.TropicalCycloneTrackVariables(min_track_timesteps=5)
        ]

    for display_name, s3_key in NOAA_MODELS.items():
        if only_model and display_name != only_model:
            continue
        print(f"   → {display_name} (S3)")
        try:
            lb = lookback_days if lookback_days is not None else max(10, -(-MAX_LEAD_HOURS // 24))
            ds = assemble_noaa_forecast(
                display_name,
                s3_key,
                case.start_date,
                case.end_date,
                event_type=case.event_type,
                lookback_days=lb,
            )

            # NOAA z variable is true geopotential (m²/s²)
            preprocess_fn = _make_preprocess(
                display_name, case.event_type, is_true_geopotential=True
            )
            ds = preprocess_fn(ds)
            missing_vars = _missing_required_variables(ds, required_vars)
            if missing_vars:
                print(
                    f"      ⏭ Skipping {display_name}: missing required variables "
                    f"{missing_vars} for event_type={case.event_type}"
                )
                continue
            
            # Debug: print dataset info
            print(f"      Dataset info before XarrayForecast:")
            print(f"        Dims: {dict(ds.dims)}")
            print(f"        Coords: {list(ds.coords)}")
            if "valid_time" in ds.coords:
                vt = ds.valid_time.values
                print(f"        valid_time range: {vt.min()} to {vt.max()}")
            else:
                print(f"        ⚠ valid_time MISSING!")
            if "init_time" in ds.coords:
                it = ds.init_time.values
                if hasattr(it, 'min'):
                    print(f"        init_time range: {it.min()} to {it.max()}")
                else:
                    print(f"        init_time: {it.min() if it.size > 1 else it}")
            print(f"        Variables: {list(ds.data_vars)[:5]}...")

            forecast = ewb.inputs.XarrayForecast(
                ds=ds,
                name=display_name,
                variables=variables,
            )
            forecasts.append(forecast)
            print(f"      ✓ {len(ds.init_time)} inits, "
                  f"{len(ds.lead_time)} lead times")
        except Exception as e:
            print(f"      ✗ Failed: {e}")

    return forecasts


def load_gcs_forecasts(
    case: ewb.IndividualCase,
    lookback_days: int | None = None,
    wn2_use_mapper: bool = False,
    wn2_max_workers: int = 1,
    only_model: str | None = None,
) -> list[ewb.inputs.XarrayForecast]:
    """Load Google Cloud models (WeatherNext2) as XarrayForecast objects."""
    forecasts = []
    variables = get_variables_for_event_type(case.event_type)
    required_vars = _required_base_variables(case.event_type)

    if case.event_type == "tropical_cyclone":
        variables = variables + [
            ewb.derived.TropicalCycloneTrackVariables(min_track_timesteps=5)
        ]

    for display_name in GCS_MODELS.keys():
        if only_model and display_name != only_model:
            continue
        print(f"   → {display_name} (GCS)")
        try:
            lb = lookback_days if lookback_days is not None else max(10, -(-MAX_LEAD_HOURS // 24))
            ds = assemble_weathernext_forecast(
                display_name,
                case.start_date,
                case.end_date,
                event_type=case.event_type,
                max_workers=wn2_max_workers,
                lookback_days=lb,
                use_mapper=wn2_use_mapper,
            )

            preprocess_fn = _make_preprocess(
                display_name, case.event_type, is_true_geopotential=False
            )
            ds = preprocess_fn(ds)
            missing_vars = _missing_required_variables(ds, required_vars)
            if missing_vars:
                print(
                    f"      ⏭ Skipping {display_name}: missing required variables "
                    f"{missing_vars} for event_type={case.event_type}"
                )
                continue

            print(f"      Dataset info before XarrayForecast:")
            print(f"        Dims: {dict(ds.dims)}")
            print(f"        Coords: {list(ds.coords)}")
            if "valid_time" in ds.coords:
                vt = ds.valid_time.values
                print(f"        valid_time range: {vt.min()} to {vt.max()}")
            else:
                print(f"        ⚠ valid_time MISSING!")
            if "init_time" in ds.coords:
                it = ds.init_time.values
                if hasattr(it, 'min'):
                    print(f"        init_time range: {it.min()} to {it.max()}")
                else:
                    print(f"        init_time: {it.min() if it.size > 1 else it}")
            print(f"        Variables: {list(ds.data_vars)[:5]}...")

            forecast = ewb.inputs.XarrayForecast(
                ds=ds,
                name=display_name,
                variables=variables,
            )
            forecasts.append(forecast)
            print(f"      ✓ {len(ds.init_time)} inits, "
                  f"{len(ds.lead_time)} lead times")
        except Exception as e:
            print(f"      ✗ Failed: {e}")

    return forecasts


# ─────────────────────────────────────────────────────────────────────────────
#  Case + target loading
# ─────────────────────────────────────────────────────────────────────────────

def load_case_metadata(case_id: int) -> ewb.IndividualCase:
    """Load case metadata from events.yaml."""
    print(f"Loading case {case_id} metadata...")

    all_cases = ewb.cases.load_ewb_events_yaml_into_case_list()
    matching = [c for c in all_cases if c.case_id_number == case_id]

    if not matching:
        raise ValueError(f"Case {case_id} not found in events.yaml")

    case = matching[0]
    print(f"\n📋 Case Information:")
    print(f"   Title: {case.title}")
    print(f"   Type: {case.event_type}")
    print(f"   Dates: {case.start_date} to {case.end_date}")
    print(f"   Region: {case.location}")

    return case


def load_target(
    case: ewb.IndividualCase,
    target_type: str = "era5",
    ghcn_source: str | None = None,
):
    """Load target data based on event type and target type.

    Args:
        case: The case metadata.
        target_type: ``"era5"`` (default) for gridded ERA5 reanalysis, or
            ``"ghcn"`` for GHCN-H hourly station observations.
        ghcn_source: Optional path/URI to a custom GHCN parquet file.
            Defaults to the built-in GCS URI when *None*.
    """
    print(f"\n🎯 Loading target data...")

    if case.event_type == "tropical_cyclone":
        print("   Using IBTrACS (observed TC tracks)")
        return ewb.inputs.IBTrACS()

    if case.event_type == "heavy_precip":
        print("   Using MRMS (radar precipitation observations)")
        variables = get_variables_for_event_type(case.event_type)
        return ewb.inputs.MRMS(
            source=str(LOCAL_ZARR_ROOT),
            start_date=case.start_date,
            end_date=case.end_date,
            variables=variables,
        )

    if case.event_type == "atmospheric_river":
        print("   Using ERA5 reanalysis (AR events require gridded upper-air data)")
        variables = get_variables_for_event_type(case.event_type)
        return ewb.inputs.ERA5(variables=variables)

    if target_type == "ghcn":
        variables = get_variables_for_event_type(case.event_type)
        kwargs: dict = {"variables": variables}
        if ghcn_source is not None:
            kwargs["source"] = ghcn_source
        print(f"   Using GHCN-H station observations (source={kwargs.get('source', 'default GCS')})")
        return ewb.inputs.GHCN(**kwargs)

    # Default: ERA5
    print("   Using ERA5 reanalysis")
    variables = get_variables_for_event_type(case.event_type)
    return ewb.inputs.ERA5(variables=variables)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-station GHCN metrics
# ─────────────────────────────────────────────────────────────────────────────


def _compute_station_results(
    case: ewb.IndividualCase,
    target,
    forecasts: list,
    output_file: Path,
) -> None:
    """Compute per-station RMSE and bias for every forecast, save as CSV.

    This bypasses the EWB metric pipeline to produce a station-level breakdown
    that can be used for geographic station map plots.  Modelled on the
    vectorised aggregation in verif_metar.py.
    """
    from extremeweatherbench.utils import build_station_bilinear_map, apply_bilinear_weights

    # --- Build target at station locations ---
    t0 = time.time()
    # Use the EWB InputBase pipeline: open -> subset -> convert
    raw = target.open_and_maybe_preprocess_data_from_source()
    raw = target.subset_data_to_case(raw, case)
    target_ds = target.maybe_convert_to_dataset(raw)
    if not isinstance(target_ds, xr.Dataset) or target_ds.sizes.get("valid_time", 0) == 0:
        print("   ⚠️  Empty GHCN target — skipping station results")
        return

    # Extract actual occupied station (lat, lon) pairs from the sparse target.
    # The GHCN Dataset has dims (valid_time, latitude, longitude) but is sparse —
    # only certain (lat, lon) cells contain data.  We must inspect the data
    # itself rather than just pairing the sorted coordinate axes.
    import sparse as sp
    lat_vals = np.asarray(target_ds["latitude"].values, dtype=np.float64)
    lon_vals = np.asarray(target_ds["longitude"].values, dtype=np.float64)

    station_lats = station_lons = None
    for _vname in target_ds.data_vars:
        _da = target_ds[_vname]
        if isinstance(_da.data, sp.COO) and _da.data.ndim >= 3:
            coo = _da.data
            lat_idx, lon_idx = coo.coords[1], coo.coords[2]
            _pairs = np.column_stack([lat_vals[lat_idx], lon_vals[lon_idx]])
            _upairs = np.unique(np.round(_pairs, decimals=6), axis=0)
            station_lats, station_lons = _upairs[:, 0], _upairs[:, 1]
            break
        else:
            _vals = np.asarray(_da.values)
            if _vals.ndim == 3:
                _has = np.any(np.isfinite(_vals), axis=0)
                _li, _lj = np.where(_has)
                _pairs = np.column_stack([lat_vals[_li], lon_vals[_lj]])
                _upairs = np.unique(np.round(_pairs, decimals=6), axis=0)
                station_lats, station_lons = _upairs[:, 0], _upairs[:, 1]
                break

    if station_lats is None:
        station_lats = np.atleast_1d(lat_vals.ravel())
        station_lons = np.atleast_1d(lon_vals.ravel())

    n_stations = len(station_lats)
    print(f"   Found {n_stations} occupied stations in GHCN target")

    # Build station lookup
    station_lookup: dict[tuple[float, float], int] = {}
    for idx in range(n_stations):
        key = (round(float(station_lats[idx]), 6), round(float(station_lons[idx]), 6))
        station_lookup[key] = idx

    # Reshape target into (valid_time, station) dense arrays per variable
    variables = get_variables_for_event_type(case.event_type)
    target_valid_times = target_ds["valid_time"].values
    n_times = len(target_valid_times)

    target_arrays: dict[str, np.ndarray] = {}
    for var in variables:
        if var not in target_ds:
            continue
        da = target_ds[var]
        if isinstance(da.data, sp.COO):
            da_dense = da.data.todense()
        else:
            da_dense = np.asarray(da.values)

        out = np.full((n_times, n_stations), np.nan, dtype=np.float32)
        da_lat_vals = np.round(np.asarray(da["latitude"].values, dtype=np.float64), 6)
        da_lon_vals = np.round(np.asarray(da["longitude"].values, dtype=np.float64), 6)
        for li, lat_val in enumerate(da_lat_vals):
            for lj, lon_val in enumerate(da_lon_vals):
                key = (round(float(lat_val), 6), round(float(lon_val), 6))
                si = station_lookup.get(key)
                if si is not None and da_dense.ndim == 3:
                    out[:, si] = da_dense[:, li, lj]
        target_arrays[var] = out
    print(f"   Target reshaped: {n_stations} stations, {n_times} times ({time.time()-t0:.1f}s)")

    # --- For each forecast, interpolate to stations and aggregate per-station ---
    # Fully vectorized: load once, bilinear interp once, numpy aggregation.
    agg_rows: list[pd.DataFrame] = []

    # ── Prepare event-metric infrastructure (heat_wave / freeze only) ────────
    event_kind = case.event_type
    do_event_metrics = event_kind in ("heat_wave", "freeze")
    if do_event_metrics:
        from extremeweatherbench.metrics import (
            _load_t2m_climatology,
            _find_events_vectorized,
        )
        climo_ds = _load_t2m_climatology()
        climo_mean = climo_ds["t2m_mean"]   # (DOY, hour, latitude, longitude)
        climo_std = climo_ds["t2m_std"]
        climo_lat = np.asarray(climo_mean["latitude"].values, dtype=np.float32)
        climo_lon = np.asarray(climo_mean["longitude"].values, dtype=np.float32)
        climo_bmap = build_station_bilinear_map(
            climo_lon, climo_lat,
            station_lons.astype(np.float32), station_lats.astype(np.float32),
        )
        # Cache: (doy, hour) → mean / std at stations
        _mean_cache: dict[tuple[int, int], np.ndarray] = {}
        _std_cache: dict[tuple[int, int], np.ndarray] = {}

        def _threshold_at_stations(vt_np) -> np.ndarray:
            """Climatology threshold at each station for a single valid time.

            Heat wave: P90 = mean + 1.282 * std (Perkins & Alexander 2013).
            Freeze:    mean - 1.645 * std.
            """
            ts = pd.Timestamp(vt_np)
            doy = ts.month * 100 + ts.day
            hr = ts.hour
            key = (doy, hr)
            if key not in _mean_cache:
                m2d = climo_mean.sel(DOY=doy, hour=hr, method="nearest").values
                _mean_cache[key] = apply_bilinear_weights(
                    m2d[np.newaxis], climo_bmap,
                )[0]
                s2d = climo_std.sel(DOY=doy, hour=hr, method="nearest").values
                _std_cache[key] = apply_bilinear_weights(
                    s2d[np.newaxis], climo_bmap,
                )[0]
            if event_kind == "heat_wave":
                return _mean_cache[key] + 1.282 * _std_cache[key]
            else:
                return _mean_cache[key] - 1.645 * _std_cache[key]

        _ev_min_days = 3
        print(f"   📊 Event-metric infrastructure ready ({event_kind}, min_days={_ev_min_days}, threshold=P90)")
    # ─────────────────────────────────────────────────────────────────────────

    for fi, fc in enumerate(forecasts):
        fc_name = fc.name
        t_fc = time.time()
        print(f"   [{fi+1}/{len(forecasts)}] {fc_name}: loading...", flush=True)
        try:
            if hasattr(fc, "ds") and fc.ds is not None:
                fc_ds = fc.ds
                # Force compute if dask-backed to avoid stale handles
                if hasattr(fc_ds, "chunks") and fc_ds.chunks:
                    print(f"      Computing dask arrays for {fc_name}...", flush=True)
                    fc_ds = fc_ds.compute()
            else:
                fc_raw = fc.open_and_maybe_preprocess_data_from_source()
                fc_ds = fc.maybe_convert_to_dataset(fc_raw)
        except Exception as e:
            print(f"   ⚠️  Skipping {fc_name}: {e}", flush=True)
            continue

        if not isinstance(fc_ds, xr.Dataset):
            print(f"   ⚠️  Skipping {fc_name}: not an xr.Dataset", flush=True)
            continue

        # Check forecast has required spatial dims
        if "latitude" not in fc_ds.dims or "longitude" not in fc_ds.dims:
            print(f"   ⚠️  Skipping {fc_name}: missing lat/lon dims ({list(fc_ds.dims)})", flush=True)
            continue

        print(f"      Grid: {dict(fc_ds.sizes)}", flush=True)
        fc_lon = np.asarray(fc_ds["longitude"].values, dtype=np.float32)
        fc_lat = np.asarray(fc_ds["latitude"].values, dtype=np.float32)
        bmap = build_station_bilinear_map(
            fc_lon, fc_lat,
            station_lons.astype(np.float32), station_lats.astype(np.float32),
        )

        for var in variables:
            if var not in fc_ds or var not in target_arrays:
                continue
            obs = target_arrays[var]  # (n_times, n_stations)
            da = fc_ds[var]
            has_lead = "lead_time" in da.dims

            # EWB convention: init dimension may be called "init_time" or
            # "valid_time" (when lead_time is also present, valid_time acts
            # as the initialization time).
            if "init_time" in da.dims:
                init_dim = "init_time"
            elif has_lead and "valid_time" in da.dims:
                init_dim = "valid_time"       # EWB default convention
            else:
                init_dim = None

            if init_dim is not None and has_lead:
                # ---------- (init, lead, lat, lon) path ----------
                ordered = da.transpose(init_dim, "lead_time", "latitude", "longitude")
                print(f"      Loading {fc_name}/{var} ({dict(ordered.sizes)})...", flush=True)
                vals_4d = np.asarray(ordered.values, dtype=np.float32)
                # Bilinear interp: (init, lead, lat, lon) -> (init, lead, station)
                n_init, n_lead = vals_4d.shape[:2]
                vals_flat = vals_4d.reshape(n_init * n_lead, *vals_4d.shape[2:])
                fc_stations_flat = apply_bilinear_weights(vals_flat, bmap)
                fc_stations = fc_stations_flat.reshape(n_init, n_lead, n_stations)

                lead_times = da["lead_time"].values
                init_times = da[init_dim].values
                lead_hours = np.array([
                    int(pd.Timedelta(lt).total_seconds() / 3600) for lt in lead_times
                ], dtype=np.int32)

                print(f"      Interpolated → ({n_init} inits × {n_lead} leads × {n_stations} stations), computing errors...", flush=True)

                # Vectorized valid-time matching & error computation per lead time
                for li in range(n_lead):
                    lh = lead_hours[li]
                    # valid_times for all inits at this lead: shape (n_init,)
                    vt_all = np.array([
                        np.datetime64(pd.Timestamp(it) + pd.Timedelta(lead_times[li]))
                        for it in init_times
                    ])
                    # Find matching target indices
                    t_idx = np.searchsorted(target_valid_times, vt_all)
                    valid_mask = (t_idx < n_times)
                    valid_mask[valid_mask] &= (target_valid_times[np.clip(t_idx, 0, n_times - 1)][valid_mask] == vt_all[valid_mask])
                    n_valid = valid_mask.sum()
                    if n_valid == 0:
                        continue

                    # fc_slice: (n_valid_inits, n_stations), obs_slice: same
                    fc_slice = fc_stations[valid_mask, li, :]
                    obs_slice = obs[t_idx[valid_mask], :]
                    diff = fc_slice - obs_slice  # (n_valid_inits, n_stations)

                    # Aggregate over inits: per-station bias and rmse for this lead time
                    finite = np.isfinite(diff)
                    n_pairs = finite.sum(axis=0)  # (n_stations,)
                    has_data = n_pairs > 0

                    if not has_data.any():
                        continue

                    bias = np.where(has_data, np.nanmean(diff, axis=0), np.nan)
                    rmse = np.where(has_data, np.sqrt(np.nanmean(diff ** 2, axis=0)), np.nan)

                    si_idx = np.where(has_data)[0]
                    agg_rows.append(pd.DataFrame({
                        "station_idx": si_idx,
                        "latitude": station_lats[si_idx],
                        "longitude": station_lons[si_idx],
                        "variable": var,
                        "forecast_source": fc_name,
                        "lead_time_hours": lh,
                        "bias": bias[si_idx],
                        "rmse": rmse[si_idx],
                        "n_pairs": n_pairs[si_idx].astype(int),
                    }))

                # ── Per-station event metrics ────────────────────────────
                if do_event_metrics:
                    from extremeweatherbench.metrics import _to_daily_agg

                    # Aggregate to daily max (heat waves) or daily min (freezes).
                    _use_daily = event_kind in ("heat_wave", "freeze")
                    _daily_agg = "max" if event_kind == "heat_wave" else "min"
                    if _use_daily:
                        ev_min_steps = 3       # 3 consecutive days
                        ev_time_res_h = 24.0
                    else:
                        time_res_h = float(lead_hours[1] - lead_hours[0]) if len(lead_hours) > 1 else 6.0
                        ev_min_steps = max(1, int(np.ceil(_ev_min_days * 24.0 / time_res_h)))
                        ev_time_res_h = time_res_h

                    # Accumulators across inits, per station
                    _s_onset = np.zeros(n_stations, dtype=np.float64)
                    _s_end = np.zeros(n_stations, dtype=np.float64)
                    _s_dur = np.zeros(n_stations, dtype=np.float64)
                    _s_ptim = np.zeros(n_stations, dtype=np.float64)
                    _s_pval = np.zeros(n_stations, dtype=np.float64)
                    _s_cnt = np.zeros(n_stations, dtype=np.int32)      # total contributing inits (detected + penalty)
                    _s_detected = np.zeros(n_stations, dtype=np.int32) # inits where fc actually detected event
                    # Track obs-only events (obs detected, forecast missed)
                    _s_obs_event = np.zeros(n_stations, dtype=np.int32)
                    _s_fc_missed = np.zeros(n_stations, dtype=np.int32)

                    for ii in range(n_init):
                        # Valid-time array for this init
                        vt_arr = np.array([
                            np.datetime64(pd.Timestamp(init_times[ii]) + pd.Timedelta(lt))
                            for lt in lead_times
                        ])
                        tidx = np.searchsorted(target_valid_times, vt_arr)
                        vmask = (tidx < n_times)
                        vmask[vmask] &= (
                            target_valid_times[np.clip(tidx, 0, n_times - 1)][vmask]
                            == vt_arr[vmask]
                        )
                        n_matched = int(vmask.sum())
                        if n_matched < ev_min_steps:
                            continue

                        # Matched forecast / obs at stations: (n_matched, n_stations)
                        fc_m = fc_stations[ii][vmask]
                        obs_m = obs[tidx[vmask]]

                        # Threshold at stations for matched valid times
                        thr_m = np.stack([
                            _threshold_at_stations(vt) for vt in vt_arr[vmask]
                        ])  # (n_matched, n_stations)

                        # ── Daily aggregation (max for heat waves, min for freezes) ──
                        if _use_daily:
                            fc_m, obs_m, thr_m = _to_daily_agg(
                                fc_m, obs_m, thr_m,
                                valid_times=vt_arr[vmask],
                                agg=_daily_agg,
                            )
                            n_matched = fc_m.shape[0]
                            if n_matched < ev_min_steps:
                                continue

                        # Exceedance masks
                        valid_vals = np.isfinite(fc_m) & np.isfinite(obs_m) & np.isfinite(thr_m)
                        if event_kind == "heat_wave":
                            f_mask = (fc_m >= thr_m) & valid_vals
                            t_mask = (obs_m >= thr_m) & valid_vals
                        else:
                            f_mask = (fc_m <= thr_m) & valid_vals
                            t_mask = (obs_m <= thr_m) & valid_vals

                        # Vectorized event detection (time_axis=0)
                        f_has, f_start, f_end = _find_events_vectorized(
                            f_mask, ev_min_steps, time_axis=0,
                        )
                        t_has, t_start, t_end = _find_events_vectorized(
                            t_mask, ev_min_steps, time_axis=0,
                        )

                        # Track where obs detected an event
                        obs_only = t_has & (~f_has)
                        _s_obs_event[t_has] += 1
                        _s_fc_missed[obs_only] += 1

                        both = f_has & t_has

                        # Peak timing setup (needed for both detected & penalty)
                        n_mt = fc_m.shape[0]
                        tidx_bcast = np.arange(n_mt).reshape(n_mt, 1)
                        si_all = np.arange(n_stations)

                        t_in = (
                            (tidx_bcast >= t_start[np.newaxis, :])
                            & (tidx_bcast < t_end[np.newaxis, :])
                            & t_has[np.newaxis, :]
                        )
                        if event_kind == "heat_wave":
                            t_peak_idx = np.where(t_in, obs_m, -np.inf).argmax(axis=0)
                        else:
                            t_peak_idx = np.where(t_in, obs_m, np.inf).argmin(axis=0)

                        if both.any():
                            # Normal errors for stations where both detected
                            _onset = (f_start[both].astype(float) - t_start[both].astype(float)) * ev_time_res_h
                            _end = (f_end[both].astype(float) - t_end[both].astype(float)) * ev_time_res_h
                            _dur = (
                                (f_end[both].astype(float) - f_start[both].astype(float))
                                - (t_end[both].astype(float) - t_start[both].astype(float))
                            ) * ev_time_res_h

                            f_in = (
                                (tidx_bcast >= f_start[np.newaxis, :])
                                & (tidx_bcast < f_end[np.newaxis, :])
                                & f_has[np.newaxis, :]
                            )
                            if event_kind == "heat_wave":
                                f_peak_idx = np.where(f_in, fc_m, -np.inf).argmax(axis=0)
                            else:
                                f_peak_idx = np.where(f_in, fc_m, np.inf).argmin(axis=0)

                            _ptim = (f_peak_idx[both].astype(float) - t_peak_idx[both].astype(float)) * ev_time_res_h
                            _pval_both = fc_m[t_peak_idx[np.where(both)[0]], np.where(both)[0]] - obs_m[t_peak_idx[np.where(both)[0]], np.where(both)[0]]

                            b_idx = np.where(both)[0]
                            _s_onset[b_idx] += _onset
                            _s_end[b_idx] += _end
                            _s_dur[b_idx] += _dur
                            _s_ptim[b_idx] += _ptim
                            _s_pval[b_idx] += _pval_both
                            _s_cnt[b_idx] += 1
                            _s_detected[b_idx] += 1

                        # Penalty for stations where obs detected event but fc missed
                        if obs_only.any():
                            o_idx = np.where(obs_only)[0]
                            obs_dur = (t_end[obs_only].astype(float) - t_start[obs_only].astype(float)) * ev_time_res_h
                            _s_onset[o_idx] += obs_dur        # model was late by entire event
                            _s_end[o_idx] += (-obs_dur)       # model ended early by entire event
                            _s_dur[o_idx] += (-obs_dur)       # model predicted 0 duration
                            _s_ptim[o_idx] += obs_dur         # peak timing off by entire event
                            # Peak value: actual fc-minus-obs at observed peak time
                            _s_pval[o_idx] += fc_m[t_peak_idx[o_idx], o_idx] - obs_m[t_peak_idx[o_idx], o_idx]
                            _s_cnt[o_idx] += 1

                    has_ev = _s_cnt > 0
                    if has_ev.any():
                        ev_idx = np.where(has_ev)[0]
                        cnt = _s_cnt[ev_idx].astype(float)
                        n_det = _s_detected[ev_idx]
                        n_miss = _s_fc_missed[ev_idx]
                        agg_rows.append(pd.DataFrame({
                            "station_idx": ev_idx,
                            "latitude": station_lats[ev_idx],
                            "longitude": station_lons[ev_idx],
                            "variable": var,
                            "forecast_source": fc_name,
                            "lead_time_hours": np.nan,
                            "bias": np.nan,
                            "rmse": np.nan,
                            "n_pairs": np.nan,
                            "onset_error": _s_onset[ev_idx] / cnt,
                            "end_error": _s_end[ev_idx] / cnt,
                            "duration_error": _s_dur[ev_idx] / cnt,
                            "peak_timing_error": _s_ptim[ev_idx] / cnt,
                            "peak_value_error": _s_pval[ev_idx] / cnt,
                            "n_events": n_det,
                            "n_events_penalized": _s_cnt[ev_idx],
                            "obs_event_detected": True,
                            "fc_event_missed": n_det == 0,
                        }))
                        n_all_missed = int((n_det == 0).sum())
                        n_partial = int(((n_det > 0) & (n_miss > 0)).sum())
                        print(
                            f"      📊 Event metrics: {has_ev.sum()} stations "
                            f"({int((n_det > 0).sum())} detected, "
                            f"{n_partial} partial, "
                            f"{n_all_missed} penalty-only)",
                            flush=True,
                        )

            elif "valid_time" in da.dims and not has_lead:
                # ---------- (valid_time, lat, lon) path — no lead dim ----------
                ordered = da.transpose("valid_time", "latitude", "longitude")
                print(f"      Loading {fc_name}/{var} ({dict(ordered.sizes)})...", flush=True)
                vals = np.asarray(ordered.values, dtype=np.float32)
                fc_at_stations = apply_bilinear_weights(vals, bmap)

                fc_vt = da["valid_time"].values
                t_idx = np.searchsorted(target_valid_times, fc_vt)
                valid_mask = (t_idx < n_times) & (target_valid_times[np.clip(t_idx, 0, n_times - 1)] == fc_vt)
                if valid_mask.any():
                    diff = fc_at_stations[valid_mask] - obs[t_idx[valid_mask]]
                    finite = np.isfinite(diff)
                    n_pairs = finite.sum(axis=0)
                    has_data = n_pairs > 0
                    if has_data.any():
                        bias = np.where(has_data, np.nanmean(diff, axis=0), np.nan)
                        rmse = np.where(has_data, np.sqrt(np.nanmean(diff ** 2, axis=0)), np.nan)
                        si_idx = np.where(has_data)[0]
                        agg_rows.append(pd.DataFrame({
                            "station_idx": si_idx,
                            "latitude": station_lats[si_idx],
                            "longitude": station_lons[si_idx],
                            "variable": var,
                            "forecast_source": fc_name,
                            "lead_time_hours": 0,
                            "bias": bias[si_idx],
                            "rmse": rmse[si_idx],
                            "n_pairs": n_pairs[si_idx].astype(int),
                        }))
            else:
                print(f"      ⚠️  {fc_name}/{var}: unrecognized dims {da.dims}, skipping", flush=True)
        print(f"   → {fc_name} done ({time.time()-t_fc:.1f}s)")

    if not agg_rows:
        print("   ⚠️  No station-level results produced")
        return

    agg = pd.concat(agg_rows, ignore_index=True)
    agg.to_csv(output_file, index=False)
    print(f"   {len(agg)} aggregated station-metric rows saved ({time.time()-t0:.1f}s total)")


# ─────────────────────────────────────────────────────────────────────────────
#  Heavy Precipitation Evaluation (MRMS-based)
# ─────────────────────────────────────────────────────────────────────────────

# Precipitation thresholds for categorical metrics (in meters, matching zarr units)
PRECIP_THRESHOLDS_M = [0.00025, 0.001, 0.0025, 0.005, 0.01, 0.025]  # 0.25, 1, 2.5, 5, 10, 25 mm


# Composite event types: when one of these events is evaluated,
# also run the heavy_precip evaluation against MRMS.
COMPOSITE_EVENTS = {
    "atmospheric_river": ["heavy_precip"],
    # Future: "tropical_cyclone": ["heavy_precip"],
    # Future: "severe_convection": ["heavy_precip"],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    case_id: int,
    force: bool = False,
    clear_tc_track_cache_flag: bool = False,
    tc_lookback_days: int | None = None,
    wn2_use_mapper: bool = False,
    wn2_max_workers: int = 1,
    local_only: bool = False,
    basic_metrics_only: bool = False,
    n_jobs: int | None = None,
    debug_heat_values: bool = False,
    debug_heat_max_inits: int = 2,
    debug_heat_max_points: int = 3,
    only_model: str | None = None,
    max_lead_hours: int = 240,
    target_type: str = "era5",
    ghcn_source: str | None = None,
    cache_derived_dir: str | None = None,
):
    """Run evaluation for a 2025 case."""
    global MAX_LEAD_HOURS
    MAX_LEAD_HOURS = max_lead_hours
    print("=" * 80)
    print("ExtremeWeatherBench — 2025 Case Evaluation")
    print("=" * 80)

    # Load case metadata
    case = load_case_metadata(case_id)

    # Check existing results
    target_suffix = "_ghcn" if target_type == "ghcn" else ""
    output_file = OUTPUT_DIR / f"case_{case_id}_{case.event_type}_2025{target_suffix}_results.csv"

    if output_file.exists() and not force:
        print(f"\n✓ Results already exist: {output_file}")
        print("  Use --force to regenerate")
        results = pd.read_csv(output_file)
        print(f"\n📊 Existing results: {len(results)} rows")
        if "forecast_source" in results.columns:
            print(f"   Forecasts: {sorted(results['forecast_source'].unique())}")
        if "metric" in results.columns:
            print(f"   Metrics: {sorted(results['metric'].unique())}")
        return

    if output_file.exists() and force:
        print(f"\n⚠️  Overwriting existing results: {output_file}")

    print(f"\n📅 Case Time Range:")
    print(f"   Start: {case.start_date}")
    print(f"   End: {case.end_date}")
    print(f"   Duration: {(case.end_date - case.start_date).days} days")
    print(f"   Max lead time: {MAX_LEAD_HOURS} h ({MAX_LEAD_HOURS // 24} days)")

    if clear_tc_track_cache_flag and case.event_type == "tropical_cyclone":
        clear_tc_track_cache(case.case_id_number)

    # Load target
    target = load_target(case, target_type=target_type, ghcn_source=ghcn_source)

    # Load forecasts
    effective_lookback = None
    if case.event_type == "tropical_cyclone" and tc_lookback_days is not None:
        effective_lookback = tc_lookback_days
        print(
            f"\n⏱️  Using TC init lookback override: {effective_lookback} day(s) "
            f"(earliest init search: {case.start_date - timedelta(days=effective_lookback)})"
        )

    print(f"\n🌐 Loading local forecasts...")
    local_forecasts = load_local_forecasts(
        case,
        lookback_days=effective_lookback,
        only_model=only_model,
    )

    if local_only:
        print("\n⏭️  Skipping NOAA S3 forecasts (--local-only)")
        noaa_forecasts = []
        print("⏭️  Skipping Google Cloud forecasts (--local-only)")
        gcs_forecasts = []
    else:
        print(f"\n☁️  Loading NOAA S3 forecasts...")
        noaa_forecasts = load_noaa_forecasts(
            case,
            lookback_days=effective_lookback,
            only_model=only_model,
        )
        print(f"\n☁️  Loading Google Cloud forecasts...")
        gcs_forecasts = load_gcs_forecasts(
            case,
            lookback_days=effective_lookback,
            wn2_use_mapper=wn2_use_mapper,
            wn2_max_workers=wn2_max_workers,
            only_model=only_model,
        )

    all_forecasts = local_forecasts + noaa_forecasts + gcs_forecasts
    if not all_forecasts:
        print("\n❌ No forecasts loaded! Cannot proceed.")
        sys.exit(1)

    print(f"\n✓ Loaded {len(all_forecasts)} models total:")
    for f in all_forecasts:
        print(f"   • {f.name}")
    if case.event_type == "tropical_cyclone":
        print_tc_track_cache_status(case, all_forecasts, min_track_timesteps=5)
        print_track_csv_cache_status(all_forecasts)

    # Get metrics
    print(f"\n📊 Configuring metrics for event type: {case.event_type}")
    if basic_metrics_only:
        print("   Using fast metric subset (--basic-metrics-only)")
        metrics = get_metrics_for_event_type_fast(case.event_type)
    else:
        metrics = get_metrics_for_event_type(case.event_type)
    print(f"   Using {len(metrics)} metrics")

    # Create evaluation objects
    print(f"\n🔧 Creating evaluation objects...")
    evaluation_objects = []
    for forecast in all_forecasts:
        eval_obj = ewb.inputs.EvaluationObject(
            event_type=case.event_type,
            metric_list=metrics,
            target=target,
            forecast=forecast,
        )
        evaluation_objects.append(eval_obj)

    print(f"   Created {len(evaluation_objects)} evaluation objects")

    # Run evaluation
    print(f"\n🚀 Starting evaluation...")
    ewb_runner = ewb.evaluate.ExtremeWeatherBench(
        case_metadata=[case],
        evaluation_objects=evaluation_objects,
    )

    if n_jobs is None:
        # Non-TC runs are often I/O-bound on target reads; default to 1 to avoid
        # parallel contention and reduce "stuck at 0%" behavior.
        effective_n_jobs = 3 if case.event_type == "tropical_cyclone" else 1
    else:
        effective_n_jobs = n_jobs
    print(f"   Evaluation workers: n_jobs={effective_n_jobs}")
    eval_kwargs = dict(
        n_jobs=effective_n_jobs,
        debug_heat_values=debug_heat_values,
        debug_heat_max_inits=debug_heat_max_inits,
        debug_heat_max_points=debug_heat_max_points,
    )
    if cache_derived_dir is not None:
        eval_kwargs["cache_derived_dir"] = cache_derived_dir
        print(f"   Caching derived variables → {cache_derived_dir}")
    results = ewb_runner.run_evaluation(**eval_kwargs)

    # Save aggregate results
    results.to_csv(output_file, index=False)
    print(f"\n✅ Evaluation complete!")
    print(f"   Results saved to: {output_file}")
    print(f"   Total rows: {len(results)}")

    # --- Composite event sub-evaluations (e.g., heavy precip for AR events) ---
    composite_subs = COMPOSITE_EVENTS.get(case.event_type, [])
    if composite_subs:
        print(f"\n🔗 Composite event: running sub-evaluations for {case.event_type}")
        for sub_eval_type in composite_subs:
            if sub_eval_type == "heavy_precip":
                precip_csv = OUTPUT_DIR / f"case_{case_id}_{case.event_type}_2025_heavy_precip_results.csv"
                try:
                    import copy as _copy
                    precip_case = _copy.copy(case)
                    object.__setattr__(precip_case, "event_type", "heavy_precip")

                    precip_forecasts = load_local_forecasts(
                        precip_case, only_model=only_model,
                    )
                    precip_target = load_target(precip_case)
                    precip_metrics = get_metrics_for_event_type("heavy_precip")
                    precip_eval_objs = [
                        ewb.inputs.EvaluationObject(
                            event_type="heavy_precip",
                            metric_list=precip_metrics,
                            target=precip_target,
                            forecast=fc,
                        )
                        for fc in precip_forecasts
                    ]
                    precip_runner = ewb.evaluate.ExtremeWeatherBench(
                        case_metadata=[case],
                        evaluation_objects=precip_eval_objs,
                    )
                    precip_results = precip_runner.run_evaluation(n_jobs=1)

                    if not precip_results.empty:
                        precip_results.to_csv(precip_csv, index=False)
                        print(f"   Heavy precip results: {precip_csv}")
                        results = pd.concat([results, precip_results], ignore_index=True)
                        results.to_csv(output_file, index=False)
                        print(f"   Updated main results: {len(results)} total rows")
                except Exception as e:
                    print(f"   ⚠️  Heavy precip evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()

    # --- Per-station results (only for GHCN target) ---
    if target_type == "ghcn" and len(all_forecasts) > 0:
        station_csv = OUTPUT_DIR / f"case_{case_id}_{case.event_type}_2025_station_results.csv"
        print(f"\n📍 Computing per-station metrics for GHCN verification...")
        try:
            _compute_station_results(
                case=case,
                target=target,
                forecasts=all_forecasts,
                output_file=station_csv,
            )
            print(f"   Station results saved to: {station_csv}")
        except Exception as e:
            print(f"   ⚠️  Per-station computation failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n📈 Sample results:")
    print(results.head(10))

    if not results.empty:
        forecast_names = {f.name for f in all_forecasts}
        if "forecast_source" in results.columns:
            result_forecasts = set(results["forecast_source"].unique())
        elif "forecast_name" in results.columns:
            result_forecasts = set(results["forecast_name"].unique())
        else:
            result_forecasts = forecast_names

        missing = forecast_names - result_forecasts
        if missing:
            print(f"\n⚠️  Missing forecasts in results: {missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate forecasts for a 2025 case from ExtremeWeatherBench"
    )
    parser.add_argument(
        "--case-id", type=int, required=True,
        help="Case ID from events.yaml (e.g. 338 for Hurricane Melissa)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force regeneration even if CSV already exists",
    )
    parser.add_argument(
        "--clear-tc-track-cache",
        action="store_true",
        help=(
            "Delete case-specific TC track cache entries before running "
            "(forces track recomputation for tropical cyclone cases)"
        ),
    )
    parser.add_argument(
        "--tc-lookback-days",
        type=int,
        default=None,
        help=(
            "Override tropical-cyclone init discovery lookback window in days "
            "(applies to local, NOAA, and WeatherNext2 sources for this run only)"
        ),
    )
    parser.add_argument(
        "--wn2-use-mapper",
        action="store_true",
        help=(
            "Force WeatherNext2 GCS opens via gcsfs mapper (Colab-style) "
            "instead of direct gs:// URI open_zarr."
        ),
    )
    parser.add_argument(
        "--wn2-max-workers",
        type=int,
        default=1,
        help=(
            "Number of concurrent WeatherNext2 init downloads. Lower values "
            "often improve per-init wall time by avoiding bandwidth contention."
        ),
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help=(
            "Evaluate only local forecasts; skip models that require download "
            "from NOAA S3 or Google Cloud."
        ),
    )
    parser.add_argument(
        "--basic-metrics-only",
        action="store_true",
        help=(
            "Use only RMSE/MAE for non-TC cases (skips event-timing metrics) "
            "for faster iteration."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Override parallel workers for evaluation. Default is 1 for non-TC "
            "and 3 for tropical cyclone."
        ),
    )
    parser.add_argument(
        "--debug-heat-values",
        action="store_true",
        help=(
            "Print detailed value-level diagnostics for heat/freeze event timing "
            "metrics (ranges, masks, windows, and per-sample errors)."
        ),
    )
    parser.add_argument(
        "--debug-heat-max-inits",
        type=int,
        default=2,
        help="Maximum init indices to print in detailed heat debug output.",
    )
    parser.add_argument(
        "--debug-heat-max-points",
        type=int,
        default=3,
        help="Maximum stacked spatial sample points to print per init in heat debug output.",
    )
    parser.add_argument(
        "--only-model",
        type=str,
        default=None,
        help=(
            "Run only one model by exact display name (e.g. 'WeatherMesh-4'). "
            "Useful for focused debugging."
        ),
    )
    parser.add_argument(
        "--max-lead-hours",
        type=int,
        default=240,
        help=(
            "Maximum forecast lead time in hours to retain (default 240). "
            "Only 12-hourly steps up to this value are kept."
        ),
    )
    parser.add_argument(
        "--target-type",
        type=str,
        choices=["era5", "ghcn"],
        default="era5",
        help=(
            "Target observation source: 'era5' (gridded reanalysis, default) or "
            "'ghcn' (GHCN-H hourly station observations)."
        ),
    )
    parser.add_argument(
        "--ghcn-source",
        type=str,
        default=None,
        help=(
            "Path or GCS URI to a custom GHCN-H parquet file. "
            "Only used when --target-type=ghcn. "
            "Defaults to the built-in GCS dataset (2020-2024)."
        ),
    )
    parser.add_argument(
        "--cache-derived",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Cache derived variables (IVT, AR mask, etc.) to DIR as NetCDF "
            "for later bulk plotting. Structure: DIR/case_{id}/{source}/derived.nc"
        ),
    )

    args = parser.parse_args()

    try:
        main(
            args.case_id,
            force=args.force,
            clear_tc_track_cache_flag=args.clear_tc_track_cache,
            tc_lookback_days=args.tc_lookback_days,
            wn2_use_mapper=args.wn2_use_mapper,
            wn2_max_workers=args.wn2_max_workers,
            local_only=args.local_only,
            basic_metrics_only=args.basic_metrics_only,
            n_jobs=args.n_jobs,
            debug_heat_values=args.debug_heat_values,
            debug_heat_max_inits=args.debug_heat_max_inits,
            debug_heat_max_points=args.debug_heat_max_points,
            only_model=args.only_model,
            max_lead_hours=args.max_lead_hours,
            target_type=args.target_type,
            ghcn_source=args.ghcn_source,
            cache_derived_dir=args.cache_derived,
        )
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

