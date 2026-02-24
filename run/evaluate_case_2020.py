#!/usr/bin/env python3
"""
Evaluate forecasts for any 2020 case from ExtremeWeatherBench events.yaml

This script generalizes the 2020 case evaluation to work with any case by:
1. Loading case metadata from events.yaml based on case number
2. Dynamically selecting appropriate variables based on event type
3. Using appropriate metrics for each event type
4. Loading WeatherBench2 AI models with standardized 12-hourly temporal resolution

Usage:
    python evaluate_2020_case.py --case-id <case_number> [--force]

Options:
    --case-id: Case ID from events.yaml (required)
    --force: Force regeneration even if results CSV already exists

Example:
    python evaluate_2020_case.py --case-id 29  # 2020 Australia heatwave
    python evaluate_2020_case.py --case-id 236  # Hurricane Laura
    python evaluate_2020_case.py --case-id 236 --force  # Regenerate results
"""

import argparse
import sys
import warnings
from pathlib import Path

import extremeweatherbench as ewb
import pandas as pd
import xarray as xr
import numpy as np

warnings.filterwarnings("ignore")

# Paths
WEATHERMESH_ZARR_PATH = "/huge/proc/weathermesh4_2020.zarr"
OUTPUT_DIR = Path("./")

# WeatherBench2 variable mapping (handles variable name differences)
# NOTE: Coordinate renaming is handled in preprocess_12hourly()
ALLOWED_INIT_HOURS = (0, 12)

WEATHERBENCH2_VARIABLE_MAPPING = {
    # Variable mappings
    "2m_temperature": "surface_air_temperature",
    "mean_sea_level_pressure": "air_pressure_at_mean_sea_level",
    "geopotential": "geopotential",
    "10m_u_component_of_wind": "surface_eastward_wind",
    "10m_v_component_of_wind": "surface_northward_wind",
    "u_component_of_wind": "eastward_wind",
    "v_component_of_wind": "northward_wind",
    "temperature": "air_temperature",
    "specific_humidity": "specific_humidity",
}


def preprocess_12hourly(ds: xr.Dataset, forecast_name: str = "Unknown") -> xr.Dataset:
    """
    Preprocess forecast data to standardize temporal resolution:
    - Filter lead_time to 12h, 24h, 36h, ... 240h (exclude hour 0)
    - Rename coordinates and variables to EWB conventions
    - Maximum lead time: 240 hours (10 days)
    """
    # Filter lead times to 12-hourly, excluding hour 0, max 240h
    if "lead_time" in ds.dims:
        lead_time_hours = ds.lead_time / pd.Timedelta(hours=1)
        valid_lead_times = (lead_time_hours > 0) & (lead_time_hours % 12 == 0) & (lead_time_hours <= 240)
        ds = ds.sel(lead_time=ds.lead_time[valid_lead_times])
    elif "prediction_timedelta" in ds.dims:
        pred_hours = ds.prediction_timedelta / pd.Timedelta(hours=1)
        valid_pred_times = (pred_hours > 0) & (pred_hours % 12 == 0) & (pred_hours <= 240)
        ds = ds.sel(prediction_timedelta=ds.prediction_timedelta[valid_pred_times])
    
    # Rename coordinates and variables to match EWB conventions
    renames = {}
    if 'time' in ds.coords and 'init_time' not in ds.coords:
        renames['time'] = 'init_time'
    if 'prediction_timedelta' in ds.coords and 'lead_time' not in ds.coords:
        renames['prediction_timedelta'] = 'lead_time'
    if 'lat' in ds.coords and 'latitude' not in ds.coords:
        renames['lat'] = 'latitude'
    if 'lon' in ds.coords and 'longitude' not in ds.coords:
        renames['lon'] = 'longitude'
    
    # Variable renames
    for wb2_name, ewb_name in WEATHERBENCH2_VARIABLE_MAPPING.items():
        if wb2_name in ds.variables and ewb_name not in ds.variables:
            renames[wb2_name] = ewb_name
    
    if renames:
        ds = ds.rename(renames)
    
    # Keep lead_time as timedelta64 - the framework will handle it
    
    return ds


def preprocess_init_hours(ds: xr.Dataset, forecast_name: str = "Unknown") -> xr.Dataset:
    """Keep only common init cycles (00Z/12Z) across models."""
    if "init_time" not in ds.dims:
        # Check pre-rename name
        if "time" in ds.dims:
            dim_name = "time"
        else:
            return ds
    else:
        dim_name = "init_time"

    init_vals = pd.to_datetime(ds[dim_name].values, errors="coerce")
    keep_mask = np.array(
        [(not pd.isna(ts)) and (ts.hour in ALLOWED_INIT_HOURS) for ts in init_vals],
        dtype=bool,
    )
    before = int(len(init_vals))
    after = int(np.sum(keep_mask))
    if after < before:
        print(
            f"      → Filtering {dim_name} to 00/12Z for {forecast_name}: "
            f"{before} -> {after}"
        )
    if after == 0:
        print(f"      ⚠ No 00/12Z {dim_name} values for {forecast_name} after filtering")
        return ds.isel({dim_name: slice(0, 0)})
    return ds.sel({dim_name: ds[dim_name][keep_mask]})


def preprocess_weathermesh(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess WeatherMesh to match 12-hourly temporal resolution."""
    ds = preprocess_init_hours(ds, forecast_name="WeatherMesh-4")
    return preprocess_12hourly(ds, forecast_name="WeatherMesh-4")


def preprocess_tc_forecast(ds: xr.Dataset, forecast_name: str = "Unknown") -> xr.Dataset:
    """
    Preprocess forecast data for tropical cyclone cases.
    1. Standardizes longitude to -180 to 180 format (IBTrACS uses this convention)
    2. Calculates geopotential thickness (300-500 hPa) if available for TC tracking.
    3. If not available, adds dummy variable filled with NaN (TC tracking will skip warm-core check).
    """
    # Standardize longitude to -180 to 180 for TC tracking
    # IBTrACS uses -180 to 180, so we need to match that for spatial filtering
    if "longitude" in ds.coords:
        lon = ds.longitude
        # Check if longitude is in 0-360 format
        if float(lon.min()) >= 0 and float(lon.max()) > 180:
            print(f"   → Converting longitude from 0-360 to -180-180 for TC tracking")
            # Convert: values > 180 become negative
            ds = ds.assign_coords(longitude=(lon + 180) % 360 - 180)
            # Sort longitude to maintain monotonic order
            ds = ds.sortby("longitude")
    
    # Try to calculate geopotential thickness if we have geopotential with required levels
    if "geopotential" in ds.variables and "level" in ds.dims:
        try:
            ds["geopotential_thickness"] = ewb.calc.geopotential_thickness(
                ds["geopotential"], top_level=300, bottom_level=500
            )
        except (KeyError, ValueError):
            pass  # Will create dummy variable below
    
    # If geopotential_thickness doesn't exist, create dummy variable
    if "geopotential_thickness" not in ds.variables:
        # Check both original and mapped variable names for SLP
        template_var = None
        for var_name in ["air_pressure_at_mean_sea_level", "mean_sea_level_pressure"]:
            if var_name in ds.variables:
                template_var = ds[var_name]
                break
        
        if template_var is not None:
            ds["geopotential_thickness"] = xr.full_like(template_var, float('nan'), dtype=float)
            ds["geopotential_thickness"].attrs["note"] = "Dummy variable - TC tracking will skip warm-core check"
    
    return ds


def get_variables_for_event_type(event_type: str) -> list[str]:
    """
    Get the required variables for a given event type.
    
    Args:
        event_type: One of 'heat_wave', 'tropical_cyclone', 'atmospheric_river', etc.
    
    Returns:
        List of variable names needed for evaluation
    """
    if event_type == "heat_wave":
        return ["surface_air_temperature"]
    elif event_type == "tropical_cyclone":
        # For TC tracking: need SLP, geopotential (for thickness), and 10m winds
        return [
            "air_pressure_at_mean_sea_level",
            "geopotential",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]
    elif event_type == "atmospheric_river":
        # ARs need IVT components: moisture and winds
        return [
            "specific_humidity",
            "eastward_wind",
            "northward_wind",
        ]
    elif event_type == "severe_convection":
        # Severe weather: temperature, humidity, winds for instability
        return [
            "surface_air_temperature",
            "air_temperature",
            "specific_humidity",
            "eastward_wind",
            "northward_wind",
        ]
    else:
        # Default: just temperature
        print(f"⚠️  Unknown event type '{event_type}', using default variables")
        return ["surface_air_temperature"]


def get_metrics_for_event_type(event_type: str) -> list:
    """
    Get appropriate metrics for a given event type.
    
    Args:
        event_type: One of 'heat_wave', 'tropical_cyclone', 'atmospheric_river', etc.
    
    Returns:
        List of metric instances
    """
    if event_type == "heat_wave":
        return [
            ewb.metrics.RootMeanSquaredError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            ewb.metrics.MeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            ewb.metrics.MaximumMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            ewb.metrics.MinimumMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            ewb.metrics.DurationMeanError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
                threshold=305.0,  # ~32°C in Kelvin
            ),
        ]
    elif event_type == "tropical_cyclone":
        # TC metrics: track-based metrics for position, intensity, landfall
        return [
            ewb.metrics.RootMeanSquaredError(
                forecast_variable="air_pressure_at_mean_sea_level",
                target_variable="air_pressure_at_mean_sea_level",
            ),
            ewb.metrics.MeanAbsoluteError(
                forecast_variable="air_pressure_at_mean_sea_level",
                target_variable="air_pressure_at_mean_sea_level",
            ),
            # Landfall-specific metrics
            ewb.metrics.LandfallDisplacement(
                forecast_variable="air_pressure_at_mean_sea_level",
                target_variable="air_pressure_at_mean_sea_level",
            ),
            ewb.metrics.LandfallTimeMeanError(
                forecast_variable="air_pressure_at_mean_sea_level",
                target_variable="air_pressure_at_mean_sea_level",
            ),
            ewb.metrics.LandfallIntensityMeanAbsoluteError(
                forecast_variable="air_pressure_at_mean_sea_level",
                target_variable="air_pressure_at_mean_sea_level",
            ),
            # Track error decomposition metrics
            ewb.metrics.AlongTrackError(
                forecast_variable="air_pressure_at_mean_sea_level",
                target_variable="air_pressure_at_mean_sea_level",
            ),
            ewb.metrics.CrossTrackError(
                forecast_variable="air_pressure_at_mean_sea_level",
                target_variable="air_pressure_at_mean_sea_level",
            ),
            ewb.metrics.TotalTrackError(
                forecast_variable="air_pressure_at_mean_sea_level",
                target_variable="air_pressure_at_mean_sea_level",
            ),
        ]
    elif event_type == "atmospheric_river":
        # AR metrics: IVT-based
        return [
            ewb.metrics.RootMeanSquaredError(
                forecast_variable="specific_humidity",
                target_variable="specific_humidity",
            ),
            ewb.metrics.MeanAbsoluteError(
                forecast_variable="specific_humidity",
                target_variable="specific_humidity",
            ),
        ]
    elif event_type == "severe_convection":
        # Severe weather: field-based metrics
        return [
            ewb.metrics.RootMeanSquaredError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            ewb.metrics.MeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
        ]
    else:
        # Default: basic temperature metrics
        print(f"⚠️  Unknown event type '{event_type}', using default metrics")
        return [
            ewb.metrics.RootMeanSquaredError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            ewb.metrics.MeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
        ]


def load_case_metadata(case_id: int) -> ewb.IndividualCase:
    """Load case metadata from events.yaml."""
    print(f"Loading case {case_id} metadata...")
    
    # Load all cases from events.yaml
    all_cases = ewb.cases.load_ewb_events_yaml_into_case_list()
    
    # Filter to the requested case
    matching_cases = [c for c in all_cases if c.case_id_number == case_id]
    
    if not matching_cases:
        raise ValueError(f"Case {case_id} not found in events.yaml")
    
    case = matching_cases[0]
    
    print(f"\n📋 Case Information:")
    print(f"   Title: {case.title}")
    print(f"   Type: {case.event_type}")
    print(f"   Dates: {case.start_date} to {case.end_date}")
    print(f"   Region: {case.location}")
    
    return case


def load_target(case: ewb.IndividualCase):
    """Load target data based on event type."""
    print(f"\n🎯 Loading target data...")
    
    if case.event_type == "tropical_cyclone":
        # Use IBTrACS for tropical cyclone targets
        print("   Using IBTrACS (observed TC tracks)")
        return ewb.inputs.IBTrACS()
    else:
        # Use ERA5 for gridded targets
        print("   Using ERA5 reanalysis")
        variables = get_variables_for_event_type(case.event_type)
        return ewb.inputs.ERA5(variables=variables)


def load_weathermesh_forecast(case: ewb.IndividualCase):
    """Load WeatherMesh-4 forecast."""
    print(f"\n🌐 Loading WeatherMesh-4 forecast...")
    
    variables = get_variables_for_event_type(case.event_type)
    
    # For TC cases, add derived variables for track detection
    preprocessing_func = preprocess_weathermesh
    
    if case.event_type == "tropical_cyclone":
        print("   → Adding TC-specific preprocessing and derived variables")
        # Chain preprocessing: 12-hourly + TC thickness calculation
        def preprocess_tc_weathermesh(ds):
            ds = preprocess_weathermesh(ds)
            ds = preprocess_tc_forecast(ds, forecast_name="WeatherMesh-4")
            return ds
        preprocessing_func = preprocess_tc_weathermesh
        
        # Add TC track detection as derived variable to the variables list
        # Note: min_track_timesteps=5 means we need 5 consecutive 12-hourly timesteps (60 hours)
        variables = variables + [
            ewb.derived.TropicalCycloneTrackVariables(min_track_timesteps=5)
        ]
    
    forecast = ewb.inputs.ZarrForecast(
        name="WeatherMesh-4",
        source=WEATHERMESH_ZARR_PATH,
        variables=variables,
        preprocess=preprocessing_func,
    )
    
    return forecast


def validate_forecast_data(forecast_source: str, storage_options: dict = None) -> tuple[bool, str]:
    """
    Validate that a forecast data source has accessible data.
    Returns (is_valid, error_message).
    """
    try:
        import xarray as xr
        print(f"    Checking {forecast_source}...")
        ds = xr.open_zarr(forecast_source, storage_options=storage_options)
        
        # Check if dataset has data
        if len(ds.dims) == 0:
            return False, "Dataset has no dimensions"
        
        # Check for time-related dimensions
        time_coords = [c for c in ds.coords if 'time' in c.lower() or 'lead' in c.lower()]
        if not time_coords:
            return False, f"No time coordinates found. Available: {list(ds.coords)}"
        
        print(f"      ✓ Accessible, dims: {dict(ds.dims)}")
        return True, ""
        
    except Exception as e:
        return False, str(e)


def load_benchmark_forecasts(case: ewb.IndividualCase) -> list:
    """Load benchmark AI model forecasts from WeatherBench2."""
    print(f"\n🤖 Loading benchmark forecasts from WeatherBench2...")
    
    benchmarks = []
    variables = get_variables_for_event_type(case.event_type)
    
    # Determine preprocessing and add derived variables based on event type
    if case.event_type == "tropical_cyclone":
        def preprocess_combined(ds, name="Unknown"):
            ds = preprocess_init_hours(ds, forecast_name=name)
            ds = preprocess_12hourly(ds, forecast_name=name)
            ds = preprocess_tc_forecast(ds, forecast_name=name)
            return ds
        
        # Create preprocessing functions with forecast names
        def make_preprocess_func(forecast_name):
            return lambda ds: preprocess_combined(ds, name=forecast_name)
        
        # Add TC track detection to variables list
        # Note: min_track_timesteps=5 means we need 5 consecutive 12-hourly timesteps (60 hours)
        variables = variables + [
            ewb.derived.TropicalCycloneTrackVariables(min_track_timesteps=5)
        ]
    else:
        def make_preprocess_func(forecast_name):
            def _preprocess(ds):
                ds = preprocess_init_hours(ds, forecast_name=forecast_name)
                ds = preprocess_12hourly(ds, forecast_name=forecast_name)
                return ds
            return _preprocess
    
    # HRES (ECMWF high-resolution deterministic)
    print("   → HRES")
    try:
        preprocessing_func = make_preprocess_func("HRES")
        benchmarks.append(
            ewb.inputs.ZarrForecast(
                name="HRES",
                source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
                variables=variables,
                preprocess=preprocessing_func,
                storage_options={"remote_options": {"anon": True}},
            )
        )
    except Exception as e:
        print(f"      ✗ Failed to load HRES: {e}")
    
    # GraphCast
    print("   → GraphCast")
    try:
        preprocessing_func = make_preprocess_func("GraphCast")
        benchmarks.append(
            ewb.inputs.ZarrForecast(
                name="GraphCast",
                source="gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours.zarr",
                variables=variables,
                preprocess=preprocessing_func,
                storage_options={"remote_options": {"anon": True}},
            )
        )
    except Exception as e:
        print(f"      ✗ Failed to load GraphCast: {e}")
    
    # Pangu-Weather
    print("   → Pangu-Weather")
    try:
        preprocessing_func = make_preprocess_func("Pangu-Weather")
        benchmarks.append(
            ewb.inputs.ZarrForecast(
                name="Pangu-Weather",
                source="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
                variables=variables,
                preprocess=preprocessing_func,
                storage_options={"remote_options": {"anon": True}},
            )
        )
    except Exception as e:
        print(f"      ✗ Failed to load Pangu-Weather: {e}")
    
    # GenCast (ensemble mean)
    print("   → GenCast")
    try:
        preprocessing_func = make_preprocess_func("GenCast")
        benchmarks.append(
            ewb.inputs.ZarrForecast(
                name="GenCast",
                source="gs://weatherbench2/datasets/gencast/2020-1440x721_mean.zarr",
                variables=variables,
                preprocess=preprocessing_func,
                storage_options={"remote_options": {"anon": True}},
            )
        )
    except Exception as e:
        print(f"      ✗ Failed to load GenCast: {e}")
    print(f"   ✓ Loaded {len(benchmarks)} benchmark models")
    return benchmarks


def main(case_id: int, force: bool = False):
    """Main evaluation function."""
    print("=" * 80)
    print(f"ExtremeWeatherBench - 2020 Case Evaluation")
    print("=" * 80)
    
    # Load case metadata first to determine output filename
    case = load_case_metadata(case_id)
    
    # Check if results already exist
    output_file = OUTPUT_DIR / f"case_{case_id}_{case.event_type}_results.csv"
    
    if output_file.exists() and not force:
        print(f"\n✓ Results already exist: {output_file}")
        print(f"  Use --force to regenerate")
        print(f"  Loading existing results...")
        
        # Load and display summary
        import pandas as pd
        results = pd.read_csv(output_file)
        print(f"\n📊 Existing results summary:")
        print(f"   Total rows: {len(results)}")
        if 'forecast_source' in results.columns:
            print(f"   Forecasts: {sorted(results['forecast_source'].unique())}")
        if 'metric' in results.columns:
            print(f"   Metrics: {sorted(results['metric'].unique())}")
        return
    
    if output_file.exists() and force:
        print(f"\n⚠️  Overwriting existing results: {output_file}")
    
    # Continue with evaluation
    print(f"\n📅 Case Time Range:")
    print(f"   Start: {case.start_date}")
    print(f"   End: {case.end_date}")
    print(f"   Duration: {(case.end_date - case.start_date).days} days")
    
    # Load target
    target = load_target(case)
    
    # Load forecasts
    weathermesh = load_weathermesh_forecast(case)
    benchmarks = load_benchmark_forecasts(case)
    
    # Get metrics for this event type
    print(f"\n📊 Configuring metrics for event type: {case.event_type}")
    metrics = get_metrics_for_event_type(case.event_type)
    print(f"   Using {len(metrics)} metrics")
    
    # Combine all forecasts
    all_forecasts = [weathermesh] + benchmarks
    
    # Create evaluation objects (one per forecast)
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
    
    # Run evaluation using ExtremeWeatherBench class
    print(f"\n🚀 Starting evaluation...")
    ewb_runner = ewb.evaluate.ExtremeWeatherBench(
        case_metadata=[case],
        evaluation_objects=evaluation_objects,
    )
    
    results = ewb_runner.run_evaluation(
        n_jobs=6,
    )
    
    # Save results (overwrite if force=True)
    results.to_csv(output_file, index=False)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Results saved to: {output_file}")
    print(f"   Total rows: {len(results)}")
    
    # Show sample results
    print(f"\n📈 Sample results:")
    print(results.head(10))
    
    # Check for missing forecasts
    if not results.empty:
        forecast_names = {f.name for f in all_forecasts}
        # Check what column name is actually used
        if "forecast_name" in results.columns:
            result_forecasts = set(results["forecast_name"].unique())
        elif "forecast" in results.columns:
            result_forecasts = set(results["forecast"].unique())
        else:
            print(f"\n📋 Results columns: {list(results.columns)}")
            result_forecasts = forecast_names  # Skip check if column not found
        
        missing = forecast_names - result_forecasts
        
        if missing:
            print(f"\n⚠️  WARNING: Missing forecasts in results: {missing}")
            print(f"   These forecasts may have failed during evaluation!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate forecasts for a 2020 case from ExtremeWeatherBench"
    )
    parser.add_argument(
        "--case-id",
        type=int,
        required=True,
        help="Case ID from events.yaml (e.g., 29 for Australia heatwave, 236 for Hurricane Laura)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of results even if CSV already exists",
    )
    
    args = parser.parse_args()
    
    try:
        main(args.case_id, force=args.force)
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

