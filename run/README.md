# run/ — Evaluation and Analysis Scripts

Scripts for evaluating weather forecast models against observations using ExtremeWeatherBench cases defined in `src/extremeweatherbench/data/events.yaml`.

## Prerequisites

- Conda environment `main` activated: `conda activate main`
- Access to `/huge/proc/met-data/zarr/` (local forecast archives)
- For GHCN station verification (2025+ cases): parquet files in `/huge/users/larissa/ExtremeWeatherBench/data_prep/`

## Workflow

The typical workflow is:

1. **Evaluate** a case → produces a results CSV
2. **Analyze** the CSV → produces plots and summary tables
3. Optionally use specialized plotting scripts for deeper investigation

---

## Evaluation Scripts

### `evaluate_case_2025.py`

Evaluates forecast models against observations for cases from 2025 onward. Pulls forecasts from three sources: local zarr archives, NOAA S3, and WeatherNext2 on GCS. Results (and per-init forecast data) are cached locally so subsequent runs are fast.

**Models evaluated:**
- **Local:** WeatherMesh-4, WeatherMesh-4p5-Ens-Mean, WeatherMesh-5c-Ens-Mean, IFS-Ens-Mean, AIFS-Ens-Mean, GFS-Ens-Mean
- **NOAA S3:** FourCastNet-v2, GraphCast, Pangu-Weather, Aurora (all IFS-initialized)
- **GCS:** WeatherNext2

Edit the `LOCAL_MODELS`, `NOAA_MODELS`, and `GCS_MODELS` lists at the top of the file to enable/disable models.

**Basic usage:**
```bash
# ERA5 gridded verification (default)
python evaluate_case_2025.py --case-id 341

# GHCN station verification with extended lead time
python evaluate_case_2025.py --case-id 341 --target-type ghcn \
    --ghcn-source /huge/users/larissa/ExtremeWeatherBench/data_prep/ghcnh_all_2026.parq \
    --max-lead-hours 360
```

**Key flags:**
| Flag | Description |
|------|-------------|
| `--case-id` | Case ID from `events.yaml` (required) |
| `--target-type` | `era5` (default) or `ghcn` for station obs |
| `--ghcn-source` | Path to GHCN parquet file (needed for 2025+ cases) |
| `--max-lead-hours` | Max lead time in hours (default: 240) |
| `--force` | Regenerate even if output CSV exists |
| `--local-only` | Skip NOAA S3 and GCS models |
| `--only-model` | Run a single model by name |
| `--basic-metrics-only` | RMSE/MAE only, skip event-timing metrics |

**Output:** `case_{id}_{event_type}_{year}_{target}_results.csv` (and `*_station_results.csv` for GHCN)

**Caching:** Downloaded forecasts are cached under `/huge/proc/larissa/`. If you increase `--max-lead-hours` beyond what was previously cached, the script will automatically fetch only the missing lead times and extend the cache.

### `evaluate_case_2020.py`

Evaluates 2020 cases (e.g., Hurricane Laura, case 236). Uses WeatherBench2 AI models from GCS and local WeatherMesh-4 zarr data.

```bash
python evaluate_case_2020.py --case-id 236
python evaluate_case_2020.py --case-id 29 --force
```

**Output:** Same CSV format as the 2025 script.

---

## Analysis & Plotting

### `analyze_results.py`

The main analysis script. Takes a results CSV and produces a full suite of plots: metric vs lead time, metric vs init time, metric vs valid time, station maps, and model comparison tables.

```bash
python analyze_results.py case_341_freeze_2025_ghcn_results.csv
```

**Key flags:**
| Flag | Description |
|------|-------------|
| `--by-valid-only` | Only generate valid-time plots |
| `--by-landfall-relative-only` | Only generate landfall-relative panel plots (TC cases) |

**Output:** A `plots_{csv_stem}/` directory with PNG files for each metric and plot type.

**Filtering:** The script automatically filters out initialization times where any model is missing data, ensuring apples-to-apples comparisons. For event-based metrics (onset error, duration error, etc.), init times where no event is forecast are excluded from plots.

**Model styling:** WeatherMesh models have fixed colors (5c=black, 4p5=dark gray, 4=medium gray). Physical models (IFS, GFS) use the Pastel2 palette; AI models use the seaborn deep palette.

### `plot_heatmaps.py`

Produces forecast-hour vs init-time heatmaps for non-event-based metrics (RMSE, MAE, bias). Events appear as diagonal stripes. Generates both raw-value heatmaps and difference-from-reference heatmaps.

```bash
python plot_heatmaps.py case_341_freeze_2025_ghcn_results.csv
python plot_heatmaps.py results.csv --reference-model IFS-Ens-Mean --diff-percentile 95
```

**Key flags:**
| Flag | Description |
|------|-------------|
| `--reference-model` | Baseline for difference plots (default: IFS-Ens-Mean) |
| `--output-dir` | Output directory (default: `plots_heatmap/` next to CSV) |
| `--metrics` | Specific metrics to plot (default: auto-detect) |
| `--diff-percentile` | Colorbar clipping percentile (default: 90) |

### `plot_station_forecast.py`

Plots a single station's forecast time series vs GHCN observations for debugging event detection. Shows the raw temperature forecast, observed values, climatological thresholds, and detected event windows.

```bash
python plot_station_forecast.py 341 \
    --station 42 \
    --model WeatherMesh-5c-Ens-Mean IFS-Ens-Mean \
    --init 2026012000
```

**Required args:**
| Arg | Description |
|-----|-------------|
| `case_id` | Case ID from `events.yaml` |
| `--station` | Station index (from station_results CSV) or `lat,lon` |
| `--model` | One or more model names |
| `--init` | Init time as `YYYYMMDDHH` |

**Optional:** `--ghcn-source`, `--output`, `--kelvin`

### `plot_hurricane_tracks.py`

Plots tropical cyclone forecast tracks against IBTrACS best-track observations. Supports both 2020 and 2025+ cases. Optionally generates SLP contour PDFs.

```bash
python plot_hurricane_tracks.py 338
python plot_hurricane_tracks.py 236 --slp-pdfs
```

**Key flags:**
| Flag | Description |
|------|-------------|
| `case_id` | Case ID (positional) |
| `--tracks-dir` | Directory with `forecast_tracks_*.csv` files |
| `--output-dir` | Output directory |
| `--slp-pdfs` | Generate SLP contour PDFs (slow) |

---

## Example End-to-End

```bash
# 1. Evaluate case 341 (Jan 2026 freeze) against GHCN stations, 15-day lead
python evaluate_case_2025.py --case-id 341 \
    --target-type ghcn \
    --ghcn-source /huge/users/larissa/ExtremeWeatherBench/data_prep/ghcnh_all_2026.parq \
    --max-lead-hours 360

# 2. Generate all analysis plots
python analyze_results.py case_341_freeze_2025_ghcn_results.csv

# 3. Generate heatmaps
python plot_heatmaps.py case_341_freeze_2025_ghcn_results.csv

# 4. Debug a specific station forecast
python plot_station_forecast.py 341 \
    --station 42 \
    --model WeatherMesh-5c-Ens-Mean IFS-Ens-Mean \
    --init 2026012000
```
