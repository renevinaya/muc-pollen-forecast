"""
Optional client for the Copernicus CAMS European air-quality pollen forecast.

CAMS runs a physics-based ensemble that forecasts a handful of pollen species
across Europe (alder, birch, grass, mugwort, ragweed). Used as an extra model
feature it injects long-range transport and season-onset signal that a purely
local statistical model can't see.

This integration is **optional and fail-open**, mirroring the NDVI client: if
the heavy dependencies (``cdsapi``/``xarray``) aren't installed, no Atmosphere
Data Store (ADS) credentials are configured, or any error occurs, every fetch
returns an empty DataFrame. The model then treats ``cams_pollen`` as 0 and is
completely unaffected — so production never breaks when CAMS is inactive.

To activate:
  1. ``uv sync --extra cams``  (installs cdsapi, xarray, netCDF4)
  2. Set ``CAMS_ADS_URL`` and ``CAMS_ADS_KEY`` (ADS credentials), or configure
     ``~/.cdsapirc``.
  3. Backfill historical CAMS into history.csv and retrain so the feature is
     populated for training, then it becomes live at inference automatically.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date

import pandas as pd

from .types import LAT, LON, FORECAST_DAYS

CAMS_DATASET = "cams-europe-air-quality-forecasts"

# CAMS variable name → our canonical species name. CAMS covers only these.
CAMS_SPECIES_MAP: dict[str, str] = {
    "alder_pollen": "Alnus",
    "birch_pollen": "Betula",
    "grass_pollen": "Poaceae",
    "mugwort_pollen": "Artemisia",
    "ragweed_pollen": "Ambrosia",
}


def _ads_client():
    """Return a configured cdsapi client, or None if CAMS is not available."""
    try:
        import cdsapi  # type: ignore[import-untyped]
    except ImportError:
        return None

    url = os.environ.get("CAMS_ADS_URL")
    key = os.environ.get("CAMS_ADS_KEY")
    try:
        if url and key:
            return cdsapi.Client(url=url, key=key)
        # Fall back to ~/.cdsapirc if present; otherwise cdsapi raises.
        return cdsapi.Client()
    except Exception:
        return None


def _download_and_parse(client, start: date, days: int) -> pd.DataFrame:
    """Download a CAMS pollen forecast and parse it to 3h windows × species."""
    import xarray as xr  # type: ignore[import-untyped]

    variables = list(CAMS_SPECIES_MAP.keys())
    leadtimes = [str(h) for h in range(0, days * 24)]
    # Small box around Munich: [North, West, South, East]
    area = [LAT + 0.1, LON - 0.1, LAT - 0.1, LON + 0.1]

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        target = tmp.name
    client.retrieve(
        CAMS_DATASET,
        {
            "variable": variables,
            "model": "ensemble",
            "level": "0",
            "type": "forecast",
            "time": "00:00",
            "leadtime_hour": leadtimes,
            "date": f"{start.isoformat()}/{start.isoformat()}",
            "format": "netcdf",
            "area": area,
        },
        target,
    )

    ds = xr.open_dataset(target)
    # Nearest grid point to Munich.
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    point = ds.sel({lat_name: LAT, lon_name: LON}, method="nearest")

    # Build an hourly series per CAMS variable, keyed by valid time.
    base = pd.Timestamp(start)
    cols: dict[str, pd.Series] = {}
    for cams_var, species in CAMS_SPECIES_MAP.items():
        if cams_var not in point:
            continue
        series = point[cams_var].to_series()
        # Index is leadtime (hours); convert to absolute timestamps.
        values = series.to_numpy().ravel()
        times = [base + pd.Timedelta(hours=int(h)) for h in range(len(values))]
        cols[species] = pd.Series(values, index=pd.DatetimeIndex(times))

    ds.close()
    try:
        os.unlink(target)
    except OSError:
        pass

    if not cols:
        return pd.DataFrame()

    hourly = pd.DataFrame(cols)
    # Aggregate to 3h windows (mean concentration) to match the weather windows.
    windows = hourly.groupby(hourly.index.floor("3h")).mean()
    return windows


def fetch_cams_forecast(days: int = FORECAST_DAYS) -> pd.DataFrame:
    """Fetch the CAMS pollen forecast as a 3h-window × species DataFrame.

    Fail-open: returns an empty DataFrame whenever CAMS is unavailable
    (missing deps, no credentials, or any download/parse error).
    """
    client = _ads_client()
    if client is None:
        return pd.DataFrame()
    try:
        df = _download_and_parse(client, date.today(), days)
        if not df.empty:
            print(f"  CAMS: {len(df)} windows, species: {list(df.columns)}")
        return df
    except Exception as exc:  # pragma: no cover - network/deps dependent
        print(f"  CAMS fetch failed ({exc}); CAMS feature will be 0")
        return pd.DataFrame()


def cams_value(cams_df: pd.DataFrame, dt: pd.Timestamp, species: str) -> float:
    """Look up the CAMS pollen value for a (window, species), or 0.0 if absent."""
    if cams_df is None or cams_df.empty or species not in cams_df.columns:
        return 0.0
    if dt not in cams_df.index:
        return 0.0
    val = cams_df.at[dt, species]
    if pd.isna(val):
        return 0.0
    return float(val)
