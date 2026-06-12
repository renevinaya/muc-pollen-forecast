"""Client for the Open-Meteo weather API (free, no API key).

Fetches hourly data and aggregates to 3-hour windows to match
the LGL Bayern pollen measurement intervals.
"""

from datetime import date
from typing import Any

import httpx
import numpy as np
import pandas as pd

from .types import LAT, LON

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_PARAMS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "sunshine_duration",
    "shortwave_radiation",
    # Diurnal pollen-relevant parameters
    "boundary_layer_height",
    "dew_point_2m",
    "cape",
    "direct_radiation",
    "is_day",
]

# Soil parameters — root-zone temperature/moisture drive grass & herbaceous
# flowering onset better than air temperature alone. The forecast model and the
# ERA5 archive expose soil under *different* variable names, so we request the
# endpoint-appropriate names and normalise both to the canonical columns
# ``soil_temperature_mean`` / ``soil_moisture_mean`` in _parse_hourly_response.
FORECAST_SOIL_PARAMS = ["soil_temperature_6cm", "soil_moisture_3_to_9cm"]
ARCHIVE_SOIL_PARAMS = ["soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm"]

# Candidate source columns for each canonical soil feature (first present wins).
_SOIL_TEMP_SOURCES = ["soil_temperature_6cm", "soil_temperature_0_to_7cm"]
_SOIL_MOISTURE_SOURCES = ["soil_moisture_3_to_9cm", "soil_moisture_0_to_7cm"]


def _parse_hourly_response(data: dict[str, Any]) -> pd.DataFrame:
    """Parse an Open-Meteo hourly response and aggregate to 3-hour windows.

    Returns a DataFrame indexed by window-start datetime (naive, local time)
    with the same canonical feature columns as before:
    temperature_max, temperature_min, temperature_mean, precipitation_sum,
    wind_speed_max, humidity_mean, sunshine_duration, shortwave_radiation_sum.
    """
    hourly = data["hourly"]
    times = pd.to_datetime(hourly["time"])

    df = pd.DataFrame({"datetime": times})
    # Pull every hourly variable the response actually contains (base params
    # plus whichever endpoint-specific soil params were requested).
    for param, values in hourly.items():
        if param == "time":
            continue
        df[param] = values

    # Floor to 3h window boundaries
    df["window"] = df["datetime"].dt.floor("3h")
    grouped = df.groupby("window")

    result = pd.DataFrame(index=sorted(grouped.groups.keys(), key=str))
    result["temperature_max"] = grouped["temperature_2m"].max()
    result["temperature_min"] = grouped["temperature_2m"].min()
    result["temperature_mean"] = grouped["temperature_2m"].mean()
    result["precipitation_sum"] = grouped["precipitation"].sum()
    result["wind_speed_max"] = grouped["wind_speed_10m"].max()
    result["humidity_mean"] = grouped["relative_humidity_2m"].mean()
    result["sunshine_duration"] = grouped["sunshine_duration"].sum()
    result["shortwave_radiation_sum"] = grouped["shortwave_radiation"].sum()

    # Wind direction: circular mean (can't just average degrees)
    dir_rad = np.radians(df["wind_direction_10m"].fillna(0).astype(float))
    df["_wd_sin"] = np.sin(dir_rad)
    df["_wd_cos"] = np.cos(dir_rad)
    grouped_wd = df.groupby("window")
    result["wind_direction"] = np.degrees(
        np.arctan2(grouped_wd["_wd_sin"].mean(), grouped_wd["_wd_cos"].mean())
    ) % 360

    # Diurnal weather features (aggregated to 3h windows)
    result["boundary_layer_height"] = grouped["boundary_layer_height"].mean()
    result["dew_point_mean"] = grouped["dew_point_2m"].mean()
    result["cape_max"] = grouped["cape"].max()
    result["direct_radiation_sum"] = grouped["direct_radiation"].sum()
    result["is_day"] = grouped["is_day"].max()  # 1.0 if any hour in window is daytime

    # Finer resolution features: slopes within each 3h window
    # These capture rapid changes (warming ramp, drying) that aggregated stats miss.
    # Slope = last hour value - first hour value within the window.
    result["temp_slope_3h"] = grouped["temperature_2m"].last() - grouped["temperature_2m"].first()
    result["humidity_slope_3h"] = (
        grouped["relative_humidity_2m"].last() - grouped["relative_humidity_2m"].first()
    )
    # Temperature variance within window (high variance = changing conditions)
    result["temp_variance_3h"] = grouped["temperature_2m"].var().fillna(0)

    # Soil features — normalise the endpoint-specific source column to a
    # canonical name. Default to 0.0 when the soil variable is unavailable so
    # downstream code never sees NaN (consistent with the rest of the pipeline).
    soil_temp_src = next((c for c in _SOIL_TEMP_SOURCES if c in df.columns), None)
    result["soil_temperature_mean"] = (
        grouped[soil_temp_src].mean() if soil_temp_src else 0.0
    )
    soil_moist_src = next((c for c in _SOIL_MOISTURE_SOURCES if c in df.columns), None)
    result["soil_moisture_mean"] = (
        grouped[soil_moist_src].mean() if soil_moist_src else 0.0
    )
    result["soil_temperature_mean"] = result["soil_temperature_mean"].fillna(0.0)
    result["soil_moisture_mean"] = result["soil_moisture_mean"].fillna(0.0)

    result.index.name = None
    return result


def _get_weather(
    url: str, params: dict[str, Any], soil_params: list[str], timeout: float = 30
) -> pd.DataFrame:
    """GET Open-Meteo with soil params, falling back to base params on failure.

    Soil variable names differ between the forecast model and the ERA5 archive.
    If a soil param is ever rejected (HTTP 4xx), retry without soil rather than
    let the whole weather request — and thus the forecast — fail. The canonical
    soil columns then default to 0.0 in _parse_hourly_response.
    """
    try:
        response = httpx.get(
            url,
            params={**params, "hourly": ",".join(HOURLY_PARAMS + soil_params)},
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError:
        if not soil_params:
            raise
        print("  Weather API rejected soil params; retrying without soil features")
        response = httpx.get(
            url,
            params={**params, "hourly": ",".join(HOURLY_PARAMS)},
            timeout=timeout,
        )
        response.raise_for_status()
    return _parse_hourly_response(response.json())


_forecast_cache: tuple[pd.DataFrame, int] | None = None


def fetch_weather_forecast(days: int = 5) -> pd.DataFrame:
    """
    Fetch weather forecast from Open-Meteo at 3-hour resolution.

    Results are cached in memory so that collect() and generate_forecast()
    don't make duplicate API calls within the same process.

    Returns a DataFrame indexed by window-start datetime with weather feature columns.
    """
    global _forecast_cache
    if _forecast_cache is not None:
        cached_df, cached_days = _forecast_cache
        if cached_days >= days:
            cutoff = cached_df.index.min() + pd.Timedelta(days=days)
            return cached_df[cached_df.index < cutoff]

    result = _get_weather(
        FORECAST_URL,
        {
            "latitude": LAT,
            "longitude": LON,
            "timezone": "Europe/Berlin",
            "forecast_days": days,
        },
        FORECAST_SOIL_PARAMS,
        timeout=30,
    )
    _forecast_cache = (result, days)
    return result


def fetch_historical_weather(start: date, end: date) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo archive at 3-hour resolution.

    The archive API has data up to ~5 days ago.
    Returns a DataFrame indexed by window-start datetime with weather feature columns.
    """
    return _get_weather(
        HISTORICAL_URL,
        {
            "latitude": LAT,
            "longitude": LON,
            "timezone": "Europe/Berlin",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        ARCHIVE_SOIL_PARAMS,
        timeout=60,
    )
