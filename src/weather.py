"""Client for the Open-Meteo weather API (free, no API key).

Fetches hourly data and aggregates to 3-hour windows to match
the LGL Bayern pollen measurement intervals.
"""

import time
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
    # Diurnal pollen-relevant parameters
    "boundary_layer_height",
    "dew_point_2m",
    "is_day",
]

# Shortwave and direct radiation and CAPE were requested, aggregated and
# stored until November 2026, and so were the window's temperature maximum
# and minimum. Nothing read them: none is a model input and none feeds a
# derived feature (`cape_max` was NaN on 99.7% of the history's rows).

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
    with the canonical columns of :data:`src.types.WEATHER_COLUMNS`.
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
    result["temperature_mean"] = grouped["temperature_2m"].mean()
    result["precipitation_sum"] = grouped["precipitation"].sum()
    result["wind_speed_max"] = grouped["wind_speed_10m"].max()
    result["humidity_mean"] = grouped["relative_humidity_2m"].mean()
    result["sunshine_duration"] = grouped["sunshine_duration"].sum()

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


# Four attempts, backing off 2s/4s/8s between them. Weather is on the critical
# path: every 3-hourly run dies if this call does.
_MAX_ATTEMPTS = 4

# Waits after a 429, in seconds: long enough to outlast a per-minute limit.
_RATE_LIMIT_DELAYS = (5, 20, 60)


def _fetch_json(url: str, params: dict[str, Any], timeout: float) -> Any:
    """GET a JSON payload, retrying transient failures with exponential backoff.

    Three kinds of failure are transient and retried: transport errors, 5xx,
    and — the one that actually bit us — a *successful* response whose body is
    not JSON. On 2026-08-29 Open-Meteo answered the archive endpoint with a 2xx
    and an empty body, which sailed straight past raise_for_status() and killed
    the run on the decode instead. The run before and after it succeeded.

    4xx is deliberately not retried: it means the request itself is wrong, so
    repeating it changes nothing, and _get_weather has a fallback ready for the
    one 4xx we expect.

    429 is the exception among the 4xx. It is Open-Meteo's rate limit — per
    minute, hour or day, counted per address, and a shared runner address is
    counted with everyone else on it — and says nothing about the request. It
    is retried on a slower schedule and, if it persists, raised as a failure
    of the service: read as a client error it used to send _get_weather into
    its "soil parameters rejected" fallback, which either died on the same
    limit or published a forecast with the soil features zeroed.
    """
    last_error: str | None = None

    for attempt in range(_MAX_ATTEMPTS):
        rate_limited = False
        try:
            response = httpx.get(url, params=params, timeout=timeout)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 429:
                rate_limited = True
                last_error = "HTTP 429 (rate limit): " + " ".join(response.text[:200].split())
            elif 400 <= response.status_code < 500:
                response.raise_for_status()
            elif response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
            else:
                try:
                    return response.json()
                except ValueError:
                    # Say what came back — a bare JSONDecodeError names only the
                    # byte offset, which tells you nothing about the response.
                    preview = " ".join(response.text[:200].split())
                    last_error = (
                        f"HTTP {response.status_code} with a non-JSON body: {preview!r}"
                    )

        if attempt < _MAX_ATTEMPTS - 1:
            delay = _RATE_LIMIT_DELAYS[attempt] if rate_limited else 2**attempt
            print(
                f"  Weather API: {last_error} — retrying in {delay}s "
                f"({attempt + 1}/{_MAX_ATTEMPTS - 1})"
            )
            time.sleep(delay)

    raise RuntimeError(f"{url} failed after {_MAX_ATTEMPTS} attempts — {last_error}")


def _get_weather(
    url: str, params: dict[str, Any], soil_params: list[str], timeout: float = 30
) -> pd.DataFrame:
    """GET Open-Meteo with soil params, falling back to base params on failure.

    Soil variable names differ between the forecast model and the ERA5 archive.
    If a soil param is ever rejected (HTTP 4xx), retry without soil rather than
    let the whole weather request — and thus the forecast — fail. The canonical
    soil columns then default to 0.0 in _parse_hourly_response.

    Transient failures are handled a level down, in _fetch_json.
    """
    try:
        payload = _fetch_json(
            url,
            {**params, "hourly": ",".join(HOURLY_PARAMS + soil_params)},
            timeout,
        )
    except httpx.HTTPStatusError:
        if not soil_params:
            raise
        print("  Weather API rejected soil params; retrying without soil features")
        payload = _fetch_json(
            url,
            {**params, "hourly": ",".join(HOURLY_PARAMS)},
            timeout,
        )
    return _parse_hourly_response(payload)


_forecast_cache: tuple[pd.DataFrame, int, int] | None = None


def fetch_weather_forecast(days: int = 5, past_days: int = 0) -> pd.DataFrame:
    """
    Fetch weather forecast from Open-Meteo at 3-hour resolution.

    *days* forecast days starting today, preceded by *past_days* days of the
    forecast model's own analysis. The archive lags real time by about five
    days, so those past days are the only weather there is for the windows
    in between.

    Results are cached in memory so that collect() and generate_forecast()
    don't make duplicate API calls within the same process: a request the
    cached one covers is cut out of it.

    Returns a DataFrame indexed by window-start datetime with weather feature columns.
    """
    global _forecast_cache
    if _forecast_cache is None or _forecast_cache[1] < days or _forecast_cache[2] < past_days:
        params: dict[str, Any] = {
            "latitude": LAT,
            "longitude": LON,
            "timezone": "Europe/Berlin",
            "forecast_days": days,
        }
        if past_days:
            params["past_days"] = past_days
        result = _get_weather(FORECAST_URL, params, FORECAST_SOIL_PARAMS, timeout=30)
        result.index = pd.DatetimeIndex(result.index)
        _forecast_cache = (result, days, past_days)

    frame, _, cached_past = _forecast_cache
    # The response starts *cached_past* days before today, on the API's own
    # (Munich) calendar; today is counted from there rather than from this
    # machine's clock, which may already be a day ahead of the response.
    today = frame.index.min().normalize() + pd.Timedelta(days=cached_past)
    start = today - pd.Timedelta(days=past_days)
    return frame[(frame.index >= start) & (frame.index < today + pd.Timedelta(days=days))]


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
