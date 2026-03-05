"""Client for the Open-Meteo weather API (free, no API key).

Fetches hourly data and aggregates to 3-hour windows to match
the LGL Bayern pollen measurement intervals.
"""

from datetime import date

import httpx
import pandas as pd

from .types import LAT, LON

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_PARAMS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
    "sunshine_duration",
    "shortwave_radiation",
]


def _parse_hourly_response(data: dict) -> pd.DataFrame:
    """Parse an Open-Meteo hourly response and aggregate to 3-hour windows.

    Returns a DataFrame indexed by window-start datetime (naive, local time)
    with the same canonical feature columns as before:
    temperature_max, temperature_min, temperature_mean, precipitation_sum,
    wind_speed_max, humidity_mean, sunshine_duration, shortwave_radiation_sum.
    """
    hourly = data["hourly"]
    times = pd.to_datetime(hourly["time"])

    df = pd.DataFrame({"datetime": times})
    for param in HOURLY_PARAMS:
        df[param] = hourly.get(param, [None] * len(times))

    # Floor to 3h window boundaries
    df["window"] = df["datetime"].dt.floor("3h")
    grouped = df.groupby("window")

    result = pd.DataFrame(index=sorted(grouped.groups.keys()))
    result["temperature_max"] = grouped["temperature_2m"].max()
    result["temperature_min"] = grouped["temperature_2m"].min()
    result["temperature_mean"] = grouped["temperature_2m"].mean()
    result["precipitation_sum"] = grouped["precipitation"].sum()
    result["wind_speed_max"] = grouped["wind_speed_10m"].max()
    result["humidity_mean"] = grouped["relative_humidity_2m"].mean()
    result["sunshine_duration"] = grouped["sunshine_duration"].sum()
    result["shortwave_radiation_sum"] = grouped["shortwave_radiation"].sum()
    result.index.name = None
    return result


def fetch_weather_forecast(days: int = 5) -> pd.DataFrame:
    """
    Fetch weather forecast from Open-Meteo at 3-hour resolution.

    Returns a DataFrame indexed by window-start datetime with weather feature columns.
    """
    response = httpx.get(
        FORECAST_URL,
        params={
            "latitude": LAT,
            "longitude": LON,
            "hourly": ",".join(HOURLY_PARAMS),
            "timezone": "Europe/Berlin",
            "forecast_days": days,
        },
        timeout=30,
    )
    response.raise_for_status()
    return _parse_hourly_response(response.json())


def fetch_historical_weather(start: date, end: date) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo archive at 3-hour resolution.

    The archive API has data up to ~5 days ago.
    Returns a DataFrame indexed by window-start datetime with weather feature columns.
    """
    response = httpx.get(
        HISTORICAL_URL,
        params={
            "latitude": LAT,
            "longitude": LON,
            "hourly": ",".join(HOURLY_PARAMS),
            "timezone": "Europe/Berlin",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=60,
    )
    response.raise_for_status()
    return _parse_hourly_response(response.json())
