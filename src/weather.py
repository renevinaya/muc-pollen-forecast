"""Client for the Open-Meteo weather API (free, no API key)."""

from datetime import date, timedelta

import httpx
import pandas as pd

from .types import LAT, LON

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_PARAMS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "sunshine_duration",
    "shortwave_radiation_sum",
]

# Mapping from Open-Meteo field names to our canonical feature names
FIELD_MAP = {
    "temperature_2m_max": "temperature_max",
    "temperature_2m_min": "temperature_min",
    "temperature_2m_mean": "temperature_mean",
    "precipitation_sum": "precipitation_sum",
    "wind_speed_10m_max": "wind_speed_max",
    "relative_humidity_2m_mean": "humidity_mean",
    "sunshine_duration": "sunshine_duration",
    "shortwave_radiation_sum": "shortwave_radiation_sum",
}


def _parse_daily_response(data: dict) -> pd.DataFrame:
    """Parse an Open-Meteo daily response into a DataFrame."""
    daily = data["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for api_field, our_field in FIELD_MAP.items():
        df[our_field] = daily.get(api_field, [None] * len(daily["time"]))
    df = df.set_index("date")
    return df


def fetch_weather_forecast(days: int = 5) -> pd.DataFrame:
    """
    Fetch weather forecast from Open-Meteo.

    Returns a DataFrame indexed by date with weather feature columns.
    """
    response = httpx.get(
        FORECAST_URL,
        params={
            "latitude": LAT,
            "longitude": LON,
            "daily": ",".join(DAILY_PARAMS),
            "timezone": "Europe/Berlin",
            "forecast_days": days,
        },
        timeout=30,
    )
    response.raise_for_status()
    return _parse_daily_response(response.json())


def fetch_historical_weather(start: date, end: date) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo archive.

    The archive API has data up to ~5 days ago.
    Returns a DataFrame indexed by date with weather feature columns.
    """
    response = httpx.get(
        HISTORICAL_URL,
        params={
            "latitude": LAT,
            "longitude": LON,
            "daily": ",".join(DAILY_PARAMS),
            "timezone": "Europe/Berlin",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=60,
    )
    response.raise_for_status()
    return _parse_daily_response(response.json())
