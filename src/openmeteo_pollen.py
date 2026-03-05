"""Client for the Open-Meteo Air Quality API — historical pollen data.

Provides modeled pollen concentration data (CAMS European Air Quality) for
Munich going back to 2021. Available species: Alder, Birch, Grass, Mugwort,
Ragweed (= Alnus, Betula, Poaceae, Artemisia, Ambrosia).

This is complementary to the LGL Bayern real-time measurements — it provides
years of historical data for model training, while LGL gives precise local
ground-truth for recent periods.
"""

from datetime import date, timedelta

import httpx
import pandas as pd

from .types import LAT, LON

API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Mapping from Open-Meteo API parameter names to our canonical species names
_PARAM_TO_SPECIES: dict[str, str] = {
    "alder_pollen": "Alnus",
    "birch_pollen": "Betula",
    "grass_pollen": "Poaceae",
    "mugwort_pollen": "Artemisia",
    "ragweed_pollen": "Ambrosia",
}

_HOURLY_PARAMS = list(_PARAM_TO_SPECIES.keys())

# Open-Meteo pollen data available from this date onward
EARLIEST_DATE = date(2021, 1, 1)

# Maximum days per request to avoid timeouts
_CHUNK_DAYS = 90


def fetch_openmeteo_pollen(start: date, end: date) -> pd.DataFrame:
    """
    Fetch historical pollen data from Open-Meteo Air Quality API.

    Returns a DataFrame with columns: date, species, value
    at 3-hour window resolution (matching LGL pollen format).

    The data comes from the CAMS European Air Quality reanalysis/forecast
    model at ~11 km resolution, in units of grains/m³.
    """
    start = max(start, EARLIEST_DATE)
    if start > end:
        return pd.DataFrame(columns=["date", "species", "value"])

    # Fetch in chunks to avoid timeouts with large date ranges
    all_chunks: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS - 1), end)
        df = _fetch_chunk(chunk_start, chunk_end)
        if not df.empty:
            all_chunks.append(df)
        chunk_start = chunk_end + timedelta(days=1)

    if not all_chunks:
        return pd.DataFrame(columns=["date", "species", "value"])

    return pd.concat(all_chunks, ignore_index=True)


def _fetch_chunk(start: date, end: date) -> pd.DataFrame:
    """Fetch a single chunk of pollen data and aggregate to 3h windows."""
    response = httpx.get(
        API_URL,
        params={
            "latitude": LAT,
            "longitude": LON,
            "hourly": ",".join(_HOURLY_PARAMS),
            "timezone": "Europe/Berlin",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    hourly = data.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []))
    if times.empty:
        return pd.DataFrame(columns=["date", "species", "value"])

    # Build hourly DataFrame
    df = pd.DataFrame({"datetime": times})
    for param in _HOURLY_PARAMS:
        df[param] = hourly.get(param, [None] * len(times))

    # Floor to 3h windows and aggregate (mean concentration per window)
    df["window"] = df["datetime"].dt.floor("3h")

    rows: list[dict] = []
    for param, species in _PARAM_TO_SPECIES.items():
        grouped = df.groupby("window")[param].mean()
        for window_dt, value in grouped.items():
            if value is not None and pd.notna(value):
                rows.append({
                    "date": window_dt,
                    "species": species,
                    "value": float(value),
                })

    if not rows:
        return pd.DataFrame(columns=["date", "species", "value"])

    return pd.DataFrame(rows)
