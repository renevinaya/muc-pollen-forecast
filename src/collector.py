"""
Data collector: fetches pollen measurements, weather, and NDVI, appends
to historical CSV.

Run daily to accumulate training data. The collector merges pollen data from the
LGL Bayern API with weather data from Open-Meteo's historical archive and NDVI
from MODIS, producing one row per (datetime, species) at 3-hour resolution
with all features attached.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .pollenscience import fetch_pollenscience
from .pollen import pivot_pollen
from .weather import fetch_historical_weather, fetch_weather_forecast
from .ndvi import ndvi_features
from .types import ALL_SPECIES

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_FILE = DATA_DIR / "history.csv"


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical calendar and time-of-day features from the datetime index."""
    import numpy as np

    df = df.copy()
    doy = df.index.dayofyear
    df["day_of_year"] = doy
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df.index.month
    # Time-of-day features (3h window)
    hour = df.index.hour
    df["hour_of_day"] = hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df


def collect(days: int = 14) -> pd.DataFrame:
    """
    Collect recent pollen + weather data and return a tidy DataFrame.

    One row per (datetime, species) at 3-hour resolution — ready to append
    to the history file.
    """
    print(f"Collecting last {days} days of data...")

    # 1. Pollen data (3h resolution) from pollenscience.eu
    end = date.today()
    start = end - timedelta(days=days)
    pollen_raw = fetch_pollenscience(start, end)
    if pollen_raw.empty:
        print("No pollen data available.")
        return pd.DataFrame()

    pollen = pivot_pollen(pollen_raw)
    print(f"  Pollen: {len(pollen)} windows, species: {list(pollen.columns)}")

    # 2. Weather data (3h resolution): use archive for older dates, forecast API for recent
    archive_end = date.today() - timedelta(days=5)
    archive_start = pollen.index.min().date() - timedelta(days=1)

    weather_parts: list[pd.DataFrame] = []

    # 2a. Historical archive (available up to ~5 days ago)
    if archive_start <= archive_end:
        archive_weather = fetch_historical_weather(archive_start, archive_end)
        weather_parts.append(archive_weather)
        print(f"  Weather archive: {len(archive_weather)} windows")

    # 2b. Forecast API for the last ~5 days + today (fills the gap)
    recent_weather = fetch_weather_forecast(days=7)
    # Only keep past/today windows from the forecast
    now = pd.Timestamp.now().floor("3h")
    recent_weather = recent_weather[recent_weather.index <= now]
    if not recent_weather.empty:
        weather_parts.append(recent_weather)
        print(f"  Weather recent (from forecast API): {len(recent_weather)} windows")

    if not weather_parts:
        print("  No weather data available.")
        return pd.DataFrame()

    weather = pd.concat(weather_parts)
    # Deduplicate (archive and forecast may overlap); prefer archive data
    weather = weather[~weather.index.duplicated(keep="first")]
    weather = weather.sort_index()

    # 3. Add calendar features to weather
    weather = _add_calendar_features(weather)

    # 4. Join: only keep windows where we have both pollen AND weather
    common_dts = pollen.index.intersection(weather.index)
    if common_dts.empty:
        print("  No overlapping windows between pollen and weather data.")
        return pd.DataFrame()

    pollen = pollen.loc[common_dts]
    weather = weather.loc[common_dts]

    # 5. NDVI features (per date, not per window — same for all windows of a day)
    try:
        unique_dates = pd.DatetimeIndex(common_dts.normalize().unique())
        ndvi_df = ndvi_features(unique_dates)
        if not ndvi_df.empty:
            print(f"  NDVI: {len(ndvi_df)} days")
        else:
            ndvi_df = pd.DataFrame()
    except Exception as exc:
        print(f"  NDVI fetch failed ({exc}), continuing without NDVI.")
        ndvi_df = pd.DataFrame()

    # 6. Melt pollen to long format and merge weather + NDVI
    rows: list[dict] = []
    for dt in common_dts:
        w = weather.loc[dt]
        day = dt.normalize()
        for species in ALL_SPECIES:
            pollen_value = pollen.loc[dt].get(species, 0.0) if species in pollen.columns else 0.0
            row = {"date": dt, "species": species, "value": float(pollen_value)}
            for col in weather.columns:
                row[col] = float(w[col])
            # Add NDVI columns (daily resolution, shared across windows)
            if not ndvi_df.empty and day in ndvi_df.index:
                for col in ndvi_df.columns:
                    row[col] = float(ndvi_df.loc[day, col])
            else:
                row["ndvi"] = 0.0
                row["evi"] = 0.0
                row["ndvi_delta"] = 0.0
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Merged: {len(df)} rows")
    return df


def update_history(new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Append new data to the history CSV, deduplicating by (date, species).
    Returns the full updated history.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if HISTORY_FILE.exists():
        history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
        print(f"Existing history: {len(history)} rows")
    else:
        history = pd.DataFrame()
        print("No existing history, starting fresh.")

    if new_data.empty:
        return history

    combined = pd.concat([history, new_data], ignore_index=True)
    # Keep the latest value for each (date, species) pair
    combined = combined.drop_duplicates(subset=["date", "species"], keep="last")
    combined = combined.sort_values(["date", "species"]).reset_index(drop=True)

    combined.to_csv(HISTORY_FILE, index=False)
    print(f"Updated history: {len(combined)} rows -> {HISTORY_FILE}")
    return combined
