"""
Data collector: fetches pollen measurements and weather, appends to historical CSV.

Run daily to accumulate training data. The collector merges pollen data from the
LGL Bayern API with weather data from Open-Meteo's historical archive, producing
one row per (date, species) with all weather features attached.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .pollen import fetch_pollen, pivot_pollen
from .weather import fetch_historical_weather
from .types import ALL_SPECIES

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_FILE = DATA_DIR / "history.csv"


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical calendar features from the date index."""
    import numpy as np

    df = df.copy()
    doy = df.index.dayofyear
    df["day_of_year"] = doy
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df.index.month
    return df


def collect(days: int = 14) -> pd.DataFrame:
    """
    Collect recent pollen + weather data and return a tidy DataFrame.

    One row per (date, species) — ready to append to the history file.
    """
    print(f"Collecting last {days} days of data...")

    # 1. Pollen data
    pollen_raw = fetch_pollen(days=days)
    if pollen_raw.empty:
        print("No pollen data available.")
        return pd.DataFrame()

    pollen = pivot_pollen(pollen_raw)
    print(f"  Pollen: {len(pollen)} days, species: {list(pollen.columns)}")

    # 2. Historical weather for the same date range
    start = pollen.index.min().date() - timedelta(days=1)
    # Archive API is ~5 days behind; clamp end date
    end = min(pollen.index.max().date(), date.today() - timedelta(days=5))
    if start > end:
        # All pollen data is too recent for the archive API
        print("  Weather archive not yet available for these dates, skipping.")
        return pd.DataFrame()

    weather = fetch_historical_weather(start, end)
    print(f"  Weather: {len(weather)} days")

    # 3. Add calendar features to weather
    weather = _add_calendar_features(weather)

    # 4. Join: only keep dates where we have both pollen AND weather
    common_dates = pollen.index.intersection(weather.index)
    if common_dates.empty:
        print("  No overlapping dates between pollen and weather data.")
        return pd.DataFrame()

    pollen = pollen.loc[common_dates]
    weather = weather.loc[common_dates]

    # 5. Melt pollen to long format and merge weather
    rows: list[dict] = []
    for dt in common_dates:
        w = weather.loc[dt]
        for species in ALL_SPECIES:
            pollen_value = pollen.loc[dt].get(species, 0.0) if species in pollen.columns else 0.0
            row = {"date": dt, "species": species, "value": float(pollen_value)}
            for col in weather.columns:
                row[col] = float(w[col])
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
