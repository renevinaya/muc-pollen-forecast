"""Client for the LGL Bayern pollen measurement API."""

import time

import httpx
import pandas as pd

from .types import LOCATION

API_URL = "https://d1ppjuhp1nvtc2.cloudfront.net/measurements"


def fetch_pollen(days: int = 7) -> pd.DataFrame:
    """
    Fetch historical pollen measurements from the LGL Bayern API.

    Returns a DataFrame with columns: date, species, value
    """
    now = int(time.time())
    from_ts = now - (days * 24 * 60 * 60)

    response = httpx.get(
        API_URL,
        params={"from": from_ts, "to": now, "locations": LOCATION},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    rows: list[dict] = []
    for measurement in data.get("measurements", []):
        species = measurement["polle"]
        if species == "Varia":
            continue
        for point in measurement.get("data", []):
            rows.append(
                {
                    "date": pd.Timestamp(point["from"], unit="s", tz="Europe/Berlin")
                    .normalize()
                    .tz_localize(None),
                    "species": species,
                    "value": point["value"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["date", "species", "value"])

    df = pd.DataFrame(rows)
    # Aggregate to daily values (the API returns 3-hour intervals)
    df = df.groupby(["date", "species"], as_index=False)["value"].mean()
    return df


def pivot_pollen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot pollen data so each species is a column.
    Index = date, columns = species names, values = daily mean pollen count.
    Missing species/days are filled with 0.
    """
    if df.empty:
        return pd.DataFrame()

    pivoted = df.pivot_table(
        index="date", columns="species", values="value", aggfunc="mean"
    ).fillna(0)
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted = pivoted.sort_index()
    return pivoted
