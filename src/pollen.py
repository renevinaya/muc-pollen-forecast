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
    where date is a Timestamp at 3-hour window boundaries (00, 03, 06, ..., 21).
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

    rows: list[dict[str, object]] = []
    for measurement in data.get("measurements", []):
        species = measurement["polle"]
        if species == "Varia":
            continue
        for point in measurement.get("data", []):
            rows.append(
                {
                    "date": pd.Timestamp(point["from"], unit="s", tz="Europe/Berlin")
                    .floor("3h")
                    .tz_localize(None),
                    "species": species,
                    "value": point["value"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["date", "species", "value"])

    df = pd.DataFrame(rows)
    return df


def pivot_pollen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot pollen data so each species is a column.
    Index = datetime (3h windows), columns = species names, values = pollen count.
    Missing species/windows are filled with 0.
    """
    if df.empty:
        return pd.DataFrame()

    pivoted = df.pivot_table(
        index="date", columns="species", values="value", aggfunc="mean"
    ).fillna(0)
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted = pivoted.sort_index()
    return pivoted
