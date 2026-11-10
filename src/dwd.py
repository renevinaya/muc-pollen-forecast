"""
Client for DWD (Deutscher Wetterdienst) pollen data.

fetch_dwd_forecast() — today/tomorrow/day-after categorical pollen forecast
for the Oberbayern region (partregion_id 121).

A fetch of DWD's phenology observations lived here until November 2026. It
fed nothing: the onset features have been calibrated on the station's own
seasons since B.2, while the file was still downloaded from the release on
every run, uploaded again on every run and refetched from DWD monthly. The
fetch itself could only ever match the oldest archive file of each plant
(``..._1930_2018_hist.txt``), never the current one, and did not map alder
— which is why it looked like one year of data. TASKS.md notes what the
source could be used for.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
import pandas as pd

from .clock import local_today

# ── DWD Pollenflug-Gefahrenindex (categorical forecast, updated ~daily) ──────

DWD_FORECAST_URL = (
    "https://opendata.dwd.de/climate_environment/health/alerts/s31fg.json"
)

# Munich is in "Allgäu / Oberbayern / Bay. Wald"
PARTREGION_ID = 121

# DWD German species names → our canonical names
DWD_SPECIES_MAP: dict[str, str] = {
    "Erle": "Alnus",
    "Ambrosia": "Ambrosia",
    "Beifuss": "Artemisia",
    "Birke": "Betula",
    "Hasel": "Corylus",
    "Esche": "Fraxinus",
    "Graeser": "Poaceae",
    "Roggen": "Poaceae",  # rye pollen grouped with grasses
}

# DWD level codes → numeric midpoint (scale 0–3)
_LEVEL_MAP: dict[str, float] = {
    "0": 0.0,
    "0-1": 0.5,
    "1": 1.0,
    "1-2": 1.5,
    "2": 2.0,
    "2-3": 2.5,
    "3": 3.0,
}


def fetch_dwd_forecast() -> pd.DataFrame:
    """
    Fetch the current DWD Pollenflug-Gefahrenindex for Oberbayern.

    Returns a DataFrame with columns:
        date, species, dwd_level  (float 0–3)

    Rows for today, tomorrow, and day-after-tomorrow.
    """
    resp = httpx.get(DWD_FORECAST_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Parse "last_update" → base date
    last_update = data.get("last_update", "")
    # Format: "2026-03-04 11:00 Uhr"
    try:
        base_date = pd.Timestamp(last_update.split(" ")[0]).date()
    except Exception:
        base_date = local_today()

    # Find our region
    region = None
    for r in data.get("content", []):
        if r.get("partregion_id") == PARTREGION_ID:
            region = r
            break
    if region is None:
        return pd.DataFrame(columns=["date", "species", "dwd_level"])

    day_keys = {
        "today": base_date,
        "tomorrow": base_date + timedelta(days=1),
        "dayafter_to": base_date + timedelta(days=2),
    }

    rows: list[dict[str, object]] = []
    for dwd_name, pollen_data in region.get("Pollen", {}).items():
        species = DWD_SPECIES_MAP.get(dwd_name)
        if species is None:
            continue
        for day_key, dt in day_keys.items():
            raw = pollen_data.get(day_key, "0")
            level = _LEVEL_MAP.get(raw, 0.0)
            rows.append({"date": pd.Timestamp(dt), "species": species, "dwd_level": level})

    df = pd.DataFrame(rows)
    # If Roggen and Graeser both map to Poaceae, take the max per day
    agg = df.groupby(["date", "species"], as_index=False)["dwd_level"].max()
    return agg if isinstance(agg, pd.DataFrame) else agg.to_frame()
