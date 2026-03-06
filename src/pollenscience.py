"""Client for the pollenscience.eu pollen measurement API.

Provides real-time and historical pollen measurements from the PollenScience
network (TUM / Helmholtz). Data available from ~2019 at 3-hour resolution.

API: https://pollenscience.eu/api/measurements
  - from/to: Unix timestamps
  - locations: station code (e.g. DEMUNC for Munich)
  - pollen: comma-separated species names
"""

import time
from datetime import date, timedelta

import httpx
import pandas as pd

from .types import ALL_SPECIES

API_URL = "https://pollenscience.eu/api/measurements"

# Munich pollen station codes on pollenscience.eu — the station appears
# under different codes in different periods.
MUNICH_LOCATIONS = ["DEMUNC", "DEBIED"]

# Species to request (all from ALL_SPECIES)
_POLLEN_PARAM = ",".join(ALL_SPECIES)

# Chunk size for backfill: 28 days keeps response sizes reasonable
_CHUNK_DAYS = 28


def _fetch_single_location(
    start: date,
    end: date,
    location: str,
    species: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch pollen measurements from pollenscience.eu for a single station.

    Returns a DataFrame with columns: date, species, value
    at 3-hour window resolution.
    """
    from_ts = int(pd.Timestamp(str(start), tz="Europe/Berlin").timestamp())
    to_ts = int(pd.Timestamp(str(end + timedelta(days=1)), tz="Europe/Berlin").timestamp())

    pollen_param = ",".join(species) if species else _POLLEN_PARAM

    response = httpx.get(
        API_URL,
        params={
            "from": from_ts,
            "to": to_ts,
            "locations": location,
            "pollen": pollen_param,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    rows: list[dict] = []
    for measurement in data.get("measurements", []):
        sp = measurement["polle"]
        if sp not in ALL_SPECIES:
            continue
        for point in measurement.get("data", []):
            rows.append(
                {
                    "date": pd.Timestamp(point["from"], unit="s", tz="Europe/Berlin")
                    .floor("3h")
                    .tz_localize(None),
                    "species": sp,
                    "value": point["value"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["date", "species", "value"])

    return pd.DataFrame(rows)


def fetch_pollenscience(
    start: date,
    end: date,
    locations: list[str] | None = None,
    species: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch pollen measurements from pollenscience.eu, querying multiple
    Munich station codes and merging the results.

    Returns a DataFrame with columns: date, species, value
    at 3-hour window resolution.
    """
    locs = locations or MUNICH_LOCATIONS
    parts: list[pd.DataFrame] = []
    for loc in locs:
        df = _fetch_single_location(start, end, loc, species)
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["date", "species", "value"])

    combined = pd.concat(parts, ignore_index=True)
    # Keep the highest value when both stations report for the same window
    combined = (
        combined.groupby(["date", "species"], as_index=False)["value"]
        .max()
        .sort_values(["date", "species"])
        .reset_index(drop=True)
    )
    return combined


def fetch_pollenscience_chunked(
    start: date,
    end: date,
    locations: list[str] | None = None,
    species: list[str] | None = None,
    delay: float = 5.0,
) -> pd.DataFrame:
    """
    Fetch pollen data in small chunks with delays between requests
    to avoid overloading the server.

    Args:
        start: First date to fetch.
        end: Last date to fetch.
        locations: Station codes (default: MUNICH_LOCATIONS).
        species: Pollen species to request (default: ALL_SPECIES).
        delay: Seconds to sleep between API requests.

    Returns a combined DataFrame with columns: date, species, value.
    """
    locs = locations or MUNICH_LOCATIONS
    all_chunks: list[pd.DataFrame] = []
    chunk_start = start

    total_days = (end - start).days
    fetched_days = 0

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS - 1), end)
        print(f"  Fetching {chunk_start} to {chunk_end} "
              f"({fetched_days}/{total_days} days done)...")

        try:
            # Fetch from each station with a delay between requests
            parts: list[pd.DataFrame] = []
            for loc in locs:
                df = _fetch_single_location(chunk_start, chunk_end, loc, species)
                if not df.empty:
                    parts.append(df)
                if loc != locs[-1]:
                    time.sleep(delay)

            if parts:
                merged = pd.concat(parts, ignore_index=True)
                merged = (
                    merged.groupby(["date", "species"], as_index=False)["value"]
                    .max()
                )
                n_nonzero = (merged["value"] > 0).sum()
                print(f"    -> {len(merged)} rows, {n_nonzero} nonzero")
                all_chunks.append(merged)
            else:
                print(f"    -> no data")
        except httpx.HTTPStatusError as exc:
            print(f"    -> HTTP error {exc.response.status_code}, skipping chunk")
        except Exception as exc:
            print(f"    -> error: {exc}, skipping chunk")

        fetched_days += (chunk_end - chunk_start).days + 1
        chunk_start = chunk_end + timedelta(days=1)

        if chunk_start <= end:
            print(f"    sleeping {delay}s...")
            time.sleep(delay)

    if not all_chunks:
        return pd.DataFrame(columns=["date", "species", "value"])

    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "species"], keep="last")
    return combined.sort_values(["date", "species"]).reset_index(drop=True)
