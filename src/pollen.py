"""Client for the LGL Bayern ePIN measurement API — the official station series.

This is the one source of pollen and spore measurements, for Munich and for
the upwind stations alike. It serves exactly one value per station, taxon and
3-hour window.

pollenscience.eu, which the collector read until November 2026, serves the
same official series *plus* a second one under the same station codes (and,
for Munich, a second instrument under the code DEBIED), up to two values per
window. Taking the larger, as the collector did, built a series no station
measured: yearly sums 1.02–1.93 times the official ones, the daily level
different on up to 45% of the pollen days of a year, and season starts moved
by up to 20 days (README, *One station, one series*).

API: https://d1ppjuhp1nvtc2.cloudfront.net/measurements
  - from/to: Unix timestamps
  - locations: comma-separated station codes (e.g. DEMUNC)
  - pollen: comma-separated taxa (optional; default every taxon the station reports)
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

import httpx
import pandas as pd

from .types import ALL_SPECIES, LOCATION

API_URL = "https://d1ppjuhp1nvtc2.cloudfront.net/measurements"

COLUMNS = ["date", "station", "species", "value"]

# A quarter per request: about 45 000 points for five stations and twelve
# taxa, which the API answers in a few seconds.
_CHUNK_DAYS = 90

_MAX_ATTEMPTS = 4


def _get(params: dict[str, Any], timeout: float = 90.0) -> dict[str, Any]:
    """GET the API, retrying transport errors, 5xx and non-JSON bodies."""
    last_error = "no attempt made"
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = httpx.get(API_URL, params=params, timeout=timeout)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if 400 <= response.status_code < 500:
                response.raise_for_status()
            elif response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
            else:
                try:
                    return response.json()  # type: ignore[no-any-return]
                except ValueError:
                    last_error = f"HTTP {response.status_code} with a non-JSON body"
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(2**attempt)
    raise RuntimeError(f"{API_URL} failed after {_MAX_ATTEMPTS} attempts — {last_error}")


def parse_measurements(payload: dict[str, Any], species: list[str] | None = None) -> pd.DataFrame:
    """The API's answer as a long frame: date, station, species, value.

    ``date`` is the window start on Munich's wall clock, naive, floored to
    the 3-hour grid — the convention of the whole history. Taxa outside
    *species* (default: :data:`ALL_SPECIES`) are dropped. Two points that
    land on one window (the repeated hour of the autumn clock change) are
    averaged.
    """
    wanted = set(species) if species else set(ALL_SPECIES)
    parts: list[pd.DataFrame] = []
    for series in payload.get("measurements", []):
        taxon = series.get("polle")
        points = series.get("data") or []
        if taxon not in wanted or not points:
            continue
        frame = pd.DataFrame(points, columns=["from", "value"])
        frame["station"] = series.get("location", LOCATION)
        frame["species"] = taxon
        parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=COLUMNS)

    raw = pd.concat(parts, ignore_index=True)
    raw["date"] = (
        pd.to_datetime(raw["from"], unit="s", utc=True)
        .dt.tz_convert("Europe/Berlin")
        .dt.floor("3h")
        .dt.tz_localize(None)
    )
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["value"])
    out = raw.groupby(["date", "station", "species"], as_index=False)["value"].mean()
    return out[COLUMNS].sort_values(["date", "station", "species"]).reset_index(drop=True)


def fetch_measurements(
    start: date,
    end: date,
    locations: list[str] | None = None,
    species: list[str] | None = None,
) -> pd.DataFrame:
    """Measurements of *locations* from *start* to *end* inclusive, in one request."""
    from_ts = int(pd.Timestamp(str(start), tz="Europe/Berlin").timestamp())
    to_ts = int(pd.Timestamp(str(end + timedelta(days=1)), tz="Europe/Berlin").timestamp())
    payload = _get(
        {
            "from": from_ts,
            "to": to_ts,
            "locations": ",".join(locations or [LOCATION]),
            "pollen": ",".join(species or ALL_SPECIES),
        }
    )
    return parse_measurements(payload, species)


def fetch_measurements_chunked(
    start: date,
    end: date,
    locations: list[str] | None = None,
    species: list[str] | None = None,
    delay: float = 1.0,
    strict: bool = False,
) -> pd.DataFrame:
    """A long span in quarterly requests, with a pause between them.

    A chunk that fails after its retries is reported and skipped: a backfill
    of eight years should not die on one bad quarter, and the gap it leaves
    is a gap, not a wrong value. With *strict* the failure is raised instead
    — for a caller that would read a missing quarter as "the station did not
    report".
    """
    chunks: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS - 1), end)
        try:
            chunk = fetch_measurements(chunk_start, chunk_end, locations, species)
            print(f"  {chunk_start} to {chunk_end}: {len(chunk)} values")
            if not chunk.empty:
                chunks.append(chunk)
        except Exception as exc:  # noqa: BLE001 — report, skip, continue
            if strict:
                raise
            print(f"  {chunk_start} to {chunk_end}: failed ({exc}), skipped")
        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start <= end:
            time.sleep(delay)
    if not chunks:
        return pd.DataFrame(columns=COLUMNS)
    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "station", "species"], keep="last")
    return combined.sort_values(["date", "station", "species"]).reset_index(drop=True)


def fetch_pollen(start: date, end: date, species: list[str] | None = None) -> pd.DataFrame:
    """The Munich station's measurements: date, species, value."""
    frame = fetch_measurements(start, end, [LOCATION], species)
    return frame[["date", "species", "value"]]


def pivot_pollen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot pollen data so each species is a column.
    Index = datetime (3h windows), columns = species names, values = pollen count.
    A taxon the station did not list in a window it reported is 0.
    """
    if df.empty:
        return pd.DataFrame()

    pivoted = df.pivot_table(
        index="date", columns="species", values="value", aggfunc="mean"
    ).fillna(0)
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted = pivoted.sort_index()
    return pivoted
