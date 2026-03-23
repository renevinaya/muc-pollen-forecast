"""
Client for MODIS NDVI satellite vegetation data (ORNL DAAC REST API).

Provides:
  - fetch_ndvi()        — download 16-day NDVI composites for Munich
  - interpolate_ndvi()  — interpolate 16-day composites to daily values
  - ndvi_features()     — compute NDVI-derived features for the model

The MODIS MOD13Q1 product provides 250 m NDVI at 16-day intervals,
available from 2000 to near-present.  No authentication required.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .types import LAT, LON

MODIS_API = "https://modis.ornl.gov/rst/api/v1"
PRODUCT = "MOD13Q1"  # 250m 16-day NDVI
NDVI_BAND = "250m_16_days_NDVI"
EVI_BAND = "250m_16_days_EVI"
SCALE_FACTOR = 10_000.0  # MODIS stores NDVI as int × 10000

DATA_DIR = Path(__file__).parent.parent / "data"
NDVI_CACHE = DATA_DIR / "ndvi_cache.csv"


def _modis_date(d: date) -> str:
    """Convert a Python date to MODIS date format: A{year}{doy:03d}."""
    return f"A{d.year}{d.timetuple().tm_yday:03d}"


def fetch_ndvi(
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """
    Download MODIS NDVI & EVI composites for Munich from the ORNL DAAC API.

    Returns a DataFrame with columns:
        date, ndvi, evi

    Results are cached locally.  On subsequent calls only the missing
    tail is fetched and appended.
    """
    import httpx

    if end is None:
        end = date.today()
    if start is None:
        # Default: 2 years back — covers the pollen history range
        start = date(end.year - 2, 1, 1)

    # Load cache
    cached = pd.DataFrame()
    if NDVI_CACHE.exists():
        cached = pd.read_csv(NDVI_CACHE, parse_dates=["date"])
        if not cached.empty:
            earliest_cached: date = cached["date"].min().date()
            latest_cached: date = cached["date"].max().date()
            # Cache is sufficient if it covers the full requested range
            cache_covers_start = earliest_cached <= start + timedelta(days=32)
            cache_covers_end = latest_cached >= end - timedelta(days=32)
            if cache_covers_start and cache_covers_end:
                print(f"  NDVI cache up-to-date ({earliest_cached} to {latest_cached})")
                return cached
            # Otherwise re-fetch from uncovered start
            if earliest_cached <= start + timedelta(days=32):
                start = max(start, latest_cached - timedelta(days=16))

    print(f"  Fetching MODIS NDVI from {start} to {end}...")

    # The ORNL API limits to 10 tiles per request (10 × 16 days = 160 days).
    # Chunk into ~150-day intervals.
    chunk_days = 150
    all_records: list[dict[str, object]] = []
    current_start = start

    while current_start < end:
        chunk_end = min(current_start + timedelta(days=chunk_days), end)
        try:
            resp = httpx.get(
                f"{MODIS_API}/{PRODUCT}/subset",
                params={
                    "latitude": LAT,
                    "longitude": LON,
                    "startDate": _modis_date(current_start),
                    "endDate": _modis_date(chunk_end),
                    "kmAboveBelow": 0,
                    "kmLeftRight": 0,
                },
                timeout=30,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  Warning: MODIS fetch failed for {current_start}–{chunk_end}: {exc}")
            current_start = chunk_end + timedelta(days=1)
            continue

        subset = data.get("subset", [])
        if not isinstance(subset, list):
            current_start = date(current_start.year + 1, 1, 1)
            continue

        # Group by calendar_date
        by_date: dict[str, dict[str, float]] = {}
        for rec in subset:
            cal = rec["calendar_date"]
            band = rec["band"]
            val = rec["data"][0] if rec.get("data") else None
            if cal not in by_date:
                by_date[cal] = {}
            if band == NDVI_BAND and val is not None:
                by_date[cal]["ndvi"] = val / SCALE_FACTOR
            elif band == EVI_BAND and val is not None:
                by_date[cal]["evi"] = val / SCALE_FACTOR

        for cal, vals in by_date.items():
            if "ndvi" in vals:
                all_records.append(
                    {
                        "date": pd.Timestamp(cal),
                        "ndvi": vals["ndvi"],
                        "evi": vals.get("evi", np.nan),
                    }
                )

        current_start = chunk_end + timedelta(days=1)

    if not all_records:
        return cached if not cached.empty else pd.DataFrame(columns=["date", "ndvi", "evi"])

    new_df = pd.DataFrame(all_records).sort_values("date").reset_index(drop=True)

    # Merge with cache
    if not cached.empty:
        combined = pd.concat([cached, new_df]).drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
    else:
        combined = new_df

    # Save cache
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(NDVI_CACHE, index=False)
    print(
        f"  NDVI: {len(combined)} composites"
        f" ({combined['date'].min().date()} to {combined['date'].max().date()})"
    )

    return combined


def interpolate_ndvi(composites: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Interpolate 16-day MODIS NDVI composites to daily values.

    Uses cubic interpolation for smooth green-up/senescence curves.

    Returns a DataFrame indexed by the requested dates with columns:
        ndvi, evi, ndvi_delta  (change rate)
    """
    if composites.empty:
        return pd.DataFrame(
            {"ndvi": 0.0, "evi": 0.0, "ndvi_delta": 0.0},
            index=dates,
        )

    comp = composites.set_index("date").sort_index()

    # Filter out clearly bad values (MODIS fill values)
    comp = comp[(comp["ndvi"] > -0.2) & (comp["ndvi"] < 1.0)]

    # Reindex to daily frequency and interpolate
    full_range = pd.date_range(comp.index.min(), max(comp.index.max(), dates.max()), freq="D")
    daily = comp.reindex(full_range)

    # Use cubic interpolation when enough points, otherwise linear
    method: str = "cubic" if comp["ndvi"].notna().sum() >= 4 else "linear"
    try:
        daily["ndvi"] = daily["ndvi"].interpolate(method=method).ffill().bfill()  # type: ignore[arg-type]
        daily["evi"] = daily["evi"].interpolate(method=method).ffill().bfill()  # type: ignore[arg-type]
    except (ValueError, TypeError):
        # Fall back to linear if cubic fails (e.g., boundary issues)
        daily["ndvi"] = daily["ndvi"].interpolate(method="linear").ffill().bfill()
        daily["evi"] = daily["evi"].interpolate(method="linear").ffill().bfill()

    # NDVI change rate (green-up speed)
    daily["ndvi_delta"] = daily["ndvi"].diff().fillna(0)

    # Return only requested dates
    result = daily.reindex(dates).ffill().fillna(0)
    return result[["ndvi", "evi", "ndvi_delta"]]


_composites_cache: pd.DataFrame | None = None


def set_composites_cache(composites: pd.DataFrame) -> None:
    """Populate the in-memory composites cache (used by parallel fetching)."""
    global _composites_cache
    _composites_cache = composites


def ndvi_features(history_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Convenience wrapper: fetch NDVI, interpolate to history dates,
    return feature columns ready to merge.

    Caches the fetched composites in memory so that repeated calls within
    the same process (e.g. collect then forecast) don't re-fetch.
    """
    global _composites_cache
    if _composites_cache is None:
        _composites_cache = fetch_ndvi()
    return interpolate_ndvi(_composites_cache, history_dates)
