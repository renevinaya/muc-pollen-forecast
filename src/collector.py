"""
Data collector: fetches pollen measurements, weather, and NDVI, appends
to historical CSV.

Run every three hours to accumulate training data. The collector merges the
official station series from the LGL Bayern API with weather from Open-Meteo
and NDVI from MODIS, producing one row per (window, species) at 3-hour
resolution with all features attached — and, for a window the station did
not report, a single weather-only row (:data:`src.types.WEATHER_ONLY`), so
the weather of the history has no holes where the station has outages.
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd

from .upwind import UPWIND_STATIONS, fetch_upwind, update_upwind
from .pollen import fetch_measurements_chunked, fetch_pollen, pivot_pollen
from .weather import fetch_historical_weather, fetch_weather_forecast
from .ndvi import fetch_ndvi, interpolate_ndvi, set_composites_cache
from .types import ALL_SPECIES, LOCATION, RETIRED_COLUMNS, WEATHER_ONLY
from .clock import local_now, local_today

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_FILE = DATA_DIR / "history.csv"


def _upwind_result(future) -> pd.DataFrame:
    """The upwind fetch's frame, or an empty one — the stations are optional."""
    if future is None:
        return pd.DataFrame()
    try:
        return future.result()
    except Exception as exc:  # noqa: BLE001 — a missing upwind read costs features, not the run
        print(f"  Upwind fetch failed ({exc}), continuing without it.")
        return pd.DataFrame()


# Days before today the collector asks the forecast endpoint for. The archive
# ends about five days back and the forecast used to start today, so the days
# in between had no weather to attach a measurement to: a window measured on
# such a day could only be written while that day was still "today". The
# station publishes with a few hours' delay, so each evening's last two
# windows arrived after midnight and were dropped — to be picked up five
# days later, when the archive reached them. Every night's forecast was
# therefore built without the previous evening.
RECENT_PAST_DAYS = 5


def _archive_result(future) -> pd.DataFrame | None:
    """The archive's frame, or None — the forecast endpoint covers the recent days.

    A failed archive request costs the refresh of windows 5-14 days old with
    archive values, which the next run repeats. It must not cost the run.
    """
    if future is None:
        return None
    try:
        return future.result()
    except Exception as exc:  # noqa: BLE001 — optional refresh, never fatal
        print(f"  Weather archive failed ({exc}), continuing on the forecast endpoint alone.")
        return None


def _fetch_station(start, end) -> pd.DataFrame:
    """The Munich station's values: one request, or quarterly ones for a long span."""
    if (end - start).days <= 90:
        return fetch_pollen(start, end)
    frame = fetch_measurements_chunked(start, end, [LOCATION])
    return frame[["date", "species", "value"]]


def collect(days: int = 14) -> pd.DataFrame:
    """
    Collect recent pollen + weather data and return a tidy DataFrame.

    One row per (datetime, species) at 3-hour resolution, plus one
    weather-only row per window the station did not report — ready to append
    to the history file.
    """
    from concurrent.futures import ThreadPoolExecutor

    print(f"Collecting last {days} days of data...")

    end = local_today()
    start = end - timedelta(days=days)
    archive_end = end - timedelta(days=5)
    archive_start = start - timedelta(days=1)

    # Fetch pollen, weather, and NDVI in parallel
    with ThreadPoolExecutor(max_workers=5) as pool:
        fut_pollen = pool.submit(_fetch_station, start, end)
        fut_upwind = pool.submit(fetch_upwind, start, end) if UPWIND_STATIONS else None
        fut_archive = (
            pool.submit(fetch_historical_weather, archive_start, archive_end)
            if archive_start <= archive_end
            else None
        )
        fut_forecast = pool.submit(fetch_weather_forecast, 7, RECENT_PAST_DAYS)
        fut_ndvi = pool.submit(fetch_ndvi)

        pollen_raw = fut_pollen.result()
        upwind_raw = _upwind_result(fut_upwind)
        archive_weather = _archive_result(fut_archive)
        recent_weather_full = fut_forecast.result()
        ndvi_composites = fut_ndvi.result()

    # Populate NDVI in-memory cache for later use by forecaster
    set_composites_cache(ndvi_composites)

    # The upwind stations keep their own file; they never touch the history.
    if not upwind_raw.empty:
        n_st = upwind_raw["station"].nunique()
        print(f"  Upwind: {len(upwind_raw)} rows from {n_st} station(s)")
        update_upwind(upwind_raw)

    if pollen_raw.empty:
        print("No pollen data available.")
        return pd.DataFrame()

    pollen = pivot_pollen(pollen_raw)
    print(f"  Pollen: {len(pollen)} windows, species: {list(pollen.columns)}")

    # 2. Combine weather data
    weather_parts: list[pd.DataFrame] = []

    if archive_weather is not None:
        weather_parts.append(archive_weather)
        print(f"  Weather archive: {len(archive_weather)} windows")

    # Only keep past/today windows from the forecast
    now = local_now().floor("3h")
    recent_weather = recent_weather_full[recent_weather_full.index <= now]
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

    # 4. Windows the station reported get one row per species; windows it
    #    did not (an outage, or the last few hours it has not published yet)
    #    get one weather-only row. Only windows inside the span the station
    #    was asked about can be called unreported.
    asked = weather.index >= pd.Timestamp(start)
    common_dts = pollen.index.intersection(weather.index)
    unreported = weather.index[asked].difference(pollen.index)
    if common_dts.empty:
        print("  No overlapping windows between pollen and weather data.")
        return pd.DataFrame()
    if len(unreported):
        print(f"  {len(unreported)} window(s) without a station value kept as weather only")

    pollen = pollen.loc[common_dts]
    all_dts = common_dts.union(unreported)
    weather = weather.loc[all_dts]

    # 5. NDVI features (per date, not per window — same for all windows of a day)
    try:
        unique_dates = pd.DatetimeIndex(all_dts).normalize().unique()
        ndvi_df = interpolate_ndvi(ndvi_composites, unique_dates)
        if not ndvi_df.empty:
            print(f"  NDVI: {len(ndvi_df)} days")
        else:
            ndvi_df = pd.DataFrame()
    except Exception as exc:
        print(f"  NDVI interpolation failed ({exc}), continuing without NDVI.")
        ndvi_df = pd.DataFrame()

    # 6. Melt pollen to long format and merge weather + NDVI
    rows: list[dict[str, object]] = []
    reported = set(common_dts)
    for dt in all_dts:
        w = weather.loc[dt]
        day = dt.normalize()
        for species in ALL_SPECIES if dt in reported else [WEATHER_ONLY]:
            if species == WEATHER_ONLY:
                pollen_value = float("nan")
            else:
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
                row["ndvi_delta"] = 0.0
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Merged: {len(df)} rows")
    return df


def drop_superseded_weather_rows(history: pd.DataFrame) -> pd.DataFrame:
    """Remove a window's weather-only row once the window has measurements.

    The station publishes with a few hours' delay, so the newest windows are
    first written as weather only and get their species rows a run or two
    later. The species rows carry the same weather; the placeholder goes.
    """
    placeholder = history["species"] == WEATHER_ONLY
    if not placeholder.any():
        return history
    measured = set(history.loc[~placeholder, "date"])
    return history[~(placeholder & history["date"].isin(measured))]


def write_history(history: pd.DataFrame) -> pd.DataFrame:
    """Write *history* to the history file, without the retired columns.

    Every writer of the file goes through here, so a column nothing reads
    (:data:`src.types.RETIRED_COLUMNS`) leaves the stored file the first
    time it is written after being retired.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history = history.drop(columns=[c for c in RETIRED_COLUMNS if c in history.columns])
    history.to_csv(HISTORY_FILE, index=False)
    return history


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
    combined = drop_superseded_weather_rows(combined)
    combined = combined.sort_values(["date", "species"]).reset_index(drop=True)

    combined = write_history(combined)
    print(f"Updated history: {len(combined)} rows -> {HISTORY_FILE}")
    return combined
