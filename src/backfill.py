"""Refresh the weather and NDVI columns of an existing history in place.

Ten of the model's sixty features were added to the collector long after the
pollen history was backfilled, so ``history.csv`` carried them as NaN (or, for
NDVI, a literal 0.0) for five to seven of its eight seasons: the six diurnal
weather features exist from 2025-03-24, soil from 2026-06-17, NDVI from
2024-01-01. Seven of the eight hazel, alder and birch season starts in the
training set therefore had none of that signal.

Re-running the pollen backfill would fix it at the cost of re-downloading
eight years of pollen at five seconds per chunk. The two functions here fetch
only the missing half — the Open-Meteo ERA5 archive and the MODIS composites —
and overwrite the corresponding columns of rows that already exist. Pollen
values are never touched, and no rows are added or removed.

Both are idempotent: running them again re-fetches and rewrites the same
values. Both are also the only place in the pipeline that writes history
columns without going through the collector, so they report exactly which
columns changed on how many rows.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .clock import local_today
from .types import ALL_SPECIES, WEATHER_COLUMNS, WEATHER_ONLY

# The archive lags real time by about this much; the collector fills the gap
# from the forecast API and those rows are left alone here.
ARCHIVE_LAG_DAYS = 5

# Days of history to refresh per archive request. A calendar year of hourly
# data for the ~17 requested variables is a few megabytes and returns in
# seconds; anything much larger risks the 60 s read timeout.
WEATHER_CHUNK_DAYS = 366

# An interpolated NDVI value further than this from a real composite is a
# guess; MODIS composites are 16 days apart, so one missed composite is fine
# and two are not.
NDVI_MAX_GAP_DAYS = 32

NDVI_COLUMNS = ["ndvi", "ndvi_delta"]


def _history_dates(history: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(history["date"]))


def _overwrite(
    history: pd.DataFrame, table: pd.DataFrame, columns: list[str], key: pd.Index
) -> dict[str, tuple[int, int]]:
    """Write *table*'s *columns* onto *history* rows whose *key* is in the table.

    A NaN in the table leaves the existing value alone — a fetch that came
    back short must not erase data we already had. Returns, per column, how
    many rows changed and how many of those were previously NaN.
    """
    aligned = table.reindex(key)
    changed: dict[str, tuple[int, int]] = {}
    for col in columns:
        if col not in aligned.columns:
            continue
        new = aligned[col].to_numpy(dtype=float)
        have = ~np.isnan(new)
        if col in history.columns:
            old = history[col].to_numpy(dtype=float)
        else:
            old = np.full(len(history), np.nan)
        was_nan = np.isnan(old)
        differs = have & (was_nan | ~np.isclose(old, new, equal_nan=False))
        if differs.any():
            history.loc[differs, col] = new[differs]
        changed[col] = (int(differs.sum()), int((differs & was_nan).sum()))
    return changed


def _print_changes(changed: dict[str, tuple[int, int]], n_rows: int) -> None:
    print(f"    {'Column':<26} {'rows changed':>13} {'of which NaN':>13}")
    for col, (n, n_nan) in changed.items():
        print(f"    {col:<26} {n:>13} {n_nan:>13}")
    untouched = [c for c, (n, _) in changed.items() if n == 0]
    if untouched:
        print(f"    unchanged: {', '.join(untouched)}")
    print(f"    ({n_rows} history rows)")


def refresh_weather(
    history: pd.DataFrame,
    start: date | None = None,
    end: date | None = None,
    fetch=None,
) -> pd.DataFrame:
    """Overwrite every weather column of *history* from the ERA5 archive.

    Covers ``start``..``end`` (default: the whole history up to the archive's
    edge). All :data:`WEATHER_COLUMNS` are rewritten, including the ones the
    model currently ignores, so un-pruning a column later needs no second
    pass. *fetch* is the archive client, injectable for tests.
    """
    if fetch is None:
        from .weather import fetch_historical_weather as fetch

    if history.empty:
        return history
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    dates = _history_dates(history)

    first = dates.min().date()
    last = dates.max().date()
    archive_edge = local_today() - timedelta(days=ARCHIVE_LAG_DAYS)
    start = max(start or first, first)
    end = min(end or last, last, archive_edge)
    if start > end:
        print(f"  Nothing to refresh: history ends before {start}")
        return history

    print(f"  Refreshing weather columns from the archive: {start} to {end}")
    parts: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=WEATHER_CHUNK_DAYS - 1), end)
        print(f"    fetching {chunk_start} .. {chunk_end}")
        parts.append(fetch(chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)

    weather = pd.concat(parts)
    weather.index = pd.DatetimeIndex(weather.index)
    weather = weather[~weather.index.duplicated(keep="last")].sort_index()
    # Only windows inside the requested span may change; the forecast-API
    # tail after the archive edge keeps what the collector wrote.
    span_end = pd.Timestamp(end) + pd.Timedelta(days=1)
    weather = weather[(weather.index >= pd.Timestamp(start)) & (weather.index < span_end)]

    changed = _overwrite(history, weather, WEATHER_COLUMNS, dates)
    _print_changes(changed, len(history))
    return history


def refresh_ndvi(
    history: pd.DataFrame,
    start: date | None = None,
    end: date | None = None,
    fetch=None,
) -> pd.DataFrame:
    """Overwrite the NDVI columns of *history* from MODIS composites.

    Days more than :data:`NDVI_MAX_GAP_DAYS` from the nearest real composite
    are left as they are rather than filled with an extrapolation — the
    interpolator pads its ends, and a padded value is not a measurement.
    """
    if fetch is None:
        from .ndvi import fetch_ndvi as fetch
    from .ndvi import interpolate_ndvi

    if history.empty:
        return history
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    dates = _history_dates(history)
    days = dates.normalize()

    start = start or days.min().date()
    end = end or local_today()
    print(f"  Refreshing NDVI columns from MODIS composites: {start} to {end}")
    composites = fetch(start, end)
    if composites is None or composites.empty:
        print("  No composites returned; NDVI columns left unchanged.")
        return history

    comp_days = pd.DatetimeIndex(pd.to_datetime(composites["date"])).sort_values()
    wanted = pd.DatetimeIndex(days.unique()).sort_values()
    table = interpolate_ndvi(composites, wanted)

    # Distance from each wanted day to the nearest composite.
    pos = comp_days.searchsorted(wanted)
    before = comp_days[np.clip(pos - 1, 0, len(comp_days) - 1)]
    after = comp_days[np.clip(pos, 0, len(comp_days) - 1)]
    gap = np.minimum(abs((wanted - before).days), abs((wanted - after).days))
    supported = pd.Series(gap <= NDVI_MAX_GAP_DAYS, index=wanted)
    table = table[supported.reindex(table.index).fillna(False).to_numpy()]
    n_unsupported = int((~supported).sum())
    if n_unsupported:
        print(f"    {n_unsupported} day(s) further than {NDVI_MAX_GAP_DAYS} d from any "
              "composite left unchanged")

    changed = _overwrite(history, table, NDVI_COLUMNS, days)
    _print_changes(changed, len(history))
    return history


def relabel_history(
    history: pd.DataFrame, official: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rewrite the measurements of *history* from the official station series.

    *official* is the station's long frame (``date``, ``species``, ``value``).
    The history keeps its windows and every non-measurement column exactly as
    they are; only what counts as a measurement changes:

    * a window the station reported gets one row per species with the
      station's value (0 for a taxon it did not list in that window);
    * a window the history has but the station did not report keeps its
      weather as a single weather-only row — its values came from somewhere
      else and are dropped;
    * a station window the history has no weather for is left out, and
      counted, since a row cannot be written without weather.

    Returns the new history and a report: windows relabelled, turned into
    weather only, missing weather, and values that changed.
    """
    if history.empty:
        return history, {}
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    carried = [c for c in history.columns if c not in ("date", "species", "value")]
    per_window = history.groupby("date")[carried].first()

    wide = (
        official[official["species"].isin(ALL_SPECIES)]
        .pivot_table(index="date", columns="species", values="value", aggfunc="mean")
        .reindex(columns=ALL_SPECIES)
        .fillna(0.0)
    )
    wide.index = pd.DatetimeIndex(wide.index)

    reported = per_window.index.intersection(wide.index)
    unreported = per_window.index.difference(wide.index)

    values = wide.loc[reported].stack().rename("value").reset_index()
    values.columns = ["date", "species", "value"]
    rows = values.join(per_window, on="date")
    weather_rows = per_window.loc[unreported].reset_index()
    weather_rows["species"] = WEATHER_ONLY
    weather_rows["value"] = np.nan

    out = pd.concat([rows, weather_rows], ignore_index=True)[list(history.columns)]
    out = out.sort_values(["date", "species"]).reset_index(drop=True)

    old = history[history["species"].isin(ALL_SPECIES)].set_index(["date", "species"])["value"]
    old = old[~old.index.duplicated(keep="last")]
    new = out[out["species"] != WEATHER_ONLY].set_index(["date", "species"])["value"]
    both = new.index.intersection(old.index)
    changed = int((~np.isclose(new.loc[both].to_numpy(float), old.loc[both].to_numpy(float),
                               atol=0.05, equal_nan=True)).sum())
    report = {
        "windows_reported": int(len(reported)),
        "windows_weather_only": int(len(unreported)),
        "station_windows_without_weather": int(len(wide.index.difference(per_window.index))),
        "values_changed": changed,
        "values_compared": int(len(both)),
    }
    return out, report
