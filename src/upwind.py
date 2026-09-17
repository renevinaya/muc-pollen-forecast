"""Upwind pollen stations: what is in the air *before* it reaches Munich.

Pre-onset pollen is transport. In 2026 Munich measured 15–24 birch grains/m³
on 27 Feb – 5 Mar, five weeks before the local trees opened, and the first two
weeks of every heavy season are forecast at a tenth to a third of the truth,
because nothing the model reads precedes the local count. Stations 50–200 km
away see the same air mass hours to a day earlier.

pollenscience.eu serves the other stations of the Bavarian ePIN network
through the same endpoint as Munich. Their measurements are kept in a second,
long-format file (``data/upwind.csv``: date, station, species, value) that is
appended on every run and backed up to the data release beside the history,
so they never touch the history's own columns.

The features are lag-like: they describe the upwind stations as of the
forecast origin, and are anchored there for every lead exactly as the local
lag block is. They are built on the *time* grid rather than by row, so a
station outage shifts nothing.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"
UPWIND_FILE = DATA_DIR / "upwind.csv"
UPWIND_ASSET = "upwind.csv.gz"

# pollenscience.eu station codes read as upwind context for Munich. The ePIN
# network's automatic stations (3 h resolution, 2019+) within ~150 km, one in
# each direction so whichever way the air arrives from, a station saw it
# first. Garmisch (75 km S, alpine, flowers later) and the two stations
# 250 km north (Hof, Marktheidenfeld) are left out.
UPWIND_STATIONS: dict[str, str] = {
    "DEMIND": "Mindelheim (85 km WSW)",
    "DEALTO": "Altötting (85 km E)",
    "DEFEUC": "Feucht (140 km N)",
    "DEVIEC": "Viechtach (140 km NE)",
}

WINDOW = pd.Timedelta(hours=3)
# 24 h and 7 d in 3h windows — the same spans the local lag block uses.
SHORT_WINDOWS = 8
LONG_WINDOWS = 56

_CHUNK_DAYS = 28
_COLUMNS = ["date", "station", "species", "value"]


# --- Fetching ---------------------------------------------------------------


def fetch_upwind(
    start: date, end: date, stations: list[str] | None = None
) -> pd.DataFrame:
    """Measurements of every upwind station between *start* and *end*."""
    from .pollenscience import _fetch_single_location

    parts: list[pd.DataFrame] = []
    for code in stations or list(UPWIND_STATIONS):
        df = _fetch_single_location(start, end, code)
        if not df.empty:
            df = df.assign(station=code)
            parts.append(df[_COLUMNS])
    if not parts:
        return pd.DataFrame(columns=_COLUMNS)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby(["date", "station", "species"], as_index=False)["value"].max()
    return combined.sort_values(["date", "station", "species"]).reset_index(drop=True)


def fetch_upwind_chunked(
    start: date, end: date, stations: list[str] | None = None, delay: float = 5.0
) -> pd.DataFrame:
    """Backfill: fetch in 28-day chunks with a pause between requests."""
    codes = stations or list(UPWIND_STATIONS)
    chunks: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS - 1), end)
        print(f"  Fetching {chunk_start} to {chunk_end} ({', '.join(codes)})...")
        for code in codes:
            try:
                df = fetch_upwind(chunk_start, chunk_end, [code])
            except Exception as exc:  # noqa: BLE001 — one bad chunk must not end the backfill
                print(f"    {code}: error {exc}, skipping chunk")
                df = pd.DataFrame(columns=_COLUMNS)
            if not df.empty:
                chunks.append(df)
                print(f"    {code}: {len(df)} rows, {int((df['value'] > 0).sum())} nonzero")
            time.sleep(delay)
        chunk_start = chunk_end + timedelta(days=1)
    if not chunks:
        return pd.DataFrame(columns=_COLUMNS)
    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "station", "species"], keep="last")
    return combined.sort_values(["date", "station", "species"]).reset_index(drop=True)


# --- Storage ----------------------------------------------------------------


def load_upwind(path: Path = UPWIND_FILE) -> pd.DataFrame:
    """The stored upwind measurements, or an empty frame when there are none."""
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(path, parse_dates=["date"])
    return df[_COLUMNS]


def update_upwind(new_data: pd.DataFrame, path: Path = UPWIND_FILE) -> pd.DataFrame:
    """Merge *new_data* into the stored file, the latest value winning."""
    existing = load_upwind(path)
    if new_data.empty:
        return existing
    combined = pd.concat([existing, new_data[_COLUMNS]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "station", "species"], keep="last")
    combined = combined.sort_values(["date", "station", "species"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    print(f"Updated upwind measurements: {len(combined)} rows -> {path}")
    return combined


def sync_upwind(path: Path = UPWIND_FILE) -> pd.DataFrame:
    """Load the upwind file, restoring it from the data release when absent."""
    from .store import download_csv

    if not path.exists():
        download_csv(UPWIND_ASSET, path)
    return load_upwind(path)


# --- Features ---------------------------------------------------------------


def upwind_series(upwind: pd.DataFrame | None, species: str) -> pd.Series:
    """log1p of the highest upwind reading per 3h window, on a full time grid.

    Windows no station reported are NaN, so a gap stays a gap instead of
    becoming "nothing in the air".
    """
    if upwind is None or upwind.empty:
        return pd.Series(dtype=float)
    sp = upwind[upwind["species"] == species]
    if sp.empty:
        return pd.Series(dtype=float)
    per_window = sp.groupby("date")["value"].max().sort_index()
    grid = pd.date_range(per_window.index.min(), per_window.index.max(), freq=WINDOW)
    return np.log1p(per_window.reindex(grid).astype(float))


def upwind_block(series: pd.Series, dates: pd.Series | pd.DatetimeIndex, lead: int) -> pd.DataFrame:
    """The upwind features for *dates*, anchored *lead* windows before each.

    Mirrors the local lag block: at ``lead=1`` the features describe the 24 h
    and 7 d ending one window before the target; at ``lead=L`` the same spans
    ending L windows before it.
    """
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    columns = ["upwind_max_8", "upwind_max_56"]
    if series.empty or len(index) == 0:
        return pd.DataFrame(np.nan, index=index, columns=columns)
    # Carry the grid past the last reading so a target after the station's
    # last report still sees the 7 days before its origin, as serving would.
    grid = pd.date_range(series.index.min(), max(series.index.max(), index.max()), freq=WINDOW)
    on_grid = series.reindex(grid)
    short = on_grid.rolling(SHORT_WINDOWS, min_periods=1).max().shift(lead)
    long = on_grid.rolling(LONG_WINDOWS, min_periods=1).max().shift(lead)
    return pd.DataFrame(
        {"upwind_max_8": short.reindex(index).to_numpy(),
         "upwind_max_56": long.reindex(index).to_numpy()},
        index=index,
    )


def with_lead_feature(block: pd.DataFrame, local_max_8: pd.Series | np.ndarray | float) -> pd.DataFrame:
    """Add ``upwind_lead_8``: how far the upwind 24 h maximum sits above Munich's."""
    block = block.copy()
    block["upwind_lead_8"] = block["upwind_max_8"] - np.asarray(local_max_8, dtype=float)
    return block


def upwind_state(upwind: pd.DataFrame | None, species: str, origin: pd.Timestamp) -> dict[str, float]:
    """The upwind features as of *origin*, from readings strictly before it."""
    if upwind is None or upwind.empty:
        return {"upwind_max_8": float("nan"), "upwind_max_56": float("nan")}
    origin = pd.Timestamp(origin)
    sp = upwind[(upwind["species"] == species)]
    dates = pd.to_datetime(sp["date"])
    recent = sp[(dates < origin) & (dates >= origin - LONG_WINDOWS * WINDOW)]
    if recent.empty:
        return {"upwind_max_8": float("nan"), "upwind_max_56": float("nan")}
    rdates = pd.to_datetime(recent["date"])
    short = recent[rdates >= origin - SHORT_WINDOWS * WINDOW]
    return {
        "upwind_max_8": float(np.log1p(short["value"].max())) if not short.empty else float("nan"),
        "upwind_max_56": float(np.log1p(recent["value"].max())),
    }
