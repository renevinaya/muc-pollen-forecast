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
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .season_load import boundary_month, season_year


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

MAX_COLUMNS = ["upwind_max_8", "upwind_max_56"]
SEASON_COLUMNS = ["upwind_season_sum", "upwind_season_anom"]


@dataclass
class UpwindTables:
    """One species' upwind readings as lookup tables on the 3h grid.

    ``series`` is log1p of the highest reading at any station per window, NaN
    where nobody reported. ``season`` holds the season-load pair per window:
    the log1p sum of the readings since the season-year start, and that sum
    against the median of the earlier season years at the same day of the
    season (0 while there is no earlier year). Both are as of the window's
    end, so the value at window ``t`` is what is known one window later.
    """

    series: pd.Series
    season: pd.DataFrame

    @property
    def empty(self) -> bool:
        return self.series.empty


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


def _season_start(species: str, years: np.ndarray) -> pd.DatetimeIndex:
    """First day of each season year: the boundary month of the year before."""
    month = boundary_month(species)
    start_year = years if month == 1 else years - 1
    return pd.to_datetime(
        {"year": start_year, "month": np.full(len(years), month), "day": np.ones(len(years), dtype=int)}
    )


def season_frame(series: pd.Series, species: str) -> pd.DataFrame:
    """The season-load pair per window of *series* (see :class:`UpwindTables`)."""
    if series.empty:
        return pd.DataFrame(columns=SEASON_COLUMNS, dtype=float)
    index = pd.DatetimeIndex(series.index)
    linear = np.expm1(series).fillna(0.0)  # an unreported window adds nothing
    years = season_year(species, index)
    cum = linear.groupby(years).cumsum()
    season_sum = np.log1p(cum)

    # Reference: the median cumulative sum of the earlier season years at the
    # same 3h slot of the same day of the season.
    day = (index.normalize() - pd.DatetimeIndex(_season_start(species, years))).days
    slot = day.to_numpy() * 8 + index.hour.to_numpy() // 3
    per_slot = pd.DataFrame({"year": years, "slot": slot, "cum": cum.to_numpy()})
    table = per_slot.groupby(["slot", "year"])["cum"].max().unstack("year").sort_index()
    reference = pd.DataFrame(index=table.index, columns=table.columns, dtype=float)
    for i, year in enumerate(table.columns):
        if i:
            reference[year] = table.iloc[:, :i].median(axis=1)
    ref_at = reference.stack(future_stack=True)
    keys = pd.MultiIndex.from_arrays([slot, years])
    ref_values = ref_at.reindex(keys).to_numpy(dtype=float)
    anomaly = np.where(np.isnan(ref_values), 0.0, season_sum.to_numpy() - np.log1p(ref_values))
    return pd.DataFrame(
        {"upwind_season_sum": season_sum.to_numpy(), "upwind_season_anom": anomaly}, index=index
    )


_TABLES_CACHE: dict[tuple[int, int, str], UpwindTables] = {}


def upwind_tables(upwind: pd.DataFrame | None, species: str) -> UpwindTables:
    """The lookup tables for *species*, cached per upwind frame.

    The rollout seeds one lag state per origin and the tables cover eight
    years, so they are built once per (frame, species) and reused.
    """
    if upwind is None or upwind.empty:
        return UpwindTables(pd.Series(dtype=float), pd.DataFrame(columns=SEASON_COLUMNS, dtype=float))
    key = (id(upwind), len(upwind), species)
    tables = _TABLES_CACHE.get(key)
    if tables is None:
        if len(_TABLES_CACHE) > 64:
            _TABLES_CACHE.clear()
        series = upwind_series(upwind, species)
        tables = UpwindTables(series, season_frame(series, species))
        _TABLES_CACHE[key] = tables
    return tables


def upwind_block(tables: UpwindTables, dates: pd.Series | pd.DatetimeIndex, lead: int) -> pd.DataFrame:
    """The upwind features for *dates*, anchored *lead* windows before each.

    Mirrors the local lag block: at ``lead=1`` the features describe the state
    one window before the target; at ``lead=L`` the state L windows before it.
    """
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    columns = MAX_COLUMNS + SEASON_COLUMNS
    if tables.empty or len(index) == 0:
        return pd.DataFrame(np.nan, index=index, columns=columns)
    series = tables.series
    # Carry the grid past the last reading so a target after the station's
    # last report still sees the state before its origin, as serving would.
    grid = pd.date_range(series.index.min(), max(series.index.max(), index.max()), freq=WINDOW)
    on_grid = series.reindex(grid)
    short = on_grid.rolling(SHORT_WINDOWS, min_periods=1).max().shift(lead)
    long = on_grid.rolling(LONG_WINDOWS, min_periods=1).max().shift(lead)
    season = tables.season.reindex(grid).ffill().shift(lead)
    return pd.DataFrame(
        {
            "upwind_max_8": short.reindex(index).to_numpy(),
            "upwind_max_56": long.reindex(index).to_numpy(),
            "upwind_season_sum": season["upwind_season_sum"].reindex(index).to_numpy(),
            "upwind_season_anom": season["upwind_season_anom"].reindex(index).to_numpy(),
        },
        index=index,
    )


def upwind_state(tables: UpwindTables, origin: pd.Timestamp) -> dict[str, float]:
    """The upwind features as of *origin*, from readings strictly before it."""
    nan = float("nan")
    if tables.empty:
        return {c: nan for c in MAX_COLUMNS + SEASON_COLUMNS}
    origin = pd.Timestamp(origin)
    last = origin - WINDOW
    series = tables.series
    short = series.loc[origin - SHORT_WINDOWS * WINDOW : last]
    long = series.loc[origin - LONG_WINDOWS * WINDOW : last]
    if last < series.index.min():
        season = {c: nan for c in SEASON_COLUMNS}
    else:
        row = tables.season.reindex([last], method="ffill").iloc[0]
        season = {c: float(row[c]) for c in SEASON_COLUMNS}
    return {
        "upwind_max_8": float(short.max()) if short.notna().any() else nan,
        "upwind_max_56": float(long.max()) if long.notna().any() else nan,
        **season,
    }


def with_local_features(
    block: pd.DataFrame,
    local_max_8: pd.Series | np.ndarray | float,
    local_season_sum: pd.Series | np.ndarray | float,
) -> pd.DataFrame:
    """Add the features that set the upwind block against Munich's own.

    ``upwind_lead_8`` is how far the upwind 24 h maximum sits above Munich's;
    ``pollen_season_sum`` is Munich's own season-to-date sum (log1p) and
    ``upwind_season_lead`` how far the region's is above it.
    """
    block = block.copy()
    block["upwind_lead_8"] = block["upwind_max_8"] - np.asarray(local_max_8, dtype=float)
    block["pollen_season_sum"] = np.asarray(local_season_sum, dtype=float)
    block["upwind_season_lead"] = block["upwind_season_sum"] - block["pollen_season_sum"]
    return block


def local_season_sum(values: np.ndarray, years: np.ndarray, lead: int) -> np.ndarray:
    """log1p of the sum of *values* over the origin's season year, as of the origin.

    Row-based like the lag block: the origin of row *i* is row ``i - lead + 1``
    and the sum runs over the rows before it that share its season year, so it
    restarts at 0 on the first window of a season year.
    """
    v = pd.Series(np.asarray(values, dtype=float))
    y = pd.Series(np.asarray(years, dtype=float))
    cum = v.groupby(y.to_numpy()).cumsum().shift(lead).to_numpy()
    origin_year = y.shift(lead - 1).to_numpy()
    last_year = y.shift(lead).to_numpy()
    out = np.where(origin_year == last_year, cum, 0.0)
    out[np.isnan(last_year)] = np.nan  # no row before the origin at all, like the lags
    return np.log1p(out)
