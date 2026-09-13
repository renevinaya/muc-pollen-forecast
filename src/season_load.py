"""Interannual season load: how heavy the previous seasons were.

Birch and oak mast; alder and hazel alternate less regularly; every tree
species here varies several-fold between years (Betula: 2,842 grains·days in
2025, 24,640 in 2026). Nothing else in the feature set crosses a season
boundary, so at onset — when the lag block is all zeros by definition — the
model has no way to know what kind of year it is entering. Phase A showed it
had been reading one off a data artefact instead: eight weather columns were
missing before March 2025, and "present" doubled as "recent, heavy years".
These features are the honest version of that signal.

Three features per species, all expressed as *log ratios against the species'
own history* so that 0 means "average, or unknown" and the trainer's NaN→0
fill is neutral rather than a claim:

* ``load_prev_anom`` — last season's total against the mean of the seasons
  before it (heavy or light year just gone);
* ``load_2y_anom`` — the last two seasons against the seasons before them
  (a slower trend, what the Phase A artefact was really encoding);
* ``load_trend`` — last season against the one before (alternation).

Causality: a season year runs from one boundary month to the next, and the
boundary sits after the species' season *and* its shoulder month. A row in
season year *Y* only ever reads totals of seasons strictly before *Y*, every
one of which ended before the row's date, and a season only counts once the
history reaches its boundary. The trainer and the forecaster call the same
two functions on their own history, so there is nothing to keep in step.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .types import LOAD_FEATURES, SEASON_SHOULDER_MONTHS, SPECIES_SEASON

# A season counts as observed when this share of the days in its core months
# carry measurements. Below that the total is a fragment, not a season.
MIN_SEASON_COVERAGE = 0.8

_DEFAULT_BOUNDARY_MONTH = 1


def boundary_month(species: str) -> int:
    """First month of the next season year: the month after the shoulder."""
    window = SPECIES_SEASON.get(species)
    if window is None:
        return _DEFAULT_BOUNDARY_MONTH
    return (window[1] + SEASON_SHOULDER_MONTHS) % 12 + 1


def season_year(species: str, dates: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """The season year each date belongs to, named for the year the season is in.

    Dates from the boundary month onwards belong to the *next* year's season,
    so a warm-December hazel start and the February peak that follows it land
    in the same season year.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    years = idx.year.to_numpy()
    return np.where(idx.month.to_numpy() >= boundary_month(species), years + 1, years)


def _core_days(species: str, year: int) -> int:
    start, end = SPECIES_SEASON.get(species, (1, 12))
    first = pd.Timestamp(year=year, month=start, day=1)
    last = pd.Timestamp(year=year, month=end, day=1) + pd.offsets.MonthEnd(0)
    return (last - first).days + 1


def season_totals(history: pd.DataFrame, species: str) -> dict[int, float]:
    """Total pollen (sum of daily means) per completed, observed season year.

    Completed: the history reaches the season's boundary. Observed: at least
    :data:`MIN_SEASON_COVERAGE` of the core-season days have a measurement.
    The season the history ends inside is never returned.
    """
    if history.empty:
        return {}
    sp = history[history["species"] == species]
    if sp.empty:
        return {}

    dates = pd.to_datetime(sp["date"])
    days = dates.dt.normalize()
    daily = sp.groupby(days)["value"].mean()
    day_index = pd.DatetimeIndex(daily.index)
    years = season_year(species, day_index)

    last_day = pd.to_datetime(history["date"]).max()
    boundary = boundary_month(species)
    start_m, end_m = SPECIES_SEASON.get(species, (1, 12))

    totals: dict[int, float] = {}
    for year in np.unique(years):
        year = int(year)
        # The boundary that closes season year *year* is boundary_month of
        # that same year (or of the following year when the boundary wraps
        # to January, i.e. the season runs to December).
        closes = pd.Timestamp(year=year if boundary > 1 else year + 1, month=boundary, day=1)
        if last_day < closes:
            continue
        in_year = years == year
        core = in_year & (day_index.month >= start_m) & (day_index.month <= end_m)
        if core.sum() < MIN_SEASON_COVERAGE * _core_days(species, year):
            continue
        totals[year] = float(daily.to_numpy()[in_year].sum())
    return totals


def load_features_for_years(
    totals: dict[int, float], years: Iterable[int]
) -> dict[int, dict[str, float]]:
    """The three load features for each season year in *years*.

    Every value is a difference of ``log1p`` totals, and every value a season
    year cannot yet have (no previous season, no baseline before that) is 0.
    """
    logs = {y: float(np.log1p(t)) for y, t in totals.items()}

    def mean_before(year: int) -> float | None:
        earlier = [v for y, v in logs.items() if y < year]
        return float(np.mean(earlier)) if earlier else None

    out: dict[int, dict[str, float]] = {}
    for year in years:
        year = int(year)
        prev = logs.get(year - 1)
        prev2 = logs.get(year - 2)
        row = {name: 0.0 for name in LOAD_FEATURES}
        if prev is not None:
            base = mean_before(year - 1)
            if base is not None:
                row["load_prev_anom"] = prev - base
            if prev2 is not None:
                row["load_trend"] = prev - prev2
                base2 = mean_before(year - 2)
                if base2 is not None:
                    row["load_2y_anom"] = (prev + prev2) / 2.0 - base2
        out[year] = row
    return out


def load_feature_frame(
    history: pd.DataFrame, species: str, dates: pd.DatetimeIndex | pd.Series
) -> pd.DataFrame:
    """Load features for every date, computed from *history* — one row each."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    years = season_year(species, idx)
    per_year = load_features_for_years(season_totals(history, species), np.unique(years))
    rows = [per_year[int(y)] for y in years]
    return pd.DataFrame(rows, index=idx, columns=LOAD_FEATURES)
