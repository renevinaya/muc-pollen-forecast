"""
Data-driven pollen season onset.

Four model features are parameterised by *when the season is expected to
start*: ``days_since_typical_onset``, ``onset_anomaly``, ``gdd_above_threshold``
and ``cold_to_warm_flip``. They previously drew that parameter from two sources
that measurement showed to be unreliable:

  * ``data/phenology.csv`` — the DWD fetch yields a single year (2017) for
    Munich, with no Alnus at all, so ``typical_onset_doy`` was returning a
    one-observation mean for Corylus (56) against a measured median of 32.
  * ``SPECIES_GDD_THRESHOLD`` — hand-set constants that crossed 15–23 days
    *after* the observed onset, so the burst features opened their gate long
    after the season had begun.

This module derives both from the accumulated history instead, which now holds
eight complete seasons. Two rules keep it honest:

  * **Walk-forward.** Every estimate for year *Y* is calibrated only on years
    strictly before *Y*, so backtests cannot see their own answer.
  * **Causal within the year.** The per-row estimate uses prior-year
    climatology until *this* year's warmth actually crosses the threshold, and
    the observed crossing date afterwards. Training and serving therefore run
    the identical rule — at no point does a training row use a value the
    forecaster could not have computed on that date.

Two GDD accumulations appear here, deliberately:

  * ``ONSET_GDD_BASE`` (0 °C) drives the onset *projection*. Leave-one-out over
    the eight seasons put base 0 well ahead of the alternatives for Corylus
    (4.6 d mean absolute error vs 8.6 d at base 5).
  * ``GDD_T_BASE`` (5 °C) is what the existing ``gdd`` feature accumulates, so
    thresholds meant for comparison against that feature are calibrated in the
    same units. Changing the feature's base would perturb every model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import (
    SPECIES_GDD_THRESHOLD,
    SPECIES_THRESHOLDS,
    SPECIES_TYPICAL_ONSET_DOY,
    SPECIES_SEASON,
    GDD_T_BASE,
    _DEFAULT_GDD_THRESHOLD,
    _DEFAULT_THRESHOLDS,
    is_season_active,
)

# Base temperature for the onset projection (see module docstring).
ONSET_GDD_BASE = 0.0

# Onset = the first day of a run of this many consecutive days at or above the
# species' low/moderate boundary. A run rather than a single day so that one
# transported cloud does not open the season.
ONSET_RUN_DAYS = 3

# Calibrating a threshold on one prior year is not calibration. Below this many
# usable prior seasons we keep the static constants.
MIN_CALIBRATION_YEARS = 2

# --- Forcing rules for the onset projection ---------------------------------
#
# One rule — forcing from 1 January at base 0 °C — used to project every
# species' onset. It was chosen because it fit hazel (4.6 d leave-one-out),
# and it does not fit the others: for alder it is no better than the calendar
# (12.8 d vs 11.6 d climatology) and for birch it is twice as bad (20.8 d vs
# 9.0 d), because January warmth counts fully towards a tree that does not
# respond to it until March. So the rule is now chosen per species, at
# calibration time, by leave-one-out over the seasons before the year being
# projected: the (start date, base) pair with the smallest LOO error wins, and
# if none beats the climatology, the climatology is used on its own.

FORCING_STARTS: tuple[tuple[int, int], ...] = ((1, 1), (1, 15), (2, 1), (2, 15), (3, 1))
FORCING_BASES: tuple[float, ...] = (0.0, 3.0, 5.0)
ForcingRule = tuple[tuple[int, int], float]

# What ran before there was a choice, and what still runs when there are too
# few seasons to choose on: leave-one-out over three seasons is a coin toss.
DEFAULT_FORCING_RULE: ForcingRule = ((1, 1), ONSET_GDD_BASE)
MIN_RULE_SELECTION_YEARS = 4


def _fingerprint(history: pd.DataFrame) -> tuple:
    """Cheap identity for the cache below.

    Row count and date range alone would collide across species — each species'
    slice of the same history has both in common — which is harmless while the
    weather really is identical, but silently wrong if two slices cover
    different windows. The temperature sum separates those for one cheap pass.
    """
    if history.empty:
        return (0, None, None, 0.0)
    dates = history["date"]
    return (
        len(history),
        dates.iloc[0],
        dates.iloc[-1],
        float(history["temperature_mean"].sum()),
    )


_daily_temp_cache: dict[tuple, pd.Series] = {}


def daily_temperature(history: pd.DataFrame) -> pd.Series:
    """Mean temperature per calendar day, indexed by normalised timestamp.

    Weather is identical across species, so this is computed once per history
    and shared. Cached because the benchmark rebuilds features per species per
    fold and the input is a 200k-row frame.
    """
    key = _fingerprint(history)
    cached = _daily_temp_cache.get(key)
    if cached is not None:
        return cached

    days = pd.to_datetime(history["date"]).dt.normalize()
    daily = history.groupby(days)["temperature_mean"].mean().sort_index()
    daily = daily.dropna()
    _daily_temp_cache[key] = daily
    return daily


def cumulative_gdd(daily_temp: pd.Series, base: float) -> pd.Series:
    """Growing degree days accumulated from 1 January, reset each year."""
    contrib = (daily_temp - base).clip(lower=0)
    return contrib.groupby(pd.DatetimeIndex(daily_temp.index).year).cumsum()


def _daily_pollen(history: pd.DataFrame, species: str) -> pd.Series:
    """Mean measured concentration per calendar day for one species."""
    sp = history[history["species"] == species]
    if sp.empty:
        return pd.Series(dtype=float)
    days = pd.to_datetime(sp["date"]).dt.normalize()
    return sp.groupby(days)["value"].mean().sort_index()


def observed_onsets(history: pd.DataFrame, species: str) -> dict[int, int]:
    """Measured season start per year: ``{year: day_of_year}``.

    A year qualifies when the species reaches its low/moderate boundary on
    ``ONSET_RUN_DAYS`` consecutive days inside its core season window. The
    window requirement is what keeps long-range transport out — a February
    birch cloud over Munich is not Munich's birches flowering.
    """
    daily = _daily_pollen(history, species)
    if daily.empty:
        return {}

    threshold = SPECIES_THRESHOLDS.get(species, _DEFAULT_THRESHOLDS)[0]
    idx = pd.DatetimeIndex(daily.index)
    frame = pd.DataFrame(
        {"value": daily.values, "year": idx.year, "doy": idx.dayofyear, "month": idx.month},
        index=idx,
    )
    frame = frame[[is_season_active(species, m) for m in frame["month"]]]

    onsets: dict[int, int] = {}
    for year, grp in frame.groupby("year"):
        grp = grp.sort_values("doy")
        above = (grp["value"] >= threshold).to_numpy()
        doys = grp["doy"].to_numpy()
        for i in range(len(above) - ONSET_RUN_DAYS + 1):
            if above[i : i + ONSET_RUN_DAYS].all():
                onsets[int(year)] = int(doys[i])
                break
    return onsets


def _gdd_at_onsets(history: pd.DataFrame, species: str, base: float) -> dict[int, float]:
    """Accumulated GDD on each year's observed onset day."""
    onsets = observed_onsets(history, species)
    if not onsets:
        return {}
    gdd = cumulative_gdd(daily_temperature(history), base)
    at: dict[int, float] = {}
    for year, doy in onsets.items():
        day = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        if day in gdd.index:
            at[year] = float(gdd.loc[day])
    return at


def _median_before(values: dict[int, float], before_year: int | None) -> float | None:
    """Median of the entries for years strictly before *before_year*."""
    if before_year is None:
        usable = list(values.values())
    else:
        usable = [v for y, v in values.items() if y < before_year]
    if len(usable) < MIN_CALIBRATION_YEARS:
        return None
    return float(np.median(usable))


def calibrated_gdd_threshold(
    history: pd.DataFrame, species: str, before_year: int | None = None
) -> float:
    """GDD at which *species* has historically started, in feature units.

    Accumulated at :data:`GDD_T_BASE` so the result is directly comparable with
    the ``gdd`` column the trainer builds. Falls back to the static constant
    when there are too few prior seasons to calibrate on.
    """
    measured = _median_before(_gdd_at_onsets(history, species, GDD_T_BASE), before_year)
    if measured is None:
        return float(SPECIES_GDD_THRESHOLD.get(species, _DEFAULT_GDD_THRESHOLD))
    return measured


def climatological_onset_doy(
    history: pd.DataFrame, species: str, before_year: int | None = None
) -> float:
    """Median observed onset day-of-year, or the static baseline if unmeasurable."""
    onsets = {y: float(d) for y, d in observed_onsets(history, species).items()}
    measured = _median_before(onsets, before_year)
    if measured is not None:
        return measured
    return _static_onset_doy(species)


def _static_onset_doy(species: str) -> float:
    """Central-European baseline, then the 15th of the season's first month."""
    if species in SPECIES_TYPICAL_ONSET_DOY:
        return float(SPECIES_TYPICAL_ONSET_DOY[species])
    import calendar

    window = SPECIES_SEASON.get(species)
    if window is None:
        return float("nan")
    return float(
        sum(calendar.monthrange(2025, m)[1] for m in range(1, window[0])) + 15
    )


def forcing_series(daily_temp: pd.Series, rule: ForcingRule) -> pd.Series:
    """Forcing accumulated per year from the rule's start date at its base.

    Zero before the start date within each year, so a crossing can never be
    found before the rule says accumulation begins.
    """
    (month, day), base = rule
    idx = pd.DatetimeIndex(daily_temp.index)
    contrib = (daily_temp - base).clip(lower=0)
    started = (idx.month > month) | ((idx.month == month) & (idx.day >= day))
    contrib = contrib.where(started, 0.0)
    return contrib.groupby(idx.year).cumsum()


def _crossing_doy(year_forcing: pd.Series, threshold: float) -> float | None:
    crossed = (year_forcing >= threshold).to_numpy()
    if not crossed.any():
        return None
    return float(pd.Timestamp(year_forcing.index[int(np.argmax(crossed))]).dayofyear)


def _forcing_at(forcing: pd.Series, onsets: dict[int, int]) -> dict[int, float]:
    out: dict[int, float] = {}
    for year, doy in onsets.items():
        day = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        if day in forcing.index:
            out[year] = float(forcing.loc[day])
    return out


def _rule_loo_error(forcing: pd.Series, onsets: dict[int, int]) -> float:
    """Mean absolute leave-one-out projection error of one rule, in days.

    Each year is projected from the median forcing-at-onset of the *other*
    years; a year the rule never crosses counts as a full-season miss.
    """
    at = _forcing_at(forcing, onsets)
    years = pd.DatetimeIndex(forcing.index).year
    errors: list[float] = []
    for year, doy in onsets.items():
        others = [v for y, v in at.items() if y != year]
        if len(others) < 2:
            continue
        crossing = _crossing_doy(forcing[years == year], float(np.median(others)))
        errors.append(abs(crossing - doy) if crossing is not None else 120.0)
    return float(np.mean(errors)) if errors else float("inf")


def _climatology_loo_error(onsets: dict[int, int]) -> float:
    errors = [
        abs(float(np.median([v for y, v in onsets.items() if y != year])) - doy)
        for year, doy in onsets.items()
        if len(onsets) > 1
    ]
    return float(np.mean(errors)) if errors else float("inf")


def select_forcing_rule(
    history: pd.DataFrame, species: str, before_year: int | None = None
) -> tuple[ForcingRule | None, float | None, float]:
    """The projection rule for *species*, calibrated on seasons before *before_year*.

    Returns ``(rule, threshold, loo_error)``. ``rule`` is None when the
    climatology beats every rule on leave-one-out — the projection is then
    switched off for that species and year. With fewer than
    :data:`MIN_RULE_SELECTION_YEARS` prior seasons the default rule is used,
    exactly as before there was a choice; with fewer than
    :data:`MIN_CALIBRATION_YEARS` there is no threshold either.
    """
    onsets = {
        y: d for y, d in observed_onsets(history, species).items()
        if before_year is None or y < before_year
    }
    daily_temp = daily_temperature(history)
    if daily_temp.empty or len(onsets) < MIN_CALIBRATION_YEARS:
        return DEFAULT_FORCING_RULE, None, float("inf")

    if len(onsets) < MIN_RULE_SELECTION_YEARS:
        forcing = forcing_series(daily_temp, DEFAULT_FORCING_RULE)
        at = _forcing_at(forcing, onsets)
        threshold = float(np.median(list(at.values()))) if len(at) >= MIN_CALIBRATION_YEARS else None
        return DEFAULT_FORCING_RULE, threshold, _rule_loo_error(forcing, onsets)

    best_rule: ForcingRule | None = None
    best_error = _climatology_loo_error(onsets)
    best_forcing: pd.Series | None = None
    for start in FORCING_STARTS:
        for base in FORCING_BASES:
            rule: ForcingRule = (start, base)
            forcing = forcing_series(daily_temp, rule)
            error = _rule_loo_error(forcing, onsets)
            if error < best_error:
                best_rule, best_error, best_forcing = rule, error, forcing

    if best_rule is None or best_forcing is None:
        return None, None, best_error
    threshold = float(np.median(list(_forcing_at(best_forcing, onsets).values())))
    return best_rule, threshold, best_error


def describe_forcing_rule(rule: ForcingRule | None) -> str:
    if rule is None:
        return "climatology"
    (month, day), base = rule
    return f"from {month:02d}-{day:02d} base {base:g}"


def onset_doy_by_day(history: pd.DataFrame, species: str) -> pd.Series:
    """Causal onset estimate for every day in *history*, indexed by day.

    For each day the answer is the best estimate available *on that day*:

      * prior-year climatology, until this year's forcing — accumulated under
        the rule chosen for this species on the seasons before this year —
        reaches the threshold calibrated on those same seasons;
      * the actual crossing day-of-year from then on;
      * the climatology throughout, for a species-year whose rule selection
        found nothing better than the calendar.

    The switch is what carries the new information — it tells the model the
    season is running early or late as soon as the warmth confirms it, and it
    is the same rule the forecaster applies, so there is no train/serve skew.
    """
    daily_temp = daily_temperature(history)
    if daily_temp.empty:
        return pd.Series(dtype=float)

    onsets = {y: float(d) for y, d in observed_onsets(history, species).items()}
    idx = pd.DatetimeIndex(daily_temp.index)
    estimate = pd.Series(np.nan, index=idx, dtype=float)

    for year in sorted(set(idx.year)):
        mask = idx.year == year
        climatology = _median_before(onsets, year)
        if climatology is None:
            climatology = _static_onset_doy(species)
        year_est = np.full(int(mask.sum()), climatology, dtype=float)

        rule, threshold, _ = select_forcing_rule(history, species, before_year=year)
        if rule is not None and threshold is not None:
            year_forcing = forcing_series(daily_temp, rule)[mask]
            crossed = (year_forcing >= threshold).to_numpy()
            if crossed.any():
                crossing_day = pd.Timestamp(year_forcing.index[int(np.argmax(crossed))])
                # Only from the crossing onwards — before it, we did not know.
                year_est[crossed] = float(crossing_day.dayofyear)

        estimate[mask] = year_est

    return estimate


def onset_doy_for_date(history: pd.DataFrame, species: str, when: pd.Timestamp) -> float:
    """Causal onset estimate for a single date, for use at forecast time.

    Falls back to the nearest earlier day when *when* is beyond the history
    (the forecaster runs several days ahead of the last observation).
    """
    day = pd.Timestamp(when).normalize()
    return onset_doy_lookup(
        onset_doy_by_day(history, species),
        day,
        climatological_onset_doy(history, species, before_year=day.year),
    )


def gdd_threshold_by_year(history: pd.DataFrame, species: str) -> dict[int, float]:
    """Per-year GDD threshold in feature units, calibrated walk-forward.

    Year *Y* gets the median GDD-at-onset of the seasons before it, so a
    backtest of year *Y* is never scored against a threshold that saw year
    *Y*'s answer. Years without enough history behind them keep the static
    constant. The year after the history ends is included so a forecast run
    crossing New Year still finds a threshold.
    """
    static = float(SPECIES_GDD_THRESHOLD.get(species, _DEFAULT_GDD_THRESHOLD))
    at_onset = _gdd_at_onsets(history, species, GDD_T_BASE)
    if history.empty:
        return {}

    years = pd.DatetimeIndex(pd.to_datetime(history["date"])).year
    span = range(int(years.min()), int(years.max()) + 2)
    return {
        year: (measured if (measured := _median_before(at_onset, year)) is not None else static)
        for year in span
    }


def gdd_threshold_for_year(
    thresholds: dict[int, float], year: int, species: str
) -> float:
    """One year's threshold, falling back to the static constant."""
    return float(
        thresholds.get(int(year), SPECIES_GDD_THRESHOLD.get(species, _DEFAULT_GDD_THRESHOLD))
    )


def gdd_threshold_series(
    thresholds: dict[int, float], index: pd.DatetimeIndex, species: str
) -> pd.Series:
    """Expand a per-year threshold map over a datetime index."""
    return pd.Series(
        [gdd_threshold_for_year(thresholds, year, species) for year in index.year],
        index=index,
        dtype=float,
    )


def onset_doy_lookup(series: pd.Series, when: pd.Timestamp, fallback: float) -> float:
    """Onset estimate for *when* from a precomputed series.

    Forecast windows run past the last observation, so a date beyond the series
    resolves to the most recent day we do have — which is what was known when
    the forecast was made.
    """
    if series.empty:
        return fallback
    day = pd.Timestamp(when).normalize()
    if day in series.index:
        return float(series.loc[day])
    earlier = series[series.index <= day]
    if not earlier.empty:
        return float(earlier.iloc[-1])
    return fallback
