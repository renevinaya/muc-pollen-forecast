"""The season outlook: when a species' season is expected to start.

The forecast itself cannot say this. It publishes values for a few days, a
species below the emit threshold is simply absent from it, and a start counts
as one only after a run of days above the low level — so even a perfect
7-day forecast could announce a start four days ahead at most. The start
date is a different question with its own answer.

The answer is thermal. A species' onset candidates are forcing rules — warmth
accumulated from a start date above a base temperature, to a threshold
calibrated on earlier seasons — and the calendar (:mod:`src.onset`). For any
day, each rule can say when it will cross:

* up to that day it has the temperatures that were measured;
* for the days after, it has the weather forecast, which the caller appends
  to the temperature series;
* beyond the forecast it adds what that part of the year has contributed on
  average in the years before (per day of year, smoothed over a fortnight).
  The average is taken over the *contributions*, not over the temperatures:
  a January that averages 0 °C still has days at 6 °C, and a rule with a
  base of 3 °C gains from them.

The projected start is the median of the ensemble's projected crossings,
exactly as the measured estimate is the median of its actual ones. It is
causal by construction — nothing later than the day asked about goes in —
and it approaches the crossing continuously instead of jumping to it.

How wrong it is, is measured rather than assumed: :func:`replay` issues the
projection on every day of the six weeks before each measured start, from
the seasons before it only, and the residuals of those replays — pooled over
species, by how far ahead the projected start lay — give the interval and
the probability of a start within a week that are published with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .onset import (
    Candidate,
    ForcingRule,
    MIN_CALIBRATION_YEARS,
    MIN_RULE_SELECTION_YEARS,
    _static_onset_doy,
    daily_temperature,
    ensemble_crossing_doy,
    forcing_series,
    observed_onsets,
    onset_ensemble,
)
from .types import ALL_SPECIES, SEASON_SHOULDER_MONTHS, SPECIES_SEASON, SPORE_TAXA

# Days of weather forecast the projection is given beyond the day it is
# issued on. The weather API serves sixteen; a week is what its temperature
# is good for (daily mean error 0.5 °C at day 1, 2.0 °C at day 7 on the
# archived forecasts of January-June 2024, and a crossing projected seven
# days ahead with them lands within a day of the one computed from the
# measured temperatures).
OUTLOOK_FORECAST_DAYS = 7

# Years of temperature needed before a part of the year has an "average
# contribution" to project with. With fewer, a rule can only vote once it
# has actually crossed.
MIN_CLIMATE_YEARS = 2

# Smoothing of the per-day-of-year average contribution, in days.
CLIMATE_SMOOTHING_DAYS = 15

# The replay issues a projection on each of this many days before a start.
# Long enough that the far lead buckets are filled by projections that were
# too early as well as by ones that were too late.
REPLAY_DAYS = 90

# Residuals are pooled by how far ahead the projected start lies, in days:
# overdue, inside the weather forecast, the week after, the month, beyond.
LEAD_BUCKETS: tuple[tuple[float, float], ...] = (
    (-np.inf, 0), (1, 7), (8, 14), (15, 30), (31, np.inf),
)
MIN_BUCKET_SAMPLES = 30

# Central coverage of the published earliest-latest interval.
INTERVAL_COVERAGE = 0.8

INF = float("inf")


# --- The projection ----------------------------------------------------------


_increment_cache: dict[tuple, np.ndarray | None] = {}


def expected_increments(
    daily_temp: pd.Series, rule: ForcingRule, before_year: int
) -> np.ndarray | None:
    """What each day of the year adds to *rule*'s forcing, on average, before *before_year*.

    An array over day of year 1..366, zero before the rule's start date. None
    when fewer than :data:`MIN_CLIMATE_YEARS` earlier years have temperatures.
    """
    idx = pd.DatetimeIndex(daily_temp.index)
    key = (len(daily_temp), idx[0] if len(idx) else None, float(daily_temp.sum()), rule, before_year)
    if key in _increment_cache:
        return _increment_cache[key]

    (month, day), base = rule
    prior = daily_temp[idx.year < before_year]
    years = pd.DatetimeIndex(prior.index).year
    covered = [y for y, n in pd.Series(years).value_counts().items() if n >= 300]
    result: np.ndarray | None = None
    if len(covered) >= MIN_CLIMATE_YEARS:
        prior = prior[np.isin(years, covered)]
        contribution = (prior - base).clip(lower=0)
        by_doy = contribution.groupby(pd.DatetimeIndex(prior.index).dayofyear).mean()
        by_doy = by_doy.reindex(range(1, 367)).interpolate(limit_direction="both")
        by_doy = by_doy.rolling(CLIMATE_SMOOTHING_DAYS, center=True, min_periods=1).mean()
        start_doy = pd.Timestamp(year=2001, month=month, day=day).dayofyear
        values = by_doy.to_numpy(dtype=float).copy()
        values[: start_doy - 1] = 0.0
        result = values
    _increment_cache[key] = result
    return result


def _rule_votes(
    year_temp: pd.Series, rule: ForcingRule, threshold: float, expected: np.ndarray | None
) -> tuple[np.ndarray, float]:
    """*rule*'s vote on every day of *year_temp*, and its vote before the year begins.

    A vote is the day of year the rule reaches *threshold*: the actual
    crossing from the day it happens, and before that the day the forcing so
    far plus the *expected* contributions of the days ahead gets there.
    ``inf`` where the rule has not crossed and cannot be projected.
    """
    cumulative = np.cumsum(expected) if expected is not None else None

    def projected(have: np.ndarray, last_doy: np.ndarray) -> np.ndarray:
        if cumulative is None:
            return np.full(len(have), INF)
        already = np.where(last_doy >= 1, cumulative[np.maximum(last_doy, 1) - 1], 0.0)
        position = np.searchsorted(cumulative, threshold - have + already, side="left")
        return np.where(position >= len(cumulative), INF, position + 1.0)

    before_year = float(projected(np.zeros(1), np.zeros(1, dtype=int))[0])
    if len(year_temp) == 0:
        return np.zeros(0), before_year
    forcing = forcing_series(year_temp, rule).to_numpy(dtype=float)
    doys = pd.DatetimeIndex(year_temp.index).dayofyear.to_numpy()
    votes = projected(forcing, doys)
    crossed = forcing >= threshold
    if crossed.any():
        first = int(np.argmax(crossed))
        votes[first:] = float(doys[first])
    return votes, before_year


def _calendar_doy(onsets: dict[int, int], year: int, species: str) -> float:
    prior = [float(d) for y, d in onsets.items() if y < year]
    if len(prior) >= MIN_CALIBRATION_YEARS:
        return float(np.median(prior))
    return _static_onset_doy(species)


def project_year(
    daily_temp: pd.Series, ensemble: list[Candidate], calendar: float, year: int
) -> tuple[pd.Series, float]:
    """The ensemble's projected onset day of *year*, as of every day of that year.

    Returns a series indexed by the year's days in *daily_temp* — the
    projection knowing the temperatures through that day — and the
    projection made before the year has any (from the average contributions
    alone). Each is the median of the members' votes: a rule votes its
    crossing, actual once it has crossed and projected until then, and the
    calendar votes *calendar*. Where the median member cannot be projected
    the calendar stands in.
    """
    idx = pd.DatetimeIndex(daily_temp.index)
    year_temp = daily_temp[idx.year == year]
    if not ensemble:
        return pd.Series(float(calendar), index=year_temp.index), float(calendar)

    per_day: list[np.ndarray] = []
    before: list[float] = []
    for rule, threshold, _ in ensemble:
        if rule is None or threshold is None:
            per_day.append(np.full(len(year_temp), float(calendar)))
            before.append(float(calendar))
            continue
        votes, start = _rule_votes(
            year_temp, rule, threshold, expected_increments(daily_temp, rule, year)
        )
        per_day.append(votes)
        before.append(start)
    member = (len(ensemble) - 1) // 2
    median = np.sort(np.vstack(per_day), axis=0)[member] if len(year_temp) else np.zeros(0)
    median = np.where(np.isfinite(median), median, float(calendar))
    start = sorted(before)[member]
    return (
        pd.Series(median, index=year_temp.index),
        float(calendar) if start == INF else start,
    )


def project_onset(
    daily_temp: pd.Series,
    ensemble: list[Candidate],
    calendar: float,
    year: int,
    as_of: pd.Timestamp,
) -> float:
    """The ensemble's projected onset day of *year*, knowing temperatures through *as_of*."""
    as_of = pd.Timestamp(as_of).normalize()
    idx = pd.DatetimeIndex(daily_temp.index)
    series, before_year = project_year(daily_temp[idx <= as_of], ensemble, calendar, year)
    return float(series.iloc[-1]) if len(series) else before_year


def projected_onset_for(
    history: pd.DataFrame,
    species: str,
    year: int,
    as_of: pd.Timestamp,
    daily_temp: pd.Series | None = None,
    onsets: dict[int, int] | None = None,
) -> tuple[float, str]:
    """Projected onset day of *year* for *species*, and what it rests on.

    The basis is ``"forcing"`` when forcing rules are among the voters,
    ``"calendar"`` when only the median of earlier seasons is, and
    ``"baseline"`` for a species that has never been measured to start.
    """
    if daily_temp is None:
        daily_temp = daily_temperature(history)
    if onsets is None:
        onsets = observed_onsets(history, species)
    calendar = _calendar_doy(onsets, year, species)
    ensemble = onset_ensemble(history, species, before_year=year, onsets=onsets)
    if any(rule is not None for rule, _, _ in ensemble):
        basis = "forcing"
    elif sum(1 for y in onsets if y < year) >= MIN_CALIBRATION_YEARS:
        basis = "calendar"
    else:
        basis = "baseline"
    if daily_temp.empty:
        return calendar, basis
    return project_onset(daily_temp, ensemble, calendar, year, as_of), basis


# --- Replaying it on the measured seasons --------------------------------------


def replay(
    history: pd.DataFrame,
    species: list[str] | None = None,
    forecast_days: int = OUTLOOK_FORECAST_DAYS,
    replay_days: int = REPLAY_DAYS,
) -> pd.DataFrame:
    """The projection issued on each day before each measured start, walk-forward.

    For every season that has :data:`MIN_RULE_SELECTION_YEARS` seasons
    behind it, and every day ``issued`` of the *replay_days* before its
    start: the projection made from the earlier seasons only, with the
    temperatures up to ``issued`` plus *forecast_days* − 1 further days
    standing in for the weather forecast (the measured ones — on real
    forecasts the difference was within a day).

    Columns: ``species``, ``year``, ``lead`` (days from ``issued`` to the
    measured start), ``observed``, ``projected``, ``calendar``, ``crossing``
    (what was shipped before: the calendar until the ensemble has actually
    crossed), ``ahead`` (days from ``issued`` to the projected start) and
    ``residual`` (observed − projected).
    """
    daily_temp = daily_temperature(history)
    rows: list[dict[str, Any]] = []
    shift = pd.Timedelta(days=forecast_days - 1)
    for name in species or [s for s in ALL_SPECIES if s not in SPORE_TAXA]:
        onsets = observed_onsets(history, name)
        for year, onset in sorted(onsets.items()):
            if sum(1 for y in onsets if y < year) < MIN_RULE_SELECTION_YEARS:
                continue
            calendar = _calendar_doy(onsets, year, name)
            ensemble = onset_ensemble(history, name, before_year=year, onsets=onsets)
            # A projection as of day t never reads a temperature after t, so
            # one pass over the year gives it for every issue day.
            by_day, before_year = project_year(daily_temp, ensemble, calendar, year)
            crossed = ensemble_crossing_doy(daily_temp, year, ensemble, calendar)
            new_year = pd.Timestamp(year=year, month=1, day=1)
            start = new_year + pd.Timedelta(days=onset - 1)
            for lead in range(1, replay_days + 1):
                issued = start - pd.Timedelta(days=lead)
                as_of = issued + shift
                known = by_day[by_day.index <= as_of]
                projected = float(known.iloc[-1]) if len(known) else before_year
                as_of_doy = (as_of - new_year).days + 1
                crossing = calendar if crossed is None or crossed > as_of_doy else crossed
                rows.append({
                    "species": name,
                    "year": year,
                    "lead": lead,
                    "observed": float(onset),
                    "projected": projected,
                    "calendar": calendar,
                    "crossing": crossing,
                    "ahead": projected - ((issued - new_year).days + 1),
                    "residual": float(onset) - projected,
                })
    return pd.DataFrame(
        rows,
        columns=["species", "year", "lead", "observed", "projected", "calendar", "crossing",
                 "ahead", "residual"],
    )


def _bucket(ahead: float) -> int:
    for i, (low, high) in enumerate(LEAD_BUCKETS):
        if low <= ahead <= high:
            return i
    return len(LEAD_BUCKETS) - 1


def residual_table(replayed: pd.DataFrame) -> dict[int, np.ndarray]:
    """Sorted replay residuals per lead bucket; key -1 holds all of them.

    A bucket with fewer than :data:`MIN_BUCKET_SAMPLES` residuals is left
    out, and a lookup falls back to the pooled set.
    """
    if replayed.empty:
        return {}
    table: dict[int, np.ndarray] = {-1: np.sort(replayed["residual"].to_numpy(dtype=float))}
    buckets = replayed["ahead"].map(_bucket)
    for i in range(len(LEAD_BUCKETS)):
        values = replayed.loc[buckets == i, "residual"].to_numpy(dtype=float)
        if len(values) >= MIN_BUCKET_SAMPLES:
            table[i] = np.sort(values)
    return table


def uncertainty(
    table: dict[int, np.ndarray], ahead: float
) -> tuple[float, float, float] | None:
    """Earliest and latest start (days from today) and P(start within 7 days).

    Read off the residuals of the bucket *ahead* falls in, conditional on
    the season not having started today: a start is at least a day away.
    None when there are no residuals to read.
    """
    residuals = table.get(_bucket(ahead), table.get(-1))
    if residuals is None or len(residuals) == 0:
        return None
    lead = ahead + residuals
    lead = lead[lead >= 1] if (lead >= 1).any() else np.array([1.0])
    tail = (1.0 - INTERVAL_COVERAGE) / 2.0
    earliest = float(np.quantile(lead, tail))
    latest = float(np.quantile(lead, 1.0 - tail))
    within_week = float((lead <= 7).mean())
    return earliest, latest, within_week


# --- The published outlook -----------------------------------------------------


@dataclass
class SeasonOutlook:
    """One species' season as of today.

    ``state`` is ``"started"`` once this year's start has been measured and
    ``"before"`` otherwise — before this year's season, or, after it, before
    the next one. ``expected_start`` is the measured start in the first case
    and the projection in the second, where ``earliest``/``latest`` bound its
    central 80% and ``p_start_7d`` is the probability that the season starts
    within the next seven days. ``basis`` says what the projection rests on:
    ``forcing`` (thermal rules), ``calendar`` (the median of earlier
    seasons) or ``baseline`` (never measured; a textbook date, no interval).
    """

    species: str
    state: str
    expected_start: date
    basis: str
    earliest: date | None = None
    latest: date | None = None
    p_start_7d: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        out: dict[str, Any] = {
            "polle": self.species,
            "state": self.state,
            "expected_start": self.expected_start.isoformat(),
            "basis": self.basis,
        }
        if self.earliest is not None and self.latest is not None:
            out["earliest"] = self.earliest.isoformat()
            out["latest"] = self.latest.isoformat()
        if self.p_start_7d is not None:
            out["p_start_7d"] = round(self.p_start_7d, 2)
        return out


def _season_months(species: str) -> tuple[int, int]:
    """Last month of the core season, and last month of its shoulder."""
    window = SPECIES_SEASON.get(species, (1, 12))
    return window[1], min(12, window[1] + SEASON_SHOULDER_MONTHS)


def _date_of(year: int, doy: float) -> date:
    return (pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(round(doy)) - 1)).date()


def season_outlook(
    history: pd.DataFrame,
    today: pd.Timestamp,
    daily_temp: pd.Series | None = None,
    species: list[str] | None = None,
    table: dict[int, np.ndarray] | None = None,
) -> list[SeasonOutlook]:
    """The outlook for every species that has a season, as of *today*.

    *daily_temp* is the temperature the projection reads — the caller passes
    the measured series with the weather forecast appended; by default the
    history's alone. *table* is :func:`residual_table` of a replay; by
    default it is computed from *history*.
    """
    today = pd.Timestamp(today).normalize()
    if daily_temp is None:
        daily_temp = daily_temperature(history)
    names = species or [s for s in ALL_SPECIES if s not in SPORE_TAXA]
    if table is None:
        table = residual_table(replay(history, names))
    as_of = pd.Timestamp(daily_temp.index.max()) if len(daily_temp) else today

    out: list[SeasonOutlook] = []
    for name in names:
        onsets = observed_onsets(history, name)
        year = today.year
        core_end, shoulder_end = _season_months(name)
        if year in onsets and today.month <= shoulder_end:
            out.append(SeasonOutlook(
                species=name, state="started",
                expected_start=_date_of(year, onsets[year]), basis="measured",
            ))
            continue
        # Past the core season the next start is next year's — whether this
        # year's season is over or, for a species the station barely
        # registers, never measurably began.
        if today.month > core_end:
            year += 1
        projected, basis = projected_onset_for(history, name, year, as_of, daily_temp, onsets)
        start = pd.Timestamp(_date_of(year, projected))
        # A season that has not started cannot start before tomorrow.
        ahead = max(1.0, float((start - today).days))
        expected = (today + pd.Timedelta(days=ahead)).date()
        outlook = SeasonOutlook(species=name, state="before", expected_start=expected, basis=basis)
        spread = uncertainty(table, float((start - today).days)) if basis != "baseline" else None
        if spread is not None:
            earliest, latest, within_week = spread
            outlook.earliest = (today + pd.Timedelta(days=int(round(earliest)))).date()
            outlook.latest = (today + pd.Timedelta(days=int(round(latest)))).date()
            outlook.p_start_7d = within_week
            # The point estimate never sits outside its own interval.
            outlook.expected_start = min(max(expected, outlook.earliest), outlook.latest)
        out.append(outlook)
    return out


# --- Scoring -----------------------------------------------------------------


REPORT_LEADS: tuple[int, ...] = (21, 14, 10, 7, 5, 3, 1)


def score_replay(replayed: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute start-date error by lead, for the projection and its baselines.

    One row per (species, lead) for the leads in :data:`REPORT_LEADS`, with
    the error of the projection, of the calendar, and of what was shipped
    before (the calendar until the ensemble has crossed), and the number of
    seasons behind each figure.
    """
    rows: list[dict[str, Any]] = []
    at = replayed[replayed["lead"].isin(REPORT_LEADS)]
    for (name, lead), grp in at.groupby(["species", "lead"]):
        rows.append({
            "species": name,
            "lead": int(lead),
            "projection": float((grp["projected"] - grp["observed"]).abs().mean()),
            "calendar": float((grp["calendar"] - grp["observed"]).abs().mean()),
            "crossing": float((grp["crossing"] - grp["observed"]).abs().mean()),
            "seasons": int(len(grp)),
        })
    return pd.DataFrame(rows, columns=["species", "lead", "projection", "calendar", "crossing", "seasons"])


def score_uncertainty(replayed: pd.DataFrame) -> dict[str, float]:
    """How the published interval and probability would have done, leave-one-year-out.

    Every replayed projection is given the residuals of all the *other*
    years, as a forecast in a new year is. Returns the coverage of the 80%
    interval, the Brier score of P(start within 7 days) and of the base
    rate, and the hit rate and false-alarm rate of announcing a start when
    that probability is at least one half.
    """
    if replayed.empty:
        return {}
    inside: list[bool] = []
    probability: list[float] = []
    happened: list[bool] = []
    for year, grp in replayed.groupby("year"):
        table = residual_table(replayed[replayed["year"] != year])
        for row in grp.itertuples():
            spread = uncertainty(table, row.ahead)
            if spread is None:
                continue
            earliest, latest, within_week = spread
            inside.append(earliest - 0.5 <= row.lead <= latest + 0.5)
            probability.append(within_week)
            happened.append(row.lead <= 7)
    p = np.array(probability)
    y = np.array(happened, dtype=float)
    base = float(y.mean())
    announced = p >= 0.5
    return {
        "n": float(len(p)),
        "coverage": float(np.mean(inside)),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_base_rate": float(np.mean((base - y) ** 2)),
        "hit_rate": float(announced[y == 1].mean()) if (y == 1).any() else float("nan"),
        "false_alarm_rate": float(announced[y == 0].mean()) if (y == 0).any() else float("nan"),
    }


def print_outlook_report(replayed: pd.DataFrame) -> None:
    """The start-date report: error by how many days ahead the projection was issued."""
    if replayed.empty:
        print("\n  No season has enough seasons behind it to replay the outlook.")
        return
    scores = score_replay(replayed)
    print("\n" + "=" * 70)
    print("SEASON OUTLOOK (start date projected N days ahead, walk-forward)")
    print("=" * 70)
    print(f"\nEach season is projected from the seasons before it, with {OUTLOOK_FORECAST_DAYS} days of")
    print("measured temperature standing in for the weather forecast and the average")
    print("contribution of earlier years beyond it.")
    print("\nMean |start-date error| in days, by days ahead of the measured start:")
    leads = sorted(scores["lead"].unique(), reverse=True)
    header = f"  {'Species':<11}{'seasons':>8}" + "".join(f"{str(lead) + ' d':>7}" for lead in leads)
    for label, column in (("projection (published)", "projection"),
                          ("calendar", "calendar"),
                          ("calendar until crossed (before)", "crossing")):
        print(f"\n  {label}")
        print(header)
        for name in scores["species"].unique():
            sp = scores[scores["species"] == name].set_index("lead")
            cells = "".join(f"{sp.loc[lead, column]:>7.1f}" if lead in sp.index else f"{'-':>7}"
                            for lead in leads)
            print(f"  {name:<11}{int(sp['seasons'].max()):>8}{cells}")
        pooled = scores.groupby("lead")[column].mean()
        print(f"  {'mean':<11}{'':>8}" + "".join(f"{pooled.loc[lead]:>7.1f}" for lead in leads))

    quality = score_uncertainty(replayed)
    if quality:
        print(f"\n  Published uncertainty, each year scored on the residuals of the others "
              f"({int(quality['n'])} projections):")
        print(f"    {INTERVAL_COVERAGE:.0%} interval covers the start        {quality['coverage']:.0%}")
        print(f"    P(start within 7 days), Brier score   {quality['brier']:.3f}"
              f"   (always saying the base rate: {quality['brier_base_rate']:.3f})")
        print(f"    announcing at P >= 0.5: catches {quality['hit_rate']:.0%} of the starts "
              f"that are within a week, false alarm on {quality['false_alarm_rate']:.0%} of the days they are not")
