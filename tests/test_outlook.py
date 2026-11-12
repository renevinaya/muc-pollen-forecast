"""The season outlook: a projected start date that never looks ahead.

What is pinned here is what cannot be seen in the output when it breaks:
that a projection made on a day uses nothing from after it, that a rule's
vote turns into its actual crossing on the day it crosses, that the replay
scores each season on the seasons before it, and that the published block
says "started" only for a start that was measured.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.outlook import (
    INTERVAL_COVERAGE,
    LEAD_BUCKETS,
    SeasonOutlook,
    _bucket,
    _rule_votes,
    expected_increments,
    project_onset,
    project_year,
    replay,
    residual_table,
    score_replay,
    score_uncertainty,
    season_outlook,
    uncertainty,
)
from src.types import ForecastOutput, SPECIES_THRESHOLDS

SPECIES = "Betula"
LOW_MAX = SPECIES_THRESHOLDS[SPECIES][0]
JAN0 = ((1, 1), 0.0)


def _temps(start: str, end: str, value: float = 5.0) -> pd.Series:
    return pd.Series(value, index=pd.date_range(start, end, freq="D"))


def _history(onsets: dict[int, int], warm_from: dict[int, int] | None = None) -> pd.DataFrame:
    """Daily rows: Betula switched on at each onset; 2 °C until *warm_from*, 10 °C after."""
    rows = []
    for year, onset in onsets.items():
        turn = (warm_from or {}).get(year, onset - 10)
        for doy in range(1, 366):
            day = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
            rows.append({
                "date": day,
                "species": SPECIES,
                "value": float(LOW_MAX * 3) if onset <= doy < onset + 40 else 0.0,
                "temperature_mean": 10.0 if doy >= turn else 2.0,
            })
    return pd.DataFrame(rows)


SEASONS = {2019: 95, 2020: 90, 2021: 100, 2022: 93, 2023: 97, 2024: 88, 2025: 99}


# --- The projection ----------------------------------------------------------


def test_expected_increments_average_contributions_not_temperatures() -> None:
    """Two years at -4 and +6 °C average to 1 °C — and to 3 degree-days at base 0, not 1."""
    days = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    temps = pd.Series(np.where(days.year == 2020, -4.0, 6.0), index=days)
    expected = expected_increments(temps, JAN0, before_year=2022)
    assert expected is not None and len(expected) == 366
    assert expected[40] == pytest.approx(3.0)
    # Nothing accrues before the rule's start date.
    late = expected_increments(temps, ((3, 1), 0.0), before_year=2022)
    assert late is not None and late[30] == 0.0 and late[90] == pytest.approx(3.0)
    # One earlier year is not a climate.
    assert expected_increments(temps, JAN0, before_year=2021) is None


def test_a_rule_votes_its_projection_until_it_crosses_and_its_crossing_after() -> None:
    year = _temps("2024-01-01", "2024-03-31", 5.0)  # 5 degree-days a day
    expected = np.full(366, 5.0)                     # and the same on average
    votes, before = _rule_votes(year, JAN0, threshold=200.0, expected=expected)
    # 200 degree-days at 5 a day: day 40, seen from any day and before the year.
    assert before == 40.0
    assert np.all(votes == 40.0)

    # A warm year crosses earlier than the average says. Before the
    # crossing the vote moves towards it; from the crossing on it is fixed.
    warm = _temps("2024-01-01", "2024-03-31", 10.0)
    votes, before = _rule_votes(warm, JAN0, threshold=200.0, expected=expected)
    assert before == 40.0
    assert votes[0] == pytest.approx(39.0)   # 10 so far, 190 to go at 5 a day
    assert votes[9] == pytest.approx(30.0)   # 100 so far, 100 to go
    assert np.all(votes[19:] == 20.0)        # crossed on day 20
    assert np.all(np.diff(votes[:20]) <= 0)


def test_projection_never_looks_ahead() -> None:
    """The projection as of a day is the same whether or not later days exist."""
    days = pd.date_range("2019-01-01", "2024-06-30", freq="D")
    rng = np.random.default_rng(3)
    temps = pd.Series(8.0 * np.sin((days.dayofyear - 80) / 58.0) + rng.normal(0, 3, len(days)),
                      index=days)
    ensemble = [(JAN0, 150.0, 1.0), (((2, 1), 3.0), 60.0, 1.2), (None, None, 1.5)]
    full, _ = project_year(temps, ensemble, calendar=70.0, year=2024)
    for cut in ("2024-01-20", "2024-02-25", "2024-04-10"):
        partial, _ = project_year(temps[temps.index <= cut], ensemble, calendar=70.0, year=2024)
        assert np.allclose(partial.to_numpy(), full.loc[partial.index].to_numpy()), cut
        assert project_onset(temps, ensemble, 70.0, 2024, pd.Timestamp(cut)) == full.loc[cut]


def test_projection_before_the_year_begins_is_the_climate_alone() -> None:
    """In November next year's start is projected from average contributions only."""
    days = pd.date_range("2020-01-01", "2023-11-10", freq="D")
    temps = pd.Series(5.0, index=days)
    ensemble = [(JAN0, 200.0, 1.0)]
    assert project_onset(temps, ensemble, calendar=60.0, year=2024,
                         as_of=pd.Timestamp("2023-11-10")) == 40.0


def test_an_unprojectable_ensemble_falls_back_to_the_calendar() -> None:
    temps = _temps("2024-01-01", "2024-01-20", 0.0)  # no earlier years, nothing crossed
    ensemble = [(JAN0, 200.0, 1.0)]
    assert project_onset(temps, ensemble, 60.0, 2024, pd.Timestamp("2024-01-20")) == 60.0
    assert project_onset(temps, [], 60.0, 2024, pd.Timestamp("2024-01-20")) == 60.0


# --- The replay --------------------------------------------------------------


def test_replay_scores_each_season_on_the_seasons_before_it() -> None:
    history = _history(SEASONS)
    replayed = replay(history, [SPECIES], replay_days=30)
    # Four seasons are needed behind a season before it is replayed.
    assert sorted(replayed["year"].unique()) == [2023, 2024, 2025]
    assert (replayed["lead"] >= 1).all() and replayed["lead"].max() == 30

    # Moving the last season does not move what was projected for an earlier one.
    moved = replay(_history({**SEASONS, 2025: 120}), [SPECIES], replay_days=30)
    before = replayed[replayed["year"] == 2024].reset_index(drop=True)
    after = moved[moved["year"] == 2024].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_replay_improves_on_the_calendar_when_warmth_decides_the_start() -> None:
    """In this fixture the season starts ten days after it turns warm, every year."""
    replayed = replay(_history(SEASONS), [SPECIES], replay_days=30)
    scores = score_replay(replayed).set_index("lead")
    assert scores.loc[7, "projection"] < scores.loc[7, "calendar"]
    assert scores.loc[1, "projection"] <= 2.0


def test_uncertainty_is_conditional_on_not_having_started() -> None:
    table = {-1: np.array([-20.0, -10.0, -5.0, 0.0, 5.0, 10.0])}
    earliest, latest, within_week = uncertainty(table, ahead=3.0)
    # Projected three days out with residuals down to -20: the season has
    # not started today, so the earliest start is tomorrow at the soonest.
    assert earliest >= 1.0
    assert latest <= 13.0
    assert 0.0 < within_week < 1.0
    assert uncertainty({}, 3.0) is None


def test_a_fractional_lead_falls_in_the_bucket_of_its_nearest_day() -> None:
    # A replayed lead is x.5 whenever the calendar median is; it used to fall
    # between two buckets and land in the last one (31+ days).
    last = len(LEAD_BUCKETS) - 1
    for ahead, expected in [
        (-3.0, 0), (0.0, 0), (0.4, 0), (0.5, 1), (1.0, 1), (7.0, 1), (7.4, 1),
        (7.5, 2), (14.5, 3), (30.4, 3), (30.5, last), (90.0, last),
    ]:
        assert _bucket(ahead) == expected, ahead


def test_uncertainty_scoring_uses_other_years_only() -> None:
    replayed = replay(_history(SEASONS), [SPECIES], replay_days=40)
    quality = score_uncertainty(replayed)
    assert 0.0 <= quality["coverage"] <= 1.0
    assert quality["brier"] <= quality["brier_base_rate"]
    assert set(residual_table(replayed)) >= {-1}


# --- The published block -------------------------------------------------------


def test_outlook_says_started_only_for_a_measured_start() -> None:
    history = _history(SEASONS)
    # 20 March 2025: day 79, twenty days before this fixture's start.
    before = season_outlook(history[history["date"] < "2025-03-20"], pd.Timestamp("2025-03-20"),
                            species=[SPECIES])[0]
    assert before.state == "before" and before.basis == "forcing"
    assert before.expected_start > pd.Timestamp("2025-03-20").date()
    assert before.earliest <= before.expected_start <= before.latest
    assert 0.0 <= before.p_start_7d <= 1.0

    # 20 April 2025: the run above the threshold has been measured.
    started = season_outlook(history[history["date"] < "2025-04-20"], pd.Timestamp("2025-04-20"),
                             species=[SPECIES])[0]
    assert started.state == "started" and started.basis == "measured"
    assert started.expected_start == (pd.Timestamp("2025-01-01") + pd.Timedelta(days=98)).date()


def test_outlook_after_the_season_is_next_years() -> None:
    history = _history(SEASONS)
    outlook = season_outlook(history[history["date"] < "2025-11-10"], pd.Timestamp("2025-11-10"),
                             species=[SPECIES])[0]
    assert outlook.state == "before"
    assert outlook.expected_start.year == 2026
    assert outlook.p_start_7d == 0.0


def test_outlook_probability_rises_as_the_start_approaches() -> None:
    history = _history(SEASONS)
    far = season_outlook(history[history["date"] < "2025-03-05"], pd.Timestamp("2025-03-05"),
                         species=[SPECIES])[0]
    # Two days before the start, with the warm spell that triggers it under way.
    near = season_outlook(history[history["date"] < "2025-04-07"], pd.Timestamp("2025-04-07"),
                          species=[SPECIES])[0]
    assert near.p_start_7d > far.p_start_7d
    assert near.p_start_7d > 0.5


def test_a_species_never_measured_gets_a_baseline_date_and_no_interval() -> None:
    history = _history(SEASONS)
    outlook = season_outlook(history[history["date"] < "2025-03-20"], pd.Timestamp("2025-03-20"),
                             species=["Ambrosia"])[0]
    assert outlook.basis == "baseline"
    assert outlook.earliest is None and outlook.p_start_7d is None
    assert "earliest" not in outlook.to_dict()


def test_outlook_is_published_with_the_forecast() -> None:
    season = SeasonOutlook(
        species="Corylus", state="before", expected_start=pd.Timestamp("2027-01-27").date(),
        basis="forcing", earliest=pd.Timestamp("2027-01-10").date(),
        latest=pd.Timestamp("2027-02-06").date(), p_start_7d=0.0312,
    )
    output = ForecastOutput(generated="2026-11-10T05:00:00Z", location="DEMUNC", seasons=[season])
    web = json.loads(json.dumps(output.to_web_dict()))
    assert web["seasons"] == [{
        "polle": "Corylus", "state": "before", "expected_start": "2027-01-27",
        "basis": "forcing", "earliest": "2027-01-10", "latest": "2027-02-06", "p_start_7d": 0.03,
    }]
    # A forecast without an outlook simply has no block.
    assert "seasons" not in ForecastOutput(generated="x", location="DEMUNC").to_web_dict()
    assert INTERVAL_COVERAGE == 0.8
