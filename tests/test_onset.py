"""Tests for the data-driven onset estimates.

These pin down the two properties the estimates are only useful if they have:
that a backtest of year *Y* never sees year *Y*'s answer, and that a training
row never uses a value the forecaster could not have computed on that date.
Both are invisible in the output when they break — the numbers just quietly
get better than they should be — so they are worth asserting.
"""

import numpy as np
import pandas as pd
import pytest

from src.onset import (
    ONSET_RUN_DAYS,
    calibrated_gdd_threshold,
    climatological_onset_doy,
    gdd_threshold_by_year,
    observed_onsets,
    onset_doy_by_day,
    onset_doy_lookup,
)
from src.types import SPECIES_GDD_THRESHOLD, SPECIES_THRESHOLDS

SPECIES = "Corylus"
LOW_MAX = SPECIES_THRESHOLDS[SPECIES][0]


def build_history(
    onsets_by_year: dict[int, int],
    temperature: float = 8.0,
    species: str = SPECIES,
) -> pd.DataFrame:
    """A minimal history: one row per day, pollen switched on at each onset."""
    rows = []
    for year, onset_doy in onsets_by_year.items():
        for doy in range(1, 366):
            day = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
            if day.year != year:
                break
            in_season = onset_doy <= doy < onset_doy + 40
            rows.append(
                {
                    "date": day,
                    "species": species,
                    "value": float(LOW_MAX * 3) if in_season else 0.0,
                    "temperature_mean": temperature,
                }
            )
    return pd.DataFrame(rows)


def test_onset_needs_a_sustained_run():
    """A single day at the threshold is a cloud passing through, not a season."""
    history = build_history({2020: 50})
    spike = history["date"] == pd.Timestamp("2020-02-01")  # doy 32, well before onset
    history.loc[spike, "value"] = float(LOW_MAX * 5)

    assert observed_onsets(history, SPECIES) == {2020: 50}


def test_onset_detects_a_run_of_exactly_the_required_length():
    history = build_history({2020: 50})
    short = history["date"].between("2020-02-01", "2020-02-02")  # two days only
    history.loc[short, "value"] = float(LOW_MAX * 5)
    assert observed_onsets(history, SPECIES) == {2020: 50}

    run = history["date"].between(
        "2020-02-01", pd.Timestamp("2020-02-01") + pd.Timedelta(days=ONSET_RUN_DAYS - 1)
    )
    history.loc[run, "value"] = float(LOW_MAX * 5)
    assert observed_onsets(history, SPECIES) == {2020: 32}


def test_onset_ignores_pollen_outside_the_core_season():
    """Betula in February is transported, not Munich's birches flowering."""
    history = build_history({2020: 100}, species="Betula")
    out_of_season = history["date"].between("2020-02-05", "2020-02-15")
    history.loc[out_of_season, "value"] = 500.0

    assert observed_onsets(history, "Betula") == {2020: 100}


def test_threshold_calibration_ignores_the_year_being_predicted():
    """Year Y's threshold must not move when year Y's own onset moves."""
    base = {2019: 40, 2020: 42, 2021: 44, 2022: 46}
    shifted = {**base, 2022: 120}

    before = gdd_threshold_by_year(build_history(base), SPECIES)[2022]
    after = gdd_threshold_by_year(build_history(shifted), SPECIES)[2022]

    assert before == pytest.approx(after)


def test_threshold_falls_back_before_there_is_anything_to_calibrate_on():
    history = build_history({2019: 40, 2020: 42, 2021: 44})
    thresholds = gdd_threshold_by_year(history, SPECIES)

    static = SPECIES_GDD_THRESHOLD[SPECIES]
    assert thresholds[2019] == static  # no prior seasons
    assert thresholds[2020] == static  # one prior season is not a calibration
    assert thresholds[2021] != static  # two is enough


def test_climatology_excludes_the_year_asked_about():
    history = build_history({2019: 40, 2020: 40, 2021: 40, 2022: 200})
    assert climatological_onset_doy(history, SPECIES, before_year=2022) == 40.0


def test_per_day_estimate_is_climatology_until_the_crossing():
    """Before this year's warmth confirms anything, we only know prior years."""
    history = build_history({2019: 40, 2020: 42, 2021: 44, 2022: 46})
    estimates = onset_doy_by_day(history, SPECIES)

    idx = pd.DatetimeIndex(estimates.index)
    year_2022 = estimates[idx.year == 2022].to_numpy()
    climatology = climatological_onset_doy(history, SPECIES, before_year=2022)

    assert year_2022[0] == pytest.approx(climatology)
    # Once it switches it stays switched — cumulative GDD never goes back down.
    switches = np.flatnonzero(np.diff(year_2022) != 0)
    assert len(switches) <= 1


def test_per_day_estimate_never_looks_ahead():
    """Truncating the history must not change any estimate that survives."""
    history = build_history({2019: 40, 2020: 42, 2021: 44, 2022: 46})
    full = onset_doy_by_day(history, SPECIES)

    cutoff = pd.Timestamp("2022-03-01")
    truncated = onset_doy_by_day(history[history["date"] <= cutoff], SPECIES)

    shared = truncated.index.intersection(full.index)
    assert len(shared) > 0
    np.testing.assert_allclose(full.loc[shared].to_numpy(), truncated.loc[shared].to_numpy())


def test_lookup_past_the_end_holds_the_last_known_value():
    """Forecast windows run past the last observation and must not lose the estimate."""
    history = build_history({2019: 40, 2020: 42, 2021: 44, 2022: 46})
    estimates = onset_doy_by_day(history, SPECIES)

    beyond = pd.Timestamp("2023-06-01")
    assert onset_doy_lookup(estimates, beyond, fallback=999.0) == float(estimates.iloc[-1])
    assert onset_doy_lookup(pd.Series(dtype=float), beyond, fallback=999.0) == 999.0


def test_calibrated_threshold_survives_a_species_with_no_measurements():
    history = build_history({2019: 40, 2020: 42})
    assert calibrated_gdd_threshold(history, "Ambrosia") == SPECIES_GDD_THRESHOLD["Ambrosia"]


def test_trainer_and_forecaster_resolve_the_same_onset():
    """The two paths must agree, or the model sees different features than it learnt on.

    The forecaster resolves the estimate through ``onset_doy_lookup``; the
    trainer resolves it inside ``_add_phenology_features``. This asserts the
    day-by-day answers match, which is the parity the causal estimate is for.
    """
    from src.trainer import _add_phenology_features

    history = build_history({2019: 40, 2020: 42, 2021: 44, 2022: 46})
    featured = _add_phenology_features(history, SPECIES)

    estimates = onset_doy_by_day(history, SPECIES)
    days = pd.to_datetime(featured["date"])
    # Sample across the year; skip rows where the -60 clip hides the estimate.
    sampled = featured.iloc[::37]
    assert len(sampled) > 20

    for (_, row), day in zip(sampled.iterrows(), days[::37]):
        trainer_onset = day.dayofyear - row["days_since_typical_onset"]
        forecaster_onset = onset_doy_lookup(estimates, day, fallback=float("nan"))
        if row["days_since_typical_onset"] <= -60:
            continue
        assert trainer_onset == pytest.approx(forecaster_onset)


def test_rule_selection_needs_enough_seasons_to_choose_on():
    """Three prior seasons is a coin toss: keep the rule that always ran."""
    from src.onset import DEFAULT_FORCING_RULE, select_forcing_rule

    history = build_history({2019: 40, 2020: 42, 2021: 44, 2022: 46})
    rule, threshold, _ = select_forcing_rule(history, SPECIES, before_year=2022)
    assert rule == DEFAULT_FORCING_RULE
    assert threshold is not None


def test_rule_selection_falls_back_to_climatology_when_nothing_beats_it():
    """Constant temperature makes every rule a calendar in disguise, except
    the calendar is exact and the rules round to a day — so nothing beats it."""
    from src.onset import select_forcing_rule

    history = build_history({2019: 40, 2020: 40, 2021: 40, 2022: 40, 2023: 40, 2024: 40})
    rule, threshold, loo = select_forcing_rule(history, SPECIES, before_year=2024)
    assert rule is None and threshold is None
    assert loo == 0.0


def test_rule_selection_is_walk_forward():
    """Year Y's rule and threshold must not move when year Y's onset moves."""
    from src.onset import select_forcing_rule

    base = {2019: 40, 2020: 45, 2021: 38, 2022: 47, 2023: 41, 2024: 44}
    shifted = {**base, 2024: 120}
    before = select_forcing_rule(build_history(base), SPECIES, before_year=2024)
    after = select_forcing_rule(build_history(shifted), SPECIES, before_year=2024)
    assert before[0] == after[0]
    assert before[1] == after[1]


def test_projection_switches_off_for_a_climatology_species():
    """With the calendar winning, the per-day estimate is flat all year."""
    history = build_history({2019: 40, 2020: 40, 2021: 40, 2022: 40, 2023: 40, 2024: 40})
    estimates = onset_doy_by_day(history, SPECIES)
    year = estimates[pd.DatetimeIndex(estimates.index).year == 2024]
    assert year.nunique() == 1 and float(year.iloc[0]) == 40.0


def test_a_run_with_implausibly_little_forcing_is_not_the_onset():
    """Six days of transported birch in early March is not Munich's birches.

    With a constant temperature the accumulated forcing is proportional to the
    day of year, so an early episode at DOY 20 carries a fifth of the forcing
    the other years flowered with. The year's next run is the onset instead.
    """
    history = build_history({2019: 90, 2020: 90, 2021: 90, 2022: 90}, species="Betula")
    episode = history["date"].between("2022-01-20", "2022-01-25")
    history.loc[episode, "value"] = float(SPECIES_THRESHOLDS["Betula"][0] * 3)

    assert observed_onsets(history, "Betula") == {2019: 90, 2020: 90, 2021: 90, 2022: 90}


def test_a_genuinely_early_year_is_kept():
    """Early because it was warm is early: forcing at onset is still typical."""
    history = build_history({2019: 90, 2020: 90, 2021: 90, 2022: 60}, species="Betula")
    # DOY 60 carries two thirds of the forcing of DOY 90 — well above the cut.
    assert observed_onsets(history, "Betula")[2022] == 60


def test_a_year_with_only_implausible_runs_drops_out():
    history = build_history({2019: 90, 2020: 90, 2021: 90, 2022: 20}, species="Betula")
    assert 2022 not in observed_onsets(history, "Betula")


def test_readiness_keeps_accumulating_past_the_history():
    """The forecaster hands in the combined temperature; forcing must climb
    through the forecast days instead of stopping at the last measurement."""
    from src.onset import daily_temperature, readiness_by_day

    history = build_history({2019: 40, 2020: 42, 2021: 44, 2022: 46, 2023: 41})
    measured = daily_temperature(history)
    extended = pd.concat(
        [measured, pd.Series(8.0, index=pd.date_range("2024-01-01", "2024-01-20", freq="D"))]
    )
    table = readiness_by_day(history, SPECIES, daily_temp=extended)
    jan = table.loc["2024-01-01":"2024-01-20", "forcing"].to_numpy()
    assert np.all(np.diff(jan) > 0)
    assert np.isfinite(table.loc["2024-01-10", "threshold"])


def test_readiness_threshold_is_walk_forward():
    from src.onset import readiness_by_day

    base = {2019: 40, 2020: 45, 2021: 38, 2022: 47, 2023: 41, 2024: 44}
    shifted = {**base, 2024: 120}
    before = readiness_by_day(build_history(base), SPECIES).loc["2024-03-01", "threshold"]
    after = readiness_by_day(build_history(shifted), SPECIES).loc["2024-03-01", "threshold"]
    assert before == pytest.approx(after)


# --- The ensemble (G.2b) ------------------------------------------------------


def test_the_estimate_is_the_median_vote_not_the_best_candidate():
    """One candidate that crosses weeks early does not move the estimate."""
    from src.onset import ensemble_crossing_doy, forcing_series

    days = pd.date_range("2024-01-01", "2024-06-30", freq="D")
    daily_temp = pd.Series(5.0, index=days)
    # At a constant 5 °C a rule from 1 Jan at base 0 gains 5 degree-days a day.
    jan0 = ((1, 1), 0.0)
    forcing = forcing_series(daily_temp, jan0)

    def threshold_for(doy: int) -> float:
        return float(forcing.iloc[doy - 1])

    wild = (jan0, threshold_for(10), 0.5)  # best-ranked, and far off
    ensemble = [wild, (jan0, threshold_for(40), 1.0), (jan0, threshold_for(42), 1.1),
                (None, None, 1.2), (jan0, threshold_for(60), 1.3)]
    # Votes: 10, 40, 42, calendar 45, 60 -> the third.
    assert ensemble_crossing_doy(daily_temp, 2024, ensemble, calendar=45.0) == 42
    # The single best candidate, which is what shipped before, says day 10.
    assert ensemble_crossing_doy(daily_temp, 2024, ensemble[:1], calendar=45.0) == 10


def test_the_ensemble_stays_silent_until_its_median_has_crossed():
    """Rules that have not crossed vote "later"; no day is named before it is known."""
    from src.onset import ensemble_crossing_doy

    days = pd.date_range("2024-01-01", "2024-02-10", freq="D")  # 41 days of history
    daily_temp = pd.Series(5.0, index=days)
    jan0 = ((1, 1), 0.0)
    forcing_per_day = 5.0
    ensemble = [(jan0, 20 * forcing_per_day, 1.0), (jan0, 60 * forcing_per_day, 1.1),
                (jan0, 70 * forcing_per_day, 1.2)]
    # Only one of three has crossed by day 41: the median is still unknown.
    assert ensemble_crossing_doy(daily_temp, 2024, ensemble, calendar=50.0) is None
    # With the calendar among the votes its date counts from the start.
    with_calendar = ensemble[:2] + [(None, None, 1.2)]
    assert ensemble_crossing_doy(daily_temp, 2024, with_calendar, calendar=30.0) == 30.0


def test_ensemble_estimate_never_looks_ahead():
    """Cutting the year off after any day leaves every earlier estimate as it was."""
    onsets = {2019: 40, 2020: 35, 2021: 50, 2022: 44, 2023: 38, 2024: 47}
    history = build_history(onsets)
    # A year with real structure: cold, then a warm spell, then cold again.
    doy = pd.to_datetime(history["date"]).dt.dayofyear
    history["temperature_mean"] = np.where((doy > 25) & (doy < 60), 9.0, 1.0)
    full = onset_doy_by_day(history, SPECIES)
    for cut in ("2024-02-01", "2024-02-20", "2024-03-15"):
        truncated = onset_doy_by_day(history[history["date"] <= cut], SPECIES)
        year = truncated[truncated.index.year == 2024]
        assert np.allclose(year.to_numpy(), full.loc[year.index].to_numpy()), cut


def test_walk_forward_report_scores_each_year_on_the_years_before_it():
    from src.onset import MIN_RULE_SELECTION_YEARS, walk_forward_onsets

    onsets = {2019: 40, 2020: 35, 2021: 50, 2022: 44, 2023: 38, 2024: 47}
    history = build_history(onsets)
    report = walk_forward_onsets(history, SPECIES)
    assert list(report["year"]) == sorted(onsets)[MIN_RULE_SELECTION_YEARS:]
    # The calendar of 2023 is the median of 2019-2022 and knows nothing of 2023.
    row = report.set_index("year").loc[2023]
    assert row["calendar"] == np.median([40, 35, 50, 44])
    assert row["observed"] == 38

    shifted = {**onsets, 2024: 70}
    again = walk_forward_onsets(build_history(shifted), SPECIES).set_index("year")
    assert again.loc[2023, "estimate"] == row["estimate"]
