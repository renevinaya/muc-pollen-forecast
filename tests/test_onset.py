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
    """With the calendar winning, the per-day estimate is flat all season."""
    history = build_history({2019: 40, 2020: 40, 2021: 40, 2022: 40, 2023: 40, 2024: 40})
    estimates = onset_doy_by_day(history, SPECIES)
    idx = pd.DatetimeIndex(estimates.index)
    season = estimates[(idx.year == 2024) & (idx.month < 11)]
    assert season.nunique() == 1 and float(season.iloc[0]) == 40.0
    # From November the estimate is next season's, past the year's end.
    winter = estimates[(idx.year == 2024) & (idx.month >= 11)]
    assert winter.nunique() == 1 and float(winter.iloc[0]) == 366.0 + 40.0


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


# --- December continuity (B.7) ----------------------------------------------


def build_winter_history(december_temp: dict[int, float], threshold: float = 700.0) -> pd.DataFrame:
    """Seasons whose onset is fixed by the forcing accumulated *from 1 November*.

    Every day is 8 °C except November and December, whose temperature is set
    per year; each season opens on the day the forcing since 1 November of
    the year before reaches *threshold*. A rule that starts on 1 January sees
    only ``8 × doy`` at those onsets, which varies with the winter, so the
    November rule is the only one that projects them.
    """
    years = sorted(december_temp)
    rows = []
    for year in [years[0] - 1] + years:
        winter_before = december_temp.get(year - 1, 6.0)
        onset = int(round((threshold - 61 * winter_before) / 8.0)) if year in years else 999
        day = pd.Timestamp(year=year, month=1, day=1)
        while day.year == year:
            doy = day.dayofyear
            rows.append({
                "date": day,
                "species": SPECIES,
                "value": float(LOW_MAX * 3) if onset <= doy < onset + 40 else 0.0,
                "temperature_mean": december_temp.get(year, 6.0) if day.month >= 11 else 8.0,
            })
            day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_autumn_rule_accumulates_across_the_year_boundary():
    from src.onset import daily_temperature, forcing_series

    history = build_winter_history({2019: 4.0, 2020: 6.0, 2021: 8.0})
    temp = daily_temperature(history)
    november = forcing_series(temp, ((11, 1), 0.0))
    january = forcing_series(temp, ((1, 1), 0.0))
    assert november.loc["2020-01-01"] == pytest.approx(61 * 4.0 + 8.0)
    assert january.loc["2020-01-01"] == pytest.approx(8.0)
    assert november.loc["2020-11-01"] == pytest.approx(6.0), "the next season restarts on 1 November"
    assert november.loc["2020-10-31"] > 2000, "and the old one keeps counting until then"


def test_a_warm_december_season_selects_the_november_rule():
    from src.onset import select_forcing_rule

    history = build_winter_history(
        {2018: 6.0, 2019: 4.0, 2020: 8.0, 2021: 5.0, 2022: 7.0, 2023: 3.0, 2024: 6.0}
    )
    rule, threshold, loo = select_forcing_rule(history, SPECIES, before_year=2025)
    assert rule == ((11, 1), 0.0), rule
    assert loo < 2.0
    assert threshold == pytest.approx(700.0, abs=8.0)


def test_the_estimate_counts_down_through_december_without_a_reset():
    """days-since-onset must step by one from 31 December to 1 January."""
    history = build_winter_history(
        {2018: 6.0, 2019: 4.0, 2020: 8.0, 2021: 5.0, 2022: 7.0, 2023: 3.0, 2024: 6.0, 2025: 5.0}
    )
    estimates = onset_doy_by_day(history, SPECIES)
    dec31 = pd.Timestamp("2024-12-31")
    jan1 = pd.Timestamp("2025-01-01")
    since_dec31 = dec31.dayofyear - estimates.loc[dec31]
    since_jan1 = jan1.dayofyear - estimates.loc[jan1]
    assert since_jan1 - since_dec31 == pytest.approx(1.0)
    assert since_dec31 < 0, "December sits before the coming season, not 300 days after the last"
    # A January rule species gets the same continuity from the climatology.
    plain = build_history({2019: 40, 2020: 42, 2021: 44, 2022: 46, 2023: 41, 2024: 43})
    est = onset_doy_by_day(plain, SPECIES)
    d, j = pd.Timestamp("2024-12-30"), pd.Timestamp("2024-12-29")
    assert est.loc[d] == pytest.approx(est.loc[j])
    assert est.loc[d] > 365


def test_readiness_in_december_is_the_coming_seasons():
    from src.onset import readiness_by_day, rules_by_year

    history = build_winter_history(
        {2018: 6.0, 2019: 4.0, 2020: 8.0, 2021: 5.0, 2022: 7.0, 2023: 3.0, 2024: 6.0, 2025: 5.0}
    )
    table = readiness_by_day(history, SPECIES)
    (_, thr_2025), = rules_by_year(history, SPECIES, [2025]).values()
    assert table.loc["2024-12-15", "threshold"] == pytest.approx(thr_2025)
    dec = table.loc["2024-11-01":"2024-12-31", "forcing"].to_numpy()
    assert dec[0] == pytest.approx(6.0) and np.all(np.diff(dec) > 0)
    assert table.loc["2025-01-01", "forcing"] == pytest.approx(dec[-1] + 8.0)
