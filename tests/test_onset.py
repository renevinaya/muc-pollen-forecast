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
