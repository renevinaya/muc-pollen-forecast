"""The season-start scorer must find the onset the forecast implies.

Built on a synthetic rollout frame where the answers are known: a season
that starts on a given day, a forecast that starts it three days late at one
horizon and never at another, and a burst of predicted pollen weeks early.
"""

import numpy as np
import pandas as pd

from src.onset_report import ramp_ratios, score_onsets
from src.types import SPECIES_THRESHOLDS

SPECIES = "Betula"
LOW = SPECIES_THRESHOLDS[SPECIES][0]
ONSET = pd.Timestamp("2024-03-21")


def build_history() -> pd.DataFrame:
    """One species, daily rows, 40 days of pollen from ONSET; ONSET is the
    measured onset by construction (first 3-day run >= low)."""
    rows = []
    for year in (2022, 2023, 2024):
        onset = pd.Timestamp(year=year, month=3, day=21)
        for day in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
            in_season = onset <= day < onset + pd.Timedelta(days=40)
            rows.append({"date": day, "species": SPECIES, "value": LOW * 5.0 if in_season else 0.0,
                         "temperature_mean": 8.0})
    return pd.DataFrame(rows)


def build_results(late_days: int = 3, never_horizon: int = 5, false_start: bool = False) -> pd.DataFrame:
    """Rollout-shaped frame for March–April 2024 at horizons 1..5."""
    rows = []
    days = pd.date_range("2024-02-20", "2024-04-30", freq="D")
    for horizon in range(1, 6):
        for day in days:
            actual = LOW * 5.0 if ONSET <= day < ONSET + pd.Timedelta(days=40) else 0.0
            if horizon == never_horizon:
                predicted = 0.0
            else:
                predicted = LOW * 2.0 if day >= ONSET + pd.Timedelta(days=late_days) else 0.0
            if false_start and horizon == 1 and pd.Timestamp("2024-02-25") <= day <= pd.Timestamp("2024-02-28"):
                predicted = LOW * 3.0
            for hour in (0, 12):
                rows.append({"date": day + pd.Timedelta(hours=hour), "origin": day, "horizon_day": horizon,
                             "species": SPECIES, "actual": actual, "predicted": predicted, "fold": 1,
                             "error": predicted - actual, "abs_error": abs(predicted - actual),
                             "level_actual": "x", "level_predicted": "x"})
    return pd.DataFrame(rows)


def test_timing_error_is_days_late_and_misses_are_nan():
    scores = score_onsets(build_results(late_days=3, never_horizon=5), build_history(), [SPECIES], years=1)
    by_h = scores.set_index("horizon_day")
    assert by_h.loc[1, "timing_error"] == 3.0
    assert by_h.loc[4, "timing_error"] == 3.0
    assert np.isnan(by_h.loc[5, "timing_error"])
    assert by_h["scored"].all()


def test_window_metrics_compare_against_predicting_zero():
    scores = score_onsets(build_results(late_days=0, never_horizon=5), build_history(), [SPECIES], years=1)
    by_h = scores.set_index("horizon_day")
    # Predicting 0 throughout costs the full actual; the model predicts 2×LOW vs 5×LOW.
    assert by_h.loc[5, "window_mae"] == by_h.loc[5, "zero_mae"]
    assert by_h.loc[1, "window_mae"] < by_h.loc[1, "zero_mae"]
    assert by_h.loc[1, "window_bias"] < 0


def test_false_starts_are_counted_before_the_window():
    scores = score_onsets(build_results(false_start=True), build_history(), [SPECIES], years=1)
    by_h = scores.set_index("horizon_day")
    assert by_h.loc[1, "false_starts"] >= 1
    assert by_h.loc[2, "false_starts"] == 0


def test_ramp_ratio_is_predicted_over_actual():
    ratios = ramp_ratios(build_results(late_days=0), build_history(), [SPECIES], years=1, horizon=3)
    assert len(ratios) == 1
    assert ratios.iloc[0]["0-4"] == 0.4  # 2×LOW / 5×LOW
    assert ratios.iloc[0]["15-19"] == 0.4


def test_unscored_horizon_when_the_window_is_not_covered():
    results = build_results()
    results = results[~((results["horizon_day"] == 2) & (pd.to_datetime(results["date"]) > "2024-03-18"))]
    scores = score_onsets(results, build_history(), [SPECIES], years=1).set_index("horizon_day")
    assert not scores.loc[2, "scored"]
    assert scores.loc[1, "scored"]
