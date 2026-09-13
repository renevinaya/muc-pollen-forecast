"""The load features must be causal, neutral when unknown, and season-aligned.

They are the only features that cross a season boundary, so the property that
matters most is the one a bug would break silently: a row in season year Y
must never read anything measured in Y or later.
"""

import numpy as np
import pandas as pd
import pytest

from src.season_load import (
    boundary_month,
    load_feature_frame,
    load_features_for_years,
    season_totals,
    season_year,
)
from src.types import LOAD_FEATURES


def build_history(totals_by_year: dict[int, float], species: str = "Betula") -> pd.DataFrame:
    """Daily rows; each April carries that year's total spread over 30 days."""
    rows = []
    years = sorted(totals_by_year)
    days = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    for day in days:
        value = totals_by_year[day.year] / 30.0 if day.month == 4 else 0.0
        rows.append({"date": day, "species": species, "value": value})
    return pd.DataFrame(rows)


def test_boundary_sits_after_the_shoulder_month():
    assert boundary_month("Betula") == 7      # Mar–May core, June shoulder
    assert boundary_month("Corylus") == 6     # Jan–Apr core, May shoulder
    assert boundary_month("Ambrosia") == 12   # Jul–Oct core, Nov shoulder


def test_season_year_wraps_a_december_start_into_the_next_year():
    dates = pd.DatetimeIndex(["2023-12-15", "2024-02-01", "2024-05-31", "2024-06-01"])
    assert list(season_year("Corylus", dates)) == [2024, 2024, 2024, 2025]


def test_totals_only_count_completed_observed_seasons():
    history = build_history({2020: 300.0, 2021: 900.0, 2022: 600.0})
    totals = season_totals(history, "Betula")
    assert totals == pytest.approx({2020: 300.0, 2021: 900.0, 2022: 600.0})

    # Cut the history off inside 2022's season year: 2022 is not complete.
    partial = history[history["date"] < "2022-06-15"]
    assert set(season_totals(partial, "Betula")) == {2020, 2021}

    # A season with hardly any core-month rows is a fragment, not a season.
    sparse = history[~((history["date"].dt.year == 2021) & (history["date"].dt.month.isin([3, 4])))]
    assert 2021 not in season_totals(sparse, "Betula")


def test_features_are_log_ratios_and_zero_when_unknown():
    totals = {2020: 300.0, 2021: 900.0, 2022: 600.0}
    feats = load_features_for_years(totals, [2020, 2021, 2022, 2023])
    log = np.log1p
    assert feats[2020] == {name: 0.0 for name in LOAD_FEATURES}          # nothing before
    assert feats[2021] == {name: 0.0 for name in LOAD_FEATURES}          # prev, no baseline
    assert feats[2022]["load_prev_anom"] == pytest.approx(log(900) - log(300))
    assert feats[2022]["load_trend"] == pytest.approx(log(900) - log(300))
    assert feats[2022]["load_2y_anom"] == 0.0                            # no baseline before 2020
    assert feats[2023]["load_prev_anom"] == pytest.approx(log(600) - np.mean([log(300), log(900)]))
    assert feats[2023]["load_trend"] == pytest.approx(log(600) - log(900))
    assert feats[2023]["load_2y_anom"] == pytest.approx((log(600) + log(900)) / 2 - log(300))


def test_a_row_never_reads_its_own_or_a_later_season():
    base = build_history({2020: 300.0, 2021: 900.0, 2022: 600.0, 2023: 100.0})
    louder = base.copy()
    louder.loc[louder["date"].dt.year >= 2022, "value"] *= 10  # change 2022 and 2023 only

    dates = pd.DatetimeIndex(pd.date_range("2022-01-01", "2022-06-30", freq="7D"))
    pd.testing.assert_frame_equal(
        load_feature_frame(base, "Betula", dates), load_feature_frame(louder, "Betula", dates)
    )
    # ...while rows in 2023 do see the changed 2022.
    later = pd.DatetimeIndex(["2023-04-01"])
    assert not load_feature_frame(base, "Betula", later).equals(
        load_feature_frame(louder, "Betula", later)
    )


def test_features_are_constant_within_a_season_year():
    history = build_history({2020: 300.0, 2021: 900.0, 2022: 600.0})
    dates = pd.DatetimeIndex(pd.date_range("2021-07-01", "2022-06-30", freq="D"))
    frame = load_feature_frame(history, "Betula", dates)
    assert (frame.nunique() == 1).all()
