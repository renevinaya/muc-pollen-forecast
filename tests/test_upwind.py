"""The upwind-station block: anchored at the origin, built on the time grid."""

import numpy as np
import pandas as pd

from src.features import LagState
from src.upwind import (
    UPWIND_STATIONS,
    upwind_block,
    upwind_series,
    upwind_state,
    update_upwind,
    with_lead_feature,
)

WINDOW = pd.Timedelta(hours=3)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2025-03-01", periods=8 * 20, freq="3h")
    rows = []
    for i, dt in enumerate(dates):
        rows.append({"date": dt, "station": "DEAAAA", "species": "Betula", "value": float(i % 7)})
        if i % 3 == 0:  # the second station reports every third window
            rows.append({"date": dt, "station": "DEBBBB", "species": "Betula", "value": 20.0})
    return pd.DataFrame(rows)


def test_series_takes_the_highest_station_per_window() -> None:
    series = upwind_series(_frame(), "Betula")
    assert series.iloc[0] == np.log1p(20.0)  # DEBBBB reports at i=0
    assert series.iloc[1] == np.log1p(1.0)   # only DEAAAA at i=1
    assert upwind_series(_frame(), "Alnus").empty
    assert upwind_series(None, "Betula").empty


def test_block_is_anchored_lead_windows_back() -> None:
    frame = _frame()
    series = upwind_series(frame, "Betula")
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    lead1 = upwind_block(series, dates, lead=1)
    lead8 = upwind_block(series, dates, lead=8)
    # The 24 h max ending one window before t at lead 1 equals the one ending
    # eight windows before t + 7 at lead 8.
    a = lead1["upwind_max_8"].to_numpy()[10:-7]
    b = lead8["upwind_max_8"].to_numpy()[17:]
    assert np.allclose(a, b, equal_nan=True)
    assert np.isnan(lead1["upwind_max_8"].iloc[0])  # nothing before the first reading


def test_state_matches_block_at_every_origin() -> None:
    frame = _frame()
    series = upwind_series(frame, "Betula")
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    for lead in (1, 4, 24):
        block = upwind_block(series, dates, lead=lead)
        for dt in dates[60:70]:
            origin = dt - (lead - 1) * WINDOW
            state = upwind_state(frame, "Betula", origin)
            assert np.isclose(state["upwind_max_8"], block.loc[dt, "upwind_max_8"])
            assert np.isclose(state["upwind_max_56"], block.loc[dt, "upwind_max_56"])


def test_gap_and_absence_are_nan_not_zero() -> None:
    frame = _frame()
    origin = pd.Timestamp("2025-03-01")  # nothing strictly before the first reading
    state = upwind_state(frame, "Betula", origin)
    assert np.isnan(state["upwind_max_8"]) and np.isnan(state["upwind_max_56"])
    late = upwind_state(frame, "Betula", pd.Timestamp("2025-06-01"))
    assert np.isnan(late["upwind_max_56"])
    assert np.isnan(upwind_state(None, "Betula", origin)["upwind_max_8"])
    empty = upwind_block(pd.Series(dtype=float), pd.DatetimeIndex([origin]), lead=1)
    assert empty.isna().all().all()


def test_block_extends_past_the_last_reading() -> None:
    """A target after the last upwind report still sees the week before its origin."""
    frame = _frame()
    series = upwind_series(frame, "Betula")
    last = series.index.max()
    later = pd.DatetimeIndex([last + 5 * WINDOW])
    block = upwind_block(series, later, lead=1)
    assert not np.isnan(block["upwind_max_8"].iloc[0])
    assert np.isclose(
        block["upwind_max_8"].iloc[0], upwind_state(frame, "Betula", later[0])["upwind_max_8"]
    )


def test_lead_feature_is_upwind_minus_local() -> None:
    block = pd.DataFrame({"upwind_max_8": [2.0, np.nan], "upwind_max_56": [3.0, 3.0]})
    out = with_lead_feature(block, np.array([0.5, 0.5]))
    assert out["upwind_lead_8"].iloc[0] == 1.5
    assert np.isnan(out["upwind_lead_8"].iloc[1])


def test_lag_state_carries_the_upwind_block() -> None:
    frame = _frame()
    history = pd.DataFrame(
        {"date": pd.date_range("2025-03-01", periods=80, freq="3h"),
         "species": "Betula", "value": 1.0}
    )
    origin = pd.Timestamp("2025-03-10")
    feats = LagState.from_history(history, "Betula", origin, upwind=frame).lag_features()
    expected = upwind_state(frame, "Betula", origin)
    assert feats["upwind_max_8"] == expected["upwind_max_8"]
    assert np.isclose(feats["upwind_lead_8"], expected["upwind_max_8"] - np.log1p(1.0))
    without = LagState.from_history(history, "Betula", origin).lag_features()
    assert np.isnan(without["upwind_max_8"]) and np.isnan(without["upwind_lead_8"])


def test_update_merges_by_station(tmp_path) -> None:
    path = tmp_path / "upwind.csv"
    first = _frame()
    update_upwind(first, path)
    later = first.head(3).assign(value=99.0)
    merged = update_upwind(later, path)
    assert len(merged) == len(first)
    assert (merged.merge(later, on=["date", "station", "species"])["value_x"] == 99.0).all()


def test_station_codes_are_pollenscience_codes() -> None:
    for code in UPWIND_STATIONS:
        assert len(code) == 6 and code.isupper()
