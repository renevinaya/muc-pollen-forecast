"""The upwind-station block: anchored at the origin, built on the time grid."""

import numpy as np
import pandas as pd

from src.features import LagState
from src.season_load import season_year
from src.upwind import (
    SEASON_COLUMNS,
    UPWIND_STATIONS,
    local_season_sum,
    season_frame,
    update_upwind,
    upwind_block,
    upwind_series,
    upwind_state,
    upwind_tables,
    with_local_features,
)

WINDOW = pd.Timedelta(hours=3)


def _frame(start: str = "2025-03-01", days: int = 20) -> pd.DataFrame:
    dates = pd.date_range(start, periods=8 * days, freq="3h")
    rows = []
    for i, dt in enumerate(dates):
        rows.append({"date": dt, "station": "DEAAAA", "species": "Betula", "value": float(i % 7)})
        if i % 3 == 0:  # the second station reports every third window
            rows.append({"date": dt, "station": "DEBBBB", "species": "Betula", "value": 20.0})
    return pd.DataFrame(rows)


def _two_years() -> pd.DataFrame:
    """Two birch seasons, the second twice as heavy, plus a quiet third start."""
    parts = []
    for year, scale in ((2024, 1.0), (2025, 2.0), (2026, 0.5)):
        dates = pd.date_range(f"{year}-03-20", periods=8 * 30, freq="3h")
        doy = np.arange(len(dates))
        value = scale * 100.0 * np.exp(-((doy - 100) / 60.0) ** 2)
        parts.append(pd.DataFrame({"date": dates, "station": "DEAAAA", "species": "Betula", "value": value}))
    return pd.concat(parts, ignore_index=True)


def test_series_takes_the_highest_station_per_window() -> None:
    series = upwind_series(_frame(), "Betula")
    assert series.iloc[0] == np.log1p(20.0)  # DEBBBB reports at i=0
    assert series.iloc[1] == np.log1p(1.0)   # only DEAAAA at i=1
    assert upwind_series(_frame(), "Alnus").empty
    assert upwind_series(None, "Betula").empty


def test_block_is_anchored_lead_windows_back() -> None:
    frame = _frame()
    tables = upwind_tables(frame, "Betula")
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    lead1 = upwind_block(tables, dates, lead=1)
    lead8 = upwind_block(tables, dates, lead=8)
    # The state one window before t at lead 1 equals the state eight windows
    # before t + 7 at lead 8, for every column.
    for col in ("upwind_max_8", "upwind_season_sum", "upwind_season_anom"):
        a = lead1[col].to_numpy()[10:-7]
        b = lead8[col].to_numpy()[17:]
        assert np.allclose(a, b, equal_nan=True), col
    assert np.isnan(lead1["upwind_max_8"].iloc[0])  # nothing before the first reading


def test_state_matches_block_at_every_origin() -> None:
    frame = _two_years()
    tables = upwind_tables(frame, "Betula")
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    for lead in (1, 4, 24):
        block = upwind_block(tables, dates, lead=lead)
        for dt in list(dates[60:66]) + list(dates[300:306]):
            origin = dt - (lead - 1) * WINDOW
            state = upwind_state(tables, origin)
            for col in block.columns:
                assert np.isclose(state[col], block.loc[dt, col], equal_nan=True), (col, lead, dt)


def test_gap_and_absence_are_nan_not_zero() -> None:
    tables = upwind_tables(_frame(), "Betula")
    origin = pd.Timestamp("2025-03-01")  # nothing strictly before the first reading
    state = upwind_state(tables, origin)
    assert all(np.isnan(state[c]) for c in state)
    late = upwind_state(tables, pd.Timestamp("2025-06-01"))
    assert np.isnan(late["upwind_max_56"])
    assert not np.isnan(late["upwind_season_sum"]), "the season sum persists past the last reading"
    assert np.isnan(upwind_state(upwind_tables(None, "Betula"), origin)["upwind_max_8"])
    empty = upwind_block(upwind_tables(None, "Betula"), pd.DatetimeIndex([origin]), lead=1)
    assert empty.isna().all().all()


def test_block_extends_past_the_last_reading() -> None:
    """A target after the last upwind report still sees the state before its origin."""
    tables = upwind_tables(_frame(), "Betula")
    last = tables.series.index.max()
    later = pd.DatetimeIndex([last + 5 * WINDOW])
    block = upwind_block(tables, later, lead=1)
    state = upwind_state(tables, later[0])
    assert not np.isnan(block["upwind_max_8"].iloc[0])
    for col in block.columns:
        assert np.isclose(block[col].iloc[0], state[col], equal_nan=True), col


def test_season_sum_accumulates_and_anomaly_reads_the_heavier_year() -> None:
    frame = _two_years()
    season = season_frame(upwind_series(frame, "Betula"), "Betula")
    years = season_year("Betula", season.index)
    first, second, third = (season[years == y] for y in (2024, 2025, 2026))
    assert (np.diff(first["upwind_season_sum"].to_numpy()) >= -1e-9).all()
    assert (first["upwind_season_anom"] == 0).all(), "no earlier year to compare with"
    # The second season is twice the first at every day: anomaly ≈ log 2 in
    # April, where both are flowering (the leap year shifts the day by one).
    april = second[second.index.month == 4]["upwind_season_anom"].to_numpy()
    assert np.allclose(april, np.log(2.0), atol=0.2)       # early April: leap-year day offset
    assert np.allclose(april[-80:], np.log(2.0), atol=0.01)  # last ten days: converged
    # The third is half the median of the first two (1.5x): anomaly ≈ log(1/3).
    april3 = third[third.index.month == 4]["upwind_season_anom"].to_numpy()
    assert np.allclose(april3[-80:], np.log(0.5 / 1.5), atol=0.02)  # log1p curvature at small counts


def test_local_season_sum_restarts_at_the_boundary() -> None:
    dates = pd.date_range("2025-06-29", periods=8 * 4, freq="3h")  # Betula boundary: 1 July
    years = season_year("Betula", dates)
    values = np.ones(len(dates))
    out = local_season_sum(values, years, lead=1)
    july = np.flatnonzero(dates.month == 7)
    assert out[july[0]] == 0.0, "first window of the new season year sees nothing"
    assert np.isclose(out[july[1]], np.log1p(1.0))
    assert np.isclose(out[july[0] - 1], np.log1p(15.0))
    assert np.isnan(out[0])
    # At lead 4 the origin sits three rows earlier; the boundary shifts with it.
    out4 = local_season_sum(values, years, lead=4)
    assert out4[july[0] + 3] == 0.0 and np.isclose(out4[july[0] + 4], np.log1p(1.0))


def test_local_features_are_upwind_minus_local() -> None:
    block = pd.DataFrame(
        {"upwind_max_8": [2.0, np.nan], "upwind_max_56": [3.0, 3.0],
         "upwind_season_sum": [5.0, 5.0], "upwind_season_anom": [0.1, 0.1]}
    )
    out = with_local_features(block, np.array([0.5, 0.5]), np.array([4.0, np.nan]))
    assert out["upwind_lead_8"].iloc[0] == 1.5 and np.isnan(out["upwind_lead_8"].iloc[1])
    assert out["upwind_season_lead"].iloc[0] == 1.0 and np.isnan(out["upwind_season_lead"].iloc[1])
    assert list(out["pollen_season_sum"]) == [4.0] or np.isnan(out["pollen_season_sum"].iloc[1])


def test_lag_state_carries_the_upwind_block() -> None:
    frame = _frame()
    history = pd.DataFrame(
        {"date": pd.date_range("2025-03-01", periods=80, freq="3h"),
         "species": "Betula", "value": 1.0}
    )
    origin = pd.Timestamp("2025-03-10")
    feats = LagState.from_history(history, "Betula", origin, upwind=frame).lag_features()
    expected = upwind_state(upwind_tables(frame, "Betula"), origin)
    for col in SEASON_COLUMNS + ["upwind_max_8", "upwind_max_56"]:
        assert feats[col] == expected[col]
    assert np.isclose(feats["upwind_lead_8"], expected["upwind_max_8"] - np.log1p(1.0))
    assert np.isclose(feats["pollen_season_sum"], np.log1p(72.0))  # 9 days × 8 windows of 1
    assert np.isclose(feats["upwind_season_lead"], expected["upwind_season_sum"] - np.log1p(72.0))
    without = LagState.from_history(history, "Betula", origin).lag_features()
    assert np.isnan(without["upwind_max_8"]) and np.isnan(without["upwind_season_lead"])
    assert np.isclose(without["pollen_season_sum"], np.log1p(72.0))


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
