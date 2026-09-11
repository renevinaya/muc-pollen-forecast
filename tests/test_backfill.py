"""The column refresh must change only what it fetched, and nothing else.

Both refreshers write into an existing history without going through the
collector, so the properties that matter are the ones a bug would break
silently: pollen values untouched, rows outside the fetched span untouched,
a short fetch never erasing data, and no rows added or lost.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backfill import NDVI_MAX_GAP_DAYS, refresh_ndvi, refresh_weather
from src.types import ALL_SPECIES, WEATHER_COLUMNS


def build_history(days: int = 30, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(1)
    windows = pd.date_range(start, periods=days * 8, freq="3h")
    rows = []
    for dt in windows:
        for sp in ALL_SPECIES[:3]:
            row = {"date": dt, "species": sp, "value": float(rng.integers(0, 50))}
            for col in WEATHER_COLUMNS:
                row[col] = float(rng.random())
            row["boundary_layer_height"] = np.nan
            row["soil_temperature_mean"] = np.nan
            row["ndvi"] = 0.0
            row["evi"] = 0.0
            row["ndvi_delta"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def fake_archive(values: float = 5.0):
    """An archive client returning constant values for every requested window."""
    calls = []

    def fetch(start: date, end: date) -> pd.DataFrame:
        calls.append((start, end))
        idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=21), freq="3h")
        frame = pd.DataFrame({col: values for col in WEATHER_COLUMNS}, index=idx)
        return frame

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    import src.backfill as backfill

    monkeypatch.setattr(backfill, "local_today", lambda: date(2024, 3, 1))


def test_weather_refresh_rewrites_weather_and_nothing_else():
    history = build_history()
    out = refresh_weather(history, fetch=fake_archive(5.0))

    assert len(out) == len(history)
    pd.testing.assert_series_equal(out["value"], history["value"])
    pd.testing.assert_series_equal(out["species"], history["species"])
    for col in WEATHER_COLUMNS:
        assert (out[col] == 5.0).all(), col
    # NDVI is not the weather refresh's business.
    assert (out["ndvi"] == 0.0).all()


def test_weather_refresh_respects_the_requested_span():
    history = build_history()
    out = refresh_weather(
        history, start=date(2024, 1, 10), end=date(2024, 1, 12), fetch=fake_archive(5.0)
    )
    day = out["date"].dt.normalize()
    inside = (day >= "2024-01-10") & (day <= "2024-01-12")
    assert (out.loc[inside, "temperature_mean"] == 5.0).all()
    assert (out.loc[~inside, "temperature_mean"] == history.loc[~inside, "temperature_mean"]).all()


def test_weather_refresh_never_erases_with_a_short_fetch():
    """A window the archive did not return keeps the value it had."""
    history = build_history()

    def short(start, end):
        frame = fake_archive(5.0)(start, end)
        return frame.iloc[:8]  # one day only

    out = refresh_weather(history, fetch=short)
    first_day = out["date"].dt.normalize() == pd.Timestamp("2024-01-01")
    assert (out.loc[first_day, "temperature_mean"] == 5.0).all()
    rest = ~first_day
    assert (out.loc[rest, "temperature_mean"] == history.loc[rest, "temperature_mean"]).all()
    assert out.loc[rest, "boundary_layer_height"].isna().all()


def test_weather_refresh_stops_at_the_archive_edge():
    """Rows newer than the archive lag come from the forecast API and stay."""
    history = build_history(days=70)  # runs to 2024-03-10, past today - 5 d
    fetch = fake_archive(5.0)
    out = refresh_weather(history, fetch=fetch)
    assert max(e for _, e in fetch.calls) == date(2024, 2, 25)
    late = out["date"] >= "2024-02-26"
    assert (out.loc[late, "temperature_mean"] == history.loc[late, "temperature_mean"]).all()


def test_weather_refresh_chunks_long_spans():
    history = build_history(days=30)
    import src.backfill as backfill

    backfill_chunk = backfill.WEATHER_CHUNK_DAYS
    backfill.WEATHER_CHUNK_DAYS = 10
    try:
        fetch = fake_archive(5.0)
        refresh_weather(history, fetch=fetch)
    finally:
        backfill.WEATHER_CHUNK_DAYS = backfill_chunk
    assert len(fetch.calls) == 3
    # Contiguous, non-overlapping.
    for (_, end), (next_start, _) in zip(fetch.calls, fetch.calls[1:]):
        assert (next_start - end).days == 1


def fake_composites(dates: list[str]):
    def fetch(start, end):
        return pd.DataFrame({"date": pd.to_datetime(dates), "ndvi": 0.4, "evi": 0.3})

    return fetch


def test_ndvi_refresh_fills_the_zero_placeholder():
    history = build_history(days=30)
    out = refresh_ndvi(history, fetch=fake_composites(["2024-01-01", "2024-01-17", "2024-02-02"]))
    assert len(out) == len(history)
    pd.testing.assert_series_equal(out["value"], history["value"])
    assert np.allclose(out["ndvi"], 0.4)
    assert np.allclose(out["evi"], 0.3)
    for col in WEATHER_COLUMNS:
        pd.testing.assert_series_equal(out[col], history[col])


def test_ndvi_refresh_leaves_days_far_from_any_composite():
    history = build_history(days=60)  # to 2024-02-29
    out = refresh_ndvi(history, fetch=fake_composites(["2024-01-01"]))
    day = out["date"].dt.normalize()
    near = (day - pd.Timestamp("2024-01-01")).dt.days <= NDVI_MAX_GAP_DAYS
    assert np.allclose(out.loc[near, "ndvi"], 0.4)
    assert (out.loc[~near, "ndvi"] == 0.0).all()


def test_ndvi_refresh_with_nothing_fetched_changes_nothing():
    history = build_history()
    out = refresh_ndvi(history, fetch=lambda s, e: pd.DataFrame())
    pd.testing.assert_frame_equal(out, history.assign(date=pd.to_datetime(history["date"])))
