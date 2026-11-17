"""NDVI on a run MODIS does not answer, and how much a run asks MODIS for."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src import collector, ndvi
from src.types import ALL_SPECIES, WEATHER_ONLY


def _write_history(path, days, values):
    rows = []
    for day, value in zip(days, values):
        for hour in (0, 12):
            rows.append({"date": day + pd.Timedelta(hours=hour), "species": "Fungus",
                         "value": 1.0, "ndvi": value, "ndvi_delta": 0.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_composites_from_history_leave_out_what_a_run_without_ndvi_wrote(tmp_path):
    path = tmp_path / "history.csv"
    days = pd.date_range("2026-11-01", periods=6, freq="D")
    _write_history(path, days, [0.21, 0.20, 0.19, 0.18, 0.0, 0.0])

    comps = ndvi.composites_from_history(path)

    assert list(comps["date"]) == list(days[:4])
    assert comps["ndvi"].iloc[-1] == 0.18
    # Interpolated onto the days after, the last stored value carries on.
    later = pd.date_range("2026-11-05", periods=10, freq="D")
    assert np.allclose(ndvi.interpolate_ndvi(comps, later)["ndvi"], 0.18)


def test_composites_from_history_without_a_history(tmp_path):
    assert ndvi.composites_from_history(tmp_path / "absent.csv").empty


def test_collect_carries_the_stored_ndvi_when_modis_fails(monkeypatch, tmp_path):
    """2026-11-14 14:19: every ORNL request timed out and the run wrote zeros."""
    path = tmp_path / "history.csv"
    _write_history(path, pd.date_range("2026-10-15", periods=20, freq="D"),
                   np.linspace(0.25, 0.17, 20))
    start = pd.Timestamp("2026-11-08 00:00")
    windows = pd.date_range(start, "2026-11-10 09:00", freq="3h")
    weather = pd.DataFrame({"temperature_mean": 4.0, "precipitation_sum": 0.0}, index=windows)
    station = pd.DataFrame({"date": [start], "species": ["Fungus"], "value": [12.0]})
    cached = []

    monkeypatch.setattr(collector, "local_today", lambda: date(2026, 11, 10))
    monkeypatch.setattr(collector, "local_now", lambda: pd.Timestamp("2026-11-10 10:30"))
    monkeypatch.setattr(collector, "_fetch_station", lambda s, e: station)
    monkeypatch.setattr(collector, "fetch_upwind", lambda s, e: pd.DataFrame())
    monkeypatch.setattr(collector, "fetch_historical_weather", lambda s, e: weather.iloc[:0])
    monkeypatch.setattr(collector, "fetch_weather_forecast", lambda days, past_days=0: weather)
    monkeypatch.setattr(collector, "fetch_ndvi", lambda: pd.DataFrame(columns=["date", "ndvi"]))
    monkeypatch.setattr(collector, "set_composites_cache", cached.append)
    monkeypatch.setattr(collector, "HISTORY_FILE", path)

    frame = collector.collect(days=2)

    assert len(frame) == len(ALL_SPECIES) + (len(windows) - 1)
    assert np.allclose(frame["ndvi"], 0.17)
    assert (frame["ndvi_delta"] == 0).all()
    assert frame[frame["species"] == WEATHER_ONLY]["ndvi"].notna().all()
    # The forecaster reads the same composites, not an empty table.
    assert len(cached) == 1 and not cached[0].empty


def test_a_run_asks_modis_for_one_request_worth(monkeypatch, tmp_path):
    asked = []

    def fake_chunk(start, end):
        asked.append((start, end))
        return {"subset": [{"calendar_date": "2026-10-16", "band": ndvi.NDVI_BAND, "data": [1800]}]}

    monkeypatch.setattr(ndvi, "_fetch_chunk", fake_chunk)
    monkeypatch.setattr(ndvi, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ndvi, "NDVI_CACHE", tmp_path / "ndvi_cache.csv")
    monkeypatch.setattr(ndvi, "local_today", lambda: date(2026, 11, 17))

    comps = ndvi.fetch_ndvi()

    assert asked == [(date(2026, 11, 17) - timedelta(days=ndvi.RECENT_DAYS), date(2026, 11, 17))]
    assert comps["ndvi"].tolist() == [0.18]
