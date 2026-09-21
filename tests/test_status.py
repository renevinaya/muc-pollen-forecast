"""The forecast says what it was built from (D.2 staleness guard, D.3 defaults).

A forecast built on measurements two days old used to look exactly like one
built on this morning's: same lead, same confidence, nothing in the file. Now
the origin is the newest measurement, the confidence is the confidence of the
horizon the forecast really is, and a ``status`` block names the observation
age and every input group that fell back to a default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import forecaster
from src.confidence import (
    BEYOND_HORIZON_SCALE,
    NO_MODEL_SCALE,
    build_table,
    confidence_for,
    load_table,
)
from src.forecaster import (
    WINDOW,
    generate_forecast,
    observation_status,
    windows_since,
    worst_observation,
)
from src.types import (
    ALL_SPECIES,
    FORECAST_DAYS,
    STALE_AFTER_WINDOWS,
    WEATHER_COLUMNS,
    DayForecast,
    ForecastOutput,
    ObservationStatus,
    RunStatus,
    SpeciesForecast,
    WindowForecast,
)
from tests.test_confidence import synthetic_results
from tests.test_feature_parity import SPECIES, build_history


# --- Staleness arithmetic ---------------------------------------------------


def test_windows_since_counts_complete_unmeasured_windows() -> None:
    last = pd.Timestamp("2026-09-21 06:00")
    assert windows_since(last, pd.Timestamp("2026-09-21 08:59")) == 0  # window still running
    assert windows_since(last, pd.Timestamp("2026-09-21 09:30")) == 0  # 09:00 in progress
    assert windows_since(last, pd.Timestamp("2026-09-21 13:26")) == 1  # 09:00 passed unmeasured
    assert windows_since(last, pd.Timestamp("2026-09-22 13:26")) == 9
    assert windows_since(last, pd.Timestamp("2026-09-20 13:26")) == 0  # never negative


def test_observation_status_is_per_species_and_flags_a_day_of_silence() -> None:
    now = pd.Timestamp("2026-09-21 13:26")
    history = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-09-21 06:00"), pd.Timestamp("2026-09-20 06:00")],
            "species": ["Betula", "Alnus"],
            "value": [1.0, 2.0],
        }
    )
    status, last = observation_status(history, now)
    assert set(status) == set(ALL_SPECIES)
    assert status["Betula"] == ObservationStatus("2026-09-21T06:00:00", 1, False)
    assert status["Alnus"].age_windows == 9 and status["Alnus"].stale
    assert status["Poaceae"] == ObservationStatus(None, None, True)  # never measured
    assert last["Betula"] == pd.Timestamp("2026-09-21 06:00")
    assert worst_observation({"a": status["Betula"], "b": status["Alnus"]}) is status["Alnus"]
    assert worst_observation({}).stale


def test_stale_threshold_is_a_day() -> None:
    assert STALE_AFTER_WINDOWS == 8


# --- Confidence beyond the shipped horizon -----------------------------------


def test_confidence_beyond_the_measured_horizon_is_marked_down() -> None:
    table = build_table(synthetic_results())  # horizons 1..5
    last_measured, _ = confidence_for(table, "Betula", "low", 5)
    beyond, _ = confidence_for(table, "Betula", "low", 6)
    far_beyond, _ = confidence_for(table, "Betula", "low", 30)
    assert beyond == pytest.approx(last_measured * BEYOND_HORIZON_SCALE, abs=1e-6)
    assert far_beyond == beyond
    without_model, _ = confidence_for(table, "Betula", "low", 6, has_model=False)
    assert without_model == pytest.approx(beyond * NO_MODEL_SCALE, abs=1e-6)


def test_table_overall_rate_is_the_shipped_horizons_only() -> None:
    """Rows past FORECAST_DAYS calibrate stale runs; they must not move the headline."""
    results = synthetic_results(accuracy=0.60)
    far = synthetic_results(accuracy=0.10, seed=5)
    far["horizon_day"] = far["horizon_day"] + FORECAST_DAYS  # days 6..10, much worse
    table = build_table(pd.concat([results, far], ignore_index=True))
    assert table["overall"]["exact"] == pytest.approx(0.60, abs=0.03)
    assert set(table["horizon_delta"]) == {str(d) for d in range(1, 2 * FORECAST_DAYS + 1)}
    assert table["source"]["horizon_days"] == 2 * FORECAST_DAYS
    day_10 = table["overall"]["exact"] + table["horizon_delta"]["10"]["exact"]
    assert day_10 == pytest.approx(0.10, abs=0.05)


# --- Output --------------------------------------------------------------------


def _status() -> RunStatus:
    obs = ObservationStatus("2026-09-19T06:00:00", 17, True)
    return RunStatus(
        observations=obs,
        species={"Betula": obs, "Alnus": ObservationStatus("2026-09-21T06:00:00", 1, False)},
        defaulted={"ndvi": "fetch failed (timeout); zeros", "dwd": "unavailable; blend skipped"},
    )


def test_status_is_published_in_both_output_shapes() -> None:
    window = WindowForecast("06:00", "09:00", [SpeciesForecast("Betula", "low", 3.0, 0.3, 0.9)])
    out = ForecastOutput(
        "2026-09-21T11:00:00Z", "DEMUNC", [DayForecast("2026-09-21", [window])], status=_status()
    )
    for shape in (out.to_dict(), out.to_web_dict()):
        status = shape["status"]
        assert status["degraded"] is True
        assert status["observations"]["stale"] is True
        assert status["observations"]["age_windows"] == 17
        assert status["observations"]["species"]["Alnus"]["stale"] is False
        assert set(status["defaulted"]) == {"dwd", "ndvi"}
    # A run with nothing missing publishes the block too, so its absence is
    # never mistaken for "all fine".
    fresh = RunStatus(ObservationStatus("2026-09-21T06:00:00", 1, False))
    assert not fresh.degraded
    assert "all inputs present" in fresh.describe()
    assert "STALE" in _status().describe()
    assert "status" not in ForecastOutput("x", "DEMUNC").to_web_dict()


# --- The forecast run ----------------------------------------------------------


class _LeadModel:
    """A stand-in model whose prediction encodes the lead it was asked for."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.log1p(x["lead_windows"].to_numpy(dtype=float))


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return build_history(days=140)  # Jan–May 2021: takes in the birch season


def _run(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame, last_obs: pd.Timestamp):
    """Forecast from a history that ends at *last_obs*, weather from 04-20 on."""
    start = pd.Timestamp("2021-04-20")
    weather_cols = [c for c in WEATHER_COLUMNS if c in frame.columns]
    weather = (
        frame[(frame["date"] >= start) & (frame["date"] < start + pd.Timedelta(days=FORECAST_DAYS))]
        .groupby("date")[weather_cols]
        .first()
        .sort_index()
    )
    history = frame[frame["date"] <= last_obs].reset_index(drop=True)

    monkeypatch.setattr(forecaster, "fetch_weather_forecast", lambda days: weather)
    monkeypatch.setattr(forecaster, "fetch_cams_forecast", lambda days: pd.DataFrame())
    monkeypatch.setattr(forecaster, "local_now", lambda: pd.Timestamp("2021-04-20 13:30"))

    import src.dwd
    import src.ndvi

    monkeypatch.setattr(src.dwd, "fetch_dwd_forecast", lambda: pd.DataFrame(columns=["date", "species", "dwd_level"]))

    def no_ndvi(days):
        raise RuntimeError("timeout")

    monkeypatch.setattr(src.ndvi, "ndvi_features", no_ndvi)
    return generate_forecast(history, models={SPECIES: _LeadModel()})


def _first_value(out: ForecastOutput, species: str, day: int = 0) -> float:
    for window in out.forecast[day].windows:
        for sp in window.species:
            if sp.name == species:
                return sp.value
    raise AssertionError("species not emitted")


def test_fresh_run_reports_its_inputs(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    out = _run(monkeypatch, frame, last_obs=pd.Timestamp("2021-04-20 06:00"))
    status = out.status
    assert status is not None
    assert status.observations.age_windows == 1 and not status.observations.stale
    assert status.species[SPECIES].last == "2021-04-20T06:00:00"
    assert set(status.defaulted) == {"ndvi", "dwd", "upwind", "model"}
    assert "timeout" in status.defaulted["ndvi"]
    assert SPECIES not in status.defaulted["model"]
    assert status.degraded
    # 00:00–06:00 are observed; 09:00 is the first forecast window at lead 1.
    first = out.forecast[0].windows
    assert first[3].from_time == "09:00"
    assert _first_value_at(first[3], SPECIES) == pytest.approx(1.0)


def _first_value_at(window: WindowForecast, species: str) -> float:
    return next(sp.value for sp in window.species if sp.name == species)


def test_stale_run_counts_its_lead_from_the_newest_measurement(
    monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame
) -> None:
    last = pd.Timestamp("2021-04-18 06:00")  # the station went quiet two days ago
    out = _run(monkeypatch, frame, last_obs=last)
    status = out.status
    assert status is not None
    assert status.observations.stale
    assert status.observations.age_windows == windows_since(last, pd.Timestamp("2021-04-20 13:30"))
    assert status.observations.age_windows == 17

    # The first forecast window (04-20 00:00) is 13 windows after the origin
    # (04-18 09:00), so it is lead 14 — not lead 1 over a two-day-old block.
    first = out.forecast[0].windows[0]
    expected_lead = int((pd.Timestamp("2021-04-20") - (last + WINDOW)) / WINDOW) + 1
    assert expected_lead == 14
    assert _first_value_at(first, SPECIES) == pytest.approx(float(expected_lead))

    # And its confidence is that of the horizon it really is, read from the
    # shipped table: lead 14 is horizon day 2, and the last forecast day
    # (lead 46) is horizon day 6 — beyond the shipped five, which the table
    # covers only because the calibration rollout runs to ten days.
    table = load_table()
    assert table is not None

    def published(day: int, window: int) -> float:
        return next(sp for sp in out.forecast[day].windows[window].species if sp.name == SPECIES).confidence

    def expected(lead: int) -> float:
        level = "low"  # unused by the table, see confidence_for
        return confidence_for(table, SPECIES, level, horizon_day=(lead - 1) // 8 + 1)[0]

    assert published(0, 0) == pytest.approx(expected(14), abs=1e-6)
    assert published(4, 0) == pytest.approx(expected(46), abs=1e-6)
    assert (46 - 1) // 8 + 1 == 6 and "6" in table["horizon_delta"]

    fresh = _run(monkeypatch, frame, last_obs=pd.Timestamp("2021-04-20 06:00"))
    fresh_day1 = next(sp for sp in fresh.forecast[0].windows[3].species if sp.name == SPECIES).confidence
    assert fresh_day1 == pytest.approx(expected(1), abs=1e-6)
    assert published(4, 0) < published(0, 0) < fresh_day1


def test_far_beyond_the_table_the_rate_is_halved(
    monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame
) -> None:
    """Twelve silent days put every window past the ten measured horizons."""
    out = _run(monkeypatch, frame, last_obs=pd.Timestamp("2021-04-08 06:00"))
    table = load_table()
    assert table is not None
    furthest = max(int(k) for k in table["horizon_delta"])
    last_measured = confidence_for(table, SPECIES, "low", horizon_day=furthest)[0]
    first = next(sp for sp in out.forecast[0].windows[0].species if sp.name == SPECIES)
    lead = int((pd.Timestamp("2021-04-20") - pd.Timestamp("2021-04-08 09:00")) / WINDOW) + 1
    assert (lead - 1) // 8 + 1 > furthest
    assert first.confidence == pytest.approx(last_measured * BEYOND_HORIZON_SCALE, abs=1e-6)
    assert out.status is not None and out.status.observations.age_windows == 97
