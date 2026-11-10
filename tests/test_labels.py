"""The official station series as the one source of measurements (G.1).

Three things are pinned here:

* the LGL client returns one value per station, taxon and window;
* relabelling a history replaces what counts as a measurement and nothing
  else — the windows and their weather stay;
* a window the station did not report keeps its weather as a weather-only
  row, which the weather features see and the measurement code does not.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import collector
from src.backfill import relabel_history
from src.collector import drop_superseded_weather_rows
from src.features import LagState
from src.forecaster import observation_status
from src.onset import daily_temperature
from src.pollen import parse_measurements, pivot_pollen
from src.rollout import rollout_evaluate
from src.trainer import prepare_training_data
from src.types import ALL_SPECIES, WEATHER_ONLY

from tests.test_feature_parity import build_history

SPECIES = "Betula"


def _ts(text: str) -> int:
    return int(pd.Timestamp(text, tz="Europe/Berlin").timestamp())


# --- The client -------------------------------------------------------------


def test_parse_measurements_is_one_value_per_station_taxon_window() -> None:
    payload = {
        "measurements": [
            {"polle": "Betula", "location": "DEMUNC", "data": [
                {"from": _ts("2026-04-05 09:00"), "to": _ts("2026-04-05 12:00"), "value": 92.9},
                {"from": _ts("2026-04-05 12:00"), "to": _ts("2026-04-05 15:00"), "value": 95.5},
            ]},
            {"polle": "Betula", "location": "DEMIND", "data": [
                {"from": _ts("2026-04-05 09:00"), "to": _ts("2026-04-05 12:00"), "value": 3.3},
            ]},
            # Not one of our taxa: dropped.
            {"polle": "Varia", "location": "DEMUNC", "data": [
                {"from": _ts("2026-04-05 09:00"), "to": _ts("2026-04-05 12:00"), "value": 7.0},
            ]},
            {"polle": "Alnus", "location": "DEMUNC", "data": []},
        ]
    }
    frame = parse_measurements(payload)
    assert list(frame.columns) == ["date", "station", "species", "value"]
    assert len(frame) == 3
    assert not frame.duplicated(["date", "station", "species"]).any()
    munich = frame[frame["station"] == "DEMUNC"].set_index("date")["value"]
    # Naive Munich wall clock, on the 3h grid.
    assert munich.loc[pd.Timestamp("2026-04-05 09:00")] == pytest.approx(92.9)
    assert set(frame["species"]) == {"Betula"}


def test_parse_measurements_averages_the_repeated_hour_of_the_clock_change() -> None:
    """On the last Sunday of October 02:00–03:00 happens twice; both land in one window."""
    first = int(pd.Timestamp("2025-10-26 00:00", tz="UTC").timestamp())  # 02:00 CEST
    second = int(pd.Timestamp("2025-10-26 01:00", tz="UTC").timestamp())  # 02:00 CET
    payload = {"measurements": [{"polle": "Fungus", "location": "DEMUNC", "data": [
        {"from": first, "value": 10.0}, {"from": second, "value": 30.0},
    ]}]}
    frame = parse_measurements(payload)
    assert len(frame) == 1
    assert frame.iloc[0]["value"] == pytest.approx(20.0)


def test_pivot_fills_an_unlisted_taxon_with_zero() -> None:
    long = pd.DataFrame({
        "date": pd.to_datetime(["2026-04-05 09:00", "2026-04-05 09:00", "2026-04-05 12:00"]),
        "species": ["Betula", "Alnus", "Betula"],
        "value": [5.0, 1.0, 7.0],
    })
    wide = pivot_pollen(long)
    assert wide.loc[pd.Timestamp("2026-04-05 12:00"), "Alnus"] == 0.0


# --- Relabelling ------------------------------------------------------------


def _small_history() -> pd.DataFrame:
    windows = pd.date_range("2025-02-04 06:00", periods=4, freq="3h")
    rows = []
    for i, dt in enumerate(windows):
        for species, value in (("Alnus", 7640.0), ("Betula", 0.0)):
            rows.append({"date": dt, "species": species, "value": value,
                         "temperature_mean": 3.0 + i, "ndvi": 0.2})
    return pd.DataFrame(rows)


def test_relabel_replaces_measurements_and_keeps_windows_and_weather() -> None:
    history = _small_history()
    windows = sorted(history["date"].unique())
    # The station reported the first two windows, at zero, and one window
    # the history has never seen. It did not report the last two.
    official = pd.DataFrame({
        "date": [windows[0], windows[1], pd.Timestamp("2025-02-05 00:00")],
        "species": ["Alnus", "Alnus", "Alnus"],
        "value": [0.0, 12.5, 4.0],
    })
    out, report = relabel_history(history, official)

    assert report["windows_reported"] == 2
    assert report["windows_weather_only"] == 2
    assert report["station_windows_without_weather"] == 1
    assert list(out.columns) == list(history.columns)
    # Same windows as before, nothing added or lost.
    assert sorted(out["date"].unique()) == windows

    reported = out[out["date"].isin(windows[:2])]
    # One row per taxon of the model, the station's value or 0.
    assert sorted(reported["species"].unique()) == sorted(ALL_SPECIES)
    alnus = reported[reported["species"] == "Alnus"].set_index("date")["value"]
    assert alnus.loc[windows[0]] == 0.0 and alnus.loc[windows[1]] == 12.5
    assert (reported[reported["species"] == "Fungus"]["value"] == 0.0).all()

    rest = out[out["date"].isin(windows[2:])]
    assert (rest["species"] == WEATHER_ONLY).all() and len(rest) == 2
    assert rest["value"].isna().all()
    # Weather travels with the window, whatever happens to its measurements.
    for dt in windows:
        before = history[history["date"] == dt]["temperature_mean"].iloc[0]
        assert (out[out["date"] == dt]["temperature_mean"] == before).all()


def test_relabel_is_idempotent() -> None:
    history = _small_history()
    official = history[history["species"] == "Alnus"].assign(value=1.0)[["date", "species", "value"]]
    once, _ = relabel_history(history, official)
    twice, report = relabel_history(once, official)
    pd.testing.assert_frame_equal(once, twice)
    assert report["values_changed"] == 0


def test_placeholder_goes_when_the_window_is_measured() -> None:
    dt = pd.Timestamp("2026-11-10 06:00")
    later = pd.Timestamp("2026-11-10 09:00")
    frame = pd.DataFrame({
        "date": [dt, dt, later],
        "species": [WEATHER_ONLY, "Betula", WEATHER_ONLY],
        "value": [np.nan, 0.0, np.nan],
    })
    out = drop_superseded_weather_rows(frame)
    assert list(out["species"]) == ["Betula", WEATHER_ONLY]


# --- Weather-only rows through the pipeline ----------------------------------


@pytest.fixture(scope="module")
def full() -> pd.DataFrame:
    return build_history(days=200, species_list=[SPECIES])


def _with_outage(history: pd.DataFrame, start: str, end: str, keep_weather: bool) -> pd.DataFrame:
    """The station silent in [start, end): rows dropped, or turned into weather-only rows."""
    dates = pd.to_datetime(history["date"])
    inside = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    if not keep_weather:
        return history[~inside].reset_index(drop=True)
    placeholders = history[inside].drop_duplicates("date").assign(species=WEATHER_ONLY, value=np.nan)
    return pd.concat([history[~inside], placeholders], ignore_index=True).sort_values(
        ["date", "species"]
    ).reset_index(drop=True)


OUTAGE = ("2021-04-20 00:00", "2021-04-24 00:00")
AFTER = pd.Timestamp("2021-04-24 09:00")


def test_weather_features_do_not_notice_a_station_outage(full: pd.DataFrame) -> None:
    """With weather-only rows the rolling weather after an outage is the true one.

    Without them — the history as it was — "the last seven days" at the
    first window after a four-day outage reaches eleven days back.
    """
    reference, _, _ = prepare_training_data(full, SPECIES, leads=(1,))
    kept, _, _ = prepare_training_data(_with_outage(full, *OUTAGE, keep_weather=True), SPECIES, leads=(1,))
    lost, _, _ = prepare_training_data(_with_outage(full, *OUTAGE, keep_weather=False), SPECIES, leads=(1,))

    def at(matrix: pd.DataFrame, history: pd.DataFrame, when: pd.Timestamp) -> pd.Series:
        sp = history[(history["species"] == SPECIES) & history["value"].notna()]
        # prepare_training_data keeps the species' rows in date order, minus
        # the leading rows without a full lag block; align from the end.
        dates = pd.DatetimeIndex(sp.sort_values("date")["date"])[-len(matrix):]
        return matrix.iloc[int(np.where(dates == when)[0][0])]

    cols = ["temp_rolling_7d", "rain_rolling_7d", "gdd", "dry_streak", "temp_delta_3d"]
    ref = at(reference, full, AFTER)[cols]
    with_rows = at(kept, _with_outage(full, *OUTAGE, keep_weather=True), AFTER)[cols]
    without = at(lost, _with_outage(full, *OUTAGE, keep_weather=False), AFTER)[cols]
    assert np.allclose(ref.to_numpy(float), with_rows.to_numpy(float))
    assert not np.allclose(ref.to_numpy(float), without.to_numpy(float))


def test_outage_windows_are_not_training_rows(full: pd.DataFrame) -> None:
    gapped = _with_outage(full, *OUTAGE, keep_weather=True)
    X, y, raw = prepare_training_data(gapped, SPECIES, leads=(1,))
    X_ref, _, _ = prepare_training_data(full, SPECIES, leads=(1,))
    assert len(X) == len(X_ref) - 32  # four days of windows
    assert not y.isna().any() and not raw.isna().any()


def test_daily_temperature_weights_windows_not_rows(full: pd.DataFrame) -> None:
    """A day with four measured windows and four weather-only ones is still eight windows."""
    gapped = _with_outage(full, "2021-04-20 00:00", "2021-04-20 12:00", keep_weather=True)
    day = pd.Timestamp("2021-04-20")
    assert daily_temperature(gapped).loc[day] == pytest.approx(daily_temperature(full).loc[day])


def test_measurement_code_ignores_weather_only_rows(full: pd.DataFrame) -> None:
    """The newest measurement is the newest *measurement*; the lag block has a gap, not zeros."""
    end = pd.Timestamp(full["date"].max()) + pd.Timedelta(hours=3)
    gapped = _with_outage(full, "2021-07-10 00:00", str(end), keep_weather=True)
    last_measured = pd.Timestamp("2021-07-09 21:00")

    status, newest = observation_status(gapped, pd.Timestamp(end))
    assert newest[SPECIES] == last_measured
    assert status[SPECIES].stale

    lag = LagState.from_history(gapped, SPECIES, pd.Timestamp("2021-07-12 00:00"))
    reference = LagState.from_history(
        _with_outage(full, "2021-07-10 00:00", str(end), keep_weather=False),
        SPECIES, pd.Timestamp("2021-07-12 00:00"),
    )
    assert lag.log_vals == reference.log_vals
    assert lag.days_since_active == reference.days_since_active


def test_rollout_launches_and_scores_on_reported_windows_only(full: pd.DataFrame) -> None:
    history = build_history(days=400, species_list=[SPECIES])
    gapped = _with_outage(history, "2021-12-05 00:00", "2021-12-09 00:00", keep_weather=True)
    results = rollout_evaluate(
        gapped, horizon_days=2, species=[SPECIES], months=[pd.Period("2021-12", "M")]
    )
    assert not results.empty
    outage = (results["date"] >= "2021-12-05") & (results["date"] < "2021-12-09")
    assert not outage.any()
    origins = pd.DatetimeIndex(results["origin"].unique())
    assert not ((origins >= "2021-12-05") & (origins < "2021-12-09")).any()
    assert results["actual"].notna().all()


# --- The collector ----------------------------------------------------------


def test_collect_writes_weather_only_rows_for_unreported_windows(monkeypatch) -> None:
    """The station reported the first two windows of the span; the rest keep their weather."""
    today = date(2026, 11, 10)
    start = pd.Timestamp("2026-11-08 00:00")
    weather_windows = pd.date_range(start - pd.Timedelta(days=1), "2026-11-10 09:00", freq="3h")
    weather = pd.DataFrame(
        {"temperature_mean": np.arange(len(weather_windows), dtype=float),
         "precipitation_sum": 0.0},
        index=weather_windows,
    )
    station = pd.DataFrame({
        "date": [start, start, start + pd.Timedelta(hours=3)],
        "species": ["Betula", "Fungus", "Fungus"],
        "value": [0.0, 12.0, 15.0],
    })
    monkeypatch.setattr(collector, "local_today", lambda: today)
    monkeypatch.setattr(collector, "local_now", lambda: pd.Timestamp("2026-11-10 10:30"))
    monkeypatch.setattr(collector, "_fetch_station", lambda s, e: station)
    monkeypatch.setattr(collector, "fetch_upwind", lambda s, e: pd.DataFrame())
    monkeypatch.setattr(collector, "fetch_historical_weather", lambda s, e: weather.iloc[:0])
    monkeypatch.setattr(collector, "fetch_weather_forecast", lambda days: weather)
    monkeypatch.setattr(collector, "fetch_ndvi", lambda: pd.DataFrame(columns=["date", "ndvi", "evi"]))
    monkeypatch.setattr(collector, "set_composites_cache", lambda composites: None)

    frame = collector.collect(days=2)

    reported = frame[frame["species"] != WEATHER_ONLY]
    assert sorted(reported["date"].unique()) == [start, start + pd.Timedelta(hours=3)]
    assert len(reported) == 2 * len(ALL_SPECIES)
    fungus = reported[reported["species"] == "Fungus"].set_index("date")["value"]
    assert fungus.loc[start] == 12.0
    # Betula was not listed in the second window the station reported: zero.
    assert reported[(reported["species"] == "Betula")]["value"].tolist() == [0.0, 0.0]

    placeholders = frame[frame["species"] == WEATHER_ONLY]
    assert placeholders["value"].isna().all()
    # Every later window up to now, once each; none before the span asked for.
    expected = pd.date_range(start + pd.Timedelta(hours=6), "2026-11-10 09:00", freq="3h")
    assert list(placeholders["date"]) == list(expected)
    assert placeholders["temperature_mean"].notna().all()
