"""Train/serve parity for the 72-feature vector.

The trainer builds features in vectorised batches over a whole history frame;
the forecaster builds them one row at a time so each prediction can feed the
next window's lags. Those are two implementations of one definition, and when
they disagree the model is served inputs it never trained on — silently, since
a forecast built from skewed features still looks like a forecast.

The check is direct: take windows out of a synthetic history, build their
features both ways, and require the numbers to match. It caught a
``days_since_active`` that was a constant 0 in training and a live count at
serving time, which is exactly the class of bug that is otherwise invisible.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import LagState, build_context, build_feature_row, ndvi_from_history
from src.trainer import (
    _add_intraday_features,
    _add_lag_features,
    _add_load_features,
    _add_ndvi_features,
    _add_phenology_features,
    _add_season_feature,
    _add_upwind_features,
    _add_weather_derived_features,
)
from src.upwind import upwind_series
from src.types import ALL_SPECIES, FEATURE_COLS, WEATHER_COLUMNS

WINDOW = pd.Timedelta(hours=3)

SPECIES = "Betula"


def _ndvi(doy: int) -> float:
    """A smooth green-up curve, used to keep NDVI daily in the fixture."""
    return 0.3 + 0.4 * max(0.0, float(np.sin(np.pi * doy / 365)))

# Features the two paths are not expected to agree on at an arbitrary window.
# Empty on purpose: any exemption here is a skew the model actually sees, so it
# belongs in a fix rather than in this list.
EXEMPT: set[str] = set()


def build_history(
    days: int = 400, seed: int = 7, species_list: list[str] | None = None
) -> pd.DataFrame:
    """A synthetic history with a plausible season and real weather variation.

    *species_list* narrows the frame to a subset; the row count is otherwise
    multiplied by all eleven species, which only matters for how long a test
    that rolls forecasts over the frame takes.
    """
    species_names = list(species_list) if species_list else list(ALL_SPECIES)
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2021-01-01")
    windows = pd.date_range(start, periods=days * 8, freq="3h")

    doy = windows.dayofyear.to_numpy()
    hour = windows.hour.to_numpy()
    temp = (
        10.0
        - 9.0 * np.cos(2 * np.pi * doy / 365.25)
        + 4.0 * np.sin(2 * np.pi * (hour - 3) / 24)
        + rng.normal(0, 1.5, len(windows))
    )

    rows = []
    for i, dt in enumerate(windows):
        # A birch season in April/May, daytime-weighted, with quiet gaps.
        in_season = 100 <= dt.dayofyear <= 140
        base = 60.0 if in_season else 0.0
        daylight = max(0.0, np.sin(np.pi * (dt.hour - 6) / 12))
        value = max(0.0, base * daylight * rng.gamma(2.0, 0.5)) if in_season else 0.0
        if in_season and rng.random() < 0.15:
            value = 0.0

        # Weather is a property of the window, not of the species: every
        # species row for one window carries identical columns, which is what
        # the collector writes and what the derived features assume.
        window = {
            "temperature_max": temp[i] + 1.0,
            "temperature_min": temp[i] - 1.0,
            "temperature_mean": temp[i],
            "precipitation_sum": float(rng.random() < 0.2) * rng.random() * 3,
            "wind_speed_max": 5 + rng.random() * 15,
            "wind_direction": rng.random() * 360,
            "humidity_mean": 50 + rng.random() * 40,
            "sunshine_duration": daylight * 3600,
            "shortwave_radiation_sum": daylight * 800,
            "boundary_layer_height": 200 + daylight * 1200,
            "dew_point_mean": temp[i] - 4,
            "cape_max": rng.random() * 50,
            "direct_radiation_sum": daylight * 600,
            "is_day": float(6 <= dt.hour <= 18),
            "temp_slope_3h": rng.normal(0, 1),
            "humidity_slope_3h": rng.normal(0, 4),
            "temp_variance_3h": abs(rng.normal(0, 1)),
            "soil_temperature_mean": temp[i] - 2,
            "soil_moisture_mean": 0.2 + rng.random() * 0.2,
            # NDVI is a daily product: the collector writes one value across
            # all eight windows of a day, so it must not vary within a day.
            "ndvi": _ndvi(dt.dayofyear),
            "evi": 0.2 + 0.3 * max(0.0, np.sin(np.pi * dt.dayofyear / 365)),
            "ndvi_delta": _ndvi(dt.dayofyear) - _ndvi(dt.dayofyear - 1),
            "cams_pollen": 0.0,
            # Calendar features live in history.csv: the collector writes them
            # at fetch time, so the trainer reads rather than derives them. The
            # forecaster derives them from the timestamp, and that pair is
            # exactly what this test has to compare.
            "day_of_year": float(dt.dayofyear),
            "day_of_year_sin": float(np.sin(2 * np.pi * dt.dayofyear / 365.25)),
            "day_of_year_cos": float(np.cos(2 * np.pi * dt.dayofyear / 365.25)),
            "month": float(dt.month),
            "hour_of_day": float(dt.hour),
            "hour_sin": float(np.sin(2 * np.pi * dt.hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * dt.hour / 24)),
        }

        for species in species_names:
            rows.append({
                "date": dt,
                "species": species,
                "value": float(value) if species == SPECIES else 0.0,
                **window,
            })
    return pd.DataFrame(rows)


def trainer_features(
    history: pd.DataFrame, species: str, lead: int = 1, upwind: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The trainer's batch path, in the order prepare_training_data applies it."""
    df = history[history["species"] == species].sort_values("date").reset_index(drop=True)
    df = _add_weather_derived_features(df, species)
    df = _add_ndvi_features(df)
    df = _add_intraday_features(df)
    df = _add_season_feature(df, species)
    df = _add_phenology_features(df, species)
    df = _add_load_features(df, species)
    df = _add_lag_features(df, lead=lead)
    df = _add_upwind_features(df, upwind_series(upwind, species), lead=lead)
    return df.set_index("date")


def build_upwind(history: pd.DataFrame, species: str, seed: int = 3) -> pd.DataFrame:
    """Two upwind stations that run a few days ahead of the local series, with
    one station dark for a stretch and a gap no station covers."""
    rng = np.random.default_rng(seed)
    sp = history[history["species"] == species].sort_values("date")
    dates = pd.DatetimeIndex(sp["date"])
    values = sp["value"].to_numpy(dtype=float)
    rows = []
    for shift, code in ((16, "DEAAAA"), (32, "DEBBBB")):
        led = np.roll(values, -shift) * rng.uniform(0.5, 2.0, len(values))
        for dt, v in zip(dates, led):
            if code == "DEBBBB" and pd.Timestamp("2021-04-10") <= dt < pd.Timestamp("2021-04-20"):
                continue  # one station dark
            if pd.Timestamp("2021-05-01") <= dt < pd.Timestamp("2021-05-03"):
                continue  # nobody reports
            rows.append({"date": dt, "station": code, "species": species, "value": float(v)})
    return pd.DataFrame(rows)


def serving_features(
    history: pd.DataFrame,
    species: str,
    windows: pd.DatetimeIndex,
    lead: int = 1,
    upwind: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The forecaster's row path over *windows*, all from one origin.

    The forecaster is direct: it snapshots the measured past once at the origin
    and predicts every window from it, varying only ``lead_windows``. So the
    comparison here fixes a lead and checks that the snapshot the forecaster
    takes equals the lead-anchored block the trainer builds.
    """
    origin = windows[0]
    past = history[history["date"] < origin]
    future = history[history["date"] >= origin]

    weather_cols = [c for c in WEATHER_COLUMNS if c in history.columns]
    weather = future.groupby("date")[weather_cols].first().sort_index()

    days = pd.DatetimeIndex(weather.index).normalize().unique()
    ctx = build_context(
        past, weather, ndvi=ndvi_from_history(history, days), species=[species],
        parallel=False,
    )

    rows = {}
    for dt in windows:
        # Each window gets its own origin, `lead` windows back, which is the
        # state the forecaster would have had when predicting it at that lead.
        lag = LagState.from_history(
            history, species, dt - lead * WINDOW + WINDOW, upwind=upwind
        )
        rows[dt] = build_feature_row(ctx, species, dt, lag, lead=lead)
    return pd.DataFrame.from_dict(rows, orient="index")


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return build_history()


@pytest.fixture(scope="module")
def upwind(history: pd.DataFrame) -> pd.DataFrame:
    return build_upwind(history, SPECIES)


@pytest.mark.parametrize("lead", [1, 8, 40])
def test_serving_matches_training_features(
    history: pd.DataFrame, upwind: pd.DataFrame, lead: int
) -> None:
    """Every feature agrees between the batch and row paths, at every lead.

    Checking lead 1 alone would miss the whole point of the direct model: the
    forecast spends 39 of its 40 windows at leads above 1, and those are exactly
    the rows a train/serve mismatch would silently corrupt.
    """
    batch = trainer_features(history, SPECIES, lead=lead, upwind=upwind)

    # Start well inside the frame so both paths have their lags warmed up, and
    # cover the season so the phenology and burst features are exercised.
    all_windows = pd.DatetimeIndex(batch.index)
    windows = all_windows[(all_windows >= all_windows[600]) & (all_windows <= all_windows[1400])]
    served = serving_features(history, SPECIES, windows, lead=lead, upwind=upwind)
    assert batch.loc[windows, "upwind_max_8"].notna().mean() > 0.9
    assert batch.loc[windows, "upwind_max_8"].isna().any(), "the uncovered gap must stay NaN"

    mismatches = []
    for col in FEATURE_COLS:
        if col in EXEMPT:
            continue
        expected = batch.loc[windows, col].to_numpy(dtype=float)
        actual = served[col].to_numpy(dtype=float)
        if not np.allclose(expected, actual, rtol=1e-6, atol=1e-6, equal_nan=True):
            worst = int(np.nanargmax(np.abs(expected - actual)))
            mismatches.append(
                f"{col}: max |Δ|={np.nanmax(np.abs(expected - actual)):.6g} "
                f"at {windows[worst]} (train={expected[worst]:.6g}, serve={actual[worst]:.6g})"
            )

    assert not mismatches, "train/serve feature skew:\n  " + "\n  ".join(mismatches)


def test_days_since_active_is_not_constant(history: pd.DataFrame) -> None:
    """The feature has to actually vary — it was a constant 0 for both paths.

    A parity test alone would have passed happily on two matching constants,
    so the range is asserted separately from the agreement.
    """
    batch = trainer_features(history, SPECIES)
    dsa = batch["days_since_active"]
    assert dsa.nunique() > 50, f"days_since_active is degenerate ({dsa.nunique()} values)"
    assert dsa.max() > 100, f"days_since_active never reaches an off-season gap ({dsa.max()})"


def test_lag_state_seeds_days_since_active_from_full_history(history: pd.DataFrame) -> None:
    """Seeding must look past the 56-window tail the other lags need.

    Off-season gaps run to hundreds of windows. Counting only within the tail
    would cap the feature at 56 at serving time while training saw the real
    distance.
    """
    origin = pd.Timestamp(history["date"].max())
    lag = LagState.from_history(history, SPECIES, origin)
    assert lag.days_since_active > 56


# --- Time-based lag alignment (E.1) ------------------------------------------


def _with_gap(history: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """The history with every row in [start, end) removed: a station outage."""
    dates = pd.to_datetime(history["date"])
    return history[(dates < pd.Timestamp(start)) | (dates >= pd.Timestamp(end))].reset_index(drop=True)


def test_lags_are_time_based_across_a_gap(history: pd.DataFrame) -> None:
    """"24 h ago" stays 24 h ago through an outage, on both paths.

    Row-based shifting made the first window after a three-day gap read its
    ``pollen_lag_8`` from four days earlier. On the grid the value is the one
    the station last reported before the gap, carried forward, and the
    ``days_since_active`` distance counts the gap's windows too.
    """
    gap_start, gap_end = "2021-04-20 00:00", "2021-04-23 00:00"
    gapped = _with_gap(history, gap_start, gap_end)
    batch = trainer_features(gapped, SPECIES, lead=1)
    first_after = pd.Timestamp(gap_end)
    last_before = pd.Timestamp(gap_start) - WINDOW

    sp = gapped[gapped["species"] == SPECIES].set_index("date")["value"]
    carried = float(np.log1p(sp.loc[last_before]))
    row = batch.loc[first_after]
    # Every lag inside the gap carries the last measurement across it...
    for col in ("pollen_lag_1", "pollen_lag_8", "pollen_lag_16"):
        assert row[col] == pytest.approx(carried), col
    # ...while a lag that reaches past the gap reads the real value there.
    assert row["pollen_lag_56"] == pytest.approx(float(np.log1p(sp.loc[first_after - 56 * WINDOW])))
    # The 24 h mean before the first window after the gap is the carried
    # value alone; row-based it would have been the mean of a whole day of
    # real windows from before the gap.
    assert row["pollen_rolling_8"] == pytest.approx(carried)

    # And the serving path agrees on the whole block.
    served = serving_features(gapped, SPECIES, pd.DatetimeIndex([first_after, first_after + 5 * WINDOW]), lead=1)
    for col in FEATURE_COLS:
        if col.startswith("upwind"):
            continue
        assert np.allclose(batch.loc[served.index, col].to_numpy(dtype=float), served[col].to_numpy(dtype=float), atol=1e-6), col


def test_days_since_active_counts_gap_windows(history: pd.DataFrame) -> None:
    """A quiet stretch plus an outage is measured in time on both paths."""
    # Birch is dormant in the fixture from day 141 on; cut a gap in June and
    # look just after it.
    gap_start, gap_end = "2021-06-10 00:00", "2021-06-14 00:00"
    gapped = _with_gap(history, gap_start, gap_end)
    origin = pd.Timestamp(gap_end)
    state = LagState.from_history(gapped, SPECIES, origin)
    sp = gapped[gapped["species"] == SPECIES]
    last_active = pd.to_datetime(sp.loc[sp["value"] > 0, "date"]).max()
    expected = float((origin - WINDOW - last_active) / WINDOW)
    assert state.days_since_active == expected
    batch = trainer_features(gapped, SPECIES, lead=1)
    assert batch.loc[origin, "days_since_active"] == expected
    # Row-based counting would have been shorter by the gap's 32 windows.
    assert expected > 32
