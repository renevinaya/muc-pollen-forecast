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
    _add_ndvi_features,
    _add_phenology_features,
    _add_season_feature,
    _add_weather_derived_features,
)
from src.types import ALL_SPECIES, FEATURE_COLS, WEATHER_FEATURES

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


def trainer_features(history: pd.DataFrame, species: str) -> pd.DataFrame:
    """The trainer's batch path, in the order prepare_training_data applies it."""
    df = history[history["species"] == species].sort_values("date").reset_index(drop=True)
    df = _add_weather_derived_features(df, species)
    df = _add_ndvi_features(df)
    df = _add_intraday_features(df)
    df = _add_lag_features(df)
    df = _add_season_feature(df, species)
    df = _add_phenology_features(df, species)
    return df.set_index("date")


def serving_features(
    history: pd.DataFrame, species: str, windows: pd.DatetimeIndex
) -> pd.DataFrame:
    """The forecaster's row path, walking *windows* with observations fed back.

    Feeding the measured value back after each window is what the forecaster
    does for assimilated windows, and it is what makes the comparison fair:
    both paths then see the same lag inputs, so any difference is a real
    definition mismatch rather than accumulated prediction error.
    """
    origin = windows[0]
    past = history[history["date"] < origin]
    future = history[history["date"] >= origin]

    weather_cols = [c for c in WEATHER_FEATURES if c in history.columns]
    weather = future.groupby("date")[weather_cols].first().sort_index()

    days = pd.DatetimeIndex(weather.index).normalize().unique()
    ctx = build_context(
        past, weather, ndvi=ndvi_from_history(history, days), species=[species],
        parallel=False,
    )

    measured = (
        future[future["species"] == species].set_index("date")["value"].to_dict()
    )
    lag = LagState.from_history(history, species, origin)

    rows = {}
    for dt in windows:
        lag.begin_window(dt)
        rows[dt] = build_feature_row(ctx, species, dt, lag)
        lag.record(float(np.log1p(measured.get(dt, 0.0))))
    return pd.DataFrame.from_dict(rows, orient="index")


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return build_history()


def test_serving_matches_training_features(history: pd.DataFrame) -> None:
    """Every feature agrees between the batch and row paths, on every window."""
    batch = trainer_features(history, SPECIES)

    # Start well inside the frame so both paths have their lags warmed up, and
    # cover the season so the phenology and burst features are exercised.
    all_windows = pd.DatetimeIndex(batch.index)
    windows = all_windows[(all_windows >= all_windows[600]) & (all_windows <= all_windows[1400])]
    served = serving_features(history, SPECIES, windows)

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
