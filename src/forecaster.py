"""
Forecaster: generates a multi-day pollen forecast at 3-hour resolution
using trained two-stage models.

Loads trained XGBoost models (classifier + regressor), fetches the hourly
weather forecast from Open-Meteo (aggregated to 3h windows), and predicts
pollen counts per species per 3h window. Predictions are made in log-space and
converted back. Lag features start from the most recent measurements and are
then fed autoregressively from the predictions themselves.

The feature vectors come from :mod:`src.features`, which the rollout benchmark
also uses — a backtest that built its own features would stop measuring the
thing being shipped the moment the two drifted.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .types import (
    ALL_SPECIES,
    FORECAST_DAYS,
    LOCATION,
    FEATURE_COLS,
    SPECIES_THRESHOLDS,
    _DEFAULT_THRESHOLDS,
    season_gate_active,
    value_to_level,
    SpeciesForecast,
    WindowForecast,
    DayForecast,
    ForecastOutput,
)
from .weather import fetch_weather_forecast
from .trainer import TwoStageModel, load_models, inv_log_transform
from .features import FeatureContext, LagState, build_context, build_feature_row
from .cams import fetch_cams_forecast

# Our categorical level → DWD-style numeric scale (0–3).
_LEVEL_TO_NUM = {"none": 0, "low": 1, "moderate": 2, "high": 3, "very_high": 3}


def _level_band_midpoint(species: str, level_num: int) -> float:
    """Representative pollen value for a 0–3 level band, per species thresholds."""
    low_max, mod_max, high_max = SPECIES_THRESHOLDS.get(species, _DEFAULT_THRESHOLDS)
    if level_num <= 0:
        return 0.0
    if level_num == 1:
        return low_max / 2.0
    if level_num == 2:
        return (low_max + mod_max) / 2.0
    return (mod_max + high_max) / 2.0


def _blend_with_dwd(value: float, species: str, dwd_num: float) -> float:
    """Nudge a model value one level toward the DWD categorical forecast.

    Conservative: moves at most one level step (never fabricates pollen far
    outside the model's range) and blends 50% (never fully overrides the model,
    so a nonzero prediction is never zeroed). Used only for DWD-covered dates.
    """
    our_num = _LEVEL_TO_NUM[value_to_level(value, species).value]
    target_round = int(round(dwd_num))
    if target_round == our_num:
        return value
    step = 1 if target_round > our_num else -1
    target_num = max(0, min(3, our_num + step))
    target_value = _level_band_midpoint(species, target_num)
    return max(0.0, value * 0.5 + target_value * 0.5)


def _confidence_for_day(day_index: int, has_model: bool) -> float:
    """Confidence decreases with forecast distance; lower if no model."""
    base = 0.90 - day_index * 0.08
    if not has_model:
        base *= 0.5
    return max(0.2, min(0.95, base))


def predict_window(
    model: TwoStageModel, ctx: FeatureContext, species: str, dt: pd.Timestamp, lag: LagState
) -> float:
    """Model prediction for one (window, species), in log space."""
    features = build_feature_row(ctx, species, dt, lag)
    x_features = pd.DataFrame([features])[FEATURE_COLS]
    return max(0.0, float(model.predict(x_features)[0]))


def _fetch_ndvi(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily NDVI for the forecast dates; zeros when the fetch fails."""
    try:
        from .ndvi import ndvi_features

        return ndvi_features(days)
    except Exception as exc:
        print(f"  NDVI fetch failed ({exc}), using defaults")
        return pd.DataFrame(
            {"ndvi": 0.0, "evi": 0.0, "ndvi_delta": 0.0}, index=days
        )


def _fetch_dwd_levels() -> dict[tuple[object, str], float]:
    """DWD categorical levels keyed by (date, species); empty when unavailable.

    DWD only covers today/tomorrow/day-after, so most forecast windows are
    untouched. Fail-open: an empty lookup leaves predictions unchanged.
    """
    levels: dict[tuple[object, str], float] = {}
    try:
        from .dwd import fetch_dwd_forecast

        for _, row in fetch_dwd_forecast().iterrows():
            levels[(pd.Timestamp(row["date"]).date(), str(row["species"]))] = float(
                row["dwd_level"]
            )
        if levels:
            print(f"DWD blend: {len(levels)} (date, species) levels available")
    except Exception as exc:
        print(f"  DWD forecast unavailable ({exc}); skipping DWD blend")
    return levels


def generate_forecast(
    history: pd.DataFrame,
    models: dict[str, TwoStageModel] | None = None,
) -> ForecastOutput:
    """
    Generate a multi-day pollen forecast at 3-hour window resolution.

    All internal lag/prediction values are in log-space (log1p).
    Final output values are converted back to original pollen-count scale.

    Args:
        history: Full historical data (date, species, value, weather features...).
        models: Pre-loaded models. If None, loads from disk.
    """
    if models is None:
        models = load_models()
        print(f"Loaded {len(models)} species models")

    weather = fetch_weather_forecast(FORECAST_DAYS)
    print(f"Weather forecast: {len(weather)} windows ({FORECAST_DAYS} days)")

    forecast_days_index = pd.DatetimeIndex(weather.index).normalize().unique()
    ctx = build_context(
        history,
        weather,
        ndvi=_fetch_ndvi(forecast_days_index),
        cams=fetch_cams_forecast(FORECAST_DAYS),
    )
    dwd_levels = _fetch_dwd_levels()

    origin = pd.Timestamp(weather.index.min())
    lags = {sp: LagState.from_history(history, sp, origin) for sp in ALL_SPECIES}

    # --- Real-time observation assimilation ---
    # Forecast windows that already have a measurement use it instead of a
    # prediction, which breaks the autoregressive error cascade for same-day
    # windows and grounds every later window's lags in real data.
    observed: dict[tuple[pd.Timestamp, str], float] = {}
    if not history.empty:
        recent = history[history["date"] >= origin]
        for _, row in recent.iterrows():
            observed[(pd.Timestamp(row["date"]), str(row["species"]))] = float(row["value"])

    n_obs_windows = len({dt for dt, _ in observed})
    if n_obs_windows > 0:
        print(f"Real-time assimilation: {n_obs_windows} observed windows will use actual data")

    window_results: list[tuple[str, WindowForecast]] = []
    prev_date_str: str | None = None
    day_idx = -1

    for dt_key in weather.index:
        dt = pd.Timestamp(str(dt_key))
        date_str = dt.strftime("%Y-%m-%d")

        if date_str != prev_date_str:
            day_idx += 1
            prev_date_str = date_str

        window_species: list[SpeciesForecast] = []

        for species in ALL_SPECIES:
            lag = lags[species]
            lag.begin_window(dt)
            has_model = species in models
            has_observation = (dt, species) in observed

            if has_observation:
                prediction = observed[(dt, species)]
                pred_log = float(np.log1p(prediction))
                confidence = min(0.95, _confidence_for_day(day_idx, has_model) + 0.05)
            elif has_model:
                pred_log = predict_window(models[species], ctx, species, dt, lag)
                prediction = float(inv_log_transform(np.array([pred_log]))[0])
                confidence = _confidence_for_day(day_idx, has_model)
            else:
                # Fallback: last known value with seasonal decay.
                prediction = float(np.expm1(lag.lag_features()["pollen_lag_1"])) * 0.8
                pred_log = float(np.log1p(prediction))
                confidence = _confidence_for_day(day_idx, has_model)

            # DWD inference-time blend, for the dates DWD covers. Skips real
            # observations (keep actual data) and runs before the season gate,
            # so the gate has the final say.
            if not has_observation:
                dwd_num = dwd_levels.get((dt.date(), species))
                if dwd_num is not None:
                    prediction = _blend_with_dwd(prediction, species, dwd_num)

            # Force to zero only outside the *widened* season window (core ±
            # shoulder), so early-onset events are no longer structurally zeroed.
            if not season_gate_active(species, dt.month):
                prediction = 0.0
                pred_log = 0.0

            window_species.append(
                SpeciesForecast(
                    name=species,
                    level=value_to_level(prediction, species).value,
                    value=prediction,
                    confidence=confidence,
                )
            )
            lag.record(pred_log)

        window_species.sort(key=lambda s: s.value, reverse=True)
        window_species = [s for s in window_species if s.value > 0.5]

        window_results.append((date_str, WindowForecast(
            from_time=dt.strftime("%H:%M"),
            to_time=(dt + pd.Timedelta(hours=3)).strftime("%H:%M"),
            species=window_species,
        )))

    days_dict: OrderedDict[str, list[WindowForecast]] = OrderedDict()
    for date_str, wf in window_results:
        days_dict.setdefault(date_str, []).append(wf)

    return ForecastOutput(
        generated=datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        location=LOCATION,
        forecast=[
            DayForecast(date=date_str, windows=windows)
            for date_str, windows in days_dict.items()
        ],
    )
