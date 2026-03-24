"""
Forecaster: generates a multi-day pollen forecast at 3-hour resolution
using trained two-stage models.

Loads trained XGBoost models (classifier + regressor), fetches hourly weather
forecast from Open-Meteo (aggregated to 3h windows), and predicts pollen counts
per species per 3h window.  Predictions are made in log-space and converted back.
For lag features it uses the most recent historical data and then autoregressively
feeds (log-space) predictions into subsequent windows.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd

from .types import (
    ALL_SPECIES,
    FORECAST_DAYS,
    LOCATION,
    WEATHER_FEATURES,
    WEATHER_DERIVED_FEATURES,
    NDVI_FEATURES,
    FEATURE_COLS,
    SPECIES_SEASON,
    SPECIES_GDD_THRESHOLD,
    SPECIES_ACTIVATION_TEMP,
    _DEFAULT_GDD_THRESHOLD,
    _DEFAULT_ACTIVATION_TEMP,
    is_season_active,
    value_to_level,
    SpeciesForecast,
    WindowForecast,
    DayForecast,
    ForecastOutput,
)
from .weather import fetch_weather_forecast
from .trainer import (
    TwoStageModel,
    load_models,
    inv_log_transform,
    _add_weather_derived_features,
)


def _calendar_features_for_datetime(dt: pd.Timestamp) -> dict[str, float]:
    """Compute calendar and time-of-day features for a single datetime."""
    doy = dt.day_of_year
    hour = dt.hour
    return {
        "day_of_year": float(doy),
        "day_of_year_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "day_of_year_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "month": float(dt.month),
        "hour_of_day": float(hour),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
    }


def _get_recent_pollen_log(history: pd.DataFrame, species: str, n: int = 56) -> list[float]:
    """
    Get the last n pollen values for a species from history, in log-space.
    Each value is a 3h window measurement.
    These will be used directly as lag features (model trains on log1p values).
    """
    sp = history[history["species"] == species].sort_values("date")
    raw_values = sp["value"].tail(n).tolist()
    # Pad with zeros if not enough history
    while len(raw_values) < n:
        raw_values.insert(0, 0.0)
    return [float(np.log1p(v)) for v in raw_values]


def _confidence_for_day(day_index: int, has_model: bool) -> float:
    """Confidence decreases with forecast distance; lower if no model."""
    base = 0.90 - day_index * 0.08
    if not has_model:
        base *= 0.5
    return max(0.2, min(0.95, base))


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

    # Fetch weather forecast (3h resolution)
    weather = fetch_weather_forecast(FORECAST_DAYS)
    print(f"Weather forecast: {len(weather)} windows ({FORECAST_DAYS} days)")

    # --- Pre-compute weather-derived features per species ---
    # The burst potential features (#2) depend on species-specific thresholds,
    # so we pre-compute a table for each species.
    hist_weather = (
        history.groupby("date")[WEATHER_FEATURES].first()
        if not history.empty
        else pd.DataFrame(columns=WEATHER_FEATURES)
    )
    combined_weather = pd.concat([hist_weather, weather])
    combined_weather = combined_weather[~combined_weather.index.duplicated(keep="last")]
    combined_weather = combined_weather.sort_index()
    # Add a dummy 'species' and 'value' so _add_weather_derived_features works
    base_weather = combined_weather.reset_index().rename(columns={"index": "date"})
    base_weather["species"] = "__dummy__"
    base_weather["value"] = 0.0

    def _derive_for_species(sp: str) -> tuple[str, pd.DataFrame]:
        return sp, _add_weather_derived_features(base_weather.copy(), sp).set_index("date")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(ALL_SPECIES)) as pool:
        futures = {pool.submit(_derive_for_species, sp): sp for sp in ALL_SPECIES}
        weather_derived_by_species = dict(f.result() for f in futures)

    # --- Pre-compute NDVI features for forecast dates (daily resolution) ---
    try:
        from .ndvi import ndvi_features
        forecast_dates = pd.DatetimeIndex(weather.index).normalize().unique()
        ndvi_df = ndvi_features(forecast_dates)
    except Exception as exc:
        print(f"  NDVI fetch failed ({exc}), using defaults")
        ndvi_df = pd.DataFrame(
            {"ndvi": 0.0, "evi": 0.0, "ndvi_delta": 0.0},
            index=pd.DatetimeIndex(weather.index).normalize().unique(),
        )

    # Build recent pollen history per species in LOG-SPACE (for lag features)
    # Keep 56 windows (7 days) for rolling_56 computation
    recent_log: dict[str, list[float]] = {}
    for species in ALL_SPECIES:
        recent_log[species] = _get_recent_pollen_log(history, species, n=56)

    # --- Real-time observation assimilation ---
    # Build a lookup of observed pollen values from history so that forecast
    # windows that already have real measurements use those instead of model
    # predictions.  This breaks the autoregressive error cascade for same-day
    # windows where data has already been collected.
    observed: dict[tuple[pd.Timestamp, str], float] = {}
    if not history.empty:
        recent_cutoff = weather.index.min()
        recent_hist = history[history["date"] >= recent_cutoff]
        for _, row in recent_hist.iterrows():
            key = (pd.Timestamp(row["date"]), str(row["species"]))
            observed[key] = float(row["value"])

    n_obs_windows = len(set(dt for dt, _ in observed))
    if n_obs_windows > 0:
        print(f"Real-time assimilation: {n_obs_windows} observed windows will use actual data")

    # Iterate over all 3h weather windows
    window_results: list[tuple[str, WindowForecast]] = []
    prev_date_str: str | None = None
    day_idx = -1

    for dt_key, weather_row in weather.iterrows():
        dt = pd.Timestamp(str(dt_key))
        date_str = dt.strftime("%Y-%m-%d")

        # Track day index for confidence scoring
        if date_str != prev_date_str:
            day_idx += 1
            prev_date_str = date_str

        cal = _calendar_features_for_datetime(dt)
        month = dt.month
        window_species: list[SpeciesForecast] = []

        for species in ALL_SPECIES:
            log_vals = recent_log[species]
            has_model = species in models
            active = is_season_active(species, month)

            # Check if we have a real observation for this window
            obs_key = (dt, species)
            has_observation = obs_key in observed

            if has_observation:
                # Use real observation — breaks autoregressive error cascade
                prediction = observed[obs_key]
                pred_log = float(np.log1p(prediction))
                # Higher confidence for observed windows
                confidence = min(0.95, _confidence_for_day(day_idx, has_model) + 0.05)
            elif has_model:
                # Build feature vector for model prediction
                features: dict[str, float] = {}

                # Weather features
                for f in WEATHER_FEATURES:
                    features[f] = float(weather_row.get(f, 0) or 0)

                # Calendar + time-of-day features
                features.update(cal)

                # Season-active feature
                features["season_active"] = 1.0 if active else 0.0

                # Weather-derived features (GDD, rolling, interaction, burst, explosion)
                weather_derived = weather_derived_by_species[species]
                if dt in weather_derived.index:
                    wd_row = weather_derived.loc[dt]
                    if isinstance(wd_row, pd.DataFrame):
                        wd_row = wd_row.iloc[0]
                    for f in WEATHER_DERIVED_FEATURES:
                        features[f] = float(wd_row.get(f, 0) or 0)
                else:
                    for f in WEATHER_DERIVED_FEATURES:
                        features[f] = 0.0

                # NDVI features (daily resolution, lookup by date)
                day = dt.normalize()
                if day in ndvi_df.index:
                    ndvi_row = ndvi_df.loc[day]
                    if isinstance(ndvi_row, pd.DataFrame):
                        ndvi_row = ndvi_row.iloc[0]
                    for f in NDVI_FEATURES:
                        features[f] = float(ndvi_row.get(f, 0) or 0)
                else:
                    for f in NDVI_FEATURES:
                        features[f] = 0.0

                # Phenology features
                import calendar as _cal
                window = SPECIES_SEASON.get(species)
                if window:
                    mean_onset_doy = sum(
                        _cal.monthrange(2025, m)[1] for m in range(1, window[0])
                    ) + 15
                else:
                    mean_onset_doy = 1
                features["days_since_typical_onset"] = float(
                    max(-60, dt.day_of_year - mean_onset_doy)
                )
                features["onset_anomaly"] = 0.0

                # Lag features in log-space (autoregressive, 3h windows)
                features["pollen_lag_1"] = log_vals[-1]
                features["pollen_lag_2"] = log_vals[-2]
                features["pollen_lag_3"] = log_vals[-3]
                features["pollen_lag_8"] = log_vals[-8] if len(log_vals) >= 8 else log_vals[0]
                features["pollen_lag_16"] = log_vals[-16] if len(log_vals) >= 16 else log_vals[0]
                features["pollen_lag_56"] = log_vals[-56] if len(log_vals) >= 56 else log_vals[0]
                features["pollen_rolling_8"] = float(np.mean(log_vals[-8:]))
                features["pollen_rolling_56"] = float(np.mean(log_vals[-56:]))
                features["pollen_max_8"] = float(np.max(log_vals[-8:]))
                features["pollen_max_56"] = float(np.max(log_vals[-56:]))
                # days_since_active: count back from end of log_vals to last > 0
                dsa = 0
                for v in reversed(log_vals):
                    if v > 0:
                        break
                    dsa += 1
                features["days_since_active"] = float(dsa)

                x_features = pd.DataFrame([features])[FEATURE_COLS]
                pred_log = float(models[species].predict(x_features)[0])
                pred_log = max(0.0, pred_log)  # log-space, 0 = pollen count of 0
                prediction = float(inv_log_transform(np.array([pred_log]))[0])
                confidence = _confidence_for_day(day_idx, has_model)
            else:
                # Fallback: use last known value with seasonal decay
                prediction = float(inv_log_transform(np.array([log_vals[-1]]))[0]) * 0.8
                pred_log = float(np.log1p(prediction))
                confidence = _confidence_for_day(day_idx, has_model)

            # Force to zero if out of season
            if not active:
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

            # Autoregressive: feed log-space value back for next window
            # (real observation or prediction — real data grounds future predictions)
            recent_log[species].append(pred_log)

        # Sort by value descending, filter out zeros
        window_species.sort(key=lambda s: s.value, reverse=True)
        window_species = [s for s in window_species if s.value > 0.5]

        from_time = dt.strftime("%H:%M")
        to_time = (dt + pd.Timedelta(hours=3)).strftime("%H:%M")
        window_results.append((date_str, WindowForecast(
            from_time=from_time,
            to_time=to_time,
            species=window_species,
        )))

    # Group windows into days
    days_dict: OrderedDict[str, list[WindowForecast]] = OrderedDict()
    for date_str, wf in window_results:
        days_dict.setdefault(date_str, []).append(wf)

    forecast_days = [
        DayForecast(date=date_str, windows=windows)
        for date_str, windows in days_dict.items()
    ]

    return ForecastOutput(
        generated=datetime.utcnow().isoformat() + "Z",
        location=LOCATION,
        forecast=forecast_days,
    )
