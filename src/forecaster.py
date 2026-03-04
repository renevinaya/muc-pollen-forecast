"""
Forecaster: generates a multi-day pollen forecast using trained models.

Loads trained XGBoost models, fetches weather forecast from Open-Meteo,
and predicts pollen counts per species per day.  Predictions are made in
log-space and converted back.  For lag features it uses the most recent
historical data and then autoregressively feeds (log-space) predictions
into subsequent days.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .types import (
    ALL_SPECIES,
    FORECAST_DAYS,
    LOCATION,
    WEATHER_FEATURES,
    FEATURE_COLS,
    is_season_active,
    value_to_level,
    SpeciesForecast,
    DayForecast,
    ForecastOutput,
)
from .weather import fetch_weather_forecast
from .trainer import load_models, log_transform, inv_log_transform, _add_weather_derived_features


def _calendar_features_for_date(dt: pd.Timestamp) -> dict[str, float]:
    """Compute calendar features for a single date."""
    doy = dt.day_of_year
    return {
        "day_of_year": float(doy),
        "day_of_year_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "day_of_year_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "month": float(dt.month),
    }


def _get_recent_pollen_log(history: pd.DataFrame, species: str, n: int = 7) -> list[float]:
    """
    Get the last n pollen values for a species from history, in log-space.
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
    models: dict[str, XGBRegressor] | None = None,
) -> ForecastOutput:
    """
    Generate a multi-day pollen forecast.

    All internal lag/prediction values are in log-space (log1p).
    Final output values are converted back to original pollen-count scale.

    Args:
        history: Full historical data (date, species, value, weather features...).
        models: Pre-loaded models. If None, loads from disk.
    """
    if models is None:
        models = load_models()
        print(f"Loaded {len(models)} species models")

    # Fetch weather forecast
    weather = fetch_weather_forecast(FORECAST_DAYS)
    print(f"Weather forecast: {len(weather)} days")

    # --- Pre-compute weather-derived features (GDD, rolling, etc.) ---
    # We need recent historical weather to compute rolling features correctly.
    # Extract unique-date weather rows from history, append forecast weather,
    # then compute derived features on the combined series.
    hist_weather = (
        history.groupby("date")[WEATHER_FEATURES].first()
        if not history.empty
        else pd.DataFrame(columns=WEATHER_FEATURES)
    )
    combined_weather = pd.concat([hist_weather, weather])
    combined_weather = combined_weather[~combined_weather.index.duplicated(keep="last")]
    combined_weather = combined_weather.sort_index()
    # Add a dummy 'species' and 'value' so _add_weather_derived_features works
    combined_weather = combined_weather.reset_index().rename(columns={"index": "date"})
    combined_weather["species"] = "__dummy__"
    combined_weather["value"] = 0.0
    combined_weather = _add_weather_derived_features(combined_weather)
    # Index by date for quick lookup
    weather_derived = combined_weather.set_index("date")

    # Build recent pollen history per species in LOG-SPACE (for lag features)
    recent_log: dict[str, list[float]] = {}
    for species in ALL_SPECIES:
        recent_log[species] = _get_recent_pollen_log(history, species, n=7)

    forecast_days: list[DayForecast] = []

    for day_idx, (dt, weather_row) in enumerate(weather.iterrows()):
        dt = pd.Timestamp(dt)
        day_species: list[SpeciesForecast] = []

        cal = _calendar_features_for_date(dt)
        month = dt.month

        for species in ALL_SPECIES:
            log_vals = recent_log[species]
            has_model = species in models
            active = is_season_active(species, month)

            if has_model:
                # Build feature vector
                features: dict[str, float] = {}

                # Weather features
                for f in WEATHER_FEATURES:
                    features[f] = float(weather_row.get(f, 0) or 0)

                # Calendar features
                features.update(cal)

                # Season-active feature
                features["season_active"] = 1.0 if active else 0.0

                # Weather-derived features (GDD, rolling, interaction)
                from .types import WEATHER_DERIVED_FEATURES
                if dt in weather_derived.index:
                    wd_row = weather_derived.loc[dt]
                    for f in WEATHER_DERIVED_FEATURES:
                        features[f] = float(wd_row.get(f, 0) or 0)
                else:
                    for f in WEATHER_DERIVED_FEATURES:
                        features[f] = 0.0

                # Lag features in log-space (autoregressive)
                features["pollen_lag_1"] = log_vals[-1]
                features["pollen_lag_2"] = log_vals[-2]
                features["pollen_lag_3"] = log_vals[-3]
                features["pollen_rolling_3"] = float(np.mean(log_vals[-3:]))
                features["pollen_rolling_7"] = float(np.mean(log_vals[-7:]))

                X = pd.DataFrame([features])[FEATURE_COLS]
                pred_log = float(models[species].predict(X)[0])
                pred_log = max(0.0, pred_log)  # log-space, 0 = pollen count of 0
                prediction = float(inv_log_transform(np.array([pred_log]))[0])
            else:
                # Fallback: use last known value with seasonal decay
                prediction = float(inv_log_transform(np.array([log_vals[-1]]))[0]) * 0.8
                pred_log = float(np.log1p(prediction))

            # Force to zero if out of season
            if not active:
                prediction = 0.0
                pred_log = 0.0

            day_species.append(
                SpeciesForecast(
                    name=species,
                    level=value_to_level(prediction, species).value,
                    value=prediction,
                    confidence=_confidence_for_day(day_idx, has_model),
                )
            )

            # Autoregressive: feed log-space prediction back for next day
            recent_log[species].append(pred_log)

        # Sort by value descending, filter out zeros
        day_species.sort(key=lambda s: s.value, reverse=True)
        day_species = [s for s in day_species if s.value > 0.5]

        forecast_days.append(
            DayForecast(date=dt.strftime("%Y-%m-%d"), species=day_species)
        )

    return ForecastOutput(
        generated=datetime.utcnow().isoformat() + "Z",
        location=LOCATION,
        forecast=forecast_days,
    )
