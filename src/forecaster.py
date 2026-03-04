"""
Forecaster: generates a multi-day pollen forecast using trained models.

Loads trained XGBoost models, fetches weather forecast from Open-Meteo,
and predicts pollen counts per species per day. For lag features, it uses
the most recent historical data and then autoregressively feeds predictions
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
    value_to_level,
    SpeciesForecast,
    DayForecast,
    ForecastOutput,
)
from .weather import fetch_weather_forecast
from .trainer import load_models


def _calendar_features_for_date(dt: pd.Timestamp) -> dict[str, float]:
    """Compute calendar features for a single date."""
    doy = dt.day_of_year
    return {
        "day_of_year": float(doy),
        "day_of_year_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "day_of_year_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "month": float(dt.month),
    }


def _get_recent_pollen(history: pd.DataFrame, species: str, n: int = 7) -> list[float]:
    """Get the last n pollen values for a species from history."""
    sp = history[history["species"] == species].sort_values("date")
    values = sp["value"].tail(n).tolist()
    # Pad with zeros if not enough history
    while len(values) < n:
        values.insert(0, 0.0)
    return values


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

    # Build recent pollen history per species (for lag features)
    recent: dict[str, list[float]] = {}
    for species in ALL_SPECIES:
        recent[species] = _get_recent_pollen(history, species, n=7)

    forecast_days: list[DayForecast] = []

    for day_idx, (dt, weather_row) in enumerate(weather.iterrows()):
        dt = pd.Timestamp(dt)
        day_species: list[SpeciesForecast] = []

        cal = _calendar_features_for_date(dt)

        for species in ALL_SPECIES:
            vals = recent[species]
            has_model = species in models

            if has_model:
                # Build feature vector
                features: dict[str, float] = {}

                # Weather features
                for f in WEATHER_FEATURES:
                    features[f] = float(weather_row.get(f, 0) or 0)

                # Calendar features
                features.update(cal)

                # Lag features (from recent history / autoregressive)
                features["pollen_lag_1"] = vals[-1]
                features["pollen_lag_2"] = vals[-2]
                features["pollen_lag_3"] = vals[-3]
                features["pollen_rolling_3"] = float(np.mean(vals[-3:]))
                features["pollen_rolling_7"] = float(np.mean(vals[-7:]))

                X = pd.DataFrame([features])[FEATURE_COLS]
                prediction = float(models[species].predict(X)[0])
                prediction = max(0.0, prediction)  # pollen can't be negative
            else:
                # Fallback: use last known value with seasonal decay
                prediction = vals[-1] * 0.8

            day_species.append(
                SpeciesForecast(
                    name=species,
                    level=value_to_level(prediction).value,
                    value=prediction,
                    confidence=_confidence_for_day(day_idx, has_model),
                )
            )

            # Autoregressive: feed prediction back for next day's lag
            recent[species].append(prediction)

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
