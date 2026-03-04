"""
Model trainer: trains one XGBoost regressor per pollen species.

Reads the accumulated history CSV, engineers lag/rolling features,
and trains models that predict daily pollen count from weather + calendar + lags.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .types import (
    ALL_SPECIES,
    FEATURE_COLS,
    LAG_FEATURES,
)

MODELS_DIR = Path(__file__).parent.parent / "models"


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features for a single species' time series.
    Expects df sorted by date with a 'value' column.
    """
    df = df.copy().sort_values("date")
    df["pollen_lag_1"] = df["value"].shift(1)
    df["pollen_lag_2"] = df["value"].shift(2)
    df["pollen_lag_3"] = df["value"].shift(3)
    df["pollen_rolling_3"] = df["value"].rolling(3, min_periods=1).mean().shift(1)
    df["pollen_rolling_7"] = df["value"].rolling(7, min_periods=1).mean().shift(1)
    return df


def prepare_training_data(history: pd.DataFrame, species: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target y for a single species.
    Drops rows where lag features are NaN (first few days).
    """
    species_df = history[history["species"] == species].copy()
    if species_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    species_df = species_df.sort_values("date").reset_index(drop=True)
    species_df = _add_lag_features(species_df)

    # Drop rows with missing lags (first 3 days)
    species_df = species_df.dropna(subset=LAG_FEATURES)

    if species_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    X = species_df[FEATURE_COLS].copy()
    y = species_df["value"].copy()

    # Fill any remaining NaN in features with 0
    X = X.fillna(0)

    return X, y


def train_species_model(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """Train an XGBoost regressor for one species."""
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def train_all(history: pd.DataFrame) -> dict[str, XGBRegressor]:
    """
    Train one model per species. Returns a dict of species -> model.
    Only trains if there are enough data points (>= 14 days).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models: dict[str, XGBRegressor] = {}

    for species in ALL_SPECIES:
        X, y = prepare_training_data(history, species)
        if len(X) < 14:
            print(f"  {species}: skipped (only {len(X)} training samples, need >= 14)")
            continue

        model = train_species_model(X, y)

        # Quick evaluation: RMSE on training data
        preds = model.predict(X)
        rmse = np.sqrt(np.mean((preds - y.values) ** 2))
        print(f"  {species}: trained on {len(X)} samples, train RMSE = {rmse:.1f}")

        # Save model
        model_path = MODELS_DIR / f"{species}.joblib"
        joblib.dump(model, model_path)
        models[species] = model

    print(f"\nTrained {len(models)} / {len(ALL_SPECIES)} species models")
    return models


def load_models() -> dict[str, XGBRegressor]:
    """Load all trained models from disk."""
    models: dict[str, XGBRegressor] = {}
    for species in ALL_SPECIES:
        model_path = MODELS_DIR / f"{species}.joblib"
        if model_path.exists():
            models[species] = joblib.load(model_path)
    return models
