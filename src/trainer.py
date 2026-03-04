"""
Model trainer: trains one XGBoost regressor per pollen species.

Key design decisions:
  - Log-transform: trains on log1p(value) to handle extreme skew (Corylus 0→1700)
  - Season-active feature: binary flag so the model learns dormancy periods
  - Quantile regression: predicts the 75th percentile to avoid under-predicting peaks
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
    GDD_T_BASE,
    is_season_active,
)

MODELS_DIR = Path(__file__).parent.parent / "models"

# --- Log-transform helpers ---

def log_transform(values: pd.Series | np.ndarray) -> np.ndarray:
    """Apply log1p transform to pollen counts."""
    return np.log1p(np.asarray(values, dtype=float))


def inv_log_transform(values: np.ndarray) -> np.ndarray:
    """Inverse of log1p: expm1."""
    return np.expm1(values)


# --- Feature engineering ---

def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features for a single species' time series.
    Lag features are computed in log-space for consistency with the target.
    Expects df sorted by date with a 'value' column.
    """
    df = df.copy().sort_values("date")
    log_val = log_transform(df["value"])
    df["pollen_lag_1"] = pd.Series(log_val, index=df.index).shift(1)
    df["pollen_lag_2"] = pd.Series(log_val, index=df.index).shift(2)
    df["pollen_lag_3"] = pd.Series(log_val, index=df.index).shift(3)
    df["pollen_rolling_3"] = pd.Series(log_val, index=df.index).rolling(3, min_periods=1).mean().shift(1)
    df["pollen_rolling_7"] = pd.Series(log_val, index=df.index).rolling(7, min_periods=1).mean().shift(1)
    return df


def _add_season_feature(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Add a binary season_active feature based on the species' known pollen window."""
    df = df.copy()
    months = pd.to_datetime(df["date"]).dt.month
    df["season_active"] = months.apply(lambda m: 1.0 if is_season_active(species, m) else 0.0)
    return df


def _add_weather_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weather-derived features from the raw weather columns already in *df*.

    Features added:
    - GDD (Growing Degree Days) — cumulative thermal time from Jan 1
    - 3 / 7-day rolling means for temperature, sunshine, precipitation
    - Day-over-day and 3-day temperature deltas (warming trend)
    - Warm × sunny interaction (peak dispersal signal)
    - Dry + warm interaction
    """
    df = df.copy().sort_values("date")

    # De-duplicate so rolling computations are per-date
    temp_mean = df.groupby("date")["temperature_mean"].first()
    sunshine = df.groupby("date")["sunshine_duration"].first()
    precip = df.groupby("date")["precipitation_sum"].first()
    humidity = df.groupby("date")["humidity_mean"].first()

    # --- GDD (cumsum of max(0, T_mean - T_base), reset each Jan 1) ---
    daily_gdd = (temp_mean - GDD_T_BASE).clip(lower=0)
    years = pd.to_datetime(temp_mean.index).year
    gdd = daily_gdd.groupby(years).cumsum()

    # --- Rolling weather ---
    temp_r3 = temp_mean.rolling(3, min_periods=1).mean()
    temp_r7 = temp_mean.rolling(7, min_periods=1).mean()
    sun_r3 = sunshine.rolling(3, min_periods=1).mean()
    sun_r7 = sunshine.rolling(7, min_periods=1).mean()
    rain_r3 = precip.rolling(3, min_periods=1).sum()
    rain_r7 = precip.rolling(7, min_periods=1).sum()

    # --- Temperature deltas ---
    td1 = temp_mean.diff(1)
    td3 = temp_mean.diff(3)

    # --- Interactions ---
    temp_x_sun = temp_mean * sunshine / 3600.0  # normalize sunshine to hours
    dry_warm = temp_mean * (100.0 - humidity) / 100.0

    # Build a lookup dict indexed by date
    derived = pd.DataFrame({
        "gdd": gdd,
        "temp_rolling_3d": temp_r3,
        "temp_rolling_7d": temp_r7,
        "sunshine_rolling_3d": sun_r3,
        "sunshine_rolling_7d": sun_r7,
        "rain_rolling_3d": rain_r3,
        "rain_rolling_7d": rain_r7,
        "temp_delta_1d": td1,
        "temp_delta_3d": td3,
        "temp_x_sunshine": temp_x_sun,
        "dry_warm": dry_warm,
    })

    for col in derived.columns:
        mapping = derived[col].to_dict()
        df[col] = df["date"].map(mapping)

    df = df.fillna({col: 0.0 for col in derived.columns})
    return df


def prepare_training_data(
    history: pd.DataFrame, species: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare feature matrix X, target y (log-transformed), and raw_values
    (original scale) for a single species.
    Drops rows where lag features are NaN (first few days).
    """
    species_df = history[history["species"] == species].copy()
    if species_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    species_df = species_df.sort_values("date").reset_index(drop=True)
    species_df = _add_weather_derived_features(species_df)
    species_df = _add_lag_features(species_df)
    species_df = _add_season_feature(species_df, species)

    # Drop rows with missing lags (first 3 days)
    species_df = species_df.dropna(subset=LAG_FEATURES)

    if species_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    X = species_df[FEATURE_COLS].copy()
    raw_values = species_df["value"].reset_index(drop=True)
    y = pd.Series(log_transform(species_df["value"]), index=species_df.index)

    # Fill any remaining NaN in features with 0
    X = X.fillna(0)

    return X, y, raw_values


def train_species_model(X: pd.DataFrame, y: pd.Series, raw_values: pd.Series | None = None) -> XGBRegressor:
    """
    Train an XGBoost regressor for one species.

    Uses quantile regression (alpha=0.80) to bias toward higher predictions,
    reducing the systematic under-prediction of pollen peaks.

    When *raw_values* (original-scale pollen counts aligned with *y*) are
    provided, high-pollen samples receive higher sample weights so the model
    pays more attention to rare peak events.
    """
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="reg:quantileerror",
        quantile_alpha=0.80,
        random_state=42,
        verbosity=0,
    )

    # Sample weighting: upweight high-pollen days so the model learns peaks
    sample_weight = None
    if raw_values is not None:
        # weight = 1 + log1p(value)  →  zero-pollen=1, value=100→5.6, value=1700→8.4
        w = 1.0 + np.log1p(raw_values.values.astype(float))
        sample_weight = w

    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_all(history: pd.DataFrame) -> dict[str, XGBRegressor]:
    """
    Train one model per species. Returns a dict of species -> model.
    Only trains if there are enough data points (>= 14 days).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models: dict[str, XGBRegressor] = {}

    for species in ALL_SPECIES:
        X, y, raw_values = prepare_training_data(history, species)
        if len(X) < 14:
            print(f"  {species}: skipped (only {len(X)} training samples, need >= 14)")
            continue

        model = train_species_model(X, y, raw_values=raw_values)

        # Quick evaluation: RMSE on training data (in original scale)
        preds_log = model.predict(X)
        preds = inv_log_transform(preds_log)
        actuals = inv_log_transform(y.values)
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
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
