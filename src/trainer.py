"""
Model trainer: two-stage XGBoost pipeline per pollen species.

Stage 1 – **Classifier**: predicts P(pollen > 0) for a given day.
Stage 2 – **Regressor**: predicts log1p(pollen count) for active days.

The combined prediction is:
    if P(active) < 0.5 and not in-season  →  0
    else  →  expm1(stage2_prediction)

Key design decisions:
  - Log-transform on the target to handle extreme skew
  - Season-active + NDVI + phenology features capture biological timing
  - Sample weighting up-weights rare peak events
  - Quantile regression (α = 0.80) biases toward higher predictions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from .types import (
    ALL_SPECIES,
    FEATURE_COLS,
    LAG_FEATURES,
    GDD_T_BASE,
    is_season_active,
    SPECIES_SEASON,
    SPECIES_GDD_THRESHOLD,
    SPECIES_ACTIVATION_TEMP,
    _DEFAULT_GDD_THRESHOLD,
    _DEFAULT_ACTIVATION_TEMP,
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
    Lag features are computed in log-space. Each row is a 3h window.
    Expects df sorted by date with a 'value' column.
    """
    df = df.copy().sort_values("date")
    log_val = log_transform(df["value"])
    s = pd.Series(log_val, index=df.index)
    df["pollen_lag_1"] = s.shift(1)           # previous 3h window
    df["pollen_lag_2"] = s.shift(2)           # 6h ago
    df["pollen_lag_3"] = s.shift(3)           # 9h ago
    df["pollen_lag_8"] = s.shift(8)           # same time yesterday (24h)
    df["pollen_lag_16"] = s.shift(16)         # 48h ago (#3)
    df["pollen_lag_56"] = s.shift(56)         # 7 days ago (#3)
    df["pollen_rolling_8"] = s.rolling(8, min_periods=1).mean().shift(1)    # 24h mean
    df["pollen_rolling_56"] = s.rolling(56, min_periods=1).mean().shift(1)  # 7-day mean
    df["pollen_max_8"] = s.rolling(8, min_periods=1).max().shift(1)         # 24h max (#3)
    df["pollen_max_56"] = s.rolling(56, min_periods=1).max().shift(1)       # 7-day max (#3)
    # Days (windows) since pollen was last > 0 (#3)
    active_mask = (s > 0).astype(int)
    cumactive = active_mask.cumsum()
    last_active = cumactive.where(active_mask == 1).ffill().fillna(0)
    df["days_since_active"] = (cumactive - last_active).shift(1).fillna(999).astype(float)
    return df


def _add_season_feature(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Add a binary season_active feature based on the species' known pollen window."""
    df = df.copy()
    months = pd.to_datetime(df["date"]).dt.month
    df["season_active"] = months.apply(lambda m: 1.0 if is_season_active(species, m) else 0.0)
    return df


def _add_phenology_features(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Add phenology-derived features: days since typical flowering onset, onset anomaly."""
    df = df.copy()
    window = SPECIES_SEASON.get(species)
    if window is None:
        df["days_since_typical_onset"] = 0.0
        df["onset_anomaly"] = 0.0
        return df

    # Approximate mean onset as day 15 of start_month
    start_month = window[0]
    # Convert to approximate day-of-year
    import calendar
    mean_onset_doy = sum(calendar.monthrange(2025, m)[1] for m in range(1, start_month)) + 15

    doys = pd.to_datetime(df["date"]).dt.dayofyear
    df["days_since_typical_onset"] = (doys - mean_onset_doy).clip(lower=-60).astype(float)
    # onset_anomaly: 0 by default; will be overwritten when real phenology data is loaded
    df["onset_anomaly"] = 0.0
    return df


def _add_ndvi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add NDVI features from cached satellite data."""
    df = df.copy()
    # Default to 0 — will be populated by collector; if columns already present, skip
    for col in ("ndvi", "evi", "ndvi_delta"):
        if col not in df.columns:
            df[col] = 0.0
    return df


def _add_weather_derived_features(
    df: pd.DataFrame, species: str = "",
) -> pd.DataFrame:
    """
    Compute weather-derived features from the raw weather columns already in *df*.

    Operates on 3-hour window data. Rolling windows are sized in number of
    3h windows: 24 windows = 3 days, 56 windows = 7 days.

    Features added:
    - GDD (Growing Degree Days) — cumulative thermal time from Jan 1 (daily)
    - 3 / 7-day rolling means for temperature, sunshine, precipitation
    - Day-over-day and 3-day temperature deltas (warming trend)
    - Warm × sunny interaction (peak dispersal signal)
    - Dry + warm interaction
    - Burst potential: GDD above threshold, cold→warm flip, consecutive warm (#2)
    - Explosion likelihood: dry streak, warm-after-cold, wind×dry_warm (#6)
    """
    df = df.copy().sort_values("date")

    # De-duplicate so rolling computations are per-window
    temp_mean = df.groupby("date")["temperature_mean"].first()
    sunshine = df.groupby("date")["sunshine_duration"].first()
    precip = df.groupby("date")["precipitation_sum"].first()
    humidity = df.groupby("date")["humidity_mean"].first()
    wind = df.groupby("date")["wind_speed_max"].first()

    # --- GDD (cumsum of daily max(0, T_mean - T_base), reset each Jan 1) ---
    window_dates = pd.to_datetime(temp_mean.index).normalize()
    daily_temp = pd.Series(temp_mean.values, index=window_dates).groupby(level=0).mean()
    daily_gdd_contrib = (daily_temp - GDD_T_BASE).clip(lower=0)
    gdd_daily = daily_gdd_contrib.groupby(daily_temp.index.year).cumsum()
    date_to_gdd = gdd_daily.to_dict()
    gdd = pd.Series(
        [date_to_gdd.get(d, 0.0) for d in window_dates],
        index=temp_mean.index,
    )

    # --- Rolling weather (3 days = 24 windows, 7 days = 56 windows) ---
    temp_r3 = temp_mean.rolling(24, min_periods=1).mean()
    temp_r7 = temp_mean.rolling(56, min_periods=1).mean()
    sun_r3 = sunshine.rolling(24, min_periods=1).mean()
    sun_r7 = sunshine.rolling(56, min_periods=1).mean()
    rain_r3 = precip.rolling(24, min_periods=1).sum()
    rain_r7 = precip.rolling(56, min_periods=1).sum()

    # --- Temperature deltas (1 day = 8 windows, 3 days = 24 windows) ---
    td1 = temp_mean.diff(8)
    td3 = temp_mean.diff(24)

    # --- Interactions ---
    temp_x_sun = temp_mean * sunshine / 3600.0  # normalize sunshine to hours
    dry_warm = temp_mean * (100.0 - humidity) / 100.0

    # --- Burst potential features (#2) ---
    gdd_thresh = SPECIES_GDD_THRESHOLD.get(species, _DEFAULT_GDD_THRESHOLD)
    activation_temp = SPECIES_ACTIVATION_TEMP.get(species, _DEFAULT_ACTIVATION_TEMP)

    gdd_above = (gdd - gdd_thresh).clip(lower=0)

    # Cold→warm flip: rapid warming while GDD is ready
    cold_to_warm = ((temp_r3 > activation_temp) & (temp_r7 < activation_temp)
                    & (gdd >= gdd_thresh)).astype(float)

    # Consecutive warm windows: count streak of temp > activation_temp
    warm_mask = (temp_mean > activation_temp).astype(int)
    # Compute streak: reset counter when not warm
    streak = warm_mask.copy()
    for i in range(1, len(streak)):
        if streak.iloc[i] == 1:
            streak.iloc[i] = streak.iloc[i - 1] + 1
    consec_warm = streak.astype(float)

    # --- Explosion likelihood features (#6) ---
    # Dry streak: consecutive windows with precipitation < 0.1mm
    dry_mask = (precip < 0.1).astype(int)
    dry_str = dry_mask.copy()
    for i in range(1, len(dry_str)):
        if dry_str.iloc[i] == 1:
            dry_str.iloc[i] = dry_str.iloc[i - 1] + 1

    warm_after_cold = temp_r3 - temp_r7  # positive = warming trend
    wind_x_dw = wind * dry_warm  # dispersal capacity

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
        "gdd_above_threshold": gdd_above,
        "cold_to_warm_flip": cold_to_warm,
        "consecutive_warm_hrs": consec_warm,
        "dry_streak": dry_str.astype(float),
        "warm_after_cold": warm_after_cold,
        "wind_x_dry_warm": wind_x_dw,
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
    species_df = _add_weather_derived_features(species_df, species)
    species_df = _add_ndvi_features(species_df)
    species_df = _add_lag_features(species_df)
    species_df = _add_season_feature(species_df, species)
    species_df = _add_phenology_features(species_df, species)

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


@dataclass
class TwoStageModel:
    """Container for a multi-stage pollen model (classifier + regressor + optional extreme)."""
    classifier: XGBClassifier
    regressor: XGBRegressor
    extreme_regressor: XGBRegressor | None  # (#7) trained only on high-pollen samples
    species: str
    extreme_threshold: float = 50.0  # pollen count above which extreme model activates

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Combined prediction in log-space.

        Returns log1p(pollen_count) predictions.  Caller should apply
        inv_log_transform() to get original-scale values.
        """
        prob_active = self.classifier.predict_proba(X)[:, 1]
        reg_pred = self.regressor.predict(X)
        # Blend: scale regression output by activation probability
        # When prob_active < 0.3, strongly suppress
        result = np.where(prob_active < 0.3, 0.0, reg_pred * np.clip(prob_active, 0.5, 1.0))
        result = np.maximum(0.0, result)

        # (#7) Blend extreme regressor when available and classifier is confident
        if self.extreme_regressor is not None:
            extreme_pred = self.extreme_regressor.predict(X)
            extreme_pred = np.maximum(0.0, extreme_pred)
            # Use extreme model when classifier is very confident (> 0.7)
            # Blend: weighted average biased toward extreme model at high confidence
            extreme_weight = np.clip((prob_active - 0.7) / 0.3, 0.0, 1.0)
            result = result * (1.0 - extreme_weight * 0.5) + extreme_pred * (extreme_weight * 0.5)

        return np.maximum(0.0, result)


# --- Species-specific hyperparameters (#5) ---

# High-variance species need deeper trees and more estimators
_SPECIES_HYPERPARAMS: dict[str, dict] = {
    "Corylus": {"clf_depth": 5, "reg_depth": 7, "reg_n": 500, "quantile": 0.92},
    "Alnus":   {"clf_depth": 5, "reg_depth": 7, "reg_n": 500, "quantile": 0.92},
    "Urtica":  {"clf_depth": 5, "reg_depth": 6, "reg_n": 400, "quantile": 0.90},
    "Poaceae": {"clf_depth": 5, "reg_depth": 6, "reg_n": 400, "quantile": 0.90},
    "Quercus": {"clf_depth": 5, "reg_depth": 6, "reg_n": 400, "quantile": 0.88},
    "Populus": {"clf_depth": 4, "reg_depth": 6, "reg_n": 400, "quantile": 0.88},
}
_DEFAULT_HYPERPARAMS = {"clf_depth": 4, "reg_depth": 5, "reg_n": 300, "quantile": 0.85}


def train_species_model(
    X: pd.DataFrame,  # pylint: disable=invalid-name
    y: pd.Series,
    raw_values: pd.Series | None = None,
    species: str = "",
) -> TwoStageModel:
    """
    Train a multi-stage model for one species.

    Stage 1: XGBClassifier — is pollen > 0 today?
    Stage 2: XGBRegressor  — how much? (quantile regression on all data)
    Stage 3: XGBRegressor  — extreme regressor (trained only on high-pollen samples) (#7)

    Improvements applied:
    - Stronger sample weighting for extreme events (#1)
    - Species-specific hyperparameters (#5)
    - Raised quantile target (#4)
    """
    hp = _SPECIES_HYPERPARAMS.get(species, _DEFAULT_HYPERPARAMS)

    # --- Stage 1: binary classifier ---
    y_binary = (raw_values > 0).astype(int) if raw_values is not None else (y > 0).astype(int)

    # Balance: weight active days more if they're rare
    n_active = y_binary.sum()
    n_total = len(y_binary)
    scale_pos = max(1.0, (n_total - n_active) / max(1, n_active))

    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=hp["clf_depth"],
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )
    classifier.fit(X, y_binary)

    # --- Stage 2: quantile regressor (trained on ALL data, but weighted) (#4, #5) ---
    regressor = XGBRegressor(
        n_estimators=hp["reg_n"],
        max_depth=hp["reg_depth"],
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="reg:quantileerror",
        quantile_alpha=hp["quantile"],
        random_state=42,
        verbosity=0,
    )

    # (#1) Stronger sample weighting: sqrt-based + tier bonuses for extreme events
    sample_weight = None
    if raw_values is not None:
        rv = raw_values.values.astype(float)
        w = 1.0 + np.sqrt(rv)
        w += (rv > 100) * 5.0
        w += (rv > 500) * 10.0
        w += (rv > 1000) * 20.0
        sample_weight = w

    regressor.fit(X, y, sample_weight=sample_weight)

    # --- Stage 3: extreme regressor (#7) ---
    # Trained only on samples where pollen > extreme_threshold, using squared error
    # on raw (non-log) values to directly optimize for high-count accuracy
    extreme_regressor = None
    extreme_threshold = 50.0
    if raw_values is not None:
        extreme_mask = raw_values.values > extreme_threshold
        n_extreme = extreme_mask.sum()
        if n_extreme >= 10:
            X_extreme = X[extreme_mask]
            y_extreme = y[extreme_mask]
            raw_extreme = raw_values.values[extreme_mask]
            # Weight by raw value — biggest events matter most
            w_extreme = 1.0 + np.sqrt(raw_extreme)

            extreme_regressor = XGBRegressor(
                n_estimators=hp["reg_n"],
                max_depth=hp["reg_depth"],
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                objective="reg:squarederror",
                random_state=42,
                verbosity=0,
            )
            extreme_regressor.fit(X_extreme, y_extreme, sample_weight=w_extreme)

    return TwoStageModel(
        classifier=classifier,
        regressor=regressor,
        extreme_regressor=extreme_regressor,
        species=species,
        extreme_threshold=extreme_threshold,
    )


def train_all(history: pd.DataFrame) -> dict[str, TwoStageModel]:
    """
    Train one two-stage model per species.  Returns dict[species → TwoStageModel].
    Only trains if there are enough data points (>= 14 days).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models: dict[str, TwoStageModel] = {}

    for species in ALL_SPECIES:
        X, y, raw_values = prepare_training_data(history, species)
        if len(X) < 14:
            print(f"  {species}: skipped (only {len(X)} training samples, need >= 14)")
            continue

        model = train_species_model(X, y, raw_values=raw_values, species=species)

        # Quick evaluation on training data
        preds_log = model.predict(X)
        preds = inv_log_transform(preds_log)
        actuals = raw_values.values
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))

        clf_acc = (model.classifier.predict(X) == (actuals > 0).astype(int)).mean()
        print(f"  {species}: trained on {len(X)} samples, "
              f"train RMSE={rmse:.1f}, classifier acc={clf_acc:.0%}")

        # Save model (both stages together)
        model_path = MODELS_DIR / f"{species}.joblib"
        joblib.dump(model, model_path)
        models[species] = model

    print(f"\nTrained {len(models)} / {len(ALL_SPECIES)} species models")
    return models


def load_models() -> dict[str, TwoStageModel]:
    """Load all trained two-stage models from disk."""
    models: dict[str, TwoStageModel] = {}
    for species in ALL_SPECIES:
        model_path = MODELS_DIR / f"{species}.joblib"
        if model_path.exists():
            models[species] = joblib.load(model_path)
    return models
