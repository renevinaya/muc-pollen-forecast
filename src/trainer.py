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

from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from .types import (
    ALL_SPECIES,
    CALENDAR_FEATURES,
    CAMS_FEATURES,
    FEATURE_COLS,
    INTRADAY_FEATURES,
    LAG_FEATURES,
    NDVI_FEATURES,
    PHENOLOGY_FEATURES,
    SEASON_FEATURE,
    WEATHER_DERIVED_FEATURES,
    LEAD_FEATURES,
    WEATHER_FEATURES,
    WINDOW_FEATURES,
    GDD_T_BASE,
    is_season_active,
    SPECIES_GDD_THRESHOLD,
    SPECIES_ACTIVATION_TEMP,
    _DEFAULT_GDD_THRESHOLD,
    _DEFAULT_ACTIVATION_TEMP,
)
from .onset import (
    calibrated_gdd_threshold,
    climatological_onset_doy,
    gdd_threshold_by_year,
    gdd_threshold_series,
    observed_onsets,
    onset_doy_by_day,
    _static_onset_doy,
)

MODELS_DIR = Path(__file__).parent.parent / "models"

# Stage 3 blend gate. The extreme regressor sees only samples above
# ``extreme_threshold``, so it has no idea what an ordinary window looks like and
# must not be consulted about one. The gate is therefore a calibrated
# P(value > threshold): the weight ramps in from EXTREME_GATE_LO ("more likely
# than not") to EXTREME_GATE_HI, and peaks at EXTREME_MAX_WEIGHT.
EXTREME_GATE_LO = 0.5
EXTREME_GATE_HI = 0.9
EXTREME_MAX_WEIGHT = 0.7

# --- Log-transform helpers ---

def log_transform(values: pd.Series | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Apply log1p transform to pollen counts."""
    return np.log1p(np.asarray(values, dtype=float))  # type: ignore[no-any-return]


def inv_log_transform(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Inverse of log1p: expm1."""
    return np.expm1(values)  # type: ignore[no-any-return]


# --- Feature engineering ---

def _add_lag_features(df: pd.DataFrame, lead: int = 1) -> pd.DataFrame:
    """
    Add lag and rolling features for a single species' time series.
    Lag features are computed in log-space. Each row is a 3h window.
    Expects df sorted by date with a 'value' column.

    *lead* is how many windows ahead of the last known measurement each row is
    being predicted. At ``lead=1`` the block is the state of the world one
    window before the target, which is all a one-step forecast ever needs. At
    ``lead=L`` the whole block is instead anchored L windows back — it describes
    what was known at the forecast origin, not at the target.

    That anchoring is what makes the model *direct* rather than recursive. The
    recursive version fed each prediction back in as the next window's lag, and
    since the predictions carry an upward bias, the bias compounded: over six
    folds the mean prediction climbed 12.97 -> 17.35 from day 1 to day 5 while
    the mean actual was flat at 9.0 -> 7.7. Anchoring at the origin means every
    lag the model ever sees is a real measurement, in training and in serving
    alike, so there is no loop for a bias to go round.

    The shift is uniform across the block, which keeps its internal structure
    intact (most recent known window, the one before it, the 24h mean ending
    there, ...) and simply moves the whole thing back to the origin.
    """
    df = df.copy().sort_values("date")
    log_val = log_transform(df["value"])
    s = pd.Series(log_val, index=df.index)
    df["pollen_lag_1"] = s.shift(1)           # previous 3h window
    df["pollen_lag_2"] = s.shift(2)           # 6h ago
    df["pollen_lag_3"] = s.shift(3)           # 9h ago
    df["pollen_lag_8"] = s.shift(8)           # same time yesterday (24h)
    df["pollen_lag_16"] = s.shift(16)         # 48h ago (#3)
    df["pollen_lag_24"] = s.shift(24)         # same time 3 days ago (72h)
    df["pollen_lag_56"] = s.shift(56)         # 7 days ago (#3)
    df["pollen_rolling_8"] = s.rolling(8, min_periods=1).mean().shift(1)    # 24h mean
    df["pollen_rolling_56"] = s.rolling(56, min_periods=1).mean().shift(1)  # 7-day mean
    df["pollen_max_8"] = s.rolling(8, min_periods=1).max().shift(1)         # 24h max (#3)
    df["pollen_max_56"] = s.rolling(56, min_periods=1).max().shift(1)       # 7-day max (#3)
    # Mean of today's earlier windows (intra-day trend signal)
    # Group by calendar day, use expanding mean of log-values within the day, shifted
    day_groups = pd.to_datetime(df["date"]).dt.normalize()
    morning_avg = s.groupby(day_groups.values).apply(
        lambda g: g.expanding(min_periods=1).mean().shift(1)
    )
    if hasattr(morning_avg.index, 'droplevel'):
        try:
            morning_avg = morning_avg.droplevel(0)
        except (ValueError, IndexError):
            pass
    df["pollen_morning_avg"] = morning_avg.reindex(df.index).fillna(0).values
    # Windows since pollen was last > 0.
    # The cumsum formulation this replaces was degenerate: cumsum does not
    # advance while a species is inactive, so ``cumactive - last_active`` was 0
    # on every row after the first active one — the model trained on a constant
    # while the forecaster served it a real count. Position arithmetic gives
    # the distance the feature was always meant to carry (0–980 windows here).
    positions = pd.Series(np.arange(len(s), dtype=float), index=s.index)
    last_active = positions.where(s > 0).ffill()
    df["days_since_active"] = (positions - last_active).shift(1).fillna(999).astype(float)

    # Move the finished block back to the forecast origin. Every column above
    # describes the state as of one window before its row, so shifting by
    # lead - 1 makes it describe the state as of `lead` windows before instead.
    if lead > 1:
        df[LAG_FEATURES] = df[LAG_FEATURES].shift(lead - 1)
    df["lead_windows"] = float(lead)
    return df


def _add_season_feature(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Add a binary season_active feature based on the species' known pollen window."""
    df = df.copy()
    months = pd.to_datetime(df["date"]).dt.month
    df["season_active"] = months.apply(lambda m: 1.0 if is_season_active(species, m) else 0.0)
    return df


def typical_onset_doy(species: str, history: pd.DataFrame | None = None) -> float:
    """Resolve a species' typical flowering-onset day-of-year.

    Resolution order:
      1. Median onset measured from *history* (eight seasons and counting).
      2. Central-European baseline (SPECIES_TYPICAL_ONSET_DOY).
      3. Approximation: the 15th of the species' season-start month.

    The DWD phenology file used to sit at the top of this list. It yields one
    year of Munich observations with no Alnus at all, which put Corylus 24 days
    and Alnus 17 days off their measured medians, so the measurements win now.
    """
    if history is not None and not history.empty:
        return climatological_onset_doy(history, species)
    return _static_onset_doy(species)


def onset_anomaly_from_gdd(
    gdd: "pd.Series | np.ndarray[Any, Any] | float",
    species: str,
    threshold: "pd.Series | np.ndarray[Any, Any] | float | None" = None,
):
    """Signed, normalised thermal readiness relative to the species GDD threshold.

    ``(gdd - threshold) / threshold`` — strongly negative before the plant has
    accumulated enough warmth (pre-onset), ~0 around the threshold crossing, and
    positive afterwards. Because a warm year crosses the threshold at an earlier
    calendar date, this encodes whether the current season is running early or
    late, which is the signal the old constant-0 feature never delivered.

    *threshold* may be per-row, which is how callers pass the walk-forward
    calibration; without it the static constant applies.
    """
    if threshold is None:
        threshold = SPECIES_GDD_THRESHOLD.get(species, _DEFAULT_GDD_THRESHOLD)
    thresh = np.maximum(np.asarray(threshold, dtype=float), 1.0)
    anomaly = (np.asarray(gdd, dtype=float) - thresh) / thresh
    return np.clip(anomaly, -3.0, 5.0)


def _add_phenology_features(
    df: pd.DataFrame,
    species: str,
    onset_by_day: pd.Series | None = None,
    gdd_thresholds: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Add phenology-derived features: days since flowering onset, onset anomaly.

    ``days_since_typical_onset`` measures against a *per-year* onset estimate
    rather than one constant for all time. Until this year's warmth reaches the
    threshold the estimate is the median of previous seasons; from the crossing
    onwards it is the crossing day itself. That switch is the whole point — it
    is how the model learns the season is running early or late — and because
    it only ever looks backwards it is equally computable while training and
    while forecasting.

    ``onset_anomaly`` is the same thermal-readiness signal expressed against the
    walk-forward GDD threshold. Expects the ``gdd`` column to already be present
    (added by _add_weather_derived_features).

    *onset_by_day* and *gdd_thresholds* let a caller supply both from the real
    history; the forecaster must, because it builds features on a weather-only
    frame that carries no measurements to derive them from.
    """
    df = df.copy()
    if onset_by_day is None:
        onset_by_day = onset_doy_by_day(df, species)

    days = pd.to_datetime(df["date"]).dt.normalize()
    doys = pd.to_datetime(df["date"]).dt.dayofyear

    # The static baseline, not a median over *df*, is the fallback: a median
    # taken here would span every year in the frame, including the one being
    # predicted, which is exactly the leak onset_doy_by_day exists to avoid.
    fallback = _static_onset_doy(species)
    if onset_by_day.empty:
        onset = pd.Series(fallback, index=df.index, dtype=float)
    else:
        # Days past the end of the estimates (forecast windows) hold the last
        # value that was actually available. Done by reindexing rather than by
        # forward-filling *df*, so the result does not depend on row order.
        resolved = onset_by_day.reindex(
            onset_by_day.index.union(pd.DatetimeIndex(days.unique()))
        ).ffill()
        onset = days.map(resolved).astype(float).fillna(fallback)

    if onset.isna().all():  # unknown species — no season to be early or late for
        df["days_since_typical_onset"] = 0.0
        df["onset_anomaly"] = 0.0
        return df

    df["days_since_typical_onset"] = (doys - onset).clip(lower=-60).astype(float)

    gdd = df["gdd"] if "gdd" in df.columns else pd.Series(0.0, index=df.index)
    if gdd_thresholds is None:
        gdd_thresholds = gdd_threshold_by_year(df, species)
    threshold = gdd_threshold_series(
        gdd_thresholds, pd.DatetimeIndex(days), species
    ).to_numpy()
    df["onset_anomaly"] = onset_anomaly_from_gdd(gdd, species, threshold)
    return df


def _add_ndvi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add NDVI features from cached satellite data."""
    df = df.copy()
    # Default to 0 — will be populated by collector; if columns already present, skip
    for col in ("ndvi", "evi", "ndvi_delta"):
        if col not in df.columns:
            df[col] = 0.0
    return df


def _add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add intra-day relative features that capture diurnal position.

    - temp_vs_daily_max: ratio of window temperature to the day's max
    - precip_in_prior_window: binary flag for rain in the previous 3h window
    - temp_rate_of_change: temperature difference from the previous 3h window
    """
    df = df.copy().sort_values("date")

    temp_mean = df["temperature_mean"].fillna(0)

    # Daily max temperature (group by calendar day)
    day_col = pd.to_datetime(df["date"]).dt.normalize()
    daily_max = temp_mean.groupby(day_col.values).transform("max")
    df["temp_vs_daily_max"] = np.where(
        daily_max > 0, temp_mean / daily_max, 0.0
    )

    # Precipitation in prior window (binary)
    precip = df["precipitation_sum"].fillna(0)
    df["precip_in_prior_window"] = (precip.shift(1) > 0.1).astype(float).fillna(0)

    # Temperature rate of change (diff from previous 3h window)
    df["temp_rate_of_change"] = temp_mean.diff(1).fillna(0)

    return df


def _add_weather_derived_features(
    df: pd.DataFrame, species: str = "", gdd_thresholds: dict[int, float] | None = None,
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

    The GDD threshold behind the burst features is calibrated per year from the
    seasons before it. Pass *gdd_thresholds* when *df* holds no measurements to
    calibrate from — the forecaster derives weather features on a dummy frame.
    """
    df = df.copy().sort_values("date")

    # De-duplicate so rolling computations are per-window
    temp_mean = df.groupby("date")["temperature_mean"].first()
    sunshine = df.groupby("date")["sunshine_duration"].first()
    precip = df.groupby("date")["precipitation_sum"].first()
    humidity = df.groupby("date")["humidity_mean"].first()
    wind = df.groupby("date")["wind_speed_max"].first()

    # --- GDD (cumsum of daily max(0, T_mean - T_base), reset each Jan 1) ---
    window_dates = pd.DatetimeIndex(pd.to_datetime(temp_mean.index)).normalize()
    daily_temp = pd.Series(temp_mean.values, index=window_dates).groupby(level=0).mean()
    daily_gdd_contrib = (daily_temp - GDD_T_BASE).clip(lower=0)
    gdd_daily = daily_gdd_contrib.groupby(pd.DatetimeIndex(daily_temp.index).year).cumsum()
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
    if gdd_thresholds is None:
        gdd_thresholds = gdd_threshold_by_year(df, species)
    gdd_thresh = gdd_threshold_series(gdd_thresholds, window_dates, species)
    gdd_thresh.index = temp_mean.index
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

    # --- Upwind transport features ---
    # Wind direction: where the wind is coming FROM (meteorological convention)
    # 0°/360° = N, 90° = E, 180° = S, 270° = W
    # Munich is surrounded by alpine forests (S) and agricultural plains (N)
    if "wind_direction" in df.columns:
        wind_dir = df.groupby("date")["wind_direction"].first()
    else:
        wind_dir = pd.Series(0.0, index=temp_mean.index)
    dir_rad = np.radians(wind_dir)
    wd_sin = np.sin(dir_rad)              # positive = from east
    wd_cos = np.cos(dir_rad)              # positive = from north
    # Directional indicators (how much wind comes from that direction)
    wind_from_south = (-wd_cos).clip(lower=0)   # 180° → 1.0
    wind_from_north = wd_cos.clip(lower=0)       # 0°   → 1.0
    # Transport potential = wind speed × directional strength
    transport_s = wind * wind_from_south
    transport_n = wind * wind_from_north

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
        "wind_dir_sin": wd_sin,
        "wind_dir_cos": wd_cos,
        "wind_from_south": wind_from_south,
        "wind_from_north": wind_from_north,
        "transport_south": transport_s,
        "transport_north": transport_n,
    })

    for col in derived.columns:
        mapping = derived[col].to_dict()
        df[col] = df["date"].map(mapping)

    df = df.fillna({col: 0.0 for col in derived.columns})
    return df


# Leads the model is trained at, in 3h windows. A 5-day forecast spans leads
# 1..40, and one row per lead would be 40x the training data, so these sample
# the range: dense early, where a window's own recent past still dominates, and
# sparse later, where it barely matters. ``lead_windows`` is a feature, so the
# model interpolates between them.
TRAINING_LEADS = (1, 4, 8, 16, 24, 32, 40)


def prepare_training_data(
    history: pd.DataFrame, species: str, leads: tuple[int, ...] = TRAINING_LEADS
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare feature matrix X, target y (log-transformed), and raw_values
    (original scale) for a single species.

    One copy of the history per entry in *leads*: the same target windows, each
    time with the lag block anchored that many windows further back. This is
    what teaches the model to forecast several days ahead directly, instead of
    forecasting one window ahead and being fed its own output forty times.

    Drops rows where lag features are NaN (the first few days, plus the first
    *lead* windows of each copy).
    """
    species_df = history[history["species"] == species].copy()
    if species_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    species_df = species_df.sort_values("date").reset_index(drop=True)
    # Ensure all expected feature columns exist (backward compat with old history)
    for col in FEATURE_COLS:
        if col not in species_df.columns:
            species_df[col] = 0.0
    species_df = _add_weather_derived_features(species_df, species)
    species_df = _add_ndvi_features(species_df)
    species_df = _add_intraday_features(species_df)
    species_df = _add_season_feature(species_df, species)
    species_df = _add_phenology_features(species_df, species)

    # Everything above is independent of the lead, so it is built once and only
    # the lag block is rebuilt per lead.
    per_lead: list[pd.DataFrame] = []
    for lead in leads:
        frame = _add_lag_features(species_df, lead=lead)
        frame = frame.dropna(subset=LAG_FEATURES)
        if not frame.empty:
            per_lead.append(frame)

    if not per_lead:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    combined = pd.concat(per_lead, ignore_index=True)

    X = combined[FEATURE_COLS].copy()
    raw_values = combined["value"].reset_index(drop=True)
    y = pd.Series(log_transform(combined["value"]), index=combined.index)

    # Fill any remaining NaN in features with 0
    X = X.fillna(0)

    return X, y, raw_values


@dataclass
class TwoStageModel:
    """Container for a multi-stage pollen model (classifier + regressor + optional extreme)."""
    classifier: XGBClassifier
    regressor: XGBRegressor
    extreme_regressor: XGBRegressor | None  # trained only on high-pollen samples
    species: str
    extreme_threshold: float = 50.0  # pollen count above which extreme model activates
    # P(value > extreme_threshold), the gate for the stage-3 blend. Defaults to
    # None so models pickled before this field existed still load; those are
    # served without the blend rather than with the gate that used to be wrong.
    extreme_classifier: XGBClassifier | None = None
    # The feature columns this model was fitted on. Serving a model a different
    # set is never merely degraded — XGBoost raises on a name mismatch, and a
    # silent reordering would be worse — so load_models refuses the mismatch
    # rather than letting it reach a running pipeline. Defaults to None for
    # models pickled before this field existed.
    feature_names: list[str] | None = None

    def predict(self, X: pd.DataFrame | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:  # pylint: disable=invalid-name
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

        # Blend in the extreme regressor, in proportion to how likely this
        # window is to *be* extreme.
        #
        # This used to ramp on ``prob_active``, which is P(pollen > 0), not
        # P(pollen > threshold). In peak season that sits near 1.0 for weeks, so
        # a regressor fitted only to samples above the threshold was given its
        # full weight on ordinary windows: measured over this history the blend
        # fired on 20–28% of all windows, and at those windows the truth was at
        # or below the threshold ~75% of the time and exactly zero 14–28% of the
        # time. It was a large, permanent upward bias.
        #
        # getattr keeps models pickled before ``extreme_classifier`` existed
        # loadable; without a gate the blend is skipped rather than fall back to
        # the one that caused the bias.
        extreme_regressor = getattr(self, "extreme_regressor", None)
        extreme_classifier = getattr(self, "extreme_classifier", None)
        if extreme_regressor is not None and extreme_classifier is not None:
            prob_extreme = extreme_classifier.predict_proba(X)[:, 1]
            gate = np.clip(
                (prob_extreme - EXTREME_GATE_LO) / (EXTREME_GATE_HI - EXTREME_GATE_LO),
                0.0,
                1.0,
            )
            extreme_pred = np.maximum(0.0, extreme_regressor.predict(X))
            weight = gate * EXTREME_MAX_WEIGHT
            result = result * (1.0 - weight) + extreme_pred * weight

        return np.maximum(0.0, result)


# --- Species-specific hyperparameters (#5) ---

# High-variance species need deeper trees and more estimators
_SPECIES_HYPERPARAMS: dict[str, dict[str, int | float]] = {
    "Corylus": {"clf_depth": 5, "reg_depth": 7, "reg_n": 500, "quantile": 0.92},
    "Alnus":   {"clf_depth": 5, "reg_depth": 7, "reg_n": 500, "quantile": 0.92},
    "Urtica":  {"clf_depth": 5, "reg_depth": 6, "reg_n": 400, "quantile": 0.90},
    "Poaceae": {"clf_depth": 5, "reg_depth": 6, "reg_n": 400, "quantile": 0.90},
    "Quercus": {"clf_depth": 5, "reg_depth": 6, "reg_n": 400, "quantile": 0.88},
    "Populus": {"clf_depth": 4, "reg_depth": 6, "reg_n": 400, "quantile": 0.88},
}
_DEFAULT_HYPERPARAMS: dict[str, int | float] = {"clf_depth": 4, "reg_depth": 5, "reg_n": 300, "quantile": 0.85}


def train_species_model(
    X: pd.DataFrame,  # pylint: disable=invalid-name
    y: pd.Series,
    raw_values: pd.Series | None = None,
    species: str = "",
) -> TwoStageModel | None:
    """
    Train a multi-stage model for one species.

    Stage 1: XGBClassifier — is pollen > 0 today?
    Stage 2: XGBRegressor  — how much? (quantile regression on all data)
    Stage 3: XGBRegressor  — extreme regressor (fitted only to high-pollen
             samples), gated by an XGBClassifier for P(value > threshold)

    Improvements applied:
    - Stronger sample weighting for extreme events (#1)
    - Species-specific hyperparameters (#5)
    - Raised quantile target (#4)
    """
    hp = _SPECIES_HYPERPARAMS.get(species, _DEFAULT_HYPERPARAMS)

    # --- Stage 1: binary classifier ---
    y_binary = (raw_values > 0).astype(int) if raw_values is not None else (y > 0).astype(int)

    # Skip species with only one class (e.g., all zeros when out of season)
    n_active = int(y_binary.sum())
    n_total = len(y_binary)
    if n_active == 0 or n_active == n_total:
        print(f"  {species}: skipped (single class — {'all zero' if n_active == 0 else 'all active'})")
        return None

    # Balance: weight active days more if they're rare
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
        rv = raw_values.to_numpy(dtype=float)
        w = 1.0 + np.sqrt(rv)
        w += (rv > 100) * 8.0
        w += (rv > 500) * 20.0
        w += (rv > 1000) * 40.0
        sample_weight = w

    regressor.fit(X, y, sample_weight=sample_weight)

    # --- Stage 3: extreme regressor, plus the gate that decides when to use it ---
    # The regressor is fitted only to samples above extreme_threshold, with
    # squared error in log space (the same target the stage-2 regressor uses;
    # squared error on raw counts would be dominated by the largest events to
    # the point of ignoring everything else).
    #
    # Because it never sees an ordinary window, it cannot be asked about one.
    # The gate is a separate classifier for P(value > extreme_threshold),
    # trained on all the data. It deliberately does *not* balance classes:
    # scale_pos_weight would inflate the probabilities, and this gate is only
    # meaningful if 0.5 really means "more likely than not".
    extreme_regressor = None
    extreme_classifier = None
    extreme_threshold = 50.0
    if raw_values is not None:
        rv_arr = raw_values.to_numpy(dtype=float)
        extreme_mask = rv_arr > extreme_threshold
        n_extreme = int(extreme_mask.sum())
        # Both classes must be present for the gate to be learnable at all.
        if n_extreme >= 10 and n_extreme < len(rv_arr):
            X_extreme = X[extreme_mask]
            y_extreme = y[extreme_mask]
            raw_extreme = rv_arr[extreme_mask]
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

            extreme_classifier = XGBClassifier(
                n_estimators=200,
                max_depth=hp["clf_depth"],
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                eval_metric="logloss",
            )
            extreme_classifier.fit(X, extreme_mask.astype(int))

    return TwoStageModel(
        classifier=classifier,
        regressor=regressor,
        extreme_regressor=extreme_regressor,
        species=species,
        extreme_threshold=extreme_threshold,
        extreme_classifier=extreme_classifier,
        feature_names=list(X.columns),
    )


def _print_onset_calibration(history: pd.DataFrame) -> None:
    """Report the onset calibration this run is training against.

    Both numbers drift as seasons accumulate, and both silently change what the
    phenology features mean, so the retrain log is the right place to see them.
    """
    print("\n  Onset calibration (measured from history):")
    print(f"    {'Species':<12} {'onset DOY':>10} {'seasons':>8} {'GDD thr':>9}")
    print(f"    {'-'*12} {'-'*10} {'-'*8} {'-'*9}")
    for species in ALL_SPECIES:
        seasons = len(observed_onsets(history, species))
        onset = typical_onset_doy(species, history)
        threshold = calibrated_gdd_threshold(history, species)
        measured = "" if seasons else "  (baseline)"
        print(f"    {species:<12} {onset:>10.0f} {seasons:>8} {threshold:>9.1f}{measured}")
    print()


# Feature families, for the gain report. Keyed in the order they are printed.
FEATURE_FAMILIES: dict[str, list[str]] = {
    "weather": WEATHER_FEATURES,
    "lead": LEAD_FEATURES,
    "calendar": CALENDAR_FEATURES + WINDOW_FEATURES,
    "season": SEASON_FEATURE,
    "weather_derived": WEATHER_DERIVED_FEATURES,
    "ndvi": NDVI_FEATURES,
    "phenology": PHENOLOGY_FEATURES,
    "cams": CAMS_FEATURES,
    "intraday": INTRADAY_FEATURES,
    "lag": LAG_FEATURES,
}


def feature_gain(models: dict[str, TwoStageModel]) -> pd.DataFrame:
    """Total XGBoost gain per feature, summed over species and stages.

    Gain is normalised per model before summing so that one species with a
    large absolute gain does not decide the ranking for all of them.
    """
    totals: dict[str, float] = {col: 0.0 for col in FEATURE_COLS}
    for model in models.values():
        stages = [
            model.classifier,
            model.regressor,
            model.extreme_regressor,
            getattr(model, "extreme_classifier", None),
        ]
        for stage in stages:
            if stage is None:
                continue
            scores = stage.get_booster().get_score(importance_type="gain")
            total = sum(scores.values()) or 1.0
            for name, value in scores.items():
                if name in totals:
                    totals[name] += value / total

    family_of = {
        col: family
        for family, cols in FEATURE_FAMILIES.items()
        for col in cols
    }
    frame = pd.DataFrame(
        {
            "feature": list(totals),
            "family": [family_of.get(c, "other") for c in totals],
            "gain": list(totals.values()),
        }
    )
    grand = frame["gain"].sum() or 1.0
    frame["share"] = frame["gain"] / grand
    return frame.sort_values("gain", ascending=False).reset_index(drop=True)


def _print_feature_gain(models: dict[str, TwoStageModel], top: int = 15) -> None:
    """Report which features the trained models actually use.

    72 features accumulated without anything reporting what they earn, so
    pruning arguments had no evidence to run on. Both views matter: the family
    totals say which *groups* are dead weight, and the per-feature list says
    which single columns carry a family.
    """
    if not models:
        return
    frame = feature_gain(models)

    print("\n  Feature gain by family (share of total gain across all models):")
    print(f"    {'Family':<18} {'Share':>8} {'Features':>9} {'Top feature':<24}")
    print(f"    {'-'*18} {'-'*8} {'-'*9} {'-'*24}")
    families = frame.groupby("family")["share"].sum().sort_values(ascending=False)
    for family, share in families.items():
        members = frame[frame["family"] == family]
        best = members.iloc[0]["feature"] if not members.empty else "—"
        print(f"    {str(family):<18} {share:>7.1%} {len(members):>9} {best:<24}")

    print(f"\n  Top {top} features:")
    print(f"    {'Feature':<26} {'Family':<18} {'Share':>8}")
    print(f"    {'-'*26} {'-'*18} {'-'*8}")
    for _, row in frame.head(top).iterrows():
        print(f"    {row['feature']:<26} {row['family']:<18} {row['share']:>7.2%}")

    dead = frame[frame["gain"] <= 0.0]
    if not dead.empty:
        print(f"\n  Never split on ({len(dead)}): {', '.join(dead['feature'])}")
    print()


# Raw history columns for which an exact 0.0 is a fill value rather than a
# measurement: NDVI over Munich never reads 0, a 3 h mean dew point or boundary
# layer height of exactly 0.0 does not happen, and the weather parser writes
# 0.0 for soil when the source column is missing. Everything else in the
# feature list (rain, sunshine, is_day, ndvi_delta on a flat day, the lags) is
# legitimately zero often.
ZERO_MEANS_MISSING = frozenset(
    [
        "ndvi",
        "soil_temperature_mean",
        "soil_moisture_mean",
        "boundary_layer_height",
        "dew_point_mean",
    ]
)

# A feature missing on more than this share of training rows is not a feature
# the model can learn; the retrain refuses rather than fit an era marker.
MAX_MISSING_SHARE = 0.5


class FeatureCoverageError(ValueError):
    """Raised when a model input is missing on most of the training rows."""


def feature_coverage(history: pd.DataFrame) -> pd.DataFrame:
    """Share of rows on which each *raw* model input is missing.

    Checked on the raw columns rather than the prepared matrix because the
    derived features are computed from these, and ``prepare_training_data``
    fills every NaN with 0 before XGBoost sees it — so a column that is NaN
    for six seasons reaches the model as six seasons of zeros, which is why
    this was invisible until someone looked. "Missing" is NaN, or an exact 0.0
    for the columns in :data:`ZERO_MEANS_MISSING`.

    One row per feature, sorted by missing share: ``missing`` (share),
    ``first_present`` (first date with a real value), ``mode_share`` (share of
    rows holding the single most common value, as a hint for other fills).
    """
    raw = [c for c in FEATURE_COLS if c in history.columns]
    if history.empty or not raw:
        return pd.DataFrame(columns=["feature", "missing", "first_present", "mode_share"])

    dates = pd.to_datetime(history["date"])
    rows: list[dict[str, object]] = []
    for col in raw:
        values = pd.to_numeric(history[col], errors="coerce")
        missing = values.isna()
        if col in ZERO_MEANS_MISSING:
            missing = missing | (values == 0.0)
        present = dates[~missing]
        counts = values.value_counts(dropna=False)
        rows.append(
            {
                "feature": col,
                "missing": float(missing.mean()),
                "first_present": present.min() if not present.empty else pd.NaT,
                "mode_share": float(counts.iloc[0] / len(values)) if len(counts) else 1.0,
            }
        )
    return pd.DataFrame(rows).sort_values("missing", ascending=False).reset_index(drop=True)


def check_feature_coverage(
    history: pd.DataFrame, max_missing: float = MAX_MISSING_SHARE
) -> pd.DataFrame:
    """Print the coverage report and refuse to train on mostly-missing inputs."""
    report = feature_coverage(history)
    flagged = report[report["missing"] > 0]
    print("\n  Feature coverage (raw columns missing on more than 0% of rows):")
    if flagged.empty:
        print("    every model input is present on every row")
    else:
        print(f"    {'Feature':<26} {'missing':>8} {'first present':>14} {'mode share':>11}")
        for _, r in flagged.iterrows():
            first = r["first_present"].date() if pd.notna(r["first_present"]) else "never"
            print(f"    {r['feature']:<26} {r['missing']:>7.1%} {str(first):>14} "
                  f"{r['mode_share']:>10.1%}")

    bad = report[report["missing"] > max_missing]
    if not bad.empty:
        names = ", ".join(f"{r.feature} ({r.missing:.0%})" for r in bad.itertuples())
        raise FeatureCoverageError(
            f"{len(bad)} model input(s) missing on more than {max_missing:.0%} of "
            f"training rows: {names}. Run `python -m src.main run-backfill` to "
            "fill the history, or drop the feature from FEATURE_COLS."
        )
    return report


def train_all(history: pd.DataFrame) -> dict[str, TwoStageModel]:
    """
    Train one two-stage model per species.  Returns dict[species → TwoStageModel].
    Only trains if there are enough data points (>= 14 days).

    Refuses (``FeatureCoverageError``) when a model input is missing on most
    of the rows — see :func:`check_feature_coverage`.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models: dict[str, TwoStageModel] = {}

    _print_onset_calibration(history)
    check_feature_coverage(history)

    for species in ALL_SPECIES:
        X, y, raw_values = prepare_training_data(history, species)
        if len(X) < 14:
            print(f"  {species}: skipped (only {len(X)} training samples, need >= 14)")
            continue

        model = train_species_model(X, y, raw_values=raw_values, species=species)
        if model is None:
            continue

        # Quick evaluation on training data
        preds_log = model.predict(X)
        preds = inv_log_transform(preds_log)
        actuals = raw_values.to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))

        clf_acc = float((model.classifier.predict(X) == (actuals > 0).astype(int)).mean())
        print(f"  {species}: trained on {len(X)} samples, "
              f"train RMSE={rmse:.1f}, classifier acc={clf_acc:.0%}")

        # Save model (both stages together)
        model_path = MODELS_DIR / f"{species}.joblib"
        joblib.dump(model, model_path)
        models[species] = model

    print(f"\nTrained {len(models)} / {len(ALL_SPECIES)} species models")
    _print_feature_gain(models)
    return models


def load_models() -> dict[str, TwoStageModel]:
    """Load trained models from disk, skipping any that no longer fit.

    The pipeline checks out new code every run but keeps serving the models on
    the data release until the next retrain, so a commit that changes
    FEATURE_COLS is briefly live against models fitted on the old set. XGBoost
    raises on a feature-name mismatch, which would take the whole forecast down.

    A skipped species falls back to the forecaster's no-model path for one
    cycle — a worse forecast, but a published one — and comes back at the next
    retrain. Models pickled before ``feature_names`` existed carry None; those
    predate the field rather than disagreeing with it, so they are checked
    against the count XGBoost itself reports.
    """
    models: dict[str, TwoStageModel] = {}
    stale: list[str] = []
    for species in ALL_SPECIES:
        model_path = MODELS_DIR / f"{species}.joblib"
        if not model_path.exists():
            continue
        loaded: TwoStageModel = joblib.load(model_path)

        names = getattr(loaded, "feature_names", None)
        if names is None:
            fitted = getattr(loaded.regressor, "n_features_in_", len(FEATURE_COLS))
            matches = int(fitted) == len(FEATURE_COLS)
        else:
            matches = list(names) == list(FEATURE_COLS)

        if matches:
            models[species] = loaded
        else:
            stale.append(species)

    if stale:
        print(
            f"  Skipped {len(stale)} model(s) fitted on a different feature set "
            f"({', '.join(stale)}) — they will be replaced at the next retrain."
        )
    return models
