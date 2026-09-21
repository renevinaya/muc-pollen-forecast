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
  - A raised quantile target is the one peak-emphasis mechanism (E.2)
  - Quantile regression (α = 0.85–0.92 per species) biases toward higher predictions
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
    LOAD_FEATURES,
    UPWIND_FEATURES,
    SEASON_FEATURE,
    WEATHER_DERIVED_FEATURES,
    LEAD_FEATURES,
    WEATHER_FEATURES,
    WINDOW_FEATURES,
    GDD_T_BASE,
    is_season_active,
    SPECIES_ACTIVATION_TEMP,
    _DEFAULT_ACTIVATION_TEMP,
)
from .upwind import upwind_block, upwind_series, with_lead_feature
from .onset import (
    climatological_onset_doy,
    observed_onsets,
    onset_doy_by_day,
    readiness_by_day,
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

WINDOW = pd.Timedelta(hours=3)


def grid_series(values: pd.Series, end: pd.Timestamp | None = None) -> tuple[pd.Series, pd.Series]:
    """A species' log values on the full 3 h grid, raw and gap-filled.

    *values* is indexed by measurement time. The raw series is NaN wherever a
    window was not measured. The filled one is "the state of the world as
    last known" at a window inside an outage: the same time of day one day
    earlier when that was measured (pollen is diurnal, so yesterday's noon
    is a better stand-in for a missing noon than this morning's 03:00), and
    the last measurement of any kind otherwise. The grid runs to *end* when
    that lies past the last measurement.
    """
    index = pd.DatetimeIndex(values.index)
    if len(index) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty
    stop = index.max() if end is None else max(index.max(), pd.Timestamp(end))
    grid = pd.date_range(index.min(), stop, freq=WINDOW)
    raw = pd.Series(log_transform(values), index=index)
    # A window measured twice (a re-collected day) keeps its latest value,
    # as the collector's own de-duplication does.
    raw = raw[~raw.index.duplicated(keep="last")].reindex(grid)
    same_slot = raw.groupby(raw.index.hour).ffill(limit=1)
    return raw, same_slot.ffill()


def lag_block_on_grid(values: pd.Series, lead: int = 1) -> pd.DataFrame:
    """The lag block for every window of the 3 h grid spanned by *values*.

    Time-based (E.1): a lag of 8 is the window 24 h earlier whether or not
    the station reported in between, and ``days_since_active`` counts
    windows of time, not rows. The block at grid window *t* describes the
    state as of ``lead`` windows before *t*.
    """
    raw, s = grid_series(values)
    out = pd.DataFrame(index=s.index)
    if s.empty:
        return out.reindex(columns=LAG_FEATURES)
    out["pollen_lag_1"] = s.shift(1)           # previous 3h window
    out["pollen_lag_2"] = s.shift(2)           # 6h ago
    out["pollen_lag_3"] = s.shift(3)           # 9h ago
    out["pollen_lag_8"] = s.shift(8)           # same time yesterday (24h)
    out["pollen_lag_16"] = s.shift(16)         # 48h ago (#3)
    out["pollen_lag_24"] = s.shift(24)         # same time 3 days ago (72h)
    out["pollen_lag_56"] = s.shift(56)         # 7 days ago (#3)
    out["pollen_rolling_8"] = s.rolling(8, min_periods=1).mean().shift(1)    # 24h mean
    out["pollen_rolling_56"] = s.rolling(56, min_periods=1).mean().shift(1)  # 7-day mean
    out["pollen_max_8"] = s.rolling(8, min_periods=1).max().shift(1)         # 24h max (#3)
    out["pollen_max_56"] = s.rolling(56, min_periods=1).max().shift(1)       # 7-day max (#3)
    # Mean of the day's earlier windows (intra-day trend signal).
    day_groups = s.index.normalize()
    out["pollen_morning_avg"] = (
        s.groupby(day_groups).transform(lambda g: g.expanding(min_periods=1).mean().shift(1))
        .fillna(0.0)
    )
    # Windows since pollen was last *measured* above 0: a gap does not count
    # as activity, and the distance is in time. The cumsum formulation this
    # replaces was degenerate (constant 0 after the first active row).
    positions = pd.Series(np.arange(len(s), dtype=float), index=s.index)
    last_active = positions.where(raw > 0).ffill()
    out["days_since_active"] = (positions - last_active).shift(1).fillna(999).astype(float)

    # Move the finished block back to the forecast origin. Every column above
    # describes the state as of one window before its row, so shifting by
    # lead - 1 makes it describe the state as of `lead` windows before instead.
    if lead > 1:
        out[LAG_FEATURES] = out[LAG_FEATURES].shift(lead - 1)
    return out


def _add_lag_features(df: pd.DataFrame, lead: int = 1) -> pd.DataFrame:
    """
    Add lag and rolling features for a single species' time series.
    Lag features are computed in log-space on the full 3 h time grid
    (:func:`lag_block_on_grid`), so a station outage leaves a gap in time
    rather than silently shortening "24 h ago" to "8 rows ago". Expects a
    'value' column and a 'date' column.

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
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    block = lag_block_on_grid(pd.Series(df["value"].to_numpy(dtype=float), index=dates), lead)
    for col in LAG_FEATURES:
        df[col] = block[col].reindex(dates).to_numpy()
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


def _add_phenology_features(
    df: pd.DataFrame,
    species: str,
    onset_by_day: pd.Series | None = None,
    readiness: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add phenology-derived features: days since flowering onset, onset anomaly.

    ``days_since_typical_onset`` measures against a *per-year* onset estimate
    rather than one constant for all time. Until this year's warmth reaches the
    threshold the estimate is the median of previous seasons; from the crossing
    onwards it is the crossing day itself. That switch is the whole point — it
    is how the model learns the season is running early or late — and because
    it only ever looks backwards it is equally computable while training and
    while forecasting.

    ``onset_anomaly`` and ``gdd_above_threshold`` are thermal readiness under
    the species' selected forcing rule (src/onset.py), against that rule's
    walk-forward threshold: signed and normalised, and the positive excess.

    *onset_by_day* and *readiness* let a caller supply both from the real
    history; the forecaster must, because it builds features on a weather-only
    frame that carries no measurements to derive them from.
    """
    df = df.copy()
    if onset_by_day is None:
        onset_by_day = onset_doy_by_day(df, species)
    if readiness is None:
        readiness = readiness_by_day(df, species)

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
        df["gdd_above_threshold"] = 0.0
        return df

    df["days_since_typical_onset"] = (doys - onset).clip(lower=-60).astype(float)

    anomaly, above = readiness_features(readiness, pd.DatetimeIndex(days))
    df["onset_anomaly"] = anomaly
    df["gdd_above_threshold"] = above
    return df


def readiness_features(
    readiness: pd.DataFrame, days: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    """``onset_anomaly`` and ``gdd_above_threshold`` for *days* from a readiness table.

    Both are 0 where there is no threshold yet (too few seasons) — "average",
    the same neutral value the trainer's NaN fill would give.
    """
    if readiness.empty:
        zeros = np.zeros(len(days))
        return zeros, zeros.copy()
    table = readiness.reindex(days)
    forcing = table["forcing"].to_numpy(dtype=float)
    threshold = table["threshold"].to_numpy(dtype=float)
    ok = np.isfinite(forcing) & np.isfinite(threshold) & (threshold > 0)
    anomaly = np.where(ok, (forcing - threshold) / np.where(ok, threshold, 1.0), 0.0)
    above = np.where(ok, np.clip(forcing - threshold, 0.0, None), 0.0)
    return anomaly.astype(float), above.astype(float)


def _add_load_features(
    df: pd.DataFrame, species: str, totals: dict[int, float] | None = None
) -> pd.DataFrame:
    """Add the interannual load features (see :mod:`src.season_load`).

    *totals* lets a caller supply the completed-season totals from the real
    history when *df* is a weather-only frame; by default they come from the
    measurements in *df* itself, which for a species frame is the history.
    """
    from .season_load import load_features_for_years, season_totals, season_year

    df = df.copy()
    if totals is None:
        totals = season_totals(df, species)
    years = season_year(species, pd.to_datetime(df["date"]))
    per_year = load_features_for_years(totals, np.unique(years))
    for name in LOAD_FEATURES:
        df[name] = [per_year[int(y)][name] for y in years]
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


def _add_weather_derived_features(df: pd.DataFrame, species: str = "") -> pd.DataFrame:
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
    - Burst potential: consecutive warm windows (#2); the threshold-gated
      readiness features live with the phenology block since they read the
      species' selected forcing rule
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
    activation_temp = SPECIES_ACTIVATION_TEMP.get(species, _DEFAULT_ACTIVATION_TEMP)

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


# --- Onset ramp weighting (B.6) ------------------------------------------------
#
# In a heavy year the forecast predicts a tenth to a third of what arrives for
# the first two weeks after onset, and only converges once the lag block has
# filled with big numbers. Those rows are rare — two weeks a year against a
# 7-day lag window that is zero by definition — so the quantile regressor
# treats them as the tail they are and predicts low. This weights them up.
#
# The weight comes from the training year's *measured* onset, which is label
# information, not a feature: nothing the forecaster reads changes, only how
# much a training row counts.
RAMP_DAYS = 14
RAMP_BOOST = 3.0


def onset_ramp_flag(history: pd.DataFrame, species: str, dates: pd.Series) -> np.ndarray:
    """1.0 for rows within RAMP_DAYS after the year's measured onset, else 0.0."""
    days = pd.to_datetime(dates).dt.normalize()
    flag = np.zeros(len(dates), dtype=float)
    for year, doy in observed_onsets(history, species).items():
        onset = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        rel = (days - onset).dt.days.to_numpy()
        flag[(rel >= 0) & (rel <= RAMP_DAYS)] = 1.0
    return flag


def _add_upwind_features(df: pd.DataFrame, series: pd.Series, lead: int = 1) -> pd.DataFrame:
    """Upwind-station block anchored *lead* windows back, like the lag block.

    Built on the time grid (:func:`src.upwind.upwind_block`), so it is
    attached by date rather than by row. Must run after the lag block: the
    lead feature is the difference against ``pollen_max_8``.
    """
    df = df.copy()
    block = upwind_block(series, df["date"], lead=lead)
    block = with_lead_feature(block, df["pollen_max_8"].to_numpy(dtype=float))
    for col in UPWIND_FEATURES:
        df[col] = block[col].to_numpy()
    return df


def finalize_features(X: pd.DataFrame) -> pd.DataFrame:
    """The last step before a feature frame reaches XGBoost, in training and
    serving alike: every NaN becomes 0.

    One function so the trainer, the rollout benchmark and the forecaster
    cannot disagree about what a missing value looks like to the model.
    """
    return X.fillna(0)


def prepare_training_data(
    history: pd.DataFrame,
    species: str,
    leads: tuple[int, ...] = TRAINING_LEADS,
    with_ramp: bool = False,
    upwind: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series] | tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    """
    Prepare feature matrix X, target y (log-transformed), and raw_values
    (original scale) for a single species. With *with_ramp*, also the
    onset-ramp flag per row (see :func:`onset_ramp_flag`). *upwind* is the
    long-format upwind-station frame (see :mod:`src.upwind`); without it the
    upwind features are NaN.

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
    species_df = _add_load_features(species_df, species)
    upwind_by_window = upwind_series(upwind, species)

    # Everything above is independent of the lead, so it is built once and only
    # the lag block is rebuilt per lead.
    per_lead: list[pd.DataFrame] = []
    for lead in leads:
        frame = _add_lag_features(species_df, lead=lead)
        frame = _add_upwind_features(frame, upwind_by_window, lead=lead)
        frame = frame.dropna(subset=LAG_FEATURES)
        if not frame.empty:
            per_lead.append(frame)

    if not per_lead:
        empty = (pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float))
        return (*empty, np.zeros(0)) if with_ramp else empty

    combined = pd.concat(per_lead, ignore_index=True)

    X = combined[FEATURE_COLS].copy()
    raw_values = combined["value"].reset_index(drop=True)
    y = pd.Series(log_transform(combined["value"]), index=combined.index)

    X = finalize_features(X)

    if with_ramp:
        return X, y, raw_values, onset_ramp_flag(species_df, species, combined["date"])
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
        # Blend: scale the *log-space* regression output by the activation
        # probability, suppressing entirely below 0.3. In real space that is
        # a power transform, count^p with p in [0.5, 1] — a shrinkage of
        # uncertain windows toward zero, not a hurdle model. Kept on the
        # benchmark's word (E.3, same six folds): this form scores MAE 6.1 /
        # level 74.5% / bias -1.7; the hurdle form (scale after expm1, so
        # the count itself is multiplied by the probability) 6.6 / 73.5% /
        # -0.5; and a plain threshold (zero below 0.5, the regressor's value
        # above it) 6.9 / 72.0% / +0.1. The shrinkage is the cheapest bias
        # of the three where the level is concerned; the hurdle form buys
        # bias and hazel/alder onset timing at the price of level accuracy
        # and 52 onset false starts against 8.
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
    ramp: np.ndarray | None = None,
) -> TwoStageModel | None:
    """
    Train a multi-stage model for one species.

    Stage 1: XGBClassifier — is pollen > 0 today?
    Stage 2: XGBRegressor  — how much? (quantile regression on all data)
    Stage 3: XGBRegressor  — extreme regressor (fitted only to high-pollen
             samples), gated by an XGBClassifier for P(value > threshold)

    Peak emphasis is one mechanism, the raised quantile target (#4, E.2),
    with species-specific hyperparameters (#5). Rows in the first RAMP_DAYS
    after the year's measured onset weigh (1 + RAMP_BOOST) times more in the
    regressor and the extreme gate (B.6); that weight is label-driven, not
    value-driven, so it does not shift the quantile.
    """
    ramp_weight = 1.0 + RAMP_BOOST * ramp if ramp is not None else None
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

    # One peak-emphasis mechanism (E.2): the raised quantile. The regressor
    # used to be weighted by 1 + sqrt(value) with tier bonuses on top of the
    # quantile, and weighting by the target inside a quantile loss shifts the
    # effective quantile above the nominal one, so the two compounded. A/B on
    # the same six folds (E.1 baseline MAE 6.9 / level 73.3% / bias -0.1):
    # quantile alone 6.1 / 74.5% / -1.7, weights alone (median regression)
    # 6.1 / 74.0% / -3.9, and raising the quantile by 0.03 on top 6.5 / 74.4%
    # / -0.8. Only the onset-ramp weight (B.6, label-driven) remains.
    regressor.fit(X, y, sample_weight=ramp_weight)

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
            extreme_classifier.fit(X, extreme_mask.astype(int), sample_weight=ramp_weight)

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
    from .onset import describe_forcing_rule, select_forcing_rule

    print("\n  Onset calibration (measured from history):")
    print(f"    {'Species':<12} {'onset DOY':>10} {'seasons':>8}"
          f"   {'projection rule':<22} {'threshold':>10} {'LOO':>6}")
    print(f"    {'-'*12} {'-'*10} {'-'*8}   {'-'*22} {'-'*10} {'-'*6}")
    for species in ALL_SPECIES:
        seasons = len(observed_onsets(history, species))
        onset = typical_onset_doy(species, history)
        rule, threshold, loo = select_forcing_rule(history, species)
        loo_text = f"{loo:5.1f}d" if loo != float("inf") else "     -"
        thr_text = f"{threshold:10.1f}" if threshold is not None else f"{'-':>10}"
        measured = "" if seasons else "  (baseline)"
        print(f"    {species:<12} {onset:>10.0f} {seasons:>8}"
              f"   {describe_forcing_rule(rule):<22} {thr_text} {loo_text}{measured}")
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
    "load": LOAD_FEATURES,
    "cams": CAMS_FEATURES,
    "intraday": INTRADAY_FEATURES,
    "lag": LAG_FEATURES,
    "upwind": UPWIND_FEATURES,
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
# 0.0 for soil moisture when the source column is missing. Everything else in
# the feature list is legitimately zero at times — rain, sunshine, is_day,
# ndvi_delta on a flat day, the lags, and soil *temperature*, which the
# backfilled archive reports as exactly 0.0 °C on a few frozen winter windows.
ZERO_MEANS_MISSING = frozenset(
    [
        "ndvi",
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


def train_all(history: pd.DataFrame, upwind: pd.DataFrame | None = None) -> dict[str, TwoStageModel]:
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
        X, y, raw_values, ramp = prepare_training_data(
            history, species, with_ramp=True, upwind=upwind
        )
        if len(X) < 14:
            print(f"  {species}: skipped (only {len(X)} training samples, need >= 14)")
            continue

        model = train_species_model(X, y, raw_values=raw_values, species=species, ramp=ramp)
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
