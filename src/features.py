"""Row-wise feature assembly, shared by the forecaster and its backtest.

The trainer builds features in vectorised batches over a whole history frame
(``trainer._add_*_features``); the forecaster has to build them one
(window, species) at a time, because each prediction feeds the next window's
lag inputs. That is two implementations of one definition, and when they drift
the model is served inputs it never saw while training — silently, because
nothing about a forecast with skewed features *looks* wrong.

This module owns the row-wise half of that pair so the forecaster and any
backtest of it run the same code instead of two copies.
``tests/test_feature_parity.py`` pins it against the trainer's batch path.

Two lag features need state that outlives a single row, and both are seeded
from the real history rather than from the forecast window alone:

* ``days_since_active`` counts windows back to the last non-zero measurement.
  Off-season that runs into the hundreds, so seeding it from a 56-window tail
  would cap it far below anything training ever showed the model.
* ``pollen_morning_avg`` averages the earlier windows of the *calendar day*. A
  forecast that starts at midday has those windows in history, not ahead of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .types import (
    ALL_SPECIES,
    WEATHER_FEATURES,
    WEATHER_DERIVED_FEATURES,
    NDVI_FEATURES,
    INTRADAY_FEATURES,
    is_season_active,
)
from .onset import (
    gdd_threshold_by_year,
    gdd_threshold_for_year,
    onset_doy_by_day,
    onset_doy_lookup,
    _static_onset_doy,
)
from .cams import cams_value

# Longest lag the model asks for (7 days of 3h windows). The rolling state
# keeps exactly this many past values.
LAG_WINDOW = 56

# What days_since_active reports before the species has ever been seen active.
# Matches the trainer's fill for the same situation.
NEVER_ACTIVE = 999.0


def _f(value: object) -> float:
    """Coerce a lookup result to float, treating a missing key as 0.

    NaN is deliberately preserved: XGBoost treats it as a missing value, and
    the trainer's batch path leaves NaN in place too.
    """
    return float(value if value is not None else 0.0)  # type: ignore[arg-type]


# --- Rolling lag state ------------------------------------------------------


@dataclass
class LagState:
    """The autoregressive state one species carries between windows.

    ``log_vals`` holds the last :data:`LAG_WINDOW` values in log space, oldest
    first; the final entry is always the window immediately before the one
    being predicted.
    """

    log_vals: list[float] = field(default_factory=list)
    morning: list[float] = field(default_factory=list)
    days_since_active: float = NEVER_ACTIVE
    day: pd.Timestamp | None = None
    # Before the species has ever been seen active the counter is a sentinel,
    # not a distance, so it must not tick upward — the trainer's batch path
    # holds those rows at NEVER_ACTIVE too.
    seen_active: bool = False

    @classmethod
    def from_history(
        cls, history: pd.DataFrame, species: str, origin: pd.Timestamp
    ) -> "LagState":
        """Seed the state from every measurement strictly before *origin*."""
        if history.empty:
            return cls(log_vals=[0.0] * LAG_WINDOW)

        sp = history[history["species"] == species]
        sp = sp[pd.to_datetime(sp["date"]) < pd.Timestamp(origin)]
        sp = sp.sort_values("date")
        if sp.empty:
            return cls(log_vals=[0.0] * LAG_WINDOW)

        values = sp["value"].to_numpy(dtype=float)
        log_vals = list(np.log1p(values[-LAG_WINDOW:]))
        while len(log_vals) < LAG_WINDOW:
            log_vals.insert(0, 0.0)

        # days_since_active over the *whole* history, not just the tail.
        active = np.flatnonzero(values > 0)
        seen_active = active.size > 0
        dsa = float(len(values) - 1 - active[-1]) if seen_active else NEVER_ACTIVE

        # Earlier windows of the origin's own calendar day are already measured.
        day = pd.Timestamp(origin).normalize()
        same_day = sp[pd.to_datetime(sp["date"]).dt.normalize() == day]
        morning = list(np.log1p(same_day["value"].to_numpy(dtype=float)))

        return cls(
            log_vals=log_vals,
            morning=morning,
            days_since_active=dsa,
            day=day,
            seen_active=seen_active,
        )

    def begin_window(self, dt: pd.Timestamp) -> None:
        """Reset the intra-day accumulator when the calendar day turns over."""
        day = pd.Timestamp(dt).normalize()
        if self.day is None or day != self.day:
            self.day = day
            self.morning = []

    def record(self, log_value: float) -> None:
        """Fold a window's value (predicted or observed) into the state."""
        self.log_vals.append(log_value)
        if len(self.log_vals) > LAG_WINDOW:
            del self.log_vals[: len(self.log_vals) - LAG_WINDOW]
        self.morning.append(log_value)
        if log_value > 0:
            self.days_since_active = 0.0
            self.seen_active = True
        elif self.seen_active:
            self.days_since_active += 1.0

    def lag_features(self) -> dict[str, float]:
        """The 13 lag features implied by the current state."""
        vals = self.log_vals
        if not vals:
            vals = [0.0]

        def at(n: int) -> float:
            return float(vals[-n]) if len(vals) >= n else float(vals[0])

        return {
            "pollen_lag_1": at(1),
            "pollen_lag_2": at(2),
            "pollen_lag_3": at(3),
            "pollen_lag_8": at(8),
            "pollen_lag_16": at(16),
            "pollen_lag_24": at(24),
            "pollen_lag_56": at(56),
            "pollen_rolling_8": float(np.mean(vals[-8:])),
            "pollen_rolling_56": float(np.mean(vals[-56:])),
            "pollen_max_8": float(np.max(vals[-8:])),
            "pollen_max_56": float(np.max(vals[-56:])),
            "pollen_morning_avg": float(np.mean(self.morning)) if self.morning else 0.0,
            "days_since_active": float(self.days_since_active),
        }


# --- Calendar ---------------------------------------------------------------


def calendar_features(dt: pd.Timestamp) -> dict[str, float]:
    """Calendar and time-of-day features for a single window."""
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


# --- Context ----------------------------------------------------------------


@dataclass
class FeatureContext:
    """Everything that can be precomputed once for a whole forecast run.

    All the tables are indexed by window start except ``ndvi``, which is daily.
    """

    weather: pd.DataFrame
    weather_derived: dict[str, pd.DataFrame]
    ndvi: pd.DataFrame
    intraday: pd.DataFrame
    cams: pd.DataFrame
    onset_by_species: dict[str, pd.Series]
    gdd_thresholds: dict[str, dict[int, float]]
    onset_fallback: dict[str, float]


def combined_weather(history: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """History's weather columns followed by the forecast's, de-duplicated.

    The rolling and streak features reach back days, so they cannot be derived
    from the forecast window alone. Where the two overlap the forecast wins,
    which is what the collector would have written anyway.
    """
    cols = [c for c in WEATHER_FEATURES if c in history.columns]
    hist_weather = (
        history.groupby("date")[cols].first()
        if not history.empty and cols
        else pd.DataFrame(columns=WEATHER_FEATURES)
    )
    combined = pd.concat([hist_weather, weather])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def intraday_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Intra-day relative features, mirroring ``trainer._add_intraday_features``.

    Derived on the *combined* frame rather than the forecast window, so the
    first forecast window has a real predecessor to difference against and the
    first (partial) day has its true daily maximum.
    """
    temp = weather["temperature_mean"].fillna(0)
    days = pd.to_datetime(weather.index).normalize()
    daily_max = temp.groupby(days).transform("max")

    out = pd.DataFrame(index=weather.index)
    out["temp_vs_daily_max"] = np.where(daily_max > 0, temp / daily_max, 0.0)
    out["precip_in_prior_window"] = (
        (weather["precipitation_sum"].fillna(0).shift(1) > 0.1).astype(float).fillna(0)
    )
    out["temp_rate_of_change"] = temp.diff(1).fillna(0)
    return out


def ndvi_from_history(history: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily NDVI table taken from columns the collector already stored."""
    if history.empty or not all(c in history.columns for c in NDVI_FEATURES):
        return pd.DataFrame(0.0, index=days, columns=NDVI_FEATURES)
    hist_days = pd.to_datetime(history["date"]).dt.normalize()
    table = history.groupby(hist_days)[NDVI_FEATURES].first().sort_index()
    return table.reindex(days).ffill().fillna(0.0)


def build_context(
    history: pd.DataFrame,
    weather: pd.DataFrame,
    ndvi: pd.DataFrame,
    cams: pd.DataFrame | None = None,
    species: list[str] | None = None,
    parallel: bool = True,
) -> FeatureContext:
    """Precompute every per-run lookup table the row builder needs.

    *weather* holds the windows to be predicted; *history* supplies the past
    the rolling features need, plus the measurements the onset calibration is
    derived from.
    """
    species_list = list(species) if species else list(ALL_SPECIES)
    from .trainer import _add_weather_derived_features

    combined = combined_weather(history, weather)
    base = combined.reset_index().rename(columns={"index": "date"})
    base["species"] = "__dummy__"
    base["value"] = 0.0

    # Onset estimates and GDD thresholds come from the real measurements: the
    # weather-only frame above carries a dummy species and cannot derive them.
    onset_by_species = {sp: onset_doy_by_day(history, sp) for sp in species_list}
    gdd_thresholds = {sp: gdd_threshold_by_year(history, sp) for sp in species_list}
    onset_fallback = {sp: _static_onset_doy(sp) for sp in species_list}

    def derive(sp: str) -> tuple[str, pd.DataFrame]:
        frame = _add_weather_derived_features(
            base.copy(), sp, gdd_thresholds=gdd_thresholds[sp]
        )
        return sp, frame.set_index("date")

    if parallel and len(species_list) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(species_list)) as pool:
            futures = [pool.submit(derive, sp) for sp in species_list]
            weather_derived = dict(f.result() for f in futures)
    else:
        weather_derived = dict(derive(sp) for sp in species_list)

    return FeatureContext(
        weather=combined,
        weather_derived=weather_derived,
        ndvi=ndvi,
        intraday=intraday_features(combined),
        cams=cams if cams is not None else pd.DataFrame(),
        onset_by_species=onset_by_species,
        gdd_thresholds=gdd_thresholds,
        onset_fallback=onset_fallback,
    )


# --- Row assembly -----------------------------------------------------------


def _row_from(table: pd.DataFrame, key: object, columns: list[str]) -> dict[str, float]:
    """Read *columns* out of *table* at *key*, defaulting to 0 when absent."""
    if key not in table.index:
        return {c: 0.0 for c in columns}
    row = table.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {c: _f(row.get(c, 0.0)) for c in columns}


def static_features(ctx: FeatureContext, species: str, dt: pd.Timestamp) -> dict[str, float]:
    """Every feature for one (window, species) *except* the lag group.

    Split out because none of these depend on the autoregressive state, so a
    backtest that rolls many forecast origins through the same windows can
    build them once instead of once per origin per step.

    The order the groups are filled in matters in one place: ``onset_anomaly``
    reads the ``gdd`` produced by the weather-derived group.
    """
    features: dict[str, float] = {}

    weather_row = ctx.weather.loc[dt] if dt in ctx.weather.index else None
    if weather_row is None:
        features.update({f: 0.0 for f in WEATHER_FEATURES})
    else:
        if isinstance(weather_row, pd.DataFrame):
            weather_row = weather_row.iloc[0]
        features.update({f: _f(weather_row.get(f, 0.0)) for f in WEATHER_FEATURES})

    features.update(calendar_features(dt))
    features["season_active"] = 1.0 if is_season_active(species, dt.month) else 0.0
    features.update(
        _row_from(ctx.weather_derived[species], dt, WEATHER_DERIVED_FEATURES)
    )
    features.update(_row_from(ctx.ndvi, dt.normalize(), NDVI_FEATURES))

    # Phenology: this year's causal onset estimate plus the thermal early/late
    # signal, resolved exactly as the trainer resolves them.
    onset_doy = onset_doy_lookup(
        ctx.onset_by_species[species], dt, ctx.onset_fallback[species]
    )
    if onset_doy == onset_doy:  # not NaN
        from .trainer import onset_anomaly_from_gdd

        features["days_since_typical_onset"] = float(
            max(-60.0, dt.day_of_year - onset_doy)
        )
        features["onset_anomaly"] = float(
            onset_anomaly_from_gdd(
                features.get("gdd", 0.0),
                species,
                gdd_threshold_for_year(ctx.gdd_thresholds[species], dt.year, species),
            )
        )
    else:
        features["days_since_typical_onset"] = 0.0
        features["onset_anomaly"] = 0.0

    features["cams_pollen"] = cams_value(ctx.cams, dt, species)
    features.update(_row_from(ctx.intraday, dt, INTRADAY_FEATURES))
    return features


def build_feature_row(
    ctx: FeatureContext, species: str, dt: pd.Timestamp, lag: LagState
) -> dict[str, float]:
    """The full feature vector for one (window, species)."""
    return {**static_features(ctx, species, dt), **lag.lag_features()}


def static_feature_frame(
    ctx: FeatureContext, species: str, windows: pd.DatetimeIndex
) -> pd.DataFrame:
    """:func:`static_features` for many windows at once, indexed by window."""
    return pd.DataFrame(
        [static_features(ctx, species, pd.Timestamp(dt)) for dt in windows],
        index=pd.DatetimeIndex(windows),
    )
