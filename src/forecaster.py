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
    STALE_AFTER_WINDOWS,
    WINDOWS_PER_DAY,
    _DEFAULT_THRESHOLDS,
    season_gate_active,
    value_to_level,
    SpeciesForecast,
    WindowForecast,
    DayForecast,
    ForecastOutput,
    ObservationStatus,
    RunStatus,
)
from .clock import local_now
from .weather import fetch_weather_forecast
from .trainer import TwoStageModel, finalize_features, load_models, inv_log_transform
from .confidence import confidence_for, load_table, value_interval
from .features import FeatureContext, LagState, build_context, build_feature_row
from .cams import fetch_cams_forecast

WINDOW = pd.Timedelta(hours=3)

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


def blend_day_with_dwd(day_mean: float, species: str, dwd_num: float) -> float | None:
    """The daily mean to aim for after nudging one level toward DWD's index.

    Levels are daily means (see :func:`src.types.value_to_level`), so the
    comparison with DWD's daily index is made on the day's mean, not on a
    single window. Conservative: moves at most one level step (never
    fabricates pollen far outside the model's range) and blends 50% (never
    fully overrides the model). Returns None when the day already sits at
    DWD's level.
    """
    our_num = _LEVEL_TO_NUM[value_to_level(day_mean, species).value]
    target_round = int(round(dwd_num))
    if target_round == our_num:
        return None
    step = 1 if target_round > our_num else -1
    target_num = max(0, min(3, our_num + step))
    target_value = _level_band_midpoint(species, target_num)
    return max(0.0, day_mean * 0.5 + target_value * 0.5)


def apply_day_blend(values: dict[pd.Timestamp, float], target_mean: float) -> dict[pd.Timestamp, float]:
    """Scale a day's predicted windows so their mean moves to *target_mean*.

    Keeps the diurnal shape; a day the model has at zero gets the target
    flat across its windows, since there is no shape to keep.
    """
    if not values:
        return values
    current = float(np.mean(list(values.values())))
    if current <= 0:
        return {dt: target_mean for dt in values}
    factor = target_mean / current
    return {dt: v * factor for dt, v in values.items()}


def _no_model_prediction(lag: LagState) -> float:
    """Last known value with a flat decay, for a species with no model."""
    return float(np.expm1(lag.lag_features()["pollen_lag_1"])) * 0.8


def predict_window(
    model: TwoStageModel,
    ctx: FeatureContext,
    species: str,
    dt: pd.Timestamp,
    lag: LagState,
    lead: int,
) -> float:
    """Model prediction for one (window, species), in log space."""
    features = build_feature_row(ctx, species, dt, lag, lead)
    x_features = finalize_features(pd.DataFrame([features])[FEATURE_COLS])
    return max(0.0, float(model.predict(x_features)[0]))


def _fetch_ndvi(days: pd.DatetimeIndex, defaulted: dict[str, str]) -> pd.DataFrame:
    """Daily NDVI for the forecast dates; zeros when the fetch fails.

    A failure is recorded in *defaulted* (D.3) rather than only printed.
    """
    try:
        from .ndvi import ndvi_features

        table = ndvi_features(days)
    except Exception as exc:
        print(f"  NDVI fetch failed ({exc}), using defaults")
        defaulted["ndvi"] = f"fetch failed ({exc}); zeros"
        return pd.DataFrame(
            {"ndvi": 0.0, "evi": 0.0, "ndvi_delta": 0.0}, index=days
        )
    if table.empty or not (table["ndvi"] != 0).any():
        defaulted["ndvi"] = "no composites; zeros"
    return table


def _fetch_dwd_levels(defaulted: dict[str, str]) -> dict[tuple[object, str], float]:
    """DWD categorical levels keyed by (date, species); empty when unavailable.

    DWD only covers today/tomorrow/day-after, so most forecast windows are
    untouched. Fail-open: an empty lookup leaves predictions unchanged, and
    is recorded in *defaulted* (D.3).
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
        else:
            defaulted["dwd"] = "no levels for the region; blend skipped"
    except Exception as exc:
        print(f"  DWD forecast unavailable ({exc}); skipping DWD blend")
        defaulted["dwd"] = f"unavailable ({exc}); blend skipped"
    return levels


def _soil_defaulted(weather: pd.DataFrame) -> bool:
    """True when the forecast's soil variables are the all-zero fallback."""
    cols = [c for c in ("soil_temperature_mean", "soil_moisture_mean") if c in weather.columns]
    if not cols or weather.empty:
        return True
    return not (weather[cols].fillna(0.0) != 0.0).any().any()


def windows_since(last: pd.Timestamp, now: pd.Timestamp) -> int:
    """Complete 3 h windows since the window that starts at *last*, with no measurement.

    The window at *last* itself and the one in progress do not count: a
    measurement for 06:00 seen at 13:30 is one window old (09:00 passed
    unmeasured; 12:00 is still running).
    """
    return max(0, int((pd.Timestamp(now) - pd.Timestamp(last)) // WINDOW) - 1)


def observation_status(
    history: pd.DataFrame, now: pd.Timestamp
) -> tuple[dict[str, ObservationStatus], dict[str, pd.Timestamp]]:
    """Per-species newest measurement and its age at *now* (D.2).

    Returns the status per species and the newest measurement's timestamp per
    species (absent when a species has never been measured).
    """
    last_by_species: dict[str, pd.Timestamp] = {}
    if not history.empty:
        newest = history.groupby("species")["date"].max()
        last_by_species = {str(sp): pd.Timestamp(ts) for sp, ts in newest.items()}

    status: dict[str, ObservationStatus] = {}
    for species in ALL_SPECIES:
        last = last_by_species.get(species)
        if last is None:
            status[species] = ObservationStatus(last=None, age_windows=None, stale=True)
            continue
        age = windows_since(last, now)
        status[species] = ObservationStatus(
            last=last.isoformat(), age_windows=age, stale=age >= STALE_AFTER_WINDOWS
        )
    return status, last_by_species


def worst_observation(status: dict[str, ObservationStatus]) -> ObservationStatus:
    """The oldest newest-measurement over all species: the run's headline."""
    if not status:
        return ObservationStatus(last=None, age_windows=None, stale=True)
    return max(
        status.values(),
        key=lambda st: (st.age_windows if st.age_windows is not None else float("inf")),
    )


def generate_forecast(
    history: pd.DataFrame,
    models: dict[str, TwoStageModel] | None = None,
    upwind: pd.DataFrame | None = None,
) -> ForecastOutput:
    """
    Generate a multi-day pollen forecast at 3-hour window resolution.

    Every window is predicted directly from the measurements available at the
    forecast origin, with ``lead_windows`` saying how far ahead it is. Nothing
    is fed back: the forecast is 40 independent predictions off one shared block
    of real data, not a chain of 40 predictions each standing on the last.

    All internal lag/prediction values are in log-space (log1p).
    Final output values are converted back to original pollen-count scale.

    Args:
        history: Full historical data (date, species, value, weather features...).
        models: Pre-loaded models. If None, loads from disk.
        upwind: Upwind-station measurements (src/upwind.py); optional.
    """
    if models is None:
        models = load_models()
        print(f"Loaded {len(models)} species models")

    # Which input groups fell back to a default this run (D.3), by name.
    defaulted: dict[str, str] = {}

    weather = fetch_weather_forecast(FORECAST_DAYS)
    print(f"Weather forecast: {len(weather)} windows ({FORECAST_DAYS} days)")
    if _soil_defaulted(weather):
        defaulted["weather_soil"] = "soil variables not in the weather response; zeros"

    forecast_days_index = pd.DatetimeIndex(weather.index).normalize().unique()
    ctx = build_context(
        history,
        weather,
        ndvi=_fetch_ndvi(forecast_days_index, defaulted),
        cams=fetch_cams_forecast(FORECAST_DAYS),
    )
    dwd_levels = _fetch_dwd_levels(defaulted)

    calibration = load_table()
    if calibration is None:
        print("  No calibration table; publishing fallback confidence")
        defaulted["confidence"] = "no calibration table; fallback rates"
    else:
        print(f"Confidence calibrated {calibration['generated'][:10]}: "
              f"exact {calibration['overall']['exact']:.2f}, "
              f"within one level {calibration['overall']['within_one']:.2f}")

    # --- Real-time observation assimilation ---
    # Windows that already have a measurement emit it directly, and the origin
    # for a species is the window after its newest measurement. Everything
    # measured before that origin feeds the lag block, so the forecast starts
    # from as much real data as exists — and ``lead_windows`` counts from
    # there, so when the station has been silent for two days the first
    # forecast window is honestly a lead of two days plus one, not lead 1
    # over a block that quietly ended two days ago (D.2).
    observed: dict[tuple[pd.Timestamp, str], float] = {}
    if not history.empty:
        recent = history[history["date"] >= pd.Timestamp(weather.index.min())]
        for _, row in recent.iterrows():
            observed[(pd.Timestamp(row["date"]), str(row["species"]))] = float(row["value"])

    windows = [pd.Timestamp(str(w)) for w in weather.index]

    def first_unobserved(species: str) -> pd.Timestamp:
        for window in windows:
            if (window, species) not in observed:
                return window
        return windows[-1] + WINDOW

    obs_status, last_by_species = observation_status(history, local_now())

    def origin_for(species: str) -> pd.Timestamp:
        origin = first_unobserved(species)
        last = last_by_species.get(species)
        if last is not None:
            origin = min(origin, last + WINDOW)
        return origin

    origins = {sp: origin_for(sp) for sp in ALL_SPECIES}
    lags = {
        sp: LagState.from_history(history, sp, origins[sp], upwind=upwind) for sp in ALL_SPECIES
    }

    n_obs_windows = len({dt for dt, _ in observed})
    if n_obs_windows > 0:
        print(f"Real-time assimilation: {n_obs_windows} observed windows will use actual data")

    no_upwind = [
        sp for sp in ALL_SPECIES if np.isnan(lags[sp].upwind.get("upwind_max_56", np.nan))
    ]
    if len(no_upwind) == len(ALL_SPECIES):
        defaulted["upwind"] = "no station readings in the last 7 days; zeros"
    elif no_upwind:
        defaulted["upwind"] = (
            "no station readings in the last 7 days for " + ", ".join(no_upwind) + "; zeros"
        )
    no_model = [sp for sp in ALL_SPECIES if sp not in models]
    if no_model:
        defaulted["model"] = "no trained model, persistence for " + ", ".join(no_model)

    status = RunStatus(
        observations=worst_observation(obs_status), species=obs_status, defaulted=defaulted
    )
    print(f"Inputs: {status.describe()}")

    # Pass 1: one value per (window, species) — the observation where there
    # is one, the model otherwise, the season gate last.
    values: dict[tuple[pd.Timestamp, str], float] = {}
    for dt in windows:
        for species in ALL_SPECIES:
            if (dt, species) in observed:
                prediction = observed[(dt, species)]
            else:
                lead = max(1, int((dt - origins[species]) / WINDOW) + 1)
                if species in models:
                    pred_log = predict_window(
                        models[species], ctx, species, dt, lags[species], lead
                    )
                    prediction = float(inv_log_transform(np.array([pred_log]))[0])
                else:
                    prediction = _no_model_prediction(lags[species])
            # Force to zero only outside the *widened* season window (core ±
            # shoulder), so early-onset events are no longer structurally zeroed.
            if not season_gate_active(species, dt.month):
                prediction = 0.0
            values[(dt, species)] = prediction

    # Pass 2: the DWD blend, on the days DWD covers. Levels are daily means,
    # so the day's mean is what is set against DWD's index, and the nudge is
    # spread over the day's *predicted* windows; observations stay as they
    # are and gated days are left at zero.
    by_day: dict[tuple[object, str], list[pd.Timestamp]] = {}
    for dt in windows:
        for species in ALL_SPECIES:
            by_day.setdefault((dt.date(), species), []).append(dt)
    for (day, species), dts in by_day.items():
        dwd_num = dwd_levels.get((day, species))
        if dwd_num is None or not season_gate_active(species, dts[0].month):
            continue
        predicted = {dt: values[(dt, species)] for dt in dts if (dt, species) not in observed}
        if not predicted:
            continue
        day_mean = float(np.mean([values[(dt, species)] for dt in dts]))
        target = blend_day_with_dwd(day_mean, species, dwd_num)
        if target is None:
            continue
        for dt, v in apply_day_blend(predicted, target).items():
            values[(dt, species)] = v

    # Pass 3: the level of a window is the level of its day's mean.
    day_mean_of: dict[tuple[object, str], float] = {
        key: float(np.mean([values[(dt, species)] for dt in dts]))
        for (key, dts) in by_day.items()
        for species in [key[1]]
    }

    window_results: list[tuple[str, WindowForecast]] = []

    for dt in windows:
        date_str = dt.strftime("%Y-%m-%d")
        window_species: list[SpeciesForecast] = []

        for species in ALL_SPECIES:
            has_model = species in models
            has_observation = (dt, species) in observed
            prediction = values[(dt, species)]

            day_mean = day_mean_of[(dt.date(), species)]
            level = value_to_level(day_mean, species).value
            # The horizon is counted from the newest measurement, as the
            # calibration rollout counts it: eight windows per day from the
            # origin. A stale run therefore reads a longer horizon's rate.
            lead = max(1, int((dt - origins[species]) / WINDOW) + 1)
            horizon_day = (lead - 1) // WINDOWS_PER_DAY + 1
            exact, within_one = confidence_for(
                calibration,
                species,
                level,
                horizon_day=horizon_day,
                has_model=has_model,
                observed=has_observation,
                log_day_mean=float(np.log1p(max(0.0, day_mean))),
            )
            if has_observation:
                interval: tuple[float, float] | None = (prediction, prediction)
            else:
                interval = value_interval(calibration, prediction, horizon_day)
            window_species.append(
                SpeciesForecast(
                    name=species,
                    level=level,
                    value=prediction,
                    confidence=exact,
                    confidence_within_one=within_one,
                    value_low=interval[0] if interval else None,
                    value_high=interval[1] if interval else None,
                )
            )

        window_species.sort(key=lambda s: s.value, reverse=True)
        window_species = [s for s in window_species if s.value > 0.5]

        window_results.append((date_str, WindowForecast(
            from_time=dt.strftime("%H:%M"),
            to_time=(dt + WINDOW).strftime("%H:%M"),
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
        status=status,
    )
