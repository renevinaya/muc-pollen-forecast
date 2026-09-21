"""
Measured forecast confidence.

The confidence attached to each species in ``forecast.json`` used to be
``0.90 - 0.08 * day``, clipped, with a +0.05 bonus for assimilated windows.
Those numbers were invented, and the rollout benchmark said both the level and
the slope were wrong: on the rows the forecast actually emits (value > 0.5) the
level was exactly right about 35% of the time under the old 3-hour level
definition (65% under the daily-mean one, D.4), and accuracy is nearly flat
across the horizon because the model is direct rather than recursive.

Two figures are published per prediction, because "is this level right" has
two defensible readings and they differ a lot:

* ``confidence`` — P(the emitted level is exactly right).
* ``confidence_within_one`` — P(the truth is within one level of it), which is
  closer to "would a reader have behaved correctly". That runs around 99% on
  daily-mean levels.

Both are **conformal** (D.5): the calibration keeps, per forecast horizon, the
distribution of the rollout's residuals in log space — ``log1p(actual daily
mean) - log1p(predicted daily mean)`` over the emitted day-rows — and a
prediction's confidence is the probability mass of that distribution that
lands inside its level's band. A prediction sitting in the middle of a wide
band gets a high number; one a few grains from a threshold gets a low one.
The same residuals, per window, give the ``value_low``/``value_high``
interval published beside each value (80% central, in the log domain).

That replaced a *flat* table — one rate for everything plus a small
per-horizon offset — which was the best that keying on species, level or
horizon could do. Scored leave-one-fold-out on the 10-day rollout
(day-rows, emitted):

    scheme                              ECE    corr. with being right   Brier
    flat + horizon offset (before)     0.072          -0.072            0.239
    conformal, pooled                  0.039           0.316            0.210
    conformal per horizon day (now)    0.037           0.323            0.209
    conformal per magnitude bin        0.049           0.298            0.215
    conformal per species              0.073           0.225            0.229

The flat table calibrated the average and could not tell which predictions
were more reliable (correlation ~0, for every keying). The conformal number
can, without any new key: what carries the information is *where the
prediction sits relative to the thresholds*, which the point estimate always
had and the lookup table threw away. Finer conditioning (species, magnitude
bin) still calibrates worse held-out, for the same reason as before: the
differences are season- and year-specific. Reliability of the shipped
scheme, leave-one-fold-out: stated 0.27 → right 11%, 0.45 → 46%, 0.65 →
70%, 0.82 → 78%.

The flat rate is kept in the table as the fallback for a prediction without a
residual distribution to consult.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .types import FORECAST_DAYS, SPECIES_THRESHOLDS, _DEFAULT_THRESHOLDS

TABLE_PATH = Path(__file__).parent / "confidence.json"

LEVEL_ORDER = ["none", "low", "moderate", "high", "very_high"]
_LEVEL_INDEX = {level: i for i, level in enumerate(LEVEL_ORDER)}

# Values at or below this are dropped from the forecast, so they are also
# excluded from calibration — a confidence is only ever attached to a row that
# is actually published. Must match the filter in forecaster.generate_forecast.
EMIT_THRESHOLD = 0.5

# Shrinkage strength for thin cells: a cell with N samples is pulled toward the
# level's pooled accuracy with weight K/(N+K). At K=20 a 900-sample cell is
# essentially untouched and a 40-sample one is pulled a third of the way.
SHRINKAGE_K = 20.0

# Bounds on any published confidence. Neither end is ever really justified: 0
# would claim certain failure and 1 certain success.
MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.99

# What to publish when no calibration exists for a species at all.
FALLBACK = {"exact": 0.30, "within_one": 0.85}

# An assimilated window carries a measurement rather than a prediction.
OBSERVED_CONFIDENCE = {"exact": 1.0, "within_one": 1.0}

# Species with no trained model fall back to persistence-with-decay, which the
# benchmark never scores. Published low rather than left implicitly high.
NO_MODEL_SCALE = 0.5

# How far out the calibration rollout looks. The forecast ships FORECAST_DAYS,
# but a forecast built on stale observations *is* a longer-range forecast — a
# window one day ahead of a block that ended two days ago sits at lead three
# days — so the table is measured to twice the shipped horizon and a stale
# run reads the rate of the horizon it really is (D.2).
CALIBRATION_HORIZON_DAYS = 2 * FORECAST_DAYS

# The residual distributions are stored as this many evenly spaced quantiles
# (0, 1, ..., 100%): one percent resolution on the ECDF, and a table that
# stays a few kilobytes.
RESIDUAL_QUANTILES = 101

# Central coverage of the published value interval.
INTERVAL_COVERAGE = 0.8

# A horizon beyond the last one measured has no rate to publish. It gets the
# furthest measured rate marked down by this factor: "never measured" is
# published low, like a species without a model, rather than as the last
# number that happened to exist.
BEYOND_HORIZON_SCALE = 0.5


def _annotate(results: pd.DataFrame) -> pd.DataFrame:
    """Add exact-hit and within-one-level columns."""
    frame = results.copy()
    actual = frame["level_actual"].map(_LEVEL_INDEX)
    predicted = frame["level_predicted"].map(_LEVEL_INDEX)
    frame["exact"] = actual == predicted
    frame["within_one"] = (actual - predicted).abs() <= 1
    return frame


def build_table(results: pd.DataFrame) -> dict[str, Any]:
    """Build the calibration table from rollout benchmark results.

    Only the overall rate and the per-horizon offset are stored, because only
    those generalise — see the module docstring.
    """
    frame = _annotate(results)
    emitted = frame[frame["predicted"] > EMIT_THRESHOLD]
    if emitted.empty:
        raise ValueError("no emitted rows to calibrate on")

    # The overall rate is the shipped horizon's: rows beyond FORECAST_DAYS are
    # only reached by a stale run and enter the table as per-horizon offsets.
    shipped = emitted[emitted["horizon_day"] <= FORECAST_DAYS]
    if shipped.empty:
        shipped = emitted
    overall = {
        "exact": float(shipped["exact"].mean()),
        "within_one": float(shipped["within_one"].mean()),
    }
    horizon_delta = {
        str(int(horizon)): {
            "exact": float(group["exact"].mean()) - overall["exact"],
            "within_one": float(group["within_one"].mean()) - overall["within_one"],
        }
        for horizon, group in emitted.groupby("horizon_day")
    }

    return {
        "generated": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "residuals": residual_tables(results),
        "source": {
            "rows_scored": int(len(frame)),
            "rows_emitted": int(len(emitted)),
            "folds": int(frame["fold"].nunique()) if "fold" in frame else None,
            "origins": int(frame["origin"].nunique()) if "origin" in frame else None,
            "horizon_days": int(frame["horizon_day"].max()),
        },
        "overall": overall,
        "horizon_delta": horizon_delta,
    }


def day_mean_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Rollout rows collapsed to (fold, species, origin, day): the daily means.

    Levels are daily means, so the residual that decides whether a level is
    right is the residual of the day's mean, not of a window. Only complete
    days (all eight windows scored) count.
    """
    frame = results.copy()
    frame["day"] = pd.to_datetime(frame["date"]).dt.normalize()
    keys = [k for k in ("fold", "species", "origin", "day") if k in frame]
    days = frame.groupby(keys, as_index=False).agg(
        predicted=("predicted", "mean"),
        actual=("actual", "mean"),
        horizon_day=("horizon_day", "min"),
        n=("actual", "size"),
    )
    return days[days["n"] == 8].drop(columns="n")


def _quantile_grid(residuals: np.ndarray) -> list[float]:
    grid = np.linspace(0.0, 1.0, RESIDUAL_QUANTILES)
    return [float(v) for v in np.quantile(residuals, grid)]


def residual_tables(results: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    """Per-horizon residual quantiles, for day means and for windows.

    ``day_mean`` feeds the level probabilities, ``window`` the value interval;
    both on the emitted rows, in log space, ``log1p(actual) - log1p(predicted)``.
    """
    out: dict[str, dict[str, list[float]]] = {"day_mean": {}, "window": {}}
    if not {"date", "actual", "predicted", "horizon_day"} <= set(results.columns):
        return out  # level-only results (older benchmark files): flat rates only
    days = day_mean_rows(results)
    days = days[days["predicted"] > EMIT_THRESHOLD]
    for horizon, group in days.groupby("horizon_day"):
        res = np.log1p(group["actual"].to_numpy(float)) - np.log1p(group["predicted"].to_numpy(float))
        if len(res) >= 30:
            out["day_mean"][str(int(horizon))] = _quantile_grid(res)
    windows = results[results["predicted"] > EMIT_THRESHOLD]
    for horizon, group in windows.groupby("horizon_day"):
        res = np.log1p(group["actual"].to_numpy(float)) - np.log1p(group["predicted"].to_numpy(float))
        if len(res) >= 30:
            out["window"][str(int(horizon))] = _quantile_grid(res)
    return out


def _ecdf(quantiles: list[float], x: float) -> float:
    """P(residual <= x) read off the stored quantile grid."""
    if x == float("inf"):
        return 1.0
    if x == float("-inf"):
        return 0.0
    grid = np.linspace(0.0, 1.0, len(quantiles))
    return float(np.interp(x, quantiles, grid, left=0.0, right=1.0))


def _band(species: str, level: str) -> tuple[float, float]:
    """The (low, high] range of daily means that reads as *level*."""
    low_max, mod_max, high_max = SPECIES_THRESHOLDS.get(species, _DEFAULT_THRESHOLDS)
    return {
        "none": (float("-inf"), 0.0),
        "low": (0.0, low_max),
        "moderate": (low_max, mod_max),
        "high": (mod_max, high_max),
        "very_high": (high_max, float("inf")),
    }[level]


def _mass_in(quantiles: list[float], log_pred: float, low: float, high: float) -> float:
    """Residual mass that puts the truth in (low, high], given the log prediction."""
    lo = np.log1p(low) - log_pred if np.isfinite(low) else float("-inf")
    hi = np.log1p(high) - log_pred if np.isfinite(high) else float("inf")
    return max(0.0, _ecdf(quantiles, hi) - _ecdf(quantiles, lo))


def _residuals_for(table: dict[str, Any] | None, kind: str, horizon_day: int) -> tuple[list[float] | None, bool]:
    """The stored residual quantiles for a horizon, and whether it lay beyond the measured ones."""
    if not table:
        return None, False
    stored = table.get("residuals", {}).get(kind, {})
    if not stored:
        return None, False
    measured = sorted(int(k) for k in stored)
    if int(horizon_day) in measured:
        return stored[str(int(horizon_day))], False
    return stored[str(max(measured))], True


def level_probability(
    table: dict[str, Any] | None, species: str, level: str, log_day_mean: float, horizon_day: int
) -> tuple[float, float] | None:
    """Conformal P(level exactly right) and P(within one level) for one day.

    None when the table has no residual distribution to consult; the caller
    then falls back to the flat rate.
    """
    quantiles, beyond = _residuals_for(table, "day_mean", horizon_day)
    if quantiles is None:
        return None
    index = _LEVEL_INDEX[level]
    exact = _mass_in(quantiles, log_day_mean, *_band(species, level))
    low = _band(species, LEVEL_ORDER[max(0, index - 1)])[0]
    high = _band(species, LEVEL_ORDER[min(len(LEVEL_ORDER) - 1, index + 1)])[1]
    within = _mass_in(quantiles, log_day_mean, low, high)
    if beyond:
        exact *= BEYOND_HORIZON_SCALE
        within *= BEYOND_HORIZON_SCALE
    return exact, max(exact, within)


def value_interval(
    table: dict[str, Any] | None, value: float, horizon_day: int, coverage: float = INTERVAL_COVERAGE
) -> tuple[float, float] | None:
    """The central *coverage* interval for one window's value, from the window residuals."""
    quantiles, _ = _residuals_for(table, "window", horizon_day)
    if quantiles is None or value <= 0:
        return None
    grid = np.linspace(0.0, 1.0, len(quantiles))
    tail = (1.0 - coverage) / 2.0
    lo = float(np.interp(tail, grid, quantiles))
    hi = float(np.interp(1.0 - tail, grid, quantiles))
    log_value = float(np.log1p(value))
    return max(0.0, float(np.expm1(log_value + lo))), float(np.expm1(log_value + hi))


def breakdown(results: pd.DataFrame) -> pd.DataFrame:
    """Per-species, per-level accuracy — diagnostics only, never published.

    Kept because it is the evidence for why the table is flat: these numbers
    look informative and are not, so anyone tempted to key confidence on them
    should see them alongside the leave-one-fold-out result above.
    """
    frame = _annotate(results)
    emitted = frame[frame["predicted"] > EMIT_THRESHOLD]
    return (
        emitted.groupby(["species", "level_predicted"])
        .agg(n=("exact", "size"), exact=("exact", "mean"), within_one=("within_one", "mean"))
        .reset_index()
    )


def load_table(path: Path | None = None) -> dict[str, Any] | None:
    """Load the calibration table, or None when it has not been generated."""
    target = path or TABLE_PATH
    if not target.exists():
        return None
    try:
        with target.open() as handle:
            return json.load(handle)  # type: ignore[no-any-return]
    except (OSError, json.JSONDecodeError):
        return None


def confidence_for(
    table: dict[str, Any] | None,
    species: str,
    level: str,
    horizon_day: int,
    has_model: bool = True,
    observed: bool = False,
    log_day_mean: float | None = None,
) -> tuple[float, float]:
    """Published (exact, within-one) confidence for one emitted prediction.

    With *log_day_mean* — ``log1p`` of the predicted daily mean the level was
    read from — the answer is conformal: the residual mass of the horizon's
    distribution that keeps the truth inside the level's band (see the module
    docstring). Without it, or without a residual table, it is the flat rate
    plus the horizon offset, which is not keyed on *species* or *level*
    because finer flat tables calibrate worse held-out.

    *horizon_day* is measured from the newest observation, not from today, so
    a forecast built on stale data asks for the horizon it really is. Beyond
    the furthest horizon the table measured, the furthest rate is published
    scaled by :data:`BEYOND_HORIZON_SCALE`.
    """
    if observed:
        return OBSERVED_CONFIDENCE["exact"], OBSERVED_CONFIDENCE["within_one"]

    if log_day_mean is not None:
        conformal = level_probability(table, species, level, log_day_mean, horizon_day)
        if conformal is not None:
            scale = 1.0 if has_model else NO_MODEL_SCALE
            return tuple(  # type: ignore[return-value]
                min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, v * scale)) for v in conformal
            )

    base = FALLBACK
    if table and table.get("overall"):
        base = table["overall"]

    delta = {"exact": 0.0, "within_one": 0.0}
    scale = 1.0 if has_model else NO_MODEL_SCALE
    deltas = table.get("horizon_delta", {}) if table else {}
    if deltas:
        measured = sorted(int(k) for k in deltas)
        if int(horizon_day) in measured:
            delta = deltas[str(int(horizon_day))]
        else:
            delta = deltas[str(max(measured))]
            scale *= BEYOND_HORIZON_SCALE
    out = []
    for key in ("exact", "within_one"):
        value = (float(base.get(key, FALLBACK[key])) + float(delta.get(key, 0.0))) * scale
        out.append(min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, value)))
    return out[0], out[1]


def expected_calibration_error(stated: pd.Series, correct: pd.Series, bins: int = 10) -> float:
    """Mean gap between stated confidence and observed accuracy, bin-weighted.

    The single number that says whether a confidence means anything: 0 is
    perfect, and a scheme that says 0.90 while being right 35% of the time
    scores about 0.55.
    """
    frame = pd.DataFrame({"stated": stated.to_numpy(), "correct": correct.to_numpy()})
    edges = pd.cut(frame["stated"], bins=bins, labels=False, include_lowest=True)
    total = 0.0
    for _, group in frame.groupby(edges):
        if group.empty:
            continue
        gap = abs(group["stated"].mean() - group["correct"].mean())
        total += gap * len(group) / len(frame)
    return float(total)
