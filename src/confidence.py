"""
Measured forecast confidence.

The confidence attached to each species in ``forecast.json`` used to be
``0.90 - 0.08 * day``, clipped, with a +0.05 bonus for assimilated windows.
Those numbers were invented, and the rollout benchmark says both the level and
the slope were wrong:

* On the rows the forecast actually emits (value > 0.5), the emitted level is
  exactly right **34-36%** of the time, not 90%. The pooled 77% accuracy the
  benchmark reports is carried almost entirely by ``none`` predictions, and
  those are filtered out of the output before a user ever sees them.
* Accuracy is **flat** across the horizon — 35.8% at day 1 to 33.9% at day 5 —
  because the model is direct rather than recursive. The old decay dropped 32
  points over the same span.

So the shipped number overstated day-1 reliability by about 2.5x and then
decayed for a reason that does not exist.

This module replaces it with a table measured from the benchmark. Two figures
are published per prediction, because "is this level right" has two defensible
readings and they differ a lot:

* ``confidence`` — P(the emitted level is exactly right).
* ``confidence_within_one`` — P(the truth is within one level of it), which is
  closer to "would a reader have behaved correctly". That runs around 87%.

The table is **flat**: one rate for everything, plus a small per-horizon
offset. That is not the obvious design — the plan called for per-species,
per-horizon rates — but it is what the evidence supports. Scored
leave-one-fold-out, finer tables calibrate *worse*:

    scheme                  ECE     correlation with being right
    old 0.90-0.08/day      0.393     0.013
    flat + horizon         0.053    -0.059
    by level               0.063     0.046
    by species             0.077     0.003
    by species x level     0.120    -0.020

Per-species/level accuracy does vary a lot in-sample (0.12 to 0.62 across
cells), but those differences do not survive to a held-out fold: they are
season- and year-specific, and re-publishing them as confidence is false
precision.

The correlation column is the more sobering result. It is ~0 for every scheme,
including the fine-grained ones. **The average can be calibrated; which
individual predictions are more reliable cannot currently be told.** A
confidence that barely varies is the honest consequence, not an oversight.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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

    overall = {
        "exact": float(emitted["exact"].mean()),
        "within_one": float(emitted["within_one"].mean()),
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
        "source": {
            "rows_scored": int(len(frame)),
            "rows_emitted": int(len(emitted)),
            "folds": int(frame["fold"].nunique()) if "fold" in frame else None,
            "origins": int(frame["origin"].nunique()) if "origin" in frame else None,
        },
        "overall": overall,
        "horizon_delta": horizon_delta,
    }


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
) -> tuple[float, float]:
    """Published (exact, within-one) confidence for one emitted prediction.

    *species* and *level* are accepted but deliberately unused: the
    leave-one-fold-out comparison in the module docstring shows that keying on
    them makes calibration worse. They stay in the signature so a future
    re-calibration that *does* find a generalising split has somewhere to put
    it, and so callers do not have to change.
    """
    if observed:
        return OBSERVED_CONFIDENCE["exact"], OBSERVED_CONFIDENCE["within_one"]

    base = FALLBACK
    if table and table.get("overall"):
        base = table["overall"]

    delta = {"exact": 0.0, "within_one": 0.0}
    if table:
        delta = table.get("horizon_delta", {}).get(str(int(horizon_day)), delta)

    scale = 1.0 if has_model else NO_MODEL_SCALE
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
