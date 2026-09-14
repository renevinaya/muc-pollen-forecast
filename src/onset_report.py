"""Score the shipped forecast around the season starts.

The general rollout benchmark samples six months across the year and almost
never lands on a season start: its January and March folds bracket the
February hazel/alder onset without covering it. So it can say nothing about
the thing an allergy sufferer most wants from a pollen forecast, which is
*when the season begins and how hard*. This module scores exactly that, on
the same direct rollout the product ships (:func:`src.rollout.rollout_evaluate`
restricted to the months containing each measured onset).

Three questions, per species-year and forecast horizon:

* **Timing** — on which day did the forecast first show a
  :data:`ONSET_RUN_DAYS`-day run at or above the low level, against the day
  the measurements did? A species-year is only scored at a horizon whose
  predictions cover the ±7 days around the onset.
* **Amount** — MAE and bias in the ±``window_days`` around the onset, next to
  the MAE of predicting zero throughout, and the predicted/actual ratio by
  days since onset, which is where the model's onset problem actually lives.
* **False starts** — predicted runs ending more than ``window_days`` before
  the real onset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .onset import ONSET_RUN_DAYS, observed_onsets
from .types import SPECIES_THRESHOLDS, _DEFAULT_THRESHOLDS

ONSET_SPECIES = ["Corylus", "Alnus", "Betula"]

# Days on either side of the onset that make up its window.
WINDOW_DAYS = 10
# How far before the onset a predicted run may start and still count as
# "early" rather than as a false start.
SEARCH_DAYS = 21
# Predictions needed inside ±7 days for a horizon to be scored on timing.
MIN_COVERED_DAYS = 12
# Bins of days since onset for the amplitude table.
RAMP_BINS: list[tuple[int, int]] = [(0, 4), (5, 9), (10, 14), (15, 19)]
RAMP_HORIZON = 3


def _daily(results: pd.DataFrame) -> pd.DataFrame:
    """Daily mean actual and prediction per (species, horizon, day)."""
    frame = results.copy()
    frame["day"] = pd.to_datetime(frame["date"]).dt.normalize()
    return (
        frame.groupby(["species", "horizon_day", "day"], as_index=False)
        .agg(actual=("actual", "mean"), predicted=("predicted", "mean"))
    )


def _first_run(days: pd.DatetimeIndex, above: np.ndarray) -> pd.Timestamp | None:
    for i in range(len(above) - ONSET_RUN_DAYS + 1):
        if above[i : i + ONSET_RUN_DAYS].all():
            return pd.Timestamp(days[i])
    return None


def _run_days(above: np.ndarray) -> int:
    """Number of positions at which a full run starts."""
    return int(sum(above[i : i + ONSET_RUN_DAYS].all() for i in range(len(above) - ONSET_RUN_DAYS + 1)))


def onset_dates(history: pd.DataFrame, species: list[str], years: int | None) -> dict[tuple[str, int], pd.Timestamp]:
    """Measured onset per (species, year), newest *years* seasons only when asked."""
    out: dict[tuple[str, int], pd.Timestamp] = {}
    for name in species:
        onsets = observed_onsets(history, name)
        chosen = sorted(onsets)[-years:] if years else sorted(onsets)
        for year in chosen:
            out[(name, year)] = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=onsets[year] - 1)
    return out


def score_onsets(
    results: pd.DataFrame,
    history: pd.DataFrame,
    species: list[str] | None = None,
    years: int | None = None,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """One row per (species, year, horizon) with timing and window metrics.

    Columns: ``timing_error`` (days, + = late; NaN when the horizon's
    predictions never form a run in the search span), ``scored`` (whether the
    ±7-day window was covered at that horizon), ``window_mae``, ``window_bias``,
    ``zero_mae`` (MAE of predicting 0 in the window), ``false_starts``.
    """
    targets = list(species) if species else ONSET_SPECIES
    daily = _daily(results)
    rows: list[dict[str, object]] = []
    for (name, year), onset in onset_dates(history, targets, years).items():
        threshold = SPECIES_THRESHOLDS.get(name, _DEFAULT_THRESHOLDS)[0]
        sp = daily[(daily["species"] == name) & (daily["day"].dt.year == year)]
        for horizon in sorted(sp["horizon_day"].unique()):
            h = sp[sp["horizon_day"] == horizon].set_index("day").sort_index()
            near = h[(h.index >= onset - pd.Timedelta(days=7)) & (h.index <= onset + pd.Timedelta(days=7))]
            scored = len(near) >= MIN_COVERED_DAYS

            span = h[(h.index >= onset - pd.Timedelta(days=SEARCH_DAYS)) & (h.index <= onset + pd.Timedelta(days=SEARCH_DAYS))]
            first = _first_run(pd.DatetimeIndex(span.index), (span["predicted"] >= threshold).to_numpy()) if scored else None
            timing = float((first - onset).days) if first is not None else float("nan")

            window = h[(h.index >= onset - pd.Timedelta(days=window_days)) & (h.index <= onset + pd.Timedelta(days=window_days))]
            err = window["predicted"] - window["actual"]
            before = h[h.index < onset - pd.Timedelta(days=window_days)]
            rows.append(
                {
                    "species": name,
                    "year": year,
                    "onset": onset,
                    "horizon_day": int(horizon),
                    "scored": scored,
                    "timing_error": timing,
                    "window_days": int(len(window)),
                    "window_mae": float(err.abs().mean()) if len(window) else float("nan"),
                    "window_bias": float(err.mean()) if len(window) else float("nan"),
                    "zero_mae": float(window["actual"].abs().mean()) if len(window) else float("nan"),
                    "false_starts": _run_days((before["predicted"] >= threshold).to_numpy()) if len(before) else 0,
                }
            )
    return pd.DataFrame(rows)


def ramp_ratios(
    results: pd.DataFrame,
    history: pd.DataFrame,
    species: list[str] | None = None,
    years: int | None = None,
    horizon: int = RAMP_HORIZON,
) -> pd.DataFrame:
    """Predicted / actual daily mean by days since onset, one row per species-year."""
    targets = list(species) if species else ONSET_SPECIES
    daily = _daily(results)
    daily = daily[daily["horizon_day"] == horizon]
    rows: list[dict[str, object]] = []
    for (name, year), onset in onset_dates(history, targets, years).items():
        sp = daily[(daily["species"] == name) & (daily["day"].dt.year == year)].copy()
        if sp.empty:
            continue
        sp["rel"] = (sp["day"] - onset).dt.days
        row: dict[str, object] = {"species": name, "year": year}
        for lo, hi in RAMP_BINS:
            cell = sp[(sp["rel"] >= lo) & (sp["rel"] <= hi)]
            act, pred = cell["actual"].mean(), cell["predicted"].mean()
            row[f"{lo}-{hi}"] = float(pred / act) if len(cell) and act > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def print_onset_rollout_report(
    results: pd.DataFrame,
    history: pd.DataFrame,
    species: list[str] | None = None,
    years: int | None = None,
) -> None:
    """The season-start report for a rollout restricted to onset months."""
    scores = score_onsets(results, history, species, years)
    if scores.empty:
        print("\n  No measured onset falls inside the scored months.")
        return

    print("\n" + "=" * 70)
    print("SEASON-START EVALUATION (direct rollout over the onset months)")
    print("=" * 70)
    horizons = sorted(scores["horizon_day"].unique())

    print(f"\nTiming: first predicted {ONSET_RUN_DAYS}-day run at or above the low level,"
          " days from the measured onset (+ = late):")
    print("  {:<9} {:<11}".format("Species", "onset") + "".join(f"{'d' + str(h):>7}" for h in horizons)
          + f"{'±' + str(WINDOW_DAYS) + 'd MAE':>10} {'zero':>7} {'bias':>7}")
    for (name, year), grp in scores.groupby(["species", "year"], sort=False):
        grp = grp.set_index("horizon_day")
        cells = ""
        for h in horizons:
            if h not in grp.index or not grp.loc[h, "scored"]:
                cells += f"{'n/a':>7}"
            elif np.isnan(grp.loc[h, "timing_error"]):
                cells += f"{'miss':>7}"
            else:
                cells += f"{int(grp.loc[h, 'timing_error']):>+6d}d"
        first = grp.iloc[0]
        print(f"  {name:<9} {str(first['onset'].date()):<11}{cells}"
              f"{first['window_mae']:>10.1f} {first['zero_mae']:>7.1f} {first['window_bias']:>+7.1f}")

    print("\n  Timing summary (scored species-years only):")
    print(f"    {'Species':<9}" + "".join(f"{'d' + str(h):>9}" for h in horizons) + "   mean |err| in days, (n)")
    for name in scores["species"].unique():
        cells = ""
        for h in horizons:
            sub = scores[(scores["species"] == name) & (scores["horizon_day"] == h) & scores["scored"]]
            if sub.empty:
                cells += f"{'-':>9}"
            else:
                cells += f"{sub['timing_error'].abs().mean():>5.1f} ({len(sub)})"
        print(f"    {name:<9}{cells}")

    false = scores[scores["false_starts"] > 0]
    print(f"\n  False starts (predicted runs ending more than {WINDOW_DAYS} d before the onset): "
          f"{int(scores['false_starts'].sum())} run-days"
          + (", in " + ", ".join(f"{r.species} {r.year} d{r.horizon_day}" for r in false.itertuples()) if len(false) else ""))

    ratios = ramp_ratios(results, history, species, years)
    if not ratios.empty:
        print(f"\nAmount: predicted / actual daily mean by days since onset, day-{RAMP_HORIZON} horizon:")
        cols = [f"{lo}-{hi}" for lo, hi in RAMP_BINS]
        print("  {:<9} {:>6}".format("Species", "year") + "".join(f"{c:>8}" for c in cols))
        for _, r in ratios.iterrows():
            cells = "".join(f"{r[c]:>8.2f}" if not np.isnan(r[c]) else f"{'-':>8}" for c in cols)
            print(f"  {r['species']:<9} {int(r['year']):>6}{cells}")
        heavy = ratios.dropna(subset=["5-9"])
        print(f"\n  Mean ratio in days 5-9 across species-years: {heavy['5-9'].mean():.2f}"
              f"  (1.00 = right on average; the heavy-year problem is ratios of 0.1-0.3)")
