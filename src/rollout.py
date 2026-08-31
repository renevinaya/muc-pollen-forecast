"""
Autoregressive rollout backtest — the benchmark that scores what ships.

``evaluate.temporal_split_evaluate`` builds every test row's lag features from
*measured* pollen, so it answers "given yesterday's true counts, how good is
the next window?". The product answers something else: a 5-day forecast whose
lag features, after the first few windows, are made entirely of its own earlier
predictions. Thirteen of the 72 features are lags, and they carry most of the
in-season signal, so the two questions can have very different answers — and
only the first one was ever measured.

This module rolls the model forward exactly as :func:`forecaster.generate_forecast`
does, through the same :mod:`src.features` code path, and reports skill per
forecast day.

Two things it deliberately does not simulate, both of which flatter the model
and are called out in the report rather than hidden:

* **Weather is actual, not forecast.** A real 5-day forecast carries the
  weather model's own error; this backtest hands the model perfect weather.
* **No DWD blend.** The DWD index is only published for today + 2 days and is
  not archived, so the inference-time blend cannot be replayed historically.

What it *does* reproduce faithfully: models trained only on data before the
fold, lag state seeded only from measurements before the forecast origin, and
onset/GDD calibration drawn only from earlier seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import (
    ALL_SPECIES,
    FEATURE_COLS,
    FORECAST_DAYS,
    LAG_FEATURES,
    WEATHER_FEATURES,
    WINDOWS_PER_DAY,
    season_gate_active,
    value_to_level,
)
from .features import (
    LagState,
    build_context,
    ndvi_from_history,
    static_feature_frame,
)
from .trainer import prepare_training_data, train_species_model

WINDOW = pd.Timedelta(hours=3)


def eligible_months(
    history: pd.DataFrame, n_folds: int | None = None, months: list[pd.Period] | None = None
) -> list[pd.Period]:
    """Monthly test blocks that have enough history behind them to train on.

    Shared with the classic benchmark so both score the same folds.
    """
    dates = sorted(history["date"].unique())
    if not dates:
        return []
    unique_days = len({pd.Timestamp(d).date() for d in dates})
    min_train_days = max(60, unique_days // 5)

    periods = pd.DatetimeIndex(dates).to_period("M").unique().sort_values()
    eligible: list[pd.Period] = []
    for period in periods:
        first = min(d for d in dates if pd.Timestamp(d).to_period("M") == period)
        train_days = len({pd.Timestamp(d).date() for d in dates if d < first})
        if train_days >= min_train_days:
            eligible.append(period)

    if months is not None:
        wanted = set(months)
        missing = wanted - set(eligible)
        if missing:
            print(f"  {len(missing)} requested month(s) lack enough training history; skipped")
        return [p for p in eligible if p in wanted]

    if n_folds and len(eligible) > n_folds:
        eligible = _spread_over_season(eligible, n_folds)
        picked = ", ".join(str(p) for p in eligible)
        print(f"  Sub-sampled to {n_folds} folds spread across the year: {picked}")
    return eligible


def _spread_over_season(periods: list[pd.Period], n_folds: int) -> list[pd.Period]:
    """Pick *n_folds* test months that cover different parts of the pollen year.

    Sampling evenly along the timeline — which is what this replaces — spaces
    folds by *index*, and because the months run in calendar order that lands
    on nearly the same month of the year every time: over this history, three
    folds all came out in August, and so did four, when every tree species is
    dormant. The benchmark was reporting on an empty season.

    Spreading over the calendar instead puts folds in the hazel/alder, birch
    and grass/nettle parts of the year, preferring the most recent instance of
    each so the model is judged on the data it will actually face.
    """
    targets = [round(1 + i * 12 / n_folds) for i in range(n_folds)]
    by_month: dict[int, list[pd.Period]] = {}
    for period in periods:
        by_month.setdefault(period.month, []).append(period)

    chosen: list[pd.Period] = []
    for target in targets:
        # Nearest calendar month that still has an unused year, around the year.
        candidates = sorted(
            (m for m, ps in by_month.items() if ps),
            key=lambda m: min(abs(m - target), 12 - abs(m - target)),
        )
        if not candidates:
            break
        chosen.append(by_month[candidates[0]].pop())  # most recent year
    return sorted(chosen)


def _fold_context(
    history: pd.DataFrame,
    past: pd.DataFrame,
    windows: pd.DatetimeIndex,
    species: list[str],
):
    """Feature context for one fold.

    The onset estimate has to keep updating *through* the fold: it switches
    from prior-year climatology to the observed GDD crossing as soon as the
    year's warmth confirms it, and that switch is driven by temperature, not by
    pollen. So the frame the calibration sees is extended over the fold with
    the fold's real weather and zeroed measurements — the temperature is data a
    forecaster genuinely has, while the zeros cannot invent a season, because
    the climatology and threshold are medians over *earlier* years only.
    """
    weather_cols = [c for c in WEATHER_FEATURES if c in history.columns]
    weather = (
        history[history["date"].isin(windows)]
        .groupby("date")[weather_cols]
        .first()
        .sort_index()
    )

    extension = weather.reset_index().rename(columns={"index": "date"})
    extension["species"] = species[0]
    extension["value"] = 0.0
    calibration_frame = pd.concat([past, extension], ignore_index=True)

    days = pd.DatetimeIndex(weather.index).normalize().unique()
    ctx = build_context(
        calibration_frame,
        weather,
        ndvi=ndvi_from_history(history, days),
        species=species,
        parallel=False,
    )
    return ctx, weather


def rollout_evaluate(
    history: pd.DataFrame,
    horizon_days: int = FORECAST_DAYS,
    n_folds: int = 3,
    species: list[str] | None = None,
    months: list[pd.Period] | None = None,
    origin_hour: int = 0,
) -> pd.DataFrame:
    """Walk-forward backtest of the real autoregressive forecast.

    For every day in a test month, a forecast is launched from that day at
    *origin_hour* and rolled *horizon_days* forward, feeding each prediction
    into the next window's lag features. Lag state is seeded only from
    measurements strictly before the origin, so the rollout starts knowing
    exactly what production knows.

    Returns one row per scored (origin, window, species) with a ``horizon_day``
    column running 1…*horizon_days*.
    """
    evaluated = list(species) if species else list(ALL_SPECIES)
    horizon_windows = horizon_days * WINDOWS_PER_DAY
    folds = eligible_months(history, n_folds=n_folds, months=months)
    if not folds:
        print("  No eligible folds (not enough training history).")
        return pd.DataFrame()

    all_dates = pd.DatetimeIndex(sorted(history["date"].unique()))
    results: list[dict[str, object]] = []

    for fold_num, period in enumerate(folds, start=1):
        month_start = period.to_timestamp()
        month_end = period.to_timestamp(how="end")

        past = history[history["date"] < month_start]
        if past.empty:
            continue

        origins = pd.DatetimeIndex(
            sorted(
                {
                    d.normalize() + pd.Timedelta(hours=origin_hour)
                    for d in all_dates
                    if month_start <= d <= month_end
                }
            )
        )
        if len(origins) == 0:
            continue

        span_end = origins.max() + horizon_windows * WINDOW
        span = all_dates[(all_dates >= origins.min()) & (all_dates <= span_end)]
        if len(span) == 0:
            continue

        print(
            f"  Fold {fold_num} ({period}): train<{month_start.date()}, "
            f"{len(origins)} origins × {horizon_days}d"
        )

        for name in evaluated:
            x_train, y_train, raw_train = prepare_training_data(past, name)
            if len(x_train) < 14:
                continue
            model = train_species_model(x_train, y_train, raw_values=raw_train, species=name)
            if model is None:
                continue

            ctx, _ = _fold_context(history, past, span, [name])
            static = static_feature_frame(ctx, name, span)

            actuals = (
                history[(history["species"] == name) & history["date"].isin(span)]
                .set_index("date")["value"]
                .to_dict()
            )

            states = [LagState.from_history(history, name, o) for o in origins]
            for step in range(horizon_windows):
                dts = [o + step * WINDOW for o in origins]
                live = [i for i, dt in enumerate(dts) if dt in static.index]
                if not live:
                    break

                lag_rows = []
                for i in live:
                    states[i].begin_window(dts[i])
                    lag_rows.append(states[i].lag_features())

                frame = static.loc[[dts[i] for i in live]].reset_index(drop=True)
                lag_frame = pd.DataFrame(lag_rows, columns=LAG_FEATURES)
                x_step = pd.concat([frame, lag_frame], axis=1)[FEATURE_COLS].fillna(0)
                preds_log = np.maximum(0.0, model.predict(x_step))

                horizon_day = step // WINDOWS_PER_DAY + 1
                for slot, i in enumerate(live):
                    dt = dts[i]
                    pred_log = float(preds_log[slot])
                    predicted = float(np.expm1(pred_log))
                    # Same season gate the forecaster applies before emitting.
                    if not season_gate_active(name, dt.month):
                        predicted, pred_log = 0.0, 0.0
                    states[i].record(pred_log)

                    if dt not in actuals:
                        continue
                    actual = float(actuals[dt])
                    results.append({
                        "date": dt,
                        "origin": origins[i],
                        "horizon_day": horizon_day,
                        "species": name,
                        "actual": actual,
                        "predicted": predicted,
                        "fold": fold_num,
                        "error": predicted - actual,
                        "abs_error": abs(predicted - actual),
                        "level_actual": value_to_level(actual, name).value,
                        "level_predicted": value_to_level(predicted, name).value,
                    })

    return pd.DataFrame(results)


# --- Reporting --------------------------------------------------------------


def _metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
    return (
        float(df["abs_error"].mean()),
        float(np.sqrt((df["error"] ** 2).mean())),
        float((df["level_actual"] == df["level_predicted"]).mean()),
        float(df["error"].mean()),
    )


def persistence_baseline(results: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Score "tomorrow looks like the last window we measured" on the same rows.

    Without a baseline a horizon curve is unreadable: MAE rises with horizon
    partly because the model degrades and partly because the days being scored
    differ. Persistence degrades for the second reason only.
    """
    if results.empty:
        return pd.DataFrame()

    measured = (
        history.set_index(["species", "date"])["value"].groupby(level=[0, 1]).first()
    )
    rows: list[dict[str, object]] = []
    for (name, origin), grp in results.groupby(["species", "origin"], sort=False):
        key = (name, pd.Timestamp(origin) - WINDOW)
        if key not in measured.index:
            continue
        anchor = float(measured.loc[key])
        for _, r in grp.iterrows():
            rows.append({
                "species": name,
                "horizon_day": r["horizon_day"],
                "error": anchor - r["actual"],
                "abs_error": abs(anchor - r["actual"]),
                "level_actual": r["level_actual"],
                "level_predicted": value_to_level(anchor, str(name)).value,
            })
    return pd.DataFrame(rows)


def print_rollout_report(results: pd.DataFrame, history: pd.DataFrame | None = None) -> None:
    """Report skill per forecast day — the number the product actually makes."""
    if results.empty:
        print("No rollout results.")
        return

    print("\n" + "=" * 70)
    print("AUTOREGRESSIVE ROLLOUT EVALUATION")
    print("=" * 70)
    print("\nLag features are fed from the model's own predictions, as in production.")
    print("Weather is actual (not forecast) and the DWD blend is not replayed,")
    print("so these numbers are, if anything, optimistic.")

    mae, rmse, lvl, bias = _metrics(results)
    n_origins = results["origin"].nunique()
    print(f"\nOverall ({len(results)} predictions from {n_origins} forecast origins):")
    print(f"  MAE {mae:.1f}   RMSE {rmse:.1f}   Level accuracy {lvl:.1%}   Bias {bias:+.1f}")

    baseline = persistence_baseline(results, history) if history is not None else pd.DataFrame()

    print("\nBy forecast day:")
    header = "  {:>10} {:>9} {:>9} {:>9} {:>9} {:>8}"
    print(header.format("Horizon", "MAE", "RMSE", "LvlAcc", "Bias", "N"))
    print("  " + " ".join("-" * w for w in (10, 9, 9, 9, 9, 8)))
    for day in sorted(results["horizon_day"].unique()):
        sub = results[results["horizon_day"] == day]
        d_mae, d_rmse, d_lvl, d_bias = _metrics(sub)
        print(f"  {'day ' + str(int(day)):>10} {d_mae:>9.1f} {d_rmse:>9.1f} "
              f"{d_lvl:>8.1%} {d_bias:>+9.1f} {len(sub):>8}")

    if not baseline.empty:
        print("\n  Persistence baseline (last measured window held flat):")
        print(header.format("Horizon", "MAE", "RMSE", "LvlAcc", "Bias", "N"))
        print("  " + " ".join("-" * w for w in (10, 9, 9, 9, 9, 8)))
        for day in sorted(baseline["horizon_day"].unique()):
            sub = baseline[baseline["horizon_day"] == day]
            b_mae, b_rmse, b_lvl, b_bias = _metrics(sub)
            print(f"  {'day ' + str(int(day)):>10} {b_mae:>9.1f} {b_rmse:>9.1f} "
                  f"{b_lvl:>8.1%} {b_bias:>+9.1f} {len(sub):>8}")

        print("\n  Skill vs persistence (positive = model better):")
        for day in sorted(results["horizon_day"].unique()):
            m = results[results["horizon_day"] == day]["abs_error"].mean()
            b = baseline[baseline["horizon_day"] == day]["abs_error"].mean()
            if b > 0:
                print(f"    day {int(day)}: MAE {m:.1f} vs {b:.1f}  "
                      f"({(b - m) / b:+.1%} skill)")

    print("\nPer-species × horizon (in-season rows only, MAE):")
    in_season = results[results["actual"] > 0]
    if in_season.empty:
        print("  No non-zero actuals in the scored windows.")
    else:
        days = sorted(results["horizon_day"].unique())
        print("  {:<12}".format("Species") + "".join(f"{'d' + str(int(d)):>9}" for d in days)
              + f"{'N':>8}")
        print("  " + "-" * (12 + 9 * len(days) + 8))
        for name in sorted(in_season["species"].unique()):
            sp = in_season[in_season["species"] == name]
            cells = ""
            for d in days:
                cell = sp[sp["horizon_day"] == d]
                cells += f"{cell['abs_error'].mean():>9.1f}" if len(cell) else f"{'—':>9}"
            print(f"  {name:<12}{cells}{len(sp):>8}")

    print("\nDegradation from day 1 to the last day:")
    days = sorted(results["horizon_day"].unique())
    first = results[results["horizon_day"] == days[0]]
    last = results[results["horizon_day"] == days[-1]]
    f_mae, _, f_lvl, _ = _metrics(first)
    l_mae, _, l_lvl, _ = _metrics(last)
    print(f"  MAE            {f_mae:.1f} → {l_mae:.1f}  ({(l_mae - f_mae) / max(f_mae, 1e-9):+.0%})")
    print(f"  Level accuracy {f_lvl:.1%} → {l_lvl:.1%}  ({l_lvl - f_lvl:+.1%})")
