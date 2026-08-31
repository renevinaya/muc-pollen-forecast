"""
Evaluation module: backtesting of forecast quality.

Uses a time-series train/test split (last N days as test set),
compares predictions to actual measurements, and reports
per-species and aggregate metrics. Also supports k-fold
temporal cross-validation for more robust estimates.
"""

from datetime import date

import numpy as np
import pandas as pd

from .types import (
    ALL_SPECIES,
    FEATURE_COLS,
    LAG_FEATURES,
    value_to_level,
    is_season_active,
    season_gate_active,
    SPECIES_THRESHOLDS,
    _DEFAULT_THRESHOLDS,
)
from .onset import ONSET_RUN_DAYS, observed_onsets
from .rollout import eligible_months
from .trainer import (
    prepare_training_data,
    train_species_model,
    _add_lag_features,
    _add_season_feature,
    _add_weather_derived_features,
    _add_ndvi_features,
    _add_intraday_features,
    _add_phenology_features,
    inv_log_transform,
)


def temporal_split_evaluate(
    history: pd.DataFrame,
    test_days: int = 60,
    n_folds: int = 3,
    species: list[str] | None = None,
    months: list[pd.Period] | None = None,
) -> pd.DataFrame:
    """
    Monthly forward-chaining cross-validation.

    Splits the history into monthly test windows (starting from the earliest
    month with at least 60 days of prior training data).  This ensures test
    windows cover all seasons — including the pollen-active months —
    instead of only the most recent (often dormant) period.

    When *n_folds* is set, only that many evenly-spaced monthly folds are
    evaluated (speeds up benchmarking significantly). Passing *months* selects
    the test folds explicitly instead, and *species* narrows which species are
    evaluated — together they make it affordable to evaluate every consecutive
    month around a season start rather than a scattered sample.

    **This scores one window ahead with measured lag features**, not the
    5-day autoregressive forecast the product ships: every test row is handed
    the true recent pollen counts. It is the right diagnostic for "are the
    weather and phenology features doing anything", and the wrong one for "how
    good is the forecast" — use :mod:`src.rollout` for that.

    Returns a DataFrame with columns:
        date, species, actual, predicted, fold, error, abs_error,
        level_actual, level_predicted
    """
    dates = sorted(history["date"].unique())
    eligible = eligible_months(history, n_folds=n_folds, months=months)

    evaluated_species = list(species) if species else ALL_SPECIES

    results: list[dict[str, object]] = []
    fold_num = 0

    for period in eligible:
        month_dates = [d for d in dates if pd.Timestamp(d).to_period("M") == period]
        # Everything before this month is training
        train_dates = [d for d in dates if d < month_dates[0]]
        train_days = len(set(pd.to_datetime(d).date() for d in train_dates))
        fold_num += 1
        test_dates = month_dates
        test_days = len(set(pd.to_datetime(d).date() for d in test_dates))

        train_data = history[history["date"].isin(train_dates)]
        test_data = history[history["date"].isin(test_dates)]

        print(f"  Fold {fold_num}: train={train_days}d ({len(train_dates)} windows), "
              f"test={test_days}d ({len(test_dates)} windows) "
              f"({pd.Timestamp(test_dates[0]).strftime('%Y-%m-%d')} to "
              f"{pd.Timestamp(test_dates[-1]).strftime('%Y-%m-%d')})")

        for species_name in evaluated_species:
            x_train, y_train, raw_train = prepare_training_data(train_data, species_name)
            if len(x_train) < 14:
                continue

            model = train_species_model(x_train, y_train, raw_values=raw_train, species=species_name)
            if model is None:
                continue

            # Prepare test features
            species_test = test_data[test_data["species"] == species_name].copy()
            if species_test.empty:
                continue

            # We need lag features for the test set that include the training tail
            species_all = pd.concat([
                train_data[train_data["species"] == species_name],
                species_test
            ]).sort_values("date").reset_index(drop=True)
            species_all = _add_weather_derived_features(species_all, species_name)
            species_all = _add_ndvi_features(species_all)
            species_all = _add_intraday_features(species_all)
            species_all = _add_lag_features(species_all)
            species_all = _add_season_feature(species_all, species_name)
            species_all = _add_phenology_features(species_all, species_name)

            # Only evaluate on test dates
            species_eval = species_all[
                species_all["date"].isin(test_dates)
            ].dropna(subset=LAG_FEATURES)
            if species_eval.empty:
                continue

            x_test = species_eval[FEATURE_COLS].fillna(0)
            y_test = species_eval["value"]
            preds_log = model.predict(x_test)
            preds = inv_log_transform(np.maximum(0, preds_log))

            # Match forecaster: force predictions to zero only outside the
            # widened season window (core ± shoulder).
            for i, (_, row) in enumerate(species_eval.iterrows()):
                month = pd.Timestamp(row["date"]).month
                if not season_gate_active(species_name, month):
                    preds[i] = 0.0

            for i, (_, row) in enumerate(species_eval.iterrows()):
                results.append({
                    "date": row["date"],
                    "species": species_name,
                    "actual": y_test.iloc[i],
                    "predicted": float(preds[i]),
                    "fold": fold_num,
                    "error": float(preds[i]) - y_test.iloc[i],
                    "abs_error": abs(float(preds[i]) - y_test.iloc[i]),
                    "level_actual": value_to_level(y_test.iloc[i], species_name).value,
                    "level_predicted": value_to_level(float(preds[i]), species_name).value,
                })

    df = pd.DataFrame(results)
    return df


def print_evaluation_report(results: pd.DataFrame) -> None:
    """Print a human-readable evaluation report."""
    if results.empty:
        print("No evaluation results.")
        return

    print("\n" + "=" * 70)
    print("FORECAST EVALUATION REPORT")
    print("=" * 70)

    # Overall metrics
    mae = results["abs_error"].mean()
    rmse = np.sqrt((results["error"] ** 2).mean())
    median_ae = results["abs_error"].median()
    level_accuracy = (results["level_actual"] == results["level_predicted"]).mean()

    print(f"\nOverall ({len(results)} predictions):")
    print(f"  MAE  (Mean Absolute Error):   {mae:.1f}")
    print(f"  RMSE (Root Mean Sq Error):     {rmse:.1f}")
    print(f"  Median Absolute Error:         {median_ae:.1f}")
    print(f"  Level Accuracy:                {level_accuracy:.1%}")

    # Per-species
    print("\nPer-species breakdown:")
    hdr = "  {:<12} {:>8} {:>8} {:>8} {:>8} {:>6} {:>8}"  # pylint: disable=consider-using-f-string
    print(hdr.format("Species", "MAE", "RMSE", "MedAE", "LvlAcc", "N", "ActMax"))
    sep = "  {} {} {} {} {} {} {}"  # pylint: disable=consider-using-f-string
    print(sep.format("-" * 12, "-" * 8, "-" * 8, "-" * 8, "-" * 8, "-" * 6, "-" * 8))

    for species in sorted(results["species"].unique()):
        sp = results[results["species"] == species]
        sp_mae = sp["abs_error"].mean()
        sp_rmse = np.sqrt((sp["error"] ** 2).mean())
        sp_median = sp["abs_error"].median()
        sp_level = (sp["level_actual"] == sp["level_predicted"]).mean()
        sp_max = sp["actual"].max()
        print(f"  {species:<12} {sp_mae:>8.1f} {sp_rmse:>8.1f} {sp_median:>8.1f}"
              f" {sp_level:>7.0%} {len(sp):>6} {sp_max:>8.1f}")

    # Per-fold breakdown
    if "fold" in results.columns and results["fold"].nunique() > 1:
        print("\nPer-fold breakdown:")
        for fold in sorted(results["fold"].unique()):
            fr = results[results["fold"] == fold]
            f_mae = fr["abs_error"].mean()
            f_rmse = np.sqrt((fr["error"] ** 2).mean())
            print(f"  Fold {fold}: MAE={f_mae:.1f}, RMSE={f_rmse:.1f}, N={len(fr)}")

    # Worst predictions (largest absolute errors)
    print("\nWorst 10 predictions:")
    worst = results.nlargest(10, "abs_error")
    for _, r in worst.iterrows():
        dt_str = pd.Timestamp(r['date']).strftime('%Y-%m-%d %H:%M')
        print(f"  {dt_str} {r['species']:<12} "
              f"actual={r['actual']:>7.1f}  predicted={r['predicted']:>7.1f}  "
              f"error={r['error']:>+8.1f}")

    # Bias analysis: does the model systematically under- or over-predict?
    print("\nBias analysis (mean error, negative = under-prediction):")
    for species in sorted(results["species"].unique()):
        sp = results[results["species"] == species]
        bias = sp["error"].mean()
        if abs(bias) > 1:
            direction = "OVER" if bias > 0 else "UNDER"
            print(f"  {species:<12} bias={bias:>+8.1f}  ({direction}-predicts)")

    # Check very-high pollen days specifically
    very_high = results[results["level_actual"] == "very_high"]
    if not very_high.empty:
        print(f"\nVery-high pollen days: {len(very_high)} predictions")
        vh_mae = very_high["abs_error"].mean()
        vh_bias = very_high["error"].mean()
        print(f"  MAE: {vh_mae:.1f}, Bias: {vh_bias:+.1f}")
        vh_level = (very_high["level_actual"] == very_high["level_predicted"]).mean()
        print(f"  Level accuracy: {vh_level:.0%}")
        # Per-species breakdown for very-high days
        for species in sorted(very_high["species"].unique()):
            sp_vh = very_high[very_high["species"] == species]
            sp_acc = (sp_vh["level_actual"] == sp_vh["level_predicted"]).mean()
            print(f"    {species:<12} {sp_acc:>5.0%}  (n={len(sp_vh)})")

    # --- In-season evaluation (excludes dormant months) ---
    print("\nIn-season evaluation (active months only):")
    in_season_mask = results.apply(
        lambda r: is_season_active(r["species"], pd.Timestamp(r["date"]).month), axis=1
    )
    season_results = results[in_season_mask]
    if not season_results.empty:
        s_mae = season_results["abs_error"].mean()
        s_rmse = np.sqrt((season_results["error"] ** 2).mean())
        s_level = (season_results["level_actual"] == season_results["level_predicted"]).mean()
        print(f"  {len(season_results)} predictions in season")
        print(f"  MAE: {s_mae:.1f}  RMSE: {s_rmse:.1f}  Level accuracy: {s_level:.1%}")

        hdr = "\n  {:<12} {:>8} {:>8} {:>8} {:>6}"  # pylint: disable=consider-using-f-string
        print(hdr.format("Species", "MAE", "RMSE", "LvlAcc", "N"))
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for species in sorted(season_results["species"].unique()):
            sp = season_results[season_results["species"] == species]
            if len(sp) == 0:
                continue
            sp_mae = sp["abs_error"].mean()
            sp_rmse = np.sqrt((sp["error"] ** 2).mean())
            sp_level = (sp["level_actual"] == sp["level_predicted"]).mean()
            print(f"  {species:<12} {sp_mae:>8.1f} {sp_rmse:>8.1f} {sp_level:>7.0%} {len(sp):>6}")
    else:
        print("  No in-season predictions in test windows.")


def compare_with_dwd(results: pd.DataFrame) -> None:
    """
    Fetch the current DWD pollen forecast for Oberbayern and compare
    level accuracy against our own model predictions.

    The DWD only publishes a short-horizon forecast (today/tomorrow/day-after),
    so the comparison is limited to whatever dates overlap with our evaluation.
    """
    from .dwd import fetch_dwd_forecast, DWD_SPECIES_MAP

    print("\n" + "=" * 70)
    print("DWD POLLEN FORECAST COMPARISON (Oberbayern)")
    print("=" * 70)

    try:
        dwd_df = fetch_dwd_forecast()
    except Exception as exc:
        print(f"  Could not fetch DWD forecast: {exc}")
        return

    if dwd_df.empty:
        print("  No DWD forecast data available.")
        return

    # DWD forecast has columns: date, species, dwd_level (0-3)
    # Our results have: date, species, level_actual, level_predicted
    print(f"\n  DWD forecast covers: {dwd_df['date'].min()} to {dwd_df['date'].max()}")
    print(f"  DWD species: {sorted(dwd_df['species'].unique())}")

    # Only DWD species that we also track
    dwd_species = set(DWD_SPECIES_MAP.values()) & set(ALL_SPECIES)
    print(f"  Overlapping species: {sorted(dwd_species)}")

    # Try to match DWD forecast dates with our evaluation results
    if results.empty:
        print("  No evaluation results to compare against.")
        return

    results_dates = set(pd.to_datetime(results["date"]).dt.date)
    dwd_dates = set(pd.to_datetime(dwd_df["date"]).dt.date)
    overlap_dates = results_dates & dwd_dates

    if overlap_dates:
        print(f"  Overlapping eval dates: {len(overlap_dates)}")
        _compare_overlapping(results, dwd_df, overlap_dates, dwd_species)
    else:
        print("  No overlapping dates between DWD forecast and evaluation results.")
        print("  (DWD only covers today–day after tomorrow; eval covers historical data)")
        print("\n  Showing DWD forecast summary instead:")
        _summarise_dwd(dwd_df)


def _compare_overlapping(
    results: pd.DataFrame,
    dwd_df: pd.DataFrame,
    overlap_dates: set[date],
    species_set: set[str],
) -> None:
    """Compare our predictions vs DWD for overlapping (date, species) pairs."""
    # Map our PollenLevel string values to a 0-3 numeric scale matching DWD
    level_to_num = {"none": 0, "low": 1, "moderate": 2, "high": 3, "very_high": 3}

    our_right = 0
    dwd_right = 0
    both_right = 0
    total = 0

    for dt in sorted(overlap_dates):
        for sp in sorted(species_set):
            our_row = results[
                (pd.to_datetime(results["date"]).dt.date == dt) & (results["species"] == sp)
            ]
            dwd_row = dwd_df[
                (pd.to_datetime(dwd_df["date"]).dt.date == dt) & (dwd_df["species"] == sp)
            ]
            if our_row.empty or dwd_row.empty:
                continue

            actual_str = str(our_row.iloc[0]["level_actual"])
            our_str = str(our_row.iloc[0]["level_predicted"])
            actual_num = level_to_num.get(actual_str, 0)
            our_num = level_to_num.get(our_str, 0)
            # DWD levels are already numeric 0-3 (with halves like 2.5); round
            dwd_num = round(float(dwd_row.iloc[0]["dwd_level"]))

            our_ok = our_num == actual_num
            dwd_ok = dwd_num == actual_num
            our_right += our_ok
            dwd_right += dwd_ok
            both_right += our_ok and dwd_ok
            total += 1

    if total == 0:
        print("  No comparable (date, species) pairs found.")
        return

    print(f"\n  Head-to-head comparison ({total} pairs):")
    print(f"    Our level accuracy:  {our_right/total:.1%}  ({our_right}/{total})")
    print(f"    DWD level accuracy:  {dwd_right/total:.1%}  ({dwd_right}/{total})")
    print(f"    Both correct:        {both_right/total:.1%}  ({both_right}/{total})")

    if our_right > dwd_right:
        print(f"    → Our model is BETTER by {(our_right - dwd_right)/total:+.1%}")
    elif dwd_right > our_right:
        print(f"    → DWD forecast is better by {(dwd_right - our_right)/total:+.1%}")
    else:
        print("    → Tied")


def _summarise_dwd(dwd_df: pd.DataFrame) -> None:
    """Print a summary of the current DWD forecast when no overlap is available."""
    for _, row in dwd_df.iterrows():
        print(f"    {row['date']}  {row['species']:<12}  level={int(row['dwd_level'])}")


def onset_windows(history: pd.DataFrame, window_days: int) -> dict[tuple[str, int], pd.Timestamp]:
    """Measured onset date per (species, year), for slicing evaluation results."""
    windows: dict[tuple[str, int], pd.Timestamp] = {}
    for species in ALL_SPECIES:
        for year, doy in observed_onsets(history, species).items():
            windows[(species, year)] = (
                pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
            )
    return windows


def _onset_slice_mask(
    results: pd.DataFrame, windows: dict[tuple[str, int], pd.Timestamp], window_days: int
) -> pd.Series:
    """True for result rows falling within ±*window_days* of their season's onset."""
    dates = pd.to_datetime(results["date"])
    onset_dates = [
        windows.get((sp, ts.year)) for sp, ts in zip(results["species"], dates)
    ]
    deltas = [
        abs((ts - onset).days) if onset is not None else None
        for ts, onset in zip(dates, onset_dates)
    ]
    return pd.Series(
        [d is not None and d <= window_days for d in deltas], index=results.index
    )


def _predicted_onset_doy(
    daily: pd.DataFrame, column: str, species: str
) -> int | None:
    """Apply the observed-onset rule to a daily series of predictions or actuals."""
    threshold = SPECIES_THRESHOLDS.get(species, _DEFAULT_THRESHOLDS)[0]
    daily = daily.sort_values("doy")
    above = (daily[column] >= threshold).to_numpy()
    doys = daily["doy"].to_numpy()
    for i in range(len(above) - ONSET_RUN_DAYS + 1):
        if above[i : i + ONSET_RUN_DAYS].all():
            return int(doys[i])
    return None


def print_onset_window_report(
    results: pd.DataFrame, history: pd.DataFrame, window_days: int = 21
) -> None:
    """Report accuracy in the season-boundary windows specifically.

    Onset windows are a tiny share of all predictions, so a change that helps
    them a lot barely moves the aggregate MAE. This slice is where the
    phenology features earn or lose their keep, and it reports two different
    things: how well the model predicts *concentrations* around onset, and how
    close it gets the *start date* itself.
    """
    if results.empty:
        return

    print("\n" + "=" * 70)
    print(f"ONSET-WINDOW EVALUATION (±{window_days} days around measured onset)")
    print("=" * 70)

    windows = onset_windows(history, window_days)
    mask = _onset_slice_mask(results, windows, window_days)
    near = results[mask]
    rest = results[~mask]

    if near.empty:
        print("\nNo evaluation rows fall inside an onset window "
              "(the folds may not cover any season start).")
        return

    def _metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
        return (
            df["abs_error"].mean(),
            np.sqrt((df["error"] ** 2).mean()),
            (df["level_actual"] == df["level_predicted"]).mean(),
            df["error"].mean(),
        )

    n_mae, n_rmse, n_lvl, n_bias = _metrics(near)
    print(f"\nOnset windows ({len(near)} of {len(results)} predictions, "
          f"{len(near) / len(results):.1%}):")
    print(f"  MAE: {n_mae:.1f}   RMSE: {n_rmse:.1f}   "
          f"Level accuracy: {n_lvl:.1%}   Bias: {n_bias:+.1f}")
    if not rest.empty:
        r_mae, r_rmse, r_lvl, r_bias = _metrics(rest)
        print(f"Everything else ({len(rest)} predictions):")
        print(f"  MAE: {r_mae:.1f}   RMSE: {r_rmse:.1f}   "
              f"Level accuracy: {r_lvl:.1%}   Bias: {r_bias:+.1f}")

    hdr = "\n  {:<12} {:>8} {:>8} {:>8} {:>9} {:>6}"  # pylint: disable=consider-using-f-string
    print(hdr.format("Species", "MAE", "RMSE", "LvlAcc", "Bias", "N"))
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*6}")
    for species in sorted(near["species"].unique()):
        sp = near[near["species"] == species]
        s_mae, s_rmse, s_lvl, s_bias = _metrics(sp)
        print(f"  {species:<12} {s_mae:>8.1f} {s_rmse:>8.1f} {s_lvl:>7.0%}"
              f" {s_bias:>+9.1f} {len(sp):>6}")

    _print_onset_timing(results, history, window_days)


def _print_onset_timing(
    results: pd.DataFrame, history: pd.DataFrame, window_days: int
) -> None:
    """Compare the season start the model predicts with the one that happened.

    Only reported for (species, year) pairs whose test folds cover the whole
    window contiguously — a monthly fold that clips the onset would otherwise
    show up as a spurious miss.
    """
    daily = results.copy()
    daily["day"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.groupby(["species", "day"], as_index=False)[["actual", "predicted"]].mean()
    daily["year"] = daily["day"].dt.year
    daily["doy"] = daily["day"].dt.dayofyear

    rows: list[tuple[str, int, int, int | None]] = []
    for species in sorted(daily["species"].unique()):
        for year, obs_doy in observed_onsets(history, species).items():
            grp = daily[(daily["species"] == species) & (daily["year"] == year)]
            if grp.empty:
                continue
            covered = set(grp["doy"])
            needed = set(range(max(1, obs_doy - window_days), obs_doy + window_days + 1))
            if not needed.issubset(covered):
                continue  # fold does not cover the window; a miss here means nothing
            rows.append((species, year, obs_doy, _predicted_onset_doy(grp, "predicted", species)))

    if not rows:
        print("\n  Onset timing: no season start is fully covered by the test folds.")
        return

    print("\n  Onset timing (predicted season start vs measured, in days):")
    print(f"    {'Species':<12} {'Year':>6} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
    print(f"    {'-'*12} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    errors: list[int] = []
    for species, year, obs_doy, pred_doy in rows:
        if pred_doy is None:
            print(f"    {species:<12} {year:>6} {obs_doy:>8} {'never':>10} {'—':>8}")
            continue
        err = pred_doy - obs_doy
        errors.append(err)
        print(f"    {species:<12} {year:>6} {obs_doy:>8} {pred_doy:>10} {err:>+8d}")

    if errors:
        print(f"\n    Mean absolute timing error: {np.mean(np.abs(errors)):.1f} days"
              f"   (bias {np.mean(errors):+.1f}, n={len(errors)})")
    missed = sum(1 for _, _, _, p in rows if p is None)
    if missed:
        print(f"    Seasons the model never started: {missed} of {len(rows)}")


def onset_focus_months(
    history: pd.DataFrame,
    species: list[str],
    window_days: int = 21,
    years: int | None = None,
) -> list[pd.Period]:
    """Test months that fully cover each species' season start.

    The onset-timing diagnostic needs contiguous daily coverage across the whole
    window, which evenly-spaced sample folds never give it. This returns every
    month those windows touch, newest *years* seasons only when asked.
    """
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for name in species:
        onsets = observed_onsets(history, name)
        chosen = sorted(onsets)[-years:] if years else sorted(onsets)
        for year in chosen:
            onset = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(
                days=onsets[year] - 1
            )
            windows.append(
                (onset - pd.Timedelta(days=window_days), onset + pd.Timedelta(days=window_days))
            )

    periods: set[pd.Period] = set()
    for start, end in windows:
        periods.update(pd.period_range(start, end, freq="M"))
    return sorted(periods)
