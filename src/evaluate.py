"""
Evaluation module: backtesting of forecast quality.

Uses a time-series train/test split (last N days as test set),
compares predictions to actual measurements, and reports
per-species and aggregate metrics. Also supports k-fold
temporal cross-validation for more robust estimates.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from .types import ALL_SPECIES, FEATURE_COLS, WEATHER_FEATURES, LAG_FEATURES, value_to_level, is_season_active
from .trainer import (
    prepare_training_data,
    train_species_model,
    _add_lag_features,
    _add_season_feature,
    _add_weather_derived_features,
    _add_ndvi_features,
    _add_phenology_features,
    inv_log_transform,
)


def temporal_split_evaluate(
    history: pd.DataFrame,
    test_days: int = 60,
    n_folds: int = 3,
) -> pd.DataFrame:
    """
    Monthly forward-chaining cross-validation.

    Splits the history into monthly test windows (starting from the earliest
    month with at least 60 days of prior training data).  This ensures test
    windows cover all seasons — including the pollen-active months —
    instead of only the most recent (often dormant) period.

    Returns a DataFrame with columns:
        date, species, actual, predicted, fold, error, abs_error,
        level_actual, level_predicted
    """
    dates = sorted(history["date"].unique())
    total_days = len(dates)

    min_train = max(60, total_days // 5)  # need a minimum training set

    # Build monthly test blocks
    all_dates_ts = pd.to_datetime(dates)
    months = all_dates_ts.to_period("M").unique().sort_values()

    results: list[dict] = []
    fold_num = 0

    for period in months:
        month_dates = [d for d in dates if pd.Timestamp(d).to_period("M") == period]
        # Everything before this month is training
        train_dates = [d for d in dates if d < month_dates[0]]
        if len(train_dates) < min_train:
            continue
        fold_num += 1
        test_dates = month_dates

        train_data = history[history["date"].isin(train_dates)]
        test_data = history[history["date"].isin(test_dates)]

        print(f"  Fold {fold_num}: train={len(train_dates)}d, test={len(test_dates)}d "
              f"({pd.Timestamp(test_dates[0]).strftime('%Y-%m-%d')} to "
              f"{pd.Timestamp(test_dates[-1]).strftime('%Y-%m-%d')})")

        for species in ALL_SPECIES:
            X_train, y_train, raw_train = prepare_training_data(train_data, species)
            if len(X_train) < 14:
                continue

            model = train_species_model(X_train, y_train, raw_values=raw_train, species=species)

            # Prepare test features
            species_test = test_data[test_data["species"] == species].copy()
            if species_test.empty:
                continue

            # We need lag features for the test set that include the training tail
            species_all = pd.concat([
                train_data[train_data["species"] == species],
                species_test
            ]).sort_values("date").reset_index(drop=True)
            species_all = _add_weather_derived_features(species_all)
            species_all = _add_ndvi_features(species_all)
            species_all = _add_lag_features(species_all)
            species_all = _add_season_feature(species_all, species)
            species_all = _add_phenology_features(species_all, species)

            # Only evaluate on test dates
            species_eval = species_all[species_all["date"].isin(test_dates)].dropna(subset=LAG_FEATURES)
            if species_eval.empty:
                continue

            X_test = species_eval[FEATURE_COLS].fillna(0)
            y_test = species_eval["value"]
            preds_log = model.predict(X_test)
            preds = inv_log_transform(np.maximum(0, preds_log))

            # Match forecaster: force out-of-season predictions to zero
            for i, (_, row) in enumerate(species_eval.iterrows()):
                month = pd.Timestamp(row["date"]).month
                if not is_season_active(species, month):
                    preds[i] = 0.0

            for i, (_, row) in enumerate(species_eval.iterrows()):
                results.append({
                    "date": row["date"],
                    "species": species,
                    "actual": y_test.iloc[i],
                    "predicted": float(preds[i]),
                    "fold": fold_num,
                    "error": float(preds[i]) - y_test.iloc[i],
                    "abs_error": abs(float(preds[i]) - y_test.iloc[i]),
                    "level_actual": value_to_level(y_test.iloc[i], species).value,
                    "level_predicted": value_to_level(float(preds[i]), species).value,
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
    print(f"\nPer-species breakdown:")
    print(f"  {'Species':<12} {'MAE':>8} {'RMSE':>8} {'MedAE':>8} {'LvlAcc':>8} {'N':>6} {'ActMax':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")

    for species in sorted(results["species"].unique()):
        sp = results[results["species"] == species]
        sp_mae = sp["abs_error"].mean()
        sp_rmse = np.sqrt((sp["error"] ** 2).mean())
        sp_median = sp["abs_error"].median()
        sp_level = (sp["level_actual"] == sp["level_predicted"]).mean()
        sp_max = sp["actual"].max()
        print(f"  {species:<12} {sp_mae:>8.1f} {sp_rmse:>8.1f} {sp_median:>8.1f} {sp_level:>7.0%} {len(sp):>6} {sp_max:>8.1f}")

    # Per-fold breakdown
    if "fold" in results.columns and results["fold"].nunique() > 1:
        print(f"\nPer-fold breakdown:")
        for fold in sorted(results["fold"].unique()):
            fr = results[results["fold"] == fold]
            f_mae = fr["abs_error"].mean()
            f_rmse = np.sqrt((fr["error"] ** 2).mean())
            print(f"  Fold {fold}: MAE={f_mae:.1f}, RMSE={f_rmse:.1f}, N={len(fr)}")

    # Worst predictions (largest absolute errors)
    print(f"\nWorst 10 predictions:")
    worst = results.nlargest(10, "abs_error")
    for _, r in worst.iterrows():
        print(f"  {pd.Timestamp(r['date']).strftime('%Y-%m-%d')} {r['species']:<12} "
              f"actual={r['actual']:>7.1f}  predicted={r['predicted']:>7.1f}  "
              f"error={r['error']:>+8.1f}")

    # Bias analysis: does the model systematically under- or over-predict?
    print(f"\nBias analysis (mean error, negative = under-prediction):")
    for species in sorted(results["species"].unique()):
        sp = results[results["species"] == species]
        bias = sp["error"].mean()
        if abs(bias) > 1:
            direction = "OVER" if bias > 0 else "UNDER"
            print(f"  {species:<12} bias={bias:>+8.1f}  ({direction}-predicts)")

    # Check high-pollen days specifically
    high_pollen = results[results["actual"] > 50]
    if not high_pollen.empty:
        print(f"\nHigh-pollen days (actual > 50): {len(high_pollen)} predictions")
        hp_mae = high_pollen["abs_error"].mean()
        hp_bias = high_pollen["error"].mean()
        print(f"  MAE: {hp_mae:.1f}, Bias: {hp_bias:+.1f}")
        hp_level = (high_pollen["level_actual"] == high_pollen["level_predicted"]).mean()
        print(f"  Level accuracy: {hp_level:.0%}")

    # --- In-season evaluation (excludes dormant months) ---
    print(f"\nIn-season evaluation (active months only):")
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

        print(f"\n  {'Species':<12} {'MAE':>8} {'RMSE':>8} {'LvlAcc':>8} {'N':>6}")
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
    overlap_dates: set,
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
        print(f"    → Tied")


def _summarise_dwd(dwd_df: pd.DataFrame) -> None:
    """Print a summary of the current DWD forecast when no overlap is available."""
    for _, row in dwd_df.iterrows():
        print(f"    {row['date']}  {row['species']:<12}  level={int(row['dwd_level'])}")
