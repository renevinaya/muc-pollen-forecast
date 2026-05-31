"""
Evaluation module: single-fold peak-month benchmark.

Trains on all history older than 12 months, finds the calendar month
within the last 12 months with the highest total pollen, and evaluates
forecast quality against that month.
"""

import numpy as np
import pandas as pd

from .types import ALL_SPECIES, FEATURE_COLS, LAG_FEATURES, is_season_active
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


def peak_month_evaluate(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Single-fold benchmark against the highest-pollen month in the last 12 months.

    Training data: all history with date < (today − 12 months).
    Test month: the calendar month in the last 12 months whose total pollen
                sum (across all species and windows) is the largest.

    Returns:
        (results_df, peak_month_label) where results_df has columns:
            date, species, actual, predicted, error, abs_error
        and peak_month_label is a string like "2025-04".
    """
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)

    train_data = history[pd.to_datetime(history["date"]) < cutoff]
    recent_data = history[pd.to_datetime(history["date"]) >= cutoff].copy()

    if train_data.empty:
        print("  Not enough history: no data older than 12 months found.")
        return pd.DataFrame(), ""

    if recent_data.empty:
        print("  Not enough history: no data in the last 12 months found.")
        return pd.DataFrame(), ""

    unique_train_days = len(pd.to_datetime(train_data["date"]).dt.normalize().unique())
    print(f"  Training data: {len(train_data)} rows, {unique_train_days} unique days "
          f"(up to {cutoff.date()})")

    # Find the calendar month with the highest total pollen in the last 12 months
    recent_data["_month"] = pd.to_datetime(recent_data["date"]).dt.to_period("M")
    monthly_totals = recent_data.groupby("_month")["value"].sum()
    peak_month = monthly_totals.idxmax()
    peak_total = monthly_totals[peak_month]
    peak_month_label = str(peak_month)

    print(f"  Peak pollen month: {peak_month_label} "
          f"(total pollen across all species: {peak_total:.0f})")

    test_dates = list(
        history["date"][
            pd.to_datetime(history["date"]).dt.to_period("M") == peak_month
        ].unique()
    )
    test_data = history[history["date"].isin(test_dates)]

    unique_test_days = len(pd.to_datetime(test_dates).normalize().unique())
    print(f"  Test data: {len(test_data)} rows, {unique_test_days} unique days "
          f"({pd.Timestamp(min(test_dates)).strftime('%Y-%m-%d')} to "
          f"{pd.Timestamp(max(test_dates)).strftime('%Y-%m-%d')})")

    results: list[dict[str, object]] = []

    for species in ALL_SPECIES:
        x_train, y_train, raw_train = prepare_training_data(train_data, species)
        if len(x_train) < 14:
            continue

        model = train_species_model(x_train, y_train, raw_values=raw_train, species=species)
        if model is None:
            continue

        species_test = test_data[test_data["species"] == species].copy()
        if species_test.empty:
            continue

        # Compute lag features using training tail + test data to avoid leakage
        species_all = pd.concat([
            train_data[train_data["species"] == species],
            species_test,
        ]).sort_values("date").reset_index(drop=True)
        species_all = _add_weather_derived_features(species_all, species)
        species_all = _add_ndvi_features(species_all)
        species_all = _add_intraday_features(species_all)
        species_all = _add_lag_features(species_all)
        species_all = _add_season_feature(species_all, species)
        species_all = _add_phenology_features(species_all, species)

        species_eval = species_all[
            species_all["date"].isin(test_dates)
        ].dropna(subset=LAG_FEATURES)
        if species_eval.empty:
            continue

        x_test = species_eval[FEATURE_COLS].fillna(0)
        y_test = species_eval["value"]
        preds_log = model.predict(x_test)
        preds = inv_log_transform(np.maximum(0, preds_log))

        # Force out-of-season predictions to zero
        for i, (_, row) in enumerate(species_eval.iterrows()):
            if not is_season_active(species, pd.Timestamp(row["date"]).month):
                preds[i] = 0.0

        for i, (_, row) in enumerate(species_eval.iterrows()):
            results.append({
                "date": row["date"],
                "species": species,
                "actual": y_test.iloc[i],
                "predicted": float(preds[i]),
                "error": float(preds[i]) - y_test.iloc[i],
                "abs_error": abs(float(preds[i]) - y_test.iloc[i]),
            })

    return pd.DataFrame(results), peak_month_label


def print_evaluation_report(results: pd.DataFrame, peak_month: str) -> None:
    """Print benchmark results: MAE and signal-normalized accuracy."""
    if results.empty:
        print("No evaluation results.")
        return

    print("\n" + "=" * 70)
    print("FORECAST EVALUATION REPORT")
    print("=" * 70)

    mae = results["abs_error"].mean()
    rms_pollen = np.sqrt((results["actual"] ** 2).mean())
    accuracy = (1.0 - mae / rms_pollen) * 100.0 if rms_pollen > 0 else 0.0

    print(f"\nTest month : {peak_month}  (highest-pollen month in last 12 months)")
    print(f"Predictions: {len(results)}")
    print(f"\n  MAE      : {mae:.1f} pollen/m³")
    print(f"  Accuracy : {accuracy:.1f}%   (= 1 − MAE / RMS_pollen)")

    # Per-species breakdown
    print("\nPer-species:")
    hdr = "  {:<12} {:>10} {:>10} {:>6}"
    print(hdr.format("Species", "MAE", "Accuracy", "N"))
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*6}")

    for species in sorted(results["species"].unique()):
        sp = results[results["species"] == species]
        sp_mae = sp["abs_error"].mean()
        sp_rms = np.sqrt((sp["actual"] ** 2).mean())
        if sp_rms > 0:
            sp_acc = (1.0 - sp_mae / sp_rms) * 100.0
            acc_str = f"{sp_acc:>9.1f}%"
        else:
            acc_str = "       n/a"
        print(f"  {species:<12} {sp_mae:>10.1f} {acc_str} {len(sp):>6}")
