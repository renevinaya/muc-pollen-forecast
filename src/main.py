"""
CLI entry point for the Munich pollen forecast system.

Usage:
    python -m src.main collect      # Fetch recent data and append to history
    python -m src.main train        # Train models on accumulated history
    python -m src.main forecast     # Generate forecast and upload to S3
    python -m src.main run          # All three steps in sequence (daily cron)
    python -m src.main backfill N   # Backfill N days of historical data
    python -m src.main benchmark    # Walk-forward evaluation of forecast quality
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

from .collector import collect, update_history, HISTORY_FILE, DATA_DIR
from .trainer import train_all, load_models
from .forecaster import generate_forecast
from .s3 import upload_forecast, upload_csv, sync_historical_data
from .pollen import fetch_pollen, pivot_pollen
from .weather import fetch_historical_weather, fetch_weather_forecast as fetch_weather_fc
from .evaluate import temporal_split_evaluate, print_evaluation_report
from .types import ALL_SPECIES

import pandas as pd


def cmd_collect(days: int = 14) -> pd.DataFrame:
    """Collect recent pollen+weather data and update history."""
    print("=" * 60)
    print("STEP 1: COLLECT DATA")
    print("=" * 60)
    new_data = collect(days=days)
    history = update_history(new_data)
    return history


def cmd_train(history: pd.DataFrame | None = None) -> None:
    """Train models on accumulated history."""
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN MODELS")
    print("=" * 60)
    if history is None:
        if not HISTORY_FILE.exists():
            print("No history file found. Run 'collect' first.")
            return
        history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])

    if len(history) < 50:
        print(f"Only {len(history)} rows in history. Need more data for training.")
        print("Run 'collect' daily to accumulate data, or use 'backfill' for bulk import.")
        return

    train_all(history)


def cmd_forecast(history: pd.DataFrame | None = None) -> None:
    """Generate forecast and optionally upload to S3."""
    print("\n" + "=" * 60)
    print("STEP 3: GENERATE FORECAST")
    print("=" * 60)
    if history is None:
        if HISTORY_FILE.exists():
            history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
        else:
            history = pd.DataFrame()

    forecast = generate_forecast(history)

    # Print summary
    print("\nForecast summary:")
    for day in forecast.forecast:
        top = ", ".join(f"{s.name}({s.level}:{s.value:.0f})" for s in day.species[:3])
        print(f"  {day.date}: {top or 'no pollen'}")

    # Upload to S3 if configured
    import os

    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        upload_forecast(forecast, bucket)
        # Also back up history
        if HISTORY_FILE.exists():
            upload_csv(HISTORY_FILE, bucket, "data/history.csv")
    else:
        # Local dev: write to file
        output_path = DATA_DIR / "forecast.json"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(forecast.to_dict(), indent=2, ensure_ascii=False)
        )
        print(f"\nForecast written to {output_path}")


def cmd_backfill(days: int = 365) -> pd.DataFrame:
    """
    Backfill historical pollen data by fetching from the LGL API in chunks,
    combined with historical weather from Open-Meteo's archive.
    
    Note: The LGL Bayern API may only have limited history available.
    Open-Meteo historical archive goes back to 1940.
    """
    import numpy as np

    print("=" * 60)
    print(f"BACKFILL: Fetching up to {days} days of history")
    print("=" * 60)

    # Fetch pollen data (the API may limit how far back we can go)
    pollen_raw = fetch_pollen(days=days)
    if pollen_raw.empty:
        print("No pollen data available for backfill.")
        return pd.DataFrame()

    pollen = pivot_pollen(pollen_raw)
    print(f"Pollen data: {len(pollen)} days ({pollen.index.min()} to {pollen.index.max()})")

    # Fetch matching weather data: archive + recent forecast API to cover gap
    start = pollen.index.min().date() - timedelta(days=1)
    archive_end = min(pollen.index.max().date(), date.today() - timedelta(days=5))

    weather_parts: list[pd.DataFrame] = []

    if start <= archive_end:
        print(f"Fetching weather archive from {start} to {archive_end}...")
        weather_parts.append(fetch_historical_weather(start, archive_end))

    # Use forecast API for the most recent days (archive is ~5 days behind)
    recent_weather = fetch_weather_fc(days=7)
    today = pd.Timestamp(date.today())
    recent_weather = recent_weather[recent_weather.index <= today]
    if not recent_weather.empty:
        weather_parts.append(recent_weather)
        print(f"Fetched {len(recent_weather)} recent weather days from forecast API")

    if not weather_parts:
        print("No weather data available.")
        return pd.DataFrame()

    weather = pd.concat(weather_parts)
    weather = weather[~weather.index.duplicated(keep="first")].sort_index()

    # Add calendar features
    doy = weather.index.dayofyear
    weather["day_of_year"] = doy
    weather["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    weather["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    weather["month"] = weather.index.month

    # Join
    common_dates = pollen.index.intersection(weather.index)
    print(f"Overlapping dates: {len(common_dates)}")

    rows: list[dict] = []
    for dt in common_dates:
        w = weather.loc[dt]
        for species in ALL_SPECIES:
            val = pollen.loc[dt].get(species, 0.0) if species in pollen.columns else 0.0
            row = {"date": dt, "species": species, "value": float(val)}
            for col in weather.columns:
                row[col] = float(w[col])
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Backfill data: {len(df)} rows")

    history = update_history(df)
    return history


def cmd_benchmark(horizon: int = 1) -> None:
    """Run walk-forward evaluation on accumulated history."""
    print("=" * 60)
    print("BENCHMARK: Walk-Forward Evaluation")
    print("=" * 60)
    if not HISTORY_FILE.exists():
        print("No history file found. Run 'collect' or 'backfill' first.")
        return

    history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    n_days = history["date"].nunique()
    print(f"History: {len(history)} rows, {n_days} unique days")

    results = temporal_split_evaluate(history, test_days=min(90, n_days // 3), n_folds=horizon)
    if not results.empty:
        print_evaluation_report(results)

        # Save results for further analysis
        results_path = DATA_DIR / "benchmark_results.csv"
        results.to_csv(results_path, index=False)
        print(f"\nDetailed results saved to {results_path}")


def cmd_run() -> None:
    """Run full pipeline: collect -> train -> forecast."""
    history = cmd_collect()
    cmd_train(history)
    cmd_forecast(history)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "collect":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
        cmd_collect(days)
    elif command == "train":
        cmd_train()
    elif command == "forecast":
        cmd_forecast()
    elif command == "backfill":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
        cmd_backfill(days)
    elif command == "benchmark":
        horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        cmd_benchmark(horizon)
    elif command == "run":
        cmd_run()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
