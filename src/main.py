"""
CLI entry point for the Munich pollen forecast system.

Usage:
    python -m src.main collect      # Fetch recent data and append to history
    python -m src.main train        # Train models on accumulated history
    python -m src.main forecast     # Generate forecast and upload to S3
    python -m src.main run          # All three steps in sequence (daily cron)
    python -m src.main backfill N   # Backfill N days of historical data
    python -m src.main backfill-om  # Backfill from Open-Meteo pollen (2021+)
    python -m src.main backfill-ps  # Backfill from pollenscience.eu (2019+, slow)
    python -m src.main benchmark    # Walk-forward evaluation of forecast quality
    python -m src.main dwd          # Show DWD pollen forecast for Oberbayern
    python -m src.main phenology    # Download DWD phenology data for Munich
"""

import json
import sys
from datetime import date, timedelta
from typing import Any

import pandas as pd

from .collector import collect, update_history, HISTORY_FILE, DATA_DIR
from .trainer import train_all
from .forecaster import generate_forecast
from .s3 import upload_forecast, upload_csv
from .pollen import fetch_pollen, pivot_pollen
from .pollenscience import fetch_pollenscience_chunked
from .openmeteo_pollen import fetch_openmeteo_pollen, EARLIEST_DATE as OM_EARLIEST
from .weather import fetch_historical_weather, fetch_weather_forecast as fetch_weather_fc
from .evaluate import temporal_split_evaluate, print_evaluation_report, compare_with_dwd
from .types import ALL_SPECIES


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

    # Print summary (show daily peak per species across windows)
    print("\nForecast summary:")
    for day in forecast.forecast:
        all_species: dict[str, Any] = {}
        for w in day.windows:
            for s in w.species:
                if s.name not in all_species or s.value > all_species[s.name].value:
                    all_species[s.name] = s
        top = sorted(all_species.values(), key=lambda s: s.value, reverse=True)[:3]
        top_str = ", ".join(f"{s.name}({s.level}:{s.value:.0f})" for s in top)
        print(f"  {day.date} ({len(day.windows)} windows): {top_str or 'no pollen'}")

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
    combined with historical weather from Open-Meteo's archive at 3h resolution.

    Note: The LGL Bayern API may only have limited history available.
    Open-Meteo historical archive goes back to 1940.

    Important: Old daily-resolution history.csv data is incompatible with
    the 3h model. Delete data/history.csv before running backfill.
    """
    import numpy as np

    print("=" * 60)
    print(f"BACKFILL: Fetching up to {days} days of history (3h resolution)")
    print("=" * 60)

    if HISTORY_FILE.exists():
        old = pd.read_csv(HISTORY_FILE, parse_dates=["date"], nrows=100)
        if not old.empty:
            hours = pd.to_datetime(old["date"]).dt.hour.unique()
            if len(hours) == 1 and hours[0] == 0:
                print("WARNING: Existing history.csv appears to contain old daily-resolution data.")
                print("  The 3h model requires 3h-resolution data.")
                print("  Consider deleting data/history.csv before backfilling.")

    # Fetch pollen data at 3h resolution
    pollen_raw = fetch_pollen(days=days)
    if pollen_raw.empty:
        print("No pollen data available for backfill.")
        return pd.DataFrame()

    pollen = pivot_pollen(pollen_raw)
    n_days = len(pollen.index.normalize().unique())
    print(f"Pollen data: {len(pollen)} windows ({n_days} days, "
          f"{pollen.index.min()} to {pollen.index.max()})")

    # Fetch matching weather data at 3h resolution
    start = pollen.index.min().date() - timedelta(days=1)
    archive_end = min(pollen.index.max().date(), date.today() - timedelta(days=5))

    weather_parts: list[pd.DataFrame] = []

    if start <= archive_end:
        print(f"Fetching weather archive from {start} to {archive_end}...")
        weather_parts.append(fetch_historical_weather(start, archive_end))

    # Use forecast API for the most recent days (archive is ~5 days behind)
    recent_weather = fetch_weather_fc(days=7)
    now = pd.Timestamp.now().floor("3h")
    recent_weather = recent_weather[recent_weather.index <= now]
    if not recent_weather.empty:
        weather_parts.append(recent_weather)
        print(f"Fetched {len(recent_weather)} recent weather windows from forecast API")

    if not weather_parts:
        print("No weather data available.")
        return pd.DataFrame()

    weather = pd.concat(weather_parts)
    weather = weather[~weather.index.duplicated(keep="first")].sort_index()

    # Add calendar + time-of-day features
    doy = weather.index.dayofyear
    weather["day_of_year"] = doy
    weather["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    weather["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    weather["month"] = weather.index.month
    hour = weather.index.hour
    weather["hour_of_day"] = hour
    weather["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    weather["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Join
    common_dts = pollen.index.intersection(weather.index)
    print(f"Overlapping windows: {len(common_dts)}")

    # Fetch NDVI for the backfill period (daily resolution)
    try:
        from .ndvi import ndvi_features
        unique_dates = pd.DatetimeIndex(common_dts.normalize().unique())
        ndvi_df = ndvi_features(unique_dates)
        if not ndvi_df.empty:
            print(f"NDVI data: {len(ndvi_df)} days")
        else:
            ndvi_df = pd.DataFrame()
    except Exception as exc:
        print(f"NDVI fetch failed ({exc}), continuing without.")
        ndvi_df = pd.DataFrame()

    rows: list[dict] = []
    for dt in common_dts:
        w = weather.loc[dt]
        day = dt.normalize()
        for species in ALL_SPECIES:
            val = pollen.loc[dt].get(species, 0.0) if species in pollen.columns else 0.0
            row = {"date": dt, "species": species, "value": float(val)}
            for col in weather.columns:
                row[col] = float(w[col])
            # Add NDVI (daily resolution, shared across windows)
            if not ndvi_df.empty and day in ndvi_df.index:
                for col in ndvi_df.columns:
                    row[col] = float(ndvi_df.loc[day, col])
            else:
                row["ndvi"] = 0.0
                row["evi"] = 0.0
                row["ndvi_delta"] = 0.0
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Backfill data: {len(df)} rows")

    history = update_history(df)
    return history


def cmd_backfill_openmeteo(start_year: int = 2021) -> pd.DataFrame:
    """
    Backfill historical pollen data from Open-Meteo Air Quality API (CAMS model).

    Provides ~5 years (2021–present) of modeled hourly pollen data for
    Alnus, Betula, Poaceae, Artemisia, and Ambrosia at 3h resolution,
    combined with historical weather, calendar and NDVI features.

    Species not available in Open-Meteo (Corylus, Fraxinus, Populus,
    Quercus, Salix, Urtica) are excluded from this backfill.
    """
    import numpy as np
    from .ndvi import ndvi_features

    start = date(start_year, 1, 1)
    end = date.today() - timedelta(days=5)  # archive lag
    if start < OM_EARLIEST:
        start = OM_EARLIEST

    print("=" * 60)
    print(f"BACKFILL OPEN-METEO: {start} to {end}")
    print("=" * 60)

    # 1. Fetch pollen data from Open-Meteo
    print(f"Fetching Open-Meteo pollen data ({start} to {end})...")
    pollen_raw = fetch_openmeteo_pollen(start, end)
    if pollen_raw.empty:
        print("No pollen data returned.")
        return pd.DataFrame()

    om_species = sorted(pollen_raw["species"].unique())
    print(f"  Species: {om_species}")
    print(f"  Rows: {len(pollen_raw)}")

    # Pivot to wide format (index=datetime, columns=species)
    pollen = pollen_raw.pivot_table(
        index="date", columns="species", values="value", aggfunc="mean"
    ).fillna(0)
    pollen.index = pd.DatetimeIndex(pollen.index)
    pollen = pollen.sort_index()
    n_days = len(pollen.index.normalize().unique())
    print(f"  Windows: {len(pollen)} ({n_days} days, "
          f"{pollen.index.min()} to {pollen.index.max()})")

    # 2. Fetch matching historical weather
    print(f"Fetching weather archive ({start} to {end})...")
    weather = fetch_historical_weather(start, end)
    print(f"  Weather windows: {len(weather)}")

    # 3. Calendar features
    doy = weather.index.dayofyear
    weather["day_of_year"] = doy
    weather["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    weather["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    weather["month"] = weather.index.month
    hour = weather.index.hour
    weather["hour_of_day"] = hour
    weather["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    weather["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # 4. Intersect pollen and weather windows
    common_dts = pollen.index.intersection(weather.index)
    print(f"  Overlapping windows: {len(common_dts)}")
    if common_dts.empty:
        print("No overlapping windows.")
        return pd.DataFrame()

    # 5. NDVI features
    try:
        unique_dates = pd.DatetimeIndex(common_dts.normalize().unique())
        ndvi_df = ndvi_features(unique_dates)
        if not ndvi_df.empty:
            print(f"  NDVI: {len(ndvi_df)} days")
        else:
            ndvi_df = pd.DataFrame()
    except Exception as exc:
        print(f"  NDVI fetch failed ({exc}), continuing without.")
        ndvi_df = pd.DataFrame()

    # 6. Build rows — only for species available in Open-Meteo
    rows: list[dict] = []
    for dt in common_dts:
        w = weather.loc[dt]
        day = dt.normalize()
        for species in om_species:
            val = pollen.loc[dt].get(species, 0.0)
            row = {"date": dt, "species": species, "value": float(val)}
            for col in weather.columns:
                row[col] = float(w[col])
            if not ndvi_df.empty and day in ndvi_df.index:
                for col in ndvi_df.columns:
                    row[col] = float(ndvi_df.loc[day, col])
            else:
                row["ndvi"] = 0.0
                row["evi"] = 0.0
                row["ndvi_delta"] = 0.0
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nBackfill data: {len(df)} rows")

    history = update_history(df)
    return history


def cmd_backfill_pollenscience(start_year: int = 2019) -> pd.DataFrame:
    """
    Backfill historical pollen data from pollenscience.eu (TUM / Helmholtz).

    Fetches data from start_year to present in small 28-day chunks with
    5-second delays between requests to avoid overloading the server.
    Combined with historical weather, calendar and NDVI features.

    All species in ALL_SPECIES are available via the DEMUNC/DEBIED stations.
    """
    import numpy as np
    from .ndvi import ndvi_features

    start = date(start_year, 1, 1)
    end = date.today() - timedelta(days=1)

    print("=" * 60)
    print(f"BACKFILL POLLENSCIENCE.EU: {start} to {end}")
    print(f"  Stations: DEMUNC + DEBIED (Munich)")
    print(f"  Species: {', '.join(ALL_SPECIES)}")
    print(f"  Fetching slowly with 5s delay between requests...")
    print("=" * 60)

    # 1. Fetch pollen data slowly
    print(f"\nFetching pollen data ({start} to {end})...")
    pollen_raw = fetch_pollenscience_chunked(start, end, delay=5.0)
    if pollen_raw.empty:
        print("No pollen data returned.")
        return pd.DataFrame()

    ps_species = sorted(pollen_raw["species"].unique())
    print(f"\n  Species found: {ps_species}")
    print(f"  Total rows: {len(pollen_raw)}")

    # Pivot to wide format
    pollen = pollen_raw.pivot_table(
        index="date", columns="species", values="value", aggfunc="mean"
    ).fillna(0)
    pollen.index = pd.DatetimeIndex(pollen.index)
    pollen = pollen.sort_index()
    n_days = len(pollen.index.normalize().unique())
    print(f"  Windows: {len(pollen)} ({n_days} days, "
          f"{pollen.index.min()} to {pollen.index.max()})")

    # 2. Fetch matching historical weather
    weather_start = pollen.index.min().date() - timedelta(days=1)
    weather_end = min(end, date.today() - timedelta(days=5))
    print(f"\nFetching weather archive ({weather_start} to {weather_end})...")
    weather = fetch_historical_weather(weather_start, weather_end)
    print(f"  Weather windows: {len(weather)}")

    # 3. Calendar features
    doy = weather.index.dayofyear
    weather["day_of_year"] = doy
    weather["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    weather["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    weather["month"] = weather.index.month
    hour = weather.index.hour
    weather["hour_of_day"] = hour
    weather["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    weather["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # 4. Intersect pollen and weather windows
    common_dts = pollen.index.intersection(weather.index)
    print(f"  Overlapping windows: {len(common_dts)}")
    if common_dts.empty:
        print("No overlapping windows.")
        return pd.DataFrame()

    # 5. NDVI features
    try:
        unique_dates = pd.DatetimeIndex(common_dts.normalize().unique())
        ndvi_df = ndvi_features(unique_dates)
        if not ndvi_df.empty:
            print(f"  NDVI: {len(ndvi_df)} days")
        else:
            ndvi_df = pd.DataFrame()
    except Exception as exc:
        print(f"  NDVI fetch failed ({exc}), continuing without.")
        ndvi_df = pd.DataFrame()

    # 6. Build rows
    rows: list[dict] = []
    for dt in common_dts:
        w = weather.loc[dt]
        day = dt.normalize()
        for species in ALL_SPECIES:
            val = pollen.loc[dt].get(species, 0.0) if species in pollen.columns else 0.0
            row = {"date": dt, "species": species, "value": float(val)}
            for col in weather.columns:
                row[col] = float(w[col])
            if not ndvi_df.empty and day in ndvi_df.index:
                for col in ndvi_df.columns:
                    row[col] = float(ndvi_df.loc[day, col])
            else:
                row["ndvi"] = 0.0
                row["evi"] = 0.0
                row["ndvi_delta"] = 0.0
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nBackfill data: {len(df)} rows")

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
    unique_days = len(set(pd.to_datetime(history["date"]).dt.date))
    print(f"History: {len(history)} rows, {unique_days} unique days")

    results = temporal_split_evaluate(history, test_days=min(90, unique_days // 3), n_folds=horizon)
    if not results.empty:
        print_evaluation_report(results)

        # Compare with DWD forecast
        compare_with_dwd(results)

        # Save results for further analysis
        results_path = DATA_DIR / "benchmark_results.csv"
        results.to_csv(results_path, index=False)
        print(f"\nDetailed results saved to {results_path}")


def cmd_dwd() -> None:
    """Fetch and display the current DWD pollen forecast for Oberbayern."""
    from .dwd import fetch_dwd_forecast

    print("=" * 60)
    print("DWD POLLEN FORECAST — Oberbayern")
    print("=" * 60)

    try:
        df = fetch_dwd_forecast()
    except Exception as exc:
        print(f"Failed to fetch DWD forecast: {exc}")
        return

    if df.empty:
        print("No DWD forecast data.")
        return

    for dt in sorted(df["date"].unique()):
        day_df = df[df["date"] == dt]
        print(f"\n  {dt}:")
        for _, row in day_df.iterrows():
            level = int(row["dwd_level"])
            level_bar = "█" * level + "░" * (3 - level)
            print(f"    {row['species']:<12} {level_bar}  ({level}/3)")


def cmd_phenology() -> None:
    """Download DWD phenology data for Munich and show season statistics."""
    from .dwd import fetch_dwd_phenology, phenology_season_stats

    print("=" * 60)
    print("DWD PHENOLOGY DATA — Munich")
    print("=" * 60)

    pheno = fetch_dwd_phenology()
    if pheno.empty:
        print("No phenology data available.")
        return

    print(f"\nDownloaded {len(pheno)} records")
    for sp in sorted(pheno["species"].unique()):
        sp_data = pheno[pheno["species"] == sp]
        years = sp_data["year"]
        print(f"  {sp}: {len(sp_data)} years ({int(years.min())}-{int(years.max())})")

    stats = phenology_season_stats(pheno)
    print("\nFlowering onset statistics (day of year):")
    hdr = "  {:<12}  {:>6}  {:>6}  {:>6}  {:>6}"  # pylint: disable=consider-using-f-string
    print(hdr.format("Species", "Mean", "Std", "Min", "Max"))
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for sp, s in sorted(stats.items()):
        print(f"  {sp:<12}  {s['mean_onset_doy']:>6.0f}  "
              f"{s['std_onset_doy']:>6.1f}  {s['earliest_onset_doy']:>6}  "
              f"{s['latest_onset_doy']:>6}")

    # Save phenology data
    pheno_path = DATA_DIR / "phenology.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pheno.to_csv(pheno_path, index=False)
    print(f"\nSaved to {pheno_path}")


def cmd_run() -> None:
    """Run full pipeline: collect -> train -> forecast."""
    history = cmd_collect()
    cmd_train(history)
    cmd_forecast(history)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand."""
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
    elif command == "backfill-om":
        start_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2021
        cmd_backfill_openmeteo(start_year)
    elif command == "backfill-ps":
        start_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
        cmd_backfill_pollenscience(start_year)
    elif command == "benchmark":
        horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        cmd_benchmark(horizon)
    elif command == "dwd":
        cmd_dwd()
    elif command == "phenology":
        cmd_phenology()
    elif command == "run":
        cmd_run()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
