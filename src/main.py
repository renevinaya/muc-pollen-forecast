"""
CLI entry point for the Munich pollen forecast system.

Usage:
    python -m src.main collect      # Fetch recent data and append to history
    python -m src.main train        # Train models on accumulated history
    python -m src.main forecast     # Generate forecast and write data/forecast.json
    python -m src.main backfill-species Fungus  # Add one taxon's history (Munich + upwind)
    python -m src.main run          # Collect + forecast (3-hourly cron)
    python -m src.main run-train    # Collect + train + forecast (monthly cron)
    python -m src.main backfill N   # Collect N days in one go (bootstrap)
    python -m src.main backfill-labels [START_YEAR]  # Rewrite every measurement from the official LGL series
    python -m src.main backfill-upwind [START_YEAR]  # Fetch the upwind stations' history (2019+)
    python -m src.main backfill-weather [START] [END]  # Rewrite weather columns from the archive
    python -m src.main backfill-ndvi [START]           # Rewrite NDVI columns from MODIS
    python -m src.main run-backfill # Both of the above against the data release (Actions)
    python -m src.main benchmark [--folds N | --months YYYY-MM,...]  # Walk-forward rollout of the 5-day forecast
    python -m src.main benchmark-onset [--species A,B] [--years N] [--classic]  # The shipped forecast around the season starts
    python -m src.main outlook      # The season outlook and its start-date error by lead
    python -m src.main calibrate    # Regenerate the confidence table
    python -m src.main dwd          # Show DWD pollen forecast for Oberbayern
    python -m src.main phenology    # Download DWD phenology data for Munich
"""

import json
import sys
from datetime import date
from typing import Any

import pandas as pd

from .collector import collect, update_history, HISTORY_FILE, DATA_DIR
from .trainer import train_all, MODELS_DIR
from .forecaster import generate_forecast
from .store import (
    upload_csv,
    download_csv,
    upload_models,
    download_models,
    sync_historical_data,
    can_upload,
    HISTORY_ASSET,
    PHENOLOGY_ASSET,
    asset_exists,
)
from .pollen import fetch_measurements_chunked
from .evaluate import (
    temporal_split_evaluate,
    print_evaluation_report,
    print_onset_window_report,
    onset_focus_months,
    compare_with_dwd,
)
from .rollout import rollout_evaluate, print_rollout_report, record_months
from .onset_report import WINDOW_DAYS
from .types import ALL_SPECIES, FORECAST_DAYS, LOCATION
from .clock import local_now, local_today
from .upwind import (
    UPWIND_ASSET,
    UPWIND_FILE,
    UPWIND_STATIONS,
    fetch_upwind_chunked,
    load_upwind,
    replace_upwind,
    sync_upwind,
    update_upwind,
)


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

    train_all(history, upwind=load_upwind())


def cmd_forecast(history: pd.DataFrame | None = None) -> None:
    """Generate forecast, write it locally and back up accumulated state."""
    print("\n" + "=" * 60)
    print("STEP 3: GENERATE FORECAST")
    print("=" * 60)
    if history is None:
        if HISTORY_FILE.exists():
            history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
        else:
            history = pd.DataFrame()

    forecast = generate_forecast(history, upwind=load_upwind())

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

    _print_outlook(forecast.seasons)

    # The workflow publishes this file to GitHub Pages.
    output_path = DATA_DIR / "forecast.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(forecast.to_web_dict(), indent=2, ensure_ascii=False)
    )
    print(f"\nForecast written to {output_path}")

    # Back up accumulated state to the data release (CI only — needs a token)
    if can_upload():
        if HISTORY_FILE.exists():
            upload_csv(HISTORY_FILE, HISTORY_ASSET)
        # Back up phenology data so the live onset features have it next run
        pheno_file = DATA_DIR / "phenology.csv"
        if pheno_file.exists():
            upload_csv(pheno_file, PHENOLOGY_ASSET)
        if UPWIND_FILE.exists():
            upload_csv(UPWIND_FILE, UPWIND_ASSET)


def _print_outlook(seasons: list[Any]) -> None:
    """The season outlook, one line per species, soonest start first."""
    if not seasons:
        return
    print("\nSeason outlook:")
    for season in sorted(seasons, key=lambda s: (s.state != "before", s.expected_start)):
        if season.state == "started":
            print(f"  {season.species:<10} started {season.expected_start}")
            continue
        line = f"  {season.species:<10} expected {season.expected_start}"
        if season.earliest is not None and season.latest is not None:
            line += f"  ({season.earliest} to {season.latest})"
        if season.p_start_7d is not None:
            line += f"  P(within 7 days) {season.p_start_7d:.0%}"
        print(f"{line}  [{season.basis}]")


def cmd_outlook() -> None:
    """Print the season outlook and how it would have done on the measured seasons.

    The outlook as the forecast publishes it (with the weather forecast when
    it can be fetched, on the history alone otherwise), then the replay
    report: start-date error by how many days ahead the projection was
    issued, walk-forward, next to the calendar. Thermal only — no model is
    trained, so it runs in seconds.
    """
    from .features import combined_weather
    from .outlook import OUTLOOK_FORECAST_DAYS, print_outlook_report, replay, season_outlook
    from .weather import fetch_weather_forecast

    print("=" * 60)
    print("SEASON OUTLOOK")
    print("=" * 60)
    history = sync_historical_data(HISTORY_FILE)
    if history.empty:
        print("No history.")
        return
    daily_temp = None
    try:
        combined = combined_weather(history, fetch_weather_forecast(OUTLOOK_FORECAST_DAYS))
        daily_temp = (
            combined["temperature_mean"]
            .groupby(pd.DatetimeIndex(combined.index).normalize()).mean().dropna().sort_index()
        )
    except Exception as exc:  # noqa: BLE001 — the report does not need the forecast
        print(f"Weather forecast unavailable ({exc}); outlook from the history alone")
    replayed = replay(history)
    from .outlook import residual_table

    _print_outlook(
        season_outlook(history, local_now(), daily_temp=daily_temp, table=residual_table(replayed))
    )
    print_outlook_report(replayed)


def cmd_backfill(days: int = 365) -> pd.DataFrame:
    """Collect *days* of history in one go — the bootstrap of an empty store.

    The same code path as the 3-hourly collection (the official LGL series,
    archive weather, the forecast API for the last five days), only over a
    longer span. NDVI reaches two years back; ``run-backfill`` fills the rest.
    """
    print("=" * 60)
    print(f"BACKFILL: collecting {days} days of history (3h resolution)")
    print("=" * 60)
    return update_history(collect(days=days))


def _describe_upwind(upwind: pd.DataFrame) -> str:
    """One line on the upwind file: which stations, over which dates."""
    if upwind.empty:
        return "none (upwind features NaN)"
    dates = pd.to_datetime(upwind["date"])
    stations = ", ".join(sorted(upwind["station"].unique()))
    return f"{stations}, {dates.min().date()} to {dates.max().date()}, {len(upwind)} rows"


def cmd_backfill_upwind(start_year: int = 2019) -> pd.DataFrame:
    """Fetch the upwind stations' full history from the LGL API.

    Merges into ``data/upwind.csv`` and, with a token, backs the file up to
    the data release so the next forecast run starts from it.
    """
    if not UPWIND_STATIONS:
        print("No upwind stations configured (src/upwind.py).")
        return pd.DataFrame()
    start = date(start_year, 1, 1)
    end = local_today()
    print("=" * 60)
    print(f"BACKFILL UPWIND STATIONS: {start} to {end}")
    print(f"  Stations: {', '.join(f'{c} ({n})' for c, n in UPWIND_STATIONS.items())}")
    print("=" * 60)
    sync_upwind()
    fetched = fetch_upwind_chunked(start, end)
    if fetched.empty:
        print("No upwind data returned.")
        return load_upwind()
    upwind = update_upwind(fetched)
    print(f"Upwind stations: {_describe_upwind(upwind)}")
    if can_upload():
        upload_csv(UPWIND_FILE, UPWIND_ASSET)
    else:
        print("No GITHUB_TOKEN — upwind file kept locally only")
    return upwind


def cmd_benchmark(
    horizon: int = FORECAST_DAYS,
    folds: int | None = None,
    classic: bool = False,
    species: list[str] | None = None,
    months: list[pd.Period] | None = None,
) -> None:
    """Walk-forward backtest of the real autoregressive forecast.

    *horizon* is the forecast depth in days — the thing the product ships and
    the thing this reports skill for, per day. It used to be passed straight in
    as the fold count, so asking for a longer horizon silently just changed how
    many months were sampled; *folds* is now its own argument.

    *classic* additionally runs the one-window-ahead evaluation with measured
    lag features, which is the only mode the DWD comparison and the onset-timing
    diagnostic make sense in.

    By default the six months the record is measured on
    (:data:`src.rollout.RECORD_MONTHS`), so the default command reproduces
    it. ``--folds N`` samples N months spread across the year instead, and
    ``--months`` names them. Six because three — an earlier default — picked
    Sep, Jan and May out of this history, all quiet months, and reported the
    model beating persistence from day 3 on; adding Nov, Mar and Jul reversed
    that. A fold count small enough to miss the season is worse than no
    benchmark, because it reads as evidence.
    """
    if months is None and folds is None:
        months = record_months()
    print("=" * 60)
    print("BENCHMARK: Autoregressive Rollout")
    print("=" * 60)
    if not HISTORY_FILE.exists():
        print("No history file found. Run 'collect' or 'backfill' first.")
        return

    history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    unique_days = len(set(pd.to_datetime(history["date"]).dt.date))
    print(f"History: {len(history)} rows, {unique_days} unique days")
    if months:
        print(f"Horizon: {horizon} days   Months: {', '.join(str(m) for m in months)}\n")
    else:
        print(f"Horizon: {horizon} days   Folds: {folds}\n")

    upwind = load_upwind()
    print(f"Upwind stations: {_describe_upwind(upwind)}")
    results = rollout_evaluate(
        history, horizon_days=horizon, n_folds=folds, species=species, months=months,
        upwind=upwind,
    )
    if results.empty:
        print("No rollout results.")
        return

    print_rollout_report(results, history)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_path = DATA_DIR / "benchmark_rollout.csv"
    results.to_csv(results_path, index=False)
    print(f"\nDetailed rollout results saved to {results_path}")

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_rollout_report(results, history)
    (DATA_DIR / "benchmark_rollout.txt").write_text(buf.getvalue())

    if not classic:
        return

    print("\n" + "=" * 60)
    print("BENCHMARK: One Window Ahead (measured lag features)")
    print("=" * 60)
    print("Not the shipped forecast — every row is handed the true recent counts.")
    lagged = temporal_split_evaluate(
        history, n_folds=folds, species=species, months=months, upwind=upwind
    )
    if lagged.empty:
        return
    print_evaluation_report(lagged)
    print_onset_window_report(lagged, history)
    compare_with_dwd(lagged)
    lagged.to_csv(DATA_DIR / "benchmark_results.csv", index=False)
    print(f"\nDetailed results saved to {DATA_DIR / 'benchmark_results.csv'}")


def _parse_onset_args(argv: list[str]) -> dict[str, Any]:
    """Parse ``benchmark-onset [--species A,B] [--years N] [--classic]``.

    Bare words are species names, for compatibility with the old form.
    """
    args: dict[str, Any] = {}
    rest = list(argv)
    bare: list[str] = []
    while rest:
        flag = rest.pop(0)
        if flag == "--classic":
            args["classic"] = True
        elif flag == "--years":
            args["years"] = int(rest.pop(0))
        elif flag == "--species":
            args["species"] = [s.strip() for s in rest.pop(0).split(",") if s.strip()]
        elif flag.startswith("-"):
            raise SystemExit(f"Unknown benchmark-onset option: {flag}")
        else:
            bare.append(flag)
    if bare:
        args["species"] = bare
    return args


def cmd_benchmark_onset(
    species: list[str] | None = None, years: int = 3, classic: bool = False
) -> None:
    """Score the shipped forecast around the season starts.

    The full benchmark samples months evenly across the history and almost
    never lands on a season start, so its aggregate says nothing about when a
    season begins or how hard. This runs the same direct rollout over the
    months containing each measured onset of the last *years* seasons and
    reports timing, amount and false starts per species-year and horizon
    (:mod:`src.onset_report`).

    *classic* runs the older one-window-ahead evaluation with measured lag
    features instead — a diagnostic for the phenology features, not the
    product.
    """
    from .onset_report import ONSET_SPECIES, print_onset_rollout_report

    print("=" * 60)
    print("BENCHMARK: Season-Start Accuracy")
    print("=" * 60)
    if not HISTORY_FILE.exists():
        print("No history file found. Run 'collect' or 'backfill' first.")
        return

    targets = species or list(ONSET_SPECIES)
    unknown = [s for s in targets if s not in ALL_SPECIES]
    if unknown:
        print(f"Unknown species: {', '.join(unknown)}")
        print(f"Known species: {', '.join(ALL_SPECIES)}")
        return

    history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    upwind = load_upwind()
    months = onset_focus_months(history, targets, window_days=WINDOW_DAYS, years=years)
    print(f"History: {len(history)} rows")
    print(f"Upwind stations: {_describe_upwind(upwind)}")
    print(f"Species: {', '.join(targets)}")
    print(f"Months: {', '.join(str(m) for m in months)} "
          f"(±{WINDOW_DAYS} d around each onset of the last {years} season(s))\n")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if classic:
        results = temporal_split_evaluate(history, species=targets, months=months, upwind=upwind)
        if results.empty:
            print("No evaluation results.")
            return
        print_evaluation_report(results)
        print_onset_window_report(results, history)
        results_path = DATA_DIR / "benchmark_onset_results.csv"
        results.to_csv(results_path, index=False)
        print(f"\nDetailed results saved to {results_path}")
        return

    results = rollout_evaluate(
        history, horizon_days=FORECAST_DAYS, species=targets, months=months, upwind=upwind
    )
    if results.empty:
        print("No rollout results.")
        return
    print_rollout_report(results, history)
    print_onset_rollout_report(results, history, targets, years)
    from .outlook import print_outlook_report, replay

    print_outlook_report(replay(history))
    results_path = DATA_DIR / "benchmark_onset_rollout.csv"
    results.to_csv(results_path, index=False)
    print(f"\nDetailed rollout results saved to {results_path}")


def cmd_calibrate(rebuild: bool = False) -> None:
    """Regenerate the confidence table from a rollout benchmark.

    Reads ``data/benchmark_rollout.csv`` — pass *rebuild* to run the benchmark
    first, over ``CALIBRATION_HORIZON_DAYS`` so the table also covers the
    horizons a stale run lands on. Writes ``src/confidence.json``, which is
    committed, so the published confidence is auditable against the run that
    produced it.

    Re-run this whenever the model changes materially; a stale table publishes
    the accuracy of a model that no longer exists.
    """
    from .confidence import CALIBRATION_HORIZON_DAYS, TABLE_PATH, breakdown, build_table

    print("=" * 60)
    print("CALIBRATE: Measured Forecast Confidence")
    print("=" * 60)

    results_path = DATA_DIR / "benchmark_rollout.csv"
    if rebuild or not results_path.exists():
        if not HISTORY_FILE.exists():
            print("No history file found. Run 'collect' or 'backfill' first.")
            return
        history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
        # Twice the shipped horizon: a stale run publishes the rate of the
        # horizon it really is, which can lie beyond FORECAST_DAYS.
        results = rollout_evaluate(
            history, horizon_days=CALIBRATION_HORIZON_DAYS, months=record_months(),
            upwind=load_upwind(),
        )
        if results.empty:
            print("Benchmark produced no results.")
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_path, index=False)
    else:
        results = pd.read_csv(results_path)
        print(f"Using existing {results_path}")

    table = build_table(results)
    TABLE_PATH.write_text(json.dumps(table, indent=2) + "\n")

    src = table["source"]
    print(f"\n  Scored {src['rows_scored']:,} predictions, "
          f"{src['rows_emitted']:,} of them emitted (value > 0.5)")
    print(f"  Exact level correct:      {table['overall']['exact']:.1%}")
    print(f"  Within one level:         {table['overall']['within_one']:.1%}")
    print("\n  Per horizon (offset from the overall rate):")
    for day, delta in sorted(table["horizon_delta"].items(), key=lambda kv: int(kv[0])):
        print(f"    day {day}: exact {delta['exact']:+.3f}   "
              f"within one {delta['within_one']:+.3f}")

    residuals = table.get("residuals", {})
    if residuals.get("window"):
        from .confidence import INTERVAL_COVERAGE, value_interval

        print(f"\n  Residual distributions stored for horizons "
              f"{', '.join(sorted(residuals['day_mean'], key=int))} (day means) — the level")
        print("  confidence is conformal: the residual mass inside the level's band.")
        for day in ("1", "5", str(max(int(k) for k in residuals["window"]))):
            interval = value_interval(table, 10.0, int(day))
            if interval:
                print(f"    day {day}: {INTERVAL_COVERAGE:.0%} interval for a value of 10 = "
                      f"{interval[0]:.1f} – {interval[1]:.1f}")

    print("\n  Per species x level (diagnostic only — NOT published).")
    print("  These vary 5x in-sample but do not generalise to a held-out fold,")
    print("  which is why the published table is flat:")
    detail = breakdown(results)
    print("    " + detail.to_string(index=False,
                                    float_format=lambda v: f"{v:.2f}").replace("\n", "\n    "))
    print(f"\nWritten to {TABLE_PATH}")


def cmd_backfill_species(species: str, start_year: int = 2019) -> None:
    """Add one taxon's full history — Munich and the upwind stations — to the stored data.

    For a taxon added after the history was built: the Munich values are
    fetched from the LGL API and joined to the weather and NDVI columns the
    history already carries for those windows, then merged into
    ``data/history.csv``; the upwind stations' values go into
    ``data/upwind.csv``. Both files are backed up to the data release when a
    token is present.
    """
    if species not in ALL_SPECIES:
        print(f"Unknown species {species!r}; known: {', '.join(ALL_SPECIES)}")
        return
    start = date(start_year, 1, 1)
    end = local_today()
    print("=" * 60)
    print(f"BACKFILL {species}: {start} to {end}")
    print("=" * 60)

    history = sync_historical_data(HISTORY_FILE)
    if history.empty:
        print("No history to attach the new taxon to; run backfill first.")
        return

    fetched = fetch_measurements_chunked(
        start, end, [LOCATION] + list(UPWIND_STATIONS), species=[species]
    )
    munich = fetched[fetched["station"] == LOCATION]
    print(f"\nMunich station: {len(munich)} windows")
    if not munich.empty:
        # Every non-pollen column of a window is shared by all its rows, so
        # the new taxon borrows them from any existing row there.
        carried = [c for c in history.columns if c not in ("date", "species", "value")]
        template = history.groupby("date")[carried].first()
        rows = munich[munich["date"].isin(template.index)][["date", "species", "value"]].copy()
        rows = rows.join(template, on="date")
        print(f"  {len(rows)} inside the history's weather coverage")
        update_history(rows[history.columns])
        if can_upload():
            upload_csv(HISTORY_FILE, HISTORY_ASSET)

    upwind_rows = fetched[fetched["station"] != LOCATION]
    if not upwind_rows.empty:
        sync_upwind()
        upwind = update_upwind(upwind_rows)
        print(f"Upwind stations: {_describe_upwind(upwind)}")
        if can_upload():
            upload_csv(UPWIND_FILE, UPWIND_ASSET)


# Where the pre-relabelling files are kept on the data release. Written once,
# never overwritten, so the merged series can be restored or compared.
HISTORY_BACKUP_ASSET = "history_merged.csv.gz"
UPWIND_BACKUP_ASSET = "upwind_merged.csv.gz"

# The relabelling refuses to run when the station series it fetched covers
# less of the history than this: a failed request must not turn a year of
# measurements into weather-only rows.
MIN_OFFICIAL_COVERAGE = 0.8


def cmd_backfill_labels(start_year: int = 2019) -> None:
    """Rewrite every measurement — Munich and upwind — from the official LGL series.

    Until November 2026 both files were filled from pollenscience.eu, whose
    station codes carry a second series beside the official one; the stored
    value was the larger of the two (TASKS.md, G.1). This fetches the
    official series for the whole span and replaces the values in place: the
    history keeps its windows and weather (:func:`src.backfill.relabel_history`),
    the upwind file is rewritten whole. The files as they were are stored on
    the release once, under ``history_merged.csv.gz`` / ``upwind_merged.csv.gz``.
    Idempotent.
    """
    from .backfill import relabel_history

    print("=" * 60)
    print("BACKFILL LABELS: measurements from the official LGL series")
    print("=" * 60)
    history = sync_historical_data(HISTORY_FILE)
    if history.empty:
        print("No history to relabel.")
        return
    sync_upwind()

    start = date(start_year, 1, 1)
    end = local_today()
    stations = [LOCATION] + list(UPWIND_STATIONS)
    print(f"Fetching {', '.join(stations)} from {start} to {end}...")
    official = fetch_measurements_chunked(start, end, stations, strict=True)
    munich = official[official["station"] == LOCATION][["date", "species", "value"]]

    windows = pd.DatetimeIndex(pd.to_datetime(history["date"]).unique())
    inside = windows[windows >= pd.Timestamp(start)]
    coverage = float(inside.isin(pd.DatetimeIndex(munich["date"].unique())).mean())
    print(f"The station series covers {coverage:.1%} of the history's windows since {start}")
    if coverage < MIN_OFFICIAL_COVERAGE:
        raise SystemExit(
            f"refusing to relabel: the fetched series covers only {coverage:.0%} of the history"
        )

    if can_upload():
        for path, name in ((HISTORY_FILE, HISTORY_BACKUP_ASSET), (UPWIND_FILE, UPWIND_BACKUP_ASSET)):
            if path.exists() and not asset_exists(name):
                upload_csv(path, name)

    untouched = history[pd.to_datetime(history["date"]) < pd.Timestamp(start)]
    relabelled, report = relabel_history(
        history[pd.to_datetime(history["date"]) >= pd.Timestamp(start)], munich
    )
    history = pd.concat([untouched, relabelled], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)
    print(f"History: {report['windows_reported']} windows relabelled "
          f"({report['values_changed']} of {report['values_compared']} values changed), "
          f"{report['windows_weather_only']} windows kept as weather only, "
          f"{report['station_windows_without_weather']} station windows have no weather yet")
    print(f"Wrote {len(history)} rows -> {HISTORY_FILE}")

    upwind = replace_upwind(official[official["station"] != LOCATION])
    print(f"Upwind stations: {_describe_upwind(upwind)}")

    if can_upload():
        upload_csv(HISTORY_FILE, HISTORY_ASSET)
        upload_csv(UPWIND_FILE, UPWIND_ASSET)
    else:
        print("No GITHUB_TOKEN — relabelled files kept locally only")


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


def _sync_phenology() -> None:
    """Download phenology.csv from the data release if absent locally (best-effort)."""
    pheno_file = DATA_DIR / "phenology.csv"
    if pheno_file.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_csv(PHENOLOGY_ASSET, pheno_file)


def cmd_run() -> None:
    """Run forecast pipeline: collect -> forecast (every 3 hours)."""
    # Runners start empty: restore history, models and phenology first.
    if not HISTORY_FILE.exists():
        sync_historical_data(HISTORY_FILE)
    download_models(MODELS_DIR)
    _sync_phenology()
    sync_upwind()
    history = cmd_collect()
    cmd_forecast(history)


def cmd_run_train() -> None:
    """Run full pipeline with training: collect -> train -> forecast (monthly).

    A retrain that fails is reported (``::error::`` so it shows on the run) but
    does not cost the forecast: the models on the release are restored and
    served for one more cycle, exactly as the 3-hourly run does.
    """
    if not HISTORY_FILE.exists():
        sync_historical_data(HISTORY_FILE)
    _sync_phenology()
    sync_upwind()
    # Monthly: refresh phenology onset data so the onset features stay current
    # (best-effort — failures are non-fatal; cmd_forecast backs it up to the release).
    try:
        cmd_phenology()
    except Exception as exc:
        print(f"Phenology refresh failed ({exc}); continuing with existing data.")
    history = cmd_collect()
    try:
        cmd_train(history)
    except Exception as exc:
        print(f"::error::Retrain failed, serving the previous models: {exc}")
        download_models(MODELS_DIR)
        cmd_forecast(history)
        raise SystemExit(1)
    upload_models(MODELS_DIR)
    cmd_forecast(history)


def _parse_date_args(argv: list[str]) -> tuple[date | None, date | None]:
    """``[START] [END]`` as ISO dates, both optional."""
    parsed = [date.fromisoformat(a) for a in argv[:2]]
    return (parsed[0] if parsed else None, parsed[1] if len(parsed) > 1 else None)


def cmd_backfill_weather(start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Rewrite every weather column of the history from the ERA5 archive."""
    from .backfill import refresh_weather

    print("=" * 60)
    print("BACKFILL WEATHER: rewrite weather columns from the archive")
    print("=" * 60)
    history = sync_historical_data(HISTORY_FILE)
    if history.empty:
        print("No history to refresh.")
        return history
    history = refresh_weather(history, start, end)
    history.to_csv(HISTORY_FILE, index=False)
    print(f"Wrote {len(history)} rows -> {HISTORY_FILE}")
    return history


def cmd_backfill_ndvi(start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Rewrite the NDVI columns of the history from MODIS composites."""
    from .backfill import refresh_ndvi

    print("=" * 60)
    print("BACKFILL NDVI: rewrite NDVI columns from MODIS")
    print("=" * 60)
    history = sync_historical_data(HISTORY_FILE)
    if history.empty:
        print("No history to refresh.")
        return history
    history = refresh_ndvi(history, start, end)
    history.to_csv(HISTORY_FILE, index=False)
    print(f"Wrote {len(history)} rows -> {HISTORY_FILE}")
    return history


def cmd_run_backfill() -> None:
    """Refresh weather and NDVI columns of the release history, then store it.

    Runs in Actions (``mode: backfill``). Either half failing leaves the other
    half's result in place: the history is written after each step and only
    uploaded once both have been attempted, so a partial refresh still lands
    and the failure still shows.
    """
    from .trainer import feature_coverage

    failures: list[str] = []
    for step in (cmd_backfill_weather, cmd_backfill_ndvi):
        try:
            step()
        except Exception as exc:  # noqa: BLE001 — report, continue, fail at the end
            print(f"::error::{step.__name__} failed: {exc}")
            failures.append(step.__name__)

    history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    report = feature_coverage(history)
    print("\nCoverage after backfill (features still missing on any row):")
    still = report[report["missing"] > 0]
    if still.empty:
        print("  none")
    for _, r in still.iterrows():
        print(f"  {r['feature']:<26} {r['missing']:>7.1%}")

    if can_upload():
        upload_csv(HISTORY_FILE, HISTORY_ASSET)
    else:
        print("No GITHUB_TOKEN — refreshed history kept locally only")

    if failures:
        raise SystemExit(f"backfill incomplete: {', '.join(failures)}")


def _parse_benchmark_args(argv: list[str]) -> dict[str, Any]:
    """Parse ``benchmark [horizon] [--folds N] [--months 2026-02,2026-04] [--species A,B] [--classic]``.

    The bare positional stays the horizon in days, which is what the flag was
    always documented to mean. ``--months`` names the test months explicitly
    instead of sampling them, which is how a result is reproduced after the
    history has grown (the sampled set shifts as months become eligible) and
    how the season starts are scored (they never fall on a sampled month).
    """
    args: dict[str, Any] = {}
    rest = list(argv)
    if rest and not rest[0].startswith("-"):
        args["horizon"] = int(rest.pop(0))
    while rest:
        flag = rest.pop(0)
        if flag == "--classic":
            args["classic"] = True
        elif flag == "--folds":
            args["folds"] = int(rest.pop(0))
        elif flag == "--months":
            args["months"] = [
                pd.Period(m.strip(), "M") for m in rest.pop(0).split(",") if m.strip()
            ]
        elif flag == "--horizon":
            args["horizon"] = int(rest.pop(0))
        elif flag == "--species":
            args["species"] = [s.strip() for s in rest.pop(0).split(",") if s.strip()]
        else:
            raise SystemExit(f"Unknown benchmark option: {flag}")
    return args


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
    elif command == "backfill-labels":
        cmd_backfill_labels(int(sys.argv[2]) if len(sys.argv) > 2 else 2019)
    elif command == "backfill-species":
        if len(sys.argv) < 3:
            print("Usage: backfill-species <species> [start_year]")
            return
        cmd_backfill_species(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2019)
    elif command == "backfill-upwind":
        start_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
        cmd_backfill_upwind(start_year)
    elif command == "backfill-weather":
        cmd_backfill_weather(*_parse_date_args(sys.argv[2:]))
    elif command == "backfill-ndvi":
        cmd_backfill_ndvi(*_parse_date_args(sys.argv[2:]))
    elif command == "run-backfill":
        cmd_run_backfill()
    elif command == "benchmark":
        cmd_benchmark(**_parse_benchmark_args(sys.argv[2:]))
    elif command == "benchmark-onset":
        cmd_benchmark_onset(**_parse_onset_args(sys.argv[2:]))
    elif command == "outlook":
        cmd_outlook()
    elif command == "calibrate":
        cmd_calibrate(rebuild="--rebuild" in sys.argv[2:])
    elif command == "dwd":
        cmd_dwd()
    elif command == "phenology":
        cmd_phenology()
    elif command == "run":
        cmd_run()
    elif command == "run-train":
        cmd_run_train()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
