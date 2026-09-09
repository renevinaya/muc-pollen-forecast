# muc-pollen-forecast

ML-based pollen forecast for Munich at 3-hour resolution, using a three-stage XGBoost pipeline (classifier + quantile regressor + extreme regressor) trained on historical pollen measurements, weather data, satellite vegetation indices, and phenological observations.

**Species covered (11):** Alnus, Ambrosia, Artemisia, Betula, Corylus, Fraxinus, Poaceae, Populus, Quercus, Salix, Urtica

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Python backend (this repo)                              │
│                                                          │
│  1. Collector  — fetch pollen + weather + NDVI (3h res)  │
│     → parallel fetching via ThreadPoolExecutor            │
│     → append to data/history.csv                         │
│                                                          │
│  2. Trainer    — train three-stage XGBoost per species   │
│     → save models to models/*.joblib                     │
│                                                          │
│  3. Forecaster — predict 5-day pollen forecast           │
│     → real-time observation assimilation                 │
│     → write data/forecast.json                           │
│                                                          │
│  4. Evaluator  — autoregressive rollout, scored per day  │
│     → data/benchmark_rollout.csv                         │
└───────────────────────┬──────────────────────────────────┘
                        │ forecast.json
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Vue frontend (muc-pollen)                               │
│  → fetches forecast.json from GitHub Pages               │
└──────────────────────────────────────────────────────────┘
```

## Data Sources

| Source | API | What it provides |
|--------|-----|------------------|
| [pollenscience.eu](https://pollenscience.eu/api/measurements) | Pollen measurements | Primary source: 3-hour pollen counts for Munich (station DEMUNC, 2019+) |
| [LGL Bayern](https://d1ppjuhp1nvtc2.cloudfront.net/measurements) | Pollen measurements | Alternative: real-time 3-hour pollen counts for Munich |
| [Open-Meteo](https://open-meteo.com/) | Weather forecast + historical archive | Hourly weather aggregated to 3-hour windows: temperature, precipitation, wind, humidity, sunshine, radiation, boundary layer height, dew point, CAPE, direct radiation, soil temperature + moisture (no API key required) |
| [MODIS (ORNL DAAC)](https://modis.ornl.gov/rst/api/v1) | NDVI / EVI satellite data | MOD13Q1 250 m 16-day vegetation indices, cubic-interpolated to daily resolution |
| [DWD Open Data](https://opendata.dwd.de/) | Pollenflug-Gefahrenindex + CDC Phenology | Official pollen danger levels for Oberbayern (partregion 121) — used both for benchmarking and an inference-time level blend; the phenology archive is fetched for reference but no longer feeds the model — see [Season onset](#season-onset) |
| [Copernicus CAMS](https://ads.atmosphere.copernicus.eu/) (optional) | European pollen forecast | Physics-based ensemble forecast for alder, birch, grass, mugwort, ragweed. **Off by default** — see [Optional: CAMS pollen feature](#optional-cams-pollen-feature) |

## Model

Each species gets a **three-stage pipeline** with species-specific hyperparameters:

1. **Stage 1 — XGBClassifier**: predicts P(pollen > 0). 200 estimators, learning rate 0.08, adaptive `scale_pos_weight`.
2. **Stage 2 — XGBRegressor**: predicts log1p(pollen count) via quantile regression. Sample-weighted by `1 + √(value)` with tier bonuses (+8 for >100, +20 for >500, +40 for >1000).
3. **Stage 3 — Extreme Regressor** (optional): fitted only to high-pollen samples (>50), squared error in log space. Blended into Stage 2 in proportion to a **separate gate classifier for P(pollen > 50)**, ramping from 0.5 to 0.9 and capped at 70% weight.

   The gate used to be the Stage 1 classifier's P(pollen > 0) at a 0.6 cutoff. In peak season that sits near 1.0 for weeks, so a regressor that has never seen an ordinary window was given its full weight on one: measured over this history the blend fired on **20–28% of all windows**, and at those windows the truth was at or below the threshold ~75% of the time and exactly zero 14–28% of the time. With the dedicated gate it fires on 4–7% of windows, and the truth is above the threshold 97–99% of the time.

**Species-specific tuning:**

| Species | Clf depth | Reg depth | Reg estimators | Quantile α |
|---------|-----------|-----------|----------------|------------|
| Corylus, Alnus | 5 | 7 | 500 | 0.92 |
| Urtica, Poaceae | 5 | 6 | 400 | 0.90 |
| Quercus | 5 | 6 | 400 | 0.88 |
| Populus | 4 | 6 | 400 | 0.88 |
| Others (default) | 4 | 5 | 300 | 0.85 |

**Combined prediction**: regression output is scaled by clamped activation probability. Out-of-season species are forced to zero, but only outside a **widened season window** (the core month range ± a one-month shoulder) so unusually early onsets — e.g. a warm-December hazel/alder bloom — are no longer structurally suppressed. On the 1–3 days the DWD Pollenflug-Gefahrenindex covers, the emitted level is **blended** one step toward DWD's expert forecast (most useful at season onset, when lag features are near zero).

**Pollen levels** are assigned using species-specific thresholds (based on DWD/ePIN): `none`, `low`, `moderate`, `high`, `very_high`.

**Real-time observation assimilation**: when the pipeline runs every 3 hours, forecast windows that already have real pollen measurements use the observed values instead of model predictions. This breaks the autoregressive error cascade and grounds lag features for subsequent windows in actual data.

## Features (72 total)

| Category | Count | Features |
|----------|-------|----------|
| Weather | 19 | temp max/min/mean, precipitation, wind speed, wind direction, humidity, sunshine duration, shortwave radiation, boundary layer height, dew point, CAPE, direct radiation, is_day, temp slope (3h), humidity slope (3h), temp variance (3h), soil temperature (0–7 cm), soil moisture (0–7 cm) |
| Calendar | 4 | day of year, sin/cos encoding, month |
| Time-of-day | 3 | hour of day (0/3/6/.../21), sin/cos hour encoding |
| Season | 1 | binary `season_active` per species |
| Weather-derived | 23 | GDD + species GDD threshold, 3/7-day rolling temp/sunshine/rain, temp deltas (1d/3d), cold-to-warm flip, consecutive warm hours, dry streak, temp×sunshine, dry+warm, warming trend, wind×dry+warm, wind direction sin/cos, wind from south/north, transport south/north |
| NDVI | 3 | NDVI, EVI, NDVI delta (green-up rate) |
| Phenology | 2 | days since flowering onset (measured from history, per year — see below), onset anomaly (GDD-driven early/late signal against a walk-forward threshold) |
| CAMS | 1 | `cams_pollen` — Copernicus CAMS forecast for the species (0 when CAMS is inactive) |
| Intra-day | 3 | temp vs. daily max (ratio), precipitation in prior window (binary), temperature rate of change |
| Lag | 13 | pollen at t-1/t-2/t-3/t-8(24h)/t-16(48h)/t-24(72h)/t-56(7d), 24h + 7d rolling mean, 24h + 7d rolling max, morning average (today's earlier windows), days since active (all log-space) |

Lag features carry about half of all model gain, which is why forecast skill is
reported per horizon day — see [Evaluation](#evaluation).

Lag features are computed autoregressively during forecasting — each 3-hour window's prediction feeds into subsequent lag inputs.

### Train/serve parity

The trainer builds features in vectorised batches over the whole history; the
forecaster has to build them one window at a time, because each prediction
feeds the next window's lags. Those were two independent implementations of one
definition, and a mismatch is invisible in the output — a forecast built from
skewed features still looks like a forecast.

`src/features.py` now owns the row-wise half, and both the forecaster and the
rollout benchmark call it. `tests/test_feature_parity.py` builds the same
windows both ways and requires all 72 features to agree.

It found one immediately: `days_since_active` was computed with a cumulative
sum that does not advance while a species is inactive, so training saw a
**constant 0** on every row after the first active one, while the forecaster
served a live count. The feature is worth 6.5% of model gain once it actually
varies (0–980 windows).

### Season onset

Four features are parameterised by when the season is expected to start:
`days_since_typical_onset`, `onset_anomaly`, `gdd_above_threshold` and
`cold_to_warm_flip`. Together they carry 2–7% of model gain, so what feeds them
matters. `src/onset.py` derives it from the accumulated history rather than from
constants:

- **Onset** is the first of three consecutive days at or above the species'
  low/moderate boundary, inside its core season window. The window requirement
  is what keeps long-range transport out — a February birch cloud over Munich
  is not Munich's birches flowering.
- **The GDD threshold** is the median accumulated GDD at those onsets, in the
  same units the `gdd` feature uses. The hand-set constants it replaces crossed
  15–23 days *after* the observed onset, so the burst features opened their gate
  long after the season had begun.
- **The onset estimate is per year, and causal.** Until the year's warmth
  reaches the threshold it is the median of previous seasons; from the crossing
  onwards it is the crossing day. That switch is what carries the signal — it
  tells the model the season is running early or late as soon as the weather
  confirms it — and because it only looks backwards, training and forecasting
  compute it identically.

Every estimate for year *Y* is calibrated only on seasons before *Y*, so a
backtest never sees its own answer. `tests/test_onset.py` pins both properties
down; they are invisible in the output when they break.

The DWD phenology fetch (`python -m src.main phenology`) is unrelated to these
features now. It returns a single year of Munich observations with no Alnus at
all, which put Corylus 24 days and Alnus 17 days off their measured medians.

## Setup

```bash
uv sync
```

Requires Python ≥ 3.12 (the version the pipeline runs, and the minimum for current numpy/xgboost). Dependencies: httpx, pandas, numpy, xgboost, scikit-learn, joblib.

### Optional: CAMS pollen feature

The model can consume the [Copernicus CAMS](https://ads.atmosphere.copernicus.eu/)
physics-based European pollen forecast as an extra feature (`cams_pollen`),
which adds long-range-transport and season-onset signal a purely local model
can't see. It is **off by default and fail-open**: with no Atmosphere Data Store
(ADS) credentials or extra dependencies installed, the feature is simply `0`
everywhere and the model is unaffected — production never breaks.

To activate:

```bash
# 1. Install the optional dependencies (cdsapi, xarray, netCDF4)
uv sync --extra cams

# 2. Provide ADS credentials (or configure ~/.cdsapirc)
export CAMS_ADS_URL="https://ads.atmosphere.copernicus.eu/api"
export CAMS_ADS_KEY="<your-ads-key>"

# 3. Backfill historical CAMS into history.csv and retrain so the feature is
#    populated for training; it then becomes live at inference automatically.
```

## Usage

```bash
# Backfill historical data (run once to bootstrap)
python -m src.main backfill 365

# Train models on accumulated data (needs ≥ 14 data points per species)
python -m src.main train

# Generate forecast (writes data/forecast.json)
python -m src.main forecast

# 3-hourly pipeline (collect → forecast)
python -m src.main run

# Monthly pipeline (collect → train → forecast)
python -m src.main run-train

# Backtest the 5-day forecast the way it actually runs
python -m src.main benchmark 5 --folds 3
```

## Commands

| Command | Description |
|---------|-------------|
| `collect [days]` | Fetch recent pollen + weather + NDVI at 3h resolution, append to history (default: 14 days) |
| `train` | Train three-stage XGBoost models per species on all history |
| `forecast` | Generate 5-day forecast at 3h resolution using trained models |
| `backfill [days]` | Bulk import historical pollen, weather, and NDVI data (default: 365 days) |
| `backfill-ps [start_year]` | Bulk import from pollenscience.eu at 3h resolution (default: 2019, 5s rate limit) |
| `benchmark [days]` | Walk-forward **rollout** of the real autoregressive forecast, scored per forecast day (default: 5). `--folds N`, `--species A,B`, `--classic` |
| `benchmark-onset [species...]` | Walk-forward evaluation restricted to the months around each season start (default: Corylus, Alnus, Betula) |
| `dwd` | Display the current DWD pollen danger index for Oberbayern |
| `phenology` | Download DWD phenology data and show flowering-onset statistics |
| `run` | Execute collect → forecast in sequence (every 3 hours) |
| `run-train` | Execute collect → train → forecast in sequence (monthly retraining) |

## Evaluation

Two benchmarks, measuring two different things. The distinction matters more
than it sounds: **lag features carry ~50% of total model gain**, and they are
the only features whose quality depends on how far ahead you are forecasting.

### `benchmark` — autoregressive rollout (the shipped forecast)

`src/rollout.py` replays the forecast exactly as `generate_forecast` runs it:
lag features start from measurements before the forecast origin and are then
fed from the model's own predictions, through the same `src/features.py` code
path production uses. A forecast is launched from every day of a test month and
rolled five days out, and skill is reported **per forecast day** against a
persistence baseline.

Folds are spread across the calendar year rather than evenly along the
timeline. Even spacing put all three folds in August — dormant for every tree
species — so the benchmark was scoring an empty season.

What it deliberately does not simulate, both of which flatter the model and are
printed with the report rather than hidden:

- **Weather is actual, not forecast.** A real 5-day forecast also carries the
  weather model's error.
- **No DWD blend.** The DWD index covers only today + 2 days and is not
  archived, so it cannot be replayed historically.

### What it currently says

Six folds spread across the year (Sep, Nov, Jan, Mar, May, Jul), 184 forecast
origins, 80 680 scored predictions, before and after the stage-3 gate fix
(3.1). Persistence is the same baseline in both runs — it does not depend on
the model.

| Horizon | MAE (was) | RMSE (was) | Level acc. (was) | Bias (was) | Persistence MAE |
|---------|-----------|------------|------------------|------------|-----------------|
| day 1 | **9.8** (11.9) | 51.1 (52.0) | **75.6%** (73.7%) | **+4.0** (+6.5) | 10.4 |
| day 2 | **11.7** (14.7) | 52.3 (53.7) | **74.0%** (71.8%) | **+6.2** (+9.5) | 10.9 |
| day 3 | **12.4** (15.6) | 50.5 (52.3) | **73.7%** (71.5%) | **+7.4** (+11.0) | 11.2 |
| day 4 | **13.1** (16.3) | 51.2 (52.8) | **73.6%** (71.3%) | **+8.6** (+12.1) | 11.8 |
| day 5 | **14.0** (17.1) | 53.6 (54.6) | **73.4%** (71.2%) | **+9.7** (+13.2) | 11.4 |

The rollout's first job was to show that the model was **losing to persistence**
— "hold the last measured window flat" — at every horizon (−14% to −50% on
MAE), while its RMSE was far *better* than persistence (50–54 against 85–92).
That combination says the model captures large events well and pays for it with
a standing over-prediction on ordinary windows.

Fixing one of the three stacked upward pressures (3.1, the stage-3 gate) cut
MAE by ~18% at every horizon and bias by ~38%, without giving anything up at
the peaks. The model now beats persistence at **day 1** (+6.2%) and still
loses from day 2 out (−7.7%, −10.5%, −11.7%, −22.9%).

Two open problems, in order of size:

1. **Horizon degradation is the dominant issue.** MAE rises 43% from day 1 to
   day 5 and bias more than doubles (+4.0 → +9.7). 3.1 barely touched this
   (+44% before, +43% after), so it is the autoregressive lag cascade itself,
   not calibration leaking into it. Fraxinus (63.8 → 115.5) and Betula
   (25.2 → 57.0) degrade worst, and they are among the species that matter most.
2. **A +4.0 bias remains even at day 1.** Two upward pressures are still stacked
   — quantile α = 0.85–0.92 and √-value sample weights inside that same quantile
   loss — which is what task 3.2 unpicks.

> **On fold counts.** An earlier version of this section reported three folds
> (Sep, Jan, May) and concluded that the model beat persistence from day 3 on
> and that horizon degradation had essentially vanished. Both were artefacts of
> that sample: the three months it happened to pick are quiet ones. Adding Nov,
> Mar and Jul — birch/alder onset and the grass/nettle peak — reversed both
> conclusions. Six folds is now the default for this reason; prefer more, not
> fewer, when a result is going to be acted on.

### `benchmark --classic` — one window ahead (diagnostic only)

`temporal_split_evaluate` hands every test row the *measured* recent counts.
That is the right diagnostic for "are the weather and phenology features doing
anything", and the wrong one for "how good is the forecast": on the same fold
and species it reported MAE 76 / 37% level accuracy where the real day-1
forecast delivers MAE 121 / 22.5%, and day 5 delivers MAE 172 / 16.7%. The DWD
comparison and onset-timing diagnostics only make sense in this mode, so they
live here.

### Feature gain

`train` prints the share of XGBoost gain each feature and feature family earns,
so pruning decisions have evidence behind them. The current split is roughly:
lag 50%, weather-derived 20%, weather 12%, calendar 10%, with NDVI, intra-day
and phenology at 2–3% each and `cams_pollen` never split on.

## Deployment

Everything runs on GitHub Actions and GitHub Pages, at no cost — both are free
for public repositories.

### Pipeline

`.github/workflows/pipeline.yml` runs on `ubuntu-latest`:

| Trigger | Cron (UTC) | Command |
|---------|-----------|---------|
| Forecast | `17 2,5,8,11,14,17,20,23 * * *` (every 3h) | `python -m src.main run` |
| Retrain | `43 4 1 * *` (1st of the month) | `python -m src.main run-train` |
| Manual | `workflow_dispatch` with a `mode` input | either |

The schedules sit at `:17` and `:43` on purpose: GitHub queues scheduled
workflows and the top of the hour is the most congested slot. Scheduled runs
can still be delayed by several minutes under load — acceptable for a 3-hourly
forecast, but it is not a hard guarantee the way a hosted scheduler is.

A `concurrency` group serialises runs so two jobs never rewrite the same
release asset.

### State

Runners are ephemeral, so accumulated state lives on a GitHub release tagged
`data` (see `src/store.py`):

| Asset | Contents |
|-------|----------|
| `history.csv.gz` | Accumulated 3h pollen + weather + NDVI observations |
| `phenology.csv.gz` | DWD flowering-onset records |
| `models.tar.gz` | All trained `*.joblib` models |

Release assets are used rather than commits because `history.csv` is far too
large to commit on every run — git rejects any single file over 100 MB, and
eight commits a day would bloat the repository permanently. Gzip takes the CSV
to roughly a tenth of its size.

Reads are unauthenticated, so a local checkout picks up production history with
no credentials:

```bash
python -m src.main run    # downloads history + models, forecasts locally
```

Writes need `GITHUB_TOKEN`, which Actions injects automatically. Without a
token the pipeline still runs end to end and simply skips the backup step.

### Publishing

The forecast job writes `data/forecast.json` and force-pushes it as a single
orphan commit to the `gh-pages` branch, which GitHub Pages serves at:

```
https://renevinaya.github.io/muc-pollen-forecast/forecast.json
```

Force-pushing an orphan commit keeps the branch at exactly one commit, so the
branch never grows. The push has a second purpose: GitHub disables scheduled
workflows after 60 days without repository activity, and a push resets that
clock on every run.

Pages serves `Access-Control-Allow-Origin: *`, so the frontend can fetch the
file cross-origin.

### Environment Variables

| Variable | Set by | Description |
|----------|--------|-------------|
| `GITHUB_TOKEN` | Actions (automatic) | Required to write release assets and push `gh-pages`. Unset locally, in which case uploads are skipped. |
| `GITHUB_REPOSITORY` | Actions (automatic) | `owner/repo` holding the data release. Falls back to `renevinaya/muc-pollen-forecast`. |
| `DATA_REPO` | optional | Overrides `GITHUB_REPOSITORY` when pointing at a fork. |
| `DATA_RELEASE_TAG` | optional | Release tag holding the data assets (default: `data`). |
| `CAMS_ADS_URL` / `CAMS_ADS_KEY` | repo secrets (optional) | Activates the Copernicus CAMS feature. |

### Repository setup

One-time, in repository settings:

1. **Settings → Actions → General → Workflow permissions**: *Read and write
   permissions*. Without this the workflow's `contents: write` cannot be
   granted, and both the release upload and the `gh-pages` push fail.
2. **Settings → Pages → Source**: *Deploy from a branch* → `gh-pages` / `/ (root)`.
   The branch only exists after the first successful run.

## Forecast Confidence

Confidence scores start at **0.90** for day 1 and decrease by **0.08** per additional forecast day (0.90, 0.82, 0.74, 0.66, 0.58). If no trained model exists for a species, confidence is halved. Windows with real-time observations get a +0.05 confidence boost. Scores are clipped to [0.20, 0.95]. Species with predicted value ≤ 0.5 are filtered from the output.

## Output Format

The forecast JSON (consumed by the Vue frontend) uses 3-hour windows:

```json
{
  "generated": "2026-03-04T05:00:00.000Z",
  "location": "DEMUNC",
  "forecast": [
    {
      "date": "2026-03-04",
      "windows": [
        {
          "from": "06:00",
          "to": "09:00",
          "species": [
            {
              "name": "Alnus",
              "level": "moderate",
              "value": 35.2,
              "confidence": 0.9
            }
          ]
        }
      ]
    }
  ]
}
```

A secondary `to_web_dict()` format is also available, restructured as species-centric measurements with Unix timestamps matching the LGL Bayern API format.
