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
| [pollenscience.eu](https://pollenscience.eu/api/measurements) | Pollen and spore measurements | Primary source: 3-hour pollen counts and the fungal-spore aggregate (`Fungus`) for Munich (station DEMUNC, 2019+), plus the four ePIN stations around it (Mindelheim, Altötting, Feucht, Viechtach) as upwind context — see [Upwind stations](#upwind-stations) and [Mould spores](#mould-spores) |
| [LGL Bayern](https://d1ppjuhp1nvtc2.cloudfront.net/measurements) | Pollen measurements | Alternative: real-time 3-hour pollen counts for Munich |
| [Open-Meteo](https://open-meteo.com/) | Weather forecast + historical archive | Hourly weather aggregated to 3-hour windows: temperature, precipitation, wind, humidity, sunshine, radiation, boundary layer height, dew point, CAPE, direct radiation, soil temperature + moisture (no API key required) |
| [MODIS (ORNL DAAC)](https://modis.ornl.gov/rst/api/v1) | NDVI / EVI satellite data | MOD13Q1 250 m 16-day vegetation indices, cubic-interpolated to daily resolution |
| [DWD Open Data](https://opendata.dwd.de/) | Pollenflug-Gefahrenindex + CDC Phenology | Official pollen danger levels for Oberbayern (partregion 121) — used both for benchmarking and an inference-time level blend; the phenology archive is fetched for reference but no longer feeds the model — see [Season onset](#season-onset) |
| [Copernicus CAMS](https://ads.atmosphere.copernicus.eu/) (optional) | European pollen forecast | Physics-based ensemble forecast for alder, birch, grass, mugwort, ragweed. **Off by default** — see [Optional: CAMS pollen feature](#optional-cams-pollen-feature) |

## Model

Each species gets a **three-stage pipeline** with species-specific hyperparameters:

1. **Stage 1 — XGBClassifier**: predicts P(pollen > 0). 200 estimators, learning rate 0.08, adaptive `scale_pos_weight`.
2. **Stage 2 — XGBRegressor**: predicts log1p(pollen count) via quantile regression (α 0.85–0.92 per species). That raised quantile is the one peak-emphasis mechanism; the `1 + √(value)` sample weights with tier bonuses it used to carry as well were removed in E.2 (they shifted the effective quantile above the nominal one, and the A/B without them is MAE 6.9 → 6.1).
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

**Combined prediction**: regression output is scaled by clamped activation probability. Out-of-season species are forced to zero, but only outside a **widened season window** (the core month range ± a one-month shoulder) so unusually early onsets — e.g. a warm-December hazel/alder bloom — are no longer structurally suppressed. On the 1–3 days the DWD Pollenflug-Gefahrenindex covers, the day's mean is **blended** one step toward DWD's expert forecast — the nudge is spread over the day's predicted windows, keeping their diurnal shape (most useful at season onset, when lag features are near zero).

**Pollen levels** (`none`, `low`, `moderate`, `high`, `very_high`) use the
species-specific DWD/ePIN thresholds, which are defined on **daily means**.
So a window's level is the level of its calendar day's mean — every window
of a day carries the same level, while the values keep the 3-hour shape.
Applying the daily thresholds to the 3-hour values directly, as the app did
until September 2026, read 15% of the pollen-bearing windows a level above
their day (a midday window runs at two to three times the daily mean) and
the night windows of any pollen day as `none`. The DWD blend and the
benchmark's level accuracy follow the same definition (`src/types.py`,
`daily_levels`).

**Real-time observation assimilation**: when the pipeline runs every 3 hours, forecast windows that already have real pollen measurements use the observed values instead of model predictions. This breaks the autoregressive error cascade and grounds lag features for subsequent windows in actual data.

## Features (65 total)

| Category | Count | Features |
|----------|-------|----------|
| Weather | 13 | temp mean, precipitation, wind speed, humidity, sunshine duration, boundary layer height, dew point, is_day, temp slope (3h), humidity slope (3h), temp variance (3h), soil temperature, soil moisture |
| Calendar | 2 | day-of-year sin/cos encoding |
| Time-of-day | 3 | hour of day (0/3/6/.../21), sin/cos hour encoding |
| Season | 1 | binary `season_active` per species |
| Weather-derived | 18 | GDD, 3/7-day rolling temp/sunshine/rain, temp deltas (1d/3d), consecutive warm hours, dry streak, temp×sunshine, dry+warm, warming trend, wind×dry+warm, wind direction sin/cos, transport south |
| NDVI | 2 | NDVI, NDVI delta (green-up rate) |
| Phenology | 3 | days since flowering onset (measured from history, per year — see below), onset anomaly and forcing above threshold (thermal readiness under the species' selected forcing rule, against that rule's walk-forward threshold) |
| Season load | 3 | last season's total vs. the seasons before it, the last two seasons vs. the seasons before them, last season vs. the one before — all log ratios per species, 0 when unknown; constant within a season year and computed only from seasons that ended before it (`src/season_load.py`) |
| Intra-day | 3 | temp vs. daily max (ratio), precipitation in prior window (binary), temperature rate of change |
| Lead | 1 | `lead_windows` — 3h windows between the last measurement and this one |
| Lag | 13 | pollen at t-1/t-2/t-3/t-8(24h)/t-16(48h)/t-24(72h)/t-56(7d), 24h + 7d rolling mean, 24h + 7d rolling max, morning average (today's earlier windows), days since active (all log-space) |
| Upwind | 3 | highest reading at any of the four upwind stations over the 24 h and 7 d before the origin (log-space), and the 24 h maximum minus Munich's own; anchored at the origin like the lag block, NaN where no station reported (`src/upwind.py`) |

Lag features carry about a third of all model gain, which is why forecast skill
is reported per horizon day — see [Evaluation](#evaluation).

**Twelve features were removed after measuring what they earned** (`temperature_max`,
`temperature_min`, `shortwave_radiation_sum`, `direct_radiation_sum`,
`wind_direction`, `cape_max`, `wind_from_south`, `wind_from_north`,
`transport_north`, `evi`, `cams_pollen`, plus raw `day_of_year` and `month`).
They accounted for 13.3% of measured gain, and dropping them made the model
slightly *better* at every horizon — gain share counts how often trees could
split on a feature, not whether it helped.

Note `WEATHER_COLUMNS` (what the pipeline carries) is deliberately larger than
`WEATHER_FEATURES` (what the model reads): `wind_direction`, for one, is no
longer a model input but is still needed to derive `wind_dir_sin/cos` and
`transport_south`. Removing it from the carried set would silently turn those
into constants, which is what `tests/test_feature_columns.py` guards.

### Direct forecasting

Lag features are anchored at the **forecast origin**, not at the window being
predicted, and a `lead_windows` feature says how far ahead that window is. So
all 40 windows of a forecast are predicted from one shared block of *measured*
values.

They used to be autoregressive: each window's prediction became the next
window's lag. Predictions carry an upward bias, so the bias went round the
loop — over six folds the mean prediction climbed 12.97 → 17.35 from day 1 to
day 5 while the mean actual was flat at 9.0 → 7.7, and MAE degraded 43% across
the horizon. Fixing the calibration (task 3.1) barely dented it, because it was
never a calibration problem.

Training covers seven leads (1, 4, 8, 16, 24, 32, 40) — dense early, where a
window's own recent past dominates, sparse later, where it barely matters — so
the model learns directly how much a day-old lag is worth versus a five-day-old
one. That is 145k rows per species and about 2.3 minutes for a full retrain.

Models record the feature columns they were fitted on, and `load_models`
refuses any that disagree. The pipeline checks out new code every run but keeps
serving the release's models until the next retrain, so a commit that changes
the feature set is briefly live against models fitted on the old one — and
XGBoost raises on that mismatch, which would take the forecast down rather than
degrade it. A refused species falls back to the no-model path for one cycle.

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

The lag block is built on the **3-hour time grid**, not by row (E.1). The
history has 231 gaps longer than a window — 184 of them whole days in
2019–2020, when the source reported once a day, the longest 13.5 days in
June 2026 — and a row-based `shift(8)` turned "24 h ago" into "8 rows ago,
whenever that was" through every one of them, in training and at forecast
time alike. Now `trainer.lag_block_on_grid` and `LagState.from_history`
both put the species' values on the full grid first: a lag of 8 is the
window 24 h earlier whether or not the station reported in between, and
`days_since_active` is a distance in time from the last window measured
above zero. A window inside an outage carries the same time of day one day
earlier when that was measured (pollen is diurnal, so yesterday's noon is a
better stand-in for a missing noon than this morning's 03:00), and the last
measurement of any kind otherwise; plain carry-forward was benchmarked too
and was worse (MAE 7.0 against 6.9, and hazel's onset timing 1.3→6.0 days
at day 2). The parity test cuts a gap into its fixture and requires both
paths to agree through it.

### Season onset

Three features are parameterised by when the season is expected to start:
`days_since_typical_onset`, `onset_anomaly` and `gdd_above_threshold`.
Together they carry 3–9% of model gain per species, so what feeds them
matters. `src/onset.py` derives it from the accumulated history rather than from
constants:

- **Onset** is the first of three consecutive days at or above the species'
  low/moderate boundary, inside its core season window. The window requirement
  is what keeps long-range transport out — a February birch cloud over Munich
  is not Munich's birches flowering. A run whose accumulated forcing is under
  half the median at the other years' onsets is skipped for the year's next
  run: the trees have never flowered with half the warmth, so it is transport
  (the 2025 birch "onset" on 4 March, six days of Italian birch then a week of
  zeros) or a calendar artefact (hazel already running on 1 January after a
  warm December). On the real history that moves exactly those two onsets.
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
- **The forcing rule behind that crossing is chosen per species.** One rule
  (forcing from 1 January at base 0 °C) used to project every species; it fit
  hazel and was worse than the calendar for alder and birch, because January
  warmth counts fully towards a tree that does not respond to it until March.
  At calibration time the rule is now picked from a grid of start dates
  (1 Jan – 1 Mar) and bases (0 / 3 / 5 °C) by leave-one-out over the seasons
  before the year being projected, and switched off — climatology only — when
  no rule beats the calendar. Current picks: hazel 1 Jan / 3 °C (3.4 d LOO),
  alder 15 Jan / 0 °C (6.5 d), birch 1 Mar / 5 °C (6.5 d), ash and poplar
  1 Mar / 0 °C, oak and nettle climatology. The retrain log prints them.

Every estimate for year *Y* is calibrated only on seasons before *Y*, so a
backtest never sees its own answer. `tests/test_onset.py` pins both properties
down; they are invisible in the output when they break.

The DWD phenology fetch (`python -m src.main phenology`) is unrelated to these
features now. It returns a single year of Munich observations with no Alnus at
all, which put Corylus 24 days and Alnus 17 days off their measured medians.

### Upwind stations

Pre-onset pollen is transport: in 2026 Munich measured 15–24 birch grains/m³
on 27 Feb – 5 Mar, five weeks before the local trees opened, and the first
two weeks of every heavy season were forecast at a tenth to a third of the
truth, because nothing the model read preceded the local count. The ePIN
network's other automatic stations see the same air hours to a day earlier.

`src/upwind.py` keeps the 3h measurements of the four stations within
~150 km — Mindelheim (85 km WSW), Altötting (85 km E), Feucht (140 km N),
Viechtach (140 km NE), one in each direction so whichever way the air arrives
from, a station saw it first — in `data/upwind.csv`, a second long-format
file beside the history. The collector appends the last two weeks on every
run and backs the file up to the data release; `backfill-upwind` fetches the
history from 2019. Three features come out of it, all anchored at the forecast
origin like the lag block: the highest reading at any station over the 24 h
and 7 d before the origin, and the 24 h maximum minus Munich's own (positive
= the region is ahead of the city). They are built on the time grid, so a
station outage shifts nothing, and they are NaN where no station reported.
Garmisch (75 km S, alpine, flowers later) and the two stations 250 km north
are not read.

What it bought, on the same six folds and onset months as everything else
(B.6 → B.5): general MAE 7.0 → 6.8, RMSE 44.0 → 43.6, level accuracy 75.7% →
76.3%, skill against persistence up 2–3 points at every horizon — the best
of the series on every number, mostly through poplar (March in-season MAE
82 → 70) and ash (27 → 24). Alder's day-1 onset timing goes from 6.0 to 2.0
days off (the 2025 start, 17 days late before, is 3 days late now) and false
starts fall 48 → 46 run-days. **What it did not buy is the heavy-year ramp**:
Betula 2024 and 2026 still get 6–18% of what arrives in days 5–9 of the
season, exactly as before. The stations do see those years coming —
Viechtach's 2024 and 2026 peaks are 16 000 and 17 500 grains against 1 000–
3 000 in the light years — but a 24 h or 7 d maximum at onset is a level the
model has seen in every year, and it still predicts the average of them. The
year's *amount* is a property of the season, not of the last day, and no
feature yet carries it (TASKS.md, B.5).

### Mould spores

`Fungus` is the ePIN samplers' fungal-spore aggregate, reported by
pollenscience.eu for Munich and the four upwind stations since 2019 and
forecast here as a twelfth taxon since F.1 (TASKS.md). It runs through the
same collector, model, benchmark and calibration as the pollen, with three
differences: it has no flowering season (the gate never zeroes it), there
is no DWD index to blend with, and its level thresholds — 50 / 150 / 300
spores/m³ for low / moderate / high — are quantiles of the Munich station's
own 2024+ daily means (median 46, 85th percentile ~150, 99th ~300), because
no official classification exists and the sampler's regime changed in 2024
(the 2019–2023 median is 9). The model's quantile target and stage-3
threshold were tuned on the rollout benchmark rather than inherited from the
pollen defaults; the pollen default (α 0.85) would have published a mould
forecast with a bias of +11 spores/m³ and only 12% skill against
persistence. On the standard six folds the mould model scores, against persistence:

| Horizon | MAE | Persistence MAE | Skill | Level acc. | Persistence level acc. | Bias |
|---------|-----|-----------------|-------|------------|------------------------|------|
| day 1 | 27.8 | 33.2 | +16% | 81.1% | 56.5% | +1.9 |
| day 3 | 29.5 | 39.4 | +25% | 78.0% | 56.5% | +0.9 |
| day 5 | 30.3 | 42.3 | +29% | 76.7% | 53.3% | +0.8 |

Spores are a persistent, slowly varying series — tomorrow looks like today
far more than for any pollen — so the skill is smaller than the pollen's
but the level is right four days in five. Of the emitted mould points the
level is exactly right 79% of the time and within one level 98.5%. The
residuals have the same spread as the pollen's (log-space standard
deviation 0.80 against 0.83), and the conformal confidence stays pooled
across all twelve taxa: a table split into pollen and spores calibrates
worse held-out (ECE 0.052 against 0.042). The pollen models are untouched
by the addition — their per-species benchmark rows are identical before
and after.

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
| `backfill-upwind [start_year]` | Fetch the upwind stations' history into `data/upwind.csv` (default: 2019, same pacing) and back it up to the data release |
| `backfill-weather [start] [end]` | Rewrite every weather column of the existing history from the Open-Meteo archive; pollen values untouched |
| `backfill-ndvi [start] [end]` | Rewrite the NDVI columns of the existing history from MODIS composites |
| `run-backfill` | Both of the above against the data release (what `mode: backfill` runs in Actions) |
| `benchmark [days]` | Walk-forward **rollout** of the real autoregressive forecast, scored per forecast day (default: 5). `--folds N` or `--months 2026-02,2026-04`, `--species A,B`, `--classic` |
| `benchmark-onset` | The shipped rollout over the months containing each measured season start of the last three seasons (default: Corylus, Alnus, Betula): timing, amount and false starts per species-year and horizon. `--species A,B`, `--years N`, `--classic` (old one-window-ahead diagnostic) |
| `calibrate [--rebuild]` | Regenerate `src/confidence.json` from `data/benchmark_rollout.csv`; `--rebuild` runs the rollout first, over 10 days so stale runs read a measured rate |
| `dwd` | Display the current DWD pollen danger index for Oberbayern |
| `phenology` | Download DWD phenology data and show flowering-onset statistics |
| `run` | Execute collect → forecast in sequence (every 3 hours) |
| `run-train` | Execute collect → train → forecast in sequence (monthly retraining); a refused retrain falls back to the released models |

## Evaluation

Two benchmarks, measuring two different things. The distinction matters more
than it sounds: **lag features carry ~50% of total model gain**, and they are
the only features whose quality depends on how far ahead you are forecasting.

### `benchmark` — autoregressive rollout (the shipped forecast)

`src/rollout.py` replays the forecast exactly as `generate_forecast` runs it:
every window is predicted directly from the measured lag block at the forecast
origin, through the same `src/features.py` code path production uses. A
forecast is launched from every day of a test month and rolled five days out,
and skill is reported **per forecast day** against a persistence baseline.
That comparison is the report's first block and its verdict — a forecast that
loses to "nothing changes" has no claim on a user's attention, whatever its
MAE — and everything else follows it.

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
origins, 80 680 scored predictions. Persistence is the same baseline throughout
— it does not depend on the model.

| Horizon | MAE | RMSE | Level acc. | Bias | Persistence MAE | Skill |
|---------|-----|------|-----------|------|-----------------|-------|
| day 1 | **6.4** | 45.1 | 75.2% | **−1.7** | 10.4 | **+38.3%** |
| day 2 | **6.3** | 44.9 | 74.8% | **−1.9** | 10.9 | **+42.2%** |
| day 3 | **6.1** | 42.0 | 74.5% | **−1.8** | 11.2 | **+45.8%** |
| day 4 | **5.9** | 40.2 | 74.2% | **−1.7** | 11.8 | **+49.5%** |
| day 5 | **5.9** | 39.6 | 73.6% | **−1.5** | 11.4 | **+48.2%** |

The model beats persistence at every horizon by 38–50%, with no decay across
the five days and a bias of about −1.7 grains/m³ (it leans low since E.2
removed the value weights that were pushing it up; see the stage table).
These are the eleven pollen taxa; the mould-spore taxon added in F.1 is
scored in its own section below, and a pooled figure that includes it
(MAE 8.2 at day 1) is not comparable with this table.
Level accuracy is measured on daily-mean levels (D.4; see *Model*): 74.5%
exact and 99.8% within one level, against persistence's 66.6% / 64.5% at
days 1 / 5. Under the old 3-hour definition the B.5 predictions scored
76.3% exact — and persistence 78.7%, because "none at night" was free; the
daily definition is the one under which the model's level skill is visible
at all.

These numbers are measured on the **complete** history (see *Data coverage*
below). Before the backfill the model scored 7.3 / 76.9%; on the complete
history the same model scored 7.8 / 75.6%. That earlier score was partly an
artefact: eight weather columns were NaN before March 2025 and filled to 0,
which gave the model a "2025 or later" flag, and all six folds lie in that
era. Restoring the NaN block recovers 7.3 exactly; dropping the eight
features on complete data stays at 7.8. The flag was a proxy for how heavy
recent seasons are; the three season-load features are the honest
replacement and take the model to 7.0 — better than the artefact — though
they leave the level accuracy where it was and the season-start amplitude
untouched (TASKS.md, B.6). The per-species onset rules and the transport-aware
onset detector (B.2/B.3) take it to 6.9, mostly through hazel, and the four
upwind stations (B.5) to 6.8. Building the lag block on the time grid
(E.1) costs 0.1 of that back for half a point of level accuracy, and
dropping the value weights that compounded with the quantile target (E.2)
takes it to 6.1.

Three fixes got here, each measured on these same folds:

| | day 1 MAE | day 5 MAE | day 1 bias | day 5 bias | day 5 skill |
|---|---|---|---|---|---|
| original | 11.9 | 17.1 | +6.5 | +13.2 | −50.2% |
| 3.1 stage-3 gate | 9.8 | 14.0 | +4.0 | +9.7 | −22.9% |
| 3.5 direct forecast | 7.9 | 7.4 | +0.3 | +0.6 | +35.6% |
| Phase 2 prune | **7.6** | **7.2** | **−0.2** | **+0.4** | **+37.3%** |
| Phase A complete data (same model) | 8.2 | 7.7 | +0.9 | +1.5 | +32.5% |
| C.1 season load | 7.4 | 6.8 | −0.2 | +0.1 | +40.7% |
| B.2 + B.3 onset calibration | 7.3 | 6.7 | −0.5 | −0.2 | +41.4% |
| B.4 rule-based readiness | 7.5 | 6.9 | +0.1 | +0.3 | +39.9% |
| B.6 onset-ramp weighting | 7.3 | 6.8 | +0.1 | +0.5 | +40.3% |
| B.5 upwind stations | 7.1 | 6.6 | −0.4 | −0.0 | +42.0% |
| E.1 time-based lags | 7.3 | 6.8 | −0.0 | +0.2 | +40.8% |
| E.2 one peak-emphasis mechanism | **6.4** | **5.9** | **−1.7** | **−1.5** | **+48.2%** |

**3.1** stopped the extreme regressor being consulted about ordinary windows.
**3.5** removed the feedback loop that let a residual bias compound into the
horizon. **Phase 2** cut 73 features to 60 — and the smaller model is better at
every horizon, not merely equal, so those features were adding variance rather
than signal.

Day-5 MAE has gone 17.1 → 5.9 and day-5 bias +13.2 → −1.5.

> **On fold counts.** An earlier version of this section reported three folds
> (Sep, Jan, May) and concluded that the model beat persistence from day 3 on
> and that horizon degradation had vanished. Both were artefacts of that sample
> — those three months are quiet ones. Adding Nov, Mar and Jul reversed both.
> Six folds is the default for this reason; prefer more, not fewer, when a
> result is going to be acted on.

### `benchmark-onset` — the season starts

The six sampled folds bracket the February hazel/alder start without covering
it, so the general benchmark cannot say when a season begins or how hard.
`benchmark-onset` runs the same direct rollout over the months containing
each measured onset of the last three seasons and reports, per species-year
and horizon, the timing of the first predicted 3-day run at or above the low
level, MAE and bias in the ±10-day window next to the MAE of predicting zero,
false starts, and the predicted/actual ratio by days since onset. The
current picture (`src/onset_report.py`): timing within 1–7 days at every
horizon for all three species, and amplitude a tenth to a third of what
arrives in heavy years for the first two weeks — the loss is in the amount,
not the date. Weighting the two weeks after each measured onset four times
in training (B.6) improved the general benchmark and alder's timing and did
not move the heavy-year ramp: nothing the model reads precedes the local
count, so it cannot tell a 25,000-grain year from a 3,000-grain one on day
one. TASKS.md tracks the numbers per change.

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
so pruning decisions have evidence behind them. On the complete history the
split is: lag 36%, weather-derived 24%, weather 12%, calendar 10%, season 5%,
season load 4%, NDVI 4%, phenology 4%, intra-day 2%. Gain share counts how often trees could
split on a feature, not whether the split helped — Phase 2 dropped 13% of
gain and got a better model, and Phase A's eight diurnal/soil features earn
2.6% between them while moving the benchmark by nothing.

### Data coverage

`train` also prints, per model input, the share of history rows on which it
is missing (NaN, or an exact 0.0 for NDVI, soil moisture, dew point and
boundary-layer height, which never legitimately read zero), and **refuses to
train** when any input is missing on more than half of the rows. The trainer
fills NaN with 0 before XGBoost sees it, so without this check an empty
column silently becomes a column of zeros — which is how ten of the sixty
features spent five to seven of the eight seasons before September 2026: the
six diurnal weather features existed from 2025-03-24, soil from 2026-06-17,
NDVI from 2024-01-01. `run-backfill` rewrites those columns from the archives
in place; it changes no pollen value and adds or removes no row.

One known gap remains: the archive returns no `boundary_layer_height` for
January–June 2024 (7% of rows).

## Deployment

Everything runs on GitHub Actions and GitHub Pages, at no cost — both are free
for public repositories.

### Pipeline

`.github/workflows/pipeline.yml` runs on `ubuntu-latest`:

| Trigger | Cron (UTC) | Command |
|---------|-----------|---------|
| Forecast | `17 2,5,8,11,14,17,20,23 * * *` (every 3h) | `python -m src.main run` |
| Retrain | `43 4 1 * *` (1st of the month) | `python -m src.main run-train` |
| Manual | `workflow_dispatch` with a `mode` input | `forecast`, `train`, `backfill` (`run-backfill`) or `backfill-upwind`; the last two publish nothing |

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
| `upwind.csv.gz` | 3h measurements of the four upwind stations (date, station, species, value) |
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

Confidence is **measured, not assumed**, and since D.5 it is also
**discriminative**: it differs from one prediction to the next, and the
difference means something. Two numbers are published per prediction, because
"is this level right" has two defensible readings that differ a lot:

| Field | Meaning | Typical range |
|-------|---------|---------------|
| `confidence` | P(the emitted level is exactly right) | 0.3 – 0.9 (mean ~0.7) |
| `confidence_within_one` | P(the truth is within one level of it) | ~0.9 – 0.99 |
| `value_low`, `value_high` | central 80% interval for `value` | ×0.09 – ×2.3 of the value at day 1 |

All of them are **conformal**: `src/confidence.json`, generated by
`python -m src.main calibrate` from the 10-day rollout benchmark, stores per
forecast horizon the distribution of the benchmark's residuals in log space —
`log1p(actual) − log1p(predicted)` — once for daily means (levels are daily
means) and once for 3-hour windows. A prediction's confidence is the share of
that distribution that keeps the truth inside its level's band: a daily mean of
30 grains/m³ sits in the middle of birch's *moderate* band (10–50) and gets
about 0.7; one of 12 sits a few grains above the threshold and gets about 0.5.
The interval is the same distribution's 10th and 90th percentile applied to
the window's value. The table is committed so every published number is
auditable against the run that produced it. **Re-run `calibrate` whenever the
model changes materially** — a stale table advertises the accuracy of a model
that no longer exists.

Windows with a real measurement (assimilated) publish 1.0 and an interval of
the value itself; a species with no trained model is marked down by half.

The horizon a window's confidence is read at is counted from the newest
measurement, not from today, because that is what the model's `lead_windows`
counts from and what the benchmark measured. On a normal run the two agree.
When the station has been silent for two days, a window "tomorrow" is a
three-day forecast, and it is published as one. The table is therefore
measured to twice the shipped horizon (`calibrate --rebuild` runs the rollout
over 10 days), so a run up to five days stale still reads a measured
distribution; a window beyond the furthest measured horizon gets that
horizon's rate halved, the same "never measured, so published low" rule as a
species without a model. The `status` block in the output says when this is
happening.

### What this replaced, and why

Two schemes came before. The original was `0.90 − 0.08 × day`, clipped, with
a +0.05 bonus for assimilated windows — invented numbers, wrong on both the
level (the emitted level was right 35% of the time under the 3-hour level
definition, 65% under the daily-mean one, not 90%) and the slope (accuracy
barely moves across the horizon — 67.6% at day 1, 62.6% at day 5, 58.7% at
day 10 — because the model is direct rather than recursive; the old decay
dropped 32 points over the first five days alone).

The second was a **flat table**: one measured rate for everything plus a
small per-horizon offset. It was flat because the evidence rejected anything
finer — keyed on species, level or both, a lookup table calibrates *worse* on
a held-out fold, and none of them correlates with being right. That ceiling
was the lookup table, not the model: what tells reliable predictions from
unreliable ones is *where the prediction sits relative to the thresholds*,
which the point estimate always carried and a table keyed on the level threw
away. Scored leave-one-fold-out on the 10-day rollout's emitted day-rows:

| Scheme | ECE | Correlation with being right | Brier |
|--------|-----|------------------------------|-------|
| old `0.90 − 0.08/day` | 0.393 | 0.013 | — |
| flat + horizon offset | 0.072 | −0.072 | 0.239 |
| **conformal, per horizon day** | **0.037** | **0.323** | **0.209** |
| conformal, pooled | 0.039 | 0.316 | 0.210 |
| conformal, per magnitude bin | 0.049 | 0.298 | 0.215 |
| conformal, per species | 0.073 | 0.225 | 0.229 |

Finer conditioning of the residuals (species, magnitude bin) still calibrates
worse held-out, for the same reason the finer flat tables did: those
differences are season- and year-specific. Reliability of the shipped scheme,
leave-one-fold-out: when it says 0.27 it is right 11% of the time, at 0.45
46%, at 0.65 70%, at 0.82 78%. The within-one figure stays as calibrated as
the flat one was (ECE 0.007 vs 0.008) and now also discriminates (correlation
0.17 vs 0.01). The 80% value interval covers 70–90% of the windows per
held-out fold.

## Output Format

`data/forecast.json` — the file GitHub Pages serves and the Vue frontend
reads — is species-centric, in the shape of the LGL Bayern measurement API
so the frontend can treat forecast points like measured ones. Each point
carries the emitted level and the calibrated confidence pair beside the
value:

```json
{
  "generated": "2026-03-04T05:00:00Z",
  "location": "DEMUNC",
  "measurements": [
    {
      "polle": "Alnus",
      "location": "DEMUNC",
      "data": [
        {
          "from": 1772600400,
          "to": 1772611200,
          "value": 35.2,
          "value_low": 3.4,
          "value_high": 81.0,
          "level": "moderate",
          "confidence": 0.71,
          "confidence_within_one": 0.99
        }
      ]
    }
  ]
}
```

`from`/`to` are Unix seconds bounding the 3-hour window. `confidence` is
P(the level is exactly right) and `confidence_within_one` P(the truth is
within one level of it); `value_low`/`value_high` bound the central 80%
interval for `value`, and are absent on a point that has no interval (a
value of 0). All from the calibration described above; a frontend that
ignores the extra keys keeps working. `ForecastOutput.to_dict()` is the
window-centric form of the same forecast (date → window → species), used
for logging and tests.

Beside `measurements` the file carries a `status` block saying what the run
was built from — the part of the output that used to be silent:

```json
"status": {
  "degraded": true,
  "observations": {
    "last": "2026-09-21T06:00:00",
    "age_windows": 1,
    "stale": false,
    "species": { "Alnus": { "last": "...", "age_windows": 1, "stale": false }, "...": {} }
  },
  "defaulted": {
    "dwd": "unavailable (HTTP 503); blend skipped",
    "ndvi": "fetch failed (timeout); zeros"
  }
}
```

- `observations` is the newest measurement (local time) and how many complete
  3-hour windows have passed since it without one; the headline is the worst
  species, and `species` has each one. `stale` is set from eight windows —
  a day of silence from the station. A stale run is not just flagged: its
  windows are forecast at the lead they really are, counted from the newest
  measurement, and their confidence is the confidence of that horizon (see
  [Forecast Confidence](#forecast-confidence)).
- `defaulted` names every input group that fell back to a default this run,
  with the reason: `ndvi` (fetch failed or no composites), `dwd` (no index to
  blend), `weather_soil` (Open-Meteo answered without soil variables),
  `upwind` (no station readings in the last 7 days, for some or all species),
  `model` (a species with no trained model, forecast by persistence), and
  `confidence` (no calibration table). CAMS is not listed: it has never been
  active, in training or live, so a zero there is the norm, not a fallback.
- `degraded` is true when observations are stale or anything was defaulted.
  The same line is printed in the run log (`Inputs: ...`).
