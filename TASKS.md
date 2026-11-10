# Open points

Review of 2026-11-10 on that day's release data. **Nothing below is
implemented.** The numbers come from one-off scripts run against
`history.csv`, `upwind.csv`, the released models and the public APIs; none
is a repo command yet, so the first step of each item is to turn its check
into one. Ordered by what blocks what: G.1 changes the labels every other
number in this file was measured on.

## G.1 The Munich series is a merge of several instruments — first

`fetch_pollenscience` reads the codes DEMUNC and DEBIED and keeps the larger
value per window, on the assumption that they are one station under two
names. They are not. pollenscience.eu serves up to two values per window
under each code. One of those series is the official ePIN station — the one
the LGL API returns for `locations=DEMUNC`, and the one the forecast is
named after. It is filed under DEMUNC in 2019 and from 2023 and under DEBIED
in 2020–2022 (the two APIs agree on every sampled window). Everything else
is a second instrument or a second processing, which LGL does not publish
(`locations=DEBIED` is empty). The history is the maximum over all of them
(94% of the rows where both codes report).

* The LGL series starts on 2019-01-01, has one value per window and 37
  taxa. Against it the history is identical on 85–95% of a year's rows,
  higher on 5–15%, never lower: 1.04 / 1.06 / 1.12 / 1.17 / 1.29 / 1.93 /
  1.66 / 1.02 times the official sum for 2019…2026. 8.5% of the history's
  rows have no official measurement at all (19% in 2026).
* Season totals, history / official: Alnus 2025 3.1, Urtica 2024 2.5,
  Quercus 2023 2.4, Poaceae 2024 2.2, Corylus 2024–25 1.4–1.5, Fungus
  2.2–2.5 in 2020–2023, 7.7 in 2024, 1.9 in 2025, 1.0 in 2026. Betula is
  1.00–1.02 throughout. The **daily level** differs from the official one
  on 6–24% of the pollen days in 2019–2023 and 2025, 45% in 2024, 35% in
  2026.
* What the second series contains: on 2025-02-04 alder at 3 579 / 7 640 /
  4 574 while the official station reports 0 in all eight windows; summed
  over 1–28 November 2024 oak at 21 498 against 223 and hazel at 13 849
  against 34; over 1–10 February 2025 oak at 34 308 against 0.
* Measured season starts move: Alnus 2025 4 Feb → 20 Feb, Alnus 2026 +6 d,
  Corylus 2023 −7 d, Quercus 2023 +20 d, Poaceae 2026 +7 d. On the 7-day
  onset rollout of G.2, retrained and scored on each, alder's day-1 timing
  error is 7.7 d on the merged labels and 3.3 d on the official ones: the
  2025 start scored "17 days late" was a date the official station never
  measured.
* The mould "regime change" (old point 2) is this merge: the second series
  dominates `Fungus` in 2020–2025 and is absent in 2026. The 50 / 150 / 300
  thresholds are quantiles of it.
* The four upwind stations have the same shape: two values in every window
  of April 2021 and April 2023, different in 25–44% of them, the larger sum
  1.7–4.4 times the smaller; one value in 2026. LGL serves one series for
  them too, equal to neither the maximum nor the minimum.
* The official station has outages the merge was papering over: 15–21
  whole days a year in 2023–2025, 50 in 2026 (34 d from 18 June; 9 d from
  2 February, just before the hazel and alder start).

To do: read Munich and the upwind stations from the LGL API (`src/pollen.py`
does already; six-week chunks work back to 2019), rewrite `value` in the
history and `upwind.csv` from it, leave outages as gaps (the lag block is
on the time grid), keep the second Munich series only as a candidate
*feature*. Then retrain, re-derive the mould thresholds, rerun both
benchmarks and `calibrate --rebuild`. Every figure in the README's
evaluation sections is on the merged labels.

## G.2 The season start, one week ahead

The shipped model form with leads trained to 56 windows, a 7-day rollout
over the onset months of the last three seasons (hazel, alder, birch,
grass), actual weather, merged labels:

| | d1 | d2 | d3 | d4 | d5 | d6 | d7 |
|---|---|---|---|---|---|---|---|
| MAE | 82.1 | 85.0 | 87.0 | 88.2 | 89.8 | 92.8 | 95.4 |
| skill vs persistence | −6% | +13% | +19% | +23% | +24% | +24% | +26% |
| timing error (d), hazel | 4.3 | 4.3 | 4.3 | 4.3 | 4.3 | 4.3 | 6.0 |
| alder | 7.7 | 7.7 | 7.7 | 7.3 | 4.3 | 7.0 | 7.0 |
| birch | 2.3 | 2.0 | 2.0 | 2.0 | 2.3 | 2.3 | 7.3 |
| grass | 3.0 | 2.7 | 2.0 | 2.0 | 2.3 | 1.3 | 3.3 |

The date holds to day 6 and slips at day 7 (birch 2.3 → 7.3 d), and in
these months the day-1 forecast loses to holding the last window. What
decides the timing is the amount: the forecast shows under half of what
arrives in the first ten days in six of the nine hazel, alder and birch
seasons, so its threshold crossing is late. The weather forecast is *not*
the limit. With the archived forecasts of Open-Meteo's previous-runs API
(Jan–Jun 2024) the daily mean temperature error is 0.5 °C at day 1 and
2.0 °C at day 7 (correlation 0.90), and the thermal crossing projected
seven days ahead lands within 0–1 day of the one computed from measured
temperatures, for all four species that have a rule.

**G.2a Ship seven days.** `FORECAST_DAYS` 7, `TRAINING_LEADS` to 56,
calibration to 14. The weather API returns every variable for 16 days and
the collector already fetches 7.

**G.2b The onset estimate is weaker than its log says.** The rule is the
argmin of a leave-one-out error over 15 candidates on 4–7 seasons, and the
retrain log prints that minimum (3.0–6.5 d today; 1.5–5 d when the rules
for 2023–2026 were chosen). Scored walk-forward on the official labels —
rule chosen on the seasons before *Y*, applied to *Y*, 2023–2026 — the
crossing is off by 7.8 d for hazel (climatology 14.6), 12.0 d for alder
(climatology **8.5**), 6.5 d for birch (6.5), 1.3 d for ash (7.1), 2.5 d
for poplar (2.6). The chosen rule changes from year to year (alder: 1 Jan /
3 °C, 15 Jan / 5, 15 Jan / 5, 15 Jan / 3). Fixed rules sit on plateaus the
selection does not find: alder 3.0–5.0 d for any start between 15 Jan and
1 Feb at any base, birch 2.8–4.2 d for 1–15 Feb at base 0–3 (best in
hindsight over four years, so optimistic — but a plateau, not a point).
To do: select on a smoothed criterion (a rule's error averaged with its
grid neighbours, or the median crossing over all rules within a day of the
best); print the walk-forward error, not the selection error; test a
chilling-dependent forcing requirement for hazel (2025 and 2026 started 17
and 12 d before the crossing). Eight seasons cannot calibrate that; DWD's
annual reporters can (Hasel 1930–2024, Schwarz-Erle 1936–2024, Hänge-Birke
1930–2024, Wiesen-Fuchsschwanz 1936–2024) with archive temperatures from
1940.

**G.2c The detector is clipped by the core season window.** Poplar's
"onset" falls on 1 or 2 March in five of eight years, with up to 20 of
February's days already above the threshold (2024, official series); oak's
on 1 or 2 April in three. Poplar's rule (from 1 March, 6.7 degree-days, LOO
3.0 d) is the calendar. The ramp weight and the three phenology features
are built on those dates. Search from the gate window and let the forcing
filter do what the window does now. `Fungus` gets an "onset" too (LOO 32 d)
and the ×4 ramp weight after it; exclude `SPORE_TAXA`.

**G.2d Production does not carry the estimate through the forecast.**
`build_context` computes `onset_by_species` from the history alone, so a
crossing inside the forecast window is not seen:
`days_since_typical_onset` reads 16 where training and the benchmark read
0 (hazel, issued 2026-02-11), 4 for 0 (birch, 2026-04-02), −11 for 0
(alder, 2024-02-01). `readiness_by_day` already uses the combined
temperature, and `_fold_context` extends the frame for exactly this
reason. The parity test cannot see it: its one-year fixture never
calibrates a threshold. Predictions moved by up to 4% in the four replays,
all of which had pollen in the lag block already; the pre-onset case is
untested.

**G.2e Publish the start, and score it as an event.** A species below 0.5
is absent from `forecast.json`, so "hazel starts around the 10th" cannot
be read from it. Add a per-species block — state, expected start, earliest
and latest, P(start within 7 days) — from the forcing to date plus the
16-day forecast plus climatological temperatures beyond, with the spread
from the walk-forward errors of G.2b (the D.5 construction). Give the
model the same projected date as a continuous `days_to_projected_onset`
in place of the feature that jumps at the crossing. In `benchmark-onset`
add, for every origin in the 30 days before a start: did the forecast
issued that day announce it within the next 7, and how many days ahead was
the first correct announcement.

**G.2f Stage 1 answers a different question.** Its target is P(window > 0)
and about half of its gain (48–58%) is the 7-day and 24-hour lag
statistics; no phenology feature is in the top eight for hazel, alder,
birch or grass. It confirms a season, it does not open one. Test P(daily
mean ≥ the low threshold) as the target.

**G.2g Leading indicators, measured.**
* *CAMS* is available without credentials from Open-Meteo's air-quality
  API (2021 on; the live forecast reaches about three days). Birch onset
  against Munich: +19, +2, −12, +4, −1, +3 d over 2021–2026; grass
  −4 … +16; daily rank correlation in season 0.64–0.90; alder never
  reaches 10 grains in five of six years. **It does not know the heavy
  years**: its birch totals are 1 119–2 138 every year against 2 817–24 525
  measured, and 2026, the record, is its third lowest. The old point 1 is
  therefore not "blocked on a CAMS backfill"; CAMS is a timing signal at
  best.
* *DWD immediate reporters* (flowering reports, updated daily; history
  from 1979 for hazel, 1992 birch, 2009 alder): 31–54% of the German
  stations report hazel flowering at least 7 days before Munich's pollen
  start (2024–26), 27–68% meadow foxtail before the grass start; for alder
  and birch 2–17% — the trap is ahead of the observers. A feature for
  hazel and grass only.
* *The other species.* On the official labels alder has never started
  before hazel (0 … +44 d, median +6), and birch follows ash by 1–21 d
  (median 12) and poplar by 13–39 d, in all eight years. The models read
  only their own taxon; a predecessor's 7-day maximum in the lag block is
  untested.

**G.2h Benchmark on the forecasts that were actually issued.** The
previous-runs API serves each hourly variable as it was forecast 1–7 days
earlier (2024 on). Replace the fold's weather with it per lead, at least
for the 2024+ folds: today a day-7 window is trained and scored on weather
known to the hour, and the confidence table inherits that.

**Tried in the review, not adoptable as it stands.** Scaling the forecast
by the ratio of measured to forecast daily mean over the day before the
origin (exponent 1.0 → 0.3 over days 1–7, clipped to 0.2–8) needs no
retraining. On the onset months: MAE 88.6 → 80.6 (day 1 82.1 → 64.0),
level accuracy 49.9 → 52.8%, timing error 4.3 → 1.5 d at day 1 and
3.3 → 2.8 d at day 5 — and false starts 1 → 11 run-days. On six general
folds (five taxa) it costs MAE 8.6 → 9.6 and a point of level accuracy,
and no restricted form (3-day ratio, half power, upward only) did better
than 8.6 → 9.0. It is a season-start device; the next test is to gate it
to the weeks after a detected start.

## G.3 Inputs that differ between training and serving

* **Weather exists only where pollen does.** The history has one row per
  measured window, so forcing, `gdd` and every rolling weather feature
  skip the station's gaps, and a day's "mean" temperature is the mean of
  whichever windows were measured. In January–April 2019 and 2020, 69 and
  50 of the 120 days hold only part of the day: every day of February 2019
  and of January 2020 is the 12:00 window alone, which on complete days
  runs 2.0–3.1 °C above the day's mean. That is the forcing period of
  hazel and alder, and those two seasons calibrate every onset threshold.
  G.1 adds gaps to 2023–2026 (9 absent days before 30 April in 2025 and in
  2026). Store the weather once per window on the complete grid,
  independent of the pollen rows.
* **NDVI.** The history holds a two-sided cubic interpolation written by
  the September backfill: through June 2026 the value changes every day
  and `ndvi_delta` is never zero, using a composite that did not exist
  yet. Serving holds the last composite flat: since the backfill the value
  is unchanged on 75% of days and `ndvi_delta` is exactly 0 on 79%. The
  pair earns 3.2% of gain and 4.8% of all splits, and the benchmark reads
  the look-ahead version. Make it as-of (the last composite that was
  available, with its latency) in the history and the fold context, then
  decide whether it stays. The composites cache is a local file on an
  ephemeral runner, so every 3-hourly run refetches three years from ORNL.
* **Temperature.** The forecast model's analysis runs 0.83 °C above the
  archive values the history is trained on (Jan–Jun 2024, daily means, MAE
  0.86, correlation 0.996): 13% more degree-days above 5 °C. Thresholds
  are calibrated on the archive; the last five days and every forecast day
  accumulate on the forecast model. Correct the offset or train on the
  historical-forecast API. Soil and boundary-layer height come from
  different models and depths on the two endpoints and were not compared
  (the archive API was rate-limited on the day).
* **HTTP 429.** `_get_weather` reads any 4xx as "soil parameters rejected"
  and retries without them; Open-Meteo's daily-limit answer is a 429. On a
  shared address that is a dead run, or a forecast without soil features
  flagged for the wrong reason. Retry it like a 5xx.

## G.4 Unnecessary or inefficient

* **Features.** Gain in the released models: six of the seven point lags
  (all but `pollen_lag_1`) and `pollen_morning_avg` earn 1.7% between them
  against 32% for the four rolling statistics; `hour_sin`, `hour_cos`,
  `is_day` 1.6% beside `hour_of_day`; `gdd_above_threshold` 1.25%, a
  monotone function of `onset_anomaly` within a year; the three
  within-window slopes 1.5%; `upwind_lead_8` 0.23%. **They are not free as
  a block**: dropping those fifteen and the two NDVI features (65 → 48),
  six folds, five taxa, costs MAE 8.6 → 8.8 and level accuracy 64.5 →
  62.9%, most of it at day 1 (9.1 → 9.7) and in alder (55.7 → 63.0; grass
  improves, 14.9 → 13.5) for a 17% faster retrain. The point lags are the
  likely cost, and NDVI's look-ahead confounds the block. Test one family
  at a time, after G.3.
* **Fetched and stored, never read:** hourly `shortwave_radiation`, `cape`
  and `direct_radiation` (`cape_max` is NaN on 99.7% of rows),
  `temperature_max`/`_min`, `evi`, `cams_pollen` (NaN on 94.5%, 0
  elsewhere), `day_of_year`, `month`; the stored calendar and hour columns
  are functions of the timestamp.
* **Steps that feed nothing:** the phenology asset is downloaded on every
  run, uploaded on every run and refetched from DWD monthly, and is not a
  model input (the fetch can only match the `_1930_2018_hist` file, never
  `_akt`, and does not map Schwarz-Erle — which is why it looked empty);
  `fetch_cams_forecast` runs twice a run; the `cams` extra (cdsapi, xarray,
  netCDF4) is superseded by G.2g.
* **A minute per run in `build_context`:** `_add_weather_derived_features`
  runs twelve times over the whole eight-year frame with Python streak
  loops (41 s; only `consecutive_warm_hrs` differs by taxon) and
  `select_forcing_rule` 192 times (12 s) for an answer that changes once a
  year. The 480 predictions take about 6 s. The same cost recurs per taxon
  and fold in both benchmarks.
* **`history.csv`** is 90 MB because each window's weather is stored
  twelve times; pollen (long) plus weather (one row per window) is about a
  sixth of it, and is what the first point of G.3 needs anyway.

## G.5 Carried over

* **The heavy-year ramp.** Betula 2024 and 2026 are still at 0.07–0.10 in
  days 5–9. Birch is the one taxon G.1 leaves untouched and CAMS does not
  carry the amount (G.2g); the correction above lifts them only to 0.09
  and 0.22.
* **Parked variants** on `claude/forecast-app-review-xwisqv`: A.5, B.8,
  B.7, verdicts below. B.7 again once twelve seasons exist.

## Rules that still hold

* Nothing that changes model behaviour merges without a benchmark that can
  see the difference: `benchmark` (six folds, Sep/Nov/Jan/Mar/May/Jul, 184
  origins) for the general forecast, `benchmark-onset` for the season starts
  (timing, ramp ratios, false starts), one dedicated commit per step.
* Feature changes are edits to the lists in `src/types.py` plus a retrain;
  data changes are a backfill plus a retrain. `calibrate --rebuild` after
  any model change, or the published confidence describes a model that no
  longer exists.
* Two XGBoost benchmarks on one 4-core machine starve each other; cap each
  at `OMP_NUM_THREADS=2`.

# Record

Same six folds throughout (184 origins, 80 680 predictions over five days).
Details of each step are in the README's *What it currently says*,
*Season onset*, *Upwind stations*, *Forecast Confidence* and *Mould spores*
sections and in the commit messages.

| Stage | day-1 MAE | day-5 MAE | day-1 bias | day-5 bias | day-5 skill | level acc. |
|---|---|---|---|---|---|---|
| original | 11.9 | 17.1 | +6.5 | +13.2 | −50.2% | — |
| 3.1 extreme gate | 9.8 | 14.0 | +4.0 | +9.7 | −22.9% | — |
| 3.5 direct forecast | 7.9 | 7.4 | +0.3 | +0.6 | +35.6% | — |
| Phase 2 pruning (73 → 60 features) | 7.6 | 7.2 | −0.2 | +0.4 | +37.3% | — |
| Phase A complete history (same model) | 8.2 | 7.7 | +0.9 | +1.5 | +32.5% | — |
| C.1 season load | 7.4 | 6.8 | −0.2 | +0.1 | +40.7% | — |
| B.2 + B.3 onset calibration | 7.3 | 6.7 | −0.5 | −0.2 | +41.4% | — |
| B.4 rule-based readiness | 7.5 | 6.9 | +0.1 | +0.3 | +39.9% | 75.6% (3 h) |
| B.6 onset-ramp weighting | 7.3 | 6.8 | +0.1 | +0.5 | +40.3% | 75.7% (3 h) |
| B.5 upwind stations (65 features) | 7.1 | 6.6 | −0.4 | −0.0 | +42.0% | 76.3% (3 h) / 72.8% (daily, D.4) |
| E.1 time-based lag block | 7.3 | 6.8 | −0.0 | +0.2 | +40.8% | 73.3% |
| E.2 one peak-emphasis mechanism | **6.4** | **5.9** | −1.7 | −1.5 | **+48.2%** | **74.5%** |

Level accuracy is on daily-mean levels from D.4 on (the 3-hour definition
before it let persistence score 78.7% by predicting "none" at night).
Persistence: MAE 10.4 / 11.4, level 66.6% / 64.5% at days 1 / 5.

Mould (F.1, `Fungus`, separate model, pollen rows unchanged): MAE 27.8 /
29.5 / 30.3 at days 1 / 3 / 5 against persistence 33.2 / 39.4 / 42.3, level
accuracy 81.1 / 78.0 / 76.7% against 56.5 / 56.5 / 53.3%.

## Done

- **Phase 1** — rollout benchmark per horizon, feature-gain report,
  train/serve parity test, shared row-wise feature assembly
  (`src/features.py`); `days_since_active` was a constant 0 in training.
- **3.1** — stage 3 gated on a dedicated P(value > threshold) classifier.
- **3.5** — direct multi-horizon forecast with `lead_windows`; horizon decay
  +43% → −6%.
- **Phase 2** — 73 → 60 features; the smaller model was better at every
  horizon.
- **Phase A** — weather, NDVI and soil backfilled to 2019; coverage guard at
  retrain; 0.5 MAE of the earlier score was an accidental "recent years"
  flag.
- **C.1** — three interannual season-load features.
- **B.1** — `benchmark-onset`: the shipped rollout around the season starts.
- **B.2 + B.3** — transport-aware onset detection; per-species forcing rules
  chosen leave-one-out (hazel 1 Jan / 0 °C, alder 15 Jan / 0 °C, birch
  1 Mar / 5 °C, ...).
- **B.4** — readiness features follow the per-species rule;
  `cold_to_warm_flip` dropped.
- **B.6** — onset-ramp weighting (first 14 days after the measured onset
  weigh ×4); timing better, amplitude unchanged.
- **B.5** — four upwind ePIN stations (Mindelheim, Altötting, Feucht,
  Viechtach) as a time-gridded feature block; alder day-1 onset timing
  6.0 → 2.0 d.
- **D.4** — levels are daily-mean levels, in the forecaster and both
  benchmarks; exact 76.3% → 72.8% but within-one 96.2% → 99.5% and
  over-predicted levels 16.4% → 10.1%.
- **D.1** — level and confidence published per point.
- **D.2 / D.3** — origin anchored at the newest measurement, confidence read
  at the horizon a stale run really is, calibration to 10 days; `status`
  block with observation age and every defaulted input group.
- **D.5** — conformal confidence and 80% value intervals from per-horizon
  residual distributions; correlation with being right −0.07 → 0.32, ECE
  0.072 → 0.037.
- **E.4** — skill vs persistence is the first block of the benchmark report.
- **E.1** — lag block on the 3 h grid on both paths; same-hour-yesterday
  fill inside gaps (plain carry-forward and dropping gap-spanning training
  rows both lost).
- **E.2** — the raised quantile is the one peak-emphasis mechanism; the
  `1 + √value` weights compounded with it. MAE 6.9 → 6.1, false starts
  43 → 8, birch onset d1 11.7 → 2.3 d; bias −0.1 → −1.7.
- **E.3** — log-space probability scaling benchmarked against a hurdle form
  (6.6 / 73.5%) and a plain threshold (6.9 / 72.0%) and kept (6.1 / 74.5%).
- **F.1** — mould forecast: `Fungus` backfilled 2019–2026 for Munich and the
  upwind stations, thresholds from the 2024+ regime, quantile 0.70 / depth
  5 / 600 trees tuned on the benchmark (the pollen default 0.85 was a +11
  bias at 12% skill). The web app needed no change: it reads the current
  load from the same source and took the new series as it was.

## Tried, not adopted

- **A.5 NaN passthrough** (XGBoost sees missing raw inputs instead of 0):
  general a wash (6.79 → 6.77), onset false starts 46 → 60.
- **B.8 upwind season load** (cumulative upwind features): general 7.07,
  false starts 60, heavy-year ramp unchanged — season load is a mid-season
  signal, not an onset-amplitude one.
- **B.7 December continuity** (forcing from 1 Nov / 1 Dec): the autumn rules
  lose the leave-one-out for hazel (8.6 d vs 4.0 d) and alder (21.5 vs
  6.5 d); the November turn alone raised false starts 46 → 59.
- **E.1 variants**: plain carry-forward (MAE 7.0, hazel onset d2 1.3 → 6.0 d)
  and dropping every gap-spanning training block (level 72.7%).
- **E.2 variants**: value weights with median regression (RMSE 45.5, bias
  −3.9); quantile +0.03 (bias −0.8 at +0.4 MAE).
- **E.3 variants**: hurdle scaling after `expm1`, plain threshold at p = 0.5.
- **F.1 variants**: quantile 0.5–0.95 (MAE rises monotonically), stage-3
  threshold none–400 (±0.4), tree depth 3–7 (±0.5).
