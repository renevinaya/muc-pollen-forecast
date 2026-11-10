# Open points

State of 2026-11-10, after the G series (record below). Everything here is
measured on the official LGL series and the six record months unless it
says otherwise. Ordered by what it is expected to buy.

Production is on this state: the data release was relabelled from the LGL
series and the twelve models retrained on 2026-11-10 (pipeline runs 623
and 624), and the forecast published since then is seven days long and
carries the `seasons` block.

## H.1 The season start in the forecast's own curve

The date of a start is projected by the season outlook (5.0 days off a
week ahead, calendar 9.2). The 3-hour forecast is a different matter.
`benchmark-onset`, the live model, official labels — first predicted
3-day run at or above the low level against the measured start, by
horizon day:

| | 2024 | 2025 | 2026 |
|---|---|---|---|
| hazel | d1 0, d2 +1, d3–7 not called | d1 +14, then not called | not called |
| alder | +2 … +3 at every horizon | d1 +13, then not called | not called |
| birch | +4 … +7 | −1 … 0 to d6; 14 d early at d7 | −2 … 0 |
| grass | −2 … +3 to d6; +8 at d7 | 0 … +2 | not called |

Birch and grass: 1.3–2.3 and 0–1 days off out to day 6, 7.0 and 5.0 at
day 7. Hazel and alder: it is the amount, not the date, both times. The
2025 "start" on 8 February is three days at 10–17 grains, a station gap
and a week of zeros; the forecast starts both on 21 February, a day after
the season that lasted did. In 2026 both came back from a nine-day outage at
130–380 grains a day against a day-1 forecast of 13–24. Grass 2026 hovers
at the threshold (3–15 against 5). In these months day 1 loses to
persistence (MAE 91.7 against 80.0); days 2–7 win by 13–27%; false starts
are 7 run-days, all birch. What is left:

* **Day 1 of a heavy start.** On 13 February 2026 hazel's lag block read
  264 in the last window and 76 over the last 24 hours, and the forecast
  for the day was 24; the day brought 329. In training a block with that
  24-hour mean is followed by a median of 71 and a mean of 134. The nine
  silent days before it were the suspect — the 7-day statistics were those
  of the zeros carried through the outage — and are not the cause: with
  the statistics over measured windows (G.5) the same days are forecast at
  13–24 instead of 9–24. What holds the forecast down is everything that
  says "nothing yet" (`gdd` at 2.65, `days_since_typical_onset` at 1)
  outvoting the measurement. In these months day 1 loses to persistence
  by 14%. Scaling the forecast by yesterday's error fixes exactly this
  and breaks the general folds (*Tried, not adopted*); a version confined
  to the fortnight after a detected start is untested.
* **The projected date as a model input.** Replacing the estimate behind
  `days_since_typical_onset` with the outlook's continuous projection did
  nothing (*Tried, not adopted*). Untested: the projection as a feature of
  its own — days *to* the projected start, clipped — beside the three that
  exist.
* **Stage 1 answers a different question.** Its target is P(window > 0)
  and about half of its gain is the 7-day and 24-hour lag statistics; no
  phenology feature is in its top eight for hazel, alder, birch or grass.
  It confirms a season, it does not open one. Test P(daily mean ≥ the low
  threshold) as the target.
* **Per-species quantiles on the official series.** They were tuned on the
  merged labels (0.92 for hazel and alder), and the mould quantile turned
  out to be 0.15 off on the official ones (G.1b). Per taxon over days 1–7
  the bias is −4.4 for alder and −7.5 for poplar against +0.4 for hazel
  and grass.
* **Hazel's date.** 9.0 d a week ahead, and the pooled 80% interval covers
  under half of its starts: it is the first of the year, with the least
  warmth behind it to read. A chilling-dependent forcing requirement
  cannot be calibrated on eight seasons; DWD's annual reporters can (Hasel
  1930–2024, Schwarz-Erle 1936–2024, Hänge-Birke 1930–2024,
  Wiesen-Fuchsschwanz 1936–2024) with archive temperatures from 1940.
* **Leading indicators, measured in the review, not built.** DWD's
  immediate reporters: 31–54% of the German stations report hazel
  flowering at least 7 days before Munich's pollen start (2024–26),
  27–68% meadow foxtail before the grass start; for alder and birch 2–17%
  — a feature for hazel and grass only. The other species: alder has never
  started before hazel (0 … +44 d, median +6), birch follows ash by
  1–21 d (median 12) and poplar by 13–39 d in all eight years, and the
  models read only their own taxon.
* **Score the announcement as an event** in `benchmark-onset`: for every
  origin in the 30 days before a start, did the forecast issued that day
  put the start inside the next 7. The outlook's replay scores the date
  and P(start within 7 days); the model's curve is only scored by the
  table above.

## H.2 Inputs that still differ between training and serving

* **The weather grid.** `backfill-weather` can complete it (`c2c300d`) and
  has **not** been run on the release. 1 630 windows have no row at all
  (2019: 1 177, 2020: 383 — the months the station reported once a day —
  2021: 19, 2023: 2, 2026: 49), so every day of February 2019 and of
  January 2020 has the 12:00 window's temperature for its mean, 2.0–3.1 °C
  above a real one, in the forcing period of hazel and alder. Filled
  locally and replayed, the more correct temperatures do not score better:

  | walk-forward, mean abs. error in days | history as released | grid completed |
  |---|---|---|
  | onset estimate, five candidates (hazel … grass) | 4.0 | 4.7 |
  | — alder / ash / hazel | 5.0 / 1.2 / 5.0 | 1.8 / 5.5 / 6.2 |
  | — best candidate alone | 8.7 | 6.1 |
  | — six or seven candidates | 4.2 / 5.2 | 4.5 / 4.5 |
  | outlook 7 days ahead (six species) | 5.0 | 5.8 |
  | outlook interval coverage | 81% | 76% |
  | forecast, eleven pollen taxa, days 1–7: MAE | 5.87 | 5.78 |
  | — level accuracy / bias / RMSE | 81.6% / −1.43 / 42.8 | 81.4% / −1.84 / 44.1 |
  | — alder / poplar MAE | 20.1 / 14.6 | 18.2 / 13.4 |
  | — hazel / ash / oak MAE | 5.6 / 5.6 / 6.2 | 6.0 / 6.3 / 6.7 |
  | mould MAE / level accuracy | 23.1 / 78.4% | 23.2 / 78.2% |

  The ensemble is still ahead of the single rule and of the calendar (9.4)
  on either history, so G.2b stands — but "4.0" is the favourable end of
  what four test years per species can tell apart, and five candidates
  was chosen on the incomplete grid. The forecast is a wash: alder and
  poplar gain, eight taxa lose a little, the level is right slightly less
  often. Nothing here can see the difference in the grid's favour, so it
  stays off the release. If it is adopted later: `mode: backfill`, then
  `backfill-labels` (it gives the station's values to the 40 windows of
  the June 2026 outage that it has since filled in), `calibrate
  --rebuild` and `train`.
* **The benchmark's weather is measured, not forecast.** A day-7 window is
  trained and scored on weather known to the hour, and the confidence
  table inherits that. Open-Meteo's previous-runs API serves each hourly
  variable as it was forecast 1–7 days earlier (2024 on; daily mean
  temperature off by 0.5 °C at day 1 and 2.0 °C at day 7). Replace the
  fold's weather with it per lead. The thermal crossing projected seven
  days ahead from those archived forecasts landed within 0–1 day of the
  one from measured temperatures, so the outlook is the part least
  affected.
* **Temperature.** The forecast model's analysis runs 0.83 °C above the
  archive values the history is trained on (Jan–Jun 2024, daily means):
  13% more degree-days above 5 °C. Thresholds are calibrated on the
  archive; the last five days (since 3843904 explicitly) and every
  forecast day accumulate on the forecast model. Correct the offset or
  train on the historical-forecast API.
* **NDVI.** The history holds a two-sided cubic interpolation written by
  the September backfill — through June 2026 it uses composites that did
  not exist yet — while serving holds the last composite flat
  (`ndvi_delta` exactly 0 on 79% of days since). The pair earns 3.2% of
  gain and the benchmark reads the look-ahead version. Without the pair
  (63 features, record months, seven days) the pollen models are no
  worse — MAE 5.87 → 5.83, level accuracy 81.6 → 82.0%, with the
  look-ahead version to lose — and the mould model reads the level right
  75.9% of the time instead of 78.4% (MAE 23.1 → 22.9): the pollen models
  do not need it, the mould model uses it as a season clock. Left in for
  that reason. Either make it as-of in the history and the fold context
  and measure again, or take it out together with its fetch — the
  composites cache is a local file on an ephemeral runner, so every run
  refetches three years from ORNL — once the mould model has something
  better to read the season from.

## H.3 The amount

* **The heavy-year ramp.** Betula 2024 and 2026 are at 0.07–0.10 of what
  arrives in days 5–9 of the season, as before. Birch is the one taxon
  the relabel left untouched, and CAMS does not carry the amount (its
  birch totals are 1 119–2 138 every year against 2 817–24 525 measured).
* **Mould never reaches "high".** The station's spore counts rose in 2025
  and again after its June 2026 outage (yearly median 0–5 through 2024,
  27 in 2025, 57 in 2026); most training years are far below that and the
  model calls no "high" day (2% of days).

## H.4 Smaller

* **Features, one family at a time.** As a block, the fifteen
  lowest-gain features and NDVI are not free (*Tried, not adopted*). The
  point lags are the likely cost; test them, the three hour encodings,
  the within-window slopes and `gdd_above_threshold` separately, after
  H.2's NDVI.
* **`history.csv`** stores each window's weather twelve times. Pollen
  (long) plus weather (one row per window) is about a sixth of the size
  and would retire the weather-only rows.
* **Confidence on the official series.** Held out, the level confidence
  overstates by three points (stated 0.68, right 0.66; ECE 0.072 against
  0.037 on the merged labels) and correlates 0.20 with being right
  (0.32). Most of the overstatement is the March fold. The comparison of
  schemes (per horizon day, pooled, per magnitude bin, per species) was
  made on the merged labels and has not been repeated.
* **The web app** (`muc-pollen`) reads `forecast.json` as before; it has
  not been checked against seven days of points, and it does not show the
  `seasons` block yet.
* **Parked variants** on `claude/forecast-app-review-xwisqv`: A.5, B.8,
  B.7, verdicts below. B.7 again once twelve seasons exist.

## Rules that still hold

* Nothing that changes model behaviour merges without a benchmark that can
  see the difference: `benchmark` (the six record months, Sep 2025 – Jul
  2026) for the general forecast, `benchmark-onset` for the season starts
  (timing, ramp ratios, false starts, and the outlook's replay), one
  dedicated commit per step.
* Feature changes are edits to the lists in `src/types.py` plus a retrain;
  data changes are a backfill plus a retrain. `calibrate --rebuild` after
  any model change, or the published confidence describes a model that no
  longer exists.
* Two XGBoost benchmarks on one 4-core machine starve each other; run one
  at a time or cap each at `OMP_NUM_THREADS=2`.
* A local run with `GITHUB_TOKEN` set writes to the production data
  release. Unset it for anything that is not meant to.

# Record

Two eras, not comparable row to row. Up to E.2 the labels were the merge
of two series and the six folds were resampled from the history as it grew
(184 origins, 80 680 predictions over five days). From G.1 on the labels
are the official LGL series and the folds are the six record months
(September and November 2025, January, March, May, July 2026; 157 origins,
every day the station reported). Details of each step are in the README
and in the commit messages.

On the merged labels:

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
| E.2 one peak-emphasis mechanism | 6.4 | 5.9 | −1.7 | −1.5 | +48.2% | 74.5% |

Persistence there: MAE 10.4 / 11.4, level 66.6% / 64.5% at days 1 / 5.
Mould (F.1): MAE 27.8 / 29.5 / 30.3 at days 1 / 3 / 5 against persistence
33.2 / 39.4 / 42.3.

On the official series, eleven pollen taxa, the 65 747 day-1-to-5
predictions common to every row (persistence MAE 10.3 / 11.3):

| Stage | day-1 MAE | day-5 MAE | day-1 bias | day-5 bias | day-5 skill | level acc. |
|---|---|---|---|---|---|---|
| E.2 as released (trained on the merged labels, scored on the official ones) | 6.42 | 5.86 | −1.37 | −0.97 | +48.2% | 78.9% |
| G.1 trained on the official series | 6.50 | 5.93 | −1.59 | −1.29 | +47.6% | 81.6% |
| G.2c poplar's window; spores have no onset | 6.48 | 5.85 | −1.28 | −1.06 | +48.3% | 81.7% |
| G.2b onset estimate from five candidates | 6.34 | 5.59 | −1.67 | −1.45 | +50.6% | 81.6% |
| G.2a trained to seven days | 6.42 | 5.67 | −1.55 | −1.32 | +49.9% | 81.7% |
| G.5 rolling statistics over measured windows | **6.36** | **5.69** | **−1.75** | **−1.46** | **+49.7%** | **82.0%** |

On all seven days (91 399 predictions): MAE 6.4 / 6.1 / 5.8 / 5.7 / 5.7 /
5.5 / 5.8, skill +38 / +42 / +46 / +51 / +50 / +54 / +58%, level accuracy
81.8% (persistence 77.7% at day 1, 75.5% at day 7).

Mould on the same rows (5 977 predictions; persistence 30.8 / 37.3):

| Stage | day-1 MAE | day-5 MAE | day-1 bias | day-1 skill | day-5 skill | level acc. |
|---|---|---|---|---|---|---|
| F.1 as released (trained on the merged labels) | 30.8 | 30.3 | +9.2 | −0.1% | +18.6% | 78.2% |
| G.1 trained on the official series (quantile 0.70) | 22.6 | 22.1 | −12.9 | +26.6% | +40.8% | 69.8% |
| G.2c no spore onset | 23.1 | 22.7 | −14.4 | +24.9% | +39.0% | 75.6% |
| G.1b quantile 0.85 | 22.8 | 22.6 | −8.3 | +25.8% | +39.3% | 78.9% |
| G.2a trained to seven days | 23.3 | 22.9 | −8.2 | +24.4% | +38.6% | 78.2% |
| G.5 rolling statistics over measured windows | 22.7 | 22.5 | −7.8 | +26.4% | +39.7% | 77.5% |

The season start, each of 2023–2026 projected from the seasons before it,
mean absolute error of the date in days (hazel, alder, birch, ash, grass,
poplar; per species in the README):

| days ahead | 21 | 14 | 7 | 3 | 1 |
|---|---|---|---|---|---|
| calendar | 9.2 | 9.2 | 9.2 | 9.2 | 9.2 |
| the estimate before G.2e (the calendar until the forcing has crossed) | 9.4 | 9.3 | 8.8 | 5.1 | 4.9 |
| G.2e projection (published as the season outlook) | 6.4 | 6.1 | **5.0** | 4.2 | 4.1 |

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
  chosen leave-one-out.
- **B.4** — readiness features follow the per-species rule;
  `cold_to_warm_flip` dropped.
- **B.6** — onset-ramp weighting (first 14 days after the measured onset
  weigh ×4); timing better, amplitude unchanged.
- **B.5** — four upwind ePIN stations (Mindelheim, Altötting, Feucht,
  Viechtach) as a time-gridded feature block.
- **D.4** — levels are daily-mean levels, in the forecaster and both
  benchmarks.
- **D.1** — level and confidence published per point.
- **D.2 / D.3** — origin anchored at the newest measurement, confidence read
  at the horizon a stale run really is; `status` block with observation
  age and every defaulted input group.
- **D.5** — conformal confidence and 80% value intervals from per-horizon
  residual distributions.
- **E.4** — skill vs persistence is the first block of the benchmark report.
- **E.1** — lag block on the 3 h grid on both paths; same-hour-yesterday
  fill inside gaps.
- **E.2** — the raised quantile is the one peak-emphasis mechanism.
- **E.3** — log-space probability scaling benchmarked against a hurdle form
  and a plain threshold, and kept.
- **F.1** — mould forecast: `Fungus` for Munich and the upwind stations.
- **G.0** (`f1d63cc`, `9d7f9eb`) — weather-derived features computed once
  per window grid instead of once per taxon over Python loops, the rule
  ranking memoised (`build_context` for twelve taxa 21.2 s → 2.8 s;
  identical feature matrices); the benchmark's default folds are the six record months.
- **G.1** (`8b43111`) — the official LGL series is the only measurement,
  for Munich and the four upwind stations; `backfill-labels` rewrote the
  stored values (7.9% of the Munich values changed, 1 848 windows have no
  official measurement) and keeps the files as they were on the release;
  a window the station did not report is a weather-only row, so forcing
  and the rolling weather features run on the windows that happened.
  Level accuracy 78.9 → 81.6%, mould MAE −25%; pollen MAE unchanged.
- **G.1b** (`13f888e`) — mould quantile 0.70 → 0.85: bias −13.8 → −7.9,
  level 75.6 → 78.9%, MAE unchanged.
- **G.2c** (`4055f5d`) — poplar's season is February–April (its "onset"
  sat on 1–2 March in five of eight years, clipped by the window); spore
  taxa have no onset, no ramp weight, no phenology features.
- **G.2b** (`6b1b3f0`) — the onset estimate is the median vote of the five
  best candidates (forcing rules and the calendar) instead of the single
  leave-one-out winner: walk-forward date error 8.7 → 4.0 d (calendar
  9.4), day-5 MAE 5.85 → 5.59. The retrain log prints the walk-forward
  error.
- **G.2d** (`014feeb`) — production carries the onset estimate through the
  weather forecast, as training and the benchmark always did
  (`days_since_typical_onset` read 16 for 0 on the day hazel started in
  2026); a parity test that calibrates a threshold.
- **G.2e** (`a079a70`) — season outlook: projected start per species,
  80% interval and P(start within 7 days) from a walk-forward replay,
  published as `seasons` in `forecast.json`; `outlook` command;
  `benchmark-onset` ends with the replay. 5.0 d a week ahead against 8.8
  before and 9.2 for the calendar; the interval covers 81% of the starts.
- **G.2a** (`eb349b4`) — seven days: leads to 56 windows, calibration to
  14 days. Costs 0.07 MAE on days 1–5, days 6 and 7 at +53% and +57%
  skill on measured weather.
- **G.5** — the 24-hour and 7-day rolling statistics of the lag block are
  taken over the windows that were measured, and over the carried values
  only where the whole span is dark; the official station is silent for
  15–50 days a year. General: pollen MAE 5.87 → 5.85, level accuracy
  81.6 → 81.8%, mould MAE 23.1 → 22.5 at 78.4 → 77.2%. Onset months: level
  accuracy 45.5 → 47.9%, false starts 18 → 7 run-days, MAE 101.3 → 102.2.
- **G.3** (`569fd02`, `3843904`, `c2c300d`) — a 429 from Open-Meteo is
  retried as a rate limit instead of being read as "soil parameters
  rejected"; the collector asks the forecast API for the last five days,
  so the evening windows the station publishes after midnight are no
  longer dropped until the archive catches up, and an archive failure no
  longer kills the run; `backfill-weather` adds a weather-only row for
  every window the history lacks.
- **G.4** (`2dd0cf8`, `384020c`, `fbef64f`, `a60c358`) — removed: the
  phenology download/upload/refetch (fed nothing since B.2, and its fetch
  could only ever match the 1930–2018 file); the Copernicus client and
  the `cams` extra (never ran, never populated); the `--classic`
  one-window benchmark (had been raising a KeyError since C.1); three
  hourly weather variables, the window's temperature extremes, EVI,
  `cams_pollen` and seven stored calendar columns — fourteen of the
  history's thirty-three columns. Feature matrices identical.

## Tried, not adopted

- **A.5 NaN passthrough** (XGBoost sees missing raw inputs instead of 0):
  general a wash (6.79 → 6.77), onset false starts 46 → 60.
- **B.8 upwind season load** (cumulative upwind features): general 7.07,
  false starts 60, heavy-year ramp unchanged.
- **B.7 December continuity** (forcing from 1 Nov / 1 Dec): the autumn rules
  lose the leave-one-out for hazel (8.6 d vs 4.0 d) and alder (21.5 vs
  6.5 d); the November turn alone raised false starts 46 → 59.
- **E.1 variants**: plain carry-forward (MAE 7.0, hazel onset d2 1.3 → 6.0 d)
  and dropping every gap-spanning training block (level 72.7%).
- **E.2 variants**: value weights with median regression (RMSE 45.5, bias
  −3.9); quantile +0.03 (bias −0.8 at +0.4 MAE).
- **E.3 variants**: hurdle scaling after `expm1`, plain threshold at p = 0.5.
- **F.1 variants**: stage-3 threshold none–400 (±0.4), tree depth 3–7
  (±0.5).
- **Pruning the low-gain block** (G.4; the six point lags beyond
  `pollen_lag_1`, `pollen_morning_avg`, the hour encodings, `is_day`,
  `gdd_above_threshold`, the within-window slopes, `upwind_lead_8`, NDVI;
  65 → 48 features, five taxa, merged labels): MAE 8.6 → 8.8, level
  accuracy 64.5 → 62.9%, most of it at day 1 and in alder, for a 17%
  faster retrain.
- **Scaling the forecast by yesterday's error** (ratio of measured to
  forecast daily mean before the origin, fading over the horizon; merged
  labels): onset months MAE 88.6 → 80.6 and timing 4.3 → 1.5 d at day 1,
  but false starts 1 → 11 run-days and the general folds 8.6 → 9.6; no
  restricted form did better than 8.6 → 9.0.
- **CAMS** (Open-Meteo air-quality API, no key): finds the birch start to
  +19, +2, −12, +4, −1, +3 d over 2021–2026, no better than the outlook
  does from temperature; does not know a heavy year (its birch totals are
  1 119–2 138 against 2 817–24 525 measured; 2026, the record, is its
  third lowest).
- **G.2b variants**: selecting on a smoothed leave-one-out criterion
  (8.2 d); four, six, seven candidates instead of five (5.8, 4.2, 5.2 d).
- **G.2c variant**: searching the gate window instead of the core season
  for every species moves birch 2024 to 13 February and oak into March.
- **G.2e variants**: residuals per species instead of pooled (the 80%
  interval covers 52%).
- **G.1b variants**: quantile 0.75, 0.80, 0.90 (see the README).
- **The projection inside the model** (G.2e): `days_since_typical_onset`
  counted from the outlook's continuous projection instead of from the
  estimate that jumps at the crossing. Record months, nine taxa: MAE
  7.20 → 7.29, level accuracy 78.2 → 78.4%; onset months: MAE 97.0 → 96.1,
  timing unchanged, false starts 26 → 29 run-days.

## Corrections

The review that opened the G series (`f882889`) made two claims that did
not survive implementation, both from one mistake: its analysis copy of
the relabelled history had dropped the weather of the windows the official
station did not report, so the forcing filter of the onset detector
rejected a real start.

* "Alnus 2025 moves from 4 February to 20 February" — it moves to
  8 February, four days. The starts that the relabel moves are oak 2023
  (+20 d), hazel 2023 (−7), grass 2026 (+7), alder 2025 (+4), oak 2024
  (+4), alder 2023 (−3), alder 2026 (+1), and willow 2024, which no longer
  has one.
* "Alder's day-1 timing error falls from 7.7 to 3.3 days on the official
  labels" — it does not. On the official labels the forecast's day-1
  timing error for alder is 7.5 d over the two of three starts it calls at
  all (H.1).

Also: the review called the mould series' jump in 2024 an artefact of the
merge. The second series did dominate `Fungus` in 2020–2025, but the
official series has its own rise, a year later (yearly median 3 in 2024,
27 in 2025, 57 in 2026).
