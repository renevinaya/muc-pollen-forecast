# Open points

State of 2026-11-10, after the G series. Everything here is measured on the
official LGL series and the six record months unless it says otherwise.
Ordered by what it is expected to buy. What is done, what was tried and not
adopted, and the numbers of every step are in the README
([Development history](README.md#development-history),
[Evaluation](README.md#evaluation)).

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
  and breaks the general folds (README, *Tried, not adopted*); a version
  confined to the fortnight after a detected start is untested.
* **The projected date as a model input.** Replacing the estimate behind
  `days_since_typical_onset` with the outlook's continuous projection did
  nothing (README, *Tried, not adopted*). Untested: the projection as a
  feature of its own — days *to* the projected start, clipped — beside the
  three that exist.
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
  lowest-gain features and NDVI are not free (README, *Tried, not
  adopted*). The point lags are the likely cost; test them, the three hour
  encodings, the within-window slopes and `gdd_above_threshold`
  separately, after H.2's NDVI.
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
  B.7, verdicts in the README (*Tried, not adopted*). B.7 again once
  twelve seasons exist.

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
