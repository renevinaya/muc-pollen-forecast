# Task list — improving the forecast

Rewritten after the second review (2026-09-10). The first task list is
finished through Phase 3.5, Phase 2 and 4.4; what it achieved is summarised
at the bottom. This list is ordered by expected value for the thing the review
asked about — **the start of the birch, alder and hazel seasons** — and every
task states the measurement that accepts or rejects it.

Two rules carried over from the first list still hold:

* Nothing that changes model behaviour merges without a benchmark that can
  see the difference. The general rollout benchmark **cannot** see onset: its
  six folds fall in Sep, Nov, Jan, Mar, May and Jul, so the Jan and Mar folds
  bracket the February hazel/alder start without covering it. Phase B.1
  exists to fix that before anything else in Phase B is judged.
* Feature changes are edits to the lists in `src/types.py` plus a retrain;
  data changes are a backfill plus a retrain. Both are cheap to A/B.

## Open points (2026-09-25)

Everything on the list below has been done, tried, or benchmarked and kept
as it was. What remains open is what none of it could settle:

1. **The heavy-year ramp.** Betula 2024 and 2026 still get 6–18% of what
   arrives in days 5–9 of the season (B.6, B.8). The season's *amount* is
   not visible in any feature the model has; it needs a source that sees
   the season before Munich does — a station far to the south-west, or a
   CAMS backfill (the CAMS feature exists but has never had data). This
   is the one item with real product value left, and it is blocked on
   data, not modelling.
2. **Mould, sampler regime.** The Munich station's spore counts changed
   scale in 2024 (2019–2023 median 9, 2024+ median 46) and the model trains
   across both. A regime feature or a 2024+ training window is the next
   experiment if the level accuracy (81% at day 1) is not enough (F.1). The
   upwind stations' spore series is in the block but did not measurably
   help.
3. **Mould, frontend.** `forecast.json` now carries a `Fungus` series; the
   Vue app has no label or thresholds for it. That is the other repository.
4. **Parked variants** on `claude/forecast-app-review-xwisqv`: A.5 (NaN
   passthrough), B.8 (upwind season load), B.7 (autumn forcing starts).
   B.7 is worth re-running once twelve seasons exist; the other two lost
   on the benchmark and are kept only as a record.

## What the second review found

**The feature list is fine. The data behind ten of the sixty features is not.**
Measured on `data/history.csv` (2019-01-01 → 2026-08-31, eight seasons):

| Feature(s) | First non-missing row | Seasons with real values |
|---|---|---|
| `boundary_layer_height`, `dew_point_mean`, `is_day`, `temp_slope_3h`, `humidity_slope_3h`, `temp_variance_3h` | 2025-03-24 | 1.5 of 8 |
| `soil_temperature_mean`, `soil_moisture_mean` | 2026-06-17 | 0.2 of 8 (NaN in 97% of rows) |
| `ndvi`, `ndvi_delta` | 2024-01-01 (a literal `0.0` before that, not NaN) | 2.7 of 8 |

Seven of the eight hazel, alder and birch season starts in the training set
therefore carry none of the diurnal, soil or vegetation signal. Whatever those
features earn today (NDVI 2.2–3.7% of gain per species, soil ≈ 0) is learned
from at most two onsets, and part of it is an era marker: "NDVI is non-zero"
identifies 2024–2026, which are also the three heaviest birch years.

**The onset projection is wrong for two of the three species.** `src/onset.py`
projects every species' onset with one rule — forcing accumulated from 1 Jan
at base 0 °C — chosen because it fit hazel. Leave-one-out over the eight
seasons (predict year *Y*'s onset from the other seven):

| Species | Climatology only | Current rule (1 Jan, base 0) | Best single rule found | Rule |
|---|---|---|---|---|
| Corylus | 14.6 d | **4.6 d** | 3.8 d | 1 Jan, base 3 |
| Alnus | 11.6 d | 12.8 d | **6.1 d** | 15 Jan, base 3 |
| Betula | 9.0 d | 20.8 d | **6.9 d** | 1 Mar, base 5 |

For alder the projection is no better than the calendar; for birch it is more
than twice as bad, because January and February warmth counts fully towards a
tree that does not respond to it until March. The `days_since_typical_onset`
and `onset_anomaly` features, and the `gdd_above_threshold` /
`cold_to_warm_flip` gates, all inherit that error.

**The onset detector is fooled by transport.** The 2025 birch "onset" is 4
March: six days of 16–37 grains/m³ followed by seven days of zero, then real
flowering three weeks later. It is the largest error (24–30 d) in every rule
tested and drags the calibrated GDD threshold for every later year.

**Chilling did not help.** Sequential chill-then-forcing models (chill days
< 7 °C from 1 Nov, forcing from when the requirement is met) tied plain
forcing for hazel and were worse for alder and birch, at every requirement
tried. With eight seasons the data cannot support a chill term; the earlier
task 5.3 is downgraded accordingly.

**Season load is unmodelled and large.** Birch season totals span 2,842
(2025) to 24,640 (2026), lag-1 correlation −0.41 (alternation); the mean
concentration in the first onset week ranges 20–811. No feature crosses a
season boundary, so the model cannot know which kind of year it is entering.

**The confidence work never reached the frontend.** `to_web_dict()` in
`src/types.py` emits only `from`/`to`/`value`; `confidence` and
`confidence_within_one` exist in `to_dict()` only. Task 4.4 is correct and
invisible.

**The shipped forecast gets the onset *date* right and the onset *amount*
wrong.** A first run of the onset rollout (B.1 below: the direct forecast
scored over the seven months containing the 2024–2026 hazel, alder and birch
starts; 210 origins, 25,181 predictions):

| | day 1 | day 3 | day 5 |
|---|---|---|---|
| Timing error of the first predicted 3-day run ≥ low, mean abs. (9 species-years) | 4.0 d | 4.0 d | 3.1 d |
| False starts (predicted run > 10 d before the real one) | 0 of 8 | 0 of 8 | 0 of 8 |
| Bias in the onset months, model | −65.6 | −69.2 | −68.9 |
| Bias in the onset months, persistence | −14.4 | −17.6 | −16.9 |
| MAE, model vs persistence | 84.7 vs 93.0 | 87.1 vs 122.3 | 85.6 vs 128.6 |
| Level accuracy, model vs persistence | 48.3% vs 56.6% | 47.5% vs 48.1% | 46.3% vs 45.8% |

The only false start in nine species-years is Betula 2026, where the model
called the season on the 27 Feb – 5 Mar transport episode, as a person reading
the trap would have. Timing at day 5 is as good as at day 1, which is what a
direct model with a calendar-and-warmth signal should give.

Amplitude is another matter. Predicted-over-actual daily means, day-3
horizon, by days since onset:

| Species-year | 0–4 d | 5–9 d | 10–14 d | 15–19 d |
|---|---|---|---|---|
| Betula 2024 (heavy) | 0.07 | 0.06 | 0.18 | — |
| Betula 2026 (heaviest) | 0.65 | 0.18 | 0.28 | 0.22 |
| Alnus 2025 (heavy) | 0.01 | 0.32 | 0.02 | 0.78 |
| Alnus 2026 | 0.89 | 0.67 | 0.39 | 0.17 |
| Corylus 2025 | 0.30 | 0.11 | 0.17 | 0.37 |
| Corylus 2026 | 0.14 | 0.16 | 0.04 | 0.23 |
| Corylus 2024 (light) | 1.31 | 1.37 | — | — |

In a heavy year the model predicts a tenth to a third of what arrives for the
first two to three weeks and only converges once the 7-day lag block has
filled with big numbers. In the light hazel year it over-predicts by a third.
That is a season-load problem (Phase C), not a timing one: the lag block is
near zero at onset by definition, so the only things that can set the scale
are the year's load and the weather of the day, and the model has no feature
for the former. The general benchmark hides this because onset weeks are a
few percent of its rows and it never samples February.

## Phase A — Fill the data behind the features — **DONE**

Result: the history is complete (run #125, 2026-09-11), and **the general
benchmark got worse for it**: MAE 7.3 → 7.8, level accuracy 76.9% → 75.6%,
bias 0.0 → +1.0, on the same six folds. Three attribution arms on those folds
say why:

| Arm (history variant, 60 features unless noted) | MAE | Level acc. | Bias |
|---|---|---|---|
| before backfill (baseline) | 7.3 | 76.9% | 0.0 |
| complete | 7.8 | 75.6% | +1.0 |
| complete, old NDVI columns restored | 7.8 | 75.7% | +1.2 |
| complete, old diurnal/soil columns restored (NaN before 2025-03) | **7.3** | **77.1%** | +0.2 |
| complete, the eight diurnal/soil features dropped (52 features) | 7.8 | 75.1% | +1.2 |

Restoring the NaN block recovers the baseline exactly; dropping the eight
features on complete data changes nothing. So the eight features never
carried the 0.5 MAE — their *absence before March 2025* did. NaN filled to 0
made "these columns are non-zero" a flag for "2025 or later", and every fold
lies in that era, so the model could learn a level offset for recent seasons
that it has no honest feature for. That flag is a crude proxy for interannual
load, which is Phase C.1; it moves to the front of the queue.

The onset rollout is unchanged by the backfill (MAE 86.4 → 88.5, timing within
3–5 days at every horizon, the same amplitude gap), as expected: the model's
onset problem was never a data-coverage problem.

What did change, and is kept:

* Pre-2025 weather now comes from the same archive request the collector makes
  every day. The old rows disagreed with it by 0.7 °C mean absolute difference
  in temperature (corr 0.993, no time shift), i.e. they were fetched from a
  different model years ago; 2026 rows were near-identical. Training and
  serving now see one source.
* NDVI is measured back to 2019 (175 composites). Spring 2026 was wrong
  before: the routine fetch had timed out on the January–June chunk, and the
  interpolator bridged the hole, so April 2026 read 0.16 where the composites
  say 0.26.
* Soil temperature and moisture are complete; dew point and the 3 h slopes and
  variance are complete; `boundary_layer_height` is still missing for
  January–June 2024 (the archive returns null there; 7% of rows, below the
  guard's threshold, and 0-filled by the trainer).

- [x] **A.1 `backfill-weather` / `run-backfill`.** `src/backfill.py`
  refreshes every weather column in place, yearly archive chunks, pollen
  never touched. The workflow's `mode: backfill` runs it against the release.
- [x] **A.2 NDVI backfill 2019–2023.** `backfill-ndvi`; MODIS chunk fetches
  now retry. The NaN-instead-of-0.0 half of this task is **not done**:
  `prepare_training_data` fills every NaN with 0 before XGBoost, so the
  history's fill value never reaches the model. Changing that is a model
  change (A.5).
- [x] **A.3 Re-benchmark.** Above. Decision: keep the complete history and
  the 60 features. Soil earns ~1% of gain each and moves Poaceae/Urtica by
  ≤0.2 MAE either way; NDVI's share rose from 3.2% to 4.2% with real values
  and costs nothing. Neither justifies a change on its own; both are
  re-judged once C.1 exists.
- [x] **A.4 Coverage guard.** `check_feature_coverage` prints a per-feature
  table at every retrain and raises when a model input is missing on more
  than half of the rows (NaN, or 0.0 for NDVI, soil moisture, dew point and
  boundary layer height). `run-train` restores the released models and still
  publishes a forecast when the retrain is refused.
- [ ] **A.5 Let XGBoost see missing values.** **Tried — not adopted**
  (parked on `claude/forecast-app-review-xwisqv`). `finalize_features` —
  since B.5 the one place a feature frame is filled, shared by the trainer,
  the rollout benchmark and the forecaster — was changed to keep NaN for the
  raw weather, NDVI and upwind columns and to turn the stored 0.0 markers of
  the coverage guard's four columns into NaN. Same six folds, B.5 → A.5:
  MAE 6.79 → 6.77, level accuracy 76.3% → 76.5%, bias −0.34 → −0.30, skill
  +0.0…+0.4 points by day — a wash. The onset benchmark went the other way:
  false starts 46 → 60 run-days, level accuracy 51.9% → 51.3%, and new
  runs 21 days early for Corylus 2024 (d1–d3) and 18 days early for Betula
  2024 (d2–d3). Both are the 2024 folds: their training data predates the
  only gap in the history (boundary-layer height, Jan–Jun 2024), so the test
  rows meet a "missing" the model never learned a branch for and XGBoost
  routes them to its arbitrary default — while the zero fill sends them
  down the "stable, low boundary layer" side, which happens to be the
  conservative one. That is not only a backtest artefact: a feed that
  fails next March would put the live model in the same position. Revisit
  when there is a second season with gaps to learn from, or with a fill
  that is conservative by construction (the column's training median)
  instead of NaN.

## Phase B — Make the season start a first-class target

- [x] **B.1 Onset rollout benchmark.** **Done.** `benchmark-onset` now runs
  the shipped direct rollout over the months containing each measured onset
  of the last three seasons (eleven months for hazel, alder and birch) and
  reports, per species-year and horizon: the timing of the first predicted
  3-day run at or above the low level, MAE and bias in the ±10-day window
  next to the MAE of predicting zero, false starts, and the predicted/actual
  ratio by days since onset (`src/onset_report.py`; `--classic` keeps the
  old one-window-ahead diagnostic). `benchmark --months` names folds
  explicitly. This is the baseline every B/C change is judged against; the
  first run, on the B.3 model (331 origins, 39,701 predictions):

  | | d1 | d3 | d5 |
  |---|---|---|---|
  | Timing, mean abs. error (days) — Corylus / Alnus / Betula | 1.3 / 7.0 / 7.0 | 2.0 / 6.0 / 11.3 | 4.3 / 6.3 / 7.0 |
  | In-season MAE — Alnus / Betula / Corylus | 112.3 / 332.6 / 67.3 | 111.2 / 329.1 / 67.5 | 109.7 / 319.7 / 67.2 |
  | Overall MAE / level acc. / bias | 69.1 / 53.3% / −41.4 | 70.5 / 52.3% / −44.4 | 71.2 / 51.1% / −46.5 |
  | Skill vs persistence | +5.0% | +24.9% | +29.9% |

  Predicted/actual ratio in days 5–9 after onset, day-3 horizon: Betula 2024
  0.09, Betula 2026 0.17, Alnus 2025 0.29, Corylus 2026 0.39; Alnus 2026
  2.24 and Betula 2025 2.48 the other way. Mean 0.83 across the nine
  species-years, made of under-prediction in heavy years and over-prediction
  in light ones — the ramp problem, B.6. False starts: 43 run-days, all
  Betula, most of them the model reproducing the March 2025 transport
  episode that the B.2 detector no longer counts as the onset.
- [x] **B.2 Robust onset detection for calibration.** **Adopted with B.3**
  (on its own it was a wash — see below). Measured on
  the eight seasons, none of the definitions in the task improves all three
  species: "no return to zero within 7 days" fixes the 2025 birch episode
  (4 March → 3 April) but moves the 2022 alder onset from mid-February to
  April, because a light alder year *is* gappy; the 2.5% / 5%-of-season-sum
  rules do not catch the 2025 episode at all, because it was 4.6% of a tiny
  season, and they cost hazel 2 days of LOO accuracy. What actually marks a
  transported episode is thermal implausibility, so the branch rejects a
  run whose accumulated forcing is under half the median at the other years'
  onsets and takes the year's next run. That moves exactly two onsets —
  Betula 2025 (63 → 93) and Corylus 2023 (1 → 11, a calendar-reset artefact
  after a warm December) — and improves the LOO table for both species
  (Betula projection 18.9 → 14.8 d, climatology 9.0 → 5.2 d; Corylus 4.2 →
  4.0 d) with Alnus untouched. Betula's base-5 threshold still moves 22%
  when one year is dropped: that is the accumulation rule (B.3), not the
  detector.

  **But the model does not care.** Same six folds: MAE 7.0 / RMSE 44.8 /
  level 75.2% / bias −0.2 before and after, to the decimal; Corylus in-season
  MAE 23.0 → 22.3 and Betula day-5 18.4 → 20.0. The onset rollout: Betula
  438 → 443, Corylus 74.6 → 74.0, Alnus identical; ±10-day windows better for
  Corylus 2026 (61 → 53) and worse for Corylus 2024 (11 → 14) and Betula
  2026 (392 → 405). Two onsets out of sixteen calibration points move the
  medians the features are built on by a day or two, and the forecast is
  insensitive to that. With B.3's per-species rules calibrated on the
  corrected onsets it does earn its keep: general MAE 7.0 → 6.9 and Corylus
  in-season MAE 23.1 → 21.4 against B.3 alone (Betula 14.9 → 15.3, within
  noise), so it shipped together with B.3.
- [x] **B.3 Species-specific forcing rules.** **Done.** `select_forcing_rule`
  picks a (start date, base) pair per species from a 5 × 3 grid by
  leave-one-out over the seasons before the year being projected, keeps the
  old rule below four prior seasons, and switches the projection off when the
  climatology wins. Picks on the full history: Corylus 1 Jan / 3 °C (LOO
  3.4 d), Alnus 15 Jan / 0 °C (6.5 d), Betula 1 Mar / 5 °C (6.5 d), Fraxinus
  and Populus 1 Mar / 0 °C, Salix 15 Feb / 0 °C, Poaceae 1 Mar / 3 °C,
  Quercus and Urtica climatology. The retrain log prints rule and LOO error.

  Same six folds and onset months, C.1 → B.3 alone → B.3 + B.2 (shipped):

  | | General MAE / RMSE / level / bias | Onset MAE Alnus / Betula / Corylus | Onset level acc. |
  |---|---|---|---|
  | C.1 | 7.0 / 44.8 / 75.2% / −0.2 | 112.0 / 438.1 / 74.6 | 41.8% |
  | B.3 | 7.0 / 45.0 / 75.2% / −0.3 | 110.2 / 436.3 / 73.9 | 41.9% |
  | **B.3 + B.2** | **6.9** / 45.0 / 75.2% / −0.4 | 110.2 / 439.0 / **72.8** | **42.3%** |

  Small, consistent, and in the right place: the Corylus 2024 false start
  three weeks early is gone at every horizon, and the Alnus 2026 and Betula
  2026 ±10-day windows improve (33.5 → 28.2, 392 → 387–396). The amplitude
  ratios in heavy years are unchanged (B.6). The `onset_anomaly` and
  `gdd_above_threshold` features still use the `gdd` column (1 Jan, base 5)
  against a threshold in those units; making them read the selected rule's
  forcing is the natural follow-up and is folded into B.4.
- [x] **B.4 Onset-phase features that survive B.3.** **Done.** Per-species
  gain on the B.3 models: `days_since_typical_onset` up to 6.1% (Alnus),
  `gdd_above_threshold` up to 2.3% (Corylus), `onset_anomaly` up to 1.2%,
  `consecutive_warm_hrs` up to 2.3% (Salix), `cold_to_warm_flip` at most
  0.05% for any tree species — dropped. `onset_anomaly` and
  `gdd_above_threshold` now read the species' selected forcing rule against
  that rule's walk-forward threshold (`readiness_by_day` in `src/onset.py`,
  accumulated over the combined measured-plus-forecast temperature so it
  keeps climbing through the forecast days) instead of the base-5 `gdd`
  column against a threshold in those units. 62 features.

  Against B.3 on the same folds and the same eleven onset months:

  | | General MAE / RMSE / level / bias | Onset timing d1 / d3 / d5 (mean abs. days, C/A/B) | Onset in-season MAE A / B / C |
  |---|---|---|---|
  | B.3 | 6.9 / 45.0 / 75.2% / −0.4 | 1.3-7.0-7.0 / 2.0-6.0-11.3 / 4.3-6.3-7.0 | 112.3 / 332.6 / 67.3 |
  | **B.4** | 7.1 / **44.2** / **75.6%** / **+0.1** | 1.3-7.0-7.0 / **0.7-6.7-7.0** / **4.0-3.7-7.3** | 114.7 / 332.8 / **66.2** |

  Timing improves at days 3–5 for every species (Betula's −18-day run at day
  3 is gone), false starts fall 43 → 33 run-days, the Corylus ±10-day windows
  improve (2026: 42.5 → 30.9), RMSE, level accuracy and bias improve on the
  general folds — and general MAE slips 0.2, mostly Populus (54 → 62) and
  Alnus (63 → 65). Adopted for the onset gains and the coherent feature set;
  the MAE cost is real and is noted. Amplitude ratios unchanged (B.6).
- [x] **B.5 Upwind stations (was 5.2).** **Done — the general benchmark's
  best; the premise did not hold.** `src/upwind.py` reads the four ePIN
  automatic stations within ~150 km of Munich (Mindelheim WSW, Altötting E,
  Feucht N, Viechtach NE; pollenscience.eu serves them from 2019 at 3 h)
  into a second long-format file backed up to the data release, and turns
  them into three features anchored at the forecast origin: the highest
  reading at any station over the last 24 h and 7 d, and the 24 h maximum
  minus Munich's own. Built on the time grid, NaN where nobody reported.
  `backfill-upwind` (also a workflow mode) fetched the history; the
  collector appends the last two weeks every run. 65 features; the live
  retrain gives the family 2.3% of total gain, `upwind_max_56` first.

  | | General MAE / RMSE / level / bias | Onset timing d1 / d3 / d5 (C/A/B) | Ramp ratio days 5–9, heavy years (B24 / B26 / A25 / C26) | False starts |
  |---|---|---|---|---|
  | B.6 | 7.0 / 44.0 / 75.7% / +0.1 | 1.3-6.0-7.3 / 1.7-3.3-6.7 / 4.0-3.7-6.7 | 0.07 / 0.18 / 0.42 / 0.40 | 48 |
  | **B.5** | **6.8 / 43.6 / 76.3% / −0.3** | 1.3-2.0-7.3 / 1.3-5.0-6.7 / 1.0-3.7-7.0 | 0.06 / 0.18 / 0.33 / 0.35 | 46 |

  Skill against persistence is up 2–3 points at every horizon (day 1 +29.6%
  → +32.0%, day 5 +40.3% → +42.0%), mostly through Populus (March MAE 82 →
  70) and Fraxinus (27 → 24); Betula's March and May 2026 in-season MAE is
  0.9–1.2 worse with a more positive March bias — the stations report the
  pre-onset transport and the model believes them a little. Alder's day-1
  onset timing 6.0 → 2.0 days (2025: +17 d → +3 d), hazel's day-5 4.0 → 1.0.
  **The heavy-year ramp did not move**: Betula 2024 and 2026 stay at 6–18%
  of the truth in days 5–9. The stations do carry the information — Viechtach
  peaked at 16 000 (2024) and 17 500 (2026) grains against 1 000–3 000 in
  the light years — but a maximum over the last day or week at onset is a
  level the model has seen in every year, so it still predicts their
  average. The amount of a season is a property of the season, and only a
  feature that summarises the *region's* season so far (upwind cumulative
  load since 1 Jan, or the upwind stations' own onset-to-date total) could
  tell a 25 000-grain year from a 3 000-grain one on its fifth day. That is
  the follow-up (B.8); the three features stay because everything else
  improved.
- [x] **B.6 Ramp amplitude at onset.** **Done — option (a); the premise only
  half held.** Rows in the first 14 days after each year's *measured* onset
  weigh four times more in the stage-2 regressor and the extreme gate
  (`RAMP_DAYS`, `RAMP_BOOST` in `src/trainer.py`; label information at
  training time, no feature changes). Against B.4:

  | | General MAE / RMSE / level / bias | Onset timing d1 / d3 / d5 (C/A/B) | Ramp ratio days 5–9, heavy years (B24 / B26 / A25 / C26) |
  |---|---|---|---|
  | B.4 | 7.1 / 44.2 / 75.6% / +0.1 | 1.3-7.0-7.0 / 0.7-6.7-7.0 / 4.0-3.7-7.3 | 0.10 / 0.17 / 0.28 / 0.54 |
  | **B.6** | **7.0 / 44.0 / 75.7% / +0.1** | 1.3-6.0-7.3 / 1.7-3.3-6.7 / 4.0-3.7-6.7 | 0.07 / 0.18 / 0.42 / 0.40 |

  The general benchmark is the best of the series on all four numbers
  (Alnus 65 → 63, Populus 62 → 56, Poaceae 22 → 20; Quercus 26 → 30 and
  Fraxinus 21 → 23 the other way), alder's onset timing improves at every
  horizon and the Alnus 2026 over-prediction is gone (±10-day MAE 34.5 →
  7.2, ratio 2.5 → 0.65). **But the heavy-year ramp is where it was**:
  Betula 2024 and 2026 still get 7–18% of what arrives in days 5–9, and
  false starts rise 33 → 48 run-days. Weighting the ramp rows makes the
  model fit them better *on average*, and the average of a 25,000-grain
  year and a 3,000-grain year is neither. The model cannot know at onset
  which it is in: season load (C.1) points the wrong way as often as not at
  eight seasons, and nothing else in the feature set is upstream of the
  count. Options (b) and (c) from the task were not tried — they change how
  the same information is combined, not what is known — and are not
  expected to reach it either. What would: a measured signal that precedes
  the local season, i.e. upwind stations (B.5), or the DWD forecast level
  for the first days, which already blends in for today/tomorrow.
- [ ] **B.8 Upwind season load.** **Tried — not adopted** (parked on
  `claude/forecast-app-review-xwisqv`, commit `598fdd2`). Four features on
  top of the B.5 block, anchored at the origin: the log1p sum of the upwind
  readings since the season-year start, that sum against the median of the
  earlier season years at the same 3h slot of the same season day (0 while
  there is no earlier year), Munich's own season-to-date sum, and the
  difference between the two. 69 features. Same six folds, B.5 → B.8: MAE
  6.79 → 7.07, RMSE 43.6 → 44.0, bias −0.34 → +0.28, level accuracy
  unchanged; all of it March 2026, where the region's heavy-year signal is
  read before the local season starts (Quercus in-season MAE 24 → 34,
  Fraxinus 24 → 29, Populus 70 → 74, Betula 11 → 14). Onset benchmark:
  hazel and alder timing a day better at most horizons, but false starts
  46 → 60 run-days (Betula 2024 19 days early at d1–d3), Betula 2025's
  ±10-day MAE 93 → 115, and **the heavy-year ramp where it was**: Betula
  2024 / 2026 at 0.08 / 0.17 of the truth in days 5–9 (B.5: 0.06 / 0.18).

  Why it cannot work as posed: the four stations share Munich's phenology,
  so on day 5 of Munich's season the region's season is also five days
  old and its season-to-date sum is as small and as timing-dominated as
  Munich's own; the 16 000-grain Viechtach peaks that mark 2024 and 2026
  arrive two weeks in, when the local lag block already knows. A season's
  amount is visible early only from somewhere that flowers earlier — a
  station 300+ km south-west, or a physics forecast (CAMS, off by default,
  see README) that carries the emission inventory. Both are new data
  sources, not features; neither is on this list yet.
- [ ] **B.7 December continuity (was part of 5.3).** **Tried — not adopted**
  (parked on `claude/forecast-app-review-xwisqv`, commit `cd0c8cd`). Two
  changes: forcing rules may start on 1 November or 1 December of the year
  before the season and accumulate across New Year, with the leave-one-out
  selection deciding per species; and from November a day carries the
  *coming* season's rule, threshold and onset climatology, so the readiness
  features and days-since-onset run continuously into January instead of
  reading last season's total in December and resetting.

  The premise failed first. On the eight measured seasons the autumn starts
  project the hazel and alder onsets *worse* than the January ones —
  leave-one-out, base 0 °C: Corylus 1 Jan 4.0 d, 1 Dec 8.6 d, 1 Nov 11.6 d;
  Alnus 15 Jan 6.5 d, 1 Nov 21.5 d, 1 Dec 24.4 d (bases 3 and 5 are worse
  still) — so the selection keeps the January rules for both and nothing
  in the live projection changes. What the benchmark priced is the
  November turn alone. Six general folds, D.4 → B.7: MAE 6.79 → 6.82, RMSE
  43.6 → 42.0, bias −0.34 → −0.20, level accuracy unchanged, per-species
  moves of ±2 that look like retraining noise (no fold holds a December).
  Onset benchmark: alder d2/d3 timing 4.7/5.0 → 2.3/3.3 d, but hazel
  worse at three of five horizons (d5 1.0 → 4.0 d), Alnus 2026 over-
  predicted (ratio 1.00 → 1.29) and **false starts 46 → 59 run-days**,
  Betula 2024 now false-starting at every horizon: giving every December
  row a negative days-since-onset moves the early-season shape the model
  learned. Two seasons with warm Decembers (2020 onset DOY 13, 2023 DOY 11)
  are not enough to teach a rule; revisit with twelve, as the chill-unit
  note already says.

## Phase C — Season load (was 5.1) — **DONE**

- [x] **C.1 Prior-season features.** `src/season_load.py`: three features
  per species, all log ratios against the species' own history so that 0 is
  "average or unknown" — last season vs. the seasons before it, the last two
  seasons vs. the seasons before them, last season vs. the one before.
  Constant within a season year (boundary = the month after the shoulder
  month), read only from seasons that ended before the row, same two
  functions in the trainer and the forecaster context. 63 features.

  Same six folds, complete history:

  | | MAE | RMSE | Level acc. | Bias | Skill d1 / d5 |
  |---|---|---|---|---|---|
  | before backfill (60, era flag) | 7.3 | 46.4 | 76.9% | 0.0 | +27% / +37% |
  | complete history (60) | 7.8 | 46.7 | 75.6% | +1.0 | +22% / +33% |
  | **+ season load (63)** | **7.0** | **44.8** | 75.2% | **−0.2** | **+29% / +41%** |

  Better than both the honest baseline and the artefact it replaced, on MAE,
  RMSE, bias and skill; level accuracy is flat. Per species, the gain is in
  the mid-season: Fraxinus day-1 MAE 42.5 → 23.0, Quercus 34.9 → 26.6,
  Populus 65.5 → 54.4; Betula, Corylus and Poaceae move by less than 1.5.
  The family earns 4.3% of gain, `load_trend` most (2.0%).

  **What it did not do: fix the onset ramp.** The onset rollout is unchanged
  (MAE 88.5 → 89.9, timing within 2–5 days at every horizon, the same
  amplitude ratios of 0.1–0.3 in heavy years), and Alnus 2026 is now
  *over*-predicted 3.4× in its first five days. That is what alternation
  looks like at eight seasons: 2025 was the lightest birch year on record and
  2026 the heaviest, so "last season" pointed the wrong way, and Alnus had
  three heavy years in a row before a moderate start. The premise that
  season load was the largest missing signal for *onset amplitude* was
  wrong; it is a mid-season signal. The ramp is B.6.

## Phase D — Honesty of the output

- [x] **D.1 Publish confidence.** **Done.** Every point in the published
  `forecast.json` (the `to_web_dict()` measurement format) now carries
  `level`, `confidence` and `confidence_within_one` beside `value`. Additive
  — a frontend that ignores the keys keeps working — so the Vue app can pick
  them up whenever it likes; until it does, the numbers are at least there.
  The README's output section now documents the file that is actually
  published (it described `to_dict()`, which nothing publishes).
- [x] **D.2 Staleness guard (was 4.2).** **Done — a stale run is forecast
  and published as the longer-range forecast it is.** The live history has
  231 gaps longer than a window, most of them a day, the longest 13.5 days
  (June 2026). Before, the forecaster's origin was the first unobserved
  window of *today*, so after two silent days a window this afternoon was
  predicted at lead 1 over a lag block that had quietly ended two days
  earlier, with the day-1 confidence attached and nothing in the file. Now
  the origin is the window after the newest measurement: `lead_windows`
  counts from there (the model was trained on exactly that anchoring), and
  the confidence horizon is counted the way the calibration rollout counts
  it, eight windows per day from the origin. A run two days stale therefore
  publishes day-3 rates for tomorrow. So that those rates exist, the
  calibration rollout now runs to twice the shipped horizon
  (`CALIBRATION_HORIZON_DAYS = 10`; `calibrate --rebuild`), and a window
  beyond the furthest measured horizon gets that horizon's rate halved, the
  same rule as a species without a model. The output's `status.observations`
  block carries the newest measurement, its age in complete windows and a
  `stale` flag from eight windows (a day), per species and as a worst-case
  headline. On a normal run nothing changes: the newest measurement is one
  window old and the origin is where it always was.

  Measured on the same six folds, rolled out to 10 days (same origins, so
  days 1–5 are the standard benchmark and are unchanged):

  | | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | MAE | 7.1 | 7.0 | 6.7 | 6.5 | 6.6 | 6.4 | 6.5 | 7.0 | 8.0 | 8.6 |
  | level accuracy | 73.5% | 73.0% | 73.0% | 72.2% | 72.2% | 71.8% | 72.2% | 71.3% | 70.8% | 71.0% |
  | skill vs persistence | +32% | +36% | +40% | +45% | +42% | +46% | +50% | +49% | +46% | +45% |
  | Betula in-season MAE | 16.8 | 16.5 | 15.1 | 15.2 | 19.4 | 24.7 | 33.0 | 55.6 | 111.1 | 146.2 |
  | exact level, emitted rows | 67.6% | 66.5% | 65.8% | 63.8% | 62.6% | 61.6% | 62.2% | 59.9% | 58.7% | 58.9% |

  The pooled numbers barely move because most rows are quiet; the birch row
  is what a stale run costs, and it is why the guard matters: a forecast
  built on a block five days old misses the birch ramp by 6–9× more than a
  fresh one, and before D.2 it was published with the day-1 confidence.
  `src/confidence.json` is regenerated from this rollout: the overall rate
  and days 1–5 are unchanged (0.652 / 0.989), days 6–10 are new.
- [x] **D.3 Degradation flags (was 4.3).** **Done.** `status.defaulted` in
  the published file names every input group that fell back to a default
  this run, with the reason: `ndvi` (fetch failed or no composites → zeros),
  `dwd` (no index → blend skipped), `weather_soil` (Open-Meteo answered
  without soil variables → zeros), `upwind` (no station readings in the last
  7 days, for some or all species → zeros), `model` (species with no trained
  model → persistence) and `confidence` (no calibration table → fallback
  rates). `status.degraded` is the one-bit summary and the same line goes to
  the run log. CAMS is deliberately not listed: it has never been active in
  training or live, so its zero is the norm rather than a fallback. The
  weather forecast itself has no fallback — the run fails without it, which
  is the right behaviour and needs no flag.
- [x] **D.4 Level-threshold semantics (was 4.5).** **Done — levels are
  daily-mean levels.** The DWD/ePIN thresholds are defined on daily means;
  the app applied them to 3 h values. Now a window's level is the level of
  its calendar day's mean (`daily_levels` in `src/types.py`), in the
  forecaster, the rollout benchmark and the classic one alike; the values
  keep their 3 h shape, and the DWD blend nudges the day's mean and spreads
  it over the day's predicted windows. The model and its predictions are
  unchanged, so the benchmark is a re-scoring of the B.5 rollout under both
  definitions:

  | | exact level | within one | over-predicted | under-predicted |
  |---|---|---|---|---|
  | 3 h values (before) | 76.3% | 96.2% | 16.4% | 7.3% |
  | daily means (D.4) | 72.8% | 99.5% | 10.1% | 17.1% |

  Under the old definition 15% of the pollen-bearing windows in the history
  read a level above their own day's, and 78% of all windows were `none`
  (every night window); under the daily one `none` is 55% and `low` 31%,
  because the night windows of a pollen day are part of that day. That is
  why exact accuracy falls while within-one rises: the easy `none` rows
  became `low` rows the model must get within ±10 grains of daily mean, and
  it misses low on 17% of them. The onset benchmark is unaffected (its
  timing and amount scores were already on daily means). Confidence
  recalibrated on the same rollout under the new definition: on the rows the
  forecast emits (value > 0.5, n = 22,358) the published level is exactly
  right 65% of the time and within one level 99% (was 36% / 89%) — a level
  that belongs to the day is easier to get right than one that belongs to
  a window, and it is the one the user now sees. Persistence's level
  accuracy under the daily definition is 66.6% against the model's 73.5%
  at day 1, where the 3-hour definition had persistence *ahead* (78.7% vs
  76.8%) because predicting `none` at night was free.
- [x] **D.5 Discriminative confidence (was 4.6).** **Done — conformal.**
  `calibrate` now stores, per forecast horizon, the quantiles of the rollout's
  log-space residuals (`log1p(actual) − log1p(predicted)`), once over the
  emitted day means and once over the emitted windows. A prediction's
  `confidence` is the residual mass that keeps its daily mean inside its
  level's band, `confidence_within_one` the mass inside the neighbouring
  bands too, and each point carries `value_low`/`value_high`, the 80%
  central interval of the window residuals applied to its value (additive
  keys; a value of 0 has none). Leave-one-fold-out on the 10-day rollout's
  emitted day-rows:

  | scheme | ECE | corr. with being right | Brier |
  |---|---|---|---|
  | flat + horizon offset (D.1/D.4) | 0.072 | −0.072 | 0.239 |
  | conformal per horizon day (adopted) | 0.037 | 0.323 | 0.209 |
  | conformal pooled | 0.039 | 0.316 | 0.210 |
  | conformal per magnitude bin | 0.049 | 0.298 | 0.215 |
  | conformal per species | 0.073 | 0.225 | 0.229 |

  The flat table's ceiling was the table, not the model: the information
  that separates reliable predictions from unreliable ones is where the
  prediction sits relative to the thresholds, which the lookup keyed on the
  level threw away. Reliability held-out: stated 0.27 → right 11%, 0.45 →
  46%, 0.65 → 70%, 0.82 → 78%. Within-one stays calibrated (ECE 0.007) and
  now discriminates (corr. 0.17). The interval's coverage is 70–90% per
  held-out fold. The quantile-ensemble alternative (several `quantile_alpha`
  regressors) was not tried: it needs retraining and a second model per
  quantile, and the residual approach already gets the correlation from
  −0.07 to 0.32 with no model change. The flat rate stays in the table as
  the fallback for a prediction without a residual distribution.

## Phase E — Model correctness leftovers

- [x] **E.1 Time-based lag alignment (was 4.1).** **Done.** The lag block
  is built on the full 3 h grid (`trainer.lag_block_on_grid`) and
  `LagState.from_history` builds the same block the same way, so a lag of 8
  is the window 24 h earlier through an outage and `days_since_active` is
  a distance in time. The history has 231 gaps longer than a window — 184
  whole days in 2019–2020 when the source reported once a day, the longest
  13.5 days in June 2026 — and every one of them used to compact the block.
  Three fills were benchmarked on the same six folds (D.5 baseline 6.8 /
  43.6 / 72.8% / −0.3):

  | fill of a window inside a gap | MAE | RMSE | level acc. | bias | onset false starts |
  |---|---|---|---|---|---|
  | last measurement carried forward | 7.0 | 43.5 | 73.0% | +0.1 | 49 |
  | **same hour of the previous day, then carry-forward** | **6.9** | **43.4** | **73.3%** | **−0.1** | **43** |
  | as above, but training drops every block that spans a gap (−10% rows) | 6.9 | 43.8 | 72.7% | −0.3 | — |

  The diurnal fill is adopted: level accuracy +0.5 points at every horizon,
  bias at zero, false starts 46 → 43, for +0.1 MAE. Onset timing moved by
  one to three days in either direction on cells of three species-years
  (hazel d2 1.3 → 3.0, alder d1 2.0 → 6.3, birch d1 7.3 → 11.7 — see the
  E.2 entry, which then takes birch's d1 timing to 2.3), which is the size
  of the swing between two retrains of the same code. The benchmark's test
  months barely touch a gap, so what it measures is the training-set
  change; the serving-side correctness — an outage no longer shifts the
  block — is the point of the task and is pinned by a parity test that
  cuts a three-day gap into the fixture.
- [x] **E.2 One peak-emphasis mechanism (was 3.2).** **Done — the raised
  quantile survives, the value weights go.** It was not tidiness: the two
  compounded. Stage 2 was a quantile regressor (α 0.85–0.92 per species)
  *and* weighted by `1 + √value` plus tier bonuses, and weighting by the
  target inside a quantile loss shifts the effective quantile above the
  nominal one. A/B on the same six folds, on E.1:

  | arm | MAE | RMSE | level acc. | bias | onset false starts | birch onset d1 |
  |---|---|---|---|---|---|---|
  | E.1 (both mechanisms) | 6.9 | 43.4 | 73.3% | −0.1 | 43 | 11.7 d |
  | **quantile only** | **6.1** | **42.4** | **74.5%** | −1.7 | **8** | **2.3 d** |
  | value weights only (median regression) | 6.1 | 45.5 | 74.0% | −3.9 | — | — |
  | quantile only, α + 0.03 | 6.5 | 43.0 | 74.4% | −0.8 | — | — |

  Quantile-only is adopted: MAE −12%, level accuracy +1.2 points, skill vs
  persistence +30% → +38% at day 1, onset false starts 43 → 8 run-days and
  birch's onset timing 11.7 → 2.3 days at day 1. Its cost is a bias of
  −1.7 (the weights were pushing predictions up, which is what the near-zero
  bias was made of) and hazel/alder onset timing about two days later
  (hazel 1.3 → 4.3 d, alder 6.3 → 7.3 d at day 1, cells of three
  species-years). Tuning the survivor against the bias — every species'
  quantile up by 0.03 — buys half the bias back at +0.4 MAE, so the
  quantiles stay where they were. The extreme regressor keeps its own
  `1 + √value` weight: it is fitted to peak samples only and has no
  quantile to shift. The onset-ramp weight (B.6) stays too: it is
  label-driven, not value-driven.
- [x] **E.3 Log-space probability scaling (was 3.3).** **Benchmarked —
  the current form is kept, and now justified.** `TwoStageModel.predict`
  multiplies the log-space regression output by the clamped activation
  probability, which is a power transform (`count^p`, p ∈ [0.5, 1]) in
  real space rather than a hurdle model. Two alternatives, on the same six
  folds on top of E.2:

  | form | MAE | RMSE | level acc. | bias | onset false starts | hazel / alder / birch onset d1 |
  |---|---|---|---|---|---|---|
  | **log-space scaling (kept)** | **6.1** | 42.4 | **74.5%** | −1.7 | **8** | 4.3 / 7.3 / 2.3 d |
  | hurdle: scale after `expm1` | 6.6 | 42.2 | 73.5% | −0.5 | 52 | 0.7 / 2.7 / 7.3 d |
  | threshold: zero below p = 0.5, the regressor's value above | 6.9 | 42.3 | 72.0% | +0.1 | — | — |

  The shrinkage is what the level accuracy and the false-start count are
  made of: the hurdle form hands uncertain windows their full regressed
  value, which lifts the bias to −0.5 and brings hazel and alder onsets in
  two to four days earlier, but it costs a point of level accuracy, half a
  grain of MAE and 44 run-days of false starts, mostly birch. A model
  whose bias is the complaint should raise its quantile (E.2's tuning arm:
  −0.8 at +0.4 MAE), not change the blend. The justification lives next to
  the code.
- [x] **E.4 Beat persistence as the headline metric (was 3.4).** **Done.**
  The rollout report now opens with skill vs persistence per horizon — MAE
  and level accuracy against "the last measured window held flat" — and a
  one-line verdict (beats persistence at every horizon, or loses at some),
  before any absolute number. Currently +32% at day 1 rising to +45% at
  day 5, level accuracy 73.5% vs 66.6% at day 1. The report's preamble also
  stopped claiming the lags are fed from the model's own predictions; they
  have been measured since the direct model.

## Phase F — Mould spores

- [x] **F.1 Mould forecast.** **Done.** Munich has had a lot of mould in the
  air (September 2026 daily means of 110–375 spores/m³ against a 2024+
  median of 46), and the data was there all along: pollenscience.eu reports
  the ePIN samplers' fungal-spore aggregate as the taxon `Fungus`, for the
  Munich station and all four upwind stations, back to 2019 (asked without
  a `pollen` filter, the API lists every taxon a station recognises; a
  `probe-species` diagnostic prints them). `Fungus` is now a twelfth taxon
  in `ALL_SPECIES`, so it goes through the same collector, trainer,
  forecaster, benchmark and calibration as the pollen — with the
  differences that matter made explicit:

  - **History.** `backfill-species Fungus` (Actions mode
    `backfill-species`) fetched the Munich series and joined it to the
    weather and NDVI columns the history already carried for those windows
    (20 854 windows, 2019-01-01 to 2026-09-21), and the four upwind
    stations' series into `upwind.csv`.
  - **Levels.** No DWD index exists for spores. The thresholds (50 / 150 /
    300) are quantiles of the Munich station's own daily means in the 2024+
    sampler regime — median 46, 85th percentile ~150, 99th ~300 — because
    the regime changed: 2019–2023 has a median of 9 and a 99th percentile
    of 116, 2024–2026 a median of 46. "Moderate" is an above-average day,
    "very high" a top-1% one; September 2026 reads high to very high.
  - **Season.** All year (`(1, 12)`); the gate never zeroes it. The monthly
    means run 6 in December–January to 87 in July, and the diurnal shape is
    a midday peak (100 at 12:00 against 46 at 03:00, 2024+).
  - **No DWD blend, no phenology.** The onset machinery runs (it finds a
    "season start" in January and the rules are calibrated against it), but
    the features it produces carry nothing for a taxon without a flowering
    season; they are left in rather than special-cased.
  - **Hyperparameters, tuned on the benchmark** (six folds, the standard
    months, `Fungus` alone; persistence MAE 38.1, level 58.9%):

    | stage | setting | MAE | level acc. | bias | skill vs persistence |
    |---|---|---|---|---|---|
    | quantile α (depth 5, 300 trees, stage 3 at 50) | 0.50 | 27.6 | — | −5.8 | +27.5% |
    | | 0.60 | 28.2 | — | −2.4 | +26.0% |
    | | **0.70** | 28.6 | — | +0.8 | +25.1% |
    | | 0.80 | 31.3 | — | +7.4 | +17.9% |
    | | 0.85 (pollen default) | 33.4 | — | +11.5 | +12.3% |
    | | 0.90 | 37.3 | — | +18.2 | +2.3% |
    | | 0.95 | 43.1 | — | +27.1 | −13.1% |
    | stage-3 threshold (α 0.70) | none / **50** / 100 / 200 / 400 | 28.9 / 28.6 / 29.0 / 28.9 / 28.9 | 0.789 / 0.792 / 0.783 / 0.784 / 0.789 | −0.5 / +0.8 / +0.5 / −0.4 / −0.5 | +24.4% / +25.1% / +23.9% / +24.1% / +24.4% |
    | tree shape (α 0.70, stage 3 at 50, with the upwind series) | depth 3 / 300 | 29.6 | 0.778 | +2.7 | +22.4% |
    | | depth 4 / 300 | 29.4 | 0.781 | +2.1 | +23.0% |
    | | depth 5 / 300 | 29.3 | 0.783 | +2.3 | +23.1% |
    | | **depth 5 / 600** | 29.1 | **0.791** | +1.4 | +23.6% |
    | | depth 6 / 400 | 29.4 | 0.782 | +0.9 | +23.0% |
    | | depth 7 / 500 | **28.8** | 0.776 | −0.8 | +24.5% |

    The quantile is the parameter that matters: MAE rises monotonically
    with α and the pollen default is a +11 bias, because a taxon that is
    present all year has none of the zero-inflated, peaky distribution the
    high quantile was chosen for. Everything else moves the score by less
    than a point, which is the size of the swing between two retrains. The
    classifier depth (4 vs 5) was tied throughout. Adopted: α 0.70, stage 3
    at the default 50, depth 5 with 600 trees — the best level accuracy, and
    a slightly positive bias is the right side for a warning. The upwind
    stations' `Fungus` series is in the block (the level-accuracy rows are
    with it; MAE 28.6 → 29.3 for the same shape without and with it, inside
    the noise). Level accuracy is on the (50 / 150 / 300) thresholds;
    persistence scores 58.9%.

  Final, on the standard six folds with all twelve taxa (the pollen rows
  are identical to E.2's): mould MAE 27.8 / 29.5 / 30.3 at days 1 / 3 / 5
  against persistence 33.2 / 39.4 / 42.3 (+16% / +25% / +29% skill), level
  accuracy 81.1% / 78.0% / 76.7% against 56.5% / 56.5% / 53.3%, bias
  +0.8 to +1.9. Emitted mould points: level exactly right 79%, within one
  98.5%. `confidence.json` recalibrated on the 12-taxon 10-day rollout; the
  spore residuals have the pollen's spread (log-space std 0.80 vs 0.83)
  and a table split into pollen and spores calibrates worse held-out (ECE
  0.052 vs 0.042 pooled), so it stays pooled. What is left open: the
  upwind stations' spore series did not measurably help (MAE 28.6 → 29.3
  for the same shape without and with it, inside the noise), and the 2024
  regime change in the sampler means the model trains on two different
  scales of the same thing; a year-regime feature or a 2024+ training
  window is the obvious next experiment if the level accuracy is not
  enough.

## Suggested order

Phases A and C are done, and B.1–B.6 with them; A.5 and B.8 were tried and
parked, and so was B.7. Phases D and E are done; E.3 was benchmarked and the existing form kept. F.1 added the mould forecast. Nothing left on
the list claims the heavy-year ramp; that now needs a data source that sees
a season before Munich does (B.8, last paragraph).

## Done so far (first task list)

Measured over six folds (184 origins, 80,680 predictions), same folds throughout:

| Stage | day-1 MAE | day-5 MAE | day-1 bias | day-5 bias | day-5 skill vs persistence |
|---|---|---|---|---|---|
| original | 11.9 | 17.1 | +6.5 | +13.2 | −50.2% |
| 3.1 extreme gate | 9.8 | 14.0 | +4.0 | +9.7 | −22.9% |
| 3.5 direct forecast | 7.9 | 7.4 | +0.3 | +0.6 | +35.6% |
| Phase 2 pruning (73 → 60 features) | 7.6 | 7.2 | −0.2 | +0.4 | +37.3% |
| Phase A complete history (same model) | 8.2 | 7.7 | +0.9 | +1.5 | +32.5% |
| C.1 season load (63 features) | 7.4 | 6.8 | −0.2 | +0.1 | +40.7% |
| B.2 + B.3 onset calibration | 7.3 | 6.7 | −0.5 | −0.2 | +41.4% |
| B.4 rule-based readiness (62 features) | 7.5 | 6.9 | +0.1 | +0.3 | +39.9% |
| B.6 onset-ramp weighting | 7.3 | 6.8 | +0.1 | +0.5 | +40.3% |
| B.5 upwind stations (65 features) | 7.1 | 6.6 | −0.4 | −0.0 | +42.0% |

- Phase 1: rollout benchmark per horizon, feature-gain report, train/serve
  parity test, shared row-wise feature assembly (`src/features.py`). Fixed
  `days_since_active` (was a constant 0 in training).
- 3.1: extreme stage gated on a dedicated classifier.
- 3.5: direct multi-horizon forecast with a `lead_windows` feature — horizon
  decay went from +43% to −6%.
- Phase 2: feature set pruned to 60; the dropped 13% of gain made the model
  better at every horizon.
- 4.4: confidence calibrated from the benchmark (ECE 0.393 → 0.053), flat
  table because per-species/per-level tables overfit. Not yet published (D.1).
- Phase A: history backfilled (weather from 2019, NDVI from 2019, soil), a
  coverage guard at retrain, and the finding that 0.5 MAE of the previous
  score was an accidental "recent years" flag rather than model skill.
- C.1: three interannual season-load features; MAE 7.8 → 7.0 on complete
  data, the first model to beat the pre-backfill score honestly.
- B.2 + B.3: transport-aware onset detection and per-species forcing rules;
  MAE 7.0 → 6.9, onset-month MAE better for all three tree species.
- B.1: `benchmark-onset` scores the shipped rollout around the season starts.
- B.4: readiness features follow the per-species rule; `cold_to_warm_flip`
  dropped (62 features). Timing better, MAE 6.9 → 7.1, level 75.2 → 75.6%.
- B.6: onset-ramp weighting; MAE 7.0, RMSE 44.0, level 75.7% — best general
  numbers so far; heavy-year ramp amplitude unchanged.
