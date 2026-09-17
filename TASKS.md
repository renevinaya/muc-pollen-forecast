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
- [ ] **B.8 Upwind season load.** B.5's follow-up: a cumulative feature —
  the upwind stations' season-to-date total (log, since 1 Jan or since the
  species' upwind onset) and its ratio to the same total in Munich — so the
  model can read on day 5 of a season whether the region is having a heavy
  one. Accept on the B.1 ramp ratios for Betula 2024/2026, which every
  change so far has left at 0.06–0.18.
- [ ] **B.7 December continuity (was part of 5.3).** `gdd` and the forcing
  accumulation reset on 1 Jan, so a hazel season that starts in a warm
  December (2023 onset = 1 Jan, i.e. already running) is invisible to the
  onset features. Start the accumulation on 1 Nov of the previous year for
  Corylus and Alnus (LOO-check the start date as in B.3). Chill units
  themselves are **not** supported by the data at eight seasons; revisit
  when there are twelve.

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

- [ ] **D.1 Publish confidence.** Add `confidence` and `confidence_within_one`
  to `to_web_dict()`. This changes the schema the Vue frontend reads, so it
  wants the frontend in the loop — but until it is done 4.4 has no effect.
- [ ] **D.2 Staleness guard (was 4.2).** If the last observation is older
  than N windows, cap confidence and say so in the output.
- [ ] **D.3 Degradation flags (was 4.3).** Emit a per-run list of feature
  groups that were defaulted (NDVI, DWD, weather forecast fallback).
- [ ] **D.4 Level-threshold semantics (was 4.5).** Daily-mean DWD/ePIN
  thresholds are applied to 3 h values, overstating midday peaks. Either
  calibrate 3 h thresholds or compute levels on a daily aggregate. This also
  changes what "onset" means to a user, so do it before tuning B further.
- [ ] **D.5 Discriminative confidence (was 4.6).** Quantile-ensemble spread or
  conformal intervals over the rollout residuals. Schema change.

## Phase E — Model correctness leftovers

- [ ] **E.1 Time-based lag alignment (was 4.1).** Row-based `shift(n)` turns
  a station outage into "8 rows ago, whenever that was". Reindex each species
  frame to the full 3 h grid before shifting.
- [ ] **E.2 One peak-emphasis mechanism (was 3.2).** Bias is ~0 now, so this
  is tidiness, not accuracy.
- [ ] **E.3 Log-space probability scaling (was 3.3).**
- [ ] **E.4 Beat persistence as the headline metric (was 3.4).** Already
  reported; make it the first line of the benchmark output.

## Suggested order

Phases A and C are done, and B.1–B.6 with them; A.5 was tried and parked.
Next: D.1 → B.8 → D.4 → B.7 → D.2/D.3 → E.x → D.5. B.8 (upwind season load) is now the only
item on the list with a claim on the heavy-year ramp: B.5 showed the upwind
stations carry the information and that a last-day or last-week maximum
does not deliver it. D.1 early because it is a five-line change that makes
finished work visible.

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
