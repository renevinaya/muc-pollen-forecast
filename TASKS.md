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
- [ ] **A.5 Let XGBoost see missing values.** Replace the blanket
  `X.fillna(0)` in `prepare_training_data` (and the matching fills in
  `src/features.py` / `src/rollout.py`) with NaN passed through for the raw
  weather and NDVI columns, so a gap like the 2024 boundary-layer hole is
  "missing" rather than "zero". Benchmark on the six folds; accept if it does
  not regress.

## Phase B — Make the season start a first-class target

- [ ] **B.1 Onset rollout benchmark.** `benchmark --months 2026-02,2026-04`
  exists now (Phase A needed it to reproduce the baseline folds after the
  history grew). Still to do: the per-species-year timing, ±10-day and
  false-start report, so the *shipped* direct forecast is scored
  over the months containing each measured onset for Corylus, Alnus and
  Betula, for every season with enough history behind it. Report per
  species-year and horizon: (a) timing error of the first predicted 3-day run
  at or above the low threshold, (b) MAE and bias in the ±10-day window,
  (c) false starts (predicted runs more than 10 days before the real one).
  The existing `benchmark-onset` scores one-window-ahead with measured lags,
  which is not the product. The numbers in the box above are the first run
  of this benchmark and are the baseline every B task is judged against.
- [ ] **B.2 Robust onset detection for calibration.** `observed_onsets` is
  only ever applied to completed past seasons, so it does not need to be
  causal. Replace "first 3-day run ≥ low" with a definition that ignores a
  transport episode: either require the run to be followed by no return to
  zero within the next 7 days, or use "first day at which 5% of the season's
  total has accumulated". Under the 5% definition birch climatology alone is
  5.2 d LOO and the 2025 outlier moves from 4 March to 10 March — still an
  outlier, still a transport season, so B.5 matters too. Accept when the
  per-species LOO table above improves for all three species and the
  calibrated thresholds stop moving by >10% when a single year is dropped.
- [ ] **B.3 Species-specific forcing rules.** Give `src/onset.py` a per-species
  (start date, base temperature) pair instead of one global rule, chosen by
  leave-one-out on the history at calibration time — walk-forward, so year
  *Y* uses a rule and threshold fitted on years before *Y*. Starting points
  from the LOO table: Corylus 1 Jan / 3 °C, Alnus 15 Jan / 3 °C, Betula
  1 Mar / 5 °C. Add the rule to `_print_onset_calibration` so the retrain
  log shows which rule each species is running. **A species whose projection
  does not beat its own climatology in LOO must fall back to climatology** —
  that is the current state for birch, and the fallback is better than what
  ships.
- [ ] **B.4 Onset-phase features that survive B.3.** After B.3, re-read gain
  for `days_since_typical_onset`, `onset_anomaly`, `gdd_above_threshold`,
  `cold_to_warm_flip`, `consecutive_warm_hrs` per species. Expect the first
  two to rise for alder and birch. Drop any of the five that stays below 0.5%
  for every tree species — `cold_to_warm_flip` is at 0.0–0.1% today.
- [ ] **B.5 Upwind stations (was 5.2).** Pre-onset birch pollen is transport:
  in 2026 Munich measured 15–24 grains/m³ on 27 Feb – 5 Mar, five weeks before
  local onset; the 2025 "onset" was the same thing. Nothing in the feature set
  can see it coming. The pollenscience.eu client already queries two Munich
  codes; add one or two upwind stations lagged 3–24 h. Accept on B.1 false
  starts and ±10-day MAE.
- [ ] **B.6 Ramp amplitude at onset.** With the lag block near zero, the
  extreme stage's gate (`P(value > threshold)`) and the quantile regressor
  both learn from rows where big counts were preceded by big counts. Test
  (a) a sample weight that up-weights the first 14 days after each measured
  onset, (b) an explicit `days_since_onset_projected` × `season_load`
  interaction once C.1 exists, and (c) whether the extreme stage should be
  allowed to fire on phenology alone. Accept on the amplitude table above:
  the 0–4 d and 5–9 d ratios for heavy years must move toward 1 without the
  light-year ratio (Corylus 2024, 1.3) getting worse.
- [ ] **B.7 December continuity (was part of 5.3).** `gdd` and the forcing
  accumulation reset on 1 Jan, so a hazel season that starts in a warm
  December (2023 onset = 1 Jan, i.e. already running) is invisible to the
  onset features. Start the accumulation on 1 Nov of the previous year for
  Corylus and Alnus (LOO-check the start date as in B.3). Chill units
  themselves are **not** supported by the data at eight seasons; revisit
  when there are twelve.

## Phase C — Season load (was 5.1)

- [ ] **C.1 Prior-season features.** Per species: last season's total, the
  two-season mean, and last season's total as a ratio of the multi-year
  mean. Computed at the season boundary so they are constant within a season
  and causal. Alternation is measurable (Betula lag-1 corr −0.41, Alnus
  +0.54), and the onset-week amplitude — 20 to 811 for birch — is the largest
  unexplained variance in the onset window. Accept on B.1 ±10-day MAE and on
  the general benchmark's Betula/Alnus/Corylus rows.

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

Phase A is done. Next: C.1 → B.1 → B.2 → B.3 → B.4 → B.6 → B.5 → A.5 → D.1 →
D.4 → B.7 → D.2/D.3 → E.x → D.5. C.1 first because Phase A showed the model
had been leaning on an accidental "recent years" flag worth 0.5 MAE, and a
season-load feature is the honest version of it; B.1 before any B change so
there is a baseline; D.1 early because it is a five-line change that makes
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
