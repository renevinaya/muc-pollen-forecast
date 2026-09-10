# Task list — parameter tidy-up and forecast improvements

Ordered so that measurement comes first: most later tasks change model behavior,
and none of them should be merged without a benchmark that can actually see the
difference. Tasks reference the review findings from the forecast-app review.

## Phase 1 — Fix the measurement — **DONE**

Result: the rollout benchmark says the model **loses to a persistence baseline
at every horizon** (day-1 MAE 6.6 vs 4.1; level accuracy 76% vs 81%), while
carrying a bias of +5 to +7 that persistence does not have. Its RMSE is *better*
than persistence (21–23 vs 26–29), so the model is buying peak capture with a
systematic over-prediction that costs it every ordinary window. Horizon
degradation is comparatively mild (MAE +33% from day 1 to day 5), so the
calibration — not the lag cascade — is the dominant problem. That promotes
Phase 3 ahead of Phase 2.

Two defects were found and fixed on the way:

* `days_since_active` was a **constant 0** in training (a cumulative sum that
  does not advance while a species is inactive) while the forecaster served a
  live count — a dead feature and a train/serve skew at once. It is worth 6.5%
  of model gain now that it varies.
* Benchmark folds sampled evenly along the timeline landed **all three in
  August**, when every tree species is dormant. Folds are now spread across the
  calendar year.

- [x] **1.1 Autoregressive rollout benchmark.** `temporal_split_evaluate`
  (src/evaluate.py) builds test-set lag features from *measured* values, so it
  scores 3h-ahead skill while the product ships a 5-day autoregressive
  forecast. Add a rollout mode that feeds predictions back into the lag
  features exactly as `generate_forecast` does, and report MAE / RMSE / level
  accuracy **per horizon day (1–5)**. This is the single most important task:
  day-3 to day-5 skill is currently unmeasured.
- [x] **1.2 Fix the `benchmark [horizon]` CLI parameter.** It is passed as
  `n_folds` (src/main.py:362), not as a forecast horizon. Rename the fold
  count, and make `horizon` select the rollout depth from 1.1.
- [x] **1.3 Per-feature gain report at retrain.** Dump XGBoost gain per feature
  (grouped by feature family) next to `_print_onset_calibration` in
  src/trainer.py, so pruning decisions in Phase 2 are data-driven instead of
  argued.
- [x] **1.4 Train/serve parity test.** Build features for the same windows via
  `prepare_training_data` and via the forecaster's feature assembly, and assert
  they match. The two code paths construct 72 features independently; today
  only the onset features have tests pinning parity.

## Phase 2 — Parameter tidy-up (validated against the Phase 1 benchmark)

Raw columns stay in `history.csv` (collection is unchanged); pruning is an edit
to the feature lists in src/types.py plus a retrain, so it is cheap to A/B.

- [ ] **2.1 Drop near-duplicate temperature stats.** `temperature_max` /
  `temperature_min` over a 3h window are nearly identical to
  `temperature_mean`; `temp_slope_3h` and `temp_variance_3h` carry the rest.
  Keep mean + slope + variance.
- [ ] **2.2 Keep one radiation measure.** `sunshine_duration`,
  `shortwave_radiation_sum` and `direct_radiation_sum` are collinear; keep one
  (candidate: shortwave) and drop the other two. Re-derive `temp_x_sunshine`
  from the survivor or drop it too.
- [ ] **2.3 Trim the wind family from 7 derived features to ~4.** Keep
  `wind_speed_max`, `wind_dir_sin`, `wind_dir_cos`, and a single transport
  interaction. While here, revisit the N/S-only transport axis — notable Munich
  transport episodes (early birch) often arrive from the NE/E.
- [ ] **2.4 Drop raw `day_of_year` and `month`.** Five overlapping calendar
  encodings exist; the raw ones let trees memorize calendar dates from ~8
  seasons and reproduce climatology, masking the weather signal. Keep the
  sin/cos pair, `season_active`, and `days_since_typical_onset`.
- [ ] **2.5 Drop `cape_max`.** Thunderstorm-asthma is real but too rare at this
  data size to be anything but noise; confirm via the 1.3 gain report.
- [ ] **2.6 NDVI: drop `evi`, widen the footprint.** EVI duplicates NDVI, and a
  single 250 m pixel at the city-center coordinates (`kmAboveBelow: 0` in
  src/ndvi.py) measures urban greenery, not the regional source areas. Either
  average a few km around Munich or drop the NDVI family if the gain report
  shows nothing.
- [ ] **2.7 CAMS hygiene.** Models trained while `cams_pollen` was constant-zero
  must not silently start receiving live CAMS values ("becomes live at
  inference automatically"). Store a trained-with-CAMS flag in the model
  container and zero the feature at inference when it is unset. Also fix the
  UTC-vs-Europe/Berlin misalignment of CAMS 3h windows (src/cams.py).

Measured gain shares from the first report (share of total across all models):
lag 50.3%, weather-derived 20.2%, weather 11.7%, calendar 10.2%, NDVI 2.6%,
intra-day 2.4%, phenology 2.1%, season 0.4%, CAMS 0.0% (never split on). The
single largest features are `pollen_max_8` (13.2%), `pollen_rolling_8` (7.7%),
`pollen_lag_1` (6.8%) and `days_since_active` (6.5%).

- [ ] **2.8 A/B the pruned set.** Run the Phase 1 benchmark with the pruned
  (~40-feature) set vs. the current 72; accept the pruning if per-horizon
  metrics do not regress.

## Phase 3 — Model correctness

**3.1 is done.** Gating the stage-3 blend on a dedicated P(value > threshold)
classifier instead of P(value > 0) improved every metric at every horizon.
Measured over six folds spread across the year (Sep, Nov, Jan, Mar, May, Jul),
184 origins, 80 680 predictions:

| Horizon | MAE | RMSE | Level acc. | Bias | vs persistence |
|---------|-----|------|-----------|------|----------------|
| day 1 | 11.9 → **9.8** | 52.0 → **51.1** | 73.7 → **75.6%** | +6.5 → **+4.0** | −14.1% → **+6.2%** |
| day 3 | 15.6 → **12.4** | 52.3 → **50.5** | 71.5 → **73.7%** | +11.0 → **+7.4** | −38.7% → **−10.5%** |
| day 5 | 17.1 → **14.0** | 54.6 → **53.6** | 71.2 → **73.4%** | +13.2 → **+9.7** | −50.2% → **−22.9%** |

MAE fell ~18% at every horizon and bias ~38%, with RMSE slightly better too —
so the peak capture the blend was supposed to buy was never being delivered.
On real data the blend now fires on 4–7% of windows (was 20–28%), and when it
fires the truth is above the threshold 97–99% of the time (was ~25%).

**Correction to an earlier version of this file.** The first write-up of 3.1
used three folds and claimed the model beat persistence from day 3 on and that
horizon degradation had gone from +33% to −3%. Both were artefacts of that
sample — Sep, Jan and May are quiet months. On six folds the model beats
persistence only at day 1, and degradation is +44% before 3.1 and +43% after.
The default fold count is now 6.

The lag cascade this exposed — MAE rising 43% across the horizon while bias
more than doubled — turned out to be the dominant problem, and 3.5 fixed it.

- [x] **3.1 Fix the extreme-regressor gate.** Stage 3 is blended whenever
  `prob_active > 0.6` (src/trainer.py:469), but that is P(pollen > 0), not
  P(pollen > 50) — in peak season the classifier sits at ~1.0 for weeks, so a
  model trained *only* on >50 samples gets its full 70 % weight on every
  ordinary in-season window. Train a dedicated P(> extreme_threshold)
  classifier and gate on that. Also fix the comment/code mismatch: the comment
  says squared error "on raw (non-log) values" but the model is fit on log `y`.
- [ ] **3.2 One peak-emphasis mechanism, not three.** Quantile α = 0.85–0.92,
  √-value + tier sample weights *inside the same quantile loss*, and the
  stage-3 blend all push predictions upward and compound. Weighting by the
  target inside quantile loss also shifts the effective quantile above the
  nominal α. Keep either the raised quantile or the tier weights, and tune the
  survivor against the benchmark's bias analysis.
- [ ] **3.3 Reconsider log-space probability scaling.**
  `TwoStageModel.predict` multiplies the log-space prediction by the clamped
  probability, which is a power transform (~`count^p`) in real space. Move to a
  hurdle formulation (scale after `expm1`) or justify the current shrinkage
  against the benchmark.

## Phase 4 — Robustness and honesty of the output

- [ ] **4.1 Time-based lag alignment.** Lag features use row-based `shift(n)`
  (src/trainer.py `_add_lag_features`), so a station outage silently turns
  "24h ago" into "8 rows ago, whenever that was" — in training and at forecast
  time. Reindex each species frame to the full 3h grid before shifting so gaps
  become NaN and are handled explicitly.
- [ ] **4.2 Staleness guard at forecast time.** If the last observation is older
  than N windows, cap confidence and say so in the output; today a stale
  history still forecasts at 0.90 confidence from outdated lags.
- [ ] **4.3 Degradation flags in forecast.json.** Fail-open is right for
  availability, but NDVI/CAMS/DWD/pollen can silently default to zeros for
  weeks. Emit a per-run list of feature groups that were defaulted.
- [ ] **4.4 Calibrated confidence.** Replace the invented 0.90 − 0.08/day decay
  with per-species, per-horizon empirical error from the Phase 1 benchmark.
- [ ] **4.5 Level-threshold semantics.** `value_to_level` applies daily-mean
  DWD/ePIN-style thresholds to 3h window values, overstating midday peaks.
  Either calibrate 3h thresholds or compute levels on a daily aggregate.

## Phase 5 — New signal (prioritized by expected value)

- [ ] **5.1 Interannual load / masting features.** Birch (and oak) alternate
  high and low years; nothing in the feature set crosses seasons. Add last
  season's cumulative total (or its anomaly vs. the species' multi-year mean)
  per species — the largest missing signal for *amplitude*, and Betula is the
  most allergologically important species here.
- [ ] **5.2 Upwind stations as features.** The pollenscience.eu client already
  queries two Munich codes; add 1–2 upwind stations (50–200 km, e.g. toward
  Augsburg / the north-east) lagged by a few hours to a day. Best available
  predictor of transport episodes — far stronger than wind direction alone.
- [ ] **5.3 Chilling accumulation.** Standard phenology is chill + forcing, not
  forcing alone. Add autumn/winter chill units, and fix the December GDD
  incoherence: `gdd` resets on Jan 1, so in the December shoulder month — the
  exact warm-December-hazel case the season shoulder was built for — the onset
  features are meaningless.
- [ ] **5.4 Post-onset frost interaction.** An explicit "frost after onset"
  feature (catkin damage) — cheap to add once 5.3's plumbing exists; low
  priority until the gain report says otherwise.

## Suggested order of execution

Phase 1 is done. The benchmark it produced changes the order of the rest:
**Phase 3 now comes before Phase 2.** The model's problem is calibration and
horizon decay, not feature count — a pruning pass would shave training time
without touching either.

Phase 1's finding that horizon degradation was mild, and that calibration was
therefore the dominant problem, was drawn from three folds and does not hold on
six: degradation is ~+44%, and 3.1 did not reduce it. Both problems are real;
the lag cascade is the larger one.

3.1 and 3.5 are done, and between them took day-5 MAE from 17.1 to 7.4 and
day-5 bias from +13.2 to +0.6. The model now beats persistence at every horizon
by 25–38%. Next: 3.2 → 3.3 (each re-benchmarked against
`benchmark 5 --folds 3`, with beating persistence on MAE *and* level accuracy
as the bar), then 2.x as a single pruning PR gated on 2.8, then 4.x
independently, then 5.1 and 5.2 (the two highest-value modeling additions),
then 5.3 → 5.4.

Add to Phase 3, from what the rollout showed:

- [x] **3.5 Attack the horizon decay directly.** **Done — the largest single
  improvement so far.** The forecast is now direct: lag features are anchored
  at the forecast origin instead of the target window, with a `lead_windows`
  feature saying how far ahead the window is, so nothing is fed back and no
  bias can compound. Over the same six folds:

  | Horizon | MAE | Bias | Level acc. | vs persistence |
  |---------|-----|------|-----------|----------------|
  | day 1 | 9.8 → **7.9** | +4.0 → **+0.3** | 75.6 → **76.9%** | +6.2% → **+24.6%** |
  | day 3 | 12.4 → **7.5** | +7.4 → **+0.3** | 73.7 → **76.7%** | −10.5% → **+33.0%** |
  | day 5 | 14.0 → **7.4** | +9.7 → **+0.6** | 73.4 → **76.5%** | −22.9% → **+35.6%** |

  Horizon decay is gone (MAE −7% from day 1 to day 5, was +43%), and the model
  now beats persistence at every horizon instead of only at day 1. Bias is
  near zero — better calibrated than persistence, which sits at +1.5 to +2.8.
  Worst-decaying species gained most at day 5: Fraxinus 115.5 → 47.2, Betula
  57.0 → 17.7, Corylus 35.3 → 20.6.

  Note this also resolves most of what 3.2 was for: the residual +4.0 day-1
  bias that the stacked peak-emphasis mechanisms were blamed for is now +0.3.
  3.2 is still worth doing on its merits (two mechanisms doing one job is hard
  to reason about), but it is no longer urgent.
- [ ] **3.4 Beat persistence.** Track model-vs-persistence MAE per horizon as
  the headline metric. A forecast that loses to "nothing changes" has no claim
  on a user's attention, whatever its RMSE.
