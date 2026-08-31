# Task list — parameter tidy-up and forecast improvements

Ordered so that measurement comes first: most later tasks change model behavior,
and none of them should be merged without a benchmark that can actually see the
difference. Tasks reference the review findings from the forecast-app review.

## Phase 1 — Fix the measurement (prerequisite for everything else)

- [ ] **1.1 Autoregressive rollout benchmark.** `temporal_split_evaluate`
  (src/evaluate.py) builds test-set lag features from *measured* values, so it
  scores 3h-ahead skill while the product ships a 5-day autoregressive
  forecast. Add a rollout mode that feeds predictions back into the lag
  features exactly as `generate_forecast` does, and report MAE / RMSE / level
  accuracy **per horizon day (1–5)**. This is the single most important task:
  day-3 to day-5 skill is currently unmeasured.
- [ ] **1.2 Fix the `benchmark [horizon]` CLI parameter.** It is passed as
  `n_folds` (src/main.py:362), not as a forecast horizon. Rename the fold
  count, and make `horizon` select the rollout depth from 1.1.
- [ ] **1.3 Per-feature gain report at retrain.** Dump XGBoost gain per feature
  (grouped by feature family) next to `_print_onset_calibration` in
  src/trainer.py, so pruning decisions in Phase 2 are data-driven instead of
  argued.
- [ ] **1.4 Train/serve parity test.** Build features for the same windows via
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
- [ ] **2.8 A/B the pruned set.** Run the Phase 1 benchmark with the pruned
  (~40-feature) set vs. the current 72; accept the pruning if per-horizon
  metrics do not regress.

## Phase 3 — Model correctness

- [ ] **3.1 Fix the extreme-regressor gate.** Stage 3 is blended whenever
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

1.1 → 1.2 → 1.3 → 1.4 (one PR: measurement), then 2.x as a single pruning PR
gated on 2.8, then 3.1 → 3.2 → 3.3 (each re-benchmarked), then 4.x
independently, then 5.1 and 5.2 (the two highest-value modeling additions),
then 5.3 → 5.4.
