# Open points

The two review task lists (2026-09-10 and before) are worked through: every
item is done, tried and rejected on the benchmark, or benchmarked and kept as
it was. The record is below; this is what is still open.

## 1. The heavy-year ramp — blocked on data

Betula 2024 and 2026 still get 6–18% of what arrives in days 5–9 of the
season, exactly as before the season-load (C.1), ramp-weighting (B.6),
upwind (B.5) and upwind-load (B.8) work. The year's *amount* is a property
of the season, not of the last day, and no feature the model has can see it
in advance: the upwind stations see the same first days Munich sees, and
last season points the wrong way when years alternate (2025 was the lightest
birch year on record, 2026 the heaviest). What would carry it is a source
that sees the season before Munich does — a station far to the south-west,
or a CAMS backfill (the `cams_pollen` feature exists and has never had data;
`src/cams.py` needs ADS credentials and a historical download). This is the
one item with real product value left.

## 2. Mould — the sampler's regime change

The Munich station's spore counts changed scale in 2024 (2019–2023 median of
daily means 9, 2024+ median 46) and the `Fungus` model trains across both.
Level accuracy is 81% at day 1 against the (50 / 150 / 300) thresholds,
which are quantiles of the 2024+ regime. If that is not enough, the next
experiment is a regime feature or a 2024+ training window. The upwind
stations' spore series is in the block and did not measurably help (MAE
28.6 → 29.3 for the same shape without and with it, inside the noise).

## 3. Parked variants

On `claude/forecast-app-review-xwisqv`, each with its verdict below: A.5
(NaN passthrough), B.8 (upwind season load), B.7 (autumn forcing starts).
B.7 is worth re-running once twelve seasons exist; two warm Decembers in
eight are not enough to teach a rule. The other two lost and are kept only
as a record.

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
