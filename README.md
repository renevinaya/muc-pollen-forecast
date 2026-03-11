# muc-pollen-forecast

ML-based pollen forecast for Munich at 3-hour resolution, using a three-stage XGBoost pipeline (classifier + quantile regressor + extreme regressor) trained on historical pollen measurements, weather data, satellite vegetation indices, and phenological observations.

**Species covered (11):** Alnus, Ambrosia, Artemisia, Betula, Corylus, Fraxinus, Poaceae, Populus, Quercus, Salix, Urtica

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Python backend (this repo)                              │
│                                                          │
│  1. Collector  — fetch pollen + weather + NDVI (3h res)  │
│     → append to data/history.csv                         │
│                                                          │
│  2. Trainer    — train three-stage XGBoost per species   │
│     → save models to models/*.joblib                     │
│                                                          │
│  3. Forecaster — predict 5-day pollen forecast           │
│     → upload forecast.json to S3                         │
│                                                          │
│  4. Evaluator  — walk-forward benchmark + DWD comparison │
│     → data/benchmark_results.csv                         │
└───────────────────────┬──────────────────────────────────┘
                        │ forecast.json
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Vue frontend (muc-pollen)                               │
│  → fetches forecast.json from CloudFront                 │
└──────────────────────────────────────────────────────────┘
```

## Data Sources

| Source | API | What it provides |
|--------|-----|------------------|
| [pollenscience.eu](https://pollenscience.eu/api/measurements) | Pollen measurements | Primary source: 3-hour pollen counts for Munich (station DEMUNC, 2019+) |
| [LGL Bayern](https://d1ppjuhp1nvtc2.cloudfront.net/measurements) | Pollen measurements | Alternative: real-time 3-hour pollen counts for Munich |
| [Open-Meteo](https://open-meteo.com/) | Weather forecast + historical archive | Hourly weather aggregated to 3-hour windows: temperature, precipitation, wind speed/direction, humidity, sunshine, radiation (no API key required) |
| [MODIS (ORNL DAAC)](https://modis.ornl.gov/rst/api/v1) | NDVI / EVI satellite data | MOD13Q1 250 m 16-day vegetation indices, cubic-interpolated to daily resolution |
| [DWD Open Data](https://opendata.dwd.de/) | Pollenflug-Gefahrenindex + CDC Phenology | Official pollen danger levels for Oberbayern (partregion 121); multi-decade flowering-onset observations near Munich |

## Model

Each species gets a **three-stage pipeline** with species-specific hyperparameters:

1. **Stage 1 — XGBClassifier**: predicts P(pollen > 0). 200 estimators, learning rate 0.08, adaptive `scale_pos_weight`.
2. **Stage 2 — XGBRegressor**: predicts log1p(pollen count) via quantile regression. Sample-weighted by `1 + √(value)` with tier bonuses (+8 for >100, +20 for >500, +40 for >1000).
3. **Stage 3 — Extreme Regressor** (optional): trained only on high-pollen samples (>50), uses squared error on log-space. Blended with Stage 2 when classifier confidence > 0.6.

**Species-specific tuning:**

| Species | Clf depth | Reg depth | Reg estimators | Quantile α |
|---------|-----------|-----------|----------------|------------|
| Corylus, Alnus | 5 | 7 | 500 | 0.92 |
| Urtica, Poaceae | 5 | 6 | 400 | 0.90 |
| Quercus | 5 | 6 | 400 | 0.88 |
| Populus | 4 | 6 | 400 | 0.88 |
| Others (default) | 4 | 5 | 300 | 0.85 |

**Combined prediction**: regression output is scaled by clamped activation probability. Out-of-season species are forced to zero using species-specific season windows.

**Pollen levels** are assigned using species-specific thresholds (based on DWD/ePIN): `none`, `low`, `moderate`, `high`, `very_high`.

## Features (57 total)

| Category | Count | Features |
|----------|-------|----------|
| Weather | 9 | temp max/min/mean, precipitation, wind speed, wind direction, humidity, sunshine duration, radiation |
| Calendar | 4 | day of year, sin/cos encoding, month |
| Time-of-day | 3 | hour of day (0/3/6/.../21), sin/cos hour encoding |
| Season | 1 | binary `season_active` per species |
| Weather-derived | 24 | GDD + species GDD threshold, 3/7-day rolling temp/sunshine/rain, temp deltas (1d/3d), cold-to-warm flip, consecutive warm hours, dry streak, temp×sunshine, dry+warm, warming trend, wind×dry+warm, wind direction sin/cos, wind from south/north, transport south/north |
| NDVI | 3 | NDVI, EVI, NDVI delta (green-up rate) |
| Phenology | 2 | days since typical flowering onset, onset anomaly vs. historical mean |
| Lag | 11 | pollen at t-1/t-2/t-3/t-8(24h)/t-16(48h)/t-56(7d), 24h + 7d rolling mean, 24h + 7d rolling max, days since active (all log-space) |

Lag features are computed autoregressively during forecasting — each 3-hour window's prediction feeds into subsequent lag inputs.

## Setup

```bash
uv sync
```

Requires Python ≥ 3.11. Dependencies: httpx, pandas, numpy, xgboost, scikit-learn, joblib, boto3.

## Usage

```bash
# Backfill historical data (run once to bootstrap)
python -m src.main backfill 365

# Train models on accumulated data (needs ≥ 14 data points per species)
python -m src.main train

# Generate forecast (writes to data/forecast.json locally, or uploads to S3 when S3_BUCKET is set)
python -m src.main forecast

# Full daily pipeline (collect → train → forecast)
python -m src.main run
```

## Commands

| Command | Description |
|---------|-------------|
| `collect [days]` | Fetch recent pollen + weather + NDVI at 3h resolution, append to history (default: 14 days) |
| `train` | Train three-stage XGBoost models per species on all history |
| `forecast` | Generate 5-day forecast at 3h resolution using trained models |
| `backfill [days]` | Bulk import historical pollen, weather, and NDVI data (default: 365 days) |
| `backfill-ps [start_year]` | Bulk import from pollenscience.eu at 3h resolution (default: 2019, 5s rate limit) |
| `benchmark [horizon]` | Walk-forward evaluation with monthly CV folds and DWD comparison (default horizon: 1) |
| `dwd` | Display the current DWD pollen danger index for Oberbayern |
| `phenology` | Download DWD phenology data and show flowering-onset statistics |
| `run` | Execute collect → train → forecast in sequence |

## AWS Deployment

The project runs on AWS, using CodeBuild for daily forecast generation, S3 for storage, and CloudFront for serving the forecast JSON to the Vue frontend.

### Prerequisites

- AWS CLI installed and configured with appropriate credentials
- Permissions to create CloudFormation stacks, S3 buckets, CloudFront distributions, CodeBuild projects, EventBridge rules, and IAM roles
- A GitHub repository containing this code (the CodeBuild project pulls source from GitHub)

### Infrastructure Overview

The CloudFormation template at `infrastructure/template.yaml` provisions:

| Resource | Type | Purpose |
|----------|------|---------|
| `ForecastBucket` | S3 Bucket | Stores `forecast.json` (public read), `data/history.csv` backup, model artifacts. Named `muc-pollen-forecast-<AccountId>`. CORS enabled for `*`. Lifecycle rule expires objects under `data/` after 730 days. |
| `ForecastBucketPolicy` | S3 Bucket Policy | Allows public `s3:GetObject` on `forecast.json` only |
| `ForecastDistribution` | CloudFront Distribution | HTTPS CDN in front of the S3 bucket. Default root object: `forecast.json`. Price class: `PriceClass_100` (North America + Europe) |
| `ForecastCodeBuild` | CodeBuild Project | ARM container (`amazonlinux2-aarch64-standard:3.0`, Python 3.12). Pulls source from GitHub, runs `buildspec.yml`. `S3_BUCKET` env var set automatically. Timeout: 15 min |
| `DailyScheduleRule` | EventBridge Rule | Cron trigger at `0 5 * * ? *` (5:00 UTC / 6:00 CET) to start the CodeBuild build daily |
| `CodeBuildServiceRole` | IAM Role | Grants CodeBuild access to CloudWatch Logs and read/write on the S3 bucket |
| `EventBridgeRole` | IAM Role | Grants EventBridge permission to call `codebuild:StartBuild` |

### Deploy the Stack

```bash
aws cloudformation deploy \
  --template-file infrastructure/template.yaml \
  --stack-name muc-pollen-forecast \
  --capabilities CAPABILITY_IAM
```

To use a different GitHub repo or schedule:

```bash
aws cloudformation deploy \
  --template-file infrastructure/template.yaml \
  --stack-name muc-pollen-forecast \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    GitHubRepo=https://github.com/<your-user>/muc-pollen-forecast.git \
    ScheduleExpression="cron(0 6 * * ? *)"
```

### After Deployment

1. **Get the stack outputs** (bucket name, CloudFront URL, CodeBuild project name):

   ```bash
   aws cloudformation describe-stacks \
     --stack-name muc-pollen-forecast \
     --query "Stacks[0].Outputs" \
     --output table
   ```

2. **Connect CodeBuild to GitHub**: on first deploy, the CodeBuild project needs a GitHub connection. In the AWS Console, navigate to CodeBuild → Build projects → `muc-pollen-forecast` → Edit → Source, and authorize access to your repository. Alternatively, create a CodeBuild credential beforehand:

   ```bash
   aws codebuild import-source-credentials \
     --server-type GITHUB \
     --auth-type PERSONAL_ACCESS_TOKEN \
     --token <your-github-pat>
   ```

3. **Trigger a manual build** to verify the pipeline works end-to-end:

   ```bash
   aws codebuild start-build --project-name muc-pollen-forecast
   ```

4. **Verify the forecast** is accessible via CloudFront:

   ```bash
   # Get the CloudFront URL from stack outputs, then:
   curl https://<distribution-id>.cloudfront.net/forecast.json
   ```

### Build Pipeline

The CodeBuild project runs `buildspec.yml`, which:

1. Installs Python 3.12 and uv, then `uv pip install --system .`
2. Runs `python -m src.main run` (collect → train → forecast)
3. The forecast step uploads `forecast.json` and backs up `data/history.csv` to S3 (because `S3_BUCKET` is set)

### Environment Variables

| Variable | Set by | Description |
|----------|--------|-------------|
| `S3_BUCKET` | CloudFormation (CodeBuild env) | S3 bucket name. When set, forecast is uploaded to S3 and history is backed up. When unset, forecast is written to `data/forecast.json` locally. |

### Local Development

Without AWS, the app works entirely locally — forecast output is written to `data/forecast.json` and no S3 interaction occurs.

## Forecast Confidence

Confidence scores start at **0.90** for day 1 and decrease by **0.08** per additional forecast day (0.90, 0.82, 0.74, 0.66, 0.58). If no trained model exists for a species, confidence is halved. Scores are clipped to [0.20, 0.95]. Species with predicted value ≤ 0.5 are filtered from the output.

## Output Format

The forecast JSON (consumed by the Vue frontend) uses 3-hour windows:

```json
{
  "generated": "2026-03-04T05:00:00.000Z",
  "location": "DEMUNC",
  "forecast": [
    {
      "date": "2026-03-04",
      "windows": [
        {
          "from": "06:00",
          "to": "09:00",
          "species": [
            {
              "name": "Alnus",
              "level": "moderate",
              "value": 35.2,
              "confidence": 0.9
            }
          ]
        }
      ]
    }
  ]
}
```

A secondary `to_web_dict()` format is also available, restructured as species-centric measurements with Unix timestamps matching the LGL Bayern API format.
