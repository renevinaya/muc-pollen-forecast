# muc-pollen-forecast

ML-based daily pollen forecast for Munich, using XGBoost trained on historical pollen measurements and weather data.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Python backend (this repo)                     │
│                                                 │
│  1. Collector  — fetch pollen + weather daily   │
│     → append to data/history.csv                │
│                                                 │
│  2. Trainer    — train XGBoost per species       │
│     → save models to models/*.joblib            │
│                                                 │
│  3. Forecaster — predict 5-day pollen forecast  │
│     → upload forecast.json to S3                │
└────────────────────┬────────────────────────────┘
                     │ forecast.json
                     ▼
┌─────────────────────────────────────────────────┐
│  Vue frontend (muc-pollen)                      │
│  → fetches forecast.json from CloudFront        │
└─────────────────────────────────────────────────┘
```

## Data Sources

- **Pollen**: [LGL Bayern API](https://d1ppjuhp1nvtc2.cloudfront.net/measurements) — real-time pollen measurements for Munich (DEMUNC)
- **Weather**: [Open-Meteo](https://open-meteo.com/) — free forecast + historical archive API (no key required)

## Features Used by the Model

| Category | Features |
|----------|----------|
| Weather | temp max/min/mean, precipitation, wind speed, humidity, sunshine duration, radiation |
| Calendar | day of year (+ sin/cos encoding), month |
| Lag | pollen count at t-1/t-2/t-3, 3-day rolling mean, 7-day rolling mean |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Backfill historical data (run once to bootstrap)
python -m src.main backfill 365

# Train models on accumulated data
python -m src.main train

# Generate forecast
python -m src.main forecast

# Full daily pipeline (collect → train → forecast)
python -m src.main run
```

## Commands

| Command | Description |
|---------|-------------|
| `collect [days]` | Fetch recent pollen + weather, append to history (default: 14 days) |
| `train` | Train XGBoost model per species on all history |
| `forecast` | Generate 5-day forecast using trained models |
| `backfill [days]` | Bulk import historical data (default: 365 days) |
| `run` | Execute collect → train → forecast in sequence |

## Deployment

The project runs on AWS CodeBuild, triggered daily by EventBridge at 6 AM CET.

```bash
# Deploy infrastructure
aws cloudformation deploy \
  --template-file infrastructure/template.yaml \
  --stack-name muc-pollen-forecast \
  --capabilities CAPABILITY_IAM
```

Set `S3_BUCKET` environment variable to enable S3 uploads.

## Output Format

The forecast JSON (consumed by the Vue frontend) looks like:

```json
{
  "generated": "2026-03-04T05:00:00.000Z",
  "location": "DEMUNC",
  "forecast": [
    {
      "date": "2026-03-04",
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
```
