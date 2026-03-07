"""S3 helpers for uploading forecasts and syncing historical data."""

import json
from pathlib import Path

import boto3
import pandas as pd

from .types import ForecastOutput


S3_REGION = "eu-central-1"


def upload_forecast(forecast: ForecastOutput, bucket: str, key: str = "forecast.json") -> None:
    """Upload forecast JSON to S3."""
    s3 = boto3.client("s3", region_name=S3_REGION)
    body = json.dumps(forecast.to_dict(), indent=2, ensure_ascii=False)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
        CacheControl="max-age=3600",
    )
    print(f"Uploaded forecast to s3://{bucket}/{key}")


def upload_csv(local_path: Path, bucket: str, key: str) -> None:
    """Upload a CSV file to S3 for backup / persistence."""
    s3 = boto3.client("s3", region_name=S3_REGION)
    s3.upload_file(str(local_path), bucket, key)
    print(f"Uploaded {local_path.name} to s3://{bucket}/{key}")


def download_csv(bucket: str, key: str, local_path: Path) -> bool:
    """Download a CSV from S3. Returns True if the file existed."""
    s3 = boto3.client("s3", region_name=S3_REGION)
    try:
        s3.download_file(bucket, key, str(local_path))
        print(f"Downloaded s3://{bucket}/{key} -> {local_path}")
        return True
    except s3.exceptions.ClientError:
        print(f"No existing file at s3://{bucket}/{key}")
        return False


def sync_historical_data(
    local_path: Path, bucket: str | None, key: str = "data/history.csv"
) -> pd.DataFrame:
    """
    Load historical data from local CSV, falling back to S3.
    Returns empty DataFrame if no data exists yet.
    """
    if local_path.exists():
        df = pd.read_csv(local_path, parse_dates=["date"])
        print(f"Loaded {len(df)} rows from {local_path}")
        return df

    if bucket:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if download_csv(bucket, key, local_path):
            return pd.read_csv(local_path, parse_dates=["date"])

    return pd.DataFrame()
