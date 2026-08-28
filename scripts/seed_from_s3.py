#!/usr/bin/env python3
"""
One-shot migration: copy pipeline state out of the S3 bucket into the data release.

This is temporary scaffolding for the AWS -> GitHub Actions move. Once the data
release is seeded, delete this file and the `seed` mode in the pipeline workflow.

The bucket only needs public `s3:GetObject` on `data/*` and `models/*` while this
runs — no bucket listing, because every key is derivable:

    data/history.csv
    data/phenology.csv
    models/<species>.joblib   for each species in ALL_SPECIES

Re-lock the bucket as soon as the run succeeds.
"""

import os
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import store  # noqa: E402
from src.collector import DATA_DIR, HISTORY_FILE  # noqa: E402
from src.types import ALL_SPECIES  # noqa: E402

# Mirrors src.trainer.MODELS_DIR. Imported by value rather than from trainer so
# this script does not pull in xgboost/joblib just to resolve a path; a mismatch
# would show up immediately as "0/11 models retrieved".
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# Overridable so the import can be pointed at a different endpoint (or a local
# stand-in) without editing this file.
BASE = os.environ.get("SEED_S3_BASE", "https://muc-pollen-forecast.s3.eu-central-1.amazonaws.com").rstrip("/")
TIMEOUT = httpx.Timeout(30.0, read=600.0)


def fetch(key: str) -> bytes | None:
    """Download one object. Returns None if it is absent; exits if it is locked."""
    url = f"{BASE}/{key}"
    resp = httpx.get(url, follow_redirects=True, timeout=TIMEOUT)

    if resp.status_code == 403:
        sys.exit(
            f"\n403 Access Denied for {key}.\n"
            f"The bucket policy must allow public s3:GetObject on data/* and "
            f"models/* while this runs.\n"
        )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    print(f"  {key:<32} {len(resp.content) / 1e6:>8.1f} MB")
    return resp.content


def main() -> None:
    print(f"Seeding the data release from {BASE}\n")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- history (required) ---
    print("Downloading:")
    history = fetch("data/history.csv")
    if history is None:
        sys.exit("data/history.csv not found in the bucket — nothing to seed.")
    HISTORY_FILE.write_bytes(history)

    df = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    if df.empty:
        sys.exit("history.csv is empty — refusing to seed.")
    days = df["date"].dt.normalize().nunique()
    print(
        f"\nHistory: {len(df):,} rows, {days:,} distinct days, "
        f"{df['date'].min()} -> {df['date'].max()}"
    )

    # --- phenology (optional) ---
    pheno_file = DATA_DIR / "phenology.csv"
    phenology = fetch("data/phenology.csv")
    if phenology is not None:
        pheno_file.write_bytes(phenology)
    else:
        print("  data/phenology.csv absent — the monthly run will refetch it from DWD.")

    # --- models (optional individually; a missing one is just retrained) ---
    print("\nModels:")
    found = 0
    for species in ALL_SPECIES:
        blob = fetch(f"models/{species}.joblib")
        if blob is None:
            print(f"  {species:<32} {'missing':>8}")
            continue
        (MODELS_DIR / f"{species}.joblib").write_bytes(blob)
        found += 1
    print(f"  -> {found}/{len(ALL_SPECIES)} models retrieved")

    # --- upload to the data release ---
    if not store.can_upload():
        sys.exit("\nNo GITHUB_TOKEN — cannot write the data release.")

    print("\nUploading to the data release:")
    store.upload_csv(HISTORY_FILE, store.HISTORY_ASSET)
    if pheno_file.exists():
        store.upload_csv(pheno_file, store.PHENOLOGY_ASSET)
    if found:
        store.upload_models(MODELS_DIR)

    # --- verify by reading the release back ---
    release = store._get_release()
    assert release is not None, "release missing after upload"
    print("\nRelease now holds:")
    for asset in sorted(release["assets"], key=lambda a: a["name"]):
        print(f"  {asset['name']:<24} {asset['size'] / 1e6:>8.1f} MB")

    names = {a["name"] for a in release["assets"]}
    if store.HISTORY_ASSET not in names:
        sys.exit(f"\n{store.HISTORY_ASSET} missing from the release after upload.")
    print("\nSeeding complete — re-lock the S3 bucket now.")


if __name__ == "__main__":
    main()
