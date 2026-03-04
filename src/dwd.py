"""
Client for DWD (Deutscher Wetterdienst) pollen data.

Provides:
  - fetch_dwd_forecast()  — today/tomorrow/day-after categorical pollen forecast
    for the Oberbayern region (partregion_id 121).
  - fetch_dwd_phenology() — multi-decade flowering-onset dates from DWD/CDC
    phenology observations for Munich-area stations.
"""

from __future__ import annotations

from datetime import date, timedelta
from io import StringIO

import httpx
import pandas as pd

# ── DWD Pollenflug-Gefahrenindex (categorical forecast, updated ~daily) ──────

DWD_FORECAST_URL = (
    "https://opendata.dwd.de/climate_environment/health/alerts/s31fg.json"
)

# Munich is in "Allgäu / Oberbayern / Bay. Wald"
PARTREGION_ID = 121

# DWD German species names → our canonical names
DWD_SPECIES_MAP: dict[str, str] = {
    "Erle": "Alnus",
    "Ambrosia": "Ambrosia",
    "Beifuss": "Artemisia",
    "Birke": "Betula",
    "Hasel": "Corylus",
    "Esche": "Fraxinus",
    "Graeser": "Poaceae",
    "Roggen": "Poaceae",  # rye pollen grouped with grasses
}

# DWD level codes → numeric midpoint (scale 0–3)
_LEVEL_MAP: dict[str, float] = {
    "0": 0.0,
    "0-1": 0.5,
    "1": 1.0,
    "1-2": 1.5,
    "2": 2.0,
    "2-3": 2.5,
    "3": 3.0,
}


def fetch_dwd_forecast() -> pd.DataFrame:
    """
    Fetch the current DWD Pollenflug-Gefahrenindex for Oberbayern.

    Returns a DataFrame with columns:
        date, species, dwd_level  (float 0–3)

    Rows for today, tomorrow, and day-after-tomorrow.
    """
    resp = httpx.get(DWD_FORECAST_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Parse "last_update" → base date
    last_update = data.get("last_update", "")
    # Format: "2026-03-04 11:00 Uhr"
    try:
        base_date = pd.Timestamp(last_update.split(" ")[0]).date()
    except Exception:
        base_date = date.today()

    # Find our region
    region = None
    for r in data.get("content", []):
        if r.get("partregion_id") == PARTREGION_ID:
            region = r
            break
    if region is None:
        return pd.DataFrame(columns=["date", "species", "dwd_level"])

    day_keys = {
        "today": base_date,
        "tomorrow": base_date + timedelta(days=1),
        "dayafter_to": base_date + timedelta(days=2),
    }

    rows: list[dict] = []
    for dwd_name, pollen_data in region.get("Pollen", {}).items():
        species = DWD_SPECIES_MAP.get(dwd_name)
        if species is None:
            continue
        for day_key, dt in day_keys.items():
            raw = pollen_data.get(day_key, "0")
            level = _LEVEL_MAP.get(raw, 0.0)
            rows.append({"date": pd.Timestamp(dt), "species": species, "dwd_level": level})

    df = pd.DataFrame(rows)
    # If Roggen and Graeser both map to Poaceae, take the max per day
    df = df.groupby(["date", "species"], as_index=False)["dwd_level"].max()
    return df


# ── DWD CDC Phenology – flowering onset observations ────────────────────────

# Base URL for annual reporters, wild plants, historical + recent
_PHENO_BASE = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/phenology/annual_reporters/wild"
)

# Phase_id 5 = "Beginn der Blüte" (beginning of flowering)
_PHASE_FLOWERING = 5

# DWD phenology file stems → our species.  Only species that produce
# allergenic wind-dispersed pollen are included.
_PHENO_FILES: dict[str, str] = {
    "Hasel": "Corylus",            # 1930–2023
    "Haenge-Birke": "Betula",      # 1930–2023
    "Esche": "Fraxinus",           # 1940–2023
    "Beifuss": "Artemisia",        # 1991–2023
}

# Munich & nearby station IDs (from PH station catalogue)
_MUNICH_STATIONS: set[int] = {
    10952,  # Freising
    10955,  # München-Freimann
    10956,  # München-Trudering
    10957,  # München-Ludwigsvorstadt
    10958,  # München-Schwabing
    10959,  # München-Neufriedenheim
    10960,  # München-Feldmoching
    10961,  # München-Pasing
    10992,  # Karlsfeld
    11006,  # Zorneding
}


def _parse_pheno_file(text: str) -> pd.DataFrame:
    """Parse a semicolon-delimited DWD phenology file."""
    # Clean header — strip whitespace from column names
    lines = text.splitlines()
    header = [h.strip() for h in lines[0].split(";")]
    data_lines = "\n".join(lines[1:])
    df = pd.read_csv(
        StringIO(data_lines),
        sep=";",
        header=None,
        names=header,
        skipinitialspace=True,
    )
    # Trim whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def fetch_dwd_phenology() -> pd.DataFrame:
    """
    Download flowering-onset dates for Munich-area stations from DWD CDC.

    Returns a DataFrame:
        species, year, onset_doy  (day-of-year when flowering begins)

    This covers decades of data (1930–2023 for some species) and is used
    to learn typical season-start timing and year-to-year variability.
    """
    all_rows: list[dict] = []

    for plant_name, species in _PHENO_FILES.items():
        for period in ("historical", "recent"):
            # Try to discover the file name via directory listing
            dir_url = f"{_PHENO_BASE}/{period}/"
            try:
                listing = httpx.get(dir_url, timeout=15, follow_redirects=True)
                listing.raise_for_status()
            except httpx.HTTPError:
                continue

            # Find the matching file (name pattern includes year range)
            import re

            pattern = rf'href="(PH_Jahresmelder_Wildwachsende_Pflanze_{plant_name}_\d+_\d+_\w+\.txt)"'
            match = re.search(pattern, listing.text)
            if not match:
                continue

            file_url = f"{dir_url}{match.group(1)}"
            try:
                print(f"    Downloading {match.group(1)}...")
                resp = httpx.get(file_url, timeout=120, follow_redirects=True)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  Warning: could not download {file_url}: {exc}")
                continue

            df = _parse_pheno_file(resp.text)

            # Filter to Munich stations and flowering phase
            df["Stations_id"] = pd.to_numeric(df["Stations_id"], errors="coerce")
            df["Phase_id"] = pd.to_numeric(df["Phase_id"], errors="coerce")
            df["Jultag"] = pd.to_numeric(df["Jultag"], errors="coerce")

            mask = (
                df["Stations_id"].isin(_MUNICH_STATIONS)
                & (df["Phase_id"] == _PHASE_FLOWERING)
                & df["Jultag"].notna()
            )
            filtered = df[mask]

            for _, row in filtered.iterrows():
                all_rows.append(
                    {
                        "species": species,
                        "year": int(row["Referenzjahr"]),
                        "onset_doy": int(row["Jultag"]),
                        "station_id": int(row["Stations_id"]),
                    }
                )

            n = len(filtered)
            if n > 0:
                print(f"  {species} ({period}): {n} flowering-onset records")

    result = pd.DataFrame(all_rows)
    if not result.empty:
        # Average across Munich stations per year → one onset_doy per (species, year)
        result = (
            result.groupby(["species", "year"], as_index=False)["onset_doy"]
            .mean()
            .round()
            .astype({"onset_doy": int})
        )
    return result


def phenology_season_stats(phenology: pd.DataFrame) -> dict[str, dict]:
    """
    Compute per-species flowering statistics from multi-year phenology data.

    Returns dict[species] → {
        "mean_onset_doy": float,
        "std_onset_doy": float,
        "earliest_onset_doy": int,
        "latest_onset_doy": int,
    }
    """
    stats: dict[str, dict] = {}
    for species, grp in phenology.groupby("species"):
        stats[str(species)] = {
            "mean_onset_doy": grp["onset_doy"].mean(),
            "std_onset_doy": grp["onset_doy"].std(),
            "earliest_onset_doy": int(grp["onset_doy"].min()),
            "latest_onset_doy": int(grp["onset_doy"].max()),
        }
    return stats
